#!/usr/bin/env python3
"""Summarize completed paired runs; use client timing for both runtimes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def load(path):
    data = json.loads(path.read_text())
    if data['status'] not in ('complete', 'complete_smoke'):
        raise ValueError(f'Incomplete measurement: {path}')
    return data


def normalized(path):
    data = load(path)
    exl3 = 'runs' in data
    rows = data['runs'] if exl3 else [dict(row, phase=phase)
        for phase in ('checks', 'fresh', 'repeat') for row in data[phase]]
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'context': data['profile']['context_tokens'] if exl3 else data['context'],
            'output_cap': None if exl3 else data['output_cap'],
            'fixture_sha256': data['fixture_sha256'], 'rows': [{
                'id': r['id'], 'phase': 'check' if r['phase'] == 'checks' else r['phase'],
                'tok_s': r['decode_tok_s'] if exl3 else r['client_decode_tok_s'],
                'ttft_seconds': r['ttft_seconds'], 'passed': r.get('passed'),
                'input_tokens': r['input_tokens'] if exl3 else r['server_timings']['prompt_n'],
                'output_tokens': r['output_tokens'] if exl3 else len(r['output_ids']),
            } for r in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True,
                    help='JSON mapping workload names to GGUF/EXL3 result-path lists, plus quality paths')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = {'workloads': {}, 'quality': {}, 'metric':
              'Tokens delivered after the first token divided by time from first to last token'}
    for name, pair in manifest['workloads'].items():
        group = {model: [normalized(Path(p)) for p in paths] for model, paths in pair.items()}
        hashes = {run['fixture_sha256'] for runs in group.values() for run in runs}
        if len(hashes) != 1 or set(group) != {'GGUF', 'EXL3'}:
            raise ValueError(f'Unpaired fixtures: {name}')
        if len({run['context'] for runs in group.values() for run in runs}) != 1:
            raise ValueError(f'Different context allocations: {name}')
        result['workloads'][name] = group
    for model, path in manifest['quality'].items():
        d = load(Path(path))
        result['quality'][model] = {key: d[key] for key in (
            'ppl', 'reference_ppl', 'mean_kl', 'top1_agreement_percent',
            'scored_positions', 'context', 'corpus_sha256')}
        result['quality'][model]['path'] = str(Path(path))
        result['quality'][model]['sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    q = list(result['quality'].values())
    assert len(q) == 2 and len({(d['scored_positions'], d['context'], d['corpus_sha256']) for d in q}) == 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bg, grid, ink, muted = '#07110d', '#284338', '#e8f3ed', '#a3bcae'
    colors = {'GGUF': '#b2ff68', 'EXL3': '#51d6ed'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': ink,
                         'axes.labelcolor': muted, 'xtick.color': muted, 'ytick.color': ink})

    def chart(title, subtitle, groups, unit, filename, note):
        fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=bg)
        ax.set_facecolor(bg)
        fig.subplots_adjust(left=.22, right=.91, top=.68, bottom=.25)
        maximum = 0
        for i, (label, values) in enumerate(groups):
            for j, model in enumerate(('GGUF', 'EXL3')):
                points = values[model]
                assert points and all(v > 0 for v in points)
                mid, y, color = statistics.median(points), i + (j - .5) * .24, colors[model]
                maximum = max(maximum, *points)
                ax.barh(y, mid, height=.17, color=color, alpha=.3,
                        label=model if i == 0 else None, zorder=2)
                ax.plot([min(points), max(points)], [y, y], color=color, lw=1.5, zorder=3)
                ax.scatter(points, [y] * len(points), s=28, color=color,
                           edgecolor=bg, linewidth=.6, zorder=4)
                ax.plot([mid, mid], [y-.09, y+.09], color=color, lw=2.2, zorder=5)
                ax.annotate(f'{mid:.2f}', (max(points), y), xytext=(9, 0), textcoords='offset points',
                            va='center', color=color, fontsize=12, weight='bold')
        ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=11)
        ax.set_ylim(len(groups)-.6, -.4)
        ax.set_xlim(0, maximum*1.23)
        ax.xaxis.grid(True, color=grid, lw=.8)
        ax.tick_params(length=0, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(unit, labelpad=14)
        legend = ax.legend(loc='lower left', bbox_to_anchor=(-.02, 1.08), ncol=2, frameon=False)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
        fig.text(.075, .94, 'FIELD NOTES / RTX 5090 / DEEPSEEK V4.1 FLASH 1017',
                 color=colors['GGUF'], fontsize=10, family='monospace')
        fig.text(.075, .855, title, fontsize=25, weight='bold')
        fig.text(.075, .795, subtitle, fontsize=11, color=muted)
        fig.text(.075, .10, note, fontsize=9, color=muted)
        fig.text(.075, .05, 'acrogenesis.com · 18 OCT 2026', fontsize=9, color=muted, family='monospace')
        for extension in ('png', 'svg'):
            fig.savefig(args.out / f'{filename}.{extension}', dpi=180, facecolor=bg)
        plt.close(fig)

    for workload, pair in result['workloads'].items():
        cap = pair['GGUF'][0]['output_cap']
        context = pair['GGUF'][0]['context']
        if workload == 'long':
            for key, title, unit in [('tok_s', 'Generation with longer inputs', 'output tokens / second'),
                                     ('ttft_seconds', 'The wait with longer inputs', 'seconds to first token')]:
                lengths = sorted({row['input_tokens'] for runs in pair.values() for run in runs for row in run['rows']})
                groups = [(f'{length:,} input tokens', {model: [row[key] for run in runs for row in run['rows']
                          if row['phase'] == 'fresh' and row['input_tokens'] == length] for model, runs in pair.items()})
                          for length in lengths]
                chart(title, f'Fresh KV state · {context:,} configured context · {cap}-token output cap', groups, unit,
                      f'long-{key}', 'First use of each prompt. Each dot is one request; repeated prompts are in the raw data.')
        else:
            groups = [(label, {model: [row['tok_s'] for run in runs for row in run['rows'] if row['phase'] == phase]
                              for model, runs in pair.items()})
                      for phase, label in [('fresh', 'Fresh prompts'), ('repeat', 'Repeated prompts')]]
            chart('Generation speed', f'{workload} workload · {context:,} configured context · {cap}-token output cap',
                  groups, 'output tokens / second · higher is better', f'{workload}-speed',
                  'Bars + labels: median. Dots: individual requests. Fresh follows checks; no cold-cache claim.')
    chart('Measured language-model quality', 'Same 2,040 held-out token positions · lower perplexity is better',
          [('Held-out perplexity', {model: [d['ppl']] for model, d in result['quality'].items()})],
          'perplexity · converted reference: 6.34', 'quality',
          'WikiText-2 · 8 × 512-token chunks. Whole model/runtime comparison; not a percentage of intelligence.')


if __name__ == '__main__':
    main()
