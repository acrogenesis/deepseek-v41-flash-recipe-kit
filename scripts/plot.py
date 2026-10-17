#!/usr/bin/env python3
"""Plot saved measurements only; distinguish fresh prompts from repeated prompts."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+", type=Path, help="Completed probe.json files")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--note", default="Same Q2_K weights · no prefix-cache reuse · short capped benchmark")
args = ap.parse_args()
rows = [json.loads(p.read_text()) for p in args.results]
assert rows and all(r["status"] == "complete_smoke" for r in rows)
assert len(rows) <= 3
assert len({(r["context"], r["output_cap"], r["fixture_sha256"]) for r in rows}) == 1, "Use matched fixtures/context/output caps"
prompt_lengths = [run["server_timings"]["prompt_n"] for run in rows[0]["fresh"]]
prompt_range = f"{min(prompt_lengths)}–{max(prompt_lengths)}" if min(prompt_lengths) != max(prompt_lengths) else str(prompt_lengths[0])
args.out.mkdir(parents=True, exist_ok=True)

background, grid, ink, muted = "#07110d", "#284338", "#e8f3ed", "#a3bcae"
colors = ["#b2ff68", "#51d6ed", "#ffbd70"]
display_names = {"baseline": "CPU experts", "streaming": "GPU expert cache", "gpu-offload": "GPU offload, no cache"}
profile_colors = {"baseline": colors[1], "streaming": colors[0], "gpu-offload": colors[2]}
if any(r["label"] in ("reference-streaming", "iq2-streaming") for r in rows):
    display_names.update({"reference-streaming": "MXFP4 experts", "iq2-streaming": "IQ2_XXS experts", "streaming": "Q2_K mix"})
    profile_colors.update({"reference-streaming": colors[1], "iq2-streaming": colors[0], "streaming": colors[2]})
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": ink,
                     "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": ink})
for key, title, unit, filename in [
    ("decode_tok_s", "Generation speed", "output tokens / second · higher is better", "generation-speed"),
    ("ttft_seconds", "The wait before the answer", "seconds to first token · lower is better", "time-to-first-token"),
]:
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=background)
    ax.set_facecolor(background)
    fig.subplots_adjust(left=.20, right=.91, top=.69, bottom=.29)
    all_values = []
    for phase_index, phase in enumerate(("fresh", "repeat")):
        for i, row in enumerate(rows):
            values = [r[key] for r in row[phase] if r[key] is not None]
            if not values:
                continue
            all_values += values
            y = phase_index * 1.15 + (i - (len(rows) - 1) / 2) * .22
            median = statistics.median(values)
            color = profile_colors.get(row["label"], colors[i])
            ax.barh(y, median, height=.15, color=color, alpha=.28,
                    label=display_names.get(row["label"], row["label"]) if phase_index == 0 else None, zorder=2)
            ax.plot([min(values), max(values)], [y, y], color=color, linewidth=1.4, zorder=3)
            ax.scatter(values, [y] * len(values), s=30, color=color, edgecolor=background, linewidth=.7, zorder=4)
            ax.plot([median, median], [y - .08, y + .08], color=color, linewidth=2.2, zorder=5)
            ax.annotate(f"{median:.2f}", (max(values), y), xytext=(9, 0), textcoords="offset points",
                        va="center", color=color, fontsize=12, weight="bold")
    ax.set_yticks([0, 1.15], ["Fresh prompts", "Repeated prompts"], fontsize=12)
    ax.set_ylim(1.55, -.4)
    ax.set_xlim(0, max(all_values) * 1.23)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.grid(True, color=grid, linewidth=.8)
    ax.tick_params(axis="both", length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(unit, labelpad=15, fontsize=11)
    fig.text(.075, .94, "FIELD NOTES / RTX 5090 / DEEPSEEK V4.1 FLASH 1017", color=colors[0], fontsize=10, family="monospace")
    fig.text(.075, .855, title, fontsize=25, weight="bold")
    legend = ax.legend(loc="lower left", bbox_to_anchor=(-.02, 1.05), ncol=len(rows), frameon=False, fontsize=10 if len(rows) == 3 else 11)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    fig.text(.075, .105, f"Bars + labels: median. Dots: individual runs. {prompt_range} input tokens; {rows[0]['output_cap']}-token output cap.", color=muted, fontsize=10)
    fig.text(.075, .055, f"{rows[0]['context']:,} configured context · {args.note}", color=muted, fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(args.out / f"{filename}.{extension}", dpi=180, facecolor=background)
    plt.close(fig)
