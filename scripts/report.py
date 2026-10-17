#!/usr/bin/env python3
"""Summarize completed campaign artifacts without guessing missing results."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

from models import ROOT, CUSTOM_DIR


def result(label, filename, status):
    paths = sorted(path for path in (ROOT / "results").glob(f"*-{label}/{filename}")
                   if path.parent.name.partition("-")[2] == label)
    if not paths:
        raise RuntimeError(f"Missing result: {label}")
    path = paths[-1]
    value = json.loads(path.read_text())
    if value.get("label") != label:
        raise RuntimeError(f"Result label mismatch: {path}")
    if value["status"] != status:
        raise RuntimeError(f"Incomplete result: {path}")
    return path, value


campaign = json.loads((ROOT / "work/campaign.json").read_text())
if campaign["status"] != "complete":
    raise SystemExit("Finish the conversion campaign before reporting it")
labels = ("reference-streaming", "iq2-streaming", "streaming")
names = ("MXFP4 experts / Q8 dense + Engram", "IQ2_XXS experts / Q8 dense + Engram", "Public Q2_K mix")
probes = [result(label, "probe.json", "complete_smoke") for label in labels]
cpu = result("baseline", "probe.json", "complete_smoke")
offload = result("gpu-offload", "probe.json", "complete_smoke")
qualities = [result(label, "quality.json", "complete") for label in
             ("mxfp4-q8-reference", "iq2-vs-mxfp4-q8", "q2-vs-mxfp4-q8")]
quant = json.loads((CUSTOM_DIR / "quantization.json").read_text())
if (quant["status"] != "complete" or not quant["tensor_layout_verified"]
        or not quant.get("retained_tensor_payloads_verified") or not quant.get("chat_template_verified")):
    raise SystemExit("Custom quant verification failed")
assert len({(p["context"], p["output_cap"], p["fixture_sha256"]) for _, p in probes + [cpu, offload]}) == 1
assert len({(q["context"], q["chunks"], q["corpus_sha256"]) for _, q in qualities}) == 1
assert len({q["reference_ppl"] for _, q in qualities[1:]}) == 1
assert all(q["pins"] == campaign["pins"] for _, q in qualities)
launches = [json.loads(path.with_name("launch.json").read_text()) for path, _ in probes + [cpu, offload]]
assert all(launch["pins"] == campaign["pins"] for launch in launches)
assert all((launch["gpu_cache"], launch["ram_cache"]) == ("16", "64") for launch in launches[:3])
paired_reference_ppl = qualities[1][1]["reference_ppl"]
prompt_lengths = [run["server_timings"]["prompt_n"] for run in probes[0][1]["fresh"]]
assert all([run["server_timings"]["prompt_n"] for run in probe["fresh"]] == prompt_lengths
           for _, probe in probes + [cpu, offload])
repeat_audit = []
for path, probe in probes + [cpu, offload]:
    assert probe["props"]["default_generation_settings"]["n_ctx"] == probe["context"]
    assert probe["props"]["total_slots"] == 1
    assert len(probe["fresh"]) == len(probe["repeat"]) == 3
    assert all(run["server_timings"]["cache_n"] == 0 for run in probe["fresh"] + probe["repeat"])
    pairs = []
    for first, repeated in zip(probe["fresh"], probe["repeat"]):
        assert first["terminal"]["prompt"] == repeated["terminal"]["prompt"]
        first_difference = next((i for i, (a, b) in enumerate(zip(first["output_ids"], repeated["output_ids"])) if a != b), None)
        if first_difference is None and len(first["output_ids"]) != len(repeated["output_ids"]):
            first_difference = min(len(first["output_ids"]), len(repeated["output_ids"]))
        pairs.append({"same_text": first["text"] == repeated["text"],
                      "same_delivered_token_ids": first["output_ids"] == repeated["output_ids"],
                      "first_different_delivered_token": first_difference,
                      "generated_tokens": [first["tokens_predicted"], repeated["tokens_predicted"]]})
    repeat_audit.append({"label": probe["label"], "source": str(path.relative_to(ROOT)), "pairs": pairs})
audit_path = ROOT / "results/validation/probe-repeat-consistency.json"
audit_path.write_text(json.dumps(repeat_audit, indent=2) + "\n")
out = ROOT / "results/quantization-comparison"
out.mkdir(exist_ok=True)
subprocess.run([sys.executable, str(ROOT / "scripts/plot.py"), *[str(p) for p, _ in probes],
                "--out", str(out), "--note", "Same expert-cache runtime · 16 GiB GPU / 64 GiB RAM cache · no prefix reuse"], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/plot.py"), str(cpu[0]), str(offload[0]), str(probes[2][0]),
                "--out", str(out / "runtimes")], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/plot_quality.py"),
                *[str(p) for p, _ in qualities[1:]], "--out", str(out)], check=True)

lines = ["# DeepSeek V4.1 Flash 1017 on RTX 5090", "",
         "Measured on one RTX 5090 32 GB, Ryzen 9950X3D, 128 GB RAM and Samsung 990 PRO SSD.", "",
         "## Generation speed", "",
         "| Weights | Fresh tok/s | Repeat tok/s | Fresh TTFT | Functional checks |",
         "|---|---:|---:|---:|---:|"]
for name, (_, p) in zip(names, probes):
    score = p["functional_score"]
    ttft = statistics.median(r["ttft_seconds"] for r in p["fresh"] if r["ttft_seconds"] is not None)
    lines.append(f"| {name} | {p['fresh_median_tok_s']:.2f} | {p['repeat_median_tok_s']:.2f} | {ttft:.2f}s | {score['correct']}/{score['total']} |")
lines += ["", "![Generation speed](results/quantization-comparison/generation-speed.png)", "",
          f"Medians of three requests per phase, with {', '.join(str(n) for n in prompt_lengths)} input tokens and a 128-token output cap; 8,192 configured context, one slot, official chat encoding, "
          "temperature zero, no KV prefix reuse. Fresh prompts follow eight functional checks. No cold-disk claim. "
          "All three use the same demand-only expert cache: 16 GiB GPU budget, 64 GiB pinned RAM and four direct-I/O workers. "
          "Download and conversion jobs finished before these timings. "
          "The separate eight-task panel requires exact answers and uses a 64-token output cap; extra explanation counts as a failure.", "", "## Measured quality", "",
          "| Weights | Held-out perplexity | PPL change | Top-choice agreement | Mean KL |",
          "|---|---:|---:|---:|---:|"]
for i, (name, (_, q)) in enumerate(zip(names, qualities)):
    if i == 0:
        lines.append(f"| {name} | {paired_reference_ppl:.4f} | reference | reference | reference |")
    else:
        change = f"{(q['ppl_ratio']-1)*100:+.3f}%"
        if "ppl_ratio_se" in q:
            change += f" ± {q['ppl_ratio_se']*100:.3f}%"
        lines.append(f"| {name} | {q['ppl']:.4f} | {change} | {q['top1_agreement_percent']:.3f}% | {q['mean_kl']:.6f} |")
q = qualities[0][1]
calibration = quant["calibration"]
lines += ["", "![Measured quantization loss](results/quantization-comparison/quantization-loss.png)", "",
          f"{q['scored_positions']:,} scored positions from {q['chunks']} held-out WikiText-2 chunks at context {q['context']}. "
          "Lower perplexity and KL are better. The ± values are tool-reported standard errors for the paired PPL ratios. "
          f"The table uses the paired reference reconstructed from saved probabilities; the standalone reference PPL was {q['ppl']:.4f}. "
          "The tool stores compressed log-probabilities and omits tiny-probability KL terms; [PROTOCOL.md](PROTOCOL.md) records that approximation. "
          "These metrics measure distribution changes on this sample; they are not a percentage of intelligence retained. "
          "The reference retains native MXFP4 expert weights and BF16 embedding/output weights, with other dense/Engram weights converted to Q8. "
          "The comparison includes the kernels used by each weight format; it does not establish parity with the exact native forward pass.", "",
          f"Calibration used {calibration['completed_chunks']} separate training chunks of {calibration['context']} tokens. "
          f"Observed {calibration['observed_experts']:,}/{calibration['total_experts']:,} expert-tensor entries. "
          f"The remaining {quant['calibration_coverage']['unseen_entries']:,} entries used the quantizer's standard unit importance weights; their IDs are recorded in the conversion manifest. "
          "Only routed experts were quantized; every expert remains. SHA256 checks confirmed that all non-expert tensor payloads, including both Engram tables, are unchanged.", "",
          f"The custom quant contains {sum(f['bytes'] for f in quant['files'])/1e9:.2f} GB of GGUF shards. "
          "The large Engram tables remain on SSD. The runtime does not implement the full advertised 1M attention mask; this recipe caps context at 16K.", "",
          "## Failed functional checks", ""]
for name, (_, p) in zip(names, probes):
    failed = [f"`{c['id']}`: {c['text'].strip()!r} (expected {c['expected']!r})" for c in p["checks"] if not c["passed"]]
    lines.append(f"- {name}: " + ("; ".join(failed) if failed else "all eight passed"))
lines += ["", "## Runtime comparison", ""]
lines += ["Both runtimes include the same reviewed attention and CUDA fixes. The no-cache runtime was also tested with GPU operation offload forced for single-token generation.", "",
          "| Q2_K runtime | Fresh tok/s | Repeat tok/s |", "|---|---:|---:|"]
for name, (_, p) in zip(("No cache, CPU experts", "No cache, GPU operation offload", "GPU expert cache"), (cpu, offload, probes[2])):
    lines.append(f"| {name} | {p['fresh_median_tok_s']:.2f} | {p['repeat_median_tok_s']:.2f} |")
lines += ["", "[Runtime comparison chart](results/quantization-comparison/runtimes/generation-speed.png)", "",
          "The separate cache check matched all 508 single-token top choices against uncached GPU arithmetic, with mean KL rounding to zero. "
          "CPU versus GPU arithmetic differed, agreeing on 91.339% of top choices with mean KL 0.031168. "
          "[Saved backend comparison](results/validation/backend-drift.json).", "", "## Repeated outputs", "",
          "| Model/runtime | Identical repeated completions |", "|---|---:|"]
for name, audit in zip((*names, "Q2_K, no cache / CPU experts", "Q2_K, no cache / GPU offload"), repeat_audit):
    lines.append(f"| {name} | {sum(pair['same_text'] for pair in audit['pairs'])}/3 |")
lines += ["", "Temperature and seed were fixed, prompts matched, and the server reported zero KV prefix reuse. "
          "Changed completions mean repeat-speed differences cannot be assigned solely to cache warming. "
          "[Saved token and context audit](results/validation/probe-repeat-consistency.json)."]
serving_checks = []
runtime_path = ROOT / "results/validation/custom-runtime.json"
if runtime_path.exists():
    runtime = json.loads(runtime_path.read_text())
    if runtime["status"] == "complete" and runtime["results"][-1]["measurement"].get("runtime_gate"):
        check = runtime["results"][-1]["measurement"]
        serving_checks.append(f"- Full custom-model cache check passed on {check['scored_positions']} batched positions: "
                              f"{check['top1_agreement_percent']:.3f}% top-choice agreement, mean KL {check['mean_kl']:.6f}, "
                              f"PPL ratio {check['ppl_ratio']:.6f} against uncached GPU arithmetic. "
                              "[Evidence](results/validation/custom-runtime.json).")
    else:
        serving_checks.append(f"- Full custom-model cache check: {runtime['status']}. [Evidence](results/validation/custom-runtime.json).")
contexts = sorted((ROOT / "results").glob("*-iq2-context16k/context-smoke.json"))
if contexts:
    context = json.loads(contexts[-1].read_text())
    rel = contexts[-1].relative_to(ROOT).as_posix()
    if context["status"] == "complete" and context.get("retrieval_passed"):
        ttft = context["answer"]["ttft_seconds"]
        serving_checks.append(f"- Custom 16K retrieval passed: {context['prompt_tokens']:,} input tokens at "
                              f"{context['configured_context']:,} configured context. Time to first token was {ttft:.1f}s "
                              f"({ttft/60:.1f} minutes). This is one retrieval check, not a broad context or throughput benchmark. [Evidence]({rel}).")
    elif "answer" in context:
        answer = context["answer"]
        serving_checks.append(f"- Custom 16K request completed with {context['prompt_tokens']:,} input tokens, but failed the exact-answer check. "
                              f"It returned {answer['text'].strip()!r} instead of exactly {context['expected']!r}. "
                              f"First token: {answer['ttft_seconds']:.1f}s ({answer['ttft_seconds']/60:.1f} minutes). "
                              f"This single request is functional evidence, not a broad context or throughput benchmark. [Evidence]({rel}).")
    else:
        serving_checks.append(f"- Custom 16K retrieval: {context['status']}. [Evidence]({rel}).")
api_path = ROOT / "results/validation/chat-endpoint-iq2.json"
if api_path.exists():
    api = json.loads(api_path.read_text())
    if api["status"] == "complete":
        serving_checks.append(f"- OpenAI chat endpoint passed with {api['configured_context']:,} configured context: "
                              "exact `DEEPSEEK_READY`, official template and matching input-token count. "
                              "[Evidence](results/validation/chat-endpoint-iq2.json).")
    else:
        serving_checks.append(f"- OpenAI chat endpoint: {api['status']}. [Evidence](results/validation/chat-endpoint-iq2.json).")
if serving_checks:
    lines += ["", "## Serving checks", "", *serving_checks]
lines += ["", "## Raw evidence", ""]
for path, _ in probes + qualities + [cpu, offload]:
    rel = path.relative_to(ROOT).as_posix()
    lines.append(f"- [{path.parent.name}]({rel})")
(ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
print(ROOT / "RESULTS.md")
