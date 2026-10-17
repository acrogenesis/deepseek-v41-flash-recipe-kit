#!/usr/bin/env python3
"""Plot paired perplexity changes from completed measurement artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs=2, type=Path, help="IQ2 and public Q2 comparison quality.json files")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
rows = [json.loads(path.read_text()) for path in args.results]
assert all(r["status"] == "complete" for r in rows)
assert len({(r["context"], r["chunks"], r["corpus_sha256"], r["scored_positions"], r["reference_ppl"]) for r in rows}) == 1
labels = {"iq2-vs-mxfp4-q8": "IQ2_XXS experts\nQ8 dense + Engram",
          "q2-vs-mxfp4-q8": "Public Q2_K mix"}
assert all(r["label"] in labels for r in rows)
background, grid, ink, muted = "#07110d", "#284338", "#e8f3ed", "#a3bcae"
colors = ["#b2ff68", "#ffbd70"]
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": ink,
                     "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": ink})
fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=background)
ax.set_facecolor(background)
fig.subplots_adjust(left=.27, right=.90, top=.68, bottom=.28)
changes = [(r["ppl_ratio"] - 1) * 100 for r in rows]
errors = [r["ppl_ratio_se"] * 100 for r in rows]
span = max(abs(v) + e for v, e in zip(changes, errors)) or 1
for y, (row, value, error, color) in enumerate(zip(rows, changes, errors, colors)):
    ax.barh(y, value, height=.3, color=color, alpha=.28, zorder=2)
    ax.errorbar(value, y, xerr=error, fmt="o", color=color, capsize=5,
                markersize=6, markeredgecolor=background, linewidth=1.6, zorder=3)
    ax.annotate(f"{value:+.2f}%", (max(0, value) + error, y), xytext=(12, 0),
                textcoords="offset points", va="center", color=color, fontsize=15, weight="bold")
ax.set_yticks(range(len(rows)), [labels[r["label"]] for r in rows], fontsize=12)
ax.set_ylim(len(rows) - .5, -.5)
ax.set_xlim(min(0, min(v-e for v, e in zip(changes, errors))) - span*.04,
            max(0, max(v+e for v, e in zip(changes, errors))) + span*.27)
ax.xaxis.set_major_formatter(PercentFormatter(100))
ax.xaxis.grid(True, color=grid, linewidth=.8)
ax.axvline(0, color=muted, linewidth=.8)
ax.tick_params(axis="both", length=0, pad=10)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel("perplexity change relative to the converted reference", labelpad=15, fontsize=11)
fig.text(.075, .94, "FIELD NOTES / RTX 5090 / DEEPSEEK V4.1 FLASH 1017", color=colors[0], fontsize=10, family="monospace")
fig.text(.075, .855, "Measured quantization loss", fontsize=25, weight="bold")
fig.text(.075, .78, "Held-out text · lower is better", fontsize=12, color=muted)
fig.text(.075, .11, f"{rows[0]['scored_positions']:,} scored WikiText-2 positions. Whiskers: tool-reported standard error.", color=muted, fontsize=10)
fig.text(.075, .06, "Reference: native MXFP4 experts, Q8 dense + Engram, BF16 embedding/output. Includes format-specific kernels.", color=muted, fontsize=9)
args.out.mkdir(parents=True, exist_ok=True)
for extension in ("png", "svg"):
    fig.savefig(args.out / f"quantization-loss.{extension}", dpi=180, facecolor=background)
plt.close(fig)
