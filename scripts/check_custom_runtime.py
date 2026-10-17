#!/usr/bin/env python3
"""Compare the verified custom model with and without caching on real held-out text.

Run after the speed campaign, inside the same bounded systemd configuration.
This keeps the original runtime gate: >=99% top choices, KL <=0.001, >=256 positions.
"""
import datetime
import json
import os
import subprocess
import sys

from models import ROOT, model_info

if subprocess.run(["systemctl", "--user", "is-active", "--quiet",
                   "deepseek41-conversion-campaign.service"]).returncode == 0:
    raise SystemExit("Wait for the speed campaign to finish")

model = model_info("iq2")["model"]
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
logits = ROOT / f"work/quality/iq2-runtime-gpu-{stamp}.logits"
env = os.environ.copy()
env.update(GPU_CACHE_GIB="16", RAM_CACHE_GIB="64")
common = [sys.executable, str(ROOT / "scripts/run_quality.py"), "--model", model,
          "--context", "512", "--chunks", "2", "--ubatch", "256"]
stages = [
    ("iq2-runtime-gpu-reference", ["--profile", "baseline", "--force-gpu-ops", "--save-logits", str(logits)]),
    ("iq2-runtime-cache-compare", ["--profile", "streaming", "--compare-logits", str(logits), "--runtime-gate"]),
]
summary = {"status": "running", "purpose": "Full custom-model batched cache comparison; not a speed benchmark",
           "results": [], "pins": json.loads((ROOT / "pins.json").read_text())}
out = ROOT / "results/validation/custom-runtime.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2) + "\n")
try:
    for label, arguments in stages:
        print(f"Starting {label}", flush=True)
        before = set((ROOT / "results").glob(f"*-{label}/quality.json"))
        run = subprocess.run([*common, "--label", label, *arguments], cwd=ROOT,
                             env=env, text=True, stdout=subprocess.PIPE)
        if run.stdout:
            print(run.stdout, flush=True)
        artifacts = sorted(set((ROOT / "results").glob(f"*-{label}/quality.json")) - before)
        if len(artifacts) == 1:
            result = json.loads(artifacts[0].read_text())
            summary["results"].append({"artifact": str(artifacts[0].relative_to(ROOT)), "measurement": result})
        elif run.returncode == 0:
            raise RuntimeError(f"Expected exactly one new measurement for {label}")
        out.write_text(json.dumps(summary, indent=2) + "\n")
        run.check_returncode()
    summary["status"] = "complete"
except Exception as error:
    summary.update(status="failed", error=str(error))
    raise
finally:
    out.write_text(json.dumps(summary, indent=2) + "\n")
