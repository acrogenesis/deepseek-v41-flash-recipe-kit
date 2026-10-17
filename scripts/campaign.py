#!/usr/bin/env python3
"""Run conversion and paired measurements serially, retaining every artifact.

Launch inside the documented bounded systemd unit. No throughput benchmark starts
until the reference download and all conversion work have finished.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from models import ROOT, CUSTOM_DIR, model_info

(ROOT / "work/logs").mkdir(parents=True, exist_ok=True)
state_path = ROOT / "work/campaign.json"
ap = argparse.ArgumentParser()
ap.add_argument("--resume", action="store_true", help="Skip completed stages with the same commands and source pins")
args = ap.parse_args()
lock = (ROOT / "work/campaign.lock").open("a")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("Another campaign is active")
pins = json.loads((ROOT / "pins.json").read_text())
if state_path.exists() and not args.resume:
    raise SystemExit("Existing campaign state; inspect it, then use --resume")
state = (json.loads(state_path.read_text()) if state_path.exists() else
         {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "stages": []})
if any(s["status"] == "complete" for s in state["stages"]) and state.get("pins") != pins:
    raise SystemExit("Completed stages use different source pins; preserve them and start a separate campaign")
state.update(status="running", pins=pins)
state.pop("error", None)
environment = os.environ.copy()
environment.update(GPU_CACHE_GIB="16", RAM_CACHE_GIB="64")
python = str(ROOT / ".venv/bin/python")


def save():
    state_path.write_text(json.dumps(state, indent=2) + "\n")


def stage(name, arguments):
    previous = [s for s in state["stages"] if s["name"] == name]
    if previous and previous[-1]["status"] == "complete":
        if previous[-1]["command"] != [python, *arguments]:
            raise RuntimeError(f"Completed stage {name} has different arguments")
        print(f"Already complete: {name}", flush=True)
        return
    row = {"name": name, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "command": [python, *arguments], "status": "running", "attempt": len(previous)+1}
    state["stages"].append(row)
    save()
    print(f"Starting {name}", flush=True)
    suffix = "" if not previous else f"-attempt{len(previous)+1}"
    row["log"] = f"work/logs/campaign-{name}{suffix}.log"
    save()
    with (ROOT / row["log"]).open("w") as log:
        result = subprocess.run(row["command"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    row.update(returncode=result.returncode, status="complete" if result.returncode == 0 else "failed",
               finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {row['log']}")


def quality(name, variant, logits, compare=False):
    stage(name, ["scripts/run_quality.py", "--profile", "streaming", "--model", model_info(variant)["model"],
                 "--label", name, "--context", "512", "--chunks", "8",
                 "--compare-logits" if compare else "--save-logits", str(logits)])


try:
    save()
    receipt = ROOT / "work/feasibility/download-reference-verified.json"
    while not receipt.exists():
        if subprocess.run(["systemctl", "--user", "is-active", "--quiet", "deepseek41-reference-download.service"]).returncode:
            raise RuntimeError("Reference download stopped before verification")
        state["waiting_for"] = "reference download and SHA256 verification"
        save()
        time.sleep(15)
    # Do not overlap with the separate diagnostic jobs or an unrelated server.
    while True:
        used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        if used <= 3000:
            break
        state["waiting_for"] = f"GPU idle ({used} MiB currently used)"
        save()
        time.sleep(15)
    state.pop("waiting_for", None)
    contexts = sorted((ROOT / "results").glob("*-baseline-context16k/context-smoke.json"))
    if contexts:
        context_check = json.loads(contexts[-1].read_text())
        if context_check["status"] != "complete" or not context_check.get("retrieval_passed"):
            raise RuntimeError("The 16K context check has not completed successfully; inspect its server log")
    logits = ROOT / f"work/quality/mxfp4-q8-reference-{pins['streaming_patch']['sha256'][:12]}.logits"
    quality("mxfp4-q8-reference", "reference", logits)
    matrix = ROOT / "work/calibration/mxfp4-q8-32.gguf"
    stage("calibration-32", ["scripts/calibrate.py", "--profile", "streaming", "--model", model_info("reference")["model"],
                             "--out", str(matrix), "--chunks", "32"])
    calibration = json.loads(matrix.with_suffix(".json").read_text())
    if calibration["observed_experts"] != calibration["total_experts"]:
        prefix = matrix
        matrix = ROOT / "work/calibration/mxfp4-q8-128.gguf"
        stage("calibration-128", ["scripts/calibrate.py", "--profile", "streaming", "--model", model_info("reference")["model"],
                                  "--out", str(matrix), "--chunks", "128", "--resume-from", str(prefix),
                                  "--data", str(ROOT / "work/data/calibration-mixed.txt"), "--no-escape"])
    stage("quantize-iq2", ["scripts/quantize.py", "--imatrix", str(matrix), "--out", str(CUSTOM_DIR)])
    quality("iq2-vs-mxfp4-q8", "iq2", logits, compare=True)
    quality("q2-vs-mxfp4-q8", "baseline", logits, compare=True)
    fixture = ROOT / "results/20261017T145141Z-baseline/fixture.json"
    for variant in ("reference", "iq2", "baseline"):
        stage(f"probe-{variant}", ["scripts/run_probe.py", "--profile", "streaming", "--variant", variant,
                                   "--fixture", str(fixture)])
    stage("probe-unchanged-gpu-offload", ["scripts/run_probe.py", "--profile", "gpu-offload", "--variant", "baseline",
                                         "--fixture", str(fixture)])
    stage("probe-unchanged-cpu", ["scripts/run_probe.py", "--profile", "baseline", "--variant", "baseline",
                                 "--fixture", str(fixture)])
    state["status"] = "complete"
except Exception as error:
    state.update(status="failed", error=str(error))
    print(str(error), file=sys.stderr, flush=True)
    raise
finally:
    save()

subprocess.run([python, "scripts/report.py"], cwd=ROOT, check=True)
