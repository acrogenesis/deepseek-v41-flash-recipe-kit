#!/usr/bin/env python3
"""Exercise prefill past sparse top-k, then decode at the recipe's 16K limit.

This is one functional retrieval check, not a long-context capability benchmark.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from models import ROOT, model_info
from probe import render, generate, request
from run_probe import monitor

gpu = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
if gpu > 3000:
    raise SystemExit("GPU busy; preserve the existing workload")
variant = os.environ.get("MODEL_VARIANT", "baseline")
model = model_info(variant)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = ROOT / "results" / f"{stamp}-{variant}-context16k"
out.mkdir(parents=True)
env = os.environ.copy()
env.update(PROFILE="streaming", MODEL_VARIANT=variant, CONTEXT="16384", PORT="5017",
           GPU_CACHE_GIB="16", RAM_CACHE_GIB="64")
active_units = [unit for unit in ("deepseek41-download", "deepseek41-reference-download", "deepseek41-conversion-campaign")
                if subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0]
report = {"status": "running", "purpose": "One retrieval smoke check at 16K configured context; not a capability benchmark",
          "configured_context": 16384, "target_prompt_tokens": 16128, "output_cap": 64,
          "expected": "K7M2R9", "model": model, "pins": json.loads((ROOT / "pins.json").read_text()),
          "timings_are_benchmark": False, "active_recipe_units_at_start": active_units,
          "timing_note": "Single-request smoke timings; inspect recorded concurrent recipe units and resource samples"}
path = out / "context-smoke.json"
path.write_text(json.dumps(report, indent=2) + "\n")
stop = threading.Event()
with (out / "server.log").open("w") as log:
    server = subprocess.Popen(["bash", "scripts/serve.sh"], cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    watcher = threading.Thread(target=monitor, args=(server.pid, out / "resources.jsonl", stop), daemon=True)
    watcher.start()
    try:
        deadline = time.monotonic() + 600
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited {server.returncode}")
            try:
                if request("http://127.0.0.1:5017/health", timeout=2)["status"] == "ok":
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Server startup")
            time.sleep(2)
        filler = (ROOT / "work/data/heldout.txt").read_text()
        text = ("Remember this access code: K7M2R9. The following reference material is unrelated to the code.\n\n"
                + filler * 3 + "\n\nWhat is the access code at the beginning? Return only that code.")
        prompt = render("http://127.0.0.1:5017", [{"role": "user", "content": text}])
        ids = prompt["input_ids"]
        assert len(ids) > report["target_prompt_tokens"]
        # Trim only the middle of the user document, preserving both official
        # message framing and the final question/assistant generation suffix.
        ids = ids[:report["target_prompt_tokens"] - 128] + ids[-128:]
        fixture = ROOT / "work/data/context16k-input-ids.json"
        fixture.write_text(json.dumps(ids) + "\n")
        report.update(prompt_tokens=len(ids), input_ids_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
        path.write_text(json.dumps(report, indent=2) + "\n")
        report["answer"] = generate("http://127.0.0.1:5017", ids, report["output_cap"])
        report["retrieval_passed"] = report["answer"]["text"].strip() == report["expected"]
        if not report["retrieval_passed"]:
            raise RuntimeError("The server completed the request but did not retrieve the expected code")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        stop.set()
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        watcher.join(timeout=10)
        path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("model", "pins", "answer")}, indent=2))
