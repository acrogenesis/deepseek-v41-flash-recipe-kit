#!/usr/bin/env python3
"""Run one bounded-memory baseline campaign after download verification."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from models import model_info

ROOT = Path(__file__).resolve().parents[1]


def monitor(pid, output, stop):
    with output.open("w") as log:
        while not stop.is_set():
            row = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            row["system_memory"] = {
                k: v.strip() for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Cached", "Dirty")
            }
            for name in ("status", "io", "stat"):
                try:
                    row[name] = Path(f"/proc/{pid}/{name}").read_text()
                except FileNotFoundError:
                    return
            row["gpu"] = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.used,power.draw,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"
            ], text=True).strip()
            log.write(json.dumps(row) + "\n")
            log.flush()
            stop.wait(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("baseline", "streaming", "gpu-offload"), default="baseline")
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--variant", choices=("baseline", "reference", "iq2"), default="baseline")
    ap.add_argument("--wait-download", action="store_true")
    args = ap.parse_args()
    receipt = ROOT / "work/feasibility" / ("download-reference-verified.json" if args.variant == "reference" else "download-verified.json")
    while not receipt.exists():
        if not args.wait_download:
            raise SystemExit("Download has not been verified")
        service = "deepseek41-reference-download" if args.variant == "reference" else "deepseek41-download"
        status = subprocess.run(["systemctl", "--user", "is-active", "--quiet", service], check=False)
        if status.returncode:
            raise SystemExit("Download stopped without a verification receipt; inspect work/logs/download.log")
        time.sleep(15)
    model = model_info(args.variant)
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
    if used > 3000:
        raise SystemExit(f"GPU is busy ({used} MiB used); preserving the existing workload")
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0]) < 80 * 1024**2:
        raise SystemExit("Less than 80 GiB RAM available; preserving other workloads")
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", 5017))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.profile if args.variant == "baseline" else f"{args.variant}-{args.profile}"
    out = ROOT / "results" / f"{stamp}-{label}"
    out.mkdir(parents=True)
    env = os.environ.copy()
    env.update(PROFILE=args.profile, MODEL_VARIANT=args.variant, CONTEXT="8192", PORT="5017")
    launch = {"profile": args.profile, "context": 8192, "port": 5017,
              "wave_preloading": False,
              "pins": json.loads((ROOT / "pins.json").read_text()), "model": model,
              "gpu_cache": env.get("GPU_CACHE_GIB", "18"), "ram_cache": env.get("RAM_CACHE_GIB", "64"),
              "command": ["bash", str(ROOT / "scripts/serve.sh")],
              "systemd_cgroup": Path('/proc/self/cgroup').read_text()}
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    stop = threading.Event()
    with (out / "server.log").open("w") as log:
        server = subprocess.Popen(launch["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        watcher = threading.Thread(target=monitor, args=(server.pid, out / "resources.jsonl", stop), daemon=True)
        watcher.start()
        try:
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; inspect {out}/server.log")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:5017/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Server did not become ready within 30 minutes")
                time.sleep(3)
            cmd = [sys.executable, str(ROOT / "scripts/probe.py"), "--label", label, "--out", str(out)]
            if args.fixture:
                cmd.extend(["--fixture", str(args.fixture.resolve())])
            subprocess.run(cmd, check=True, timeout=14400)
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
    print(f"Probe saved to {out}", flush=True)


if __name__ == "__main__":
    main()
