#!/usr/bin/env python3
"""Measure teacher-forced perplexity and runtime/quant drift with llama.cpp's saved-probability format."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import threading
import time

from run_probe import monitor

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("baseline", "streaming", "gpu"), required=True,
                    help="gpu is an unchanged fully-GPU reference for small validation models")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--ubatch", type=int, default=256,
                    help="Use 1 to exercise single-token kernels rather than batched prefill")
    ap.add_argument("--force-gpu-ops", action="store_true",
                    help="Unchanged runtime: offload CPU-resident expert operations even for one token")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--save-logits", type=Path)
    group.add_argument("--compare-logits", type=Path)
    ap.add_argument("--runtime-gate", action="store_true")
    args = ap.parse_args()
    if args.runtime_gate and not args.compare_logits:
        ap.error("--runtime-gate requires --compare-logits")
    if not 0 < args.context <= 16384 or args.chunks < 1 or args.ubatch < 1:
        ap.error("Invalid context or chunk count")
    gpu = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
    if gpu > 3000:
        raise SystemExit(f"GPU busy ({gpu} MiB); stop the other campaign before quality evaluation")
    data = ROOT / "work/data/heldout.txt"
    if not data.exists():
        raise SystemExit("Run scripts/prepare_data.py first")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"{stamp}-{args.label}"
    out.mkdir(parents=True)
    build = "build-streaming" if args.profile == "streaming" else "build"
    command = [str(ROOT / f"work/{build}/bin/llama-perplexity"), "-m", str(args.model.resolve()),
               "-c", str(args.context), "-b", str(args.context), "-ub", str(args.ubatch), "--chunks", str(args.chunks),
               "-f", str(data), "-t", os.environ.get("THREADS", "16"), "-ngl", "99", "-ot", "engram_embd.weight=CPU",
               "-lm", "mmap", "-lzm", "on", "--no-repack", "--fit", "off", "-fa", "on", "--no-warmup"]
    if os.environ.get('BATCH_THREADS'):
        command += ['-tb', os.environ['BATCH_THREADS']]
    if args.profile == "baseline":
        command += ["-cmoe"]
    elif args.profile == "streaming":
        command += ["--moe-stream", "--moe-stream-cache", os.environ.get("GPU_CACHE_GIB", "16"),
                    "--moe-stream-l2", os.environ.get("RAM_CACHE_GIB", "64"),
                    "--moe-stream-io-threads", os.environ.get("IO_THREADS", "4"), "--moe-stream-direct"]
    logits = (args.save_logits or args.compare_logits).resolve()
    if args.save_logits:
        if logits.exists():
            raise SystemExit(f"Refusing to overwrite existing reference logits: {logits}")
        logits.parent.mkdir(parents=True, exist_ok=True)
        command += ["--save-all-logits", str(logits)]
    else:
        with logits.open("rb") as f:
            magic, context, vocab, chunks = struct.unpack("<8sIii", f.read(20))
        if magic != b"_logits_" or context != args.context or chunks != args.chunks:
            raise SystemExit(f"Logits/context/chunk mismatch: {magic!r}, {context}, {chunks}")
        command += ["--kl-divergence", "--kl-divergence-base", str(logits)]
    env = os.environ.copy()
    env["LLAMA_MOE_STREAM_NO_PRELOAD"] = "1"
    if args.force_gpu_ops:
        env["GGML_OP_OFFLOAD_MIN_BATCH"] = "1"
    result = {"label": args.label, "profile": args.profile, "command": command, "status": "running",
              "context": args.context, "chunks": args.chunks, "ubatch": args.ubatch, "wave_preloading": False,
              "op_offload_min_batch": env.get("GGML_OP_OFFLOAD_MIN_BATCH", "32"),
              "scored_positions": (args.context - 1 - args.context // 2) * args.chunks,
              "corpus_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
              "pins": json.loads((ROOT / "pins.json").read_text()),
              "cgroup": Path("/proc/self/cgroup").read_text()}
    result['environment'] = {k: v for k, v in env.items()
                             if k.startswith(('LLAMA_MOE_STREAM_', 'GGML_CUDA_'))
                             or k in ('GPU_CACHE_GIB', 'RAM_CACHE_GIB', 'IO_THREADS', 'THREADS', 'BATCH_THREADS')}
    result['binary_sha256'] = {name: hashlib.sha256((ROOT / f'work/{build}/bin' / name).read_bytes()).hexdigest()
                              for name in ('llama-perplexity', 'libllama.so.0.4.1')}
    if args.profile == 'streaming':
        result['source_sha256'] = {name: hashlib.sha256((ROOT / 'work/upstream/llama-streaming' / name).read_bytes()).hexdigest()
                                   for name in ('src/llama-moe-stream.cpp', 'src/llama-moe-stream.h')}
    result_file = out / "quality.json"
    result_file.write_text(json.dumps(result, indent=2) + "\n")
    log_file = out / "quality.log"
    stop = threading.Event()
    started = time.monotonic()
    with log_file.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        watcher = threading.Thread(target=monitor, args=(process.pid, out / "resources.jsonl", stop), daemon=True)
        watcher.start()
        try:
            result["returncode"] = process.wait(timeout=21600)
        except Exception as error:
            result.update(status="failed", error=str(error), returncode=process.poll())
        finally:
            stop.set()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            watcher.join(timeout=10)
    result["seconds"] = time.monotonic() - started
    log = log_file.read_text(errors="replace")
    if args.compare_logits:
        patterns = {"ppl": r"Mean PPL\(Q\)\s*:\s*([\d.]+)",
                    "reference_ppl": r"Mean PPL\(base\)\s*:\s*([\d.]+)",
                    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)",
                    "mean_kl": r"Mean\s+KLD:\s*(-?[\d.]+)",
                    "top1_agreement_percent": r"Same top p:\s*([\d.]+)"}
    else:
        patterns = {"ppl": r"Final estimate: PPL = ([\d.]+)"}
    for key, pattern in patterns.items():
        match = re.search(pattern, log)
        if match:
            result[key] = float(match[1])
    uncertainty = {
        "ppl_ratio_se": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*[\d.]+\s*±\s*([\d.]+)",
        "mean_kl_se": r"Mean\s+KLD:\s*-?[\d.]+\s*±\s*([\d.]+)",
        "top1_agreement_se": r"Same top p:\s*[\d.]+\s*±\s*([\d.]+)",
        "ppl_se": (r"Mean PPL\(Q\)\s*:\s*[\d.]+\s*±\s*([\d.]+)" if args.compare_logits else
                   r"Final estimate: PPL = [\d.]+\s*\+/-\s*([\d.]+)"),
    }
    for key, pattern in uncertainty.items():
        match = re.search(pattern, log)
        if match:
            result[key] = float(match[1])
    result["status"] = "complete" if result.get("returncode") == 0 and all(k in result for k in patterns) else "failed"
    if args.runtime_gate and result["status"] == "complete":
        result["runtime_gate"] = (result["scored_positions"] >= 256
                                  and result["top1_agreement_percent"] >= 99
                                  and result["mean_kl"] <= .001)
    result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "complete" or result.get("runtime_gate") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
