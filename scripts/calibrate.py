#!/usr/bin/env python3
"""Collect expert importance values with original IDs and recorded coverage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--chunks", type=int, default=32)
ap.add_argument("--context", type=int, default=512)
ap.add_argument("--data", type=Path, help="Calibration corpus; default is the pinned WikiText training sample")
ap.add_argument("--resume-from", type=Path, help="Completed imatrix to extend; --chunks is the total target")
ap.add_argument("--no-escape", action="store_true", help="Keep code backslashes literal")
ap.add_argument("--profile", choices=("baseline", "streaming", "gpu"), default="baseline",
                help="gpu is an unchanged fully-GPU reference for small validation models")
args = ap.parse_args()
used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
if used > 3000:
    raise SystemExit(f"GPU busy ({used} MiB); stop the other campaign before calibration")
if args.out.exists():
    raise SystemExit(f"Refusing to overwrite {args.out}")
args.out.parent.mkdir(parents=True, exist_ok=True)
data = (args.data or ROOT / "work/data/calibration.txt").resolve()
data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
resume = None
previous_chunks = skip_chunks = 0
sources = []
if args.resume_from:
    previous = json.loads(args.resume_from.with_suffix(".json").read_text())
    if (previous["status"] != "complete" or not previous.get("all_statistics_finite")
            or previous["source_model"] != str(args.model.resolve())
            or previous["context"] != args.context or previous["pins"] != json.loads((ROOT / "pins.json").read_text())):
        raise SystemExit("Resume matrix has incompatible source, context or pins")
    with args.resume_from.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != previous["sha256"]:
            raise SystemExit("Resume matrix hash mismatch")
    previous_chunks = previous["completed_chunks"]
    if not 0 < previous_chunks < args.chunks:
        raise SystemExit("Target chunks must exceed the completed prefix")
    if previous["corpus_sha256"] == data_hash:
        skip_chunks = previous.get("chunks_from_current_corpus", previous_chunks)
    sources = previous.get("calibration_sources", [{"sha256": previous["corpus_sha256"], "chunks": previous_chunks}])
    resume = {"file": str(args.resume_from.resolve()), "sha256": previous["sha256"], "completed_chunks": previous_chunks}
new_chunks = args.chunks-previous_chunks
sources.append({"file": str(data), "sha256": data_hash, "start_chunk": skip_chunks, "chunks": new_chunks})
build = "build-streaming" if args.profile == "streaming" else "build"
command = [str(ROOT / f"work/{build}/bin/llama-imatrix"), "-m", str(args.model.resolve()),
           "-f", str(data), "-o", str(args.out.resolve()), "--output-format", "gguf",
           "-c", str(args.context), "-b", str(args.context), "-ub", "256", "--chunks", str(new_chunks), "--no-ppl",
           "-t", "16", "-ngl", "99", "-ot", "engram_embd.weight=CPU",
           "-lm", "mmap", "-lzm", "on", "--no-repack", "--fit", "off", "-fa", "on"]
if resume:
    command += ["--in-file", str(args.resume_from.resolve()), "--from-chunk", str(skip_chunks)]
if args.no_escape:
    command += ["--no-escape"]
# The imatrix executable disables warmup internally; it has no --no-warmup flag.
if args.profile == "baseline":
    command += ["-cmoe"]
elif args.profile == "streaming":
    command += ["--moe-stream", "--moe-stream-cache", os.environ.get("GPU_CACHE_GIB", "16"),
                "--moe-stream-l2", os.environ.get("RAM_CACHE_GIB", "64"),
                "--moe-stream-io-threads", "4", "--moe-stream-direct"]
environment = os.environ.copy()
environment["LLAMA_MOE_STREAM_NO_PRELOAD"] = "1"
report = {"status": "running", "command": command, "requested_chunks": args.chunks, "context": args.context,
          "source_model": str(args.model.resolve()), "profile": args.profile,
          "runtime": "original expert IDs; masked parking pairs excluded",
          "corpus_sha256": data_hash, "calibration_sources": sources, "resumed_from": resume,
          "new_chunks": new_chunks, "chunks_from_current_corpus": skip_chunks+new_chunks,
          "pins": json.loads((ROOT / "pins.json").read_text())}
manifest = args.out.with_suffix(".json")
manifest.write_text(json.dumps(report, indent=2) + "\n")
start = time.monotonic()
with args.out.with_suffix(".log").open("w") as log:
    result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
report.update(returncode=result.returncode, seconds=time.monotonic() - start, status="failed")
if result.returncode == 0 and args.out.exists():
    reader = gguf.GGUFReader(args.out)
    report["completed_chunks"] = int(reader.fields["imatrix.chunk_count"].contents())
    report["recorded_chunk_size"] = int(reader.fields["imatrix.chunk_size"].contents())
    finite = all(bool(np.isfinite(tensor.data).all()) for tensor in reader.tensors)
    report["all_statistics_finite"] = finite
    model_header = gguf.GGUFReader(args.model)
    architecture = model_header.fields["general.architecture"].contents()
    experts_used = int(model_header.fields[f"{architecture}.expert_used_count"].contents())
    expected_routes = args.chunks * args.context * experts_used
    counts = []
    for tensor in reader.tensors:
        if "_exps.weight.counts" in tensor.name:
            values = tensor.data.reshape(-1)
            counts.append({"tensor": tensor.name.removesuffix(".counts"), "experts": len(values),
                           "observed": int(np.count_nonzero(values)), "min_tokens": float(values.min()),
                           "max_tokens": float(values.max()), "total_routes": int(values.sum())})
    report["expected_routes_per_tensor"] = expected_routes
    report["routing_counts_consistent"] = bool(counts) and all(c["total_routes"] == expected_routes for c in counts)
    if (counts and finite and report["completed_chunks"] == args.chunks
            and report["recorded_chunk_size"] == args.context
            and report["routing_counts_consistent"]
            and all("stream_cache" not in t.name for t in reader.tensors)):
        report.update(status="complete", coverage=counts,
                      observed_experts=sum(x["observed"] for x in counts),
                      total_experts=sum(x["experts"] for x in counts))
        with args.out.open("rb") as stream:
            report["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
manifest.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("coverage", "pins")}, indent=2))
if report["status"] != "complete":
    raise SystemExit(1)
