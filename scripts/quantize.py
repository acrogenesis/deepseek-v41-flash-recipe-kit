#!/usr/bin/env python3
"""Quantize native MXFP4 routed experts; preserve every other reference tensor.

Requires the verified converted reference and a completed importance matrix.
This never uses the downloaded Q2_K baseline as its input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import gguf

from calibration_coverage import summarize
from embed_chat_template import embed

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--imatrix", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True, help="New directory for quantized shards")
ap.add_argument("--threads", type=int, default=24)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--require-full-coverage", action="store_true",
                help="Reject unseen experts instead of using the quantizer's recorded unit-weight fallback")
args = ap.parse_args()
receipt = ROOT / "work/feasibility/download-reference-verified.json"
if not receipt.exists():
    raise SystemExit("Download and verify the converted reference first")
calibration = json.loads(args.imatrix.with_suffix(".json").read_text())
if (calibration["status"] != "complete" or not calibration.get("all_statistics_finite")
        or not calibration.get("routing_counts_consistent")
        or calibration.get("completed_chunks") != calibration["requested_chunks"]):
    raise SystemExit("Calibration is not complete")
coverage = calibration.get("coverage", [])
if len(coverage) != 120 or any(x["experts"] != 384 or x["observed"] == 0 for x in coverage):
    raise SystemExit("Calibration must contain observations for all 120 routed tensors, each with 384 expert slots")
with args.imatrix.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != calibration["sha256"]:
        raise SystemExit("Importance-matrix hash mismatch")
coverage_detail = summarize(args.imatrix)
if (coverage_detail["expert_tensors"] != 120 or coverage_detail["total_entries"] != 46080
        or coverage_detail["observed_entries"] != calibration["observed_experts"]
        or coverage_detail["chunks"] != calibration["completed_chunks"]
        or any(t["total_routes"] != calibration["expected_routes_per_tensor"] for t in coverage_detail["tensors"])):
    raise SystemExit("Importance-matrix coverage does not match its completed calibration manifest")
unseen = coverage_detail["unseen_entries"]
if unseen and args.require_full_coverage:
    raise SystemExit(f"Strict coverage requested: {unseen}/46080 expert-tensor entries are unseen")
if unseen:
    print(f"Calibration did not observe {unseen}/46080 expert-tensor entries; "
          "the pinned quantizer uses unit importance weights for those entries. "
          "Their IDs will be recorded; this is not a quality claim.", flush=True)
if args.out.exists() and any(args.out.iterdir()):
    raise SystemExit(f"Output directory must be empty: {args.out}")
args.out.mkdir(parents=True, exist_ok=True)
source = ROOT / "work/models/DeepSeek-V4.1-Flash-1017-GGUF/Q8_0"
files = sorted(source.glob("*.gguf"))
if len(files) != 10:
    raise SystemExit("Expected ten verified reference shards")
if Path(calibration["source_model"]).resolve() != files[0].resolve():
    raise SystemExit("Importance matrix must be collected from this MXFP4/Q8 reference")
rules, tensors, expected_bytes, expert_count = [], [], 0, 0
retained_sources = {}
for file in files:
    reader = gguf.GGUFReader(file)
    for tensor in reader.tensors:
        kind = tensor.tensor_type
        expert = re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", tensor.name) is not None
        if expert:
            if kind != gguf.GGMLQuantizationType.MXFP4 or int(tensor.shape[2]) != 384:
                raise SystemExit(f"Unexpected reference expert type/shape: {tensor.name}")
            target = gguf.GGMLQuantizationType.IQ2_XXS
            expert_count += 1
        else:
            target = kind
            retained_sources[tensor.name] = (file, int(tensor.data_offset), int(tensor.n_bytes))
        block, size = gguf.GGML_QUANT_SIZES[target]
        count = 1
        for dimension in tensor.shape:
            count *= int(dimension)
        expected_bytes += count * size // block
        rules.append(f"^{re.escape(tensor.name)}$={target.name.lower()}")
        tensors.append({"name": tensor.name, "source_type": kind.name, "target_type": target.name,
                        "shape": [int(x) for x in tensor.shape]})
if expert_count != 120:
    raise SystemExit(f"Expected 120 routed expert tensors, found {expert_count}")
metadata_scratch_bytes = files[0].stat().st_size + 1024**2
if shutil.disk_usage(args.out).free < expected_bytes + metadata_scratch_bytes + 50 * 1024**3:
    raise SystemExit("Insufficient free space for the quant, first-shard metadata copy and a 50 GiB reserve")
rules_file = args.out / "tensor-types.txt"
rules_file.write_text("\n".join(rules) + "\n")
prefix = args.out / "DeepSeek-V4.1-Flash-1017-IQ2_XXS-Experts-Q8-Dense-Engram"
command = [str(ROOT / "work/build/bin/llama-quantize"), "--allow-requantize", "--pure", "--keep-split",
           "--max-buffer-size", "2048", "--imatrix", str(args.imatrix.resolve()),
           "--tensor-type-file", str(rules_file.resolve())]
if args.dry_run:
    command += ["--dry-run"]
command += [str(files[0]), str(prefix.resolve()), "IQ2_XXS", str(args.threads)]
report = {"status": "running", "phase": "quantizing", "command": command, "reference": json.loads(receipt.read_text()),
          "calibration": calibration, "expert_tensors": expert_count, "tensor_payload_bytes": expected_bytes,
          "metadata_scratch_budget_bytes": metadata_scratch_bytes,
          "calibration_coverage": coverage_detail,
          "unobserved_importance_policy": "unit weights, as implemented by the pinned llama.cpp quantizer",
          "require_full_coverage": args.require_full_coverage,
          "tensors": tensors, "policy": "IQ2_XXS routed experts; all non-expert tensor types and payloads retained"}
manifest = args.out / "quantization.json"

def save_report():
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(manifest)

save_report()
started = time.monotonic()
with (args.out / "quantize.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
report.update(returncode=result.returncode, seconds=time.monotonic() - started, status="failed")
if result.returncode == 0:
    report["status"] = "dry_run_complete" if args.dry_run else "verifying"
    if not args.dry_run:
        report["phase"] = "embedding the verified chat template"
        save_report()
        print("Embedding the verified V4.1 text-chat template in the first shard", flush=True)
        first_output = sorted(args.out.glob("*.gguf"))[0]
        report["chat_template"] = embed(first_output, ROOT / "templates/deepseek-v41-chat.jinja")
        report["chat_template_verified"] = True
        report["phase"] = "tensor layout verification"
        save_report()
        print("Quantization finished; verifying output tensor layout", flush=True)
        outputs = sorted(args.out.glob("*.gguf"))
        if len(outputs) != 10:
            report["status"] = "failed"
        expected = {t["name"]: t for t in tensors}
        observed = {}
        retained_outputs = {}
        for file in outputs:
            reader = gguf.GGUFReader(file)
            for tensor in reader.tensors:
                if tensor.name in observed:
                    raise RuntimeError(f"Duplicate output tensor: {tensor.name}")
                observed[tensor.name] = {"type": tensor.tensor_type.name, "shape": [int(x) for x in tensor.shape]}
                if tensor.name in retained_sources:
                    retained_outputs[tensor.name] = (file, int(tensor.data_offset), int(tensor.n_bytes))
        report["tensor_layout_verified"] = (
            observed.keys() == expected.keys()
            and all(t["type"] == expected[n]["target_type"] and t["shape"] == expected[n]["shape"]
                    for n, t in observed.items()))
        if not report["tensor_layout_verified"]:
            report["status"] = "failed"
        # Verify the retained payloads themselves, including the two large Engram
        # tables. Bounded reads avoid materializing these tensors in RAM.
        report["phase"] = "retained tensor payload verification"
        save_report()
        print("Verifying that retained tensor payloads are byte-identical", flush=True)
        def payload_digest(location):
            file, offset, remaining = location
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                stream.seek(offset)
                while remaining:
                    data = stream.read(min(remaining, 32 * 1024**2))
                    if not data:
                        raise RuntimeError(f"Truncated tensor payload: {file}")
                    digest.update(data)
                    remaining -= len(data)
            return digest.hexdigest()

        report["retained_tensor_checks"] = []
        for name, location in retained_sources.items():
            if location[2] >= 1024**3:
                print(f"Checking retained payload {name} ({location[2]/1e9:.2f} GB)", flush=True)
            other = retained_outputs.get(name)
            source_digest = payload_digest(location)
            output_digest = payload_digest(other) if other else None
            report["retained_tensor_checks"].append({
                "name": name, "source_sha256": source_digest, "output_sha256": output_digest,
                "equal": other is not None and location[2] == other[2] and source_digest == output_digest,
            })
        report["retained_tensor_payloads_verified"] = all(t["equal"] for t in report["retained_tensor_checks"])
        if not report["retained_tensor_payloads_verified"]:
            report["status"] = "failed"
        report["phase"] = "output shard SHA256 verification"
        report["files"] = []
        save_report()
        for file in outputs:
            print(f"Hashing output shard {file.name}", flush=True)
            with file.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            report["files"].append({"file": file.name, "bytes": file.stat().st_size, "sha256": digest})
            save_report()
        if report["status"] != "failed":
            report["status"] = "complete"
report["phase"] = report["status"]
save_report()
print(json.dumps({k: v for k, v in report.items() if k not in ("reference", "calibration", "calibration_coverage", "tensors", "retained_tensor_checks")}, indent=2))
if report["status"] == "failed":
    raise SystemExit(1)
