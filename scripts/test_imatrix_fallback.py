#!/usr/bin/env python3
"""Verify the pinned quantizer's zero-count fallback on a tiny CPU fixture.

This compares tensor bytes, not DeepSeek quality. A missing expert observation
must produce the same result as explicitly supplying unit importance weights.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--imatrix", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
model = gguf.GGUFReader(args.model)
matrix = gguf.GGUFReader(args.imatrix)
expert_names = [t.name for t in model.tensors if re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", t.name)]
assert len(expert_names) == 6, "Use the tiny two-layer validation fixture"
assert all(int(t.shape[2]) == 16 for t in model.tensors if t.name in expert_names)

rules = []
for tensor in model.tensors:
    target = "iq2_xxs" if tensor.name in expert_names else tensor.tensor_type.name.lower()
    rules.append(f"^{re.escape(tensor.name)}$={target}")
rules_file = args.out / "tensor-types.txt"
rules_file.write_text("\n".join(rules) + "\n")

commands = []
for label in ("missing", "explicit-unit"):
    path = args.out / f"{label}-imatrix.gguf"
    writer = gguf.GGUFWriter(path, "")
    writer.kv_data[0].pop("general.architecture", None)
    for key, field in matrix.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        writer.add_key_value(key, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
    for tensor in matrix.tensors:
        data = tensor.data.copy()
        if tensor.name.endswith(".counts") and tensor.name.removesuffix(".counts") in expert_names:
            data.reshape(-1)[0] = 0 if label == "missing" else 1
        elif tensor.name.endswith(".in_sum2") and tensor.name.removesuffix(".in_sum2") in expert_names:
            data.reshape(16, -1)[0, :] = 0 if label == "missing" else 1
        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    command = [str(ROOT / "work/build/bin/llama-quantize"), "--allow-requantize", "--pure",
               "--imatrix", str(path), "--tensor-type-file", str(rules_file),
               str(args.model), str(args.out / f"{label}.gguf"), "IQ2_XXS", "2"]
    commands.append(command)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    with (args.out / f"{label}.log").open("w") as log:
        subprocess.run(command, env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)

readers = [gguf.GGUFReader(args.out / f"{label}.gguf") for label in ("missing", "explicit-unit")]
left, right = [{t.name: t for t in reader.tensors} for reader in readers]
assert left.keys() == right.keys()
checks = [{"tensor": name, "equal": bool(np.array_equal(left[name].data, right[name].data))} for name in left]
report = {"purpose": "Synthetic quantizer fallback check; not model quality",
          "source": "tools/quantize/quantize.cpp load_imatrix: count == 0 gives unit weights",
          "pins": json.loads((ROOT / "pins.json").read_text()), "commands": commands,
          "forced_missing_expert_ids": {name: [0] for name in expert_names},
          "checks": checks, "passed": all(row["equal"] for row in checks)}
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"passed": report["passed"], "compared_tensors": len(checks), "forced_missing_entries": len(expert_names)}))
if not report["passed"]:
    raise SystemExit(1)
