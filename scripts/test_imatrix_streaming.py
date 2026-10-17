#!/usr/bin/env python3
"""Compare original-expert calibration statistics with and without the cache."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True, help="Small synthetic 16-expert model only")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(GPU_CACHE_GIB="6s", RAM_CACHE_GIB="1")
report = {"model": str(args.model), "purpose": "Synthetic imatrix capture validation only", "cases": []}
for context in (1, 32, 512):
    matrices = {}
    for profile in ("gpu", "streaming"):
        target = args.out / f"{profile}-{context}.gguf"
        command = [sys.executable, str(ROOT / "scripts/calibrate.py"), "--profile", profile,
                   "--model", str(args.model), "--out", str(target), "--context", str(context), "--chunks", "4"]
        with (args.out / f"{profile}-{context}-run.log").open("w") as log:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        reader = gguf.GGUFReader(target)
        matrices[profile] = {t.name: t.data.copy() for t in reader.tensors}
    reference, candidate = matrices["gpu"], matrices["streaming"]
    assert reference.keys() == candidate.keys(), (reference.keys(), candidate.keys())
    comparisons = []
    for name in reference:
        r, c = reference[name], candidate[name]
        is_counts = name.endswith(".counts")
        passed = np.array_equal(r, c) if is_counts else np.allclose(r, c, rtol=1e-5, atol=1e-6)
        comparisons.append({"name": name, "shape": list(r.shape), "passed": bool(passed),
                            "max_abs_difference": float(np.max(np.abs(r - c)))})
    case = {"context": context, "chunks": 4, "passed": all(x["passed"] for x in comparisons), "tensors": comparisons}
    report["cases"].append(case)
    print(f"context {context}: {'PASS' if case['passed'] else 'FAIL'}", flush=True)
report["passed"] = all(c["passed"] for c in report["cases"])
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
if not report["passed"]:
    raise SystemExit(1)
