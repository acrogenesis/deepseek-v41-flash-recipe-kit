#!/usr/bin/env python3
"""Summarize observed and unseen routed experts in an imatrix snapshot."""
import argparse
import json
from pathlib import Path
import re

import gguf
import numpy as np


def summarize(path):
    reader = gguf.GGUFReader(path)
    rows = []
    for tensor in reader.tensors:
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\.counts", tensor.name)
        if not match:
            continue
        values = tensor.data.reshape(-1)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid counts in {tensor.name}")
        rows.append({"layer": int(match[1]), "projection": match[2],
                     "tensor": tensor.name.removesuffix(".counts"), "experts": len(values),
                     "observed": int(np.count_nonzero(values)),
                     "unseen_ids": np.flatnonzero(values == 0).tolist(),
                     "fewer_than_10_observations": int(np.count_nonzero(values < 10)),
                     "total_routes": int(values.sum()), "min_count": int(values.min()),
                     "median_count": float(np.median(values)), "max_count": int(values.max())})
    rows.sort(key=lambda row: (row["layer"], row["projection"]))
    return {"file": str(Path(path).resolve()),
            "chunks": int(reader.fields["imatrix.chunk_count"].contents()),
            "context": int(reader.fields["imatrix.chunk_size"].contents()),
            "expert_tensors": len(rows), "observed_entries": sum(row["observed"] for row in rows),
            "total_entries": sum(row["experts"] for row in rows),
            "fully_observed_tensors": sum(row["observed"] == row["experts"] for row in rows),
            "unseen_entries": sum(len(row["unseen_ids"]) for row in rows),
            "fewer_than_10_observations": sum(row["fewer_than_10_observations"] for row in rows),
            "tensors": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = summarize(args.matrix)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "tensors"}, indent=2))
