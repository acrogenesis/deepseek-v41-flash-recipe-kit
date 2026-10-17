#!/usr/bin/env python3
"""Prepare disjoint, deterministic calibration and held-out WikiText samples."""
import hashlib
import json
from pathlib import Path
import random
import urllib.request

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
BASE = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}"
out = ROOT / "work/data"
out.mkdir(parents=True, exist_ok=True)
manifest = {"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "revision": REVISION,
            "source": "https://huggingface.co/datasets/Salesforce/wikitext", "seed": 1017, "files": []}
for split, name, count in [("train", "calibration", 512), ("test", "heldout", 64)]:
    parquet = out / f"{split}.parquet"
    if not parquet.exists():
        urllib.request.urlretrieve(f"{BASE}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet", parquet)
    paragraphs = [s.strip() for s in pq.read_table(parquet)["text"].to_pylist() if len(s.strip()) >= 300]
    random.Random(1017).shuffle(paragraphs)
    # Sample complete paragraphs. Training and evaluation never share a dataset split.
    text = "\n\n".join(paragraphs[:count]) + "\n"
    target = out / f"{name}.txt"
    target.write_text(text)
    manifest["files"].append({"split": split, "file": target.name, "paragraphs": count,
                              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                              "source_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest()})
urllib.request.urlretrieve(f"{BASE}/README.md", out / "WikiText-README.md")
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
