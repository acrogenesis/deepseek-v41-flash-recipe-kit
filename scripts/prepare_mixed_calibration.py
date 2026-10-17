#!/usr/bin/env python3
"""Prepare a pinned, varied calibration extension with held-out overlap removal."""
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "lemon07r/bartowski-imatrix-v5-semantic"
REVISION = "a306f203ee4323e0afe846ae02c2daafe17384d9"
DIRECTORY = ROOT / "work/data"
DIRECTORY.mkdir(parents=True, exist_ok=True)
source = DIRECTORY / "bartowski-v5-a306f203.jsonl"
if not source.exists():
    urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/bartowski-imatrix-v5-semantic.jsonl", source)
readme = DIRECTORY / "Bartowski-v5-README.md"
if not readme.exists():
    urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md", readme)


def shingles(text):
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i+12]) for i in range(len(words)-11)}


def bucket(text):
    for name, pattern in [("kana", r"[\u3040-\u30ff]"), ("cjk", r"[\u4e00-\u9fff]"),
                          ("cyrillic", r"[\u0400-\u04ff]"), ("arabic", r"[\u0600-\u06ff]"),
                          ("devanagari", r"[\u0900-\u097f]"), ("hangul", r"[\uac00-\ud7af]"),
                          ("accented_latin", r"[\u00c0-\u024f]")]:
        if len(re.findall(pattern, text)) >= 10:
            return name
    if re.search(r"\bdef \w+\(|#include|\bfunction\b|\bpublic class\b|\breturn\b.*;|\bSELECT\b.*\bFROM\b", text):
        return "code"
    return "other_text"


heldout = DIRECTORY / "heldout.txt"
forbidden = shingles(heldout.read_text())
fixture = ROOT / "results/20261017T145141Z-baseline/fixture.json"
benchmark = json.loads(fixture.read_text())
evaluation_prompts = [" ".join(re.findall(r"\w+", message["content"].lower()))
                      for case in benchmark["checks"]+benchmark["speed"] for message in case["messages"]]
groups = {}
excluded = []
for index, line in enumerate(source.read_text().splitlines()):
    text = json.loads(line)["text"].strip()
    normalized = " ".join(re.findall(r"\w+", text.lower()))
    if shingles(text) & forbidden or any(prompt in normalized for prompt in evaluation_prompts):
        excluded.append(index)
        continue
    group = bucket(text)
    groups.setdefault(group, []).append({"source_row": index, "bucket": group, "text": text[:6000]})
rng = random.Random(1017)
available = {name:len(rows) for name,rows in groups.items()}
for rows in groups.values(): rng.shuffle(rows)
queues = {name:deque(groups[name]) for name in sorted(groups)}
selected = []
while len(selected) < 288 and any(queues.values()):
    for queue in queues.values():
        if queue and len(selected) < 288:
            selected.append(queue.popleft())
output = DIRECTORY / "calibration-mixed.txt"
output.write_text("\n\n".join(row["text"] for row in selected)+"\n")
report = {"source_repo": REPO, "source_revision": REVISION,
          "source_url": f"https://huggingface.co/datasets/{REPO}",
          "upstream_credit": "Bartowski v5, semantically segmented by lemon07r",
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "heldout_sha256": hashlib.sha256(heldout.read_bytes()).hexdigest(),
          "benchmark_fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
          "overlap_policy": "Exclude any source document sharing a normalized 12-word sequence with held-out text or containing a complete benchmark prompt",
          "excluded_source_rows": excluded, "selection_seed": 1017,
          "selection": "Round-robin across Unicode-script/code/other buckets; shuffle within each bucket; first 6000 characters of each selected document",
          "available_buckets": available, "selected_buckets": dict(Counter(r["bucket"] for r in selected)),
          "selected_rows": [{k:v for k,v in r.items() if k != "text"} for r in selected],
          "file": output.name, "bytes": output.stat().st_size,
          "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
(DIRECTORY / "calibration-mixed.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k != "selected_rows"},indent=2))
