#!/usr/bin/env python3
"""Check sampled GGUF MXFP4 experts against exact native checkpoint bytes.

This checks sampled lossless packing, not complete conversion correctness.
HTTP ranges avoid downloading the full native checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, nargs="+", default=[0])
parser.add_argument("--experts", type=int, nargs="+", default=[0, 383])
parser.add_argument("--local-reference", action="store_true")
parser.add_argument("--out", type=Path, default=ROOT / "work/feasibility/reference-native-check.json")
args = parser.parse_args()
if any(not 0 <= x < 40 for x in args.layers) or any(not 0 <= x < 384 for x in args.experts):
    parser.error("Layers must be 0..39 and experts 0..383")
if not args.local_reference and args.layers != [0]:
    parser.error("Additional layers require --local-reference")
pins = json.loads((ROOT / "pins.json").read_text())
headers = {}


def fetch(pin, file, start, size):
    url = f"https://huggingface.co/{pin['repo']}/resolve/{pin['revision']}/{file}?byte_range={start}-{start + size - 1}"
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"Server did not honor range: {response.status}")
        data = response.read(size + 1)
    if len(data) != size:
        raise RuntimeError(f"Wrong range length: expected {size}, got {len(data)}")
    return data


def native(name):
    file = index[name]
    if file not in headers:
        length, = struct.unpack("<Q", fetch(pins["native"], file, 0, 8))
        headers[file] = (json.loads(fetch(pins["native"], file, 8, length)), 8 + length)
    header, offset = headers[file]
    item = header[name]
    start, end = item["data_offsets"]
    data = fetch(pins["native"], file, offset + start, end - start)
    return np.frombuffer(data, dtype=np.uint8).reshape(item["shape"])


class HeaderReader(gguf.GGUFReader):
    def _build_tensors(self, start_offs, fields):
        self.tensor_metadata = []
        for field in fields:
            _, name, _, dims, dtype, offset = field.parts
            kind = gguf.GGMLQuantizationType(dtype[0])
            block, size = gguf.GGML_QUANT_SIZES[kind]
            elements = 1
            for dimension in dims:
                elements *= int(dimension)
            self.tensor_metadata.append({"name": bytes(name).decode(), "type": kind.name,
                                         "bytes": elements * size // block,
                                         "offset": start_offs + int(offset[0])})


cache = ROOT / "work/feasibility"
cache.mkdir(parents=True, exist_ok=True)
index_path = cache / "model.safetensors.index.json"
if not index_path.exists():
    pin = pins["native"]
    urllib.request.urlretrieve(f"https://huggingface.co/{pin['repo']}/resolve/{pin['revision']}/model.safetensors.index.json", index_path)
index = json.loads(index_path.read_text())["weight_map"]
remote_file = "Q8_0/DeepSeek-V4.1-Flash-1017-Q8_0-00001-of-00010.gguf"
if args.local_reference:
    tensors = []
    for path in sorted((ROOT / "work/models/DeepSeek-V4.1-Flash-1017-GGUF/Q8_0").glob("*.gguf")):
        reader = gguf.GGUFReader(path)
        tensors.extend({"name": t.name, "type": t.tensor_type.name, "bytes": int(t.n_bytes),
                        "offset": int(t.data_offset), "path": path} for t in reader.tensors)
else:
    prefix = cache / "reference-header.gguf.prefix"
    if not prefix.exists():
        prefix.write_bytes(fetch(pins["reference"], remote_file, 0, 8 * 1024**2))
    tensors = HeaderReader(prefix).tensor_metadata
report = {"native": pins["native"], "reference": pins["reference"],
          "scope": "Selected layers and experts, gate/up/down only; not whole-model parity",
          "layers": args.layers, "experts": args.experts, "checks": []}
for layer, expert in ((layer, expert) for layer in args.layers for expert in args.experts):
    for projection, gguf_name in [("w1", "gate"), ("w2", "down"), ("w3", "up")]:
        name = f"layers.{layer}.ffn.experts.{expert}.{projection}"
        weight, scale = native(name + ".weight"), native(name + ".scale")
        rows, packed_cols = weight.shape
        blocks = packed_cols // 16
        source = weight.reshape(rows, blocks, 16)
        values = np.stack((source & 15, source >> 4), axis=-1).reshape(rows, blocks, 32)
        codes = values[:, :, :16] | (values[:, :, 16:] << 4)
        packed = np.concatenate((scale[:, :, None], codes), axis=-1).tobytes()
        tensor = next(t for t in tensors if t["name"] == f"blk.{layer}.ffn_{gguf_name}_exps.weight")
        assert tensor["type"] == "MXFP4" and tensor["bytes"] // 384 == len(packed)
        offset = tensor["offset"] + expert * len(packed)
        if args.local_reference:
            with tensor["path"].open("rb") as stream:
                stream.seek(offset)
                actual = stream.read(len(packed))
        else:
            actual = fetch(pins["reference"], remote_file, offset, len(packed))
        row = {"native_tensor": name, "gguf_tensor": tensor["name"], "expert": expert,
               "bytes": len(packed), "sha256_native_repacked": hashlib.sha256(packed).hexdigest(),
               "sha256_gguf": hashlib.sha256(actual).hexdigest(), "equal": packed == actual}
        report["checks"].append(row)
        print(name, "MATCH" if row["equal"] else "MISMATCH", flush=True)
report["passed"] = all(c["equal"] for c in report["checks"])
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
if not report["passed"]:
    raise SystemExit(1)
