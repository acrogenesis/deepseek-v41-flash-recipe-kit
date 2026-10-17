#!/usr/bin/env python3
"""Compare sampled converted Engram rows with the pinned native FP8 checkpoint."""
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
pins = json.loads((ROOT / "pins.json").read_text())
pin = pins["native"]


def fetch(file, start, size):
    url = f"https://huggingface.co/{pin['repo']}/resolve/{pin['revision']}/{file}?byte_range={start}-{start+size-1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{start+size-1}"}), timeout=120) as response:
        if response.status != 206:
            raise RuntimeError("Native checkpoint server did not honor the byte range")
        data = response.read(size+1)
    if len(data) != size:
        raise RuntimeError("Native checkpoint range length mismatch")
    return data


def decode_e4m3(raw):
    bits = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    if np.any((exponent == 15) & (mantissa == 7)):
        raise RuntimeError("Non-finite native Engram sample")
    values = np.ldexp(np.where(exponent == 0, mantissa/8, 1+mantissa/8),
                      np.where(exponent == 0, -6, exponent-7))
    return (values * np.where(bits >> 7, -1, 1)).astype(np.float32)


report = {"native": pins["native"], "reference": pins["reference"], "checks": [],
          "scope": "Six rows in each Engram table; sampled conversion check, not whole-model parity",
          "thresholds": {"relative_rmse_max": .01, "cosine_min": .9999}}
for layer, native_shard, gguf_shard in ((1, 47, 8), (14, 48, 9)):
    file = f"model-{native_shard:05d}-of-00048.safetensors"
    header_path = ROOT / f"work/feasibility/native-shard-{native_shard}-header.json"
    if not header_path.exists():
        length, = struct.unpack("<Q", fetch(file, 0, 8))
        header_path.write_text(json.dumps({"file": file, "data_start": length+8,
                                          "header": json.loads(fetch(file, 8, length))}, indent=2)+"\n")
    metadata = json.loads(header_path.read_text())
    header, offset = metadata["header"], metadata["data_start"]
    weight = header[f"layers.{layer}.engram.embed.weight"]
    scale = header[f"layers.{layer}.engram.embed.scale"]
    assert weight["dtype"] == "F8_E4M3" and scale["dtype"] == "F8_E8M0"
    n_rows, columns = weight["shape"]
    assert columns == 256 and scale["shape"] == [n_rows, columns//32]
    path = ROOT / f"work/models/DeepSeek-V4.1-Flash-1017-GGUF/Q8_0/DeepSeek-V4.1-Flash-1017-Q8_0-{gguf_shard:05d}-of-00010.gguf"
    reader = gguf.GGUFReader(path)
    tensor = next(t for t in reader.tensors if t.name == f"blk.{layer}.engram_embd.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0 and list(tensor.shape) == [columns, n_rows]
    row_bytes = columns//32*34
    for row in (0, 1, n_rows//3, n_rows//2, n_rows-2, n_rows-1):
        raw = fetch(file, offset+weight["data_offsets"][0]+row*columns, columns)
        scales = fetch(file, offset+scale["data_offsets"][0]+row*(columns//32), columns//32)
        native = decode_e4m3(raw).reshape(-1, 32)
        native *= np.ldexp(np.ones(columns//32, dtype=np.float32),
                          np.frombuffer(scales, dtype=np.uint8).astype(np.int32)-127)[:, None]
        native = native.reshape(-1)
        with path.open("rb") as stream:
            stream.seek(int(tensor.data_offset)+row*row_bytes)
            packed = stream.read(row_bytes)
        converted = gguf.dequantize(np.frombuffer(packed, dtype=np.uint8), tensor.tensor_type)
        repacked = gguf.quantize(native, tensor.tensor_type).tobytes()
        norm = float(np.linalg.norm(native))
        relative_rmse = float(np.linalg.norm(converted-native))/norm if norm else float(np.linalg.norm(converted))
        cosine = float(np.dot(converted, native)/(np.linalg.norm(converted)*norm)) if norm else float(not np.any(converted))
        check = {"layer": layer, "row": row, "relative_rmse": relative_rmse, "cosine": cosine,
                 "same_q8_bytes": packed == repacked,
                 "native_sample_sha256": hashlib.sha256(raw+scales).hexdigest(),
                 "gguf_sample_sha256": hashlib.sha256(packed).hexdigest(),
                 "passed": packed == repacked and relative_rmse <= .01 and cosine >= .9999}
        report["checks"].append(check)
        print(f"layer {layer}, row {row}: relative RMSE {relative_rmse:.6f}, cosine {cosine:.8f}, exact Q8 bytes {packed == repacked}", flush=True)
report["all_sampled_q8_bytes_identical"] = all(c["same_q8_bytes"] for c in report["checks"])
report["passed"] = all(c["passed"] for c in report["checks"])
(ROOT / "results/validation/reference-native-engram-samples.json").write_text(json.dumps(report, indent=2)+"\n")
if not report["passed"]:
    raise SystemExit(1)
