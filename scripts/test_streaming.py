#!/usr/bin/env python3
"""Exercise cache eviction and multipass prefill on a small synthetic MoE.

This verifies runtime plumbing; these are not DeepSeek quality or speed results.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer-model", type=Path, required=True)
ap.add_argument("--out", type=Path, default=ROOT / "work/streaming-selftest")
ap.add_argument("--allow-preload", action="store_true", help="Reproduce the experimental preloading path; disabled in the serving recipe")
args = ap.parse_args()
env = os.environ.copy()
if args.allow_preload:
    env.pop("LLAMA_MOE_STREAM_NO_PRELOAD", None)
else:
    env["LLAMA_MOE_STREAM_NO_PRELOAD"] = "1"
out = args.out
out.mkdir(parents=True, exist_ok=True)
reader = gguf.GGUFReader(args.tokenizer_model)
writer = gguf.GGUFWriter(out / "synthetic-f32.gguf", "llama")
for key, field in reader.fields.items():
    if key.startswith("tokenizer."):
        writer.add_key_value(key, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
writer.add_name("Synthetic MoE cache test, not a trained model")
writer.add_context_length(512)
writer.add_embedding_length(256)
writer.add_block_count(2)
writer.add_feed_forward_length(256)
writer.add_head_count(8)
writer.add_head_count_kv(4)
writer.add_rope_dimension_count(32)
writer.add_layer_norm_rms_eps(1e-5)
writer.add_expert_count(16)
writer.add_expert_used_count(2)
rng = np.random.default_rng(1017)
vocab = len(reader.fields["tokenizer.ggml.tokens"].contents())

def tensor(name, shape, norm=False):
    value = np.ones(shape, dtype=np.float32) if norm else rng.normal(0, .04, shape).astype(np.float32)
    writer.add_tensor(name, value)

tensor("token_embd.weight", (vocab, 256))
tensor("output.weight", (vocab, 256))
tensor("output_norm.weight", (256,), True)
for layer in range(2):
    prefix = f"blk.{layer}."
    for name, shape in [("attn_q", (256, 256)), ("attn_k", (128, 256)), ("attn_v", (128, 256)),
                        ("attn_output", (256, 256)), ("ffn_gate_inp", (16, 256)),
                        ("ffn_gate_exps", (16, 256, 256)), ("ffn_down_exps", (16, 256, 256)),
                        ("ffn_up_exps", (16, 256, 256))]:
        tensor(prefix + name + ".weight", shape)
    tensor(prefix + "attn_norm.weight", (256,), True)
    tensor(prefix + "ffn_norm.weight", (256,), True)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
with (out / "quantize.log").open("w") as log:
    subprocess.run([str(ROOT / "work/build/bin/llama-quantize"), "--pure", str(out / "synthetic-f32.gguf"),
                    str(out / "synthetic-q2.gguf"), "Q2_K", "4"], check=True, stdout=log, stderr=subprocess.STDOUT)

prompts = [rng.integers(3, vocab, size=n).tolist() for n in (32, 128, 311)]
results = {}
for label, build, extra in [
    ("unchanged", "build", []),
    ("stream-buffered", "build-streaming", ["--moe-stream-cache", "6s", "--moe-stream-io-threads", "4"]),
    ("stream-direct", "build-streaming", ["--moe-stream-cache", "6s", "--moe-stream-io-threads", "4", "--moe-stream-direct"]),
    ("stream-buffered-l2", "build-streaming", ["--moe-stream-cache", "6s", "--moe-stream-io-threads", "4", "--moe-stream-l2", "1"]),
    ("stream-direct-l2", "build-streaming", ["--moe-stream-cache", "6s", "--moe-stream-io-threads", "4", "--moe-stream-direct", "--moe-stream-l2", "1"]),
]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [str(ROOT / f"work/{build}/bin/llama-server"), "-m", str(out / "synthetic-q2.gguf"),
               "--host", "127.0.0.1", "--port", str(port), "-ngl", "99", "-c", "512", "-np", "1",
               "-b", "512", "-ub", "128", "-t", "4", "--fit", "off", "--no-warmup", "--no-repack"] + extra
    with (out / f"{label}.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"{label} exited {server.returncode}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1):
                        break
                except OSError:
                    time.sleep(.2)
            else:
                raise TimeoutError(label)
            results[label] = []
            for prompt in prompts:
                payload = {"prompt": prompt, "n_predict": 16, "temperature": -1, "n_probs": vocab,
                           "cache_prompt": False, "return_tokens": True, "repeat_penalty": 1}
                req = urllib.request.Request(url + "/completion", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    results[label].append(json.load(response))
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    (out / f"{label}.json").write_text(json.dumps(results[label]) + "\n")

report = {"model": "synthetic two-layer 16-expert Q2_K", "purpose": "cache plumbing only",
          "preloading": args.allow_preload, "comparisons": []}
for label in ("stream-buffered", "stream-direct", "stream-buffered-l2", "stream-direct-l2"):
    for i, (reference, candidate) in enumerate(zip(results["unchanged"], results[label])):
        rprobs = reference.get("completion_probabilities", reference.get("probs", []))
        cprobs = candidate.get("completion_probabilities", candidate.get("probs", []))
        assert rprobs and cprobs, (reference.keys(), candidate.keys())
        same = reference["tokens"] == candidate["tokens"]
        same_prob_positions = [p["id"] for p in rprobs] == [p["id"] for p in cprobs]
        errors = []
        for r, c in zip(rprobs, cprobs):
            rp = {v["id"]: v["logprob"] for v in r["top_logprobs"]}
            cp = {v["id"]: v["logprob"] for v in c["top_logprobs"]}
            assert rp.keys() == cp.keys()
            errors.extend(abs(rp[k] - cp[k]) for k in rp)
        maximum = max(errors)
        report["comparisons"].append({"profile": label, "prompt_tokens": len(prompts[i]), "same_tokens": same,
                                       "max_abs_logprob_difference": maximum, "positions": len(rprobs),
                                       "passed": same and same_prob_positions and maximum < .001})
report["passed"] = all(row["passed"] for row in report["comparisons"])
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(1)
