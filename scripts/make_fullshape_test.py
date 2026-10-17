#!/usr/bin/env python3
"""Place one real DeepSeek expert layer in a synthetic, GPU-sized Llama wrapper.

This tests original cache tensor bytes at the actual 384-expert dimensions. The
random attention/router/output weights make it unsuitable for capability scores.
"""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--experts", type=Path, required=True, help="First DeepSeek Q2_K or MXFP4 reference shard")
ap.add_argument("--tokenizer-model", type=Path, required=True, help="Small 512-token test tokenizer")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
if args.out.exists():
    raise SystemExit(f"Refusing to overwrite {args.out}")
args.out.parent.mkdir(parents=True, exist_ok=True)
source = gguf.GGUFReader(args.experts)
tokenizer = gguf.GGUFReader(args.tokenizer_model)
experts = [t for t in source.tensors if t.name in {
    "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"}]
assert len(experts) == 3
writer = gguf.GGUFWriter(args.out, "llama")
for key, field in tokenizer.fields.items():
    if key.startswith("tokenizer."):
        writer.add_key_value(key, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
writer.add_name("Synthetic wrapper with actual DeepSeek layer-0 expert bytes; not a trained model")
writer.add_context_length(512)
writer.add_embedding_length(5120)
writer.add_block_count(1)
writer.add_feed_forward_length(2304)
writer.add_head_count(80)
writer.add_head_count_kv(8)
writer.add_rope_dimension_count(64)
writer.add_layer_norm_rms_eps(1e-5)
writer.add_expert_count(384)
writer.add_expert_used_count(6)
rng = np.random.default_rng(1017)
vocab = len(tokenizer.fields["tokenizer.ggml.tokens"].contents())
assert vocab == 512
for name, shape in [("token_embd.weight", (vocab, 5120)), ("output.weight", (vocab, 5120)),
                    ("blk.0.attn_q.weight", (5120, 5120)), ("blk.0.attn_k.weight", (512, 5120)),
                    ("blk.0.attn_v.weight", (512, 5120)), ("blk.0.attn_output.weight", (5120, 5120)),
                    ("blk.0.ffn_gate_inp.weight", (384, 5120))]:
    # Keep token-specific variation larger than the random attention residual so
    # the router exercises many experts and the 1 GiB test L2 must evict slabs.
    scale = .05 if name == "token_embd.weight" else .005
    writer.add_tensor(name, rng.normal(0, scale, shape).astype(np.float32))
for name in ("output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"):
    writer.add_tensor(name, np.ones((5120,), dtype=np.float32))
for tensor in experts:
    writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
report = {"purpose": "Full expert dimensions and actual mixed-quant bytes in a synthetic wrapper; not model quality",
          "seed": 1017, "embedding_std": .05, "other_random_weight_std": .005,
          "expert_source": str(args.experts.resolve()), "expert_tensors": [
              {"name": t.name, "shape": [int(x) for x in t.shape], "type": t.tensor_type.name, "bytes": t.n_bytes} for t in experts],
          "output": str(args.out.resolve()), "bytes": args.out.stat().st_size}
args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
