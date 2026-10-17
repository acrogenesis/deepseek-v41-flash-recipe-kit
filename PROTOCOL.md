# DeepSeek V4.1 Flash 1017 measurement plan

Exact model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, pinned in `pins.json`.
The initial Q2_K is mrfakename's public conversion, not a quantization made by this project.
All 40 layers and 384 experts per layer remain. No expert pruning or disabled Engram.

## First gate: a usable baseline

- Build the pinned V4.1 runtime for SM120a and verify every downloaded shard with its published SHA256.
- Run inside a systemd user unit with MemoryHigh=72 GiB, MemoryMax=80 GiB and no model swap. The initial CPU baseline used an 88 GiB limit; the final matched campaign uses 80 GiB for every variant. Keep the existing desktop and GLM setup intact.
- Keep experts mapped in CPU memory and Engram rows on SSD. Disable CPU repacking, which can materialize a full extra copy of expert weights.
- Start with 8,192 configured context, one request at a time, FP16 KV, no speculation. The current runtime lacks the full hierarchical candidate mask; its usable limit must not be advertised as the model's 1M context.
- Render prompts with the exact release's official Python encoding, including its special tokens. Test chat mode first. Retain rendered prompts, token IDs and completions.

## Speed

Measure TTFT, decode tokens/s from first to last token, the server's own decode timing, total request latency, GPU memory and power, process memory, page faults and physical reads.

The primary decode rate is the server's generated-token count divided by its
decode interval (excluding the first token). Keep the independently measured
client stream rate as a cross-check. Streaming can withhold incomplete UTF-8
bytes and omit EOS, so delivered IDs alone can undercount generated tokens.
TTFT measures the first delivered token; an empty visible response has no TTFT
and fails answer checks even if the engine generated tokens internally.

First run a short real completion, then several distinct fresh prompts. Keep repeats separate: streaming expert caches can make identical prompts look much faster than new conversation content. Prefix KV reuse is disabled for measured requests. A warm-repeat result is not a fresh-content result, even with KV reuse disabled.

Compare runtime variants on identical saved prompt IDs, sampling settings, context allocation and output caps. Record natural EOS or a length cutoff. Do not mix reasoning and chat mode results. Initial capped smoke timings are not a completed throughput campaign.

## Quality

First score deterministic extraction, arithmetic, ordering and JSON tasks against exact expected answers. Retain failed answers. This is a functional check of the serving path, not a measurement of quantization loss or a substitute for a broad capability benchmark.

Measure held-out negative log likelihood against the converted MXFP4/Q8 source used for quantization, plus a fixed task panel. Use the same tokenizer, Engram release, prompts and answer budget. Measure runtime drift separately from quantization loss; a fluent completion alone does not establish either. The exact native FP8/MXFP4 forward pass is not measured in this campaign.

The conversion reference is mrfakename's Q8_0
GGUF: routed experts retain native MXFP4, while dense weights and Engram are
converted to Q8. Any comparison to this file must be labeled as relative to that
converted reference, not exact native-forward parity. Custom expert quantization
must start from its MXFP4 tensors, never from the Q2_K baseline.

Use pinned WikiText-2 raw training paragraphs for importance-matrix calibration
and disjoint test paragraphs for held-out loss. `scripts/prepare_data.py` records
the source revision, selection seed and file hashes. Keep evaluation prompts out
of calibration. Report the number of scored tokens and expert calibration coverage.

The pinned perplexity tool saves per-token log-probabilities as uint16 values,
with a float scale and offset, clipping the logit range to 16 below the maximum.
Its KL calculation omits reconstructed log-probabilities at or below -16.
These are approximate distribution comparisons. Paired PPL ratios use the
reference reconstructed from that file, which can differ slightly from the
standalone reference PPL; report both and use the paired value in comparison
tables. This describes the existing tool format, not a change to the measurements
or acceptance thresholds. See `log_softmax` in the pinned
[perplexity implementation](https://github.com/vcruz305/llama.cpp/blob/2076412598e99127ea2554467b2b7530856895d3/tools/perplexity/perplexity.cpp#L79).

For the runtime-only gate, compare the same quant in the unchanged CPU-expert
runtime and the GPU streaming runtime using teacher-forced distributions. The
initial thresholds are at least 99% top-1 agreement and mean KL divergence no
greater than 0.001 across at least 256 scored positions. Report failures rather
than changing the thresholds after seeing the result. Do not apply this runtime
equivalence threshold to a deliberately different quantization.

Exercise batched prefill and single-token kernels separately. In this pinned
runtime, operations on CPU-resident weights can move to CUDA when the batch
reaches 32 tokens. A passing 256-token-microbatch comparison therefore cannot
establish CPU-versus-GPU decode parity. Use `run_quality.py --ubatch 1` for the
second comparison and retain any changed generated answers.

The custom quantization campaign uses eight 512-token held-out chunks (2,040
scored positions) and a common streaming path for the reference and both quants.
Start calibration with 32 WikiText training chunks. If some experts are not
observed, reuse the completed statistics and add 96 chunks from a pinned
[Bartowski-v5 selection](https://huggingface.co/datasets/lemon07r/bartowski-imatrix-v5-semantic).
The selection spans available Unicode-script, code and prose buckets and excludes
documents with a normalized 12-word held-out overlap or a complete benchmark prompt.
Its source revision, selected rows and hashes are recorded separately. Continuation
was checked against a single uninterrupted synthetic calibration run. Require
finite statistics and exact routing totals for all 120 routed tensors. Record
the IDs and count of experts that remain unseen; for those entries, use the
pinned quantizer's standard unit importance weights. A synthetic byte comparison
verified that fallback against explicitly supplied unit weights. This replaces
the initial full-coverage prerequisite; the original runtime-equivalence
thresholds above are unchanged. `quantize.py --require-full-coverage` retains
the stricter option. This is a bounded initial calibration, not a claim of
optimal quality. The subsequent speed comparison fixes GPU/RAM cache budgets at
16/64 GiB for all three weight variants.

The native release, quantization output and conversion scratch must fit within a written disk budget. Never convert a community Q2 model again and describe it as a fresh quantization from the native source.

Embed the verified text-chat template in the custom GGUF's first shard. Check
that this rewrite preserves every copied tensor byte and all unrelated encoded
metadata, especially the tokenizer. Allow scratch space for a first-shard copy
before conversion starts. The serving path exercises the custom artifact's
embedded template; it supplies an override only for the public baseline/reference.

Initial disk budget (decimal GB): 264.5 baseline + 508.0 converted reference +
approximately 359.3 for IQ2_XXS experts with reference dense/Engram retained =
approximately 1,131.8 GB, plus small runtime/data/log artifacts. The machine had
about 1.3 TiB free at the start. Recheck actual free space before downloading the
reference and before writing the quant. Avoid a separate full native checkpoint
copy alongside this plan. Quantization uses bounded RAM buffers; it must not
materialize the full dequantized model in RAM or on disk.
