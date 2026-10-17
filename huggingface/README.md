---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - deepseek
  - mixture-of-experts
  - quantized
  - iq2-xxs
---

# DeepSeek V4.1 Flash 1017 — IQ2_XXS experts / Q8 dense and Engram

**21.37 tok/s on one RTX 5090 32 GB with 128 GB RAM**, using the
[recipe and patched llama.cpp runtime](https://github.com/acrogenesis/deepseek-v41-flash-recipe-kit)
to stream experts from a Samsung 990 PRO SSD. Ryzen 9950X3D, Linux,
16 GiB GPU expert cache and 64 GiB RAM expert cache.

This is a **359.32 GB mixed-precision GGUF**, split into ten shards:

- IQ2_XXS routed experts; every expert is retained (384 per layer, 40 layers).
- Q8 dense and Engram weights, with BF16 embedding/output and existing F32 tensors retained.
- Both full Engram tables retained. No abliteration, expert pruning or router bias.
- Text-chat template embedded and checked against DeepSeek's official encoder.

This is not a 2-bit quant of every weight. The two Engram shards alone total
208.90 GB. The patched runtime is required for the tested SSD-streaming setup;
ordinary GGUF support does not imply support for this architecture or memory layout.

## Measured performance

| Weights | Fresh tok/s | Repeated tok/s | Fresh first token | Held-out perplexity ↓ |
|---|---:|---:|---:|---:|
| This IQ2_XXS / Q8 quant | **21.37** | **25.04** | **1.12s** | 7.57 |
| Public Q2_K mix | 10.92 | 11.01 | 3.67s | 12.65 |
| MXFP4 / Q8 converted reference | 7.08 | 7.43 | 5.02s | **6.34** |

![Generation speed](figures/generation-speed.png)

Speed: median of three runs per phase, **30–31 input tokens**, 128 output
tokens, **8,192 configured context**, one slot, temperature 0, no KV prefix reuse.
Fresh prompts follow eight functional checks; these are not cold-disk runs.
All three use the same runtime and cache budgets. These are short-prompt decode
measurements, not throughput with a full 8K input.

Quality: 2,040 scored held-out positions. Perplexity rose **19.51%** versus the
paired converted reference (tool-reported standard error 2.94 percentage points).
Mean KL was 0.365186 and top-choice agreement 76.863%. The reference preserves
native MXFP4 experts but has converted Q8 dense/Engram weights; it is not a full
native-forward comparison. Saved reference probabilities use llama.cpp's uint16
encoding and tail clipping, documented in the protocol.

This quant passed **5/8 exact-answer checks**: two returned the correct number
with extra explanation; one hit the 64-token cap before giving its final answer.
The reference passed 8/8 and public Q2 passed 7/8. Cached public Q2 produced
different repeated outputs at temperature 0; this did not occur in the three
custom-quant repeat pairs. The cause of the public Q2 variation is unresolved.

[Full results and raw evidence](https://github.com/acrogenesis/deepseek-v41-flash-recipe-kit/blob/main/RESULTS.md)
include uncertainty, failed answers, runtime checks and all launch settings.

## Run

Requires Linux, CUDA 12.9, GCC 14, CMake, Ninja, Python, Hugging Face CLI (`hf`),
a 5090-class 32 GB GPU, 128 GB host RAM and fast local NVMe. The build is pinned
to SM120a. The launcher requires at least 80 GiB available RAM and a free GPU.
Reserve the 359.32 GB download plus at least 50 GiB of free disk space.

```bash
git clone https://github.com/acrogenesis/deepseek-v41-flash-recipe-kit
cd deepseek-v41-flash-recipe-kit
bash scripts/build.sh streaming
python scripts/download.py --variant iq2
bash scripts/start.sh iq2
```

The downloader verifies all ten shard hashes. The OpenAI-compatible endpoint is
`http://127.0.0.1:5017/v1`, model `deepseek-v41-flash-1017-iq2xxs-q8`.
The recipe's real chat endpoint check passed. Stop with
`systemctl --user stop deepseek41-server`.

## Context and functionality

The tested runtime defaults to **8K**, with a **16K cap**. A 16,128-token input
took **360.54 seconds to first token**. It returned the correct retrieval code
with extra words and **failed the strict exact-answer check**. This is one
functional request, not evidence of broad long-context quality. The runtime does
not implement the complete advertised 1M attention mask.

Text chat only in this recipe: tool calls, vision, DSpark and thinking mode are
not enabled. The custom cache path matched uncached GPU execution on the sampled
510-position check (100% top-choice agreement, KL approximately zero); that is a
limited runtime check, not a claim that quantization preserves original quality.

## Quantization and provenance

Source release:
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
at `1bae0e1088993b9b82de08f59de5be5a34e42170`.
The conversion input is the Q8_0 GGUF from
[`mrfakename/DeepSeek-V4.1-Flash-1017-GGUF`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF)
at `2c97dba3934f27a99767a90830e7519cc7b8b2d9`, which retains native MXFP4
expert weights. **The public Q2_K quant was not used as conversion input.**
Sampled native expert packing and Engram conversions were checked against the
official release; all non-expert tensor payloads were preserved from the converted reference.

Importance calibration used 65,536 tokens: 32 × 512 from WikiText train and
96 × 512 from a pinned, held-out-filtered sample of
[`lemon07r/bartowski-imatrix-v5-semantic`](https://huggingface.co/datasets/lemon07r/bartowski-imatrix-v5-semantic).
Credits to Bartowski and lemon07r for that corpus. Calibration observed
46,020/46,080 expert-tensor entries. The 60 unseen entries use the pinned
quantizer's standard unit-weight fallback; their IDs and counts are recorded.

[`quantization.json`](quantization.json) records the source, calibration,
tensor checks and every shard's SHA256. [`SHA256SUMS`](SHA256SUMS) is also included.
The [recipe](https://github.com/acrogenesis/deepseek-v41-flash-recipe-kit) contains
the pinned build, conversion commands, patches, protocol and measurement scripts.

GGUF was chosen for this PC's memory budget. The
[published Sybil EXL3 recipe](https://github.com/sybil-solutions/dsv41-flash-offload#host-requirements-exact-measured-on-d119)
keeps 190.5 GiB of experts pinned in RAM and specifies a 256 GiB host. This recipe
streams experts from SSD instead. EXL3 was not benchmarked on this PC.

Runtime credits: [vcruz305's V4.1 support](https://github.com/ggml-org/llama.cpp/pull/28696),
[Crow](https://github.com/nibor1896/Crow),
[JigSawPT's streaming work](https://github.com/JigSawPT/deepseek-v41-flash-on-5090)
and the contributors credited in the recipe's patch notes. DeepSeek's MIT
license is included. This community quant is not an official DeepSeek release.
