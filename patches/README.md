# Experimental expert streaming

`moe-stream.patch` adapts the expert streaming change from
[JigSawPT/llama.cpp commit 4aa9d62](https://github.com/JigSawPT/llama.cpp/commit/4aa9d62c871a55021bc2c2d35ee84bdd5dd28c6e)
to the pinned V4.1 PR runtime. Its original implementation is from
[nibor1896/Crow](https://github.com/nibor1896/Crow), with JigSawPT's CLOCK host-cache eviction.
The included MIT license applies.

This is an experimental adaptation. The measured speed and quality results are
in [RESULTS.md](../RESULTS.md), with check-specific limits in
[VALIDATION.md](../VALIDATION.md). It has not been submitted upstream.
It retains the PR runtime's Engram representation and V4.1 graph. Experts stream into bounded VRAM and pinned-RAM caches. The adaptation resolves the new lazy-tensor flag collision, keeps the current graph-node budgets and sampling code, merges new command-line flags, and adds the streaming source to the current CMake source list.

Streaming is opt-in. Router bias, expert pinning and speculative prefetch are not enabled. Compare against the unchanged runtime with identical weights and prompt IDs before drawing conclusions about accuracy or speed.

The serving script also disables wave preloading (`LLAMA_MOE_STREAM_NO_PRELOAD=1`).
A synthetic two-layer MoE test found changed output probabilities with preloading
and the direct-I/O/RAM-cache combination. With preloading disabled, all four
buffered/direct and RAM-cache/no-RAM-cache configurations produced identical
tokens and returned log probabilities to the unchanged runtime on 32-, 128- and
311-token prompts. This validates those small cache tests only; it does not establish
DeepSeek accuracy. Failed and passing raw runs remain under ignored `work/`.

Two local additions support this machine and the quantization workflow:

- Streaming skips whole-checkpoint mmap prefetch at load. The expert cache reads
  weights on demand instead of filling the OS cache before allocating its host tier.
- `llama-imatrix` captures original expert IDs, keeps full expert dimensions and
  excludes masked parking pairs during multi-pass prefill. Capture is enabled only
  by imatrix, with wave preloading disabled. Single-token, 32-token and 512-token
  synthetic calibration runs produced identical tensor names, expert counts and
  activation sums to the unchanged fully-GPU runtime. CPU-expert calibration in
  the unchanged runtime remains available as a reference.

## Common runtime fixes

`runtime-fixes.patch` is applied to the no-cache runtime and is also included in
`moe-stream.patch`, so later comparisons share the same attention and CUDA fixes.
The initial short measurements predate these fixes; their saved pins identify
that original source state.

- Reuse the most recent index source's top-k tensor in following compressed
  attention layers, asserting that its source matches `dsv41_topk_source`. This
  follows [nicholasshirley's September 28 review report](https://github.com/ggml-org/llama.cpp/pull/28696)
  and the exact release's `Attention._compress_topk_idxs`. The pinned PR otherwise
  attends densely in reuse layers once more than 512 compressed rows exist. The
  source-ID assertion follows the same approach documented in saturnsky/ranma.cpp.
- Preserve [Mermiges' MMQ tail-padding fix and regression test](https://github.com/Mermiges/pli-local-llm-serving/blob/main/patches/xid31-mul-mat-id-padding.patch),
  adapting the test insertion location and using the existing MUL_MAT_ID
  tolerance for native Blackwell FP4 activations. The padding covers the widest
  selectable tile instead of deriving zero padding from a broadcast dimension.
  Upstream also fixes this in [fc9ce6b](https://github.com/ggml-org/llama.cpp/commit/fc9ce6b9d52a8504edcb262abc92737c2289f96c).
- Filter non-matrix operations before `llama-imatrix` reads their source tensors.
  DeepSeek's Engram scale dequantization creates a source-free `ARANGE`; the
  original collector dereferenced its null source while looking for a tensor
  name. This was reproduced under GDB. The fixed full-model calibration chunk
  collected all 120 expert tensors with finite statistics and exactly 3,072
  routed observations each (512 tokens times six experts).

The two-level candidate mask is still absent, so these fixes do not enable the
model's advertised 1M context. The recipe remains capped at 16K.

## Optional transfer and cache settings

The following additions preserve the stored weight bytes and selected expert IDs.
They are disabled by default and have been exercised on CUDA/RTX 5090 only.

- `LLAMA_MOE_STREAM_UPLOAD_BATCH=3` groups up to three weight transfers on one
  worker-owned CUDA backend before synchronizing. Disk reads remain parallel;
  uploads remain serialized. A host-cache lease stays live until its transfer
  finishes, and a GPU slot becomes resident only after every tensor arrives.
  Accepted values are 1–6; 1 uses the original worker path.
- `LLAMA_MOE_STREAM_KEEP_SELECTED=1` protects every already-resident expert needed
  by the current token before reserving slots for missing experts.
- `LLAMA_MOE_STREAM_GPU_ONLY_GIB=N` removes redundant host copies of frequently
  used GPU experts, up to the per-layer share of N GiB. Before evicting such an
  expert, its original bytes are copied back to a reserved host-cache slot.
  At least six GPU slots per layer keep the ordinary policy. This requires an
  allocated host tier; 0 disables it. It never writes to model files.
- `LLAMA_MOE_STREAM_TRANSFER_PROFILE=1` records aggregate read, transfer and wait
  counters. Worker times overlap and must not be read as elapsed request time.

`KEEP_SELECTED` and `TRANSFER_PROFILE` enable when present: unset them to disable
them. CUDA's existing `GGML_CUDA_GRAPH_OPT=1` can be tested independently.
Neither cache setting changes routing scores, activation precision or sampling.
Changing the GPU cache capacity can change prefill wave grouping and floating
point rounding; compare outputs and distributions for the chosen capacity.

`test-moe-host-cache`, included in `scripts/build.sh streaming`, checks host-slot
leases, failed reads, explicit recycling and concurrent eviction. Real-expert
and full-model measurements are saved with the comparison results.
