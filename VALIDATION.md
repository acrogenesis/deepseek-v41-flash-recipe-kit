# Validation evidence

The runtime and quantizer are experimental. Each saved measurement records its
source pins. Synthetic fixtures below test implementation behavior; they are not
DeepSeek capability benchmarks. Final speed and held-out quantization results
are generated separately in `RESULTS.md`.

| Check | Result | Evidence |
|---|---|---|
| Demand-only cache, tiny MoE | Matching tokens and returned probabilities with buffered/direct I/O and optional RAM cache | [Passing cache test](results/validation/synthetic-cache-demand.json) |
| Wave preloading, tiny MoE | Failed; disabled in this recipe | [Preserved failure](results/validation/synthetic-cache-preload-failure.json) |
| Original expert IDs in calibration | Counts and activation sums matched the uncached GPU path at context 1, 32 and 512 | [Imatrix comparison](results/validation/synthetic-imatrix-streaming.json) |
| Calibration continuation | Eight uninterrupted chunks matched four plus a resumed four | [Continuation check](results/validation/imatrix-resume.json) |
| Unseen-expert weighting | Forced zero-count experts produced the same quantized tensor bytes as explicit unit weights | [Fallback check](results/validation/imatrix-unit-fallback.json) |
| Real expert dimensions in a synthetic wrapper | Cache comparisons passed for Q2/Q3, MXFP4 and IQ2_XXS, including eviction and single-token cases | [Q2/Q3](results/validation/fullshape-eviction-gpu-streaming.json), [MXFP4](results/validation/fullshape-mxfp4-gpu-streaming.json), [IQ2 prefill](results/validation/fullshape-iq2-gpu-streaming.json), [IQ2 decode](results/validation/fullshape-iq2-decode-streaming.json) |
| Native source samples | 42 expert projection payloads matched lossless native MXFP4 packing; 12 Engram rows reproduced native FP8-to-Q8 conversion exactly | [Experts](results/validation/reference-native-layer-samples.json), [Engram](results/validation/reference-native-engram-samples.json) |
| CUDA MMQ tail padding | Both broadcast cases passed the real GPU regression | [Test log](results/validation/mmq-tail-regression.log) |
| Engram calibration crash | GDB identified a null-source read for ARANGE; filtering operation types fixed full-model collection | [Diagnosis and check](results/validation/imatrix-null-source.json) |
| Reference distributions after imatrix fix | Saved eight-chunk logit files were byte-identical | [Hash comparison](results/validation/imatrix-guard-reference-stability.json) |
| Text chat template | Ten message histories matched the official encoder byte for byte through the server's template endpoint; tool requests rejected | [Template check](results/validation/chat-template.json) |
| Embedded chat template | Native metadata, including tokenizer bytes, survived the rewrite; all 23 tiny-fixture tensor payloads stayed identical; ten histories passed using the embedded template | [Metadata and payload checks](results/validation/chat-template-metadata.json), [Embedded rendering](results/validation/chat-template-embedded.json) |
| Baseline OpenAI chat endpoint | Exact `DEEPSEEK_READY` response with a system prompt; prompt token count matched the official encoder | [API check](results/validation/chat-endpoint-baseline.json) |
| Completed custom artifact | All 1,046 tensor layouts and all 926 retained payloads verified; checked embedded template; hashes recorded for ten shards totaling 359.32 GB | [Conversion manifest](results/validation/custom-quantization.json) |
| Paired quantization loss | Custom IQ2: PPL 7.5721, +19.512%; public Q2: PPL 12.6486, +99.636%; paired reference 6.3358 on 2,040 positions | [Custom](results/20261017T205714Z-iq2-vs-mxfp4-q8/quality.json), [Public Q2](results/20261017T205910Z-q2-vs-mxfp4-q8/quality.json) |
| Full custom-model cache comparison | Original gate passed on 510 batched positions: 100% top-choice agreement, KL rounding to zero, PPL ratio 1.000002 against uncached GPU arithmetic | [Custom runtime check](results/validation/custom-runtime.json) |
| Repeated generation | Custom/reference cached and both uncached Q2 runs reproduced all three completions; cached Q2 changed all three. Cause is unresolved | [Token/context audit](results/validation/probe-repeat-consistency.json) |
| Custom 16K request | Completed normally after 360.5s to first token. Included `K7M2R9` with extra words, so the exact-answer check failed | [Preserved result](results/20261017T214241Z-iq2-context16k/context-smoke.json) |
| Custom OpenAI chat endpoint | Passed at 8K context: exact `DEEPSEEK_READY`, normal stop, official embedded template and matching input-token count | [API check](results/validation/chat-endpoint-iq2.json) |

The source samples verify the tested conversions, not an exact native forward
pass. The reference uses Q8 dense/Engram weights and retains native MXFP4 experts
and BF16 embedding/output tensors.
The paired comparison uses the tool's compressed saved log-probabilities; its
reconstructed reference differs slightly from the standalone PPL of 6.3430.
The encoding and KL cutoff are documented in [PROTOCOL.md](PROTOCOL.md).

## Full-model runtime differences

On 508 teacher-forced single-token positions, the CPU-expert and GPU-cache paths
agreed on 91.339% of top choices, with mean KL 0.031168. This failed the original
runtime-equivalence gate and remains recorded. Forcing GPU arithmetic in the
uncached runtime matched the cache on all 508 top choices, with mean KL rounding
to zero. That isolates the observed difference to backend arithmetic on this
sample. It is not a claim that quantization preserves all model capabilities.

See the [backend comparison](results/validation/backend-drift.json) for the
measurements and source artifacts.

## Context

The current graph lacks the full hierarchical candidate mask. The wrapper caps
context at 16,384. A Q2_K retrieval check recovered the exact code at the beginning
of a 16,128-token prompt. That exercises the corrected top-k reuse beyond 512
compressed rows; one retrieval check does not establish broad long-context
accuracy. Its timing is not used as a throughput result because a download was
running concurrently.

[Saved 16K check](results/20261017T155804Z-baseline-context16k/context-smoke.json).

The custom model completed the same 16,128-token request without another recipe
job running. It returned `tacking down the code: K7M2R9.` with a normal EOS, after
360.5 seconds to first token. The correct code was present, but the extra prose
failed the original exact-answer requirement. That failure remains recorded;
the serving default remains 8K. This does not demonstrate general 16K accuracy.
