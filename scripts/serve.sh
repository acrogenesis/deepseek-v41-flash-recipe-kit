#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model_info=$(python scripts/models.py "${MODEL_VARIANT:-baseline}")
mapfile -t model_fields <<< "$model_info"
model="${model_fields[0]}"
alias="${model_fields[1]}"
# This implementation does not yet implement the full 1M-context candidate mask.
context="${CONTEXT:-8192}"
(( context > 0 && context <= 16384 )) || { echo 'This recipe currently limits context to 16384'; exit 1; }
profile="${PROFILE:-baseline}"
template_args=(--chat-template-file "$PWD/templates/deepseek-v41-chat.jinja")
if [[ "${MODEL_VARIANT:-baseline}" == iq2 ]]; then
    # Our verified artifact embeds this template; exercise its own metadata.
    template_args=()
fi
binary=work/build/bin/llama-server
placement=(-cmoe)
if [[ "$profile" == streaming ]]; then
    binary=work/build-streaming/bin/llama-server
    # Wave preloading changed logits in our synthetic eviction test. Demand-only
    # loading matched the unchanged runtime with both buffered and direct I/O.
    export LLAMA_MOE_STREAM_NO_PRELOAD=1
    placement=(--moe-stream --moe-stream-cache "${GPU_CACHE_GIB:-18}" --moe-stream-l2 "${RAM_CACHE_GIB:-64}" --moe-stream-io-threads "${IO_THREADS:-4}" --moe-stream-direct)
elif [[ "$profile" == gpu-offload ]]; then
    # Unchanged runtime with GPU math also for single-token CPU-resident experts.
    export GGML_OP_OFFLOAD_MIN_BATCH=1
elif [[ "$profile" != baseline ]]; then
    echo 'PROFILE must be baseline, gpu-offload or streaming'; exit 1
fi
exec "$binary" -m "$model" \
    --host 127.0.0.1 --port "${PORT:-5017}" \
    --alias "$alias" \
    -c "$context" -np 1 -b 512 -ub 256 -t "${THREADS:-16}" \
    -ngl 99 "${placement[@]}" -ot 'engram_embd.weight=CPU' \
    -lm mmap -lzm on --no-repack --fit off -fa on \
    --jinja "${template_args[@]}" --reasoning off \
    --no-warmup --no-context-shift --perf "$@"
