#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-baseline}"
[[ "$profile" == baseline || "$profile" == streaming ]] || { echo 'Usage: build.sh [baseline|streaming]'; exit 1; }
source_dir="$PWD/work/upstream/llama.cpp"
build_dir=work/build
tests=ON
targets=(llama-server llama-cli llama-bench llama-perplexity llama-quantize llama-imatrix llama-tokenize)
targets+=(test-backend-ops)
if [[ "$profile" == streaming ]]; then
    source_dir="$PWD/work/upstream/llama-streaming"
    build_dir=work/build-streaming
    tests=ON
    targets+=(test-llama-file)
fi
revision=2076412598e99127ea2554467b2b7530856895d3
if [[ ! -e "$source_dir/.git" ]]; then
    git init "$source_dir"
    git -C "$source_dir" remote add origin https://github.com/vcruz305/llama.cpp
    git -C "$source_dir" fetch --depth 1 origin "$revision"
    git -C "$source_dir" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]]
if [[ "$profile" == streaming ]]; then
    if git -C "$source_dir" diff HEAD --quiet; then
        git -C "$source_dir" apply --index "$PWD/patches/moe-stream.patch"
    fi
    git -C "$source_dir" diff HEAD --binary | cmp - patches/moe-stream.patch
else
    if git -C "$source_dir" diff HEAD --quiet; then
        git -C "$source_dir" apply --index "$PWD/patches/runtime-fixes.patch"
    fi
    git -C "$source_dir" diff HEAD --binary | cmp - patches/runtime-fixes.patch
fi
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120a-real \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-14 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
    -DLLAMA_BUILD_TESTS="$tests" -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_TOOLS_INSTALL=OFF -DLLAMA_BUILD_UI=OFF
cmake --build "$build_dir" --target "${targets[@]}" -j "${BUILD_JOBS:-8}"
"$build_dir/bin/llama-server" --version
if [[ "$profile" == streaming ]]; then
    "$build_dir/bin/test-llama-file"
fi
