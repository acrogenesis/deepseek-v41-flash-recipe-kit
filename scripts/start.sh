#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:-iq2}"
python scripts/models.py "$variant" >/dev/null
speed_profile="${SPEED_PROFILE:-default}"
gpu_cache=16 ram_cache=64 memory_high=72G memory_max=80G
extra_env=()
case "$speed_profile" in
    default) ;;
    fast)
        [[ "$variant" == iq2 ]] || { echo 'The fast profile was measured with the custom IQ2 model'; exit 1; }
        gpu_cache=18 ram_cache=84 memory_high=92G memory_max=96G
        extra_env=(LLAMA_MOE_STREAM_UPLOAD_BATCH=3 LLAMA_MOE_STREAM_KEEP_SELECTED=1
                   LLAMA_MOE_STREAM_GPU_ONLY_GIB=14 GGML_CUDA_GRAPH_OPT=1)
        ;;
    *) echo 'SPEED_PROFILE must be default or fast'; exit 1 ;;
esac
if systemctl --user is-active --quiet deepseek41-server.service; then
    echo 'DeepSeek is already running. Stop it with: systemctl --user stop deepseek41-server'
    exit 0
fi
python - "$speed_profile" <<'PY'
from pathlib import Path
import socket
import subprocess
import sys
gpu = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
if gpu > 3000:
    raise SystemExit(f'GPU busy ({gpu} MiB); stop your other model before starting DeepSeek')
memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
minimum = 100 if sys.argv[1] == 'fast' else 80
if int(memory['MemAvailable'].split()[0]) < minimum * 1024**2:
    raise SystemExit(f'Less than {minimum} GiB RAM available; preserving the existing workload')
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 5017))
PY
systemd-run --user --collect --unit=deepseek41-server \
    -p MemoryHigh="$memory_high" -p MemoryMax="$memory_max" -p MemorySwapMax=0 \
    --working-directory="$PWD" \
    env PROFILE=streaming MODEL_VARIANT="$variant" \
    GPU_CACHE_GIB="$gpu_cache" RAM_CACHE_GIB="$ram_cache" CONTEXT="${CONTEXT:-8192}" \
    "${extra_env[@]}" \
    bash "$PWD/scripts/serve.sh"
python - <<'PY'
import json
import subprocess
import time
import urllib.request
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'deepseek41-server.service']).returncode:
        raise SystemExit('Server stopped. Inspect: journalctl --user -u deepseek41-server -n 50')
    try:
        with urllib.request.urlopen('http://127.0.0.1:5017/health', timeout=2) as response:
            if json.load(response)['status'] == 'ok':
                print('DeepSeek ready on http://127.0.0.1:5017')
                print("Chat: python scripts/chat.py 'Hello'")
                break
    except OSError:
        pass
    time.sleep(2)
else:
    raise SystemExit('Startup timed out. Inspect: journalctl --user -u deepseek41-server -n 50')
PY
