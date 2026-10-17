#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:-iq2}"
python scripts/models.py "$variant" >/dev/null
if systemctl --user is-active --quiet deepseek41-server.service; then
    echo 'DeepSeek is already running. Stop it with: systemctl --user stop deepseek41-server'
    exit 0
fi
python - <<'PY'
from pathlib import Path
import socket
import subprocess
gpu = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
if gpu > 3000:
    raise SystemExit(f'GPU busy ({gpu} MiB); stop your other model before starting DeepSeek')
memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
if int(memory['MemAvailable'].split()[0]) < 80 * 1024**2:
    raise SystemExit('Less than 80 GiB RAM available; preserving the existing workload')
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 5017))
PY
systemd-run --user --collect --unit=deepseek41-server \
    -p MemoryHigh=72G -p MemoryMax=80G -p MemorySwapMax=0 \
    --working-directory="$PWD" \
    env PROFILE=streaming MODEL_VARIANT="$variant" \
    GPU_CACHE_GIB=16 RAM_CACHE_GIB=64 CONTEXT="${CONTEXT:-8192}" \
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
