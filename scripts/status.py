#!/usr/bin/env python3
"""Show the current local campaign state without changing any job."""
import json
import datetime
from pathlib import Path
import re
import subprocess
import struct

from models import ROOT


def active(unit):
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


pin = json.loads((ROOT / "pins.json").read_text())["reference"]
hub = ROOT / "work/models/DeepSeek-V4.1-Flash-1017-GGUF"
complete = sum(p.stat().st_size for p in (hub / "Q8_0").glob("*.gguf"))
partial = sum(p.stat().st_blocks * 512 for p in (hub / ".cache/huggingface/download/Q8_0").glob("*.incomplete"))
verified = (ROOT / "work/feasibility/download-reference-verified.json").exists()
download_state = "verified" if verified else ("running" if active("deepseek41-reference-download.service") else "stopped")
download_log = ROOT / "work/logs/reference-download.log"
if download_state == "running" and download_log.exists():
    checks = re.findall(r"Verifying .*-(\d+)-of-(\d+)\.gguf", download_log.read_text())
    if checks:
        current, total = checks[-1]
        download_state = f"verifying shard {int(current)}/{int(total)}"
print(f"Reference: {min(complete+partial, pin['bytes'])/1e9:.1f}/{pin['bytes']/1e9:.1f} GB, {download_state}")

for variant, label, unit in (("baseline", "Q2_K", "deepseek41-context16k.service"),
                             ("iq2", "Custom IQ2", "deepseek41-custom-context16k.service")):
    contexts = sorted((ROOT / "results").glob(f"*-{variant}-context16k/context-smoke.json"))
    if not contexts:
        continue
    p = contexts[-1]
    r = json.loads(p.read_text())
    progress = re.findall(r"prompt processing, n_tokens =\s*(\d+)", p.with_name("server.log").read_text())
    if r["status"] == "complete":
        print(f"{label} 16K retrieval: {'passed' if r['retrieval_passed'] else 'failed'}")
    elif r["status"] == "failed":
        print(f"{label} 16K retrieval: failed, {r.get('error', 'inspect the result')}")
    else:
        state = "running" if active(unit) else "stopped"
        print(f"{label} 16K request: {progress[-1] if progress else 0}/{r['target_prompt_tokens']} prompt tokens, {state}")

runtime_check = ROOT / "results/validation/custom-runtime.json"
if runtime_check.exists():
    check = json.loads(runtime_check.read_text())
    passed = check["status"] == "complete" and check["results"][-1]["measurement"].get("runtime_gate") is True
    print(f"Custom cache check: {'passed' if passed else check['status']}")

path = ROOT / "work/campaign.json"
if path.exists():
    state = json.loads(path.read_text())
    running = active("deepseek41-conversion-campaign.service")
    stage = state["stages"][-1]["name"] if state["stages"] else state.get("waiting_for", "not started")
    print(f"Campaign: {state['status']} ({'service active' if running else 'service stopped'}), {stage}")
    if state["stages"]:
        current = state["stages"][-1]
        if current["status"] == "running":
            elapsed = datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(current["started_at"])
            print(f"Stage elapsed: {int(elapsed.total_seconds())//60}m {int(elapsed.total_seconds())%60}s")
            if current["name"].startswith("calibration-"):
                command = current["command"]
                matrix = Path(command[command.index("--out")+1])
                if matrix.exists():
                    # Read only the small metadata header. Intermediate snapshots
                    # can contain partial current-chunk statistics; they are not
                    # a completed calibration artifact.
                    with matrix.open("rb") as stream:
                        head = stream.read(65536)
                    key = b"imatrix.chunk_count" + struct.pack("<I", 4)
                    index = head.find(key)
                    if head[:4] == b"GGUF" and index >= 0 and index+len(key)+4 <= len(head):
                        chunks, = struct.unpack_from("<I", head, index+len(key))
                        print(f"Last imatrix snapshot: chunk {chunks} (partial; final validation pending)")
    if state.get("error"):
        print(f"Error: {state['error']}")
quant_log = ROOT / "work/models/DeepSeek-V4.1-Flash-1017-IQ2_XXS/quantize.log"
if quant_log.exists():
    quant_text = quant_log.read_text(errors="replace")
    rows = re.findall(r"\[\s*(\d+)/\s*(\d+)\]\s+(\S+)", quant_text)
    if rows:
        current, total, tensor = rows[-1]
        print(f"Quantizer: tensor {current}/{total}, {tensor}")
    converted = len(re.findall(r"size =\s*[\d.]+ MiB ->\s*[\d.]+ MiB", quant_text))
    print(f"Expert tensors converted: {converted}/120")
    manifest = quant_log.with_name("quantization.json")
    if manifest.exists():
        quant = json.loads(manifest.read_text())
        print(f"Quant artifact: {quant['status']}, {quant.get('phase', 'unknown phase')}")
        if quant.get("phase") == "output shard SHA256 verification":
            print(f"Output shards hashed: {len(quant.get('files', []))}/10")
print("Details: STATUS.md and work/logs/")
