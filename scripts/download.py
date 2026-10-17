#!/usr/bin/env python3
"""Download and hash-check published custom, baseline or reference weights."""
import argparse
import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("baseline", "reference", "iq2"), default="baseline")
parser.add_argument("--metadata-only", action="store_true",
                    help="Check remote sizes and hashes without downloading or changing files")
parser.add_argument("--hf-cli", default=os.environ.get("HF_CLI", "hf"),
                    help="Hugging Face CLI executable (or set HF_CLI)")
args = parser.parse_args()
custom_manifest = None
destination = ROOT / "work/models/DeepSeek-V4.1-Flash-1017-GGUF"
if args.variant == "iq2":
    pin = json.loads((ROOT / "huggingface/release.json").read_text())
    custom_manifest = json.loads((ROOT / "results/validation/custom-quantization.json").read_text())
    destination = ROOT / "work/models/DeepSeek-V4.1-Flash-1017-IQ2_XXS"
else:
    pin = json.loads((ROOT / "pins.json").read_text())[args.variant]
report_dir = ROOT / "work/feasibility"
url = f"https://huggingface.co/api/models/{pin['repo']}/revision/{pin['revision']}?blobs=true"
with urllib.request.urlopen(url, timeout=60) as response:
    metadata = json.load(response)
if pin["revision"] != "main":
    assert metadata["sha"] == pin["revision"]
revision = metadata["sha"]
files = [f for f in metadata["siblings"] if fnmatch.fnmatch(f["rfilename"], pin["include"])]
if len(files) != 10:
    raise SystemExit(f"Expected ten complete shards, found {len(files)}; the publication may still be uploading")
assert sum(f["size"] for f in files) == pin["bytes"]
if custom_manifest is not None:
    expected = {f["file"]: f for f in custom_manifest["files"]}
    assert {f["rfilename"] for f in files} == set(expected)
    for item in files:
        original = expected[item["rfilename"]]
        assert item["size"] == original["bytes"], item["rfilename"]
        assert item["lfs"]["sha256"] == original["sha256"], item["rfilename"]
if args.metadata_only:
    print(f"Verified remote manifest: {pin['repo']} @ {revision}; ten shards, {pin['bytes']:,} bytes")
    raise SystemExit(0)
destination.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / "gguf-info.json").write_text(json.dumps(metadata, indent=2) + "\n")
remaining = sum(f["size"] for f in files if not (destination / f["rfilename"]).is_file()
                or (destination / f["rfilename"]).stat().st_size != f["size"])
if shutil.disk_usage(destination).free < remaining + 50 * 1024**3:
    raise SystemExit("Insufficient disk space: retain at least 50 GiB beyond this download")
print(f"Downloading {pin['repo']} @ {revision} ({pin['bytes']/1e9:.1f} GB)", flush=True)
environment = os.environ.copy()
environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
environment["HF_XET_HIGH_PERFORMANCE"] = "0"
subprocess.run([
    args.hf_cli, "download", pin["repo"], "--revision", revision,
    "--include", pin["include"], "--local-dir", str(destination), "--max-workers", "2",
], env=environment, check=True)
verified = []
for item in files:
    path = destination / item["rfilename"]
    assert path.stat().st_size == item["size"], path
    print(f"Verifying {path.name}", flush=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == item["lfs"]["sha256"], f"SHA256 mismatch: {path}"
    verified.append({"file": item["rfilename"], "bytes": item["size"], "sha256": digest})
report = {"repo": pin["repo"], "revision": revision, "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "files": verified}
receipt = {"baseline": "download-verified.json", "reference": "download-reference-verified.json",
           "iq2": "download-iq2-verified.json"}[args.variant]
(report_dir / receipt).write_text(json.dumps(report, indent=2) + "\n")
if custom_manifest is not None:
    # Only install the launch receipt after every local shard passes SHA256.
    temporary = destination / "quantization.json.tmp"
    temporary.write_text(json.dumps(custom_manifest, indent=2) + "\n")
    temporary.replace(destination / "quantization.json")
print(f"All {args.variant} shards verified", flush=True)
