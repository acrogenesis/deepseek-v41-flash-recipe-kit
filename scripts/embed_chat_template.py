#!/usr/bin/env python3
"""Embed the checked chat template while verifying every copied tensor byte."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import gguf

ROOT = Path(__file__).resolve().parents[1]


def payload_digest(path, tensor):
    remaining = int(tensor.n_bytes)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(int(tensor.data_offset))
        while remaining:
            data = stream.read(min(remaining, 32 * 1024**2))
            if not data:
                raise RuntimeError(f"Truncated tensor: {path}: {tensor.name}")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def embed(model, template):
    model, template = Path(model), Path(template)
    temporary = model.with_suffix(".chat-template.tmp")
    if temporary.exists():
        raise RuntimeError(f"Refusing to overwrite metadata scratch: {temporary}")
    if shutil.disk_usage(model.parent).free < model.stat().st_size + 50 * 1024**3:
        raise RuntimeError("Insufficient space to copy the first shard while retaining a 50 GiB reserve")
    command = [sys.executable, "-m", "gguf.scripts.gguf_new_metadata", str(model), str(temporary),
               "--chat-template-file", str(template)]
    with model.with_suffix(".chat-template.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    original, updated = gguf.GGUFReader(model), gguf.GGUFReader(temporary)
    assert updated.fields["tokenizer.chat_template"].contents() == template.read_text()
    # Compare encoded fields, including all tokenizer bytes, without relying on
    # a potentially lossy decode/re-encode comparison of unusual token strings.
    def metadata(reader):
        return {key: b"".join(part.tobytes() for part in field.parts)
                for key, field in reader.fields.items()
                if not key.startswith(("GGUF.", "tokenizer.chat_template"))}
    assert metadata(original) == metadata(updated), "Unrelated GGUF metadata changed"
    before, after = [{tensor.name: tensor for tensor in reader.tensors} for reader in (original, updated)]
    assert before.keys() == after.keys()
    checks = []
    for name, tensor in before.items():
        other = after[name]
        assert tensor.tensor_type == other.tensor_type and tensor.shape.tolist() == other.shape.tolist()
        assert tensor.n_bytes == other.n_bytes
        source_hash, output_hash = payload_digest(model, tensor), payload_digest(temporary, other)
        assert source_hash == output_hash, f"Metadata rewrite changed tensor payload: {name}"
        checks.append({"tensor": name, "bytes": int(tensor.n_bytes), "sha256": source_hash})
    report = {"command": command, "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
              "metadata_preserved": True, "tensor_payloads_preserved": True, "tensors": checks}
    temporary.replace(model)
    return report


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="Owned output shard to replace after verification")
    parser.add_argument("--template", type=Path, default=ROOT / "templates/deepseek-v41-chat.jinja")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = embed(args.model, args.template)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Embedded template; preserved {len(result['tensors'])} tensor payloads and all other metadata")
