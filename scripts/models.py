"""Resolve only verified model artifacts used by this recipe."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUSTOM_DIR = ROOT / "work/models/DeepSeek-V4.1-Flash-1017-IQ2_XXS"


def model_info(variant):
    if variant in ("baseline", "reference"):
        quant = "Q2_K" if variant == "baseline" else "Q8_0"
        receipt = ROOT / "work/feasibility" / (
            "download-verified.json" if variant == "baseline" else "download-reference-verified.json")
        if not receipt.exists():
            raise RuntimeError(f"Run scripts/download.py --variant {variant} and finish verification first")
        manifest = json.loads(receipt.read_text())
        directory = ROOT / "work/models/DeepSeek-V4.1-Flash-1017-GGUF" / quant
        model = directory / f"DeepSeek-V4.1-Flash-1017-{quant}-00001-of-00010.gguf"
        alias = "deepseek-v41-flash-1017-" + ("q2k" if variant == "baseline" else "mxfp4-q8")
        for item in manifest["files"]:
            path = directory.parent / item["file"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise RuntimeError(f"Verified shard missing or resized: {path}")
    elif variant == "iq2":
        receipt = CUSTOM_DIR / "quantization.json"
        if not receipt.exists():
            raise RuntimeError("Run scripts/download.py --variant iq2, or finish scripts/campaign.py first")
        manifest = json.loads(receipt.read_text())
        if (manifest["status"] != "complete" or not manifest.get("tensor_layout_verified")
                or not manifest.get("retained_tensor_payloads_verified")
                or not manifest.get("chat_template_verified")):
            raise RuntimeError("Custom quantization has not completed verification")
        for item in manifest["files"]:
            path = CUSTOM_DIR / item["file"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise RuntimeError(f"Verified shard missing or resized: {path}")
        model = CUSTOM_DIR / sorted(x["file"] for x in manifest["files"])[0]
        alias = "deepseek-v41-flash-1017-iq2xxs-q8"
    else:
        raise ValueError(variant)
    return {"variant": variant, "model": str(model), "alias": alias,
            "receipt": str(receipt), "manifest": manifest}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=("baseline", "reference", "iq2"))
    info = model_info(ap.parse_args().variant)
    print(info["model"])
    print(info["alias"])
