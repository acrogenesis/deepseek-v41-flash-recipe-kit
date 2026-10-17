#!/usr/bin/env python3
"""Check a running recipe server through its OpenAI-compatible chat endpoint."""
import argparse
import datetime
import json
from pathlib import Path
import time

from models import ROOT, model_info
from probe import encode_messages, request

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("baseline", "reference", "iq2"), required=True)
parser.add_argument("--url", default="http://127.0.0.1:5017")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
model = model_info(args.variant)
messages = [{"role": "system", "content": "For this connection test, reply with exactly DEEPSEEK_READY and nothing else."},
            {"role": "user", "content": "Run the connection test."}]
report = {"status": "running", "purpose": "Real OpenAI chat completion with system and user roles; not a capability benchmark",
          "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "variant": args.variant, "model": model["alias"], "url": args.url,
          "pins": json.loads((ROOT / "pins.json").read_text()), "messages": messages}
try:
    report["health"] = request(args.url + "/health")
    assert report["health"]["status"] == "ok"
    report["models"] = request(args.url + "/v1/models")
    assert model["alias"] in [row["id"] for row in report["models"]["data"]]
    report["configured_context"] = request(args.url + "/props")["default_generation_settings"]["n_ctx"]
    rendered = request(args.url + "/apply-template", {"messages": messages, "add_generation_prompt": True})
    report["template_matches_official"] = rendered["prompt"] == encode_messages(messages, thinking_mode="chat")
    assert report["template_matches_official"]
    report["expected_input_ids"] = request(args.url + "/tokenize", {
        "content": rendered["prompt"], "add_special": False, "parse_special": True,
    })["tokens"]
    started = time.monotonic()
    report["response"] = request(args.url + "/v1/chat/completions", {
        "model": model["alias"], "messages": messages, "max_tokens": 32,
        "temperature": 0, "seed": 1017, "stream": False, "cache_prompt": False,
    })
    report["seconds"] = time.monotonic() - started
    report["prompt_token_count_matches"] = report["response"]["usage"]["prompt_tokens"] == len(report["expected_input_ids"])
    assert report["prompt_token_count_matches"], "Chat endpoint input count differs from the official encoding"
    choice = report["response"]["choices"][0]
    report["answer"] = choice["message"]["content"]
    report["exact_answer"] = report["answer"].strip() == "DEEPSEEK_READY"
    assert report["exact_answer"] and choice["finish_reason"] == "stop"
    report["status"] = "complete"
except Exception as error:
    report.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: report[key] for key in ("status", "model", "answer", "seconds", "template_matches_official")}))
