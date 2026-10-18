#!/usr/bin/env python3
"""Save real completions, stream timings and exactly scored functional checks."""
import argparse
import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/deepseek"))
from encoding import encode_messages


def request(url, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def render(url, messages):
    prompt = encode_messages(messages, thinking_mode="chat")
    ids = request(url + "/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
    return {"messages": messages, "rendered_prompt": prompt, "input_ids": ids,
            "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}


def generate(url, ids, cap):
    payload = {"prompt": ids, "n_predict": cap, "temperature": 0, "seed": 1017,
               "cache_prompt": False, "stream": True, "return_tokens": True,
               "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0}
    req = urllib.request.Request(url + "/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    events, output_ids, chunks = [], [], []
    first = last = final = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            elapsed = time.perf_counter() - start
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("stop"):
                final = event
            else:
                tokens = event.get("tokens", [])
                chunks.append(event.get("content", ""))
                output_ids.extend(tokens)
                if tokens:
                    if first is None:
                        first = elapsed
                    last = elapsed
            events.append({"seconds": elapsed, "response": event})
    if final is None:
        raise RuntimeError("The server did not return a terminal stream event")
    timing = final.get("timings", {})
    cached = final.get("n_prompt_tokens_cache", timing.get("cache_n"))
    if cached not in (None, 0):
        raise RuntimeError(f"Unexpected prompt-cache reuse: {cached}")
    return {"text": "".join(chunks), "output_ids": output_ids, "streamed_tokens": len(output_ids),
            "tokens_predicted": timing.get("predicted_n", final.get("tokens_predicted")),
            "ttft_seconds": first, "last_token_seconds": last,
            "decode_tok_s": timing.get("predicted_per_second"),
            "client_decode_tok_s": (len(output_ids) - 1) / (last - first) if len(output_ids) > 1 and last > first else None,
            "stream_note": "Delivered token IDs can omit EOS or bytes withheld by the server's UTF-8 buffering. The primary decode rate uses the server's full generated-token count and decode interval.",
            "total_seconds": time.perf_counter() - start, "server_timings": timing,
            "terminal": final, "events": events}


CHECKS = [
    ("multiply", "Compute 137 * 29. Return only the integer.", "3973"),
    ("remaining", "There are 84 bolts. Six machines each use 9 bolts. How many bolts remain? Return only the integer.", "30"),
    ("percentage", "A price is 240. Apply a 15 percent discount, then add 10 percent tax. Return only the final decimal number.", "224.4"),
    ("extract", "Record: order=K7N4; customer=Mara; status=packed. Return only the order identifier.", "K7N4"),
    ("ordering", "Lena finishes before Omar. Omar finishes before Pia. Who finishes last? Return only the name.", "Pia"),
    ("json", 'Return exactly this JSON object, without Markdown: a key "count" whose value is the number of odd integers in [2, 7, 9, 12, 15].', {"count": 3}),
    ("sort", 'Sort these integers ascending: 14, -3, 8, 0, 14. Return only a JSON array.', [-3, 0, 8, 14, 14]),
    ("python", "What does this Python expression evaluate to: sum(x*x for x in range(1, 6) if x % 2)? Return only the integer.", "35"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5017")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    props = request(args.url + "/props")
    context = props["default_generation_settings"]["n_ctx"]
    if args.fixture:
        fixture = json.loads(args.fixture.read_text())
        assert context == fixture["context"], (context, fixture["context"])
    else:
        fixture = {"context": context, "mode": "chat", "temperature": 0, "output_cap": args.tokens, "checks": [], "speed": []}
        for name, text, expected in CHECKS:
            fixture["checks"].append({"id": name, "expected": expected, **render(args.url, [{"role": "user", "content": text}])})
        topics = ["how database transactions recover after a power failure", "how a compiler detects and optimizes repeated expressions", "how TCP congestion control adapts to changing bandwidth"]
        for i, topic in enumerate(topics):
            text = f"Explain {topic}. Include concrete worked examples and discuss limitations. Write at least 1000 words."
            fixture["speed"].append({"id": f"fresh-{i}", **render(args.url, [{"role": "user", "content": text}])})
    (args.out / "fixture.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
    result = {"label": args.label, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "context": context, "output_cap": fixture["output_cap"],
              "fixture_sha256": hashlib.sha256((args.out / "fixture.json").read_bytes()).hexdigest(),
              "status": "running", "props": props, "checks": [], "fresh": [], "repeat": [],
              "limitation": "Functional smoke checks and short capped timings. Quantization loss against the native release is not measured here."}
    path = args.out / "probe.json"

    def save():
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        temp.replace(path)

    save()
    try:
        for case in fixture["checks"]:
            answer = generate(args.url, case["input_ids"], fixture.get("check_output_cap", 64))
            actual = answer["text"].strip()
            if not isinstance(case["expected"], str):
                try:
                    actual = json.loads(actual)
                except json.JSONDecodeError:
                    pass
            passed = actual == case["expected"]
            if case["id"] in ("multiply", "remaining", "percentage", "python"):
                try:
                    passed = Decimal(actual) == Decimal(case["expected"])
                except InvalidOperation:
                    passed = False
            result["checks"].append({"id": case["id"], "passed": passed, "expected": case["expected"], **answer})
            save()
            print(f"{case['id']}: {'PASS' if passed else 'FAIL'} {answer['text']!r}", flush=True)
        for phase in ("fresh", "repeat"):
            for case in fixture["speed"]:
                answer = generate(args.url, case["input_ids"], fixture["output_cap"])
                result[phase].append({"id": case["id"], **answer})
                save()
                rate = answer["decode_tok_s"]
                if (rate is None or not math.isfinite(rate) or rate <= 0
                        or answer["ttft_seconds"] is None or (answer["tokens_predicted"] or 0) < 2):
                    raise RuntimeError(f"No valid throughput measurement for {phase}/{case['id']}")
                print(f"{phase} {case['id']}: {answer['decode_tok_s']} tok/s, TTFT {answer['ttft_seconds']}s", flush=True)
        result["functional_score"] = {"correct": sum(x["passed"] for x in result["checks"]), "total": len(result["checks"])}
        for phase in ("fresh", "repeat"):
            rates = [x["decode_tok_s"] for x in result[phase] if x["decode_tok_s"] is not None]
            result[phase + "_median_tok_s"] = statistics.median(rates) if rates else None
        result["status"] = "complete_smoke"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
