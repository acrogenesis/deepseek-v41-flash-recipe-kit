#!/usr/bin/env python3
"""Check actual server template rendering against DeepSeek's official encoder.

Uses a tiny CPU-only model solely to initialize the HTTP template endpoint.
No generations or capability measurements are made with that fixture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/deepseek"))
from encoding import encode_messages

ap = argparse.ArgumentParser()
ap.add_argument("--tiny-model", type=Path, required=True)
ap.add_argument("--embedded", action="store_true", help="Use the model's embedded template without a file override")
args = ap.parse_args()
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 5018))
template = ROOT / "templates/deepseek-v41-chat.jinja"
label = "chat-template-embedded" if args.embedded else "chat-template"
out = ROOT / f"results/validation/{label}.json"
log = ROOT / f"work/logs/{label}.log"


def request(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:5018"+path, data=body,
                                headers={"Content-Type": "application/json"}), timeout=10) as response:
        return json.load(response)


def user(text): return {"role": "user", "content": text}
def system(text): return {"role": "system", "content": text}
def assistant(text, **kw): return {"role": "assistant", "content": text, **kw}

cases = [
    [user("Hello")],
    [system("Be concise."), user("Hello")],
    [system("Be concise."), user("Hello"), assistant("Hi."), user("What is 2+2?")],
    [user("First"), user("Second")],
    [system("First"), system("Second"), user("Third")],
    [user("First"), assistant("Reply"), system("New instruction"), user("Continue")],
    [user("First"), assistant("Reply"), system("New instruction")],
    [user("First"), assistant("Reply", reasoning_content="Old reasoning"), user("Continue")],
    [user("")],
    [system("Español / 中文"), user("  café\n\n你好\t  ")],
]
command = [str(ROOT / "work/build/bin/llama-server"), "-m", str(args.tiny_model.resolve()),
           "--host", "127.0.0.1", "--port", "5018", "-ngl", "0", "-c", "512", "-np", "1",
           "-t", "2", "--no-warmup", "--jinja",
           "--reasoning", "off"]
if not args.embedded:
    command += ["--chat-template-file", str(template)]
report = {"status": "running", "purpose": "Template rendering only; tiny CPU fixture does not measure DeepSeek quality",
          "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(), "checks": []}
with log.open("w") as stream:
    process = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                               stdout=stream, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic()+90
        while True:
            if process.poll() is not None:
                raise RuntimeError("Tiny template server stopped; inspect its log")
            try:
                if request("/health")["status"] == "ok": break
            except OSError:
                pass
            if time.monotonic() > deadline: raise RuntimeError("Tiny template server did not start")
            time.sleep(.2)
        for messages in cases:
            expected = encode_messages(messages, thinking_mode="chat")
            actual = request("/apply-template", {"messages": messages})["prompt"]
            report["checks"].append({"messages": messages, "expected": expected, "actual": actual,
                                     "equal": actual == expected})
        for name, extra in [("tools", {"tools": [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object"}}}]})]:
            try:
                request("/apply-template", {"messages": [user("Hello")], **extra})
                rejected = False
            except urllib.error.HTTPError as error:
                rejected = error.code == 400
            report[name+"_rejected"] = rejected
        report["status"] = "complete" if all(c["equal"] for c in report["checks"]) and report["tools_rejected"] else "failed"
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
print(json.dumps({k:v for k,v in report.items() if k != "checks"}, indent=2))
for i, check in enumerate(report["checks"]):
    print(i, "PASS" if check["equal"] else "FAIL")
if report["status"] != "complete": raise SystemExit(1)
