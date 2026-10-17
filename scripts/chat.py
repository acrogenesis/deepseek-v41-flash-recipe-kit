#!/usr/bin/env python3
"""Text chat using the exact release's official encoder instead of GGUF's template."""
import argparse
from probe import generate, render

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--url", default="http://127.0.0.1:5017")
ap.add_argument("--tokens", type=int, default=512)
args = ap.parse_args()
case = render(args.url, [{"role": "user", "content": args.prompt}])
answer = generate(args.url, case["input_ids"], args.tokens)
print(answer["text"])
