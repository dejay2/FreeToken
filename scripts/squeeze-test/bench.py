#!/usr/bin/env python3
"""Benchmark script for FreeToken memory governor evaluation.

Performs three 200-token completions at temperature 0 against the model server,
reporting per-run usage and the median decode tok/s and wall time.

Spec:
  three 200-token completions at temperature 0 (the prompt in the recipe above),
  median decode tok/s and wall time; --label prefix for logs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3x 200-token completion benchmark against FreeToken.")
    parser.add_argument("--url", default="http://127.0.0.1:2020", help="Base URL of the FreeToken server (default: http://127.0.0.1:2020)")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=200, help="max_tokens per run (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0)")
    parser.add_argument("--model", default="x", help="Model name (default: x)")
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds (default: 180)")
    parser.add_argument("--label", default="", help="Optional label prefix for log output (e.g. baseline or post-release)")

    args = parser.parse_args()

    endpoint = f"{args.url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Write a 300 word story about a lighthouse keeper."}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    payload_bytes = json.dumps(body).encode("utf-8")

    prefix = f"[{args.label}] " if args.label else ""
    tok_s_list: list[float] = []
    wall_list: list[float] = []

    for i in range(1, args.runs + 1):
        req = urllib.request.Request(
            endpoint,
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
        )
        t_start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                data = json.load(resp)
            dt = time.time() - t_start
            usage = data.get("usage", {})
            completion_tokens = int(usage.get("completion_tokens", args.max_tokens))
            tok_s = completion_tokens / dt if dt > 0 else 0.0

            wall_list.append(dt)
            tok_s_list.append(tok_s)
            print(f"{prefix}run {i}/{args.runs}: usage {usage} wall {dt:.2f}s decode tok/s {tok_s:.1f}")
            sys.stdout.flush()
        except Exception as exc:
            dt = time.time() - t_start
            print(f"{prefix}run {i}/{args.runs} FAILED after {dt:.2f}s: {exc}", file=sys.stderr)
            sys.exit(1)

    median_tok_s = statistics.median(tok_s_list) if tok_s_list else 0.0
    median_wall = statistics.median(wall_list) if wall_list else 0.0

    summary_line = f"{prefix}median decode tok/s {median_tok_s:.1f}, median wall {median_wall:.2f}s"
    print(summary_line)


if __name__ == "__main__":
    main()
