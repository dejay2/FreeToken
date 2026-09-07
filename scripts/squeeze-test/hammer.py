#!/usr/bin/env python3
"""Hammer script for FreeToken memory governor squeeze tests.

Sends continuous short chat completions to the model server (default: 127.0.0.1:2020)
and verifies that requests succeed without errors even during runtime layer moves
and cache rebuilds.

Spec:
  logs per request: t, http_status, wall_s, completion_tokens, tok_s, error
  summary line: requests N, failed M, p50 tok/s, min tok/s, max first-token wait
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def run_one_request(
    endpoint: str,
    payload_bytes: bytes,
    timeout: float,
    stream: bool,
) -> tuple[float, int, float, int, float, float, str]:
    """Execute a single request and measure timings.

    Returns:
      (t_wall_clock, http_status, wall_s, completion_tokens, tok_s, first_token_wait, error)
    """
    req = urllib.request.Request(
        endpoint,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
    )
    t_wall_clock = time.time()
    t_start = time.perf_counter()
    http_status = 0
    completion_tokens = 0
    first_token_wait = 0.0
    error = ""

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_status = resp.status
            t_first = time.perf_counter()
            first_token_wait = t_first - t_start

            if stream:
                # Parse server-sent events
                tokens_counted = 0
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        # Check for usage in chunk
                        if "usage" in chunk and chunk["usage"]:
                            tokens_counted = chunk["usage"].get("completion_tokens", tokens_counted)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                tokens_counted += 1
                    except Exception:
                        pass
                completion_tokens = tokens_counted
            else:
                body = resp.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(body)
                    usage = data.get("usage", {})
                    completion_tokens = int(usage.get("completion_tokens", 0))
                except Exception as exc:
                    error = f"JSON parse error: {exc}"

            t_end = time.perf_counter()
            wall_s = t_end - t_start
            if not (200 <= http_status < 300):
                error = f"HTTP {http_status}"

    except urllib.error.HTTPError as exc:
        t_end = time.perf_counter()
        wall_s = t_end - t_start
        first_token_wait = wall_s
        http_status = exc.code
        error = f"HTTP {exc.code}: {exc.reason}"
    except (TimeoutError, urllib.error.URLError) as exc:
        t_end = time.perf_counter()
        wall_s = t_end - t_start
        first_token_wait = wall_s
        http_status = 0
        error = "timeout" if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower() else str(exc)
    except Exception as exc:
        t_end = time.perf_counter()
        wall_s = t_end - t_start
        first_token_wait = wall_s
        http_status = 0
        error = str(exc)

    tok_s = (completion_tokens / wall_s) if wall_s > 0 else 0.0
    return (t_wall_clock, http_status, wall_s, completion_tokens, tok_s, first_token_wait, error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hammer FreeToken server with continuous chat requests.")
    parser.add_argument("--url", default="http://127.0.0.1:2020", help="Base URL of the FreeToken server (default: http://127.0.0.1:2020)")
    parser.add_argument("--seconds", type=float, default=600.0, help="Duration to run in seconds (default: 600)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel request workers (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=64, help="max_tokens per request (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0)")
    parser.add_argument("--model", default="x", help="Model name (default: x)")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds (default: 180)")
    parser.add_argument("--stream", action="store_true", help="Use streaming SSE chat completions")
    parser.add_argument("--log", default=None, help="Optional CSV file to write logs")
    parser.add_argument("--quiet", action="store_true", help="Do not print individual request lines to stdout")

    args = parser.parse_args()

    endpoint = f"{args.url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Write a short 20-word description of a coastal lighthouse."}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.stream:
        payload["stream"] = True
    payload_bytes = json.dumps(payload).encode("utf-8")

    stop_event = threading.Event()
    results_lock = threading.Lock()
    all_results: list[tuple[float, int, float, int, float, float, str]] = []

    log_file = open(args.log, "w", encoding="utf-8") if args.log else None
    csv_header = "t,http_status,wall_s,completion_tokens,tok_s,error"

    if log_file:
        log_file.write(csv_header + "\n")
        log_file.flush()
    if not args.quiet:
        print(csv_header)

    t_start = time.time()
    t_deadline = t_start + args.seconds

    def worker() -> None:
        while not stop_event.is_set() and time.time() < t_deadline:
            res = run_one_request(
                endpoint=endpoint,
                payload_bytes=payload_bytes,
                timeout=args.timeout,
                stream=args.stream,
            )
            with results_lock:
                all_results.append(res)
                t_w, status, wall, toks, speed, wait, err = res
                line = f"{t_w:.2f},{status},{wall:.3f},{toks},{speed:.1f},{err}"
                if not args.quiet:
                    print(line)
                    sys.stdout.flush()
                if log_file:
                    log_file.write(line + "\n")
                    log_file.flush()

    try:
        if args.parallel <= 1:
            worker()
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = [executor.submit(worker) for _ in range(args.parallel)]
                for f in futures:
                    f.result()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nInterrupted by user.", file=sys.stderr)
    finally:
        stop_event.set()
        if log_file:
            log_file.close()

    total_requests = len(all_results)
    failed_requests = sum(
        1 for r in all_results if r[1] < 200 or r[1] >= 300 or bool(r[6])
    )
    successful = [r for r in all_results if 200 <= r[1] < 300 and not r[6]]

    p50_tok_s = statistics.median([r[4] for r in successful]) if successful else 0.0
    min_tok_s = min([r[4] for r in successful]) if successful else 0.0
    max_first_token_wait = max([r[5] for r in all_results]) if all_results else 0.0

    # Summary line per P1-spec.md:
    # "requests N, failed M, p50 tok/s, min tok/s, max first-token wait"
    summary_line = (
        f"requests {total_requests}, failed {failed_requests}, "
        f"p50 tok/s {p50_tok_s:.1f}, min tok/s {min_tok_s:.1f}, "
        f"max first-token wait {max_first_token_wait:.3f}s"
    )
    print(summary_line)


if __name__ == "__main__":
    main()
