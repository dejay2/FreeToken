"""D1: loop the four-at-once concurrency probe hunting the once-seen crossed reply.

Same prompt construction and request shape as kv_concurrency_live.py (L1's driver), run
N times against a live server. Every reply is checked against the marker its OWN request
asked for; any mismatch dumps the full request/response pair and stops.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import threading
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer


def http_json(url, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def wait_ready(base_url, timeout):
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, body = http_json(f"{base_url}/v1/cache/status", timeout=5)
            last = f"HTTP {status}: {body}"
            if status == 200 and body.get("state") == "serving":
                return body
        except Exception as exc:
            last = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not reach state=serving: {last}")


def model_id(base_url):
    status, body = http_json(f"{base_url}/v1/models", timeout=30)
    if status != 200 or not body.get("data"):
        raise RuntimeError(f"no served model: HTTP {status} {body}")
    return body["data"][0]["id"]


def build_prompt(tokenizer, target_tokens, request_id):
    marker = f"request-{request_id}-marker-{7000 + request_id}"
    head = f"Remember this marker: {marker}.\n"
    filler = f"Request {request_id} has an ordinary independent context line with no other marker.\n"
    tail = f"Reply with exactly this marker and nothing else: {marker}"
    fixed = len(tokenizer.encode(head + tail, add_special_tokens=False))
    unit = max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    copies = max(1, math.ceil((target_tokens - fixed) / unit))
    text = head + filler * copies + tail
    count = len(tokenizer.encode(text, add_special_tokens=False))
    while count < target_tokens:
        copies += 1
        text = head + filler * copies + tail
        count = len(tokenizer.encode(text, add_special_tokens=False))
    return text, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:2020")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--stop-on-mismatch", type=int, default=1)
    parser.add_argument("--label", default="loop")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_ready(base_url, args.ready_timeout)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    model = model_id(base_url)
    prompts = [build_prompt(tokenizer, args.prompt_tokens, i) for i in range(4)]
    markers = [f"request-{i}-marker-{7000 + i}" for i in range(4)]
    print(f"LOOP_START label={args.label} runs={args.runs} model={model} "
          f"prompt_tokens={[p[1] for p in prompts]}", flush=True)

    mismatches = 0
    for run in range(1, args.runs + 1):
        gate = threading.Barrier(5)

        def worker(request_id):
            prompt, local_count = prompts[request_id]
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 48,
                "ignore_eos": True,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            gate.wait()
            started = time.perf_counter()
            status, response = http_json(f"{base_url}/v1/chat/completions", body, timeout=3600)
            elapsed = time.perf_counter() - started
            message = response.get("choices", [{}])[0].get("message", {})
            answer = message.get("content") or message.get("reasoning_content") or ""
            return {
                "request_id": request_id,
                "local_prompt_tokens": local_count,
                "http_status": status,
                "answer": answer,
                "elapsed_s": elapsed,
                "usage": response.get("usage", {}),
                "response_id": response.get("id"),
                "prompt_head": prompt[:80],
                "prompt_tail": prompt[-80:],
            }

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, i) for i in range(4)]
            gate.wait()
            results = [f.result() for f in futures]
        wall = time.perf_counter() - started
        results.sort(key=lambda item: item["request_id"])

        bad = []
        for r in results:
            own = markers[r["request_id"]]
            if own not in r["answer"]:
                other = [m for m in markers if m in r["answer"]]
                bad.append((r, other))
        if bad:
            mismatches += 1
            print(f"MISMATCH run={run}", flush=True)
            for r, other in bad:
                print("  EXPECTED", markers[r["request_id"]], "GOT_MARKERS", other, flush=True)
            for r in results:
                print("  PAIR", json.dumps(r, ensure_ascii=False, sort_keys=True), flush=True)
            if args.stop_on_mismatch:
                print(f"LOOP_STOP_ON_MISMATCH run={run} mismatches={mismatches}", flush=True)
                return
        cached = [r["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
                  for r in results]
        comp = [r["usage"].get("completion_tokens") for r in results]
        print(f"RUN {run}/{args.runs} ok wall={wall:.3f}s completion={comp} cached={cached}",
              flush=True)

    print(f"LOOP_DONE runs={args.runs} mismatches={mismatches}", flush=True)


if __name__ == "__main__":
    main()
