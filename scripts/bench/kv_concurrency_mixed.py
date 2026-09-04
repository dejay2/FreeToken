"""D1 phase 3: recreate the batch geometry the crossed reply actually happened in.

L1's bad run is fingerprinted in its transcript: request 0 reported NO cached_tokens
while requests 1-3 reported 8192. So that run was NOT the steady warm state - one request
re-prefilled its whole ~8192-token prompt in the same prefill batch as three 13-token warm
extends (the server log shows exactly this shape: "#new-seq: 4, #new-token: 8167").
A mixed batch is a different GEMM/reduction shape from the all-warm batch, so a
temperature-0 argmax whose top-2 margin is small can land on a different digit.

This driver forces that geometry EVERY run: request 0 gets a fresh prompt nonce (cold,
full prefill), requests 1-3 keep fixed prompts (warm, 13-token extends). Same barrier,
same request shape as L1's driver otherwise.
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


def build_prompt(tokenizer, target_tokens, request_id, nonce=""):
    marker = f"request-{request_id}-marker-{7000 + request_id}"
    head = f"Remember this marker: {marker}.\n"
    filler = (f"Request {request_id} has an ordinary independent context line "
              f"with no other marker{nonce}.\n")
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
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--cold-request", type=int, default=0)
    parser.add_argument("--ready-timeout", type=int, default=900)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_ready(base_url, args.ready_timeout)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    model = model_id(base_url)
    markers = [f"request-{i}-marker-{7000 + i}" for i in range(4)]
    warm = {i: build_prompt(tokenizer, args.prompt_tokens, i) for i in range(4)
            if i != args.cold_request}
    print(f"MIXED_START runs={args.runs} cold_request={args.cold_request} model={model}",
          flush=True)

    mismatches = 0
    for run in range(1, args.runs + 1):
        prompts = dict(warm)
        prompts[args.cold_request] = build_prompt(
            tokenizer, args.prompt_tokens, args.cold_request, f" v{run:04d}"
        )
        gate = threading.Barrier(5)

        def worker(request_id):
            prompt, local_count = prompts[request_id]
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "top_p": 1, "max_tokens": 48,
                "ignore_eos": True, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            gate.wait()
            started = time.perf_counter()
            status, response = http_json(f"{base_url}/v1/chat/completions", body, timeout=3600)
            elapsed = time.perf_counter() - started
            message = response.get("choices", [{}])[0].get("message", {})
            answer = message.get("content") or message.get("reasoning_content") or ""
            return {
                "request_id": request_id, "local_prompt_tokens": local_count,
                "http_status": status, "answer": answer, "elapsed_s": elapsed,
                "usage": response.get("usage", {}), "response_id": response.get("id"),
                "prompt_head": prompt[:80], "prompt_tail": prompt[-80:],
            }

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, i) for i in range(4)]
            gate.wait()
            results = [f.result() for f in futures]
        wall = time.perf_counter() - started
        results.sort(key=lambda item: item["request_id"])

        bad = [(r, [m for m in markers if m in r["answer"]])
               for r in results if markers[r["request_id"]] not in r["answer"]]
        cached = [r["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
                  for r in results]
        comp = [r["usage"].get("completion_tokens") for r in results]
        if bad:
            mismatches += 1
            print(f"MISMATCH run={run}", flush=True)
            for r, other in bad:
                print("  EXPECTED", markers[r["request_id"]], "GOT_MARKERS", other, flush=True)
            for r in results:
                print("  PAIR", json.dumps(r, ensure_ascii=False, sort_keys=True), flush=True)
            # Discriminator: re-issue the offending prompt ALONE against the same warm cache.
            # Still wrong -> the cached prefix itself is poisoned (a real routing/reuse bug).
            # Correct   -> the batched forward diverged; the cache is fine.
            for r, _other in bad:
                rid = r["request_id"]
                prompt, _n = prompts[rid]
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "top_p": 1, "max_tokens": 48,
                    "ignore_eos": True, "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                _st, resp = http_json(f"{base_url}/v1/chat/completions", body, timeout=3600)
                msg = resp.get("choices", [{}])[0].get("message", {})
                solo = msg.get("content") or msg.get("reasoning_content") or ""
                verdict = "CACHE_POISONED" if markers[rid] not in solo else "BATCH_ONLY"
                print(f"  SOLO_RETRY rid={rid} {verdict} "
                      f"answer={json.dumps(solo, ensure_ascii=False)} "
                      f"usage={json.dumps(resp.get('usage', {}), sort_keys=True)}", flush=True)
        print(f"RUN {run}/{args.runs} {'BAD' if bad else 'ok'} wall={wall:.3f}s "
              f"completion={comp} cached={cached}", flush=True)
    print(f"MIXED_DONE runs={args.runs} mismatches={mismatches}", flush=True)


if __name__ == "__main__":
    main()
