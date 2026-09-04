"""D1 phase 4: does the mixed batch perturb a request's output at all?

Phases 2 and 3 showed the digit markers flip in a mixed prefill batch (one ~8,192-token
re-prefill packed beside three 13-token warm extends) but distinct word passwords do not.
That is consistent with "the mixed batch changes the numbers slightly and a knife-edge greedy
choice lands the other way", but it is not proof: it could also be that the mixed batch
changes nothing and the digit flips come from somewhere else.

So measure the perturbation directly. For each run: take request 1's FULL 47-token answer in
the mixed batch, then re-issue the SAME prompt alone against the SAME warm cache and take its
full answer. Byte-identical => the mixed batch does not perturb this request. Different =>
it does, and the flipped digit is that perturbation landing on a near-tie.

Runs both prompt families: "digit" (L1's, markers differ by one digit) and "word" (distinct
passwords), so the perturbation is measured independently of whether the answer flips.
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

WORDS = ["maple", "cobalt", "quartz", "indigo"]


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


def build_prompt(tokenizer, target_tokens, request_id, family, nonce=""):
    if family == "digit":
        marker = f"request-{request_id}-marker-{7000 + request_id}"
        head = f"Remember this marker: {marker}.\n"
        filler = (f"Request {request_id} has an ordinary independent context line "
                  f"with no other marker{nonce}.\n")
        tail = f"Reply with exactly this marker and nothing else: {marker}"
    else:
        marker = WORDS[request_id]
        head = f"Remember this password: {marker}.\n"
        filler = f"This line is filler and the password is still {marker}{nonce}.\n"
        tail = f"Reply with exactly the password and nothing else: {marker}"
    fixed = len(tokenizer.encode(head + tail, add_special_tokens=False))
    unit = max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    copies = max(1, math.ceil((target_tokens - fixed) / unit))
    text = head + filler * copies + tail
    count = len(tokenizer.encode(text, add_special_tokens=False))
    while count < target_tokens:
        copies += 1
        text = head + filler * copies + tail
        count = len(tokenizer.encode(text, add_special_tokens=False))
    return text, marker


def ask(base_url, model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "max_tokens": 48,
        "ignore_eos": True, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    _st, resp = http_json(f"{base_url}/v1/chat/completions", body, timeout=3600)
    msg = resp.get("choices", [{}])[0].get("message", {})
    return msg.get("content") or msg.get("reasoning_content") or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:2020")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--family", choices=["digit", "word"], default="digit")
    parser.add_argument("--probe-request", type=int, default=1)
    parser.add_argument("--cold-request", type=int, default=0)
    parser.add_argument("--ready-timeout", type=int, default=900)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_ready(base_url, args.ready_timeout)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    model = model_id(base_url)
    warm = {i: build_prompt(tokenizer, args.prompt_tokens, i, args.family)
            for i in range(4) if i != args.cold_request}
    print(f"PERTURB_START family={args.family} runs={args.runs} "
          f"probe_request={args.probe_request} cold_request={args.cold_request}", flush=True)

    differed = 0
    flipped = 0
    for run in range(1, args.runs + 1):
        prompts = dict(warm)
        prompts[args.cold_request] = build_prompt(
            tokenizer, args.prompt_tokens, args.cold_request, args.family, f" v{run:04d}"
        )
        gate = threading.Barrier(5)

        def worker(request_id):
            gate.wait()
            return request_id, ask(base_url, model, prompts[request_id][0])

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, i) for i in range(4)]
            gate.wait()
            batched = dict(f.result() for f in futures)

        rid = args.probe_request
        prompt, marker = prompts[rid]
        solo = ask(base_url, model, prompt)
        same = batched[rid] == solo
        if not same:
            differed += 1
        batched_ok = batched[rid].startswith(marker)
        solo_ok = solo.startswith(marker)
        if batched_ok != solo_ok:
            flipped += 1
        print(f"RUN {run}/{args.runs} identical={same} batched_marker_ok={batched_ok} "
              f"solo_marker_ok={solo_ok}", flush=True)
        if not same:
            print("   BATCHED", json.dumps(batched[rid], ensure_ascii=False), flush=True)
            print("   SOLO   ", json.dumps(solo, ensure_ascii=False), flush=True)
    print(f"PERTURB_DONE family={args.family} runs={args.runs} "
          f"batched_differs_from_solo={differed} marker_verdict_flipped={flipped}", flush=True)


if __name__ == "__main__":
    main()
