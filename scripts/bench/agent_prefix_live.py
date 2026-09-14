"""Compare unregistered cold, registered cold fanout and warm local agent prefixes.

Run against a validation server with --enable-cache-report. This driver never rebuilds
or clears its caches, changes retention settings, or restarts the server. Each cold
trial has an early random nonce; only aliases created by this run are deleted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid


def http(base, path, payload=None, *, method=None, timeout=1800):
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("FREETOKEN_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base + path,
        data=None if payload is None else json.dumps(payload).encode(), headers=headers,
        method=method or ("GET" if payload is None else "POST"))
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(base, path, payload=None, **kwargs):
    try:
        with http(base, path, payload, **kwargs) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{path}: HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def encode(manager, request):
    from freetoken.message import TokenizeMsg
    from freetoken.server.openai_api import chat_request_to_genspec
    from freetoken.server.api_models import ChatCompletionRequest

    spec = chat_request_to_genspec(ChatCompletionRequest.model_validate(request), {})
    return manager.tokenize([TokenizeMsg(uid=0, text=spec.messages,
        sampling_params=spec.sampling_params, tools=spec.template_tools,
        chat_template_kwargs=spec.chat_template_kwargs,
        preserve_system_order=spec.preserve_system_order)])[0]


def common_length(a, b):
    unequal = (a[:min(len(a), len(b))] != b[:min(len(a), len(b))]).nonzero()
    return int(unequal[0, 0]) if len(unequal) else min(len(a), len(b))


def build_trial(manager, model, target_tokens, output_tokens):
    nonce = uuid.uuid4().hex
    copies = max(1, target_tokens // 12)
    while True:
        shared = (f"Local agent trial {nonce}.\n"
                  "Follow the task at the end; answer with its marker only.\n" +
                  "Shared manual: inspect the workspace, keep records, verify the result.\n" * copies)
        request = {"model": model, "messages": [
            {"role": "system", "content": shared},
            {"role": "user", "content": "Return exactly alpha-marker."}],
            "temperature": 0, "top_p": 1, "max_tokens": output_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True, "stream_options": {"include_usage": True}}
        alternate = copy.deepcopy(request)
        alternate["messages"][-1]["content"] = "Different private task: beta-marker."
        tokens = encode(manager, request)
        common = common_length(tokens, encode(manager, alternate))
        if common >= target_tokens:
            return request, tokens[:target_tokens], common
        copies *= 2


def fanout(base, seed, count, tag):
    gate = threading.Barrier(count)

    def one(index):
        request = copy.deepcopy(seed)
        marker = f"{tag}-{index}-marker"
        request["messages"][-1]["content"] = f"Return exactly {marker}."
        gate.wait(timeout=30)
        started, ttft, usage, text = time.perf_counter(), None, None, ""
        with http(base, "/v1/chat/completions", request) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                event = json.loads(payload)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or delta.get("reasoning_content") or ""
                    if (content or delta.get("tool_calls")) and ttft is None:
                        ttft = time.perf_counter() - started
                    text += delta.get("content") or ""
        if usage is None:
            raise RuntimeError("Server did not return streaming usage")
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        return {"agent": index, "ttft_s": ttft, "elapsed_s": time.perf_counter() - started,
                "usage": usage, "cached_tokens": cached,
                "reported_uncached_prompt_tokens": usage["prompt_tokens"] - cached,
                "marker": marker, "answer": text, "marker_matched": text.strip() == marker}

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(one, range(count)))
    return {"agents": count, "phase": tag, "elapsed_s": time.perf_counter() - start,
            "requests": results, "uncached_prompt_tokens": sum(r["reported_uncached_prompt_tokens"] for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True, help="The same tokenizer revision as the running model")
    parser.add_argument("--tokens", type=int, default=25000)
    parser.add_argument("--agents", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("agent-prefix-results.json"))
    args = parser.parse_args()
    if args.tokens <= 0 or args.output_tokens <= 0 or any(n < 1 or n > 64 for n in args.agents):
        parser.error("positive token budgets and 1–64 agents are required")
    from freetoken.tokenizer.tokenize import TokenizeManager
    from freetoken.utils import load_tokenizer

    base = args.base_url.rstrip("/")
    health = get_json(base, "/health")
    if not health.get("instance_id"):
        raise RuntimeError("Server health response lacks an instance ID")
    model = get_json(base, "/v1/models")["data"][0]["id"]
    manager = TokenizeManager(load_tokenizer(args.model_path))
    report = {"complete": False, "model": model, "health": health,
              "initial_cache_status": get_json(base, "/v1/cache/status"), "rounds": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        if get_json(base, "/health").get("instance_id") != health.get("instance_id"):
            raise RuntimeError("Server restarted during the benchmark")
        row["valid_output"] = all(r["marker_matched"] and r["ttft_s"] is not None for r in row["requests"])
        report["rounds"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
        if not row["valid_output"]:
            raise RuntimeError("A request failed its answer-marker/TTFT check; report remains incomplete")

    for count in args.agents:
        seed, _, _ = build_trial(manager, model, args.tokens, args.output_tokens)
        record(fanout(base, seed, count, "baseline"))
        seed, selected, common = build_trial(manager, model, args.tokens, args.output_tokens)
        name = "bench-" + uuid.uuid4().hex
        created = False
        try:
            result = get_json(base, "/v1/cache/prefixes", {"name": name, "format": "openai",
                "request": seed, "prefix_tokens": len(selected), "ttl_seconds": 300})
            created = True
            prefix = result.get("result", result)
            length = prefix["prefix_tokens"]
            expected = hashlib.sha256(selected[:length].numpy().tobytes()).hexdigest()
            if prefix["id"] != expected:
                raise RuntimeError("Local and server rendered prefix IDs differ; check tokenizer/template settings")
            for phase in ("coalesced", "warm"):
                before_result = get_json(base, f"/v1/cache/prefixes/{name}")
                before = before_result.get("result", before_result)
                row = fanout(base, seed, count, phase)
                after_result = get_json(base, f"/v1/cache/prefixes/{name}")
                after = after_result.get("result", after_result)
                prepared = after["forwarded_tokens"] - before["forwarded_tokens"]
                row.update(prefix_tokens=length, common_tokens=common, prefix=after,
                           prepared_tokens=prepared,
                           total_prefill_tokens=prepared + row["uncached_prompt_tokens"],
                           residency=get_json(base, "/v1/cache/prefixes"))
                record(row)
                if any(r["cached_tokens"] < length for r in row["requests"]):
                    raise RuntimeError("Prefix was not reused by every agent; enable --enable-cache-report and inspect pool pressure")
                if phase == "coalesced" and after["preparations"] != 1:
                    raise RuntimeError("Cold fanout did not use exactly one preparation")
                if phase == "warm" and prepared:
                    raise RuntimeError("Warm prefix was recomputed (check retention and admission pressure)")
        finally:
            if created:
                get_json(base, f"/v1/cache/prefixes/{name}", method="DELETE")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
