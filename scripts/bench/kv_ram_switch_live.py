"""Prove two independent 200k-token conversations survive RAM parking.

Run ``record`` with parking off, then ``verify`` with RAM parking and the same
model/settings. Synthetic prompts and canonical replies are stored in --record.
Each request must recall distinct facts at the beginning, middle and end.
Verification requires actual RAM hits after idle GPU eviction, bounded tail
prefill, identical answers to the cold run, and an unchanged server instance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from kv_long_chat_live import http_json, model_id


def status(url):
    return http_json(url + "/v1/cache/status", timeout=15)[1]


def health(url):
    return http_json(url + "/health", timeout=15)[1]


def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)


def prompt(tokenizer, family, target):
    codes = ("maple7319", "copper2048", "violet9631") if family == "A" else (
        "amber5826", "silver1073", "willow6402"
    )
    filler = ("Archive A: ordinary gray stones were counted beside the river.\n"
              if family == "A" else
              "Archive B: ordinary blue crates were stacked near the station.\n")
    begin = f"This is independent archive {family}. Remember its three facts. FIRST={codes[0]}.\n"
    middle = f"\nThe central archive fact is MIDDLE={codes[1]}.\n"
    end = f"\nThe closing archive fact is LAST={codes[2]}.\n"
    question = "Return FIRST, MIDDLE and LAST in that order, separated by |. No other text."
    copies = target // max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    while True:
        text = begin + filler * (copies // 2) + middle + filler * (copies - copies // 2) + end + question
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False,
        )
        if target <= len(ids) <= target + 64:
            return text, codes, len(ids)
        copies += max(1, (target - len(ids)) // 14) if len(ids) < target else -1


def send(url, model, messages):
    started = time.monotonic()
    code, body = http_json(url + "/v1/chat/completions", {
        "model": model, "messages": messages, "temperature": 0,
        "max_tokens": 128, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=1800)
    assert code == 200, body
    choice = body["choices"][0]
    answer = choice["message"].get("content") or ""
    assert choice["finish_reason"] == "stop", choice
    return answer.strip(), body["usage"], time.monotonic() - started


def settle(url, instance, minimum_count):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        current = status(url)
        p = current.get("parking", {})
        assert health(url)["instance_id"] == instance, "server restarted"
        assert not p.get("disabled") and not p.get("last_error"), p
        if p.get("parked_count", 0) >= minimum_count and current.get("active_requests", 0) == 0:
            # Let the scheduler detach the already-saved prompt leaf; hits on the next
            # request independently prove the host path was exercised.
            time.sleep(2)
            return status(url)
        time.sleep(1)
    raise TimeoutError(f"RAM park did not settle: {current}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("record", "verify"))
    parser.add_argument("--base-url", default="http://127.0.0.1:2020")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=200_000)
    args = parser.parse_args()
    url = args.base_url.rstrip("/")
    initial = status(url)
    instance = health(url)["instance_id"]
    required_mode = "off" if args.phase == "record" else "ram"
    assert initial["parking"]["mode"] == required_mode, initial
    model = model_id(url)
    emit({"event": "start", "phase": args.phase, "health": health(url), "cache": initial})
    if args.phase == "record":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        archives = {family: prompt(tokenizer, family, args.tokens) for family in "AB"}
        data = {"model": model, "tokens": args.tokens, "requests": []}
        histories = {family: [{"role": "user", "content": archives[family][0]}] for family in "AB"}
        args.record.parent.mkdir(parents=True, exist_ok=True)
        for turn in range(3):
            for family in "AB":
                codes = archives[family][1]
                order = ((0, 1, 2), (2, 0, 1), (1, 2, 0))[turn]
                if turn:
                    names = ("FIRST", "MIDDLE", "LAST")
                    histories[family].append({"role": "user", "content":
                        "Now return " + ", ".join(names[i] for i in order) +
                        " from this archive, separated by |. No other text."})
                expected = "|".join(codes[i] for i in order)
                request = {"family": family, "turn": turn, "messages": list(histories[family]), "expected": expected}
                answer, usage, elapsed = send(url, model, request["messages"])
                emit({"event": "cold", "family": family, "turn": turn,
                      "answer": answer, "expected": expected, "usage": usage, "seconds": elapsed})
                assert answer == expected, "cold model recall failed"
                assert health(url)["instance_id"] == instance, "server restarted"
                request.update(answer=answer, usage=usage, seconds=elapsed)
                data["requests"].append(request)
                args.record.write_text(json.dumps(data))
                histories[family].append({"role": "assistant", "content": answer})
        emit({"event": "record_pass", "requests": len(data["requests"])})
        return
    data = json.loads(args.record.read_text())
    assert data["model"] == model and len(data["requests"]) == 6, "incomplete or mismatched cold reference"
    # Growing A/B conversations, then shorter historical branches after both grew.
    requests = data["requests"] + data["requests"][:2] + data["requests"][2:4]
    results = []
    for index, request in enumerate(requests):
        before = status(url)["parking"]
        answer, usage, elapsed = send(url, model, request["messages"])
        after = settle(url, instance, min(index + 1, 2))["parking"]
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        result = {"event": "ram", "index": index, "family": request["family"],
                  "turn": request["turn"], "answer": answer, "expected": request["answer"],
                  "usage": usage, "seconds": elapsed, "cold_seconds": request["seconds"],
                  "hits_delta": after["hits"] - before["hits"], "parking": after}
        emit(result)
        results.append(result)
        args.record.with_suffix(".results.json").write_text(json.dumps(results, indent=2))
        assert answer == request["answer"], "RAM restore changed answer"
        assert usage["prompt_tokens"] >= args.tokens, usage
        if index >= 2:
            assert after["hits"] > before["hits"], "no RAM restore hit"
            assert cached >= args.tokens - 64, "old context was reprocessed"
            assert usage["prompt_tokens"] - cached <= 512, "unexpected large prefill"
            transfer = after["last_restore_breakdown_ms"]
            previous = before["last_restore_breakdown_ms"].get("sequence", 0)
            assert transfer["sequence"] > previous, "lookup did not complete a restore"
            assert transfer["page_offset"] == 0, "some old pages stayed on the GPU"
            units = initial["geometry"]["unit_bytes"]
            assert transfer["kv_bytes"] == cached * units["kv_per_token"], "not a full KV reload"
            assert transfer["state_bytes"] == units["mamba_per_slot"], "incomplete recurrent state"
    emit({"event": "verify_pass", "requests": len(results), "health": health(url)})


if __name__ == "__main__":
    main()
