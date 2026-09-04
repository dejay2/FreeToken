"""Two-turn live chat with an idle pause between turns, for L1 steps 6 and 7.

Turn 1 sends a long prompt and takes a 48-token temperature-0 answer. The script then sits
idle for longer than the server's --kv-park-idle-ms so the scheduler can park the finished
prefix. Turn 2 continues the same conversation, which must reuse that prefix. The parking
block of /v1/cache/status is sampled before turn 1, after the pause, and after turn 2, so a
park and a restore are visible as counters rather than inferred from timing.

Run it once against a parking-off server to record the baseline second-turn answer, then
against each parking mode with --identity-mode verify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from transformers import AutoTokenizer

import kv_long_chat_live as base


def build_turn_one(tokenizer, target_tokens: int) -> tuple[str, int]:
    # Deliberately mundane wording. An earlier draft framed these as BEGIN-CODE/END-CODE to
    # be "acknowledged", and the model read that as a prompt-injection test and refused, which
    # made the two turns useless as an identity check.
    begin = "First item on the packing list: maple syrup.\n"
    filler = "Middle item: one ordinary grey pebble was counted and nothing else changed.\n"
    end = (
        "Last item on the packing list: cobalt paint. That is the whole list. "
        "Reply with the single word: ready."
    )
    fixed = len(tokenizer.encode(begin + end, add_special_tokens=False))
    unit = max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    copies = max(1, math.ceil((target_tokens - fixed) / unit))
    text = begin + filler * copies + end
    count = len(tokenizer.encode(text, add_special_tokens=False))
    while count < target_tokens:
        copies += 1
        text = begin + filler * copies + end
        count = len(tokenizer.encode(text, add_special_tokens=False))
    return text, count


TURN_TWO = (
    "Now reply with the first item on the list, then a |, then the last item. Use no other words."
)


def parking(base_url: str) -> dict:
    _, body = base.http_json(f"{base_url}/v1/cache/status", timeout=10)
    return body.get("parking", {})


def send(base_url: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    status, response = base.http_json(f"{base_url}/v1/chat/completions", body, timeout=3600)
    elapsed = time.perf_counter() - started
    message = response.get("choices", [{}])[0].get("message", {})
    answer = message.get("content") or message.get("reasoning_content") or ""
    usage = response.get("usage", {})
    return {
        "http_status": status,
        "elapsed_s": elapsed,
        "answer": answer,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:2020")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--turn-tokens", type=int, default=65536)
    parser.add_argument("--idle-pause-s", type=float, default=12.0)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--identity-file", required=True)
    parser.add_argument("--identity-mode", choices=("record", "verify"), required=True)
    args = parser.parse_args()

    url = args.base_url.rstrip("/")
    base.wait_ready(url, args.ready_timeout)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    prompt, prompt_tokens = build_turn_one(tokenizer, args.turn_tokens)
    print("TURN_ONE_TOKENS", prompt_tokens)
    model = base.model_id(url)

    print("PARKING_BEFORE", json.dumps(parking(url), sort_keys=True))
    first = send(url, model, [{"role": "user", "content": prompt}], 48)
    print("TURN_ONE", json.dumps(first, ensure_ascii=False, sort_keys=True))
    print("PARKING_AFTER_TURN_ONE", json.dumps(parking(url), sort_keys=True))

    # Wait for the park to actually happen rather than for a fixed wall time. A leaf only
    # becomes a park candidate once the finished request has released it, so a fixed sleep can
    # end before the scheduler's idle pass has parked anything, and turn two then finds the
    # prefix still live on the GPU and never exercises a restore.
    print("IDLE_PAUSE_S", args.idle_pause_s)
    deadline = time.monotonic() + args.idle_pause_s
    while time.monotonic() < deadline:
        parked = parking(url)
        if parked.get("parked_count", 0) > 0 or parked.get("disabled"):
            break
        time.sleep(1.0)
    parked = parking(url)
    print("PARKING_AFTER_PAUSE", json.dumps(parked, sort_keys=True))
    if parked.get("disabled"):
        print("PARKING_SELF_DISABLED_BEFORE_TURN_TWO")

    second = send(
        url,
        model,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": first["answer"]},
            {"role": "user", "content": TURN_TWO},
        ],
        48,
    )
    print("TURN_TWO", json.dumps(second, ensure_ascii=False, sort_keys=True))
    restored = parking(url)
    print("PARKING_AFTER_TURN_TWO", json.dumps(restored, sort_keys=True))

    if second["http_status"] != 200:
        raise AssertionError(f"turn two returned HTTP {second['http_status']}")

    record = {
        "answer": second["answer"],
        "answer_sha256": second["answer_sha256"],
        "completion_tokens": second["usage"].get("completion_tokens"),
    }
    path = Path(args.identity_file)
    if args.identity_mode == "record":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("IDENTITY_RECORDED", path)
    else:
        expected = json.loads(path.read_text(encoding="utf-8"))
        if expected != record:
            raise AssertionError(
                f"second-turn identity mismatch: expected {expected}, got {record}"
            )
        print("IDENTITY_PASS", record["answer_sha256"])

    print("SUMMARY", json.dumps({
        "turn_one_s": first["elapsed_s"],
        "turn_two_s": second["elapsed_s"],
        "turn_two_prompt_tokens": second["usage"].get("prompt_tokens"),
        "turn_two_cached_tokens": second["usage"].get("prompt_tokens_details", {}).get("cached_tokens"),
        "parking_after_pause": parked,
        "parking_after_turn_two": restored,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
