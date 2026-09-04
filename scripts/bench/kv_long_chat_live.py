from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def http_json(url: str, payload: dict | None = None, timeout: int = 1800):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def wait_ready(base_url: str, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, body = http_json(f"{base_url}/v1/cache/status", timeout=5)
            last = f"HTTP {status}: {body}"
            if status == 200 and body.get("state") == "serving":
                print("READY", json.dumps(body, sort_keys=True))
                return body
        except Exception as exc:
            last = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not reach state=serving: {last}")


def model_id(base_url: str) -> str:
    status, body = http_json(f"{base_url}/v1/models", timeout=30)
    if status != 200 or not body.get("data"):
        raise RuntimeError(f"no served model: HTTP {status} {body}")
    return body["data"][0]["id"]


# Each prompt carries one code at the very beginning and a different one at the very end, so a
# correct answer proves the whole prompt was read. F2 2026-09-04 added short/medium/long: the
# 2026-09-03 FP8 gate compared one 250k prompt only, which is too narrow to judge a KV dtype.
PROMPT_VARIANTS = {
    "primary": (
        "BEGIN-CODE: maple-7319. Remember this exact code from the beginning.\n",
        "Middle archive line: ordinary gray stones were counted and no code changed.\n",
        "END-CODE: cobalt-4826. This is the end. Reply with the BEGIN-CODE value, then a |, "
        "then the END-CODE value. Use no other words.",
    ),
    "displacement": (
        "BEGIN-CODE: amber-2048. Remember this different opening.\n",
        "Separate archive line: ordinary blue boxes were counted and no code changed.\n",
        "END-CODE: violet-9631. This is the end. Reply with the two values separated by |.",
    ),
    "short": (
        "BEGIN-CODE: topaz-1204. Remember this exact code from the beginning.\n",
        "Short archive line: ordinary green tiles were counted and no code changed.\n",
        "END-CODE: saffron-8815. This is the end. Reply with the BEGIN-CODE value, then a |, "
        "then the END-CODE value. Use no other words.",
    ),
    "medium": (
        "BEGIN-CODE: jasper-3370. Remember this exact code from the beginning.\n",
        "Medium archive line: ordinary amber wheels were counted and no code changed.\n",
        "END-CODE: crimson-6402. This is the end. Reply with the BEGIN-CODE value, then a |, "
        "then the END-CODE value. Use no other words.",
    ),
    "long": (
        "BEGIN-CODE: onyx-5581. Remember this exact code from the beginning.\n",
        "Long archive line: ordinary copper rings were counted and no code changed.\n",
        "END-CODE: willow-2937. This is the end. Reply with the BEGIN-CODE value, then a |, "
        "then the END-CODE value. Use no other words.",
    ),
}


def build_prompt(tokenizer, target_tokens: int, variant: str) -> tuple[str, int]:
    begin, filler, end = PROMPT_VARIANTS[variant]
    fixed = len(tokenizer.encode(begin + end, add_special_tokens=False))
    unit = max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    copies = max(1, math.ceil((target_tokens - fixed) / unit))
    text = begin + filler * copies + end
    count = len(tokenizer.encode(text, add_special_tokens=False))
    while count < target_tokens:
        text = begin + filler * (copies + 1) + end
        copies += 1
        count = len(tokenizer.encode(text, add_special_tokens=False))
    return text, count


def send_chat(base_url: str, model: str, prompt: str, max_tokens: int,
              ignore_eos: bool = True) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    status, response = http_json(f"{base_url}/v1/chat/completions", body, timeout=3600)
    elapsed = time.perf_counter() - started
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    answer = message.get("content") or message.get("reasoning_content") or ""
    usage = response.get("usage", {})
    result = {
        "http_status": status,
        "elapsed_s": elapsed,
        "answer": answer,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "completion_tok_s": usage.get("completion_tokens", 0) / max(elapsed, 1e-9),
        "total_tok_s": usage.get("total_tokens", 0) / max(elapsed, 1e-9),
    }
    print("RESULT", json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def end_of_turn_ids(tokenizer) -> set[int]:
    """Token ids that end the assistant's turn.

    F2 2026-09-04: the 2026-09-03 FP8 gate compared 48 ignore_eos-forced tokens, but the
    answer ends after about 20. Everything past <|im_end|> is padding that exists only
    because the request forces generation onward, and it is unstable even BF16 against BF16
    (a cache-hit run and a cold run produce different tails on the same prompt), so the gate
    failed on the one part of the output that carries no information.
    """
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    unknown = tokenizer.unk_token_id
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(tid, int) and tid >= 0 and tid != unknown:
        ids.add(int(tid))
    if not ids:
        raise RuntimeError("tokenizer exposes no end-of-turn token id")
    return ids


def gate_tokens(tokenizer, answer: str, eot_ids: set[int], cut_at_eot: bool,
                finish_reason: str | None = None) -> dict:
    """The token ids the gate compares: up to and including the first end-of-turn token."""
    ids = [int(token) for token in tokenizer.encode(answer, add_special_tokens=False)]
    ended = False
    if cut_at_eot:
        for index, token in enumerate(ids):
            if token in eot_ids:
                ids = ids[: index + 1]
                ended = True
                break
        else:
            # Measured 2026-09-04: without ignore_eos the server ends the turn itself and does
            # not put <|im_end|> in `content` (16-17 tokens, finish_reason "stop"), so the two
            # boundaries the job names are the same one and finish_reason is the one that
            # shows up here. With ignore_eos the marker is in the text and the loop above cuts
            # at it. "length" means the answer ran into the cap and was never finished.
            ended = finish_reason == "stop"
    return {"token_ids": ids, "tokens_compared": len(ids), "ended_at_eot": ended}


def first_divergence(reference: list[int], candidate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:2020")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-tokens", type=int, default=250000)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--identity-file", required=True)
    parser.add_argument("--identity-mode", choices=("record", "verify"), required=True)
    # F2 2026-09-04: "eot" is the gate. "forced-48" keeps the 2026-09-03 comparison for
    # reference - 48 ignore_eos-forced tokens, most of them post-end-of-turn padding.
    parser.add_argument("--gate-mode", choices=("eot", "forced-48"), default="eot")
    parser.add_argument("--gate-max-tokens", type=int, default=64,
                        help="cap on the eot gate's generation; the answer ends well inside it")
    args = parser.parse_args()

    if args.target_tokens < 250000:
        raise ValueError("--target-tokens must be at least 250000 for L1")
    base = args.base_url.rstrip("/")
    ready = wait_ready(base, args.ready_timeout)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    eot_ids = end_of_turn_ids(tokenizer)
    cut_at_eot = args.gate_mode == "eot"
    ignore_eos = not cut_at_eot
    max_tokens = args.gate_max_tokens if cut_at_eot else 48
    # The extra lengths only run in eot mode; forced-48 stays exactly the 2026-09-03 shape.
    lengths = {"primary": args.target_tokens}
    if cut_at_eot:
        lengths.update({"short": 2048, "medium": 16384, "long": 65536})

    prompts = {name: build_prompt(tokenizer, count, name) for name, count in lengths.items()}
    displacement, displacement_tokens = build_prompt(tokenizer, args.target_tokens, "displacement")
    print("PROMPTS", json.dumps(
        {name: count for name, (_, count) in prompts.items()}
        | {"displacement_tokens": displacement_tokens}, sort_keys=True))
    for name, (_, count) in prompts.items():
        if count + max_tokens > 262144:
            raise ValueError(f"{name} prompt plus answer exceeds the 262144 model limit")
    if displacement_tokens + 1 > 262144:
        raise ValueError("displacement prompt exceeds the 262144 model limit")

    model = model_id(base)
    print("GATE_MODE", args.gate_mode, "eot_ids", sorted(eot_ids))
    first = send_chat(base, model, prompts["primary"][0], max_tokens, ignore_eos)
    send_chat(base, model, displacement, 1, True)
    replay = send_chat(base, model, prompts["primary"][0], max_tokens, ignore_eos)

    # L1 2026-09-03: the live server reports 47 completion tokens for the 48-token answer
    # (the emitted <|im_end|> is not counted in usage). The gate is answer identity, not the
    # accounting, so require the two runs to agree with each other and store the observed
    # count in the identity record, which the candidate boot must then match exactly.
    counts = {name: result["usage"].get("completion_tokens") for name, result in
              (("first", first), ("replay", replay))}
    if counts["first"] != counts["replay"]:
        raise AssertionError(f"first and replay token counts differ: {counts}")
    if not cut_at_eot and not 47 <= (counts["first"] or 0) <= 48:
        raise AssertionError(f"expected 47-48 generated tokens, got {counts}")
    if first["answer_sha256"] != replay["answer_sha256"] or first["answer"] != replay["answer"]:
        raise AssertionError("cold and replayed answers differ in the same boot")

    results = {"primary": replay}
    for name, (text, _) in prompts.items():
        if name != "primary":
            results[name] = send_chat(base, model, text, max_tokens, ignore_eos)

    record = {"gate_mode": args.gate_mode, "prompts": {}}
    for name, result in results.items():
        gate = gate_tokens(tokenizer, result["answer"], eot_ids, cut_at_eot,
                           result["finish_reason"])
        record["prompts"][name] = {
            "answer": result["answer"],
            "answer_sha256": result["answer_sha256"],
            "completion_tokens": result["usage"].get("completion_tokens"),
            "finish_reason": result["finish_reason"],
            "gate_text": tokenizer.decode(gate["token_ids"]),
            "gate_token_ids": gate["token_ids"],
            "tokens_compared": gate["tokens_compared"],
            "ended_at_eot": gate["ended_at_eot"],
        }
        if cut_at_eot and not gate["ended_at_eot"]:
            raise AssertionError(
                f"{name}: no end-of-turn token inside {max_tokens} tokens "
                f"(finish_reason {result['finish_reason']!r}); raise --gate-max-tokens"
            )

    path = Path(args.identity_file)
    if args.identity_mode == "record":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        for name, entry in record["prompts"].items():
            print(f"GATE {name} RECORDED tokens_compared={entry['tokens_compared']} "
                  f"sha={entry['answer_sha256'][:16]}")
            print(f"GATE {name} TEXT {json.dumps(entry['gate_text'], ensure_ascii=False)}")
        print("IDENTITY_RECORDED", path)
    else:
        expected = json.loads(path.read_text(encoding="utf-8"))
        if expected.get("gate_mode") != args.gate_mode:
            raise AssertionError(
                f"identity file was recorded in {expected.get('gate_mode')!r} mode, "
                f"not {args.gate_mode!r}"
            )
        failures = []
        for name, entry in record["prompts"].items():
            reference = expected["prompts"].get(name)
            if reference is None:
                raise AssertionError(f"identity file has no reference for prompt {name!r}")
            divergence = first_divergence(reference["gate_token_ids"], entry["gate_token_ids"])
            verdict = "PASS" if divergence is None else "FAIL"
            if divergence is not None:
                failures.append(name)
            print(f"GATE {name} {verdict} tokens_compared="
                  f"{reference['tokens_compared']}/{entry['tokens_compared']} "
                  f"first_divergence={divergence}")
            print(f"GATE {name} REFERENCE {json.dumps(reference['gate_text'], ensure_ascii=False)}")
            print(f"GATE {name} CANDIDATE {json.dumps(entry['gate_text'], ensure_ascii=False)}")
        print("GATE OVERALL", "FAIL" if failures else "PASS",
              json.dumps({"failed": failures}, sort_keys=True))
        if failures:
            raise AssertionError(f"gate mismatch on {failures}")
        print("IDENTITY_PASS", record["prompts"]["primary"]["answer_sha256"])

    print("STATUS", json.dumps(ready, sort_keys=True))
    print("ANSWER", replay["answer"])
    print("USAGE", json.dumps(replay["usage"], sort_keys=True))


if __name__ == "__main__":
    main()
