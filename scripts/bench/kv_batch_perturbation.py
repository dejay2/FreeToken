"""F4: how far does packed-batch numerical noise move a greedy answer?

D1 (2026-09-03) closed the "crossed reply" scare: nothing leaked between requests, the four
probes differed by a single digit and a mixed prefill batch nudged a near-tie the other way.
It also measured that the mixed batch changes the token run of essentially every request in
it (15/15 with distinct-word prompts) while solo answers are stable. This driver bounds that:
for the same prompt and the same warm cache, how often does batch shape change the FIRST
token, and how often does it change the finished answer?

Three prompt classes, so the flip rate can be split by how tied the top-2 choice is:
  digit    - D1's marker probes; the answer is a number whose neighbours are one token away.
  word     - distinct-word passwords; the correct answer is restated in the prompt.
  ordinary - plain questions with no planted answer.

The server has no logprobs (`server/openai_api.py:630` rejects `logprobs` outright, and the
chat request model carries no `top_logprobs`), so the top-2 gap and the max abs logit shift
asked for in the F4 brief cannot be read from the API. The fallback is text: first-token
match, full-text match, and the token index where batched and solo first diverge. Divergence
index is the "how much" proxy - index 0 means the very first choice flipped.

Each repeat re-prefills the three fillers with a fresh nonce, so the probe sits in the same
mixed batch shape D1 used (one long cold prefill packed beside warm extends), and the probe
itself is pre-warmed so batched and solo runs read the same cached prefix.
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

WORDS = ["maple", "cobalt", "quartz", "indigo", "saffron",
         "basalt", "juniper", "lantern", "marigold", "onyx"]

ORDINARY = [
    "What is the capital of France? Answer with one word and nothing else.",
    "How many days are in a leap year? Answer with the number and nothing else.",
    "What colour do you get when you mix blue and yellow? One word only.",
    "Which planet is closest to the Sun? One word only.",
    "What is 17 plus 26? Answer with the number and nothing else.",
    "Name the largest ocean on Earth. Two words only.",
    "What gas do plants take in to grow? Answer with one short phrase.",
    "How many sides does a hexagon have? Answer with the number and nothing else.",
    "What is the chemical symbol for gold? Answer with the symbol and nothing else.",
    "In which season do deciduous trees lose their leaves? One word only.",
]

FILLER_TOPICS = [
    "warehouse stock rotation",
    "coastal tide timetables",
    "printing press maintenance",
]


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


def pad(tokenizer, head, tail, target_tokens, tag):
    """head + repeated neutral lines + tail, grown to at least target_tokens."""
    if target_tokens <= 0:
        return head + tail
    line = f"This line is context for {tag} and states nothing that changes the answer.\n"
    fixed = len(tokenizer.encode(head + tail, add_special_tokens=False))
    unit = max(1, len(tokenizer.encode(line, add_special_tokens=False)))
    copies = max(1, math.ceil((target_tokens - fixed) / unit))
    text = head + line * copies + tail
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        copies += 1
        text = head + line * copies + tail
    return text


def build_probes(tokenizer, per_class, probe_tokens):
    """(probe_id, class, prompt, expected) rows; expected is None for ordinary questions."""
    probes = []
    for i in range(per_class):
        marker = f"request-{i}-marker-{7000 + i}"
        probes.append((f"digit-{i}", "digit",
                       pad(tokenizer, f"Remember this marker: {marker}.\n",
                           f"Reply with exactly this marker and nothing else: {marker}",
                           probe_tokens, f"marker {i}"),
                       marker))
    for i in range(per_class):
        word = WORDS[i % len(WORDS)]
        probes.append((f"word-{i}", "word",
                       pad(tokenizer, f"Remember this password: {word}.\n",
                           f"Reply with exactly the password and nothing else: {word}",
                           probe_tokens, f"password {i}"),
                       word))
    for i in range(per_class):
        probes.append((f"ordinary-{i}", "ordinary",
                       pad(tokenizer, "", ORDINARY[i % len(ORDINARY)],
                           probe_tokens, f"question {i}"),
                       None))
    return probes


def build_filler(tokenizer, target_tokens, slot, nonce):
    """A long, cold prompt: the nonce forces a fresh prefill on every repeat."""
    topic = FILLER_TOPICS[slot % len(FILLER_TOPICS)]
    head = f"Read the following notes on {topic} (batch {nonce}).\n"
    line = (f"Note {slot}: the {topic} schedule for pass {nonce} is reviewed weekly "
            f"and no exception was recorded.\n")
    tail = f"In one sentence, say what the notes on {topic} are about."
    fixed = len(tokenizer.encode(head + tail, add_special_tokens=False))
    unit = max(1, len(tokenizer.encode(line, add_special_tokens=False)))
    copies = max(1, math.ceil((target_tokens - fixed) / unit))
    text = head + line * copies + tail
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        copies += 1
        text = head + line * copies + tail
    return text


def ask(base_url, model, prompt, max_tokens, ignore_eos=False, clock=None):
    """clock, when given, receives (start, end) monotonic stamps for overlap accounting."""
    started = time.monotonic()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
        "ignore_eos": ignore_eos, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    _st, resp = http_json(f"{base_url}/v1/chat/completions", body, timeout=3600)
    if clock is not None:
        clock.append((started, time.monotonic()))
    msg = resp.get("choices", [{}])[0].get("message", {})
    return msg.get("content") or msg.get("reasoning_content") or ""


def divergence_index(tokenizer, a, b):
    """Token index of the first difference; -1 when the two answers are identical."""
    ta = tokenizer.encode(a, add_special_tokens=False)
    tb = tokenizer.encode(b, add_special_tokens=False)
    for i in range(min(len(ta), len(tb))):
        if ta[i] != tb[i]:
            return i, ta, tb
    if len(ta) == len(tb):
        return -1, ta, tb
    return min(len(ta), len(tb)), ta, tb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:2020")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--per-class", type=int, default=10,
                        help="probes per class; 10 gives the 30 prompts F4 asks for")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--filler-tokens", type=int, default=2048,
                        help="length of each of the three cold fillers packed beside the probe")
    parser.add_argument("--probe-tokens", type=int, default=0,
                        help="pad each probe to this many tokens; 0 keeps the short probe")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ignore-eos", action="store_true",
                        help="force max_tokens of output, as D1's drivers did")
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--out", default=None, help="optional JSON results file")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    status = wait_ready(base_url, args.ready_timeout)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    model = model_id(base_url)
    probes = build_probes(tokenizer, args.per_class, args.probe_tokens)

    print("BATCH_PERTURBATION_START "
          f"prompts={len(probes)} repeats={args.repeats} probe_tokens={args.probe_tokens} "
          f"filler_tokens={args.filler_tokens} max_tokens={args.max_tokens} "
          f"ignore_eos={args.ignore_eos} kv_dtype={status['geometry']['kv_dtype']} "
          f"parking={status['parking']['mode']} logprobs=unsupported", flush=True)

    # Warm every probe's prefix once, so the batched and solo runs read the same cache.
    for pid, _cls, prompt, _exp in probes:
        ask(base_url, model, prompt, args.max_tokens, args.ignore_eos)
    print(f"WARMED {len(probes)} probe prefixes", flush=True)

    records = []
    started = time.monotonic()
    for rep in range(1, args.repeats + 1):
        for pid, cls, prompt, expected in probes:
            nonce = f"{rep:02d}-{pid}"
            fillers = [build_filler(tokenizer, args.filler_tokens, s, nonce) for s in range(3)]
            gate = threading.Barrier(4)
            answers = {}
            clock = []

            def worker(slot, text):
                gate.wait()
                answers[slot] = ask(base_url, model, text, args.max_tokens,
                                    args.ignore_eos, clock)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(worker, 0, prompt)]
                futures += [pool.submit(worker, s + 1, fillers[s]) for s in range(3)]
                for f in futures:
                    f.result()

            # Serial time / wall time: 1.0 means the four requests ran one after another
            # (nothing was packed and a 0 flip count would be vacuous), 4.0 means they were
            # in flight together for their whole life.
            wall = max(e for _s, e in clock) - min(s for s, _e in clock)
            serial = sum(e - s for s, e in clock)
            batched = answers[0]
            solo = ask(base_url, model, prompt, args.max_tokens, args.ignore_eos)
            idx, tb, ts = divergence_index(tokenizer, batched, solo)
            rec = {
                "probe": pid, "class": cls, "repeat": rep,
                "first_token_match": bool(tb[:1] == ts[:1]),
                "full_text_match": batched == solo,
                "divergence_index": idx,
                "batched_tokens": len(tb), "solo_tokens": len(ts),
                "batch_wall_s": round(wall, 3),
                "batch_concurrency": round(serial / wall, 2) if wall > 0 else 0.0,
            }
            if expected is not None:
                rec["batched_correct"] = expected in batched
                rec["solo_correct"] = expected in solo
                rec["verdict_flipped"] = rec["batched_correct"] != rec["solo_correct"]
            records.append(rec)
            if not rec["full_text_match"]:
                print(f"  DIFF {pid} rep={rep} first_token_match={rec['first_token_match']} "
                      f"diverge_at={idx}", flush=True)
        print(f"REPEAT {rep}/{args.repeats} done "
              f"({time.monotonic() - started:.1f}s elapsed)", flush=True)

    print("\nCLASS               n  first_token_flips  full_text_diffs  "
          "median_diverge  answer_verdict_flips", flush=True)
    for cls in ("digit", "word", "ordinary"):
        rows = [r for r in records if r["class"] == cls]
        if not rows:
            continue
        ft = sum(1 for r in rows if not r["first_token_match"])
        fx = sum(1 for r in rows if not r["full_text_match"])
        div = sorted(r["divergence_index"] for r in rows if r["divergence_index"] >= 0)
        med = div[len(div) // 2] if div else "n/a"
        has_v = any("verdict_flipped" in r for r in rows)
        vf = sum(1 for r in rows if r.get("verdict_flipped"))
        print(f"{cls:<12} {len(rows):>5}  {ft:>17}  {fx:>15}  {str(med):>14}  "
              f"{(str(vf) if has_v else 'n/a'):>20}", flush=True)

    total = len(records)
    ft_all = sum(1 for r in records if not r["first_token_match"])
    fx_all = sum(1 for r in records if not r["full_text_match"])
    vf_all = sum(1 for r in records if r.get("verdict_flipped"))
    conc = sorted(r["batch_concurrency"] for r in records)
    print(f"\nBATCH_PERTURBATION_DONE samples={total} first_token_flips={ft_all} "
          f"full_text_diffs={fx_all} answer_verdict_flips={vf_all} "
          f"median_batch_concurrency={conc[len(conc) // 2]:.2f} "
          f"min_batch_concurrency={conc[0]:.2f} "
          f"elapsed_s={time.monotonic() - started:.1f}", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"records": records, "prompts": len(probes),
                       "repeats": args.repeats, "probe_tokens": args.probe_tokens,
                       "filler_tokens": args.filler_tokens,
                       "max_tokens": args.max_tokens,
                       "ignore_eos": args.ignore_eos}, fh, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
