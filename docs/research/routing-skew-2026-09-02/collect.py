"""Collect per-workload MoE decode routing histograms from a live FreeToken server.

Usage: python routing_run.py <port> <out_dir>

Four workloads, ~500 generated tokens each. Before each one the counters are zeroed
(GET /v1/cache/routing?reset=true, result discarded); after it they are read and zeroed
again, so each JSON holds exactly that workload's routing. A final read after all four is
saved as "all" (it only holds the last workload, so the union is rebuilt offline).
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "Qwen3.8-Flash-Next-NVFP4"
BASE = f"http://127.0.0.1:{PORT}"
SRC = Path(r"D:\FreeToken\python\freetoken")

CODE8K = (SRC / "engine/spec_graph.py").read_text(encoding="utf-8")[:30000]

TOOLS_PREAMBLE = """You are an API planning assistant. You answer ONLY with a JSON array of
tool calls, one object per call, each of the form
{"tool": "<name>", "arguments": {...}, "why": "<one short sentence>"}.
Available tools: search_flights, search_hotels, book_flight, book_hotel, get_weather,
convert_currency, send_email, create_calendar_event, list_calendar_events, cancel_booking,
get_exchange_rate, geocode, reverse_geocode, translate_text, summarize_document.
"""

WORKLOADS = {
    "code": [
        {"role": "user", "content":
         "Write a complete, production-quality Python module implementing a thread-safe LRU "
         "cache with per-entry TTL expiry: type hints, docstrings, an internal doubly-linked "
         "list, a background sweeper thread, and a unittest block at the end. Output only "
         "code, no prose. Make it long and complete."},
    ],
    "prose": [
        {"role": "user", "content":
         "Write a thoughtful, flowing essay about the history of navigation at sea, from dead "
         "reckoning and the log line through lunar distances, the marine chronometer, radio "
         "direction finding, LORAN, and satellite positioning. Continuous prose only: no "
         "lists, no headings, no bullet points. At least 600 words."},
    ],
    "chat8k": [
        {"role": "user", "content":
         "Here is a Python module from my project:\n\n```python\n" + CODE8K +
         "\n```\n\nExplain in detail how this module decides whether a batch can be replayed "
         "from a captured graph, what invariants the capture relies on, and where you would "
         "add a new guard. Be thorough."},
    ],
    "toolcall": [
        {"role": "system", "content": TOOLS_PREAMBLE},
        {"role": "user", "content":
         "Plan a two-week research trip: Berlin -> Tokyo -> Sydney -> Berlin, departing "
         "2026-10-03, budget EUR 4500, one carry-on. I need flights, hotels near each "
         "conference venue, the local weather forecast for each leg, currency conversions "
         "into EUR, calendar events for every flight and check-in, and a summary email to "
         "my co-author. Emit the full JSON array of tool calls, at least 20 of them."},
    ],
}


def get_json(path, timeout=120):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(messages, max_tokens=520):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    tokens = 0
    text = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    text.append(d["content"])
            u = ev.get("usage")
            if u and u.get("completion_tokens"):
                tokens = u["completion_tokens"]
    return {
        "completion_tokens": tokens,
        "seconds": time.perf_counter() - t0,
        "sample": "".join(text)[:400],
    }


def main():
    summaries = {}
    for name, messages in WORKLOADS.items():
        get_json("/v1/cache/routing?reset=true")  # zero the window
        gen = chat(messages)
        stats = get_json("/v1/cache/routing?reset=true")
        stats["workload"] = name
        stats["generation"] = gen
        (OUT / f"{name}.json").write_text(json.dumps(stats), encoding="utf-8")
        summaries[name] = {
            "completion_tokens": gen["completion_tokens"],
            "seconds": round(gen["seconds"], 2),
            "tok_s": round(gen["completion_tokens"] / max(gen["seconds"], 1e-9), 1),
            "summary": stats.get("summary", {}),
            "total_counts": sum(sum(r) for r in stats.get("decode_freq") or []),
        }
        print(f"[{name}] tokens={gen['completion_tokens']} "
              f"{summaries[name]['tok_s']} tok/s counts={summaries[name]['total_counts']}",
              flush=True)
        print("   sample:", gen["sample"][:160].replace("\n", " "), flush=True)

    # One more pass over all four back-to-back, without resetting in between: the union.
    get_json("/v1/cache/routing?reset=true")
    for name, messages in WORKLOADS.items():
        chat(messages, max_tokens=520)
    union = get_json("/v1/cache/routing?reset=true")
    union["workload"] = "union"
    (OUT / "union.json").write_text(json.dumps(union), encoding="utf-8")
    summaries["union"] = {
        "summary": union.get("summary", {}),
        "total_counts": sum(sum(r) for r in union.get("decode_freq") or []),
    }
    print("[union] counts=", summaries["union"]["total_counts"], flush=True)

    (OUT / "summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
