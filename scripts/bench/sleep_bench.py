#!/usr/bin/env python3
"""Sleep/wake acceptance bench for a live FreeToken server (stdlib only; run on the serving box).

Per cycle: two greedy chats awake (A1 cold, A2 warm), POST /v1/sleep (timed), read the card and
Windows free RAM while asleep, then a greedy chat that wakes the model by itself (B, timed end
to end), then sleep again and an explicit POST /v1/wake (timed). A cycle passes when B equals
A1 or A2 token for token, every sleep and wake answers 200, /v1/cache/status says "sleeping"
while asleep, the explicit wake is within --max-wake-s, and the card reading while asleep is
at most --max-asleep-gib above --baseline-gib (the card with FreeToken stopped).

    python3 scripts/bench/sleep_bench.py --cycles 3 --baseline-gib 2.1 --cold-boot-s 155 \
        --out ~/sleep-bench.json

The JSON file keeps everything the acceptance doc needs (Task 13 of the plan): the card
readings, sleep and wake times against the cold boot, the greedy outputs, Windows free RAM at
each point, and the server's own sleep block from /health. The server is never left asleep:
the bench wakes it in a ``finally`` (also on Ctrl-C).

Design: docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md (section 1 criteria).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPT = (
    "List the first twelve prime numbers, then explain in three sentences why there are "
    "infinitely many primes."
)

# nvidia-smi inside WSL lives under /usr/lib/wsl/lib and is often not on PATH; it reports the
# whole card (Windows-side view), which is the number Jay's game sees.
_WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"
# The absolute powershell path works without any interop env (the governor uses it for the
# same reason: python/freetoken/daemon/settings/governor.py, measured on the box 2026-09-07).
_POWERSHELL_ABS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def call(base: str, path: str, body: dict | None = None, timeout: float = 600.0) -> tuple[int, dict]:
    """GET (body None) or POST JSON; returns (status, decoded body, {} when empty)."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read() or b"{}"
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"raw": raw.decode(errors="replace")}


def card_used_gib() -> float | None:
    """Card memory in use (GiB) as nvidia-smi reports it; None when there is no nvidia-smi."""
    exe = shutil.which("nvidia-smi") or _WSL_NVIDIA_SMI
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return round(int(out.split()[0]) / 1024, 2)
    except Exception:  # noqa: BLE001 - the bench still reports the server's own numbers
        return None


def windows_free_ram_gb() -> float | None:
    """Windows free physical RAM in GB (Task Manager's "Available"-ish number the box rules
    watch); None off WSL or when powershell is unreachable."""
    candidates = [c for c in (shutil.which("powershell.exe"), _POWERSHELL_ABS) if c and os.path.exists(c)]
    for exe in candidates:
        try:
            proc = subprocess.run(
                [exe, "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            kb = int(proc.stdout.strip())
        except Exception:  # noqa: BLE001
            continue
        return round(kb * 1024 / 1e9, 2)
    return None


def probes() -> dict:
    return {"card_gib": card_used_gib(), "win_free_ram_gb": windows_free_ram_gb()}


def chat(base: str, model: str, tokens: int) -> tuple[str, float, int]:
    """One greedy, non-streamed chat: (text, seconds end to end, completion tokens)."""
    t0 = time.monotonic()
    status, body = call(base, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens, "temperature": 0, "stream": False,
    })
    if status != 200:
        raise SystemExit(f"chat failed: HTTP {status} {json.dumps(body)[:500]}")
    text = body["choices"][0]["message"]["content"]
    completion = int((body.get("usage") or {}).get("completion_tokens") or 0)
    return text, round(time.monotonic() - t0, 2), completion


def timed_post(base: str, path: str, timeout: float) -> tuple[int, dict, float]:
    t0 = time.monotonic()
    status, body = call(base, path, {}, timeout=timeout)
    return status, body, round(time.monotonic() - t0, 2)


def judge(row: dict, *, baseline_gib: float, max_asleep_gib: float, max_wake_s: float) -> list[str]:
    """Pure: the reasons a cycle row fails (empty = pass)."""
    reasons = []
    if not row["same_output"]:
        reasons.append("output after wake differs from the awake outputs")
    for key in ("sleep_http", "sleep2_http", "wake_http"):
        if row[key] != 200:
            reasons.append(f"{key}={row[key]}")
    if row["state_asleep"] != "sleeping":
        reasons.append(f"state while asleep was {row['state_asleep']!r}")
    if row["wake_s"] > max_wake_s:
        reasons.append(f"wake took {row['wake_s']} s > {max_wake_s} s")
    asleep = row["card_asleep_gib"]
    if asleep is not None and asleep - baseline_gib > max_asleep_gib:
        reasons.append(f"card asleep {asleep} GiB is more than {max_asleep_gib} GiB above the {baseline_gib} GiB baseline")
    return reasons


def summarize(cycles: list[dict], *, cold_boot_s: float | None) -> dict:
    """Pure: the numbers the acceptance table quotes, over the cycles that ran."""
    def stat(key):
        vals = [c[key] for c in cycles if isinstance(c.get(key), (int, float))]
        if not vals:
            return None
        return {"min": min(vals), "median": round(statistics.median(vals), 2), "max": max(vals)}

    out = {
        "cycles_run": len(cycles),
        "cycles_passed": sum(1 for c in cycles if c.get("pass")),
        "sleep_s": stat("sleep_s"),
        "wake_s": stat("wake_s"),
        "chat_after_sleep_s": stat("chat_after_sleep_s"),
        "wake_inside_chat_s": stat("wake_inside_chat_s"),
        "card_awake_gib": stat("card_awake_gib"),
        "card_asleep_gib": stat("card_asleep_gib"),
        "win_free_ram_gb_asleep": stat("win_free_ram_gb_asleep"),
        "cold_boot_s": cold_boot_s,
    }
    wake = out["wake_s"]
    if cold_boot_s and wake:
        out["wake_vs_cold_boot"] = f"{wake['max']} s wake vs {cold_boot_s} s cold boot ({cold_boot_s / wake['max']:.1f}x faster)"
    return out


def is_asleep(base: str) -> bool | None:
    try:
        _, health = call(base, "/health", timeout=10)
    except Exception:  # noqa: BLE001
        return None
    return health.get("maintenance") == "sleeping"


def ensure_awake(base: str, timeout: float) -> None:
    """Never leave the server asleep: the panel and llama-swap expect the shape they had."""
    if is_asleep(base):
        status, body, secs = timed_post(base, "/v1/wake", timeout)
        print(json.dumps({"final_wake_http": status, "final_wake_s": secs, "reply": body}), flush=True)


def run_cycle(n: int, args, model: str) -> dict:
    base = args.server
    a1, a1_s, a1_tokens = chat(base, model, args.tokens)
    a2, a2_s, _ = chat(base, model, args.tokens)
    awake = probes()

    s_status, s_body, sleep_s = timed_post(base, "/v1/sleep", args.sleep_timeout)
    time.sleep(args.settle_s)  # let WSL/WDDM hand the freed memory back before reading the card
    asleep = probes()
    _, status_doc = call(base, "/v1/cache/status")
    _, health_asleep = call(base, "/health")

    b, b_s, _ = chat(base, model, args.tokens)  # wakes the model by itself
    after_chat = probes()

    s2_status, _, _ = timed_post(base, "/v1/sleep", args.sleep_timeout)
    w_status, w_body, wake_s = timed_post(base, "/v1/wake", args.wake_timeout)
    after_wake = probes()

    row = {
        "cycle": n, "same_output": b in (a1, a2), "same_as": "a1" if b == a1 else ("a2" if b == a2 else None),
        "tokens": a1_tokens, "a1_s": a1_s, "a2_s": a2_s,
        "sleep_http": s_status, "sleep_s": sleep_s, "sleep_reply": s_body,
        "state_asleep": status_doc.get("state"), "health_sleep_block": health_asleep.get("sleep"),
        "card_awake_gib": awake["card_gib"], "card_asleep_gib": asleep["card_gib"],
        "card_after_wake_gib": after_wake["card_gib"],
        "win_free_ram_gb_awake": awake["win_free_ram_gb"], "win_free_ram_gb_asleep": asleep["win_free_ram_gb"],
        "win_free_ram_gb_after_chat": after_chat["win_free_ram_gb"],
        "win_free_ram_gb_after_wake": after_wake["win_free_ram_gb"],
        "chat_after_sleep_s": b_s,
        # the wake cost hidden inside chat B: its end-to-end time less a warm awake chat
        "wake_inside_chat_s": round(b_s - a2_s, 2),
        "sleep2_http": s2_status, "wake_http": w_status, "wake_s": wake_s, "wake_reply": w_body,
        "outputs": {"a1": a1, "a2": a2, "b": b},
    }
    row["fail_reasons"] = judge(row, baseline_gib=args.baseline_gib, max_asleep_gib=args.max_asleep_gib,
                                max_wake_s=args.max_wake_s)
    row["pass"] = not row["fail_reasons"]
    return row


def _printable(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.endswith("_reply") and k not in ("outputs", "health_sleep_block")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:2020")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=200, help="max_tokens per greedy chat")
    ap.add_argument("--baseline-gib", type=float, required=True,
                    help="card used with FreeToken stopped (desktop only), measured first")
    ap.add_argument("--max-asleep-gib", type=float, default=7.5,
                    help="pass ceiling for card used while asleep, above the baseline (spec section 1)")
    ap.add_argument("--max-wake-s", type=float, default=45.0, help="pass ceiling for POST /v1/wake (target 30)")
    ap.add_argument("--cold-boot-s", type=float, default=None,
                    help="last cold boot's load time from logs/server-2020.log, for the comparison line")
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="pause after sleep before reading the card (WDDM returns memory lazily)")
    ap.add_argument("--sleep-timeout", type=float, default=180.0)
    ap.add_argument("--wake-timeout", type=float, default=330.0,
                    help="a little over the server's own WAKE_WAIT_S (300 s) so its verdict arrives")
    ap.add_argument("--out", default="sleep-bench.json")
    args = ap.parse_args()

    status, health = call(args.server, "/health", timeout=10)
    if status != 200 or health.get("status") != "ok":
        raise SystemExit(f"server not serving: HTTP {status} {json.dumps(health)[:300]}")
    model = health.get("model") or "default"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    cycles, ok = [], True
    try:
        for n in range(1, args.cycles + 1):
            row = run_cycle(n, args, model)
            ok &= row["pass"]
            cycles.append(row)
            print(json.dumps(_printable(row)), flush=True)
    finally:
        ensure_awake(args.server, args.wake_timeout)
        summary = summarize(cycles, cold_boot_s=args.cold_boot_s)
        report = {
            "model": model, "server": args.server, "started": started, "prompt": PROMPT,
            "tokens": args.tokens, "baseline_gib": args.baseline_gib, "max_asleep_gib": args.max_asleep_gib,
            "max_wake_s": args.max_wake_s, "health_at_start": health,
            "summary": summary, "cycles": cycles, "pass": ok and len(cycles) == args.cycles,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(json.dumps({"summary": summary, "out": args.out}), flush=True)
    print("PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
