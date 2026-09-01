"""Per-stage kernel breakdown of a FREETOKEN_DIAG_PROFILE_DIR chrome trace.

    python scripts/diag/analyze_profile.py <dir-or-trace.json> [--top 15]

Reads the trace written by ``freetoken/diag.py`` and, for every ``diag.*`` range, reports
how long the stage took on the host, how much GPU kernel time landed inside it, and which
kernels those were. The attribution is only meaningful because the seam device-syncs at the
END of each range: that is what puts a kernel's GPU timestamp inside the CPU range of the
stage that launched it.

Nested ranges (``diag.ple_gather`` inside ``diag.spec_verify_replay``) each count the kernels
inside them, so the columns are per-stage totals and do not sum to the trace.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from collections import defaultdict

# Chrome-trace categories that are device work. "kernel" is the compute; the two memory
# categories are the PCIe expert fetch and the staging copies, which are the point here.
KERNEL_CATS = ("kernel",)
MEMORY_CATS = ("gpu_memcpy", "gpu_memset")
DEVICE_CATS = KERNEL_CATS + MEMORY_CATS

# record_function emits a CPU "user_annotation" range and (when the range covered device
# work) a mirrored "gpu_user_annotation" one. The CPU range is the stage; the mirror would
# double-count it.
_MIRROR_CATS = {"gpu_user_annotation"}


def load_events(path: str) -> list:
    """Load a chrome trace. Linear in file size: one json.load, one pass, nothing quadratic."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        return raw.get("traceEvents") or []
    return raw


def split_events(events: list):
    """-> ({range name: [(ts, end)]}, sorted device events)."""
    ranges: dict[str, list] = defaultdict(list)
    device: list = []
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = event.get("cat") or ""
        name = event.get("name") or ""
        ts = event.get("ts")
        dur = event.get("dur")
        if ts is None or dur is None:
            continue
        if name.startswith("diag.") and cat not in _MIRROR_CATS:
            ranges[name].append((float(ts), float(ts) + float(dur)))
        elif cat in DEVICE_CATS:
            device.append((float(ts), float(dur), name, cat))
    device.sort(key=lambda row: row[0])
    for spans in ranges.values():
        spans.sort()
    return ranges, device


def kernels_in(device: list, starts: list, lo: float, hi: float):
    """Device events whose start timestamp falls in [lo, hi]. Binary search, then walk."""
    index = bisect.bisect_left(starts, lo)
    while index < len(device) and device[index][0] <= hi:
        yield device[index]
        index += 1


def analyze(path: str, top: int) -> int:
    events = load_events(path)
    ranges, device = split_events(events)
    if not ranges:
        print(f"no diag.* ranges in {path} (was FREETOKEN_DIAG_PROFILE_DIR armed?)")
        return 1
    starts = [row[0] for row in device]

    print(f"trace: {path}")
    print(f"events: {len(events)}  device events: {len(device)}  diag ranges: {len(ranges)}")
    print()

    per_instance_kernels: dict[str, list] = {}
    for name in sorted(ranges):
        spans = ranges[name]
        wall_us = sum(end - start for start, end in spans)
        totals: dict[tuple, list] = defaultdict(lambda: [0.0, 0])
        gpu_us = 0.0
        counts = []
        for start, end in spans:
            n = 0
            for _, dur, kname, cat in kernels_in(device, starts, start, end):
                key = (kname, "kernel" if cat in KERNEL_CATS else cat)
                slot = totals[key]
                slot[0] += dur
                slot[1] += 1
                gpu_us += dur
                n += 1
            counts.append(n)
        per_instance_kernels[name] = counts
        instances = len(spans)
        print(f"== {name}")
        print(
            f"   count {instances}  mean wall {wall_us / instances / 1e3:.3f} ms  "
            f"mean gpu-kernel {gpu_us / instances / 1e3:.3f} ms  "
            f"mean kernels/instance {sum(counts) / instances:.1f}"
        )
        rows = sorted(totals.items(), key=lambda item: -item[1][0])[:top]
        if rows:
            print(f"   {'kernel':<70} {'total ms':>10} {'count':>8} {'mean us':>10}  cat")
        for (kname, cat), (total, count) in rows:
            print(
                f"   {kname[:70]:<70} {total / 1e3:>10.3f} {count:>8} "
                f"{total / count:>10.1f}  {cat}"
            )
        print()

    print("== kernels launched per step")
    for name in ("diag.plain_decode_step", "diag.spec_verify_replay"):
        counts = per_instance_kernels.get(name)
        if not counts:
            print(f"   {name}: not in this trace")
            continue
        ordered = sorted(counts)
        print(
            f"   {name}: instances {len(counts)}  mean {sum(counts) / len(counts):.1f}  "
            f"min {ordered[0]}  median {ordered[len(ordered) // 2]}  max {ordered[-1]}"
        )
    return 0


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="the profile directory, or a trace.json directly")
    parser.add_argument("--top", type=int, default=15, help="kernels listed per range")
    args = parser.parse_args(argv)

    path = args.path
    if os.path.isdir(path):
        path = os.path.join(path, "trace.json")
    if not os.path.isfile(path):
        print(f"no trace at {path}")
        return 2
    return analyze(path, args.top)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
