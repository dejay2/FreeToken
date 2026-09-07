#!/usr/bin/env python3
r"""Memory grabber tool for FreeToken memory governor squeeze tests.

Intended to run on Windows (C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe)
to allocate card VRAM and host RAM in stepped amounts, hold at peak, and release in reverse.
Logs memory status every second to a CSV file.

Supports --dry-run to simulate and log the timetable without allocating memory,
allowing verification on machines without CUDA/Windows.

Spec:
  --vram-steps 2,4,8,12 --ram-steps 4,8,16 --hold 60 --step-interval 30 --log grab.csv
  Allocates CUDA memory in the named GB steps with torch.empty(..., device="cuda") and touches it,
  RAM with bytearray touched page by page, holds, releases in reverse;
  writes one CSV row per second:
  t, phase, vram_free_mb (nvidia-smi), win_free_mb (Win32_OperatingSystem), grabbed_vram_gb, grabbed_ram_gb.
  Frees everything on any exception (try/finally) and on Ctrl-C.
  Must not exceed --max-vram-gb (default total-1) and refuses to run if nvidia-smi shows less than 1 GB free.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import shutil
import subprocess
import sys
import time
from typing import Any


def parse_steps(steps_str: str) -> list[float]:
    """Parse comma-separated GB step string (e.g. '2,4,8,12')."""
    steps_str = steps_str.strip()
    if not steps_str or steps_str == "0":
        return []
    parts = [p.strip() for p in steps_str.split(",") if p.strip()]
    return [float(p) for p in parts if float(p) > 0.0]


def query_nvidia_smi() -> tuple[float | None, float | None]:
    """Query nvidia-smi for (total_mb, free_mb). Returns (None, None) on failure."""
    cmd = shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        res = subprocess.run(
            [cmd, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            first_line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            total_mb = float(parts[0])
            free_mb = float(parts[1])
            return total_mb, free_mb
    except Exception:
        pass
    return None, None


def query_win_free_mb() -> float:
    """Query free physical RAM in MB (Win32_OperatingSystem or /proc/meminfo fallback)."""
    if sys.platform == "win32":
        # Fast path: GlobalMemoryStatusEx via ctypes
        try:
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("avail_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.avail_phys / (1024.0 * 1024.0)
        except Exception:
            pass

        # Fallback to PowerShell Get-CimInstance Win32_OperatingSystem
        try:
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip()) / 1024.0
        except Exception:
            pass
        return 0.0
    else:
        # Linux / devbox fallback for dry-run testing
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # Value is in kB
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0


def format_duration(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def build_schedule(
    vram_steps: list[float],
    ram_steps: list[float],
    hold_sec: float,
    step_sec: float,
) -> list[dict[str, Any]]:
    """Construct the sequence of timetable intervals for stepping up, holding, and stepping down."""
    schedule: list[dict[str, Any]] = []
    num_steps = max(len(vram_steps), len(ram_steps))
    if num_steps == 0:
        return []

    # 1. Step up
    for i in range(num_steps):
        v = vram_steps[min(i, len(vram_steps) - 1)] if vram_steps else 0.0
        r = ram_steps[min(i, len(ram_steps) - 1)] if ram_steps else 0.0
        schedule.append({
            "phase": "step_up",
            "duration": step_sec,
            "target_vram": v,
            "target_ram": r,
        })

    # 2. Hold at peak
    peak_v = vram_steps[-1] if vram_steps else 0.0
    peak_r = ram_steps[-1] if ram_steps else 0.0
    schedule.append({
        "phase": "hold",
        "duration": hold_sec,
        "target_vram": peak_v,
        "target_ram": peak_r,
    })

    # 3. Step down in reverse
    for i in range(num_steps - 2, -1, -1):
        v = vram_steps[min(i, len(vram_steps) - 1)] if vram_steps else 0.0
        r = ram_steps[min(i, len(ram_steps) - 1)] if ram_steps else 0.0
        schedule.append({
            "phase": "step_down",
            "duration": step_sec,
            "target_vram": v,
            "target_ram": r,
        })

    # Final release back to 0
    schedule.append({
        "phase": "step_down",
        "duration": step_sec,
        "target_vram": 0.0,
        "target_ram": 0.0,
    })

    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Grab GPU VRAM and host RAM in steps for governor testing.")
    parser.add_argument("--vram-steps", default="2,4,8,12", help="Comma-separated VRAM steps in GB (default: 2,4,8,12)")
    parser.add_argument("--ram-steps", default="4,8,16", help="Comma-separated host RAM steps in GB (default: 4,8,16)")
    parser.add_argument("--hold", type=float, default=60.0, help="Hold duration at peak in seconds (default: 60)")
    parser.add_argument("--step-interval", type=float, default=30.0, help="Step duration in seconds (default: 30)")
    parser.add_argument("--log", default="grab.csv", help="CSV log filename (default: grab.csv)")
    parser.add_argument("--max-vram-gb", type=float, default=None, help="Maximum allowed VRAM allocation in GB (default: total-1)")
    parser.add_argument("--dry-run", action="store_true", help="Log the timetable and metrics without allocating memory")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")

    args = parser.parse_args()

    vram_steps = parse_steps(args.vram_steps)
    ram_steps = parse_steps(args.ram_steps)

    # Query GPU
    total_vram_mb, free_vram_mb = query_nvidia_smi()

    if not args.dry_run:
        if total_vram_mb is None or free_vram_mb is None:
            print("Error: nvidia-smi failed or no NVIDIA GPU detected. Cannot monitor VRAM.", file=sys.stderr)
            sys.exit(1)

        # Refuse to run if nvidia-smi shows less than 1 GB free
        if free_vram_mb < 1024.0:
            print(
                f"Refusing to run: nvidia-smi reports only {free_vram_mb:.1f} MiB free (< 1024 MiB / 1 GB). "
                "Serving engine or another process is occupying VRAM; aborting to prevent OOM crash.",
                file=sys.stderr,
            )
            sys.exit(1)

        max_vram_gb = args.max_vram_gb if args.max_vram_gb is not None else max(0.0, (total_vram_mb / 1024.0) - 1.0)
    else:
        max_vram_gb = args.max_vram_gb if args.max_vram_gb is not None else (max(0.0, (total_vram_mb / 1024.0) - 1.0) if total_vram_mb is not None else 31.0)
        if total_vram_mb is None:
            print("[dry-run] Notice: nvidia-smi unavailable on this host; VRAM metrics will be reported as 0.0.")

    # Must not exceed --max-vram-gb (default total - 1)
    for s in vram_steps:
        if s > max_vram_gb:
            print(
                f"Refusing to run: requested VRAM step {s:.1f} GB exceeds max allowed {max_vram_gb:.2f} GB (total VRAM - 1 GB).",
                file=sys.stderr,
            )
            sys.exit(1)

    schedule = build_schedule(vram_steps, ram_steps, args.hold, args.step_interval)
    total_duration = sum(item["duration"] for item in schedule)

    # Print timetable
    print("=== Grabber Configuration ===")
    print(f"VRAM steps (GB): {vram_steps}")
    print(f"RAM steps (GB):  {ram_steps}")
    print(f"Hold duration:   {args.hold}s")
    print(f"Step interval:   {args.step_interval}s")
    print(f"Max VRAM limit:  {max_vram_gb:.2f} GB")
    print(f"Dry run mode:    {args.dry_run}")
    print(f"Log file:        {args.log}")
    print("\n=== Timetable ===")
    cur_t = 0.0
    for idx, item in enumerate(schedule, 1):
        t_start_s = format_duration(cur_t)
        t_end_s = format_duration(cur_t + item["duration"])
        cur_t += item["duration"]
        print(
            f"{idx:2d}  {t_start_s} - {t_end_s} ({item['duration']:4.0f}s)  "
            f"[{item['phase']:<9}]  target VRAM: {item['target_vram']:5.2f} GB, RAM: {item['target_ram']:5.2f} GB"
        )
    print(f"Total planned duration: {total_duration:.1f}s ({format_duration(total_duration)})\n")

    # Allocation state
    current_vram_gb = 0.0
    current_ram_gb = 0.0
    vram_allocations: list[Any] = []
    ram_allocations: list[bytearray] = []

    def set_vram_target(target_gb: float) -> None:
        nonlocal current_vram_gb
        if args.dry_run:
            current_vram_gb = target_gb
            return
        import torch

        target_bytes = int(target_gb * (1024**3))
        current_bytes = int(current_vram_gb * (1024**3))
        diff = target_bytes - current_bytes
        if diff > 0:
            t = torch.empty((diff,), dtype=torch.uint8, device="cuda")
            t.fill_(1)
            torch.cuda.synchronize()
            vram_allocations.append(t)
        elif diff < 0:
            to_free = -diff
            while vram_allocations and to_free > 0:
                last = vram_allocations.pop()
                last_bytes = last.nelement() * last.element_size()
                del last
                to_free -= last_bytes
                if to_free < 0:
                    realloc_bytes = -to_free
                    t = torch.empty((realloc_bytes,), dtype=torch.uint8, device="cuda")
                    t.fill_(1)
                    vram_allocations.append(t)
                    to_free = 0
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        current_vram_gb = target_gb

    def set_ram_target(target_gb: float) -> None:
        nonlocal current_ram_gb
        if args.dry_run:
            current_ram_gb = target_gb
            return

        target_bytes = int(target_gb * (1024**3))
        current_bytes = int(current_ram_gb * (1024**3))
        diff = target_bytes - current_bytes
        if diff > 0:
            b = bytearray(diff)
            # Touch each 4096-byte page to force page fault and commit
            num_pages = len(b) // 4096
            if num_pages > 0:
                b[:num_pages * 4096:4096] = b"\x01" * num_pages
            ram_allocations.append(b)
        elif diff < 0:
            to_free = -diff
            while ram_allocations and to_free > 0:
                last = ram_allocations.pop()
                last_bytes = len(last)
                del last
                to_free -= last_bytes
                if to_free < 0:
                    realloc_bytes = -to_free
                    b = bytearray(realloc_bytes)
                    num_pages = len(b) // 4096
                    if num_pages > 0:
                        b[:num_pages * 4096:4096] = b"\x01" * num_pages
                    ram_allocations.append(b)
                    to_free = 0
            gc.collect()
        current_ram_gb = target_gb

    def cleanup() -> None:
        nonlocal current_vram_gb, current_ram_gb
        ram_allocations.clear()
        vram_allocations.clear()
        current_vram_gb = 0.0
        current_ram_gb = 0.0
        gc.collect()
        if not args.dry_run:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass

    csv_header = "t,phase,vram_free_mb,win_free_mb,grabbed_vram_gb,grabbed_ram_gb"
    try:
        log_f = open(args.log, "w", encoding="utf-8")
        log_f.write(csv_header + "\n")
        log_f.flush()
    except Exception as exc:
        print(f"Error opening log file {args.log}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(csv_header)

    try:
        for item in schedule:
            phase = item["phase"]
            target_vram = item["target_vram"]
            target_ram = item["target_ram"]
            duration = item["duration"]

            # Adjust allocation to match target
            set_vram_target(target_vram)
            set_ram_target(target_ram)

            step_end = time.time() + duration
            while time.time() < step_end:
                t_now = time.time()
                _, free_vram = query_nvidia_smi()
                v_free = free_vram if free_vram is not None else 0.0
                r_free = query_win_free_mb()

                row = f"{t_now:.2f},{phase},{v_free:.1f},{r_free:.1f},{current_vram_gb:.2f},{current_ram_gb:.2f}"
                print(row)
                sys.stdout.flush()
                log_f.write(row + "\n")
                log_f.flush()

                # Sleep until next sampling interval
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl-C). Cleaning up allocations...", file=sys.stderr)
    except Exception as exc:
        print(f"\nException encountered: {exc}. Cleaning up allocations...", file=sys.stderr)
        raise
    finally:
        cleanup()
        log_f.close()
        print("All allocations cleanly released.")


if __name__ == "__main__":
    main()
