"""Deterministic native-Windows benchmark for FreeToken's mmap-backed PLE path."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCENARIOS = [
    {"name": "warmup", "input_tokens": 256, "output_tokens": 128, "repeats": 1, "recorded": False},
    {"name": "short", "input_tokens": 512, "output_tokens": 512, "repeats": 3, "recorded": True},
    {"name": "long", "input_tokens": 8192, "output_tokens": 512, "repeats": 3, "recorded": True},
    {"name": "context_check", "input_tokens": 32768, "output_tokens": 128, "repeats": 1, "recorded": True},
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def run_text(command: list[str], timeout: int = 30) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def powershell_json(script: str) -> Any:
    text = run_text(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"$ProgressPreference='SilentlyContinue'; {script} | ConvertTo-Json -Compress -Depth 5",
        ],
        timeout=60,
    )
    return json.loads(text)


def capture_system(disk_counter: str) -> dict[str, Any]:
    drive_match = re.search(r"\((?:\d+ )?([A-Za-z]):\)", disk_counter)
    drive_letter = drive_match.group(1).upper() if drive_match else None
    disk_lookup = (
        f"$disk=Get-Partition -DriveLetter '{drive_letter}' | Get-Disk | Select-Object -First 1; "
        if drive_letter
        else "$disk=$null; "
    )
    system_script = (
        "$os=Get-CimInstance Win32_OperatingSystem; "
        "$cpu=Get-CimInstance Win32_Processor | Select-Object -First 1; "
        + disk_lookup
        + "[pscustomobject]@{"
        "os_caption=$os.Caption; os_version=$os.Version; os_build=$os.BuildNumber; "
        "cpu=$cpu.Name.Trim(); ram_bytes=[int64]$os.TotalVisibleMemorySize*1024; "
        "disk_name=$disk.FriendlyName; disk_media_type=[string]$disk.MediaType; disk_bytes=[int64]$disk.Size}"
    )
    windows = powershell_json(system_script)
    gpu_line = run_text(
        [
            "nvidia-smi.exe",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()[0]
    gpu_name, gpu_memory_mib, driver = [part.strip() for part in gpu_line.split(",")]
    try:
        nvcc = run_text(["nvcc.exe", "--version"])
        cuda_build = next(
            line.strip() for line in nvcc.splitlines() if "release" in line.lower()
        )
    except (OSError, subprocess.SubprocessError, StopIteration):
        cuda_build = None
    return {
        **windows,
        "gpu": gpu_name,
        "gpu_memory_mib": int(gpu_memory_mib),
        "nvidia_driver": driver,
        "cuda_toolkit": cuda_build,
        "python": sys.version.split()[0],
    }


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


def memory_snapshot() -> dict[str, float | int | None]:
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    try:
        line = run_text(
            [
                "nvidia-smi.exe",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        ).splitlines()[0]
        used_mib, total_mib = [int(part.strip()) for part in line.split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        used_mib, total_mib = None, None
    return {
        "system_used_bytes": status.total_phys - status.avail_phys,
        "system_total_bytes": status.total_phys,
        "gpu_used_mib": used_mib,
        "gpu_total_mib": total_mib,
    }


class ResourceSampler:
    def __init__(self, disk_counter: str) -> None:
        self.disk_counter = disk_counter
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._typeperf: subprocess.Popen[str] | None = None
        self._disk_samples: list[dict[str, Any]] = []

    def start(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._typeperf = subprocess.Popen(
            ["typeperf.exe", self.disk_counter, "-si", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        assert self._typeperf.stdout is not None
        self._typeperf.stdout.readline()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._typeperf and self._typeperf.stdout
        disk_thread = threading.Thread(target=self._read_disk, daemon=True)
        disk_thread.start()
        while not self._stop.is_set():
            snapshot = memory_snapshot()
            snapshot["perf_counter"] = time.perf_counter()
            snapshot["utc"] = utc_now()
            self.samples.append(snapshot)
            self._stop.wait(1.0)

    def _read_disk(self) -> None:
        assert self._typeperf and self._typeperf.stdout
        for line in self._typeperf.stdout:
            try:
                row = next(csv.reader([line.strip()]))
                self._disk_samples.append(
                    {
                        "perf_counter": time.perf_counter(),
                        "reported_time": row[0],
                        "read_bytes_per_second": float(row[1]),
                    }
                )
            except (csv.Error, ValueError, IndexError):
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._typeperf:
            self._typeperf.terminate()
        if self._thread:
            self._thread.join(timeout=10)
        if self._typeperf:
            try:
                self._typeperf.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._typeperf.kill()

    def summary(self, started: float, ended: float) -> dict[str, Any]:
        memory = [s for s in self.samples if started <= s["perf_counter"] <= ended]
        disk = [s for s in self._disk_samples if started <= s["perf_counter"] <= ended]
        return {
            "peak_system_used_bytes": max(
                (s["system_used_bytes"] for s in memory), default=None
            ),
            "peak_gpu_used_mib": max(
                (s["gpu_used_mib"] for s in memory if s["gpu_used_mib"] is not None),
                default=None,
            ),
            "disk_read_bytes_per_second": [s["read_bytes_per_second"] for s in disk],
            "peak_disk_read_bytes_per_second": max(
                (s["read_bytes_per_second"] for s in disk), default=None
            ),
        }


def generate_prompt(tokenizer: Any, token_count: int, seed: int) -> str:
    rng = random.Random(seed)
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [rng.randint(0, vocab_size) for _ in range(token_count)]
    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == token_count:
            return prompt
        if len(token_ids) < token_count:
            token_ids.extend(
                rng.randint(0, vocab_size) for _ in range(token_count - len(token_ids))
            )
        else:
            token_ids = token_ids[:token_count]
    raise RuntimeError(f"could not generate exactly {token_count} tokenizer tokens")


def stream_request(
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # FreeToken's current max_tokens cap is exclusive: N yields N-1 usage.
        # Ask for one extra slot so the measured completion has the advertised size.
        "max_tokens": output_tokens + 1,
        "temperature": 0.0,
        "reasoning_effort": "off",
        "ignore_eos": True,
        "top_k": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] | None = None
    content: list[str] = []
    chunk_count = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                text = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    content.append(text)
                    chunk_count += 1
    ended = time.perf_counter()
    if first_token_at is None or usage is None:
        raise RuntimeError("stream omitted content timing or final usage")
    completion_tokens = int(usage["completion_tokens"])
    if completion_tokens != output_tokens:
        raise RuntimeError(
            f"expected {output_tokens} completion tokens, got {completion_tokens}"
        )
    decode_seconds = ended - first_token_at
    decode_intervals = max(completion_tokens - 1, 1)
    output_text = "".join(content)
    return {
        "started_perf_counter": started,
        "ended_perf_counter": ended,
        "ttft_seconds": first_token_at - started,
        "decode_seconds": decode_seconds,
        "end_to_end_seconds": ended - started,
        "tpot_milliseconds": decode_seconds * 1000 / decode_intervals,
        "output_tokens_per_second": decode_intervals / decode_seconds,
        "usage": usage,
        "stream_content_chunks": chunk_count,
        "output_characters": len(output_text),
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }


def usable_context(geometry: dict[str, Any]) -> int:
    if not geometry["num_pages"]:
        return 0
    return (geometry["num_pages"] - 1) * geometry["page_size"]


def scenario_manifest() -> dict[str, Any]:
    return {
        "seed": 3805090,
        "scenarios": SCENARIOS,
        "recorded_request_count": sum(
            item["repeats"] for item in SCENARIOS if item["recorded"]
        ),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for scenario in ("short", "long", "context_check"):
        rows = [r for r in result["runs"] if r["scenario"] == scenario and r["recorded"]]
        if not rows:
            continue
        summaries[scenario] = {
            "repeats": len(rows),
            "mean_actual_prompt_tokens": mean(
                [r["usage"]["prompt_tokens"] for r in rows]
            ),
            "mean_ttft_seconds": mean([r["ttft_seconds"] for r in rows]),
            "mean_tpot_milliseconds": mean([r["tpot_milliseconds"] for r in rows]),
            "mean_output_tokens_per_second": mean(
                [r["output_tokens_per_second"] for r in rows]
            ),
            "mean_end_to_end_seconds": mean(
                [r["end_to_end_seconds"] for r in rows]
            ),
            "peak_system_used_bytes": max(
                (r["resources"]["peak_system_used_bytes"] for r in rows if r["resources"]["peak_system_used_bytes"] is not None),
                default=None,
            ),
            "peak_gpu_used_mib": max(
                (r["resources"]["peak_gpu_used_mib"] for r in rows if r["resources"]["peak_gpu_used_mib"] is not None),
                default=None,
            ),
            "peak_disk_read_bytes_per_second": max(
                (r["resources"]["peak_disk_read_bytes_per_second"] for r in rows if r["resources"]["peak_disk_read_bytes_per_second"] is not None),
                default=None,
            ),
        }
    return summaries


def validate_result(result: dict[str, Any], require_both: bool = False) -> None:
    if require_both:
        contexts = result.get("contexts") or []
        usable = {item["usable_context_tokens"] for item in contexts}
        if usable != {49984, 262144}:
            raise ValueError(f"expected 49984 and 262144 usable contexts, got {usable}")
        for item in contexts:
            validate_result(item)
        return

    recorded = [item for item in result.get("runs", []) if item.get("recorded")]
    if len(recorded) != 7:
        raise ValueError(f"expected 7 recorded requests, got {len(recorded)}")
    counts = {name: sum(r["scenario"] == name for r in recorded) for name in ("short", "long", "context_check")}
    if counts != {"short": 3, "long": 3, "context_check": 1}:
        raise ValueError(f"unexpected scenario counts: {counts}")
    for row in recorded:
        for key in ("ttft_seconds", "tpot_milliseconds", "output_tokens_per_second", "end_to_end_seconds"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError(f"invalid {key}: {row[key]}")
        if row["usage"]["completion_tokens"] != row["requested_output_tokens"]:
            raise ValueError("completion-token count differs from request")


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    base_url = args.base_url.rstrip("/")
    status = get_json(f"{base_url}/cache/status")
    if status.get("state") != "serving":
        raise RuntimeError(f"server state is {status.get('state')!r}, not 'serving'")
    geometry = status["geometry"]
    actual_context = usable_context(geometry)
    expected_context = int(args.context_label)
    if actual_context != expected_context:
        raise RuntimeError(
            f"server has {actual_context} usable tokens, expected {expected_context}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    sampler = ResourceSampler(args.disk_counter)
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "model": args.model,
        "checkpoint": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "pr_279_commit": "feaeaa31c0cea385a1c9ee107d4b1053f83b35db",
        "context_label": args.context_label,
        "total_kv_pool_tokens": geometry["num_pages"] * geometry["page_size"],
        "reserved_dummy_page_tokens": geometry["page_size"],
        "usable_context_tokens": actual_context,
        "startup_seconds": args.startup_seconds,
        "seed": args.seed,
        "request_settings": {
            "reasoning_effort": "off",
            "temperature": 0.0,
            "top_k": 1,
            "ignore_eos": True,
            "max_running_requests": 1,
            "api_max_tokens": "requested output tokens + 1 (current cap is exclusive)",
        },
        "cache_status_before": status,
        "system": capture_system(args.disk_counter),
        "resource_scope": {
            "ram": "whole Windows system",
            "vram": "whole GPU",
            "disk": args.disk_counter,
        },
        "runs": [],
    }

    sampler.start()
    try:
        case_number = 0
        for scenario in SCENARIOS:
            for repeat in range(1, scenario["repeats"] + 1):
                prompt_seed = args.seed + scenario["input_tokens"] * 100 + repeat
                prompt = generate_prompt(tokenizer, scenario["input_tokens"], prompt_seed)
                print(
                    f"{scenario['name']} {repeat}/{scenario['repeats']}: "
                    f"{scenario['input_tokens']} in, {scenario['output_tokens']} out",
                    flush=True,
                )
                row = stream_request(
                    f"{base_url}/chat/completions",
                    args.model,
                    prompt,
                    scenario["output_tokens"],
                    args.timeout,
                )
                row.update(
                    {
                        "scenario": scenario["name"],
                        "repeat": repeat,
                        "recorded": scenario["recorded"],
                        "requested_input_tokens": scenario["input_tokens"],
                        "requested_output_tokens": scenario["output_tokens"],
                        "api_max_tokens": scenario["output_tokens"] + 1,
                        "prompt_seed": prompt_seed,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    }
                )
                row["resources"] = sampler.summary(
                    row["started_perf_counter"], row["ended_perf_counter"]
                )
                del row["started_perf_counter"]
                del row["ended_perf_counter"]
                result["runs"].append(row)
                case_number += 1
    finally:
        sampler.stop()

    result["cache_status_after"] = get_json(f"{base_url}/cache/status")
    result["summary"] = summarize(result)
    validate_result(result)
    return result


def combine(paths: list[str], output: str) -> None:
    contexts = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    combined = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "method": scenario_manifest(),
        "contexts": sorted(contexts, key=lambda item: item["usable_context_tokens"]),
    }
    validate_result(combined, require_both=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"PASS combined benchmark: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:2020/v1")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--tokenizer")
    parser.add_argument("--context-label", choices=("49984", "262144"))
    parser.add_argument("--seed", type=int, default=3805090)
    parser.add_argument("--startup-seconds", type=float)
    parser.add_argument("--disk-counter", default=r"\PhysicalDisk(1 D:)\Disk Read Bytes/sec")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate")
    parser.add_argument("--combine", nargs=2, metavar=("RESULT_50K", "RESULT_262144"))
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(scenario_manifest(), indent=2))
        return 0
    if args.validate:
        result = json.loads(Path(args.validate).read_text(encoding="utf-8"))
        validate_result(result, require_both="contexts" in result)
        print(f"PASS benchmark result: {args.validate}")
        return 0
    if args.combine:
        if not args.output:
            parser.error("--combine requires --output")
        combine(args.combine, args.output)
        return 0
    if not args.tokenizer or not args.context_label or not args.output:
        parser.error("benchmark mode requires --tokenizer, --context-label, and --output")

    try:
        result = benchmark(args)
    except (OSError, RuntimeError, ValueError, KeyError, urllib.error.URLError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS benchmark result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
