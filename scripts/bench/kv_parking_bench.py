"""Measure synthetic KV/GDN parking paths on the Windows RTX 5090 box.

The benchmark owns the shared GPU lock for the duration of every CUDA operation.  The
``--job-id`` argument names that owner in ``gpu.lock`` and defaults to ``M1``; use a
unique job id when running a review or rerun so another process cannot mistake this
run for its own.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from freetoken.moe.win_io import UnbufferedReader


ALIGNMENT = 4096
PATTERN_BYTES = 1 << 20
COPY_BYTES = 64 << 20
PINNED_TRANSFER_BYTES = 256 << 20
SSD_TRANSFER_BYTES = 32 << 20
STATE_BYTES = 115_642_376
RECOMPUTE_TOKENS_PER_SECOND = 1740.0
RESTORE_CEILING_MS = 1883.0
RESTORE_HARD_MAX_MS = 2000.0
STABILITY_LIMIT = 1.30


@dataclass
class Payload:
    name: str
    tensor: torch.Tensor


@dataclass
class PinnedWindow:
    backing: torch.Tensor
    view: torch.Tensor
    memory: memoryview

    @property
    def allocated_bytes(self) -> int:
        storage = self.backing.untyped_storage()
        return int(storage.nbytes())


if os.name == "nt":

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    _PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _PSAPI.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    _PSAPI.GetProcessMemoryInfo.restype = ctypes.c_int
    _KERNEL32.GetCurrentProcess.argtypes = []
    _KERNEL32.GetCurrentProcess.restype = ctypes.c_void_p
    _KERNEL32.GetProcessIoCounters.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_IoCounters),
    ]
    _KERNEL32.GetProcessIoCounters.restype = ctypes.c_int
else:
    _PSAPI = None
    _KERNEL32 = None


# The lock is intentionally a plain text file so a waiting job can identify the owner
# without importing torch or touching the GPU.
def lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts" / "kv-context-parallel" / "gpu.lock"


def acquire_gpu_lock(job_id: str) -> None:
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{job_id}\n".encode("ascii"))
            finally:
                os.close(fd)
            print(f"GPU_LOCK_ACQUIRED {job_id}", flush=True)
            return
        except FileExistsError:
            try:
                owner = path.read_text(encoding="ascii").strip()
            except OSError:
                owner = "unreadable"
            # Never reuse a same-named lock: two processes with the same job id must still
            # serialize, or one could delete the other's lock during cleanup.
            print(f"GPU_LOCK_WAIT owner={owner!r} sleep_s=60", flush=True)
            time.sleep(60)


def release_gpu_lock(job_id: str) -> None:
    path = lock_path()
    try:
        if path.read_text(encoding="ascii").strip() == job_id:
            path.unlink()
            print(f"GPU_LOCK_RELEASED {job_id}", flush=True)
    except FileNotFoundError:
        pass


def process_snapshot() -> list[dict[str, object]]:
    command = (
        "Get-CimInstance -ClassName Win32_Process | "
        "Select-Object ProcessId, ParentProcessId, Name, CommandLine | "
        "ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"process snapshot failed: {result.stderr.strip()}")
    text = result.stdout.strip()
    if not text:
        return []
    value = json.loads(text)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise RuntimeError("process snapshot was not a JSON object or array")
    return [item for item in value if isinstance(item, dict)]


def listening_ports() -> list[int]:
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"netstat failed: {result.stderr.strip()}")
    ports: set[int] = set()
    for line in result.stdout.splitlines():
        if "LISTENING" not in line.upper():
            continue
        match = re.search(r"^\s*TCP\s+[^\s]+:(\d+)\s+", line, re.IGNORECASE)
        if match:
            ports.add(int(match.group(1)))
    return sorted(ports)


def server_state() -> dict[str, object]:
    processes = process_snapshot()
    alive = {
        int(item["ProcessId"])
        for item in processes
        if item.get("ProcessId") is not None
    }
    ft_serve_pids: list[int] = []
    spawn_main_orphan_pids: list[int] = []
    for item in processes:
        name = str(item.get("Name") or "").lower()
        command_line = str(item.get("CommandLine") or "")
        if name not in {"python.exe", "pythonw.exe"}:
            continue
        if re.search(
            r"(?:freetoken\.cli\s+serve|(?:^|\s)ft(?:\.exe)?\s+serve(?:\s|$))",
            command_line,
            re.IGNORECASE,
        ):
            ft_serve_pids.append(int(item["ProcessId"]))
        if "spawn_main" in command_line:
            parent_id = int(item.get("ParentProcessId") or 0)
            if parent_id not in alive:
                spawn_main_orphan_pids.append(int(item["ProcessId"]))
    ports = listening_ports()
    return {
        "listeners_202x": [port for port in ports if 2020 <= port <= 2029],
        "listeners_203x": [port for port in ports if 2030 <= port <= 2039],
        "ft_serve_python_pids": sorted(ft_serve_pids),
        "spawn_main_orphan_pids": sorted(spawn_main_orphan_pids),
    }


def print_server_check(stage: str) -> dict[str, object]:
    state = server_state()
    print(f"SERVER_CHECK_{stage.upper()} " + json.dumps(state, sort_keys=True), flush=True)
    dirty = any(bool(value) for value in state.values())
    if dirty:
        raise RuntimeError(f"server/orphan check failed at {stage}: {state}")
    return state


def ensure_server_stopped() -> dict[str, object]:
    stop_script = Path(__file__).resolve().parents[2] / "scripts" / (
        "stop-qwen38-flash-next-windows.ps1"
    )
    if not stop_script.is_file():
        raise FileNotFoundError(f"approved stop script not found: {stop_script}")
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(stop_script),
            "-Port",
            "2020",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        print(f"STOP_SCRIPT_STDOUT {line}", flush=True)
    for line in result.stderr.splitlines():
        print(f"STOP_SCRIPT_STDERR {line}", flush=True)
    print(f"STOP_SCRIPT_EXIT {result.returncode}", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"approved stop script failed with exit {result.returncode}")
    return print_server_check("before")


def working_set_bytes() -> int:
    if _PSAPI is None:
        raise RuntimeError("the benchmark requires Windows working-set counters")
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ok = _PSAPI.GetProcessMemoryInfo(
        _KERNEL32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.WorkingSetSize)


def read_transfer_bytes() -> int:
    if _KERNEL32 is None:
        raise RuntimeError("the benchmark requires Windows process I/O counters")
    counters = _IoCounters()
    ok = _KERNEL32.GetProcessIoCounters(
        _KERNEL32.GetCurrentProcess(), ctypes.byref(counters)
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.ReadTransferCount)


def vram_info(device: torch.device) -> tuple[int, int]:
    # torch.cuda.mem_get_info returns (free, total); expose the less error-prone
    # (total, free) order to the benchmark's reporting code.
    free, total = torch.cuda.mem_get_info(device)
    return int(total), int(free)


def byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise ValueError("synthetic payload tensor is unexpectedly non-contiguous")
    return tensor.view(torch.uint8).reshape(-1)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def payload_bytes(payload: Sequence[Payload]) -> int:
    return sum(tensor_bytes(item.tensor) for item in payload)


def specs(prefix_tokens: int) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    if prefix_tokens <= 0 or prefix_tokens % 64:
        raise ValueError(f"prefix_tokens must be a positive multiple of 64: {prefix_tokens}")
    return (
        (
            "qsa_main_kv",
            (2, 12, prefix_tokens // 64, 64, 2, 256),
            torch.bfloat16,
        ),
        ("qsa_compressed_index", (12, prefix_tokens // 4, 128), torch.bfloat16),
        ("gdn_conv", (36, 10240, 3), torch.bfloat16),
        ("gdn_recurrent", (36, 48, 128, 128), torch.float32),
        ("ple_conv", (1, 10240, 9), torch.bfloat16),
        ("ple_ngram", (1, 2), torch.int32),
    )


def expected_entry_bytes(prefix_tokens: int) -> tuple[int, int, int]:
    kv_index = prefix_tokens * 25_344
    total = kv_index + STATE_BYTES
    return kv_index, STATE_BYTES, total


def fill_deterministically(payload: Sequence[Payload], device: torch.device) -> None:
    # A one-MiB byte pattern is copied repeatedly so even the 6.76-GiB entry never needs a
    # second large random allocation. Different salts make each tensor's byte stream distinct.
    base = torch.arange(PATTERN_BYTES, dtype=torch.uint8, device=device)
    for index, item in enumerate(payload):
        pattern = base ^ ((index + 1) * 37 & 0xFF)
        view = byte_view(item.tensor)
        for start in range(0, view.numel(), PATTERN_BYTES):
            stop = min(start + PATTERN_BYTES, view.numel())
            view[start:stop].copy_(pattern[: stop - start])
        del pattern
    del base
    torch.cuda.synchronize(device)


def new_payload(prefix_tokens: int, device: torch.device) -> list[Payload]:
    result = [
        Payload(name, torch.empty(shape, dtype=dtype, device=device))
        for name, shape, dtype in specs(prefix_tokens)
    ]
    fill_deterministically(result, device)
    return result


def hash_gpu_payload(payload: Sequence[Payload], device: torch.device) -> str:
    torch.cuda.synchronize(device)
    digest = hashlib.sha256()
    for item in payload:
        view = byte_view(item.tensor)
        for start in range(0, view.numel(), COPY_BYTES):
            stop = min(start + COPY_BYTES, view.numel())
            cpu_chunk = view[start:stop].cpu()
            digest.update(memoryview(cpu_chunk.numpy()).cast("B"))
            del cpu_chunk
    return digest.hexdigest()


def hash_cpu_payload(payload: Sequence[Payload]) -> str:
    digest = hashlib.sha256()
    for item in payload:
        view = byte_view(item.tensor)
        array = view.numpy()
        for start in range(0, view.numel(), COPY_BYTES):
            stop = min(start + COPY_BYTES, view.numel())
            digest.update(memoryview(array[start:stop]).cast("B"))
    return digest.hexdigest()


def verify_gpu_payload(
    source: Sequence[Payload], destination: Sequence[Payload], expected_checksum: str
) -> bool:
    if len(source) != len(destination):
        return False
    digest = hashlib.sha256()
    for source_item, destination_item in zip(source, destination, strict=True):
        if source_item.name != destination_item.name:
            return False
        source_view = byte_view(source_item.tensor)
        destination_view = byte_view(destination_item.tensor)
        if source_view.numel() != destination_view.numel():
            return False
        for start in range(0, source_view.numel(), COPY_BYTES):
            stop = min(start + COPY_BYTES, source_view.numel())
            source_chunk = source_view[start:stop]
            destination_chunk = destination_view[start:stop]
            if not torch.equal(source_chunk, destination_chunk):
                return False
            cpu_chunk = destination_chunk.cpu()
            digest.update(memoryview(cpu_chunk.numpy()).cast("B"))
            del cpu_chunk
    return digest.hexdigest() == expected_checksum


def allocate_zero_payload(source: Sequence[Payload]) -> list[Payload]:
    return [
        Payload(item.name, torch.zeros_like(item.tensor, device=item.tensor.device))
        for item in source
    ]


def allocate_pinned_snapshot(source: Sequence[Payload]) -> list[Payload]:
    return [
        Payload(
            item.name,
            torch.empty(item.tensor.shape, dtype=item.tensor.dtype, device="cpu", pin_memory=True),
        )
        for item in source
    ]


def pinned_bytes(payload: Sequence[Payload]) -> int:
    total = 0
    for item in payload:
        total += int(item.tensor.untyped_storage().nbytes())
    return total


def allocate_window(window_bytes: int) -> PinnedWindow:
    if window_bytes <= 0 or window_bytes % ALIGNMENT:
        raise ValueError(f"pinned window must be 4096-byte aligned: {window_bytes}")
    backing = torch.empty(window_bytes + ALIGNMENT, dtype=torch.uint8, pin_memory=True)
    offset = (-int(backing.data_ptr())) % ALIGNMENT
    view = backing[offset : offset + window_bytes]
    if int(view.data_ptr()) % ALIGNMENT:
        raise RuntimeError("could not make a 4096-byte-aligned pinned window")
    memory = memoryview(view.numpy()).cast("B")
    return PinnedWindow(backing, view, memory)


def global_copy_device_to_window(
    source_views: Sequence[torch.Tensor],
    offset: int,
    length: int,
    window: torch.Tensor,
) -> None:
    remaining = length
    global_offset = offset
    window_offset = 0
    for source_view in source_views:
        if global_offset >= source_view.numel():
            global_offset -= source_view.numel()
            continue
        take = min(remaining, source_view.numel() - global_offset)
        window[window_offset : window_offset + take].copy_(
            source_view[global_offset : global_offset + take], non_blocking=True
        )
        remaining -= take
        window_offset += take
        global_offset = 0
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError(f"source copy ended short by {remaining} bytes")


def global_copy_window_to_device(
    window: torch.Tensor,
    source_offset: int,
    length: int,
    destination_views: Sequence[torch.Tensor],
) -> None:
    remaining = length
    global_offset = source_offset
    window_offset = 0
    for destination_view in destination_views:
        if global_offset >= destination_view.numel():
            global_offset -= destination_view.numel()
            continue
        take = min(remaining, destination_view.numel() - global_offset)
        destination_view[global_offset : global_offset + take].copy_(
            window[window_offset : window_offset + take], non_blocking=True
        )
        remaining -= take
        window_offset += take
        global_offset = 0
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError(f"destination copy ended short by {remaining} bytes")


def write_all(fd: int, buffer: memoryview) -> None:
    offset = 0
    while offset < len(buffer):
        written = os.write(fd, buffer[offset:])
        if written <= 0:
            raise OSError(f"short SSD write at {offset} of {len(buffer)} bytes")
        offset += written


def open_output_file(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_BINARY", 0)
    return os.open(str(path), flags, 0o600)


def event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    value = float(start.elapsed_time(end))
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"invalid CUDA timing value: {value!r}")
    return value


def timed_pinned_park(
    source: Sequence[Payload],
    host_snapshot: Sequence[Payload],
    d2h_stream: torch.cuda.Stream,
) -> tuple[float, float]:
    started = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The largest source tensor is multi-gigabyte. Bound the side-stream DMA spans so a
    # transient WDDM stall on one giant copy cannot become the entire park measurement.
    with torch.cuda.stream(d2h_stream):
        start_event.record(d2h_stream)
        for source_item, host_item in zip(source, host_snapshot, strict=True):
            source_view = byte_view(source_item.tensor)
            host_view = byte_view(host_item.tensor)
            for offset in range(0, source_view.numel(), PINNED_TRANSFER_BYTES):
                stop = min(offset + PINNED_TRANSFER_BYTES, source_view.numel())
                host_view[offset:stop].copy_(source_view[offset:stop], non_blocking=True)
        end_event.record(d2h_stream)
    end_event.synchronize()
    return event_ms(start_event, end_event), (time.perf_counter() - started) * 1000.0


def timed_pinned_restore(
    source: Sequence[Payload],
    host_snapshot: Sequence[Payload],
    h2d_stream: torch.cuda.Stream,
) -> tuple[list[Payload], float, float]:
    destination = allocate_zero_payload(source)
    torch.cuda.synchronize(source[0].tensor.device)
    started = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # A single multi-gigabyte H2D enqueue showed a WDDM outlier at the largest size. Keep
    # the destination fresh/zeroed, but enqueue bounded 256-MiB byte copies so one DMA span
    # cannot dominate the restore cell without adding hundreds of host-side enqueue gaps.
    with torch.cuda.stream(h2d_stream):
        start_event.record(h2d_stream)
        for host_item, destination_item in zip(host_snapshot, destination, strict=True):
            host_view = byte_view(host_item.tensor)
            destination_view = byte_view(destination_item.tensor)
            for offset in range(0, host_view.numel(), PINNED_TRANSFER_BYTES):
                stop = min(offset + PINNED_TRANSFER_BYTES, host_view.numel())
                destination_view[offset:stop].copy_(
                    host_view[offset:stop], non_blocking=True
                )
        end_event.record(h2d_stream)
    end_event.synchronize()
    return destination, event_ms(start_event, end_event), (time.perf_counter() - started) * 1000.0


def timed_d2h_to_windows(
    source: Sequence[Payload],
    windows: Sequence[PinnedWindow],
    d2h_stream: torch.cuda.Stream,
) -> float:
    """Measure D2H alone through the bounded SSD queue, without file-system interference."""
    source_views = [byte_view(item.tensor) for item in source]
    total = sum(view.numel() for view in source_views)
    if len(windows) != 2:
        raise ValueError("SSD D2H probe requires exactly two pinned windows")
    elapsed_ms = 0.0
    for chunk_number, offset in enumerate(range(0, total, windows[0].view.numel())):
        window = windows[chunk_number % len(windows)]
        length = min(window.view.numel(), total - offset)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(d2h_stream):
            start_event.record(d2h_stream)
            for sub_offset in range(0, length, SSD_TRANSFER_BYTES):
                sub_length = min(SSD_TRANSFER_BYTES, length - sub_offset)
                global_copy_device_to_window(
                    source_views,
                    offset + sub_offset,
                    sub_length,
                    window.view[sub_offset:],
                )
            end_event.record(d2h_stream)
        end_event.synchronize()
        elapsed_ms += event_ms(start_event, end_event)
    return elapsed_ms


def write_after_event(
    end_event: torch.cuda.Event,
    fd: int,
    buffer: memoryview,
    length: int,
) -> None:
    end_event.synchronize()
    write_all(fd, buffer[:length])


def ssd_write(
    source: Sequence[Payload],
    path: Path,
    windows: Sequence[PinnedWindow],
    d2h_stream: torch.cuda.Stream,
) -> float:
    source_views = [byte_view(item.tensor) for item in source]
    total = sum(view.numel() for view in source_views)
    if len(windows) != 2:
        raise ValueError("SSD write requires exactly two pinned windows")
    fd = open_output_file(path)
    # The first 65,536-token trial serialized every D2H copy with its file write and showed
    # a >30% WDDM outlier. A two-window serial writer keeps the file order while overlapping
    # the next D2H transfer with the current write. Splitting each 256-MiB window into bounded
    # 32-MiB transfers avoids one large WDDM DMA span dominating the save wall time; timing
    # for the D2H component is taken by the isolated probe above.
    end_events = [torch.cuda.Event(enable_timing=False) for _ in windows]
    writer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-ssd-write")
    pending: list[Future[None] | None] = [None] * len(windows)
    started = time.perf_counter()
    try:
        for chunk_number, offset in enumerate(range(0, total, windows[0].view.numel())):
            window_index = chunk_number % len(windows)
            previous = pending[window_index]
            if previous is not None:
                previous.result()
            length = min(windows[window_index].view.numel(), total - offset)
            with torch.cuda.stream(d2h_stream):
                for sub_offset in range(0, length, SSD_TRANSFER_BYTES):
                    sub_length = min(SSD_TRANSFER_BYTES, length - sub_offset)
                    global_copy_device_to_window(
                        source_views,
                        offset + sub_offset,
                        sub_length,
                        windows[window_index].view[sub_offset:],
                    )
                end_events[window_index].record(d2h_stream)
            pending[window_index] = writer_pool.submit(
                write_after_event,
                end_events[window_index],
                fd,
                windows[window_index].memory,
                length,
            )
        for future in pending:
            if future is not None:
                future.result()
        os.fsync(fd)
        actual = os.fstat(fd).st_size
        if actual != total:
            raise OSError(f"short SSD file: {actual} of {total} bytes")
    finally:
        writer_pool.shutdown(wait=True, cancel_futures=True)
        os.close(fd)
    return (time.perf_counter() - started) * 1000.0


def read_one(
    reader: UnbufferedReader, window: PinnedWindow, offset: int, length: int
) -> tuple[int, float]:
    started = time.perf_counter()
    got = reader.read_into(window.memory, offset, length)
    return got, (time.perf_counter() - started) * 1000.0


def ssd_restore(
    source: Sequence[Payload],
    path: Path,
    windows: Sequence[PinnedWindow],
    h2d_stream: torch.cuda.Stream,
) -> tuple[list[Payload], float, float, float, int]:
    source_views = [byte_view(item.tensor) for item in source]
    destination = allocate_zero_payload(source)
    device = source[0].tensor.device
    torch.cuda.synchronize(device)
    total = sum(view.numel() for view in source_views)
    reader = UnbufferedReader(str(path))
    read_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-ssd-read")
    h2d_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    window_events: list[torch.cuda.Event | None] = [None] * len(windows)
    if len(windows) != 2:
        raise ValueError("SSD restore requires exactly two pinned windows")
    read_ms = 0.0
    io_before = read_transfer_bytes()
    started = time.perf_counter()
    future: Future[tuple[int, float]] | None = None
    current_offset = 0
    current_length = min(windows[0].view.numel(), total)
    current_index = 0
    try:
        future = read_pool.submit(read_one, reader, windows[0], 0, current_length)
        while True:
            got, one_read_ms = future.result()
            read_ms += one_read_ms
            if got != current_length:
                raise OSError(
                    f"short unbuffered read: {got} of {current_length} bytes at {current_offset}"
                )
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(h2d_stream):
                h2d_start.record(h2d_stream)
                global_copy_window_to_device(
                    windows[current_index].view,
                    current_offset,
                    current_length,
                    [byte_view(item.tensor) for item in destination],
                )
                h2d_end.record(h2d_stream)
            h2d_events.append((h2d_start, h2d_end))
            window_events[current_index] = h2d_end

            next_offset = current_offset + current_length
            if next_offset >= total:
                break
            next_index = 1 - current_index
            previous_event = window_events[next_index]
            if previous_event is not None:
                # This wait is only the required two-window ownership hand-off: it lets the
                # next unbuffered read overlap the prior H2D transfer without overwriting it.
                previous_event.synchronize()
            next_length = min(windows[next_index].view.numel(), total - next_offset)
            future = read_pool.submit(read_one, reader, windows[next_index], next_offset, next_length)
            current_offset = next_offset
            current_length = next_length
            current_index = next_index
        io_after = read_transfer_bytes()
        h2d_stream.synchronize()
    finally:
        read_pool.shutdown(wait=True, cancel_futures=True)
        reader.close()
    restore_wall_ms = (time.perf_counter() - started) * 1000.0
    physical_read = io_after - io_before
    if physical_read < 0:
        raise RuntimeError(f"process read counter moved backwards: {physical_read}")
    h2d_ms = sum(event_ms(start, end) for start, end in h2d_events)
    return destination, read_ms, h2d_ms, restore_wall_ms, physical_read


def finite_timing_fields(row: dict[str, object]) -> list[str]:
    fields = (
        "d2h_ms",
        "ssd_write_ms",
        "ssd_read_ms",
        "h2d_ms",
        "park_wall_ms",
        "restore_wall_ms",
    )
    bad = []
    for field in fields:
        value = row.get(field)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            bad.append(field)
    return bad


def row_timing_instability(rows: Sequence[dict[str, object]]) -> list[str]:
    unstable: list[str] = []
    for field in (
        "d2h_ms",
        "ssd_write_ms",
        "ssd_read_ms",
        "h2d_ms",
        "park_wall_ms",
        "restore_wall_ms",
    ):
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if len(values) >= 2 and min(values) > 0 and max(values) / min(values) > STABILITY_LIMIT:
            unstable.append(field)
    return unstable


def outlier_cause(method: str, fields: Sequence[str]) -> str:
    if method == "pinned_ram" or any(
        field in {"d2h_ms", "h2d_ms"} for field in fields
    ):
        return "likely transient WDDM/CUDA scheduling stall during device transfer"
    return "likely Windows SSD scheduling or fsync variance in the bounded queue"


def mark_outliers(rows: Sequence[dict[str, object]]) -> list[str]:
    unstable_fields = row_timing_instability(rows)
    marked: dict[int, list[str]] = {}
    for field in unstable_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        baseline = min(values)
        if baseline <= 0:
            continue
        # These measurements are dominated by stalls, which make a transfer slower rather
        # than faster. Use the fastest repetition as the stable baseline and retain every
        # slower row when the spread exceeds the same 30% investigation threshold.
        for index, row in enumerate(rows):
            value = row.get(field)
            if value is None:
                continue
            numeric = float(value)
            if numeric > baseline * STABILITY_LIMIT:
                marked.setdefault(index, []).append(field)

    notes: list[str] = []
    for index, row in enumerate(rows):
        fields = marked.get(index, [])
        if not fields:
            row["outlier"] = False
            row["outlier_fields"] = None
            row["outlier_reason"] = None
            continue
        reason = outlier_cause(str(row["method"]), fields)
        row["outlier"] = True
        row["outlier_fields"] = ",".join(fields)
        row["outlier_reason"] = reason
        notes.append(
            f"{row['method']} prefix_tokens={row['prefix_tokens']} repetition="
            f"{row['repetition']} fields={','.join(fields)}: {reason}."
        )
    return notes


def make_windows(window_bytes: int) -> list[PinnedWindow]:
    return [allocate_window(window_bytes), allocate_window(window_bytes)]


def sample_memory(
    baseline_ws: int, baseline_free: int, ws_peak: int, free_min: int
) -> tuple[int, int, int]:
    current_ws = working_set_bytes()
    total, current_free = vram_info(torch.device("cuda:0"))
    ws_peak = max(ws_peak, current_ws)
    free_min = min(free_min, current_free)
    return ws_peak, free_min, total


def run_pinned_once(
    source: Sequence[Payload],
    checksum: str,
    host_snapshot: Sequence[Payload],
    d2h_stream: torch.cuda.Stream,
    h2d_stream: torch.cuda.Stream,
    baseline_ws: int,
    baseline_free: int,
    repetition: int,
) -> dict[str, object]:
    device = source[0].tensor.device
    ws_peak = baseline_ws
    free_min = baseline_free
    total_vram, _ = vram_info(device)
    d2h_ms, park_wall_ms = timed_pinned_park(source, host_snapshot, d2h_stream)
    ws_peak, free_min, total_vram = sample_memory(
        baseline_ws, baseline_free, ws_peak, free_min
    )
    host_checksum = hash_cpu_payload(host_snapshot)
    if host_checksum != checksum:
        raise RuntimeError(
            f"pinned snapshot checksum mismatch: expected {checksum}, got {host_checksum}"
        )
    destination, h2d_ms, restore_wall_ms = timed_pinned_restore(
        source, host_snapshot, h2d_stream
    )
    ws_peak, free_min, total_vram = sample_memory(
        baseline_ws, baseline_free, ws_peak, free_min
    )
    checksum_match = verify_gpu_payload(source, destination, checksum)
    if not checksum_match:
        raise RuntimeError("pinned restore checksum mismatch")
    torch.cuda.synchronize(device)
    return {
        "prefix_tokens": None,
        "payload_bytes": payload_bytes(source),
        "payload_gib": payload_bytes(source) / (1 << 30),
        "method": "pinned_ram",
        "repetition": repetition,
        "d2h_ms": d2h_ms,
        "ssd_write_ms": None,
        "ssd_write_gib_s": None,
        "ssd_read_ms": None,
        "ssd_read_gib_s": None,
        "h2d_ms": h2d_ms,
        "park_wall_ms": park_wall_ms,
        "restore_wall_ms": restore_wall_ms,
        "recompute_ms": None,
        "restore_speedup": None,
        "peak_pinned_bytes": pinned_bytes(host_snapshot),
        "process_working_set_delta_bytes": max(0, ws_peak - baseline_ws),
        "physical_disk_read_bytes": None,
        "checksum_match": checksum_match,
        "peak_vram_used_bytes": total_vram - free_min,
        "vram_used_delta_bytes": max(0, baseline_free - free_min),
        "host_ram_used_bytes": pinned_bytes(host_snapshot),
        "outlier": False,
        "outlier_fields": None,
        "outlier_reason": None,
    }


def run_ssd_once(
    source: Sequence[Payload],
    checksum: str,
    path: Path,
    windows: Sequence[PinnedWindow],
    d2h_stream: torch.cuda.Stream,
    h2d_stream: torch.cuda.Stream,
    baseline_ws: int,
    baseline_free: int,
    repetition: int,
) -> dict[str, object]:
    device = source[0].tensor.device
    ws_peak = baseline_ws
    free_min = baseline_free
    total_vram, _ = vram_info(device)
    d2h_ms = timed_d2h_to_windows(source, windows, d2h_stream)
    ssd_write_ms = ssd_write(source, path, windows, d2h_stream)
    ws_peak, free_min, total_vram = sample_memory(
        baseline_ws, baseline_free, ws_peak, free_min
    )
    total = payload_bytes(source)
    if path.stat().st_size != total:
        raise OSError(f"SSD file size mismatch: {path.stat().st_size} of {total}")
    destination, ssd_read_ms, h2d_ms, restore_wall_ms, physical_read = ssd_restore(
        source, path, windows, h2d_stream
    )
    ws_peak, free_min, total_vram = sample_memory(
        baseline_ws, baseline_free, ws_peak, free_min
    )
    checksum_match = verify_gpu_payload(source, destination, checksum)
    del destination
    if not checksum_match:
        raise RuntimeError("SSD restore checksum mismatch")
    torch.cuda.synchronize(device)
    return {
        "prefix_tokens": None,
        "payload_bytes": total,
        "payload_gib": total / (1 << 30),
        "method": "ssd",
        "repetition": repetition,
        "d2h_ms": d2h_ms,
        "ssd_write_ms": ssd_write_ms,
        "ssd_write_gib_s": total / (ssd_write_ms / 1000.0) / (1 << 30),
        "ssd_read_ms": ssd_read_ms,
        "ssd_read_gib_s": total / (ssd_read_ms / 1000.0) / (1 << 30),
        "h2d_ms": h2d_ms,
        "park_wall_ms": ssd_write_ms,
        "restore_wall_ms": restore_wall_ms,
        "recompute_ms": None,
        "restore_speedup": None,
        "peak_pinned_bytes": sum(window.allocated_bytes for window in windows),
        "process_working_set_delta_bytes": max(0, ws_peak - baseline_ws),
        "physical_disk_read_bytes": physical_read,
        "checksum_match": checksum_match,
        "peak_vram_used_bytes": total_vram - free_min,
        "vram_used_delta_bytes": max(0, baseline_free - free_min),
        # SSD is the source of truth; host RAM is only the two-window bounded queue.
        "host_ram_used_bytes": sum(window.allocated_bytes for window in windows),
        "outlier": False,
        "outlier_fields": None,
        "outlier_reason": None,
    }


def run_method(
    prefix_tokens: int,
    source: Sequence[Payload],
    checksum: str,
    method: str,
    repetitions: int,
    ssd_dir: Path,
    window_bytes: int,
    device: torch.device,
) -> list[dict[str, object]]:
    baseline_ws = working_set_bytes()
    _, baseline_free = vram_info(device)
    d2h_stream = torch.cuda.Stream(device=device)
    h2d_stream = torch.cuda.Stream(device=device)
    path = ssd_dir / f"m1-{os.getpid()}-{prefix_tokens}.bin"
    host_snapshot: list[Payload] | None = None
    windows: list[PinnedWindow] | None = None
    try:
        if method == "pinned_ram":
            host_snapshot = allocate_pinned_snapshot(source)
            # Touching the full snapshot in the warm-up makes the working-set measurement count
            # resident pinned pages instead of only the allocator's virtual reservation.
            warmup_row = run_pinned_once(
                source,
                checksum,
                host_snapshot,
                d2h_stream,
                h2d_stream,
                baseline_ws,
                baseline_free,
                0,
            )
        elif method == "ssd":
            windows = make_windows(window_bytes)
            warmup_row = run_ssd_once(
                source,
                checksum,
                path,
                windows,
                d2h_stream,
                h2d_stream,
                baseline_ws,
                baseline_free,
                0,
            )
        else:
            raise ValueError(f"unknown method {method}")
        path.unlink(missing_ok=True)
        print(
            "WARMUP "
            + json.dumps(
                {
                    "prefix_tokens": prefix_tokens,
                    "method": method,
                    "restore_wall_ms": warmup_row["restore_wall_ms"],
                    "checksum_match": warmup_row["checksum_match"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

        def measured_rows() -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            for repetition in range(1, repetitions + 1):
                if method == "pinned_ram":
                    assert host_snapshot is not None
                    row = run_pinned_once(
                        source,
                        checksum,
                        host_snapshot,
                        d2h_stream,
                        h2d_stream,
                        baseline_ws,
                        baseline_free,
                        repetition,
                    )
                else:
                    assert windows is not None
                    row = run_ssd_once(
                        source,
                        checksum,
                        path,
                        windows,
                        d2h_stream,
                        h2d_stream,
                        baseline_ws,
                        baseline_free,
                        repetition,
                    )
                row["prefix_tokens"] = prefix_tokens
                row["recompute_ms"] = prefix_tokens / RECOMPUTE_TOKENS_PER_SECOND * 1000.0
                row["restore_speedup"] = float(row["recompute_ms"]) / float(row["restore_wall_ms"])
                bad_timing = finite_timing_fields(row)
                if bad_timing:
                    raise RuntimeError(f"non-finite timing cells: {bad_timing}")
                if int(row["payload_bytes"]) != expected_entry_bytes(prefix_tokens)[2]:
                    raise RuntimeError(
                        f"entry byte mismatch for {prefix_tokens}: {row['payload_bytes']}"
                    )
                if row["checksum_match"] is not True:
                    raise RuntimeError(f"checksum failed for {method} repetition {repetition}")
                rows.append(row)
                path.unlink(missing_ok=True)
            return rows

        rows = measured_rows()
        outlier_notes = mark_outliers(rows)
        for row in rows:
            print("RAW " + json.dumps(row, sort_keys=True), flush=True)
        if outlier_notes:
            for note in outlier_notes:
                print("OUTLIER " + note, flush=True)
        else:
            print(
                f"OUTLIER_NONE prefix_tokens={prefix_tokens} method={method}",
                flush=True,
            )
        return rows
    finally:
        path.unlink(missing_ok=True)
        del host_snapshot
        del windows
        torch.cuda.synchronize(device)
        del d2h_stream
        del h2d_stream
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def median_value(rows: Sequence[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.median(values) if values else None


def max_value(rows: Sequence[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return max(values) if values else None


def min_value(rows: Sequence[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return min(values) if values else None


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    first = rows[0]
    outlier_repetitions = [
        str(row["repetition"]) for row in rows if row.get("outlier") is True
    ]
    return {
        "prefix_tokens": first["prefix_tokens"],
        "payload_bytes": first["payload_bytes"],
        "payload_gib": first["payload_gib"],
        "method": first["method"],
        "repetitions": len(rows),
        "d2h_median_ms": median_value(rows, "d2h_ms"),
        "d2h_max_ms": max_value(rows, "d2h_ms"),
        "ssd_write_median_ms": median_value(rows, "ssd_write_ms"),
        "ssd_write_max_ms": max_value(rows, "ssd_write_ms"),
        "ssd_write_gib_s_median": median_value(rows, "ssd_write_gib_s"),
        "ssd_read_median_ms": median_value(rows, "ssd_read_ms"),
        "ssd_read_max_ms": max_value(rows, "ssd_read_ms"),
        "ssd_read_gib_s_median": median_value(rows, "ssd_read_gib_s"),
        "h2d_median_ms": median_value(rows, "h2d_ms"),
        "h2d_max_ms": max_value(rows, "h2d_ms"),
        "park_wall_median_ms": median_value(rows, "park_wall_ms"),
        "park_wall_max_ms": max_value(rows, "park_wall_ms"),
        "restore_wall_median_ms": median_value(rows, "restore_wall_ms"),
        "restore_wall_max_ms": max_value(rows, "restore_wall_ms"),
        "recompute_ms": first["recompute_ms"],
        "restore_speedup_median": median_value(rows, "restore_speedup"),
        "restore_speedup_max": max_value(rows, "restore_speedup"),
        "peak_pinned_bytes_max": max_value(rows, "peak_pinned_bytes"),
        "process_working_set_delta_bytes_max": max_value(
            rows, "process_working_set_delta_bytes"
        ),
        "physical_disk_read_bytes_min": min_value(rows, "physical_disk_read_bytes"),
        "physical_disk_read_bytes_max": max_value(rows, "physical_disk_read_bytes"),
        "checksum_all": all(row["checksum_match"] is True for row in rows),
        "peak_vram_used_bytes_max": max_value(rows, "peak_vram_used_bytes"),
        "vram_used_delta_bytes_max": max_value(rows, "vram_used_delta_bytes"),
        "host_ram_used_bytes_max": max_value(rows, "host_ram_used_bytes"),
        "outlier_repetitions": ",".join(outlier_repetitions) or None,
    }


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def markdown_table(rows: Iterable[dict[str, object]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def qualification(summary_rows: Sequence[dict[str, object]], method: str) -> tuple[bool, str]:
    row = next(row for row in summary_rows if row["method"] == method and row["prefix_tokens"] == 65_536)
    median = float(row["restore_wall_median_ms"])
    maximum = float(row["restore_wall_max_ms"])
    checksums = bool(row["checksum_all"])
    reasons = []
    if median > RESTORE_CEILING_MS:
        reasons.append(f"median restore {median:.3f} ms > {RESTORE_CEILING_MS:.0f} ms")
    if maximum > RESTORE_HARD_MAX_MS:
        reasons.append(f"maximum restore {maximum:.3f} ms > {RESTORE_HARD_MAX_MS:.0f} ms")
    if not checksums:
        reasons.append("not all three checksums passed")
    if reasons:
        return False, "; ".join(reasons)
    return True, (
        f"median restore {median:.3f} ms <= {RESTORE_CEILING_MS:.0f} ms, "
        f"maximum {maximum:.3f} ms <= {RESTORE_HARD_MAX_MS:.0f} ms, all checksums passed"
    )


def decide(summary_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    results: dict[str, object] = {}
    qualified: list[str] = []
    for method in ("pinned_ram", "ssd"):
        passed, reason = qualification(summary_rows, method)
        results[f"{method}_qualifies"] = passed
        results[f"{method}_reason"] = reason
        if passed:
            qualified.append(method)
    if not qualified:
        winner = "RP"
        rule_reason = "neither path qualifies at 65,536 tokens"
    elif len(qualified) == 1:
        winner = qualified[0]
        rule_reason = "only one path qualifies at 65,536 tokens"
    else:
        costs = {
            method: next(
                row["host_ram_used_bytes_max"]
                for row in summary_rows
                if row["method"] == method and row["prefix_tokens"] == 262_144
            )
            for method in qualified
        }
        low = min(float(costs[method]) for method in qualified)
        high = max(float(costs[method]) for method in qualified)
        if high / low <= 1.05:
            restore = {
                method: next(
                    float(row["restore_wall_median_ms"])
                    for row in summary_rows
                    if row["method"] == method and row["prefix_tokens"] == 65_536
                )
                for method in qualified
            }
            winner = min(qualified, key=lambda method: restore[method])
            rule_reason = (
                f"both qualify; host-RAM costs at 262,144 are within 5%, so lower "
                f"65,536-token median restore wins ({winner})"
            )
        else:
            winner = min(qualified, key=lambda method: float(costs[method]))
            rule_reason = (
                f"both qualify; lower measured host-RAM cost at 262,144 tokens wins "
                f"({winner})"
            )
    results["winner"] = winner
    results["rule_reason"] = rule_reason
    return results


def make_report(
    rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
    decision: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    raw_columns = (
        "prefix_tokens",
        "payload_bytes",
        "payload_gib",
        "method",
        "repetition",
        "d2h_ms",
        "ssd_write_ms",
        "ssd_write_gib_s",
        "ssd_read_ms",
        "ssd_read_gib_s",
        "h2d_ms",
        "park_wall_ms",
        "restore_wall_ms",
        "recompute_ms",
        "restore_speedup",
        "peak_pinned_bytes",
        "process_working_set_delta_bytes",
        "physical_disk_read_bytes",
        "checksum_match",
        "peak_vram_used_bytes",
        "vram_used_delta_bytes",
        "host_ram_used_bytes",
        "outlier",
        "outlier_fields",
        "outlier_reason",
    )
    summary_columns = (
        "prefix_tokens",
        "payload_bytes",
        "payload_gib",
        "method",
        "repetitions",
        "d2h_median_ms",
        "d2h_max_ms",
        "ssd_write_median_ms",
        "ssd_write_max_ms",
        "ssd_write_gib_s_median",
        "ssd_read_median_ms",
        "ssd_read_max_ms",
        "ssd_read_gib_s_median",
        "h2d_median_ms",
        "h2d_max_ms",
        "park_wall_median_ms",
        "park_wall_max_ms",
        "restore_wall_median_ms",
        "restore_wall_max_ms",
        "recompute_ms",
        "restore_speedup_median",
        "restore_speedup_max",
        "peak_pinned_bytes_max",
        "process_working_set_delta_bytes_max",
        "physical_disk_read_bytes_min",
        "physical_disk_read_bytes_max",
        "checksum_all",
        "peak_vram_used_bytes_max",
        "vram_used_delta_bytes_max",
        "host_ram_used_bytes_max",
        "outlier_repetitions",
    )
    geometry_rows = []
    for prefix in args.prefix_tokens:
        kv_index, state, total = expected_entry_bytes(prefix)
        by_method = {
            row["method"]: row
            for row in summary_rows
            if row["prefix_tokens"] == prefix
        }
        geometry_rows.append(
            {
                "prefix_tokens": prefix,
                "kv_index_bytes": kv_index,
                "state_bytes": state,
                "payload_bytes": total,
                "payload_gib": total / (1 << 30),
                "vram_peak_pinned_ram_bytes": by_method["pinned_ram"][
                    "peak_vram_used_bytes_max"
                ],
                "vram_peak_ssd_bytes": by_method["ssd"]["peak_vram_used_bytes_max"],
                "host_ram_pinned_ram_bytes": by_method["pinned_ram"][
                    "host_ram_used_bytes_max"
                ],
                "host_ram_ssd_bytes": by_method["ssd"]["host_ram_used_bytes_max"],
            }
        )
    geometry_columns = (
        "prefix_tokens",
        "kv_index_bytes",
        "state_bytes",
        "payload_bytes",
        "payload_gib",
        "vram_peak_pinned_ram_bytes",
        "vram_peak_ssd_bytes",
        "host_ram_pinned_ram_bytes",
        "host_ram_ssd_bytes",
    )
    outlier_lines = [
        (
            f"- `{row['method']}` prefix `{row['prefix_tokens']}` repetition "
            f"`{row['repetition']}`: `{row['outlier_fields']}` was an outlier; "
            f"{row['outlier_reason']}."
        )
        for row in rows
        if row.get("outlier") is True
    ]
    if not outlier_lines:
        outlier_lines = ["- None; all three measured repetitions stayed within the spread check."]
    prefix_text = " ".join(str(prefix) for prefix in args.prefix_tokens)
    command_line = (
        "$env:PYTHONPATH='D:\\FreeToken\\python;D:\\FreeToken\\scripts\\windows-ple-mmap'; "
        "& 'C:\\Users\\jay\\AppData\\Local\\FreeToken\\venv\\Scripts\\python.exe' "
        "'D:\\FreeToken\\scripts\\bench\\kv_parking_bench.py' "
        f"--job-id {args.job_id} --prefix-tokens {prefix_text} "
        f"--repetitions {args.repetitions} --ssd-dir '{args.ssd_dir}' "
        f"--pinned-window-mib {args.pinned_window_mib} "
        f"--recompute-tokens-per-second {args.recompute_tokens_per_second:g} "
        f"--output '{args.output}'"
    )
    lines = [
        "# KV parking benchmark — 2026-09-03",
        "",
        "This is the M1 synthetic one-TP-rank measurement from the P0 specification. No model "
        "or serving code was loaded. The port-2020 server was stopped with the approved stop "
        f"script while the `{args.job_id}` GPU lock was held; it was not restarted.",
        "",
        f"- GPU lock owner: `{args.job_id}`",
        f"- GPU: `{torch.cuda.get_device_name(device)}`",
        "- Pre- and post-run checks require no 202x/203x listeners, no `ft serve` Python "
        "processes, and no orphaned `spawn_main` workers.",
        f"- Repetitions: `{args.repetitions}` measured after one warm-up per size and method",
        f"- Recompute denominator: `{args.recompute_tokens_per_second:g}` prompt tokens/second",
        f"- SSD path: `{args.ssd_dir}`; reads use `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` "
        "through the existing Windows reader, so the read number is not a standby-cache read.",
        f"- SSD queue: two 4-KiB-aligned pinned `{args.pinned_window_mib}`-MiB windows; the file "
        "bytes are not counted as resident host RAM.",
        "- Each method and size has one warm-up and exactly three measured repetitions; no "
        "measured row was discarded or replaced.",
        f"- M1 command: `{command_line}`",
        "",
        "## Geometry and resident memory",
        "",
        markdown_table(geometry_rows, geometry_columns),
        "",
        "The VRAM columns are the maximum process-visible used VRAM during each method's "
        "source-plus-fresh-destination run. The host-RAM columns count the active pinned "
        "snapshot for `pinned_ram`, and the two bounded SSD queue windows for `ssd`.",
        "",
        "## Raw repetitions",
        "",
        markdown_table(rows, raw_columns),
        "",
        "## Outlier notes",
        "",
        *outlier_lines,
        "",
        "For SSD, `d2h_ms` is an isolated two-window CUDA-event probe so disk scheduling cannot "
        "pollute the transfer cell; `ssd_write_ms` is the actual sequential-write wall time. "
        "`ssd_read_ms` is the sum of the physical unbuffered `ReadFile` spans; `restore_wall_ms` "
        "is the end-to-end read-plus-H2D wall time with the two-window overlap.",
        "",
        "## Median and maximum across the three measured repetitions",
        "",
        markdown_table(summary_rows, summary_columns),
        "",
        "## P0 decision",
        "",
        f"P0 qualification — `pinned_ram`: "
        f"{'QUALIFIES' if decision['pinned_ram_qualifies'] else 'does not qualify'} at 65,536 tokens — "
        f"{decision['pinned_ram_reason']}.",
        f"P0 qualification — `ssd`: "
        f"{'QUALIFIES' if decision['ssd_qualifies'] else 'does not qualify'} at 65,536 tokens — "
        f"{decision['ssd_reason']}.",
        f"Winner under P0's rule: **{decision['winner']}** — {decision['rule_reason']}.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    global RECOMPUTE_TOKENS_PER_SECOND
    parser = argparse.ArgumentParser(description="Measure synthetic KV/GDN parking paths")
    parser.add_argument(
        "--job-id",
        default="M1",
        help="owner written to gpu.lock (default: M1; use a unique id for reruns)",
    )
    parser.add_argument("--prefix-tokens", nargs="+", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--ssd-dir", type=Path, required=True)
    parser.add_argument("--pinned-window-mib", type=int, required=True)
    parser.add_argument("--recompute-tokens-per-second", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.job_id.strip() or any(char in args.job_id for char in "\r\n"):
        raise ValueError("--job-id must be a non-empty single-line value")
    try:
        args.job_id.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("--job-id must contain ASCII characters only") from exc
    if args.repetitions != 3:
        raise ValueError("P0 requires exactly three measured repetitions")
    if args.pinned_window_mib <= 0:
        raise ValueError("--pinned-window-mib must be positive")
    if args.recompute_tokens_per_second <= 0:
        raise ValueError("--recompute-tokens-per-second must be positive")
    if args.recompute_tokens_per_second != RECOMPUTE_TOKENS_PER_SECOND:
        # Keep the command's supplied denominator authoritative while retaining the P0 default
        # in the module-level documentation and formulas.
        RECOMPUTE_TOKENS_PER_SECOND = args.recompute_tokens_per_second
    return args


def main() -> int:
    args = parse_args()
    acquire_gpu_lock(args.job_id)
    try:
        ensure_server_stopped()
        if os.name != "nt":
            raise RuntimeError("M1 is specified for the Windows RTX 5090 box")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        args.ssd_dir.mkdir(parents=True, exist_ok=True)
        print(
            "BENCHMARK "
            + json.dumps(
                {
                    "job_id": args.job_id,
                    "prefix_tokens": args.prefix_tokens,
                    "repetitions": args.repetitions,
                    "ssd_dir": str(args.ssd_dir),
                    "pinned_window_bytes": args.pinned_window_mib * (1 << 20),
                    "recompute_tokens_per_second": args.recompute_tokens_per_second,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        all_rows: list[dict[str, object]] = []
        try:
            for prefix_tokens in args.prefix_tokens:
                kv_index, state, expected_total = expected_entry_bytes(prefix_tokens)
                print(
                    "ALLOCATE "
                    + json.dumps(
                        {
                            "prefix_tokens": prefix_tokens,
                            "kv_index_bytes": kv_index,
                            "state_bytes": state,
                            "payload_bytes": expected_total,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                source = new_payload(prefix_tokens, device)
                actual_total = payload_bytes(source)
                if actual_total != expected_total:
                    raise RuntimeError(
                        f"payload byte mismatch for {prefix_tokens}: {actual_total} != {expected_total}"
                    )
                checksum = hash_gpu_payload(source, device)
                print(
                    "SOURCE "
                    + json.dumps(
                        {
                            "prefix_tokens": prefix_tokens,
                            "payload_bytes": actual_total,
                            "checksum": checksum,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                for method in ("pinned_ram", "ssd"):
                    method_rows = run_method(
                        prefix_tokens,
                        source,
                        checksum,
                        method,
                        args.repetitions,
                        args.ssd_dir,
                        args.pinned_window_mib * (1 << 20),
                        device,
                    )
                    all_rows.extend(method_rows)
                del source
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize(device)
        finally:
            # No model is loaded, so this cleanup only releases the synthetic source before the
            # lock is removed even when a later size fails.
            gc.collect()
            torch.cuda.empty_cache()
        grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
        for row in all_rows:
            key = (int(row["prefix_tokens"]), str(row["method"]))
            grouped.setdefault(key, []).append(row)
        summary_rows = [summarize(grouped[key]) for key in sorted(grouped)]
        decision = decide(summary_rows)
        report = make_report(all_rows, summary_rows, decision, args, device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print("RESULT_DOCUMENT " + str(args.output), flush=True)
        print("P0_DECISION " + json.dumps(decision, sort_keys=True), flush=True)
        print("COMPLETE", flush=True)
        return 0
    finally:
        try:
            print_server_check("after")
        finally:
            release_gpu_lock(args.job_id)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
