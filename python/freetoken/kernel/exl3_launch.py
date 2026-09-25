"""The one choke point for every ExLlamaV3 GPU launch FreeToken makes.

Every ``exl3_gemm`` (kernel/exl3_linear.py), ``reconstruct_had_slice`` (kernel/exl3.py) and
``exl3_mgemm`` (kernel/exl3_mgemm.py) call goes through :func:`launch`. It does three things.

1. Serialises EXL3 launches across CUDA streams (always on).
   Why: the first EXL3 Qwen Flash boot (2026-09-25 15:33) hung the RTX 5090 on its first
   prompt, and the only unbounded device spins in that path are ExLlamaV3's
   (.superpowers/sdd/2026-09-25-exl3-qwen-flash/hang-investigation.md, H-A and H-D):

   - every ExLlamaV3 kernel on a device shares ONE lock buffer, ``DevCtx::get_locks(device)``,
     a raw ``cudaMalloc`` of ``1M + 2*1024 + 66`` ints zeroed once
     (exllamav3_ext ``exl3_devctx.cu:60-73``, ``exl3_devctx.cuh:96-105``);
   - ``exl3_gemm`` spins on split-K locks from offset 0 (``barrier_acquire``, ``ptx.cuh:103-119``,
     used at ``exl3_gemm_inner.cuh:835-872``);
   - Blackwell ``exl3_mgemm`` (``__CUDA_ARCH__ > 890``, the 5090 is sm_120) also spins on a
     sense-reversal ``group_barrier`` over counters at ``locks + 1M + 2*blockIdx.z``
     (``ptx.cuh:319-347``), plus ``grid.sync()`` in cooperative launches.

   That state returns to zero only if the launches run to completion one after another. Two
   EXL3 kernels in flight at once on two streams interleave the counters and a block waits for
   a value that never arrives: no fault, no timeout, and Blackwell's compute preemption means
   no TDR either -- the card just spins under the desktop. A single-stream replay of the
   first-prefill kernel mix did NOT hang on the box (tiny-test-report.md, 2026-09-25 16:57),
   which leaves cross-stream overlap as the concrete suspect.

   The rule: remember the stream of the previous EXL3 launch on each device; when a launch
   arrives on a different stream, record an event on the old stream and make the new stream
   wait on it before launching. Stream order then puts the new kernel after everything the old
   stream had queued, including the previous EXL3 kernel.

   Why this is CUDA-graph-capture-safe:
   - ``Event.record`` and ``Stream.wait_event`` are asynchronous and capturable; nothing here
     synchronises the host, queries an event or allocates device memory.
   - CUDA forbids waiting, from a capturing stream, on an event recorded OUTSIDE the capture
     (CUDA programming guide, "Prohibited and unhandled operations": it needs
     ``cudaEventWaitExternal``), and forbids an uncaptured stream from waiting on a captured
     event (that merges the capture). So the wait is only inserted when the old and new
     streams are in the same state -- both capturing (a fork inside one capture: the wait
     becomes a graph edge) or both not capturing (plain eager ordering).
   - When a captured launch follows an eager one (or the reverse) the wait is skipped. That is
     sound because a captured kernel does not run at capture time; it runs at replay, and
     FreeToken replays every graph on ``engine.stream`` (engine/graph.py captures on the engine
     stream too), which orders it against every other engine-stream launch. What this choke
     point cannot see is a replay overlapping an eager EXL3 launch on ANOTHER stream -- which
     is exactly why the picture encode now runs on the engine stream
     (scheduler._picture_encode_stream) and why strict mode exists.
   - A fresh event per stream switch: switches are rare (none in the text-only serving path),
     so the event cost is irrelevant, and a fresh event never mixes a captured record with an
     eager one.

2. ``FREETOKEN_EXL3_STRICT_STREAM=1`` (debug): the first stream a device sees an EXL3 launch
   on outside capture becomes its home stream; an eager launch on any other stream raises a
   RuntimeError naming the op, label and shape instead of launching. Captured launches are not
   checked (they run at replay, on the replay stream). Use it in a debug boot to discover any
   cross-stream caller.

3. ``FREETOKEN_EXL3_TRACE=<file>`` (debug only, slow): one line before (BEGIN) and after (END)
   each launch -- monotonic time, pid, op, caller label, M/K/N, K bits, stream id, capturing --
   appended line-buffered, flushed and fsynced per line so it survives a frozen PC. When not
   capturing it also synchronises the current stream after each launch, so an END line proves
   the kernel FINISHED; after a hang the last BEGIN with no END names the stuck call. Stream
   switches log a WAIT (or SKIP-WAIT) line. The per-launch sync removes all CPU/GPU overlap:
   never leave it on for serving.

Labels: ``Exl3Linear.label`` is its module path (set by ``label_exl3_linears`` at workspace
preparation); routed-expert, MTP and other call sites push a :func:`scope` such as ``L3.routed``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

import torch

STRICT_ENV = "FREETOKEN_EXL3_STRICT_STREAM"
TRACE_ENV = "FREETOKEN_EXL3_TRACE"


class _CudaOps:
    """Stream queries behind a seam, so CPU tests can drive :func:`launch` with fake streams."""

    def applies(self, device: torch.device) -> bool:
        return device.type == "cuda"

    def current_stream(self, device: torch.device):
        return torch.cuda.current_stream(device)

    def is_capturing(self, stream) -> bool:
        # cudaStreamIsCapturing on that stream; switching the current stream is host-only.
        with torch.cuda.stream(stream):
            return bool(torch.cuda.is_current_stream_capturing())

    def new_event(self):
        return torch.cuda.Event()


_OPS = _CudaOps()
_LOCK = threading.RLock()
_LAST_STREAM: dict[object, object] = {}
_HOME_STREAM: dict[object, object] = {}
_SCOPE = threading.local()
_TRACE_FILE = None
_TRACE_PATH: str | None = None


def _reset_for_tests() -> None:
    global _TRACE_FILE, _TRACE_PATH
    with _LOCK:
        _LAST_STREAM.clear()
        _HOME_STREAM.clear()
        if _TRACE_FILE is not None:
            _TRACE_FILE.close()
        _TRACE_FILE = None
        _TRACE_PATH = None


class scope:  # noqa: N801 - used like a function: ``with exl3_launch.scope("L3.routed"):``
    """Prefix the labels of every EXL3 launch inside this block (e.g. ``L3.routed``).

    A plain class rather than ``@contextmanager``: it wraps every dense EXL3 GEMM, and eager
    (graphs-off) decode makes hundreds of those per step."""

    __slots__ = ("name",)

    def __init__(self, name: str | None):
        self.name = name

    def __enter__(self) -> None:
        if self.name is not None:
            stack = getattr(_SCOPE, "stack", None)
            if stack is None:
                stack = _SCOPE.stack = []
            stack.append(str(self.name))

    def __exit__(self, *exc) -> None:
        if self.name is not None:
            _SCOPE.stack.pop()


def _full_label(label: str | None) -> str:
    parts = list(getattr(_SCOPE, "stack", None) or ())
    if label:
        parts.append(label)
    return "/".join(parts) if parts else "?"


def _stream_id(stream) -> str:
    handle = getattr(stream, "cuda_stream", None)
    return hex(handle) if isinstance(handle, int) else f"py{id(stream):x}"


def _dev_key(device: torch.device):
    return (device.type, device.index)


def _trace_file():
    global _TRACE_FILE, _TRACE_PATH
    path = os.environ.get(TRACE_ENV) or None
    if path != _TRACE_PATH:
        if _TRACE_FILE is not None:
            _TRACE_FILE.close()
        _TRACE_FILE = open(path, "a", buffering=1) if path else None  # noqa: SIM115
        _TRACE_PATH = path
    return _TRACE_FILE


def _trace(fh, event: str, fields: str) -> None:
    fh.write(f"{time.monotonic():.6f} pid={os.getpid()} {event} {fields}\n")
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except OSError:  # pragma: no cover - e.g. a pipe; flush already happened
        pass


def launch(
    op: str,
    fn: Callable[[], object],
    *,
    device: torch.device,
    m: int,
    k: int,
    n: int,
    bits: int,
    label: str | None = None,
):
    """Run ``fn`` (exactly one ExLlamaV3 GPU launch) under the process-wide EXL3 rules."""
    device = torch.device(device)
    if not _OPS.applies(device):
        return fn()
    strict = os.environ.get(STRICT_ENV, "") not in ("", "0")
    with _LOCK:
        fh = _trace_file()
        stream = _OPS.current_stream(device)
        key = _dev_key(device)
        prev = _LAST_STREAM.get(key)
        switched = prev is not None and prev != stream
        if not (switched or strict or fh):
            # Hot path (every launch of a text-only serve): one stream query and a compare.
            result = fn()
            _LAST_STREAM[key] = stream
            return result
        capturing = _OPS.is_capturing(stream)
        desc = f"op={op} label={_full_label(label)} m={m} k={k} n={n} bits={bits}"

        if strict and not capturing:
            home = _HOME_STREAM.setdefault(key, stream)
            if home != stream:
                raise RuntimeError(
                    f"EXL3 launch on a second CUDA stream ({STRICT_ENV}=1): {desc} "
                    f"arrived on stream {_stream_id(stream)}, but this device's EXL3 home "
                    f"stream is {_stream_id(home)}"
                )

        if switched:
            prev_capturing = _OPS.is_capturing(prev)
            if prev_capturing == capturing:
                event = _OPS.new_event()
                event.record(prev)
                stream.wait_event(event)
                if fh:
                    _trace(fh, "WAIT", f"{desc} from={_stream_id(prev)} to={_stream_id(stream)} "
                                       f"capturing={'yes' if capturing else 'no'}")
            elif fh:
                _trace(fh, "SKIP-WAIT", f"{desc} from={_stream_id(prev)} to={_stream_id(stream)} "
                                        f"prev_capturing={prev_capturing} capturing={capturing}")

        if fh:
            fields = f"{desc} stream={_stream_id(stream)} capturing={'yes' if capturing else 'no'}"
            _trace(fh, "BEGIN", fields)
            started = time.monotonic()
        try:
            result = fn()
            _LAST_STREAM[key] = stream
            if fh and not capturing:
                # Debug only: proves the kernel finished before END is written.
                stream.synchronize()
        except BaseException as exc:
            if fh:
                _trace(fh, "FAIL", f"{fields} error={type(exc).__name__}: {exc}")
            raise
        if fh:
            _trace(fh, "END", f"{fields} ms={(time.monotonic() - started) * 1e3:.3f}")
        return result


__all__ = ["STRICT_ENV", "TRACE_ENV", "launch", "scope"]
