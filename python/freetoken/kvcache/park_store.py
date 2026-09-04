"""Host-RAM and SSD parking for page-aligned QSA/GDN prefix snapshots.

A parked entry is deliberately opaque to the attention and recurrent-state code: it is the
exact bytes returned by ``QSAKVCache.page_byte_views`` plus one complete
``LinearStatePool.slot_byte_views`` snapshot.  The scheduler copies those bytes before returning
any page or state slot to a free list, then a later match restores into ordinary newly-allocated
pages and inserts the prefix into the unchanged hybrid radix tree.

The SSD format uses a versioned 4-KiB header, verbatim int32 token ids, a SHA-256 payload digest,
and a 4-KiB-aligned payload. Every read validates the model/layout fingerprint, token ids, and
payload, so stale or damaged files and rolling-hash collisions become misses rather than numerics
changes. Restore alternates two bounded pinned windows so disk read N+1 overlaps H2D copy N; the
background writer yields those shared windows while a restore is active. The file is the source of
truth; no full SSD entry remains in RAM.
"""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Full, Queue
from threading import Event, Lock, RLock, Thread
from typing import Iterable

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_MAGIC = b"FTKVPARK"
# 3: the payload digest became the parallel-friendly chunked scheme in _ChunkedDigest.
# Version 2 files are rejected by _parse_header and deleted by _load_or_scan, which is the
# intended upgrade path: parking is off by default and an entry is cheap to rebuild.
_VERSION = 3
_HEADER_BYTES = 4096
_ALIGNMENT = 4096
_COPY_BYTES = 32 << 20
# Slice size of the payload digest. Independent of --kv-park-window-mib on purpose, so a
# store booted with a different window size still verifies files an earlier boot wrote.
_DIGEST_CHUNK_BYTES = 32 << 20
# Eight threads was the knee on this 32-thread box: 1 -> 2.40 GiB/s, 4 -> 9.1, 8 -> 18.3,
# 16 -> 27.4. Eight buys the whole win a restore needs without taking half the machine
# away from the three other requests that may be decoding at the time.
_DIGEST_WORKERS = min(8, max(1, os.cpu_count() or 1))


def _sha256_digest(raw) -> bytes:
    return hashlib.sha256(raw).digest()


class _ChunkedDigest:
    """SHA-256 over the concatenated SHA-256 digests of fixed-size payload slices.

    One thread hashes 2.40 GiB/s here (measured 2026-09-04), so the plain sequential
    ``sha256`` spent 690 ms of the 1,882 ms live SSD restore of a 1.655 GiB entry inside
    ``update()`` alone -- more than twice the 304 ms unbuffered read it guards. ``hashlib``
    releases the GIL above 2 KiB, so eight threads over 32 MiB slices reach 18.3 GiB/s and
    the same guard costs about 90 ms.

    The caller reuses its pinned window the moment ``update()`` returns, so every slice is
    hashed before then: the pool buys parallelism, not overlap.
    """

    def __init__(self, pool: ThreadPoolExecutor | None, chunk_bytes: int) -> None:
        if chunk_bytes < _ALIGNMENT or chunk_bytes % _ALIGNMENT:
            raise ValueError("park digest chunk must be a positive 4096-byte multiple")
        self._pool = pool
        self._chunk = int(chunk_bytes)
        self._carry = bytearray()
        self._digests: list[bytes] = []

    def update(self, raw) -> None:
        view = memoryview(raw).cast("B")
        if self._carry:
            take = min(self._chunk - len(self._carry), len(view))
            self._carry += view[:take]
            view = view[take:]
            if len(self._carry) == self._chunk:
                self._digests.append(_sha256_digest(self._carry))
                self._carry = bytearray()
        whole = (len(view) // self._chunk) * self._chunk
        if whole:
            slices = [view[o : o + self._chunk] for o in range(0, whole, self._chunk)]
            if self._pool is None or len(slices) == 1:
                self._digests.extend(_sha256_digest(part) for part in slices)
            else:
                self._digests.extend(self._pool.map(_sha256_digest, slices))
            view = view[whole:]
        if view:
            self._carry += view

    def hexdigest(self) -> str:
        digests = list(self._digests)
        if self._carry:
            digests.append(_sha256_digest(self._carry))
        return hashlib.sha256(b"".join(digests)).hexdigest()


class ParkEntryRejected(RuntimeError):
    """A parked entry failed a compatibility or integrity check and was dropped."""


@dataclass
class ParkedEntry:
    key: str
    token_ids: torch.Tensor
    token_count: int
    payload_bytes: int
    total_bytes: int
    last_used_ns: int
    payload_sha256: str | None = None
    path: Path | None = None
    ram_buffer: torch.Tensor | None = None


@dataclass
class PendingPark:
    """One background save whose source pages stay owned until ``copy_done``."""

    token_ids: torch.Tensor
    page_bases: torch.Tensor
    state_slot: int
    source_ready: object | None = None
    reserved_bytes: int = 0
    copy_done: Event = field(default_factory=Event)
    done: Event = field(default_factory=Event)
    success: bool = False
    error: Exception | None = None

    def wait_copied(self) -> None:
        self.copy_done.wait()

    def wait(self) -> bool:
        self.done.wait()
        return self.success


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise ValueError(f"park view must be contiguous, got shape={tuple(tensor.shape)}")
    return tensor.view(torch.uint8).reshape(-1)


@dataclass(frozen=True)
class _ByteSpan:
    """Flattened uint8 views of one parked snapshot plus their cumulative byte offsets.

    A 65,600-token entry is 1,025 pages x 36 tensors = 36,900 views. Until 2026-09-04 the
    copy walk restarted at view 0 for every pinned-window chunk and rebuilt a byte view for
    each view it skipped -- about 166,000 torch calls per seven-chunk restore, and 707 ms of
    the measured 1,882 ms. Building the byte views and their offsets once, then binary
    searching the first view of a chunk, removes that walk.
    """

    views: tuple[torch.Tensor, ...]
    offsets: tuple[int, ...]

    @classmethod
    def build(cls, views: Iterable[torch.Tensor]) -> "_ByteSpan":
        kept: list[torch.Tensor] = []
        offsets = [0]
        for view in views:
            raw = _byte_view(view)
            if raw.numel() == 0:
                continue
            kept.append(raw)
            offsets.append(offsets[-1] + raw.numel())
        return cls(tuple(kept), tuple(offsets))

    @property
    def nbytes(self) -> int:
        return self.offsets[-1]

    def locate(self, offset: int) -> int:
        """Index of the view holding byte ``offset``; ``len(views)`` when past the end."""
        return bisect.bisect_right(self.offsets, offset) - 1


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _tokens_cpu(input_ids: torch.Tensor) -> torch.Tensor:
    return input_ids.detach().to(device="cpu", dtype=torch.int32).contiguous().clone()


def _temp_writer_pid(filename: str) -> int | None:
    """Return the writer PID only for this store's two exact hidden temp-name shapes."""
    parts = filename.split(".")
    if len(parts) != 4 or parts[0] or parts[3] != "tmp" or not parts[2].isdigit():
        return None
    owner = parts[1]
    if owner != "park" and (
        len(owner) != 32 or any(ch not in "0123456789abcdef" for ch in owner)
    ):
        return None
    pid = int(parts[2])
    return pid if 0 < pid <= 0xFFFFFFFF else None


def _pid_is_alive(pid: int) -> bool:
    """Check a temp-file owner without sending a signal on Windows."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            # Access denied still means the process exists; INVALID_PARAMETER means no such PID.
            return ctypes.get_last_error() != 87
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def rolling_page_keys(
    input_ids: torch.Tensor, page_size: int, fingerprint: str
) -> list[str]:
    """BLAKE2b-128 key at every complete page boundary, chained from the fingerprint."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    tokens = _tokens_cpu(input_ids)
    root = hashlib.blake2b(fingerprint.encode("utf-8"), digest_size=16).digest()
    keys: list[str] = []
    parent = root
    fp = fingerprint.encode("utf-8")
    raw = memoryview(tokens.numpy()).cast("B")
    page_bytes = page_size * torch.int32.itemsize
    for offset in range(0, len(raw) - page_bytes + 1, page_bytes):
        parent = hashlib.blake2b(
            fp + parent + raw[offset : offset + page_bytes], digest_size=16
        ).digest()
        keys.append(parent.hex())
    return keys


def build_model_fingerprint(
    *,
    model_path: str,
    page_size: int,
    tp_rank: int,
    tp_size: int,
    kv_pool,
    state_pool,
) -> str:
    """Stable digest of the checkpoint identity and every parked byte-layout dimension."""
    root = Path(model_path).expanduser().resolve()

    def file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        return digest.hexdigest()

    def shard_sample_hash(path: Path, size: int) -> str:
        """Cheap content identity without rereading an entire frontier checkpoint at boot."""
        sample_bytes = 64 << 10
        offsets = sorted(
            {
                0,
                max(0, size // 2 - sample_bytes // 2),
                max(0, size - sample_bytes),
            }
        )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                chunk = handle.read(min(sample_bytes, size - offset))
                digest.update(struct.pack("<QQ", offset, len(chunk)))
                digest.update(chunk)
        return digest.hexdigest()

    identity: list[dict[str, int | str]] = []
    config_path = root / "config.json"
    if config_path.is_file():
        identity.append({"file": config_path.name, "sha256": file_hash(config_path)})
    shard_paths: set[Path] = set()
    ftw_index = root / "freetoken_weight.json"
    if ftw_index.is_file():
        # FTW is the loader's source of truth when present; bind persistent KV to its index and
        # every referenced freetoken-*.ftw shard rather than any leftover safetensors beside it.
        identity.append({"file": ftw_index.name, "sha256": file_hash(ftw_index)})
        try:
            index_doc = json.loads(ftw_index.read_text(encoding="utf-8"))
            shard_paths.update(root / str(row["file"]) for row in index_doc.get("shards", []))
        except Exception:
            # The loader will report malformed checkpoint metadata. The fingerprint remains stable
            # and distinct through the index file's own digest.
            pass
    else:
        index_candidates = sorted(root.glob("*.safetensors.index.json"))
        for index_path in index_candidates:
            identity.append({"file": index_path.name, "sha256": file_hash(index_path)})
            try:
                index_doc = json.loads(index_path.read_text(encoding="utf-8"))
                shard_paths.update(
                    root / str(name)
                    for name in set(index_doc.get("weight_map", {}).values())
                )
            except Exception:
                # The loader will report malformed checkpoint metadata. The fingerprint remains
                # stable and distinct through the index file's own digest.
                pass
        if not index_candidates:
            shard_paths.update(root.glob("*.safetensors"))
    for shard in sorted(shard_paths):
        try:
            stat = shard.stat()
        except OSError:
            identity.append({"file": shard.name, "missing": 1})
            continue
        # Full shard hashing would reread a frontier checkpoint at every boot. File identity,
        # change time, and fixed content samples still reject normal same-size/mtime deployment
        # replacements without pulling hundreds of GiB through the page cache.
        identity.append(
            {
                "file": shard.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "sample_sha256": shard_sample_hash(shard, stat.st_size),
            }
        )
    page_layout = [
        (tuple(int(v) for v in view.shape), str(view.dtype))
        for view in kv_pool.page_byte_views(0)
    ]
    state_layout = [
        (tuple(int(v) for v in view.shape), str(view.dtype))
        for view in state_pool.slot_byte_views(state_pool.padding_slot)
    ]
    doc = {
        "checkpoint": str(root),
        "checkpoint_identity": identity,
        "page_size": int(page_size),
        "index_ratio": int(kv_pool.index_ratio),
        "page_layout": page_layout,
        "state_layout": state_layout,
        "tp_rank": int(tp_rank),
        "tp_size": int(tp_size),
    }
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ParkStore:
    """Content-keyed RAM or SSD store for complete page-aligned QSA/GDN snapshots."""

    def __init__(
        self,
        *,
        mode: str,
        page_size: int,
        kv_pool,
        state_pool,
        fingerprint: str,
        min_tokens: int,
        ram_budget_bytes: int,
        ssd_dir: str | Path,
        disk_budget_bytes: int,
        pinned_window_bytes: int,
        idle_ms: int = 0,
    ) -> None:
        if mode not in {"ram", "ssd"}:
            raise ValueError(f"ParkStore mode must be 'ram' or 'ssd', got {mode!r}")
        if page_size < 1:
            raise ValueError("park page_size must be positive")
        if min_tokens < page_size or min_tokens % page_size:
            raise ValueError("park min_tokens must be a page-aligned positive prefix")
        if idle_ms < 0:
            raise ValueError("park idle_ms must be non-negative")
        if ram_budget_bytes < 0 or disk_budget_bytes < 0:
            raise ValueError("park budgets must be non-negative")
        if pinned_window_bytes < _ALIGNMENT or pinned_window_bytes % _ALIGNMENT:
            raise ValueError("park pinned window must be a positive 4096-byte multiple")
        self.mode = mode
        self.page_size = int(page_size)
        self.kv_pool = kv_pool
        self.state_pool = state_pool
        self.fingerprint = str(fingerprint)
        self.min_tokens = int(min_tokens)
        self.idle_ms = int(idle_ms)
        self.ram_budget_bytes = int(ram_budget_bytes)
        self.ssd_dir = Path(ssd_dir).expanduser()
        self.disk_budget_bytes = int(disk_budget_bytes)
        self.pinned_window_bytes = int(pinned_window_bytes)
        self._entries: dict[str, ParkedEntry] = {}
        self._hits = 0
        self._misses = 0
        self._last_restore_ms = 0.0
        self._last_restore_breakdown: dict[str, float] = {}
        self._disabled = False
        self._last_error: str | None = None
        self._lock = RLock()
        self._window_lock = Lock()
        self._stream = None
        self._windows: tuple[torch.Tensor, torch.Tensor] | None = None
        self._save_queue: Queue[PendingPark | None] | None = None
        self._worker: Thread | None = None
        self._digest_pool: ThreadPoolExecutor | None = None
        self._reserved_bytes = 0
        self._closed = False
        # Inference mode, the current CUDA device and the current CUDA stream are all THREAD-LOCAL,
        # and the serving process builds this store inside torch.inference_mode() (launch.py) with
        # the scheduler's metadata stream current (scheduler.py torch.cuda.set_stream). The pinned
        # windows allocated below are therefore inference tensors, and on 2026-09-03 the save
        # worker -- a plain Thread -- raised "Inplace update to inference tensor outside
        # InferenceMode is not allowed" on its very first write, so ssd parking disabled itself
        # before writing one payload. Same defect class as the GPU-owned bank fill (8a63977):
        # there the write moved back onto the loader's thread; here the worker IS the writer, so
        # it re-enters the caller's context instead. Capturing the stream also makes the copy
        # helpers' wait_stream() fence against the scheduler's real producer stream rather than
        # the worker thread's default stream.
        self._caller_inference_mode = torch.is_inference_mode_enabled()
        self._caller_stream = (
            torch.cuda.current_stream(kv_pool.device)
            if kv_pool.device.type == "cuda"
            else None
        )
        try:
            if kv_pool.device.type == "cuda":
                self._stream = torch.cuda.Stream(device=kv_pool.device)
            if mode == "ssd":
                self._digest_pool = ThreadPoolExecutor(
                    max_workers=_DIGEST_WORKERS, thread_name_prefix="kv-park-digest"
                )
                self.ssd_dir.mkdir(parents=True, exist_ok=True)
                self._windows = (
                    self._allocate_window(self.pinned_window_bytes),
                    self._allocate_window(self.pinned_window_bytes),
                )
                self._load_or_scan()
            self._save_queue = Queue(maxsize=2)
            self._worker = Thread(
                target=self._worker_loop,
                name=f"kv-park-{mode}",
                daemon=True,
            )
            self._worker.start()
        except Exception as exc:
            self._disabled = True
            self._stream = None
            self._windows = None
            if self._digest_pool is not None:
                self._digest_pool.shutdown(wait=False)
                self._digest_pool = None
            self._entries.clear()
            self.note_error(f"{mode} store setup failed: {exc!r}")
            logger.warning(
                f"KV parking disabled during {mode} store setup: {exc!r}"
            )

    @classmethod
    def from_config(cls, config, kv_pool, state_pool) -> "ParkStore":
        from freetoken.kvcache.qsa_pool import QSAKVCache

        mode = str(getattr(config, "kv_park", "off")).lower()
        if mode not in {"ram", "ssd"}:
            raise ValueError(f"--kv-park must be off, ram or ssd; got {mode!r}")
        if not isinstance(kv_pool, QSAKVCache) or state_pool is None:
            raise ValueError("--kv-park currently needs a QSA + GDN hybrid cache")
        directory = Path(
            str(getattr(config, "kv_park_ssd_dir", "~/.cache/freetoken/kv-park"))
        ).expanduser()
        if mode == "ssd":
            directory /= f"tp-{config.tp_info.rank:04d}-of-{config.tp_info.size:04d}"
        # Scheduler pins a Hub id to one immutable snapshot before Engine loading. Reuse that exact
        # path here: resolving a mutable branch again could fingerprint a newer revision than the
        # weights and PLE/GDN sibling state that produced the parked bytes.
        fingerprint = build_model_fingerprint(
            model_path=config.model_path,
            page_size=config.page_size,
            tp_rank=config.tp_info.rank,
            tp_size=config.tp_info.size,
            kv_pool=kv_pool,
            state_pool=state_pool,
        )
        return cls(
            mode=mode,
            page_size=config.page_size,
            kv_pool=kv_pool,
            state_pool=state_pool,
            fingerprint=fingerprint,
            min_tokens=int(getattr(config, "kv_park_min_tokens", 8192)),
            idle_ms=int(getattr(config, "kv_park_idle_ms", 0)),
            ram_budget_bytes=int(float(getattr(config, "kv_park_ram_gib", 2.0)) * (1 << 30)),
            ssd_dir=directory,
            disk_budget_bytes=int(float(getattr(config, "kv_park_ssd_gib", 32.0)) * (1 << 30)),
            pinned_window_bytes=int(getattr(config, "kv_park_window_mib", 256)) * (1 << 20),
        )

    def _new_payload_digest(self, chunk_bytes: int) -> _ChunkedDigest:
        return _ChunkedDigest(self._digest_pool, chunk_bytes)

    def payload_bytes(self, token_count: int) -> int:
        if token_count < 0 or token_count % self.page_size:
            raise ValueError("park token_count must be page aligned")
        kv_per_token, _ = self.kv_pool.unit_bytes()
        state = self.state_pool.bytes_per_slot()
        return token_count * int(kv_per_token) + int(state)

    def storage_bytes(self, token_count: int) -> int:
        payload = self.payload_bytes(token_count)
        if self.mode == "ram":
            return payload + token_count * torch.int32.itemsize
        return _align_up(_HEADER_BYTES + token_count * torch.int32.itemsize) + payload

    def _entry_views(self, page_bases: torch.Tensor, state_slot: int) -> _ByteSpan:
        bases = page_bases.detach().to(device="cpu", dtype=torch.int64).tolist()
        views: list[torch.Tensor] = []
        for base in bases:
            if base < 0 or base % self.page_size:
                raise ValueError(f"park page base must be aligned, got {base}")
            views.extend(self.kv_pool.page_byte_views(base // self.page_size))
        views.extend(self.state_pool.slot_byte_views(int(state_slot)))
        return _ByteSpan.build(views)

    def _allocate_window(self, nbytes: int) -> torch.Tensor:
        if self.kv_pool.device.type == "cuda":
            from freetoken.kernel.pinned import alloc_pinned_tensor

            backing = alloc_pinned_tensor(nbytes + _ALIGNMENT, dtype=torch.uint8)
        else:
            backing = torch.empty(nbytes + _ALIGNMENT, dtype=torch.uint8)
        offset = (-int(backing.data_ptr())) % _ALIGNMENT
        window = backing[offset : offset + nbytes]
        if int(window.data_ptr()) % _ALIGNMENT:
            raise RuntimeError("could not align KV park pinned window")
        # Keep the parent allocation alive; a tensor slice owns its storage, including the slack.
        return window

    def _copy_to_ram(self, span: _ByteSpan) -> torch.Tensor:
        payload_bytes = span.nbytes
        if self.kv_pool.device.type == "cuda":
            from freetoken.kernel.pinned import alloc_pinned_tensor

            host = alloc_pinned_tensor(payload_bytes, dtype=torch.uint8)
        else:
            host = torch.empty(payload_bytes, dtype=torch.uint8)
        if self._stream is None:
            self._copy_span_to_window(span, 0, payload_bytes, host)
        else:
            current = torch.cuda.current_stream(self.kv_pool.device)
            self._stream.wait_stream(current)
            with torch.cuda.stream(self._stream):
                self._copy_span_to_window(span, 0, payload_bytes, host)
            self._stream.synchronize()
        return host

    @staticmethod
    def _copy_span_to_window(
        span: _ByteSpan, offset: int, length: int, window: torch.Tensor
    ) -> None:
        index = span.locate(offset)
        views, offsets = span.views, span.offsets
        window_offset = 0
        while window_offset < length:
            if index >= len(views):
                raise RuntimeError(
                    f"park source ended {length - window_offset} bytes short"
                )
            raw = views[index]
            start = offset + window_offset - offsets[index]
            take = min(length - window_offset, raw.numel() - start)
            window[window_offset : window_offset + take].copy_(
                raw[start : start + take], non_blocking=True
            )
            window_offset += take
            index += 1

    @staticmethod
    def _copy_window_to_span(
        window: torch.Tensor, offset: int, length: int, span: _ByteSpan
    ) -> None:
        index = span.locate(offset)
        views, offsets = span.views, span.offsets
        window_offset = 0
        while window_offset < length:
            if index >= len(views):
                raise RuntimeError(
                    f"park destination ended {length - window_offset} bytes short"
                )
            raw = views[index]
            start = offset + window_offset - offsets[index]
            take = min(length - window_offset, raw.numel() - start)
            raw[start : start + take].copy_(
                window[window_offset : window_offset + take], non_blocking=True
            )
            window_offset += take
            index += 1

    def _header(
        self,
        *,
        key: str,
        token_count: int,
        payload_bytes: int,
        payload_offset: int,
        payload_sha256: str,
    ) -> bytes:
        meta = json.dumps(
            {
                "version": _VERSION,
                "fingerprint": self.fingerprint,
                "key": key,
                "token_count": token_count,
                "page_size": self.page_size,
                "payload_bytes": payload_bytes,
                "payload_offset": payload_offset,
                "payload_sha256": payload_sha256,
                "payload_digest_chunk_bytes": _DIGEST_CHUNK_BYTES,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        prefix = _MAGIC + struct.pack("<II", _VERSION, len(meta))
        if len(prefix) + len(meta) > _HEADER_BYTES:
            raise ValueError("KV park header exceeds 4096 bytes")
        return prefix + meta + bytes(_HEADER_BYTES - len(prefix) - len(meta))

    def _parse_header(self, path: Path) -> dict:
        with path.open("rb") as handle:
            raw = handle.read(_HEADER_BYTES)
        if len(raw) != _HEADER_BYTES or raw[: len(_MAGIC)] != _MAGIC:
            raise ParkEntryRejected(f"invalid KV park magic/header: {path}")
        version, meta_len = struct.unpack_from("<II", raw, len(_MAGIC))
        if version != _VERSION or meta_len < 2 or meta_len > _HEADER_BYTES - 16:
            raise ParkEntryRejected(f"unsupported KV park header version: {path}")
        try:
            meta = json.loads(raw[16 : 16 + meta_len].decode("utf-8"))
        except Exception as exc:
            raise ParkEntryRejected(f"invalid KV park header JSON: {path}") from exc
        if meta.get("fingerprint") != self.fingerprint:
            raise ParkEntryRejected(f"KV park fingerprint mismatch: {path}")
        if int(meta.get("page_size", 0)) != self.page_size:
            raise ParkEntryRejected(f"KV park page-size mismatch: {path}")
        token_count = int(meta.get("token_count", 0))
        payload_bytes = int(meta.get("payload_bytes", -1))
        payload_offset = int(meta.get("payload_offset", -1))
        payload_sha256 = meta.get("payload_sha256")
        if (
            not isinstance(payload_sha256, str)
            or len(payload_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in payload_sha256)
        ):
            raise ParkEntryRejected(f"invalid KV park payload checksum: {path}")
        if token_count < self.min_tokens or token_count % self.page_size:
            raise ParkEntryRejected(f"invalid KV park token count: {path}")
        if payload_bytes != self.payload_bytes(token_count):
            raise ParkEntryRejected(f"KV park payload-size mismatch: {path}")
        if payload_offset != _align_up(_HEADER_BYTES + token_count * torch.int32.itemsize):
            raise ParkEntryRejected(f"KV park payload-offset mismatch: {path}")
        digest_chunk = int(meta.get("payload_digest_chunk_bytes", 0))
        if digest_chunk < _ALIGNMENT or digest_chunk % _ALIGNMENT:
            raise ParkEntryRejected(f"invalid KV park digest chunk: {path}")
        if path.stat().st_size != payload_offset + payload_bytes:
            raise ParkEntryRejected(f"KV park file-size mismatch: {path}")
        return meta

    def _read_tokens(self, path: Path, token_count: int) -> torch.Tensor:
        with path.open("rb") as handle:
            handle.seek(_HEADER_BYTES)
            raw = bytearray(handle.read(token_count * torch.int32.itemsize))
        if len(raw) != token_count * torch.int32.itemsize:
            raise ParkEntryRejected(f"short KV park token list: {path}")
        return torch.frombuffer(raw, dtype=torch.int32).clone()

    @staticmethod
    def _unbuffered_reader(path: Path):
        if sys.platform != "win32":
            return None
        from freetoken.moe import win_io

        if not win_io.enabled():
            return None
        try:
            return win_io.UnbufferedReader(str(path))
        except OSError:
            # Network, compressed, and unusual Windows volumes can refuse NO_BUFFERING. The
            # ordinary reader is slower/cacheable but preserves correctness and bootability.
            return None

    def _write_ssd(
        self,
        *,
        key: str,
        tokens: torch.Tensor,
        span: _ByteSpan,
        payload_bytes: int,
        on_source_copied=None,
    ) -> ParkedEntry:
        assert self._windows is not None
        payload_offset = _align_up(_HEADER_BYTES + _tensor_nbytes(tokens))
        final = self.ssd_dir / f"{key}.park"
        temp = self.ssd_dir / f".{key}.{os.getpid()}.tmp"
        header = self._header(
            key=key,
            token_count=len(tokens),
            payload_bytes=payload_bytes,
            payload_offset=payload_offset,
            payload_sha256="0" * 64,
        )
        payload_digest = self._new_payload_digest(_DIGEST_CHUNK_BYTES)
        try:
            with temp.open("w+b", buffering=0) as handle:
                handle.write(header)
                handle.write(memoryview(tokens.numpy()).cast("B"))
                handle.write(bytes(payload_offset - _HEADER_BYTES - _tensor_nbytes(tokens)))
                for offset in range(0, payload_bytes, self.pinned_window_bytes):
                    # A restore owns both windows for its complete double-buffered pass. The writer
                    # takes one window for one chunk at a time, so a waiting restore is delayed by at
                    # most one bounded D2H-plus-write span rather than a multi-GiB file.
                    with self._window_lock:
                        window = self._windows[0]
                        length = min(self.pinned_window_bytes, payload_bytes - offset)
                        if self._stream is None:
                            self._copy_span_to_window(span, offset, length, window)
                        else:
                            current = torch.cuda.current_stream(self.kv_pool.device)
                            self._stream.wait_stream(current)
                            with torch.cuda.stream(self._stream):
                                self._copy_span_to_window(span, offset, length, window)
                            self._stream.synchronize()
                        if offset + length == payload_bytes and on_source_copied is not None:
                            on_source_copied()
                        raw = memoryview(window[:length].numpy()).cast("B")
                        payload_digest.update(raw)
                        handle.write(raw)
                checksum = payload_digest.hexdigest()
                handle.seek(0)
                handle.write(
                    self._header(
                        key=key,
                        token_count=len(tokens),
                        payload_bytes=payload_bytes,
                        payload_offset=payload_offset,
                        payload_sha256=checksum,
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, final)
        finally:
            temp.unlink(missing_ok=True)
        now = time.time_ns()
        return ParkedEntry(
            key=key,
            token_ids=tokens,
            token_count=len(tokens),
            payload_bytes=payload_bytes,
            total_bytes=final.stat().st_size,
            last_used_ns=now,
            payload_sha256=checksum,
            path=final,
        )

    def offer(
        self, input_ids: torch.Tensor, page_bases: torch.Tensor, state_slot: int
    ) -> PendingPark | None:
        """Queue a save without blocking the scheduler; ``None`` means fall back to eviction."""
        with self._lock:
            if self._disabled or self._closed or self._save_queue is None:
                return None
            tokens = _tokens_cpu(input_ids)
            if len(tokens) < self.min_tokens or len(tokens) % self.page_size:
                return None
            bases = page_bases.detach().to(device="cpu", dtype=torch.int32).flatten().clone()
            if len(bases) != len(tokens) // self.page_size:
                raise ValueError(
                    f"park got {len(bases)} page bases for {len(tokens)} tokens"
                )
            needed = self.storage_bytes(len(tokens))
            budget = self.ram_budget_bytes if self.mode == "ram" else self.disk_budget_bytes
            if needed > budget:
                return None
            key = rolling_page_keys(tokens, self.page_size, self.fingerprint)[-1]
            existing = self._entries.get(key)
            if existing is not None and torch.equal(existing.token_ids, tokens):
                existing.last_used_ns = time.time_ns()
                pending = PendingPark(tokens, bases, int(state_slot))
                pending.success = True
                pending.copy_done.set()
                pending.done.set()
                return pending
            if self._save_queue.full():
                return None
            before = len(self._entries)
            self._evict_to_fit(self._reserved_bytes + needed)
            if self.mode == "ssd" and len(self._entries) != before:
                self._write_manifest()
            occupied = sum(entry.total_bytes for entry in self._entries.values())
            if occupied + self._reserved_bytes + needed > budget:
                return None
            source_ready = None
            if self._stream is not None:
                source_ready = torch.cuda.Event(enable_timing=False)
                source_ready.record(torch.cuda.current_stream(self.kv_pool.device))
            pending = PendingPark(
                tokens,
                bases,
                int(state_slot),
                source_ready,
                reserved_bytes=needed,
            )
            self._reserved_bytes += needed
            try:
                self._save_queue.put_nowait(pending)
            except Full:
                self._reserved_bytes -= needed
                return None
            return pending

    def _worker_loop(self) -> None:
        """Re-enter the constructing thread's context, then drain saves under it.

        See ``__init__``: inference mode, CUDA device and CUDA stream are thread-local, so a bare
        worker thread cannot write the pinned windows the scheduler's inference mode created."""
        with contextlib.ExitStack() as caller_context:
            if self.kv_pool.device.type == "cuda":
                torch.cuda.set_device(self.kv_pool.device)
                if self._caller_stream is not None:
                    caller_context.enter_context(torch.cuda.stream(self._caller_stream))
            caller_context.enter_context(torch.inference_mode(self._caller_inference_mode))
            self._drain_saves()

    def _drain_saves(self) -> None:
        assert self._save_queue is not None
        while True:
            pending = self._save_queue.get()
            try:
                if pending is None:
                    return
                if pending.source_ready is not None:
                    assert self._stream is not None
                    self._stream.wait_event(pending.source_ready)
                pending.success = self.save(
                    pending.token_ids,
                    pending.page_bases,
                    pending.state_slot,
                    _on_copied=pending.copy_done.set,
                )
            except Exception as exc:
                pending.error = exc
                self.note_error(f"background {self.mode} save failed: {exc!r}")
                logger.warning(f"KV parking background save failed: {exc!r}")
            finally:
                if pending is not None:
                    # save() currently completes every D2H before returning. Source ownership may
                    # move back to the allocators even when the later store operation failed.
                    pending.copy_done.set()
                    with self._lock:
                        self._reserved_bytes -= pending.reserved_bytes
                    pending.done.set()
                self._save_queue.task_done()

    def save(
        self,
        input_ids: torch.Tensor,
        page_bases: torch.Tensor,
        state_slot: int,
        *,
        _on_copied=None,
    ) -> bool:
        """Copy one complete entry before its source pages/state are released.

        The potentially multi-second copy/write runs outside ``_lock`` so a worker can never
        block the scheduler's next non-blocking offer or status read.
        """
        tokens = _tokens_cpu(input_ids)
        if len(tokens) < self.min_tokens or len(tokens) % self.page_size:
            if _on_copied is not None:
                _on_copied()
            return False
        bases = page_bases.detach().to(device="cpu", dtype=torch.int32).flatten().clone()
        if len(bases) != len(tokens) // self.page_size:
            raise ValueError(
                f"park got {len(bases)} page bases for {len(tokens)} tokens"
            )
        key = rolling_page_keys(tokens, self.page_size, self.fingerprint)[-1]
        needed = self.storage_bytes(len(tokens))
        budget = self.ram_budget_bytes if self.mode == "ram" else self.disk_budget_bytes
        with self._lock:
            if self._disabled:
                if _on_copied is not None:
                    _on_copied()
                return False
            existing = self._entries.get(key)
            if existing is not None:
                if torch.equal(existing.token_ids, tokens):
                    existing.last_used_ns = time.time_ns()
                    if _on_copied is not None:
                        _on_copied()
                    return True
                self._drop_entry(key)
            if needed > budget:
                if _on_copied is not None:
                    _on_copied()
                return False
            before = len(self._entries)
            self._evict_to_fit(needed)
            if self.mode == "ssd" and len(self._entries) != before:
                self._write_manifest()
        span = self._entry_views(bases, state_slot)
        payload = span.nbytes
        expected = self.payload_bytes(len(tokens))
        if payload != expected:
            raise RuntimeError(f"KV park payload {payload} != expected {expected}")
        try:
            if self.mode == "ram":
                buffer = self._copy_to_ram(span)
                if _on_copied is not None:
                    _on_copied()
                entry = ParkedEntry(
                    key=key,
                    token_ids=tokens,
                    token_count=len(tokens),
                    payload_bytes=payload,
                    total_bytes=needed,
                    last_used_ns=time.time_ns(),
                    ram_buffer=buffer,
                )
            else:
                entry = self._write_ssd(
                    key=key,
                    tokens=tokens,
                    span=span,
                    payload_bytes=payload,
                    on_source_copied=_on_copied,
                )
                if _on_copied is not None:
                    _on_copied()
            with self._lock:
                self._evict_to_fit(needed)
                self._entries[key] = entry
                if self.mode == "ssd":
                    self._write_manifest()
            return True
        except Exception as exc:
            # A failed enqueue/copy can still leave work on the private CUDA stream. Fence it
            # before copy_done lets the scheduler recycle and overwrite the source pages.
            if self._stream is not None:
                self._stream.synchronize()
            if _on_copied is not None:
                _on_copied()
            with self._lock:
                self._disabled = True
                self._last_error = f"{self.mode} save failed: {exc!r}"
            logger.warning(
                f"KV parking disabled after {self.mode} save failed: {exc!r}"
            )
            return False

    def lookup(
        self,
        input_ids: torch.Tensor,
        min_len: int = 0,
        max_len: int | None = None,
    ) -> ParkedEntry | None:
        with self._lock:
            tokens = _tokens_cpu(input_ids)
            keys = rolling_page_keys(tokens, self.page_size, self.fingerprint)
            for page_number in range(len(keys), 0, -1):
                token_count = page_number * self.page_size
                if token_count < max(self.min_tokens, min_len):
                    break
                if max_len is not None and token_count > max_len:
                    continue
                key = keys[page_number - 1]
                entry = self._entries.get(key)
                if entry is None:
                    continue
                if not torch.equal(entry.token_ids, tokens[:token_count]):
                    self._drop_entry(key)
                    self._misses += 1
                    return None
                entry.last_used_ns = time.time_ns()
                self._hits += 1
                return entry
            self._misses += 1
            return None

    @staticmethod
    def _copy_restored_chunk(
        window: torch.Tensor,
        chunk_offset: int,
        chunk_length: int,
        source_offset: int,
        length: int,
        span: _ByteSpan,
    ) -> bool:
        restore_start = max(chunk_offset, source_offset)
        restore_stop = min(chunk_offset + chunk_length, source_offset + length)
        if restore_start >= restore_stop:
            return False
        window_offset = restore_start - chunk_offset
        restore_length = restore_stop - restore_start
        ParkStore._copy_window_to_span(
            window[window_offset : window_offset + restore_length],
            restore_start - source_offset,
            restore_length,
            span,
        )
        return True

    @staticmethod
    def _expected_payload_checksum(entry: ParkedEntry) -> str:
        if entry.payload_sha256 is None:
            raise ParkEntryRejected(f"KV park payload checksum missing: {entry.path}")
        return entry.payload_sha256

    def _read_payload_cpu(
        self,
        entry: ParkedEntry,
        span: _ByteSpan,
        payload_offset: int,
        source_offset: int,
        length: int,
        digest_chunk_bytes: int = _DIGEST_CHUNK_BYTES,
        timing: dict[str, float] | None = None,
    ) -> None:
        assert entry.path is not None and self._windows is not None
        expected = self._expected_payload_checksum(entry)
        digest = self._new_payload_digest(digest_chunk_bytes)
        with self._window_lock:
            with entry.path.open("rb", buffering=0) as handle:
                handle.seek(payload_offset)
                for chunk, offset in enumerate(
                    range(0, entry.payload_bytes, self.pinned_window_bytes)
                ):
                    chunk_length = min(
                        self.pinned_window_bytes, entry.payload_bytes - offset
                    )
                    window = self._windows[chunk % len(self._windows)]
                    raw = memoryview(window[:chunk_length].numpy()).cast("B")
                    mark = time.perf_counter()
                    got = handle.readinto(raw)
                    if timing is not None:
                        timing["read_ms"] += (time.perf_counter() - mark) * 1000.0
                    if got != chunk_length:
                        raise ParkEntryRejected(
                            f"short KV park payload read: {entry.path}"
                        )
                    mark = time.perf_counter()
                    digest.update(raw)
                    if timing is not None:
                        timing["hash_ms"] += (time.perf_counter() - mark) * 1000.0
                    mark = time.perf_counter()
                    self._copy_restored_chunk(
                        window,
                        offset,
                        chunk_length,
                        source_offset,
                        length,
                        span,
                    )
                    if timing is not None:
                        timing["copy_ms"] += (time.perf_counter() - mark) * 1000.0
        if digest.hexdigest() != expected:
            raise ParkEntryRejected(f"KV park payload checksum mismatch: {entry.path}")

    def _read_payload_cuda(
        self,
        entry: ParkedEntry,
        span: _ByteSpan,
        payload_offset: int,
        source_offset: int,
        length: int,
        digest_chunk_bytes: int = _DIGEST_CHUNK_BYTES,
        timing: dict[str, float] | None = None,
    ) -> None:
        assert entry.path is not None and self._windows is not None and self._stream is not None
        expected = self._expected_payload_checksum(entry)
        digest = self._new_payload_digest(digest_chunk_bytes)
        reader = None
        handle = None
        with self._window_lock:
            try:
                reader = self._unbuffered_reader(entry.path)
                handle = None if reader is not None else entry.path.open("rb", buffering=0)
                if handle is not None:
                    handle.seek(payload_offset)
                events: list[torch.cuda.Event | None] = [None] * len(self._windows)
                for chunk, offset in enumerate(
                    range(0, entry.payload_bytes, self.pinned_window_bytes)
                ):
                    index = chunk % len(self._windows)
                    event = events[index]
                    if event is not None:
                        # Only wait when this tray is about to be reused. Reading into the other
                        # tray meanwhile overlaps disk block N+1 with H2D block N, matching M1.
                        mark = time.perf_counter()
                        event.synchronize()
                        if timing is not None:
                            timing["tray_wait_ms"] += (time.perf_counter() - mark) * 1000.0
                    chunk_length = min(
                        self.pinned_window_bytes, entry.payload_bytes - offset
                    )
                    window = self._windows[index]
                    raw = memoryview(window[:chunk_length].numpy()).cast("B")
                    mark = time.perf_counter()
                    if reader is not None:
                        got = reader.read_into(
                            raw,
                            payload_offset + offset,
                            chunk_length,
                        )
                    else:
                        got = handle.readinto(raw)
                    if timing is not None:
                        timing["read_ms"] += (time.perf_counter() - mark) * 1000.0
                    if got != chunk_length:
                        raise ParkEntryRejected(
                            f"short KV park payload read: {entry.path}"
                        )
                    mark = time.perf_counter()
                    digest.update(raw)
                    if timing is not None:
                        timing["hash_ms"] += (time.perf_counter() - mark) * 1000.0
                    mark = time.perf_counter()
                    with torch.cuda.stream(self._stream):
                        copied = self._copy_restored_chunk(
                            window,
                            offset,
                            chunk_length,
                            source_offset,
                            length,
                            span,
                        )
                        if copied:
                            done = torch.cuda.Event(enable_timing=False)
                            done.record(self._stream)
                            events[index] = done
                        else:
                            events[index] = None
                    if timing is not None:
                        timing["copy_ms"] += (time.perf_counter() - mark) * 1000.0
                mark = time.perf_counter()
                self._stream.synchronize()
                if timing is not None:
                    timing["sync_ms"] += (time.perf_counter() - mark) * 1000.0
            finally:
                if reader is not None:
                    reader.close()
                if handle is not None:
                    handle.close()
        if digest.hexdigest() != expected:
            raise ParkEntryRejected(f"KV park payload checksum mismatch: {entry.path}")

    def restore(
        self,
        entry: ParkedEntry,
        page_bases: torch.Tensor,
        state_slot: int,
        *,
        page_offset: int = 0,
    ) -> None:
        """Restore a parked page suffix plus the complete state into newly-owned storage."""
        started = time.perf_counter()
        # Per-step restore timing. The 2026-09-03 live SSD restore took 1,882 ms against an
        # 1,883 ms gate while the M1 bench moved the same 1.66 GiB in 317 ms; the breakdown
        # below is what separates them and is published on /v1/cache/status so a live run can
        # be attributed without a debug-level boot.
        timing: dict[str, float] = {
            "views_ms": 0.0,
            "header_ms": 0.0,
            "read_ms": 0.0,
            "hash_ms": 0.0,
            "copy_ms": 0.0,
            "tray_wait_ms": 0.0,
            "sync_ms": 0.0,
            "total_ms": 0.0,
        }
        with self._lock:
            total_pages = entry.token_count // self.page_size
            if page_offset < 0 or page_offset > total_pages:
                raise ValueError("restore page offset is outside the parked entry")
            mark = time.perf_counter()
            bases = page_bases.detach().to(device="cpu", dtype=torch.int32).flatten()
            if len(bases) != total_pages - page_offset:
                raise ValueError("restore target page count does not match parked suffix")
            span = self._entry_views(bases, state_slot)
            per_page_views = self.kv_pool.page_byte_views(0)
            per_page_bytes = sum(_tensor_nbytes(view) for view in per_page_views)
            state_views = self.state_pool.slot_byte_views(self.state_pool.padding_slot)
            state_bytes = sum(_tensor_nbytes(view) for view in state_views)
            source_offset = page_offset * per_page_bytes
            suffix_bytes = (total_pages - page_offset) * per_page_bytes + state_bytes
            if span.nbytes != suffix_bytes:
                raise ParkEntryRejected("restore target layout does not match parked suffix")
            timing["views_ms"] = (time.perf_counter() - mark) * 1000.0
            timing["views"] = float(len(span.views))
            try:
                if entry.ram_buffer is not None:
                    if entry.ram_buffer.numel() != entry.payload_bytes:
                        raise ParkEntryRejected("RAM parked payload size changed")
                    source = entry.ram_buffer[
                        source_offset : source_offset + suffix_bytes
                    ]
                    mark = time.perf_counter()
                    if self._stream is None:
                        self._copy_window_to_span(source, 0, suffix_bytes, span)
                    else:
                        with torch.cuda.stream(self._stream):
                            self._copy_window_to_span(source, 0, suffix_bytes, span)
                        timing["copy_ms"] = (time.perf_counter() - mark) * 1000.0
                        mark = time.perf_counter()
                        self._stream.synchronize()
                        timing["sync_ms"] = (time.perf_counter() - mark) * 1000.0
                else:
                    if entry.path is None:
                        raise ParkEntryRejected("SSD parked entry has no file")
                    mark = time.perf_counter()
                    meta = self._parse_header(entry.path)
                    disk_tokens = self._read_tokens(entry.path, entry.token_count)
                    if not torch.equal(disk_tokens, entry.token_ids):
                        raise ParkEntryRejected("SSD parked token verification failed")
                    payload_offset = int(meta["payload_offset"])
                    digest_chunk = int(meta["payload_digest_chunk_bytes"])
                    timing["header_ms"] = (time.perf_counter() - mark) * 1000.0
                    if self._stream is None:
                        self._read_payload_cpu(
                            entry, span, payload_offset, source_offset, suffix_bytes,
                            digest_chunk, timing,
                        )
                    else:
                        self._read_payload_cuda(
                            entry, span, payload_offset, source_offset, suffix_bytes,
                            digest_chunk, timing,
                        )
            except Exception:
                if self._stream is not None:
                    self._stream.synchronize()
                self._drop_entry(entry.key)
                raise
            finally:
                self._last_restore_ms = (time.perf_counter() - started) * 1000.0
                timing["total_ms"] = self._last_restore_ms
                self._last_restore_breakdown = timing
                logger.debug(f"KV park restore breakdown: {timing}")

    def _evict_to_fit(self, incoming: int) -> None:
        budget = self.ram_budget_bytes if self.mode == "ram" else self.disk_budget_bytes
        while self._entries and sum(e.total_bytes for e in self._entries.values()) + incoming > budget:
            oldest = min(self._entries.values(), key=lambda entry: entry.last_used_ns)
            self._drop_entry(oldest.key, write_manifest=False)

    def _drop_entry(self, key: str, *, write_manifest: bool = True) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None and entry.path is not None:
            entry.path.unlink(missing_ok=True)
        if write_manifest and self.mode == "ssd":
            self._write_manifest()

    def _manifest_doc(self) -> dict:
        return {
            "version": _VERSION,
            "fingerprint": self.fingerprint,
            "entries": [
                {
                    "key": entry.key,
                    "file": entry.path.name,
                    "token_count": entry.token_count,
                    "payload_bytes": entry.payload_bytes,
                    "payload_sha256": entry.payload_sha256,
                    "total_bytes": entry.total_bytes,
                    "last_used_ns": entry.last_used_ns,
                }
                for entry in sorted(self._entries.values(), key=lambda item: item.key)
                if entry.path is not None
            ],
        }

    def _write_manifest(self) -> None:
        if self.mode != "ssd" or self._disabled:
            return
        path = self.ssd_dir / "park.json"
        temp = self.ssd_dir / f".park.{os.getpid()}.tmp"
        data = json.dumps(self._manifest_doc(), indent=2, sort_keys=True) + "\n"
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def _entry_from_file(self, path: Path, last_used_ns: int | None = None) -> ParkedEntry:
        meta = self._parse_header(path)
        token_count = int(meta["token_count"])
        key = str(meta["key"])
        tokens = self._read_tokens(path, token_count)
        expected_key = rolling_page_keys(tokens, self.page_size, self.fingerprint)[-1]
        if key != expected_key or path.stem != key:
            raise ParkEntryRejected(f"KV park key/token mismatch: {path}")
        return ParkedEntry(
            key=key,
            token_ids=tokens,
            token_count=token_count,
            payload_bytes=int(meta["payload_bytes"]),
            total_bytes=path.stat().st_size,
            last_used_ns=last_used_ns or path.stat().st_mtime_ns,
            payload_sha256=str(meta["payload_sha256"]),
            path=path,
        )

    def _remove_stale_temp_files(self) -> None:
        # A hard kill bypasses each writer's finally block. Keep another live process's file, but
        # remove dead-writer snapshots before admitting new bytes so repeated interrupted saves
        # cannot grow beyond the configured SSD budget across restarts.
        for path in self.ssd_dir.iterdir():
            pid = _temp_writer_pid(path.name)
            if pid is not None and path.is_file() and not _pid_is_alive(pid):
                path.unlink()

    def _load_or_scan(self) -> None:
        self._remove_stale_temp_files()
        manifest = self.ssd_dir / "park.json"
        paths: list[tuple[Path, int | None]] = []
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            if doc.get("version") != _VERSION or doc.get("fingerprint") != self.fingerprint:
                raise ValueError("manifest version/fingerprint mismatch")
            for row in doc.get("entries", []):
                filename = str(row["file"])
                candidate = Path(filename)
                stem = candidate.stem
                if (
                    candidate.name != filename
                    or candidate.suffix != ".park"
                    or len(stem) != 32
                    or any(ch not in "0123456789abcdef" for ch in stem)
                ):
                    raise ValueError("manifest entry escapes the KV parking directory")
                paths.append((self.ssd_dir / filename, int(row["last_used_ns"])))
        except Exception:
            paths = [(path, None) for path in sorted(self.ssd_dir.glob("*.park"))]
        seen: set[Path] = set()
        for path, last_used in paths:
            seen.add(path)
            try:
                entry = self._entry_from_file(path, last_used)
            except Exception:
                path.unlink(missing_ok=True)
                continue
            self._entries[entry.key] = entry
        # A valid but stale manifest must not orphan extra files forever; validate the extras too.
        for path in sorted(self.ssd_dir.glob("*.park")):
            if path in seen:
                continue
            try:
                entry = self._entry_from_file(path)
            except Exception:
                path.unlink(missing_ok=True)
                continue
            self._entries[entry.key] = entry
        self._evict_to_fit(0)
        self._write_manifest()

    def status(self) -> dict[str, int | float | str | bool]:
        with self._lock:
            return {
                "mode": self.mode,
                "parked_count": len(self._entries),
                "parked_bytes": sum(entry.total_bytes for entry in self._entries.values()),
                "hits": self._hits,
                "misses": self._misses,
                "last_restore_ms": self._last_restore_ms,
                "last_restore_breakdown_ms": dict(self._last_restore_breakdown),
                "disabled": self._disabled,
                "last_error": self._last_error,
            }

    def note_error(self, message: str) -> None:
        """Record the newest parking failure for ``/v1/cache/status``.

        The 2026-09-03 live run lost a whole ssd session to a failure that only ever appeared in
        the server log: the status line said mode ssd, zero entries, and nothing about why."""
        with self._lock:
            self._last_error = str(message)

    def flush(self) -> None:
        if self._save_queue is not None:
            self._save_queue.join()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        queue = self._save_queue
        worker = self._worker
        if queue is not None and worker is not None:
            queue.join()
            queue.put(None)
            worker.join()
        with self._lock:
            if self.mode == "ssd":
                self._write_manifest()
        if self._digest_pool is not None:
            self._digest_pool.shutdown(wait=True)
            self._digest_pool = None


__all__ = [
    "ParkEntryRejected",
    "ParkStore",
    "ParkedEntry",
    "build_model_fingerprint",
    "rolling_page_keys",
]
