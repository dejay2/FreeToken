"""Host-RAM and SSD parking for page-aligned QSA/GDN prefix snapshots.

A parked entry is deliberately opaque to the attention and recurrent-state code: it is the
exact bytes returned by ``QSAKVCache.page_byte_views`` plus one complete
``LinearStatePool.slot_byte_views`` snapshot.  The scheduler copies those bytes before returning
any page or state slot to a free list, then a later match restores into ordinary newly-allocated
pages and inserts the prefix into the unchanged hybrid radix tree.

The SSD format (version 5) stores one immutable *segment* per saved checkpoint: a versioned
4-KiB header, the verbatim int32 token ids of the whole prefix, then two independently
checksummed 4-KiB-aligned regions -- the KV pages this segment adds on top of its parent
segment, and one complete recurrent-state slot.  A root segment starts its KV at token zero; a
child names its parent's key, immutable metadata digest and the page-aligned *cut* at which it
borrows the parent's KV.  Two cuts are legal: the parent's whole length (an ordinary
continuation, which writes only its new pages plus the current state) and the child's own
length (a shorter checkpoint whose bytes already lie inside a longer committed segment, which
writes no KV at all -- only its own state at that exact cut).  A segment therefore stores
exactly one state, at its own end, and a restore never rewinds a longer segment's state to a
shorter cut.  Every read validates the model/layout fingerprint, token ids, parent chain and
region digests, so stale or damaged files, broken chains and rolling-hash collisions become
misses rather than numerics changes.  Restore walks root -> target once, copying only each
ancestor's effective range, alternating two bounded pinned windows so disk read N+1 overlaps
H2D copy N; the background writer yields those shared windows while a restore is active.  The
files are the source of truth; no full SSD entry remains in RAM.
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
from typing import Callable, Iterable

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_MAGIC = b"FTKVPARK"
# 3: the payload digest became the parallel-friendly chunked scheme in _ChunkedDigest.
# 4: parent-linked segments with separate KV and state regions/digests (2026-09-09). On the live
# 5090 box every finished turn rewrote the whole prefix as a standalone v3 entry: 37 entries
# averaging 863 MB, the 32 GB park directory pinned at its cap, 27 MB/s written while serving
# and 793 GB written in 46 hours. A v4 child writes (new tokens x KV bytes/token) plus one
# 115,642,376-byte state slot, so 65,536 + 256 tokens fall from 987,254,792 to 119,033,864
# payload bytes (8.29x).
# 5: a child's parent_token_count is a *cut* that may equal the child's own length (2026-09-10).
# Under v4 a child could only link to a parent that was a complete, exact prefix, and a lookup
# only hit when a whole entry was a prefix of the prompt. On the live 5090 with real agent
# traffic that linked almost nothing: in 50 minutes the manifest held 16 entries and 18.5 GB,
# 1 hit against 18 misses, ~18 MB/s written. The harness sends the same conversation prefix
# two or three times per turn with different tails (the real next turn, a permission
# classifier, a status line): the next real turn shared its predecessor's save up to 44-148
# tokens before the end (88,724 of 88,768; 74,860 of 75,008), and a side request was a
# shorter exact prefix of an earlier, longer save (71,424 of 71,552). Neither shape linked, so
# each was a 1-2.7 GB standalone copy. v5 lets the scheduler save the final-prefill checkpoint
# as its own segment (the shared prefix the later requests really reach) and lets a shorter
# checkpoint borrow KV from a longer committed segment through its own cut with zero KV bytes
# of its own. Older versions are rejected by _parse_header and deleted once by _load_or_scan,
# which is the intended upgrade path: parking is off by default and an entry is cheap to
# rebuild.
_VERSION = 5
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
_PLACEHOLDER_DIGEST = "0" * 64


def _sha256_digest(raw) -> bytes:
    return hashlib.sha256(raw).digest()


def _is_hex(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(ch in "0123456789abcdef" for ch in value)
    )


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
    """One committed snapshot.

    ``payload_bytes`` is the complete snapshot size for a RAM entry but, for an SSD segment,
    only the bytes *this file* holds (its KV region plus its state region): a child segment's
    earlier pages live in its ancestors. ``ParkStore.payload_bytes(n)`` keeps the logical
    full-snapshot meaning. ``parent_token_count`` is the page-aligned cut at which this segment
    borrows its parent's KV: the parent's whole length for a continuation, or this segment's
    own length for a shorter checkpoint that owns no KV bytes. ``root_key`` is in-memory only
    and names the family this segment belongs to (itself for a root); the family is the unit
    of eviction.
    """

    key: str
    token_ids: torch.Tensor
    token_count: int
    payload_bytes: int
    total_bytes: int
    last_used_ns: int
    path: Path | None = None
    ram_buffer: torch.Tensor | None = None
    parent_key: str | None = None
    parent_token_count: int = 0
    parent_metadata_sha256: str | None = None
    kv_offset: int = 0
    kv_bytes: int = 0
    state_offset: int = 0
    state_bytes: int = 0
    kv_sha256: str | None = None
    state_sha256: str | None = None
    tokens_sha256: str | None = None
    metadata_sha256: str | None = None
    digest_chunk_bytes: int = _DIGEST_CHUNK_BYTES
    root_key: str = ""


@dataclass
class _SaveOp:
    """One planned save: its pinned parent family, exact reservation and in-flight key."""

    key: str
    tokens: torch.Tensor
    keys: list[str]
    parent: ParkedEntry | None = None
    # The cut this save borrows through: parent.token_count for a continuation, len(tokens)
    # for a shorter checkpoint, 0 for a root. Sizing, writing and validation all use it.
    cut: int = 0
    chain: tuple[ParkedEntry, ...] = ()
    root_key: str | None = None
    reserved_bytes: int = 0


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
    op: _SaveOp | None = field(default=None, repr=False)

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


def _release_file_cache(handle) -> None:
    # On 2026-09-08 a 19 GB park took about one minute and Windows free RAM fell 1.1 -> 0.4
    # GB; after drop_caches it rose 2.5 -> 9.4 GB within a minute with the server untouched.
    # Release each completed file window instead of crediting WSL's cache in the memory governor.
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        # Some filesystems expose the call but refuse the advice; retaining the cache is safe.
        return


def _write_all(handle, raw) -> None:
    """Write every byte of ``raw`` or raise; an unbuffered handle may accept a short write."""
    view = memoryview(raw).cast("B")
    while len(view):
        written = handle.write(view)
        if not written:
            raise OSError("short KV park write")
        view = view[written:]


def _fsync_directory(directory: Path) -> None:
    """Make a rename durable where the platform allows; best effort elsewhere (Windows)."""
    if sys.platform == "win32":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


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


def _tokens_sha256(tokens: torch.Tensor) -> str:
    return hashlib.sha256(memoryview(tokens.numpy()).cast("B")).hexdigest()


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
        # Family bookkeeping (SSD chains). A parent may be pinned by queued/active saves of its
        # children; a pinned family is never evicted and its dropped files are unlinked only
        # once the last pin goes, so a writer can always re-validate the parent it named.
        self._children: dict[str, set[str]] = {}
        self._pins: dict[str, int] = {}
        self._inflight: set[str] = set()
        self._pending_unlinks: list[tuple[str, Path, int]] = []
        self._generation = 0
        self._on_change: Callable[[], None] | None = None
        self._hits = 0
        self._misses = 0
        self._last_restore_ms = 0.0
        self._last_restore_breakdown: dict[str, float] = {}
        self._disabled = False
        self._last_error: str | None = None
        self._lock = RLock()
        self._window_lock = Lock()
        self._writer_lock = Lock()
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
            self._children.clear()
            self.note_error(f"{mode} store setup failed: {exc!r}")
            logger.warning(
                f"KV parking disabled during {mode} store setup: {exc!r}"
            )

    @property
    def generation(self) -> int:
        return self._generation

    def bind_change_callback(self, callback: Callable[[], None] | None) -> None:
        self._on_change = callback

    def _notify_change(self) -> None:
        self._generation += 1
        if self._on_change is not None:
            self._on_change()

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

    # ---- sizes -------------------------------------------------------------------------

    def _kv_bytes_per_token(self) -> int:
        return int(self.kv_pool.unit_bytes()[0])

    def _state_bytes(self) -> int:
        return int(self.state_pool.bytes_per_slot())

    def payload_bytes(self, token_count: int) -> int:
        """Logical size of one complete snapshot of ``token_count`` tokens."""
        if token_count < 0 or token_count % self.page_size:
            raise ValueError("park token_count must be page aligned")
        return token_count * self._kv_bytes_per_token() + self._state_bytes()

    def _segment_layout(self, token_count: int, cut: int) -> tuple[int, int, int, int]:
        """``(kv_offset, kv_bytes, state_offset, state_bytes)`` of one v5 file.

        ``cut`` is the parent cut: 0 for a root, the parent's length for a continuation, or
        ``token_count`` itself for a shorter checkpoint, whose KV region is then empty.
        """
        kv_offset = _align_up(_HEADER_BYTES + token_count * torch.int32.itemsize)
        kv_bytes = (token_count - cut) * self._kv_bytes_per_token()
        state_offset = _align_up(kv_offset + kv_bytes)
        return kv_offset, kv_bytes, state_offset, self._state_bytes()

    def _segment_bytes(self, token_count: int, cut: int) -> int:
        kv_offset, kv_bytes, state_offset, state_bytes = self._segment_layout(token_count, cut)
        return state_offset + state_bytes

    def storage_bytes(self, token_count: int) -> int:
        """Bytes one *standalone* (root) entry of ``token_count`` tokens occupies."""
        payload = self.payload_bytes(token_count)
        if self.mode == "ram":
            return payload + token_count * torch.int32.itemsize
        return self._segment_bytes(token_count, 0)

    def _planned_bytes(self, token_count: int, cut: int) -> int:
        """Exact file bytes of a save at ``cut``: a zero-KV child still pays header, the full
        token list, padding and one complete state slot (about 115.6 MB on the live model)."""
        if self.mode == "ram":
            return self.storage_bytes(token_count)
        return self._segment_bytes(token_count, cut)

    def _entry_views(self, page_bases: torch.Tensor, state_slot: int) -> _ByteSpan:
        bases = page_bases.detach().to(device="cpu", dtype=torch.int64).tolist()
        views: list[torch.Tensor] = []
        for base in bases:
            if base < 0 or base % self.page_size:
                raise ValueError(f"park page base must be aligned, got {base}")
            views.extend(self.kv_pool.page_byte_views(base // self.page_size))
        views.extend(self.state_pool.slot_byte_views(int(state_slot)))
        return _ByteSpan.build(views)

    def _split_views(
        self, page_bases: torch.Tensor, state_slot: int
    ) -> tuple[_ByteSpan, _ByteSpan]:
        """The KV pages and the state slot as two spans, in the same order as ``_entry_views``."""
        bases = page_bases.detach().to(device="cpu", dtype=torch.int64).tolist()
        views: list[torch.Tensor] = []
        for base in bases:
            if base < 0 or base % self.page_size:
                raise ValueError(f"park page base must be aligned, got {base}")
            views.extend(self.kv_pool.page_byte_views(base // self.page_size))
        return (
            _ByteSpan.build(views),
            _ByteSpan.build(self.state_pool.slot_byte_views(int(state_slot))),
        )

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

    def _stage_source_chunk(
        self, span: _ByteSpan, offset: int, length: int, window: torch.Tensor
    ) -> None:
        """D2H one bounded chunk of the source span into a pinned window and wait for it."""
        if self._stream is None:
            self._copy_span_to_window(span, offset, length, window)
            return
        current = torch.cuda.current_stream(self.kv_pool.device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            self._copy_span_to_window(span, offset, length, window)
        self._stream.synchronize()

    # ---- header ------------------------------------------------------------------------

    @staticmethod
    def _metadata_sha256(meta: dict) -> str:
        doc = {name: value for name, value in meta.items() if name != "metadata_sha256"}
        return hashlib.sha256(
            json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _segment_meta(
        self,
        *,
        key: str,
        token_count: int,
        parent: ParkedEntry | None,
        cut: int,
        kv_sha256: str,
        state_sha256: str,
        tokens_sha256: str,
    ) -> dict:
        if parent is None:
            cut = 0
        kv_offset, kv_bytes, state_offset, state_bytes = self._segment_layout(token_count, cut)
        meta = {
            "version": _VERSION,
            "fingerprint": self.fingerprint,
            "key": key,
            "token_count": token_count,
            "page_size": self.page_size,
            "parent_key": parent.key if parent is not None else None,
            "parent_token_count": cut,
            "parent_metadata_sha256": (
                parent.metadata_sha256 if parent is not None else None
            ),
            "kv_offset": kv_offset,
            "kv_bytes": kv_bytes,
            "state_offset": state_offset,
            "state_bytes": state_bytes,
            "payload_bytes": kv_bytes + state_bytes,
            "kv_sha256": kv_sha256,
            "state_sha256": state_sha256,
            "payload_digest_chunk_bytes": _DIGEST_CHUNK_BYTES,
            "tokens_sha256": tokens_sha256,
        }
        meta["metadata_sha256"] = self._metadata_sha256(meta)
        return meta

    @staticmethod
    def _header(meta: dict) -> bytes:
        raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
        prefix = _MAGIC + struct.pack("<II", _VERSION, len(raw))
        if len(prefix) + len(raw) > _HEADER_BYTES:
            raise ValueError("KV park header exceeds 4096 bytes")
        return prefix + raw + bytes(_HEADER_BYTES - len(prefix) - len(raw))

    def _parse_header(self, path: Path) -> dict:
        """Validate one segment header against this store's layout; chains are checked later."""
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
        if not isinstance(meta, dict) or meta.get("version") != _VERSION:
            raise ParkEntryRejected(f"unsupported KV park metadata version: {path}")
        if meta.get("fingerprint") != self.fingerprint:
            raise ParkEntryRejected(f"KV park fingerprint mismatch: {path}")
        if int(meta.get("page_size", 0)) != self.page_size:
            raise ParkEntryRejected(f"KV park page-size mismatch: {path}")
        if not _is_hex(meta.get("key"), 32):
            raise ParkEntryRejected(f"invalid KV park key: {path}")
        for name in ("kv_sha256", "state_sha256", "tokens_sha256", "metadata_sha256"):
            if not _is_hex(meta.get(name), 64):
                raise ParkEntryRejected(f"invalid KV park {name}: {path}")
        if meta["metadata_sha256"] != self._metadata_sha256(meta):
            raise ParkEntryRejected(f"KV park metadata digest mismatch: {path}")
        token_count = int(meta.get("token_count", 0))
        if token_count < self.min_tokens or token_count % self.page_size:
            raise ParkEntryRejected(f"invalid KV park token count: {path}")
        parent_key = meta.get("parent_key")
        parent_tokens = int(meta.get("parent_token_count", -1))
        parent_digest = meta.get("parent_metadata_sha256")
        if parent_key is None:
            if parent_tokens != 0 or parent_digest is not None:
                raise ParkEntryRejected(f"KV park root names a parent: {path}")
        else:
            # A cut equal to the child's own length is the v5 shorter checkpoint (zero KV
            # bytes); whether the parent is really longer is checked against the parent
            # entry in _parent_matches, since a header cannot see its parent.
            if (
                not _is_hex(parent_key, 32)
                or parent_key == meta["key"]
                or not _is_hex(parent_digest, 64)
                or parent_tokens <= 0
                or parent_tokens % self.page_size
                or parent_tokens > token_count
            ):
                raise ParkEntryRejected(f"invalid KV park parent link: {path}")
        kv_offset, kv_bytes, state_offset, state_bytes = self._segment_layout(
            token_count, parent_tokens
        )
        if (
            int(meta.get("kv_offset", -1)) != kv_offset
            or int(meta.get("kv_bytes", -1)) != kv_bytes
            or int(meta.get("state_offset", -1)) != state_offset
            or int(meta.get("state_bytes", -1)) != state_bytes
            or int(meta.get("payload_bytes", -1)) != kv_bytes + state_bytes
        ):
            raise ParkEntryRejected(f"KV park region layout mismatch: {path}")
        digest_chunk = int(meta.get("payload_digest_chunk_bytes", 0))
        if digest_chunk < _ALIGNMENT or digest_chunk % _ALIGNMENT:
            raise ParkEntryRejected(f"invalid KV park digest chunk: {path}")
        if path.stat().st_size != state_offset + state_bytes:
            raise ParkEntryRejected(f"KV park file-size mismatch: {path}")
        return meta

    def _read_tokens(self, path: Path, token_count: int, expected_sha256: str) -> torch.Tensor:
        with path.open("rb") as handle:
            handle.seek(_HEADER_BYTES)
            raw = bytearray(handle.read(token_count * torch.int32.itemsize))
        if len(raw) != token_count * torch.int32.itemsize:
            raise ParkEntryRejected(f"short KV park token list: {path}")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ParkEntryRejected(f"KV park token digest mismatch: {path}")
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

    # ---- chains and families ---------------------------------------------------------

    def _chain(self, entry: ParkedEntry) -> tuple[ParkedEntry, ...] | None:
        """Root -> ``entry`` through committed segments only; ``None`` if any link is broken.

        A shorter checkpoint means token length is no longer a topological order, so the walk
        follows identity links with cycle protection and finally checks that the effective
        ranges tile ``[0, entry.token_count)`` exactly.
        """
        chain: list[ParkedEntry] = []
        seen: set[str] = set()
        current: ParkedEntry | None = entry
        while current is not None:
            if current.key in seen or self._entries.get(current.key) is not current:
                return None
            seen.add(current.key)
            chain.append(current)
            if current.parent_key is None:
                break
            parent = self._entries.get(current.parent_key)
            if parent is None or not self._parent_matches(parent, current):
                return None
            current = parent
        chain.reverse()
        if self._chain_ranges(chain, entry.token_count) is None:
            return None
        return tuple(chain)

    @staticmethod
    def _legal_cut(parent_tokens: int, child_tokens: int, cut: int) -> bool:
        """The only two parent relationships v5 admits (see the version note)."""
        if cut <= 0:
            return False
        continuation = cut == parent_tokens and parent_tokens < child_tokens
        shorter_checkpoint = cut == child_tokens and child_tokens < parent_tokens
        return continuation or shorter_checkpoint

    @classmethod
    def _parent_matches(cls, parent: ParkedEntry, child: ParkedEntry) -> bool:
        cut = child.parent_token_count
        return (
            parent.path is not None
            and parent.key == child.parent_key
            and cls._legal_cut(parent.token_count, child.token_count, cut)
            and parent.metadata_sha256 is not None
            and parent.metadata_sha256 == child.parent_metadata_sha256
            and torch.equal(parent.token_ids[:cut], child.token_ids[:cut])
        )

    @staticmethod
    def _chain_ranges(
        chain: tuple[ParkedEntry, ...] | list[ParkedEntry], needed_end: int
    ) -> list[tuple[ParkedEntry, int, int]] | None:
        """Effective KV token range ``[start, stop)`` each segment contributes to a target
        that needs ``[0, needed_end)``; root first. ``None`` if the ranges do not tile it.

        Walking from the target up, a segment contributes ``[cut, needed_end)`` of its own
        local KV interval and passes ``min(needed_end, cut)`` to its parent, so a longer donor
        behind a shorter checkpoint is clipped to the checkpoint's cut and a zero-KV segment
        contributes nothing. State always comes from the target alone.
        """
        target_end = needed_end
        ranges: list[tuple[ParkedEntry, int, int]] = []
        for segment in reversed(chain):
            start = segment.parent_token_count
            stop = needed_end if needed_end > start else start
            if stop > segment.token_count:
                return None
            ranges.append((segment, start, stop))
            needed_end = min(needed_end, start)
        ranges.reverse()
        cursor = 0
        for _segment, start, stop in ranges:
            if stop <= start:
                continue
            if start != cursor:
                return None
            cursor = stop
        if cursor != target_end:
            return None
        return ranges

    def _plan_parent(
        self, tokens: torch.Tensor, keys: list[str]
    ) -> tuple[ParkedEntry, int] | None:
        """``(parent, cut)`` for a new save of ``tokens``, or ``None`` for a standalone root.

        The largest reusable cut wins: a committed longer segment whose prefix equals the
        whole of ``tokens`` supplies every KV byte (cut == len(tokens), a zero-KV child);
        otherwise the longest committed chain-complete entry that is an exact page-aligned
        prefix (cut == parent length). Among equal cuts the shorter donor with the smaller key
        wins, deterministically, to keep verification reads small. Deliberately not
        ``lookup()``: planning must not move the hit/miss counters, and a colliding shorter
        entry is merely skipped here rather than dropped. The longer-donor scan compares
        committed token lists directly (tens of entries on the live box); it computes no
        arbitrary longest-common-prefix and never invents a cut without a saved state.
        """
        if self.mode != "ssd":
            return None
        token_count = len(tokens)
        best: tuple[tuple[int, str], ParkedEntry] | None = None
        for candidate in self._entries.values():
            if candidate.path is None or candidate.token_count <= token_count:
                continue
            rank = (candidate.token_count, candidate.key)
            if best is not None and rank >= best[0]:
                continue
            if not torch.equal(candidate.token_ids[:token_count], tokens):
                continue
            if self._chain(candidate) is None:
                continue
            best = (rank, candidate)
        if best is not None:
            return best[1], token_count
        for page_number in range(len(keys) - 1, 0, -1):
            cut = page_number * self.page_size
            if cut < self.min_tokens:
                break
            candidate = self._entries.get(keys[page_number - 1])
            if candidate is None or candidate.path is None:
                continue
            if candidate.token_count != cut:
                continue
            if not torch.equal(candidate.token_ids, tokens[:cut]):
                continue
            if self._chain(candidate) is None:
                continue
            return candidate, cut
        return None

    def _pin(self, root_key: str) -> None:
        self._pins[root_key] = self._pins.get(root_key, 0) + 1

    def _unpin(self, root_key: str) -> None:
        count = self._pins.get(root_key, 0) - 1
        if count > 0:
            self._pins[root_key] = count
        else:
            self._pins.pop(root_key, None)
            self._sweep_unlinks()

    def _occupied_bytes(self) -> int:
        committed = sum(entry.total_bytes for entry in self._entries.values())
        return committed + sum(nbytes for _root, _path, nbytes in self._pending_unlinks)

    def _sweep_unlinks(self) -> None:
        """Delete dropped files whose family is no longer pinned; a failed unlink stays charged."""
        kept: list[tuple[str, Path, int]] = []
        for root_key, path, nbytes in self._pending_unlinks:
            if self._pins.get(root_key, 0):
                kept.append((root_key, path, nbytes))
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                kept.append((root_key, path, nbytes))
        self._pending_unlinks = kept

    def _unlink_or_defer(self, entry: ParkedEntry) -> None:
        if entry.path is None:
            return
        record = (entry.root_key or entry.key, entry.path, entry.total_bytes)
        if self._pins.get(record[0], 0):
            self._pending_unlinks.append(record)
            return
        try:
            entry.path.unlink(missing_ok=True)
        except OSError:
            self._pending_unlinks.append(record)

    def _subtree(self, key: str) -> list[ParkedEntry]:
        root = self._entries.get(key)
        if root is None:
            return []
        members = [root]
        queue = [key]
        while queue:
            for child_key in self._children.get(queue.pop(), ()):
                child = self._entries.get(child_key)
                if child is not None:
                    members.append(child)
                    queue.append(child_key)
        return members

    def _drop_order(self, victims: list[ParkedEntry]) -> list[ParkedEntry]:
        """Child-before-parent order over the actual parent edges (iterative postorder).

        Under v4 descendants were strictly longer than ancestors, so longest-first was a safe
        order; a v5 shorter checkpoint is a *shorter* child, so length no longer orders the
        graph. An interrupted drop must still leave valid roots, never orphans.
        """
        pending = {
            victim.key: victim
            for victim in victims
            if self._entries.get(victim.key) is victim
        }
        order: list[ParkedEntry] = []
        visited: set[str] = set()
        for key in list(pending):
            if key in visited:
                continue
            stack: list[tuple[str, bool]] = [(key, False)]
            while stack:
                current, expanded = stack.pop()
                if expanded:
                    order.append(pending[current])
                    continue
                if current in visited:
                    continue
                visited.add(current)
                stack.append((current, True))
                for child_key in self._children.get(current, ()):
                    if child_key in pending and child_key not in visited:
                        stack.append((child_key, False))
        return order

    def _drop_entries(self, victims: list[ParkedEntry], *, write_manifest: bool) -> None:
        for victim in self._drop_order(victims):
            if self._entries.get(victim.key) is not victim:
                continue
            del self._entries[victim.key]
            self._children.pop(victim.key, None)
            if victim.parent_key is not None:
                siblings = self._children.get(victim.parent_key)
                if siblings is not None:
                    siblings.discard(victim.key)
            self._unlink_or_defer(victim)
        if victims:
            self._notify_change()
        if write_manifest and self.mode == "ssd":
            self._write_manifest()

    def _drop_entry(self, key: str, *, write_manifest: bool = True) -> None:
        """Forget ``key`` and every segment that depends on its bytes."""
        self._drop_entries(self._subtree(key), write_manifest=write_manifest)

    def _evict_to_fit(self, incoming: int) -> bool:
        """Evict least-recently-used unpinned *families* until ``incoming`` fits; False if stuck."""
        budget = self.ram_budget_bytes if self.mode == "ram" else self.disk_budget_bytes
        while True:
            self._sweep_unlinks()
            if self._occupied_bytes() + self._reserved_bytes + incoming <= budget:
                return True
            families: dict[str, list[ParkedEntry]] = {}
            for entry in self._entries.values():
                families.setdefault(entry.root_key or entry.key, []).append(entry)
            candidates = [
                (max(member.last_used_ns for member in members), root_key)
                for root_key, members in families.items()
                if not self._pins.get(root_key, 0)
            ]
            if not candidates:
                return False
            _used, victim = min(candidates)
            self._drop_entries(families[victim], write_manifest=False)

    # ---- save planning -------------------------------------------------------------------

    def _begin_op(self, tokens: torch.Tensor, keys: list[str]) -> _SaveOp | None:
        """Under ``_lock``: pick a parent, reserve the exact file size, pin, mark in flight."""
        key = keys[-1]
        if key in self._inflight:
            return None
        existing = self._entries.get(key)
        if existing is not None:
            # Same rolling key, different tokens: a hash collision. Invalidate the old family
            # member and its dependants rather than writing over a file a child still needs.
            self._drop_entry(key)
        if any(path.stem == key for _root, path, _nbytes in self._pending_unlinks):
            return None
        plan = self._plan_parent(tokens, keys)
        parent, cut = plan if plan is not None else (None, 0)
        needed = self._planned_bytes(len(tokens), cut)
        budget = self.ram_budget_bytes if self.mode == "ram" else self.disk_budget_bytes
        if needed > budget:
            return None
        before = len(self._entries)
        op = _SaveOp(key=key, tokens=tokens, keys=keys)
        self._adopt_parent(op, parent, cut)
        try:
            # Pin before eviction: the 2026-09-09 CPU pressure probe otherwise rewrote
            # 311,448 KV bytes instead of 152 by evicting its own selected parent. A zero-KV
            # child pins its donor family the same way.
            fits = self._evict_to_fit(needed)
            if not fits and parent is not None:
                self._adopt_parent(op, None, 0)
                needed = self._planned_bytes(len(tokens), 0)
                fits = needed <= budget and self._evict_to_fit(needed)
            if self.mode == "ssd" and len(self._entries) != before:
                self._write_manifest()
            if not fits:
                return None
            op.reserved_bytes = needed
            self._reserved_bytes += needed
            self._inflight.add(key)
            return op
        finally:
            if not op.reserved_bytes:
                self._adopt_parent(op, None, 0)

    def _adopt_parent(self, op: _SaveOp, parent: ParkedEntry | None, cut: int) -> None:
        """Pin the new parent's family before releasing the old pin."""
        new_root = parent.root_key if parent is not None else None
        if new_root is not None:
            self._pin(new_root)
        old_root = op.root_key
        op.parent = parent
        op.cut = cut if parent is not None else 0
        op.root_key = new_root
        op.chain = (self._chain(parent) or ()) if parent is not None else ()
        if old_root is not None:
            self._unpin(old_root)

    def _resize_reservation(self, op: _SaveOp, needed: int) -> bool:
        """Grow or shrink ``op``'s reservation to ``needed``; growth may evict unpinned families."""
        extra = needed - op.reserved_bytes
        if extra > 0 and not self._evict_to_fit(extra):
            return False
        self._reserved_bytes += extra
        op.reserved_bytes = needed
        return True

    def _replan(self, op: _SaveOp) -> bool:
        """At the writer: re-check the pinned parent and prefer any larger cut committed since."""
        if op.parent is not None and self._chain(op.parent) is None:
            self._adopt_parent(op, None, 0)
        plan = self._plan_parent(op.tokens, op.keys)
        if plan is not None:
            best, cut = plan
            if op.parent is None or cut > op.cut:
                needed = self._planned_bytes(len(op.tokens), cut)
                if self._resize_reservation(op, needed):
                    self._adopt_parent(op, best, cut)
        if op.parent is None:
            return self._resize_reservation(op, self._planned_bytes(len(op.tokens), 0))
        return True

    def _switch_to_root(self, op: _SaveOp) -> bool:
        """The source prefix no longer matches the parent bytes: fall back to a standalone root."""
        self._adopt_parent(op, None, 0)
        return self._resize_reservation(op, self._planned_bytes(len(op.tokens), 0))

    def _finish_op(self, op: _SaveOp) -> None:
        """Release whatever the operation still owns; safe to call exactly once per op."""
        self._reserved_bytes -= op.reserved_bytes
        op.reserved_bytes = 0
        if op.root_key is not None:
            root = op.root_key
            op.root_key = None
            op.parent = None
            op.chain = ()
            self._unpin(root)
        self._inflight.discard(op.key)
        self._sweep_unlinks()

    def _publish(self, op: _SaveOp, entry: ParkedEntry) -> bool:
        """Under ``_lock``: swap the reservation for the committed bytes and link the family."""
        if op.parent is not None and self._chain(op.parent) is None:
            return False
        if self._entries.get(op.key) is not None:
            return False
        entry.root_key = op.parent.root_key if op.parent is not None else op.key
        self._entries[op.key] = entry
        if op.parent is not None:
            self._children.setdefault(op.parent.key, set()).add(op.key)
        self._reserved_bytes -= op.reserved_bytes
        op.reserved_bytes = 0
        self._notify_change()
        if self.mode == "ssd":
            self._write_manifest()
        return True

    # ---- writing -----------------------------------------------------------------------

    def _prefix_matches(
        self, span: _ByteSpan, chain: tuple[ParkedEntry, ...], cut: int
    ) -> bool:
        """Byte-identity guard: the *current* source KV prefix ``[0, cut)`` must equal the
        bytes the chain would restore there.

        Equal tokens do not prove a cold recomputation produced identical floating-point KV,
        and today's save contract carries no provenance, so the reused prefix is streamed
        through the bounded windows and compared with what the ancestors recorded. A segment
        used in full is compared through its recorded region digest (one D2H+hash pass, no
        disk read). A donor clipped to a shorter cut has no stored digest for that prefix, so
        its whole KV region is read back and hashed through one window while the borrowed
        intersection is compared byte-for-byte against the staged source through the other
        window: extra reads, never a skipped check, and never a full prefix buffer. Both cost
        nothing on the write side, which is the SSD traffic this job targets. Restores may
        interleave between chunks as with the writer.
        """
        assert self._windows is not None
        ranges = self._chain_ranges(chain, cut)
        if ranges is None:
            return False
        per_token = self._kv_bytes_per_token()
        for segment, start_tokens, stop_tokens in ranges:
            if stop_tokens <= start_tokens:
                continue
            if stop_tokens < segment.token_count:
                if not self._clipped_prefix_matches(
                    span, segment, start_tokens * per_token, stop_tokens * per_token
                ):
                    return False
                continue
            digest = self._new_payload_digest(segment.digest_chunk_bytes)
            start = start_tokens * per_token
            stop = stop_tokens * per_token
            for offset in range(start, stop, self.pinned_window_bytes):
                length = min(self.pinned_window_bytes, stop - offset)
                with self._window_lock:
                    window = self._windows[0]
                    self._stage_source_chunk(span, offset, length, window)
                    digest.update(memoryview(window[:length].numpy()).cast("B"))
            if digest.hexdigest() != segment.kv_sha256:
                return False
        return True

    def _clipped_prefix_matches(
        self, span: _ByteSpan, segment: ParkedEntry, use_lo: int, use_hi: int
    ) -> bool:
        """Read ``segment``'s whole KV region, verify its digest, and compare the logical bytes
        ``[use_lo, use_hi)`` with the current source. A digest mismatch is corruption: the
        segment and its dependants are dropped so no later save or restore trusts them."""
        assert self._windows is not None
        if segment.path is None:
            return False
        per_token = self._kv_bytes_per_token()
        region_base = segment.parent_token_count * per_token
        digest = self._new_payload_digest(segment.digest_chunk_bytes)
        equal = True
        damaged: str | None = None
        try:
            with segment.path.open("rb", buffering=0) as handle:
                handle.seek(segment.kv_offset)
                for offset in range(0, segment.kv_bytes, self.pinned_window_bytes):
                    take = min(self.pinned_window_bytes, segment.kv_bytes - offset)
                    with self._window_lock:
                        disk, source = self._windows
                        raw = memoryview(disk[:take].numpy()).cast("B")
                        if handle.readinto(raw) != take:
                            damaged = "short read"
                            break
                        _release_file_cache(handle)
                        digest.update(raw)
                        chunk_lo = region_base + offset
                        lo = max(chunk_lo, use_lo)
                        hi = min(chunk_lo + take, use_hi)
                        if equal and lo < hi:
                            self._stage_source_chunk(span, lo, hi - lo, source)
                            if not torch.equal(
                                source[: hi - lo], disk[lo - chunk_lo : hi - chunk_lo]
                            ):
                                equal = False
        except OSError as exc:
            damaged = repr(exc)
        if damaged is None and digest.hexdigest() != segment.kv_sha256:
            damaged = "checksum mismatch"
        if damaged is not None:
            # The donor, not the source, is at fault: forget it and its dependants so no later
            # save or restore trusts it, and let this save fall back to a standalone root.
            logger.warning(f"KV park donor unreadable ({damaged}), dropping: {segment.path}")
            with self._lock:
                self._drop_entry(segment.key)
            return False
        return equal

    def _write_region(
        self,
        handle,
        span: _ByteSpan,
        source_offset: int,
        length: int,
        on_last_copied=None,
    ) -> str:
        assert self._windows is not None
        digest = self._new_payload_digest(_DIGEST_CHUNK_BYTES)
        for offset in range(0, length, self.pinned_window_bytes):
            # A restore owns both windows for its complete double-buffered pass. The writer
            # takes one window for one chunk at a time, so a waiting restore is delayed by at
            # most one bounded D2H-plus-write span rather than a multi-GiB file.
            with self._window_lock:
                window = self._windows[0]
                take = min(self.pinned_window_bytes, length - offset)
                self._stage_source_chunk(span, source_offset + offset, take, window)
                if offset + take == length and on_last_copied is not None:
                    on_last_copied()
                raw = memoryview(window[:take].numpy()).cast("B")
                digest.update(raw)
                _write_all(handle, raw)
                handle.flush()
                os.fsync(handle.fileno())
                _release_file_cache(handle)
        return digest.hexdigest()

    def _write_ssd(
        self,
        op: _SaveOp,
        span: _ByteSpan,
        on_source_copied=None,
    ) -> ParkedEntry:
        tokens = op.tokens
        parent = op.parent
        token_count = len(tokens)
        cut = op.cut if parent is not None else 0
        kv_offset, kv_bytes, state_offset, state_bytes = self._segment_layout(token_count, cut)
        per_token = self._kv_bytes_per_token()
        tokens_digest = _tokens_sha256(tokens)
        final = self.ssd_dir / f"{op.key}.park"
        temp = self.ssd_dir / f".{op.key}.{os.getpid()}.tmp"
        provisional = self._segment_meta(
            key=op.key,
            token_count=token_count,
            parent=parent,
            cut=cut,
            kv_sha256=_PLACEHOLDER_DIGEST,
            state_sha256=_PLACEHOLDER_DIGEST,
            tokens_sha256=tokens_digest,
        )
        # The provisional header carries placeholder digests, and its metadata digest is
        # recomputed over them, so a crash before the final header rewrite leaves a file no
        # scan will ever admit even if it were renamed.
        provisional["metadata_sha256"] = _PLACEHOLDER_DIGEST
        try:
            with temp.open("w+b", buffering=0) as handle:
                _write_all(handle, self._header(provisional))
                _write_all(handle, memoryview(tokens.numpy()).cast("B"))
                _write_all(
                    handle,
                    bytes(kv_offset - _HEADER_BYTES - token_count * torch.int32.itemsize),
                )
                kv_sha256 = self._write_region(handle, span, cut * per_token, kv_bytes)
                _write_all(handle, bytes(state_offset - kv_offset - kv_bytes))
                state_sha256 = self._write_region(
                    handle,
                    span,
                    token_count * per_token,
                    state_bytes,
                    on_last_copied=on_source_copied,
                )
                meta = self._segment_meta(
                    key=op.key,
                    token_count=token_count,
                    parent=parent,
                    cut=cut,
                    kv_sha256=kv_sha256,
                    state_sha256=state_sha256,
                    tokens_sha256=tokens_digest,
                )
                handle.seek(0)
                _write_all(handle, self._header(meta))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, final)
            _fsync_directory(self.ssd_dir)
        finally:
            temp.unlink(missing_ok=True)
        return ParkedEntry(
            key=op.key,
            token_ids=tokens,
            token_count=token_count,
            payload_bytes=kv_bytes + state_bytes,
            total_bytes=state_offset + state_bytes,
            last_used_ns=time.time_ns(),
            path=final,
            parent_key=parent.key if parent is not None else None,
            parent_token_count=cut,
            parent_metadata_sha256=parent.metadata_sha256 if parent is not None else None,
            kv_offset=kv_offset,
            kv_bytes=kv_bytes,
            state_offset=state_offset,
            state_bytes=state_bytes,
            kv_sha256=kv_sha256,
            state_sha256=state_sha256,
            tokens_sha256=tokens_digest,
            metadata_sha256=str(meta["metadata_sha256"]),
            digest_chunk_bytes=_DIGEST_CHUNK_BYTES,
        )

    # ---- public save surface -------------------------------------------------------------

    def _check_sources(
        self, input_ids: torch.Tensor, page_bases: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        tokens = _tokens_cpu(input_ids)
        if len(tokens) < self.min_tokens or len(tokens) % self.page_size:
            return None
        bases = page_bases.detach().to(device="cpu", dtype=torch.int32).flatten().clone()
        if len(bases) != len(tokens) // self.page_size:
            raise ValueError(
                f"park got {len(bases)} page bases for {len(tokens)} tokens"
            )
        return tokens, bases

    def offer(
        self, input_ids: torch.Tensor, page_bases: torch.Tensor, state_slot: int
    ) -> PendingPark | None:
        """Queue a save without blocking the scheduler; ``None`` means fall back to eviction."""
        with self._lock:
            if self._disabled or self._closed or self._save_queue is None:
                return None
            checked = self._check_sources(input_ids, page_bases)
            if checked is None:
                return None
            tokens, bases = checked
            keys = rolling_page_keys(tokens, self.page_size, self.fingerprint)
            existing = self._entries.get(keys[-1])
            if existing is not None and torch.equal(existing.token_ids, tokens):
                existing.last_used_ns = time.time_ns()
                pending = PendingPark(tokens, bases, int(state_slot))
                pending.success = True
                pending.copy_done.set()
                pending.done.set()
                return pending
            if self._save_queue.full():
                return None
            op = self._begin_op(tokens, keys)
            if op is None:
                return None
            try:
                source_ready = None
                if self._stream is not None:
                    source_ready = torch.cuda.Event(enable_timing=False)
                    source_ready.record(torch.cuda.current_stream(self.kv_pool.device))
                pending = PendingPark(
                    tokens,
                    bases,
                    int(state_slot),
                    source_ready,
                    reserved_bytes=op.reserved_bytes,
                    op=op,
                )
                self._save_queue.put_nowait(pending)
            except Full:
                self._finish_op(op)
                return None
            except Exception:
                self._finish_op(op)
                raise
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
                assert pending.op is not None
                pending.success = self.save(
                    pending.token_ids,
                    pending.page_bases,
                    pending.state_slot,
                    _on_copied=pending.copy_done.set,
                    _op=pending.op,
                )
            except Exception as exc:
                pending.error = exc
                self.note_error(f"background {self.mode} save failed: {exc!r}")
                logger.warning(f"KV parking background save failed: {exc!r}")
            finally:
                if pending is not None:
                    # _save_op completes every D2H before returning. Source ownership may move
                    # back to the allocators even when the later store operation failed.
                    pending.copy_done.set()
                    if pending.op is not None:
                        with self._lock:
                            self._finish_op(pending.op)
                    pending.done.set()
                self._save_queue.task_done()

    def save(
        self,
        input_ids: torch.Tensor,
        page_bases: torch.Tensor,
        state_slot: int,
        *,
        _on_copied=None,
        _op: _SaveOp | None = None,
    ) -> bool:
        """Copy one complete entry before its source pages/state are released.

        The potentially multi-second copy/write runs outside ``_lock`` so a worker can never
        block the scheduler's next non-blocking offer or status read. ``_op`` is the worker's
        already-reserved plan from ``offer``: it is consumed here, never reserved twice, and
        released by the worker.
        """

        def copied() -> None:
            if _on_copied is not None:
                _on_copied()

        if _op is not None:
            bases = page_bases.detach().to(device="cpu", dtype=torch.int32).flatten().clone()
            return self._save_op(_op, bases, int(state_slot), on_copied=copied)
        checked = self._check_sources(input_ids, page_bases)
        if checked is None:
            copied()
            return False
        tokens, bases = checked
        keys = rolling_page_keys(tokens, self.page_size, self.fingerprint)
        with self._lock:
            if self._disabled:
                copied()
                return False
            existing = self._entries.get(keys[-1])
            if existing is not None and torch.equal(existing.token_ids, tokens):
                existing.last_used_ns = time.time_ns()
                copied()
                return True
            op = self._begin_op(tokens, keys)
            if op is None:
                copied()
                return False
        try:
            return self._save_op(op, bases, int(state_slot), on_copied=copied)
        finally:
            with self._lock:
                self._finish_op(op)

    def _save_op(
        self,
        op: _SaveOp,
        bases: torch.Tensor,
        state_slot: int,
        *,
        on_copied,
    ) -> bool:
        """Shared writer for queued and synchronous saves; the caller owns ``_finish_op``."""
        try:
            with self._writer_lock:
                with self._lock:
                    if self._disabled or not self._replan(op):
                        on_copied()
                        return False
                    chain = op.chain
                span = self._entry_views(bases, state_slot)
                expected = self.payload_bytes(len(op.tokens))
                if span.nbytes != expected:
                    raise RuntimeError(f"KV park payload {span.nbytes} != expected {expected}")
                if self.mode == "ram":
                    buffer = self._copy_to_ram(span)
                    on_copied()
                    entry = ParkedEntry(
                        key=op.key,
                        token_ids=op.tokens,
                        token_count=len(op.tokens),
                        payload_bytes=span.nbytes,
                        total_bytes=op.reserved_bytes,
                        last_used_ns=time.time_ns(),
                        ram_buffer=buffer,
                    )
                else:
                    if chain and not self._prefix_matches(span, chain, op.cut):
                        with self._lock:
                            if not self._switch_to_root(op):
                                logger.info(
                                    "KV park: prefix bytes changed and a standalone entry "
                                    "does not fit; declining this park"
                                )
                                if self._stream is not None:
                                    self._stream.synchronize()
                                on_copied()
                                return False
                    entry = self._write_ssd(op, span, on_source_copied=on_copied)
                    on_copied()
                with self._lock:
                    if not self._publish(op, entry):
                        # The parent was invalidated while this file was being written.
                        if entry.path is not None:
                            entry.path.unlink(missing_ok=True)
                        return False
                return True
        except Exception as exc:
            # A failed enqueue/copy can still leave work on the private CUDA stream. Fence it
            # before copy_done lets the scheduler recycle and overwrite the source pages.
            if self._stream is not None:
                self._stream.synchronize()
            on_copied()
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
        *,
        keys: list[str] | None = None,
    ) -> ParkedEntry | None:
        with self._lock:
            tokens = _tokens_cpu(input_ids)
            if keys is None:
                keys = rolling_page_keys(tokens, self.page_size, self.fingerprint)
            elif len(keys) != len(tokens) // self.page_size:
                raise ValueError(
                    f"precomputed KV park keys cover {len(keys)} pages, "
                    f"expected {len(tokens) // self.page_size}"
                )
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
                # Every committed endpoint is independently resumable: an ordinary finish, a
                # scheduler-saved prompt checkpoint or a zero-KV shorter checkpoint, each with
                # its own state at exactly ``token_count``. A longer entry is never sliced into
                # a synthetic shorter hit: there is no saved state at that cut. An entry whose
                # chain broke underneath it is dropped here so a shorter valid endpoint can
                # still answer.
                if entry.token_count != token_count or (
                    entry.path is not None and self._chain(entry) is None
                ):
                    self._drop_entry(key)
                    continue
                entry.last_used_ns = time.time_ns()
                self._hits += 1
                return entry
            self._misses += 1
            return None

    # ---- restore -----------------------------------------------------------------------

    @staticmethod
    def _copy_restored_chunk(
        window: torch.Tensor,
        chunk_offset: int,
        chunk_length: int,
        source_offset: int,
        length: int,
        span: _ByteSpan,
        copy_lo: int | None = None,
        copy_hi: int | None = None,
    ) -> bool:
        """Copy the part of one read chunk that lands inside the destination span and, when
        given, inside the segment's effective logical range ``[copy_lo, copy_hi)``: a donor
        clipped by a shorter checkpoint must never write its unused tail over the target's
        own suffix, not even into pages the caller freshly allocated."""
        restore_start = max(chunk_offset, source_offset)
        restore_stop = min(chunk_offset + chunk_length, source_offset + length)
        if copy_lo is not None:
            restore_start = max(restore_start, copy_lo)
        if copy_hi is not None:
            restore_stop = min(restore_stop, copy_hi)
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

    def _read_region(
        self,
        ctx: dict,
        segment: ParkedEntry,
        handle,
        reader,
        *,
        file_offset: int,
        length: int,
        expected_sha256: str | None,
        span: _ByteSpan,
        logical_base: int,
        dest_offset: int,
        dest_length: int,
        cuda: bool,
        copy_lo: int | None = None,
        copy_hi: int | None = None,
    ) -> None:
        """Read one checksummed file region through the alternating windows.

        Chunk ``o`` of the region holds logical snapshot bytes ``logical_base + o``; the part
        that overlaps ``[dest_offset, dest_offset + dest_length)`` and the segment's effective
        range ``[copy_lo, copy_hi)`` lands in ``span``. Bytes before the caller's live prefix
        or past a clipping cut are still read and hashed -- chain integrity is checked
        conservatively -- but never copied anywhere.
        """
        assert self._windows is not None
        if expected_sha256 is None:
            raise ParkEntryRejected(f"KV park region checksum missing: {segment.path}")
        timing = ctx["timing"]
        events = ctx["events"]
        digest = self._new_payload_digest(segment.digest_chunk_bytes)
        if handle is not None:
            handle.seek(file_offset)
        for offset in range(0, length, self.pinned_window_bytes):
            index = ctx["chunk"] % len(self._windows)
            ctx["chunk"] += 1
            if cuda and events[index] is not None:
                # Only wait when this tray is about to be reused. Reading into the other
                # tray meanwhile overlaps disk block N+1 with H2D block N, matching M1.
                mark = time.perf_counter()
                events[index].synchronize()
                timing["tray_wait_ms"] += (time.perf_counter() - mark) * 1000.0
                events[index] = None
            chunk_length = min(self.pinned_window_bytes, length - offset)
            window = self._windows[index]
            raw = memoryview(window[:chunk_length].numpy()).cast("B")
            mark = time.perf_counter()
            if reader is not None:
                got = reader.read_into(raw, file_offset + offset, chunk_length)
            else:
                got = handle.readinto(raw)
            timing["read_ms"] += (time.perf_counter() - mark) * 1000.0
            if got != chunk_length:
                raise ParkEntryRejected(f"short KV park payload read: {segment.path}")
            if handle is not None:
                _release_file_cache(handle)
            mark = time.perf_counter()
            digest.update(raw)
            timing["hash_ms"] += (time.perf_counter() - mark) * 1000.0
            mark = time.perf_counter()
            if cuda:
                with torch.cuda.stream(self._stream):
                    copied = self._copy_restored_chunk(
                        window, logical_base + offset, chunk_length,
                        dest_offset, dest_length, span, copy_lo, copy_hi,
                    )
                    if copied:
                        done = torch.cuda.Event(enable_timing=False)
                        done.record(self._stream)
                        events[index] = done
            else:
                self._copy_restored_chunk(
                    window, logical_base + offset, chunk_length,
                    dest_offset, dest_length, span, copy_lo, copy_hi,
                )
            timing["copy_ms"] += (time.perf_counter() - mark) * 1000.0
        if digest.hexdigest() != expected_sha256:
            raise ParkEntryRejected(f"KV park payload checksum mismatch: {segment.path}")

    def _read_payload_cpu(self, ctx: dict, segment: ParkedEntry, handle, reader, **region) -> None:
        self._read_region(ctx, segment, handle, reader, cuda=False, **region)

    def _read_payload_cuda(self, ctx: dict, segment: ParkedEntry, handle, reader, **region) -> None:
        assert self._stream is not None
        self._read_region(ctx, segment, handle, reader, cuda=True, **region)

    def _restore_chain(
        self,
        chain: tuple[ParkedEntry, ...],
        kv_span: _ByteSpan,
        state_span: _ByteSpan,
        page_offset: int,
        timing: dict[str, float],
        progress: dict,
    ) -> None:
        """Read each contributing segment's whole KV region once, copying only its effective
        range, then only the target's state region.

        A segment whose effective range is empty (a zero-KV checkpoint, or a longer donor
        wholly behind a shorter one) lends no bytes, so only its identity is verified.
        """
        assert self._windows is not None
        target = chain[-1]
        per_token = self._kv_bytes_per_token()
        dest_offset = page_offset * self.page_size * per_token
        ranges = self._chain_ranges(chain, target.token_count)
        if ranges is None:
            raise ParkEntryRejected("SSD parked chain does not tile its prefix")
        ctx = {"timing": timing, "events": [None] * len(self._windows), "chunk": 0}
        cuda = self._stream is not None
        with self._window_lock:
            for segment, start_tokens, stop_tokens in ranges:
                progress["key"] = segment.key
                if segment.path is None:
                    raise ParkEntryRejected("SSD parked entry has no file")
                mark = time.perf_counter()
                meta = self._parse_header(segment.path)
                if (
                    meta["metadata_sha256"] != segment.metadata_sha256
                    or meta["key"] != segment.key
                ):
                    raise ParkEntryRejected(f"KV park file changed identity: {segment.path}")
                disk_tokens = self._read_tokens(
                    segment.path, segment.token_count, segment.tokens_sha256 or ""
                )
                if not torch.equal(disk_tokens, segment.token_ids):
                    raise ParkEntryRejected("SSD parked token verification failed")
                timing["header_ms"] += (time.perf_counter() - mark) * 1000.0
                reader = self._unbuffered_reader(segment.path) if cuda else None
                handle = None if reader is not None else segment.path.open("rb", buffering=0)
                try:
                    read = self._read_payload_cuda if cuda else self._read_payload_cpu
                    if stop_tokens > start_tokens or segment.kv_bytes == 0:
                        read(
                            ctx, segment, handle, reader,
                            file_offset=segment.kv_offset,
                            length=segment.kv_bytes,
                            expected_sha256=segment.kv_sha256,
                            span=kv_span,
                            logical_base=segment.parent_token_count * per_token,
                            dest_offset=dest_offset,
                            dest_length=kv_span.nbytes,
                            copy_lo=start_tokens * per_token,
                            copy_hi=stop_tokens * per_token,
                        )
                    if segment is target:
                        read(
                            ctx, segment, handle, reader,
                            file_offset=segment.state_offset,
                            length=segment.state_bytes,
                            expected_sha256=segment.state_sha256,
                            span=state_span,
                            logical_base=0,
                            dest_offset=0,
                            dest_length=state_span.nbytes,
                        )
                finally:
                    if reader is not None:
                        reader.close()
                    if handle is not None:
                        handle.close()
            if cuda:
                mark = time.perf_counter()
                self._stream.synchronize()
                timing["sync_ms"] += (time.perf_counter() - mark) * 1000.0

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
        progress = {"key": entry.key}
        with self._lock:
            # Staleness is not corruption: a replacement under the same key may be valid.
            if self._entries.get(entry.key) is not entry:
                raise ParkEntryRejected("parked entry is no longer current")
            total_pages = entry.token_count // self.page_size
            if page_offset < 0 or page_offset > total_pages:
                raise ValueError("restore page offset is outside the parked entry")
            mark = time.perf_counter()
            bases = page_bases.detach().to(device="cpu", dtype=torch.int32).flatten()
            if len(bases) != total_pages - page_offset:
                raise ValueError("restore target page count does not match parked suffix")
            # Build each destination view once. The redundant SSD split cost 105-109 ms
            # for 2,048 CPU pages in the 2026-09-09 review, outside reported views_ms.
            if entry.ram_buffer is not None:
                span = self._entry_views(bases, state_slot)
                view_bytes, view_count = span.nbytes, len(span.views)
            else:
                kv_span, state_span = self._split_views(bases, state_slot)
                view_bytes = kv_span.nbytes + state_span.nbytes
                view_count = len(kv_span.views) + len(state_span.views)
            per_page_bytes = self.page_size * self._kv_bytes_per_token()
            state_bytes = self._state_bytes()
            source_offset = page_offset * per_page_bytes
            suffix_bytes = (total_pages - page_offset) * per_page_bytes + state_bytes
            if view_bytes != suffix_bytes:
                raise ParkEntryRejected("restore target layout does not match parked suffix")
            timing["views_ms"] = (time.perf_counter() - mark) * 1000.0
            timing["views"] = float(view_count)
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
                    chain = self._chain(entry)
                    if chain is None:
                        raise ParkEntryRejected("SSD parked chain is incomplete")
                    timing["segments"] = float(len(chain))
                    if (
                        kv_span.nbytes != suffix_bytes - state_bytes
                        or state_span.nbytes != state_bytes
                    ):
                        raise ParkEntryRejected("restore target regions do not split cleanly")
                    self._restore_chain(chain, kv_span, state_span, page_offset, timing, progress)
            except Exception:
                if self._stream is not None:
                    self._stream.synchronize()
                # Whatever segment failed takes its dependants with it; the ancestors it does
                # not reference stay valid on their own.
                self._drop_entry(progress["key"])
                if progress["key"] != entry.key:
                    self._drop_entry(entry.key)
                raise
            finally:
                self._last_restore_ms = (time.perf_counter() - started) * 1000.0
                timing["total_ms"] = self._last_restore_ms
                self._last_restore_breakdown = timing
                logger.debug(f"KV park restore breakdown: {timing}")

    # ---- manifest and startup ----------------------------------------------------------

    def _manifest_doc(self) -> dict:
        return {
            "version": _VERSION,
            "fingerprint": self.fingerprint,
            "entries": [
                {
                    "key": entry.key,
                    "file": entry.path.name,
                    "token_count": entry.token_count,
                    "parent_key": entry.parent_key,
                    "parent_token_count": entry.parent_token_count,
                    "kv_bytes": entry.kv_bytes,
                    "state_bytes": entry.state_bytes,
                    "payload_bytes": entry.payload_bytes,
                    "kv_sha256": entry.kv_sha256,
                    "state_sha256": entry.state_sha256,
                    "metadata_sha256": entry.metadata_sha256,
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
        tokens = self._read_tokens(path, token_count, str(meta["tokens_sha256"]))
        expected_key = rolling_page_keys(tokens, self.page_size, self.fingerprint)[-1]
        if key != expected_key or path.stem != key:
            raise ParkEntryRejected(f"KV park key/token mismatch: {path}")
        stat = path.stat()
        return ParkedEntry(
            key=key,
            token_ids=tokens,
            token_count=token_count,
            payload_bytes=int(meta["payload_bytes"]),
            total_bytes=stat.st_size,
            last_used_ns=last_used_ns or stat.st_mtime_ns,
            path=path,
            parent_key=meta["parent_key"],
            parent_token_count=int(meta["parent_token_count"]),
            parent_metadata_sha256=meta["parent_metadata_sha256"],
            kv_offset=int(meta["kv_offset"]),
            kv_bytes=int(meta["kv_bytes"]),
            state_offset=int(meta["state_offset"]),
            state_bytes=int(meta["state_bytes"]),
            kv_sha256=str(meta["kv_sha256"]),
            state_sha256=str(meta["state_sha256"]),
            tokens_sha256=str(meta["tokens_sha256"]),
            metadata_sha256=str(meta["metadata_sha256"]),
            digest_chunk_bytes=int(meta["payload_digest_chunk_bytes"]),
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
        """Two-phase startup: admit valid files, then expose only complete chains.

        The manifest is an index for LRU times and file names, never the authority for parent
        links or digests: those come from each file's own validated header and token list.
        """
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
                    or not _is_hex(stem, 32)
                ):
                    raise ValueError("manifest entry escapes the KV parking directory")
                paths.append((self.ssd_dir / filename, int(row["last_used_ns"])))
        except Exception:
            paths = [(path, None) for path in sorted(self.ssd_dir.glob("*.park"))]
        candidates: dict[str, ParkedEntry] = {}
        seen: set[Path] = set()

        def admit(path: Path, last_used: int | None) -> None:
            if path in seen or path.is_symlink() or not path.is_file():
                return
            seen.add(path)
            try:
                entry = self._entry_from_file(path, last_used)
            except Exception:
                path.unlink(missing_ok=True)
                return
            candidates[entry.key] = entry

        for path, last_used in paths:
            admit(path, last_used)
        # A valid but stale manifest must not orphan extra files forever; validate the extras too.
        for path in sorted(self.ssd_dir.glob("*.park")):
            admit(path, None)
        # Under v4 ascending length was a topological order; a v5 shorter checkpoint is a child
        # shorter than its parent, so parent identity is resolved over the actual links
        # instead: walk each candidate's ancestry once with an explicit visiting set, so a
        # missing, rejected or different parent, or a cycle, rejects the child and everything
        # that depends on it, and a valid shorter child of a longer parent is admitted.
        resolved: dict[str, bool] = {}
        for start in candidates:
            if start in resolved:
                continue
            trail: list[str] = []
            visiting: set[str] = set()
            key = start
            while True:
                if key in resolved:
                    verdict = resolved[key]
                    break
                entry = candidates.get(key)
                if entry is None or key in visiting:
                    verdict = False
                    break
                visiting.add(key)
                trail.append(key)
                if entry.parent_key is None:
                    verdict = True
                    break
                key = entry.parent_key
            for key in reversed(trail):
                entry = candidates[key]
                if verdict and entry.parent_key is not None:
                    verdict = self._parent_matches(candidates[entry.parent_key], entry)
                resolved[key] = verdict
        self._entries = {}
        self._children = {}
        for key, entry in candidates.items():
            if not resolved.get(key, False):
                assert entry.path is not None
                entry.path.unlink(missing_ok=True)
                continue
            self._entries[key] = entry
        for entry in self._entries.values():
            root = entry
            while root.parent_key is not None:
                root = self._entries[root.parent_key]
            entry.root_key = root.key
            if entry.parent_key is not None:
                self._children.setdefault(entry.parent_key, set()).add(entry.key)
        # Belt and braces: a chain whose effective ranges do not tile its prefix is unusable.
        broken = [entry for entry in self._entries.values() if self._chain(entry) is None]
        for entry in broken:
            self._drop_entry(entry.key, write_manifest=False)
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
            self._sweep_unlinks()
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
