"""Host-RAM and SSD parking for page-aligned QSA/GDN prefix snapshots.

A parked entry is deliberately opaque to the attention and recurrent-state code: it is the
exact bytes returned by ``QSAKVCache.page_byte_views`` plus one complete
``LinearStatePool.slot_byte_views`` snapshot.  The scheduler copies those bytes before returning
any page or state slot to a free list, then a later match restores into ordinary newly-allocated
pages and inserts the prefix into the unchanged hybrid radix tree.

The SSD format uses a 4-KiB header, verbatim int32 token ids, and a 4-KiB-aligned payload.  Every
read validates the model/layout fingerprint and the token ids, so stale files or a rolling-hash
collision become a cache miss rather than a numerics change.  SSD reads use the same bounded,
double-buffered pinned-window shape measured by ``scripts/bench/kv_parking_bench.py``; the file is
the source of truth and no full parked entry remains resident in host RAM.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, Sequence

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_MAGIC = b"FTKVPARK"
_VERSION = 1
_HEADER_BYTES = 4096
_ALIGNMENT = 4096
_COPY_BYTES = 32 << 20


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
    path: Path | None = None
    ram_tensors: tuple[torch.Tensor, ...] | None = None


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise ValueError(f"park view must be contiguous, got shape={tuple(tensor.shape)}")
    return tensor.view(torch.uint8).reshape(-1)


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _tokens_cpu(input_ids: torch.Tensor) -> torch.Tensor:
    return input_ids.detach().to(device="cpu", dtype=torch.int32).contiguous().clone()


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
    tp_size: int,
    kv_pool,
    state_pool,
) -> str:
    """Stable digest of the checkpoint identity and every parked byte-layout dimension."""
    root = Path(model_path).expanduser().resolve()
    index_candidates = sorted(root.glob("*.safetensors.index.json"))
    identity_file = index_candidates[0] if index_candidates else root / "config.json"
    identity_hash = "missing"
    if identity_file.is_file():
        digest = hashlib.sha256()
        with identity_file.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        identity_hash = digest.hexdigest()
    qsa_shape = tuple(int(v) for v in kv_pool._kv_buffer.shape)
    cmp_shape = tuple(int(v) for v in kv_pool._cmp_k_buffer.shape)
    state_shapes = [
        (tuple(int(v) for v in view.shape), str(view.dtype))
        for view in state_pool.slot_byte_views(state_pool.padding_slot)
    ]
    doc = {
        "checkpoint": str(root),
        "checkpoint_index_sha256": identity_hash,
        "page_size": int(page_size),
        "index_ratio": int(kv_pool.index_ratio),
        "kv_dtype": str(kv_pool.dtype),
        "index_dtype": str(kv_pool._index_dtype),
        "kv_shape": qsa_shape,
        "cmp_shape": cmp_shape,
        "state_shapes": state_shapes,
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
        self._disabled = False
        self._lock = RLock()
        self._stream = None
        self._windows: tuple[torch.Tensor, torch.Tensor] | None = None
        try:
            if kv_pool.device.type == "cuda":
                self._stream = torch.cuda.Stream(device=kv_pool.device)
            if mode == "ssd":
                self.ssd_dir.mkdir(parents=True, exist_ok=True)
                self._windows = (
                    self._allocate_window(self.pinned_window_bytes),
                    self._allocate_window(self.pinned_window_bytes),
                )
                self._load_or_scan()
        except Exception as exc:
            self._disabled = True
            self._stream = None
            self._windows = None
            self._entries.clear()
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
        directory = str(
            getattr(config, "kv_park_ssd_dir", "~/.cache/freetoken/kv-park")
        )
        fingerprint = build_model_fingerprint(
            model_path=config.model_path,
            page_size=config.page_size,
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

    def payload_bytes(self, token_count: int) -> int:
        if token_count < 0 or token_count % self.page_size:
            raise ValueError("park token_count must be page aligned")
        page_zero = self.kv_pool.page_byte_views(0)
        per_page = sum(_tensor_nbytes(view) for view in page_zero)
        state = sum(
            _tensor_nbytes(view)
            for view in self.state_pool.slot_byte_views(self.state_pool.padding_slot)
        )
        return token_count // self.page_size * per_page + state

    def storage_bytes(self, token_count: int) -> int:
        payload = self.payload_bytes(token_count)
        if self.mode == "ram":
            return payload + token_count * torch.int32.itemsize
        return _align_up(_HEADER_BYTES + token_count * torch.int32.itemsize) + payload

    def _entry_views(
        self, page_bases: torch.Tensor, state_slot: int
    ) -> tuple[torch.Tensor, ...]:
        bases = page_bases.detach().to(device="cpu", dtype=torch.int64).tolist()
        views: list[torch.Tensor] = []
        for base in bases:
            if base < 0 or base % self.page_size:
                raise ValueError(f"park page base must be aligned, got {base}")
            views.extend(self.kv_pool.page_byte_views(base // self.page_size))
        views.extend(self.state_pool.slot_byte_views(int(state_slot)))
        return tuple(views)

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

    def _copy_to_ram(self, views: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        host: list[torch.Tensor] = []
        for view in views:
            if view.device.type == "cuda":
                from freetoken.kernel.pinned import alloc_pinned_tensor

                out = alloc_pinned_tensor(*view.shape, dtype=view.dtype)
            else:
                out = torch.empty_like(view, device="cpu")
            host.append(out)
        if self._stream is None:
            for source, target in zip(views, host, strict=True):
                target.copy_(source)
        else:
            current = torch.cuda.current_stream(self.kv_pool.device)
            self._stream.wait_stream(current)
            with torch.cuda.stream(self._stream):
                for source, target in zip(views, host, strict=True):
                    target.copy_(source, non_blocking=True)
            self._stream.synchronize()
        return tuple(host)

    @staticmethod
    def _copy_span_to_window(
        views: Sequence[torch.Tensor], offset: int, length: int, window: torch.Tensor
    ) -> None:
        remaining = length
        global_offset = offset
        window_offset = 0
        for view in views:
            raw = _byte_view(view)
            if global_offset >= raw.numel():
                global_offset -= raw.numel()
                continue
            take = min(remaining, raw.numel() - global_offset)
            window[window_offset : window_offset + take].copy_(
                raw[global_offset : global_offset + take], non_blocking=True
            )
            remaining -= take
            window_offset += take
            global_offset = 0
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(f"park source ended {remaining} bytes short")

    @staticmethod
    def _copy_window_to_span(
        window: torch.Tensor, offset: int, length: int, views: Sequence[torch.Tensor]
    ) -> None:
        remaining = length
        global_offset = offset
        window_offset = 0
        for view in views:
            raw = _byte_view(view)
            if global_offset >= raw.numel():
                global_offset -= raw.numel()
                continue
            take = min(remaining, raw.numel() - global_offset)
            raw[global_offset : global_offset + take].copy_(
                window[window_offset : window_offset + take], non_blocking=True
            )
            remaining -= take
            window_offset += take
            global_offset = 0
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(f"park destination ended {remaining} bytes short")

    def _header(
        self, *, key: str, token_count: int, payload_bytes: int, payload_offset: int
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
        if token_count < self.min_tokens or token_count % self.page_size:
            raise ParkEntryRejected(f"invalid KV park token count: {path}")
        if payload_bytes != self.payload_bytes(token_count):
            raise ParkEntryRejected(f"KV park payload-size mismatch: {path}")
        if payload_offset != _align_up(_HEADER_BYTES + token_count * torch.int32.itemsize):
            raise ParkEntryRejected(f"KV park payload-offset mismatch: {path}")
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

    def _write_ssd(
        self,
        *,
        key: str,
        tokens: torch.Tensor,
        views: Sequence[torch.Tensor],
        payload_bytes: int,
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
        )
        try:
            with temp.open("wb", buffering=0) as handle:
                handle.write(header)
                handle.write(memoryview(tokens.numpy()).cast("B"))
                handle.write(bytes(payload_offset - _HEADER_BYTES - _tensor_nbytes(tokens)))
                for chunk, offset in enumerate(
                    range(0, payload_bytes, self.pinned_window_bytes)
                ):
                    window = self._windows[chunk % 2]
                    length = min(self.pinned_window_bytes, payload_bytes - offset)
                    if self._stream is None:
                        self._copy_span_to_window(views, offset, length, window)
                    else:
                        current = torch.cuda.current_stream(self.kv_pool.device)
                        self._stream.wait_stream(current)
                        with torch.cuda.stream(self._stream):
                            self._copy_span_to_window(views, offset, length, window)
                        self._stream.synchronize()
                    handle.write(memoryview(window[:length].numpy()).cast("B"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, final)
        finally:
            temp.unlink(missing_ok=True)
        now = time.monotonic_ns()
        return ParkedEntry(
            key=key,
            token_ids=tokens,
            token_count=len(tokens),
            payload_bytes=payload_bytes,
            total_bytes=final.stat().st_size,
            last_used_ns=now,
            path=final,
        )

    def save(
        self, input_ids: torch.Tensor, page_bases: torch.Tensor, state_slot: int
    ) -> bool:
        """Copy one complete entry before its source pages/state are released.

        Returns ``False`` when the entry is below the floor, exceeds its tier's whole budget, or
        parking was disabled after an allocation/IO failure.  A duplicate key is a successful
        no-op: the existing immutable copy remains the source of truth.
        """
        with self._lock:
            if self._disabled:
                return False
            tokens = _tokens_cpu(input_ids)
            if len(tokens) < self.min_tokens or len(tokens) % self.page_size:
                return False
            bases = page_bases.detach().to(device="cpu", dtype=torch.int32).flatten()
            if len(bases) != len(tokens) // self.page_size:
                raise ValueError(
                    f"park got {len(bases)} page bases for {len(tokens)} tokens"
                )
            key = rolling_page_keys(tokens, self.page_size, self.fingerprint)[-1]
            existing = self._entries.get(key)
            if existing is not None:
                if torch.equal(existing.token_ids, tokens):
                    existing.last_used_ns = time.monotonic_ns()
                    return True
                self._drop_entry(key)
            views = self._entry_views(bases, state_slot)
            payload = sum(_tensor_nbytes(view) for view in views)
            expected = self.payload_bytes(len(tokens))
            if payload != expected:
                raise RuntimeError(f"KV park payload {payload} != expected {expected}")
            needed = self.storage_bytes(len(tokens))
            budget = self.ram_budget_bytes if self.mode == "ram" else self.disk_budget_bytes
            if needed > budget:
                return False
            self._evict_to_fit(needed)
            try:
                if self.mode == "ram":
                    tensors = self._copy_to_ram(views)
                    entry = ParkedEntry(
                        key=key,
                        token_ids=tokens,
                        token_count=len(tokens),
                        payload_bytes=payload,
                        total_bytes=needed,
                        last_used_ns=time.monotonic_ns(),
                        ram_tensors=tensors,
                    )
                else:
                    entry = self._write_ssd(
                        key=key, tokens=tokens, views=views, payload_bytes=payload
                    )
                self._entries[key] = entry
                if self.mode == "ssd":
                    self._write_manifest()
                return True
            except Exception as exc:
                self._disabled = True
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
                entry.last_used_ns = time.monotonic_ns()
                self._hits += 1
                return entry
            self._misses += 1
            return None

    def _read_payload_cpu(
        self,
        entry: ParkedEntry,
        views: Sequence[torch.Tensor],
        payload_offset: int,
        source_offset: int,
        length: int,
    ) -> None:
        assert entry.path is not None and self._windows is not None
        with entry.path.open("rb", buffering=0) as handle:
            handle.seek(payload_offset + source_offset)
            copied = 0
            chunk = 0
            while copied < length:
                chunk_length = min(self.pinned_window_bytes, length - copied)
                window = self._windows[chunk % 2]
                got = handle.readinto(memoryview(window[:chunk_length].numpy()).cast("B"))
                if got != chunk_length:
                    raise ParkEntryRejected(f"short KV park payload read: {entry.path}")
                self._copy_window_to_span(window, copied, chunk_length, views)
                copied += chunk_length
                chunk += 1

    def _read_payload_cuda(
        self,
        entry: ParkedEntry,
        views: Sequence[torch.Tensor],
        payload_offset: int,
        source_offset: int,
        length: int,
    ) -> None:
        assert entry.path is not None and self._windows is not None and self._stream is not None
        reader = None
        try:
            if sys.platform == "win32":
                from freetoken.moe.win_io import UnbufferedReader

                reader = UnbufferedReader(str(entry.path))
            handle = None if reader is not None else entry.path.open("rb", buffering=0)
            if handle is not None:
                handle.seek(payload_offset + source_offset)
            events: list[torch.cuda.Event | None] = [None, None]
            copied = 0
            chunk = 0
            while copied < length:
                index = chunk % 2
                event = events[index]
                if event is not None:
                    event.synchronize()
                chunk_length = min(self.pinned_window_bytes, length - copied)
                window = self._windows[index]
                if reader is not None:
                    got = reader.read_into(
                        memoryview(window[:chunk_length].numpy()).cast("B"),
                        payload_offset + source_offset + copied,
                        chunk_length,
                    )
                else:
                    got = handle.readinto(
                        memoryview(window[:chunk_length].numpy()).cast("B")
                    )
                if got != chunk_length:
                    raise ParkEntryRejected(f"short KV park payload read: {entry.path}")
                with torch.cuda.stream(self._stream):
                    self._copy_window_to_span(window, copied, chunk_length, views)
                    done = torch.cuda.Event(enable_timing=False)
                    done.record(self._stream)
                    events[index] = done
                copied += chunk_length
                chunk += 1
            self._stream.synchronize()
        finally:
            if reader is not None:
                reader.close()
            if "handle" in locals() and handle is not None:
                handle.close()

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
        with self._lock:
            total_pages = entry.token_count // self.page_size
            if page_offset < 0 or page_offset > total_pages:
                raise ValueError("restore page offset is outside the parked entry")
            bases = page_bases.detach().to(device="cpu", dtype=torch.int32).flatten()
            if len(bases) != total_pages - page_offset:
                raise ValueError("restore target page count does not match parked suffix")
            views = self._entry_views(bases, state_slot)
            per_page_views = self.kv_pool.page_byte_views(0)
            per_page_bytes = sum(_tensor_nbytes(view) for view in per_page_views)
            state_views = self.state_pool.slot_byte_views(self.state_pool.padding_slot)
            state_bytes = sum(_tensor_nbytes(view) for view in state_views)
            source_offset = page_offset * per_page_bytes
            suffix_bytes = (total_pages - page_offset) * per_page_bytes + state_bytes
            if sum(_tensor_nbytes(view) for view in views) != suffix_bytes:
                raise ParkEntryRejected("restore target layout does not match parked suffix")
            try:
                if entry.ram_tensors is not None:
                    page_view_count = len(per_page_views)
                    page_end = total_pages * page_view_count
                    sources = (
                        entry.ram_tensors[page_offset * page_view_count : page_end]
                        + entry.ram_tensors[page_end:]
                    )
                    if len(sources) != len(views):
                        raise ParkEntryRejected("RAM parked view count changed")
                    if self._stream is None:
                        for source, target in zip(sources, views, strict=True):
                            target.copy_(source)
                    else:
                        with torch.cuda.stream(self._stream):
                            for source, target in zip(sources, views, strict=True):
                                target.copy_(source, non_blocking=True)
                        self._stream.synchronize()
                else:
                    if entry.path is None:
                        raise ParkEntryRejected("SSD parked entry has no file")
                    meta = self._parse_header(entry.path)
                    disk_tokens = self._read_tokens(entry.path, entry.token_count)
                    if not torch.equal(disk_tokens, entry.token_ids):
                        raise ParkEntryRejected("SSD parked token verification failed")
                    payload_offset = int(meta["payload_offset"])
                    if self._stream is None:
                        self._read_payload_cpu(
                            entry, views, payload_offset, source_offset, suffix_bytes
                        )
                    else:
                        self._read_payload_cuda(
                            entry, views, payload_offset, source_offset, suffix_bytes
                        )
            except Exception:
                if self._stream is not None:
                    self._stream.synchronize()
                self._drop_entry(entry.key)
                raise
            finally:
                self._last_restore_ms = (time.perf_counter() - started) * 1000.0

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
            path=path,
        )

    def _load_or_scan(self) -> None:
        manifest = self.ssd_dir / "park.json"
        paths: list[tuple[Path, int | None]] = []
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            if doc.get("version") != _VERSION or doc.get("fingerprint") != self.fingerprint:
                raise ValueError("manifest version/fingerprint mismatch")
            for row in doc.get("entries", []):
                paths.append((self.ssd_dir / str(row["file"]), int(row["last_used_ns"])))
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
                "disabled": self._disabled,
            }

    def close(self) -> None:
        with self._lock:
            if self.mode == "ssd":
                self._write_manifest()


__all__ = [
    "ParkEntryRejected",
    "ParkStore",
    "ParkedEntry",
    "build_model_fingerprint",
    "rolling_page_keys",
]
