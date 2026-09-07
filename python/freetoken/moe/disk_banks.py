"""Disk-backed expert weight copy and staging readers for SSD-tier offload (J2).

When host RAM is squeezed, MoE layers step down from pinned host memory to this on-disk
copy. Experts are stored as raw pre-repacked bank files:
    <root>/L<layer>.<bank>.bin
Rows are in bank order, length = feat_bytes of that bank. A manifest.json tracks the
checkpoint identity (shard names, sizes, mtimes), bank shapes/dtypes, and per-layer
completion.

DiskLayerReader gathers routed expert rows into a shared pinned staging buffer via
os.pread, bypassing the slot-cache LRU entirely to avoid host-side pointer churn.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


def _set_lowest_io_priority() -> None:
    """Demote current thread to idle scheduling priority for non-disruptive background writes.

    Under WSL2 / Linux, background writes to the SSD must not starve interactive foreground
    decode reads (measured 4.2 GB/s roofline; unthrottled writes introduce multi-second latency
    hiccups).
    """
    try:
        os.nice(19)
    except Exception:
        pass
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        # syscall(SYS_ioprio_set=251, IOPRIO_WHO_PROCESS=1, who=0, ioprio=(IOPRIO_CLASS_IDLE=3 << 13))
        libc.syscall(251, 1, 0, 3 << 13)
    except Exception:
        pass


def _scan_checkpoint_identity(model_path: Path) -> list[dict[str, Any]]:
    """Record shard names, sizes, and mtimes to ensure disk copy matches the active checkpoint.

    If a model directory is swapped or touched, old cached expert files must be invalidated
    rather than served with silent numerical corruption.
    """
    if not model_path.is_dir():
        return []
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        # Dummy or test directories without safetensors: scan regular non-hidden files
        shards = sorted(
            p for p in model_path.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix != ".tmp"
        )
    identity = []
    for p in shards:
        st = p.stat()
        identity.append({
            "name": p.name,
            "size": st.st_size,
            "mtime": st.st_mtime,
        })
    return identity


class ExpertDiskCopy:
    """Manages the on-disk binary expert copies and manifest validation for an MoE model."""

    MANIFEST_FILENAME = "manifest.json"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        root: Path | str,
        model_path: Path | str,
        schema: tuple[str, ...] | list[str],
        shapes: dict[str, tuple[int, ...]],
        dtypes: dict[str, torch.dtype | str],
    ) -> None:
        self.root = Path(root)
        self.model_path = Path(model_path)
        self.schema = tuple(schema)
        self.shapes = {k: tuple(v) for k, v in shapes.items()}
        # Normalize dtypes to torch.dtype
        self.dtypes: dict[str, torch.dtype] = {}
        for k, v in dtypes.items():
            if isinstance(v, torch.dtype):
                self.dtypes[k] = v
            else:
                s = str(v).replace("torch.", "")
                self.dtypes[k] = getattr(torch, s)

        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / self.MANIFEST_FILENAME

        self._row_bytes = {
            name: math.prod(self.shapes[name][1:]) * torch.empty((), dtype=self.dtypes[name]).element_size()
            for name in self.schema
        }
        self._bank_bytes = {
            name: self.shapes[name][0] * self._row_bytes[name]
            for name in self.schema
        }

        self.current_identity = _scan_checkpoint_identity(self.model_path)
        self.manifest = self._load_or_create_manifest()

    def bank_path(self, layer_id: int, bank_name: str) -> Path:
        return self.root / f"L{layer_id}.{bank_name}.bin"

    def row_bytes(self, bank_name: str) -> int:
        return self._row_bytes[bank_name]

    def bank_bytes(self, bank_name: str) -> int:
        return self._bank_bytes[bank_name]

    def _fresh_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "checkpoint_identity": self.current_identity,
            "schema": list(self.schema),
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "dtypes": {k: str(v).replace("torch.", "") for k, v in self.dtypes.items()},
            "complete": {},
        }

    def _manifest_matches(self, doc: dict[str, Any]) -> bool:
        if doc.get("schema_version") != self.SCHEMA_VERSION:
            return False
        if doc.get("checkpoint_identity") != self.current_identity:
            return False
        if list(doc.get("schema", [])) != list(self.schema):
            return False
        doc_shapes = doc.get("shapes", {})
        for name in self.schema:
            if list(doc_shapes.get(name, [])) != list(self.shapes[name]):
                return False
        doc_dtypes = doc.get("dtypes", {})
        for name in self.schema:
            expected_dtype_str = str(self.dtypes[name]).replace("torch.", "")
            if doc_dtypes.get(name) != expected_dtype_str:
                return False
        return True

    def _load_or_create_manifest(self) -> dict[str, Any]:
        if self.manifest_path.is_file():
            try:
                doc = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if self._manifest_matches(doc):
                    return doc
                logger.warning_rank0(
                    "checkpoint identity or schema mismatch in %s; resetting disk copy manifest",
                    self.manifest_path,
                )
            except Exception as exc:
                logger.warning_rank0(
                    "failed reading %s (%s); recreating fresh manifest",
                    self.manifest_path,
                    exc,
                )
        doc = self._fresh_manifest()
        self._save_manifest(doc)
        return doc

    def _save_manifest(self, doc: dict[str, Any] | None = None) -> None:
        if doc is None:
            doc = self.manifest
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def layer_complete(self, layer_id: int) -> bool:
        """A layer may only be spilled when its files are completely written and verified."""
        if not self.manifest.get("complete", {}).get(str(layer_id), False):
            return False
        for name in self.schema:
            p = self.bank_path(layer_id, name)
            if not p.is_file():
                return False
            if p.stat().st_size != self.bank_bytes(name):
                return False
        return True

    def write_layer(self, layer_id: int, banks: dict[str, torch.Tensor]) -> None:
        """Write one MoE layer's expert banks to disk in 64-row chunks.

        Chunking to 64 rows bounds host/device memory during conversion: device tensors
        are copied in 64-row slices via torch.Tensor.cpu() rather than materializing a full
        1.33 GiB layer in host RAM at once.
        """
        for name in self.schema:
            assert name in banks, f"bank {name!r} missing from write_layer input"
            tensor = banks[name]
            tensor = getattr(tensor, "tensor", tensor)
            num_experts = tensor.size(0)
            dst_path = self.bank_path(layer_id, name)
            tmp_path = dst_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                for r in range(0, num_experts, 64):
                    chunk = tensor[r : r + 64]
                    if chunk.is_cuda:
                        chunk = chunk.cpu()
                    chunk = chunk.contiguous()
                    u = chunk.view(torch.uint8).reshape(-1)
                    f.write(u.numpy().tobytes())
            tmp_path.replace(dst_path)

        self.manifest.setdefault("complete", {})[str(layer_id)] = True
        self._save_manifest()

    def read_layer_into(self, layer_id: int, bank_name: str, dst_tensor: torch.Tensor) -> None:
        """Recall a full expert bank from disk into a pre-allocated host bank tensor."""
        path = self.bank_path(layer_id, bank_name)
        with open(path, "rb") as f:
            u = dst_tensor.view(torch.uint8).reshape(-1)
            mv = memoryview(u.numpy())
            f.readinto(mv)


class BackgroundWriter(threading.Thread):
    """Background worker that writes expert banks to disk at lowest I/O priority."""

    def __init__(self, copy: ExpertDiskCopy, cache: Any) -> None:
        super().__init__(daemon=True, name="freetoken-expert-disk-writer")
        self.copy = copy
        self.cache = cache
        self.bytes_written = 0
        self.elapsed_s = 0.0

    def run(self) -> None:
        _set_lowest_io_priority()
        incomplete = [
            l for l in range(self.cache.num_layers)
            if not self.copy.layer_complete(l)
        ]
        if not incomplete:
            logger.info_rank0(
                "ExpertDiskCopy: all %d layers already complete at %s",
                self.cache.num_layers,
                self.copy.root,
            )
            return

        total_bytes_expected = sum(
            sum(self.copy.bank_bytes(n) for n in self.copy.schema)
            for _ in incomplete
        )
        logger.info_rank0(
            "BackgroundWriter: writing %d incomplete MoE layers (%.2f GB) to %s",
            len(incomplete),
            total_bytes_expected / (1 << 30),
            self.copy.root,
        )

        t0 = time.perf_counter()
        bytes_written = 0
        for layer_id in incomplete:
            layer_banks = {
                name: self.cache.bank_sources[name][layer_id]
                for name in self.copy.schema
            }
            self.copy.write_layer(layer_id, layer_banks)
            bytes_written += sum(self.copy.bank_bytes(n) for n in self.copy.schema)

        dt = time.perf_counter() - t0
        self.bytes_written = bytes_written
        self.elapsed_s = dt
        gb = bytes_written / (1 << 30)
        mb_per_s = (bytes_written / 1e6) / max(dt, 1e-4)
        logger.info_rank0(
            "BackgroundWriter finished: wrote %.2f GB in %.1f s (%.1f MB/s)",
            gb,
            dt,
            mb_per_s,
        )


class DiskLayerReader:
    """Reads expert rows from the disk copy into a shared pinned staging buffer."""

    def __init__(
        self,
        copy: ExpertDiskCopy,
        layer_id: int,
        staging: dict[str, torch.Tensor],
        max_workers: int = 4,
    ) -> None:
        self.copy = copy
        self.layer_id = layer_id
        self.staging = staging
        self.row_bytes = {name: copy.row_bytes(name) for name in copy.schema}

        # 1D uint8 memoryviews for zero-copy os.pread slicing
        self._staging_mvs = {
            name: memoryview(staging[name].view(torch.uint8).reshape(-1).numpy())
            for name in copy.schema
        }

        self._fds = {
            name: os.open(str(copy.bank_path(layer_id, name)), os.O_RDONLY)
            for name in copy.schema
        }
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"disk-reader-L{layer_id}",
        )

    def read_rows(self, expert_ids: list[int]) -> tuple[torch.Tensor, ...]:
        """Read arbitrary expert rows into the first len(expert_ids) staging slots."""
        k = len(expert_ids)
        if k == 0:
            return tuple(self.staging[name][:0] for name in self.copy.schema)

        def _read_bank_expert(name: str, slot: int, exp_id: int) -> None:
            fd = self._fds[name]
            rbytes = self.row_bytes[name]
            mv = self._staging_mvs[name]
            mv[slot * rbytes : (slot + 1) * rbytes] = os.pread(fd, rbytes, exp_id * rbytes)

        tasks = [
            (name, slot, exp_id)
            for name in self.copy.schema
            for slot, exp_id in enumerate(expert_ids)
        ]
        if len(tasks) <= 1:
            for task in tasks:
                _read_bank_expert(*task)
        else:
            list(self._executor.map(lambda t: _read_bank_expert(*t), tasks))

        return tuple(self.staging[name][:k] for name in self.copy.schema)

    def iter_layer_chunks(self, rows: int = 64):
        """Yield full-layer chunks through the staging buffer for prefill materialization."""
        num_experts = self.copy.shapes[self.copy.schema[0]][0]
        for start_row in range(0, num_experts, rows):
            chunk_size = min(rows, num_experts - start_row)
            for name in self.copy.schema:
                fd = self._fds[name]
                rbytes = self.row_bytes[name]
                mv = self._staging_mvs[name]
                total_bytes = chunk_size * rbytes
                mv[:total_bytes] = os.pread(fd, total_bytes, start_row * rbytes)
            yield start_row, chunk_size, tuple(self.staging[name][:chunk_size] for name in self.copy.schema)

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __del__(self) -> None:
        self.close()
