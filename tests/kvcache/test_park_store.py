from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import freetoken.kvcache.park_store as park_module
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.park_store import (
    ParkEntryRejected,
    ParkStore,
    build_model_fingerprint,
    rolling_page_keys,
)
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _qsa_pool(num_pages: int = 6, kv_dtype: torch.dtype | None = None, device="cpu") -> QSAKVCache:
    return QSAKVCache(
        num_kv_heads=1,
        num_layers=2,
        head_dim=4,
        num_pages=num_pages,
        page_size=4,
        dtype=torch.bfloat16,
        kv_dtype=kv_dtype,
        device=torch.device(device),
        index_head_dim=3,
        num_index_layers=2,
        index_ratio=2,
        num_req_slots=2,
        layer_ids=(0, 1),
    )


def _state_pool(num_slots: int = 5, device="cpu") -> LinearStatePool:
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 1),
        num_key_heads=1,
        num_value_heads=1,
        key_head_dim=2,
        value_head_dim=2,
        conv_kernel_dim=3,
        output_gate="silu",
    )
    sibling = SlotStateSpec(
        name="ple_sibling",
        shape=(2, 3),
        layer_ids=(0,),
        dtype=torch.int32,
        fill_value=-1,
    )
    return LinearStatePool(
        group=group,
        num_slots=num_slots,
        dtype=torch.bfloat16,
        device=torch.device(device),
        tp_size=1,
        slot_states=(sibling,),
    )


def _fill_entry(
    kv_pool: QSAKVCache,
    state_pool: LinearStatePool,
    page_bases: torch.Tensor,
    state_slot: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    originals: list[torch.Tensor] = []
    value = 1
    for base in page_bases.tolist():
        for view in kv_pool.page_byte_views(base // 4):
            payload = torch.arange(view.numel(), dtype=torch.int64).reshape(view.shape) + value
            view.copy_(payload.to(view.dtype))
            originals.append(view.clone())
            value += view.numel() + 7
    state_originals: list[torch.Tensor] = []
    for view in state_pool.slot_byte_views(state_slot):
        payload = torch.arange(view.numel(), dtype=torch.int64).reshape(view.shape) + value
        view.copy_(payload.to(view.dtype))
        state_originals.append(view.clone())
        value += view.numel() + 11
    return originals, state_originals


def _assert_entry_equal(
    kv_pool: QSAKVCache,
    state_pool: LinearStatePool,
    page_bases: torch.Tensor,
    state_slot: int,
    kv_expected: list[torch.Tensor],
    state_expected: list[torch.Tensor],
) -> None:
    restored_kv = [
        view.clone()
        for base in page_bases.tolist()
        for view in kv_pool.page_byte_views(base // 4)
    ]
    restored_state = [view.clone() for view in state_pool.slot_byte_views(state_slot)]
    assert len(restored_kv) == len(kv_expected)
    assert len(restored_state) == len(state_expected)
    assert all(torch.equal(got, want) for got, want in zip(restored_kv, kv_expected, strict=True))
    assert all(
        torch.equal(got, want)
        for got, want in zip(restored_state, state_expected, strict=True)
    )


def _store(
    mode: str,
    tmp_path: Path,
    kv_pool: QSAKVCache,
    state_pool: LinearStatePool,
    *,
    fingerprint: str = "model-A",
    ram_budget_bytes: int = 1 << 20,
    disk_budget_bytes: int = 1 << 20,
    pinned_window_bytes: int = 4096,
) -> ParkStore:
    return ParkStore(
        mode=mode,
        page_size=4,
        kv_pool=kv_pool,
        state_pool=state_pool,
        fingerprint=fingerprint,
        min_tokens=8,
        ram_budget_bytes=ram_budget_bytes,
        ssd_dir=tmp_path,
        disk_budget_bytes=disk_budget_bytes,
        pinned_window_bytes=pinned_window_bytes,
    )


def test_model_fingerprint_ignores_runtime_pool_capacity(tmp_path: Path):
    state_pool = _state_pool()
    small = build_model_fingerprint(
        model_path=str(tmp_path),
        page_size=4,
        tp_rank=0,
        tp_size=1,
        kv_pool=_qsa_pool(6),
        state_pool=state_pool,
    )
    large = build_model_fingerprint(
        model_path=str(tmp_path),
        page_size=4,
        tp_rank=0,
        tp_size=1,
        kv_pool=_qsa_pool(8),
        state_pool=state_pool,
    )

    assert small == large


def test_model_fingerprint_tracks_referenced_shard_identity(tmp_path: Path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"AAAA")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"x": shard.name}}), encoding="utf-8"
    )
    kv_pool, state_pool = _qsa_pool(), _state_pool()

    before = build_model_fingerprint(
        model_path=str(tmp_path),
        page_size=4,
        tp_rank=0,
        tp_size=1,
        kv_pool=kv_pool,
        state_pool=state_pool,
    )
    prior = shard.stat().st_mtime_ns
    shard.write_bytes(b"BBBB")
    shard.touch()
    if shard.stat().st_mtime_ns == prior:
        import os

        os.utime(shard, ns=(prior + 1_000_000_000, prior + 1_000_000_000))
    after = build_model_fingerprint(
        model_path=str(tmp_path),
        page_size=4,
        tp_rank=0,
        tp_size=1,
        kv_pool=kv_pool,
        state_pool=state_pool,
    )

    assert before != after


def test_model_fingerprint_rejects_same_size_same_mtime_shard_replacement(
    tmp_path: Path,
):
    import os

    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"AAAA")
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    before = build_model_fingerprint(
        model_path=str(tmp_path),
        page_size=4,
        tp_rank=0,
        tp_size=1,
        kv_pool=kv_pool,
        state_pool=state_pool,
    )
    old_mtime = shard.stat().st_mtime_ns
    replacement = tmp_path / "replacement.tmp"
    replacement.write_bytes(b"BBBB")
    os.replace(replacement, shard)
    os.utime(shard, ns=(old_mtime, old_mtime))
    assert shard.stat().st_size == 4
    assert shard.stat().st_mtime_ns == old_mtime

    after = build_model_fingerprint(
        model_path=str(tmp_path),
        page_size=4,
        tp_rank=0,
        tp_size=1,
        kv_pool=kv_pool,
        state_pool=state_pool,
    )

    assert before != after


def test_model_fingerprint_tracks_ftw_index_and_referenced_shards(tmp_path: Path):
    shard = tmp_path / "freetoken-00000.ftw"
    shard.write_bytes(b"AAAA")
    index = tmp_path / "freetoken_weight.json"
    index.write_text(
        json.dumps(
            {
                "format": "freetoken_weight",
                "version": 1,
                "shards": [{"file": shard.name, "global_off": 0, "nbytes": 4}],
                "tensors": [],
            }
        ),
        encoding="utf-8",
    )
    kv_pool, state_pool = _qsa_pool(), _state_pool()

    before = build_model_fingerprint(
        model_path=str(tmp_path),
        page_size=4,
        tp_rank=0,
        tp_size=1,
        kv_pool=kv_pool,
        state_pool=state_pool,
    )
    shard.write_bytes(b"BBBB")
    after_shard = build_model_fingerprint(
        model_path=str(tmp_path),
        page_size=4,
        tp_rank=0,
        tp_size=1,
        kv_pool=kv_pool,
        state_pool=state_pool,
    )
    index.write_text(index.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after_index = build_model_fingerprint(
        model_path=str(tmp_path),
        page_size=4,
        tp_rank=0,
        tp_size=1,
        kv_pool=kv_pool,
        state_pool=state_pool,
    )

    assert before != after_shard
    assert after_shard != after_index


def test_from_config_uses_pre_resolved_checkpoint_without_hub_lookup(
    tmp_path: Path, monkeypatch
):
    from freetoken.distributed.info import DistributedInfo

    resolved = tmp_path / "snapshot"
    resolved.mkdir()
    (resolved / "model.safetensors").write_bytes(b"checkpoint")

    def unexpected_resolve(_model_path):
        raise AssertionError("ParkStore must reuse the snapshot pinned before Engine loading")

    monkeypatch.setattr("freetoken.utils.hf.download_hf_snapshot", unexpected_resolve)
    monkeypatch.setattr("freetoken.utils.hf.download_hf_weight", unexpected_resolve)
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    config = SimpleNamespace(
        kv_park="ram",
        kv_park_ssd_dir=str(tmp_path / "parks"),
        kv_park_min_tokens=8,
        kv_park_idle_ms=0,
        kv_park_ram_gib=1,
        kv_park_ssd_gib=1,
        kv_park_window_mib=1,
        model_path=str(resolved),
        page_size=4,
        tp_info=DistributedInfo(rank=0, size=1),
    )

    store = ParkStore.from_config(config, kv_pool, state_pool)
    try:
        expected = build_model_fingerprint(
            model_path=str(resolved),
            page_size=4,
            tp_rank=0,
            tp_size=1,
            kv_pool=kv_pool,
            state_pool=state_pool,
        )
        assert store.fingerprint == expected
    finally:
        store.close()


def test_ssd_store_is_scoped_by_tensor_parallel_rank(tmp_path: Path):
    from freetoken.distributed.info import DistributedInfo

    kv_pool, state_pool = _qsa_pool(), _state_pool()
    base = tmp_path / "parks"

    def config(rank):
        return SimpleNamespace(
            kv_park="ssd",
            kv_park_ssd_dir=str(base),
            kv_park_min_tokens=8,
            kv_park_idle_ms=0,
            kv_park_ram_gib=1,
            kv_park_ssd_gib=1,
            kv_park_window_mib=1,
            model_path=str(tmp_path),
            page_size=4,
            tp_info=DistributedInfo(rank=rank, size=2),
        )

    rank0 = ParkStore.from_config(config(0), kv_pool, state_pool)
    rank1 = ParkStore.from_config(config(1), kv_pool, state_pool)
    try:
        assert rank0.ssd_dir == base / "tp-0000-of-0002"
        assert rank1.ssd_dir == base / "tp-0001-of-0002"
        assert rank0.fingerprint != rank1.fingerprint
    finally:
        rank0.close()
        rank1.close()


def test_rolling_keys_are_prefix_chained_and_model_scoped():
    tokens = torch.arange(12, dtype=torch.int32)
    keys = rolling_page_keys(tokens, page_size=4, fingerprint="model-A")
    assert len(keys) == 3
    assert keys[:2] == rolling_page_keys(tokens[:8], page_size=4, fingerprint="model-A")
    assert keys != rolling_page_keys(tokens, page_size=4, fingerprint="model-B")
    changed = tokens.clone()
    changed[1] += 100
    assert keys[0] != rolling_page_keys(changed, page_size=4, fingerprint="model-A")[0]


def test_active_store_owns_one_bounded_worker_and_close_stops_it(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    before = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("kv-park-")
    }

    store = _store("ram", tmp_path, kv_pool, state_pool)

    worker = store._worker
    assert worker.name == "kv-park-ram"
    assert worker.is_alive()
    assert store._save_queue.maxsize == 2
    current = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("kv-park-")
    }
    assert current - before == {worker.ident}

    store.close()
    assert not worker.is_alive()


@pytest.mark.parametrize("mode", ["ram", "ssd"])
def test_background_save_runs_in_the_callers_inference_mode(tmp_path: Path, mode: str):
    # The serving process builds the store inside torch.inference_mode() (launch.py:79), and
    # inference mode is THREAD-LOCAL: the ssd windows are then inference tensors and a plain
    # worker thread's first in-place write into one raised "Inplace update to inference tensor
    # outside InferenceMode is not allowed" on 2026-09-03, so ssd parking disabled itself on its
    # first save and never wrote a payload.
    with torch.inference_mode():
        kv_pool, state_pool = _qsa_pool(), _state_pool()
        slot = state_pool.alloc(1)[0]
        pages = torch.tensor([0, 4], dtype=torch.int32)
        _fill_entry(kv_pool, state_pool, pages, slot)
        store = _store(mode, tmp_path, kv_pool, state_pool)
        if mode == "ssd":
            assert store._windows is not None
            assert store._windows[0].is_inference()

        pending = store.offer(torch.arange(8, dtype=torch.int32), pages, slot)
        try:
            assert pending is not None
            assert pending.done.wait(timeout=10)
            assert pending.error is None
            assert pending.success
            status = store.status()
            assert status["disabled"] is False
            assert status["last_error"] is None
            assert status["parked_count"] == 1
        finally:
            store.close()


def test_offer_does_not_wait_for_an_inflight_device_copy(tmp_path: Path, monkeypatch):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    slots = state_pool.alloc(2)
    store = _store("ram", tmp_path, kv_pool, state_pool)
    started = threading.Event()
    release = threading.Event()
    real_copy = store._copy_to_ram

    def blocked_copy(views):
        started.set()
        release.wait()
        return real_copy(views)

    monkeypatch.setattr(store, "_copy_to_ram", blocked_copy)
    first = store.offer(
        torch.arange(8, dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        slots[0],
    )
    assert first is not None and started.wait(timeout=1)

    result = []
    caller = threading.Thread(
        target=lambda: result.append(
            store.offer(
                torch.arange(100, 108, dtype=torch.int32),
                torch.tensor([8, 12], dtype=torch.int32),
                slots[1],
            )
        )
    )
    caller.start()
    caller.join(timeout=0.1)
    try:
        assert not caller.is_alive(), "offer blocked behind the worker's device copy"
    finally:
        release.set()
        caller.join(timeout=1)
        store.close()


def test_async_ram_offers_respect_the_configured_byte_budget(
    tmp_path: Path, monkeypatch
):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slots = state_pool.alloc(2)
    probe = _store("ram", tmp_path, kv_pool, state_pool)
    one_entry = probe.storage_bytes(8)
    probe.close()
    store = _store(
        "ram",
        tmp_path,
        kv_pool,
        state_pool,
        ram_budget_bytes=one_entry + 64,
    )
    started = threading.Event()
    release = threading.Event()
    real_copy = store._copy_to_ram

    def blocked_copy(views):
        started.set()
        release.wait()
        return real_copy(views)

    monkeypatch.setattr(store, "_copy_to_ram", blocked_copy)
    first = store.offer(torch.arange(8, dtype=torch.int32), pages, slots[0])
    assert first is not None and started.wait(timeout=1)
    try:
        assert (
            store.offer(
                torch.arange(100, 108, dtype=torch.int32), pages, slots[1]
            )
            is None
        )
    finally:
        release.set()
        first.wait()
        store.close()


def test_sync_ram_save_evicts_before_allocating_its_replacement(
    tmp_path: Path, monkeypatch
):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    probe = _store("ram", tmp_path, kv_pool, state_pool)
    one_entry = probe.storage_bytes(8)
    probe.close()
    store = _store(
        "ram",
        tmp_path,
        kv_pool,
        state_pool,
        ram_budget_bytes=one_entry + 64,
    )
    assert store.save(torch.arange(8, dtype=torch.int32), pages, slot)
    real_copy = store._copy_to_ram

    def checked_copy(views):
        assert store.status()["parked_count"] == 0
        return real_copy(views)

    monkeypatch.setattr(store, "_copy_to_ram", checked_copy)
    assert store.save(torch.arange(100, 108, dtype=torch.int32), pages, slot)
    store.close()


def test_ssd_source_copy_completes_before_manifest_update(
    tmp_path: Path, monkeypatch
):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    slot = state_pool.alloc(1)[0]
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    manifest_started = threading.Event()
    release = threading.Event()
    real_manifest = store._write_manifest

    def blocked_manifest():
        manifest_started.set()
        release.wait()
        real_manifest()

    monkeypatch.setattr(store, "_write_manifest", blocked_manifest)
    pending = store.offer(
        torch.arange(8, dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        slot,
    )
    assert pending is not None and manifest_started.wait(timeout=1)
    try:
        assert pending.copy_done.is_set()
    finally:
        release.set()
        pending.wait()
        store.close()


def test_payload_size_uses_independent_pool_accounting(tmp_path: Path, monkeypatch):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    store = _store("ram", tmp_path, kv_pool, state_pool)
    expected = 2 * kv_pool.unit_bytes()[0] * 4 + state_pool.bytes_per_slot()
    monkeypatch.setattr(
        kv_pool,
        "page_byte_views",
        lambda _page: (_ for _ in ()).throw(AssertionError("read byte views")),
    )

    assert store.payload_bytes(8) == expected
    store.close()


def test_ram_entry_uses_one_contiguous_host_buffer(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    _fill_entry(kv_pool, state_pool, pages, slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ram", tmp_path, kv_pool, state_pool)

    assert store.save(tokens, pages, slot)
    entry = store.lookup(tokens)

    assert entry is not None
    assert entry.ram_buffer is not None
    assert entry.ram_buffer.dtype is torch.uint8
    assert entry.ram_buffer.numel() == store.payload_bytes(len(tokens))
    store.close()


def test_ram_round_trip_preserves_qsa_and_all_linear_state_bytes(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    source_pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    kv_expected, state_expected = _fill_entry(kv_pool, state_pool, source_pages, source_slot)
    store = _store("ram", tmp_path, kv_pool, state_pool)
    tokens = torch.arange(8, dtype=torch.int32)

    assert store.save(tokens, source_pages, source_slot)
    for view in kv_pool.page_byte_views(2):
        view.fill_(99)
    for view in kv_pool.page_byte_views(3):
        view.fill_(99)
    for view in state_pool.slot_byte_views(target_slot):
        view.fill_(99)

    hit = store.lookup(torch.cat([tokens, torch.tensor([99], dtype=torch.int32)]))
    assert hit is not None and hit.token_count == 8
    store.restore(hit, torch.tensor([8, 12], dtype=torch.int32), target_slot)
    _assert_entry_equal(
        kv_pool,
        state_pool,
        torch.tensor([8, 12], dtype=torch.int32),
        target_slot,
        kv_expected,
        state_expected,
    )
    assert store.status()["hits"] == 1


def test_ssd_round_trip_rebuilds_index_without_manifest(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    source_pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    kv_expected, state_expected = _fill_entry(kv_pool, state_pool, source_pages, source_slot)
    tokens = torch.arange(8, dtype=torch.int32)

    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, source_pages, source_slot)
    assert (tmp_path / "park.json").is_file()
    store.close()
    (tmp_path / "park.json").unlink()

    rebuilt = _store("ssd", tmp_path, kv_pool, state_pool)
    hit = rebuilt.lookup(torch.cat([tokens, torch.tensor([88], dtype=torch.int32)]))
    assert hit is not None
    rebuilt.restore(hit, torch.tensor([8, 12], dtype=torch.int32), target_slot)
    _assert_entry_equal(
        kv_pool,
        state_pool,
        torch.tensor([8, 12], dtype=torch.int32),
        target_slot,
        kv_expected,
        state_expected,
    )
    header = next(tmp_path.glob("*.park")).read_bytes()[:16]
    assert header.startswith(b"FTKVPARK")
    assert (
        json.loads((tmp_path / "park.json").read_text(encoding="utf-8"))["version"]
        == park_module._VERSION
    )


def test_payload_digest_does_not_depend_on_the_pinned_window_size(tmp_path: Path):
    # The digest is sliced on a fixed 32 MiB grid rather than on the pinned window, so a
    # store booted with a different --kv-park-window-mib still verifies an older file.
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    source_pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    kv_expected, state_expected = _fill_entry(
        kv_pool, state_pool, source_pages, source_slot
    )
    tokens = torch.arange(8, dtype=torch.int32)

    writer = _store("ssd", tmp_path, kv_pool, state_pool, pinned_window_bytes=4096)
    assert writer.save(tokens, source_pages, source_slot)
    writer.close()

    reader = _store("ssd", tmp_path, kv_pool, state_pool, pinned_window_bytes=3 * 4096)
    hit = reader.lookup(tokens)
    assert hit is not None
    reader.restore(hit, torch.tensor([8, 12], dtype=torch.int32), target_slot)
    _assert_entry_equal(
        kv_pool,
        state_pool,
        torch.tensor([8, 12], dtype=torch.int32),
        target_slot,
        kv_expected,
        state_expected,
    )
    reader.close()


def test_an_older_header_version_is_rejected_and_the_file_removed(tmp_path: Path):
    # Version 2 wrote a single sequential SHA-256 over the payload; version 3 writes the
    # chunked digest. An old file must become a miss, not a checksum failure mid-restore.
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    _fill_entry(kv_pool, state_pool, pages, slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, pages, slot)
    store.close()

    path = next(tmp_path.glob("*.park"))
    raw = bytearray(path.read_bytes())
    raw[len(park_module._MAGIC) : len(park_module._MAGIC) + 4] = (2).to_bytes(4, "little")
    path.write_bytes(bytes(raw))
    (tmp_path / "park.json").unlink()

    rebuilt = _store("ssd", tmp_path, kv_pool, state_pool)
    assert rebuilt.lookup(tokens) is None
    assert rebuilt.status()["parked_count"] == 0
    assert not path.exists()
    rebuilt.close()


def test_ssd_reader_honors_unbuffered_gate_and_falls_back(
    tmp_path: Path, monkeypatch
):
    import freetoken.kvcache.park_store as park_module
    import freetoken.moe.win_io as win_io

    kv_pool, state_pool = _qsa_pool(), _state_pool()
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    path = tmp_path / "probe.park"
    path.write_bytes(b"")
    calls = []
    monkeypatch.setattr(park_module.sys, "platform", "win32")
    monkeypatch.setattr(win_io, "enabled", lambda: False)
    monkeypatch.setattr(
        win_io,
        "UnbufferedReader",
        lambda *_args, **_kwargs: calls.append("open"),
    )
    assert store._unbuffered_reader(path) is None
    assert calls == []

    monkeypatch.setattr(win_io, "enabled", lambda: True)

    def unsupported(*_args, **_kwargs):
        calls.append("open")
        raise OSError("volume refuses no-buffering")

    monkeypatch.setattr(win_io, "UnbufferedReader", unsupported)
    assert store._unbuffered_reader(path) is None
    assert calls == ["open"]
    store.close()


def test_ssd_manifest_uses_restart_stable_wall_clock_lru(tmp_path: Path):
    import time

    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    _fill_entry(kv_pool, state_pool, pages, slot)
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    before = time.time_ns()
    assert store.save(torch.arange(8, dtype=torch.int32), pages, slot)
    after = time.time_ns()
    store.close()

    manifest = json.loads((tmp_path / "park.json").read_text(encoding="utf-8"))
    last_used = manifest["entries"][0]["last_used_ns"]
    assert before <= last_used <= after


def test_ssd_manifest_cannot_delete_a_file_outside_the_store(tmp_path: Path):
    root = tmp_path / "parks"
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me", encoding="utf-8")
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    first = _store("ssd", root, kv_pool, state_pool)
    first.close()
    (root / "park.json").write_text(
        json.dumps(
            {
                "version": 2,
                "fingerprint": "model-A",
                "entries": [
                    {"file": "../outside.txt", "last_used_ns": 1}
                ],
            }
        ),
        encoding="utf-8",
    )

    reopened = _store("ssd", root, kv_pool, state_pool)
    try:
        assert outside.read_text(encoding="utf-8") == "keep me"
    finally:
        reopened.close()


def test_ssd_startup_removes_only_dead_writer_temp_files(tmp_path: Path, monkeypatch):
    dead = tmp_path / f".{'a' * 32}.123.tmp"
    live = tmp_path / f".{'b' * 32}.456.tmp"
    unrelated = tmp_path / ".not-a-park.789.tmp"
    dead.write_bytes(b"incomplete")
    live.write_bytes(b"in progress")
    unrelated.write_bytes(b"unrelated")
    monkeypatch.setattr(
        "freetoken.kvcache.park_store._pid_is_alive",
        lambda pid: pid == 456,
    )

    store = _store("ssd", tmp_path, _qsa_pool(), _state_pool())
    try:
        assert not dead.exists()
        assert live.read_bytes() == b"in progress"
        assert unrelated.read_bytes() == b"unrelated"
    finally:
        store.close()


def _record_fake_cuda_restore_io(monkeypatch, store: ParkStore, entry):
    assert entry.path is not None and store._windows is not None
    file_bytes = entry.path.read_bytes()
    reads: list[tuple[int, int]] = []
    window_indices: list[int] = []
    window_ptrs = [int(window.data_ptr()) for window in store._windows]

    class FakeReader:
        def read_into(self, buffer, offset, length):
            ptr = int(torch.frombuffer(buffer, dtype=torch.uint8).data_ptr())
            window_indices.append(window_ptrs.index(ptr))
            chunk = file_bytes[offset : offset + length]
            buffer[: len(chunk)] = chunk
            reads.append((offset, len(chunk)))
            return len(chunk)

        def close(self):
            pass

    class FakeEvent:
        def __init__(self, **_kwargs):
            pass

        def record(self, _stream):
            pass

        def synchronize(self):
            pass

    class FakeStream:
        def synchronize(self):
            pass

    store.pinned_window_bytes = 64
    store._stream = FakeStream()
    monkeypatch.setattr(store, "_unbuffered_reader", lambda _path: FakeReader())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: contextlib.nullcontext())
    return reads, window_indices


def test_ssd_restore_reads_each_payload_block_once_while_checking_integrity(
    tmp_path: Path, monkeypatch
):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    source_pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    kv_expected, state_expected = _fill_entry(kv_pool, state_pool, source_pages, source_slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, source_pages, source_slot)
    entry = store.lookup(tokens)
    assert entry is not None and entry.path is not None
    meta = store._parse_header(entry.path)
    reads, _window_indices = _record_fake_cuda_restore_io(monkeypatch, store, entry)

    try:
        store.restore(
            entry,
            torch.tensor([8], dtype=torch.int32),
            target_slot,
            page_offset=1,
        )
        # v4 keeps the KV pages and the state slot in two separately checksummed regions;
        # each is read exactly once in window-sized blocks, the skipped first page included.
        expected_reads = [
            (int(meta[region + "_offset"]) + offset, min(64, int(meta[region + "_bytes"]) - offset))
            for region in ("kv", "state")
            for offset in range(0, int(meta[region + "_bytes"]), 64)
        ]
        assert reads == expected_reads
        views_per_page = len(kv_pool.page_byte_views(0))
        _assert_entry_equal(
            kv_pool,
            state_pool,
            torch.tensor([8], dtype=torch.int32),
            target_slot,
            kv_expected[views_per_page:],
            state_expected,
        )
    finally:
        store.close()


def test_ssd_cuda_restore_alternates_both_pinned_windows(tmp_path: Path, monkeypatch):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    source_pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    _fill_entry(kv_pool, state_pool, source_pages, source_slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, source_pages, source_slot)
    entry = store.lookup(tokens)
    assert entry is not None
    _reads, window_indices = _record_fake_cuda_restore_io(monkeypatch, store, entry)

    try:
        store.restore(entry, torch.tensor([8, 12], dtype=torch.int32), target_slot)
        assert len(window_indices) >= 4
        assert window_indices == [index % 2 for index in range(len(window_indices))]
    finally:
        store.close()


def test_ssd_payload_checksum_rejects_corruption_during_restore(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    source_pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    _fill_entry(kv_pool, state_pool, source_pages, source_slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, source_pages, source_slot)
    entry = store.lookup(tokens)
    assert entry is not None and entry.path is not None
    meta = store._parse_header(entry.path)
    with entry.path.open("r+b") as handle:
        handle.seek(int(meta["kv_offset"]) + 3)
        original = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(ParkEntryRejected, match="checksum"):
        store.restore(entry, torch.tensor([8, 12], dtype=torch.int32), target_slot)
    assert store.lookup(tokens) is None
    store.close()


def test_ssd_pinned_window_failure_disables_store_without_failing_boot(
    tmp_path: Path, monkeypatch
):
    kv_pool, state_pool = _qsa_pool(), _state_pool()

    def fail_window(_self, _nbytes):
        raise RuntimeError("pin quota exhausted")

    monkeypatch.setattr(ParkStore, "_allocate_window", fail_window)
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.status()["disabled"] is True
    assert "pin quota exhausted" in str(store.status()["last_error"])
    assert store.lookup(torch.arange(8, dtype=torch.int32)) is None
    store.close()


def test_save_failure_waits_for_private_copy_stream_before_releasing_sources(
    tmp_path: Path, monkeypatch
):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    store = _store("ram", tmp_path, kv_pool, state_pool)

    class FakeStream:
        def __init__(self):
            self.synchronize_calls = 0

        def synchronize(self):
            self.synchronize_calls += 1

    stream = FakeStream()
    store._stream = stream
    monkeypatch.setattr(
        store,
        "_copy_to_ram",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("mid-copy failed")),
    )
    released_after_sync = []

    assert not store.save(
        torch.arange(8, dtype=torch.int32),
        pages,
        slot,
        _on_copied=lambda: released_after_sync.append(stream.synchronize_calls),
    )
    assert released_after_sync == [1]
    assert "mid-copy failed" in str(store.status()["last_error"])
    store.close()


def test_restore_failure_waits_for_private_copy_stream_before_releasing_targets(
    tmp_path: Path, monkeypatch
):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    _fill_entry(kv_pool, state_pool, pages, source_slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, pages, source_slot)
    entry = store.lookup(tokens)
    assert entry is not None

    class FakeStream:
        def __init__(self):
            self.synchronize_calls = 0

        def synchronize(self):
            self.synchronize_calls += 1

    stream = FakeStream()
    store._stream = stream
    monkeypatch.setattr(
        store,
        "_read_payload_cuda",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("mid-copy read failed")),
    )

    with pytest.raises(OSError, match="mid-copy"):
        store.restore(entry, torch.tensor([8, 12], dtype=torch.int32), target_slot)
    assert stream.synchronize_calls == 1


def test_fingerprint_mismatch_is_rejected_on_startup(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    _fill_entry(kv_pool, state_pool, pages, slot)
    tokens = torch.arange(8, dtype=torch.int32)
    first = _store("ssd", tmp_path, kv_pool, state_pool, fingerprint="model-A")
    assert first.save(tokens, pages, slot)
    first.close()

    wrong_model = _store("ssd", tmp_path, kv_pool, state_pool, fingerprint="model-B")
    assert wrong_model.lookup(tokens) is None
    assert wrong_model.status()["parked_count"] == 0


def test_token_verification_turns_a_hash_collision_into_a_miss(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    _fill_entry(kv_pool, state_pool, pages, slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ram", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, pages, slot)
    key = rolling_page_keys(tokens, 4, "model-A")[-1]
    store._entries[key].token_ids[0] = -123

    assert store.lookup(tokens) is None
    status = store.status()
    assert status["misses"] == 1
    assert status["parked_count"] == 0


def test_ram_and_ssd_budgets_evict_the_oldest_entry(tmp_path: Path):
    for mode in ("ram", "ssd"):
        root = tmp_path / mode
        kv_pool, state_pool = _qsa_pool(), _state_pool()
        pages = torch.tensor([0, 4], dtype=torch.int32)
        slot = state_pool.alloc(1)[0]
        _fill_entry(kv_pool, state_pool, pages, slot)
        probe = _store(mode, root, kv_pool, state_pool)
        one_entry = probe.storage_bytes(8)
        probe.close()
        store = _store(
            mode,
            root,
            kv_pool,
            state_pool,
            ram_budget_bytes=one_entry + 64,
            disk_budget_bytes=one_entry + 64,
        )
        first = torch.arange(8, dtype=torch.int32)
        second = torch.arange(100, 108, dtype=torch.int32)
        assert store.save(first, pages, slot)
        assert store.save(second, pages, slot)
        assert store.lookup(first) is None
        assert store.lookup(second) is not None
        assert store.status()["parked_count"] == 1


def test_lookup_uses_precomputed_page_keys(tmp_path: Path, monkeypatch):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ram", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, pages, slot)
    keys = rolling_page_keys(tokens, 4, "model-A")

    monkeypatch.setattr(
        park_module,
        "rolling_page_keys",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("lookup recomputed parked page keys")
        ),
    )

    assert store.lookup(tokens, keys=keys) is not None
    store.close()


def test_ssd_writer_releases_page_cache(tmp_path: Path, monkeypatch):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    source_pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    _fill_entry(kv_pool, state_pool, source_pages, source_slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ssd", tmp_path, kv_pool, state_pool, pinned_window_bytes=4096)
    calls = []
    events = []

    def spy_fsync(fd):
        events.append(("fsync", fd))
        return real_fsync(fd)

    def spy_fadvise(fd, offset, length, advice):
        stat = park_module.os.fstat(fd)
        calls.append((fd, offset, length, advice, stat.st_size))
        events.append(("fadvise", fd))

    real_fsync = park_module.os.fsync
    monkeypatch.setattr(park_module.os, "fsync", spy_fsync)
    monkeypatch.setattr(park_module.os, "posix_fadvise", spy_fadvise, raising=False)
    try:
        assert store.save(tokens, source_pages, source_slot)
        save_calls = list(calls)
        assert save_calls
        # One release per written window of each region: the KV pages and the state slot are
        # separate 4 KiB-aligned regions, so each starts its own window sequence.
        kv_bytes = len(tokens) * kv_pool.unit_bytes()[0]
        expected_windows = (kv_bytes + 4095) // 4096 + (state_pool.bytes_per_slot() + 4095) // 4096
        assert len(save_calls) == expected_windows
        assert all(
            offset == 0
            and length == 0
            and advice == park_module.os.POSIX_FADV_DONTNEED
            and fd >= 0
            and file_size > 0
            for fd, offset, length, advice, file_size in save_calls
        )
        assert any(
            events[index][0] == "fsync" and events[index + 1][0] == "fadvise"
            for index in range(len(events) - 1)
        )

        entry = store.lookup(tokens)
        assert entry is not None
        store.restore(entry, torch.tensor([8, 12], dtype=torch.int32), target_slot)
        restore_calls = calls[len(save_calls) :]
        assert restore_calls
        assert all(call[0] >= 0 and call[4] > 0 for call in restore_calls)

        monkeypatch.delattr(park_module.os, "posix_fadvise", raising=False)
        store.restore(entry, torch.tensor([8, 12], dtype=torch.int32), target_slot)
    finally:
        store.close()


# ---- incremental (v4) parking ------------------------------------------------------------


def _page_views(kv_pool: QSAKVCache, page_ids) -> list[torch.Tensor]:
    return [view for page_id in page_ids for view in kv_pool.page_byte_views(int(page_id))]


def _fill_raw(views, seed: int) -> None:
    """Deterministic raw-byte fill through each view's uint8 alias (works for fp8 and int32)."""
    for index, view in enumerate(views):
        raw = view.view(torch.uint8).reshape(-1)
        pattern = (torch.arange(raw.numel(), dtype=torch.int64) * 37 + seed * 101 + index * 13) % 251
        raw.copy_(pattern.to(torch.uint8))


def _raw_bytes(views) -> list[torch.Tensor]:
    return [view.view(torch.uint8).reshape(-1).clone() for view in views]


def _assert_raw_equal(got: list[torch.Tensor], want: list[torch.Tensor]) -> None:
    assert len(got) == len(want)
    for index, (a, b) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(a, b), f"raw bytes differ in view {index}"


def _bases(page_ids) -> torch.Tensor:
    return torch.tensor([int(page_id) * 4 for page_id in page_ids], dtype=torch.int32)


def _entry_for(store: ParkStore, tokens: torch.Tensor):
    return store._entries.get(rolling_page_keys(tokens, 4, store.fingerprint)[-1])


@pytest.mark.parametrize("page_offset", [0, 1, 3, 5])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_ram_two_families_share_pages_and_restore_exact_checkpoint(tmp_path, page_offset, kv_dtype):
    """A budget for two roots plus deltas cannot hold repeated full snapshots.

    Separate byte oracles cover QSA, its index, GDN and PLE sibling state after
    every source page/slot has been overwritten by a different conversation.
    """
    kv, state = _qsa_pool(num_pages=32, kv_dtype=kv_dtype), _state_pool(num_slots=8)
    slots = state.alloc(3)
    per_token = kv.unit_bytes()[0]
    state_bytes = sum(v.numel() * v.element_size() for v in state.slot_byte_views(slots[0]))
    budget = 2 * (20 * per_token + 3 * state_bytes + (12 + 8 + 20) * 4)
    store = _store("ram", tmp_path / "never-created", kv, state, ram_budget_bytes=budget)
    snapshots = []
    try:
        for family in range(2):
            tokens = torch.arange(20, dtype=torch.int32) + family * 100
            pages = [2, 4, 6, 8, 10]
            _fill_raw(_page_views(kv, pages), seed=family + 17)
            # Longer donor -> shorter prompt checkpoint -> extended next turn.
            parent = None
            for length, seed in [(12, 1), (8, 2), (20, 3)]:
                _fill_raw(state.slot_byte_views(slots[0]), seed=seed + family * 7)
                expected_kv = _raw_bytes(_page_views(kv, pages[:length // 4]))
                expected_state = _raw_bytes(state.slot_byte_views(slots[0]))
                assert store.save(tokens[:length], _bases(pages[:length // 4]), slots[0])
                entry = _entry_for(store, tokens[:length])
                if length == 8:
                    assert entry.parent_key == parent.key and entry.kv_bytes == 0
                if length == 20:
                    assert entry.parent_token_count == 12
                    assert entry.ram_buffer.numel() == 8 * per_token + state_bytes
                snapshots.append((entry, expected_kv, expected_state))
                parent = entry
        assert store.status()["parked_count"] == 6
        assert store.status()["parked_bytes"] <= budget
        assert not (tmp_path / "never-created").exists()
        _fill_raw(_page_views(kv, list(range(32))), seed=97)
        _fill_raw(state.slot_byte_views(slots[0]), seed=98)
        for entry, expected_kv, expected_state in snapshots * 2:
            offset = min(page_offset, entry.token_count // 4)
            targets = [17, 19, 21, 23, 25][offset:entry.token_count // 4]
            store.restore(entry, _bases(targets), slots[1], page_offset=offset)
            per_page = len(kv.page_byte_views(0))
            _assert_raw_equal(_raw_bytes(_page_views(kv, targets)), expected_kv[offset * per_page:])
            _assert_raw_equal(_raw_bytes(state.slot_byte_views(slots[1])), expected_state)
        assert store._reserved_bytes == 0 and not store._pins
    finally:
        store.close()


def test_ram_changed_prefix_bytes_cannot_borrow_a_stale_parent(tmp_path):
    kv, state = _qsa_pool(num_pages=12), _state_pool()
    slots = state.alloc(2)
    store = _store("ram", tmp_path, kv, state)
    try:
        tokens = torch.arange(12, dtype=torch.int32)
        pages = [1, 3, 5]
        _fill_raw(_page_views(kv, pages), seed=21)
        _fill_raw(state.slot_byte_views(slots[0]), seed=22)
        assert store.save(tokens[:8], _bases(pages[:2]), slots[0])
        kv.page_byte_views(1)[0].view(torch.uint8).reshape(-1)[0] ^= 255
        expected = _raw_bytes(_page_views(kv, pages))
        expected_state = _raw_bytes(state.slot_byte_views(slots[0]))
        assert store.save(tokens, _bases(pages), slots[0])
        entry = _entry_for(store, tokens)
        assert entry.parent_key is None
        assert entry.kv_bytes == 12 * kv.unit_bytes()[0]
        store.restore(entry, _bases([6, 7, 8]), slots[1])
        _assert_raw_equal(_raw_bytes(_page_views(kv, [6, 7, 8])), expected)
        _assert_raw_equal(_raw_bytes(state.slot_byte_views(slots[1])), expected_state)
    finally:
        store.close()


def test_ram_root_fallback_releases_evicted_payload_before_allocating(tmp_path, monkeypatch):
    import weakref

    kv, state = _qsa_pool(num_pages=12), _state_pool()
    slot = state.alloc(1)[0]
    store = _store("ram", tmp_path, kv, state)
    try:
        tokens = torch.arange(12, dtype=torch.int32)
        _fill_raw(_page_views(kv, [0, 1, 2]), seed=1)
        _fill_raw(state.slot_byte_views(slot), seed=2)
        assert store.save(tokens[:8], _bases([0, 1]), slot)
        old = _entry_for(store, tokens[:8])
        old_buffer = weakref.ref(old.ram_buffer)
        # Room for a root and delta, or one replacement root, but not both roots.
        store.ram_budget_bytes = old.total_bytes + store._planned_bytes(12, 8)
        kv.page_byte_views(0)[0].view(torch.uint8).reshape(-1)[0] ^= 255
        real_copy = store._copy_to_ram

        def checked_copy(span):
            assert old_buffer() is None, "evicted payload remains pinned during root allocation"
            return real_copy(span)

        monkeypatch.setattr(store, "_copy_to_ram", checked_copy)
        assert store.save(tokens, _bases([0, 1, 2]), slot)
        assert store.status()["disabled"] is False
    finally:
        store.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA stream/event proof")
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_ram_async_cuda_copy_owns_sources_until_event_then_restores(tmp_path, kv_dtype):
    producer = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(producer):
        kv = _qsa_pool(num_pages=16, kv_dtype=kv_dtype, device="cuda:0")
        state = _state_pool(device="cuda:0")
        source_slot, target_slot = state.alloc(2)
        with torch.cuda.stream(torch.cuda.default_stream()):
            store = _store("ram", tmp_path, kv, state)
        try:
            tokens = torch.arange(12, dtype=torch.int32)
            _fill_raw(_page_views(kv, [0, 2, 4]), seed=31)
            _fill_raw(state.slot_byte_views(source_slot), seed=32)
            assert store.save(tokens[:8], _bases([0, 2]), source_slot)
            _fill_raw(state.slot_byte_views(source_slot), seed=33)
            expected_kv = [v.cpu() for v in _raw_bytes(_page_views(kv, [0, 2, 4]))]
            expected_state = [v.cpu() for v in _raw_bytes(state.slot_byte_views(source_slot))]
            # Queue unfinished GPU work AFTER capturing the oracle. The worker must
            # wait for this producer before reading the new page and recurrent state.
            page = kv.page_byte_views(4)[0].view(torch.uint8).reshape(-1)
            recurrent = state.slot_byte_views(source_slot)[0].view(torch.uint8).reshape(-1)
            saved_page, saved_recurrent = page.clone(), recurrent.clone()
            page.zero_()
            recurrent.zero_()
            torch.cuda._sleep(20_000_000)
            page.copy_(saved_page, non_blocking=True)
            recurrent.copy_(saved_recurrent, non_blocking=True)
            pending = store.offer(tokens, _bases([0, 2, 4]), source_slot)
            assert pending is not None and pending.copy_done.wait(30)
            # This is when the scheduler may give these pages and slot to conversation B.
            _fill_raw(_page_views(kv, [0, 2, 4]), seed=61)
            _fill_raw(state.slot_byte_views(source_slot), seed=62)
            assert pending.done.wait(30) and pending.success
            hit = store.lookup(tokens)
            assert hit.parent_key is not None
            store.restore(hit, _bases([7, 9, 11]), target_slot)
            got_kv = [v.cpu() for v in _raw_bytes(_page_views(kv, [7, 9, 11]))]
            got_state = [v.cpu() for v in _raw_bytes(state.slot_byte_views(target_slot))]
            _assert_raw_equal(got_kv, expected_kv)
            _assert_raw_equal(got_state, expected_state)
            transferred = store.status()["last_restore_breakdown_ms"]
            assert transferred["kv_bytes"] == 12 * kv.unit_bytes()[0]
            assert transferred["state_bytes"] == sum(v.numel() for v in expected_state)
            assert transferred["page_offset"] == 0 and transferred["sequence"] == 1
            assert store._reserved_bytes == 0 and not store._pins
        finally:
            store.close()


def test_ram_invalidated_inflight_parent_remains_charged_until_unpinned(tmp_path, monkeypatch):
    import weakref

    kv, state = _qsa_pool(num_pages=12), _state_pool()
    slot = state.alloc(1)[0]
    store = _store("ram", tmp_path, kv, state)
    started, release = threading.Event(), threading.Event()
    try:
        tokens = torch.arange(12, dtype=torch.int32)
        _fill_raw(_page_views(kv, [0, 1, 2]), seed=11)
        _fill_raw(state.slot_byte_views(slot), seed=12)
        assert store.save(tokens[:8], _bases([0, 1]), slot)
        parent = _entry_for(store, tokens[:8])
        buffer = weakref.ref(parent.ram_buffer)
        write = store._save_op

        def paused(*args, **kwargs):
            started.set()
            assert release.wait(10)
            return write(*args, **kwargs)

        monkeypatch.setattr(store, "_save_op", paused)
        pending = store.offer(tokens, _bases([0, 1, 2]), slot)
        assert pending is not None and started.wait(10)
        with store._lock:
            store._drop_entry(parent.key)
            assert store._occupied_bytes() == parent.total_bytes
            assert buffer() is not None
        release.set()
        assert pending.done.wait(10) and pending.success
        assert buffer() is None and not store._pending_ram
        assert _entry_for(store, tokens).parent_key is None
        assert store._reserved_bytes == 0 and not store._pins
    finally:
        release.set()
        store.close()


class _WriteSpy:
    """Count bytes actually handed to ``write()`` on files opened for writing under ``root``."""

    def __init__(self, monkeypatch, root: Path) -> None:
        self.total = 0
        self.files: dict[str, int] = {}
        root = root.resolve()
        spy = self
        real_open = Path.open

        class Counting:
            def __init__(self, inner, name: str) -> None:
                self._inner = inner
                self._name = name

            def write(self, data):
                written = self._inner.write(data)
                count = len(data.encode("utf-8")) if isinstance(data, str) else int(written or 0)
                spy.total += count
                spy.files[self._name] = spy.files.get(self._name, 0) + count
                return written

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *args):
                return self._inner.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def open_(path, mode="r", *args, **kwargs):
            handle = real_open(path, mode, *args, **kwargs)
            try:
                inside = Path(path).resolve().parent == root
            except OSError:
                inside = False
            if inside and any(flag in mode for flag in "wa+"):
                return Counting(handle, Path(path).name)
            return handle

        monkeypatch.setattr(Path, "open", open_)

    def reset(self) -> None:
        self.total = 0
        self.files = {}


def test_ssd_incremental_second_turn_writes_far_fewer_bytes(tmp_path: Path, monkeypatch):
    # 8,192 tokens = 2,048 pages on the first turn, one more page on the second. The second
    # save must write its 32 KiB token list, headers, padding, ONE page and one state slot,
    # not the 2,048 pages again. On this CPU geometry that is a >= 5x bound; the live fp8 262k
    # profile is 8.29x at 65,536 + 256 tokens (plan.md arithmetic), not measured here.
    kv_pool, state_pool = _qsa_pool(num_pages=2050), _state_pool()
    first_slot, second_slot = state_pool.alloc(2)
    first_ids = list(range(2048))
    _fill_raw(_page_views(kv_pool, first_ids), seed=1)
    _fill_raw(state_pool.slot_byte_views(first_slot), seed=2)
    first_tokens = torch.arange(8192, dtype=torch.int32)
    second_tokens = torch.arange(8196, dtype=torch.int32)
    store = _store("ssd", tmp_path, kv_pool, state_pool, disk_budget_bytes=4 << 20)
    spy = _WriteSpy(monkeypatch, tmp_path)
    try:
        spy.reset()
        first = store.offer(first_tokens, _bases(first_ids), first_slot)
        assert first is not None and first.wait() and first.error is None
        store.flush()
        first_bytes = spy.total
        first_entry = _entry_for(store, first_tokens)
        assert first_entry is not None and first_entry.path is not None
        first_file = first_entry.path.read_bytes()
        assert first_bytes >= len(first_file) + 4096, "spy must see the real file plus the header rewrite"

        # Turn two: the old pages are untouched, one page is added, the whole state changes.
        _fill_raw(_page_views(kv_pool, [2048]), seed=3)
        _fill_raw(state_pool.slot_byte_views(second_slot), seed=4)
        spy.reset()
        second = store.offer(second_tokens, _bases(first_ids + [2048]), second_slot)
        assert second is not None and second.wait() and second.error is None
        store.flush()
        second_bytes = spy.total

        assert second_bytes < 0.20 * first_bytes, (first_bytes, second_bytes, spy.files)
        second_entry = _entry_for(store, second_tokens)
        assert second_entry is not None and second_entry.path is not None
        assert second_entry.parent_key == first_entry.key
        assert second_entry.parent_token_count == 8192
        assert first_entry.path.read_bytes() == first_file
        # Physical second file: header + 8,196 int32 tokens, padded; one page of KV, padded;
        # one state slot. Nothing else.
        assert second_entry.kv_bytes == 4 * kv_pool.unit_bytes()[0]
        assert second_entry.state_bytes == state_pool.bytes_per_slot()
        assert second_entry.kv_offset == park_module._align_up(4096 + 8196 * 4)
        assert second_entry.state_offset == park_module._align_up(
            second_entry.kv_offset + second_entry.kv_bytes
        )
        assert second_entry.path.stat().st_size == second_entry.state_offset + second_entry.state_bytes
        assert store.status()["parked_count"] == 2
        assert store._reserved_bytes == 0 and not store._pins
    finally:
        store.close()


@pytest.mark.parametrize("kv_dtype", [None, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize("page_offset", [0, 1, 2, 5, 6])
def test_ssd_incremental_restore_is_byte_identical_to_full_snapshot(
    tmp_path: Path, kv_dtype, page_offset: int, monkeypatch
):
    kv_pool = _qsa_pool(num_pages=48, kv_dtype=kv_dtype)
    state_pool = _state_pool(num_slots=8)
    slots = state_pool.alloc(5)
    source_ids = [4, 20, 8, 36, 12, 28]  # scattered physical pages, tree order = list order
    tokens = torch.arange(24, dtype=torch.int32) + 5
    turns = [(2, slots[0]), (4, slots[1]), (6, slots[2])]
    incremental = _store(
        "ssd", tmp_path / "inc", kv_pool, state_pool, pinned_window_bytes=4096
    )
    filled = 0
    for turn, (pages, slot) in enumerate(turns):
        _fill_raw(_page_views(kv_pool, source_ids[filled:pages]), seed=10 + turn)
        filled = pages
        _fill_raw(state_pool.slot_byte_views(slot), seed=20 + turn)
        if turn == len(turns) - 1:
            oracle_kv = _raw_bytes(_page_views(kv_pool, source_ids))
            oracle_state = _raw_bytes(state_pool.slot_byte_views(slot))
        assert incremental.save(tokens[: pages * 4], _bases(source_ids[:pages]), slot)
    keys = rolling_page_keys(tokens, 4, "model-A")
    third = incremental._entries[keys[5]]
    second = incremental._entries[keys[3]]
    assert third.parent_key == keys[3] and second.parent_key == keys[1]
    assert incremental._entries[keys[1]].parent_key is None
    incremental.close()
    (tmp_path / "inc" / "park.json").unlink()

    # The old full-snapshot semantics: the same final source parked alone is one root.
    control = _store("ssd", tmp_path / "full", kv_pool, state_pool, pinned_window_bytes=4096)
    assert control.save(tokens, _bases(source_ids), slots[2])
    assert control._entries[keys[5]].parent_key is None
    control_ids = [40, 41, 42, 43, 44, 45][page_offset:]
    control_hit = control.lookup(tokens)
    assert control_hit is not None
    control.restore(control_hit, _bases(control_ids), slots[3], page_offset=page_offset)
    control.close()

    reopened = _store(
        "ssd", tmp_path / "inc", kv_pool, state_pool, pinned_window_bytes=3 * 4096
    )
    try:
        target_ids = [1, 3, 5, 7, 9, 11][page_offset:]
        untouched_ids = [
            page for page in range(48)
            if page not in source_ids and page not in target_ids and page not in control_ids
        ]
        _fill_raw(_page_views(kv_pool, untouched_ids), seed=90)
        _fill_raw(state_pool.slot_byte_views(slots[4]), seed=91)
        sentinel_kv = _raw_bytes(_page_views(kv_pool, untouched_ids))
        sentinel_state = _raw_bytes(state_pool.slot_byte_views(slots[4]))

        hit = reopened.lookup(torch.cat([tokens, torch.tensor([999], dtype=torch.int32)]))
        assert hit is not None and hit.token_count == 24
        page_calls = []
        real_page_views = kv_pool.page_byte_views

        def counted_page_views(page):
            page_calls.append(page)
            return real_page_views(page)

        monkeypatch.setattr(kv_pool, "page_byte_views", counted_page_views)
        reopened.restore(hit, _bases(target_ids), slots[3], page_offset=page_offset)
        assert page_calls == target_ids, "restore must build each destination page once"

        views_per_page = len(kv_pool.page_byte_views(0))
        restored_kv = _raw_bytes(_page_views(kv_pool, target_ids))
        _assert_raw_equal(restored_kv, oracle_kv[page_offset * views_per_page :])
        _assert_raw_equal(_raw_bytes(state_pool.slot_byte_views(slots[3])), oracle_state)
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, control_ids)), restored_kv)
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, source_ids)), oracle_kv)
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, untouched_ids)), sentinel_kv)
        _assert_raw_equal(_raw_bytes(state_pool.slot_byte_views(slots[4])), sentinel_state)
        breakdown = reopened.status()["last_restore_breakdown_ms"]
        assert breakdown["segments"] == 3.0
    finally:
        reopened.close()


def _chain_store(tmp_path: Path, *, budget: int = 1 << 20, num_slots: int = 8, pools=None):
    if pools is None:
        pools = _qsa_pool(num_pages=64), _state_pool(num_slots=num_slots)
    kv_pool, state_pool = pools
    store = _store("ssd", tmp_path, kv_pool, state_pool, disk_budget_bytes=budget)
    return store, kv_pool, state_pool


def _save_turn(store, kv_pool, state_pool, tokens, page_ids, slot, *, seed: int, new_pages):
    _fill_raw(_page_views(kv_pool, new_pages), seed=seed)
    _fill_raw(state_pool.slot_byte_views(slot), seed=seed + 50)
    assert store.save(tokens, _bases(page_ids), slot)
    entry = _entry_for(store, tokens)
    assert entry is not None
    return entry


def test_ssd_branching_families_pick_the_longest_prefix_and_leave_counters_alone(tmp_path: Path):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(4)
    base = torch.arange(16, dtype=torch.int32)
    fork = torch.cat([base[:8], torch.tensor([70, 71, 72, 73], dtype=torch.int32)])
    try:
        generation = store.generation
        a = _save_turn(store, kv_pool, state_pool, base[:8], [2, 5], slots[0], seed=1, new_pages=[2, 5])
        b = _save_turn(store, kv_pool, state_pool, base[:12], [2, 5, 9], slots[1], seed=2, new_pages=[9])
        c = _save_turn(store, kv_pool, state_pool, base, [2, 5, 9, 13], slots[2], seed=3, new_pages=[13])
        d = _save_turn(store, kv_pool, state_pool, fork, [2, 5, 17], slots[3], seed=4, new_pages=[17])
        assert a.parent_key is None
        assert b.parent_key == a.key and c.parent_key == b.key and d.parent_key == a.key
        assert store._children[a.key] == {b.key, d.key}
        assert {entry.root_key for entry in store._entries.values()} == {a.key}
        # Parent planning is not a lookup: no hit/miss moved, and only the four publishes bumped
        # the generation.
        status = store.status()
        assert status["hits"] == 0 and status["misses"] == 0
        assert store.generation == generation + 4
        assert c.kv_bytes == 4 * kv_pool.unit_bytes()[0]
    finally:
        store.close()


def test_ssd_changed_prefix_bytes_fall_back_to_a_standalone_root(tmp_path: Path):
    # Equal tokens do not prove equal KV bytes. When the current source prefix no longer hashes
    # to the parent's recorded region, the turn is written as an independent root and the old
    # family may be evicted to make room for it.
    kv_pool, state_pool = _qsa_pool(num_pages=2052), _state_pool()
    slots = state_pool.alloc(3)
    ids = list(range(2048))
    tokens_a = torch.arange(8192, dtype=torch.int32)
    tokens_b = torch.arange(8196, dtype=torch.int32)
    probe = _store("ssd", tmp_path / "probe", kv_pool, state_pool)
    root_a = probe.storage_bytes(8192)
    delta_b = probe._segment_bytes(8196, 8192)
    root_b = probe.storage_bytes(8196)
    probe.close()
    assert root_a + delta_b < root_a + root_b
    store = _store("ssd", tmp_path / "s", kv_pool, state_pool, disk_budget_bytes=root_a + delta_b + 64)
    try:
        a = _save_turn(store, kv_pool, state_pool, tokens_a, ids, slots[0], seed=1, new_pages=ids)
        # Changed state only: still incremental.
        _fill_raw(state_pool.slot_byte_views(slots[1]), seed=7)
        _fill_raw(_page_views(kv_pool, [2048]), seed=8)
        assert store.save(tokens_b, _bases(ids + [2048]), slots[1])
        b = _entry_for(store, tokens_b)
        assert b is not None and b.parent_key == a.key
        target = state_pool.alloc(1)[0]
        hit = store.lookup(tokens_b)
        store.restore(hit, _bases([2049]), target, page_offset=2048)
        _assert_raw_equal(
            _raw_bytes(state_pool.slot_byte_views(target)),
            _raw_bytes(state_pool.slot_byte_views(slots[1])),
        )
        store._drop_entry(b.key)
        assert (tmp_path / "s" / f"{b.key}.park").exists() is False

        # Now flip one byte inside page 0 of the prefix and park the same tokens again.
        page0 = kv_pool.page_byte_views(0)[0].view(torch.uint8).reshape(-1)
        page0[3] ^= 0xFF
        _fill_raw(state_pool.slot_byte_views(slots[2]), seed=9)
        assert store.save(tokens_b, _bases(ids + [2048]), slots[2])
        b2 = _entry_for(store, tokens_b)
        assert b2 is not None and b2.parent_key is None
        assert b2.kv_bytes == 8196 * kv_pool.unit_bytes()[0]
        assert store.lookup(tokens_a) is None, "the stale family was evicted to fit the root"
        oracle = _raw_bytes(_page_views(kv_pool, ids + [2048]))
        target_ids = [2050]
        hit = store.lookup(tokens_b)
        store.restore(hit, _bases(target_ids), target, page_offset=2048)
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, target_ids)), oracle[-len(kv_pool.page_byte_views(0)):])
        assert store.status()["disabled"] is False
        assert store._reserved_bytes == 0 and not store._pins
    finally:
        store.close()


def test_ssd_root_fallback_declines_when_the_only_evictable_family_is_pinned(
    tmp_path: Path, monkeypatch
):
    kv_pool, state_pool = _qsa_pool(num_pages=2052), _state_pool()
    slots = state_pool.alloc(3)
    ids = list(range(2048))
    tokens_a = torch.arange(8192, dtype=torch.int32)
    tokens_b = torch.arange(8196, dtype=torch.int32)
    tokens_c = torch.cat([tokens_a, torch.tensor([500, 501, 502, 503], dtype=torch.int32)])
    probe = _store("ssd", tmp_path / "probe", kv_pool, state_pool)
    root_a = probe.storage_bytes(8192)
    delta = probe._segment_bytes(8196, 8192)
    probe.close()
    budget = root_a + 2 * delta + 64
    assert budget < root_a + probe.storage_bytes(8196)
    store = _store("ssd", tmp_path / "s", kv_pool, state_pool, disk_budget_bytes=budget)
    started = threading.Event()
    release = threading.Event()
    real_check = store._prefix_matches

    checks = []

    def blocked_check(span, chain, cut):
        # Only B (the first check) sees a changed prefix byte; C shares the same physical
        # page 0 and must still match A.
        checks.append(1)
        if len(checks) > 1:
            return real_check(span, chain, cut)
        started.set()
        release.wait(timeout=10)
        page0 = kv_pool.page_byte_views(0)[0].view(torch.uint8).reshape(-1)
        page0[5] ^= 0xFF
        try:
            return real_check(span, chain, cut)
        finally:
            page0[5] ^= 0xFF

    monkeypatch.setattr(store, "_prefix_matches", blocked_check)
    try:
        a = _save_turn(store, kv_pool, state_pool, tokens_a, ids, slots[0], seed=1, new_pages=ids)
        _fill_raw(_page_views(kv_pool, [2048, 2049]), seed=2)
        _fill_raw(state_pool.slot_byte_views(slots[1]), seed=3)
        _fill_raw(state_pool.slot_byte_views(slots[2]), seed=4)
        b = store.offer(tokens_b, _bases(ids + [2048]), slots[1])
        assert b is not None and started.wait(timeout=5)
        # A second child of A is queued while B is active: A's family is pinned by both.
        c = store.offer(tokens_c, _bases(ids + [2049]), slots[2])
        assert c is not None
        assert store._pins == {a.key: 2}
        release.set()
        assert b.wait() is False and b.error is None, "B must decline, not fail"
        assert c.wait() is True
        assert store.status()["disabled"] is False
        assert _entry_for(store, tokens_b) is None
        c_entry = _entry_for(store, tokens_c)
        assert c_entry is not None and c_entry.parent_key == a.key
        assert store.lookup(tokens_a) is not None
        assert store._reserved_bytes == 0 and not store._pins and not store._inflight
        assert not list((tmp_path / "s").glob(".*.tmp"))
    finally:
        release.set()
        store.close()


def test_ssd_async_offer_admits_a_delta_that_a_full_copy_could_not_fit(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(num_pages=2052), _state_pool()
    slots = state_pool.alloc(2)
    ids = list(range(2048))
    tokens_a = torch.arange(8192, dtype=torch.int32)
    tokens_b = torch.arange(8196, dtype=torch.int32)
    probe = _store("ssd", tmp_path / "probe", kv_pool, state_pool)
    root_a = probe.storage_bytes(8192)
    delta = probe._segment_bytes(8196, 8192)
    full_b = probe.storage_bytes(8196)
    probe.close()
    store = _store("ssd", tmp_path / "s", kv_pool, state_pool, disk_budget_bytes=root_a + delta + 64)
    try:
        assert root_a + full_b > store.disk_budget_bytes
        a = _save_turn(store, kv_pool, state_pool, tokens_a, ids, slots[0], seed=1, new_pages=ids)
        _fill_raw(_page_views(kv_pool, [2048]), seed=2)
        _fill_raw(state_pool.slot_byte_views(slots[1]), seed=3)
        pending = store.offer(tokens_b, _bases(ids + [2048]), slots[1])
        assert pending is not None and pending.reserved_bytes == delta
        assert pending.wait() and pending.error is None
        b = _entry_for(store, tokens_b)
        assert b is not None and b.parent_key == a.key and b.total_bytes == delta
        assert store.lookup(tokens_a) is not None
    finally:
        store.close()


def test_ssd_duplicate_inflight_key_is_rejected_and_releases_everything(
    tmp_path: Path, monkeypatch
):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(2)
    tokens = torch.arange(12, dtype=torch.int32)
    started = threading.Event()
    release = threading.Event()
    real_write = store._write_ssd

    def blocked_write(op, span, on_source_copied=None):
        started.set()
        release.wait(timeout=10)
        return real_write(op, span, on_source_copied=on_source_copied)

    monkeypatch.setattr(store, "_write_ssd", blocked_write)
    try:
        a = _save_turn(store, kv_pool, state_pool, tokens[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
        _fill_raw(_page_views(kv_pool, [3]), seed=2)
        first = store.offer(tokens, _bases([1, 2, 3]), slots[1])
        assert first is not None and started.wait(timeout=5)
        assert store.offer(tokens, _bases([1, 2, 3]), slots[1]) is None
        assert store._pins == {a.key: 1}
        release.set()
        assert first.wait() and first.error is None
        assert _entry_for(store, tokens).parent_key == a.key
        assert store._reserved_bytes == 0 and not store._pins and not store._inflight
    finally:
        release.set()
        store.close()


def test_ssd_eviction_removes_a_whole_family_and_never_orphans_a_child(tmp_path: Path):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    one_entry = store.storage_bytes(8)
    assert store._segment_bytes(12, 8) == one_entry and store._segment_bytes(16, 12) == one_entry
    store.close()
    store, kv_pool, state_pool = _chain_store(
        tmp_path, budget=4 * one_entry + 64, pools=(kv_pool, state_pool)
    )
    slots = state_pool.alloc(7)
    base = torch.arange(16, dtype=torch.int32)
    other = torch.arange(300, 308, dtype=torch.int32)
    third = torch.arange(400, 408, dtype=torch.int32)
    fourth = torch.arange(500, 508, dtype=torch.int32)

    def touch(*entries):
        for entry in entries:
            entry.last_used_ns = time.time_ns()
            time.sleep(0.001)

    try:
        a = _save_turn(store, kv_pool, state_pool, base[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
        b = _save_turn(store, kv_pool, state_pool, base[:12], [1, 2, 3], slots[1], seed=2, new_pages=[3])
        c = _save_turn(store, kv_pool, state_pool, base, [1, 2, 3, 4], slots[2], seed=3, new_pages=[4])
        r = _save_turn(store, kv_pool, state_pool, other, [5, 6], slots[3], seed=4, new_pages=[5, 6])
        # A is the oldest ENTRY but its family was used most recently through C: entry-wise LRU
        # would delete A and orphan B and C; family LRU evicts the standalone R instead.
        touch(a, r, b, c)
        assert store.status()["parked_count"] == 4
        s1 = _save_turn(store, kv_pool, state_pool, third, [7, 8], slots[4], seed=5, new_pages=[7, 8])
        assert set(store._entries) == {a.key, b.key, c.key, s1.key}
        # Now the chain is the least recently used family: it goes as a unit, longest first.
        touch(c, b, a, s1)
        s2 = _save_turn(store, kv_pool, state_pool, fourth, [9, 10], slots[5], seed=6, new_pages=[9, 10])
        remaining = set(store._entries)
        assert remaining == {s1.key, s2.key}
        for entry in store._entries.values():
            assert entry.parent_key is None or entry.parent_key in store._entries
        assert sorted(path.stem for path in tmp_path.glob("*.park")) == sorted(remaining)
        assert store.lookup(base) is None
    finally:
        store.close()


@pytest.mark.parametrize("same_family", [False, True])
def test_ssd_pressure_pins_the_parent_before_eviction_or_rolls_over(tmp_path: Path, same_family):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(3)
    tokens = torch.arange(16, dtype=torch.int32)
    store.disk_budget_bytes = 2 * store.storage_bytes(8) + 64
    try:
        a = _save_turn(store, kv_pool, state_pool, tokens[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
        middle = tokens[:12] if same_family else torch.arange(100, 108, dtype=torch.int32)
        middle_pages = [1, 2, 3] if same_family else [5, 6]
        other = _save_turn(store, kv_pool, state_pool, middle, middle_pages, slots[1], seed=2,
                           new_pages=[3] if same_family else [5, 6])
        a.last_used_ns, other.last_used_ns = 1, 2
        final = tokens if same_family else tokens[:12]
        pages = [1, 2, 3, 4] if same_family else [1, 2, 3]
        _fill_raw(_page_views(kv_pool, [pages[-1]]), seed=3)
        pending = store.offer(final, _bases(pages), slots[2])
        assert pending is not None and pending.wait() and pending.error is None
        child = _entry_for(store, final)
        assert other.key not in store._entries
        if same_family:
            assert child.parent_key is None and a.key not in store._entries
        else:
            assert child.parent_key == a.key and a.key in store._entries
            assert child.kv_bytes == 4 * kv_pool.unit_bytes()[0]
        assert store._occupied_bytes() <= store.disk_budget_bytes
        assert not store._pins and not store._inflight and store._reserved_bytes == 0
        store.restore(child, _bases([10, 11, 12, 13][:len(pages)]), slots[0])
        _assert_raw_equal(
            _raw_bytes(_page_views(kv_pool, [10, 11, 12, 13][:len(pages)])),
            _raw_bytes(_page_views(kv_pool, pages)),
        )
    finally:
        store.close()


@pytest.mark.parametrize("failure", ["create", "record"])
def test_ssd_offer_event_failure_releases_the_unqueued_operation(tmp_path: Path, monkeypatch, failure):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(2)
    tokens = torch.arange(12, dtype=torch.int32)
    try:
        a = _save_turn(store, kv_pool, state_pool, tokens[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])

        class FailedEvent:
            def __init__(self, **kwargs):
                if failure == "create":
                    raise RuntimeError("source-ready event failed")

            def record(self, stream):
                raise RuntimeError("source-ready event failed")

        monkeypatch.setattr(store, "_stream", object())
        monkeypatch.setattr(torch.cuda, "Event", FailedEvent)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: None)
        with pytest.raises(RuntimeError, match="source-ready event failed"):
            store.offer(tokens, _bases([1, 2, 3]), slots[1])
        assert store._save_queue.empty()
        assert store._reserved_bytes == 0 and not store._pins and not store._inflight
        assert store.status()["disabled"] is False
        monkeypatch.setattr(store, "_stream", None)
        pending = store.offer(tokens, _bases([1, 2, 3]), slots[1])
        assert pending is not None and pending.wait()
        assert _entry_for(store, tokens).parent_key == a.key
    finally:
        store.close()


def test_ssd_restart_rejects_children_of_a_missing_or_replaced_parent(tmp_path: Path):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(4)
    base = torch.arange(16, dtype=torch.int32)
    a = _save_turn(store, kv_pool, state_pool, base[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
    b = _save_turn(store, kv_pool, state_pool, base[:12], [1, 2, 3], slots[1], seed=2, new_pages=[3])
    c = _save_turn(store, kv_pool, state_pool, base, [1, 2, 3, 4], slots[2], seed=3, new_pages=[4])
    store.close()
    (tmp_path / "park.json").unlink()

    # Replace the parent by a fresh root with different bytes: same key, new metadata digest.
    a.path.unlink()
    _fill_raw(_page_views(kv_pool, [1]), seed=77)
    rewrite = _chain_store(tmp_path / "other", pools=(kv_pool, state_pool))[0]
    assert rewrite.save(base[:8], _bases([1, 2]), slots[0])
    rewritten = _entry_for(rewrite, base[:8])
    rewrite.close()
    assert rewritten.metadata_sha256 != a.metadata_sha256
    (tmp_path / a.path.name).write_bytes(rewritten.path.read_bytes())

    reopened = _chain_store(tmp_path, pools=(kv_pool, state_pool))[0]
    try:
        assert set(reopened._entries) == {a.key}
        assert not (tmp_path / b.path.name).exists() and not (tmp_path / c.path.name).exists()
        assert reopened.lookup(base) is not None and reopened.lookup(base).token_count == 8
    finally:
        reopened.close()

    # And with the parent gone entirely, the chain is a miss, not a partial restore.
    (tmp_path / a.path.name).unlink()
    reopened = _chain_store(tmp_path, pools=(kv_pool, state_pool))[0]
    try:
        assert reopened.status()["parked_count"] == 0
        assert reopened.lookup(base) is None
    finally:
        reopened.close()


def test_ssd_restart_recovers_a_chain_behind_a_stale_or_old_manifest(tmp_path: Path):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(4)
    base = torch.arange(16, dtype=torch.int32)
    a = _save_turn(store, kv_pool, state_pool, base[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
    stale = (tmp_path / "park.json").read_text(encoding="utf-8")
    b = _save_turn(store, kv_pool, state_pool, base[:12], [1, 2, 3], slots[1], seed=2, new_pages=[3])
    oracle = _raw_bytes(_page_views(kv_pool, [1, 2, 3]))
    oracle_state = _raw_bytes(state_pool.slot_byte_views(slots[1]))
    store.close()

    for manifest in (stale, stale.replace('"version": 5', '"version": 4')):
        (tmp_path / "park.json").write_text(manifest, encoding="utf-8")
        reopened = _chain_store(tmp_path, pools=(kv_pool, state_pool))[0]
        try:
            assert set(reopened._entries) == {a.key, b.key}
            hit = reopened.lookup(base[:12])
            assert hit is not None and hit.token_count == 12 and hit.parent_key == a.key
            reopened.restore(hit, _bases([10, 11, 12]), slots[2])
            _assert_raw_equal(_raw_bytes(_page_views(kv_pool, [10, 11, 12])), oracle)
            _assert_raw_equal(_raw_bytes(state_pool.slot_byte_views(slots[2])), oracle_state)
            doc = json.loads((tmp_path / "park.json").read_text(encoding="utf-8"))
            assert doc["version"] == 5
            assert {row["key"]: row["parent_key"] for row in doc["entries"]} == {
                a.key: None,
                b.key: a.key,
            }
        finally:
            reopened.close()


def test_ssd_restart_rejects_a_truncated_or_placeholder_segment(tmp_path: Path):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(3)
    base = torch.arange(12, dtype=torch.int32)
    a = _save_turn(store, kv_pool, state_pool, base[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
    b = _save_turn(store, kv_pool, state_pool, base, [1, 2, 3], slots[1], seed=2, new_pages=[3])
    store.close()
    raw = b.path.read_bytes()
    b.path.write_bytes(raw[:-1])  # short state region
    reopened = _chain_store(tmp_path, pools=(kv_pool, state_pool))[0]
    assert set(reopened._entries) == {a.key} and not b.path.exists()
    reopened.close()

    # A file whose header still carries the writer's placeholder digests (a crash before the
    # final header rewrite) is never admitted, even at the right size.
    meta = json.loads(raw[16 : 16 + int.from_bytes(raw[12:16], "little")].decode("utf-8"))
    meta["kv_sha256"] = "0" * 64
    meta["metadata_sha256"] = "0" * 64
    header = park_module.ParkStore._header(meta)
    b.path.write_bytes(header + raw[4096:])
    reopened = _chain_store(tmp_path, pools=(kv_pool, state_pool))[0]
    assert set(reopened._entries) == {a.key} and not b.path.exists()
    reopened.close()


def test_ssd_publish_failure_after_rename_is_rediscovered_on_restart(tmp_path: Path, monkeypatch):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(3)
    base = torch.arange(12, dtype=torch.int32)
    a = _save_turn(store, kv_pool, state_pool, base[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
    _fill_raw(_page_views(kv_pool, [3]), seed=2)
    _fill_raw(state_pool.slot_byte_views(slots[1]), seed=3)
    oracle = _raw_bytes(_page_views(kv_pool, [1, 2, 3]))
    calls = []

    real_manifest = store._write_manifest

    def failing_manifest():
        if not calls:
            calls.append(1)
            raise OSError("manifest device gone")
        real_manifest()

    monkeypatch.setattr(store, "_write_manifest", failing_manifest)
    assert store.save(base, _bases([1, 2, 3]), slots[1]) is False
    assert calls and store.status()["disabled"] is True
    store.close()
    files = sorted(path.stem for path in tmp_path.glob("*.park"))
    reopened = _chain_store(tmp_path, pools=(kv_pool, state_pool))[0]
    try:
        assert sorted(reopened._entries) == files and len(files) == 2
        hit = reopened.lookup(base)
        assert hit is not None and hit.token_count == 12 and hit.parent_key == a.key
        reopened.restore(hit, _bases([10, 11, 12]), slots[2])
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, [10, 11, 12])), oracle)
    finally:
        reopened.close()


def test_ssd_short_write_fails_the_save_and_leaves_no_partial_file(tmp_path: Path, monkeypatch):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(2)
    base = torch.arange(12, dtype=torch.int32)
    a = _save_turn(store, kv_pool, state_pool, base[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
    real_write_all = park_module._write_all
    writes = []

    def failing_write_all(handle, raw):
        writes.append(len(memoryview(raw)))
        if len(writes) == 4:  # inside the KV region, after header + tokens + padding
            raise OSError("disk full mid-region")
        real_write_all(handle, raw)

    monkeypatch.setattr(park_module, "_write_all", failing_write_all)
    try:
        assert store.save(base, _bases([1, 2, 3]), slots[1]) is False
        assert "disk full" in str(store.status()["last_error"])
        assert sorted(path.stem for path in tmp_path.glob("*.park")) == [a.key]
        assert not list(tmp_path.glob(".*.tmp"))
        assert store._reserved_bytes == 0 and not store._pins and not store._inflight
    finally:
        store.close()


def _corrupt(path: Path, offset: int) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        byte = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([byte[0] ^ 0xFF]))


@pytest.mark.parametrize("victim", ["root_kv", "middle_kv", "final_kv", "final_state"])
def test_ssd_corruption_anywhere_in_the_chain_is_a_miss_not_wrong_kv(tmp_path: Path, victim: str):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(5)
    base = torch.arange(16, dtype=torch.int32)
    a = _save_turn(store, kv_pool, state_pool, base[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
    b = _save_turn(store, kv_pool, state_pool, base[:12], [1, 2, 3], slots[1], seed=2, new_pages=[3])
    c = _save_turn(store, kv_pool, state_pool, base, [1, 2, 3, 4], slots[2], seed=3, new_pages=[4])
    oracle_a = _raw_bytes(_page_views(kv_pool, [1, 2]))
    oracle_a_state = _raw_bytes(state_pool.slot_byte_views(slots[0]))
    try:
        target, offset = {
            "root_kv": (a, a.kv_offset + 1),
            "middle_kv": (b, b.kv_offset + 1),
            "final_kv": (c, c.kv_offset + 1),
            "final_state": (c, c.state_offset + 1),
        }[victim]
        _corrupt(target.path, offset)
        hit = store.lookup(base)
        assert hit is c
        with pytest.raises(ParkEntryRejected, match="checksum"):
            store.restore(hit, _bases([20, 21, 22]), slots[3], page_offset=1)
        assert store.lookup(base) is None or store.lookup(base).token_count < target.token_count
        # The corrupt segment and everything that depends on its bytes are gone; the ancestors
        # it does not reference stay valid on their own.
        survivors = {entry.key for entry in (a, b, c) if entry.token_count < target.token_count}
        assert set(store._entries) == survivors
        if a.key in survivors:
            hit = store.lookup(base[:8])
            assert hit is a
            store.restore(hit, _bases([30, 31]), slots[4])
            _assert_raw_equal(_raw_bytes(_page_views(kv_pool, [30, 31])), oracle_a)
            _assert_raw_equal(_raw_bytes(state_pool.slot_byte_views(slots[4])), oracle_a_state)
    finally:
        store.close()


def test_ssd_child_restore_never_reads_or_validates_an_ancestor_state(tmp_path: Path):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(4)
    base = torch.arange(12, dtype=torch.int32)
    a = _save_turn(store, kv_pool, state_pool, base[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
    b = _save_turn(store, kv_pool, state_pool, base, [1, 2, 3], slots[1], seed=2, new_pages=[3])
    oracle = _raw_bytes(_page_views(kv_pool, [1, 2, 3]))
    oracle_state = _raw_bytes(state_pool.slot_byte_views(slots[1]))
    try:
        _corrupt(a.path, a.state_offset + 2)
        hit = store.lookup(base)
        assert hit is b
        store.restore(hit, _bases([10, 11, 12]), slots[2])
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, [10, 11, 12])), oracle)
        _assert_raw_equal(_raw_bytes(state_pool.slot_byte_views(slots[2])), oracle_state)
        with pytest.raises(ParkEntryRejected, match="checksum"):
            store.restore(store.lookup(base[:8]), _bases([20, 21]), slots[3])
        assert store.lookup(base[:8]) is None
    finally:
        store.close()


def test_ssd_chain_restore_reads_each_kv_region_once_and_only_the_target_state(
    tmp_path: Path, monkeypatch
):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(4)
    base = torch.arange(16, dtype=torch.int32)
    a = _save_turn(store, kv_pool, state_pool, base[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
    b = _save_turn(store, kv_pool, state_pool, base[:12], [1, 2, 3], slots[1], seed=2, new_pages=[3])
    c = _save_turn(store, kv_pool, state_pool, base, [1, 2, 3, 4], slots[2], seed=3, new_pages=[4])
    oracle = _raw_bytes(_page_views(kv_pool, [2, 3, 4]))
    oracle_state = _raw_bytes(state_pool.slot_byte_views(slots[2]))
    reads: list[tuple[str, int, int]] = []
    window_indices: list[int] = []
    window_ptrs = [int(window.data_ptr()) for window in store._windows]

    class FakeReader:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.data = path.read_bytes()

        def read_into(self, buffer, offset, length):
            ptr = int(torch.frombuffer(buffer, dtype=torch.uint8).data_ptr())
            window_indices.append(window_ptrs.index(ptr))
            chunk = self.data[offset : offset + length]
            buffer[: len(chunk)] = chunk
            reads.append((self.path.name, offset, len(chunk)))
            return len(chunk)

        def close(self):
            pass

    class FakeEvent:
        def __init__(self, **_kwargs):
            pass

        def record(self, _stream):
            pass

        def synchronize(self):
            pass

    class FakeStream:
        def synchronize(self):
            pass

    store.pinned_window_bytes = 64
    store._stream = FakeStream()
    monkeypatch.setattr(store, "_unbuffered_reader", lambda path: FakeReader(path))
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: contextlib.nullcontext())
    try:
        hit = store.lookup(base)
        store.restore(hit, _bases([20, 21, 22]), slots[3], page_offset=1)
        expected = []
        for segment, regions in ((a, ("kv",)), (b, ("kv",)), (c, ("kv", "state"))):
            for region in regions:
                offset = getattr(segment, region + "_offset")
                length = getattr(segment, region + "_bytes")
                expected.extend(
                    (segment.path.name, offset + o, min(64, length - o))
                    for o in range(0, length, 64)
                )
        assert reads == expected
        assert window_indices == [index % 2 for index in range(len(window_indices))]
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, [20, 21, 22])), oracle)
        _assert_raw_equal(_raw_bytes(state_pool.slot_byte_views(slots[3])), oracle_state)
    finally:
        store.close()


def test_restore_rejects_an_entry_the_store_no_longer_holds(tmp_path: Path):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(2)
    base = torch.arange(8, dtype=torch.int32)
    a = _save_turn(store, kv_pool, state_pool, base, [1, 2], slots[0], seed=1, new_pages=[1, 2])
    try:
        stale = store.lookup(base)
        assert stale is a
        store._drop_entry(a.key)
        with pytest.raises(ParkEntryRejected, match="no longer current"):
            store.restore(stale, _bases([3, 4]), slots[1])
        assert store.save(base, _bases([1, 2]), slots[0])
        replacement = store.lookup(base)
        assert replacement is not stale
        with pytest.raises(ParkEntryRejected, match="no longer current"):
            store.restore(stale, _bases([3, 4]), slots[1])
        assert store.lookup(base) is replacement and replacement.path.exists()
        store.restore(replacement, _bases([3, 4]), slots[1])
        _assert_raw_equal(
            _raw_bytes(_page_views(kv_pool, [3, 4])),
            _raw_bytes(_page_views(kv_pool, [1, 2])),
        )
    finally:
        store.close()


# ---- branch-aware (v5) parking ---------------------------------------------------------


def _pages(store: ParkStore, token_count: int) -> int:
    return token_count // store.page_size


def _reaches(store: ParkStore, entry, ancestor) -> bool:
    seen = set()
    current = entry
    while current is not None and current.key not in seen:
        if current is ancestor:
            return True
        seen.add(current.key)
        current = store._entries.get(current.parent_key) if current.parent_key else None
    return False


def _recurrence(state_views, steps: int = 3) -> torch.Tensor:
    """A deterministic CPU stand-in for continuing the recurrence from a restored state: any
    byte difference in the state diverges within one step."""
    value = torch.cat([view.view(torch.uint8).reshape(-1).to(torch.int64) for view in state_views])
    for _ in range(steps):
        value = (value * 31 + torch.arange(value.numel(), dtype=torch.int64) + 7) % 251
    return value


@pytest.mark.parametrize(
    "gap, shorter_side",
    [(44, False), (148, False), (46, False), (44, True)],
    ids=["gap44", "gap148", "gap46-unaligned", "gap44-shorter-side"],
)
def test_ssd_agent_shaped_saves_share_prompt_checkpoint(
    tmp_path: Path, monkeypatch, gap: int, shorter_side: bool
):
    """The 2026-09-10 live shape: one main request, two side requests that share its prefix
    with different tails, then the real next turn that diverges inside the previous side
    save's last pages (44/148 tokens before its end on the 5090). Every later request must
    attach to the main prompt checkpoint C and write only its own pages plus states, never a
    full copy; the shorter-side case is the observed shorter-exact-prefix save (71,424 of
    71,552), which must become a zero-KV child of the longer segment.

    The pool carries 32-wide heads (about 290 KV bytes per token) rather than the 38-byte
    fixture default: every segment stores its whole token list verbatim at 4 bytes per token,
    so with tiny KV the bound would measure the token list, not the pages. Live fp8 KV is
    over 10 KB per token, where the token list is noise; this is still not a live byte bound.
    """
    kv_pool = QSAKVCache(
        num_kv_heads=1,
        num_layers=2,
        head_dim=32,
        num_pages=4400,
        page_size=4,
        dtype=torch.bfloat16,
        kv_dtype=None,
        device=torch.device("cpu"),
        index_head_dim=8,
        num_index_layers=2,
        index_ratio=2,
        num_req_slots=2,
        layer_ids=(0, 1),
    )
    state_pool = _state_pool(num_slots=16)
    slots = state_pool.alloc(12)
    per_token = kv_pool.unit_bytes()[0]
    state_bytes = state_pool.bytes_per_slot()
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    control_spy = _WriteSpy(monkeypatch, control_dir)
    main_dir = tmp_path / "s"
    main_dir.mkdir()
    store = _store("ssd", main_dir, kv_pool, state_pool, disk_budget_bytes=64 << 20)
    spy = _WriteSpy(monkeypatch, main_dir)
    next_page = [0]

    def fresh_pages(count: int, seed: int) -> list[int]:
        ids = list(range(next_page[0], next_page[0] + count))
        next_page[0] += count
        _fill_raw(_page_views(kv_pool, ids), seed=seed)
        return ids

    def state(slot: int, seed: int) -> int:
        _fill_raw(state_pool.slot_byte_views(slot), seed=seed)
        return slot

    def checkpoint(tokens, pages, slot) -> None:
        # The scheduler's eager, synchronous prompt-checkpoint save.
        assert store.save(tokens, _bases(pages), slot)

    def finish(tokens, pages, slot) -> None:
        # The ordinary finish-time offer -> background write -> flush.
        pending = store.offer(tokens, _bases(pages), slot)
        assert pending is not None, (len(tokens), store.status(), store._inflight, store._pins)
        assert pending.wait() and pending.error is None, (len(tokens), store.status())
        store.flush()

    written: dict[str, int] = {}

    def measure(name: str, fn) -> None:
        spy.reset()
        fn()
        written[name] = spy.total

    def control(name: str, tokens, pages, slot) -> int:
        control_spy.reset()
        full = _store("ssd", control_dir, kv_pool, state_pool, disk_budget_bytes=64 << 20)
        try:
            assert full.save(tokens, _bases(pages), slot)
            entry = _entry_for(full, tokens)
            assert entry is not None and entry.parent_key is None
        finally:
            full.close()
        for path in control_dir.iterdir():
            path.unlink()
        return control_spy.total

    try:
        # Main: an 8,194-token prompt (checkpoint C at the aligned 8,192) and a 6-token answer.
        main_prompt = torch.arange(1, 8195, dtype=torch.int32)
        main_answer = torch.arange(50_000, 50_006, dtype=torch.int32)
        main_tokens = torch.cat([main_prompt, main_answer])          # 8,200: aligned finish
        main_pages = fresh_pages(_pages(store, 8200), seed=1)
        slot_c, slot_mf = state(slots[0], 10), state(slots[1], 11)

        def do_main():
            checkpoint(main_tokens[:8192], main_pages[:2048], slot_c)
            finish(main_tokens, main_pages, slot_mf)

        measure("main", do_main)
        c = _entry_for(store, main_tokens[:8192])
        mf = _entry_for(store, main_tokens)
        assert c is not None and c.parent_key is None
        assert mf is not None and mf.parent_key == c.key and mf.parent_token_count == 8192
        c_file = c.path.read_bytes()

        # Side 1 (permission classifier): main prefix + a 63-token tail, 8-token answer.
        side1_tokens = torch.cat([main_tokens, torch.arange(60_000, 60_068, dtype=torch.int32)])
        side1_pages = main_pages + fresh_pages(_pages(store, 8268) - 2050, seed=2)
        slot_c1, slot_f1 = state(slots[2], 12), state(slots[3], 13)

        def do_side1():
            checkpoint(side1_tokens[:8260], side1_pages[:2065], slot_c1)   # prompt 8,263 -> C1
            finish(side1_tokens, side1_pages, slot_f1)                     # answer -> 8,268

        measure("side1", do_side1)
        c1 = _entry_for(store, side1_tokens[:8260])
        f1 = _entry_for(store, side1_tokens)
        assert c1.parent_key == mf.key and c1.parent_token_count == 8200
        assert c1.kv_bytes == 60 * per_token
        assert f1.parent_key == c1.key and f1.kv_bytes == 8 * per_token

        # Side 2 (status line): either a longer different tail, or -- the observed shorter
        # case -- a prompt that is an exact prefix of the earlier, longer side-1 save.
        if shorter_side:
            side2_tokens = side1_tokens[:8256]
            side2_pages = side1_pages[:2064]
            slot_c2 = state(slots[4], 14)

            def do_side2():
                checkpoint(side2_tokens, side2_pages, slot_c2)   # prompt 8,256; unaligned finish

            measure("side2", do_side2)
            c2 = _entry_for(store, side2_tokens)
            assert c2.parent_key == c1.key, "the shortest longer donor wins, deterministically"
            assert c2.parent_token_count == 8256 == c2.token_count < c1.token_count
            assert c2.kv_bytes == 0 and c2.state_bytes == state_bytes
            assert c2.total_bytes == store._segment_bytes(8256, 8256)
            assert c2.total_bytes >= park_module._align_up(4096 + 8256 * 4) + state_bytes
            last_side_tokens, last_side_pages = side2_tokens, side2_pages
        else:
            side2_tokens = torch.cat(
                [main_tokens, torch.arange(70_000, 70_208, dtype=torch.int32)]
            )                                                            # prompt 8,407, +1 answer
            side2_pages = main_pages + fresh_pages(_pages(store, 8408) - 2050, seed=3)
            slot_c2, slot_f2 = state(slots[4], 14), state(slots[5], 15)

            def do_side2():
                checkpoint(side2_tokens[:8404], side2_pages[:2101], slot_c2)
                finish(side2_tokens, side2_pages, slot_f2)

            measure("side2", do_side2)
            c2 = _entry_for(store, side2_tokens[:8404])
            f2 = _entry_for(store, side2_tokens)
            assert c2.parent_key == mf.key and c2.kv_bytes == 204 * per_token
            assert f2.parent_key == c2.key and f2.kv_bytes == 4 * per_token
            last_side_tokens, last_side_pages = side2_tokens, side2_pages

        # The real next turn: shares the previous side save only up to ``gap`` tokens before
        # its end (inside its last pages), then continues with 300 new tokens.
        shared = len(last_side_tokens) - gap
        assert shared >= 8200, "the next turn must still contain the main finish"
        next_prompt = torch.cat(
            [last_side_tokens[:shared], torch.arange(80_000, 80_300, dtype=torch.int32)]
        )
        aligned_shared = shared - shared % 4
        assert aligned_shared < c2.token_count
        # No arbitrary-LCP state: before the next turn is saved, the longest saved endpoint
        # inside its prefix is the main finish, not a slice of the side save at the divergence.
        probe = store.lookup(next_prompt)
        assert probe is mf and probe.token_count == 8200
        l_n = len(next_prompt) - len(next_prompt) % 4
        next_tokens = torch.cat([next_prompt[:l_n], torch.arange(90_000, 90_004, dtype=torch.int32)])
        next_pages = main_pages + fresh_pages(_pages(store, l_n + 4) - 2050, seed=4)
        slot_ln, slot_fn = state(slots[6], 16), state(slots[7], 17)

        def do_next():
            checkpoint(next_tokens[:l_n], next_pages[: _pages(store, l_n)], slot_ln)
            finish(next_tokens, next_pages, slot_fn)

        measure("real_next", do_next)
        ln = _entry_for(store, next_tokens[:l_n])
        fn = _entry_for(store, next_tokens)
        assert ln.parent_key == mf.key and ln.parent_token_count == 8200
        assert ln.kv_bytes == (l_n - 8200) * per_token
        assert fn.parent_key == ln.key and fn.kv_bytes == 4 * per_token

        # Every later entry reaches the shared main checkpoint C, whose file never changed.
        for entry in store._entries.values():
            assert _reaches(store, entry, c), entry.token_count
        assert c.path.read_bytes() == c_file
        assert {entry.root_key for entry in store._entries.values()} == {c.key}

        # Each later request wrote far less than an independent full copy of its own path.
        controls = {
            "side1": control("side1", side1_tokens, side1_pages, slot_f1),
            "side2": control("side2", last_side_tokens, last_side_pages, slot_c2),
            "real_next": control("real_next", next_tokens, next_pages, slot_fn),
        }
        for name, full_bytes in controls.items():
            assert written[name] < 0.25 * full_bytes, (name, written[name], full_bytes, written)
        assert written["main"] > controls["side1"] * 0.9, "the first save is the full one"

        # Real restore hits for the side and next paths, at their own saved endpoints.
        hits_before = store.status()["hits"]
        probe_tail = torch.tensor([999], dtype=torch.int32)
        assert store.lookup(torch.cat([side1_tokens, probe_tail])) is f1
        assert store.lookup(torch.cat([next_tokens, probe_tail])) is fn
        side2_hit = store.lookup(torch.cat([last_side_tokens, probe_tail]))
        assert side2_hit is (c2 if shorter_side else f2)
        assert store.status()["hits"] == hits_before + 3
        if shorter_side:
            # The zero-KV side checkpoint restores its own state and the donor's KV bytes.
            target_pages = list(range(next_page[0], next_page[0] + 2064))
            target_slot = slots[8]
            store.restore(c2, _bases(target_pages), target_slot)
            _assert_raw_equal(
                _raw_bytes(_page_views(kv_pool, target_pages)),
                _raw_bytes(_page_views(kv_pool, side2_pages)),
            )
            _assert_raw_equal(
                _raw_bytes(state_pool.slot_byte_views(target_slot)),
                _raw_bytes(state_pool.slot_byte_views(slot_c2)),
            )
        assert store._reserved_bytes == 0 and not store._pins and not store._inflight
    finally:
        store.close()


@pytest.mark.parametrize("kv_dtype", [None, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize("page_offset", [0, 1, 5, 6], ids=["zero", "inside", "boundary", "state-only"])
@pytest.mark.parametrize("source", ["checkpoint", "shorter"])
def test_ssd_partial_prefix_restore_is_byte_identical(
    tmp_path: Path, monkeypatch, kv_dtype, page_offset: int, source: str
):
    """A prompt that starts with a page-aligned prefix of a saved chat, shorter than the whole
    save, restores KV and GDN state at that cut byte for byte -- from C's own saved state,
    never from the longer N's. ``checkpoint``: C was saved first (the eager prompt
    checkpoint) and N is its continuation. ``shorter``: N was saved first and C is a zero-KV
    child borrowing N's bytes through its own cut."""
    kv_pool = _qsa_pool(num_pages=64, kv_dtype=kv_dtype)
    state_pool = _state_pool(num_slots=8)
    slots = state_pool.alloc(4)
    source_ids = [4, 20, 8, 36, 12, 28, 40, 44, 48]     # scattered physical pages
    tokens = torch.arange(36, dtype=torch.int32) + 5
    _fill_raw(_page_views(kv_pool, source_ids), seed=10)
    _fill_raw(state_pool.slot_byte_views(slots[0]), seed=20)   # C's state
    _fill_raw(state_pool.slot_byte_views(slots[1]), seed=21)   # N's state, deliberately different
    assert not torch.equal(
        torch.cat(_raw_bytes(state_pool.slot_byte_views(slots[0]))),
        torch.cat(_raw_bytes(state_pool.slot_byte_views(slots[1]))),
    )
    oracle_source_kv = _raw_bytes(_page_views(kv_pool, source_ids))
    oracle_kv = _raw_bytes(_page_views(kv_pool, source_ids[:6]))
    oracle_state = _raw_bytes(state_pool.slot_byte_views(slots[0]))
    store = _store("ssd", tmp_path / "s", kv_pool, state_pool, pinned_window_bytes=4096)
    if source == "checkpoint":
        assert store.save(tokens[:24], _bases(source_ids[:6]), slots[0])
        assert store.save(tokens, _bases(source_ids), slots[1])
        c, n = _entry_for(store, tokens[:24]), _entry_for(store, tokens)
        assert c.parent_key is None and n.parent_key == c.key and n.parent_token_count == 24
    else:
        assert store.save(tokens, _bases(source_ids), slots[1])
        assert store.save(tokens[:24], _bases(source_ids[:6]), slots[0])
        c, n = _entry_for(store, tokens[:24]), _entry_for(store, tokens)
        assert n.parent_key is None and c.parent_key == n.key
        assert c.parent_token_count == 24 and c.kv_bytes == 0
    store.close()
    (tmp_path / "s" / "park.json").unlink()

    reopened = _store("ssd", tmp_path / "s", kv_pool, state_pool, pinned_window_bytes=3 * 4096)
    try:
        assert set(reopened._entries) == {c.key, n.key}
        target_ids = [1, 3, 5, 7, 9, 11][page_offset:]
        untouched = [
            page for page in range(64)
            if page not in source_ids and page not in target_ids
        ]
        _fill_raw(_page_views(kv_pool, untouched), seed=90)
        _fill_raw(state_pool.slot_byte_views(slots[3]), seed=91)
        sentinel_kv = _raw_bytes(_page_views(kv_pool, untouched))
        sentinel_state = _raw_bytes(state_pool.slot_byte_views(slots[3]))

        # Incoming prompt: C's 24 tokens, then 8 tokens that diverge from N before its end.
        prompt = torch.cat([tokens[:24], torch.arange(900, 908, dtype=torch.int32)])
        hit = reopened.lookup(prompt)
        assert hit is not None and hit.token_count == 24 and hit.key == c.key
        reads: list[tuple[str, int, int]] = []
        real_read = reopened._read_region

        def spying_read(ctx, segment, handle, reader, **region):
            reads.append((segment.key, region["file_offset"], region["length"]))
            return real_read(ctx, segment, handle, reader, **region)

        monkeypatch.setattr(reopened, "_read_region", spying_read)
        page_calls = []
        real_page_views = kv_pool.page_byte_views
        monkeypatch.setattr(
            kv_pool, "page_byte_views", lambda page: page_calls.append(page) or real_page_views(page)
        )
        reopened.restore(hit, _bases(target_ids), slots[2], page_offset=page_offset)
        assert page_calls == target_ids, "restore must build each destination page once"
        chain_c = reopened._entries[c.key]
        chain_n = reopened._entries[n.key]
        if source == "checkpoint":
            expected_reads = [
                (c.key, chain_c.kv_offset, chain_c.kv_bytes),
                (c.key, chain_c.state_offset, chain_c.state_bytes),
            ]
        else:
            # The whole donor KV region is read and hashed once (its unused tail included);
            # C's own empty KV region and C's state follow. N's state is never read.
            expected_reads = [
                (n.key, chain_n.kv_offset, chain_n.kv_bytes),
                (c.key, chain_c.kv_offset, 0),
                (c.key, chain_c.state_offset, chain_c.state_bytes),
            ]
        assert reads == expected_reads

        views_per_page = len(real_page_views(0))
        restored_kv = _raw_bytes(_page_views(kv_pool, target_ids))
        _assert_raw_equal(restored_kv, oracle_kv[page_offset * views_per_page :])
        restored_state = _raw_bytes(state_pool.slot_byte_views(slots[2]))
        _assert_raw_equal(restored_state, oracle_state)
        assert torch.equal(_recurrence(restored_state), _recurrence(oracle_state))
        assert not torch.equal(
            torch.cat(restored_state),
            torch.cat(_raw_bytes(state_pool.slot_byte_views(slots[1]))),
        ), "the restored state must be C's own, not N's"
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, source_ids)), oracle_source_kv)
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, untouched)), sentinel_kv)
        _assert_raw_equal(_raw_bytes(state_pool.slot_byte_views(slots[3])), sentinel_state)
        assert reopened.status()["last_restore_breakdown_ms"]["segments"] == (
            1.0 if source == "checkpoint" else 2.0
        )
    finally:
        reopened.close()

    # Negative: only the longer N exists. A request that ends before N must miss: there is no
    # saved state at any shorter cut, and a longer file is never sliced into one.
    only_n = _store("ssd", tmp_path / "n", kv_pool, state_pool)
    try:
        assert only_n.save(tokens, _bases(source_ids), slots[1])
        misses = only_n.status()["misses"]
        assert only_n.lookup(prompt) is None
        assert only_n.lookup(tokens[:24]) is None
        assert only_n.status()["misses"] == misses + 2
        assert set(only_n._entries) == {n.key}
    finally:
        only_n.close()


def test_ssd_partial_lookup_honors_min_and_max_lengths(tmp_path: Path):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(2)
    tokens = torch.arange(16, dtype=torch.int32)
    try:
        n = _save_turn(store, kv_pool, state_pool, tokens, [1, 2, 3, 4], slots[0], seed=1, new_pages=[1, 2, 3, 4])
        c = _save_turn(store, kv_pool, state_pool, tokens[:8], [1, 2], slots[1], seed=2, new_pages=[])
        assert c.parent_key == n.key and c.kv_bytes == 0
        prompt = torch.cat([tokens[:12], torch.tensor([500, 501, 502, 503], dtype=torch.int32)])
        assert store.lookup(prompt) is c
        assert store.lookup(prompt, max_len=8) is c
        assert store.lookup(prompt, min_len=12) is None
        assert store.lookup(torch.cat([tokens, tokens[:1]]), max_len=8) is c
        assert store.lookup(torch.cat([tokens, tokens[:1]])) is n
    finally:
        store.close()


def test_ssd_chain_with_several_zero_kv_checkpoints_restores_each_endpoint(tmp_path: Path):
    """A(16) <- B(20) continuation; C(12) and D(8) are zero-KV checkpoints that borrow from
    the shortest longer donor. Each endpoint restores its own state and exactly its prefix;
    a donor's unused tail never lands anywhere."""
    store, kv_pool, state_pool = _chain_store(tmp_path, num_slots=12)
    slots = state_pool.alloc(8)
    tokens = torch.arange(20, dtype=torch.int32) + 100
    try:
        a = _save_turn(store, kv_pool, state_pool, tokens[:16], [1, 2, 3, 4], slots[0], seed=1, new_pages=[1, 2, 3, 4])
        b = _save_turn(store, kv_pool, state_pool, tokens, [1, 2, 3, 4, 5], slots[1], seed=2, new_pages=[5])
        c = _save_turn(store, kv_pool, state_pool, tokens[:12], [1, 2, 3], slots[2], seed=3, new_pages=[])
        d = _save_turn(store, kv_pool, state_pool, tokens[:8], [1, 2], slots[3], seed=4, new_pages=[])
        assert b.parent_key == a.key and b.parent_token_count == 16
        assert c.parent_key == a.key and c.parent_token_count == 12 and c.kv_bytes == 0
        assert d.parent_key == c.key and d.parent_token_count == 8 and d.kv_bytes == 0
        assert store._chain(d) == (a, c, d)
        assert [(s.key, lo, hi) for s, lo, hi in store._chain_ranges((a, c, d), 8)] == [
            (a.key, 0, 8), (c.key, 12, 12), (d.key, 8, 8)
        ]
        for entry, pages, slot, target in (
            (d, [1, 2], slots[3], [10, 11]),
            (c, [1, 2, 3], slots[2], [20, 21, 22]),
            (b, [1, 2, 3, 4, 5], slots[1], [30, 31, 32, 33, 34]),
            (a, [1, 2, 3, 4], slots[0], [40, 41, 42, 43]),
        ):
            sentinel_pages = [page for page in range(50, 60)]
            _fill_raw(_page_views(kv_pool, sentinel_pages), seed=77)
            sentinel = _raw_bytes(_page_views(kv_pool, sentinel_pages))
            hit = store.lookup(torch.cat([tokens[: entry.token_count], torch.tensor([999], dtype=torch.int32)]))
            assert hit is entry
            store.restore(hit, _bases(target), slots[5])
            _assert_raw_equal(_raw_bytes(_page_views(kv_pool, target)), _raw_bytes(_page_views(kv_pool, pages)))
            _assert_raw_equal(
                _raw_bytes(state_pool.slot_byte_views(slots[5])),
                _raw_bytes(state_pool.slot_byte_views(slot)),
            )
            _assert_raw_equal(_raw_bytes(_page_views(kv_pool, sentinel_pages)), sentinel)
        # Offset restore of D through the clipped donor: only page 1 of A lands in the target.
        store.restore(d, _bases([12]), slots[6], page_offset=1)
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, [12])), _raw_bytes(_page_views(kv_pool, [2])))
    finally:
        store.close()


def test_ssd_shorter_checkpoint_falls_back_to_a_root_when_donor_bytes_differ(tmp_path: Path):
    """Equal tokens are not byte provenance: a shorter checkpoint whose current KV differs
    from the longer donor's file bytes becomes a standalone root, and a donor whose file is
    corrupt is dropped with its dependants instead of being trusted."""
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(4)
    tokens = torch.arange(16, dtype=torch.int32) + 7
    try:
        a = _save_turn(store, kv_pool, state_pool, tokens, [1, 2, 3, 4], slots[0], seed=1, new_pages=[1, 2, 3, 4])
        page1 = kv_pool.page_byte_views(1)[0].view(torch.uint8).reshape(-1)
        page1[3] ^= 0xFF
        _fill_raw(state_pool.slot_byte_views(slots[1]), seed=5)
        assert store.save(tokens[:8], _bases([1, 2]), slots[1])
        c = _entry_for(store, tokens[:8])
        assert c.parent_key is None and c.kv_bytes == 8 * kv_pool.unit_bytes()[0]
        assert a.key in store._entries
        store.restore(store.lookup(torch.cat([tokens[:8], tokens[:1]])), _bases([10, 11]), slots[2])
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, [10, 11])), _raw_bytes(_page_views(kv_pool, [1, 2])))
        page1[3] ^= 0xFF
        store._drop_entry(c.key)

        # Now the source matches again but A's file is corrupt inside the borrowed range.
        _corrupt(a.path, a.kv_offset + 5)
        _fill_raw(state_pool.slot_byte_views(slots[3]), seed=6)
        assert store.save(tokens[:8], _bases([1, 2]), slots[3])
        c2 = _entry_for(store, tokens[:8])
        assert c2.parent_key is None
        assert a.key not in store._entries and not a.path.exists()
        assert store.status()["disabled"] is False
        assert store._reserved_bytes == 0 and not store._pins and not store._inflight
    finally:
        store.close()


def test_ssd_drop_order_and_restart_follow_edges_when_a_child_is_shorter(tmp_path: Path):
    """A(16) <- C(8, zero-KV) <- D(12, cut 8, a fork after C): length no longer orders the
    graph. Dropping the family must delete D before C before A; a restart must admit the
    shorter child, reject a child whose parent is missing, and reject an injected cycle."""
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(4)
    tokens = torch.arange(16, dtype=torch.int32) + 3
    fork = torch.cat([tokens[:8], torch.arange(900, 904, dtype=torch.int32)])
    a = _save_turn(store, kv_pool, state_pool, tokens, [1, 2, 3, 4], slots[0], seed=1, new_pages=[1, 2, 3, 4])
    c = _save_turn(store, kv_pool, state_pool, tokens[:8], [1, 2], slots[1], seed=2, new_pages=[])
    d = _save_turn(store, kv_pool, state_pool, fork, [1, 2, 7], slots[2], seed=3, new_pages=[7])
    assert c.parent_key == a.key and d.parent_key == c.key and d.parent_token_count == 8
    assert d.kv_bytes == 4 * kv_pool.unit_bytes()[0]
    order = [entry.key for entry in store._drop_order([a, c, d])]
    assert order.index(d.key) < order.index(c.key) < order.index(a.key)
    oracle_d = _raw_bytes(_page_views(kv_pool, [1, 2, 7]))
    oracle_d_state = _raw_bytes(state_pool.slot_byte_views(slots[2]))
    store.close()

    (tmp_path / "park.json").unlink()
    reopened = _chain_store(tmp_path, pools=(kv_pool, state_pool))[0]
    try:
        assert set(reopened._entries) == {a.key, c.key, d.key}
        assert reopened._entries[d.key].parent_key == c.key
        assert {entry.root_key for entry in reopened._entries.values()} == {a.key}
        hit = reopened.lookup(torch.cat([fork, tokens[:1]]))
        assert hit is reopened._entries[d.key]
        reopened.restore(hit, _bases([20, 21, 22]), slots[3])
        _assert_raw_equal(_raw_bytes(_page_views(kv_pool, [20, 21, 22])), oracle_d)
        _assert_raw_equal(_raw_bytes(state_pool.slot_byte_views(slots[3])), oracle_d_state)
        # An in-memory cycle is never a valid chain and never an infinite walk; the broken
        # endpoint is dropped and nothing shorter on that path is left to answer.
        reopened._entries[a.key].parent_key = d.key
        assert reopened._chain(reopened._entries[d.key]) is None
        assert reopened.lookup(torch.cat([fork, tokens[:1]])) is None
        assert d.key not in reopened._entries
    finally:
        reopened.close()

    # Restart again with the shorter middle child's file gone: its dependant D (rewritten
    # below so it exists again) must be rejected and A survives on its own.
    (tmp_path / "park.json").unlink()
    rewrite = _chain_store(tmp_path, pools=(kv_pool, state_pool))[0]
    try:
        assert set(rewrite._entries) == {a.key}
        assert rewrite.save(tokens[:8], _bases([1, 2]), slots[1])
        assert rewrite.save(fork, _bases([1, 2, 7]), slots[2])
        c_again, d_again = _entry_for(rewrite, tokens[:8]), _entry_for(rewrite, fork)
        assert c_again.parent_key == a.key and d_again.parent_key == c_again.key
    finally:
        rewrite.close()
    (tmp_path / "park.json").unlink()
    c_again.path.unlink()
    reopened = _chain_store(tmp_path, pools=(kv_pool, state_pool))[0]
    try:
        assert set(reopened._entries) == {a.key}
        assert not d_again.path.exists()
        # Dropping the whole family removes every file, children first.
        reopened._drop_entry(a.key)
        assert not list(tmp_path.glob("*.park"))
    finally:
        reopened.close()


def test_ssd_lookup_drops_a_broken_endpoint_and_answers_with_a_shorter_valid_one(tmp_path: Path):
    store, kv_pool, state_pool = _chain_store(tmp_path)
    slots = state_pool.alloc(3)
    tokens = torch.arange(12, dtype=torch.int32)
    try:
        a = _save_turn(store, kv_pool, state_pool, tokens[:8], [1, 2], slots[0], seed=1, new_pages=[1, 2])
        b = _save_turn(store, kv_pool, state_pool, tokens, [1, 2, 3], slots[1], seed=2, new_pages=[3])
        # B names a parent digest that no longer matches: its chain is broken underneath it.
        b.parent_metadata_sha256 = "f" * 64
        hit = store.lookup(torch.cat([tokens, tokens[:1]]))
        assert hit is a
        assert b.key not in store._entries and not b.path.exists()
        assert store.status()["hits"] == 1 and store.status()["misses"] == 0
    finally:
        store.close()


def test_ssd_v4_files_and_manifest_are_invalidated_once(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    _fill_entry(kv_pool, state_pool, pages, slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, pages, slot)
    store.close()
    path = next(tmp_path.glob("*.park"))
    raw = bytearray(path.read_bytes())
    raw[len(park_module._MAGIC) : len(park_module._MAGIC) + 4] = (4).to_bytes(4, "little")
    path.write_bytes(bytes(raw))
    manifest = tmp_path / "park.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"version": 5', '"version": 4'), encoding="utf-8")
    other = tmp_path / "unrelated.txt"
    other.write_text("keep me", encoding="utf-8")

    rebuilt = _store("ssd", tmp_path, kv_pool, state_pool)
    try:
        assert rebuilt.lookup(tokens) is None
        assert rebuilt.status()["parked_count"] == 0
        assert not path.exists() and other.exists()
        assert json.loads(manifest.read_text(encoding="utf-8"))["version"] == 5
    finally:
        rebuilt.close()


def _ram_store(ttl_s: int, wall_ns) -> ParkStore:
    store = ParkStore(
        mode="ram", page_size=4, kv_pool=_qsa_pool(), state_pool=_state_pool(),
        fingerprint="fp", min_tokens=4, ram_budget_bytes=1 << 30, ssd_dir="/nonexistent",
        disk_budget_bytes=0, pinned_window_bytes=4096, idle_ms=0, ttl_s=ttl_s,
    )
    store._wall_ns = wall_ns  # the test's wall clock; the scheduler's monotonic clock is never used
    return store


def _seed_family(
    store: ParkStore, key: str, last_used_ns: int, *, path: Path | None = None
) -> None:
    """Insert one already-published entry directly, bypassing the save worker.

    ``ParkedEntry`` requires ``key``, ``token_ids``, ``token_count``, ``payload_bytes``,
    ``total_bytes`` and ``last_used_ns``; everything else here takes its natural empty
    value. ``root_key`` is left at its dataclass default (``""``), which every family-grouping
    site in park_store.py (``_evict_to_fit``, ``_family_ages``) treats as "root_key or key" --
    i.e. this entry is its own one-member family/root. ``path`` is set only for SSD-tier tests
    that need ``_manifest_doc``/``_unlink_or_defer`` to see a real on-disk file.
    """
    entry = park_module.ParkedEntry(
        key=key,
        token_ids=torch.arange(8, dtype=torch.int32),
        token_count=8,
        payload_bytes=1024,
        total_bytes=1024,
        last_used_ns=last_used_ns,
        path=path,
    )
    with store._lock:
        store._entries[key] = entry


def test_ttl_expiry_uses_the_wall_clock_and_two_clocks_with_different_origins(monkeypatch):
    """The store stamps last_used_ns with time.time_ns(); the scheduler's idle loop runs on
    time.monotonic_ns(). Ages must never mix the two (review of 702543b): a five-hour entry
    must expire after six wall-clock hours whatever the monotonic clock says.

    This is a genuine two-clock test, not just a bare constant: ``time.monotonic_ns`` is
    monkeypatched to a second, independently-advancing fake clock with an unrelated origin
    (seconds-since-boot vs. the wall clock's epoch), advanced by DIFFERENT amounts than the
    wall clock at every step, so any accidental read of the wrong clock inside the store would
    change the assertions below.
    """
    wall = {"now": 1_700_000_000 * 10**9}          # epoch-based, 2023
    mono = {"now": 12_345 * 10**9}                  # seconds since boot, unrelated origin
    monkeypatch.setattr(park_module.time, "monotonic_ns", lambda: mono["now"])
    store = _ram_store(ttl_s=5 * 3600, wall_ns=lambda: wall["now"])
    _seed_family(store, "old", last_used_ns=wall["now"])
    assert store.next_expiry_delay_ms() == 5 * 3600 * 1000
    wall["now"] += 4 * 3600 * 10**9
    mono["now"] += 60 * 10**9  # monotonic barely moves while 4 wall-hours pass
    assert store.sweep_expired() == 0 and store.status()["parked_count"] == 1
    wall["now"] += 2 * 3600 * 10**9
    mono["now"] += 60 * 10**9  # again a different, much smaller amount than the wall step
    assert store.next_expiry_delay_ms() == 0
    assert store.sweep_expired() == 1
    assert store.status()["parked_count"] == 0 and store.status()["expired_evictions"] == 1
    # Advancing ONLY the monotonic clock -- no wall-clock movement at all -- must change
    # neither the delay nor the sweep outcome for a fresh entry.
    _seed_family(store, "new", last_used_ns=wall["now"])
    delay_before = store.next_expiry_delay_ms()
    assert delay_before == 5 * 3600 * 1000
    mono["now"] += 10 * 3600 * 10**9
    assert store.next_expiry_delay_ms() == delay_before
    assert store.sweep_expired() == 0


def test_ttl_zero_never_expires_and_reports_no_deadline():
    wall = {"now": 1_700_000_000 * 10**9}
    store = _ram_store(ttl_s=0, wall_ns=lambda: wall["now"])
    _seed_family(store, "k", last_used_ns=wall["now"] - 10**15)
    assert store.next_expiry_delay_ms() is None
    assert store.sweep_expired() == 0
    assert store.status()["ttl_s"] == 0 and store.status()["next_expiry_s"] is None


def test_expiry_evicts_the_oldest_family_first_and_skips_pinned_ones():
    wall = {"now": 1_700_000_000 * 10**9}
    store = _ram_store(ttl_s=3600, wall_ns=lambda: wall["now"])
    _seed_family(store, "a", last_used_ns=wall["now"] - 7200 * 10**9)
    _seed_family(store, "b", last_used_ns=wall["now"] - 100 * 10**9)
    _seed_family(store, "pinned", last_used_ns=wall["now"] - 7200 * 10**9)
    store._pins["pinned"] = 1
    assert store.sweep_expired() == 1
    assert set(store._entries) == {"b", "pinned"}
    assert store.next_expiry_delay_ms() == 3500 * 1000


def test_ssd_sweep_rewrites_the_manifest(tmp_path: Path):
    """sweep_expired's ``if self.mode == "ssd": self._write_manifest()`` branch: an expired
    SSD family must be dropped from park.json, not just from self._entries in memory, or a
    restart's _load_or_scan would resurrect the evicted file from a stale index."""
    wall = {"now": 1_700_000_000 * 10**9}
    store = ParkStore(
        mode="ssd", page_size=4, kv_pool=_qsa_pool(), state_pool=_state_pool(),
        fingerprint="fp", min_tokens=4, ram_budget_bytes=0, ssd_dir=tmp_path,
        disk_budget_bytes=1 << 30, pinned_window_bytes=4096, idle_ms=0, ttl_s=3600,
    )
    try:
        store._wall_ns = lambda: wall["now"]
        payload_path = tmp_path / (("a" * 32) + ".park")
        payload_path.touch()
        _seed_family(store, "a" * 32, last_used_ns=wall["now"] - 7200 * 10**9, path=payload_path)

        # Write the manifest once up front so this test proves the sweep REWRITES it (an
        # already-empty/never-written file would trivially satisfy an "entries == []" check).
        store._write_manifest()
        manifest_path = tmp_path / "park.json"
        before = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert [row["key"] for row in before["entries"]] == ["a" * 32]

        assert store.sweep_expired() == 1
        assert store._entries == {}
        assert not payload_path.exists()

        after = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert after["entries"] == []
    finally:
        store.close()
