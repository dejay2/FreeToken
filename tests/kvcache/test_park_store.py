from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


def _qsa_pool(num_pages: int = 6) -> QSAKVCache:
    return QSAKVCache(
        num_kv_heads=1,
        num_layers=2,
        head_dim=4,
        num_pages=num_pages,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_head_dim=3,
        num_index_layers=2,
        index_ratio=2,
        num_req_slots=2,
        layer_ids=(0, 1),
    )


def _state_pool(num_slots: int = 5) -> LinearStatePool:
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
        device=torch.device("cpu"),
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
        pinned_window_bytes=4096,
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
    assert json.loads((tmp_path / "park.json").read_text(encoding="utf-8"))["version"] == 2


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
    payload_offset = int(store._parse_header(entry.path)["payload_offset"])
    reads, _window_indices = _record_fake_cuda_restore_io(monkeypatch, store, entry)

    try:
        store.restore(
            entry,
            torch.tensor([8], dtype=torch.int32),
            target_slot,
            page_offset=1,
        )
        expected_reads = [
            (payload_offset + offset, min(64, entry.payload_bytes - offset))
            for offset in range(0, entry.payload_bytes, 64)
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
        handle.seek(int(meta["payload_offset"]) + 3)
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
