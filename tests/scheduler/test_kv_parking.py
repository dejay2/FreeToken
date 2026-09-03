from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.park_store import ParkStore
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.scheduler.cache import CacheManager


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _pools(num_pages: int = 8):
    kv = QSAKVCache(
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
    state = LinearStatePool(
        group,
        num_slots=8,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        tp_size=1,
        slot_states=(sibling,),
    )
    return kv, state


def _store(tmp_path: Path, kv, state, *, mode="ram", idle_ms=0):
    return ParkStore(
        mode=mode,
        page_size=4,
        kv_pool=kv,
        state_pool=state,
        fingerprint="scheduler-test",
        min_tokens=8,
        idle_ms=idle_ms,
        ram_budget_bytes=1 << 20,
        ssd_dir=tmp_path,
        disk_budget_bytes=1 << 20,
        pinned_window_bytes=4096,
    )


def _manager(tmp_path: Path, *, mode="ram", idle_ms=0, num_pages=8):
    kv, state = _pools(num_pages)
    table = torch.zeros(3, num_pages * 4, dtype=torch.int32)
    store = _store(tmp_path, kv, state, mode=mode, idle_ms=idle_ms)
    cm = CacheManager(
        num_pages,
        4,
        table,
        "hybrid_radix",
        linear_state_pool=state,
        swa_pool=kv,
        park_store=store,
    )
    return cm, kv, state


def _install_prefix(cm: CacheManager, kv, state, tokens=None):
    if tokens is None:
        tokens = torch.arange(8, dtype=torch.int32)
    page_bases = cm._allocate(2)
    token_indices = cm._page_to_token(page_bases)
    value = 1
    expected_kv = []
    for base in page_bases.tolist():
        for view in kv.page_byte_views(base // 4):
            payload = torch.arange(view.numel(), dtype=torch.int64).reshape(view.shape) + value
            view.copy_(payload.to(view.dtype))
            expected_kv.append(view.clone())
            value += view.numel() + 3
    slot = state.alloc(1)[0]
    expected_state = []
    for view in state.slot_byte_views(slot):
        payload = torch.arange(view.numel(), dtype=torch.int64).reshape(view.shape) + value
        view.copy_(payload.to(view.dtype))
        expected_state.append(view.clone())
        value += view.numel() + 5
    _, existed = cm.prefix_cache.insert(tokens, token_indices, slot)
    assert not existed
    return tokens, page_bases, token_indices, slot, expected_kv, expected_state


def _pending(tokens):
    ids = torch.cat([tokens, torch.tensor([999], dtype=torch.int32)])
    return SimpleNamespace(input_ids=ids, input_len=len(ids), mm_embeds=None, cache_private=False)


def _restored_views(kv, state, handle, slot):
    bases = handle.get_matched_indices()[::4]
    kv_views = [
        view.clone()
        for base in bases.tolist()
        for view in kv.page_byte_views(base // 4)
    ]
    state_views = [view.clone() for view in state.slot_byte_views(slot)]
    return kv_views, state_views


def test_parking_engine_defaults_are_off_and_bounded():
    from freetoken.engine.config import EngineConfig

    fields = EngineConfig.__dataclass_fields__
    assert fields["kv_park"].default == "off"
    assert fields["kv_park_idle_ms"].default == 0
    assert fields["kv_park_min_tokens"].default == 8192
    assert fields["kv_park_ram_gib"].default == 2.0
    assert fields["kv_park_ssd_dir"].default == "~/.cache/freetoken/kv-park"
    assert fields["kv_park_ssd_gib"].default == 32.0
    assert fields["kv_park_window_mib"].default == 256


def test_cache_status_exposes_latest_parking_metrics(monkeypatch):
    import freetoken.server.api_server as api

    parking = {
        "mode": "ssd",
        "parked_count": 3,
        "parked_bytes": 1234,
        "hits": 2,
        "misses": 1,
        "last_restore_ms": 17.5,
        "disabled": False,
    }
    state = SimpleNamespace(
        maintenance_state="serving", last_rebuild=None, parking_status=parking
    )
    monkeypatch.setattr(api, "get_global_state", lambda: state)
    monkeypatch.setattr(api, "cache_geometry", lambda _state: {})

    result = asyncio.run(api.cache_status())
    assert result["parking"] == parking


def test_off_mode_constructs_no_store_or_worker_thread():
    before = {thread.ident for thread in threading.enumerate() if thread.name.startswith("kv-park")}
    kv, state = _pools()
    cm = CacheManager(
        8,
        4,
        torch.zeros(3, 32, dtype=torch.int32),
        "hybrid_radix",
        linear_state_pool=state,
        swa_pool=kv,
    )
    after = {thread.ident for thread in threading.enumerate() if thread.name.startswith("kv-park")}
    assert cm.park_store is None
    assert after == before


def test_idle_park_frees_only_after_the_copy_and_restore_is_byte_identical(tmp_path: Path):
    cm, kv, state = _manager(tmp_path)
    tokens, _pages, _indices, slot, expected_kv, expected_state = _install_prefix(cm, kv, state)
    free_pages_before = len(cm.free_slots)
    free_states_before = state.num_free_slots

    cm.park_idle(now_ns=10**30)
    assert len(cm.free_slots) == free_pages_before + 2
    assert state.num_free_slots == free_states_before + 1
    assert cm.park_store.status()["parked_count"] == 1

    matched = cm.match_req(_pending(tokens))
    assert matched.cuda_handle.cached_len == 8
    assert matched.mamba_value is not None
    restored_kv, restored_state = _restored_views(
        kv, state, matched.cuda_handle, matched.mamba_value
    )
    assert all(torch.equal(a, b) for a, b in zip(restored_kv, expected_kv, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(restored_state, expected_state, strict=True))
    status = cm.park_status()
    assert status["hits"] == 1
    assert status["last_restore_ms"] >= 0


def test_live_gpu_match_is_not_replaced_by_the_same_parked_prefix(tmp_path: Path):
    cm, kv, state = _manager(tmp_path)
    tokens, pages, _indices, slot, *_ = _install_prefix(cm, kv, state)
    assert cm.park_store.save(tokens, pages, slot)
    free_pages = cm.free_slots.clone()
    free_states = state.num_free_slots

    matched = cm.match_req(_pending(tokens))
    assert matched.cuda_handle.cached_len == 8
    assert torch.equal(cm.free_slots, free_pages)
    assert state.num_free_slots == free_states
    assert cm.park_store.status()["hits"] == 0


def test_page_and_state_ownership_stays_out_of_free_lists_during_save(tmp_path: Path):
    cm, kv, state = _manager(tmp_path)
    tokens, pages, _indices, slot, *_ = _install_prefix(cm, kv, state)
    real_save = cm.park_store.save
    observations = []

    def checked_save(token_ids, page_bases, state_slot):
        observations.append(
            (
                set(page_bases.tolist()).isdisjoint(set(cm.free_slots.tolist())),
                state_slot not in state._free_slots,
            )
        )
        return real_save(token_ids, page_bases, state_slot)

    cm.park_store.save = checked_save
    cm.park_idle(now_ns=10**30)
    assert observations == [(True, True)]
    assert set(pages.tolist()).issubset(set(cm.free_slots.tolist()))
    assert slot in state._free_slots
    assert cm.park_store.lookup(tokens) is not None


def test_idle_threshold_waits_until_the_leaf_is_old_enough(tmp_path: Path):
    cm, kv, state = _manager(tmp_path, idle_ms=1000)
    _install_prefix(cm, kv, state)
    node = cm.prefix_cache._leaves()[0]
    free_before = len(cm.free_slots)

    cm.park_idle(now_ns=node.timestamp + 999_000_000)
    assert len(cm.free_slots) == free_before
    assert cm.park_store.status()["parked_count"] == 0
    cm.park_idle(now_ns=node.timestamp + 1_000_000_000)
    assert len(cm.free_slots) == free_before + 2
    assert cm.park_store.status()["parked_count"] == 1


def test_temporary_speculative_lease_blocks_idle_free_list_mutation(tmp_path: Path):
    cm, kv, state = _manager(tmp_path)
    _install_prefix(cm, kv, state)
    before = cm.free_slots.clone()
    with cm.temporary_page_lease(1):
        lease_remainder = cm.free_slots.clone()
        cm.park_idle(now_ns=10**30)
        assert torch.equal(cm.free_slots, lease_remainder)
    assert torch.equal(cm.free_slots, before)
    assert cm.park_store.status()["parked_count"] == 0


def test_rebuild_parks_unlocked_prefix_before_discarding_the_tree(tmp_path: Path):
    cm, kv, state = _manager(tmp_path)
    tokens, *_ = _install_prefix(cm, kv, state)
    new_table = torch.zeros_like(cm.page_table)

    cm.rebuild(cm.num_pages, new_table)
    assert cm.park_store.lookup(tokens) is not None
    assert state.num_free_slots == state.num_slots - 1


def test_allocation_pressure_parks_the_lru_leaf_before_reusing_its_pages(tmp_path: Path):
    cm, kv, state = _manager(tmp_path, num_pages=3)
    tokens, pages, *_ = _install_prefix(cm, kv, state)
    only_free = cm.free_slots.clone()

    allocated = cm._allocate(2)
    assert len(allocated) == 2
    assert int(allocated[0]) == int(only_free[0])
    assert set(allocated.tolist()).intersection(set(pages.tolist()))
    assert cm.park_store.lookup(tokens) is not None
