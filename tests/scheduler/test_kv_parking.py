from __future__ import annotations

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


def test_async_park_keeps_sources_owned_until_copy_completion(tmp_path: Path, monkeypatch):
    cm, kv, state = _manager(tmp_path)
    _tokens, pages, _indices, slot, *_ = _install_prefix(cm, kv, state)

    class Pending:
        def __init__(self):
            self.copy_done = threading.Event()

        def wait_copied(self):
            self.copy_done.wait()

    pending = Pending()
    monkeypatch.setattr(cm.park_store, "offer", lambda *_args: pending, raising=False)
    monkeypatch.setattr(
        cm.park_store,
        "save",
        lambda *_args: (_ for _ in ()).throw(AssertionError("scheduler used sync save")),
    )

    assert cm.park_idle(now_ns=10**30) == 1
    assert set(pages.tolist()).isdisjoint(set(cm.free_slots.tolist()))
    assert slot not in state._free_slots

    pending.copy_done.set()
    cm.drain_pending_parks()

    assert set(pages.tolist()).issubset(set(cm.free_slots.tolist()))
    assert slot in state._free_slots


def test_idle_park_frees_only_after_the_copy_and_restore_is_byte_identical(tmp_path: Path):
    cm, kv, state = _manager(tmp_path)
    tokens, _pages, _indices, slot, expected_kv, expected_state = _install_prefix(cm, kv, state)
    free_pages_before = len(cm.free_slots)
    free_states_before = state.num_free_slots

    cm.park_idle(now_ns=10**30)
    assert len(cm.free_slots) == free_pages_before
    assert state.num_free_slots == free_states_before
    cm.drain_pending_parks(wait=True)
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


def test_non_fitting_longest_entry_falls_back_to_a_fitting_shorter_prefix(
    tmp_path: Path,
):
    cm, kv, state = _manager(tmp_path, num_pages=3)
    source_slot = state.alloc(1)[0]
    fill_pages = torch.tensor([0, 4, 8], dtype=torch.int32)
    for page in range(3):
        for view in kv.page_byte_views(page):
            view.fill_(page + 1)
    for view in state.slot_byte_views(source_slot):
        view.fill_(7)
    short = torch.arange(8, dtype=torch.int32)
    long = torch.arange(12, dtype=torch.int32)
    assert cm.park_store.save(short, fill_pages[:2], source_slot)
    assert cm.park_store.save(long, fill_pages, source_slot)

    matched = cm.match_req(_pending(long))

    assert matched.cuda_handle.cached_len == 8
    assert cm.park_store.status()["hits"] == 1


def test_failed_longer_restore_keeps_the_existing_live_match_valid(
    tmp_path: Path, monkeypatch
):
    cm, kv, state = _manager(tmp_path, num_pages=4)
    parked_slot = state.alloc(1)[0]
    for page in range(3):
        for view in kv.page_byte_views(page):
            view.fill_(page + 11)
    for view in state.slot_byte_views(parked_slot):
        view.fill_(13)
    long = torch.arange(12, dtype=torch.int32)
    assert cm.park_store.save(
        long, torch.tensor([0, 4, 8], dtype=torch.int32), parked_slot
    )
    live = long[:8]
    _install_prefix(cm, kv, state, tokens=live)

    monkeypatch.setattr(
        cm.park_store,
        "restore",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read failed")),
    )
    matched = cm.match_req(_pending(long))

    assert matched.cuda_handle.cached_len == 8
    rematched = cm.prefix_cache.match_prefix(live)
    assert rematched.cached_len == 8
    assert rematched.node is matched.cuda_handle.node
    assert torch.equal(rematched.kv_indices, matched.cuda_handle.kv_indices)
    assert rematched.mamba_value == matched.mamba_value


def test_page_and_state_ownership_stays_out_of_free_lists_during_save(tmp_path: Path):
    cm, kv, state = _manager(tmp_path)
    tokens, pages, _indices, slot, *_ = _install_prefix(cm, kv, state)
    real_save = cm.park_store.save
    observations = []

    def checked_save(token_ids, page_bases, state_slot, **kwargs):
        observations.append(
            (
                set(page_bases.tolist()).isdisjoint(set(cm.free_slots.tolist())),
                state_slot not in state._free_slots,
            )
        )
        return real_save(token_ids, page_bases, state_slot, **kwargs)

    cm.park_store.save = checked_save
    cm.park_idle(now_ns=10**30)
    cm.drain_pending_parks(wait=True)
    assert observations == [(True, True)]
    assert set(pages.tolist()).issubset(set(cm.free_slots.tolist()))
    assert slot in state._free_slots
    assert cm.park_store.lookup(tokens) is not None


def test_idle_threshold_exposes_a_receive_wakeup_deadline(tmp_path: Path):
    cm, kv, state = _manager(tmp_path, idle_ms=1000)
    _install_prefix(cm, kv, state)
    node = cm.prefix_cache._leaves()[0]

    assert cm.next_park_delay_ms(now_ns=node.timestamp + 999_000_000) == 1
    assert cm.next_park_delay_ms(now_ns=node.timestamp + 500_000_000) == 500


def test_blocking_receive_rechecks_idle_work_after_timeout():
    from freetoken.scheduler.io import SchedulerIOMixin

    events = []

    class Queue:
        def __init__(self):
            self.polls = [False, True]

        def poll(self, timeout_ms):
            events.append(("poll", timeout_ms))
            return self.polls.pop(0)

        def get(self):
            events.append(("get", None))
            return "message"

        def empty(self):
            return True

    io = object.__new__(SchedulerIOMixin)
    io._recv_from_tokenizer = Queue()
    io.run_when_idle = lambda: events.append(("idle", None))
    io.idle_poll_timeout_ms = lambda: 7

    assert io._recv_msg_single_rank(blocking=True) == ["message"]
    assert events == [
        ("idle", None),
        ("poll", 7),
        ("idle", None),
        ("poll", 7),
        ("get", None),
    ]


def test_idle_threshold_waits_until_the_leaf_is_old_enough(tmp_path: Path):
    cm, kv, state = _manager(tmp_path, idle_ms=1000)
    _install_prefix(cm, kv, state)
    node = cm.prefix_cache._leaves()[0]
    free_before = len(cm.free_slots)

    cm.park_idle(now_ns=node.timestamp + 999_000_000)
    assert len(cm.free_slots) == free_before
    assert cm.park_store.status()["parked_count"] == 0
    cm.park_idle(now_ns=node.timestamp + 1_000_000_000)
    assert len(cm.free_slots) == free_before
    cm.drain_pending_parks(wait=True)
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

    cm.prepare_rebuild()
    cm.rebuild(cm.num_pages, new_table)
    assert cm.park_store.lookup(tokens) is not None
    assert state.num_free_slots == state.num_slots - 1


def test_scheduler_prepares_parking_before_engine_reallocates_cache(
    monkeypatch,
):
    from freetoken.scheduler.scheduler import Scheduler

    events = []
    scheduler = object.__new__(Scheduler)
    scheduler.device = torch.device("cpu")
    scheduler.prefill_manager = SimpleNamespace(runnable=False)
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.config = SimpleNamespace(
        tp_info=SimpleNamespace(size=1),
        max_extend_tokens=128,
    )
    scheduler.cache_manager = SimpleNamespace(
        park_store=object(),
        prepare_rebuild=lambda: events.append("park"),
        rebuild=lambda *_args: events.append("manager"),
        check_integrity=lambda: None,
        prefill_chunk_budget=None,
    )
    scheduler.engine = SimpleNamespace(
        num_pages=4,
        page_table=torch.zeros(1, 16, dtype=torch.int32),
        rebuild_runtime_cache=lambda **_kwargs: events.append("engine"),
    )
    scheduler.table_manager = SimpleNamespace(
        token_pool=object(),
        rebuild=lambda _table: events.append("table"),
    )
    scheduler.token_pool = scheduler.table_manager.token_pool
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)

    scheduler.rebuild_cache(num_pages=4)

    assert events[:3] == ["park", "engine", "manager"]


def test_scheduler_shutdown_drains_parking_before_engine_teardown(monkeypatch):
    from freetoken.scheduler.scheduler import Scheduler

    events = []
    scheduler = object.__new__(Scheduler)
    scheduler.device = torch.device("cpu")
    scheduler.cache_manager = SimpleNamespace(close=lambda: events.append("park-close"))
    scheduler.sync_all_ranks = lambda: events.append("ranks")
    scheduler.engine = SimpleNamespace(shutdown=lambda: events.append("engine"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: events.append("cuda"))

    scheduler.shutdown()

    assert events == ["park-close", "cuda", "ranks", "engine"]


def test_offline_scheduler_ignores_parking_status_messages():
    from freetoken.llm.llm import LLM
    from freetoken.message import CacheParkStatusMsg

    llm = object.__new__(LLM)
    llm.status_map = {}
    llm.offline_send_result([CacheParkStatusMsg(status={"mode": "ram"})])


def test_allocation_pressure_parks_the_lru_leaf_before_reusing_its_pages(tmp_path: Path):
    cm, kv, state = _manager(tmp_path, num_pages=3)
    tokens, pages, *_ = _install_prefix(cm, kv, state)
    only_free = cm.free_slots.clone()

    allocated = cm._allocate(2)
    assert len(allocated) == 2
    assert int(allocated[0]) == int(only_free[0])
    assert set(allocated.tolist()).intersection(set(pages.tolist()))
    assert cm.park_store.lookup(tokens) is not None
