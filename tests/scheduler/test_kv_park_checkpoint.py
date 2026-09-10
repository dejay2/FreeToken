"""Branch-aware SSD parking at the scheduler seam (2026-09-10).

The final-prefill commit (``cache_req(finished=False)``) and an early finish that still
carries a frozen prefill snapshot must persist that exact checkpoint as its own SSD segment
before state-slot pressure can tombstone it, from the canonical tree pages and the frozen
tree slot, without detaching or freeing anything. RAM mode, parking-off, private and aborted
requests keep the previous behavior. CPU only: a pure-Python key compare stands in for the
native radix extension when it is absent (the devbox), so nothing here is skipped.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.park_store import ParkStore, rolling_page_keys
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager

PAGE = 4


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


@pytest.fixture(autouse=True)
def _cpu_key_compare(monkeypatch):
    """The radix walk needs ``fast_compare_key`` from the native extension; without it
    (no ``tvm_ffi`` on the devbox) use the same first-difference semantics in Python."""
    import freetoken.kernel as kernel

    probe = torch.tensor([1, 2], dtype=torch.int32)
    try:
        if kernel.fast_compare_key(probe, probe) == 2:
            return
    except Exception:
        pass

    def compare(x: torch.Tensor, y: torch.Tensor) -> int:
        n = min(len(x), len(y))
        if n == 0:
            return 0
        diff = (x[:n] != y[:n]).nonzero()
        return int(diff[0]) if len(diff) else n

    monkeypatch.setattr(kernel, "fast_compare_key", compare)


def _pools(num_pages: int = 16, num_slots: int = 12):
    kv = QSAKVCache(
        num_kv_heads=1,
        num_layers=2,
        head_dim=4,
        num_pages=num_pages,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_head_dim=3,
        num_index_layers=2,
        index_ratio=2,
        num_req_slots=4,
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
        name="ple_sibling", shape=(2, 3), layer_ids=(0,), dtype=torch.int32, fill_value=-1
    )
    state = LinearStatePool(
        group,
        num_slots=num_slots,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        tp_size=1,
        slot_states=(sibling,),
    )
    return kv, state


def _store(tmp_path: Path, kv, state, *, mode="ssd", budget=1 << 22):
    return ParkStore(
        mode=mode,
        page_size=PAGE,
        kv_pool=kv,
        state_pool=state,
        fingerprint="checkpoint-test",
        min_tokens=8,
        idle_ms=0,
        ram_budget_bytes=budget,
        ssd_dir=tmp_path,
        disk_budget_bytes=budget,
        pinned_window_bytes=4096,
    )


def _manager(tmp_path: Path, *, mode="ssd", num_pages=16, num_slots=12, store=None):
    kv, state = _pools(num_pages, num_slots)
    table = torch.zeros(4, num_pages * PAGE, dtype=torch.int32)
    if store is None and mode is not None:
        store = _store(tmp_path, kv, state, mode=mode)
    cm = CacheManager(
        num_pages, PAGE, table, "hybrid_radix",
        linear_state_pool=state, swa_pool=kv, park_store=store,
    )
    return cm, kv, state, table


def _fill(views, seed: int) -> None:
    for index, view in enumerate(views):
        raw = view.view(torch.uint8).reshape(-1)
        raw.copy_(((torch.arange(raw.numel(), dtype=torch.int64) * 37 + seed * 101 + index * 13) % 251).to(torch.uint8))


def _raw(views) -> list[torch.Tensor]:
    return [view.view(torch.uint8).reshape(-1).clone() for view in views]


def _page_views(kv, bases: torch.Tensor):
    return [view for base in bases.tolist() for view in kv.page_byte_views(base // PAGE)]


def _pending(ids: torch.Tensor):
    return SimpleNamespace(input_ids=ids, input_len=len(ids), mm_embeds=None, cache_private=False)


def _admit(cm, kv, state, table, ids: torch.Tensor, *, table_idx: int, seed: int, private=False):
    """Admit a request whose final prefill has run: pages allocated for the whole prompt and
    filled, live + ping-pong slots held, the frozen ping-pong slot filled at the last
    page-aligned boundary L (the x64 track), cached_len == len(ids)."""
    mr = cm.match_req(_pending(ids))
    req = Req(
        input_ids=ids.clone(), table_idx=table_idx, cached_len=mr.cuda_handle.cached_len,
        output_len=8, uid=table_idx, sampling_params=SamplingParams(max_tokens=8),
        cache_handle=mr.cuda_handle,
    )
    req.cache_private = private
    cm.lock(mr.cuda_handle)
    req.device_len = len(ids)
    cm.allocate_paged([req])
    req.cached_len = len(ids)
    live_len = mr.cuda_handle.cached_len
    if live_len:
        table[table_idx, :live_len] = mr.cuda_handle.get_matched_indices()
    new_bases = table[table_idx, live_len:len(ids):PAGE]
    _fill(_page_views(kv, new_bases), seed)
    req.linear_slot_idx = state.alloc(1)[0]
    req.mamba_ping_pong = tuple(state.alloc(2))
    req.mamba_next_track_idx = 1                       # frozen snapshot = ping_pong[0]
    L = (len(ids) // PAGE) * PAGE
    req.mamba_last_track_seqlen = L
    _fill(state.slot_byte_views(req.mamba_ping_pong[0]), seed + 1)
    _fill(state.slot_byte_views(req.linear_slot_idx), seed + 2)
    return req, L


def _entry(cm, ids: torch.Tensor):
    return cm.park_store._entries.get(rolling_page_keys(ids, PAGE, cm.park_store.fingerprint)[-1])


def _restore_equals(cm, kv, state, entry, expected_kv, expected_state) -> None:
    pages = cm._allocate(entry.token_count // PAGE)
    slot = state.alloc(1)[0]
    try:
        cm.park_store.restore(entry, pages, slot)
        got_kv = _raw(_page_views(kv, pages))
        got_state = _raw(state.slot_byte_views(slot))
        assert len(got_kv) == len(expected_kv)
        assert all(torch.equal(a, b) for a, b in zip(got_kv, expected_kv, strict=True))
        assert all(torch.equal(a, b) for a, b in zip(got_state, expected_state, strict=True))
    finally:
        state.free(slot)
        cm._free(cm._page_to_token(pages))


def test_final_prefill_checkpoint_is_saved_before_slot_pressure_and_survives_eviction(tmp_path):
    # 4,107 tokens: the SSD restore margin in _lookup_parked is 4,096 tokens, so a manager-level
    # restore hit needs a checkpoint at least that long (L = 4,104 here).
    cm, kv, state, table = _manager(tmp_path, num_pages=2200)
    store = cm.park_store
    ids = torch.arange(100, 100 + 4107, dtype=torch.int32)
    req, L = _admit(cm, kv, state, table, ids, table_idx=0, seed=1)
    frozen = req.mamba_ping_pong[0]
    expected_kv = _raw(_page_views(kv, table[0, :L:PAGE]))
    expected_state = _raw(state.slot_byte_views(frozen))
    free_slots_before = state.num_free_slots
    free_pages_before = len(cm.free_slots)

    cm.cache_req(req, finished=False)

    # The checkpoint at L is on disk now, from the canonical pages and the frozen tree slot.
    entry = _entry(cm, ids[:L])
    assert entry is not None and entry.token_count == L and entry.parent_key is None
    assert store.status()["parked_count"] == 1 and store.status()["disabled"] is False
    node = cm.prefix_cache.match_prefix(ids[:L]).node
    assert node.park_finished is True and node.mamba_value == frozen
    # Non-detaching: the request still owns a locked handle at L, its pages are untouched,
    # the frozen slot moved to the tree and exactly one replacement slot was allocated.
    assert req.cache_handle.cached_len == L and node.ref_count == 1
    assert state.num_free_slots == free_slots_before - 1
    assert req.mamba_ping_pong[0] != frozen
    assert len(cm.free_slots) == free_pages_before
    _restore_equals(cm, kv, state, entry, expected_kv, expected_state)

    # Finish (L + 4 tokens, aligned): the finish state is a distinct second segment linked to L.
    finish_len = L + 4
    req.cached_len = finish_len
    req.device_len = finish_len
    cm.allocate_paged([req])
    _fill(_page_views(kv, table[0, L:finish_len:PAGE]), 5)
    ids12 = torch.arange(100, 100 + finish_len, dtype=torch.int32)
    req.input_ids = ids12
    cm.cache_req(req, finished=True)
    cm.check_integrity()
    finish = cm.prefix_cache.match_prefix(ids12)
    assert finish.cached_len == finish_len and finish.node.park_finished is True

    # State pressure tombstones the internal checkpoint on the card (make it the LRU snapshot
    # explicitly: the finish walk stamped both nodes alike); the SSD copy answers.
    node.timestamp = 0
    evicted = cm.prefix_cache.evict_mamba(1)
    assert node.mamba_value is None and evicted.mamba_slots == [frozen]
    state.free(evicted.mamba_slots)
    probe = torch.cat([ids[:L], torch.tensor([777, 778, 779, 780, 781], dtype=torch.int32)])
    hit = store.lookup(probe)
    assert hit is entry
    _restore_equals(cm, kv, state, entry, expected_kv, expected_state)

    # Drain every leaf: the finish leaf is parked (one write) and detached, which exposes the
    # tombstoned checkpoint as a KV-only leaf that ordinary cleanup removes; nothing is
    # rewritten, the tree ends empty and every page/slot returns to its pool.
    parked_bytes = store.status()["parked_bytes"]
    assert cm.park_idle(now_ns=10**30) == 1
    cm.drain_pending_parks(wait=True)
    store.flush()
    assert cm.prefix_cache.full_evictable_size == 0 and cm.prefix_cache.mamba_evictable_size == 0
    assert cm.prefix_cache.root.is_leaf()
    finish_entry = _entry(cm, ids12)
    assert finish_entry is not None and finish_entry.parent_key == entry.key
    assert finish_entry.parent_token_count == L
    assert store.status()["parked_count"] == 2
    assert store.status()["parked_bytes"] == parked_bytes + finish_entry.total_bytes
    assert state.num_free_slots == state.num_slots - 1
    assert len(cm.free_slots) == cm.num_pages
    cm.check_integrity()
    assert L >= 4096, "the manager-level hit below needs the SSD restore margin"
    # The next real turn restores the checkpoint through match_req: a hit, not a cold prefill.
    hits = store.status()["hits"]
    matched = cm.match_req(_pending(probe))
    assert matched.cuda_handle.cached_len == L and matched.mamba_value is not None
    assert store.status()["hits"] == hits + 1
    assert all(
        torch.equal(a, b)
        for a, b in zip(_raw(state.slot_byte_views(matched.mamba_value)), expected_state, strict=True)
    )
    cm.close()


def test_a_saved_checkpoint_leaves_the_live_tree_without_a_rewrite(tmp_path, monkeypatch):
    """Once its children are gone, a saved prompt checkpoint is an ordinary eligible leaf:
    the duplicate key makes its park a no-write and ordinary detach releases it, so a live
    internal checkpoint never suppresses the SSD hit forever."""
    cm, kv, state, table = _manager(tmp_path)
    store = cm.park_store
    ids = torch.arange(200, 209, dtype=torch.int32)        # 9 tokens -> L = 8
    req, L = _admit(cm, kv, state, table, ids, table_idx=0, seed=3)
    cm.cache_req(req, finished=False)
    entry = _entry(cm, ids[:L])
    assert entry is not None
    # Unaligned finish (9 tokens): no finish donate, the checkpoint stays the deepest node.
    cm.cache_req(req, finished=True)
    cm.check_integrity()
    writes = []
    real_write = store._write_ssd
    monkeypatch.setattr(store, "_write_ssd", lambda *a, **k: writes.append(1) or real_write(*a, **k))
    candidates = cm._park_candidates()
    assert len(candidates) == 1 and candidates[0].node.park_finished is True
    assert cm.park_idle(now_ns=10**30) == 1
    cm.drain_pending_parks(wait=True)
    store.flush()
    assert writes == [], "a duplicate key must not rewrite the segment"
    assert cm.prefix_cache.root.is_leaf()
    assert store.status()["parked_count"] == 1 and _entry(cm, ids[:L]) is entry
    assert state.num_free_slots == state.num_slots - 1
    cm.check_integrity()
    cm.close()


@pytest.mark.parametrize("path", ["ram", "off", "private", "aborted"])
def test_no_eager_checkpoint_on_ram_off_private_or_aborted_paths(tmp_path, path):
    mode = {"ram": "ram", "off": None}.get(path, "ssd")
    cm, kv, state, table = _manager(tmp_path, mode=mode)
    ids = torch.arange(300, 311, dtype=torch.int32)
    req, L = _admit(cm, kv, state, table, ids, table_idx=0, seed=7, private=(path == "private"))
    if path == "aborted":
        req.aborted = True
        cm.cache_req(req, finished=True)            # the abort drain frees at finish
    else:
        cm.cache_req(req, finished=False)
    if cm.park_store is not None:
        assert cm.park_store.status()["parked_count"] == 0
    if path == "ram":
        node = cm.prefix_cache.match_prefix(ids[:L]).node
        assert node.park_finished is False, "RAM mode keeps the intermediate-snapshot rule"
    if path == "aborted":
        # The frozen snapshot went into the tree as before (78f118c's finish marker still
        # applies to it); the new eager save simply did not happen.
        assert cm.prefix_cache.match_prefix(ids[:L]).mamba_value is not None
        assert req.mamba_ping_pong is None and req.linear_slot_idx is None
    else:
        cm.cache_req(req, finished=True)
    cm.check_integrity()
    cm.close()


def test_intermediate_chunk_commit_never_reaches_the_store(tmp_path):
    """The scheduler drain skips ChunkedReqs (only the final prefill is committed), so an
    in-flight chunk with a tracked x64 boundary produces no checkpoint, saved or otherwise."""
    cm, kv, state, table = _manager(tmp_path)
    tm = TableManager(max_running_reqs=4, page_table=table)
    prompt = torch.arange(400, 416, dtype=torch.int32)
    chunk = ChunkedReq(
        input_ids=prompt[:8], table_idx=tm.allocate(), cached_len=0, output_len=4, uid=1,
        sampling_params=SamplingParams(max_tokens=4),
        cache_handle=cm.match_req(_pending(prompt[:8])).cuda_handle,
    )
    cm.lock(chunk.cache_handle)
    chunk.linear_slot_idx = state.alloc(1)[0]
    chunk.mamba_ping_pong = tuple(state.alloc(2))
    chunk.mamba_next_track_idx = 1
    chunk.device_len = 8
    cm.allocate_paged([chunk])
    chunk.cached_len = 8
    chunk.mamba_last_track_seqlen = 8
    stub = SimpleNamespace(
        cache_manager=cm, table_manager=tm, finished_reqs=set(),
        decode_manager=SimpleNamespace(remove_req=lambda _req: None),
        _spec_record_plain=lambda _batch: None,
        _flush_abort_acks=lambda: None,
        _ship_replies=lambda *a, **k: None,
        _pending_abort_acks=set(),
        _emit_step_tokens=lambda req, tokens: (_ for _ in ()).throw(AssertionError("chunks never emit")),
    )
    stub._free_req_resources = lambda req: Scheduler._free_req_resources(stub, req)
    batch = Batch(reqs=[chunk], phase="prefill")
    last_data = (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([42], dtype=torch.int32), SimpleNamespace(synchronize=lambda: None)),
    )
    Scheduler._process_last_data(stub, last_data)
    assert cm.park_store.status()["parked_count"] == 0
    assert cm.prefix_cache.root.is_leaf()
    assert chunk.mamba_last_track_seqlen == 8 and chunk.table_idx != -1
    cm.close()


def test_eager_save_reads_canonical_pages_not_the_freed_duplicates(tmp_path):
    """Two cold prefills of the same prompt: the second commit dedups against the first's
    tree node (mamba_exist), re-points its row and frees its own duplicate pages. The eager
    save must read the tree's canonical pages and slot, never the freed duplicates or the
    request's own frozen slot; the tree slot is what a restore will hand out."""
    cm, kv, state, table = _manager(tmp_path, num_pages=24)
    store = cm.park_store
    ids = torch.arange(500, 511, dtype=torch.int32)
    first, L = _admit(cm, kv, state, table, ids, table_idx=0, seed=11)
    # Disable the store around the first commit so only the tree learns the prefix.
    store._disabled = True
    cm.cache_req(first, finished=False)
    store._disabled = False
    assert store.status()["parked_count"] == 0
    canonical = cm.prefix_cache.match_prefix(ids[:L])
    canonical_kv = _raw(_page_views(kv, canonical.kv_indices[::PAGE]))
    canonical_state = _raw(state.slot_byte_views(canonical.mamba_value))

    # The second request prefilled the same prompt cold into its own pages with different
    # bytes (a garbage duplicate is the strongest check) before its commit dedups.
    second = Req(
        input_ids=ids.clone(), table_idx=1, cached_len=0, output_len=8, uid=1,
        sampling_params=SamplingParams(max_tokens=8),
        cache_handle=cm.match_req(_pending(torch.tensor([9999, 9998], dtype=torch.int32))).cuda_handle,
    )
    assert second.cache_handle.cached_len == 0
    cm.lock(second.cache_handle)
    second.device_len = len(ids)
    cm.allocate_paged([second])
    second.cached_len = len(ids)
    dup_bases = table[1, :L:PAGE].clone()
    _fill(_page_views(kv, dup_bases), 99)
    second.linear_slot_idx = state.alloc(1)[0]
    second.mamba_ping_pong = tuple(state.alloc(2))
    second.mamba_next_track_idx = 1
    second.mamba_last_track_seqlen = L
    _fill(state.slot_byte_views(second.mamba_ping_pong[0]), 98)
    free_before = state.num_free_slots
    pages_before = len(cm.free_slots)

    cm.cache_req(second, finished=False)

    entry = _entry(cm, ids[:L])
    assert entry is not None
    assert torch.equal(table[1, :L:PAGE], canonical.kv_indices[::PAGE]), "row re-pointed at canonical"
    assert set(dup_bases.tolist()) <= set(cm.free_slots.tolist()), "duplicates were freed"
    assert len(cm.free_slots) == pages_before + L // PAGE
    assert state.num_free_slots == free_before, "mamba_exist: no replacement slot, nothing lost"
    assert second.cache_handle.node is canonical.node and canonical.node.ref_count == 2
    _restore_equals(cm, kv, state, entry, canonical_kv, canonical_state)
    cm.cache_req(first, finished=True)
    cm.cache_req(second, finished=True)
    cm.check_integrity()
    cm.close()


def test_early_eos_checkpoint_is_saved_once_and_unaligned_finish_still_falls_back(tmp_path):
    cm, kv, state, table = _manager(tmp_path)
    store = cm.park_store
    ids = torch.arange(600, 611, dtype=torch.int32)        # 11 tokens: unaligned finish
    req, L = _admit(cm, kv, state, table, ids, table_idx=0, seed=13)
    frozen = req.mamba_ping_pong[0]
    expected_kv = _raw(_page_views(kv, table[0, :L:PAGE]))
    expected_state = _raw(state.slot_byte_views(frozen))
    slots_before = state.num_free_slots

    cm.cache_req(req, finished=True)                        # EOS right after the final prefill

    entry = _entry(cm, ids[:L])
    assert entry is not None and store.status()["parked_count"] == 1
    _restore_equals(cm, kv, state, entry, expected_kv, expected_state)
    node = cm.prefix_cache.match_prefix(ids[:L]).node
    assert node.park_finished is True and node.mamba_value == frozen
    assert node.ref_count == 0 and node.mamba_ref_count == 0, "the temporary lock was released"
    # The frozen slot is tree-owned; the live slot and the other ping-pong slot came back.
    assert state.num_free_slots == slots_before + 2
    assert req.mamba_ping_pong is None and req.linear_slot_idx is None
    cm.check_integrity()
    # The deepest snapshot is the checkpoint: the unaligned-finish fallback still parks it.
    candidates = cm._park_candidates()
    assert len(candidates) == 1 and candidates[0].mamba_slot == frozen
    # A second identical request saves nothing new (duplicate key) and frees nothing twice.
    again, _ = _admit(cm, kv, state, table, ids, table_idx=1, seed=13)
    before = store.status()["parked_bytes"]
    cm.cache_req(again, finished=True)
    assert store.status()["parked_count"] == 1 and store.status()["parked_bytes"] == before
    cm.check_integrity()
    cm.close()


def test_main_side_and_next_turn_park_incrementally_through_the_manager(tmp_path):
    """Long main -> side request -> real next turn, driven through match_req/cache_req with
    real manager marking, idle parking between turns (the box is idle between agent
    requests), pressure from another request, and SSD restore hits for the later turns."""
    # Prompts exceed the 4,096-token SSD restore margin so match_req can hit.
    cm, kv, state, table = _manager(tmp_path, num_pages=3400, num_slots=16)
    store = cm.park_store

    def turn(ids: torch.Tensor, table_idx: int, seed: int):
        hits = store.status()["hits"]
        req, L = _admit(cm, kv, state, table, ids, table_idx=table_idx, seed=seed)
        live = req.cache_handle.cached_len
        cm.cache_req(req, finished=False)
        # one decoded token, then finish
        req.input_ids = torch.cat([ids, torch.tensor([seed * 1000], dtype=torch.int32)])
        req.cached_len = len(req.input_ids)
        req.device_len = req.cached_len
        cm.allocate_paged([req])
        _fill(_page_views(kv, table[table_idx, len(ids) : req.cached_len : PAGE]), seed + 5)
        cm.cache_req(req, finished=True)
        cm.check_integrity()
        # Idle between turns: every completed leaf is parked and the tree drains.
        while cm.park_idle(now_ns=10**30):
            cm.drain_pending_parks(wait=True)
            store.flush()
        assert cm.prefix_cache.root.is_leaf()
        cm.check_integrity()
        return req, L, live, store.status()["hits"] - hits

    main = torch.arange(1000, 1000 + 4111, dtype=torch.int32)               # 4,111 -> C 4,108, finish 4,112
    main_req, L_main, live, hits = turn(main, 0, 21)
    assert live == 0 and hits == 0 and L_main == 4108
    c = _entry(cm, main[:L_main])
    mf = _entry(cm, main_req.input_ids)
    assert c is not None and c.parent_key is None
    assert mf is not None and mf.parent_key == c.key and mf.parent_token_count == 4108
    parked_after_main = store.status()["parked_bytes"]

    side = torch.cat([main_req.input_ids, torch.arange(2000, 2011, dtype=torch.int32)])   # 4,123 -> C_s 4,120
    side_req, L_side, live, hits = turn(side, 1, 31)
    assert live == 4112 and hits == 1, "the side request restored the main finish from SSD"
    c_side = _entry(cm, side[:L_side])
    assert c_side is not None and c_side.parent_key == mf.key and c_side.parent_token_count == 4112
    assert c_side.kv_bytes == 8 * kv.unit_bytes()[0]
    side_finish = _entry(cm, side_req.input_ids)
    assert side_finish is not None and side_finish.parent_key == c_side.key
    parked_after_side = store.status()["parked_bytes"]
    # Exactly the two delta segments were added (byte ratios are proved at the store level
    # with a heavier KV pool; here the 4-byte token list would dominate the 38-byte KV).
    assert parked_after_side - parked_after_main == c_side.total_bytes + side_finish.total_bytes

    # The next real turn shares the side prompt only up to 4,118 (inside its last pages),
    # so it must restore the main finish (4,112), not a slice of the side save.
    nxt = torch.cat([side[:4118], torch.arange(3000, 3013, dtype=torch.int32)])          # 4,131 -> C_n 4,128
    cm.ensure_mamba_slots(state.num_free_slots + 2)     # unrelated pressure: nothing to evict
    next_req, L_next, live, hits = turn(nxt, 2, 41)
    assert live == 4112 and hits == 1 and L_next == 4128
    c_next = _entry(cm, nxt[:L_next])
    assert c_next is not None and c_next.parent_key == mf.key and c_next.parent_token_count == 4112
    assert c_next.kv_bytes == (L_next - 4112) * kv.unit_bytes()[0]
    next_finish = _entry(cm, next_req.input_ids)
    assert next_finish is not None and next_finish.parent_key == c_next.key
    assert store.status()["parked_bytes"] - parked_after_side == c_next.total_bytes + next_finish.total_bytes
    for entry in store._entries.values():
        chain = store._chain(entry)
        assert chain is not None and chain[0] is c
    assert state.num_free_slots == state.num_slots - 1
    assert len(cm.free_slots) == cm.num_pages

    # And the turn after that hits the next turn's own checkpoint.
    hits = store.status()["hits"]
    probe = torch.cat([nxt[:L_next], torch.tensor([4242, 4243, 4244, 4245, 4246], dtype=torch.int32)])
    matched = cm.match_req(_pending(probe))
    assert matched.cuda_handle.cached_len == L_next and store.status()["hits"] == hits + 1
    cm.close()


def test_eager_checkpoint_survives_replacement_with_all_twelve_slots_occupied(tmp_path, monkeypatch):
    cm, kv, state, table = _manager(tmp_path, num_pages=24, num_slots=12)
    store = cm.park_store
    victim_ids = torch.arange(800, 811, dtype=torch.int32)
    victim, victim_L = _admit(cm, kv, state, table, victim_ids, table_idx=0, seed=20)
    cm.cache_req(victim, finished=True)
    victim_node = cm.prefix_cache.match_prefix(victim_ids[:victim_L]).node
    victim_slot = victim_node.mamba_value
    assert victim_slot is not None and victim_node.ref_count == 0

    ids = torch.arange(900, 911, dtype=torch.int32)
    req, L = _admit(cm, kv, state, table, ids, table_idx=1, seed=21)
    frozen = req.mamba_ping_pong[0]
    expected_kv = _raw(_page_views(kv, table[1, :L:PAGE]))
    expected_state = _raw(state.slot_byte_views(frozen))
    # Other active requests consume the remaining slots; only the victim can be reclaimed.
    held = state.alloc(state.num_free_slots)
    assert state.num_free_slots == 0
    real_ensure = cm.ensure_mamba_slots
    calls = []

    def ensure(n):
        calls.append(n)
        assert state.num_free_slots == 0
        entry = _entry(cm, ids[:L])
        assert entry is not None, "save must precede replacement-slot reclamation"
        match = cm.prefix_cache.match_prefix(ids[:L])
        assert match.mamba_value == frozen
        assert match.node.ref_count > 0 and match.node.mamba_ref_count > 0
        real_ensure(n)
        assert state.num_free_slots == n
        assert match.node.mamba_value == frozen, "pressure must not evict the donated state"

    monkeypatch.setattr(cm, "ensure_mamba_slots", ensure)
    cm.cache_req(req, finished=False)
    assert calls == [1]
    assert state.num_free_slots == 0
    assert req.mamba_ping_pong[0] == victim_slot
    assert cm.prefix_cache.match_prefix(victim_ids[:victim_L]).mamba_value is None
    assert all(torch.equal(a, b) for a, b in zip(
        _raw(_page_views(kv, table[1, :L:PAGE])), expected_kv, strict=True
    ))
    assert all(torch.equal(a, b) for a, b in zip(
        _raw(state.slot_byte_views(frozen)), expected_state, strict=True
    ))
    state.free(held)
    monkeypatch.setattr(cm, "ensure_mamba_slots", real_ensure)
    cm.cache_req(req, finished=True)
    cm.check_integrity()
    _restore_equals(cm, kv, state, _entry(cm, ids[:L]), expected_kv, expected_state)
    while cm.park_idle(now_ns=10**30):
        cm.drain_pending_parks(wait=True)
        store.flush()
    cm.check_integrity()
    assert state.num_free_slots == state.num_slots - 1
    assert len(cm.free_slots) == cm.num_pages
    cm.close()


def test_eager_save_failure_leaves_live_state_and_ordinary_finish_intact(tmp_path, monkeypatch):
    cm, kv, state, table = _manager(tmp_path)
    store = cm.park_store
    ids = torch.arange(700, 711, dtype=torch.int32)
    req, L = _admit(cm, kv, state, table, ids, table_idx=0, seed=17)
    calls = []

    def failing_save(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("disk gone")

    real_save = store.save
    monkeypatch.setattr(store, "save", failing_save)
    cm.cache_req(req, finished=False)
    assert calls == [1]
    assert "disk gone" in str(store.status()["last_error"])
    node = cm.prefix_cache.match_prefix(ids[:L]).node
    assert node.mamba_value is not None and node.ref_count == 1
    assert req.cache_handle.cached_len == L
    assert node.park_finished is True, "eligible for an ordinary retry through the leaf path"
    monkeypatch.setattr(store, "save", real_save)
    cm.cache_req(req, finished=True)
    cm.check_integrity()
    assert cm.park_idle(now_ns=10**30) == 1
    cm.drain_pending_parks(wait=True)
    store.flush()
    assert store.status()["parked_count"] == 1 and _entry(cm, ids[:L]) is not None
    cm.close()
