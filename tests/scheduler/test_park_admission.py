from __future__ import annotations

from threading import Event
from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.message import ErrorReplyMsg
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager, PrefillAdder
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq


class _StatePool:
    padding_slot = 0

    def __init__(self, num_slots: int = 16):
        self._free = list(range(1, num_slots))

    @property
    def num_free_slots(self) -> int:
        return len(self._free)

    def alloc(self, count: int = 1) -> list[int]:
        return [self._free.pop() for _ in range(count)]

    def free(self, slots) -> None:
        if isinstance(slots, int):
            slots = [slots]
        self._free.extend(int(slot) for slot in slots)


class _CountingParkStore:
    min_tokens = 4
    mode = "ram"
    page_size = 1
    idle_ms = 0

    def __init__(self):
        self.calls = 0

    def lookup(self, *_args, **_kwargs):
        self.calls += 1
        return None


class _OfferParkStore:
    min_tokens = 1
    mode = "ram"
    page_size = 1
    idle_ms = 0

    def __init__(self):
        self.offers = []

    def offer(self, input_ids, page_bases, state_slot):
        self.offers.append((input_ids.clone(), page_bases.clone(), state_slot))
        copied = Event()
        copied.set()
        return SimpleNamespace(copy_done=copied, wait_copied=copied.wait)


class _CandidateCache:
    def __init__(self, candidates):
        self.candidates = list(candidates)

    def park_candidates(self):
        return list(self.candidates)

    def detach_parked(self, candidate):
        self.candidates = [item for item in self.candidates if item is not candidate]
        return SimpleNamespace(
            kv_indices=candidate.kv_indices,
            mamba_slots=[candidate.mamba_slot],
            lock_node=None,
        )


def _hybrid_manager(*, num_pages: int, page_size: int = 1, park_store=None):
    state_pool = _StatePool()
    page_table = torch.zeros((4, num_pages * page_size), dtype=torch.int32)
    manager = CacheManager(
        num_pages=num_pages,
        page_size=page_size,
        page_table=page_table,
        type="hybrid_radix",
        linear_state_pool=state_pool,
        park_store=park_store,
    )
    return manager, state_pool, page_table


def test_parked_lookup_runs_once_per_generation():
    store = _CountingParkStore()
    cache_manager, _pool, _page_table = _hybrid_manager(num_pages=16, park_store=store)
    table_manager = SimpleNamespace(available_size=1)
    adder = PrefillAdder(
        token_budget=128,
        reserved_size=0,
        cache_manager=cache_manager,
        table_manager=table_manager,
    )
    pending = PendingReq(
        uid=1,
        input_ids=torch.arange(10, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=7),
    )

    results = [adder.try_add_one(pending) for _ in range(5)]

    assert results == [None] * 5
    assert store.calls == 1

    cache_manager._bump_park_generation()
    assert adder.try_add_one(pending) is None
    assert store.calls == 2


def test_never_fitting_request_is_rejected_with_reason():
    cache_manager, _pool, page_table = _hybrid_manager(num_pages=16)
    table_manager = TableManager(max_running_reqs=2, page_table=page_table)
    decode_manager = DecodeManager(page_size=1)
    prefill_manager = PrefillManager(cache_manager, table_manager, decode_manager)
    prefill_manager.pending_list = [
        PendingReq(
            uid=7,
            input_ids=torch.arange(12, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=8),
        )
    ]
    sent = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_manager = prefill_manager
    scheduler.decode_manager = decode_manager
    scheduler.prefill_budget = 128
    scheduler.send_result = sent.extend

    assert Scheduler._schedule_next_batch(scheduler) is None

    assert prefill_manager.pending_list == []
    assert len(sent) == 1
    assert isinstance(sent[0], ErrorReplyMsg)
    assert sent[0].uid == 7
    assert "KV admission gate" in sent[0].error


def test_two_turn_chat_parks_one_entry_per_turn():
    store = _OfferParkStore()
    cache_manager, state_pool, _page_table = _hybrid_manager(num_pages=16, park_store=store)
    candidates = []
    page = 0
    slot = 1
    for turn in (1, 2):
        # Three chunks per turn: only the completed turn's final leaf is eligible. The two
        # intermediate leaves model the snapshots that caused 42 entries during the 190k live run.
        for chunk in (1, 2):
            candidates.append(
                SimpleNamespace(
                    input_ids=torch.tensor([turn, chunk], dtype=torch.int32),
                    kv_indices=torch.tensor([page], dtype=torch.int32),
                    mamba_slot=slot,
                    timestamp=page,
                    node=SimpleNamespace(park_finished=False),
                )
            )
            page += 1
            slot += 1
        candidates.append(
            SimpleNamespace(
                input_ids=torch.tensor([turn, 3], dtype=torch.int32),
                kv_indices=torch.tensor([page], dtype=torch.int32),
                mamba_slot=slot,
                timestamp=page,
                node=SimpleNamespace(park_finished=True),
            )
        )
        page += 1
        slot += 1
    cache_manager.prefix_cache = _CandidateCache(candidates)

    parked = cache_manager.park_idle(now_ns=1_000_000)

    assert parked == 2
    assert len(store.offers) == 2
    assert [offer[0].tolist() for offer in store.offers] == [[1, 3], [2, 3]]
    assert state_pool.num_free_slots == 17
