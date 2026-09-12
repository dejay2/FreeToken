"""Rule 2 of the dynamic KV pool spec: the admission probe walks the tree only (no park
lookup, no restore, no lock) and reproduces the POST-lock verdict of PrefillAdder in tokens."""
from __future__ import annotations

import torch

from freetoken.scheduler.cache import AdmissionProbe, CacheManager, admission_fits


class _StatePool:
    padding_slot = 0

    def __init__(self, num_slots=16):
        self._free = list(range(1, num_slots))

    @property
    def num_free_slots(self):
        return len(self._free)

    def alloc(self, n=1):
        return [self._free.pop() for _ in range(n)]

    def free(self, slots):
        self._free.extend([slots] if isinstance(slots, int) else [int(s) for s in slots])

    def reclaim_all_slots(self):
        pass


class _RestoringParkStore:
    """A park store that records every lookup; the probe must never touch it."""
    min_tokens = 4
    mode = "ram"
    page_size = 1
    idle_ms = 0
    ttl_s = 0

    def __init__(self):
        self.lookups = 0

    def lookup(self, *_a, **_k):
        self.lookups += 1
        return None

    def next_expiry_delay_ms(self):
        return None

    def sweep_expired(self):
        return 0


def _manager(num_pages=64, store=None) -> CacheManager:
    table = torch.zeros((4, 256), dtype=torch.int32)
    return CacheManager(num_pages, 1, table, "hybrid_radix",
                        linear_state_pool=_StatePool(), park_store=store)


def _insert_prefix(cm: CacheManager, ids: torch.Tensor, slot: int) -> None:
    pages = cm._allocate(len(ids))
    cm.prefix_cache.insert(ids, cm._page_to_token(pages), slot)


def test_admission_fits_is_in_tokens_and_subtracts_the_pages_that_would_lock():
    # 98,304-token evictable prefix, 16,384 of demand: pre-lock says yes, post-lock says no.
    assert admission_fits(need_now=16_384, reserved=0, available=98_304, protect_tokens=0)
    assert not admission_fits(need_now=16_384, reserved=0, available=98_304, protect_tokens=98_304)


def test_probe_matches_a_resident_prefix_without_touching_the_park_store():
    store = _RestoringParkStore()
    cm = _manager(num_pages=64, store=store)
    ids = torch.arange(1, 33)
    _insert_prefix(cm, ids, slot=cm.linear_state_pool.alloc()[0])
    free_before = len(cm.free_slots)
    probe = cm.probe_admission(torch.cat([ids, torch.tensor([99, 100])]), output_len=8, reserved=0)
    assert isinstance(probe, AdmissionProbe)
    assert probe.cached_len == 32 and probe.need_now == 2 + 8
    assert probe.protect_tokens == 32                        # evictable today, locked on admission
    assert probe.fits_empty and probe.fits_now
    assert store.lookups == 0                               # no park lookup, no restore
    assert len(cm.free_slots) == free_before                # no allocation


def test_probe_reproduces_the_post_lock_refusal():
    cm = _manager(num_pages=64)
    big = torch.arange(1, 61)                               # 60 of 64 pages, evictable
    _insert_prefix(cm, big, slot=cm.linear_state_pool.alloc()[0])
    probe = cm.probe_admission(torch.cat([big, torch.tensor([7])]), output_len=10, reserved=0)
    # pre-lock: available 64 >= 11; post-lock: 64 - 60 = 4 < 11
    assert probe.fits_empty and not probe.fits_now


def test_probe_counts_an_already_locked_prefix_as_free_room():
    cm = _manager(num_pages=64)
    shared = torch.arange(1, 33)
    _insert_prefix(cm, shared, slot=cm.linear_state_pool.alloc()[0])
    handle = cm.match_req(__import__("freetoken.scheduler.utils", fromlist=["PendingReq"]).PendingReq(
        1, torch.cat([shared, torch.tensor([5])]), __import__("freetoken.core", fromlist=["SamplingParams"]).SamplingParams(max_tokens=1)
    )).cuda_handle
    cm.lock(handle)                                          # another request already protects it
    probe = cm.probe_admission(torch.cat([shared, torch.tensor([9])]), output_len=4, reserved=0)
    assert probe.protect_tokens == 0 and probe.fits_now


def test_probe_reports_a_request_that_can_never_fit():
    cm = _manager(num_pages=8)
    probe = cm.probe_admission(torch.arange(1, 7), output_len=8, reserved=0)
    assert not probe.fits_empty and not probe.fits_now and probe.need_now == 14


def test_next_park_delay_folds_the_ttl_expiry_when_nothing_else_is_pending():
    class _Store(_RestoringParkStore):
        def next_expiry_delay_ms(self):
            return 4200

    cm = _manager(store=_Store())
    assert cm.next_park_delay_ms(now_ns=0) == 4200


def test_park_idle_sweeps_expired_families():
    class _Store(_RestoringParkStore):
        def __init__(self):
            super().__init__()
            self.swept = 0

        def sweep_expired(self):
            self.swept += 1
            return 2

    store = _Store()
    cm = _manager(store=store)
    cm.park_idle(now_ns=10**30)
    assert store.swept == 1


def test_probe_reads_the_handle_of_a_plain_radix_cache():
    """Final review I2: the non-hybrid caches return MatchResult(cuda_handle=<handle>), so
    cached_len/node live on the handle. Reading them off the result made every match look
    empty and every follow-up turn on such a model plan a grow it did not need."""
    table = torch.zeros((4, 256), dtype=torch.int32)
    cm = CacheManager(64, 1, table, "radix")             # no linear_state_pool: plain RadixCache
    ids = torch.arange(1, 33)
    pages = cm._allocate(len(ids))
    cm.prefix_cache.insert_prefix(ids, cm._page_to_token(pages))
    probe = cm.probe_admission(torch.cat([ids, torch.tensor([99, 100])]), output_len=8, reserved=0)
    assert probe.cached_len == 32                        # the inserted prefix, not 0
    assert probe.need_now == 2 + 8
    assert probe.protect_tokens == 32                    # evictable now, locked on admission
