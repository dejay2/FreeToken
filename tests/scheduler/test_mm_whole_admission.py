"""Bidirectional image spans are admitted whole, or rejected before resource allocation."""
from types import SimpleNamespace

import torch

from freetoken.core import Context, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.message import MMItem
from freetoken.scheduler.prefill import PrefillAdder
from freetoken.scheduler.utils import PendingReq


def _case(span, budget=8, *, remaining=None, page_size=1, alignment=1, swa_capacity=None, swa_available=None, window=4):
    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))
    swa = swa_capacity is not None
    manager = SimpleNamespace(swa_paged=swa, page_size=page_size, prefill_chunk_align=alignment,
        sliding_window_size=window, swa_pool=SimpleNamespace(swa_num_tokens=swa_capacity + 1) if swa else None,
        swa_available_size=swa_available if swa_available is not None else swa_capacity)
    table = SimpleNamespace(token_pool=torch.zeros((1, 64), dtype=torch.int32))
    adder = PrefillAdder(budget if remaining is None else remaining, 0, manager, table, keep_images_whole=True, pass_budget=budget)
    item = MMItem('image', 1, 1000001, [list(span)], feature=torch.zeros(1))
    pending = PendingReq(1, torch.arange(span[1] + 2, dtype=torch.int32), SamplingParams(max_tokens=1), mm_items=[item], cache_private=True)
    return adder, pending


def test_oversized_image_rejected_before_any_allocation():
    adder, pending = _case((1, 10), budget=8)
    allocation_calls = []
    adder._try_allocate_one = lambda req: allocation_calls.append(req)
    assert adder.try_add_one(pending) is None
    assert allocation_calls == []
    assert 'image' in pending.admission_error and 'reduce image tokens' in pending.admission_error


def test_image_alignment_prefix_counts_toward_pass_budget():
    adder, pending = _case((3, 11), budget=8, alignment=4)
    allocation_calls = []
    adder._try_allocate_one = lambda req: allocation_calls.append(req)
    assert adder.try_add_one(pending) is None
    assert allocation_calls == []
    assert 'prefill' in pending.admission_error


def test_image_window_footprint_rejected_before_any_allocation():
    # Start 8 retains [2,8) for window=4 plus the extra page; image [8,16)
    # needs 14 page-rounded slots, even though the eight image tokens fit the pass.
    adder, pending = _case((8, 16), budget=8, page_size=2, swa_capacity=12)
    allocation_calls = []
    adder._try_allocate_one = lambda req: allocation_calls.append(req)
    assert adder.try_add_one(pending) is None
    assert allocation_calls == []
    assert 'sliding-window' in pending.admission_error


def test_whole_image_waits_when_other_requests_use_pass_budget():
    adder, pending = _case((0, 6), budget=8, remaining=4)
    handle = SimpleNamespace(cached_len=0)
    assert adder._add_one_req(pending, handle, 0, 0) is None
    assert adder.token_budget == 4 and adder.reserved_size == 0
    assert getattr(pending, 'admission_error', None) is None


def test_whole_image_waits_for_temporary_swa_contention():
    adder, pending = _case((0, 6), budget=8, page_size=2, swa_capacity=12, swa_available=4)
    handle = SimpleNamespace(cached_len=0)
    assert adder._add_one_req(pending, handle, 0, 0) is None
    assert adder.token_budget == 8 and adder.reserved_swa == 0


def test_whole_image_succeeds_when_budget_is_available():
    adder, pending = _case((0, 6), budget=8)
    handle = SimpleNamespace(cached_len=0)
    adder._try_allocate_one = lambda req: (handle, 0, None, None, None)
    req = adder.try_add_one(pending)
    assert req is not None and req.device_len == 8
    assert adder.token_budget == 0


def test_non_aligned_budget_rejects_image_before_permanent_wait():
    adder, pending = _case((0, 7), budget=7, alignment=4)
    calls = []
    adder._try_allocate_one = lambda req: calls.append(req)
    assert adder.try_add_one(pending) is None
    assert calls == [] and 'alignment' in pending.admission_error


def test_images_without_legal_aligned_boundary_are_checked_together():
    adder, pending = _case((0, 6), budget=8, alignment=4)
    pending.input_ids = torch.arange(12, dtype=torch.int32)
    pending.mm_items.append(MMItem('image', 2, 1000002, [[7, 10]], feature=torch.zeros(1)))
    calls = []
    adder._try_allocate_one = lambda req: calls.append(req)
    assert adder.try_add_one(pending) is None
    assert calls == [] and 'alignment' in pending.admission_error


def test_prompt_final_image_can_end_without_alignment_padding():
    adder, pending = _case((0, 7), budget=7, alignment=4)
    pending.input_ids = pending.input_ids[:7]
    handle = SimpleNamespace(cached_len=0)
    adder._try_allocate_one = lambda req: (handle, 0, None, None, None)
    req = adder.try_add_one(pending)
    assert req is not None and req.device_len == 7
