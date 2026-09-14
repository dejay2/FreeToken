"""Rule 2 'Same-batch arrivals' (b): a never-started pending request the prefill manager cannot
seat for capacity is handed back to the dynamic controller instead of waiting in place."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.table import TableManager
from tests.scheduler.test_kv_dynamic_probe import _StatePool


def _msg(uid, n, max_tokens):
    from freetoken.message import UserMsg
    return UserMsg(uid=uid, input_ids=torch.arange(1, n + 1) + uid * 1000,
                   sampling_params=SamplingParams(max_tokens=max_tokens))


def _managers(num_pages):
    table = torch.zeros((4, 4096), dtype=torch.int32)
    cm = CacheManager(num_pages, 1, table, "hybrid_radix", linear_state_pool=_StatePool())
    tm = TableManager(3, table)
    dm = DecodeManager(1)
    return cm, PrefillManager(cm, tm, dm)


def test_two_requests_in_one_batch_that_do_not_fit_together_escalate_the_second():
    cm, pm = _managers(num_pages=1024)
    blocked, reserved = [], []
    pm.on_capacity_blocked = blocked.append
    pm.on_reserved = reserved.append
    pm.add_one_req(_msg(1, 300, 320))   # need 620
    pm.add_one_req(_msg(2, 300, 320))   # need 620; together 1240 > 1024
    batch = pm.schedule_next_batch(prefill_budget=8192)
    assert [r.uid for r in batch.reqs] == [1]
    assert reserved == [1]
    assert [p.uid for p in blocked] == [2]
    assert [p.uid for p in pm.pop_capacity_blocked()] == [2]
    assert [p.uid for p in pm.pending_list] == []


def test_a_request_that_can_never_fit_is_rejected_not_escalated():
    cm, pm = _managers(num_pages=64)
    blocked = []
    pm.on_capacity_blocked = blocked.append
    pm.add_one_req(_msg(1, 60, 20))    # 80 > 64 even empty
    assert pm.schedule_next_batch(prefill_budget=8192) is None
    assert blocked == [] and pm.pop_rejections()[0][0] == 1


def test_no_hooks_means_the_old_behaviour():
    cm, pm = _managers(num_pages=1024)
    pm.add_one_req(_msg(1, 300, 320))
    pm.add_one_req(_msg(2, 300, 320))
    batch = pm.schedule_next_batch(prefill_budget=8192)
    assert [r.uid for r in batch.reqs] == [1]
    assert [p.uid for p in pm.pending_list] == [2]    # still waiting in place
    assert pm.pop_capacity_blocked() == []


def test_a_blocked_request_admitted_on_a_later_pass_is_no_longer_escalated():
    """The resurrection case. uid 2 is blocked on pass 1 and runs normally on pass 2 once uid 1
    has given its pages back. Handing it back as capacity-blocked afterwards would let the
    controller hold a RUNNING request and re-append it to the pending list: a second full
    response for one request."""
    cm, pm = _managers(num_pages=1024)
    pm.on_capacity_blocked = lambda pending: None
    pm.add_one_req(_msg(1, 300, 320))
    pm.add_one_req(_msg(2, 300, 320))
    first = pm.schedule_next_batch(prefill_budget=8192)
    assert [r.uid for r in first.reqs] == [1]
    assert [p.uid for p in pm.pending_list] == [2]      # blocked, still waiting in place
    # uid 1 finishes and returns its pages.
    done = first.reqs[0]
    cm.cache_req(done, finished=True)
    pm.table_manager.free(done.table_idx)
    second = pm.schedule_next_batch(prefill_budget=8192)
    assert [r.uid for r in second.reqs] == [2]
    assert pm.pop_capacity_blocked() == [] and pm.pending_list == []


def test_a_blocked_request_records_the_reservation_it_was_refused_against():
    """Rule 2's concurrent target needs the room the OTHER requests were using at the moment of
    the block; at the idle point where the escalation is planned it is already 0. The figure is
    the adder's running reserved_size: in-flight decode (400) PLUS what this pass has already
    charged to the requests it did seat (uid 1's 300 + 320)."""
    cm, pm = _managers(num_pages=1024)
    pm.on_capacity_blocked = lambda pending: None
    # inflight_tokens is a read-only property; the adder reads it once, so stand in for it.
    pm.decode_manager = SimpleNamespace(inflight_tokens=400)
    pm.add_one_req(_msg(1, 300, 320))
    pm.add_one_req(_msg(2, 300, 320))
    pm.schedule_next_batch(prefill_budget=8192)
    blocked = pm.pop_capacity_blocked()
    assert [p.uid for p in blocked] == [2]
    assert blocked[0].blocked_reserved_tokens == 400 + 620
