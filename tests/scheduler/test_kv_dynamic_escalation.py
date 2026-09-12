"""Rule 2 'Same-batch arrivals' (b): a never-started pending request the prefill manager cannot
seat for capacity is handed back to the dynamic controller instead of waiting in place."""
from __future__ import annotations

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
