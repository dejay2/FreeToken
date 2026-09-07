"""The between-step safe point for MoE-only rebuilds and governor steps (J1).

A rebuild that touches only the slot cache / layer residency may run with decode requests in
flight (their state lives in KV pages and the GDN pool); anything touching KV, GDN or the
window pool still waits for idle; nothing runs mid prefill chunk.
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.message import CacheRebuildBackendMsg, CacheStepBackendMsg
from freetoken.scheduler.scheduler import Scheduler


def _sched(prefill: bool, decode: bool, pending):
    s = Scheduler.__new__(Scheduler)
    s.prefill_manager = SimpleNamespace(runnable=prefill)
    s.decode_manager = SimpleNamespace(runnable=decode)
    s._pending_rebuild = pending
    return s


MOE_ONLY = CacheRebuildBackendMsg(request_id="r", moe_cache_size=3000)
MOVES = CacheRebuildBackendMsg(request_id="r", layer_moves=[(5, "pinned")])
KV = CacheRebuildBackendMsg(request_id="r", num_pages=100)
STEP = CacheStepBackendMsg(request_id="r", axis="vram", direction="down")


def test_moe_only_classification():
    assert Scheduler._is_moe_only_rebuild(MOE_ONLY)
    assert Scheduler._is_moe_only_rebuild(MOVES)
    assert Scheduler._is_moe_only_rebuild(STEP)
    assert not Scheduler._is_moe_only_rebuild(KV)
    assert not Scheduler._is_moe_only_rebuild(CacheRebuildBackendMsg(request_id="r", num_mamba_slots=8))
    assert not Scheduler._is_moe_only_rebuild(CacheRebuildBackendMsg(request_id="r", num_swa_pages=8))


def test_moe_only_runs_between_decode_steps_but_never_mid_prefill():
    for msg in (MOE_ONLY, MOVES, STEP):
        assert _sched(prefill=False, decode=True, pending=msg)._rebuild_can_run()
        assert _sched(prefill=False, decode=False, pending=msg)._rebuild_can_run()
        assert not _sched(prefill=True, decode=False, pending=msg)._rebuild_can_run()
        assert not _sched(prefill=True, decode=True, pending=msg)._rebuild_can_run()


def test_kv_rebuild_still_waits_for_idle():
    assert _sched(prefill=False, decode=False, pending=KV)._rebuild_can_run()
    assert not _sched(prefill=False, decode=True, pending=KV)._rebuild_can_run()
    assert not _sched(prefill=True, decode=False, pending=KV)._rebuild_can_run()


def test_unknown_pending_work_waits_for_idle():
    assert not Scheduler._is_moe_only_rebuild(object())
    assert not _sched(prefill=False, decode=True, pending=object())._rebuild_can_run()


def test_normal_loop_runs_a_moe_only_rebuild_while_decode_is_runnable():
    """The between-step path: a slot-cache/residency rebuild no longer waits for the chat to end."""
    from freetoken.engine.config import SpecDecodeConfig

    sched = _sched(prefill=False, decode=True, pending=MOVES)
    sched.config = SimpleNamespace(spec_decode=SpecDecodeConfig())
    sched.engine = SimpleNamespace(spec_draft=None)
    sched.receive_msg = lambda blocking: []
    sched._schedule_next_batch = lambda: None
    sched._process_last_data = lambda data: None
    calls = []

    def _exec():
        calls.append(True)
        sched._pending_rebuild = None

    sched._execute_pending_rebuild = _exec
    Scheduler.normal_loop(sched)
    assert calls == [True]
    # the same shell with a KV resize queued still defers (existing behaviour)
    sched._pending_rebuild = KV
    calls.clear()
    Scheduler.normal_loop(sched)
    assert calls == []
