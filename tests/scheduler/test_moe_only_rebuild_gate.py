"""The between-step safe point for MoE-only rebuilds and governor steps (J1).

A rebuild that touches only the slot cache / layer residency may run with decode requests in
flight (their state lives in KV pages and the GDN pool); anything touching KV, GDN or the
window pool still waits for idle; nothing runs mid prefill chunk.
"""

from __future__ import annotations

from types import SimpleNamespace

import contextlib

from freetoken.message import (
    CacheProgressMsg,
    CacheRebuildBackendMsg,
    CacheStepBackendMsg,
    CacheStepResultMsg,
)
from freetoken.scheduler import scheduler as scheduler_mod
from freetoken.scheduler.prefill import ChunkedReq
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


def test_a_residency_proven_noop_step_is_answered_inline_and_never_queued():
    """The engine's residency-only preflight proves "ram up" has nothing to recall: the
    scheduler answers on receipt instead of queueing the step to a safe point, so the API
    gate reopens in one round trip and no device sync or rebuild runs."""
    sent: list = []
    calls: list = []
    s = Scheduler.__new__(Scheduler)
    s.config = SimpleNamespace(tp_info=SimpleNamespace(size=1))
    s.cache_manager = SimpleNamespace(supports_runtime_rebuild=True)
    s._pending_rebuild = None
    s._idle_wait_logged = False
    noop = {"applied": None, "layer": None, "moe_cache_size": 6262, "at_floor": False,
            "exhausted": True, "reason": "nothing to recall", "vram_free_bytes": 0}
    s.engine = SimpleNamespace(
        step_memory_noop=lambda axis, direction: calls.append((axis, direction)) or noop,
        residency_report=lambda: {"owned": 0, "pinned": 48, "disk": 0, "ram_parked": []},
        step_memory=lambda *a, **k: (_ for _ in ()).throw(AssertionError("step ran")),
    )
    s.send_result = lambda msgs: sent.extend(msgs)
    s._process_one_msg(CacheStepBackendMsg(request_id="up-1", axis="ram", direction="up"))
    assert calls == [("ram", "up")]
    assert s._pending_rebuild is None, "a no-op is never queued"
    reply = sent[0]
    assert reply.request_id == "up-1" and reply.status == "ok"
    assert reply.applied is None and reply.exhausted is True and reply.error == "nothing to recall"
    assert reply.layers == {"owned": 0, "pinned": 48, "disk": 0, "parked": 0}

    # A step that may apply is queued exactly as before, and the API is told it was received
    # (the first progress report of the operation; see test_a_queued_operation_reports_progress).
    s.engine.step_memory_noop = lambda axis, direction: None
    msg = CacheStepBackendMsg(request_id="down-1", axis="ram", direction="down")
    s._process_one_msg(msg)
    assert s._pending_rebuild is msg and len(sent) == 2
    assert isinstance(sent[1], CacheProgressMsg)
    assert (sent[1].request_id, sent[1].phase) == ("down-1", "queued")


# ---- progress reports for a queued/executing operation (review F1, 2026-09-09) ------------


def _progress_shell(monkeypatch, clock):
    monkeypatch.setattr(scheduler_mod.time, "monotonic", lambda: clock["t"])
    sent: list = []
    s = Scheduler.__new__(Scheduler)
    s.config = SimpleNamespace(tp_info=SimpleNamespace(size=1))
    s.cache_manager = SimpleNamespace(supports_runtime_rebuild=True, lazy_free_region=contextlib.nullcontext)
    s.prefill_manager = SimpleNamespace(runnable=True, pending_list=[])
    s.decode_manager = SimpleNamespace(runnable=False)
    s._pending_rebuild = None
    s._idle_wait_logged = False
    s.finished_reqs = set()
    s.send_result = lambda msgs: sent.extend(msgs)
    s._spec_record_plain = lambda batch: None
    s._ship_replies = lambda *a, **k: None
    s._log_cache_geometry = lambda *a, **k: None
    s.engine = SimpleNamespace(
        step_memory_noop=lambda axis, direction: None,
        residency_report=lambda: {"owned": 0, "pinned": 47, "disk": 1, "ram_parked": []},
    )
    return s, sent


def _chunked_batch():
    req = ChunkedReq.__new__(ChunkedReq)
    req.aborted = False
    batch = SimpleNamespace(reqs=[req], is_prefill=True)
    return (SimpleNamespace(batch=batch), (None, None, SimpleNamespace(synchronize=lambda: None)))


def test_a_queued_operation_reports_progress_per_drained_chunk_then_per_engine_phase(monkeypatch):
    """The F1 case: a real step arrives while one long prompt is being processed in chunks and
    nothing else is running. Intermediate chunks emit no token reply, so the scheduler now
    reports each drained chunk as progress (throttled), then "executing" at the safe point,
    then every engine phase through the hook it hands the engine, and the reply last."""
    clock = {"t": 1000.0}
    s, sent = _progress_shell(monkeypatch, clock)
    s._process_one_msg(CacheStepBackendMsg(request_id="up-1", axis="ram", direction="up"))
    assert [(m.request_id, m.phase) for m in sent] == [("up-1", "queued")]

    # Five chunks drain within one second: one "waiting" report, not five ("queued" itself
    # counted as the first report, so the throttle interval has to pass first).
    clock["t"] += scheduler_mod.MAINTENANCE_PROGRESS_INTERVAL_S
    for _ in range(5):
        clock["t"] += 0.2
        s._process_last_data(_chunked_batch())
    assert [m.phase for m in sent] == ["queued", "waiting"]
    assert sent[-1].detail == "drained 1 prefill"
    # Past the throttle interval the next chunk reports again.
    clock["t"] += scheduler_mod.MAINTENANCE_PROGRESS_INTERVAL_S
    s._process_last_data(_chunked_batch())
    assert [m.phase for m in sent] == ["queued", "waiting", "waiting"]

    # The safe point: the engine's phases arrive through the hook, un-throttled, then the reply.
    def step_memory(**kwargs):
        s.engine.maintenance_progress("step:probed", "ram up")
        s.engine.maintenance_progress("rebuild:teardown")
        s.engine.maintenance_progress("rebuild:layer_move", "layer 40 -> pinned")
        return {"applied": "disk->pinned", "layer": 40, "moe_cache_size": 6144, "at_floor": False,
                "vram_free_bytes": 0}

    s.engine.step_memory = step_memory
    s.prefill_manager.runnable = False
    s._execute_pending_rebuild()
    phases = [(m.phase if isinstance(m, CacheProgressMsg) else type(m).__name__) for m in sent[3:]]
    assert phases == ["executing", "step:probed", "rebuild:teardown", "rebuild:layer_move", "CacheStepResultMsg"]
    assert all(m.request_id == "up-1" for m in sent)
    assert isinstance(sent[-1], CacheStepResultMsg) and sent[-1].applied == "disk->pinned"
    # Nothing leaks past the operation: the hook is gone and later drains report nothing.
    assert s.engine.maintenance_progress is None and s._maintenance_request_id is None
    clock["t"] += 10.0
    s._process_last_data(_chunked_batch())
    assert len(sent) == 8


def test_the_hook_is_removed_even_when_the_step_fails(monkeypatch):
    clock = {"t": 1000.0}
    s, sent = _progress_shell(monkeypatch, clock)
    s._process_one_msg(CacheStepBackendMsg(request_id="up-2", axis="ram", direction="up"))

    def boom(**kwargs):
        s.engine.rebuild_teardown_started = True
        raise RuntimeError("cuda OOM")

    s.engine.step_memory = boom
    s._execute_pending_rebuild()
    assert isinstance(sent[-1], CacheStepResultMsg) and sent[-1].status == "failed"
    assert s.engine.maintenance_progress is None and s._maintenance_request_id is None


def test_a_progress_send_failure_does_not_fail_the_operation(monkeypatch):
    clock = {"t": 1000.0}
    s, sent = _progress_shell(monkeypatch, clock)
    s._pending_rebuild = CacheStepBackendMsg(request_id="up-3", axis="ram", direction="up")
    s._maintenance_request_id = "up-3"

    def flaky(msgs):
        if any(isinstance(m, CacheProgressMsg) for m in msgs):
            raise OSError("socket closed")
        sent.extend(msgs)

    s.send_result = flaky
    s.engine.step_memory = lambda **k: {"applied": None, "layer": None, "moe_cache_size": 6144,
                                        "at_floor": False, "vram_free_bytes": 0}
    s._execute_pending_rebuild()
    assert isinstance(sent[-1], CacheStepResultMsg) and sent[-1].status == "ok"
