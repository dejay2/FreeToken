"""Scheduler glue for the dynamic KV pool on a lightweight Scheduler shell (the
test_moe_only_rebuild_gate pattern): hold/admit decisions, the idle plan execution and its
three outcomes, the drain barrier, and the finish hooks."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.message import (
    CacheRebuildBackendMsg, ErrorReplyMsg, KVDynamicStatusMsg, MaintenanceBeginMsg, UserMsg,
)
from freetoken.scheduler.cache import AdmissionProbe
from freetoken.scheduler.kv_dynamic import KVDynamicController, KVDynamicPolicy
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.utils import PendingReq

PAGE, KV_PAGE, SLOT = 64, 13_248 * 64, 2_772_480


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _shell(*, num_pages=1025, probe=None, clock=None):
    s = Scheduler.__new__(Scheduler)
    clock = clock or _Clock()
    policy = KVDynamicPolicy(floor_pages=1025, ceiling_pages=4097, step_pages=512, page_size=PAGE,
                             kv_bytes_per_page=KV_PAGE, slot_bytes=SLOT, slot_floor=1024)
    s._kv_dynamic = KVDynamicController(policy, shrink_idle_s=600, instance_id="i", clock=clock)
    s._clock = clock
    s.engine = SimpleNamespace(
        num_pages=num_pages, max_seq_len=(num_pages - 1) * PAGE,
        pool_budget_bytes=7200 * SLOT + 1025 * KV_PAGE,
        moe_offload_cache=SimpleNamespace(cache_size=7200),
        snapshot_pool_budget=lambda: s.engine.pool_budget_bytes,
        rebuild_teardown_started=False, maintenance_progress=None, kv_dynamic_floor_pages=1025,
    )
    s.config = SimpleNamespace(kv_ceiling_tokens=262_144 + 64, page_size=PAGE,
                               tp_info=SimpleNamespace(size=1))
    s.cache_manager = SimpleNamespace(
        probe_admission=lambda ids, out, reserved, cache_private=False: probe,
        supports_runtime_rebuild=True,
    )
    s.prefill_manager = SimpleNamespace(runnable=False, pending_list=[], added=[],
                                        add_one_req=lambda m: s.prefill_manager.added.append(m.uid),
                                        pop_capacity_blocked=lambda: [])
    s.decode_manager = SimpleNamespace(runnable=False, running_reqs=[], inflight_tokens=0)
    s.sent = []
    s.send_result = lambda msgs: s.sent.extend(msgs)
    s._pending_rebuild = None
    s._abort_tombstones = {}
    s._maintenance_request_id = None
    s._maintenance_progress_at = -float("inf")
    s._kv_dynamic_last_status = None
    s.executed = []
    return s


def _user(uid, n, max_tokens):
    return UserMsg(uid=uid, input_ids=torch.arange(1, n + 1), sampling_params=SamplingParams(max_tokens=max_tokens))


def test_small_request_is_admitted_at_once():
    s = _shell(probe=AdmissionProbe(need_now=39_000, protect_tokens=0, fits_empty=True, fits_now=True, cached_len=0))
    assert s._queue_for_kv_dynamic(_user(1, 7_000, 32_000)) is False
    assert s._kv_dynamic.uncommitted_tokens == 39_000


def test_request_over_the_pool_is_held_and_grows_at_idle_then_admits():
    s = _shell(probe=AdmissionProbe(need_now=92_000, protect_tokens=0, fits_empty=False, fits_now=False, cached_len=0))
    assert s._queue_for_kv_dynamic(_user(2, 60_000, 32_000)) is True
    outcomes = []

    def fake_exec():
        msg = s._pending_rebuild
        s.executed.append((msg.request_id, msg.num_pages, msg.moe_cache_size))
        s._pending_rebuild = None
        s.engine.num_pages = msg.num_pages
        s.engine.max_seq_len = (msg.num_pages - 1) * PAGE
        outcomes.append("ok")
        return "ok"

    s._execute_pending_rebuild = fake_exec
    s._run_kv_dynamic_idle()
    assert s.executed == [("auto-kv:i:1", 98_304 // PAGE + 1, 7200 - 157)]
    assert any(isinstance(m, MaintenanceBeginMsg) and m.request_id == "auto-kv:i:1" for m in s.sent)
    assert s.prefill_manager.added == [2] and not s._kv_dynamic.has_held()


def test_rejected_rebuild_still_admits_against_the_old_pool():
    s = _shell(probe=AdmissionProbe(need_now=92_000, protect_tokens=0, fits_empty=False, fits_now=False, cached_len=0))
    s._queue_for_kv_dynamic(_user(2, 60_000, 32_000))
    s._execute_pending_rebuild = lambda: (setattr(s, "_pending_rebuild", None), "rejected")[1]
    s._run_kv_dynamic_idle()
    assert s.prefill_manager.added == [2]


def test_failed_rebuild_error_replies_the_held_requests_and_disables_the_controller():
    s = _shell(probe=AdmissionProbe(need_now=92_000, protect_tokens=0, fits_empty=False, fits_now=False, cached_len=0))
    s._queue_for_kv_dynamic(_user(2, 60_000, 32_000))
    s._queue_for_kv_dynamic(_user(3, 1_000, 100))          # barrier: queued behind
    s._execute_pending_rebuild = lambda: (setattr(s, "_pending_rebuild", None), "failed")[1]
    s._run_kv_dynamic_idle()
    errors = [m for m in s.sent if isinstance(m, ErrorReplyMsg)]
    assert sorted(m.uid for m in errors) == [2, 3] and all(m.code == "server_error" for m in errors)
    assert s.prefill_manager.added == [] and not s._kv_dynamic.enabled


def test_prompt_over_the_ceiling_is_refused_with_the_existing_message():
    s = _shell(probe=None)
    assert s._queue_for_kv_dynamic(_user(4, 300_000, 10)) is True   # handled (refused), not admitted
    err = [m for m in s.sent if isinstance(m, ErrorReplyMsg)][0]
    assert err.code == "context_length_exceeded" and "300000 tokens > 262144" in err.error


def test_max_tokens_is_clipped_against_the_ceiling_not_the_small_pool():
    s = _shell(probe=AdmissionProbe(need_now=262_144, protect_tokens=0, fits_empty=False, fits_now=False, cached_len=0))
    msg = _user(5, 240_000, 32_000)
    assert s._queue_for_kv_dynamic(msg) is True
    assert msg.sampling_params.max_tokens == 262_144 - 240_000


def test_timer1_shrink_runs_from_run_when_idle_and_reports_status():
    clock = _Clock()
    s = _shell(num_pages=2049, clock=clock, probe=None)
    s._note_request_finished([object()])
    clock.now += 601
    s._execute_pending_rebuild = lambda: (s.executed.append(s._pending_rebuild.num_pages), setattr(s, "_pending_rebuild", None), "ok")[2]
    s._run_kv_dynamic_idle()
    assert s.executed == [1025]
    assert any(isinstance(m, KVDynamicStatusMsg) for m in s.sent)


def test_idle_poll_timeout_includes_the_shrink_deadline():
    clock = _Clock()
    s = _shell(num_pages=2049, clock=clock, probe=None)
    s.cache_manager.next_park_delay_ms = lambda: None
    s._note_request_finished([object()])
    assert s.idle_poll_timeout_ms() == 600_000
    s.cache_manager.next_park_delay_ms = lambda: 250
    assert s.idle_poll_timeout_ms() == 250


def _blocked(uid, n, max_tokens):
    """What PrefillManager.pop_capacity_blocked hands back: a never-started pending request
    the admission pass could not seat (rule 2(b))."""
    return PendingReq(uid, torch.arange(1, n + 1), SamplingParams(max_tokens=max_tokens))


def test_a_capacity_blocked_request_is_escalated_and_returns_to_the_pending_list():
    s = _shell(probe=None)
    blocked = _blocked(9, 60_000, 32_000)
    s.prefill_manager.pop_capacity_blocked = lambda: [blocked]
    s._execute_pending_rebuild = lambda: (
        s.executed.append(s._pending_rebuild.num_pages), setattr(s, "_pending_rebuild", None), "ok"
    )[2]
    s._run_kv_dynamic_idle()
    # It needs 92,000 tokens, so the same grow as an arrival that never got past admission.
    assert s.executed == [98_304 // PAGE + 1]
    # An escalated PendingReq is already prepared: it goes back to the pending list, never
    # through _admit_user_msg (which would re-run the picture/clip work on a UserMsg).
    assert s.prefill_manager.pending_list == [blocked] and s.prefill_manager.added == []


def test_an_aborted_capacity_blocked_request_is_not_resurrected():
    s = _shell(probe=None)
    blocked = _blocked(9, 60_000, 32_000)
    s.prefill_manager.pop_capacity_blocked = lambda: [blocked]
    s._abort_tombstones = {9: None}          # aborted after the pass that blocked it
    s._execute_pending_rebuild = lambda: (_ for _ in ()).throw(AssertionError("no rebuild"))
    s._run_kv_dynamic_idle()
    assert not s._kv_dynamic.has_held()
    assert s.prefill_manager.pending_list == [] and s.prefill_manager.added == []
