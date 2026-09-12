"""Scheduler glue for the dynamic KV pool on a lightweight Scheduler shell (the
test_moe_only_rebuild_gate pattern): hold/admit decisions, the idle plan execution and its
three outcomes, the drain barrier, and the finish hooks."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import (
    CacheRebuildBackendMsg, CacheRebuildResultMsg, CacheStepBackendMsg, CacheStepResultMsg,
    ErrorReplyMsg, KVDynamicStatusMsg, MaintenanceBeginMsg, UserMsg,
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
        # bank_sources as _build_kv_dynamic_policy reads it: one bank, one streaming layer,
        # one row per expert. expand() gives the row the real byte count with no allocation.
        moe_offload_cache=SimpleNamespace(
            cache_size=7200,
            bank_sources={"w": [torch.empty(1, dtype=torch.float16).expand(1, SLOT // 2)]},
        ),
        _gpu_owned_layer_ids=frozenset(),
        _kv_bytes_per_page=lambda: KV_PAGE,
        snapshot_pool_budget=lambda: s.engine.pool_budget_bytes,
        rebuild_teardown_started=False, maintenance_progress=None, kv_dynamic_floor_pages=1025,
    )
    s.config = SimpleNamespace(kv_ceiling_tokens=262_144 + 64, page_size=PAGE,
                               kv_floor_tokens=65_536, kv_step_tokens=32_768,
                               moe_prefill_overlap=True,
                               model_config=SimpleNamespace(num_experts=512),
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


def test_a_failed_rebuild_latches_the_backend_and_schedules_nothing_afterwards():
    """External review of 8d566de: the frontend's ``failed`` gate only stops NEW arrivals, so a
    never-started request already in the pending list would still be handed to the torn-down
    engine by _schedule_next_batch."""
    s = _shell(probe=AdmissionProbe(need_now=92_000, protect_tokens=0, fits_empty=False,
                                    fits_now=False, cached_len=0))
    pending = PendingReq(1, torch.arange(1, 500), SamplingParams(max_tokens=10))
    s.prefill_manager.pending_list.append(pending)          # A: queued, never started
    s._queue_for_kv_dynamic(_user(2, 60_000, 32_000))       # B: held for the grow
    s._execute_pending_rebuild = lambda: (setattr(s, "_pending_rebuild", None), "failed")[1]
    s._run_kv_dynamic_idle()

    errors = [m for m in s.sent if isinstance(m, ErrorReplyMsg)]
    assert sorted(m.uid for m in errors) == [1, 2]          # exactly one reply each
    assert all(m.code == "server_error" and "needs a restart" in m.error for m in errors)
    assert s.prefill_manager.pending_list == [] and s.prefill_manager.added == []
    assert s._engine_failed is not None
    # Nothing is forwarded into the dead engine (and the managers are never touched: this
    # shell has no schedule_next_batch at all).
    assert s._schedule_next_batch() is None

    # Later arrivals are answered, not queued -- through either door.
    assert s._queue_for_kv_dynamic(_user(3, 1_000, 100)) is False   # controller disabled
    s._process_one_msg(_user(3, 1_000, 100))
    late = [m for m in s.sent if isinstance(m, ErrorReplyMsg) and m.uid == 3]
    assert len(late) == 1 and late[0].code == "server_error"
    assert s.prefill_manager.added == [] and s.prefill_manager.pending_list == []


def test_an_external_rebuild_reprices_the_controller_policy():
    """External review of 8d566de: a window-only rebuild moves the per-page price, and the
    boot-time policy is frozen -- re-snapshotting the budget alone is half a re-pricing."""
    s = _shell(num_pages=2049, probe=None)
    s._kv_dynamic.on_request_finished()
    held_before = s._kv_dynamic.last_request_finished
    s._pending_rebuild = CacheRebuildBackendMsg(request_id="op-1", num_swa_pages=512)
    s._execute_pending_operation = lambda msg: "ok"
    s.engine._kv_bytes_per_page = lambda: KV_PAGE + 4_096   # the window pin moved the price

    assert s._execute_pending_rebuild() == "ok"
    assert s._kv_dynamic.policy.kv_bytes_per_page == s.engine._kv_bytes_per_page()
    assert s._kv_dynamic.policy.floor_pages == 1025 and s._kv_dynamic.policy.ceiling_pages == 4097
    assert s._kv_dynamic.last_request_finished == held_before   # a re-pricing, not a reset


def test_an_automatic_rebuild_does_not_reprice():
    """auto-kv: rebuilds never touch the window, so nothing re-reads the price for them."""
    s = _shell(num_pages=2049, probe=None)
    s._pending_rebuild = CacheRebuildBackendMsg(request_id="auto-kv:i:1", num_pages=1025)
    s._execute_pending_operation = lambda msg: "ok"
    s.engine._kv_bytes_per_page = lambda: KV_PAGE + 4_096
    assert s._execute_pending_rebuild() == "ok"
    assert s._kv_dynamic.policy.kv_bytes_per_page == KV_PAGE


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


def test_timer1_shrink_runs_at_the_idle_point_and_reports_status():
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


def _block(s, uid, n, max_tokens, *, reserved=0):
    """Put a never-started pending request into the shell's pending list and arm a
    pop_capacity_blocked that mirrors the real one: only entries STILL in the pending list come
    back, and they leave it (prefill.py, pop-time filter)."""
    pending = PendingReq(uid, torch.arange(1, n + 1), SamplingParams(max_tokens=max_tokens))
    pending.blocked_reserved_tokens = reserved
    s.prefill_manager.pending_list.append(pending)

    def pop():
        out = [p for p in [pending] if p in s.prefill_manager.pending_list]
        for p in out:
            s.prefill_manager.pending_list.remove(p)
        return out

    s.prefill_manager.pop_capacity_blocked = pop
    return pending


def test_a_capacity_blocked_request_is_escalated_and_returns_to_the_pending_list():
    s = _shell(probe=None)
    # It needs 92,000 tokens and was refused against 100,000 tokens of other requests, so the
    # plan is rule 2's CONCURRENT target: round_up(100,000 + 92,000 + 1) = 196,608.
    blocked = _block(s, 9, 60_000, 32_000, reserved=100_000)
    s._execute_pending_rebuild = lambda: (
        s.executed.append(s._pending_rebuild.num_pages), setattr(s, "_pending_rebuild", None), "ok"
    )[2]
    s._run_kv_dynamic_idle()
    assert s.executed == [196_608 // PAGE + 1]
    assert s._kv_dynamic.last_plan["reason"] == "grow-concurrent"
    # An escalated PendingReq is already prepared: it goes back to the pending list, never
    # through _admit_user_msg (which would re-run the picture/clip work on a UserMsg).
    assert s.prefill_manager.pending_list == [blocked] and s.prefill_manager.added == []


def test_an_aborted_capacity_blocked_request_is_not_resurrected():
    s = _shell(probe=None)
    _block(s, 9, 60_000, 32_000)
    s._abort_tombstones = {9: None}          # aborted after the pass that blocked it
    s._execute_pending_rebuild = lambda: (_ for _ in ()).throw(AssertionError("no rebuild"))
    s._run_kv_dynamic_idle()
    assert not s._kv_dynamic.has_held()
    assert s.prefill_manager.pending_list == [] and s.prefill_manager.added == []


# ---- final review C1: a refused plan must not be retried every millisecond -----------------

def _refusing_shrink_shell(outcome="rejected"):
    """A grown, quiet shell whose rebuilds always come back with ``outcome``."""
    clock = _Clock()
    s = _shell(num_pages=2049, clock=clock, probe=None)
    s.cache_manager.next_park_delay_ms = lambda: None
    s._note_request_finished([object()])
    clock.now += 601                                     # Timer 1 overdue
    s._execute_pending_rebuild = lambda: (
        s.executed.append(s._pending_rebuild.num_pages), setattr(s, "_pending_rebuild", None), outcome
    )[2]
    return s, clock


def test_a_rejected_shrink_rearms_timer1_instead_of_retrying_every_millisecond():
    s, _ = _refusing_shrink_shell()
    s._run_kv_dynamic_idle()
    assert s.executed == [1025]                          # the shrink was attempted once
    # Without the re-arm the deadline is 1 ms and run_when_idle re-runs the same destructive
    # rebuild on every idle poll; with it the next attempt is a full shrink_idle_s away.
    assert s.idle_poll_timeout_ms() == 600_000
    assert s._kv_dynamic.consecutive_failures == 1
    s._run_kv_dynamic_idle()                             # same idle point, no second rebuild
    assert s.executed == [1025]


def test_three_refused_shrinks_stop_the_planning_until_a_request_finishes():
    s, clock = _refusing_shrink_shell()
    for _ in range(3):
        s._run_kv_dynamic_idle()
        clock.now += 601                                 # the re-armed timer comes due again
    assert s.executed == [1025, 1025, 1025]
    assert s._kv_dynamic.consecutive_failures == 3
    s._run_kv_dynamic_idle()
    assert s.executed == [1025, 1025, 1025]              # fourth attempt suppressed
    assert s._kv_dynamic.status(current_pages=2049,
                                pool_budget_bytes=s.engine.pool_budget_bytes)["consecutive_failures"] == 3
    # Traffic re-arms it: the card's situation has changed.
    s._note_request_finished([object()])
    assert s._kv_dynamic.consecutive_failures == 0
    clock.now += 601
    s._run_kv_dynamic_idle()
    assert s.executed == [1025, 1025, 1025, 1025]


def test_a_suspended_shrink_stops_waking_the_idle_loop():
    """External review of 8d566de: with the cap reached and the re-armed deadline expired, the
    old next_deadline_ms returned its 1 ms floor for ever -- a quiet server woke every
    millisecond to re-decide not to shrink."""
    s, clock = _refusing_shrink_shell()
    for _ in range(3):
        s._run_kv_dynamic_idle()
        clock.now += 601
    assert s.executed == [1025, 1025, 1025]
    assert s.idle_poll_timeout_ms() is None               # no park delay either: block on the queue
    s.cache_manager.next_park_delay_ms = lambda: 250
    assert s.idle_poll_timeout_ms() == 250                # Timer 2 still wins on its own merit
    s.cache_manager.next_park_delay_ms = lambda: None
    s._note_request_finished([object()])                  # traffic re-arms the full window
    assert s.idle_poll_timeout_ms() == 600_000


def test_a_rejected_grow_still_admits_the_head_and_leaves_no_1ms_deadline():
    s = _shell(probe=AdmissionProbe(need_now=92_000, protect_tokens=0, fits_empty=False,
                                    fits_now=False, cached_len=0))
    s.cache_manager.next_park_delay_ms = lambda: None
    s._queue_for_kv_dynamic(_user(2, 60_000, 32_000))
    s._execute_pending_rebuild = lambda: (setattr(s, "_pending_rebuild", None), "rejected")[1]
    s._run_kv_dynamic_idle()
    assert s.prefill_manager.added == [2] and not s._kv_dynamic.has_held()
    assert s.idle_poll_timeout_ms() in (None, 600_000)   # never the 1 ms retry storm


# ---- final review I1: the same-batch charge must not leak on an early refusal ---------------

def test_a_request_refused_by_the_max_seq_len_clip_releases_its_uncommitted_charge():
    s = _shell(probe=AdmissionProbe(need_now=7_100, protect_tokens=0, fits_empty=True,
                                    fits_now=True, cached_len=0))
    # The engine's own max_seq_len sits below the pool here (the ceiling clip in
    # _queue_for_kv_dynamic passes, _admit_user_msg's clip then refuses): the one arrangement
    # that reaches the early return with a charge already on the books.
    s.engine.max_seq_len = 5_000
    msg = _user(6, 7_000, 100)
    assert s._queue_for_kv_dynamic(msg) is False         # admitted, charged
    assert s._kv_dynamic.uncommitted_tokens == 7_100
    s._admit_user_msg(msg)                               # refused: max_output_len <= 0
    assert [m.code for m in s.sent if isinstance(m, ErrorReplyMsg)] == ["context_length_exceeded"]
    assert s.prefill_manager.added == []
    assert s._kv_dynamic.uncommitted_tokens == 0         # released, not leaked for the process


# ---- final review I3: no shrink while a never-admitted request waits (spec rule 4) ----------

def test_timer1_does_not_shrink_while_a_never_admitted_request_is_queued():
    clock = _Clock()
    s = _shell(num_pages=2049, clock=clock, probe=None)
    pending = PendingReq(11, torch.arange(1, 5_000), SamplingParams(max_tokens=100))
    pending.blocked_reserved_tokens = 0
    s.prefill_manager.pending_list.append(pending)
    s._note_request_finished([object()])
    clock.now += 601
    s._execute_pending_rebuild = lambda: (_ for _ in ()).throw(AssertionError("no rebuild"))
    s._run_kv_dynamic_idle()
    assert s.executed == [] and s.engine.num_pages == 2049


# ---- final review I5: the status debounce survives the shrink countdown --------------------

def test_the_status_debounce_ignores_the_shrink_countdown():
    clock = _Clock()
    s = _shell(num_pages=2049, clock=clock, probe=None)
    s._note_request_finished([object()])
    s._send_kv_dynamic_status()
    clock.now += 0.3
    s._send_kv_dynamic_status()
    assert len([m for m in s.sent if isinstance(m, KVDynamicStatusMsg)]) == 1


# ---- PR review 2 (2026-09-12): the latch also stops MTP and the running request -------------

class _RunningReq:
    """A request decoding with a primed draft head: everything _spec_candidate reads, so the
    real predicate (not a stub) would hand this request to _speculative_decode_step. The
    decode manager holds requests in a set and SimpleNamespace is unhashable."""

    def __init__(self, uid: int):
        self.uid = uid
        self.aborted = False
        self.mm_embeds = None
        self.can_decode = True
        self.table_idx = 0
        self.extend_len = 1
        self.input_ids = torch.zeros(4, dtype=torch.int32)
        self.device_len = 4
        self.remain_len = 8


def _latched_loop_shell(loop: str):
    """A shell decoding one MTP-eligible request with a governor step queued that fails after
    teardown. Both loops dispatch speculation BEFORE _schedule_next_batch, so the latch has to
    be visible to the MTP path and to the decode manager, not only to the scheduler."""
    from freetoken.engine.config import SpecDecodeConfig
    from freetoken.scheduler.decode import DecodeManager

    s = _shell(probe=None)
    s.decode_manager = DecodeManager(page_size=PAGE)
    req = _RunningReq(uid=7)
    s.decode_manager.running_reqs.add(req)
    s.finished_reqs = set()
    s._pending_abort_acks = set()
    s._last_data = None
    s._idle_wait_logged = True
    # MTP on and ready: every one of _spec_dispatch_ready's real conditions holds but the latch.
    s.config.spec_decode = SpecDecodeConfig(enabled=True)
    s.engine.spec_draft = SimpleNamespace(is_ready=lambda r: True)
    s._spec_policy = lambda r: None                      # no adaptive cooldown in the shell
    s.engine.stream = SimpleNamespace()
    s.stream = SimpleNamespace(wait_stream=lambda other: None)
    s.engine_stream_ctx = contextlib.nullcontext()
    s._log_cache_geometry = lambda *a, **k: None

    def _step_memory(**kwargs):
        s.engine.rebuild_teardown_started = True        # past the point of no return
        raise RuntimeError("CUDA out of memory re-capturing graphs")

    s.engine.step_memory = _step_memory
    s.engine.residency_report = lambda: {"owned": 0, "pinned": 48, "disk": 0, "ram_parked": []}
    s._pending_rebuild = CacheStepBackendMsg(request_id="gov-1", axis="vram", direction="down")

    calls = {"spec": [], "forward": [], "batch": [], "blocking": []}
    s.receive_msg = lambda blocking: (calls["blocking"].append(blocking), [])[1]
    s._speculative_decode_step = lambda r: calls["spec"].append(r)
    s._forward = lambda forward_input: calls["forward"].append(forward_input)
    s._process_last_data = lambda data: None

    def _schedule():
        out = Scheduler._schedule_next_batch(s)          # the real rule-8 check
        calls["batch"].append(out)
        return out

    s._schedule_next_batch = _schedule
    run = (
        (lambda: Scheduler.normal_loop(s)) if loop == "normal"
        else (lambda: Scheduler.overlap_loop(s, None))
    )
    return s, req, calls, run


@pytest.mark.parametrize("loop", ["normal", "overlap"])
def test_a_failed_step_stops_speculation_and_the_running_request(loop):
    s, req, calls, run = _latched_loop_shell(loop)

    run()                                                # the iteration that latches
    assert s._engine_failed is not None
    assert calls["spec"] == [] and calls["forward"] == []
    assert calls["batch"] == [None]                      # nothing forwarded into the dead engine
    errors = [m for m in s.sent if isinstance(m, ErrorReplyMsg)]
    assert [(m.uid, m.code) for m in errors] == [(7, "server_error")]
    assert "needs a restart" in errors[0].error
    assert s.decode_manager.running_reqs == set()        # CPU side only: dropped, never freed
    assert req.aborted is True and req in s.finished_reqs

    run()                                                # the next iteration must not spin
    assert calls["blocking"] == [False, True]
    assert calls["spec"] == [] and calls["forward"] == []
    assert [(m.uid, m.code) for m in s.sent if isinstance(m, ErrorReplyMsg)] == [(7, "server_error")]


@pytest.mark.parametrize("loop", ["normal", "overlap"])
def test_the_mtp_gate_reads_the_latch_even_with_a_request_still_eligible(loop):
    """Belt and braces: the dispatch is refused on the latch alone, whatever the managers say."""
    s, req, calls, run = _latched_loop_shell(loop)
    assert s._spec_dispatch_ready() is True              # before the latch: the real predicate
    assert s._spec_candidate() is req                    # and it really would be dispatched
    s._latch_engine_failed("test")
    s.decode_manager.running_reqs.add(req)               # a request that somehow survived
    assert s._spec_dispatch_ready() is False
    assert s._spec_candidate() is None

    # ...and the loop still blocks on the queue rather than spinning on that request.
    s._pending_rebuild = None
    run()
    assert calls["blocking"] == [True]
    assert calls["spec"] == [] and calls["forward"] == [] and calls["batch"] == [None]


def test_run_when_idle_does_nothing_once_latched():
    """The latched loop blocks on the queue, so run_when_idle fires on every poll -- and
    parking / the integrity check would walk pools a failed teardown left unknown."""
    s = _shell(probe=None)
    s.cache_manager.park_idle = lambda: (_ for _ in ()).throw(AssertionError("parked"))
    s.cache_manager.check_integrity = lambda: (_ for _ in ()).throw(AssertionError("checked"))
    s.cache_manager.drain_pending_parks = lambda: (_ for _ in ()).throw(AssertionError("drained"))
    s._engine_failed = "torn down"
    Scheduler.run_when_idle(s)


def test_maintenance_arriving_after_the_latch_is_refused_not_queued():
    """Rule 8: a maintenance message already on the socket when the latch fired must not be
    queued for a safe point that will never come (external review of PR #5, round 3,
    2026-09-12)."""
    s = _shell(probe=None)
    s.engine.linear_state_pool = None  # _reply_rebuild's geometry readout needs this attribute
    s._latch_engine_failed("test")

    s._queue_maintenance(CacheRebuildBackendMsg(request_id="m1", moe_cache_size=3000))
    s._queue_maintenance(CacheStepBackendMsg(request_id="s1", axis="vram", direction="down"))

    assert s._pending_rebuild is None
    rebuild_replies = [m for m in s.sent if isinstance(m, CacheRebuildResultMsg)]
    step_replies = [m for m in s.sent if isinstance(m, CacheStepResultMsg)]
    assert [(m.request_id, m.status) for m in rebuild_replies] == [("m1", "failed")]
    assert [(m.request_id, m.status) for m in step_replies] == [("s1", "failed")]
