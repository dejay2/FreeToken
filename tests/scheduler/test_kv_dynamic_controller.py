"""Rules 2, 4 and 8 of the dynamic KV pool spec on a fake clock: the drain barrier, the
concurrent-growth hold, same-batch uncommitted charging, Timer 1, and the failed outcome."""
from __future__ import annotations

from types import SimpleNamespace

from freetoken.scheduler.kv_dynamic import KVDynamicController, KVDynamicPolicy

PAGE, KV_PAGE, SLOT = 64, 13_248 * 64, 2_772_480


class _Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _controller(clock, floor=65_536, ceiling=262_144):
    policy = KVDynamicPolicy(
        floor_pages=floor // PAGE + 1, ceiling_pages=ceiling // PAGE + 1, step_pages=512,
        page_size=PAGE, kv_bytes_per_page=KV_PAGE, slot_bytes=SLOT, slot_floor=1024,
    )
    return KVDynamicController(policy, shrink_idle_s=600, instance_id="inst", clock=clock)


BUDGET = 7200 * SLOT + 1025 * KV_PAGE
POOL = 65_536


def _msg(uid):
    return SimpleNamespace(uid=uid)


def test_small_request_is_admitted_and_charged_as_uncommitted_until_reserved():
    c = _controller(_Clock())
    verdict = c.decide_admission(1, _msg(1), need_total=39_936, need_now=39_936,
                                 pool_tokens=POOL, fits_empty=True, fits_now=True)
    assert verdict == "admit"
    c.note_uncommitted(1, 39_936)
    assert c.uncommitted_tokens == 39_936
    c.note_reserved(1)
    assert c.uncommitted_tokens == 0


def test_request_that_cannot_fit_the_empty_pool_is_held_and_plans_a_grow():
    c = _controller(_Clock())
    assert c.decide_admission(2, _msg(2), need_total=92_000, need_now=92_000,
                              pool_tokens=POOL, fits_empty=False, fits_now=False) == "hold"
    plan = c.plan_idle(current_pages=1025, pool_budget_bytes=BUDGET, running_need_tokens=0)
    assert plan.reason == "grow" and plan.target_pages == 98_304 // PAGE + 1
    assert c.pop_held().uid == 2 and not c.has_held()


def test_drain_barrier_queues_later_arrivals_behind_a_held_request_in_order():
    c = _controller(_Clock())
    c.decide_admission(2, _msg(2), need_total=92_000, need_now=92_000,
                       pool_tokens=POOL, fits_empty=False, fits_now=False)
    # A tiny request that would fit is still queued: it must not overtake the held one.
    assert c.decide_admission(3, _msg(3), need_total=5_000, need_now=5_000,
                              pool_tokens=POOL, fits_empty=True, fits_now=True) == "hold"
    assert [h.uid for h in c.held] == [2, 3]


def test_concurrent_hold_uses_the_probe_verdict_and_the_running_sum():
    c = _controller(_Clock())
    assert c.decide_admission(4, _msg(4), need_total=72_000, need_now=72_000,
                              pool_tokens=131_072, fits_empty=True, fits_now=False) == "hold"
    plan = c.plan_idle(current_pages=131_072 // PAGE + 1, pool_budget_bytes=BUDGET,
                       running_need_tokens=142_000)
    assert plan.reason == "grow-concurrent" and (plan.target_pages - 1) * PAGE == 229_376


def test_concurrent_hold_does_not_fire_at_the_ceiling():
    c = _controller(_Clock())
    assert c.decide_admission(4, _msg(4), need_total=72_000, need_now=72_000,
                              pool_tokens=262_144, fits_empty=True, fits_now=False) == "admit"


def test_held_head_that_now_fits_is_admitted_without_a_plan():
    c = _controller(_Clock())
    c.decide_admission(2, _msg(2), need_total=92_000, need_now=92_000,
                       pool_tokens=POOL, fits_empty=False, fits_now=False)
    # The pool grew for someone else in the meantime.
    assert c.plan_idle(current_pages=98_304 // PAGE + 1, pool_budget_bytes=BUDGET,
                       running_need_tokens=0) is None
    assert c.pop_held().uid == 2


def test_abort_removes_a_held_request_and_its_uncommitted_charge():
    c = _controller(_Clock())
    c.decide_admission(2, _msg(2), need_total=92_000, need_now=92_000,
                       pool_tokens=POOL, fits_empty=False, fits_now=False)
    c.note_uncommitted(9, 1000)
    c.on_abort(2)
    c.on_abort(9)
    assert not c.has_held() and c.uncommitted_tokens == 0


def test_timer1_shrinks_after_the_idle_window_and_not_before():
    clock = _Clock()
    c = _controller(clock)
    c.on_request_finished()
    clock.now += 599
    assert c.plan_idle(current_pages=2049, pool_budget_bytes=BUDGET, running_need_tokens=0) is None
    assert c.next_deadline_ms(current_pages=2049) == 1000
    clock.now += 1
    plan = c.plan_idle(current_pages=2049, pool_budget_bytes=BUDGET, running_need_tokens=0)
    assert plan.reason == "shrink" and plan.target_pages == 1025


def test_no_deadline_at_the_floor_or_before_any_request_finished():
    clock = _Clock()
    c = _controller(clock)
    assert c.next_deadline_ms(current_pages=2049) is None  # nothing has finished yet
    c.on_request_finished()
    assert c.next_deadline_ms(current_pages=1025) is None  # already at the floor


def test_a_held_grow_wins_over_an_overdue_shrink():
    clock = _Clock()
    c = _controller(clock)
    c.on_request_finished()
    clock.now += 10_000
    c.decide_admission(2, _msg(2), need_total=150_000, need_now=150_000,
                       pool_tokens=131_072, fits_empty=False, fits_now=False)
    plan = c.plan_idle(current_pages=2049, pool_budget_bytes=BUDGET, running_need_tokens=0)
    assert plan.reason == "grow"  # one rebuild straight to the grow target, no shrink first


def test_operation_ids_are_unique():
    c = _controller(_Clock())
    a, b = c.next_operation_id(), c.next_operation_id()
    assert a != b and a.startswith("auto-kv:inst:")


def test_disable_returns_the_held_requests_and_admits_everything_afterwards():
    c = _controller(_Clock())
    c.decide_admission(2, _msg(2), need_total=92_000, need_now=92_000,
                       pool_tokens=POOL, fits_empty=False, fits_now=False)
    held = c.disable("rebuild failed")
    assert [h.uid for h in held] == [2] and not c.enabled
    assert c.decide_admission(5, _msg(5), need_total=92_000, need_now=92_000,
                              pool_tokens=POOL, fits_empty=False, fits_now=False) == "admit"
    assert c.plan_idle(current_pages=2049, pool_budget_bytes=BUDGET, running_need_tokens=0) is None


def test_status_reports_the_dials_and_the_queue():
    clock = _Clock()
    c = _controller(clock)
    c.on_request_finished()
    s = c.status(current_pages=1025, pool_budget_bytes=BUDGET)
    assert s["enabled"] and s["floor_tokens"] == 65_536 and s["ceiling_tokens"] == 262_144
    assert s["pool_tokens"] == 65_536 and s["held"] == 0 and s["shrink_in_s"] is None
    c.decide_admission(2, _msg(2), need_total=92_000, need_now=92_000,
                       pool_tokens=POOL, fits_empty=False, fits_now=False)
    assert c.status(current_pages=1025, pool_budget_bytes=BUDGET)["held"] == 1


def test_escalate_holds_a_pending_request_once_and_not_when_disabled():
    c = _controller(_Clock())
    # Escalate uid 7 with need_total 92_000 -> has_held() and queue contains [7]
    c.escalate(7, _msg(7), need_total=92_000)
    assert c.has_held() and [h.uid for h in c.held] == [7]
    # Escalate uid 7 again -> still exactly one entry (deduplication)
    c.escalate(7, _msg(7), need_total=92_000)
    assert [h.uid for h in c.held] == [7]
    # A small fitting request is held because the drain barrier is engaged
    assert c.decide_admission(3, _msg(3), need_total=5_000, need_now=5_000,
                              pool_tokens=POOL, fits_empty=True, fits_now=True) == "hold"
    # After disable, escalate does nothing
    c.disable("x")
    c.escalate(8, _msg(8), need_total=92_000)
    assert not c.has_held()


def test_timer1_with_a_budget_below_the_floor_geometry_plans_nothing():
    clock = _Clock()
    c = _controller(clock)
    c.on_request_finished()
    clock.now += 601  # past the shrink idle window
    # Budget too small to fit floor geometry
    plan = c.plan_idle(current_pages=2049, pool_budget_bytes=1025 * KV_PAGE + 100 * SLOT, running_need_tokens=0)
    assert plan is None and c.last_plan is None
