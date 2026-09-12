# Dynamic KV Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Boot Qwen3.8-Flash-Next with a small KV pool, grow it by trading MoE expert slots when an arriving request needs the room, shrink it back after a quiet spell, and expire parked prefixes from host RAM on a second timer, all through the existing idle-only rebuild path with no conversation lost.

**Architecture:** A CUDA-free policy + controller module (`scheduler/kv_dynamic.py`) decides sizes from one engine-owned byte budget (`Engine.pool_budget_bytes`). The scheduler holds requests that need growth in a FIFO (drain barrier), executes the controller's plans through `_pending_rebuild` at idle safe points, and announces each automatic operation to the API so the maintenance gate, status and watchdog see it. The park store gains a wall-clock TTL with its own wake-up; the engine orders pool shrinks before grows and validates against the shared budget.

**Tech Stack:** Python 3.11, PyTorch (CPU-testable fakes), FastAPI, ZMQ message dataclasses, pytest. Tests run on the GPU-less devbox with `PYTHONPATH=python .venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-09-12-dynamic-kv-pool-design.md` (commit 5e9e572). Read it first; every rule number below refers to its "Behaviour" section.

## Global Constraints

- Branch `mtp-upstream-merge`, push target `origin`; never force-push; commit subjects `type: subject` with type in `feat|fix|perf|refactor|build|ci|docs|test|chore`.
- Test command: `PYTHONPATH=python .venv/bin/python -m pytest <path> -q`. The devbox has no GPU; the baseline has pre-existing failures (281 across tests/engine + tests/kvcache + tests/scheduler at 4ff86b3). A task passes when its new tests pass and the failure set of its touched test files is unchanged.
- `freetoken.daemon` stays torch-free (`tests/daemon/` asserts it). `scheduler/kv_dynamic.py` must not import torch either.
- All admission quantities are **tokens**. `available_size`, `reserved_size` and `need_*` are tokens; a page count is multiplied by `page_size` before it meets them (spec rule 2).
- Byte costs for tests and docs: 13,248 B per KV token (fp8 QSA), 2,772,480 B per slot, page 64 tokens. Engine `num_pages` includes the dummy page: usable tokens = `(num_pages - 1) * page_size`; the live 262,208-token pool is 4,097 pages.
- Defaults: floor 65,536 usable tokens, step 32,768 (minimum 8,192), Timer 1 600 s, Timer 2 18,000 s, ceiling = existing `--num-tokens` (`KVCacheTokens`).
- Automatic operation ids are `auto-kv:{instance_id}:{seq}`; never descriptive.
- No mutating call on the admission probe path: `CacheManager.match_req` and `_lookup_parked` are forbidden there.
- Two independent fake clocks (wall and monotonic, different origins) in every TTL test.
- Comments carry measurements and reasons, matching the codebase convention.

---

## File structure

| File | Responsibility |
|---|---|
| Create `python/freetoken/scheduler/kv_dynamic.py` | Pure arithmetic (`KVDynamicPolicy`, `KVPlan`) and the request-holding controller (`KVDynamicController`). No torch. |
| Modify `python/freetoken/kvcache/park_store.py` | `ttl_s`, `sweep_expired()`, `next_expiry_delay_ms()`, status fields. Wall clock only. |
| Modify `python/freetoken/scheduler/cache.py` | `probe_admission()` (tree-only), `admission_fits()` shared helper, expiry folded into `next_park_delay_ms`, sweep in `park_idle`. |
| Modify `python/freetoken/scheduler/prefill.py` | Use `admission_fits()`; escalation callback for a never-started, capacity-blocked request; clear the controller's uncommitted entry on real reservation. |
| Modify `python/freetoken/engine/config.py`, `python/freetoken/server/args.py` | Fields and flags; floor/ceiling rewrite; validation. |
| Modify `python/freetoken/kvcache/base.py`, `python/freetoken/engine/engine.py` | `pool_budget_bytes` snapshot; shrink-first ordering; shared-budget validation; `step_memory` KV rung under the dynamic pool. |
| Modify `python/freetoken/message/tokenizer.py`, `message/frontend.py`, `message/__init__.py`, `python/freetoken/tokenizer/server.py` | `MaintenanceBeginMsg`/`Reply`, `KVDynamicStatusMsg`/`Reply` passthrough. |
| Modify `python/freetoken/server/api_server.py` | Per-id `maintenance_ops`; begin handling; `kv_dynamic` block in `/v1/cache/status`. |
| Modify `python/freetoken/scheduler/scheduler.py` | Controller wiring, held FIFO, outcome contract, idle deadline, finish hooks, external geometry. |
| Modify `python/freetoken/daemon/settings/dials.py`, `linux_launch.py`, `app.py`, `static/index.html`, `python/freetoken/engine/memory_plan.py` | Six dials, launch mapping, cross-field validation, ceiling-reachable warning, status tile, helper version. |
| Modify `docs/cli.md`, `README.md` | Flag rows and one fork paragraph. |

---

### Task 1: Pure sizing policy

**Files:**
- Create: `python/freetoken/scheduler/kv_dynamic.py`
- Test: `tests/scheduler/test_kv_dynamic_policy.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class KVPlan:
      target_pages: int      # engine num_pages (dummy included)
      target_slots: int
      capped: bool           # slot floor limited the grow
      reason: str            # "grow" | "grow-concurrent" | "shrink"

  @dataclass(frozen=True)
  class KVDynamicPolicy:
      floor_pages: int; ceiling_pages: int; step_pages: int
      page_size: int; kv_bytes_per_page: int; slot_bytes: int; slot_floor: int
      def usable_tokens(self, pages: int) -> int
      def pages_for_usable(self, tokens: int) -> int
      def slots_for_pages(self, pool_budget_bytes: int, pages: int) -> int
      def plan_grow(self, *, current_pages, pool_budget_bytes, need_tokens, running_need_tokens=0, reason="grow") -> KVPlan | None
      def plan_shrink(self, *, current_pages, pool_budget_bytes) -> KVPlan | None
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/scheduler/test_kv_dynamic_policy.py
"""Rule 3 of the dynamic KV pool spec: every plan stays inside pool_budget_bytes, slots fall
monotonically as pages grow, and rounding slack never accumulates. Costs are the live 5090
fp8 numbers (13,248 B/token, 2,772,480 B/slot, 64-token pages)."""
from __future__ import annotations

import pytest

from freetoken.scheduler.kv_dynamic import KVDynamicPolicy, KVPlan

PAGE = 64
KV_PAGE = 13_248 * PAGE
SLOT = 2_772_480


def _policy(floor=65_536, ceiling=262_144, step=32_768, slot_floor=1024) -> KVDynamicPolicy:
    return KVDynamicPolicy(
        floor_pages=floor // PAGE + 1,
        ceiling_pages=ceiling // PAGE + 1,
        step_pages=step // PAGE,
        page_size=PAGE,
        kv_bytes_per_page=KV_PAGE,
        slot_bytes=SLOT,
        slot_floor=slot_floor,
    )


def _budget(policy: KVDynamicPolicy, slots_at_floor: int) -> int:
    return slots_at_floor * SLOT + policy.floor_pages * KV_PAGE


def test_pages_and_tokens_round_trip_with_the_dummy_page():
    p = _policy()
    assert p.floor_pages == 1025 and p.ceiling_pages == 4097  # the live 262,208-token pool
    assert p.usable_tokens(4097) == 262_144
    assert p.pages_for_usable(262_144) == 4097
    assert p.pages_for_usable(1) == 2


def test_every_reachable_geometry_stays_inside_the_budget_and_slots_are_monotone():
    p = _policy()
    budget = _budget(p, 7200)
    prev = None
    for pages in range(p.floor_pages, p.ceiling_pages + 1):
        slots = p.slots_for_pages(budget, pages)
        assert slots * SLOT + pages * KV_PAGE <= budget
        assert budget - (slots * SLOT + pages * KV_PAGE) < SLOT  # slack below one slot
        if prev is not None:
            assert slots <= prev
        prev = slots
    assert p.slots_for_pages(budget, p.floor_pages) == 7200


def test_one_step_grow_releases_about_157_slots_and_may_raise_the_total_by_under_a_slot():
    p = _policy()
    budget = _budget(p, 7200)
    plan = p.plan_grow(current_pages=p.floor_pages, pool_budget_bytes=budget, need_tokens=60_000 + 32_768)
    assert plan == KVPlan(target_pages=98_304 // PAGE + 1, target_slots=7200 - 157, capped=False, reason="grow")
    before = 7200 * SLOT + p.floor_pages * KV_PAGE
    after = plan.target_slots * SLOT + plan.target_pages * KV_PAGE
    assert after <= budget and abs(after - before) < SLOT


def test_grow_jumps_straight_to_the_rung_that_fits_and_clamps_at_the_ceiling():
    p = _policy()
    budget = _budget(p, 7200)
    big = p.plan_grow(current_pages=p.floor_pages, pool_budget_bytes=budget, need_tokens=200_000 + 32_768)
    assert p.usable_tokens(big.target_pages) == 262_144  # 232,768 rounds to 262,144 then clamps
    assert big.target_slots == p.slots_for_pages(budget, big.target_pages)
    at_top = p.plan_grow(current_pages=p.ceiling_pages, pool_budget_bytes=budget, need_tokens=300_000)
    assert at_top is None


def test_grow_returns_none_when_the_pool_already_holds_the_need():
    p = _policy()
    assert p.plan_grow(current_pages=p.floor_pages, pool_budget_bytes=_budget(p, 7200), need_tokens=39_000) is None


def test_exact_rung_landing_still_moves_one_step():
    p = _policy()
    plan = p.plan_grow(current_pages=p.floor_pages, pool_budget_bytes=_budget(p, 7200), need_tokens=65_536)
    assert p.usable_tokens(plan.target_pages) == 98_304


def test_concurrent_target_uses_the_running_sum():
    p = _policy()
    plan = p.plan_grow(
        current_pages=131_072 // PAGE + 1, pool_budget_bytes=_budget(p, 7200),
        need_tokens=72_000, running_need_tokens=142_000, reason="grow-concurrent",
    )
    assert p.usable_tokens(plan.target_pages) == 229_376 and plan.reason == "grow-concurrent"


def test_slot_floor_caps_the_grow_and_flags_it():
    p = _policy(slot_floor=1024)
    budget = _budget(p, 1300)  # only 276 slots above the floor: 1.5 GB, about 3.7 steps
    plan = p.plan_grow(current_pages=p.floor_pages, pool_budget_bytes=budget, need_tokens=250_000)
    assert plan.capped and plan.target_slots == 1024
    assert plan.target_slots * SLOT + plan.target_pages * KV_PAGE <= budget
    assert plan.target_pages < p.ceiling_pages


def test_capped_grow_that_cannot_move_a_page_returns_none():
    p = _policy(slot_floor=1024)
    budget = _budget(p, 1024)
    assert p.plan_grow(current_pages=p.floor_pages, pool_budget_bytes=budget, need_tokens=250_000) is None


def test_shrink_returns_to_the_floor_and_the_floor_slot_count():
    p = _policy()
    budget = _budget(p, 7200)
    plan = p.plan_shrink(current_pages=229_376 // PAGE + 1, pool_budget_bytes=budget)
    assert plan == KVPlan(target_pages=p.floor_pages, target_slots=7200, capped=False, reason="shrink")
    assert p.plan_shrink(current_pages=p.floor_pages, pool_budget_bytes=budget) is None


def test_budget_lowered_by_the_governor_lowers_every_target():
    p = _policy()
    high, low = _budget(p, 7200), _budget(p, 7200) - 512 * SLOT
    assert p.slots_for_pages(low, p.floor_pages) == 7200 - 512
    assert p.slots_for_pages(low, 2049) == p.slots_for_pages(high, 2049) - 512


@pytest.mark.parametrize("bad", [
    dict(floor_pages=0), dict(step_pages=0), dict(slot_floor=0), dict(kv_bytes_per_page=0),
    dict(floor_pages=5000),  # above the ceiling
    dict(step_pages=64),  # 4,096 tokens, under the 8,192 minimum
])
def test_invalid_policies_are_refused(bad):
    kwargs = dict(
        floor_pages=1025, ceiling_pages=4097, step_pages=512, page_size=PAGE,
        kv_bytes_per_page=KV_PAGE, slot_bytes=SLOT, slot_floor=1024,
    )
    kwargs.update(bad)
    with pytest.raises(ValueError):
        KVDynamicPolicy(**kwargs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_policy.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'freetoken.scheduler.kv_dynamic'`

- [ ] **Step 3: Write the policy**

```python
# python/freetoken/scheduler/kv_dynamic.py
"""Dynamic KV pool: pure sizing arithmetic and the request-holding controller.

Spec: docs/superpowers/specs/2026-09-12-dynamic-kv-pool-design.md. Everything here is
CUDA-free and torch-free so the boundary cases are unit-tested on the GPU-less devbox.

Units. The engine's ``num_pages`` includes the dummy page 0, so the usable pool is
``(num_pages - 1) * page_size`` tokens; the live 262,208-token pool is 4,097 pages of 64.
Bytes come from one engine-owned budget, ``pool_budget_bytes = slots * slot_bytes +
num_pages * kv_bytes_per_page``, snapshotted at boot and after every rebuild this module did
not issue. Floor division keeps every plan inside that budget with unused slack below one
slot (2.77 MB on the 5090); a 32,768-token step is 156.6 slots, so one grow releases 156 or
157 slots and the resident total may rise by up to one slot's bytes while staying inside the
budget (external review of 702543b, 2026-09-12).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque

MIN_STEP_TOKENS = 8_192


@dataclass(frozen=True)
class KVPlan:
    target_pages: int
    target_slots: int
    capped: bool
    reason: str  # "grow" | "grow-concurrent" | "shrink"


@dataclass(frozen=True)
class KVDynamicPolicy:
    floor_pages: int
    ceiling_pages: int
    step_pages: int
    page_size: int
    kv_bytes_per_page: int
    slot_bytes: int
    slot_floor: int

    def __post_init__(self) -> None:
        for name in (
            "floor_pages", "ceiling_pages", "step_pages", "page_size",
            "kv_bytes_per_page", "slot_bytes", "slot_floor",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.floor_pages > self.ceiling_pages:
            raise ValueError(
                f"floor_pages {self.floor_pages} is above ceiling_pages {self.ceiling_pages}"
            )
        if self.step_pages * self.page_size < MIN_STEP_TOKENS:
            raise ValueError(
                f"step must be at least {MIN_STEP_TOKENS} tokens "
                f"(got {self.step_pages * self.page_size})"
            )

    # ---- units -------------------------------------------------------------------------
    def usable_tokens(self, pages: int) -> int:
        return max(0, pages - 1) * self.page_size

    def pages_for_usable(self, tokens: int) -> int:
        return -(-max(tokens, 0) // self.page_size) + 1

    # ---- budget ------------------------------------------------------------------------
    def slots_for_pages(self, pool_budget_bytes: int, pages: int) -> int:
        """Slots the budget funds beside ``pages`` (floor division: always inside the budget)."""
        remaining = pool_budget_bytes - pages * self.kv_bytes_per_page
        return max(self.slot_floor, remaining // self.slot_bytes)

    def _fits_budget(self, pool_budget_bytes: int, pages: int, slots: int) -> bool:
        return slots * self.slot_bytes + pages * self.kv_bytes_per_page <= pool_budget_bytes

    # ---- plans -------------------------------------------------------------------------
    def plan_grow(
        self,
        *,
        current_pages: int,
        pool_budget_bytes: int,
        need_tokens: int,
        running_need_tokens: int = 0,
        reason: str = "grow",
    ) -> KVPlan | None:
        """Grow to the smallest rung holding ``running_need_tokens + need_tokens``.

        At least one step (PR #300's rung rule, with +1 token of breathing room when the need
        lands exactly on a rung); clamped at the ceiling; slots funded from the budget. When
        the slot floor stops the byte-neutral target, the pool grows only as far as the floor
        funds (``capped``), or not at all if that is no further than today.
        """
        current_usable = self.usable_tokens(current_pages)
        step = self.step_pages * self.page_size
        ceiling_usable = self.usable_tokens(self.ceiling_pages)
        total_need = running_need_tokens + need_tokens
        if total_need < current_usable or current_usable >= ceiling_usable:
            return None
        required_rung = -(-(total_need + 1) // step) * step
        target_usable = min(ceiling_usable, max(current_usable + step, required_rung))
        target_pages = self.pages_for_usable(target_usable)
        if target_pages <= current_pages:
            return None
        target_slots = self.slots_for_pages(pool_budget_bytes, target_pages)
        capped = False
        if not self._fits_budget(pool_budget_bytes, target_pages, target_slots):
            # The floor is binding: how many pages does the floor leave room for?
            affordable = (pool_budget_bytes - self.slot_floor * self.slot_bytes) // self.kv_bytes_per_page
            target_pages = min(target_pages, affordable)
            target_slots = self.slot_floor
            capped = True
            if target_pages <= current_pages:
                return None
        return KVPlan(target_pages=target_pages, target_slots=target_slots, capped=capped, reason=reason)

    def plan_shrink(self, *, current_pages: int, pool_budget_bytes: int) -> KVPlan | None:
        if current_pages <= self.floor_pages:
            return None
        return KVPlan(
            target_pages=self.floor_pages,
            target_slots=self.slots_for_pages(pool_budget_bytes, self.floor_pages),
            capped=False,
            reason="shrink",
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_policy.py -q`
Expected: all PASS. If `test_capped_grow_that_cannot_move_a_page_returns_none` fails, check that `affordable` is computed before comparing to `current_pages`.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/scheduler/kv_dynamic.py tests/scheduler/test_kv_dynamic_policy.py
git commit -m "feat(scheduler): pure sizing policy for the dynamic KV pool"
```

---

### Task 2: Request-holding controller

**Files:**
- Modify: `python/freetoken/scheduler/kv_dynamic.py` (append)
- Test: `tests/scheduler/test_kv_dynamic_controller.py`

**Interfaces:**
- Consumes: `KVDynamicPolicy`, `KVPlan` from Task 1.
- Produces:
  ```python
  @dataclass
  class HeldRequest: uid: int; msg: object; need_total: int; held_at: float

  class KVDynamicController:
      def __init__(self, policy, *, shrink_idle_s: int, instance_id: str, clock: Callable[[], float] = time.monotonic)
      enabled: bool
      def decide_admission(self, uid, msg, *, need_total, need_now, pool_tokens, fits_empty, fits_now) -> str   # "admit" | "hold"
      def note_uncommitted(self, uid, need_now) -> None; def note_reserved(self, uid) -> None
      uncommitted_tokens: int (property)
      def escalate(self, uid, msg, need_total) -> None            # rule 2(b)
      def on_abort(self, uid) -> None
      def on_request_finished(self, now: float | None = None) -> None
      def has_held(self) -> bool; def pop_held(self) -> HeldRequest | None
      def plan_idle(self, *, current_pages, pool_budget_bytes, running_need_tokens, now=None) -> KVPlan | None
      def next_deadline_ms(self, *, current_pages, now=None) -> int | None
      def next_operation_id(self) -> str                            # "auto-kv:{instance_id}:{seq}"
      def disable(self, reason: str) -> list[HeldRequest]           # rule 8 "failed"
      def status(self, *, current_pages, pool_budget_bytes) -> dict
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/scheduler/test_kv_dynamic_controller.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_controller.py -q`
Expected: FAIL with `ImportError: cannot import name 'KVDynamicController'`

- [ ] **Step 3: Append the controller to `kv_dynamic.py`**

```python
# append to python/freetoken/scheduler/kv_dynamic.py
import time


@dataclass
class HeldRequest:
    uid: int
    msg: object
    need_total: int
    held_at: float


class KVDynamicController:
    """Decides when the scheduler holds a request for growth and when it shrinks (rules 2, 4, 8).

    The scheduler owns every GPU action; this object only keeps the held FIFO, the same-batch
    ``uncommitted`` charges, the Timer 1 clock and the operation counter. Torch-free.
    """

    def __init__(
        self,
        policy: KVDynamicPolicy,
        *,
        shrink_idle_s: int,
        instance_id: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if shrink_idle_s <= 0:
            raise ValueError("shrink_idle_s must be positive")
        self.policy = policy
        self.shrink_idle_s = int(shrink_idle_s)
        self.instance_id = str(instance_id)
        self._clock = clock
        self.enabled = True
        self.disabled_reason: str | None = None
        self.held: Deque[HeldRequest] = deque()
        self._uncommitted: dict[int, int] = {}
        self.last_request_finished: float | None = None
        self._seq = 0
        self.last_plan: dict | None = None

    # ---- admission (rule 2) ------------------------------------------------------------
    def decide_admission(
        self, uid: int, msg: object, *, need_total: int, need_now: int, pool_tokens: int,
        fits_empty: bool, fits_now: bool,
    ) -> str:
        if not self.enabled:
            return "admit"
        ceiling_tokens = self.policy.usable_tokens(self.policy.ceiling_pages)
        if self.held:
            # Drain barrier: once anyone waits for growth, later arrivals queue behind it so a
            # stream of small requests can never keep the scheduler busy and starve the big one.
            self._hold(uid, msg, need_total)
            return "hold"
        if need_total > pool_tokens:
            self._hold(uid, msg, need_total)
            return "hold"
        if fits_empty and not fits_now and pool_tokens < ceiling_tokens:
            # It fits an empty pool but other requests hold the room: wait for them once, then
            # grow so the pair runs side by side next time.
            self._hold(uid, msg, need_total)
            return "hold"
        return "admit"

    def _hold(self, uid: int, msg: object, need_total: int) -> None:
        self.held.append(HeldRequest(uid=uid, msg=msg, need_total=need_total, held_at=self._clock()))

    def escalate(self, uid: int, msg: object, need_total: int) -> None:
        """Rule 2(b): a never-started pending request the prefill manager could not seat."""
        if self.enabled and all(h.uid != uid for h in self.held):
            self._hold(uid, msg, need_total)

    # ---- same-batch charging (rule 2, "Same-batch arrivals") ----------------------------
    def note_uncommitted(self, uid: int, need_now: int) -> None:
        self._uncommitted[uid] = int(need_now)

    def note_reserved(self, uid: int) -> None:
        self._uncommitted.pop(uid, None)

    @property
    def uncommitted_tokens(self) -> int:
        return sum(self._uncommitted.values())

    def on_abort(self, uid: int) -> None:
        self._uncommitted.pop(uid, None)
        self.held = deque(h for h in self.held if h.uid != uid)

    # ---- lifecycle -----------------------------------------------------------------------
    def on_request_finished(self, now: float | None = None) -> None:
        self.last_request_finished = self._clock() if now is None else now

    def has_held(self) -> bool:
        return bool(self.held)

    def pop_held(self) -> HeldRequest | None:
        return self.held.popleft() if self.held else None

    def plan_idle(
        self, *, current_pages: int, pool_budget_bytes: int, running_need_tokens: int,
        now: float | None = None,
    ) -> KVPlan | None:
        """The one rebuild (if any) to run at this idle point: the held head's grow first,
        else Timer 1's shrink. A held grow always wins over an overdue shrink so an arrival
        after a quiet spell costs one rebuild, never two."""
        if not self.enabled:
            return None
        if self.held:
            head = self.held[0]
            reason = "grow-concurrent" if running_need_tokens else "grow"
            plan = self.policy.plan_grow(
                current_pages=current_pages, pool_budget_bytes=pool_budget_bytes,
                need_tokens=head.need_total, running_need_tokens=running_need_tokens, reason=reason,
            )
            self._remember(plan)
            return plan
        now = self._clock() if now is None else now
        if (
            self.last_request_finished is not None
            and now - self.last_request_finished >= self.shrink_idle_s
        ):
            plan = self.policy.plan_shrink(current_pages=current_pages, pool_budget_bytes=pool_budget_bytes)
            self._remember(plan)
            return plan
        return None

    def _remember(self, plan: KVPlan | None) -> None:
        if plan is not None:
            self.last_plan = {
                "reason": plan.reason,
                "target_tokens": self.policy.usable_tokens(plan.target_pages),
                "target_slots": plan.target_slots,
                "capped": plan.capped,
                "at": self._clock(),
            }

    def next_deadline_ms(self, *, current_pages: int, now: float | None = None) -> int | None:
        """Milliseconds until Timer 1 is due, or None when no shrink can be pending."""
        if not self.enabled or self.last_request_finished is None:
            return None
        if current_pages <= self.policy.floor_pages:
            return None
        now = self._clock() if now is None else now
        remaining = self.shrink_idle_s - (now - self.last_request_finished)
        return max(1, int(remaining * 1000 + 0.999))

    def next_operation_id(self) -> str:
        self._seq += 1
        return f"auto-kv:{self.instance_id}:{self._seq}"

    def disable(self, reason: str) -> list[HeldRequest]:
        """Rule 8 ``failed``: stop planning for the life of the process and hand back the
        held requests so the scheduler can error-reply them."""
        self.enabled = False
        self.disabled_reason = reason
        held = list(self.held)
        self.held.clear()
        self._uncommitted.clear()
        return held

    def status(self, *, current_pages: int, pool_budget_bytes: int) -> dict:
        deadline = self.next_deadline_ms(current_pages=current_pages)
        return {
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "floor_tokens": self.policy.usable_tokens(self.policy.floor_pages),
            "ceiling_tokens": self.policy.usable_tokens(self.policy.ceiling_pages),
            "step_tokens": self.policy.step_pages * self.policy.page_size,
            "pool_tokens": self.policy.usable_tokens(current_pages),
            "pool_budget_bytes": int(pool_budget_bytes),
            "slots_at_floor": self.policy.slots_for_pages(pool_budget_bytes, self.policy.floor_pages),
            "held": len(self.held),
            "uncommitted_tokens": self.uncommitted_tokens,
            "shrink_in_s": None if deadline is None else round(deadline / 1000, 1),
            "last_plan": self.last_plan,
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_policy.py tests/scheduler/test_kv_dynamic_controller.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/scheduler/kv_dynamic.py tests/scheduler/test_kv_dynamic_controller.py
git commit -m "feat(scheduler): request-holding controller for the dynamic KV pool"
```

---

### Task 3: Park store TTL with a wall-clock expiry delay

**Files:**
- Modify: `python/freetoken/kvcache/park_store.py` (constructor at :525-560, `from_config` at :655-690, `status` at :2406)
- Test: `tests/kvcache/test_park_store.py` (append)

**Interfaces:**
- Produces on `ParkStore`: `ttl_s: int` (0 = never), `sweep_expired(self) -> int` (families evicted), `next_expiry_delay_ms(self) -> int | None` (relative, wall-clock arithmetic inside), `status()` gains `ttl_s`, `expired_evictions`, `next_expiry_s`. Both new methods call `self._wall_ns()` (defaults to `time.time_ns`) so tests can inject a fake wall clock without touching the monotonic clock.

- [ ] **Step 1: Write the failing tests**

Find the existing RAM-mode fixture in `tests/kvcache/test_park_store.py` (a `ParkStore(mode="ram", ...)` construction near the top; `_qsa_pool`/`_state_pool` helpers exist at lines 32-60). Append:

```python
# append to tests/kvcache/test_park_store.py


def _ram_store(ttl_s: int, wall_ns) -> ParkStore:
    store = ParkStore(
        mode="ram", page_size=4, kv_pool=_qsa_pool(), state_pool=_state_pool(),
        fingerprint="fp", min_tokens=4, ram_budget_bytes=1 << 30, ssd_dir="/nonexistent",
        disk_budget_bytes=0, pinned_window_bytes=4096, idle_ms=0, ttl_s=ttl_s,
    )
    store._wall_ns = wall_ns  # the test's wall clock; the scheduler's monotonic clock is never used
    return store


def _seed_family(store: ParkStore, key: str, last_used_ns: int) -> None:
    """Insert one already-published RAM entry directly, bypassing the save worker."""
    entry = park_module.ParkedEntry(
        key=key, token_count=8, total_bytes=1024, last_used_ns=last_used_ns,
        **{name: field.default for name, field in park_module.ParkedEntry.__dataclass_fields__.items()
           if name not in {"key", "token_count", "total_bytes", "last_used_ns"} and field.default is not field.default_factory},
    )
    with store._lock:
        store._entries[key] = entry


def test_ttl_expiry_uses_the_wall_clock_and_two_clocks_with_different_origins():
    """The store stamps last_used_ns with time.time_ns(); the scheduler's idle loop runs on
    time.monotonic_ns(). Ages must never mix the two (review of 702543b): a five-hour entry
    must expire after six wall-clock hours whatever the monotonic clock says."""
    wall = {"now": 1_700_000_000 * 10**9}          # epoch-based, 2023
    monotonic_origin = 12_345 * 10**9               # seconds since boot, unrelated
    store = _ram_store(ttl_s=5 * 3600, wall_ns=lambda: wall["now"])
    _seed_family(store, "old", last_used_ns=wall["now"])
    assert store.next_expiry_delay_ms() == 5 * 3600 * 1000
    wall["now"] += 4 * 3600 * 10**9
    assert store.sweep_expired() == 0 and store.status()["parked_count"] == 1
    wall["now"] += 2 * 3600 * 10**9
    assert store.next_expiry_delay_ms() == 0
    assert store.sweep_expired() == 1
    assert store.status()["parked_count"] == 0 and store.status()["expired_evictions"] == 1
    # A monotonic timestamp handed in by mistake would be years "younger" than any entry:
    # the API takes no now_ns argument at all, so the mistake cannot be made.
    assert monotonic_origin < wall["now"]


def test_ttl_zero_never_expires_and_reports_no_deadline():
    wall = {"now": 1_700_000_000 * 10**9}
    store = _ram_store(ttl_s=0, wall_ns=lambda: wall["now"])
    _seed_family(store, "k", last_used_ns=wall["now"] - 10**15)
    assert store.next_expiry_delay_ms() is None
    assert store.sweep_expired() == 0
    assert store.status()["ttl_s"] == 0 and store.status()["next_expiry_s"] is None


def test_expiry_evicts_the_oldest_family_first_and_skips_pinned_ones():
    wall = {"now": 1_700_000_000 * 10**9}
    store = _ram_store(ttl_s=3600, wall_ns=lambda: wall["now"])
    _seed_family(store, "a", last_used_ns=wall["now"] - 7200 * 10**9)
    _seed_family(store, "b", last_used_ns=wall["now"] - 100 * 10**9)
    _seed_family(store, "pinned", last_used_ns=wall["now"] - 7200 * 10**9)
    store._pins["pinned"] = 1
    assert store.sweep_expired() == 1
    assert set(store._entries) == {"b", "pinned"}
    assert store.next_expiry_delay_ms() == 3500 * 1000
```

If `ParkedEntry` has required fields beyond the four named, adjust `_seed_family` to pass them explicitly (read the dataclass at `park_store.py` around line 160-200); the point of the helper is one published entry with a chosen `last_used_ns`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/kvcache/test_park_store.py -q -k "ttl or expiry"`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'ttl_s'`

- [ ] **Step 3: Implement**

In `ParkStore.__init__` (signature at `park_store.py:525-538`) add `ttl_s: int = 0` after `idle_ms`, validate `if ttl_s < 0: raise ValueError("park ttl_s must be non-negative")`, and store:

```python
        self.ttl_s = int(ttl_s)
        self._expired_evictions = 0
        # Wall clock for age arithmetic: every last_used_ns in this store (RAM saves, lookups,
        # the SSD index's st_mtime_ns) is time.time_ns(). The scheduler's idle loop runs on
        # time.monotonic_ns(), whose origin is unrelated, so this store never accepts a caller's
        # now_ns for ages: it reads its own clock and hands back RELATIVE delays.
        self._wall_ns = time.time_ns
```

Add the two methods next to `_evict_to_fit` (`park_store.py:1249`):

```python
    def _family_ages(self) -> list[tuple[int, str, list["ParkedEntry"]]]:
        """(newest last_used_ns, root_key, members) per unpinned family."""
        families: dict[str, list[ParkedEntry]] = {}
        for entry in self._entries.values():
            families.setdefault(entry.root_key or entry.key, []).append(entry)
        return [
            (max(m.last_used_ns for m in members), root, members)
            for root, members in families.items()
            if not self._pins.get(root, 0)
        ]

    def sweep_expired(self) -> int:
        """Drop every unpinned family whose newest use is older than ``ttl_s`` (Timer 2)."""
        if self.ttl_s <= 0:
            return 0
        cutoff = self._wall_ns() - self.ttl_s * 1_000_000_000
        evicted = 0
        with self._lock:
            for newest, _root, members in self._family_ages():
                if newest <= cutoff:
                    self._drop_entries(members, write_manifest=False)
                    evicted += 1
            if evicted:
                self._expired_evictions += evicted
                self._notify_change()
                if self.mode == "ssd":
                    self._write_manifest()
        return evicted

    def next_expiry_delay_ms(self) -> int | None:
        """Milliseconds until the oldest unpinned family expires; None when nothing can."""
        if self.ttl_s <= 0:
            return None
        with self._lock:
            ages = self._family_ages()
        if not ages:
            return None
        oldest = min(newest for newest, _root, _members in ages)
        remaining_ns = oldest + self.ttl_s * 1_000_000_000 - self._wall_ns()
        return max(0, (remaining_ns + 999_999) // 1_000_000)
```

In `status()` add `"ttl_s": self.ttl_s, "expired_evictions": self._expired_evictions,` and `"next_expiry_s": (None if (d := self.next_expiry_delay_ms()) is None else round(d / 1000, 1))` (compute `d` before taking `self._lock` again, since `next_expiry_delay_ms` locks; `RLock` makes re-entry safe but keep it simple: call it first). In `from_config` pass `ttl_s=int(getattr(config, "kv_park_ttl_s", 0))`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/kvcache/test_park_store.py -q`
Expected: the new tests PASS; the pre-existing pass/fail set of the file is unchanged (record it before the change with the same command).

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/kvcache/park_store.py tests/kvcache/test_park_store.py
git commit -m "feat(kvcache): park store TTL sweep with a wall-clock relative expiry delay"
```

---

### Task 4: Non-mutating admission probe and the expiry wake-up in CacheManager

**Files:**
- Modify: `python/freetoken/scheduler/cache.py` (`available_size` at :274, `park_idle` at :362, `next_park_delay_ms` at :379)
- Modify: `python/freetoken/scheduler/prefill.py` (`_try_allocate_one` :77-99)
- Test: `tests/scheduler/test_kv_dynamic_probe.py`

**Interfaces:**
- Produces in `scheduler/cache.py`:
  ```python
  def admission_fits(*, need_now: int, reserved: int, available: int, protect_tokens: int) -> bool
  @dataclass(frozen=True)
  class AdmissionProbe: need_now: int; protect_tokens: int; fits_empty: bool; fits_now: bool; cached_len: int
  CacheManager.probe_admission(self, input_ids, output_len: int, *, reserved: int, cache_private: bool = False) -> AdmissionProbe
  CacheManager.next_park_delay_ms(...)   # now also folds park_store.next_expiry_delay_ms()
  CacheManager.park_idle(...)            # now also calls park_store.sweep_expired()
  ```
- Consumes: Task 3's `next_expiry_delay_ms`, `sweep_expired`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/scheduler/test_kv_dynamic_probe.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_probe.py -q`
Expected: FAIL with `ImportError: cannot import name 'AdmissionProbe'`

- [ ] **Step 3: Implement the probe and the shared check**

At module level in `scheduler/cache.py` (after the imports):

```python
from dataclasses import dataclass


def admission_fits(*, need_now: int, reserved: int, available: int, protect_tokens: int) -> bool:
    """The ONE admission inequality, in tokens. ``protect_tokens`` is the matched prefix's
    evictable length that locking will remove from ``available`` (PrefillAdder checks
    before and after lock for exactly this reason, prefill.py:96-99). Shared by the real
    admission and the dynamic KV pool's probe so they can never disagree."""
    return need_now + reserved <= available - protect_tokens


@dataclass(frozen=True)
class AdmissionProbe:
    need_now: int
    protect_tokens: int
    fits_empty: bool
    fits_now: bool
    cached_len: int
```

Add to `CacheManager` (after `available_size`):

```python
    def _evictable_tokens_on_path(self, node) -> int:
        """Tokens of the matched path that are evictable today (ref_count 0) and would become
        protected by inc_lock. Mirrors HybridRadixCache.inc_lock's walk without mutating."""
        total = 0
        cur = node
        while cur is not None and not cur.is_root():
            if getattr(cur, "ref_count", 0) == 0:
                total += cur.length
            cur = cur.parent
        return total

    def probe_admission(
        self, input_ids: torch.Tensor, output_len: int, *, reserved: int, cache_private: bool = False
    ) -> AdmissionProbe:
        """Rule 2 of the dynamic KV pool spec: what admission WOULD say, without doing it.

        Walks the radix tree only. Never calls match_req: that path looks up parked prefixes
        and may restore one (allocating pages and a GDN slot) as a side effect. A parked but
        non-resident prefix therefore counts as not cached here, which is right for capacity:
        its restore would allocate the whole prefix again."""
        input_len = int(len(input_ids))
        ids = input_ids[:0] if cache_private else input_ids[: max(input_len - 1, 0)]
        m = self.prefix_cache.match_prefix(ids) if input_len > 0 else None
        cached_len = int(m.cached_len) if m is not None else 0
        node = getattr(m, "node", None)
        protect_tokens = self._evictable_tokens_on_path(node) if node is not None else 0
        need_now = (input_len - cached_len) + int(output_len)
        empty_limit = self.num_pages * self.page_size
        fits_empty = input_len + int(output_len) <= empty_limit
        fits_now = fits_empty and admission_fits(
            need_now=need_now, reserved=int(reserved), available=self.available_size,
            protect_tokens=protect_tokens,
        )
        return AdmissionProbe(
            need_now=need_now, protect_tokens=protect_tokens,
            fits_empty=fits_empty, fits_now=fits_now, cached_len=cached_len,
        )
```

For the non-hybrid radix cache `match_prefix` returns an object with `cached_len` and `node` too (check `kvcache/radix_cache.py`; if its match result lacks `node`, `getattr` above yields 0 protect tokens, which is the pre-lock answer and acceptable for non-hybrid models the feature does not target).

In `next_park_delay_ms` (`cache.py:379`) fold the expiry:

```python
    def next_park_delay_ms(self, *, now_ns: int | None = None) -> int | None:
        """Milliseconds until idle parking, a pending-copy poll, or the next TTL expiry;
        ``None`` can block forever. The expiry delay is RELATIVE and computed inside the
        store on its own wall clock (Timer 2); ``now_ns`` here is monotonic and only ever
        meets the tree's monotonic candidate timestamps."""
        if self.park_store is None:
            return None
        expiry = getattr(self.park_store, "next_expiry_delay_ms", lambda: None)()
        if self._pending_parks:
            return 10 if expiry is None else min(10, max(1, expiry))
        candidates = self._park_candidates()
        timestamp = min((candidate.timestamp for candidate in candidates), default=None)
        if timestamp is None:
            return None if expiry is None else max(1, int(expiry))
        if now_ns is None:
            import time

            now_ns = time.monotonic_ns()
        remaining = self.park_store.idle_ms * 1_000_000 - (now_ns - timestamp)
        delay = max(1, (remaining + 999_999) // 1_000_000)
        return delay if expiry is None else min(delay, max(1, int(expiry)))
```

In `park_idle` (`cache.py:362`), before `return parked`, add:

```python
        sweep = getattr(self.park_store, "sweep_expired", None)
        if sweep is not None and sweep():
            self._bump_park_generation()
```

Keep the existing early return for `self.park_store is None or self._temporary_lease_depth`.

In `prefill.py:_try_allocate_one`, replace the two inequalities at lines 96-99 with the shared function so there is one formula:

```python
        from .cache import admission_fits

        if not admission_fits(need_now=estimated_len, reserved=self.reserved_size,
                              available=self.cache_manager.available_size, protect_tokens=0):
            return None
        self.cache_manager.lock(handle)
        if not admission_fits(need_now=estimated_len, reserved=self.reserved_size,
                              available=self.cache_manager.available_size, protect_tokens=0):
            return self.cache_manager.unlock(handle)
```

(The real path locks and re-reads `available_size`, so `protect_tokens=0` there; the probe substitutes the arithmetic for the lock.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_probe.py tests/scheduler/test_park_admission.py tests/scheduler/test_kv_parking.py -q`
Expected: new tests PASS; the two existing files keep their prior pass/fail set. If `test_probe_counts_an_already_locked_prefix_as_free_room` fails on the `PendingReq` construction, build the request the way `tests/scheduler/test_park_admission.py` does and keep the assertion.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/scheduler/cache.py python/freetoken/scheduler/prefill.py tests/scheduler/test_kv_dynamic_probe.py
git commit -m "feat(scheduler): tree-only admission probe and TTL expiry wake-up in the cache manager"
```

---

### Task 5: Prefill escalation hook and reservation hand-off

**Files:**
- Modify: `python/freetoken/scheduler/prefill.py` (`PrefillAdder` :43, `_try_allocate_one` :77-99, `PrefillManager` :273, `_admit_next_batch` :300-330)
- Test: `tests/scheduler/test_kv_dynamic_escalation.py`

**Interfaces:**
- Produces: `PrefillManager.on_reserved: Callable[[int], None] | None` (called with the uid the moment `_add_one_req` charges `reserved_size`), `PrefillManager.on_capacity_blocked: Callable[[PendingReq], None] | None` (called once per admission pass for each never-started pending request that `_try_allocate_one` refused for capacity while `fits_empty`), `PrefillManager.pop_capacity_blocked() -> list[PendingReq]` (removes them from `pending_list`; the scheduler moves them to the controller).

- [ ] **Step 1: Write the failing tests**

```python
# tests/scheduler/test_kv_dynamic_escalation.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_escalation.py -q`
Expected: FAIL with `AttributeError: ... has no attribute 'on_capacity_blocked'` (or `pop_capacity_blocked`).

- [ ] **Step 3: Implement**

In `PrefillAdder` (dataclass at `prefill.py:43`) add two fields after `reserved_swa`:

```python
    # Dynamic KV pool hooks (scheduler/kv_dynamic.py). Both None = today's behaviour.
    on_reserved: Callable[[int], None] | None = None
    capacity_blocked: List[PendingReq] = field(default_factory=list)
```

(`from typing import Callable` at the top.) In `_try_allocate_one`, the two `return None` sites that mean "fits an empty pool but not now" (the first `admission_fits` check and the post-lock unlock) record the request:

```python
        if not admission_fits(need_now=estimated_len, reserved=self.reserved_size,
                              available=self.cache_manager.available_size, protect_tokens=0):
            if req.chunked_req is None:
                self.capacity_blocked.append(req)
            return None
        self.cache_manager.lock(handle)
        if not admission_fits(need_now=estimated_len, reserved=self.reserved_size,
                              available=self.cache_manager.available_size, protect_tokens=0):
            if req.chunked_req is None:
                self.capacity_blocked.append(req)
            return self.cache_manager.unlock(handle)
```

In `_add_one_req`, right after `self.reserved_size += remain_len + pending_req.output_len` (`prefill.py:200`):

```python
        if self.on_reserved is not None:
            self.on_reserved(pending_req.uid)
```

In `PrefillManager` add fields `on_reserved: Callable[[int], None] | None = None`, `on_capacity_blocked: Callable[[PendingReq], None] | None = None`, `_capacity_blocked: List[PendingReq] = field(default_factory=list)`; pass `on_reserved=self.on_reserved` when constructing `PrefillAdder` in `_admit_next_batch`; and at the end of `_admit_next_batch`, after the admission loop has run (before building the `Batch`), add:

```python
        if self.on_capacity_blocked is not None and adder.capacity_blocked:
            for pending in adder.capacity_blocked:
                if pending in self.pending_list and pending not in self._capacity_blocked:
                    self._capacity_blocked.append(pending)
                    self.on_capacity_blocked(pending)
```

and the accessor:

```python
    def pop_capacity_blocked(self) -> List[PendingReq]:
        """Never-started requests refused for capacity this pass; removed from pending_list so
        the dynamic controller can hold them (rule 2(b)). Empty unless a hook is installed."""
        blocked, self._capacity_blocked = self._capacity_blocked, []
        for pending in blocked:
            if pending in self.pending_list:
                self.pending_list.remove(pending)
        return blocked
```

Read `_admit_next_batch` fully (`prefill.py:300-360`) to place the hook after the loop that calls `adder.try_add_one` and before `return Batch(...)`; the rejected list (`self.rejected`) already exists there as the model for a per-pass side list.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_escalation.py tests/scheduler/test_park_admission.py -q`
Expected: PASS; `test_park_admission.py` unchanged.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/scheduler/prefill.py tests/scheduler/test_kv_dynamic_escalation.py
git commit -m "feat(scheduler): prefill hands capacity-blocked requests to the dynamic KV controller"
```

---

### Task 6: Engine budget snapshot, shrink-first ordering, shared-budget validation

**Files:**
- Modify: `python/freetoken/kvcache/base.py` (`validate_rebuild` :98-151)
- Modify: `python/freetoken/engine/engine.py` (boot snapshot near :479; `_is_pure_moe_shrink` :1277; `rebuild_runtime_cache` :2021-2240)
- Test: `tests/engine/test_kv_dynamic_engine.py`, `tests/kvcache/test_kv_cache_rebuild.py` (append)

**Interfaces:**
- Produces: `Engine.pool_budget_bytes: int`, `Engine.snapshot_pool_budget(self) -> int`, `Engine.kv_dynamic_floor_pages: int | None`; `validate_rebuild(..., pool_budget_bytes: int | None = None, budget_swap: bool = False)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_kv_dynamic_engine.py
"""Engine side of the dynamic KV pool: one budget for planner and validator, shrinks resized
before grows, and the over-budget card case from the 2026-09-12 review."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from freetoken.engine.engine import Engine

PAGE, KV_PAGE, SLOT = 64, 13_248 * 64, 2_772_480


def test_snapshot_pool_budget_is_slots_plus_pages():
    eng = SimpleNamespace(
        num_pages=1025, moe_offload_cache=SimpleNamespace(cache_size=7200, bank_sources={}),
        _gpu_owned_layer_ids=frozenset(),
        _target_moe_and_expert_bytes=lambda mcs: (7200, SLOT),
        _kv_bytes_per_page=lambda: KV_PAGE,
    )
    assert Engine.snapshot_pool_budget(eng) == 7200 * SLOT + 1025 * KV_PAGE
    assert eng.pool_budget_bytes == 7200 * SLOT + 1025 * KV_PAGE


def test_rebuild_order_puts_a_kv_shrink_before_the_moe_grow():
    calls = []
    eng = _recording_engine(calls, num_pages=4097, slots=6260)
    Engine._resize_pools(eng, eng.config, moe_cache_size=7200, num_pages=1025, num_swa_pages=None,
                         num_mamba_slots=None)
    assert calls == [("kv", 1025), ("moe", 7200)]


def test_rebuild_order_keeps_moe_first_when_kv_grows():
    calls = []
    eng = _recording_engine(calls, num_pages=1025, slots=7200)
    Engine._resize_pools(eng, eng.config, moe_cache_size=7043, num_pages=1537, num_swa_pages=None,
                         num_mamba_slots=None)
    assert calls == [("moe", 7043), ("kv", 1537)]


def _recording_engine(calls, *, num_pages, slots):
    cache = SimpleNamespace(cache_size=slots, quant_format="nvfp4",
                            rebuild=lambda n: calls.append(("moe", n)))
    eng = SimpleNamespace(
        num_pages=num_pages, moe_offload_cache=cache, linear_state_pool=None,
        spec_state_ladder=None, _gpu_owned_layer_ids=frozenset(),
        config=SimpleNamespace(moe_cache_size=slots, model_config=SimpleNamespace(num_experts=512)),
        _resize_kv_pool=lambda config, n, swa: calls.append(("kv", n)),
        _report_maintenance_progress=lambda *a, **k: None,
    )
    return eng
```

```python
# append to tests/kvcache/test_kv_cache_rebuild.py


def test_validate_rebuild_accepts_a_budget_swap_inside_pool_budget_on_an_over_budget_card():
    """Review of 702543b: 131,072 -> 163,840 raises the resident total by 1.6 MB and the
    Timer 1 return to the floor by 0.7 MB; both stay inside pool_budget_bytes and must pass,
    while the old 'no larger than resident' allowance would refuse them."""
    from freetoken.engine.engine import CacheRebuildRejected
    from freetoken.kvcache.qsa_pool import QSAKVCache

    KV_PAGE, SLOT = 13_248 * 64, 2_772_480
    budget = 7200 * SLOT + 1025 * KV_PAGE
    pool = _qsa_like_pool()  # any BaseKVCache subclass instance; kv_cost is monkeypatched below
    config = SimpleNamespace(memory_ratio=0.9, page_size=64)
    type(pool).kv_cost = classmethod(lambda cls, cfg, **kw: (KV_PAGE, 0, 64, 0))
    # Boot account so tight that the ordinary budget refuses everything (over-committed card).
    tight = dict(baseline_free=1, weights_bytes=0)
    with pytest.raises(CacheRebuildRejected):
        pool.validate_rebuild(config, num_pages=2561, target_moe=6730, per_expert_bytes=SLOT,
                              current_num_pages=2049, **tight)
    pool.validate_rebuild(config, num_pages=2561, target_moe=6730, per_expert_bytes=SLOT,
                          current_num_pages=2049, pool_budget_bytes=budget, budget_swap=True, **tight)
    pool.validate_rebuild(config, num_pages=1025, target_moe=7200, per_expert_bytes=SLOT,
                          current_num_pages=2561, pool_budget_bytes=budget, budget_swap=True, **tight)
    with pytest.raises(CacheRebuildRejected):
        pool.validate_rebuild(config, num_pages=4097, target_moe=7200, per_expert_bytes=SLOT,
                              current_num_pages=1025, pool_budget_bytes=budget, budget_swap=True, **tight)
```

Use the file's existing pool constructor for `_qsa_like_pool()` (the MHA pool helper `_mha_pool()` at the top of the file works: `validate_rebuild` lives on the base class). Import `SimpleNamespace` and `pytest` at the top of the file if missing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/engine/test_kv_dynamic_engine.py tests/kvcache/test_kv_cache_rebuild.py -q -k "dynamic or budget_swap or rebuild_order or snapshot"`
Expected: FAIL (`AttributeError: type object 'Engine' has no attribute 'snapshot_pool_budget'`, `TypeError: unexpected keyword 'pool_budget_bytes'`).

- [ ] **Step 3: Implement**

`kvcache/base.py:validate_rebuild`: add parameters `pool_budget_bytes: int | None = None, budget_swap: bool = False` after `shrink_only`, and replace the final check:

```python
        if need > budget:
            if budget_swap and pool_budget_bytes is not None and need <= pool_budget_bytes:
                # Dynamic KV pool (spec rule 3 + engine section): planner and validator share
                # ONE budget, the MoE+KV bytes resident at boot / after the last external
                # rebuild. A swap inside it cannot OOM when the engine resizes the shrinking
                # pool first (transient peak = max(before, after)); a per-transition "no larger
                # than resident" rule would wrongly refuse 131,072 -> 163,840 (+1.6 MB of
                # rounding slack) and the return to the floor (+0.7 MB) on an over-budget card.
                return
            raise CacheRebuildRejected(
                f"requested cache (moe={target_moe} slots, kv={target_pages} pages{extra_note}) "
                f"needs {mem_GB(need)} > budget {mem_GB(budget)}; old cache kept, still serving"
            )
```

`engine/engine.py`:

1. After `self._initial_num_pages = self.num_pages` (line 479) add `self.kv_dynamic_floor_pages = None` and, once the MoE cache exists (after the KV pool is built and before `_log_vram_ledger`), `self.snapshot_pool_budget()`. Add the helpers next to `_target_moe_and_expert_bytes`:

```python
    def _kv_bytes_per_page(self) -> int:
        cache_per_page, _fixed, _page, _min = self._pool_cls.kv_cost(self.config)
        return int(cache_per_page)

    def snapshot_pool_budget(self) -> int:
        """MoE slots + KV pages resident right now, in bytes: the one budget the dynamic KV
        pool's planner and the rebuild validator both read (spec rule 3). Taken at boot and
        after every rebuild the controller did not issue, never raised by the controller."""
        slots, per_slot = self._target_moe_and_expert_bytes(None)
        self.pool_budget_bytes = int(slots * per_slot + self.num_pages * self._kv_bytes_per_page())
        return self.pool_budget_bytes
```

2. In `rebuild_runtime_cache`, compute the swap verdict beside `_is_pure_moe_shrink` and pass it:

```python
        budget_swap = (
            num_mamba_slots is None and num_swa_pages is None and not layer_moves
            and moe_cache_size is not None and num_pages is not None
        )
        self.kv_cache.validate_rebuild(
            ...existing kwargs...,
            pool_budget_bytes=getattr(self, "pool_budget_bytes", None),
            budget_swap=budget_swap,
        )
```

3. Extract the pool-resize block (from `if num_swa_pages is not None: object.__setattr__(...)` through `self.linear_state_pool.rebuild(...)`/`spec_state_ladder.rebind()`, lines ~2208-2242) into a method and order it shrink-first:

```python
    def _resize_pools(self, config, *, moe_cache_size, num_pages, num_swa_pages, num_mamba_slots) -> None:
        """Resize the pools free-before-alloc ACROSS pools too: a KV shrink runs before a MoE
        grow so the transient peak is max(before, after). Until 2026-09-12 the MoE cache
        always went first, which is right for grow-KV/shrink-MoE and wrong for the dynamic
        pool's Timer 1 shrink (KV down, slots up)."""
        if num_swa_pages is not None:
            object.__setattr__(config, "swa_num_pages_override", num_swa_pages)
        kv_shrinks = num_pages is not None and num_pages < self.num_pages
        if kv_shrinks:
            self._resize_kv_pool(config, num_pages, num_swa_pages)
        if moe_cache_size is not None:
            ...existing exl3 scratch clear + self.moe_offload_cache.rebuild(moe_cache_size)...
        elif layer_moves-bookkeeping branch (unchanged)...
        if num_pages is not None and not kv_shrinks:
            self._resize_kv_pool(config, num_pages, num_swa_pages)
        elif num_pages is None and num_swa_pages is not None:
            self._resize_kv_pool(config, self.num_pages, num_swa_pages)
        if num_mamba_slots is not None:
            ...existing GDN pool rebuild + ladder rebind...
```

Call it from `rebuild_runtime_cache` where the block used to be, keeping `self._report_maintenance_progress("rebuild:pools")` after it. The `layer_moves` `elif` needs `layer_moves` in scope: pass it as a keyword too.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/engine/test_kv_dynamic_engine.py tests/kvcache/test_kv_cache_rebuild.py tests/engine/test_memory_step.py tests/engine/test_spec_rearm_after_rebuild.py -q`
Expected: new tests PASS; the other files' pass/fail sets unchanged.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/kvcache/base.py python/freetoken/engine/engine.py tests/engine/test_kv_dynamic_engine.py tests/kvcache/test_kv_cache_rebuild.py
git commit -m "feat(engine): shared pool budget, shrink-first pool ordering, budget-swap validation"
```

---

### Task 7: Governor KV rung under the dynamic pool

**Files:**
- Modify: `python/freetoken/engine/engine.py` (`step_memory` KV rung :1827-1860, `step_memory_noop` :1944)
- Test: `tests/engine/test_memory_step.py` (append)

**Interfaces:**
- Consumes: `Engine.kv_dynamic_floor_pages` (set by the scheduler at boot when the controller exists, Task 10), `config.kv_dynamic` (Task 8; read via `getattr(self.config, "kv_dynamic", False)` so this task tests standalone).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/engine/test_memory_step.py


def test_vram_down_kv_rung_shrinks_to_the_dynamic_floor_when_the_dynamic_pool_is_on():
    eng = FakeEngine(num_layers=2, num_experts=4, cache_size=8, owned_layers=(), num_pages=400)
    eng.config.kv_dynamic = True
    eng.kv_dynamic_floor_pages = 100
    # slots already at the floor (num_experts=4, overlap False) so rung 3 is reached
    eng.moe_offload_cache.rebuild(4)
    eng.config.moe_cache_size = 4
    res = eng.step_memory("vram", "down", is_idle=True)
    assert res["applied"] == "kv" and eng.num_pages == 100  # not the old 300 (-25 %)


def test_vram_up_kv_rung_is_a_noop_under_the_dynamic_pool():
    eng = FakeEngine(num_layers=2, num_experts=4, cache_size=8, owned_layers=(), num_pages=100)
    eng.config.kv_dynamic = True
    eng.kv_dynamic_floor_pages = 100
    eng._initial_num_pages = 100
    assert eng.step_memory_noop("vram", "up")["exhausted"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/engine/test_memory_step.py -q -k dynamic`
Expected: FAIL (`eng.num_pages == 300`, and the noop returns `None` because `_initial_moe_cache_size` differs or the KV restore branch fires).

- [ ] **Step 3: Implement**

In `step_memory` rung 3 (`engine.py:1827`):

```python
            # 3. KV pool (idle-only). With the dynamic KV pool on, the controller owns the pool
            #    size: the governor's shrink goes straight to the controller's floor (the
            #    freed bytes ARE the cushion it asked for; the controller re-reads
            #    pool_budget_bytes afterwards and funds the next grow from slots). Otherwise
            #    the historical -25 % of the boot size.
            idle = True if is_idle is None else is_idle
            init_pages = getattr(self, "_initial_num_pages", None) or getattr(self, "num_pages", 0)
            floor_pages = getattr(self, "kv_dynamic_floor_pages", None)
            if getattr(self.config, "kv_dynamic", False) and floor_pages:
                kv_floor = max(1, int(floor_pages))
            else:
                kv_floor = max(1, int(init_pages * 0.75))
```

In the `direction == "up"` branch, guard the KV restore: `if idle and init_pages and self.num_pages < init_pages and not getattr(self.config, "kv_dynamic", False):`. In `step_memory_noop` (`engine.py:1944`): `if init_pages and getattr(self, "num_pages", 0) < init_pages and not getattr(self.config, "kv_dynamic", False): return None`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/engine/test_memory_step.py -q`
Expected: the two new tests PASS; the rest unchanged.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/engine/engine.py tests/engine/test_memory_step.py
git commit -m "feat(engine): governor KV rung defers to the dynamic pool floor"
```

---

### Task 8: Config fields and server flags

**Files:**
- Modify: `python/freetoken/engine/config.py` (after `kv_reserve_tokens` :346 and `num_token_override` :431)
- Modify: `python/freetoken/server/args.py` (flags after `--kv-park-window-mib` :295; post-parse fixups after the `moe_cache_auto` default at ~:895)
- Test: `tests/server/test_kv_dynamic_args.py`

**Interfaces:**
- Produces on `EngineConfig`: `kv_dynamic: bool = False`, `kv_floor_tokens: int = 65_536`, `kv_step_tokens: int = 32_768`, `kv_shrink_idle_s: int = 600`, `kv_park_ttl_s: int = 18_000`, `kv_ceiling_tokens: int | None = None`. Flags `--kv-dynamic`, `--kv-floor-tokens`, `--kv-step-tokens`, `--kv-shrink-idle-s`, `--kv-park-ttl-s`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_kv_dynamic_args.py
"""Flag plumbing for the dynamic KV pool (spec: Engine/args section)."""
from __future__ import annotations

import pytest

from freetoken.server.args import parse_args

BASE = ["--model", "/tmp/model", "--moe-backend", "offload", "--kv-dtype", "fp8"]


def _parse(extra):
    return parse_args(BASE + extra)


def test_dynamic_off_by_default_changes_nothing():
    a = _parse(["--num-tokens", "262208"])
    assert not a.kv_dynamic and a.num_token_override == 262208 and a.kv_ceiling_tokens is None


def test_dynamic_rewrites_the_boot_pool_to_the_floor_and_records_the_ceiling():
    a = _parse(["--kv-dynamic", "--num-tokens", "262208", "--kv-reserve-tokens", "262144"])
    assert a.kv_dynamic
    assert a.kv_ceiling_tokens == 262208
    assert a.num_token_override == 65_536 + 64      # floor plus the dummy page
    assert a.kv_reserve_tokens == 65_536
    assert a.kv_step_tokens == 32_768 and a.kv_shrink_idle_s == 600 and a.kv_park_ttl_s == 18_000


def test_dynamic_without_num_tokens_uses_the_context_as_ceiling():
    a = _parse(["--kv-dynamic", "--max-seq-len-override", "131072"])
    assert a.kv_ceiling_tokens == 131072 + 64


@pytest.mark.parametrize("extra", [
    ["--kv-dynamic", "--kv-step-tokens", "4096"],
    ["--kv-dynamic", "--num-tokens", "32768", "--kv-floor-tokens", "65536"],
    ["--kv-dynamic", "--moe-backend", "fused"],
    ["--kv-park-ttl-s", "-1"],
])
def test_bad_dynamic_flags_are_refused(extra):
    with pytest.raises(SystemExit):
        _parse(extra)
```

If `parse_args` requires a real model folder for `--model`, create one in `tmp_path` with a minimal `config.json` the way `tests/server/test_parser_auto_selection.py` does and use it in `BASE`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/server/test_kv_dynamic_args.py -q`
Expected: FAIL with `argparse` "unrecognized arguments: --kv-dynamic".

- [ ] **Step 3: Implement**

`engine/config.py`, after `kv_reserve_tokens` (line 346):

```python
    # Dynamic KV pool (docs/superpowers/specs/2026-09-12-dynamic-kv-pool-design.md): boot at
    # kv_floor_tokens, grow by kv_step_tokens rungs up to kv_ceiling_tokens by trading MoE
    # slots, shrink back after kv_shrink_idle_s with no request. kv_park_ttl_s expires parked
    # prefixes from host RAM (Timer 2) whether or not the dynamic pool is on.
    kv_dynamic: bool = False
    kv_floor_tokens: int = 65_536
    kv_step_tokens: int = 32_768
    kv_shrink_idle_s: int = 600
    kv_park_ttl_s: int = 18_000
    kv_ceiling_tokens: int | None = None  # resolved by server/args.py from --num-tokens
```

`server/args.py`, after the `--kv-park-window-mib` argument:

```python
    parser.add_argument("--kv-dynamic", action="store_true", default=ServerArgs.kv_dynamic,
                        help="Boot with a small KV pool and grow it by trading MoE slots when a request "
                             "needs the room; shrink back after --kv-shrink-idle-s idle. Needs an "
                             "offload-family MoE backend. --num-tokens becomes the ceiling.")
    parser.add_argument("--kv-floor-tokens", type=_positive_int, default=ServerArgs.kv_floor_tokens,
                        help="Usable KV tokens the dynamic pool boots with and shrinks back to (default 65536).")
    parser.add_argument("--kv-step-tokens", type=_positive_int, default=ServerArgs.kv_step_tokens,
                        help="Growth rung of the dynamic pool in tokens (default 32768, minimum 8192).")
    parser.add_argument("--kv-shrink-idle-s", type=_positive_int, default=ServerArgs.kv_shrink_idle_s,
                        help="Seconds with no request before the dynamic pool shrinks to the floor (default 600).")
    parser.add_argument("--kv-park-ttl-s", type=int, default=ServerArgs.kv_park_ttl_s,
                        help="Seconds a parked prefix may sit unused in RAM/SSD before it is dropped "
                             "(default 18000 = 5 h; 0 never).")
```

Post-parse, after the `moe_cache_auto` default block (~line 895), where `page_size` is already final (check `_adjust_config`; if `page_size` is resolved later, do the rewrite where `num_token_override` is normalised there instead):

```python
    if kwargs["kv_park_ttl_s"] < 0:
        parser.error("--kv-park-ttl-s must be >= 0")
    if kwargs["kv_dynamic"]:
        from freetoken.moe import is_offload_moe_backend

        if not is_offload_moe_backend(kwargs["moe_backend"]):
            parser.error("--kv-dynamic requires the offload, cpu or hybrid MoE backend")
        if kwargs["kv_step_tokens"] < 8192:
            parser.error("--kv-step-tokens must be at least 8192")
        page = kwargs.get("page_size") or 64
        ceiling = kwargs.get("num_token_override") or 0
        if ceiling <= 0:
            # 0/absent means "fill the card" for a fixed pool; for the dynamic pool the ceiling
            # is the context itself plus the dummy page, never the auto card-filling size.
            ceiling = int(kwargs.get("max_seq_len_override") or _model_max_seq_len(kwargs["model_path"])) + page
        floor = kwargs["kv_floor_tokens"]
        if floor + page > ceiling:
            parser.error(f"--kv-floor-tokens {floor} must not exceed the ceiling {ceiling - page}")
        kwargs["kv_ceiling_tokens"] = ceiling
        kwargs["num_token_override"] = floor + page
        kwargs["kv_reserve_tokens"] = floor
```

`_model_max_seq_len(model_path)` reads `max_position_embeddings` from the model's `config.json` (`cached_load_hf_config(model_path).max_position_embeddings`; import from `freetoken.engine.config`). If `page_size` is not in `kwargs` at this point, look at how `--num-tokens` is turned into pages (`_adjust_config` in `engine/config.py` or `scheduler/config.py`) and perform the floor rewrite in the same place, keeping the argparse errors here.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/server/test_kv_dynamic_args.py tests/server/test_parser_auto_selection.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/engine/config.py python/freetoken/server/args.py tests/server/test_kv_dynamic_args.py
git commit -m "feat(server): flags for the dynamic KV pool and the park TTL"
```

---

### Task 9: Maintenance ownership by operation id, begin/status messages

**Files:**
- Modify: `python/freetoken/message/tokenizer.py`, `python/freetoken/message/frontend.py`, `python/freetoken/message/__init__.py`
- Modify: `python/freetoken/tokenizer/server.py` (`_forward_control_msg` :395)
- Modify: `python/freetoken/server/api_server.py` (`_open_maintenance` :167, `_abort_maintenance` :188, `_reply_matches_open_operation` :211, `FrontendManager.maintenance_op` :268, `listen` :450-475, `_resolve_rebuild` :493, `_resolve_step` :534, `_note_progress` :562, `check_maintenance` :629, `cache_status` :1357)
- Test: `tests/server/test_maintenance_ownership.py`

**Interfaces:**
- Produces messages:
  ```python
  # tokenizer.py (scheduler -> detokenizer)
  @dataclass class MaintenanceBeginMsg(BaseTokenizerMsg): request_id: str; kind: str; detail: str | None = None
  @dataclass class KVDynamicStatusMsg(BaseTokenizerMsg): status: dict
  # frontend.py (detokenizer -> api)
  @dataclass class MaintenanceBeginReply(BaseFrontendMsg): request_id: str; kind: str; detail: str | None = None
  @dataclass class KVDynamicStatusReply(BaseFrontendMsg): status: dict
  ```
- Produces on `FrontendManager`: `maintenance_ops: Dict[str, dict]`, property `maintenance_op` (oldest record or None, kept for existing readers), `kv_dynamic_status: dict | None`; `_open_maintenance(state, request_id, kind, detail=None)`, `_close_maintenance(state, request_id, *, failed: bool)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_maintenance_ownership.py
"""Rule 9 of the dynamic KV pool spec: one maintenance record per operation id. The race from
the 2026-09-12 review: a manual rebuild is dispatched (record opened at dispatch), the
scheduler starts an automatic operation before receiving it, and the automatic completion
must not reopen the gate while the manual one is outstanding."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from freetoken.message import (
    CacheProgressReply, CacheRebuildReply, KVDynamicStatusReply, MaintenanceBeginReply,
)
from freetoken.server.api_server import FrontendManager, dispatch_rebuild


class _Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _manager(clock):
    m = FrontendManager(
        config=SimpleNamespace(served_model_name="m", kv_park="ram", kv_dtype="fp8"),
        send_tokenizer=None, recv_tokenizer=None, maintenance_state="serving", monotonic=clock,
    )
    m.sent = []

    async def send_one(msg):
        m.sent.append(msg)

    m.send_one = send_one
    return m


def _rebuild_reply(request_id, status="ok"):
    return CacheRebuildReply(request_id=request_id, status=status, moe_cache_size=7000,
                             num_pages=1537, mamba_slots=12, num_swa_pages=0, error=None)


def test_automatic_completion_does_not_reopen_the_gate_under_a_dispatched_manual_operation():
    clock = _Clock()
    m = _manager(clock)

    async def run():
        task = asyncio.ensure_future(dispatch_rebuild(m, moe_cache_size=7000, num_pages=1537, timeout=0.05))
        await asyncio.sleep(0)                       # dispatch has opened M and sent it
        manual_id = m.sent[-1].request_id
        m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:1", kind="auto-kv", detail="grow"))
        assert set(m.maintenance_ops) == {manual_id, "auto-kv:i:1"}
        m._note_progress(CacheProgressReply(request_id="auto-kv:i:1", phase="rebuild:pools"))
        assert m.maintenance_ops["auto-kv:i:1"]["phase"] == "rebuild:pools"
        m._resolve_rebuild(_rebuild_reply("auto-kv:i:1"))
        assert m.maintenance_state == "rebuilding"        # M still outstanding
        assert not m.rebuild_done.is_set()
        assert set(m.maintenance_ops) == {manual_id}
        m._note_progress(CacheProgressReply(request_id=manual_id, phase="executing"))
        assert m.maintenance_ops[manual_id]["phase"] == "executing"   # not ignored
        m._resolve_rebuild(_rebuild_reply(manual_id))
        assert m.maintenance_state == "serving" and m.rebuild_done.is_set()
        assert m.maintenance_ops == {}
        result = await task
        assert result["status"] == "ok"

    asyncio.run(run())


def test_failed_automatic_operation_latches_failed_even_with_others_open():
    m = _manager(_Clock())
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:2", kind="auto-kv"))
    m._resolve_rebuild(_rebuild_reply("auto-kv:i:2", status="failed"))
    assert m.maintenance_state == "failed" and m.rebuild_done.is_set()


def test_stale_reply_for_an_unknown_operation_changes_nothing():
    m = _manager(_Clock())
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:3", kind="auto-kv"))
    m._resolve_rebuild(_rebuild_reply("gone"))
    assert m.maintenance_state == "rebuilding" and "auto-kv:i:3" in m.maintenance_ops


def test_watchdog_times_the_oldest_silent_operation():
    clock = _Clock()
    m = _manager(clock)
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:4", kind="auto-kv"))
    clock.now += 5
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:5", kind="auto-kv"))
    snap = m.check_maintenance()
    assert snap["operation"]["request_id"] == "auto-kv:i:4" and snap["age_s"] == 5.0


def test_kv_dynamic_status_snapshot_is_kept_for_cache_status():
    m = _manager(_Clock())
    m._note_kv_dynamic(KVDynamicStatusReply(status={"enabled": True, "held": 1}))
    assert m.kv_dynamic_status == {"enabled": True, "held": 1}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/server/test_maintenance_ownership.py -q`
Expected: FAIL with `ImportError: cannot import name 'MaintenanceBeginReply'`.

- [ ] **Step 3: Implement**

Messages: add the four dataclasses (shapes above) in `message/tokenizer.py` (after `CacheProgressMsg`) and `message/frontend.py` (after `CacheProgressReply`); export all four in `message/__init__.py` (both the import list and `__all__`). In `tokenizer/server.py:_forward_control_msg` add, next to the `CacheProgressMsg` branch:

```python
    elif isinstance(m, MaintenanceBeginMsg):
        send_frontend.put(MaintenanceBeginReply(request_id=m.request_id, kind=m.kind, detail=m.detail))
    elif isinstance(m, KVDynamicStatusMsg):
        send_frontend.put(KVDynamicStatusReply(status=m.status))
```

and import them at the top of that file (both the import block near line 20-40 and the tuple near line 379 that lists control message types).

`api_server.py`:

```python
def _open_maintenance(state: Any, request_id: str, kind: str, detail: str | None = None) -> None:
    """Close the gate for one correlated operation. Several may be open at once: a manual
    rebuild is recorded at DISPATCH (before the scheduler sees it) and the scheduler may begin
    an automatic dynamic-KV operation in between, so the records are keyed by id and the gate
    reopens only when the map is empty (2026-09-12 review of 702543b)."""
    state.maintenance_state = "rebuilding"
    now = getattr(state, "monotonic", time.monotonic)()
    state.maintenance_ops[request_id] = {
        "request_id": request_id, "kind": kind, "started_at": now, "phase": "dispatched",
        "progress_at": now, "progress_count": 0, "detail": detail, "expired": False,
    }
    if hasattr(state, "rebuild_done"):
        state.rebuild_done.clear()


def _close_maintenance(state: Any, request_id: str, *, failed: bool) -> None:
    """Remove one record; reopen the gate only when nothing else is outstanding. A genuine
    destructive failure latches ``failed`` whatever else is open (the engine may be torn down)."""
    state.maintenance_ops.pop(request_id, None)
    if failed or state.fatal_error is not None:
        state.maintenance_state = "failed"
        if hasattr(state, "rebuild_done"):
            state.rebuild_done.set()
        return
    if not state.maintenance_ops and state.maintenance_state == "rebuilding":
        state.maintenance_state = "serving"
        state.inference_seen = state.monotonic()
        if hasattr(state, "rebuild_done"):
            state.rebuild_done.set()


def _abort_maintenance(state: Any, request_id: str) -> None:
    """The enqueue failed: the scheduler never saw the request and the engine is untouched."""
    _close_maintenance(state, request_id, failed=False)


def _reply_matches_open_operation(state: Any, request_id: str) -> bool:
    ops = getattr(state, "maintenance_ops", None) or {}
    if not ops or request_id in ops:
        return True
    logger.warning(f"ignoring a stale cache reply for {request_id}; open operations: {sorted(ops)}")
    return False
```

`FrontendManager`: replace `maintenance_op: Dict[str, Any] | None = None` with `maintenance_ops: Dict[str, Dict[str, Any]] = field(default_factory=dict)` and `kv_dynamic_status: Dict[str, Any] | None = None`, plus a read-only property:

```python
    @property
    def maintenance_op(self) -> Dict[str, Any] | None:
        """The oldest open operation (existing readers: /health, check_maintenance)."""
        if not self.maintenance_ops:
            return None
        return min(self.maintenance_ops.values(), key=lambda op: op["started_at"])
```

Update `dispatch_rebuild`/`dispatch_step` to call `_abort_maintenance(state, request_id)`. In `_resolve_rebuild` and `_resolve_step`, replace the tail (from `if self.fatal_error is not None:` to the end) with `_close_maintenance(self, msg.request_id, failed=(msg.status == "failed"))`. In `_note_progress`, look up `op = self.maintenance_ops.get(msg.request_id)`. In `check_maintenance`, `op = self.maintenance_op` still works through the property (oldest record); when it expires, latch as today. Add the two handlers and route them in `listen()` beside `CacheProgressReply`:

```python
    def _begin_maintenance(self, msg: MaintenanceBeginReply) -> None:
        """The scheduler started an operation on its own (dynamic KV pool). Same gate, own record."""
        _open_maintenance(self, msg.request_id, msg.kind, msg.detail)

    def _note_kv_dynamic(self, msg: KVDynamicStatusReply) -> None:
        self.kv_dynamic_status = dict(msg.status)
```

In `cache_status()` add `"kv_dynamic": getattr(state, "kv_dynamic_status", None),` and, in the `maintenance` snapshot, `"open_operations": sorted(state.maintenance_ops)`. Grep `maintenance_op` across `python/freetoken/server/` and `tests/server/` and update every writer (`= None` assignments at lines 530 and 559 become `_close_maintenance` calls; readers may keep the property).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/server/test_maintenance_ownership.py tests/server/test_rebuild_maintenance.py tests/server/test_maintenance_deadline.py tests/server/test_rebuild_wait_queue.py tests/server/test_message_wire.py -q`
Expected: PASS. `test_message_wire.py` may enumerate message classes; add the four new ones where it lists them.

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/message python/freetoken/tokenizer/server.py python/freetoken/server/api_server.py tests/server/test_maintenance_ownership.py tests/server/test_message_wire.py
git commit -m "feat(server): per-id maintenance ownership and automatic operation begin/status messages"
```

---

### Task 10: Scheduler wiring

**Files:**
- Modify: `python/freetoken/scheduler/scheduler.py` (`__init__` :100-222, `idle_poll_timeout_ms` :223, `run_when_idle` :226, `overlap_loop` :300-420, `normal_loop` :400-440, `_process_batch_result` :498-549, `_process_one_msg` UserMsg/Abort branches :802-880, `_execute_pending_rebuild` :1229, `_execute_pending_operation` :1240-1316, `_execute_pending_step` :1195, `_speculative_decode_step` finish :1870-1877)
- Test: `tests/scheduler/test_kv_dynamic_scheduler.py`

**Interfaces:**
- Consumes: Tasks 1-9. `engine.pool_budget_bytes`, `engine.snapshot_pool_budget()`, `engine.kv_dynamic_floor_pages`, `cache_manager.probe_admission`, `prefill_manager.on_reserved/on_capacity_blocked/pop_capacity_blocked`, `MaintenanceBeginMsg`, `KVDynamicStatusMsg`, `KVDynamicController`.
- Produces on `Scheduler`: `_kv_dynamic: KVDynamicController | None`, `_admit_user_msg(msg)`, `_queue_for_kv_dynamic(msg) -> bool`, `_run_kv_dynamic_idle() -> None`, `_note_request_finished(reqs) -> None`, `_execute_pending_operation(msg) -> str` and `_execute_pending_rebuild() -> str | None` (outcome contract), `_send_kv_dynamic_status()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/scheduler/test_kv_dynamic_scheduler.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_scheduler.py -q`
Expected: FAIL with `AttributeError: 'Scheduler' object has no attribute '_queue_for_kv_dynamic'`.

- [ ] **Step 3: Implement**

In `__init__`, after the `PrefillManager` is built and before `super().__init__`, construct the controller:

```python
        self._kv_dynamic = self._make_kv_dynamic(config) if getattr(config, "kv_dynamic", False) else None
        self._kv_dynamic_last_status = None
        if self._kv_dynamic is not None:
            self.prefill_manager.on_reserved = self._kv_dynamic.note_reserved
            self.prefill_manager.on_capacity_blocked = lambda pending: None  # collected via pop_capacity_blocked
            self.engine.kv_dynamic_floor_pages = self._kv_dynamic.policy.floor_pages
            self.engine._initial_num_pages = self._kv_dynamic.policy.floor_pages
```

```python
    def _make_kv_dynamic(self, config):
        """Bind the pure policy to this engine's measured costs (spec: Scheduler section)."""
        from freetoken.engine.cache_budget import expert_bytes_per_slot

        from .kv_dynamic import KVDynamicController, KVDynamicPolicy

        moe = self.engine.moe_offload_cache
        if moe is None:
            raise ValueError("--kv-dynamic requires an offloaded MoE slot cache")
        if not self.cache_manager.supports_runtime_rebuild:
            raise ValueError("--kv-dynamic is unsupported by this model's KV cache")
        if getattr(config.model_config, "dsv4_args", None) is not None:
            raise ValueError("--kv-dynamic does not support DSV4's owned KV tiers")
        if config.tp_info.size != 1:
            raise ValueError("--kv-dynamic requires TP=1 in this batch")
        page = config.page_size
        cache_per_page, _fixed, _pt, _min = type(self.engine.kv_cache).kv_cost(config)
        num_experts = config.model_config.num_experts
        slot_floor = 2 * num_experts if getattr(config, "moe_prefill_overlap", False) else num_experts
        policy = KVDynamicPolicy(
            floor_pages=config.kv_floor_tokens // page + 1,
            ceiling_pages=(config.kv_ceiling_tokens - page) // page + 1,
            step_pages=config.kv_step_tokens // page,
            page_size=page,
            kv_bytes_per_page=int(cache_per_page),
            slot_bytes=expert_bytes_per_slot(moe.bank_sources, self.engine._gpu_owned_layer_ids),
            slot_floor=slot_floor,
        )
        self.engine.snapshot_pool_budget()
        controller = KVDynamicController(
            policy, shrink_idle_s=config.kv_shrink_idle_s, instance_id=str(id(self))[-6:],
        )
        logger.info_rank0(
            "Dynamic KV pool on: floor %d, ceiling %d, step %d tokens; shrink after %d s idle; "
            "budget %.2f GiB", policy.usable_tokens(policy.floor_pages),
            policy.usable_tokens(policy.ceiling_pages), config.kv_step_tokens,
            config.kv_shrink_idle_s, self.engine.pool_budget_bytes / (1 << 30),
        )
        return controller
```

Admission. Replace the `UserMsg` branch body after the tombstone check with the ceiling clip + probe (`_admit_user_msg` is the old body from `has_raw_picture = ...` to `add_one_req`, factored out verbatim):

```python
        elif isinstance(msg, UserMsg):
            ...tombstone check unchanged...
            if not self._queue_for_kv_dynamic(msg):
                self._admit_user_msg(msg)
```

```python
    def _queue_for_kv_dynamic(self, msg: UserMsg) -> bool:
        """Rule 2. True when this method handled the message (held, or refused over the
        ceiling); False when the ordinary admission path should take it."""
        c = self._kv_dynamic
        if c is None or not c.enabled:
            return False
        page = self.config.page_size
        ceiling = int(self.config.kv_ceiling_tokens) - page
        input_len = len(msg.input_ids)
        if input_len >= ceiling:
            self.send_result([ErrorReplyMsg(
                uid=msg.uid,
                error=(f"prompt is too long: {input_len} tokens > {ceiling} maximum "
                       f"(prompt + generation); shorten the prompt or increase the KV cache budget"),
                code="context_length_exceeded",
            )])
            return True
        if msg.sampling_params.max_tokens > ceiling - input_len:
            msg.sampling_params.max_tokens = ceiling - input_len
            logger.warning_rank0(f"Adjust max_tokens to {ceiling - input_len} for request {msg.uid}.")
        need_total = input_len + msg.sampling_params.max_tokens
        pool_tokens = (self.engine.num_pages - 1) * page
        probe = self.cache_manager.probe_admission(
            msg.input_ids, msg.sampling_params.max_tokens,
            reserved=self.decode_manager.inflight_tokens + c.uncommitted_tokens,
            cache_private=msg.mm_embeds is not None or msg.mm_pixel_values is not None,
        )
        verdict = c.decide_admission(
            msg.uid, msg, need_total=need_total, need_now=probe.need_now, pool_tokens=pool_tokens,
            fits_empty=probe.fits_empty, fits_now=probe.fits_now,
        )
        if verdict == "hold":
            logger.info_rank0(
                "KV pool holds request %d: need %d tokens (now %d), pool %d, %d held",
                msg.uid, need_total, probe.need_now, pool_tokens, len(c.held),
            )
            self._send_kv_dynamic_status()
            return True
        c.note_uncommitted(msg.uid, probe.need_now)
        return False
```

`_admit_user_msg` keeps the existing `max_seq_len` clip (now against the grown pool) and calls `self.prefill_manager.add_one_req(msg)`.

Abort branch: add `if self._kv_dynamic is not None: self._kv_dynamic.on_abort(msg.uid)` before the tombstone bookkeeping.

Idle execution:

```python
    def _running_need_tokens(self) -> int:
        running = sum(len(r.input_ids) + r.output_len for r in self.decode_manager.running_reqs)
        pending = sum(p.input_len + p.output_len for p in self.prefill_manager.pending_list)
        return running + pending

    def _run_kv_dynamic_idle(self) -> None:
        """Rule 2/4/8 at an idle safe point: escalate capacity-blocked requests, execute the
        controller's one plan, then drain the held FIFO according to the outcome."""
        c = self._kv_dynamic
        if c is None or not c.enabled or self._pending_rebuild is not None:
            return
        for pending in self.prefill_manager.pop_capacity_blocked():
            c.escalate(pending.uid, pending, pending.input_len + pending.output_len)
        plan = c.plan_idle(
            current_pages=self.engine.num_pages, pool_budget_bytes=self.engine.pool_budget_bytes,
            running_need_tokens=self._running_need_tokens(),
        )
        outcome = "ok"
        if plan is not None:
            request_id = c.next_operation_id()
            detail = f"{plan.reason} -> {c.policy.usable_tokens(plan.target_pages)} tokens, {plan.target_slots} slots"
            self.send_result([MaintenanceBeginMsg(request_id=request_id, kind="auto-kv", detail=detail)])
            self._pending_rebuild = CacheRebuildBackendMsg(
                request_id=request_id, moe_cache_size=plan.target_slots, num_pages=plan.target_pages,
            )
            logger.info_rank0("KV pool %s: %d -> %d tokens, slots -> %d%s", plan.reason,
                              (self.engine.num_pages - 1) * self.config.page_size,
                              c.policy.usable_tokens(plan.target_pages), plan.target_slots,
                              " (capped by the slot floor)" if plan.capped else "")
            if plan.capped:
                logger.warning_rank0("KV pool grow capped: slots at their floor %d", c.policy.slot_floor)
            outcome = self._execute_pending_rebuild() or "ok"
        if outcome == "failed":
            for held in c.disable("cache rebuild failed and could not be rolled back"):
                self.send_result([ErrorReplyMsg(
                    uid=held.uid, error="cache rebuild failed; server needs a restart", code="server_error",
                )])
            self._send_kv_dynamic_status()
            return
        # ok or rejected: admit the head (and any further held entries that fit as they are).
        head = c.pop_held()
        while head is not None:
            if isinstance(head.msg, UserMsg):
                self._admit_user_msg(head.msg)
            else:  # an escalated PendingReq goes straight back to the pending list
                self.prefill_manager.pending_list.append(head.msg)
            if c.has_held() and c.plan_idle(
                current_pages=self.engine.num_pages, pool_budget_bytes=self.engine.pool_budget_bytes,
                running_need_tokens=self._running_need_tokens(),
            ) is not None:
                break  # the next one needs another rebuild: next idle point
            head = c.pop_held()
        self._send_kv_dynamic_status()
```

Wire it: in `overlap_loop` after the `_execute_pending_rebuild` gate, add `if last_data is None and not (self.prefill_manager.runnable or self.decode_manager.runnable or self._pending_rebuild is not None): self._run_kv_dynamic_idle()`; the same (without `last_data`) in `normal_loop` after its gate; and in `run_when_idle` after `park_idle()`. Include `self._kv_dynamic is not None and self._kv_dynamic.has_held()` in both loops' `blocking = not (...)` expressions.

Deadline: `idle_poll_timeout_ms` returns the min of `cache_manager.next_park_delay_ms()` and `self._kv_dynamic.next_deadline_ms(current_pages=self.engine.num_pages)` (skip `None`s).

Outcomes: change `_execute_pending_operation` to `return "ok"` / `"rejected"` / `"failed"` at each of its exits exactly as PR #300's diff did (`docs`: the three `return` sites after `_reply_rebuild`), `_execute_pending_step` to return `"ok"`/`"rejected"`/`"failed"` likewise, and `_execute_pending_rebuild` to `return self._execute_pending_operation(msg)` inside its `try`. After a successful non-`auto-kv:` operation (`request_id` not starting with `"auto-kv:"`), call `self.engine.snapshot_pool_budget()` and `self._send_kv_dynamic_status()`.

Finish hooks:

```python
    def _note_request_finished(self, reqs) -> None:
        if reqs and self._kv_dynamic is not None:
            self._kv_dynamic.on_request_finished()

    def _send_kv_dynamic_status(self) -> None:
        c = self._kv_dynamic
        if c is None:
            return
        status = c.status(current_pages=self.engine.num_pages, pool_budget_bytes=self.engine.pool_budget_bytes)
        if status != self._kv_dynamic_last_status:
            self.send_result([KVDynamicStatusMsg(status=status)])
            self._kv_dynamic_last_status = status
```

Call `self._note_request_finished(new_finished_reqs)` right after `self.finished_reqs = new_finished_reqs` (line 549), `self._note_request_finished(finished_now)` after `self.finished_reqs = finished_now` (line 1877), and `self._note_request_finished([req_to_free])` in the abort branch when a request was freed.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler/test_kv_dynamic_scheduler.py tests/scheduler/test_moe_only_rebuild_gate.py tests/scheduler/test_park_admission.py -q`
Expected: new tests PASS; the two existing files unchanged. Then the wider sweep: `PYTHONPATH=python .venv/bin/python -m pytest tests/scheduler tests/engine tests/kvcache tests/server -q 2>&1 | tail -3` and compare the failure count with the pre-task baseline (only pre-existing failures may remain).

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/scheduler/scheduler.py tests/scheduler/test_kv_dynamic_scheduler.py
git commit -m "feat(scheduler): dynamic KV pool controller wired into admission, idle points and outcomes"
```

---

### Task 11: Settings page dials, launch mapping, fit-check warning, status tile

**Files:**
- Modify: `python/freetoken/daemon/settings/dials.py` (after `KVCacheTokens` :168 and the parking dials :207-262; `validate_settings` cross-field block :1150+)
- Modify: `python/freetoken/daemon/settings/linux_launch.py` (:325-340, :374-376)
- Modify: `python/freetoken/engine/memory_plan.py` (geometry issues near :852-864)
- Modify: `python/freetoken/daemon/settings/static/index.html` (:1126 KV tile)
- Modify: `python/freetoken/daemon/settings/app.py` (`HELPER_VERSION` :34 -> `"1.5.0"`)
- Test: `tests/settings/test_kv_dynamic_dials.py`

**Interfaces:**
- Produces dials `KVDynamic` (toggle, default on), `KVFloorTokens` (65536), `KVStepTokens` (32768), `KVShrinkIdleMin` (10), `KVParkTTLHours` (5); `KVCacheTokens` help text updated. Launch: `--kv-dynamic --kv-floor-tokens N --kv-step-tokens N --kv-shrink-idle-s N*60`; `--kv-park-ttl-s N*3600` whenever `KVPark != off`. Planner issue code `kv_ceiling_unreachable`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/settings/test_kv_dynamic_dials.py
from __future__ import annotations

from freetoken.daemon.settings import dials as d
from freetoken.daemon.settings.linux_launch import build_launch
from tests.settings.test_linux_launch import _facts, _settings  # existing helpers in that file


def test_the_six_dynamic_pool_dials_exist_in_the_chats_group_with_plain_words():
    names = {"KVDynamic", "KVFloorTokens", "KVStepTokens", "KVShrinkIdleMin", "KVParkTTLHours"}
    for name in names:
        dial = d.DIAL_BY_NAME[name]
        assert dial.group == "Model & chats" and dial.plain and dial.info and dial.blurb
    assert d.DIAL_BY_NAME["KVDynamic"].default is True
    assert d.DIAL_BY_NAME["KVStepTokens"].minimum == 8192
    assert "largest size" in d.DIAL_BY_NAME["KVCacheTokens"].info


def test_validation_refuses_a_floor_above_the_ceiling_and_a_tiny_step():
    errors = d.validate_settings({"KVFloorTokens": 300_000, "KVCacheTokens": 262_208})
    assert any(e["field"] == "KVFloorTokens" for e in errors)
    errors = d.validate_settings({"KVStepTokens": 4096})
    assert any(e["field"] == "KVStepTokens" for e in errors)
    assert d.validate_settings({"KVFloorTokens": 65_536, "KVCacheTokens": 262_208, "KVStepTokens": 32_768}) == []


def test_launch_maps_the_dynamic_pool_and_the_ttl():
    settings = _settings(KVDynamic=True, KVFloorTokens=65_536, KVStepTokens=32_768, KVShrinkIdleMin=10,
                         KVPark="ram", KVParkTTLHours=5, KVCacheTokens=262_208)
    argv = build_launch(settings, _facts()).argv
    for flag, value in (("--kv-dynamic", None), ("--kv-floor-tokens", "65536"), ("--kv-step-tokens", "32768"),
                        ("--kv-shrink-idle-s", "600"), ("--kv-park-ttl-s", "18000"), ("--num-tokens", "262208")):
        assert flag in argv
        if value is not None:
            assert argv[argv.index(flag) + 1] == value


def test_launch_without_dynamic_or_parking_adds_neither():
    argv = build_launch(_settings(KVDynamic=False, KVPark="off"), _facts()).argv
    assert "--kv-dynamic" not in argv and "--kv-park-ttl-s" not in argv


def test_planner_warns_when_the_ceiling_cannot_be_reached():
    from freetoken.engine.memory_plan import kv_ceiling_issue

    KV_PAGE, SLOT = 13_248 * 64, 2_772_480
    # 940 slots needed for 65k -> 262k; only 500 above the floor
    issue = kv_ceiling_issue(floor_tokens=65_536, ceiling_tokens=262_144, lru_slots=1524, slot_floor=1024,
                             cache_per_page=KV_PAGE, page_tokens=64, per_expert_bytes=SLOT)
    assert issue["code"] == "kv_ceiling_unreachable" and "1024" in issue["message"]
    assert kv_ceiling_issue(floor_tokens=65_536, ceiling_tokens=262_144, lru_slots=7200, slot_floor=1024,
                            cache_per_page=KV_PAGE, page_tokens=64, per_expert_bytes=SLOT) is None
```

Read `tests/settings/test_linux_launch.py` first: reuse its fixture helpers by their real names (they build a settings dict and a `ModelFacts`); rename `_facts`/`_settings` in the test above to match.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_kv_dynamic_dials.py -q`
Expected: FAIL with `KeyError: 'KVDynamic'`.

- [ ] **Step 3: Implement**

`dials.py`, after the `KVCacheTokens` dial (extend its `info` with: "With Dynamic KV memory on, this is the largest size the pool may grow to; the pool boots at the smallest size and grows in steps.") add:

```python
    Dial(
        "KVDynamic", "toggle", True, "switch", "Grow the KV pool on demand by trading MoE expert slots and shrink it back when idle.",
        "Model & chats", engine_mapping="--kv-dynamic",
        plain="Dynamic KV memory", blurb="Start small, grow when a chat needs room, shrink back when quiet.",
        effects=("speed:up",),
        info=(
            "The card's chat memory starts at the smallest size below and the space saved becomes expert "
            "slots, which type faster. When a chat arrives that could outgrow the memory, the server pauses "
            "about a second, hands slots to the memory, and lets it in. After the quiet time below with no "
            "chats, it shrinks back. Nothing is forgotten: parking already keeps each finished chat in PC memory. "
            "Measured 2026-09-12 on this PC: one resize under a second; about 157 slots per 32,768 tokens; "
            "about 6 % typing speed per 1,000 slots."
        ),
    ),
    Dial(
        "KVFloorTokens", "number", 65536, "tokens", "Usable KV tokens the dynamic pool boots with and shrinks back to.",
        "Model & chats", minimum=8192, maximum=4194304, engine_mapping="--kv-floor-tokens <N>",
        plain="Smallest KV memory", blurb="Chat memory when no big chat is around.", slider=(16384, 262144, 8192),
        effects=("speed:up", "vram:down"),
        info="Chats whose prompt plus answer allowance fit in this need no resize. Must be no larger than KV cache tokens.",
    ),
    Dial(
        "KVStepTokens", "number", 32768, "tokens", "Growth rung of the dynamic KV pool.",
        "Model & chats", minimum=8192, maximum=1048576, engine_mapping="--kv-step-tokens <N>",
        plain="KV growth step", blurb="How much the chat memory grows at a time.", slider=(8192, 131072, 8192), advanced=True,
        info="A big chat jumps straight to the rung it needs, so this only sets the smallest change worth a resize. Below 8,192 is refused: a resize costs about a second.",
    ),
    Dial(
        "KVShrinkIdleMin", "number", 10, "min", "Minutes with no request before the dynamic KV pool shrinks to the floor.",
        "Model & chats", minimum=1, maximum=1440, engine_mapping="--kv-shrink-idle-s <N*60>",
        plain="Quiet time before shrinking", blurb="Timer 1: how long the card waits before giving slots back.", slider=(1, 120, 1),
        info="Counted from the last finished chat. The shrink runs only while no chat is active and pauses the server about a second.",
    ),
    Dial(
        "KVParkTTLHours", "number", 5, "h", "Hours a parked chat may sit unused in PC memory or on the SSD before it is dropped.",
        "Model & chats", minimum=0, maximum=168, engine_mapping="--kv-park-ttl-s <N*3600>",
        plain="Parked chat lifetime", blurb="Timer 2: when a quiet chat is dropped from PC memory for good.", slider=(0, 48, 1),
        effects=("ram:down",),
        info="0 keeps parked chats until the space is needed. Dropping frees the pinned RAM; a dropped chat is re-read from scratch if it returns.",
    ),
```

`validate_settings` cross-field block (where the owned-layer slot floor check reads `context`): add

```python
    floor = _num(settings, context, "KVFloorTokens")
    ceiling = _num(settings, context, "KVCacheTokens")
    if floor is not None and ceiling is not None and ceiling > 0 and floor > ceiling:
        errors.append({"field": "KVFloorTokens", "message": f"Smallest KV memory {floor} must not exceed KV cache tokens {ceiling}"})
    step = _num(settings, context, "KVStepTokens")
    if step is not None and step < 8192:
        errors.append({"field": "KVStepTokens", "message": "KV growth step must be at least 8192 tokens"})
```

using whatever helper the block already uses to read a value from `settings` falling back to `context` (there is one for the `MoECacheSize`/`GpuOwnedLayers` pair; reuse it and name it correctly).

`linux_launch.build_launch`: after the `--kv-park` block add

```python
    if kv_park != "off":
        argv += ["--kv-park-ttl-s", str(_int(_get(settings, "KVParkTTLHours"), 5) * 3600)]
    if _truthy(_get(settings, "KVDynamic")) and facts.is_moe:
        argv += [
            "--kv-dynamic",
            "--kv-floor-tokens", str(_int(_get(settings, "KVFloorTokens"), 65536)),
            "--kv-step-tokens", str(_int(_get(settings, "KVStepTokens"), 32768)),
            "--kv-shrink-idle-s", str(_int(_get(settings, "KVShrinkIdleMin"), 10) * 60),
        ]
```

`memory_plan.py`: add the pure helper and call it where the geometry issues are appended (near line 852, after `usable_kv_tokens` is known, with the same `cache_per_page`, `page_tokens`, `per_expert_bytes`, and the geometry's `lru_slots`; the settings dict is in scope as the request's settings, keys `KVDynamic`, `KVFloorTokens`, `KVCacheTokens`):

```python
def kv_ceiling_issue(*, floor_tokens: int, ceiling_tokens: int, lru_slots: int, slot_floor: int,
                     cache_per_page: int, page_tokens: int, per_expert_bytes: int) -> dict | None:
    """Dynamic KV pool: can the slots fund growth from the floor to the ceiling? None when yes."""
    needed_bytes = max(0, ceiling_tokens - floor_tokens) // page_tokens * cache_per_page
    fundable_bytes = max(0, lru_slots - slot_floor) * per_expert_bytes
    if needed_bytes <= fundable_bytes:
        return None
    reachable = floor_tokens + (fundable_bytes // cache_per_page) * page_tokens
    return {
        "code": "kv_ceiling_unreachable", "scope": "both",
        "message": (f"the largest KV size {ceiling_tokens} cannot be reached: slots would fall below "
                    f"their floor of {slot_floor}; the pool can grow to about {reachable} tokens"),
    }
```

`static/index.html` line 1126: when `status.kv_dynamic` is present and enabled, render the tile's `<strong>` as `${fmt(kv.pool_tokens)} of ${fmt(kv.ceiling_tokens)}` with the sub line `dynamic, floor ${fmt(kv.floor_tokens)}${kv.held ? `, ${kv.held} waiting` : ''}${kv.shrink_in_s != null ? `, shrinks in ${Math.round(kv.shrink_in_s / 60)} min` : ''}`; read `kv` from the same `/v1/cache/status` payload the page already polls for `geometry` (grep `parking` in the file to find the fetch).

`app.py`: `HELPER_VERSION = "1.5.0"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings tests/daemon -q`
Expected: new tests PASS; `test_dials_metadata.py` still passes (every new dial has plain/info/blurb, slider inside bounds, one group).

- [ ] **Step 5: Commit**

```bash
git add python/freetoken/daemon/settings python/freetoken/engine/memory_plan.py tests/settings/test_kv_dynamic_dials.py
git commit -m "feat(settings): dynamic KV pool dials, launch mapping, ceiling-reachable warning, status tile"
```

---

### Task 12: Docs

**Files:**
- Modify: `docs/cli.md` (table rows after `--kv-reserve-tokens` :85 and after the `--kv-park-*` rows)
- Modify: `README.md` (a paragraph under "### KV prefix parking" :116, and a new "### Dynamic KV pool" heading before it)

- [ ] **Step 1: Add the flag rows to `docs/cli.md`**

```markdown
| `--kv-dynamic` | off | Boot the KV pool at `--kv-floor-tokens`, grow it in `--kv-step-tokens` rungs by trading MoE slots when a request needs the room (up to `--num-tokens`), shrink back after `--kv-shrink-idle-s` with no request. Offload-family MoE backends, TP=1 |
| `--kv-floor-tokens` | 65536 | Usable KV tokens the dynamic pool boots with and shrinks back to |
| `--kv-step-tokens` | 32768 | Growth rung of the dynamic pool (minimum 8192) |
| `--kv-shrink-idle-s` | 600 | Seconds with no request before the dynamic pool shrinks to the floor |
| `--kv-park-ttl-s` | 18000 | Seconds a parked prefix may sit unused before it is dropped from RAM/SSD; 0 never |
```

- [ ] **Step 2: Add the README section**

```markdown
### Dynamic KV pool

`--kv-dynamic` boots with a small KV pool (default 65,536 tokens) and spends the saved VRAM on
MoE expert slots. When a request arrives whose prompt plus output allowance needs more than the
pool, the scheduler waits for an idle point, trades slots for pages (byte-for-byte inside one
engine-owned budget) and admits it; requests arriving meanwhile queue behind it. After
`--kv-shrink-idle-s` with no request the pool shrinks back and the slots return. Every resize is
the existing idle-only rebuild (under a second on the RTX 5090 for graph sizes 1 and 2, measured
2026-09-12) and loses no conversation because `--kv-park ram` already holds every finished prefix;
`--kv-park-ttl-s` drops parked prefixes after a quiet spell (default 5 h). Design and measured
inputs: `docs/superpowers/specs/2026-09-12-dynamic-kv-pool-design.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/cli.md README.md
git commit -m "docs: dynamic KV pool flags and fork section"
```

---

### Task 13: Live acceptance on the serving box

**Files:**
- Create: `docs/research/dynamic-kv-pool-live-2026-09-XX.md` (fill the date on the day)

This task runs only when Jay says the server is free. It follows the spec's "Live acceptance" list (items 1-19). Commands run from the devbox through the WSL route (`cat <<'SCRIPT' | ssh -o BatchMode=yes 5090 'wsl -d vllm -e bash -l'`).

- [ ] **Step 1: Deploy**

```bash
git push origin mtp-upstream-merge
cat <<'SCRIPT' | ssh -o BatchMode=yes 5090 'wsl -d vllm -e bash -l'
cd ~/FreeToken && git pull && systemctl --user restart freetoken-settings
SCRIPT
```

Then on the page (https://5090.tail45ff04.ts.net/) confirm the five new dials on Model & chats, `KVDynamic` on, save, Start; wait for `/health` ok.

- [ ] **Step 2: Record the boot ledger**

```bash
grep -E "Dynamic KV pool on|KV [0-9]+ pages|MoE cache [0-9]+/" ~/FreeToken/logs/server-2020.log | tail -3
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
curl -s 127.0.0.1:2020/v1/cache/status | python3 -c "import sys,json; print(json.load(sys.stdin)['kv_dynamic'])"
```

Expected: pool 65,536 tokens; slots about 940 above the previous boot's figure (write down both).

- [ ] **Step 3: Run the acceptance items and record each in the research note**

For each spec item 1-19: the command or action, the log lines (`auto-kv:` ids, `KV pool grow/shrink`, `width N: captured`), the numbers (TTFT, total time, warm tok/s; first-100-token tok/s after a resize), pass/fail. Items needing traffic: a 150k `/v1/messages` conversation (Jay's pi harness or `~/specbench.py`), a second concurrent conversation, 20 short requests during a held big one (`scripts/squeeze-test/hammer.py`), a 4 GB card grab (`scripts/squeeze-test/grab.py` natively on Windows) during a grown pool. Timer 1: leave 10 minutes idle and confirm `auto-kv:*` shrink with no message arriving. Timer 2: set `KVParkTTLHours` to a fraction (the dial accepts integers; use `--kv-park-ttl-s 120` via the Advanced env override or a one-off boot) and confirm `expired_evictions` rises with the server idle.

- [ ] **Step 4: Decide and record**

If any of items 7 (zero failed requests), 10 (failure outcomes), 14 (MTP captured) or 19 (same-batch) fails, set `KVDynamic` off on the page, restart, and open a fix task before any further live run. Commit the note:

```bash
git add docs/research/dynamic-kv-pool-live-2026-09-XX.md
git commit -m "docs(research): dynamic KV pool live acceptance on the RTX 5090"
git push origin mtp-upstream-merge
```

---

## Self-review

**Spec coverage.** Rule 1 boot (Task 8 floor rewrite + Task 10 snapshot). Rule 2 admission, ceiling clip, drain barrier, concurrent hold with `fits_now`, same-batch (a) uncommitted (Tasks 2, 4, 10) and (b) escalation (Tasks 5, 10). Rule 3 budget, floor division, slot floor cap (Tasks 1, 6). Rule 4 Timer 1 in `run_when_idle` with the poll deadline (Task 10). Rule 5 TTL, relative delay, two clocks, sweep in `park_idle` (Tasks 3, 4). Rule 6 governor KV rung (Task 7). Rule 7 aborts (Tasks 2, 10). Rule 8 outcomes, error replies, disable (Tasks 2, 10). Rule 9 per-id ownership, unique ids, begin message, status (Tasks 9, 10). Settings, launch, fit warning, tile, version (Task 11). Docs (Task 12). Live acceptance 1-19 (Task 13). Not planned on purpose: Windows launcher, TP>1, DSV4, layer_moves funding, page-preserving resize (spec "Out of scope").

**Placeholder scan.** Task 8 names `_model_max_seq_len`; it is defined in the step text (reads `max_position_embeddings` via `cached_load_hf_config`). Task 11 defers the exact helper names of `tests/settings/test_linux_launch.py` to the implementer with the instruction to read that file first; the assertions are complete. Task 13 is a runbook, not code.

**Type consistency.** `KVPlan(target_pages, target_slots, capped, reason)` everywhere. `probe_admission(input_ids, output_len, *, reserved, cache_private)` returns `AdmissionProbe(need_now, protect_tokens, fits_empty, fits_now, cached_len)` in Tasks 4 and 10. `decide_admission(uid, msg, *, need_total, need_now, pool_tokens, fits_empty, fits_now) -> "admit"|"hold"` in Tasks 2 and 10. `plan_idle(*, current_pages, pool_budget_bytes, running_need_tokens)` in Tasks 2 and 10. `next_expiry_delay_ms()` takes no argument in Tasks 3 and 4. `_execute_pending_rebuild() -> str | None` in Task 10's tests and implementation. `MaintenanceBeginMsg(request_id, kind, detail)` in Tasks 9 and 10.
