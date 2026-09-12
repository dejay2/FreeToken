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

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque

# stdlib logging on purpose: freetoken.utils.init_logger drags torch in through
# utils/torch_utils, and this module is the torch-free half of the feature.
logger = logging.getLogger(__name__)

MIN_STEP_TOKENS = 8_192
# Rule 4 backstop: after this many non-"ok" plans in a row the controller stops proposing
# shrinks until a request finishes. A rejection is usually an external VRAM grab that will
# still be there on the next attempt, and a rebuild is destructive (parking flush, graph
# teardown, API maintenance gate), so retrying it forever is worse than not shrinking.
MAX_CONSECUTIVE_FAILURES = 3


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
        """Slots the budget funds beside ``pages`` (floor division). Clamped to slot_floor; the
        floor clamp may exceed the budget when it is smaller than the floor geometry, so
        callers must check fit with _fits_budget if the budget may be below floor geometry."""
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
        if not self._fits_budget(pool_budget_bytes, self.floor_pages, self.slot_floor):
            return None
        return KVPlan(
            target_pages=self.floor_pages,
            target_slots=self.slots_for_pages(pool_budget_bytes, self.floor_pages),
            capped=False,
            reason="shrink",
        )


@dataclass
class HeldRequest:
    uid: int
    msg: object
    need_total: int
    held_at: float
    # What the requests already holding the pool needed when this one was held. Captured at
    # hold time on purpose: the plan runs at an IDLE point, by which time those requests have
    # finished and the live sum is 0 -- reading it then would collapse rule 2's concurrent
    # target back to a plain grow and the pair would keep taking turns (review, 2026-09-12).
    running_need_tokens: int = 0


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
        self.consecutive_failures = 0
        self._backoff_logged = False

    # ---- admission (rule 2) ------------------------------------------------------------
    def decide_admission(
        self, uid: int, msg: object, *, need_total: int, need_now: int, pool_tokens: int,
        fits_empty: bool, fits_now: bool, running_need_tokens: int = 0,
    ) -> str:
        if not self.enabled:
            return "admit"
        ceiling_tokens = self.policy.usable_tokens(self.policy.ceiling_pages)
        if self.held:
            # Drain barrier: once anyone waits for growth, later arrivals queue behind it so a
            # stream of small requests can never keep the scheduler busy and starve the big one.
            self._hold(uid, msg, need_total, running_need_tokens)
            return "hold"
        if need_total > pool_tokens:
            self._hold(uid, msg, need_total, running_need_tokens)
            return "hold"
        if fits_empty and not fits_now and pool_tokens < ceiling_tokens:
            # It fits an empty pool but other requests hold the room: wait for them once, then
            # grow so the pair runs side by side next time.
            self._hold(uid, msg, need_total, running_need_tokens)
            return "hold"
        return "admit"

    def _hold(self, uid: int, msg: object, need_total: int, running_need_tokens: int = 0) -> None:
        self.held.append(HeldRequest(
            uid=uid, msg=msg, need_total=need_total, held_at=self._clock(),
            running_need_tokens=int(running_need_tokens),
        ))

    def escalate(
        self, uid: int, msg: object, need_total: int, running_need_tokens: int = 0
    ) -> None:
        """Rule 2(b): a never-started pending request the prefill manager could not seat.

        ``running_need_tokens`` is the reservation the admission pass measured against when it
        refused this request, not the (by then zero) live sum.
        """
        if self.enabled and all(h.uid != uid for h in self.held):
            self._hold(uid, msg, need_total, running_need_tokens)

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
        # Traffic is the signal that the card's situation changed; give the shrink its three
        # attempts again (a grab that refused the last one may well be gone by now).
        self.consecutive_failures = 0
        self._backoff_logged = False

    def note_plan_outcome(self, reason: str, outcome: str, now: float | None = None) -> None:
        """Rule 8's outcome, fed back so a refused plan is not retried at once.

        Timer 1 is armed by ``last_request_finished`` alone: after a rejected or rolled-back
        shrink nothing else moves that clock, so ``next_deadline_ms`` would return 1 ms and
        ``run_when_idle`` would re-plan and re-execute the same destructive rebuild on every
        idle poll -- milliseconds apart, for as long as the server stayed quiet. Re-arming the
        timer puts the next attempt one full ``shrink_idle_s`` away, and three refusals in a
        row stop the attempts until a request finishes.
        """
        if outcome == "ok":
            self.consecutive_failures = 0
            self._backoff_logged = False
            return
        self.last_request_finished = self._clock() if now is None else now
        self.consecutive_failures += 1

    def has_held(self) -> bool:
        return bool(self.held)

    def pop_held(self) -> HeldRequest | None:
        return self.held.popleft() if self.held else None

    def plan_idle(
        self, *, current_pages: int, pool_budget_bytes: int, running_need_tokens: int,
        now: float | None = None, allow_shrink: bool = True, remember: bool = True,
    ) -> KVPlan | None:
        """The one rebuild (if any) to run at this idle point: the held head's grow first,
        else Timer 1's shrink. A held grow always wins over an overdue shrink so an arrival
        after a quiet spell costs one rebuild, never two.

        ``allow_shrink`` is the caller's answer to "is there a never-admitted request queued?"
        (spec rule 4: the shrink is skipped then, so one plan can go to the grow target
        instead of a shrink followed immediately by a grow). ``remember`` is False for the
        look-ahead the drain loop runs after admitting a held request: that call asks whether
        the NEXT entry would need another rebuild, and recording it as ``last_plan`` would
        report a rebuild that never ran.
        """
        if not self.enabled:
            return None
        if self.held:
            head = self.held[0]
            # The larger of what the head was waiting behind when it was held and whatever is
            # queued now: the first is the figure rule 2's concurrent case is about, the second
            # covers requests that arrived after it.
            running = max(head.running_need_tokens, running_need_tokens)
            reason = "grow-concurrent" if running else "grow"
            plan = self.policy.plan_grow(
                current_pages=current_pages, pool_budget_bytes=pool_budget_bytes,
                need_tokens=head.need_total, running_need_tokens=running, reason=reason,
            )
            if remember:
                self._remember(plan)
            return plan
        if not allow_shrink:
            return None
        now = self._clock() if now is None else now
        if (
            self.last_request_finished is not None
            and now - self.last_request_finished >= self.shrink_idle_s
        ):
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                if not self._backoff_logged:
                    logger.warning(
                        "KV pool shrink suspended after %d refused rebuilds in a row; it will "
                        "be retried once a request finishes", self.consecutive_failures,
                    )
                    self._backoff_logged = True
                return None
            plan = self.policy.plan_shrink(current_pages=current_pages, pool_budget_bytes=pool_budget_bytes)
            if remember:
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
            "consecutive_failures": self.consecutive_failures,
            "shrink_in_s": None if deadline is None else round(deadline / 1000, 1),
            "last_plan": self.last_plan,
        }
