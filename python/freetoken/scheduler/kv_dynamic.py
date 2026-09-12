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

from dataclasses import dataclass

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
