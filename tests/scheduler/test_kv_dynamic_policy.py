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


def test_shrink_refuses_a_budget_below_the_floor_geometry():
    p = _policy(slot_floor=1024)
    tiny_budget = p.floor_pages * KV_PAGE + 100 * SLOT  # below floor geometry
    assert p.plan_shrink(current_pages=p.floor_pages + 2000, pool_budget_bytes=tiny_budget) is None
    assert p.plan_grow(current_pages=p.floor_pages, pool_budget_bytes=tiny_budget, need_tokens=100_000) is None


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
