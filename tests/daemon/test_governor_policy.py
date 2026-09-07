from __future__ import annotations

import pytest

from freetoken.daemon.settings.governor import Action, GovernorPolicy, GIB


def test_decide_down_when_below_cushion():
    policy = GovernorPolicy(
        vram_cushion=4 * GIB,
        ram_cushion=8 * GIB,
        step_interval=5.0,
    )
    # VRAM below cushion, RAM above
    actions = policy.decide(
        now=10.0,
        free_vram=3 * GIB,
        free_ram=16 * GIB,
    )
    assert len(actions) == 1
    assert actions[0] == Action(axis="vram", direction="down", ram_tight=False)

    # RAM below cushion, VRAM above
    actions = policy.decide(
        now=20.0,
        free_vram=10 * GIB,
        free_ram=6 * GIB,
    )
    assert len(actions) == 1
    assert actions[0] == Action(axis="ram", direction="down", ram_tight=True)


def test_decide_one_step_per_interval():
    policy = GovernorPolicy(
        vram_cushion=4 * GIB,
        ram_cushion=8 * GIB,
        step_interval=5.0,
    )
    # First step at t=10.0
    actions = policy.decide(now=10.0, free_vram=2 * GIB, free_ram=16 * GIB)
    assert len(actions) == 1
    assert actions[0].axis == "vram" and actions[0].direction == "down"

    # Too soon at t=12.0 (< 10.0 + 5.0) -> no action
    actions = policy.decide(now=12.0, free_vram=2 * GIB, free_ram=16 * GIB)
    assert actions == []

    # Exactly at interval boundary t=15.0 -> step allowed
    actions = policy.decide(now=15.0, free_vram=2 * GIB, free_ram=16 * GIB)
    assert len(actions) == 1
    assert actions[0].axis == "vram" and actions[0].direction == "down"


def test_decide_up_only_after_hold():
    vram_cushion = 4 * GIB
    rung = int(1.33 * GIB)
    margin = int(0.5 * GIB)
    up_threshold = vram_cushion + rung + margin

    policy = GovernorPolicy(
        vram_cushion=vram_cushion,
        ram_cushion=8 * GIB,
        rung_bytes=rung,
        margin=margin,
        step_interval=5.0,
        up_hold=60.0,
    )

    # First enters generous zone at t=10.0 -> no action yet (RAM is in neutral cushion zone)
    actions = policy.decide(now=10.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    assert actions == []

    # At t=40.0 (< 60s hold) -> no action
    actions = policy.decide(now=40.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    assert actions == []

    # At t=70.0 (>= 10.0 + 60s) -> step up
    actions = policy.decide(now=70.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    assert len(actions) == 1
    assert actions[0] == Action(axis="vram", direction="up", ram_tight=False)


def test_decide_up_hold_resets_if_memory_drops():
    vram_cushion = 4 * GIB
    rung = int(1.33 * GIB)
    margin = int(0.5 * GIB)
    up_threshold = vram_cushion + rung + margin

    policy = GovernorPolicy(
        vram_cushion=vram_cushion,
        ram_cushion=8 * GIB,
        rung_bytes=rung,
        margin=margin,
        step_interval=5.0,
        up_hold=60.0,
    )

    # High at t=10.0
    policy.decide(now=10.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)

    # Dips below up_threshold at t=30.0
    policy.decide(now=30.0, free_vram=up_threshold - 100, free_ram=9 * GIB)

    # Returns above up_threshold at t=40.0
    policy.decide(now=40.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)

    # At t=75.0 (only 35s since t=40.0) -> no action
    actions = policy.decide(now=75.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    assert actions == []

    # At t=100.0 (>= 40.0 + 60s) -> step up
    actions = policy.decide(now=100.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    assert len(actions) == 1
    assert actions[0] == Action(axis="vram", direction="up", ram_tight=False)


def test_decide_doubling_hold_off_on_flap():
    vram_cushion = 4 * GIB
    rung = int(1.33 * GIB)
    margin = int(0.5 * GIB)
    up_threshold = vram_cushion + rung + margin

    policy = GovernorPolicy(
        vram_cushion=vram_cushion,
        ram_cushion=8 * GIB,
        rung_bytes=rung,
        margin=margin,
        step_interval=5.0,
        up_hold=60.0,
        max_hold=600.0,
    )

    # Generous from t=0, holds until t=60 -> step up
    policy.decide(now=0.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    actions = policy.decide(now=60.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    assert len(actions) == 1 and actions[0].direction == "up"

    # Immediately trips cushion at t=62.0 (< 60.0 + 5.0 step_interval)
    # This must double the hold-off to 120.0
    policy.decide(now=62.0, free_vram=vram_cushion - GIB, free_ram=9 * GIB)

    # Memory recovers to generous at t=70.0
    policy.decide(now=70.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)

    # At t=135.0 (65s after t=70; would have passed original 60s hold) -> still waiting on 120s hold!
    actions = policy.decide(now=135.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    assert actions == []

    # At t=190.0 (>= 70.0 + 120.0) -> step up fires
    actions = policy.decide(now=190.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    assert len(actions) == 1 and actions[0].direction == "up"


def test_decide_both_axes_independent():
    policy = GovernorPolicy(
        vram_cushion=4 * GIB,
        ram_cushion=8 * GIB,
        step_interval=5.0,
    )

    # Both below cushion
    actions = policy.decide(
        now=10.0,
        free_vram=2 * GIB,
        free_ram=4 * GIB,
    )
    assert len(actions) == 2
    axes = {a.axis for a in actions}
    assert axes == {"vram", "ram"}
    assert all(a.direction == "down" for a in actions)


def test_decide_ram_tight_flag():
    policy = GovernorPolicy(
        vram_cushion=4 * GIB,
        ram_cushion=8 * GIB,
        step_interval=5.0,
    )

    # free_ram (4 GiB) < ram_cushion (8 GiB) -> ram_tight is True
    actions = policy.decide(
        now=10.0,
        free_vram=2 * GIB,
        free_ram=4 * GIB,
    )
    for a in actions:
        assert a.ram_tight is True

    # free_ram (12 GiB) >= ram_cushion (8 GiB) -> ram_tight is False
    actions = policy.decide(
        now=20.0,
        free_vram=2 * GIB,
        free_ram=12 * GIB,
    )
    for a in actions:
        assert a.ram_tight is False
