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


def test_ram_up_threshold_requires_two_rungs_while_vram_keeps_one():
    vram_cushion = 4 * GIB
    ram_cushion = 8 * GIB
    rung = int(1.33 * GIB)
    margin = int(0.5 * GIB)
    policy = GovernorPolicy(
        vram_cushion=vram_cushion,
        ram_cushion=ram_cushion,
        rung_bytes=rung,
        margin=margin,
        step_interval=5.0,
        up_hold=0.0,
    )

    # One-rung RAM headroom is deliberately not enough for a recall.
    assert policy.decide(
        now=0.0,
        free_vram=vram_cushion,
        free_ram=ram_cushion + rung + margin + 1,
    ) == []

    # Two rungs plus the margin are enough; VRAM remains in its neutral zone here.
    actions = policy.decide(
        now=1.0,
        free_vram=vram_cushion,
        free_ram=ram_cushion + 2 * rung + margin + 1,
    )
    assert actions == [Action(axis="ram", direction="up", ram_tight=False)]


def test_post_up_grace_blocks_recall_dip_but_allows_hard_squeeze():
    vram_cushion = 4 * GIB
    ram_cushion = 8 * GIB
    rung = int(1.33 * GIB)
    margin = int(0.5 * GIB)
    policy = GovernorPolicy(
        vram_cushion=vram_cushion,
        ram_cushion=ram_cushion,
        rung_bytes=rung,
        margin=margin,
        step_interval=5.0,
        up_hold=0.0,
    )
    high_ram = ram_cushion + 2 * rung + margin + GIB
    assert policy.decide(now=0.0, free_vram=vram_cushion, free_ram=high_ram) == [
        Action(axis="ram", direction="up", ram_tight=False)
    ]

    # A recall's cushion dip two seconds after the up is protected by the 10 s grace.
    assert policy.decide(
        now=2.0,
        free_vram=vram_cushion,
        free_ram=ram_cushion - 1,
    ) == []

    # A new squeeze more than one rung below the cushion overrides that grace.
    actions = policy.decide(
        now=5.0,
        free_vram=vram_cushion,
        free_ram=ram_cushion - rung - 1,
    )
    assert actions == [Action(axis="ram", direction="down", ram_tight=True)]


def test_high_memory_bursts_after_first_hold():
    vram_cushion = 4 * GIB
    ram_cushion = 8 * GIB
    rung = int(1.33 * GIB)
    margin = int(0.5 * GIB)
    policy = GovernorPolicy(
        vram_cushion=vram_cushion,
        ram_cushion=ram_cushion,
        rung_bytes=rung,
        margin=margin,
        step_interval=5.0,
        up_hold=60.0,
    )
    high_ram = ram_cushion + 2 * rung + margin + GIB

    assert policy.decide(now=0.0, free_vram=vram_cushion, free_ram=high_ram) == []
    assert policy.decide(now=60.0, free_vram=vram_cushion, free_ram=high_ram) == [
        Action(axis="ram", direction="up", ram_tight=False)
    ]
    assert policy.decide(now=64.0, free_vram=vram_cushion, free_ram=high_ram) == []
    assert policy.decide(now=65.0, free_vram=vram_cushion, free_ram=high_ram) == [
        Action(axis="ram", direction="up", ram_tight=False)
    ]
    assert policy.decide(now=70.0, free_vram=vram_cushion, free_ram=high_ram) == [
        Action(axis="ram", direction="up", ram_tight=False)
    ]


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


def test_note_step_done_counts_interval_and_flap_window_from_completion():
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
    policy.decide(now=0.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    actions = policy.decide(now=60.0, free_vram=up_threshold + GIB, free_ram=9 * GIB)
    assert [a.direction for a in actions] == ["up"]

    # The step's rebuild took 8 s; the loop re-stamps the axis when the POST returns.
    policy.note_step_done("vram", 68.0)

    # Without the re-stamp this tick (10 s after the step was chosen) would be outside the
    # 5 s flap window and would fire a plain down step; from completion it is 2 s in.
    actions = policy.decide(now=70.0, free_vram=vram_cushion - GIB, free_ram=9 * GIB)
    assert actions == []
    assert policy._state["vram"]["up_hold"] == 120.0

    # The two-interval grace ends at 78; the next step is allowed after that grace.
    assert policy.decide(now=73.0, free_vram=vram_cushion - GIB, free_ram=9 * GIB) == []
    assert policy.decide(now=77.0, free_vram=vram_cushion - GIB, free_ram=9 * GIB) == []
    actions = policy.decide(now=79.0, free_vram=vram_cushion - GIB, free_ram=9 * GIB)
    assert [a.direction for a in actions] == ["down"]

    # An axis the policy has not seen yet is a no-op, not a KeyError.
    policy.note_step_done("ram", 73.0)


def test_powershell_candidates_include_absolute_path_when_not_on_path(monkeypatch):
    """A systemd --user service has no /mnt/c on PATH; the absolute path must still be tried."""
    from freetoken.daemon.settings import governor

    monkeypatch.setattr(governor.shutil, "which", lambda name: None)
    monkeypatch.setattr(governor.os.path, "exists", lambda p: p == governor._POWERSHELL_ABS)
    assert governor._powershell_candidates() == [governor._POWERSHELL_ABS]
    monkeypatch.setattr(governor.os.path, "exists", lambda p: False)
    assert governor._powershell_candidates() == []
    monkeypatch.setattr(governor.shutil, "which", lambda name: "/usr/bin/powershell.exe")
    monkeypatch.setattr(governor.os.path, "exists", lambda p: True)
    assert governor._powershell_candidates() == ["/usr/bin/powershell.exe", governor._POWERSHELL_ABS]
