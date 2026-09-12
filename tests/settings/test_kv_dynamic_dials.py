"""Dynamic KV pool: the five settings-page dials, their launch mapping, the cross-field
validation, and the planner's ceiling-reachable warning (see task-11-brief.md)."""

from __future__ import annotations

from freetoken.daemon.settings import dials as d
from tests.settings.test_linux_launch import _facts, _plan


def test_the_five_dynamic_pool_dials_exist_in_the_chats_group_with_plain_words():
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


def test_validation_refuses_a_floor_off_the_page_grid():
    errors = d.validate_settings({"KVFloorTokens": 65_500})
    assert any(e["field"] == "KVFloorTokens" and "multiple of 64" in e["message"] for e in errors)


def test_validation_rechecks_the_pair_when_only_kv_dynamic_is_flipped_on():
    # A patch that only turns KVDynamic on (no KVFloorTokens/KVCacheTokens of its own) must
    # still be checked against whatever floor/ceiling the boot file already holds: flipping the
    # pool on activates that stored pair, it does not introduce a fresh one.
    context = {"KVFloorTokens": 300_000, "KVCacheTokens": 262_208, "KVDynamic": False}
    errors = d.validate_settings({"KVDynamic": True}, context=context)
    assert any(e["field"] == "KVFloorTokens" for e in errors)


def test_launch_maps_the_dynamic_pool_and_the_ttl():
    plan = _plan({
        "KVDynamic": True, "KVFloorTokens": 65_536, "KVStepTokens": 32_768, "KVShrinkIdleMin": 10,
        "KVPark": "ram", "KVParkTTLHours": 5, "KVCacheTokens": 262_208,
    })
    argv = plan.argv
    for flag, value in (("--kv-dynamic", None), ("--kv-floor-tokens", "65536"), ("--kv-step-tokens", "32768"),
                        ("--kv-shrink-idle-s", "600"), ("--kv-park-ttl-s", "18000"), ("--num-tokens", "262208")):
        assert flag in argv
        if value is not None:
            assert argv[argv.index(flag) + 1] == value


def test_launch_without_dynamic_or_parking_adds_neither():
    argv = _plan({"KVDynamic": False, "KVPark": "off"}).argv
    assert "--kv-dynamic" not in argv and "--kv-park-ttl-s" not in argv


def test_launch_skips_dynamic_pool_flags_for_a_non_moe_model():
    argv = _plan({"KVDynamic": True}, is_moe=False, expert_count=0).argv
    assert "--kv-dynamic" not in argv


def test_planner_warns_when_the_ceiling_cannot_be_reached():
    from freetoken.engine.memory_plan import kv_ceiling_issue

    KV_PAGE, SLOT = 13_248 * 64, 2_772_480
    # 940 slots needed for 65k -> 262k; only 500 above the floor
    issue = kv_ceiling_issue(floor_tokens=65_536, ceiling_tokens=262_144, lru_slots=1524, slot_floor=1024,
                             cache_per_page=KV_PAGE, page_tokens=64, per_expert_bytes=SLOT)
    assert issue["code"] == "kv_ceiling_unreachable" and "1024" in issue["message"]
    assert kv_ceiling_issue(floor_tokens=65_536, ceiling_tokens=262_144, lru_slots=7200, slot_floor=1024,
                            cache_per_page=KV_PAGE, page_tokens=64, per_expert_bytes=SLOT) is None


def test_planner_geometry_wiring_appends_the_ceiling_issue_only_when_dynamic():
    """The wiring between one resolved geometry and kv_ceiling_issue (memory_plan.py, inside
    _scenario_geometry's try block, right after the geometry dict is assembled) -- exercised
    through the small function that does that wiring, not the whole geometry solve, using the
    same byte constants as the pure-helper test above (num_experts=512, overlap on -> slot
    floor 1024)."""
    from types import SimpleNamespace

    from freetoken.engine.memory_plan import _kv_ceiling_issue_for_geometry

    KV_PAGE, SLOT = 13_248 * 64, 2_772_480
    dynamic_config = SimpleNamespace(kv_dynamic=True, kv_floor_tokens=65_536, kv_ceiling_tokens=262_144)
    issue = _kv_ceiling_issue_for_geometry(
        dynamic_config, lru_slots=1524, overlap=True, num_experts=512,
        cache_per_page=KV_PAGE, page_tokens=64, per_expert=SLOT,
    )
    assert issue is not None and issue["code"] == "kv_ceiling_unreachable" and issue["scope"] == "both"

    static_config = SimpleNamespace(kv_dynamic=False, kv_floor_tokens=65_536, kv_ceiling_tokens=262_144)
    assert _kv_ceiling_issue_for_geometry(
        static_config, lru_slots=1524, overlap=True, num_experts=512,
        cache_per_page=KV_PAGE, page_tokens=64, per_expert=SLOT,
    ) is None


def test_the_geometry_wiring_prices_the_ceiling_in_usable_tokens():
    """Final review minor 1: kv_ceiling_tokens counts the dummy page, kv_floor_tokens does not.
    The wiring drops the page so the growth the slots must fund is measured in one unit."""
    from types import SimpleNamespace

    from freetoken.engine.memory_plan import _kv_ceiling_issue_for_geometry

    KV_PAGE, SLOT = 13_248 * 64, 2_772_480
    # Exactly one page of growth (65,600 - 64 == the floor): one page is 0 slots' worth after
    # the subtraction, so the slot floor is never in the way.
    config = SimpleNamespace(kv_dynamic=True, kv_floor_tokens=65_536, kv_ceiling_tokens=65_600)
    assert _kv_ceiling_issue_for_geometry(
        config, lru_slots=1024, overlap=True, num_experts=512,
        cache_per_page=KV_PAGE, page_tokens=64, per_expert=SLOT,
    ) is None


def test_launch_switches_the_pool_off_when_the_floor_reaches_the_models_context():
    """Final review I4: KVDynamic defaults on for every MoE model, but the engine refuses
    --kv-dynamic at parse time unless the floor plus one page fits under the ceiling -- and
    with KVCacheTokens on automatic that ceiling is the model's own context. A 32k-context
    model would fail to boot on a default nobody typed; the launcher drops the flags instead."""
    small = {"KVDynamic": True, "KVCacheTokens": 0, "ContextTokens": 32_768}
    plan = _plan({**small, "KVFloorTokens": 65_536}, max_context=32_768)
    assert "--kv-dynamic" not in plan.argv and "--kv-floor-tokens" not in plan.argv
    assert any("Dynamic KV memory switched off" in note and "32768" in note for note in plan.notes)
    kept = _plan({**small, "KVFloorTokens": 16_384}, max_context=32_768)
    assert "--kv-dynamic" in kept.argv


def test_launch_judges_the_ceiling_against_an_explicit_kv_cache_tokens_first():
    argv = _plan({"KVDynamic": True, "KVFloorTokens": 65_536, "KVCacheTokens": 65_536},
                 max_context=262_144).argv
    assert "--kv-dynamic" not in argv          # floor plus a page is above the explicit ceiling


def test_validation_refuses_a_floor_at_the_models_context_when_kv_cache_tokens_is_auto():
    from freetoken.daemon.settings.model_info import ModelInfo

    model = ModelInfo(path="/models/demo", name="demo", num_experts=512, max_context_tokens=32_768)
    errors = d.validate_settings({"KVFloorTokens": 65_536, "KVCacheTokens": 0}, model)
    assert any(e["field"] == "KVFloorTokens" and "longest chat" in e["message"] for e in errors)
    assert d.validate_settings({"KVFloorTokens": 16_384, "KVCacheTokens": 0}, model) == []
