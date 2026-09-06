from __future__ import annotations

from freetoken.daemon.settings.browse import BROWSE_KINDS
from freetoken.daemon.settings.dials import DIALS, EFFECT_AXES, EFFECT_DIRECTIONS, GROUP_INFO


def test_every_dial_has_plain_words_and_an_explanation():
    for dial in DIALS:
        assert dial.plain and dial.plain != dial.name, dial.name
        assert len(dial.info) >= 60, f"{dial.name} needs a real explanation, not a stub"
        assert dial.group in GROUP_INFO, dial.name


def test_effects_use_the_known_axes_and_directions():
    for dial in DIALS:
        for effect in dial.effects:
            axis, direction = effect.split(":", 1)
            assert axis in EFFECT_AXES, (dial.name, effect)
            assert direction in EFFECT_DIRECTIONS, (dial.name, effect)
        rendered = dial.as_dict(dial.default)["effects"]
        assert all(set(item) == {"axis", "direction"} for item in rendered)


def test_sliders_sit_inside_the_validation_bounds():
    for dial in DIALS:
        if dial.slider is None:
            continue
        low, high, step = dial.slider
        assert dial.control == "number", dial.name
        assert low < high and step > 0, dial.name
        factor = dial.display_factor
        assert dial.minimum is None or low * factor >= dial.minimum, dial.name
        assert dial.maximum is None or high * factor <= dial.maximum, dial.name


def test_auto_sentinels_and_browse_kinds_are_consistent():
    for dial in DIALS:
        if dial.auto_value is not None:
            assert dial.control == "number", dial.name
            assert dial.minimum is None or dial.auto_value >= dial.minimum, dial.name
        if dial.browse:
            assert dial.control == "path", dial.name
            assert dial.browse in BROWSE_KINDS, dial.name
        if dial.option_labels is not None:
            assert dial.options is not None and len(dial.option_labels) == len(dial.options), dial.name


def test_the_guess_depth_dial_is_a_small_slider_and_the_fast_path_is_not_hidden():
    from freetoken.daemon.settings.dials import DIAL_BY_NAME

    depth = DIAL_BY_NAME["FREETOKEN_MTP_SPEC_DEPTH"]
    assert depth.control == "number" and depth.source == "env"
    assert (depth.minimum, depth.maximum, depth.default) == (1, 5, 5)
    assert depth.slider == (1, 5, 1) and not depth.advanced
    assert depth.group == "Look-ahead speed trick (MTP)"
    graph = DIAL_BY_NAME["FREETOKEN_MTP_SPEC_GRAPH"]
    assert graph.default == "1" and not graph.advanced, "off is never the faster choice; measured 2026-09-05"


def test_as_dict_carries_the_page_contract():
    doc = DIALS[0].as_dict("x")
    for key in (
        "name", "group", "control", "value", "plain", "info", "effects", "slider", "displayUnit",
        "displayFactor", "autoValue", "autoLabel", "browse", "advanced", "engine", "optionLabels",
    ):
        assert key in doc, key
