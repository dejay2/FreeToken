"""Import of today's switcher config. Review focus 5: it round-trips to an equivalent config."""

from __future__ import annotations

from pathlib import Path

import pytest

from freetoken.daemon.settings.registry import find_model
from freetoken.daemon.settings.registry_import import ImportRefused, import_live
from freetoken.daemon.settings.swap_config import compare, render_config, semantic_models
from tests.settings.registry_fixtures import FT_DEFAULTS, five

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO / "engines" / "config" / "config.example.yaml").read_text(encoding="utf-8")
ENV = {"HOME": "/home/jay"}


def test_example_config_imports_to_the_five_model_registry():
    registry, warnings = import_live(EXAMPLE, FT_DEFAULTS, env=ENV)
    assert registry == five()
    assert warnings == []


def test_imported_registry_generates_an_equivalent_config():
    registry, _ = import_live(EXAMPLE, FT_DEFAULTS, env=ENV)
    generated = render_config(registry, {})
    assert compare(EXAMPLE, generated, env=ENV) == []
    assert semantic_models(generated, env=ENV)["qwen3.8-flash"]["launch"]["profile"] == "model-qwen3.8-flash"
    assert semantic_models(EXAMPLE, env=ENV)["qwen3.8-flash"]["launch"]["profile"] is None


def test_compare_notices_a_real_difference():
    registry, _ = import_live(EXAMPLE, FT_DEFAULTS, env=ENV)
    find_model(registry, "quasar-27b")["overrides"]["draft-tokens"] = 5
    diffs = compare(EXAMPLE, render_config(registry, {}), env=ENV)
    assert len(diffs) == 1 and diffs[0].startswith("quasar-27b.launch:")


def test_unknown_ninfer_option_refuses_the_import():
    text = EXAMPLE.replace("--draft-tokens 7", "--draft-tokens 7 --turbo")
    with pytest.raises(ImportRefused, match="--turbo"):
        import_live(text, FT_DEFAULTS, env=ENV)


def test_unknown_adapter_refuses_the_import():
    text = EXAMPLE.replace("${ft} ${env.HOME}/models/Qwen3.8-Flash-Next-NVFP4\n", "/usr/bin/llama-server -m x.gguf\n", 1)
    with pytest.raises(ImportRefused, match="unknown adapter"):
        import_live(text, FT_DEFAULTS, env=ENV)


def test_values_the_panel_does_not_keep_are_reported():
    old = "    unloadTimeout: 60\n    ramNeedGB: 18\n    ttl: 0\n    env:"
    assert old in EXAMPLE
    text = EXAMPLE.replace(old, "    unloadTimeout: 90\n    ramNeedGB: 18\n    ttl: 90\n    env:", 1)
    registry, warnings = import_live(text, FT_DEFAULTS, env=ENV)
    assert any("unloadTimeout 90" in warning for warning in warnings), warnings
    assert any("ttl 90" in warning for warning in warnings), warnings
    assert find_model(registry, "quasar-27b")["idleMinutes"] == 2


def test_profiles_become_presets_on_matching_freetoken_models():
    profiles = [
        {"id": "default", "kind": "default", "name": "Default", "settings": {"KVDtype": "bf16"}},
        {"id": "profile-a", "kind": "preset", "isPreset": True, "name": "Profile A (BF16 Default)",
         "settings": {"KVDtype": "bf16", "MoECacheSize": 4188}},
        {"id": "prof-1", "kind": "profile", "name": "Long chats",
         "settings": {"ModelPath": "/home/jay/models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4", "ContextTokens": 131072}},
        {"id": "model-qwen3.8-flash", "kind": "profile", "name": "ours", "settings": {"KVDtype": "bf16"}},
    ]
    registry, _ = import_live(EXAMPLE, FT_DEFAULTS, env=ENV, profiles=profiles)
    flash = find_model(registry, "qwen3.8-flash")
    ablit = find_model(registry, "qwen3.8-flash-abliterated")
    assert flash["presets"] == {"Profile A (BF16 Default)": {"KVDtype": "bf16", "MoECacheSize": 4188}}
    assert ablit["presets"]["Long chats"] == {"ContextTokens": 131072}
    assert flash["activePreset"] is None and "ours" not in flash["presets"]


def test_a_changed_check_endpoint_is_reported():
    text = EXAMPLE.replace(
        "checkEndpoint: /ready?model=Qwen3.8-Flash-Next-NVFP4",
        "checkEndpoint: /totally-custom-endpoint",
        1,
    )
    registry, warnings = import_live(text, FT_DEFAULTS, env=ENV)
    assert any("checkEndpoint" in warning and "/totally-custom-endpoint" in warning for warning in warnings), warnings
    # the value is still not stored anywhere in the registry
    flash = find_model(registry, "qwen3.8-flash")
    assert "checkEndpoint" not in flash


def test_a_changed_use_model_name_is_reported():
    text = EXAMPLE.replace('useModelName: "quasar-27b"', 'useModelName: "something-else"', 1)
    registry, warnings = import_live(text, FT_DEFAULTS, env=ENV)
    assert any("useModelName" in warning and "something-else" in warning for warning in warnings), warnings
    quasar = find_model(registry, "quasar-27b")
    assert quasar["id"] == "quasar-27b"


def test_empty_or_unreadable_config_refuses():
    with pytest.raises(ImportRefused, match="lists no models"):
        import_live("healthCheckTimeout: 600\n", FT_DEFAULTS, env=ENV)
    with pytest.raises(ImportRefused, match="not readable"):
        import_live("models: [unclosed\n", FT_DEFAULTS, env=ENV)
