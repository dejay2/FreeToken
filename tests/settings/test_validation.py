from __future__ import annotations

from freetoken.daemon.settings.dials import validate_settings


def test_validation_accepts_catalog_values():
    errors = validate_settings(
        {
            "ContextTokens": 262144,
            "KVDtype": "bf16",
            "EmbedHost": True,
            "ModelPath": r"D:\Models\Qwen3.8-Flash-Next-NVFP4",
            "KVPark": "off",
        }
    )
    assert errors == []


def test_validation_reports_bounds_and_choices():
    errors = validate_settings(
        {
            "ContextTokens": 500000,
            "KVDtype": "int4",
            "EmbedHost": "yes",
            "GpuOwnedLayers": "all",
        }
    )
    by_field = {item["field"]: item["message"] for item in errors}
    assert "ContextTokens" in by_field
    assert "exceeds maximum 262144" in by_field["ContextTokens"]
    assert "KVDtype" in by_field and "one of" in by_field["KVDtype"]
    assert "EmbedHost" in by_field
    assert "GpuOwnedLayers" in by_field


def test_validation_rejects_unknown_and_unsafe_paths():
    errors = validate_settings({"NotADial": 1, "ModelPath": "bad\x00path"})
    by_field = {item["field"]: item["message"] for item in errors}
    assert "NotADial" in by_field
    assert "ModelPath" in by_field
