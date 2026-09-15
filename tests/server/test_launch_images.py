"""Agent model declarations follow the capability reported by the serving process."""
from freetoken.launch import (
    _codex_catalog_entry,
    _input_modalities,
    _openclaw_model_entry,
    _opencode_model_entries,
)


def test_unknown_server_defaults_to_text_input():
    assert _input_modalities({}) == ("text",)
    assert _input_modalities({"input_modalities": "image"}) == ("text",)
    assert _input_modalities({"input_modalities": ["text", "image"]}) == ("text", "image")


def test_agent_entries_declare_images_only_when_served():
    for images in (False, True):
        expected = ["text", "image"] if images else ["text"]
        assert _codex_catalog_entry("model", 8192, images)["input_modalities"] == expected
        assert _openclaw_model_entry("model", 8192, 2048, images)["input"] == expected
        opencode = _opencode_model_entries(["model"], 8192, 2048, images)["model"]
        if images:
            assert opencode["modalities"] == {"input": expected, "output": ["text"]}
        else:
            assert "modalities" not in opencode
