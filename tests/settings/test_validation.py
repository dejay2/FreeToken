from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from freetoken.daemon.settings.dials import (
    DIAL_BY_NAME,
    adapt_dial,
    minimum_slot_total,
    owned_layer_count,
    streaming_floor_slots,
    validate_settings,
)
from freetoken.daemon.settings.model_info import read_model


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


def test_validation_bounds_the_guess_depth():
    def messages(settings):
        return {item["field"]: item["message"] for item in validate_settings(settings)}

    assert "exceeds maximum 5" in messages({"FREETOKEN_MTP_SPEC_DEPTH": 6})["FREETOKEN_MTP_SPEC_DEPTH"]
    assert "below minimum 1" in messages({"FREETOKEN_MTP_SPEC_DEPTH": 0})["FREETOKEN_MTP_SPEC_DEPTH"]
    assert "FREETOKEN_MTP_SPEC_DEPTH" in messages({"FREETOKEN_MTP_SPEC_DEPTH": "lots"})
    assert validate_settings({"FREETOKEN_MTP_SPEC_DEPTH": "3"}) == []


def test_validation_bounds_the_guess_safety_catches():
    def messages(settings):
        return {item["field"]: item["message"] for item in validate_settings(settings)}

    assert "exceeds maximum 1.0" in messages({"FREETOKEN_MTP_SPEC_CONF_CUT": 1.5})["FREETOKEN_MTP_SPEC_CONF_CUT"]
    assert "below minimum 0.0" in messages({"FREETOKEN_MTP_SPEC_CONF_CUT": -0.1})["FREETOKEN_MTP_SPEC_CONF_CUT"]
    assert "exceeds maximum 6.0" in messages({"FREETOKEN_MTP_SPEC_MIN_EMITTED": 7})["FREETOKEN_MTP_SPEC_MIN_EMITTED"]
    assert "FREETOKEN_MTP_SPEC_COST_AWARE" in messages({"FREETOKEN_MTP_SPEC_COST_AWARE": "maybe"})
    assert validate_settings({"FREETOKEN_MTP_SPEC_CONF_CUT": "0.8", "FREETOKEN_MTP_SPEC_MIN_EMITTED": 2.4,
                              "FREETOKEN_MTP_SPEC_COST_AWARE": "0"}) == []


def test_validation_rejects_unknown_and_unsafe_paths():
    errors = validate_settings({"NotADial": 1, "ModelPath": "bad\x00path"})
    by_field = {item["field"]: item["message"] for item in errors}
    assert "NotADial" in by_field
    assert "ModelPath" in by_field


# ---------------------------------------------------------------------------------------
# The GPU-owned-layer charge against the expert-slot total.
#
# Live failure this guards, 2026-09-07 11:22 BST on the RTX 5090 serving box: the page saved
# MoECacheSize 4288 with GpuOwnedLayers auto:8 and the boot died immediately -- 8 x 512 = 4096
# slots charged, 192 left for the LRU, floor 1024.
# ---------------------------------------------------------------------------------------

REPO = Path(__file__).parents[2]


def _moe_model(folder: Path) -> Path:
    """A 48-layer, 512-expert MoE folder: the geometry of the live failure."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "model_type": "qwen4_exp",
                "num_hidden_layers": 48,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "max_position_embeddings": 262144,
                "hidden_size": 2560,
                "moe_intermediate_size": 640,
                "quantization_config": {"quant_algo": "NVFP4", "quant_method": "modelopt"},
            }
        ),
        encoding="utf-8",
    )
    return folder


def _dense_model(folder: Path) -> Path:
    """A model with no routed experts: the slot dials do not apply to it."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_hidden_layers": 32,
                "max_position_embeddings": 8192,
                "hidden_size": 4096,
            }
        ),
        encoding="utf-8",
    )
    return folder


def test_the_live_1122_pair_is_refused_with_the_minimum_that_would_have_worked(tmp_path):
    model = read_model(_moe_model(tmp_path / "Qwen-Like"))
    errors = validate_settings({"MoECacheSize": 4288, "GpuOwnedLayers": "auto:8"}, model)

    assert len(errors) == 1
    error = errors[0]
    assert error["field"] == "MoECacheSize"
    assert error["message"] == (
        "8 layers on the card take 4,096 of the 4,288 expert slots, leaving 192; the layers "
        "that still stream need at least 1,024. Set expert slots to 5,120 or more, or keep "
        "fewer layers on the card."
    )
    # the page offers this value; it is the engine's own charge + floor
    assert error["minimum"] == 5120 == 8 * 512 + 2 * 512


def test_a_total_exactly_on_the_floor_is_accepted(tmp_path):
    model = read_model(_moe_model(tmp_path / "Qwen-Like"))
    assert validate_settings({"MoECacheSize": 5120, "GpuOwnedLayers": "auto:8"}, model) == []
    assert validate_settings({"MoECacheSize": 5119, "GpuOwnedLayers": "auto:8"}, model)


def test_every_owned_layer_spelling_the_page_accepts_is_resolved(tmp_path):
    model = read_model(_moe_model(tmp_path / "Qwen-Like"))

    def refused(owned, slots=4288):
        return validate_settings({"MoECacheSize": slots, "GpuOwnedLayers": owned}, model)

    assert refused("auto:8") and not refused("auto:8", 5120)
    assert refused(8) and not refused(8, 5120), "a bare integer count"
    assert refused("8") and not refused("8", 5120)
    assert refused("0,1,2,3,4,5,6,7") and not refused("0,1,2,3,4,5,6,7", 5120), "an explicit list"
    assert not refused("0,1,0"), "the engine parses a list into a set: two distinct layers"
    # "auto" is the launcher's spelling of six layers: 3,072 + 1,024 = 4,096
    assert refused("auto", 4095) and not refused("auto", 4096)
    assert not refused(""), "no layers on the card, no charge"
    assert not refused(0)
    # the fraction form needs the model's layer count: 0.25 of 48 layers = 12
    assert owned_layer_count("0.25", 48) == 12 and owned_layer_count("0.25", None) is None
    assert refused("0.25") and not refused("0.25", 7168)


def test_automatic_slots_and_models_without_experts_are_exempt(tmp_path):
    model = read_model(_moe_model(tmp_path / "Qwen-Like"))
    assert validate_settings({"MoECacheSize": 0, "GpuOwnedLayers": "auto:8"}, model) == [], "0 = Automatic"

    dense = read_model(_dense_model(tmp_path / "Dense"))
    assert dense.found and not dense.is_moe
    assert validate_settings({"MoECacheSize": 4288, "GpuOwnedLayers": "auto:8"}, dense) == []
    # no model folder at all: the charge per layer is unknown, so the pair is not checked
    assert validate_settings({"MoECacheSize": 4288, "GpuOwnedLayers": "auto:8"}) == []
    assert validate_settings(
        {"MoECacheSize": 4288, "GpuOwnedLayers": "auto:8"}, model, ceilings_only=True
    ) == [], "the file writer checks only the ceilings"


def test_the_pair_is_checked_against_the_saved_half_only_when_one_half_is_touched(tmp_path):
    model = read_model(_moe_model(tmp_path / "Qwen-Like"))
    saved = {"MoECacheSize": 4288, "GpuOwnedLayers": "auto:2", "Port": 2020}

    only_layers = validate_settings({"GpuOwnedLayers": "auto:8"}, model, context=saved)
    assert only_layers and only_layers[0]["field"] == "MoECacheSize"
    only_slots = validate_settings({"MoECacheSize": 1024}, model, context={**saved, "GpuOwnedLayers": "auto:8"})
    assert only_slots and only_slots[0]["minimum"] == 5120
    assert validate_settings({"GpuOwnedLayers": "auto:2"}, model, context=saved) == []
    assert validate_settings({"Port": 2021}, model, context={**saved, "MoECacheSize": 120}) == [], (
        "a save that touches neither half must not be refused for what the file already holds"
    )
    # a value that is already refused on its own terms is not also reported as a pair
    over = validate_settings({"GpuOwnedLayers": 99}, model, context=saved)
    assert [error["field"] for error in over] == ["GpuOwnedLayers"]


def test_both_dials_explain_the_rule_with_the_model_numbers(tmp_path):
    model = read_model(_moe_model(tmp_path / "Qwen-Like"))
    slots = adapt_dial(DIAL_BY_NAME["MoECacheSize"], model)
    layers = adapt_dial(DIAL_BY_NAME["GpuOwnedLayers"], model)
    for info in (slots["info"], layers["info"]):
        assert "512 x layers on the card + 1,024" in info
    # the catalogue text (no model folder) says the same with the measured build's numbers
    for dial in (DIAL_BY_NAME["MoECacheSize"], DIAL_BY_NAME["GpuOwnedLayers"]):
        assert "512 x layers on the card + 1,024" in dial.info
    # what the page needs to print the live minimum, so it never hard-codes 512 or 1,024
    assert slots["expertsPerLayer"] == 512 and slots["streamingFloor"] == 1024
    assert slots["ownedDial"] == "GpuOwnedLayers"
    doc = DIAL_BY_NAME["MoECacheSize"].as_dict(4288, model)
    assert doc["expertsPerLayer"] == 512 and doc["streamingFloor"] == 1024 and doc["ownedDial"] == "GpuOwnedLayers"
    assert DIAL_BY_NAME["Port"].as_dict(2020, model)["expertsPerLayer"] is None


def _cache_budget():
    """The engine's own module, loaded by path (importing the package pulls in torch)."""
    spec = importlib.util.spec_from_file_location(
        "cache_budget_under_test", REPO / "python" / "freetoken" / "engine" / "cache_budget.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # freetoken.utils imports transformers, absent on the devbox
        pytest.skip(f"engine.cache_budget is not importable here: {exc}")
    return module


def test_the_page_minimum_agrees_with_the_engine_refusal():
    budget = _cache_budget()
    for experts in (64, 512):
        floor = streaming_floor_slots(experts)  # prefill overlap on: the engine default
        for layers in range(0, 5):
            minimum = minimum_slot_total(layers, experts)
            for total in (minimum - 1, minimum, minimum + 1):
                if total <= 0:
                    continue
                accepted = True
                try:
                    budget.lru_slots_after_owned_charge(
                        moe_cache_size=total, owned_layers=layers, num_experts=experts, floor=floor
                    )
                except ValueError:
                    accepted = False
                assert accepted == (total >= minimum), (experts, layers, total)
