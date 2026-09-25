from __future__ import annotations

import json
import re
import struct
from pathlib import Path

from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.dials import (
    DIAL_BY_NAME,
    MODEL_AWARE_DIALS,
    adapt_dial,
    canonical_value,
    stored_count,
    validate_settings,
)
from freetoken.daemon.settings import model_info
from freetoken.daemon.settings.model_info import SUPPORTED_ARCHITECTURES, expert_format, read_model
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.profiles_manager import ProfilesManager

REPO = Path(__file__).parents[2]


def _qwen_like(folder: Path, **overrides) -> Path:
    """A config.json shaped like the shipping Qwen3.8-Flash-Next-NVFP4 build (text_config nested,
    modelopt NVFP4 on the experts) with the sizes that give 2,772,480 bytes per expert."""
    folder.mkdir(parents=True, exist_ok=True)
    text = {
        "model_type": "qwen4_exp_text",
        "max_position_embeddings": 262144,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "hidden_size": 2560,
        "moe_intermediate_size": 640,
        "ple_layer_ids": [2],
        "mtp_num_hidden_layers": 1,
    }
    text.update(overrides.pop("text", {}))
    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "quantization_config": {"quant_algo": "NVFP4", "quant_method": "modelopt"},
        "text_config": text,
        "vision_config": {"depth": 1},
    }
    config.update(overrides)
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 10)
    (folder / "model-00002-of-00002.safetensors").write_bytes(b"\0" * 6)
    return folder


def _small_fp8(folder: Path) -> Path:
    """A flat-config 8-bit MoE with two leading dense layers and a 32k context."""
    folder.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "max_position_embeddings": 32768,
        "num_hidden_layers": 26,
        "first_k_dense_replace": 2,
        "n_routed_experts": 64,
        "num_experts_per_tok": 4,
        "hidden_size": 2048,
        "moe_intermediate_size": 1024,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"\0")
    return folder


def test_reference_model_sizes_match_the_measured_numbers(tmp_path):
    info = read_model(_qwen_like(tmp_path / "Qwen-Like"))
    assert info.found and info.supported and info.is_reference
    assert info.max_context_tokens == 262144
    assert info.num_moe_layers == 48 and info.num_experts == 512 and info.experts_per_token == 10
    assert info.expert_format == "nvfp4"
    # 2.77 MB per slot, 2.58 GiB per 1,000 slots, 1.32 GiB per layer (memory audit, 2026-09-02)
    assert info.bytes_per_expert == 2_772_480
    assert round(info.bytes_per_expert * 1000 / 1024 ** 3, 2) == 2.58
    assert round(info.bytes_per_layer / 1024 ** 3, 2) == 1.32
    assert info.has_vision and info.has_ple and info.has_mtp
    assert info.weight_files == 2 and info.weight_bytes == 16
    doc = info.as_dict()
    assert doc["isReference"] is True and doc["bytesPerExpert"] == 2_772_480 and doc["isMoe"] is True


def test_read_model_counts_ple_payload_from_safetensors_header(tmp_path):
    folder = _qwen_like(tmp_path / "Qwen-With-PLE")
    header = json.dumps(
        {
            "model.layers.2.ple_table": {"dtype": "U8", "shape": [13], "data_offsets": [0, 13]},
            "model.embed_ngram.weight": {"dtype": "U8", "shape": [14], "data_offsets": [13, 27]},
            "model.layers.2.mlp.gate_proj": {"dtype": "U8", "shape": [5], "data_offsets": [27, 32]},
            "__metadata__": {"format": "pt"},
        }
    ).encode("utf-8")
    payload = struct.pack("<Q", len(header)) + header + (b"\\0" * 32)
    (folder / "model-00001-of-00002.safetensors").write_bytes(payload)

    info = read_model(folder)

    assert info.ple_bytes == 27
    assert info.as_dict()["pleBytes"] == 27


def test_other_model_gets_its_own_layers_context_and_format(tmp_path):
    info = read_model(_small_fp8(tmp_path / "Small-FP8"))
    assert info.found and info.supported and not info.is_reference
    assert info.max_context_tokens == 32768
    assert info.num_layers == 26 and info.num_moe_layers == 24 and info.num_experts == 64
    assert info.expert_format == "fp8_block" and info.bytes_per_expert
    assert not info.has_vision and not info.has_ple and not info.has_mtp


def test_unreadable_folders_come_back_with_a_plain_error(tmp_path):
    assert read_model("").error == "No model folder is set."
    assert "does not exist" in read_model(str(tmp_path / "nope")).error
    empty = tmp_path / "Pictures"
    empty.mkdir()
    assert "config.json" in read_model(str(empty)).error
    assert read_model("$env:MODELS").found is False
    broken = tmp_path / "Broken"
    broken.mkdir()
    (broken / "config.json").write_text("{not json", encoding="utf-8")
    assert read_model(str(broken)).found is False


def test_expert_format_detection():
    assert expert_format({}) == "bf16"
    assert expert_format({"quantization_config": {"quant_algo": "NVFP4"}}) == "nvfp4"
    assert expert_format({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}) == "fp8_block"
    assert expert_format({"quantization_config": {"quant_method": "mxfp4"}}) == "mxfp4"
    assert expert_format({"quantization_config": {"quant_method": "mystery"}}) == ""


def test_exl3_expert_format_and_size(tmp_path):
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"],
           "quantization_config": {"quant_method": "exl3", "bits": 3.05, "head_bits": 5},
           "text_config": {"hidden_size": 2560, "moe_intermediate_size": 640, "num_experts": 512,
                           "num_hidden_layers": 48, "num_experts_per_tok": 10}}
    assert model_info.expert_format(cfg) == "exl3"
    info = model_info.describe_config(cfg, "q")
    assert info.expert_format_label == "EXL3 (3-bit experts)"
    assert info.bytes_per_expert == 3 * 160 * 40 * 48 * 2 + 2 * (2560 + 640) * 2 + (640 + 2560) * 2
    assert info.extra["exl3_expert_k"] == 3


def test_exl3_bytes_match_engine_formula():
    from types import SimpleNamespace

    from freetoken.moe.offload_cache import bank_bytes_per_expert  # torch import is fine in tests

    assert model_info.expert_bytes("exl3", 2560, 640, k=3) == bank_bytes_per_expert(
        "exl3", 2560, 640, SimpleNamespace(exl3_expert_k=3))


def test_exl3_garbled_bits_gives_an_unknown_size_not_a_crash(tmp_path):
    """Fix round 1, Critical: int(float(None)) used to raise uncaught out of describe_config
    (and therefore out of read_model), breaking panel._model_limit_errors' all-models loop for
    every model on the page over one bad EXL3 entry. None/non-numeric/out-of-range bits must
    leave the estimate unknown instead of guessing K=2."""
    base = {"architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": {"hidden_size": 2560, "moe_intermediate_size": 640, "num_experts": 512,
                             "num_hidden_layers": 48, "num_experts_per_tok": 10}}
    for bad_bits in (None, "unknown", 12.5):
        cfg = {**base, "quantization_config": {"quant_method": "exl3", "bits": bad_bits}}
        info = model_info.describe_config(cfg, "q")
        assert info.expert_format == "exl3"
        assert info.expert_format_label == "EXL3 (unknown bits)"
        assert info.bytes_per_expert is None
        assert info.bytes_per_layer is None
        assert info.total_expert_bytes is None
        assert "exl3_expert_k" not in info.extra
        assert "exl3_bits_error" in info.extra


def test_a_garbled_exl3_model_does_not_break_reading_the_others(tmp_path):
    """The shape of panel._model_limit_errors' loop: read_model() over several registered
    models, one of which has a garbled EXL3 bits value, must not raise and must still size the
    healthy models correctly."""
    garbled = tmp_path / "Garbled-EXL3"
    garbled.mkdir()
    (garbled / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "quantization_config": {"quant_method": "exl3", "bits": None},
        "text_config": {"hidden_size": 2560, "moe_intermediate_size": 640, "num_experts": 512,
                        "num_hidden_layers": 48, "num_experts_per_tok": 10},
    }), encoding="utf-8")
    (garbled / "model.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")

    nvfp4 = _qwen_like(tmp_path / "Qwen-Like")

    infos = [read_model(str(folder)) for folder in (garbled, nvfp4)]

    assert infos[0].found and infos[0].expert_format == "exl3" and infos[0].bytes_per_expert is None
    assert infos[1].found and infos[1].expert_format == "nvfp4" and infos[1].bytes_per_expert == 2_772_480


def test_ple_bytes_skips_the_unconverted_exl3_trellis_table():
    """turboderp's ngram_embedding.safetensors is not FreeToken's PLE table; only the converted
    freetoken-ple-*.safetensors shards (no such metadata marker) should be counted."""
    header = json.dumps({
        "ngram_embedding.head_offsets": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]},
        "__metadata__": {"format": "exl3_ngram_trellis", "version": "1"},
    }).encode("utf-8")
    assert model_info.ple_bytes_from_header(header) == 0


def test_supported_architectures_mirror_the_engine_registry():
    source = (REPO / "python" / "freetoken" / "models" / "register.py").read_text(encoding="utf-8")
    # underscore included: Qwen3_5* keys were invisible to the first version of this pattern
    registered = set(re.findall(r'^\s{4}"([A-Za-z0-9_]+)":\s*ModelSpec\(', source, flags=re.M))
    assert registered, "could not find the registry keys in register.py"
    assert "Qwen3_5MoeForConditionalGeneration" in registered, "the pattern must see underscored keys"
    assert registered == set(SUPPORTED_ARCHITECTURES)


def test_adapted_dials_follow_the_model(tmp_path):
    ref = read_model(_qwen_like(tmp_path / "Qwen-Like"))
    small = read_model(_small_fp8(tmp_path / "Small-FP8"))

    context = DIAL_BY_NAME["ContextTokens"]
    assert adapt_dial(context, ref)["max"] == 262144
    small_context = adapt_dial(context, small)
    assert small_context["max"] == 32768 and small_context["slider"][1] == 32768

    layers = DIAL_BY_NAME["GpuOwnedLayers"]
    assert adapt_dial(layers, ref)["slider"] == (0, 48, 1) and adapt_dial(layers, ref)["storedAs"] == "auto:{n}"
    assert adapt_dial(layers, small)["max"] == 24 and adapt_dial(layers, small)["storedAs"] == "{n}"
    assert "1.32 GiB" in adapt_dial(layers, ref)["info"]
    assert "measured for Qwen3.8 only" in adapt_dial(layers, small)["info"]

    slots = DIAL_BY_NAME["MoECacheSize"]
    assert adapt_dial(slots, ref)["slider"][1] <= 48 * 512
    assert adapt_dial(slots, small)["slider"][1] <= 24 * 64
    assert adapt_dial(slots, small)["max"] == 24 * 64
    assert "expert pieces" in adapt_dial(slots, small)["limitNote"]
    assert "2.58 GiB" in adapt_dial(slots, ref)["info"]
    assert "not been measured" in adapt_dial(slots, small)["info"]

    # dials outside the model-aware set change nothing; an unreadable model folder falls back
    # to the limits the catalogue was measured with
    assert adapt_dial(DIAL_BY_NAME["Port"], ref) == {}
    assert adapt_dial(context, read_model(""))["max"] == 262144
    assert adapt_dial(layers, None)["max"] == 48
    for name in MODEL_AWARE_DIALS:
        assert name in DIAL_BY_NAME, name
    doc = layers.as_dict("auto:3", ref)
    assert doc["storedAs"] == "auto:{n}" and doc["max"] == 48 and doc["modelAware"] is True


def test_layers_count_round_trips_through_launcher_text():
    layers = DIAL_BY_NAME["GpuOwnedLayers"]
    assert stored_count(layers, "auto") == 6
    assert stored_count(layers, "auto:3") == 3
    assert stored_count(layers, "") == 0
    assert stored_count(layers, 4) == 4
    assert stored_count(layers, "2,4,6") == 3
    assert canonical_value(layers, 3) == "auto:3"
    assert canonical_value(layers, 0) == ""
    assert canonical_value(layers, "5") == "5", "a plain count typed in the file keeps its spelling"
    assert canonical_value(layers, 5, "{n}") == "5"
    assert canonical_value(layers, "1,3") == "1,3"


def test_validation_uses_the_model_limits(tmp_path):
    small = read_model(_small_fp8(tmp_path / "Small-FP8"))
    errors = validate_settings({"ContextTokens": 65536, "GpuOwnedLayers": 30}, small)
    fields = {error["field"]: error["message"] for error in errors}
    assert "cannot read a longer chat" in fields["ContextTokens"]
    assert "24 expert layers" in fields["GpuOwnedLayers"]
    assert validate_settings({"ContextTokens": 65536, "GpuOwnedLayers": 30}, ceilings_only=True) == [], "the file writer checks only the ceilings"
    assert validate_settings({"ContextTokens": 300000}) and validate_settings({"ContextTokens": 262144}) == [], "no model: the built-in limits apply"
    assert validate_settings({"ContextTokens": 32768, "GpuOwnedLayers": "auto:24"}, small) == []


def _client(tmp_path, model_folder: Path) -> TestClient:
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text(
        f"& $launcher `\n    -ModelPath '{model_folder}' `\n    -ContextTokens 262144 `\n    -GpuOwnedLayers auto `\n    -Port 2020\n",
        encoding="utf-8",
    )
    proc = ProcessManager(
        boot_file=boot,
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *a, **k: None,
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
    )
    app = create_app(
        boot_file=boot,
        process_manager=proc,
        profiles=ProfilesManager(tmp_path / "boot-profiles.json"),
        log_path=tmp_path / "server.log",
        static_path=tmp_path / "missing-index.html",
    )
    return TestClient(app)


def test_routes_shape_dials_for_the_saved_and_previewed_model(tmp_path):
    ref = _qwen_like(tmp_path / "Qwen-Like")
    small = _small_fp8(tmp_path / "Small-FP8")
    client = _client(tmp_path, ref)

    doc = client.get("/api/settings").json()
    assert doc["model"]["name"] == "Qwen-Like" and doc["model"]["isReference"] is True
    by_name = {dial["name"]: dial for dial in doc["dials"]}
    assert by_name["ContextTokens"]["max"] == 262144
    assert by_name["GpuOwnedLayers"]["slider"] == [0, 48, 1] and by_name["GpuOwnedLayers"]["value"] == "auto"

    preview = client.get("/api/settings", params={"model": str(small)}).json()
    assert preview["model"]["name"] == "Small-FP8"
    previewed = {dial["name"]: dial for dial in preview["dials"]}
    assert previewed["ContextTokens"]["max"] == 32768 and previewed["GpuOwnedLayers"]["max"] == 24
    assert previewed["GpuOwnedLayers"]["storedAs"] == "{n}"

    described = client.get("/api/model", params={"path": str(small)}).json()
    assert described["found"] is True and described["numMoeLayers"] == 24
    assert client.get("/api/model", params={"path": str(tmp_path / "nowhere")}).json()["found"] is False

    # Saving a longer chat than the previewed model allows is refused with the model's name.
    refused = client.put("/api/settings", json={"settings": {"ModelPath": str(small), "ContextTokens": 65536}})
    assert refused.status_code == 422
    assert "Small-FP8" in refused.json()["detail"][0]["message"]

    # The same value is fine for the saved (reference) model, and the layers slider value is
    # written back in launcher text.
    saved = client.put("/api/settings", json={"settings": {"ContextTokens": 65536, "GpuOwnedLayers": 3}})
    assert saved.status_code == 200, saved.text
    text = (tmp_path / "boot-2020.ps1").read_text(encoding="utf-8")
    assert "-ContextTokens 65536" in text and "-GpuOwnedLayers auto:3" in text
    assert client.get("/api/settings").json()["settings"]["GpuOwnedLayers"] == "auto:3"

    # Switching the saved model re-checks what the file already holds (R1 finding F2): the
    # 65,536-token chat saved above does not fit the 32k model, so the switch is refused
    # until the chat length is lowered in the same save.
    switch = client.put("/api/settings", json={"settings": {"ModelPath": str(small)}})
    assert switch.status_code == 422 and switch.json()["detail"][0]["field"] == "ContextTokens"
    slot_switch = client.put(
        "/api/settings",
        json={"settings": {"ModelPath": str(small), "ContextTokens": 32768}},
    )
    assert slot_switch.status_code == 422
    assert slot_switch.json()["detail"][0]["field"] == "MoECacheSize"
    assert "1,536 expert pieces" in slot_switch.json()["detail"][0]["message"]
    both = client.put(
        "/api/settings",
        json={
            "settings": {
                "ModelPath": str(small),
                "ContextTokens": 32768,
                "MoECacheSize": 1536,
                "GpuOwnedLayers": 2,
            }
        },
    )
    assert both.status_code == 200, both.text
    text = (tmp_path / "boot-2020.ps1").read_text(encoding="utf-8")
    assert "-ContextTokens 32768" in text
    assert re.search(r"-GpuOwnedLayers 2\b", text), "a non-reference model stores a plain count"


def test_profile_apply_is_checked_against_the_model(tmp_path):
    small = _small_fp8(tmp_path / "Small-FP8")
    client = _client(tmp_path, small)
    created = client.post("/api/profiles", json={"name": "Big chat", "description": "", "settings": {"ContextTokens": 262144}})
    assert created.status_code == 201, created.text
    profile_id = created.json()["id"]
    refused = client.post(f"/api/profiles/{profile_id}/apply")
    assert refused.status_code == 422 and "Small-FP8" in refused.json()["detail"][0]["message"]
    assert "-ContextTokens 262144" in (tmp_path / "boot-2020.ps1").read_text(encoding="utf-8"), "the file was left alone"


def test_hand_typed_fraction_is_kept_as_typed():
    layers = DIAL_BY_NAME["GpuOwnedLayers"]
    assert stored_count(layers, "0.125") == 0
    assert canonical_value(layers, "0.125") == "0.125"
    assert validate_settings({"GpuOwnedLayers": "0.125"}) == []
    assert validate_settings({"GpuOwnedLayers": "1.5"}), "fractions above 1 are refused"
