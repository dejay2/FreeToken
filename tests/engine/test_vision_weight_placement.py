from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine.engine import _materialize_loaded_weight_state_dict
from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM


def _model_state():
    return {
        "model.embed_tokens.weight": torch.empty(2, 3, dtype=torch.bfloat16, device="meta"),
        "visual.patch_embed.proj.weight": torch.empty(
            4, 3, dtype=torch.bfloat16, device="meta"
        ),
    }


def _weights():
    return [
        ("model.embed_tokens.weight", torch.arange(6, dtype=torch.float32).view(2, 3)),
        ("visual.patch_embed.proj.weight", torch.arange(12, dtype=torch.float32).view(4, 3)),
    ]


def test_materializer_uses_per_key_destination_and_preserves_expected_dtype():
    seen = []

    def destination(key: str, engine_device: torch.device) -> torch.device:
        seen.append((key, engine_device))
        return torch.device("cpu") if key.startswith("visual.") else engine_device

    state = _materialize_loaded_weight_state_dict(
        _model_state(),
        _weights(),
        device=torch.device("meta"),
        device_for_key=destination,
    )

    assert [key for key, _device in seen] == [
        "model.embed_tokens.weight",
        "visual.patch_embed.proj.weight",
    ]
    assert all(device == torch.device("meta") for _key, device in seen)
    assert state["model.embed_tokens.weight"].device.type == "meta"
    assert state["visual.patch_embed.proj.weight"].device.type == "cpu"
    assert state["visual.patch_embed.proj.weight"].dtype is torch.bfloat16
    assert torch.equal(
        state["visual.patch_embed.proj.weight"],
        torch.arange(12, dtype=torch.bfloat16).view(4, 3),
    )


def test_materializer_without_destination_hook_keeps_one_device_behavior():
    state = _materialize_loaded_weight_state_dict(
        _model_state(), _weights(), device=torch.device("meta")
    )
    assert {tensor.device.type for tensor in state.values()} == {"meta"}


def _model_shell(mode: str) -> Qwen4ExpForCausalLM:
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._vision_execution = mode
    return model


def test_qwen_layer_stream_routes_only_visual_keys_to_cpu():
    engine_device = torch.device("meta")
    model = _model_shell("layer-stream")

    visual_device = model.weight_device_for_key(
        "visual.blocks.0.attn.qkv.weight", engine_device
    )
    assert visual_device == torch.device("cpu")
    assert model.weight_device_for_key("model.embed_tokens.weight", engine_device) == engine_device
    assert model.weight_device_for_key("lm_head.weight", engine_device) == engine_device


def test_qwen_gpu_mode_keeps_every_key_on_engine_device():
    engine_device = torch.device("meta")
    model = _model_shell("gpu")
    for key in ("visual.blocks.0.attn.qkv.weight", "model.embed_tokens.weight", "lm_head.weight"):
        assert model.weight_device_for_key(key, engine_device) == engine_device


def test_qwen_picture_weight_report_records_exact_count_bytes_and_devices():
    model = _model_shell("layer-stream")
    state = {
        "one": torch.zeros(2, 3, dtype=torch.bfloat16),
        "two": torch.zeros(5, dtype=torch.float32),
    }
    model.visual = SimpleNamespace(state_dict=lambda: state)

    assert model.weight_placement_report() == (
        "Picture weights: mode=layer-stream, tensors=2, bytes=32, devices=cpu"
    )
