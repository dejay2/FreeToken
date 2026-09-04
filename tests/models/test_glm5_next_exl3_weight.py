"""CPU-safe key mapping tests for GLM-5.3's non-routed EXL3 tensors."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import pytest

from freetoken.models.glm5_next.weight import (
    _iter_dsa_layer,
    _iter_kda_layer,
    _read_linear,
)


class _Reader:
    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}

    def has(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str) -> torch.Tensor:
        return self.values[key]


def _put_exl3(
    reader: _Reader,
    base: str,
    *,
    in_features: int = 128,
    out_features: int = 128,
    k: int = 2,
) -> None:
    reader.values[f"{base}.trellis"] = torch.zeros(
        (in_features // 16, out_features // 16, 16 * k), dtype=torch.int16
    )
    reader.values[f"{base}.suh"] = torch.ones(in_features, dtype=torch.float16)
    reader.values[f"{base}.svh"] = torch.ones(out_features, dtype=torch.float16)
    reader.values[f"{base}.mul1"] = torch.tensor(0, dtype=torch.int32)


def test_read_linear_reconstructs_exl3_and_keeps_plain_weights(monkeypatch):
    from freetoken.kernel import exl3

    calls: list[tuple[int, str]] = []

    def fake_reconstruct(trellis, suh, svh, *, k, codebook, out=None, work=None):
        calls.append((k, codebook))
        result = torch.full(
            (trellis.shape[1] * 16, trellis.shape[0] * 16),
            2,
            dtype=torch.bfloat16,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    monkeypatch.setattr(exl3, "reconstruct", fake_reconstruct)
    reader = _Reader()
    _put_exl3(reader, "layer.exl3")
    reconstructed = _read_linear(reader, "layer.exl3")

    assert reconstructed.shape == (128, 128)
    assert reconstructed.dtype == torch.bfloat16
    assert calls == [(2, "mul1")]

    plain = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    reader.values["layer.plain.weight"] = plain
    assert _read_linear(reader, "layer.plain") is plain
    assert calls == [(2, "mul1")]


def test_read_linear_rejects_mcg_and_missing_components():
    reader = _Reader()
    reader.values["layer.mcg"] = torch.tensor(0, dtype=torch.int32)
    with pytest.raises(ValueError, match="mcg"):
        _read_linear(reader, "layer")

    reader = _Reader()
    _put_exl3(reader, "layer")
    del reader.values["layer.mul1"]
    with pytest.raises(KeyError, match="mul1"):
        _read_linear(reader, "layer")


def test_kda_maps_fused_qkv_and_merged_convolution(monkeypatch):
    from freetoken.kernel import exl3

    calls: list[str] = []

    def fake_reconstruct(trellis, suh, svh, *, k, codebook, out=None, work=None):
        calls.append(codebook)
        return torch.zeros(
            (trellis.shape[1] * 16, trellis.shape[0] * 16), dtype=torch.bfloat16
        )

    monkeypatch.setattr(exl3, "reconstruct", fake_reconstruct)
    reader = _Reader()
    src = "model.language_model.layers.0.self_attn"
    _put_exl3(reader, f"{src}.qkv_proj")
    _put_exl3(reader, f"{src}.o_proj")
    for name, rows in (("b_proj", 2), ("f_a_proj", 3), ("g_a_proj", 4)):
        reader.values[f"{src}.{name}.weight"] = torch.zeros((rows, 128), dtype=torch.bfloat16)
    reader.values[f"{src}.conv1d.weight"] = torch.zeros((6, 1, 4), dtype=torch.bfloat16)
    for name in ("f_b_proj", "g_b_proj"):
        reader.values[f"{src}.{name}.weight"] = torch.zeros((2, 128), dtype=torch.bfloat16)
    reader.values[f"{src}.A_log"] = torch.zeros(8)
    reader.values[f"{src}.dt_bias"] = torch.zeros(8)
    reader.values[f"{src}.o_norm.weight"] = torch.ones(128, dtype=torch.bfloat16)

    out = dict(_iter_kda_layer(reader, 0, attn_fp8=False))

    assert out["model.layers.0.self_attn.in_proj.weight"].shape == (137, 128)
    assert out["model.layers.0.self_attn.conv1d.weight"].shape == (6, 1, 4)
    assert out["model.layers.0.self_attn.o_proj.weight"].shape == (128, 128)
    assert out["model.layers.0.self_attn.A_log"].dtype == torch.float32
    assert len(calls) == 2


def test_dsa_maps_mixed_exl3_projections_and_indexer(monkeypatch):
    from freetoken.kernel import exl3

    calls: list[str] = []

    def fake_reconstruct(trellis, suh, svh, *, k, codebook, out=None, work=None):
        calls.append(codebook)
        return torch.zeros(
            (trellis.shape[1] * 16, trellis.shape[0] * 16), dtype=torch.bfloat16
        )

    monkeypatch.setattr(exl3, "reconstruct", fake_reconstruct)
    reader = _Reader()
    src = "model.language_model.layers.3.self_attn"
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
        _put_exl3(reader, f"{src}.{name}")
    reader.values[f"{src}.kv_b_proj.weight"] = torch.zeros((128, 128), dtype=torch.bfloat16)
    for name in ("q_a_layernorm", "kv_a_layernorm"):
        reader.values[f"{src}.{name}.weight"] = torch.ones(128, dtype=torch.bfloat16)
    _put_exl3(reader, f"{src}.indexer.wq_b")
    for name in ("wk", "weights_proj"):
        reader.values[f"{src}.indexer.{name}.weight"] = torch.zeros(
            (128, 128), dtype=torch.bfloat16
        )
    reader.values[f"{src}.indexer.k_norm.weight"] = torch.ones(128, dtype=torch.bfloat16)
    reader.values[f"{src}.indexer.k_norm.bias"] = torch.zeros(128, dtype=torch.bfloat16)
    reader.values[f"{src}.indexer.index_kpool_compress_gate"] = torch.zeros(8)
    reader.values[f"{src}.indexer.index_kpool_compress_ape"] = torch.zeros(8)

    out = dict(_iter_dsa_layer(reader, 3, attn_fp8=False))

    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        assert out[f"model.layers.3.self_attn.{name}.weight"].dtype == torch.bfloat16
    assert out["model.layers.3.self_attn.indexer.wq_b.weight"].shape == (128, 128)
    assert out["model.layers.3.self_attn.indexer.index_kpool_compress_ape"].dtype == torch.float32
    assert len(calls) == 5  # four attention projections plus indexer.wq_b


def test_iter_weights_routes_dense_shared_and_head_through_exl3_helper(tmp_path, monkeypatch):
    from freetoken.kernel import exl3
    from freetoken.models.glm5_next import weight as glm_weight

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}}), encoding="utf-8"
    )
    bases = (
        *(f"model.language_model.layers.0.mlp.{proj}" for proj in ("gate_proj", "up_proj", "down_proj")),
        *(f"model.language_model.layers.1.mlp.shared_experts.{proj}"
          for proj in ("gate_proj", "up_proj", "down_proj")),
        "lm_head",
    )

    class FullReader(_Reader):
        def __init__(self, folder, weight_map, device):
            del folder, weight_map, device
            super().__init__()
            for base in bases:
                _put_exl3(self, base)

        def get(self, key: str) -> torch.Tensor:
            return self.values.get(key, torch.zeros(128, dtype=torch.bfloat16))

        def close(self) -> None:
            pass

    calls: list[tuple[int, str]] = []

    def fake_reconstruct(trellis, suh, svh, *, k, codebook, out=None, work=None):
        calls.append((k, codebook))
        result = torch.zeros(
            (trellis.shape[1] * 16, trellis.shape[0] * 16), dtype=torch.bfloat16
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    config = SimpleNamespace(
        num_layers=2,
        first_k_dense_replace=1,
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        tie_word_embeddings=False,
        glm5_args=SimpleNamespace(is_kda_layer=lambda layer: False),
    )
    monkeypatch.setattr(glm_weight, "_ShardReader", FullReader)
    monkeypatch.setattr(glm_weight, "cached_load_hf_config", lambda path: object())
    monkeypatch.setattr(glm_weight, "download_hf_weight", lambda path: str(tmp_path))
    monkeypatch.setattr(glm_weight, "parse_config", lambda hf: config)
    monkeypatch.setattr(
        glm_weight,
        "get_tp_info",
        lambda: SimpleNamespace(size=1, is_primary=lambda: False),
    )
    monkeypatch.setattr(glm_weight, "_iter_dsa_layer", lambda *args: iter(()))
    monkeypatch.setattr(exl3, "reconstruct", fake_reconstruct)

    output = dict(
        glm_weight.iter_weights(
            "fixture", torch.device("cpu"), include_moe_experts=False, include_non_moe=True
        )
    )

    expected_outputs = {
        *(f"model.layers.0.mlp.{proj}.weight" for proj in ("gate_proj", "up_proj", "down_proj")),
        *(f"model.layers.1.mlp.shared_experts.{proj}.weight"
          for proj in ("gate_proj", "up_proj", "down_proj")),
        "lm_head.weight",
    }
    assert expected_outputs <= output.keys()
    assert calls == [(2, "mul1")] * len(bases)
