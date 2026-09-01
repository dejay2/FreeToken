from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from freetoken.layers import BaseOP, OPList
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.models.qwen4_exp.mtp_spike import (
    MTPWeightStore,
    Qwen4ExpMTPModel,
    build_mtp_weight_plan,
    derive_mtp_model_config,
)
from freetoken.utils.torch_utils import torch_dtype

from .common import parsed_config


def _norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * (1.0 + weight.float())).to(x.dtype)


def _new_model(config) -> Qwen4ExpMTPModel:
    with torch.device("cpu"), torch_dtype(torch.bfloat16):
        return Qwen4ExpMTPModel(config)


def _fill(op: BaseOP, seed: int = 5) -> None:
    generator = torch.Generator().manual_seed(seed)
    for tensor in op.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, 0.05, generator=generator)
        else:
            tensor.zero_()
    if isinstance(op, Qwen4ExpMTPModel):
        start = op.config.qwen4_args.hc_lowrank + op.config.qwen4_args.hc_count
        for layer_hc in (
            op.layers.op_list[0].attn_hyper_connection,
            op.layers.op_list[0].mlp_hyper_connection,
        ):
            layer_hc.input_mix_weight_down_block_inject.weight[start:].zero_()


def test_derived_config_is_one_qsa_layer_without_ple_gdn_or_vision():
    base = parsed_config(num_layers=4)
    config = derive_mtp_model_config(base)

    assert config.num_layers == 1
    assert len(config.attention_groups) == 1
    group = config.attention_groups[0]
    assert isinstance(group, FullAttentionGroupConfig)
    assert group.layer_ids == (0,)
    assert group.index_head_dim == base.qwen4_args.index_head_dim
    assert group.index_ratio == base.qwen4_args.index_ratio
    assert group.num_index_layers == 1
    assert config.is_linear_layer(0) is False
    assert config.has_linear_attention is False
    assert config.qwen4_args.ple_layer_ids == ()
    assert config.slot_states == ()
    assert config.vision_config is None
    assert config.image_token_id is None
    assert config.expert_quant == "none"


def test_input_fusion_matches_independent_reference():
    config = derive_mtp_model_config(parsed_config(num_layers=4))
    model = _new_model(config)
    _fill(model, seed=11)
    tokens = 3
    hidden = torch.randn(tokens, config.qwen4_args.hc_count * config.hidden_size).to(
        torch.bfloat16
    )
    embeddings = torch.randn(tokens, config.hidden_size).to(torch.bfloat16)

    got = model.fuse_inputs(embeddings, hidden)

    expected_embeddings = F.linear(
        _norm_ref(
            embeddings,
            model.pre_fc_norm_embedding.weight,
            config.rms_norm_eps,
        ),
        model.fc_embedding.weight,
    )
    expected_hidden = _norm_ref(
        hidden,
        model.pre_fc_norm_hidden.weight,
        config.rms_norm_eps,
    ).view(tokens, config.qwen4_args.hc_count, config.hidden_size)
    expected_hidden = F.linear(expected_hidden, model.fc_hidden.weight)
    expected = (expected_hidden + expected_embeddings.unsqueeze(1)).flatten(-2)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


class _FakeLayer(BaseOP):
    def forward(self, hidden: torch.Tensor, _batch) -> torch.Tensor:
        return hidden + torch.tensor(0.125, dtype=hidden.dtype)


def test_forward_returns_sample_and_recursive_multi_streams():
    config = derive_mtp_model_config(parsed_config(num_layers=4))
    model = _new_model(config)
    _fill(model, seed=19)
    model.layers = OPList([_FakeLayer()])
    embeddings = torch.randn(2, config.hidden_size).to(torch.bfloat16)
    target_multi = torch.randn(2, config.qwen4_args.hc_count * config.hidden_size).to(
        torch.bfloat16
    )

    sample, recursive = model.forward(embeddings, target_multi, batch=object())

    assert sample.shape == (2, config.hidden_size)
    assert recursive.shape == (
        2,
        config.qwen4_args.hc_count * config.hidden_size,
    )
    torch.testing.assert_close(
        recursive,
        model.fuse_inputs(embeddings, target_multi) + torch.tensor(0.125, dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )
    sample_2, recursive_2 = model.forward(embeddings, recursive, batch=object())
    assert sample_2.shape == sample.shape
    assert recursive_2.shape == recursive.shape
    assert not torch.equal(recursive_2, recursive)


def _raw_checkpoint_from_model(model: Qwen4ExpMTPModel) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    raw: dict[str, torch.Tensor] = {}
    qkv = "layers.0.self_attn.qkv_proj.weight"
    q_rows = 2 * model.config.num_qo_heads * model.config.head_dim
    kv_rows = model.config.num_kv_heads * model.config.head_dim
    q, k, v = state[qkv].split((q_rows, kv_rows, kv_rows), dim=0)
    for name, value in zip(("q", "k", "v"), (q, k, v)):
        raw[f"mtp.layers.0.self_attn.{name}_proj.weight"] = value.clone()

    shared = "layers.0.mlp.shared_expert.gate_up_proj.weight"
    gate, up = state[shared].chunk(2, dim=0)
    raw["mtp.layers.0.mlp.shared_expert.gate_proj.weight"] = gate.clone()
    raw["mtp.layers.0.mlp.shared_expert.up_proj.weight"] = up.clone()

    for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
        fused = f"layers.0.{hc}.input_mix_weight_down_block_inject.weight"
        value = state[fused]
        lowrank = model.config.qwen4_args.hc_lowrank
        count = model.config.qwen4_args.hc_count
        raw[f"mtp.layers.0.{hc}.input_mix_weight_down.weight"] = value[:lowrank].clone()
        raw[f"mtp.layers.0.{hc}.block_inject_weight.weight"] = value[
            lowrank : lowrank + count
        ].clone()

    fused_keys = {
        qkv,
        shared,
        "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        "layers.0.mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
    }
    for name, value in state.items():
        if name not in fused_keys:
            raw[f"mtp.{name}"] = value.clone()
    return raw


def _write_indexed_checkpoint(folder: Path, raw: dict[str, torch.Tensor]) -> None:
    names = sorted(raw)
    shards = [names[::3], names[1::3], names[2::3]]
    weight_map: dict[str, str] = {}
    for index, shard_names in enumerate(shards, 1):
        shard = f"model-bf16-{index:05d}.safetensors"
        save_file({name: raw[name] for name in shard_names}, str(folder / shard))
        weight_map.update({name: shard for name in shard_names})
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )


def test_weight_plan_consumes_every_mtp_key_and_materializes_exact_state(tmp_path):
    config = derive_mtp_model_config(parsed_config(num_layers=4))
    model = _new_model(config)
    _fill(model, seed=23)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    raw = _raw_checkpoint_from_model(model)
    assert len(raw) == 31
    _write_indexed_checkpoint(tmp_path, raw)

    with MTPWeightStore(tmp_path) as store:
        plan = build_mtp_weight_plan(store.keys, expected.keys())
        assert len(plan.raw_names) == 31
        assert set(plan.model_names) == set(expected)
        assert plan.expert_model_names == {
            "layers.0.mlp.experts.gate_up_proj",
            "layers.0.mlp.experts.down_proj",
        }
        got = {
            entry.model_name: store.materialize(entry, device=torch.device("cpu"))
            for entry in plan.entries
        }

    assert set(got) == set(expected)
    for name in expected:
        torch.testing.assert_close(got[name], expected[name], rtol=0, atol=0)


def test_weight_plan_rejects_missing_extra_and_duplicate_sources(tmp_path):
    config = derive_mtp_model_config(parsed_config(num_layers=4))
    model = _new_model(config)
    _fill(model, seed=29)
    raw = _raw_checkpoint_from_model(model)
    _write_indexed_checkpoint(tmp_path, raw)
    expected = model.state_dict().keys()

    with MTPWeightStore(tmp_path) as store:
        names = list(store.keys)
        with pytest.raises(ValueError, match="missing MTP source"):
            build_mtp_weight_plan(names[:-1], expected)
        with pytest.raises(ValueError, match="unexpected MTP source"):
            build_mtp_weight_plan(names + ["mtp.unexpected.weight"], expected)
        with pytest.raises(ValueError, match="duplicate MTP source"):
            build_mtp_weight_plan(names + [names[0]], expected)


def test_model_state_has_no_ple_gdn_embedding_or_language_head():
    config = derive_mtp_model_config(parsed_config(num_layers=4))
    with torch.device("meta"):
        model = Qwen4ExpMTPModel(config)
    names = set(model.state_dict())
    assert names
    assert not any("ple" in name for name in names)
    assert not any("linear_attn" in name for name in names)
    assert not any(name.startswith("embed_tokens") for name in names)
    assert not any(name.startswith("lm_head") for name in names)


def test_private_module_is_not_exported_from_qwen_package():
    package = Path(__file__).parents[3] / "python" / "freetoken" / "models" / "qwen4_exp" / "__init__.py"
    assert "mtp_spike" not in package.read_text(encoding="utf-8")
