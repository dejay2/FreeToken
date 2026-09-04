"""glm5_next (GLM-5.3-Flash) config parsing: attention groups, kpool spec, aliases.

Runs off a trimmed copy of the real checkpoint config.json via RawConfigShim (the
exact object cached_load_hf_config falls back to when transformers doesn't know
``glm5_next`` yet), so the parse path under test is the one production hits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from freetoken.attention.base import AttnType
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
)
from freetoken.models.glm5_next.args import load_args
from freetoken.models.glm5_next.config import parse_config
from freetoken.utils.hf import RawConfigShim

_NUM_LAYERS = 45
_DSA_IDS = tuple(range(3, _NUM_LAYERS, 4))  # 3, 7, ..., 43
_KDA_IDS = tuple(i for i in range(_NUM_LAYERS) if i not in _DSA_IDS)


def _layer_types() -> list[str]:
    return [
        "deepseek_sparse_attention" if i in _DSA_IDS else "linear_attention"
        for i in range(_NUM_LAYERS)
    ]


def _text_config() -> dict:
    # Trimmed from zai-org/GLM-5.3-Flash config.json (text_config).
    return {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": _NUM_LAYERS,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "vocab_size": 154880,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 1048576,
        "tie_word_embeddings": False,
        # MLA (NoPE)
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "mla_use_nope": True,
        # DSA indexer + kpool
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "indexer_types": ["full"] * _NUM_LAYERS,
        "indexer_rope_interleave": True,
        "index_kpool": 4,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        # KDA
        "linear_attn_config": {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": list(_KDA_IDS),
            "full_attn_layers": list(_DSA_IDS),
        },
        # layout
        "layer_types": _layer_types(),
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (_NUM_LAYERS - 3),
        "first_k_dense_replace": 3,
        # mHC (checkpoint spellings)
        "mhc": True,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        # MoE
        "n_routed_experts": 288,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "n_group": 1,
        "topk_group": 1,
        "swiglu_limit": 10.0,
        "num_nextn_predict_layers": 1,
        "attention_bias": False,
        "model_type": "glm5_next_text",
    }


def _hf_config(quantization_config: dict | None = None) -> RawConfigShim:
    data: dict = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": _text_config(),
        "vision_config": {"model_type": "glm5_next_vision", "depth": 24},
        "image_token_id": 154854,
    }
    if quantization_config is not None:
        data["quantization_config"] = quantization_config
    return RawConfigShim(data)


_CT_NVFP4_QUANT = {
    # From RedHatAI/GLM-5.3-Flash-NVFP4 (llm-compressor, experts-only calibrated NVFP4).
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*mlp\\.experts\\..*(gate|up|down)_proj$"],
            "weights": {"num_bits": 4, "type": "float", "group_size": 16},
        }
    },
}

_CT_MIXED_QUANT = {
    # From RedHatAI/GLM-5.3-Flash-NVFP4 as published: nvfp4 routed experts plus fp8 experts on the MTP layer, so the top-level format is "mixed-precision".
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*\\.layers\\.(?:[3-9]|[1-3][0-9]|4[0-4])\\.mlp\\.experts\\..*(gate|up|down)_proj$"],
            "weights": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"},
            "format": "nvfp4-pack-quantized",
        },
        "group_1": {
            "targets": ["re:.*\\.layers\\.45\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"],
            "weights": {"num_bits": 8, "type": "float", "strategy": "block"},
            "format": "float-quantized",
        },
    },
}

_NVFP4_QUANT = {
    # From LibertAIDAI/GLM-5.3-Flash-NVFP4 (ModelOpt weight-only NVFP4).
    "quant_algo": "NVFP4",
    "quant_method": "modelopt",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "float", "group_size": 16},
        }
    },
    "ignore": ["lm_head", "model.visual.*"],
}


def test_attention_groups():
    cfg = parse_config(_hf_config())

    assert cfg.num_layers == _NUM_LAYERS
    assert len(cfg.attention_groups) == 2
    linear, full = cfg.attention_groups  # ordered by first layer id (0 < 3)

    assert isinstance(linear, LinearGatedDeltaGroupConfig)
    assert linear.variant == "kda"
    assert linear.layer_ids == _KDA_IDS
    assert (linear.num_key_heads, linear.key_head_dim) == (64, 128)
    assert (linear.num_value_heads, linear.value_head_dim) == (64, 128)
    assert linear.conv_kernel_dim == 4

    assert isinstance(full, FullAttentionGroupConfig)
    assert full.layer_ids == _DSA_IDS
    assert full.mla is True
    assert full.head_dim == 512  # bare ckv latent: kv_lora_rank + 0 rope dims
    assert full.num_kv_heads == 1
    assert full.index_head_dim == 128
    assert full.num_index_layers == len(_DSA_IDS)
    assert full.index_ratio == 4

    assert cfg.has_linear_attention and cfg.has_hybrid_attention
    assert cfg.attn_type_for_layer(0) == AttnType.LINEAR
    assert cfg.attn_type_for_layer(3) == AttnType.DSA


def test_kv_cache_group_specs_skip_linear_and_carry_kpool():
    cfg = parse_config(_hf_config())
    specs = [s for s in cfg.kv_cache_group_specs() if s.num_layers > 0]
    # The linear group keeps recurrent state (no paged KV); exactly one paged spec.
    assert len(specs) == 1
    (spec,) = specs
    assert spec.attn_type == AttnType.DSA
    assert spec.layer_ids == _DSA_IDS
    assert (spec.mla, spec.head_dim, spec.index_head_dim) == (True, 512, 128)
    assert spec.index_ratio == 4
    assert spec.num_index_layers == len(_DSA_IDS)


def test_moe_and_scalars():
    cfg = parse_config(_hf_config(_NVFP4_QUANT))
    assert cfg.expert_quant == "nvfp4"
    # Only DERIVED facts are pinned here; fixture echoes (num_experts == 288
    # and friends) assert nothing the parse could get wrong.
    assert cfg.attn_sm_scale == pytest.approx(256**-0.5)
    assert cfg.num_moe_layers == _NUM_LAYERS - 3
    assert cfg.is_moe
    # Checkpoint-faithful default; the FREETOKEN_GLM5_*_FP8 env flags opt into
    # the W8A16 fp8 load.
    assert (cfg.attn_quant, cfg.dense_quant, cfg.lm_head_quant) == ("none",) * 3
    # Text-only serving: the vision tower is never built.
    assert cfg.vision_config is None


def test_args_alias_folding_and_nope():
    args = load_args(_hf_config())
    # Checkpoint spellings fold into the canonical fields.
    assert args.mla_nope is True  # from mla_use_nope
    assert args.mhc_num_residual_streams == 4  # from hc_mult
    assert args.mhc_sinkhorn_iterations == 20  # from hc_sinkhorn_iters
    assert args.linear_num_heads == 64  # from nested linear_attn_config
    assert args.linear_lower_bound == -5.0
    # NoPE geometry.
    assert args.qk_rope_head_dim == 0
    assert args.qk_head_dim == 256
    assert args.latent_dim == 512
    assert args.kda_layer_ids == _KDA_IDS
    assert args.dsa_layer_ids == _DSA_IDS
    # Defaults for fields the checkpoint doesn't ship.
    assert args.rope_theta == 10000.0
    assert args.mhc_tau == 0.05
    assert args.mhc_post_mult_value == 2.0


def test_registry_resolves_glm5_next():
    from freetoken.models.register import get_model_spec

    spec = get_model_spec("Glm5NextForConditionalGeneration")
    assert spec.module == "freetoken.models.glm5_next"
    assert spec.model_cls == "Glm5NextForCausalLM"
    assert get_model_spec("Glm5NextForCausalLM").module == spec.module


def test_dev_layer_cap(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GLM5_MAX_LAYERS", "5")
    cfg = parse_config(_hf_config())
    assert cfg.num_layers == 5
    linear, full = cfg.attention_groups
    assert linear.layer_ids == (0, 1, 2, 4)
    assert full.layer_ids == (3,)
    assert full.num_index_layers == 1
    # 3 dense + 2 sparse under the cap.
    assert cfg.first_k_dense_replace == 3


def test_rejects_unknown_layer_types():
    data = _hf_config().to_dict()
    data["text_config"]["layer_types"][0] = "full_attention"
    with pytest.raises(ValueError, match="unsupported layer_types"):
        load_args(RawConfigShim(data))


def test_compressed_tensors_nvfp4_detected():
    """RedHatAI/GLM-5.3-Flash-NVFP4 (llm-compressor): expert_quant resolves to
    nvfp4 off the ``format`` field (quant_algo is absent for compressed-tensors)."""
    cfg = parse_config(_hf_config(_CT_NVFP4_QUANT))
    assert cfg.expert_quant == "nvfp4"


def test_compressed_tensors_mixed_precision_detected():
    """The published RedHatAI export says ``format: mixed-precision`` at the top and
    ``nvfp4-pack-quantized`` only inside the routed-expert group; expert_quant must
    still resolve to nvfp4, not to the raw quant_method."""
    cfg = parse_config(_hf_config(_CT_MIXED_QUANT))
    assert cfg.expert_quant == "nvfp4"


def test_compressed_tensors_mixed_precision_reads_the_expert_group():
    """A mixed export with nvfp4 dense layers but fp8 experts must not report nvfp4
    experts: the group that targets the experts decides."""
    quant = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "targets": ["re:.*self_attn.*_proj$"],
                "weights": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"},
                "format": "nvfp4-pack-quantized",
            },
            "group_1": {
                "targets": ["re:.*mlp\\.experts\\..*(gate|up|down)_proj$"],
                "weights": {"num_bits": 8, "type": "float", "strategy": "block"},
                "format": "float-quantized",
            },
        },
    }
    cfg = parse_config(_hf_config(quant))
    assert cfg.expert_quant == "compressed-tensors"


def test_expert_source_spec_selection():
    """quant_method picks the bank source spec: compressed-tensors maps
    weight_packed/weight_global_scale onto the canonical kinds with a reciprocal
    global; modelopt stays identity."""
    from freetoken.models.glm5_next.weight import (
        _NVFP4_CT_SOURCE_SPEC,
        _NVFP4_SOURCE_SPEC,
    )

    ct = _NVFP4_CT_SOURCE_SPEC
    m = ct.key_pattern.match(
        "model.language_model.layers.5.mlp.experts.7.gate_proj.weight_packed"
    )
    assert m and m.group("kind") == "weight_packed"
    assert ct.kind_map["weight_packed"] == "weight"
    assert ct.kind_map["weight_global_scale"] == "weight_scale_2"
    assert ct.global_reciprocal
    # W4A16 serving never consumes the calibrated activation scale.
    assert ct.key_pattern.match(
        "model.language_model.layers.5.mlp.experts.7.gate_proj.input_global_scale"
    ) is None
    assert _NVFP4_SOURCE_SPEC.kind_map is None
    assert not _NVFP4_SOURCE_SPEC.global_reciprocal


def test_ingest_global_reciprocal():
    import torch
    from freetoken.models.glm5_next.weight import _NVFP4_CT_SOURCE_SPEC, _NVFP4_SOURCE_SPEC
    from freetoken.models.nvfp4_banks import _ingest_global

    g = torch.tensor(4.0)
    assert _ingest_global(_NVFP4_CT_SOURCE_SPEC, g).item() == 0.25
    assert _ingest_global(_NVFP4_SOURCE_SPEC, g).item() == 4.0


def test_real_trimmed_exl3_metadata_detects_format_and_geometry():
    fixture_path = Path(__file__).parent / "fixtures" / "glm53_exl3_quantization_trimmed.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    cfg = parse_config(_hf_config(fixture))

    assert cfg.expert_quant == "exl3"
    assert cfg.num_layers == 45
    assert cfg.first_k_dense_replace == 3
    assert cfg.num_moe_layers == 42
    assert cfg.num_experts == 288
    assert cfg.num_experts_per_tok == 8
    assert fixture["codebook"] == "mul1"
    assert fixture["bits"] == pytest.approx(2.05)
    assert fixture["head_bits"] == 5
    assert fixture["mtp_bits"] == 2
    assert fixture["vision_bits"] == 5


def test_trimmed_exl3_records_lock_the_selected_source_headers():
    # Records were copied byte-for-byte from turboderp/GLM-5.3-Flash-exl3, revision 2.05bpw.
    fixture_path = Path(__file__).parent / "fixtures" / "glm53_exl3_quantization_trimmed.json"
    storage = json.loads(fixture_path.read_text(encoding="utf-8"))["tensor_storage"]
    assert len(storage) == 9
    assert sum(len(record["stored_tensors"]) for record in storage.values()) == 37
    assert all(record["quant_format"] == "exl3" for record in storage.values())
    assert {record["bits_per_weight"] for record in storage.values()} == {2, 3, 4, 5}

    expected = {
        "model.language_model.layers.3.mlp.experts.0.gate_proj": (2, [256, 128, 32]),
        "model.language_model.layers.3.mlp.experts.0.up_proj": (2, [256, 128, 32]),
        "model.language_model.layers.3.mlp.experts.0.down_proj": (2, [128, 256, 32]),
        "model.language_model.layers.0.mlp.down_proj": (3, [768, 256, 48]),
        "model.language_model.layers.0.self_attn.qkv_proj": (4, [256, 1536, 64]),
        "model.language_model.layers.3.mlp.shared_experts.gate_proj": (4, [256, 128, 64]),
        "lm_head": (5, [256, 9680, 80]),
    }
    for base, (bits, trellis_shape) in expected.items():
        record = storage[base]
        assert record["bits_per_weight"] == bits
        assert record["mul1_multiplier"] == 2212286765
        trellis = record["stored_tensors"][f"{base}.trellis"]
        assert trellis["dtype"] == "torch.int16"
        assert trellis["shape"] == trellis_shape
        assert record["stored_tensors"][f"{base}.mul1"]["shape"] == []

    # The retained trailing MTP and vision samples are metadata-only proof that the
    # non-language tensors stay outside the text loader's 0..44 loop.
    assert "model.language_model.layers.45.eh_proj" in storage
    assert "model.visual.blocks.0.attn.q_proj" in storage
