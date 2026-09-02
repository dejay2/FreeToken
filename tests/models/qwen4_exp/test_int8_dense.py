"""FREETOKEN_DENSE_QUANT=int8: the flag, what it converts, and what it must leave alone.

The Qwen3.8-Flash-Next NVFP4 checkpoint quantizes only the routed experts -- its modelopt
ignore list excludes ``*.self_attn.*``, ``*.linear_attn.*``, ``*hyper_connection*``,
``*.mlp.shared_expert.*``, ``*.mlp.gate*`` and ``lm_head`` -- so every dense projection on the
decode path arrives bf16. This flag converts them at load to weight-only int8; unset, the
model is the bf16 one it has always been, class for class.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.kernel.triton.int8_linear import (
    Int8DenseColMerged,
    Int8DenseLinear,
    Int8DenseRowParallel,
    Int8LMHead,
)
from freetoken.layers import LinearColParallelMerged, LinearReplicated, LinearRowParallel
from freetoken.layers.embedding import ParallelLMHead
from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention
from freetoken.models.qwen4_exp.config import parse_config, resolve_dense_quant
from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
from freetoken.models.qwen4_exp.hc import GatedResidual
from freetoken.models.qwen3_5_moe.moe import _SharedExpert

from .common import requires_cuda, toy_hf_config
from .test_config import _hf_config

ENV = "FREETOKEN_DENSE_QUANT"


# ======================================================================================
# The flag
# ======================================================================================
def test_the_flag_defaults_to_bf16(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert resolve_dense_quant() == "none"


@pytest.mark.parametrize("value,expected", [("int8", "int8"), ("  INT8 ", "int8"),
                                            ("none", "none"), ("", "none")])
def test_the_flag_is_parsed_case_and_space_insensitively(value: str, expected: str):
    assert resolve_dense_quant({ENV: value}) == expected


def test_an_unknown_mode_fails_loudly():
    with pytest.raises(ValueError, match=ENV):
        resolve_dense_quant({ENV: "fp6"})


# ======================================================================================
# Config resolution
# ======================================================================================
def test_unset_leaves_the_checkpoint_quant_untouched(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    config = parse_config(_hf_config())
    # The shipping checkpoint: experts NVFP4, everything dense bf16.
    assert config.expert_quant == "nvfp4"
    assert (config.attn_quant, config.dense_quant, config.lm_head_quant) == ("none",) * 3


def test_int8_converts_exactly_the_components_the_checkpoint_left_bf16(monkeypatch):
    monkeypatch.setenv(ENV, "int8")
    config = parse_config(_hf_config())
    assert config.expert_quant == "nvfp4"  # the routed experts keep their own (cheaper) format
    assert (config.attn_quant, config.dense_quant, config.lm_head_quant) == ("int8",) * 3


def test_int8_never_downgrades_a_component_the_checkpoint_already_packed(monkeypatch):
    """An ignore list that does not exclude attention leaves those weights NVFP4 on disk;
    converting them to int8 would mean dequantizing 4-bit data to make it bigger."""
    monkeypatch.setenv(ENV, "int8")
    hf = _hf_config()
    hf.quantization_config = {"quant_algo": "NVFP4", "quant_method": "modelopt", "ignore": []}
    config = parse_config(hf)
    assert (config.attn_quant, config.dense_quant, config.lm_head_quant) == ("nvfp4",) * 3


def test_a_tied_lm_head_is_left_bf16(monkeypatch):
    """A tied head IS the token embedding; quantizing it would corrupt the embedding or
    silently duplicate it."""
    monkeypatch.setenv(ENV, "int8")
    hf = _hf_config()
    hf.text_config.tie_word_embeddings = True
    config = parse_config(hf)
    assert config.tie_word_embeddings and config.lm_head_quant == "none"
    assert config.attn_quant == "int8"  # the rest still converts


def test_a_checkpoint_with_no_quant_config_still_honors_the_flag(monkeypatch):
    monkeypatch.setenv(ENV, "int8")
    config = parse_config(toy_hf_config())
    assert (config.attn_quant, config.dense_quant, config.lm_head_quant) == ("int8",) * 3
    assert config.expert_quant == "none"  # nothing to convert: this flag is dense-only


# ======================================================================================
# Which class each module gets
# ======================================================================================
def _modules(monkeypatch, mode: str | None):
    if mode is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, mode)
    config = parse_config(toy_hf_config())
    args = config.qwen4_args
    gdn = Qwen4ExpGatedDeltaNet(
        hidden_size=args.hidden_size, num_k_heads=2, num_v_heads=2, head_k_dim=32,
        head_v_dim=32, conv_kernel_size=4, rms_norm_eps=1e-6, layer_id=0,
        expert_quant=config.expert_quant, attn_quant=config.attn_quant,
        dense_quant=config.dense_quant,
    )
    attn = Qwen4ExpAttention(config, layer_id=3)
    hc = GatedResidual(config, use_combine=True)
    mixer = GatedResidual(config, use_combine=False)
    shared = _SharedExpert(config, args.hidden_size, 64)
    return {
        "gdn.in_proj": gdn.in_proj,
        "gdn.out_proj": gdn.out_proj,
        "attn.qkv_proj": attn.qkv_proj,
        "attn.o_proj": attn.o_proj,
        "attn.indexer.index_qk_proj": attn.indexer.index_qk_proj,
        "hc.down_block_inject": hc.input_mix_weight_down_block_inject,
        "hc.up": hc.input_mix_weight_up,
        "mixer.down": mixer.input_mix_weight_down,
        "shared.gate_up_proj": shared.gate_up_proj,
        "shared.down_proj": shared.down_proj,
    }


def test_unset_leaves_every_dense_class_the_bf16_one(monkeypatch):
    expected = {
        "gdn.in_proj": LinearColParallelMerged,
        "gdn.out_proj": LinearReplicated,
        "attn.qkv_proj": LinearColParallelMerged,
        "attn.o_proj": LinearReplicated,
        "attn.indexer.index_qk_proj": LinearReplicated,
        "hc.down_block_inject": LinearReplicated,
        "hc.up": LinearReplicated,
        "mixer.down": LinearReplicated,
        "shared.gate_up_proj": LinearColParallelMerged,
        "shared.down_proj": LinearRowParallel,
    }
    for name, module in _modules(monkeypatch, None).items():
        assert type(module) is expected[name], name


def test_int8_replaces_every_dense_class_on_the_decode_path(monkeypatch):
    expected = {
        "gdn.in_proj": Int8DenseColMerged,
        "gdn.out_proj": Int8DenseLinear,
        "attn.qkv_proj": Int8DenseColMerged,
        "attn.o_proj": Int8DenseLinear,
        "attn.indexer.index_qk_proj": Int8DenseLinear,
        "hc.down_block_inject": Int8DenseLinear,
        "hc.up": Int8DenseLinear,
        "mixer.down": Int8DenseLinear,
        "shared.gate_up_proj": Int8DenseColMerged,
        "shared.down_proj": Int8DenseRowParallel,
    }
    for name, module in _modules(monkeypatch, "int8").items():
        assert type(module) is expected[name], name


def test_gdn_in_proj_uses_dense_quant_with_packed_attention(monkeypatch):
    """Packed attention storage must not hide the separate int8 dense conversion."""
    from dataclasses import replace

    from freetoken.models.qwen4_exp.model import build_linear_mixer

    monkeypatch.delenv(ENV, raising=False)
    config = replace(
        parse_config(toy_hf_config()), attn_quant="mxfp8", dense_quant="int8"
    )
    gdn = build_linear_mixer(config, layer_id=0)
    assert type(gdn.in_proj) is Int8DenseColMerged


def test_the_router_and_the_shared_gate_stay_bf16(monkeypatch):
    """A [num_experts, hidden] router is 1/500th of a layer's dense bytes and the one GEMM
    whose output decides which experts run: it is not worth a code of error."""
    monkeypatch.setenv(ENV, "int8")
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE

    config = parse_config(toy_hf_config())
    moe = Qwen4ExpMoE(config, layer_id=0)
    assert type(moe.gate) is LinearReplicated
    assert type(moe.shared_expert_gate) is LinearReplicated


def _build_model(monkeypatch, mode: str | None):
    """The whole model, built the way the engine builds it (meta device, model dtype)."""
    from freetoken.layers.rotary import set_rope_device
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    from freetoken.utils import torch_dtype

    if mode is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, mode)
    set_rope_device(torch.device("cpu"))  # rope tables refuse the meta device
    config = parse_config(toy_hf_config())
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return Qwen4ExpForCausalLM(config)


@pytest.mark.parametrize("mode,expected", [(None, ParallelLMHead), ("int8", Int8LMHead)])
def test_the_lm_head_class_follows_lm_head_quant(monkeypatch, mode, expected):
    assert type(_build_model(monkeypatch, mode).lm_head) is expected


def _count_linears(op, seen=None, counts=None):
    """Walk the assembled model and tally every dense-linear class it holds."""
    from freetoken.layers import BaseOP

    seen = set() if seen is None else seen
    counts = {} if counts is None else counts
    if id(op) in seen:
        return counts
    seen.add(id(op))
    children = list(getattr(op, "op_list", [])) + [
        v for k, v in vars(op).items() if not k.startswith("_") and isinstance(v, BaseOP)
    ]
    if isinstance(op, (LinearReplicated, LinearColParallelMerged, LinearRowParallel,
                       Int8DenseLinear, Int8DenseColMerged, Int8DenseRowParallel)):
        counts[type(op).__name__] = counts.get(type(op).__name__, 0) + 1
    for child in children:
        _count_linears(child, seen, counts)
    return counts


def test_the_assembled_model_holds_no_bf16_dense_linear_beyond_the_routers(monkeypatch):
    """The only bf16 ``LinearReplicated`` left under int8 is a router: one MoE ``gate`` and
    one ``shared_expert_gate`` per MoE layer. Everything else converted."""
    bf16 = _count_linears(_build_model(monkeypatch, None))
    int8 = _count_linears(_build_model(monkeypatch, "int8"))

    assert not any(name.startswith("Int8") for name in bf16), bf16
    routers = int8.get("LinearReplicated", 0)
    assert routers > 0 and routers % 2 == 0  # gate + shared_expert_gate, per MoE layer
    assert int8.get("LinearColParallelMerged", 0) == 0
    assert int8.get("LinearRowParallel", 0) == 0
    # Every bf16 linear that was not a router became an int8 one, one for one.
    converted = sum(v for k, v in int8.items() if k.startswith("Int8"))
    assert converted == sum(bf16.values()) - routers


# ======================================================================================
# Numerics: each converted module against the bf16 twin it replaces
# ======================================================================================
@requires_cuda
@pytest.mark.parametrize("m", [1, 6, 128])
def test_every_converted_class_matches_its_bf16_twin(monkeypatch, m: int):
    import torch.nn.functional as F

    monkeypatch.setenv(ENV, "int8")
    torch.manual_seed(0)
    cases = [
        (Int8DenseLinear(256, 128), LinearReplicated(256, 128, has_bias=False)),
        (Int8DenseColMerged(256, [64, 64]), LinearColParallelMerged(256, [64, 64], has_bias=False)),
        (Int8DenseRowParallel(256, 128), LinearRowParallel(256, 128, has_bias=False)),
    ]
    x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    for quant, bf16 in cases:
        w = torch.randn(quant.out_features, 256, device="cuda", dtype=torch.bfloat16) * 0.05
        quant.load_state_dict({"weight": w.clone()})
        bf16.weight = w
        got, want = quant.forward(x), bf16.forward(x)
        rel = ((got.float() - want.float()).abs().max()
               / want.float().abs().max().clamp(min=1e-6)).item()
        assert rel < 3e-2, (type(quant).__name__, m, rel)


@requires_cuda
def test_the_int8_lm_head_keeps_forward_alls_signature():
    """``SpecDraftHead.propose`` and the MTP teacher seam call ``forward_all(x) -> [rows,
    vocab]``; the draft's own copy is skipped entirely once the target head is already int8."""
    from freetoken.engine.spec_lmhead import build_draft_lm_head

    head = Int8LMHead(num_embeddings=512, embedding_dim=128)
    head.load_state_dict(
        {"weight": torch.randn(512, 128, device="cuda", dtype=torch.bfloat16) * 0.05}
    )
    out = head.forward_all(torch.randn(3, 128, device="cuda", dtype=torch.bfloat16))
    assert out.shape == (3, 512)
    for placement in ("bf16", "int8", "nvfp4"):
        assert build_draft_lm_head(head, placement=placement) is head


def test_a_synthetic_config_object_without_dense_quant_is_tolerated():
    """``getattr(config, "dense_quant", "none")`` -- the direct-op tests build bare configs."""
    from freetoken.models.quant_linear import make_dense_col_merged, make_dense_replicated

    bare = SimpleNamespace()
    assert type(make_dense_replicated(getattr(bare, "dense_quant", "none"), 8, 4)) is (
        LinearReplicated
    )
    assert type(make_dense_col_merged(getattr(bare, "dense_quant", "none"), 8, [2, 2])) is (
        LinearColParallelMerged
    )
