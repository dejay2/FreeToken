"""EXL3 Qwen Flash dense wiring and loader (spec 2026-09-25 section 2).

The model is built on ``meta`` exactly as the engine builds it, and the fixture checkpoint is
written back from that model's own state dict under the CHECKPOINT's names (the reverse of the
loader's rename and fusions), so "the loader emits exactly the model's keys" is checked against
the model rather than against a hand-kept list. K follows the real 3.05bpw_h5_ng5 build: the
QSA indexer is K=3, every other dense EXL3 linear K=5, so the load has to adopt mixed K.

The toy geometry is NOT common.toy_hf_config: an EXL3 linear needs out % 128 == 0, which its
kv width (64), indexer (192), GDN conv (320) and shared expert (64) all miss. The widths here
are the smallest that pass while keeping the real layer split (GDN, GDN+PLE, GDN, QSA).

The NVFP4 key set and fusions are already pinned by test_weight.py
(``test_key_map_is_exactly_the_model_state_dict`` and the fusion round-trip tests), so this
file does not re-pin them (ruling R5).
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.exl3_linear import (
    Exl3ColMerged,
    Exl3LMHead,
    Exl3Linear,
    iter_exl3_linears,
    prepare_exl3_dense_workspace,
)
from freetoken.models.qwen4_exp import weight as W

from .common import hf_config

_EXL3_QUANT = {"quant_method": "exl3", "version": "1.4.4", "bits": 3.05, "head_bits": 5,
               "codebook": "mul1", "out_scales": "always", "vision_bits": 5, "mtp_bits": 3}
_COMPONENTS = (".trellis", ".suh", ".svh", ".mul1")
_ROUTED = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _exl3_hf_config():
    hf = hf_config(
        num_layers=4, head_dim=128, num_q=2, num_kv=1, index_head_dim=128, index_heads=4,
        budget=16, hidden=128, max_position=4096, rope_theta=10000.0,
        linear_key_head_dim=64, linear_value_head_dim=64, linear_num_value_heads=4,
        shared_expert_intermediate_size=128,
    )
    hf.quantization_config = dict(_EXL3_QUANT)
    return hf


def _build_model(hf):
    from freetoken.layers import rotary
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    from freetoken.utils.torch_utils import torch_dtype

    config = parse_config(hf)
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            return Qwen4ExpForCausalLM(config)
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


@pytest.fixture(scope="module")
def exl3_model():
    # Routed experts: expert_quant="exl3" builds them through Qwen3_5MoE's bf16 make_moe_layer
    # today (Task 5 owns the EXL3 expert banks); on meta that allocates nothing, and the
    # loader never emits them (they come from the offload source banks), so they are only
    # excluded from the key comparison below.
    return _build_model(_exl3_hf_config())


# Model (nested) name -> checkpoint name; kept independent of the loader's own table.
_NEST = {
    ".self_attn.qkv_proj.q_proj.": ".self_attn.q_proj.",
    ".self_attn.qkv_proj.k_proj.": ".self_attn.k_proj.",
    ".self_attn.qkv_proj.v_proj.": ".self_attn.v_proj.",
    ".linear_attn.in_proj_qkvz.in_proj_qkv.": ".linear_attn.in_proj_qkv.",
    ".linear_attn.in_proj_qkvz.in_proj_z.": ".linear_attn.in_proj_z.",
    ".mlp.shared_expert.gate_up_proj.gate_proj.": ".mlp.shared_expert.gate_proj.",
    ".mlp.shared_expert.gate_up_proj.up_proj.": ".mlp.shared_expert.up_proj.",
}


def _checkpoint_name(key: str) -> str:
    for nested, part in _NEST.items():
        if nested in key:
            key = key.replace(nested, part, 1)
            break
    if key.startswith("model."):
        return "model.language_model." + key[len("model."):]
    return key


def _k_for(key: str) -> int:
    return 3 if ".indexer.index_qk_proj." in key else 5


def _checkpoint_tensors(model) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    args = model._config.qwen4_args
    out: dict[str, torch.Tensor] = {}
    for key, param in model.state_dict().items():
        if key.endswith(_ROUTED):
            continue
        shape = tuple(param.shape)
        if key.endswith(".trellis"):
            k = _k_for(key)
            out[_checkpoint_name(key)] = torch.randint(
                -32768, 32767, (*shape[:2], 16 * k), dtype=torch.int16, generator=gen)
        elif key.endswith((".suh", ".svh")):
            out[_checkpoint_name(key)] = torch.randn(shape, generator=gen).to(torch.float16)
        elif key.endswith(".mul1"):
            out[_checkpoint_name(key)] = torch.tensor(0x83DCD12D, dtype=torch.int64).to(torch.int32)
        elif key.endswith(".linear_attn.in_proj_ba.weight"):
            # fp16 in the real checkpoint, one [num_v_heads, hidden] each
            b, a = torch.randn(shape, generator=gen).to(torch.float16).chunk(2, dim=0)
            base = _checkpoint_name(key)[: -len("in_proj_ba.weight")]
            out[base + "in_proj_b.weight"] = b.contiguous()
            out[base + "in_proj_a.weight"] = a.contiguous()
        elif key.endswith(".input_mix_weight_down_block_inject.weight"):
            base = _checkpoint_name(key)[: -len("input_mix_weight_down_block_inject.weight")]
            out[base + "input_mix_weight_down.weight"] = torch.randn(
                args.hc_lowrank, shape[1], generator=gen).to(torch.bfloat16)
            out[base + "block_inject_weight.weight"] = torch.randn(
                args.hc_count, shape[1], generator=gen).to(torch.bfloat16)
        elif param.dtype.is_floating_point:
            out[_checkpoint_name(key)] = torch.randn(shape, generator=gen).to(torch.bfloat16)
        else:
            out[_checkpoint_name(key)] = torch.zeros(shape, dtype=param.dtype)
    # Noise the dense pass must skip: routed EXL3 experts (offload source banks) and MTP.
    lm = "model.language_model"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"{lm}.layers.0.mlp.experts.0.{proj}"
        out[f"{base}.trellis"] = torch.zeros(8, 8, 48, dtype=torch.int16)
        out[f"{base}.suh"] = torch.zeros(128, dtype=torch.float16)
        out[f"{base}.svh"] = torch.zeros(128, dtype=torch.float16)
        out[f"{base}.mul1"] = torch.zeros((), dtype=torch.int32)
    out["mtp.layers.0.self_attn.q_proj.trellis"] = torch.zeros(8, 16, 80, dtype=torch.int16)
    return out


@pytest.fixture(scope="module")
def exl3_checkpoint(tmp_path_factory, exl3_model) -> str:
    folder = tmp_path_factory.mktemp("qwen4_exl3_ckpt")
    raw = _checkpoint_tensors(exl3_model)
    names = sorted(raw)
    # Two shards, interleaved, so every fusion (HC, GDN b|a) has to survive a file boundary.
    shards = {"model-00001-of-00002.safetensors": names[::2],
              "model-00002-of-00002.safetensors": names[1::2]}
    weight_map = {}
    for file, keys in shards.items():
        save_file({n: raw[n] for n in keys}, str(folder / file))
        weight_map.update({n: file for n in keys})
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    (folder / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "quantization_config": _EXL3_QUANT}),
        encoding="utf-8")
    return str(folder)


def _loaded(folder: str) -> dict[str, torch.Tensor]:
    return {name: t for name, t in W.iter_weights(
        folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True,
        include_vision=False)}


# --------------------------------------------------------------------------------------


def test_exl3_rename_nests_fused_parts():
    assert W._exl3_rename("model.layers.3.self_attn.q_proj.trellis") == \
        "model.layers.3.self_attn.qkv_proj.q_proj.trellis"
    assert W._exl3_rename("model.layers.0.linear_attn.in_proj_z.svh") == \
        "model.layers.0.linear_attn.in_proj_qkvz.in_proj_z.svh"
    assert W._exl3_rename("model.layers.0.mlp.shared_expert.up_proj.mul1") == \
        "model.layers.0.mlp.shared_expert.gate_up_proj.up_proj.mul1"
    # not an EXL3 component or not a fused part: unchanged
    assert W._exl3_rename("model.layers.0.linear_attn.in_proj_b.weight") == \
        "model.layers.0.linear_attn.in_proj_b.weight"
    assert W._exl3_rename("model.layers.3.self_attn.o_proj.trellis") == \
        "model.layers.3.self_attn.o_proj.trellis"


def test_exl3_modules_are_built(exl3_model):
    layer_attn = exl3_model.model.layers.op_list[3].self_attn
    assert isinstance(layer_attn.qkv_proj, Exl3ColMerged)
    assert isinstance(layer_attn.o_proj, Exl3Linear)
    assert isinstance(layer_attn.indexer.index_qk_proj, Exl3Linear)
    assert layer_attn.indexer.index_qk_proj.k == 3
    gdn = exl3_model.model.layers.op_list[0].linear_attn
    assert isinstance(gdn.in_proj_qkvz, Exl3ColMerged) and isinstance(gdn.out_proj, Exl3Linear)
    assert not hasattr(gdn, "in_proj")
    assert type(gdn.in_proj_ba).__name__ == "LinearColParallelMerged"
    assert isinstance(exl3_model.model.layers.op_list[0].mlp.shared_expert.gate_up_proj, Exl3ColMerged)
    assert isinstance(exl3_model.model.layers.op_list[0].mlp.shared_expert.down_proj, Exl3Linear)
    assert isinstance(exl3_model.lm_head, Exl3LMHead)
    # HC, router, shared_expert_gate and the PLE projections stay plain linears.
    layer1 = exl3_model.model.layers.op_list[1]
    assert not isinstance(layer1.ple.key_proj, Exl3Linear)
    assert not isinstance(layer1.mlp.gate, Exl3Linear)


def test_is_exl3_checkpoint(exl3_checkpoint, tmp_path):
    assert W.is_exl3_checkpoint(exl3_checkpoint)
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}))
    assert not W.is_exl3_checkpoint(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    assert not W.is_exl3_checkpoint(str(tmp_path))
    bare = tmp_path / "no_config"
    bare.mkdir()
    assert not W.is_exl3_checkpoint(str(bare))


def test_iter_weights_matches_model_state(exl3_checkpoint, exl3_model):
    got = set(_loaded(exl3_checkpoint))
    # Routed experts come from the offload source banks, never from the dense pass.
    want = {k for k in exl3_model.state_dict() if not k.endswith(_ROUTED)}
    assert got == want


def test_fusions_and_packed_parts_round_trip(exl3_checkpoint):
    loaded = _loaded(exl3_checkpoint)
    from safetensors import safe_open

    raw = {}
    for file in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        with safe_open(f"{exl3_checkpoint}/{file}", framework="pt") as fh:
            raw.update({n: fh.get_tensor(n) for n in fh.keys()})
    lm = "model.language_model"
    # GDN b|a: fused as one bf16-path buffer, b first
    ba = loaded["model.layers.0.linear_attn.in_proj_ba.weight"]
    b = raw[f"{lm}.layers.0.linear_attn.in_proj_b.weight"]
    a = raw[f"{lm}.layers.0.linear_attn.in_proj_a.weight"]
    assert torch.equal(ba, torch.cat([b, a]))
    # packed parts are passed through untouched, under their nested names
    assert torch.equal(loaded["model.layers.3.self_attn.qkv_proj.k_proj.trellis"],
                       raw[f"{lm}.layers.3.self_attn.k_proj.trellis"])
    assert torch.equal(loaded["model.layers.0.linear_attn.in_proj_qkvz.in_proj_z.suh"],
                       raw[f"{lm}.layers.0.linear_attn.in_proj_z.suh"])
    assert torch.equal(loaded["lm_head.svh"], raw["lm_head.svh"])
    # the HC merge still pads to 16 rows
    hc = loaded["model.layers.2.attn_hyper_connection.input_mix_weight_down_block_inject.weight"]
    assert hc.shape[0] % 16 == 0


def test_model_adopts_mixed_k_from_the_checkpoint(exl3_checkpoint):
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict

    model = _build_model(_exl3_hf_config())
    model_state = model.state_dict()
    state = _materialize_loaded_weight_state_dict(
        model_state, W.iter_weights(exl3_checkpoint, torch.device("cpu"),
                                    include_moe_experts=False, include_non_moe=True,
                                    include_vision=False),
        device=torch.device("cpu"))
    for key, param in model_state.items():
        if key.endswith(_ROUTED):
            state[key] = torch.zeros(param.shape, dtype=param.dtype)
    model.load_state_dict(state)
    attn = model.model.layers.op_list[3].self_attn
    assert attn.indexer.index_qk_proj.k == 3
    assert attn.indexer.index_qk_proj.trellis.shape[-1] == 48
    assert {op.k for op in iter_exl3_linears(model) if op is not attn.indexer.index_qk_proj} == {5}
    # suh/svh stay fp16 and the trellis int16 through the engine's dtype cast
    assert attn.o_proj.suh.dtype == torch.float16 and attn.o_proj.trellis.dtype == torch.int16
    # and the dense workspace sizes itself from this tree: widest input is o_proj / GDN
    # out_proj (256), widest output the indexer (640, wider than the toy vocab of 512).
    from freetoken.kernel import exl3_linear

    try:
        ws = prepare_exl3_dense_workspace(model, torch.device("cpu"), _reset=True)
        assert ws is not None and (ws.max_in, ws.max_out) == (256, 640)
    finally:
        exl3_linear._WORKSPACES.pop(torch.device("cpu"), None)


def test_exl3_checkpoint_serves_pictures_from_ram(exl3_checkpoint, monkeypatch):
    """The mmap picture view assumes one bf16 extent; an EXL3 tower is not one."""
    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    monkeypatch.setenv("FREETOKEN_VISION_EXECUTION", "layer-stream")
    monkeypatch.setenv("FREETOKEN_VISION_WEIGHTS", "mmap")

    def refuse(_path):
        raise AssertionError("mmap picture view opened for an EXL3 checkpoint")

    monkeypatch.setattr(W, "open_mmap_vision_weights", refuse)
    names = {n for n, _ in W.iter_weights(exl3_checkpoint, torch.device("cpu"),
                                          include_moe_experts=False, include_non_moe=True,
                                          include_vision=True)}
    assert "lm_head.trellis" in names


def test_quantized_head_list_includes_exl3():
    from freetoken.engine.spec_lmhead import _QUANTIZED_HEADS

    assert "Exl3LMHead" in _QUANTIZED_HEADS
