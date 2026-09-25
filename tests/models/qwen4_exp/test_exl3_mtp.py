"""EXL3 MTP head (spec 2026-09-25 section 6).

The 3.05bpw_h5_ng5 checkpoint ships the MTP sidecar packed: fc_embedding/fc_hidden K=4,
q/k/v/o and the shared expert K=5, the indexer K=3, the 512 routed experts K=3 per expert
(``mtp.layers.0.mlp.experts.E.{gate,up,down}_proj.*``), router/HC fp16. Its final
``mtp.hyper_connection_mixer.*`` (three fp16 tensors) lives in
``mtp_hyper_connection_mixer_patch.safetensors``, which the index does not list.
"""

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.qwen4_exp import mtp_spike as M

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _raw_exl3_mtp_names(experts=4):
    names = []
    for mod in ("fc_embedding", "fc_hidden"):
        names += [f"mtp.{mod}.{c}" for c in ("trellis", "suh", "svh", "mul1")]
    for part in ("q_proj", "k_proj", "v_proj", "o_proj"):
        names += [f"mtp.layers.0.self_attn.{part}.{c}" for c in ("trellis", "suh", "svh", "mul1")]
    for part in ("gate_proj", "up_proj", "down_proj"):
        names += [f"mtp.layers.0.mlp.shared_expert.{part}.{c}" for c in ("trellis", "suh", "svh", "mul1")]
        for e in range(experts):
            names += [f"mtp.layers.0.mlp.experts.{e}.{part}.{c}" for c in ("trellis", "suh", "svh", "mul1")]
    return names


def test_plan_maps_exl3_dense_and_leaves_experts_to_banks():
    raw = _raw_exl3_mtp_names()
    model_names = [
        "fc_embedding.trellis", "fc_embedding.suh", "fc_embedding.svh", "fc_embedding.mul1",
        "layers.0.self_attn.qkv_proj.q_proj.trellis", "layers.0.self_attn.o_proj.mul1",
        "layers.0.mlp.shared_expert.gate_up_proj.up_proj.svh",
    ]
    plan = M.build_mtp_weight_plan(raw, model_names, exl3=True, strict=False)
    by_model = {e.model_name: e.raw_names for e in plan.entries}
    assert by_model["layers.0.self_attn.qkv_proj.q_proj.trellis"] == ("mtp.layers.0.self_attn.q_proj.trellis",)
    assert by_model["layers.0.mlp.shared_expert.gate_up_proj.up_proj.svh"] == (
        "mtp.layers.0.mlp.shared_expert.up_proj.svh",)


def test_plan_rejects_unknown_raw_names_in_exl3_mode():
    raw = _raw_exl3_mtp_names() + ["mtp.surprise.weight"]
    with pytest.raises(ValueError, match="unexpected MTP source"):
        M.build_mtp_weight_plan(raw, [], exl3=True)


def test_stack_exl3_experts_orders_banks():
    from freetoken.models.exl3_banks import EXL3_BANK_NAMES, stack_exl3_experts
    H, I, k, E = 128, 256, 3, 4
    def tensor_of(name):
        e = int(name.split(".experts.")[1].split(".")[0])
        if name.endswith(".trellis"):
            shape = (H // 16, I // 16, 16 * k) if "down" not in name else (I // 16, H // 16, 16 * k)
            return torch.full(shape, e, dtype=torch.int16)
        if name.endswith(".mul1"):
            return torch.tensor(0, dtype=torch.int32)
        n = (H if name.endswith("suh") else I) if "down" not in name else (I if name.endswith("suh") else H)
        return torch.full((n,), float(e), dtype=torch.float16)
    banks = stack_exl3_experts(tensor_of, prefix="mtp.layers.0.mlp.experts", experts=E,
                               hidden=H, intermediate=I, k=k)
    assert list(banks) == list(EXL3_BANK_NAMES)
    assert banks["gate_trellis"].shape == (E, H // 16, I // 16, 48)
    assert int(banks["down_trellis"][2].flatten()[0]) == 2


def test_stack_exl3_experts_refuses_a_wrong_k():
    from freetoken.models.exl3_banks import stack_exl3_experts

    def tensor_of(name):
        if name.endswith(".trellis"):
            return torch.zeros(8, 16, 32, dtype=torch.int16)  # K=2, bank wants K=3
        raise AssertionError(name)

    with pytest.raises(ValueError, match="trellis"):
        stack_exl3_experts(tensor_of, prefix="mtp.layers.0.mlp.experts", experts=1,
                           hidden=128, intermediate=256, k=3)


@pytest.mark.parametrize("stray", [
    "mtp.layers.0.mlp.experts.7.surprise.weight",
    "mtp.layers.0.mlp.experts.1.gate_proj.bias",
    "mtp.layers.0.mlp.experts.1.gate_proj.trellis.extra",
])
def test_plan_rejects_unknown_tensors_under_the_experts_prefix(stray):
    raw = _raw_exl3_mtp_names() + [stray]
    with pytest.raises(ValueError, match="unexpected MTP source") as info:
        M.build_mtp_weight_plan(raw, [], exl3=True, strict=False)
    assert stray in str(info.value)


def _expert_store(ids, H=128, I=256, k=3):
    from types import SimpleNamespace

    tensors = {}
    for e in ids:
        for proj, (fin, fout) in (("gate_proj", (H, I)), ("up_proj", (H, I)), ("down_proj", (I, H))):
            base = f"mtp.layers.0.mlp.experts.{e}.{proj}"
            tensors[f"{base}.trellis"] = torch.zeros((fin // 16, fout // 16, 16 * k), dtype=torch.int16)
            tensors[f"{base}.suh"] = torch.ones(fin, dtype=torch.float16)
            tensors[f"{base}.svh"] = torch.ones(fout, dtype=torch.float16)
            tensors[f"{base}.mul1"] = torch.tensor(0, dtype=torch.int32)
    cfg = SimpleNamespace(num_experts=2, hidden_size=H, moe_intermediate_size=I, exl3_expert_k=k)
    return SimpleNamespace(tensor=tensors.__getitem__, keys=tuple(tensors)), cfg


def test_from_store_refuses_an_out_of_range_expert_id():
    store, cfg = _expert_store([0, 1, 600])
    with pytest.raises(ValueError, match=r"experts\.600\.gate_proj\.trellis"):
        M.MTPExl3ExpertBanks.from_store(store, cfg)


def test_from_store_refuses_a_missing_expert_id():
    store, cfg = _expert_store([0])
    with pytest.raises(ValueError, match=r"missing ids \[1\]"):
        M.MTPExl3ExpertBanks.from_store(store, cfg)


def test_from_store_accepts_exactly_the_configured_ids():
    store, cfg = _expert_store([0, 1])
    assert M.MTPExl3ExpertBanks.from_store(store, cfg).num_experts == 2


def test_placement_is_exl3_for_exl3_checkpoints():
    from types import SimpleNamespace
    from freetoken.engine.spec_draft import resolve_spec_expert_placement
    cfg = SimpleNamespace(expert_quant="exl3", linear_storage="exl3")
    assert resolve_spec_expert_placement({}, model_config=cfg) == ("exl3", None)
    with pytest.raises(ValueError, match="exl3"):
        resolve_spec_expert_placement({"FREETOKEN_MTP_SPEC_EXPERT_FORMAT": "nvfp4"}, model_config=cfg)


def test_exl3_runner_type_and_bank_loader_are_wired():
    from types import SimpleNamespace

    from freetoken.engine.spec_draft import load_spec_expert_banks, spec_expert_runner_type

    assert spec_expert_runner_type("exl3") is M.MTPExl3GPUExpertRunner
    H, I, k, E = 128, 256, 3, 2
    tensors = {}
    for e in range(E):
        for proj, (fin, fout) in (("gate_proj", (H, I)), ("up_proj", (H, I)), ("down_proj", (I, H))):
            base = f"mtp.layers.0.mlp.experts.{e}.{proj}"
            tensors[f"{base}.trellis"] = torch.full((fin // 16, fout // 16, 16 * k), e, dtype=torch.int16)
            tensors[f"{base}.suh"] = torch.ones(fin, dtype=torch.float16)
            tensors[f"{base}.svh"] = torch.ones(fout, dtype=torch.float16)
            tensors[f"{base}.mul1"] = torch.tensor(0, dtype=torch.int32)
    store = SimpleNamespace(tensor=tensors.__getitem__, keys=tuple(tensors))
    cfg = SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I, exl3_expert_k=k)
    banks = load_spec_expert_banks("exl3", None, store, model_config=cfg)
    assert isinstance(banks, M.MTPExl3ExpertBanks)
    assert banks.quant_format == "exl3"
    assert (banks.num_experts, banks.hidden_size, banks.intermediate_size) == (E, H, I)
    from freetoken.moe.offload_cache import bank_bytes_per_expert

    assert banks.bytes_per_expert == bank_bytes_per_expert("exl3", H, I, cfg)
    with pytest.raises(ValueError, match="model config"):
        load_spec_expert_banks("exl3", None, store)


# ------------------------------------------------------------------ the real model's names


def _exl3_mtp_model():
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.utils.torch_utils import torch_dtype

    from .test_exl3_weight import _exl3_hf_config

    from freetoken.layers import rotary

    config = M.derive_mtp_model_config(parse_config(_exl3_hf_config()))
    assert config.linear_storage == "exl3"
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            return M.Qwen4ExpMTPModel(config)
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


def _unnest(model_name: str) -> str:
    for nested, part in (
        (".self_attn.qkv_proj.q_proj.", ".self_attn.q_proj."),
        (".self_attn.qkv_proj.k_proj.", ".self_attn.k_proj."),
        (".self_attn.qkv_proj.v_proj.", ".self_attn.v_proj."),
        (".mlp.shared_expert.gate_up_proj.gate_proj.", ".mlp.shared_expert.gate_proj."),
        (".mlp.shared_expert.gate_up_proj.up_proj.", ".mlp.shared_expert.up_proj."),
    ):
        model_name = model_name.replace(nested, part, 1)
    return "mtp." + model_name


def _checkpoint_names_for(model, experts: int) -> list[str]:
    raw = []
    for name in model.state_dict():
        if name in M._EXPERT_MODEL_NAMES:
            continue
        if name.endswith("input_mix_weight_down_block_inject.weight"):
            base = name[: -len("input_mix_weight_down_block_inject.weight")]
            raw += [f"mtp.{base}input_mix_weight_down.weight", f"mtp.{base}block_inject_weight.weight"]
        else:
            raw.append(_unnest(name))
    for e in range(experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            raw += [f"mtp.layers.0.mlp.experts.{e}.{proj}.{c}" for c in ("trellis", "suh", "svh", "mul1")]
    return raw


def test_exl3_mtp_model_is_packed_and_the_plan_maps_every_tensor():
    from freetoken.kernel.exl3_linear import Exl3ColMerged, Exl3Linear

    model = _exl3_mtp_model()
    assert isinstance(model.fc_embedding, Exl3Linear) and model.fc_embedding.k == 4
    assert isinstance(model.fc_hidden, Exl3Linear) and model.fc_hidden.k == 4
    layer = model.layers.op_list[0]
    assert isinstance(layer.self_attn.qkv_proj, Exl3ColMerged)
    assert isinstance(layer.mlp.shared_expert.gate_up_proj, Exl3ColMerged)

    names = list(model.state_dict())
    raw = _checkpoint_names_for(model, experts=3)
    plan = M.build_mtp_weight_plan(raw, names, exl3=True)  # strict: every raw name used
    assert set(plan.model_names) == set(names)
    assert plan.expert_model_names == set(M._EXPERT_MODEL_NAMES)
    used = set(plan.raw_names)
    assert not any(".mlp.experts." in name for name in used)
    assert {n for n in raw if ".mlp.experts." not in n} == used
    # the HC fusions still apply to an EXL3 checkpoint
    by_model = {e.model_name: e for e in plan.entries}
    hc = by_model["layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"]
    assert hc.raw_names == (
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    ) and hc.pad_to == 16
    # an unrelated missing dense tensor still fails loudly
    with pytest.raises(ValueError, match="missing MTP source"):
        M.build_mtp_weight_plan([n for n in raw if not n.endswith("fc_hidden.svh")], names, exl3=True)


def test_nvfp4_plan_is_unchanged_by_the_exl3_switch():
    # exl3=False keeps the q|k|v and shared gate|up fusions and rejects per-expert names.
    raw = ["mtp.layers.0.mlp.experts.0.gate_proj.trellis"]
    with pytest.raises(ValueError, match="unexpected MTP source"):
        M.build_mtp_weight_plan(raw, [])


def test_staged_runner_adopts_the_loaded_k_of_each_exl3_linear():
    model = _exl3_mtp_model()
    state = model.state_dict()
    cpu = {}
    for name, value in state.items():
        if name in M._EXPERT_MODEL_NAMES:
            continue
        if name.endswith(".trellis"):
            shape = (*value.shape[:2], 16 * (2 if name.startswith("fc_hidden.") else value.shape[2] // 16))
            cpu[name] = torch.zeros(shape, dtype=torch.int16)
        elif name.endswith(".mul1"):
            cpu[name] = torch.zeros((), dtype=torch.int32)
        else:
            cpu[name] = torch.zeros(value.shape, dtype=value.dtype)
    M.MTPStagedModelRunner(model, cpu, device=torch.device("cpu"), resident=False)
    assert model.fc_hidden.k == 2
    assert model.fc_embedding.k == 4
    assert model.layers.op_list[0].self_attn.indexer.index_qk_proj.k == 3


# ------------------------------------------------------------------ the unindexed patch file


def _header_keys(path) -> list[str]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return [k for k in json.loads(fh.read(n)) if k != "__metadata__"]


def test_weight_store_includes_an_unindexed_mtp_only_patch_file(tmp_path):
    indexed = {"mtp.pre_fc_norm_embedding.weight": torch.ones(8, dtype=torch.bfloat16)}
    save_file(indexed, str(tmp_path / "model-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"mtp.pre_fc_norm_embedding.weight": "model-00001.safetensors",
                       "model.norm.weight": "model-00001.safetensors"}}))
    patch = {
        "mtp.hyper_connection_mixer.hc_norm.weight": torch.full((8,), 2.0, dtype=torch.float16),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": torch.ones(8, 2, dtype=torch.float16),
    }
    save_file(patch, str(tmp_path / "mtp_hyper_connection_mixer_patch.safetensors"))
    # unindexed but not MTP-only (the PLE table, the converted n-gram shards): ignored
    save_file({"model.language_model.x": torch.zeros(2), "mtp.y": torch.zeros(2)},
              str(tmp_path / "mixed.safetensors"))
    save_file({"model.language_model.layers.1.ple.w": torch.zeros(2)},
              str(tmp_path / "freetoken-ple-00001-of-00016.safetensors"))

    store = M.MTPWeightStore(tmp_path)
    assert set(store.keys) == set(indexed) | set(patch)
    with store:
        got = store.tensor("mtp.hyper_connection_mixer.hc_norm.weight")
    assert got.dtype == torch.float16 and float(got[0]) == 2.0


def test_weight_store_refuses_a_patch_that_redefines_an_indexed_tensor(tmp_path):
    save_file({"mtp.a.weight": torch.ones(2)}, str(tmp_path / "model-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"mtp.a.weight": "model-00001.safetensors"}}))
    save_file({"mtp.a.weight": torch.zeros(2)}, str(tmp_path / "patch.safetensors"))
    with pytest.raises(ValueError, match="mtp.a.weight"):
        M.MTPWeightStore(tmp_path)


# ------------------------------------------------------------------ the packed expert runner


def _stub_mgemm(monkeypatch, calls):
    import freetoken.kernel.exl3_mgemm as G

    monkeypatch.setattr(G.Exl3MgemmBanks, "from_banks", classmethod(lambda cls, banks: "tables"))
    monkeypatch.setattr(G, "prepare_exl3_mgemm_scratch", lambda **kw: "scratch")

    def fake(hidden, tables, weights, ids, *, activation, swiglu_limit, scratch, **kw):
        assert (tables, scratch, activation, swiglu_limit) == ("tables", "scratch", "silu", None)
        assert ids.shape[0] * ids.shape[1] <= G.EXL3_MGEMM_MAX_INDICES
        calls.append(int(hidden.shape[0]))
        return hidden * weights.sum(-1, keepdim=True).to(hidden.dtype)

    monkeypatch.setattr(G, "fused_experts_exl3_mgemm", fake)


def _tiny_banks(E=12, H=128, I=256, k=3):
    from freetoken.models.exl3_banks import _bank_specs

    banks = {n: torch.zeros(shape, dtype=dt) for n, (shape, dt) in _bank_specs(E, H, I, k).items()}
    return M.MTPExl3ExpertBanks(banks, k)


def test_exl3_runner_tiles_rows_to_the_route_list_capacity(monkeypatch):
    calls: list[int] = []
    _stub_mgemm(monkeypatch, calls)
    banks = _tiny_banks()
    runner = M.MTPExl3GPUExpertRunner(
        banks, top_k=10, activation="silu", renormalize=True, max_tokens=128,
        num_threads=1, device=torch.device("cpu"))
    assert runner.resident_bytes == sum(t.numel() * t.element_size() for t in banks.banks.values())
    assert runner.banks.num_experts == 12 and runner.banks.hidden_size == 128
    x = torch.randn(30, 128).bfloat16()
    ids = torch.randint(0, 12, (30, 10), dtype=torch.int32)
    w = torch.full((30, 10), 0.1)
    out = runner.run_routed(x, w, ids)
    assert calls == [12, 12, 6]  # 128 route entries // top_k 10 = 12 rows a tile
    torch.testing.assert_close(out, x)
    assert runner.stats.calls == 1 and runner.stats.tokens == 30
    with pytest.raises(ValueError, match="int32"):
        runner.run_routed(x, w, ids.long())
    with pytest.raises(ValueError, match="at most 128"):
        runner.run_routed(torch.zeros(129, 128).bfloat16(), w, ids)


def test_exl3_runner_refuses_non_exl3_banks():
    banks = M.MTPBF16ExpertBanks(torch.zeros(2, 8, 8, dtype=torch.bfloat16),
                                 torch.zeros(2, 8, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="exl3"):
        M.MTPExl3GPUExpertRunner(banks, top_k=1, activation="silu", renormalize=True,
                                 max_tokens=4, num_threads=1, device=torch.device("cpu"))


@cuda
@pytest.mark.parametrize("rows", [1, 30, 40])
def test_exl3_runner_matches_reconstruct_first(rows):
    # Qwen3.8-Flash-Next 3.05bpw MTP routed experts: K=3, H=2560, I=640, SiLU. top_k 4 over 4
    # experts -> 32 rows a tile, so 40 rows crosses a tile boundary.
    from freetoken.kernel.exl3 import reconstruct
    from freetoken.models.exl3_banks import stack_exl3_experts

    dev, H, I, k, E, top_k = torch.device("cuda"), 2560, 640, 3, 4, 4
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for e in range(E):
        for proj, (fin, fout) in (("gate_proj", (H, I)), ("up_proj", (H, I)), ("down_proj", (I, H))):
            base = f"mtp.layers.0.mlp.experts.{e}.{proj}"
            tensors[f"{base}.trellis"] = torch.randint(
                -32768, 32767, (fin // 16, fout // 16, 16 * k), generator=g, dtype=torch.int32
            ).to(torch.int16)
            tensors[f"{base}.suh"] = (torch.randint(0, 2, (fin,), generator=g) * 2 - 1).half()
            tensors[f"{base}.svh"] = (torch.rand(fout, generator=g) * 0.02 + 0.01).half()
            tensors[f"{base}.mul1"] = torch.tensor(0, dtype=torch.int32)
    banks = M.MTPExl3ExpertBanks(
        stack_exl3_experts(tensors.__getitem__, prefix="mtp.layers.0.mlp.experts", experts=E,
                           hidden=H, intermediate=I, k=k), k)
    runner = M.MTPExl3GPUExpertRunner(banks, top_k=top_k, activation="silu", renormalize=True,
                                      max_tokens=128, num_threads=1, device=dev)
    x = torch.randn(rows, H, generator=g).to(dev).bfloat16()
    logits = torch.randn(rows, E, generator=g).to(dev).float()
    weights, ids = runner.route(x, logits)
    got = runner.run_routed(x, weights, ids).float()

    recon = {}
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"mtp.layers.0.mlp.experts.{e}.{proj}"
            recon[e, proj] = reconstruct(*(tensors[f"{base}.{c}"].to(dev) for c in ("trellis", "suh", "svh")),
                                         k=k, codebook="mul1").float()
    want = torch.zeros(rows, H, device=dev)
    for r in range(rows):
        xr = x[r : r + 1].float()
        for slot in range(top_k):
            e = int(ids[r, slot])
            act = torch.nn.functional.silu(xr @ recon[e, "gate_proj"].T) * (xr @ recon[e, "up_proj"].T)
            want[r] += float(weights[r, slot]) * (act @ recon[e, "down_proj"].T)[0]
    assert (got - want).norm() / want.norm() < 1e-2
