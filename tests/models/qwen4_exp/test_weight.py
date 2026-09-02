"""qwen4_exp weight loading against a synthetic checkpoint shaped like the RadixArk NVFP4 one.

The tensors are tiny but the key names, dtypes and the fusion geometry that matters
(hc_lowrank=320 + hc_count=4 -> a 12-row zero pad) are the real ones.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import random
import struct
from types import SimpleNamespace

import pytest
import safetensors
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.aot_models import SUPPORTED_MODELS, expert_bank_row_bytes
from freetoken.models.qwen4_exp import weight as weight_mod
from freetoken.models.qwen4_exp.weight import (
    _ZERO_CENTERED_NORM_SUFFIXES,
    _PleRowCache,
    _rename,
    iter_weights,
    load_mmap_ple_table,
    load_ple_table,
)
from freetoken.models.weight import _ST_DTYPE
from freetoken.moe.host_banks import HostBank, read_range_into

# safetensors dtype string per torch dtype, inverted from the reader's own map so a test
# shard can never disagree with what the loader will parse back out of the header.
_ST_NAME = {dtype: name for name, dtype in _ST_DTYPE.items()}

H = 32  # hidden_size
HC = 4  # hc_count
LR = 320  # hc_lowrank; kept real so the merged HC pad is the real (-(320+4)) % 16 = 12
HCH = HC * H  # hyper-connection stream width
KH, VH, HD = 2, 6, 8  # GDN key / value heads, head dim
QH, KVH, AHD = 4, 2, 16  # QSA q / kv heads, head dim
IHD = 8  # indexer head dim
E, I = 3, 6  # routed experts, moe_intermediate_size
NGRAM_DIM, NGRAM_ROWS, NGRAM_SHARDS = 4, 7, 4


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _hc_weights(prefix: str, inject: bool) -> dict[str, torch.Tensor]:
    w = {
        f"{prefix}.hc_norm.weight": _bf16(HCH),
        f"{prefix}.input_mix_weight_down.weight": _bf16(LR, HCH),
        f"{prefix}.input_mix_weight_up.weight": _bf16(HCH, LR),
    }
    if inject:
        w[f"{prefix}.block_inject_weight.weight"] = _bf16(HC, HCH)
    return w


def _raw_checkpoint() -> dict[str, torch.Tensor]:
    """Layer 0 = GDN + PLE, layer 1 = QSA; plus the mtp / visual / routed-expert noise."""
    lm = "model.language_model"
    raw: dict[str, torch.Tensor] = {
        f"{lm}.embed_tokens.weight": _bf16(11, H),
        "lm_head.weight": _bf16(11, H),
    }
    raw.update(_hc_weights(f"{lm}.hyper_connection_mixer", inject=False))
    for layer in (0, 1):
        raw.update(_hc_weights(f"{lm}.layers.{layer}.attn_hyper_connection", inject=True))
        raw.update(_hc_weights(f"{lm}.layers.{layer}.mlp_hyper_connection", inject=True))
        raw.update({
            f"{lm}.layers.{layer}.mlp.gate.weight": _bf16(E, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.gate_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.up_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.down_proj.weight": _bf16(H, I),
            f"{lm}.layers.{layer}.mlp.shared_expert_gate.weight": _bf16(1, H),
        })
        for expert in range(E):
            base = f"{lm}.layers.{layer}.mlp.experts.{expert}"
            for proj, out, inn in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
                raw[f"{base}.{proj}.weight"] = torch.randint(
                    0, 256, (out, inn // 2), dtype=torch.uint8
                )
                raw[f"{base}.{proj}.weight_scale"] = torch.ones(
                    out, inn // 16 or 1, dtype=torch.float8_e4m3fn
                )
                raw[f"{base}.{proj}.weight_scale_2"] = torch.tensor(0.5)
                raw[f"{base}.{proj}.input_scale"] = torch.tensor(0.25)
    gdn = f"{lm}.layers.0.linear_attn"
    raw.update({
        f"{gdn}.in_proj_qkv.weight": _bf16(2 * KH * HD + VH * HD, H),
        f"{gdn}.in_proj_z.weight": _bf16(VH * HD, H),
        f"{gdn}.in_proj_b.weight": _bf16(VH, H),
        f"{gdn}.in_proj_a.weight": _bf16(VH, H),
        f"{gdn}.conv1d.weight": _bf16(2 * KH * HD + VH * HD, 1, 4),
        f"{gdn}.A_log": _bf16(VH),
        f"{gdn}.dt_bias": _bf16(VH),
        f"{gdn}.norm.weight": _bf16(HD),
        f"{gdn}.out_proj.weight": _bf16(H, VH * HD),
    })
    ple = f"{lm}.layers.0.ple"
    raw.update({
        f"{ple}.key_proj.weight": _bf16(HCH, H),
        f"{ple}.value_proj.weight": _bf16(H, H),
        f"{ple}.norm_key.weight": _bf16(HCH),
        f"{ple}.norm_query.weight": _bf16(HCH),
        f"{ple}.norm_conv.weight": _bf16(HCH),
        f"{ple}.conv1d.weight": _bf16(HCH, 1, 4),
        f"{ple}.ple_embedding.layer_multipliers": torch.randint(1, 1 << 40, (3,)),
        f"{ple}.ple_embedding.ngram_heads_offsets": torch.arange(4),
        f"{ple}.ple_embedding.ngram_heads_vocab_sizes": torch.full((4,), 5),
    })
    attn = f"{lm}.layers.1.self_attn"
    raw.update({
        f"{attn}.q_proj.weight": _bf16(2 * QH * AHD, H),
        f"{attn}.k_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.v_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.o_proj.weight": _bf16(H, QH * AHD),
        f"{attn}.q_norm.weight": _bf16(AHD),
        f"{attn}.k_norm.weight": _bf16(AHD),
        f"{attn}.indexer.index_qk_proj.weight": _bf16(5 * IHD, H),
        f"{attn}.indexer.q_layernorm.weight": _bf16(IHD),
        f"{attn}.indexer.k_layernorm.weight": _bf16(IHD),
    })
    raw.update({
        "mtp.hyper_connection_mixer.hc_norm.weight": _bf16(HCH),
        "mtp.layers.0.self_attn.q_proj.weight": _bf16(2 * QH * AHD, H),
        "mtp.layers.0.mlp.experts.gate_up_proj": _bf16(E, 2 * I, H),
        "mtp.layers.0.mlp.experts.down_proj": _bf16(E, H, I),
        "model.visual.blocks.0.attn.qkv.weight": _bf16(3 * H, H),
        "model.visual.merger.norm.weight": _bf16(H),
    })
    return raw


def _ngram_table() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shards = {
        f"{prefix}.shard_{i}.weight": (
            torch.arange(i * NGRAM_ROWS * NGRAM_DIM, (i + 1) * NGRAM_ROWS * NGRAM_DIM)
            .remainder(200).to(torch.uint8).view(NGRAM_ROWS, NGRAM_DIM).view(torch.float8_e4m3fn)
        )
        for i in range(NGRAM_SHARDS)
    }
    scale = torch.tensor([0.125], dtype=torch.bfloat16)
    shards[f"{prefix}.weight_scale"] = scale
    return shards, scale


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    folder = tmp_path_factory.mktemp("qwen4_exp_ckpt")
    raw = _raw_checkpoint()
    table, _scale = _ngram_table()
    # Spread the dense tensors over two shards so the fusion buffer has to survive a file
    # boundary, and put the n-gram table in its own shards like the real checkpoint does.
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    shard_names = sorted(table)
    save_file({n: table[n] for n in shard_names[:2]}, str(folder / "model-plefp8-00000.safetensors"))
    save_file({n: table[n] for n in shard_names[2:]}, str(folder / "model-plefp8-00001.safetensors"))
    return str(folder), {**raw, **table}


@pytest.fixture(scope="module")
def loaded(checkpoint) -> dict[str, torch.Tensor]:
    folder, _raw = checkpoint
    return {
        name: tensor.clone()
        for name, tensor in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True
        )
    }


def _expected_names() -> set[str]:
    names = {"model.embed_tokens.weight", "lm_head.weight"}
    names |= {f"model.hyper_connection_mixer.{leaf}" for leaf in
              ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")}
    for layer in (0, 1):
        for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
            names |= {f"model.layers.{layer}.{hc}.{leaf}" for leaf in (
                "hc_norm.weight", "input_mix_weight_down_block_inject.weight",
                "input_mix_weight_up.weight")}
        names |= {f"model.layers.{layer}.mlp.{leaf}" for leaf in (
            "gate.weight", "shared_expert.gate_up_proj.weight",
            "shared_expert.down_proj.weight", "shared_expert_gate.weight")}
    names |= {f"model.layers.0.linear_attn.{leaf}" for leaf in (
        "in_proj.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight")}
    names |= {f"model.layers.0.ple.{leaf}" for leaf in (
        "key_proj.weight", "value_proj.weight", "norm_key.weight", "norm_query.weight",
        "norm_conv.weight", "conv1d.weight", "ple_embedding.layer_multipliers",
        "ple_embedding.ngram_heads_offsets", "ple_embedding.ngram_heads_vocab_sizes")}
    names |= {f"model.layers.1.self_attn.{leaf}" for leaf in (
        "qkv_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight",
        "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight",
        "indexer.k_layernorm.weight")}
    return names


def test_picture_keys_are_retained_only_when_enabled():
    assert _rename("model.visual.blocks.0.attn.qkv.weight", include_vision=False) is None
    assert _rename("visual.merger.norm.weight", include_vision=False) is None
    assert (
        _rename("model.visual.blocks.0.attn.qkv.weight", include_vision=True)
        == "visual.blocks.0.attn.qkv.weight"
    )
    assert (
        _rename("visual.merger.norm.weight", include_vision=True)
        == "visual.merger.norm.weight"
    )
    assert _rename("mtp.visual.weight", include_vision=True) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
@pytest.mark.parametrize(
    ("execution", "visual_device"), (("gpu", "cuda"), ("layer-stream", "cpu"))
)
def test_picture_weights_are_read_directly_on_their_persistent_device(
    checkpoint, monkeypatch, execution, visual_device
):
    folder, _raw = checkpoint
    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    monkeypatch.setenv("FREETOKEN_VISION_EXECUTION", execution)

    values = dict(
        iter_weights(
            folder,
            torch.device("cuda"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )

    assert values["visual.blocks.0.attn.qkv.weight"].device.type == visual_device
    assert values["visual.merger.norm.weight"].device.type == visual_device
    assert values["model.embed_tokens.weight"].device.type == "cuda"


def test_key_map_is_exactly_the_model_state_dict(loaded):
    assert set(loaded) == _expected_names()


def test_mtp_visual_experts_and_table_never_loaded(loaded):
    for name in loaded:
        assert not name.startswith(("mtp.", "model.visual."))
        assert ".mlp.experts." not in name
        assert "ngram_embedding" not in name
        assert not name.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))


def test_hc_merge_is_down_then_inject_then_zero_pad(loaded, checkpoint):
    _folder, raw = checkpoint
    key = "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    merged = loaded[key]
    assert merged.shape == (LR + HC + 12, HCH)  # pad = (-(320 + 4)) % 16
    down = raw["model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"]
    inject = raw["model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight"]
    assert torch.equal(merged[:LR], down)
    assert torch.equal(merged[LR:LR + HC], inject)
    assert torch.equal(merged[LR + HC:], torch.zeros(12, HCH, dtype=merged.dtype))


def test_top_level_mixer_keeps_the_unmerged_down(loaded, checkpoint):
    _folder, raw = checkpoint
    got = loaded["model.hyper_connection_mixer.input_mix_weight_down.weight"]
    assert got.shape == (LR, HCH)
    assert torch.equal(
        got, raw["model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"]
    )
    assert torch.equal(
        loaded["model.hyper_connection_mixer.input_mix_weight_up.weight"],
        raw["model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"],
    )


def test_qkv_fusion_slices_back_to_q_k_v(loaded, checkpoint):
    _folder, raw = checkpoint
    attn = "model.language_model.layers.1.self_attn"
    parts = [raw[f"{attn}.{p}_proj.weight"] for p in ("q", "k", "v")]
    fused = loaded["model.layers.1.self_attn.qkv_proj.weight"]
    assert fused.shape == (2 * QH * AHD + 2 * KVH * AHD, H)  # q carries the output gate
    for part, back in zip(parts, torch.split(fused, [p.shape[0] for p in parts], dim=0)):
        assert torch.equal(part, back)


def test_gdn_in_proj_slices_round_trip(loaded, checkpoint):
    _folder, raw = checkpoint
    gdn = "model.language_model.layers.0.linear_attn"
    parts = [raw[f"{gdn}.in_proj_{p}.weight"] for p in ("qkv", "z", "b", "a")]
    fused = loaded["model.layers.0.linear_attn.in_proj.weight"]
    assert fused.shape == (sum(p.shape[0] for p in parts), H)
    splits = torch.split(fused, [p.shape[0] for p in parts], dim=0)
    for part, back in zip(parts, splits):
        assert torch.equal(part, back)


def test_shared_expert_gate_up_merge(loaded, checkpoint):
    _folder, raw = checkpoint
    base = "model.language_model.layers.1.mlp.shared_expert"
    merged = loaded["model.layers.1.mlp.shared_expert.gate_up_proj.weight"]
    assert torch.equal(merged[:I], raw[f"{base}.gate_proj.weight"])
    assert torch.equal(merged[I:], raw[f"{base}.up_proj.weight"])


ZERO_CENTERED = (
    "model.layers.0.attn_hyper_connection.hc_norm.weight",
    "model.layers.0.mlp_hyper_connection.hc_norm.weight",
    "model.hyper_connection_mixer.hc_norm.weight",
    "model.layers.0.ple.norm_key.weight",
    "model.layers.0.ple.norm_query.weight",
    "model.layers.0.ple.norm_conv.weight",
    "model.layers.1.self_attn.q_norm.weight",
    "model.layers.1.self_attn.k_norm.weight",
    "model.layers.1.self_attn.indexer.q_layernorm.weight",
    "model.layers.1.self_attn.indexer.k_layernorm.weight",
)


def test_zero_centered_norms_are_loaded_raw(loaded, checkpoint):
    """(1+w) is applied at runtime in fp32, so the loader must not fold it into the bf16 weight."""
    _folder, raw = checkpoint
    for name in ZERO_CENTERED:
        raw_name = name.replace("model.", "model.language_model.", 1)
        assert torch.equal(loaded[name], raw[raw_name]), name


def test_the_zero_centered_suffix_list_covers_every_such_norm():
    assert {n for n in ZERO_CENTERED if n.endswith(_ZERO_CENTERED_NORM_SUFFIXES)} == set(ZERO_CENTERED)
    assert not "model.layers.0.linear_attn.norm.weight".endswith(_ZERO_CENTERED_NORM_SUFFIXES)


def test_gdn_gated_norm_passes_through(loaded, checkpoint):
    _folder, raw = checkpoint
    assert torch.equal(
        loaded["model.layers.0.linear_attn.norm.weight"],
        raw["model.language_model.layers.0.linear_attn.norm.weight"],
    )


def test_hash_constants_stay_int64(loaded):
    for leaf in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
        assert loaded[f"model.layers.0.ple.ple_embedding.{leaf}"].dtype is torch.int64


def test_load_ple_table_concatenates_shards_in_index_order(checkpoint):
    folder, raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    table = load_ple_table(folder, args, pin=False)
    assert table.tensor.shape == (NGRAM_SHARDS * NGRAM_ROWS, NGRAM_DIM)
    assert table.tensor.dtype is torch.float8_e4m3fn
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    for shard in range(NGRAM_SHARDS):
        rows = table.tensor[shard * NGRAM_ROWS: (shard + 1) * NGRAM_ROWS]
        assert torch.equal(rows.view(torch.uint8),
                           raw[f"{prefix}.shard_{shard}.weight"].view(torch.uint8))
    assert table.weight_scale.dtype is torch.bfloat16
    assert float(table.weight_scale) == 0.125


def test_load_ple_table_rejects_a_shard_count_mismatch(checkpoint):
    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS + 1, ngram_head_dim=NGRAM_DIM)
    with pytest.raises(ValueError, match="shards 0"):
        load_ple_table(folder, args, pin=False)


def test_mmap_ple_table_gathers_rows_without_materializing_the_table(checkpoint):
    folder, raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    table = load_mmap_ple_table(folder, args)
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    ids = torch.tensor([
        0,
        NGRAM_ROWS - 1,
        NGRAM_ROWS,
        2 * NGRAM_ROWS + 3,
        NGRAM_SHARDS * NGRAM_ROWS - 1,
        -1,
        NGRAM_SHARDS * NGRAM_ROWS,
    ], dtype=torch.int64)
    out = torch.empty(ids.numel(), NGRAM_DIM, dtype=torch.uint8)
    try:
        assert table.storage.gather(ids, out) is out
        assert table.storage.num_rows == NGRAM_SHARDS * NGRAM_ROWS
        assert table.storage.nbytes == NGRAM_SHARDS * NGRAM_ROWS * NGRAM_DIM
        for i, row_id in enumerate(ids.tolist()[:5]):
            shard, row = divmod(row_id, NGRAM_ROWS)
            want = raw[f"{prefix}.shard_{shard}.weight"][row].view(torch.uint8)
            assert torch.equal(out[i], want)
        assert out[-2:].count_nonzero() == 0
        assert float(table.weight_scale) == 0.125
    finally:
        table.storage.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_mmap_staged_backend_matches_checkpoint_rows(checkpoint):
    from freetoken.models.qwen4_exp.ple import MmapStagedTable

    folder, raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    table = load_mmap_ple_table(folder, args)
    backend = MmapStagedTable(table.storage, float(table.weight_scale))
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    source = torch.cat([
        raw[f"{prefix}.shard_{i}.weight"] for i in range(NGRAM_SHARDS)
    ]).cuda()
    ids = torch.tensor([[0, 6, 7, 17], [27, 20, 3, 14]], device="cuda")
    want = source.index_select(0, ids.reshape(-1)).to(torch.bfloat16)
    want = (want * float(table.weight_scale)).view(2, -1)
    try:
        # Match the scheduler's inference-mode context.
        with torch.inference_mode():
            assert torch.equal(backend.lookup(ids), want)
            backend.prefetch(ids)
            assert torch.equal(backend.lookup(ids), want)
    finally:
        backend.close()
        table.storage.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_mmap_staged_backend_reuses_pinned_staging_across_steps(checkpoint):
    """A decode step must not re-``pin_memory`` its staging pair; a prefill's size is not pooled."""
    import freetoken.models.qwen4_exp.ple as ple_module
    from freetoken.models.qwen4_exp.ple import MmapStagedTable

    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    table = load_mmap_ple_table(folder, args)
    backend = MmapStagedTable(table.storage, float(table.weight_scale))
    ids = torch.tensor([[0, 6, 7, 17], [27, 20, 3, 14]], device="cuda")
    try:
        with torch.inference_mode():
            for _ in range(2 * ple_module._MMAP_STAGING_SLOTS + 1):
                backend.lookup(ids)
            ring = backend._host_pool[ids.numel()]
            assert len(ring) == ple_module._MMAP_STAGING_SLOTS
            assert all(slot.rows.is_pinned() for slot in ring)
            assert len({id(slot.rows) for slot in ring}) == ple_module._MMAP_STAGING_SLOTS

            # A chunk-sized gather is allocated fresh: pooling every distinct prefill width
            # would pin tens of MB per width for the life of the process.
            wide = ple_module._MMAP_STAGING_MAX_ROWS + 1
            assert backend._acquire_staging(wide) is not backend._acquire_staging(wide)
            assert wide not in backend._host_pool
    finally:
        backend.close()
        table.storage.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_mmap_staged_backend_cuda_graph_replay_stages_new_rows(checkpoint):
    from freetoken.models.qwen4_exp.ple import MmapStagedTable

    folder, raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    table = load_mmap_ple_table(folder, args)
    backend = MmapStagedTable(table.storage, float(table.weight_scale))
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    source = torch.cat([
        raw[f"{prefix}.shard_{i}.weight"] for i in range(NGRAM_SHARDS)
    ]).cuda()
    graphs = {}
    captured = {}
    try:
        for tokens in (1, 2):
            backend.prepare_cuda_graph_capture(tokens * 4)
            captured[tokens] = torch.empty(
                tokens, 4 * NGRAM_DIM, dtype=torch.bfloat16, device="cuda"
            )
        torch.cuda.synchronize()
        for tokens in (1, 2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured[tokens].copy_(backend.cuda_graph_lookup(tokens, 4))
            graphs[tokens] = graph

        for ids in (
            torch.tensor([[0, 6, 7, 17]], device="cuda"),
            torch.tensor([[1, 8, 15, 22], [4, 11, 18, 25]], device="cuda"),
            torch.tensor([[27, 20, 3, 14]], device="cuda"),
        ):
            tokens = ids.shape[0]
            with torch.inference_mode():
                backend.prefetch(ids)
                backend.prepare_cuda_graph_replay(ids)
                graphs[tokens].replay()
            torch.cuda.synchronize()
            want = source.index_select(0, ids.reshape(-1)).to(torch.bfloat16)
            want = (want * float(table.weight_scale)).view_as(captured[tokens])
            assert torch.equal(captured[tokens], want)
    finally:
        graphs.clear()
        torch.cuda.synchronize()
        backend.reset_cuda_graph()
        backend.close()
        table.storage.close()


# ======================================================================================
# MmapPleStorage.gather: parallel row faults behind a bounded host row cache
# ======================================================================================


def _ngram_reference(raw) -> torch.Tensor:
    """The whole synthetic n-gram table as one ``[num_rows, head_dim]`` uint8 tensor."""
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    return torch.cat(
        [raw[f"{prefix}.shard_{i}.weight"] for i in range(NGRAM_SHARDS)]
    ).view(torch.uint8)


def _serial_gather(table: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Reference: one row at a time, zeros for out-of-range ids."""
    out = torch.zeros(len(ids), NGRAM_DIM, dtype=torch.uint8)
    for i, row_id in enumerate(ids):
        if 0 <= row_id < table.shape[0]:
            out[i] = table[row_id]
    return out


def _open_mmap_ple(folder: str):
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    return load_mmap_ple_table(folder, args)


def _gather(storage, ids: list[int]) -> torch.Tensor:
    out = torch.empty(len(ids), NGRAM_DIM, dtype=torch.uint8)
    return storage.gather(torch.tensor(ids, dtype=torch.int64), out)


ALL_ROWS = NGRAM_SHARDS * NGRAM_ROWS


@pytest.mark.parametrize("ids", [
    pytest.param([], id="empty"),
    pytest.param([5], id="single"),
    pytest.param([0, -1, 3, ALL_ROWS, ALL_ROWS - 1, -1000], id="mixed-invalid"),
    pytest.param([9, 9, 9, 2, 9, 2], id="duplicates"),
    pytest.param(list(range(ALL_ROWS)), id="every-shard"),
    pytest.param([r % ALL_ROWS for r in range(0, 3 * ALL_ROWS, 3)], id="strided"),
])
@pytest.mark.parametrize("cache", ["0", "65536"])
def test_parallel_gather_matches_a_serial_reference(checkpoint, monkeypatch, ids, cache):
    folder, raw = checkpoint
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", cache)
    table = _open_mmap_ple(folder)
    try:
        assert torch.equal(_gather(table.storage, ids), _serial_gather(_ngram_reference(raw), ids))
    finally:
        table.storage.close()


def test_row_cache_disabled_by_env_zero(checkpoint, monkeypatch):
    folder, raw = checkpoint
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", "0")
    table = _open_mmap_ple(folder)
    ids = [0, 13, 27, -1]
    try:
        assert table.storage._row_cache is None
        assert torch.equal(_gather(table.storage, ids), _serial_gather(_ngram_reference(raw), ids))
    finally:
        table.storage.close()


def test_row_cache_serves_a_repeated_gather_byte_identically(checkpoint, monkeypatch):
    folder, raw = checkpoint
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", "65536")
    table = _open_mmap_ple(folder)
    ids = [3, 10, 17, 24, -1, 6, ALL_ROWS + 2]
    want = _serial_gather(_ngram_reference(raw), ids)
    try:
        assert table.storage._row_cache is not None
        assert torch.equal(_gather(table.storage, ids), want)
        assert torch.equal(_gather(table.storage, ids), want)  # every row now a cache hit
        assert torch.equal(_gather(table.storage, ids[::-1]), _serial_gather(
            _ngram_reference(raw), ids[::-1]
        ))
    finally:
        table.storage.close()


def test_row_cache_stays_correct_across_evictions(checkpoint, monkeypatch):
    folder, raw = checkpoint
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", "2")  # far below the working set
    table = _open_mmap_ple(folder)
    reference = _ngram_reference(raw)
    try:
        for _ in range(4):
            for start in range(0, ALL_ROWS, 3):
                ids = [(start + k) % ALL_ROWS for k in range(5)]
                assert torch.equal(_gather(table.storage, ids), _serial_gather(reference, ids))
    finally:
        table.storage.close()


def test_gather_of_many_duplicates_is_deterministic(checkpoint, monkeypatch):
    """Enough rows to fan out over the pool, few enough ids that workers share source rows."""
    folder, raw = checkpoint
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", "65536")
    table = _open_mmap_ple(folder)
    ids = [(i % 3) * 9 for i in range(256)]
    want = _serial_gather(_ngram_reference(raw), ids)
    try:
        for _ in range(8):
            assert torch.equal(_gather(table.storage, ids), want)
    finally:
        table.storage.close()


def test_gather_runs_under_inference_mode_like_the_staged_table(checkpoint, monkeypatch):
    folder, raw = checkpoint
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", "65536")
    table = _open_mmap_ple(folder)
    ids = list(range(ALL_ROWS))
    want = _serial_gather(_ngram_reference(raw), ids)
    try:
        with torch.inference_mode():
            got = _gather(table.storage, ids)
            assert torch.equal(got, want)
            assert torch.equal(_gather(table.storage, ids), want)
    finally:
        table.storage.close()


# ======================================================================================
# The Windows PrefetchVirtualMemory fast path
# ======================================================================================


@pytest.fixture
def prefetch_enabled(monkeypatch):
    """Force the fast path on with a stub that records calls and reports success.

    The stub does not really prefetch; the copy that follows faults the pages itself, so a
    gather behind it must still return exactly the same bytes.
    """
    calls: list[tuple[int, int]] = []

    def fake(handle, entries, pointer, flags):
        calls.append((int(entries), int(flags)))
        return 1

    monkeypatch.setattr(weight_mod, "_prefetch_virtual_memory", fake)
    monkeypatch.setattr(weight_mod, "_current_process", -1)
    monkeypatch.setattr(weight_mod, "_prefetch_failed", False)
    return calls


def test_range_entries_are_16_byte_address_length_pairs(checkpoint):
    """One WIN32_MEMORY_RANGE_ENTRY per row: {PVOID VirtualAddress; SIZE_T NumberOfBytes;}."""
    folder, raw = checkpoint
    table = _open_mmap_ple(folder)
    storage = table.storage
    try:
        ids = torch.tensor(
            [0, NGRAM_ROWS - 1, NGRAM_ROWS, 2 * NGRAM_ROWS + 3, ALL_ROWS - 1, 5, 5],
            dtype=torch.int64,
        )
        entries = storage._range_entries(ids)
        assert entries.dtype == torch.int64
        assert entries.is_contiguous()
        assert tuple(entries.shape) == (ids.numel(), 2)
        assert entries.element_size() * entries.shape[1] == 16
        want = [
            [
                storage._shards[row // NGRAM_ROWS].data_ptr() + (row % NGRAM_ROWS) * NGRAM_DIM,
                NGRAM_DIM,
            ]
            for row in ids.tolist()
        ]
        assert entries.tolist() == want
        # and the addresses really name those rows in the mapping
        reference = _ngram_reference(raw)
        for row, (address, length) in zip(ids.tolist(), entries.tolist()):
            assert bytes(ctypes.string_at(address, length)) == bytes(reference[row].tolist())
    finally:
        table.storage.close()


def test_range_entries_of_an_empty_id_set_is_empty(checkpoint):
    folder, _raw = checkpoint
    table = _open_mmap_ple(folder)
    try:
        entries = table.storage._range_entries(torch.empty(0, dtype=torch.int64))
        assert tuple(entries.shape) == (0, 2)
    finally:
        table.storage.close()


@pytest.mark.parametrize("cache", ["0", "1048576"])
def test_prefetched_gather_matches_the_serial_reference(
    checkpoint, monkeypatch, prefetch_enabled, cache
):
    folder, raw = checkpoint
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", cache)
    table = _open_mmap_ple(folder)
    ids = [r % ALL_ROWS for r in range(0, 3 * ALL_ROWS, 3)] + [-1, ALL_ROWS + 4]
    try:
        assert torch.equal(_gather(table.storage, ids), _serial_gather(_ngram_reference(raw), ids))
        assert prefetch_enabled  # the fast path really ran
        assert all(flags == 0 for _entries, flags in prefetch_enabled)  # Flags is reserved
    finally:
        table.storage.close()


def test_prefetch_covers_every_missing_row_in_one_call(checkpoint, monkeypatch, prefetch_enabled):
    folder, _raw = checkpoint
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", "1048576")
    table = _open_mmap_ple(folder)
    try:
        _gather(table.storage, [3, 10, 17, 24, -1, ALL_ROWS])  # 4 valid rows, all missing
        assert prefetch_enabled == [(4, 0)]
        _gather(table.storage, [3, 10, 17, 24, 9, 16])  # 4 cached, 2 new
        assert prefetch_enabled == [(4, 0), (2, 0)]
        _gather(table.storage, [3, 10, 17, 24])  # every row cached: no syscall at all
        assert prefetch_enabled == [(4, 0), (2, 0)]
    finally:
        table.storage.close()


def test_a_single_missing_row_skips_the_syscall(checkpoint, monkeypatch, prefetch_enabled):
    """One row is cheaper to just fault (0.15 ms) than to prefetch and fault (0.17 ms)."""
    folder, raw = checkpoint
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", "0")
    table = _open_mmap_ple(folder)
    try:
        assert torch.equal(_gather(table.storage, [11]), _serial_gather(_ngram_reference(raw), [11]))
        assert prefetch_enabled == []
    finally:
        table.storage.close()


def test_gather_falls_back_when_the_api_is_unavailable(checkpoint, monkeypatch):
    """No PrefetchVirtualMemory (non-Windows, or a kernel without the symbol)."""
    folder, raw = checkpoint
    monkeypatch.setattr(weight_mod, "_prefetch_virtual_memory", None)
    monkeypatch.setattr(weight_mod, "_prefetch_failed", False)
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", "0")
    table = _open_mmap_ple(folder)
    ids = list(range(ALL_ROWS))
    try:
        assert not table.storage._prefetch_rows(torch.tensor(ids, dtype=torch.int64))
        assert torch.equal(_gather(table.storage, ids), _serial_gather(_ngram_reference(raw), ids))
    finally:
        table.storage.close()


def test_a_false_return_warns_once_and_disables_the_path(checkpoint, monkeypatch):
    folder, raw = checkpoint
    calls: list[int] = []
    warnings: list[tuple] = []

    def failing(handle, entries, pointer, flags):
        calls.append(int(entries))
        return 0

    monkeypatch.setattr(weight_mod, "_prefetch_virtual_memory", failing)
    monkeypatch.setattr(weight_mod, "_current_process", -1)
    monkeypatch.setattr(weight_mod, "_prefetch_failed", False)
    monkeypatch.setattr(
        weight_mod.logger,
        "warning",
        lambda msg, *args: warnings.append((msg, args)),
        raising=False,
    )
    monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", "0")
    table = _open_mmap_ple(folder)
    ids = [1, 8, 15, 22, -3]
    want = _serial_gather(_ngram_reference(raw), ids)
    try:
        assert torch.equal(_gather(table.storage, ids), want)  # falls back, still correct
        assert torch.equal(_gather(table.storage, ids), want)
        assert torch.equal(_gather(table.storage, ids), want)
        assert calls == [4]  # tried once, never again for the process
        assert weight_mod._prefetch_failed is True
        assert len(warnings) == 1
    finally:
        table.storage.close()


# ======================================================================================
# _PleRowCache: the slab-backed bounded row cache
# ======================================================================================


def _rows(n: int, start: int = 0) -> torch.Tensor:
    return (torch.arange(start, start + n * NGRAM_DIM) % 251).to(torch.uint8).reshape(n, NGRAM_DIM)


def test_row_cache_default_capacity_is_a_mebirow():
    assert weight_mod._PLE_ROW_CACHE_DEFAULT == 1_048_576


def test_row_cache_splits_hits_from_misses_and_serves_the_hits():
    cache = _PleRowCache(8, NGRAM_DIM)
    src = _rows(3)
    assert cache.take_hits([5, 6, 7], [0, 1, 2], src.clone()) == ([5, 6, 7], [0, 1, 2])
    cache.insert([5, 6, 7], [0, 1, 2], src)
    out = torch.zeros(4, NGRAM_DIM, dtype=torch.uint8)
    # positions deliberately unsorted and one id still absent
    assert cache.take_hits([7, 99, 5], [3, 1, 0], out) == ([99], [1])
    assert torch.equal(out[3], src[2])
    assert torch.equal(out[0], src[0])
    assert torch.equal(out[1], torch.zeros(NGRAM_DIM, dtype=torch.uint8))


def test_row_cache_holds_copies_not_views_of_the_gather_buffer():
    cache = _PleRowCache(4, NGRAM_DIM)
    src = _rows(1, start=13)
    cache.insert([1], [0], src)
    src.zero_()  # the gather buffer is reused every step; the cache must not follow it
    out = torch.zeros(1, NGRAM_DIM, dtype=torch.uint8)
    assert cache.take_hits([1], [0], out) == ([], [])
    assert torch.equal(out[0], _rows(1, start=13)[0])


def test_row_cache_evicts_in_fifo_order():
    cache = _PleRowCache(2, NGRAM_DIM)
    src = _rows(3)
    cache.insert([10, 11], [0, 1], src)
    cache.insert([12], [2], src)  # takes slot 0 back from id 10
    out = torch.zeros(3, NGRAM_DIM, dtype=torch.uint8)
    assert cache.take_hits([10, 11, 12], [0, 1, 2], out) == ([10], [0])
    assert torch.equal(out[1], src[1])
    assert torch.equal(out[2], src[2])


def test_row_cache_never_holds_more_than_capacity_rows():
    """A miss batch larger than the whole cache reuses slots inside one insert."""
    cache = _PleRowCache(3, NGRAM_DIM)
    src = _rows(10)
    cache.insert(list(range(10)), list(range(10)), src)
    assert len(cache._slot_of) == 3
    assert sorted(cache._slot_of.values()) == [0, 1, 2]  # no slot claimed twice
    for row_id, slot in cache._slot_of.items():
        assert torch.equal(cache._slab[slot], src[row_id])  # every survivor is its own row


def test_row_cache_inserts_a_repeated_id_once():
    cache = _PleRowCache(4, NGRAM_DIM)
    src = _rows(3)
    cache.insert([5, 5, 6], [0, 1, 2], src)
    assert len(cache._slot_of) == 2
    assert torch.equal(cache._slab[cache._slot_of[5]], src[0])  # the first copy wins
    assert torch.equal(cache._slab[cache._slot_of[6]], src[2])


def test_row_cache_capacity_of_one_still_works():
    cache = _PleRowCache(1, NGRAM_DIM)
    src = _rows(2)
    cache.insert([4], [0], src)
    cache.insert([9], [1], src)
    out = torch.zeros(2, NGRAM_DIM, dtype=torch.uint8)
    assert cache.take_hits([4, 9], [0, 1], out) == ([4], [0])
    assert torch.equal(out[1], src[1])


@pytest.mark.parametrize("prefetch", [False, True])
def test_gathered_bytes_are_identical_at_every_cache_size(
    checkpoint, monkeypatch, request, prefetch
):
    """Same id stream through no cache, a huge cache and a thrashing one."""
    if prefetch:
        request.getfixturevalue("prefetch_enabled")
    folder, raw = checkpoint
    reference = _ngram_reference(raw)
    rng = random.Random(11)
    streams = [[rng.randrange(-2, ALL_ROWS + 2) for _ in range(9)] for _ in range(24)]
    got: dict[str, list[torch.Tensor]] = {}
    for capacity in ("0", "1048576", "3"):
        monkeypatch.setenv("FREETOKEN_PLE_ROW_CACHE", capacity)
        table = _open_mmap_ple(folder)
        try:
            got[capacity] = [_gather(table.storage, ids).clone() for ids in streams]
        finally:
            table.storage.close()
    for index, ids in enumerate(streams):
        want = _serial_gather(reference, ids)
        for capacity in got:
            assert torch.equal(got[capacity][index], want), (capacity, ids)


# ======================================================================================
# read_range_into: the O_DIRECT byte-range read the PLE table load is built on
# ======================================================================================


@pytest.fixture(scope="module")
def blob(tmp_path_factory) -> tuple[str, bytes]:
    data = random.Random(7).randbytes(5_000_003)
    path = tmp_path_factory.mktemp("blob") / "data.bin"
    path.write_bytes(data)
    return str(path), data


@pytest.mark.parametrize("file_offset, nbytes, dest_offset", [
    (1, 4095, 0),                 # sub-block, unaligned source
    (2239, 1_000_000, 0),         # the real checkpoint's header-end phase
    (4095, 4097, 1),              # straddles two block boundaries
    (4_999_000, 1003, 123_456),   # runs to EOF
])
def test_read_range_into_matches_the_file(blob, file_offset, nbytes, dest_offset):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    got = read_range_into(view, path, file_offset=file_offset, nbytes=nbytes,
                          dest_offset=dest_offset, chunk=1 << 20)
    assert got == nbytes
    assert bytes(view[dest_offset:dest_offset + nbytes]) == data[file_offset:file_offset + nbytes]


def test_read_range_into_is_chunk_and_thread_safe(blob):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    read_range_into(view, path, file_offset=2239, nbytes=4_000_000, dest_offset=1024,
                    workers=8, chunk=64 << 10)
    assert bytes(view[1024:1024 + 4_000_000]) == data[2239:2239 + 4_000_000]


def test_read_range_into_rejects_a_short_destination(blob):
    path, _data = blob
    bank = HostBank((1024,), torch.uint8)
    with pytest.raises(ValueError, match="destination holds"):
        read_range_into(bank.memoryview(), path, file_offset=0, nbytes=1 << 20)


# ======================================================================================
# AOT shape table
# ======================================================================================


def test_aot_entry_carries_the_checkpoint_geometry():
    entry = next(m for m in SUPPORTED_MODELS
                 if m.architecture == "Qwen4ExpForConditionalGeneration")
    assert (entry.hidden_size, entry.moe_intermediate_size, entry.top_k) == (2560, 640, 10)
    assert entry.kv_groups == ((2, 256),)
    rows = expert_bank_row_bytes("nvfp4", entry.hidden_size, entry.moe_intermediate_size)
    assert set(rows) == {"gate_up_packed", "gate_up_scale", "gate_up_global",
                         "down_packed", "down_scale", "down_global"}
    for name, nbytes in rows.items():
        assert nbytes % 16 == 0, name  # fused multi-bank copy only engages on 16B multiples


def test_every_registry_architecture_is_claimed_by_an_aot_entry():
    from freetoken.models.register import _MODEL_REGISTRY

    claimed = {m.architecture for m in SUPPORTED_MODELS}
    claimed |= {a for m in SUPPORTED_MODELS for a in m.arch_aliases}
    assert "Qwen4ExpForConditionalGeneration" in claimed
    assert set(_MODEL_REGISTRY) - claimed == set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_fusion_pad_rides_the_tensor_device():
    """safetensors loads straight to cuda; a cpu-allocated pad row would break torch.cat."""
    from freetoken.models.qwen4_exp.weight import _try_fuse

    buf = {}
    down = torch.randn(320, 64, device="cuda", dtype=torch.bfloat16)
    inject = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    assert _try_fuse("model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", down, buf) == ()
    key, fused = _try_fuse("model.layers.0.attn_hyper_connection.block_inject_weight.weight", inject, buf)
    assert fused.device.type == "cuda" and fused.shape[0] == 336
    assert torch.equal(fused[324:], torch.zeros(12, 64, device="cuda", dtype=torch.bfloat16))


# ======================================================================================
# The POSIX madvise(MADV_WILLNEED) path
# ======================================================================================
#
# Off Windows there is no PrefetchVirtualMemory. The equivalent -- ask the kernel to fault a
# set of ranges in one go, then copy serially -- is madvise(WILLNEED) on each row's page
# range. Unmeasured on a real table (this branch has only ever run on Windows), so what is
# pinned here is the contract: page-aligned advice covering every requested row, then the
# same bytes as the serial reference, and a clean fall-through when the mapping cannot
# advise.


class _AdvisingMap:
    """Stands in for an ``mmap.mmap``: records ``madvise`` calls, never faults anything."""

    def __init__(self, calls: list[tuple[int, int, int]]) -> None:
        self.calls = calls

    def madvise(self, option: int, start: int, length: int) -> None:
        self.calls.append((int(option), int(start), int(length)))


class _PlainMap:
    """A mapping without ``madvise`` at all (what Windows' ``mmap.mmap`` looks like)."""


@pytest.fixture
def posix_advice(monkeypatch):
    """No Windows prefetch symbol, but the platform mmap knows MADV_WILLNEED."""
    monkeypatch.setattr(weight_mod, "_prefetch_virtual_memory", None)
    monkeypatch.setattr(weight_mod, "_prefetch_failed", False)
    monkeypatch.setattr(weight_mod.mmap, "MADV_WILLNEED", 3, raising=False)
    monkeypatch.setattr(weight_mod.mmap, "PAGESIZE", 4096, raising=False)


@pytest.mark.parametrize("cache", ["0", "1048576"])
def test_willneed_advice_covers_every_row_and_the_gather_still_matches(
    checkpoint, monkeypatch, posix_advice, cache
):
    monkeypatch.setenv(weight_mod._PLE_ROW_CACHE_ENV, cache)
    folder, raw = checkpoint
    reference = _ngram_reference(raw)
    table = _open_mmap_ple(folder)
    storage = table.storage
    calls: list[tuple[int, int, int]] = []
    try:
        bases = [base for _mapping, base in storage._shard_maps]
        storage._shard_maps = [(_AdvisingMap(calls), base) for base in bases]
        ids = [0, NGRAM_ROWS - 1, NGRAM_ROWS, 2 * NGRAM_ROWS + 3, ALL_ROWS - 1, 5, 5, -1, ALL_ROWS]
        assert torch.equal(_gather(storage, ids), _serial_gather(reference, ids))
        valid = [row for row in ids if 0 <= row < ALL_ROWS]
        assert len(calls) == len(valid)
        for row, (option, start, length) in zip(valid, calls):
            assert option == 3
            assert start % 4096 == 0
            row_start = bases[row // NGRAM_ROWS] + (row % NGRAM_ROWS) * NGRAM_DIM
            assert start <= row_start
            assert start + length >= row_start + NGRAM_DIM
        # a second gather of the same ids: with the cache on, nothing is advised again
        calls.clear()
        assert torch.equal(_gather(storage, ids), _serial_gather(reference, ids))
        assert len(calls) == (0 if cache != "0" else len(valid))
    finally:
        storage.close()


def test_a_mapping_without_madvise_falls_through_to_the_fan_out(
    checkpoint, monkeypatch, posix_advice
):
    monkeypatch.setenv(weight_mod._PLE_ROW_CACHE_ENV, "0")
    folder, raw = checkpoint
    reference = _ngram_reference(raw)
    table = _open_mmap_ple(folder)
    storage = table.storage
    try:
        storage._shard_maps = [(_PlainMap(), base) for _m, base in storage._shard_maps]
        ids = list(range(ALL_ROWS))
        assert storage._advise_rows(torch.tensor(ids[:4], dtype=torch.int64)) is False
        assert torch.equal(_gather(storage, ids), _serial_gather(reference, ids))
    finally:
        storage.close()


def test_a_single_row_is_not_worth_an_advice_syscall(checkpoint, monkeypatch, posix_advice):
    """Mirrors ``_PLE_MIN_PREFETCH_ROWS``: below two rows the syscall costs more than the
    fault it hides, so one row takes the plain copy."""
    monkeypatch.setenv(weight_mod._PLE_ROW_CACHE_ENV, "0")
    folder, raw = checkpoint
    table = _open_mmap_ple(folder)
    storage = table.storage
    calls: list[tuple[int, int, int]] = []
    try:
        storage._shard_maps = [(_AdvisingMap(calls), base) for _m, base in storage._shard_maps]
        assert torch.equal(_gather(storage, [3]), _serial_gather(_ngram_reference(raw), [3]))
        assert calls == []
    finally:
        storage.close()


def test_the_windows_prefetch_takes_precedence_over_advice(
    checkpoint, monkeypatch, prefetch_enabled
):
    """Where both exist (never in practice, but the order must be deliberate) the single
    PrefetchVirtualMemory call wins and no per-row madvise is issued."""
    monkeypatch.setattr(weight_mod.mmap, "MADV_WILLNEED", 3, raising=False)
    monkeypatch.setenv(weight_mod._PLE_ROW_CACHE_ENV, "0")
    folder, _raw = checkpoint
    table = _open_mmap_ple(folder)
    storage = table.storage
    calls: list[tuple[int, int, int]] = []
    try:
        storage._shard_maps = [(_AdvisingMap(calls), base) for _m, base in storage._shard_maps]
        _gather(storage, [1, 2, 3])
        assert prefetch_enabled == [(3, 0)]
        assert calls == []
    finally:
        storage.close()


# ======================================================================================
# Picture weights served from the mapped shard extent (FREETOKEN_VISION_WEIGHTS=mmap)
# ======================================================================================
#
# In ``mmap`` mode the 333 picture tensors are not read into process RAM at boot. Their one
# contiguous extent of the bf16 shard is mapped copy-on-write and each tensor becomes a
# zero-copy ``torch.frombuffer`` view installed exactly where the resident tensor used to
# go. What is pinned here: the extent comes out of the header (never a constant), the views
# carry the checkpoint's bytes, they live inside the mapping, ``pos_embed`` deliberately
# does not, one prefetch covers each window, and every failure lands back on ``ram``.

PW = 16  # picture hidden size, tiny


def _picture_tensors() -> dict[str, torch.Tensor]:
    """One of each real component class: patch projection, two blocks, merger, pos_embed."""
    return {
        "model.visual.patch_embed.proj.weight": _bf16(PW, 3, 1, 2, 2),
        "model.visual.patch_embed.proj.bias": _bf16(PW),
        "model.visual.pos_embed.weight": _bf16(9, PW),
        "model.visual.blocks.0.norm1.weight": _bf16(PW),
        "model.visual.blocks.0.attn.qkv.weight": _bf16(3 * PW, PW),
        "model.visual.blocks.0.attn.qkv.bias": _bf16(3 * PW),
        "model.visual.blocks.1.attn.qkv.weight": _bf16(3 * PW, PW),
        "model.visual.merger.linear_fc1.weight": _bf16(PW, PW),
        "model.visual.merger.norm.weight": _bf16(PW),
    }


def _write_shard(
    path: str,
    tensors: dict[str, torch.Tensor],
    *,
    dtype_names: dict[str, str] | None = None,
) -> int:
    """Serialize a safetensors shard by hand and return its data base offset.

    ``safetensors.torch.save_file`` pads its header to a multiple of 8, so it can only ever
    produce an EVEN data base. The real Qwen3.8 shard's header is 47,581 B and its base is
    47,589 -- odd -- which is exactly why every bf16 picture tensor is 2-byte misaligned and
    why the mapped views must be treated as copy sources only. Build that layout here rather
    than test a friendlier one than production has. ``dtype_names`` overrides the header
    dtype string, so a dtype the reader cannot map can be fabricated.
    """
    blobs: list[bytes] = []
    header: dict[str, dict] = {}
    offset = 0
    for name, tensor in tensors.items():
        raw = bytes(tensor.contiguous().flatten().view(torch.uint8).numpy())
        header[name] = {
            "dtype": (dtype_names or {}).get(name, _ST_NAME[tensor.dtype]),
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    for pad in (1, 2):
        body = json.dumps({**header, "__metadata__": {"pad": "x" * pad}}).encode("utf-8")
        if (8 + len(body)) % 2 == 1:
            break
    else:  # pragma: no cover - one extra header byte always flips the parity
        raise AssertionError("could not pad the header to an odd data base")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(body)))
        fh.write(body)
        for raw in blobs:
            fh.write(raw)
    return 8 + len(body)


@pytest.fixture(scope="module")
def picture_checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor], str, int]:
    """``(folder, raw picture tensors, the shard holding them, its odd data base)``.

    Two bf16 shards and an index, so the layout has to consult the index and map exactly the
    one shard that carries ``model.visual.*``.
    """
    torch.manual_seed(3)
    folder = tmp_path_factory.mktemp("qwen4_exp_picture_ckpt")
    picture = _picture_tensors()
    text = {
        "model.language_model.embed_tokens.weight": _bf16(11, H),
        "lm_head.weight": _bf16(11, H),
    }
    shard = str(folder / "model-bf16-00001.safetensors")
    base = _write_shard(shard, {**text, **picture})
    other_name = "model.language_model.layers.0.mlp.gate.weight"
    save_file({other_name: _bf16(E, H)}, str(folder / "model-bf16-00002.safetensors"))
    weight_map = {name: "model-bf16-00001.safetensors" for name in {**text, **picture}}
    weight_map[other_name] = "model-bf16-00002.safetensors"
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    return str(folder), picture, shard, base


def _header_ranges(shard: str) -> dict[str, tuple[int, int, list[int]]]:
    """``raw name -> (absolute offset, bytes, shape)`` read straight out of the JSON header."""
    header, base = weight_mod._safetensors_header(shard)
    ranges = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        ranges[key] = (base + begin, end - begin, meta["shape"])
    return ranges


def test_picture_layout_offsets_come_from_the_header(picture_checkpoint):
    folder, picture, shard, base = picture_checkpoint
    assert base % 2 == 1, "the interesting layout is the misaligned one"
    ranges = _header_ranges(shard)

    layout = weight_mod._vision_layout(folder)

    assert [s.path for s in layout.shards] == [shard], "only the shard with picture tensors"
    specs = {spec.name: spec for spec in layout.shards[0].tensors}
    assert set(specs) == {name.replace("model.visual.", "visual.") for name in picture}
    for raw_name, tensor in picture.items():
        spec = specs[raw_name.replace("model.visual.", "visual.")]
        offset, nbytes, shape = ranges[raw_name]
        assert (spec.offset, spec.nbytes, list(spec.shape)) == (offset, nbytes, shape)
        assert spec.offset % 2 == 1, "bf16 at an odd file offset, like the real checkpoint"
        assert spec.dtype is tensor.dtype
    assert layout.nbytes == sum(t.numel() * t.element_size() for t in picture.values())


def test_picture_layout_window_covers_the_extent_and_nothing_is_hard_coded(picture_checkpoint):
    folder, _picture, _shard, _base = picture_checkpoint
    shard_layout = weight_mod._vision_layout(folder).shards[0]

    offsets = [spec.offset for spec in shard_layout.tensors]
    ends = [spec.offset + spec.nbytes for spec in shard_layout.tensors]
    assert shard_layout.start == min(offsets)
    assert shard_layout.end == max(ends)
    assert shard_layout.span == shard_layout.end - shard_layout.start


def test_picture_layout_finds_the_shard_without_an_index(checkpoint):
    """No ``model.safetensors.index.json``: scan the shards instead. The stock fixture also
    splits its two picture tensors across both bf16 shards, so both get mapped."""
    folder, _raw = checkpoint
    layout = weight_mod._vision_layout(folder)
    assert len(layout.shards) == 2
    assert {spec.name for shard in layout.shards for spec in shard.tensors} == {
        "visual.blocks.0.attn.qkv.weight",
        "visual.merger.norm.weight",
    }


def test_picture_layout_without_picture_tensors_is_fatal(tmp_path):
    save_file({"lm_head.weight": _bf16(4, H)}, str(tmp_path / "model-bf16-00001.safetensors"))
    with pytest.raises(ValueError, match="no picture tensors"):
        weight_mod._vision_layout(str(tmp_path))


def test_picture_layout_rejects_a_mixed_dtype_extent(tmp_path):
    save_file(
        {
            "model.visual.blocks.0.attn.qkv.weight": _bf16(PW, PW),
            "model.visual.merger.norm.weight": torch.randn(PW),  # float32 among bf16
        },
        str(tmp_path / "model-bf16-00001.safetensors"),
    )
    with pytest.raises(ValueError, match="visual.merger.norm.weight"):
        weight_mod._vision_layout(str(tmp_path))


def test_picture_layout_rejects_a_dtype_it_cannot_view(tmp_path):
    tensors = {"model.visual.merger.norm.weight": torch.zeros(PW, dtype=torch.bfloat16)}
    _write_shard(
        str(tmp_path / "model-bf16-00001.safetensors"),
        tensors,
        dtype_names={"model.visual.merger.norm.weight": "U16"},
    )
    with pytest.raises(ValueError, match="unsupported dtype"):
        weight_mod._vision_layout(str(tmp_path))


@pytest.fixture
def picture_source(picture_checkpoint):
    """A mapped holder over the picture checkpoint, closed afterwards."""
    folder, picture, shard, _base = picture_checkpoint
    source = weight_mod.MmapVisionWeights(weight_mod._vision_layout(folder))
    try:
        yield source, picture, shard
    finally:
        source.close()


def test_mapped_views_are_byte_identical_to_the_checkpoint(picture_source):
    source, picture, shard = picture_source
    with safetensors.safe_open(shard, framework="pt", device="cpu") as handle:
        for raw_name, expected in picture.items():
            name = raw_name.replace("model.visual.", "visual.")
            view = source.tensor(name)
            reference = handle.get_tensor(raw_name)
            assert view.shape == expected.shape and view.dtype is expected.dtype, name
            # bf16 through int16 so a NaN payload still compares as bytes
            assert torch.equal(view.view(torch.int16), reference.view(torch.int16)), name


def test_mapped_views_live_inside_the_mapping_and_pos_embed_does_not(picture_source):
    source, picture, _shard = picture_source
    for raw_name in picture:
        name = raw_name.replace("model.visual.", "visual.")
        view = source.tensor(name)
        if name == "visual.pos_embed.weight":
            # CPU ``F.embedding`` operand, not a memcpy source: kept resident and aligned.
            assert not source.contains(view.data_ptr())
            assert view.data_ptr() % 2 == 0
        else:
            assert source.contains(view.data_ptr()), name


def test_mapped_bytes_exclude_the_resident_carve_out(picture_source):
    source, picture, _shard = picture_source
    pos_embed = picture["model.visual.pos_embed.weight"]
    assert source.resident_names == frozenset({"visual.pos_embed.weight"})
    assert source.mapped_bytes == source.layout.nbytes - pos_embed.numel() * 2


def test_mapped_windows_start_on_an_allocation_granularity_boundary(picture_source):
    """``mmap`` requires the FILE offset to be a multiple of the allocation granularity, so
    the window starts below the extent and the views carry the difference."""
    source, _picture, _shard = picture_source
    for window in source.windows:
        assert window.file_offset % mmap.ALLOCATIONGRANULARITY == 0
        assert window.span > 0
    shard_layout = source.layout.shards[0]
    assert source.windows[0].file_offset <= shard_layout.start
    assert source.windows[0].span >= shard_layout.span


def test_a_repeated_lookup_returns_the_same_view(picture_source):
    source, _picture, _shard = picture_source
    first = source.tensor("visual.merger.norm.weight")
    assert source.tensor("visual.merger.norm.weight").data_ptr() == first.data_ptr()


def test_an_unknown_picture_key_is_a_named_error(picture_source):
    source, _picture, _shard = picture_source
    with pytest.raises(KeyError, match="visual.blocks.9.attn.qkv.weight"):
        source.tensor("visual.blocks.9.attn.qkv.weight")


@pytest.fixture
def picture_prefetch_enabled(monkeypatch):
    """Force the Windows prefetch on with a stub that records ``(entries, count)``."""
    calls: list[tuple[list[tuple[int, int]], int]] = []

    def fake(_process, count, entries, flags):
        assert flags == 0
        array = (ctypes.c_size_t * (2 * count)).from_address(entries.value)
        calls.append(([(array[2 * i], array[2 * i + 1]) for i in range(count)], count))
        return 1

    monkeypatch.setattr(weight_mod, "_prefetch_virtual_memory", fake)
    monkeypatch.setattr(weight_mod, "_current_process", ctypes.c_void_p(0))
    monkeypatch.setattr(weight_mod, "_vision_prefetch_failed", False)
    return calls


def test_prefetch_asks_for_every_mapped_window_in_one_call(
    picture_source, picture_prefetch_enabled
):
    source, _picture, _shard = picture_source

    assert source.prefetch() is True
    assert len(picture_prefetch_enabled) == 1
    entries, count = picture_prefetch_enabled[0]
    assert count == len(source.windows)
    assert [span for _address, span in entries] == [w.span for w in source.windows]
    for address, span in entries:
        assert source.contains(address) and source.contains(address + span - 1)


def test_prefetch_is_not_reissued_until_the_encode_releases_it(
    picture_source, picture_prefetch_enabled
):
    """The scheduler prefetches at admission and the streamed encode prefetches defensively
    at entry; the second call must not pay the syscall again for the same picture."""
    source, _picture, _shard = picture_source

    assert source.prefetch() is True
    assert source.prefetch() is True
    assert len(picture_prefetch_enabled) == 1

    source.release_prefetch()
    assert source.prefetch() is True
    assert len(picture_prefetch_enabled) == 2


def test_a_false_prefetch_return_warns_once_and_stops_trying(
    picture_source, monkeypatch, caplog
):
    source, _picture, _shard = picture_source
    calls = []

    def fake(_process, count, _entries, _flags):
        calls.append(count)
        return 0

    monkeypatch.setattr(weight_mod, "_prefetch_virtual_memory", fake)
    monkeypatch.setattr(weight_mod, "_current_process", ctypes.c_void_p(0))
    monkeypatch.setattr(weight_mod, "_vision_prefetch_failed", False)

    with caplog.at_level("WARNING", logger=weight_mod.logger.name):
        assert source.prefetch() is False
        source.release_prefetch()
        assert source.prefetch() is False

    assert len(calls) == 1, "a documented FALSE is not fixed by retrying"
    assert sum("PrefetchVirtualMemory" in r.message for r in caplog.records) == 1
    assert weight_mod._vision_prefetch_failed is True


def test_prefetch_falls_back_to_madvise_willneed_off_windows(picture_source, monkeypatch):
    """No PrefetchVirtualMemory: one ``madvise(MADV_WILLNEED)`` per window, not per row --
    the picture extent is one sequential run, unlike the PLE table's random rows."""
    source, _picture, shard = picture_source
    advice: list[tuple[int, int, int]] = []

    class _Advising:
        def madvise(self, option, start, length):
            advice.append((int(option), int(start), int(length)))

    monkeypatch.setattr(weight_mod, "_prefetch_virtual_memory", None)
    monkeypatch.setattr(weight_mod.mmap, "MADV_WILLNEED", 3, raising=False)
    monkeypatch.setattr(weight_mod, "_vision_advise_failed", False)
    monkeypatch.setitem(source._maps, shard, _Advising())

    assert source.prefetch() is True
    assert advice == [(3, 0, source.windows[0].span)]


def test_prefetch_reports_failure_when_neither_mechanism_exists(picture_source, monkeypatch):
    source, _picture, _shard = picture_source
    monkeypatch.setattr(weight_mod, "_prefetch_virtual_memory", None)
    monkeypatch.delattr(weight_mod.mmap, "MADV_WILLNEED", raising=False)

    assert source.prefetch() is False


def test_close_releases_every_map_and_file(picture_checkpoint):
    folder, _picture, _shard, _base = picture_checkpoint
    source = weight_mod.MmapVisionWeights(weight_mod._vision_layout(folder))
    mappings = list(source._maps.values())
    files = list(source._files.values())

    source.close()

    assert all(mapping.closed for mapping in mappings)
    assert all(handle.closed for handle in files)
    assert source.close() is None, "close is idempotent"


# --------------------------------------------------------------------------------------
# iter_weights installs the views where the resident tensors used to go
# --------------------------------------------------------------------------------------


def _read_picture_weights(folder: str, backing: str, monkeypatch) -> dict[str, torch.Tensor]:
    # The mapping is process-scoped and cached per checkpoint folder, so start from a clean
    # registry: these tests assert on whether THIS load mapped anything.
    weight_mod.close_mmap_vision_weights()
    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    monkeypatch.setenv("FREETOKEN_VISION_EXECUTION", "layer-stream")
    monkeypatch.setenv("FREETOKEN_VISION_WEIGHTS", backing)
    return dict(
        iter_weights(
            folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True
        )
    )


def test_iter_weights_installs_mapped_views_for_the_picture_keys(
    picture_checkpoint, monkeypatch
):
    folder, picture, _shard, _base = picture_checkpoint
    try:
        values = _read_picture_weights(folder, "mmap", monkeypatch)
        source = weight_mod.mmap_vision_weights(folder)
        assert source is not None
        for raw_name in picture:
            name = raw_name.replace("model.visual.", "visual.")
            inside = source.contains(values[name].data_ptr())
            assert inside is (name != "visual.pos_embed.weight"), name
        assert values["lm_head.weight"].device.type == "cpu"
        assert not source.contains(values["lm_head.weight"].data_ptr())
    finally:
        weight_mod.close_mmap_vision_weights()


def test_mapped_and_resident_picture_weights_are_the_same_bytes(
    picture_checkpoint, monkeypatch
):
    """Only where the bytes live changes. Anything else is a layout bug."""
    folder, picture, _shard, _base = picture_checkpoint
    try:
        resident = _read_picture_weights(folder, "ram", monkeypatch)
        mapped = _read_picture_weights(folder, "mmap", monkeypatch)
        assert set(resident) == set(mapped)
        for raw_name in picture:
            name = raw_name.replace("model.visual.", "visual.")
            assert torch.equal(
                mapped[name].view(torch.int16), resident[name].view(torch.int16)
            ), name
    finally:
        weight_mod.close_mmap_vision_weights()


def test_resident_mode_never_opens_a_mapping(picture_checkpoint, monkeypatch):
    folder, _picture, _shard, _base = picture_checkpoint
    monkeypatch.setattr(
        weight_mod, "_vision_layout", lambda *a, **k: pytest.fail("ram must not map")
    )
    values = _read_picture_weights(folder, "ram", monkeypatch)
    assert weight_mod.mmap_vision_weights(folder) is None
    assert values["visual.merger.norm.weight"].device.type == "cpu"


def test_a_mapping_failure_falls_back_to_resident_ram_with_one_warning(
    picture_checkpoint, monkeypatch, caplog
):
    folder, picture, shard, _base = picture_checkpoint
    monkeypatch.setattr(weight_mod, "_vision_mmap_warned", False)

    def refuse(*_args, **_kwargs):
        raise OSError(8, "not enough address space")

    monkeypatch.setattr(weight_mod.mmap, "mmap", refuse)
    with caplog.at_level("WARNING", logger=weight_mod.logger.name):
        first = _read_picture_weights(folder, "mmap", monkeypatch)
        second = _read_picture_weights(folder, "mmap", monkeypatch)

    assert weight_mod.mmap_vision_weights(folder) is None
    warnings = [r for r in caplog.records if "picture weights" in r.message]
    assert len(warnings) == 1, "one warning for the process, not one per boot attempt"
    with safetensors.safe_open(shard, framework="pt", device="cpu") as handle:
        for raw_name in picture:
            name = raw_name.replace("model.visual.", "visual.")
            reference = handle.get_tensor(raw_name).view(torch.int16)
            assert torch.equal(first[name].view(torch.int16), reference), name
            assert torch.equal(second[name].view(torch.int16), reference), name
