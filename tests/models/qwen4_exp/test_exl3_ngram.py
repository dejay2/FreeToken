"""EXL3 n-gram table decode + conversion (spec 2026-09-25 section 4)."""

from __future__ import annotations

import json
import struct

import pytest
import safetensors.torch
import torch

from freetoken.models.qwen4_exp.exl3_ngram import (
    ROW_DIM, dequant_rows, head_of_rows, mul1_codebook, pack_rows, words_per_row,
)


def _rows(n: int, k: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    states = torch.randint(0, 1 << 16, (n, ROW_DIM), generator=g, dtype=torch.int64)
    # A tail-biting ring constrains neighbouring states; build legal states by packing then
    # unpacking, so the reference below compares like with like.
    scales = (torch.rand(n, generator=g) + 0.5).to(torch.float16)
    return pack_rows(states, scales, k)


def test_words_per_row_matches_checkpoint():
    assert words_per_row(5) == 51  # 3.05bpw_h5_ng5 header: shard trellis [2500012, 51]


def test_codebook_is_bit_exact_fp16():
    cb = mul1_codebook("cpu")
    assert cb.dtype == torch.float16 and cb.shape == (65536,)
    # decode_3inst<2> of state 0: bsum 0 -> (1024 * 0x1eee) + 0xc931 in fp16
    k_inv = torch.tensor([0x1EEE], dtype=torch.uint16).view(torch.float16).float()
    k_bias = torch.tensor([0xC931], dtype=torch.uint16).view(torch.float16).float()
    assert cb[0].float() == (1024.0 * k_inv + k_bias).half().float()


def test_pack_unpack_round_trip_values():
    packed = _rows(64, 5)
    cb = mul1_codebook("cpu")
    once = dequant_rows(packed, 5, cb)
    again = dequant_rows(pack_rows(*_unpack(packed, 5), 5), 5, cb)
    assert torch.equal(once, again)


def _unpack(packed, k):
    from freetoken.models.qwen4_exp.exl3_ngram import unpack_rows
    return unpack_rows(packed, k)


def test_head_boundary_rows():
    offsets = torch.tensor([0, 10, 25, 40])
    rows = torch.tensor([0, 9, 10, 24, 25, 39, 40, 41])
    assert head_of_rows(rows, offsets).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def _write_exl3_table(path, *, shards=4, rows_per_shard=8, k=5, heads=2, layer=1):
    prefix = f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"
    tensors = {}
    for s in range(shards):
        tensors[f"{prefix}.shard_{s}.trellis"] = _rows(rows_per_shard, k, seed=s)
    total = shards * rows_per_shard
    tensors[f"{prefix}.head_bias"] = torch.randn(heads, ROW_DIM).half()
    tensors[f"{prefix}.head_offsets"] = torch.tensor([0, total // 2])
    tensors[f"{prefix}.head_vocab_sizes"] = torch.tensor([total // 2, total - total // 2])
    tensors[f"{prefix}.layer_multipliers"] = torch.tensor([1, 3, 5])
    meta = {"format": "exl3_ngram_trellis", "version": "1", "K": str(k), "codebook": "mul1",
            "row_dim": str(ROW_DIM), "rows": str(total), "shard_rows": str(rows_per_shard)}
    safetensors.torch.save_file(tensors, str(path), metadata=meta)
    return tensors, prefix


def test_convert_writes_freetoken_layout(tmp_path):
    from scripts.exl3.convert_ngram_table import convert_table  # noqa: E402

    tensors, prefix = _write_exl3_table(tmp_path / "ngram_embedding.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3"}}))
    report = convert_table(str(tmp_path), device="cpu", chunk_rows=5, out_dtype="fp8")
    index = json.loads((tmp_path / "freetoken-ple.index.json").read_text())["weight_map"]
    shard_names = sorted(n for n in index if ".ngram_embedding.shard_" in n)
    assert len(shard_names) == 4 and all(n.endswith(".weight") for n in shard_names)
    assert f"{prefix}.weight_scale" in index
    # hash constants are renamed to FreeToken's NGramEmbedding state keys
    base = prefix.rsplit(".ngram_embedding", 1)[0]
    for key in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
        assert f"{base}.{key}" in index
    # decoded values agree with the fp32 reference within fp8 resolution
    cb = mul1_codebook("cpu")
    got_file = tmp_path / index[f"{prefix}.shard_1.weight"]
    got = safetensors.torch.load_file(str(got_file))
    scale = got[f"{prefix}.weight_scale"].float() if f"{prefix}.weight_scale" in got else None
    if scale is None:
        scale = safetensors.torch.load_file(str(tmp_path / index[f"{prefix}.weight_scale"]))[
            f"{prefix}.weight_scale"].float()
    rows = torch.arange(8, 16)
    heads = head_of_rows(rows, tensors[f"{prefix}.head_offsets"])
    ref = dequant_rows(tensors[f"{prefix}.shard_1.trellis"], 5, cb,
                       tensors[f"{prefix}.head_bias"][heads])
    approx = got[f"{prefix}.shard_1.weight"].float() * scale
    rel = ((approx - ref).pow(2).mean(1).sqrt() / ref.pow(2).mean(1).sqrt())
    assert rel.median() < 0.05
    assert report["median_rel_rms"] < 0.05 and report["dtype"] == "F8_E4M3"


def test_convert_refuses_mismatched_hash_constants(tmp_path):
    from scripts.exl3.convert_ngram_table import convert_table

    tensors, prefix = _write_exl3_table(tmp_path / "ngram_embedding.safetensors")
    tensors[f"{prefix}.head_offsets"] = torch.tensor([0, 3])  # not what head_vocab_sizes imply
    safetensors.torch.save_file(tensors, str(tmp_path / "ngram_embedding.safetensors"),
                                metadata={"format": "exl3_ngram_trellis", "version": "1", "K": "5",
                                          "row_dim": "160", "rows": "32", "shard_rows": "8"})
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="head_offsets"):
        convert_table(str(tmp_path), device="cpu", chunk_rows=5, out_dtype="fp8")


def test_ple_table_files_prefers_sidecar(tmp_path):
    from freetoken.models.qwen4_exp.weight import _ple_table_files

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (tmp_path / "freetoken-ple.index.json").write_text(json.dumps({"weight_map": {
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": "a.safetensors"}}))
    assert _ple_table_files(str(tmp_path)) == [str(tmp_path / "a.safetensors")]


def test_unconverted_exl3_table_names_the_converter(tmp_path):
    from types import SimpleNamespace
    from freetoken.models.qwen4_exp.weight import _ple_layout

    (tmp_path / "ngram_embedding.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="convert_ngram_table.py"):
        _ple_layout(str(tmp_path), SimpleNamespace(split_ngram_parts=128, ngram_head_dim=160))
