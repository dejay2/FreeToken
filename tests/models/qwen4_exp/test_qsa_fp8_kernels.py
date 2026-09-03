"""FP8 QSA store, dequantized sparse attention, and graph replay."""

from __future__ import annotations

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config

QSA_LAYER = 3
PAGE_SIZE = 64


def _relative_l2(got: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(got.float() - expected.float()) /
                 torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))


@requires_cuda
def test_fp8_store_quantizes_each_token_head_and_scatters_its_scale():
    fixture = Fixture(
        parsed_config(), num_pages=8, kv_dtype=torch.float8_e4m3fn
    )
    pool = fixture.pool
    rows, heads, dim = 23, 2, 256
    generator = torch.Generator(device="cuda").manual_seed(41)
    k = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16,
                    generator=generator).flatten(1)
    v = (torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16,
                     generator=generator) * 0.6).flatten(1)
    out_loc = torch.randperm(8 * PAGE_SIZE, device="cuda")[:rows].to(torch.int32)
    index_before = pool.cmp_k_cache(0).clone()

    pool.store_kv(k, v, out_loc, QSA_LAYER)

    assert pool.k_cache(QSA_LAYER).shape == (9, PAGE_SIZE, heads, dim)
    assert pool.v_cache(QSA_LAYER).shape == (9, PAGE_SIZE, heads, dim)
    assert pool.k_scale(QSA_LAYER).shape == (9, PAGE_SIZE, heads)
    assert pool.v_scale(QSA_LAYER).shape == (9, PAGE_SIZE, heads)
    flat_k = pool.k_cache(QSA_LAYER).view(-1, heads, dim)
    flat_v = pool.v_cache(QSA_LAYER).view(-1, heads, dim)
    flat_k_scale = pool.k_scale(QSA_LAYER).view(-1, heads)
    flat_v_scale = pool.v_scale(QSA_LAYER).view(-1, heads)
    k_scale = flat_k_scale.index_select(0, out_loc.long())
    v_scale = flat_v_scale.index_select(0, out_loc.long())
    got_k = flat_k.index_select(0, out_loc.long()).float() * k_scale[..., None]
    got_v = flat_v.index_select(0, out_loc.long()).float() * v_scale[..., None]
    expected_k = k.view(rows, heads, dim).float()
    expected_v = v.view(rows, heads, dim).float()

    per_head_k = torch.linalg.vector_norm(got_k - expected_k, dim=-1) / torch.linalg.vector_norm(
        expected_k, dim=-1
    ).clamp_min(1e-12)
    per_head_v = torch.linalg.vector_norm(got_v - expected_v, dim=-1) / torch.linalg.vector_norm(
        expected_v, dim=-1
    ).clamp_min(1e-12)
    aggregate_k = _relative_l2(got_k, expected_k)
    aggregate_v = _relative_l2(got_v, expected_v)
    # docs/research/fp8-kv-accuracy-2026-09-03.md measured 3.5-3.8% worst-head error;
    # E4M3 arithmetic predicts ~2.4% aggregate. The 1-4% band catches a raw-copy/no-quantization path at the low end
    # and a wrong scale/layout at the high end; 8% still catches a bad individual token/head.
    assert 0.01 < aggregate_k < 0.04
    assert 0.01 < aggregate_v < 0.04
    assert float(per_head_k.max()) < 0.08
    assert float(per_head_v.max()) < 0.08
    expected_k_scale = expected_k.abs().amax(-1).clamp_min(1e-10) / 448.0
    expected_v_scale = expected_v.abs().amax(-1).clamp_min(1e-10) / 448.0
    torch.testing.assert_close(k_scale, expected_k_scale)
    torch.testing.assert_close(v_scale, expected_v_scale)
    unwritten = torch.ones(flat_k_scale.shape[0], dtype=torch.bool, device="cuda")
    unwritten[out_loc.long()] = False
    assert flat_k_scale[unwritten].count_nonzero().item() == 0
    assert flat_v_scale[unwritten].count_nonzero().item() == 0
    assert torch.equal(pool.cmp_k_cache(0), index_before)


@requires_cuda
def test_fp8_sparse_attention_dequantizes_scales_with_masking_and_split_k():
    from freetoken.kernel.triton.qsa import (
        qsa_sparse_paged_attention,
        qsa_sparse_paged_attention_fp8,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(73)
    pages, rows, q_heads, kv_heads, dim = 6, 3, 4, 2, 256
    k = torch.randn(pages, PAGE_SIZE, kv_heads, dim, device=device,
                    dtype=torch.bfloat16, generator=generator)
    v = torch.randn_like(k)
    k_scale = k.float().abs().amax(-1).clamp_min(1e-10) / 448.0
    v_scale = v.float().abs().amax(-1).clamp_min(1e-10) / 448.0
    k_fp8 = (k.float() / k_scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    v_fp8 = (v.float() / v_scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    q = torch.randn(rows, q_heads, dim, device=device, dtype=torch.bfloat16,
                    generator=generator)
    # 130 live columns force the split-K path; trailing -1 entries exercise masked scale loads.
    selected = torch.stack([
        torch.randperm(pages * PAGE_SIZE, device=device)[:130] for _ in range(rows)
    ]).to(torch.int32)
    selected = torch.cat((selected, torch.full((rows, 7), -1, device=device,
                                                dtype=torch.int32)), dim=1)
    block_table = torch.arange(pages, device=device, dtype=torch.int32).repeat(rows, 1)
    token_to_req = torch.arange(rows, device=device, dtype=torch.int32)

    expected = qsa_sparse_paged_attention(
        q, k, v, selected, block_table, token_to_req
    )
    got = qsa_sparse_paged_attention_fp8(
        q, k_fp8, v_fp8, selected, block_table, token_to_req,
        k_scale=k_scale, v_scale=v_scale,
    )

    assert got.shape == expected.shape == (rows, q_heads, dim)
    assert torch.isfinite(got).all()
    aggregate = _relative_l2(got, expected)
    per_token_head = torch.linalg.vector_norm(got.float() - expected.float(), dim=-1) / (
        torch.linalg.vector_norm(expected.float(), dim=-1).clamp_min(1e-12)
    )
    # docs/research/fp8-kv-accuracy-2026-09-03.md measured 3.5-3.8% worst-head error;
    # E4M3 arithmetic predicts ~2.4% aggregate. Near-zero would mean the FP8 path was bypassed, while >4% aggregate
    # or >8% for one token/head points to broken dequantization, masking, or scale addressing.
    assert 0.01 < aggregate < 0.04
    assert float(per_token_head.max()) < 0.08


@requires_cuda
def test_fp8_qsa_backend_matches_its_bf16_dense_oracle():
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    config = parsed_config()
    fixture = Fixture(config, num_pages=32, kv_dtype=torch.float8_e4m3fn)
    attn = fixture.layer(QSA_LAYER, seed=91)
    length = 257
    generator = torch.Generator(device="cuda").manual_seed(19)
    x = torch.randn(length, config.hidden_size, device="cuda", dtype=torch.bfloat16,
                    generator=generator) * 0.5
    req = fixture.req(0, 0, length)
    batch = fixture.batch([req], "prefill")

    got = attn.forward(x, batch)
    fixture.ctx.attn_backend = TorchDenseQSAReference(
        config, num_slots=fixture.num_req_slots, max_len=4096,
        device=fixture.device, dtype=fixture.dtype,
    )
    expected = attn.forward(x, batch)

    assert torch.isfinite(got).all()
    assert _relative_l2(got, expected) < 0.04


@requires_cuda
def test_fp8_store_and_attention_replay_in_one_cuda_graph():
    """The startup-selected pointers and dtypes stay fixed across decode replays."""
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention_fp8

    config = parsed_config()
    fixture = Fixture(config, num_pages=8, kv_dtype=torch.float8_e4m3fn)
    pool = fixture.pool
    heads, dim = 2, 256
    layer_id = QSA_LAYER
    k = torch.zeros(1, heads * dim, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    q = torch.zeros(1, 4, dim, device="cuda", dtype=torch.bfloat16)
    out_loc = torch.tensor([67], device="cuda", dtype=torch.int32)
    selected = torch.tensor([[67, -1, -1, -1]], device="cuda", dtype=torch.int32)
    block_table = torch.arange(8, device="cuda", dtype=torch.int32).unsqueeze(0)
    token_to_req = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)

    def run():
        pool.store_kv(k, v, out_loc, layer_id)
        return qsa_sparse_paged_attention_fp8(
            q, pool.k_cache(layer_id), pool.v_cache(layer_id), selected,
            block_table, token_to_req, out,
            k_scale=pool.k_scale(layer_id), v_scale=pool.v_scale(layer_id),
        )

    k.normal_()
    v.normal_()
    q.normal_()
    run()  # compile before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    torch.cuda.synchronize()

    for seed in (101, 102, 103):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        k.copy_(torch.randn(k.shape, device="cuda", dtype=k.dtype, generator=generator))
        v.copy_(torch.randn(v.shape, device="cuda", dtype=v.dtype, generator=generator))
        q.copy_(torch.randn(q.shape, device="cuda", dtype=q.dtype, generator=generator))
        graph.replay()
        replayed = captured.clone()
        eager = run().clone()
        assert torch.equal(replayed, eager)
