"""Weight-only int8 (W8A16) dense linear: the triton GEMV/GEMM against the dequant
reference, the quantizer's round-trip contract, and the argmax agreement an LM head needs.

Shapes are the ones Qwen3.8-Flash-Next actually runs through these kernels (GDN in/out, QSA
qkv/o_proj/indexer, shared expert, hyper-connection, lm_head) plus two deliberately awkward
ones, because every tuning-table entry is keyed on (M bucket, N, K) and a shape that misses
the table has to land on the heuristic and still be right.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.int8_linear import (
    Int8DenseColMerged,
    Int8DenseLinear,
    Int8LMHead,
    _gemm_config,
    _gemv_config,
    int8_linear,
    quantize_int8_rows,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# The checkpoint's dense projections. lm_head ([248320, 2560]) is exercised on its own below:
# a bf16 copy of it is 1.27 GB, which does not belong in a parametrized sweep.
SHAPES = [
    (16480, 2560),   # GDN in_proj (qkv|z|b|a fused)
    (2560, 6144),    # GDN out_proj / QSA o_proj
    (13312, 2560),   # QSA qkv_proj
    (640, 2560),     # QSA indexer index_qk_proj
    (1280, 2560),    # shared expert gate_up_proj
    (2560, 640),     # shared expert down_proj
    (336, 10240),    # hyper-connection down_block_inject
    (10240, 320),    # hyper-connection input_mix_weight_up
]
# Off-table shapes: N not a multiple of any BLOCK_N (n-mask tail) and K not a multiple of any
# BLOCK_K (k-mask tail + the EVEN_K=False branch of every kernel).
RAGGED = [(1001, 2560), (2560, 1001), (37, 129)]


def _quantized(n: int, k: int, seed: int, device: str = "cuda"):
    torch.manual_seed(seed)
    w = (torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02)
    codes, scales = quantize_int8_rows(w)
    return codes, scales


def _reference(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """What the kernel is asked to compute, in fp32: x @ (codes * scale)^T."""
    weight = codes.float() * scales.float().unsqueeze(1)
    return x.float() @ weight.t()


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()


# ======================================================================================
# Kernel vs reference
# ======================================================================================
@requires_cuda
@pytest.mark.parametrize("m", [1, 2, 4, 6, 8, 16, 64])
@pytest.mark.parametrize("n,k", SHAPES)
def test_decode_shapes_match_the_dequant_reference(m: int, n: int, k: int):
    codes, scales = _quantized(n, k, seed=m + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = int8_linear(x, codes, scales)
    assert out.shape == (m, n) and out.dtype is torch.bfloat16
    # bf16 accumulation of a K-long dot is itself ~1e-2 relative; the kernel adds nothing
    # beyond that (its accumulator is fp32 and the int8 -> bf16 conversion is exact).
    assert _rel_err(out, _reference(x, codes, scales)) < 2e-2


@requires_cuda
@pytest.mark.parametrize("m", [1, 3, 65, 128, 300, 1100])
@pytest.mark.parametrize("n,k", RAGGED)
def test_ragged_shapes_and_prefill_m_match_the_reference(m: int, n: int, k: int):
    """Every M bucket including the three prefill ones, on shapes absent from the table."""
    codes, scales = _quantized(n, k, seed=m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = int8_linear(x, codes, scales)
    assert out.shape == (m, n)
    assert _rel_err(out, _reference(x, codes, scales)) < 2e-2


@requires_cuda
@pytest.mark.parametrize("m", [1, 6, 128])
def test_bias_and_leading_dims_are_preserved(m: int):
    n, k = 640, 2560
    codes, scales = _quantized(n, k, seed=7)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, m, k, device="cuda", dtype=torch.bfloat16)
    out = int8_linear(x, codes, scales, bias)
    assert out.shape == (2, m, n)
    ref = _reference(x.reshape(-1, k), codes, scales).reshape(2, m, n) + bias.float()
    assert _rel_err(out, ref) < 2e-2


@requires_cuda
def test_a_non_contiguous_activation_is_accepted():
    n, k = 1280, 2560
    codes, scales = _quantized(n, k, seed=11)
    wide = torch.randn(8, 2 * k, device="cuda", dtype=torch.bfloat16)
    x = wide[:, ::2]  # stride-2 view: the kernels require a contiguous [M, K]
    assert not x.is_contiguous()
    assert _rel_err(int8_linear(x, codes, scales), _reference(x, codes, scales)) < 2e-2


@requires_cuda
def test_a_bf16_weight_is_refused():
    """The layers declare a bf16 weight until load; reaching the kernel with one is a bug."""
    w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    scales = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="int8 weight"):
        int8_linear(torch.randn(1, 128, device="cuda", dtype=torch.bfloat16), w, scales)


def test_the_cpu_fallback_matches_the_dequant_reference():
    """No triton on CPU, but the classes are constructible and testable there."""
    codes, scales = _quantized(96, 128, seed=3, device="cpu")
    x = torch.randn(5, 128, dtype=torch.bfloat16)
    assert _rel_err(int8_linear(x, codes, scales), _reference(x, codes, scales)) < 2e-2


# ======================================================================================
# Quantizer
# ======================================================================================
@pytest.mark.parametrize("chunk", [8192, 7])  # one pass vs many, same answer
def test_quantize_int8_rows_round_trip(chunk: int):
    torch.manual_seed(0)
    w = torch.randn(19, 64, dtype=torch.bfloat16)
    codes, scales = quantize_int8_rows(w, chunk_rows=chunk)

    assert codes.dtype is torch.int8 and codes.shape == w.shape
    assert scales.dtype is w.dtype and scales.shape == (19,)
    assert int(codes.abs().max()) <= 127
    # The scale is amax/127 IN THE STORED DTYPE, and the codes are rounded against that
    # stored value -- so dequantizing reproduces the weight to within half a code.
    expected = (w.float().abs().amax(dim=1) / 127.0).to(w.dtype)
    assert torch.equal(scales, expected)
    err = (codes.float() * scales.float().unsqueeze(1) - w.float()).abs()
    assert (err <= 0.5 * scales.float().unsqueeze(1) + 1e-6).all()
    # Each row uses its full code range (the max-magnitude entry lands on +-127).
    assert torch.equal(codes.abs().amax(dim=1), torch.full((19,), 127, dtype=torch.int8))


def test_quantize_int8_rows_reproduces_an_all_zero_row_exactly():
    w = torch.zeros(3, 16, dtype=torch.bfloat16)
    w[1] = 0.5
    codes, scales = quantize_int8_rows(w)
    assert (scales > 0).all()  # a zero row still needs a usable (positive) scale
    assert torch.equal(codes[0], torch.zeros(16, dtype=torch.int8))
    # 0 / s == 0 for any positive s, so the zero rows come back exact whatever scale they got.
    zeros = codes[[0, 2]].float() * scales[[0, 2]].float().unsqueeze(1)
    assert torch.equal(zeros, torch.zeros(2, 16))


def test_quantize_int8_rows_refuses_a_non_matrix():
    with pytest.raises(ValueError, match=r"\[N, K\]"):
        quantize_int8_rows(torch.zeros(4, 4, 4))
    with pytest.raises(ValueError, match="chunk_rows"):
        quantize_int8_rows(torch.zeros(4, 4), chunk_rows=0)


def test_the_draft_head_shares_this_quantizer():
    """engine/spec_lmhead re-exports it rather than keeping a second copy."""
    from freetoken.engine import spec_lmhead

    assert spec_lmhead.quantize_int8_rows is quantize_int8_rows


# ======================================================================================
# Tuning-table stability: graph capture must see the same launch every replay.
# ======================================================================================
@pytest.mark.parametrize("n,k", SHAPES + RAGGED)
def test_config_choice_is_a_pure_function_of_the_shape(n: int, k: int):
    for m in (1, 2, 6, 16, 33, 100, 700, 4096):
        chooser = (lambda: _gemv_config(n, k)) if m == 1 else (lambda mm=m: _gemm_config(mm, n, k))
        assert chooser() == chooser()


@pytest.mark.parametrize("m", [2, 4, 6, 8, 16])
def test_every_decode_batch_size_shares_one_launch_shape(m: int):
    """M in 2..16 is one bucket, so a captured decode graph does not depend on batch size."""
    assert _gemm_config(m, 16480, 2560) == _gemm_config(2, 16480, 2560)


@requires_cuda
@pytest.mark.parametrize(
    "n,k,m",
    [
        (16480, 2560, 1),   # single-pass GEMV
        (336, 10240, 1),    # split-K GEMV: the fused last-arriver reduce and its counters
        (2560, 6144, 1),    # split-K GEMV, moderate N
        (16480, 2560, 6),   # dot GEMM at the speculative verify's width
        (336, 10240, 6),    # dot GEMM with a split-K partial buffer + reduce launch
    ],
)
def test_decode_paths_capture_and_replay_bit_identically(n: int, k: int, m: int):
    """Decode runs inside a CUDA graph. The split-K arrival counters are shared across calls
    and reset in-kernel, so a replay must not see a stale count -- this is the test that
    would catch it, and it also pins "no host sync, no data-dependent allocation"."""
    codes, scales = _quantized(n, k, seed=n + m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    int8_linear(x, codes, scales)  # warm the JIT and the counter cache before capture

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            int8_linear(x, codes, scales)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=torch.cuda.graphs.graph_pool_handle()):
        out = int8_linear(x, codes, scales)
    for _ in range(3):
        x.normal_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, int8_linear(x, codes, scales))


# ======================================================================================
# LM head: the property that actually matters -- which token wins.
#
# WHAT THE BAR CAN AND CANNOT BE. A raw "int8 argmax == bf16 argmax >= 99%" over iid-Gaussian
# weights at this vocabulary is not a test of the quantizer, because the bf16 reference is not
# ground truth at that resolution. Measured here on [248320, 2560], 512 Gaussian rows:
#
#     int8 vs bf16 F.linear   97.3%          int8 RMS error   0.86% of the logit std
#     int8 vs an fp32 oracle  95.9%          bf16 RMS error   0.17% of the logit std
#     bf16 vs the same oracle 95.9%          median top-2 gap  15% of the logit std
#
# bf16 disagrees with fp32 as often as int8 does: with 248,320 iid logits the top-2 gap is at
# the noise floor for BOTH, and 42 of those 512 rows do not even have a strict bf16 top-2
# (exact ties). Restricted to rows that DO have a strict top-2, int8 agrees with bf16 98.7%
# of the time, and every remaining miss sits below 5% of the logit std. So the tests below
# assert what is determinate -- once the gap clears the quantization noise the token is the
# same one, every time -- and pin int8 against bf16's own distance from the exact answer.
# ======================================================================================
@requires_cuda
def test_lm_head_picks_the_same_token_whenever_the_choice_is_determinate():
    """~1.9 GB transient; freed on the way out."""
    import torch.nn.functional as F

    n, k = 248320, 2560
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    codes, scales = quantize_int8_rows(w)
    try:
        rows = torch.randn(512, k, device="cuda", dtype=torch.bfloat16)
        ref = F.linear(rows, w).float()
        out = int8_linear(rows, codes, scales).float()
        assert out.shape == (512, n)
        assert _rel_err(out, ref) < 2e-2

        std = ref.std()
        assert ((out - ref).pow(2).mean().sqrt() / std).item() < 0.015

        top2 = ref.topk(2, dim=-1).values
        gap = (top2[:, 0] - top2[:, 1]) / std
        same = out.argmax(-1) == ref.argmax(-1)
        # Every disagreement is a near-tie: the gap is under ~6x the quantization noise.
        assert same[gap > 0.05].all(), int((~same[gap > 0.05]).sum())
        # And the bf16 winner is never pushed out of the int8 head's top 5.
        top5 = out.topk(5, dim=-1).indices
        assert (top5 == ref.argmax(-1).unsqueeze(1)).any(1).all()
    finally:
        del w, codes, scales
        torch.cuda.empty_cache()


@requires_cuda
def test_the_bf16_lm_head_is_no_closer_to_the_exact_answer_than_the_int8_one():
    """Why the test above measures determinacy instead of a raw agreement percentage.

    Against an fp32 oracle over the SAME bf16 weight, the bf16 ``F.linear`` picks the exact
    argmax no more often than the int8 kernel does: at 248,320 iid-Gaussian rows the winner is
    decided by a gap both of them are blind to. A raw ">= 99% agreement with bf16" bar would
    therefore be measuring the reference's noise, not the quantizer -- so this pins the
    comparison that IS meaningful, and it fails if int8 ever falls materially behind bf16.
    """
    import torch.nn.functional as F

    n, k, m = 248320, 2560, 128
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    codes, scales = quantize_int8_rows(w)
    try:
        rows = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        exact = torch.empty(m, n, device="cuda", dtype=torch.float32)
        for lo in range(0, n, 32768):  # chunked: a [128, 248320] fp32 oracle is 127 MB
            exact[:, lo:lo + 32768] = rows.float() @ w[lo:lo + 32768].float().t()
        oracle = exact.argmax(-1)
        bf16_hits = (F.linear(rows, w).argmax(-1) == oracle).float().mean().item()
        int8_hits = (int8_linear(rows, codes, scales).argmax(-1) == oracle).float().mean().item()
        assert int8_hits >= bf16_hits - 0.02, (int8_hits, bf16_hits)
    finally:
        del w, codes, scales
        torch.cuda.empty_cache()


# ======================================================================================
# Modules
# ======================================================================================
def _bf16_twin(module, weight: torch.Tensor):
    """The bf16 F.linear the int8 module replaces, on the same weight."""
    return torch.nn.functional.linear


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_int8_dense_linear_matches_its_bf16_twin(device: str):
    import torch.nn.functional as F

    from freetoken.layers import LinearReplicated

    torch.manual_seed(0)
    w = torch.randn(96, 128, device=device, dtype=torch.bfloat16) * 0.05
    bf16 = LinearReplicated(128, 96, has_bias=False)
    bf16.weight = w
    quant = Int8DenseLinear(128, 96)
    quant.load_state_dict({"weight": w.clone()})

    assert quant.weight.dtype is torch.int8
    assert quant.weight_scale.shape == (96,)
    assert quant.resident_bytes == 96 * 128 + 96 * 2

    x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    got, want = quant.forward(x), bf16.forward(x)
    assert got.shape == want.shape
    assert _rel_err(got, want.float()) < 3e-2


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_int8_col_merged_keeps_a_per_part_scale(device: str, monkeypatch):
    """A merged weight is exactly as accurate as the split ones: the scale is per ROW, so a
    part whose rows are 100x smaller keeps its own resolution."""
    _set_tp_info_for_test(monkeypatch, rank=0, size=1)
    torch.manual_seed(1)
    small = torch.randn(32, 128, device=device, dtype=torch.bfloat16) * 0.0005
    large = torch.randn(64, 128, device=device, dtype=torch.bfloat16) * 0.05
    merged = torch.cat([small, large], dim=0)

    quant = Int8DenseColMerged(128, [32, 64])
    quant.load_state_dict({"weight": merged.clone()})
    assert quant.output_sizes == [32, 64]
    assert quant.weight.shape == (96, 128)

    x = torch.randn(3, 128, device=device, dtype=torch.bfloat16)
    got = quant.forward(x)
    want = torch.nn.functional.linear(x, merged).float()
    head = _rel_err(got[:, :32], want[:, :32])
    assert head < 3e-2, head  # the small part is not swamped by the large one's scale


def _set_tp_info_for_test(monkeypatch, *, rank: int, size: int = 2) -> None:
    import freetoken.distributed.info as tp_info
    from freetoken.distributed import set_tp_info

    monkeypatch.setattr(tp_info, "_TP_INFO", None)
    set_tp_info(rank=rank, size=size)


@pytest.mark.parametrize("rank", [0, 1])
def test_int8_col_merged_tp2_shards_each_part_like_bf16_twin(monkeypatch, rank: int):
    from freetoken.layers import LinearColParallelMerged

    _set_tp_info_for_test(monkeypatch, rank=rank)
    input_size = 4
    non_divisible_output_sizes = [3, 5]
    assert sum(non_divisible_output_sizes) % 2 == 0
    with pytest.raises(AssertionError):
        LinearColParallelMerged(input_size, non_divisible_output_sizes, has_bias=False)
    with pytest.raises(AssertionError):
        Int8DenseColMerged(input_size, non_divisible_output_sizes)

    output_sizes = [4, 6]
    full_weight = torch.arange(40, dtype=torch.bfloat16).reshape(10, input_size)
    local_rows = torch.cat(
        [
            full_weight[offset + rank * (size // 2): offset + (rank + 1) * (size // 2)]
            for offset, size in ((0, output_sizes[0]), (output_sizes[0], output_sizes[1]))
        ]
    )

    bf16 = LinearColParallelMerged(input_size, output_sizes, has_bias=False)
    bf16.weight = local_rows.clone()
    quant = Int8DenseColMerged(input_size, output_sizes)
    quant.load_state_dict({"weight": bf16.weight.clone()})

    expected_codes, expected_scales = quantize_int8_rows(bf16.weight)
    assert quant.weight.shape == bf16.weight.shape
    assert torch.equal(quant.weight, expected_codes)
    assert torch.equal(quant.weight_scale, expected_scales)


def test_int8_lm_head_rejects_a_mis_sharded_weight(monkeypatch):
    _set_tp_info_for_test(monkeypatch, rank=0)
    head = Int8LMHead(num_embeddings=8, embedding_dim=4)

    with pytest.raises(ValueError, match=r"expected a \[4, 4\] weight"):
        head.quantize_from(torch.zeros(8, 4, dtype=torch.bfloat16))


@requires_cuda
def test_int8_row_parallel_matches_bf16_at_tp1():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    from freetoken.kernel.triton.int8_linear import Int8DenseRowParallel

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(2)
    w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.05
    quant = Int8DenseRowParallel(256, 128)
    quant.load_state_dict({"weight": w.clone()})
    x = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)
    assert _rel_err(quant.forward(x), torch.nn.functional.linear(x, w).float()) < 3e-2


@requires_cuda
def test_int8_lm_head_projects_every_row_through_forward_all():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(3)
    w = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    head = Int8LMHead(num_embeddings=512, embedding_dim=128)
    # Declared in the model's float dtype so the engine's dtype-matching materializer does not
    # truncate the checkpoint value; int8 only after load.
    assert head.weight.is_floating_point()
    head.load_state_dict({"weight": w.clone()})
    assert head.weight.dtype is torch.int8
    assert head.vocab_range == (0, 512) and head.num_embeddings_tp == 512

    x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
    out = head.forward_all(x)
    assert out.shape == (5, 512)
    assert _rel_err(out, torch.nn.functional.linear(x, w).float()) < 3e-2


def test_the_scale_is_not_a_state_dict_key():
    """It has no checkpoint key, and the dummy-weight path fabricates one tensor per key --
    a public attribute would let it overwrite the scale the module just computed. The weight
    key itself is declared FLOAT (the checkpoint's dtype, under the engine's meta+dtype
    context) because the loader casts each loaded tensor to the dtype the model declares."""
    quant = Int8DenseLinear(128, 96)
    assert set(quant.state_dict()) == {"weight"}
    assert quant.state_dict()["weight"].is_floating_point()
