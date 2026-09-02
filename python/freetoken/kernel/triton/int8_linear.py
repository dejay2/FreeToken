"""Weight-only int8 (W8A16) dense linears: bf16 activation x int8 weight, one symmetric
scale per output channel.

WHY
---
The shipping Qwen3.8-Flash-Next NVFP4 checkpoint quantizes only the *routed* experts: the
modelopt ignore list excludes ``*.self_attn.*``, ``*.linear_attn.*``, ``*hyper_connection*``,
``*.mlp.shared_expert.*``, ``*.mlp.gate*``, ``model.embed_tokens`` and ``lm_head``, so every
dense projection on the decode path is bf16 and every decode token drags ~8.0 GiB of dense
weight through HBM (~5.3 ms of a 15 ms step on an RTX 5090; ~7.9 ms of a 6-row speculative
verify, where cuBLAS falls onto a wmma GEMM at M=6). That is also 8.0 GiB of VRAM the expert
cache would rather have. Quantizing it halves both: 3.93 GiB freed, and 1.8-2.9x on every one
of those GEMMs at M = 1..16.

int8 per-output-channel weight-only quantization is what makes that near-lossless: the logit
error it adds to the 248,320-row head is 0.86% of the logit standard deviation, against the
bf16 head's own 0.17% -- and at that vocabulary the bf16 head already disagrees with an fp32
oracle on 4% of Gaussian rows, so the two are indistinguishable where it counts (see
``tests/kernels/test_int8_linear.py``, which asserts the determinate half of that).

What was missing was a kernel. ``torch._weight_int8pack_mm``'s CUDA path is naive -- measured
on this box at 1.56 ms for ``[1, 2560] x [248320, 2560]^T`` against 0.81 ms for the bf16
cuBLAS GEMV, and linear in M. This module is that kernel.

DISPATCH (``int8_linear``)
--------------------------
  * ``M == 1`` (decode): GEMV. Each program owns a ``BLOCK_N`` tile of output rows and
    streams K in ``BLOCK_K`` chunks, converting int8 -> fp32 in registers and deferring the
    K reduction to one ``tl.sum`` at the end (a per-iteration reduce serializes the ALU).
    Split-K with a fused last-arriver reduction when ``N`` is too small to fill the SMs
    (the hyper-connection down projection is ``N = 336``); the counter buffer is zeroed on
    exit so it is reusable across launches and CUDA-graph replays.
  * ``M > 1`` (batched decode, speculative verify, prefill): tensor-core dot GEMM in the
    swapped-operand form ``acc^T = W @ a^T`` -- the weight tile is the lhs, so the converted
    values feed the MMA from registers instead of a shared-memory round trip. int8 codes
    convert to bf16 EXACTLY (bf16 has 8 bits of significand; |code| <= 127), so the dot is
    bit-for-bit ``sum_k code * a`` with an fp32 accumulator and the per-channel scale applied
    once in the fp32 epilogue -- no fp16 range assumption, unlike the NVFP4 sibling.
    Programs are ordered M-fastest, so consecutive blocks share a weight tile: on these
    skinny-M / huge-N shapes the weight is read once from HBM at ANY M while the few-MB
    activation lives in L2.

The NVFP4 sibling switches to dequantize-once + cuBLAS above M = 64. That does not pay here.
Its in-kernel GEMM re-dequantizes every weight tile ``M / BLOCK_M`` times, and its scratch
costs only 0.25 (read FP4) + 1 (write bf16) + 1 (read bf16) weight-units. int8's scratch costs
0.5 + 1 + 1 = 2.5 units against the in-kernel path's 0.5 with the M-fastest order -- measured
at 0.45-0.87x of the bf16 GEMM across these shapes at M = 256..4096, where the in-kernel GEMM
runs at 0.96-1.06x. So there is one GEMM path, all the way up.

TUNING
------
No ``triton.autotune``: autotuning benchmarks inside the call, which is illegal under CUDA
graph capture and makes the chosen config depend on warm-up order. Instead a small static
table keyed on ``(M bucket, N, K)`` (:data:`_GEMV_TABLE` / :data:`_GEMM_TABLE`), measured
offline on the real shapes, with a shape-independent heuristic fallback. The choice is a pure
function of the shapes, so capture sees the same launch every time.

CUDA-graph safe on the decode paths: fixed shapes, no host sync, no allocation whose size
depends on device data.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from freetoken.layers import BaseOP
from freetoken.layers.base import _concat_prefix

from freetoken.kernel.triton.int8_tuning import _GEMM_TABLE, _GEMV_TABLE

_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# Output rows quantized per pass in :func:`quantize_int8_rows`. The fp32 working copy of the
# shipping 248,320-row LM head would be 2.5 GB in one go; at this chunk it is 84 MB.
_QUANT_CHUNK_ROWS = 8192


# ======================================================================================
# Quantizer
# ======================================================================================
def quantize_int8_rows(
    weight: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    chunk_rows: int = _QUANT_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row symmetric int8 for a ``[N, K]`` weight: ``(codes int8, scales)``.

    The scale is stored in the COMPUTE dtype and the codes are then rounded against that
    stored value, not against the fp32 scale it came from. The kernels multiply by what is
    stored, so folding the scale's own rounding in here leaves quantization as the only
    error; leaving it out would add a per-row gain error of up to one bf16 ulp (~0.4%) that
    moves an entire row's outputs together -- which is exactly the error argmax notices.

    Chunked over output rows so the fp32 intermediate stays bounded regardless of ``N``.
    """
    if weight.ndim != 2:
        raise ValueError(f"an int8 weight must be [N, K], got {tuple(weight.shape)}")
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    dtype = weight.dtype if dtype is None else dtype
    rows, width = weight.shape
    codes = torch.empty((rows, width), dtype=torch.int8, device=weight.device)
    scales = torch.empty(rows, dtype=dtype, device=weight.device)
    # An all-zero output row has no scale of its own; any positive one reproduces it exactly
    # (0 / s = 0), and the smallest normal keeps the stored value representable in bf16.
    floor = torch.finfo(torch.float32).tiny
    for lo in range(0, rows, chunk_rows):
        hi = min(lo + chunk_rows, rows)
        block = weight[lo:hi].float()
        scale = (block.abs().amax(dim=1) / 127.0).clamp_min(floor).to(dtype)
        block /= scale.float().unsqueeze(1)
        codes[lo:hi] = block.round_().clamp_(-127.0, 127.0).to(torch.int8)
        scales[lo:hi] = scale
    return codes, scales

# Heuristic fallbacks for shapes missing from int8_tuning.

_GEMV_DEFAULT = (16, 256, 4, 4)
_GEMV_SPLITK_TARGET = 2048  # target total programs before split-K stops growing
_GEMM_SPLITK_TARGET = 512
_PREFILL_TARGET = 512       # programs a prefill GEMM aims for before it stops subdividing
_PREFILL_PART_BYTES = 256 << 20  # cap on the split-K [SPLIT_K, M, N] fp32 partial buffer


def _m_bucket(m: int) -> int:
    """Coarse M classes, so the table stays small and capture-stable across batch sizes.

    Below 64, ``BLOCK_M`` is the bucket and 16 is the MMA's minimum, so every M in 2..16
    (decode batches, the speculative verify's 6 rows) shares one entry and one launch shape.
    Above it the classes are prefill lengths -- what changes with M there is not ``BLOCK_M``
    so much as how much split-K the grid still needs -- and ``0`` is the open-ended top class.
    """
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    if m <= 64:
        return 64
    if m <= 256:
        return 256
    if m <= 1024:
        return 1024
    return 0


def _pow2_floor(v: int) -> int:
    return 1 << (max(v, 1).bit_length() - 1)


def _fill_split_k(target: int, num_mn: int, num_tiles: int) -> int:
    """Power-of-two split-K that brings a ``num_mn``-program grid up to ``target``, without
    dropping any program below ~2 K-tiles (a shorter one never leaves the load pipeline's
    ramp). Power of two so the reduction order is fixed and replay-stable."""
    return _pow2_floor(min(max(target // max(num_mn, 1), 1), max(num_tiles // 2, 1)))


def _gemv_config(n: int, k: int) -> tuple[int, int, int, int, int]:
    """``(BLOCK_N, BLOCK_K, num_warps, num_stages, split_k)`` for the M==1 GEMV."""
    hit = _GEMV_TABLE.get((n, k))
    if hit is not None:
        return hit
    block_n, block_k, warps, stages = _GEMV_DEFAULT
    n_blocks = triton.cdiv(n, block_n)
    num_tiles = triton.cdiv(k, block_k)
    # Split K only far enough to fill the machine, and never below ~2 K-tiles per program
    # (a shorter program never leaves the load pipeline's ramp).
    split_k = _fill_split_k(_GEMV_SPLITK_TARGET, n_blocks, num_tiles)
    return block_n, block_k, warps, stages, split_k


def _prefill_config(m: int, n: int, k: int) -> tuple[int, int, int, int, int, int]:
    """Prefill tiles (M > 64) for a shape not in the table.

    The wide N tile that suits ``lm_head`` leaves the machine almost idle on a narrow
    projection -- the hyper-connection inject is ``N = 336``, two programs at
    ``BLOCK_N = 256`` -- so shrink the N tile and then split K until the grid reaches a few
    waves. ``_PREFILL_TARGET`` is a program count rather than a wave count, so this needs no
    device query and stays a pure function of the shapes (capture-stable). ``BLOCK_M = 64``
    / ``BLOCK_K = 64`` / 3 stages measured within ~10% of the per-shape best everywhere on
    this checkpoint, which is what a fallback has to be.
    """
    block_m, block_k, stages = 64, 64, 3
    block_n = 256
    while block_n > 64 and triton.cdiv(n, block_n) * triton.cdiv(m, block_m) < _PREFILL_TARGET:
        block_n //= 2
    num_mn = triton.cdiv(n, block_n) * triton.cdiv(m, block_m)
    split_k = _fill_split_k(_PREFILL_TARGET, num_mn, triton.cdiv(k, block_k))
    warps = 8 if block_n >= 128 else 4
    return block_m, block_n, block_k, warps, stages, split_k


def _gemm_config(m: int, n: int, k: int) -> tuple[int, int, int, int, int, int]:
    """``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, split_k)`` for M > 1."""
    bucket = _m_bucket(m)
    hit = _GEMM_TABLE.get((bucket, n, k))
    if hit is not None:
        return hit
    if bucket not in (16, 32, 64):
        return _prefill_config(m, n, k)
    block_m = bucket
    block_n = 64
    block_k = 128
    # A 32+-row activation tile has enough work per program for 8 warps; at BLOCK_M == 16
    # (decode / verify) the extra warps only split the same tile finer and lose.
    warps = 8 if bucket >= 32 else 4
    stages = 4
    num_mn = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    split_k = _fill_split_k(_GEMM_SPLITK_TARGET, num_mn, triton.cdiv(k, block_k))
    return block_m, block_n, block_k, warps, stages, split_k


# Arrival counters for the GEMV's fused split-K reduction (one int32 per N-tile). The last
# program to land a partial for a tile reduces the slices in-kernel, which removes a separate
# reduce launch (~2 us on every decode linear). Allocated zeroed; every kernel resets its slot
# after reducing, so the buffer is reusable across launches and graph replays with no memset.
# Cached per (device, size) so a captured launch keeps a stable address.
_SPLITK_COUNTERS: dict = {}


def _splitk_counters(n: int, device: torch.device) -> torch.Tensor:
    key = (device, n)
    counters = _SPLITK_COUNTERS.get(key)
    if counters is None:
        counters = torch.zeros(n, dtype=torch.int32, device=device)
        _SPLITK_COUNTERS[key] = counters
    return counters


# ======================================================================================
# Decode (M == 1) W8A16 GEMV: fp32 accumulate, deferred K reduction.
# ======================================================================================
@triton.jit
def _int8_gemv_kernel(
    a_ptr,      # [K] activation, contiguous
    w_ptr,      # [N, K] int8
    s_ptr,      # [N] per-output-channel scale
    out_ptr,    # [N]
    N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    a_ptrs = a_ptr + offs_k

    partial = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for t in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0)
            a = tl.load(a_ptrs).to(tl.float32)
        else:
            k_ok = t * BLOCK_K + offs_k < K
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_ok[None, :], other=0)
            a = tl.load(a_ptrs, mask=k_ok, other=0.0).to(tl.float32)
        partial += w.to(tl.float32) * a[None, :]
        w_ptrs += BLOCK_K * stride_wk
        a_ptrs += BLOCK_K

    acc = tl.sum(partial, axis=1)
    s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_n, (acc * s).to(OUT), mask=n_mask)


@triton.jit
def _int8_gemv_splitk_kernel(
    a_ptr, w_ptr, s_ptr,
    part_ptr,     # [SPLIT_K, N] fp32 partials (pre-scale)
    counter_ptr,  # [cdiv(N, BLOCK_N)] int32, zeroed and self-resetting
    out_ptr,
    N, K, tiles_per,
    stride_wn, stride_wk,
    stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    OUT: tl.constexpr, EVEN_K: tl.constexpr,
):
    """Split-K decode GEMV for small ``N``: each ``(pid_n, pid_k)`` reduces ``tiles_per``
    K-tiles, so the grid stays wide enough to fill the SMs when ``N`` alone does not. The
    last program to land a partial for an N-tile reduces all ``SPLIT_K`` slices and writes
    the scaled output in-kernel; the acq_rel arrival atomic orders the partial stores."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    k0 = pid_k * tiles_per * BLOCK_K
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :] * stride_wk
    a_ptrs = a_ptr + k0 + offs_k

    partial = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for t in range(tiles_per):
        if EVEN_K:
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0)
            a = tl.load(a_ptrs).to(tl.float32)
        else:
            k_ok = k0 + t * BLOCK_K + offs_k < K
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_ok[None, :], other=0)
            a = tl.load(a_ptrs, mask=k_ok, other=0.0).to(tl.float32)
        partial += w.to(tl.float32) * a[None, :]
        w_ptrs += BLOCK_K * stride_wk
        a_ptrs += BLOCK_K

    acc = tl.sum(partial, axis=1)
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)

    cnt = tl.atomic_add(counter_ptr + pid_n, 1)  # default acq_rel/gpu: publishes the store
    if cnt == SPLIT_K - 1:
        total = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):  # fixed order -> replay-stable reduction
            total += tl.load(
                part_ptr + k * stride_pk + offs_n * stride_pn, mask=n_mask, other=0.0
            )
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs_n, (total * s).to(OUT), mask=n_mask)
        tl.store(counter_ptr + pid_n, 0)  # leave zeroed for the next launch/replay


def _gemv(a: torch.Tensor, w: torch.Tensor, scale: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """M==1 W8A16 GEMV. ``a`` [K], ``w`` [N, K] int8, ``scale`` [N]."""
    out_tl = _TL_DTYPE[out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16]
    n, k = w.shape
    block_n, block_k, warps, stages, split_k = _gemv_config(n, k)
    out = torch.empty(n, dtype=out_dtype, device=a.device)
    n_blocks = triton.cdiv(n, block_n)
    num_tiles = triton.cdiv(k, block_k)
    split_k = max(1, min(split_k, num_tiles))

    if split_k == 1:
        _int8_gemv_kernel[(n_blocks,)](
            a, w, scale, out, n, k, w.stride(0), w.stride(1),
            BLOCK_N=block_n, BLOCK_K=block_k, OUT=out_tl, EVEN_K=k % block_k == 0,
            num_warps=warps, num_stages=stages,
        )
        return out

    tiles_per = triton.cdiv(num_tiles, split_k)
    even = (k % block_k == 0) and (num_tiles == tiles_per * split_k)
    part = torch.empty((split_k, n), dtype=torch.float32, device=a.device)
    counters = _splitk_counters(n_blocks, a.device)
    _int8_gemv_splitk_kernel[(n_blocks, split_k)](
        a, w, scale, part, counters, out, n, k, tiles_per,
        w.stride(0), w.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k, OUT=out_tl, EVEN_K=even,
        num_warps=warps, num_stages=stages,
    )
    return out


# ======================================================================================
# Small-M (2..64) W8A16 dot GEMM: int8 -> bf16 in registers (exact), tensor cores, split-K.
# The weight tile is the MMA's lhs (``acc^T = W @ a^T``) so the converted values stay in
# registers instead of taking a shared-memory round trip as the rhs operand would.
# ======================================================================================
@triton.jit
def _int8_gemm_kernel(
    a_ptr,      # [M, K] activations
    w_ptr,      # [N, K] int8
    s_ptr,      # [N]
    c_ptr,      # [M, N] output (written when SPLIT_K == 1)
    part_ptr,   # [SPLIT_K, M, N] fp32 partials (written when SPLIT_K > 1)
    M, N, K, tiles_per,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    stride_qk, stride_qm, stride_qn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, OUT: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_k = tl.program_id(1)
    # M varies fastest: consecutive programs share a weight tile and stream different
    # activation rows. These shapes are skinny-M / huge-N, so the weight is the tensor worth
    # keeping resident -- this ordering reads it once from HBM (the [M, K] activation is a
    # few MB and lives in L2). At M <= 64 there is a single M-tile and the order is moot.
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_n = pid_mn // num_pid_m
    pid_m = pid_mn % num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    k0 = pid_k * tiles_per * BLOCK_K
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :] * stride_wk

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for t in range(tiles_per):
        if EVEN_K:
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0)
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        else:
            k_ok = k0 + t * BLOCK_K + offs_k < K
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_ok[None, :], other=0)
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_ok[None, :], other=0.0)
        # |code| <= 127 converts to bf16 exactly (8 bits of significand), so the dot is
        # exactly sum_k code * a with an fp32 accumulator; the scale lands in the epilogue.
        acc = tl.dot(w.to(tl.bfloat16), tl.trans(a.to(tl.bfloat16)), acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    io_mask = m_mask[None, :] & n_mask[:, None]
    if SPLIT_K == 1:
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc * s[:, None]
        c_ptrs = c_ptr + offs_m[None, :] * stride_cm + offs_n[:, None] * stride_cn
        tl.store(c_ptrs, acc.to(OUT), mask=io_mask)
    else:
        q_ptrs = (part_ptr + pid_k * stride_qk + offs_m[None, :] * stride_qm
                  + offs_n[:, None] * stride_qn)
        tl.store(q_ptrs, acc, mask=io_mask)


@triton.jit
def _int8_gemm_splitk_reduce_kernel(
    part_ptr, s_ptr, out_ptr, M, N, SPLIT_K: tl.constexpr,
    stride_qk, stride_qm, stride_qn,
    stride_om, stride_on,
    BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    """Split-K reduce for the dot GEMM. Kept a separate launch (unlike the GEMV's fused
    last-arriver reduce): the ``[BLOCK_N, BLOCK_M]`` reduce tile would inflate the main
    loop's register budget, and it runs serially at the tail where the GEMV's ``[BLOCK_N]``
    reduce is small enough to be free."""
    pid_m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_qk + pid_m * stride_qm + offs * stride_qn,
                       mask=mask, other=0.0)
    s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid_m * stride_om + offs * stride_on, (acc * s).to(OUT), mask=mask)


def _gemm_inkernel(a: torch.Tensor, w: torch.Tensor, scale: torch.Tensor,
                   out_dtype: torch.dtype) -> torch.Tensor:
    m, k = a.shape
    n = w.shape[0]
    compute = out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16
    block_m, block_n, block_k, warps, stages, split_k = _gemm_config(m, n, k)
    out = torch.empty((m, n), dtype=compute, device=a.device)

    num_mn = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    num_tiles = triton.cdiv(k, block_k)
    split_k = max(1, min(split_k, num_tiles))
    # Split-K's [SPLIT_K, M, N] fp32 partial buffer is the one allocation here big enough to
    # matter once M leaves decode territory; keep it bounded whatever the table asks for.
    while split_k > 1 and split_k * m * n * 4 > _PREFILL_PART_BYTES:
        split_k //= 2
    tiles_per = triton.cdiv(num_tiles, split_k)
    even = (k % block_k == 0) and (num_tiles == tiles_per * split_k)
    part = (torch.empty((split_k, m, n), dtype=torch.float32, device=a.device)
            if split_k > 1 else out)  # unused dummy when split_k == 1
    _int8_gemm_kernel[(num_mn, split_k)](
        a, w, scale, out, part, m, n, k, tiles_per,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
        part.stride(0) if split_k > 1 else 0,
        part.stride(1) if split_k > 1 else 0,
        part.stride(2) if split_k > 1 else 0,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
        OUT=_TL_DTYPE[compute], EVEN_K=even, num_warps=warps, num_stages=stages,
    )
    if split_k > 1:
        r_block, r_warps = (128, 2) if n <= 8192 else (512, 8)
        _int8_gemm_splitk_reduce_kernel[(m, triton.cdiv(n, r_block))](
            part, scale, out, m, n, split_k,
            part.stride(0), part.stride(1), part.stride(2), out.stride(0), out.stride(1),
            BLOCK=r_block, OUT=_TL_DTYPE[compute], num_warps=r_warps,
        )
    return out


# ======================================================================================
# Weight dequantization. Not on any serving path (the GEMM reads the int8 weight directly at
# every M -- see the module docstring); this is the numeric reference the kernels are diffed
# against, and the CPU fallback's only way to compute the product.
# ======================================================================================
@triton.jit
def _int8_dequant_kernel(
    w_ptr, s_ptr, out_ptr, N, K,
    stride_wn, stride_wk, stride_on, stride_ok,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n < N)[:, None] & (offs_k < K)[None, :]
    w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=mask, other=0).to(tl.float32)
    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok,
             (w * s[:, None]).to(OUT), mask=mask)


def dequant_int8_rows(w: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out[n, k] = w[n, k] * scale[n]`` for an int8 ``[N, K]`` weight (CUDA: triton)."""
    n, k = w.shape
    if not w.is_cuda:
        torch.mul(w.to(out.dtype), scale.to(out.dtype).unsqueeze(1), out=out)
        return out
    block_n, block_k = 16, 256
    _int8_dequant_kernel[(triton.cdiv(n, block_n), triton.cdiv(k, block_k))](
        w, scale, out, n, k, w.stride(0), w.stride(1), out.stride(0), out.stride(1),
        BLOCK_N=block_n, BLOCK_K=block_k, OUT=_TL_DTYPE[out.dtype], num_warps=4,
    )
    return out


def _linear_cpu(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """CPU fallback (the triton kernels are CUDA-only): dequantize, then ``F.linear``."""
    weight = torch.empty(w.shape, dtype=torch.float32, device=w.device)
    dequant_int8_rows(w, scale, weight)
    return torch.nn.functional.linear(x.to(torch.float32), weight).to(x.dtype)


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = x @ (weight * scale[:, None])^T`` with an int8 ``[N, K]`` weight.

    ``x`` is ``[..., K]`` in a floating dtype (bf16 on the serving path), ``scale`` is ``[N]``
    in the compute dtype, and the result is ``[..., N]`` in ``x``'s dtype.
    """
    *lead, k = x.shape
    n = weight.shape[0]
    if weight.dtype is not torch.int8:
        # The layers below declare a bf16 ``weight`` until load time (see _Int8DenseBase);
        # reaching the kernel with one means the module was never given its checkpoint value.
        raise ValueError(f"int8_linear needs an int8 weight, got {weight.dtype}")
    if weight.shape[1] != k:
        raise ValueError(f"weight [N, K] = {tuple(weight.shape)} does not accept K = {k}")
    if not x.is_cuda:
        out = _linear_cpu(x.reshape(-1, k), weight, scale).reshape(*lead, n)
        return out if bias is None else out + bias.to(out.dtype)

    m = x.numel() // k
    if m == 1:
        out = _gemv(x.reshape(k).contiguous(), weight, scale, x.dtype).reshape(*lead, n)
    else:
        rows = x.reshape(-1, k)
        out = _gemm_inkernel(
            rows if rows.is_contiguous() else rows.contiguous(), weight, scale, x.dtype
        ).reshape(*lead, n)
    return out if bias is None else out + bias.to(out.dtype)


# ======================================================================================
# BaseOP linear layers.
#
# TWO RULES SHAPE THESE CLASSES, both imposed by the loader they have to live inside:
#
#  1. ``weight`` is DECLARED bf16 ``[out, in]`` -- exactly what the bf16 class it replaces
#     declares, and exactly what the checkpoint ships. The engine materializes each loaded
#     tensor as ``weight.to(device, dtype=model_state[key].dtype)``, reading the dtype off
#     the model's own ``state_dict()``; declaring int8 here would truncate every checkpoint
#     value to a raw int8 before this class ever saw it. The buffer BECOMES int8 inside
#     ``load_state_dict``, once the real values have arrived. Construction happens under
#     ``torch.device("meta")``, so the bf16 declaration costs nothing.
#
#  2. The per-channel scale is PRIVATE (``_scale``). It has no checkpoint key, and a public
#     tensor attribute would appear in ``state_dict()`` -- which the dummy-weight path
#     fabricates a random tensor for, overwriting the scale this class just computed.
#     ``weight_scale`` exposes it read-only.
# ======================================================================================
class _Int8DenseBase(BaseOP):
    """Shared load/forward for the int8 dense linears.

    ``load_state_dict`` takes the checkpoint's bf16 ``[out, in]`` tensor -- the loader, the
    key rewrites and the q/k/v merge rules are untouched -- and quantizes it into the
    resident int8 buffer plus one scale per output row. Quantization is chunked, so the fp32
    working set stays bounded even for the 248,320-row LM head.
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features)  # bf16 until load; see above
        self.bias = torch.empty(out_features) if has_bias else None
        self._scale = torch.empty(out_features)

    @property
    def weight_scale(self) -> torch.Tensor:
        """The per-output-channel scale (read-only; deliberately not a state_dict key)."""
        return self._scale

    def quantize_from(self, weight: torch.Tensor) -> None:
        """Adopt a bf16 ``[out, in]`` weight as int8 codes + per-row scales."""
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"expected a [{self.out_features}, {self.in_features}] weight, "
                f"got {tuple(weight.shape)}"
            )
        self.weight, self._scale = quantize_int8_rows(weight)

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        self.quantize_from(state_dict.pop(_concat_prefix(prefix, "weight")))
        if self.bias is not None:
            self.bias = state_dict.pop(_concat_prefix(prefix, "bias"))
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    @property
    def resident_bytes(self) -> int:
        tensors = [self.weight, self._scale]
        if self.bias is not None:
            tensors.append(self.bias)
        return sum(t.numel() * t.element_size() for t in tensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_linear(x, self.weight, self._scale, self.bias)


class Int8DenseLinear(_Int8DenseBase):
    """Replicated int8 dense linear (W8A16). Drop-in for ``LinearReplicated``."""


class Int8DenseColMerged(_Int8DenseBase):
    """Column-merged int8 dense linear mirroring ``LinearColParallelMerged``.

    One weight concatenating several projections on the output dim; each output row keeps
    its own scale, so a merged weight is exactly as accurate as the split ones. Each output
    part is sharded independently by tensor-parallel rank, and the caller splits the output
    by ``output_sizes`` as before.
    """

    def __init__(self, in_features: int, output_sizes: list[int], has_bias: bool = False):
        from freetoken.distributed import get_tp_info
        from freetoken.utils import div_even

        tp_info = get_tp_info()
        self.output_sizes = list(output_sizes)
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        super().__init__(in_features, sum(tp_output_sizes), has_bias)


class Int8DenseRowParallel(_Int8DenseBase):
    """Row-parallel int8 dense linear (drop-in for ``LinearRowParallel``).

    The input dim is sharded, so the local weight is ``[out, in // tp]`` and the partial
    products are all-reduced -- identical TP semantics to the bf16 class, with quantization
    applied per local output row of the local shard.
    """

    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        from freetoken.distributed import DistributedCommunicator, get_tp_info
        from freetoken.utils import div_even

        tp_info = get_tp_info()
        self.full_input_size = input_size
        self.full_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(div_even(input_size, tp_info.size), output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = int8_linear(x, self.weight, self._scale, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class Int8LMHead(BaseOP):
    """Weight-only int8 LM head. Mirrors ``ParallelLMHead`` (untied): the same vocab-parallel
    range, the same all-gather at TP > 1 and the same prefill last-row slice, with the bf16
    ``F.linear`` replaced by the W8A16 kernel over a half-size weight.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        from freetoken.distributed import DistributedCommunicator, get_tp_info
        from freetoken.utils import div_ceil

        tp_info = get_tp_info()
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start = self.num_embeddings_tp * tp_info.rank
        finish = min(start + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start, finish - start)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)  # bf16 until load
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self._scale = torch.empty(self.num_embeddings_tp)
        self._comm = DistributedCommunicator()

    @property
    def weight_scale(self) -> torch.Tensor:
        return self._scale

    def quantize_from(self, weight: torch.Tensor) -> None:
        expected_shape = (self.num_embeddings_tp, self.embedding_dim)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"expected a [{expected_shape[0]}, {expected_shape[1]}] weight, "
                f"got {tuple(weight.shape)}"
            )
        self.weight, self._scale = quantize_int8_rows(weight)

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        self.quantize_from(state_dict.pop(_concat_prefix(prefix, "weight")))
        if self.bias is not None:
            self.bias = state_dict.pop(_concat_prefix(prefix, "bias"))
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    @property
    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self._scale))

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        logits = int8_linear(x, self.weight, self._scale, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        gathered = self._comm.all_gather(logits)
        if input_shape[0] == 1:
            return gathered.view(1, -1)[:, : self.num_embeddings]
        gathered = gathered.view((self.tp_size,) + input_shape)
        gathered = gathered.permute(1, 0, 2).contiguous()
        gathered = gathered.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return gathered[:, : self.num_embeddings]

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Private teacher seam: project every input row without prefill last-row slicing."""
        return self._project(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return self._project(x)


__all__ = [
    "Int8DenseColMerged",
    "Int8DenseLinear",
    "Int8DenseRowParallel",
    "Int8LMHead",
    "dequant_int8_rows",
    "int8_linear",
    "quantize_int8_rows",
]
