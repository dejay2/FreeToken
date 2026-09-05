"""EXL3 K=1..8 reconstruction helpers used by the proof path.

The packed-state and ``mul1`` reference below are adapted from ExLlamaV3 v1.4.6.

MIT License
Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import functools
import operator
from typing import Tuple

import torch


# Import the optional extension before CUDA graph capture.  CPU-only imports keep the
# reference path available when the proof wheel is not installed; the card path reports a
# focused error from ``reconstruct`` instead of entering an import during graph replay.
try:
    import exllamav3_ext as _exllamav3_ext
except (ImportError, OSError):  # pragma: no cover - depends on the optional proof wheel
    _exllamav3_ext = None


# The proof fixes the codebook to EXL3's ``mul1`` (cb2) path.  The extension supports
# K=1..8, but the first checkpoint and its routed banks use K=2 throughout.
_MUL1_MULTIPLIER = 0x83DCD12D
_MUL1_ACCUMULATOR = 0x6400
_MUL1_K_INV = 0x1EEE
_MUL1_K_BIAS = 0xC931
_MAX_K = 8
# A CPU reconstruction is useful for small reference tests.  The real GLM matrices are
# deliberately rejected here: expanding one of them on the CPU would consume the same
# memory that the card-only proof is designed to avoid.
_CPU_FALLBACK_MAX_TRELLIS_WORDS = 1 << 18


@functools.lru_cache(maxsize=8)
def tensor_core_perm(device: torch.device) -> torch.Tensor:
    """Return EXL3's 16x16 tensor-core lane permutation."""
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        base = t * 8
        perm[base + 0] = r0 * 16 + c0
        perm[base + 1] = r1 * 16 + c0
        perm[base + 2] = r2 * 16 + c0
        perm[base + 3] = r3 * 16 + c0
        perm[base + 4] = r0 * 16 + c1
        perm[base + 5] = r1 * 16 + c1
        perm[base + 6] = r2 * 16 + c1
        perm[base + 7] = r3 * 16 + c1
    return torch.tensor(perm, dtype=torch.int64, device=device)


@functools.lru_cache(maxsize=8)
def normalized_sylvester_hadamard(
    size: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Return the normalized Sylvester Hadamard matrix of ``size``."""
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got {size}")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < size:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)),
            dim=0,
        )
    return h * (size ** -0.5)


def block_left_matmul(h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply ``h`` independently to every leading row block of ``x``."""
    size = h.shape[0]
    if h.shape != (size, size) or x.shape[0] % size:
        raise ValueError(f"left Hadamard shape mismatch: H={tuple(h.shape)}, x={tuple(x.shape)}")
    return torch.matmul(h, x.reshape(-1, size, x.shape[1])).reshape_as(x)


def block_right_matmul(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Apply ``h`` independently to every trailing column block of ``x``."""
    size = h.shape[0]
    if h.shape != (size, size) or x.shape[1] % size:
        raise ValueError(f"right Hadamard shape mismatch: x={tuple(x.shape)}, H={tuple(h.shape)}")
    return torch.matmul(x.reshape(x.shape[0], -1, size), h).reshape_as(x)


def _validate_inputs(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    k: int,
    codebook: str,
) -> Tuple[int, int, int]:
    if not isinstance(trellis, torch.Tensor):
        raise ValueError("trellis must be a torch.Tensor")
    if not isinstance(suh, torch.Tensor) or not isinstance(svh, torch.Tensor):
        raise ValueError("suh and svh must be torch.Tensor values")
    if codebook != "mul1":
        raise ValueError(f"unsupported EXL3 codebook {codebook!r}; only 'mul1' is supported")
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise ValueError(f"EXL3 K must be an integer, got {k!r}") from exc
    if not 1 <= k <= _MAX_K:
        raise ValueError(f"EXL3 K must be in 1..{_MAX_K}, got {k}")
    if trellis.dim() != 3:
        raise ValueError(f"trellis must have rank 3, got rank {trellis.dim()}")
    if trellis.dtype != torch.int16:
        raise ValueError(f"trellis must have dtype torch.int16, got {trellis.dtype}")
    if not trellis.is_contiguous():
        raise ValueError("trellis must be contiguous")
    if trellis.shape[-1] != 16 * k:
        raise ValueError(
            f"trellis last dimension {trellis.shape[-1]} does not match K={k} (expected {16 * k})"
        )

    in_features = int(trellis.shape[0]) * 16
    out_features = int(trellis.shape[1]) * 16
    if in_features <= 0 or out_features <= 0:
        raise ValueError("EXL3 matrices must have non-zero dimensions")
    if in_features % 128 or out_features % 128:
        raise ValueError(
            "EXL3 reconstruction dimensions must be divisible by 128, "
            f"got [{out_features}, {in_features}]"
        )
    if suh.dtype != torch.float16 or svh.dtype != torch.float16:
        raise ValueError(f"suh and svh must have dtype torch.float16, got {suh.dtype}/{svh.dtype}")
    if suh.dim() != 1 or svh.dim() != 1:
        raise ValueError("suh and svh must be one-dimensional")
    if suh.shape != (in_features,):
        raise ValueError(f"suh shape {tuple(suh.shape)} does not match [{in_features}]")
    if svh.shape != (out_features,):
        raise ValueError(f"svh shape {tuple(svh.shape)} does not match [{out_features}]")
    if not suh.is_contiguous() or not svh.is_contiguous():
        raise ValueError("suh and svh must be contiguous")
    devices = {trellis.device, suh.device, svh.device}
    if len(devices) != 1:
        raise ValueError("trellis, suh and svh must be on the same device")
    return k, in_features, out_features


def _validate_buffers(
    out: torch.Tensor | None,
    work: torch.Tensor | None,
    *,
    device: torch.device,
    in_features: int,
    out_features: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise ValueError("out must be a torch.Tensor when supplied")
        if out.shape != (out_features, in_features):
            raise ValueError(
                f"out shape {tuple(out.shape)} does not match [{out_features}, {in_features}]"
            )
        if out.dtype != torch.bfloat16:
            raise ValueError(f"out must have dtype torch.bfloat16, got {out.dtype}")
        if out.device != device:
            raise ValueError(f"out is on {out.device}, expected {device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    if work is not None:
        if not isinstance(work, torch.Tensor):
            raise ValueError("work must be a torch.Tensor when supplied")
        if work.shape != (in_features, out_features):
            raise ValueError(
                f"work shape {tuple(work.shape)} does not match [{in_features}, {out_features}]"
            )
        if work.dtype != torch.float16:
            raise ValueError(f"work must have dtype torch.float16, got {work.dtype}")
        if work.device != device:
            raise ValueError(f"work is on {work.device}, expected {device}")
        if not work.is_contiguous():
            raise ValueError("work must be contiguous")
    return out, work


def _decode_packed_states(trellis: torch.Tensor, k: int) -> torch.Tensor:
    """Decode the 256 sliding 16-bit windows in each packed trellis tile."""
    tiles = trellis.detach().cpu().contiguous().view(torch.uint16).reshape(-1, 16 * k)
    words = tiles.view(torch.uint32).to(torch.int64) & 0xFFFF_FFFF
    states = torch.empty((tiles.shape[0], 256), dtype=torch.int64)
    words32 = 8 * k

    # The C++ fshift() combines two little-endian uint32 words and performs a logical
    # right shift.  Splitting the expression keeps the operation correct on Torch CPU,
    # whose int64 right shift is arithmetic and whose uint64 right shift is unavailable.
    for t in range(256):
        b0 = t * k + k - 16 + 256 * k
        b1 = b0 + 16
        shift = (((b1 - 1) // 32 + 1) * 32) - b1
        hi = words[:, (b0 // 32) % words32]
        lo = words[:, ((b1 - 1) // 32) % words32]
        if shift:
            value = (hi << (32 - shift)) | (lo >> shift)
        else:
            value = lo
        states[:, t] = value & 0xFFFF
    return states


def _decode_mul1(states: torch.Tensor) -> torch.Tensor:
    """Decode EXL3's cb2/mul1 procedural codebook into FP16 values."""
    x = (states * _MUL1_MULTIPLIER) & 0xFFFF_FFFF
    byte_sum = (
        ((x >> 0) & 0xFF)
        + ((x >> 8) & 0xFF)
        + ((x >> 16) & 0xFF)
        + ((x >> 24) & 0xFF)
    )
    h = (byte_sum + _MUL1_ACCUMULATOR).to(torch.uint16).view(torch.float16)
    k_inv = torch.tensor([_MUL1_K_INV], dtype=torch.uint16).view(torch.float16)
    k_bias = torch.tensor([_MUL1_K_BIAS], dtype=torch.uint16).view(torch.float16)
    return (h * k_inv + k_bias).to(torch.float16)


def reconstruct_reference(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    k: int,
    codebook: str,
) -> torch.Tensor:
    """Pure-Torch EXL3 reconstruction oracle, returned as BF16 ``[out, in]``.

    The reference intentionally runs on CPU even when its inputs came from a card.  This
    keeps comparison tests from consuming card memory and mirrors the independent source
    implementation used to check the wheel path.
    """
    k, in_features, out_features = _validate_inputs(trellis, suh, svh, k=k, codebook=codebook)
    if trellis.device.type == "cpu" and trellis.numel() > _CPU_FALLBACK_MAX_TRELLIS_WORDS:
        raise ValueError(
            "CPU EXL3 reference is limited to tiny matrices; use the card reconstruction "
            f"path for {trellis.numel()} packed words"
        )

    states = _decode_packed_states(trellis, k)
    decoded = _decode_mul1(states)
    perm = tensor_core_perm(torch.device("cpu"))
    decoded = decoded[:, torch.argsort(perm)]

    tk, tn, _ = trellis.shape
    w_hat = (
        decoded.reshape(tk, tn, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(tk * 16, tn * 16)
    )

    h128 = normalized_sylvester_hadamard(128, dtype=torch.float32)
    w = block_left_matmul(h128, w_hat.float())
    w.mul_(suh.detach().cpu().float().unsqueeze(1))
    w = block_right_matmul(w, h128)
    w.mul_(svh.detach().cpu().float().unsqueeze(0))
    result = w.T.to(torch.bfloat16).contiguous()
    assert result.shape == (out_features, in_features)
    return result


def reconstruct(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    k: int,
    codebook: str,
    out: torch.Tensor | None = None,
    work: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return contiguous BF16 linear weight ``[out_features, in_features]``.

    This public seam is card-only and calls the imported ExLlamaV3 v1.4.6 extension.  The
    independent :func:`reconstruct_reference` is the CPU oracle for tests; keeping it
    separate prevents a production load from silently expanding a large matrix in host RAM.
    """
    k, in_features, out_features = _validate_inputs(trellis, suh, svh, k=k, codebook=codebook)
    if trellis.device.type != "cuda":
        raise ValueError(
            "EXL3 reconstruction is card-only and requires a CUDA tensor; "
            "use reconstruct_reference for CPU tests"
        )
    out, work = _validate_buffers(
        out,
        work,
        device=trellis.device,
        in_features=in_features,
        out_features=out_features,
    )

    if out is None:
        out = torch.empty(
            (out_features, in_features), dtype=torch.bfloat16, device=trellis.device
        )
    if work is None:
        work = torch.empty(
            (in_features, out_features), dtype=torch.float16, device=trellis.device
        )

    if _exllamav3_ext is None:
        raise RuntimeError(
            "EXL3 CUDA reconstruction needs the ExLlamaV3 v1.4.6 exllamav3_ext wheel"
        )

    _exllamav3_ext.reconstruct_had_slice(
        work,
        trellis,
        suh,
        svh,
        k,
        False,
        codebook == "mul1",
        0,
    )
    out.copy_(work.T)
    return out


__all__ = [
    "block_left_matmul",
    "block_right_matmul",
    "normalized_sylvester_hadamard",
    "reconstruct",
    "reconstruct_reference",
    "tensor_core_perm",
]
