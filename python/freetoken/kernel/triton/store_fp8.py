"""Quantizing scatter-store for QSA's E4M3 K/V cache."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_fp8_cache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    k_ptr,
    v_ptr,
    num_cache_tokens,
    stride_cache_token,
    stride_cache_head,
    stride_scale_token,
    stride_scale_head,
    stride_src_row,
    stride_src_head,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    slot = tl.load(indices_ptr + row).to(tl.int64)
    valid_slot = (slot >= 0) & (slot < num_cache_tokens)
    offsets = tl.arange(0, BLOCK_D)
    mask = valid_slot & (offsets < HEAD_DIM)
    src = row * stride_src_row + head * stride_src_head + offsets
    key = tl.load(k_ptr + src, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(v_ptr + src, mask=mask, other=0.0).to(tl.float32)
    key_scale = tl.maximum(tl.max(tl.abs(key), axis=0), 1.0e-10) / 448.0
    value_scale = tl.maximum(tl.max(tl.abs(value), axis=0), 1.0e-10) / 448.0
    key = tl.clamp(key / key_scale, -448.0, 448.0).to(tl.float8e4nv)
    value = tl.clamp(value / value_scale, -448.0, 448.0).to(tl.float8e4nv)
    dst = slot * stride_cache_token + head * stride_cache_head + offsets
    tl.store(k_cache_ptr + dst, key, mask=mask)
    tl.store(v_cache_ptr + dst, value, mask=mask)
    scale_dst = slot * stride_scale_token + head * stride_scale_head
    tl.store(k_scale_ptr + scale_dst, key_scale, mask=valid_slot)
    tl.store(v_scale_ptr + scale_dst, value_scale, mask=valid_slot)


def store_fp8_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Scatter BF16 K/V rows into E4M3 storage with FP32 token/head scales."""
    if k_cache.dtype is not torch.float8_e4m3fn or v_cache.dtype is not k_cache.dtype:
        raise ValueError("FP8 cache store needs E4M3 K/V destinations")
    if k_scale.dtype is not torch.float32 or v_scale.dtype is not torch.float32:
        raise ValueError("FP8 cache store needs FP32 scale destinations")
    if k_cache.ndim != 3 or v_cache.shape != k_cache.shape:
        raise ValueError("FP8 cache store expects [tokens, heads, head_dim] destinations")
    if k_scale.shape != k_cache.shape[:2] or v_scale.shape != k_scale.shape:
        raise ValueError("FP8 cache scales must have one row per token and head")
    if indices.dtype is not torch.int32 or indices.shape != (k.shape[0],):
        raise ValueError("FP8 cache indices must be one int32 slot per source row")
    if k.shape != v.shape or k.ndim not in (2, 3):
        raise ValueError("FP8 cache source K/V shapes differ")
    heads, head_dim = k_cache.shape[1:]
    if k.numel() != indices.numel() * heads * head_dim:
        raise ValueError("FP8 cache source width does not match destination heads")
    k = k.reshape(indices.numel(), heads, head_dim)
    v = v.reshape_as(k)
    if not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("FP8 cache source rows must be contiguous")
    if torch.cuda.get_device_capability(k_cache.device) < (8, 9):
        raise ValueError("FP8 KV storage needs a GPU with native E4M3 support (sm89+)")

    _store_fp8_cache_kernel[(indices.numel(), heads)](
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        indices,
        k,
        v,
        k_cache.shape[0],
        k_cache.stride(0),
        k_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        k.stride(0),
        k.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
    )


__all__ = ["store_fp8_cache"]
