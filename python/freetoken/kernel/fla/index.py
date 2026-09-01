# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from freetoken.kernel.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


def prime_chunk_index_cache(
    cu_seqlens: torch.LongTensor, seq_len: int
) -> tuple[torch.LongTensor, ...]:
    """Build every chunk-index tensor a GDN prefill of ``seq_len`` tokens derives from
    ``cu_seqlens``, and return them so the caller can keep them alive.

    ``tensor_cache`` is a four-entry LRU keyed on tensor identity, which a CUDA graph cannot
    rely on: the miss path does a D2H ``.tolist()`` and a pageable H2D (both illegal under
    capture), and a graph that captured a read of one of these tensors keeps reading that
    address after a later prefill has evicted the entry and freed it. Priming makes the capture
    a pure hit; holding the result keeps the addresses valid for the graph's life. The values
    depend only on ``cu_seqlens``' contents, so a pinned tensor stays correct after eviction.
    """
    from freetoken.kernel.fla.chunk import CHUNK_SIZE

    # mirrors chunk_o.py's own block choice; chunk.py and chunk_delta_h.py use CHUNK_SIZE
    block = min(CHUNK_SIZE, max(16, triton.next_power_of_2(seq_len)))
    return (
        prepare_chunk_indices(cu_seqlens, CHUNK_SIZE),
        prepare_chunk_indices(cu_seqlens, block),
        prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE),
    )
