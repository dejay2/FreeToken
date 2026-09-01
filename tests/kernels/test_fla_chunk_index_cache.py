"""``tensor_cache`` behind the fla chunk-index helpers is a four-entry LRU keyed on tensor
identity. A CUDA graph bakes in the address of whatever it read, so anything captured has to be
primed (capture cannot take the miss path) and held (later prefills evict the entry)."""

from __future__ import annotations

import torch

from freetoken.kernel.fla.chunk import CHUNK_SIZE
from freetoken.kernel.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prime_chunk_index_cache,
)


def _cu(total: int) -> torch.Tensor:
    return torch.tensor([0, total], dtype=torch.int32)


def test_priming_returns_the_objects_the_chunk_kernels_will_read():
    cu = _cu(3)

    pinned = prime_chunk_index_cache(cu, 3)

    # chunk.py asks for CHUNK_SIZE; chunk_o.py derives its own block from the row count
    assert pinned[0] is prepare_chunk_indices(cu, CHUNK_SIZE)
    assert pinned[1] is prepare_chunk_indices(cu, 16)
    assert pinned[2] is prepare_chunk_offsets(cu, CHUNK_SIZE)


def test_priming_three_graph_widths_overflows_the_identity_cache():
    widths = (2, 3, 4)
    cus = {width: _cu(width) for width in widths}
    pinned = {width: prime_chunk_index_cache(cus[width], width) for width in widths}

    # two entries per width against four slots: the first width is already gone, which is why
    # the graph runner has to hold its own reference rather than trust a later lookup
    assert prepare_chunk_indices(cus[2], CHUNK_SIZE) is not pinned[2][0]
    # the pinned tensor is still alive and still correct for the geometry it was captured with
    assert torch.equal(pinned[2][0], prepare_chunk_indices(cus[2], CHUNK_SIZE))
    assert pinned[2][0].tolist() == [[0, 0]]
