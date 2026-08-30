from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine.graph import GraphCaptureBuffer


def _batch(positions: torch.Tensor, rope_positions: torch.Tensor | None):
    size = positions.numel()
    return SimpleNamespace(
        padded_size=size,
        input_ids=torch.arange(size, dtype=torch.int32),
        out_loc=torch.arange(size, dtype=torch.int32) + 10,
        positions=positions,
        rope_positions=rope_positions,
        linear_table_idx=None,
    )


def test_graph_buffer_copies_three_axis_positions_without_stale_rows():
    buffer = GraphCaptureBuffer.init(4, 16, torch.device("cpu"))
    first_positions = torch.tensor([3, 4], dtype=torch.int32)
    first_rope = torch.tensor([[3, 4], [7, 8], [11, 12]], dtype=torch.int64)
    first = _batch(first_positions, first_rope)

    buffer.copy_from(first)
    captured = SimpleNamespace(padded_size=2)
    buffer.set_batch(captured)
    assert captured.rope_positions.data_ptr() == buffer.rope_positions.data_ptr()
    assert torch.equal(captured.rope_positions, first_rope)

    second_positions = torch.tensor([20, 21], dtype=torch.int32)
    second_rope = torch.tensor([[20, 21], [30, 31], [40, 41]], dtype=torch.int64)
    buffer.copy_from(_batch(second_positions, second_rope))
    assert torch.equal(captured.rope_positions, second_rope)


def test_graph_buffer_expands_scalar_text_positions_to_all_axes():
    buffer = GraphCaptureBuffer.init(3, 16, torch.device("cpu"))
    positions = torch.tensor([5, 6, 7], dtype=torch.int32)

    buffer.copy_from(_batch(positions, None))
    captured = SimpleNamespace(padded_size=3)
    buffer.set_batch(captured)

    assert torch.equal(
        captured.rope_positions,
        positions.to(torch.int64).expand(3, -1),
    )
