from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.graph import GraphCaptureBuffer
from freetoken.models.config import RotaryConfig
from freetoken.models.qwen4_exp.mrope import Qwen4MRoPE


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_replay_reads_two_distinct_three_axis_position_values():
    device = torch.device("cuda")
    config = SimpleNamespace(
        rotary_config=RotaryConfig(
            head_dim=256,
            rotary_dim=64,
            max_position=128,
            base=10_000_000,
            scaling=None,
        ),
        qwen4_args=SimpleNamespace(
            mrope_section=(11, 11, 10),
            mrope_interleaved=True,
        ),
    )
    rotary = Qwen4MRoPE(config)
    rotary._cos_sin_cache = rotary._cos_sin_cache.to(device)
    source_q = torch.randn(4, 2 * 256, dtype=torch.bfloat16, device=device)
    source_k = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
    work_q = torch.empty_like(source_q)
    work_k = torch.empty_like(source_k)
    first = torch.tensor([[2, 3, 4, 5], [2, 4, 6, 8], [2, 5, 8, 11]], device=device)
    second = torch.tensor([[12, 13, 14, 15], [20, 21, 22, 23], [30, 31, 32, 33]], device=device)

    def expected(positions: torch.Tensor):
        query, key = source_q.clone(), source_k.clone()
        rotary.forward(positions, query, key)
        return query, key

    expected_first = expected(first)
    expected_second = expected(second)
    buffer = GraphCaptureBuffer.init(4, 16, device)
    buffer.copy_from(_batch(torch.arange(4, dtype=torch.int32, device=device), first))
    captured = SimpleNamespace(padded_size=4)
    buffer.set_batch(captured)

    # Compile before capture, then capture stable buffer addresses.
    work_q.copy_(source_q)
    work_k.copy_(source_k)
    rotary.forward(captured.rope_positions, work_q, work_k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        work_q.copy_(source_q)
        work_k.copy_(source_k)
        rotary.forward(captured.rope_positions, work_q, work_k)

    buffer.copy_from(_batch(torch.arange(4, dtype=torch.int32, device=device), first))
    graph.replay()
    torch.testing.assert_close(work_q, expected_first[0], rtol=0, atol=0)
    torch.testing.assert_close(work_k, expected_first[1], rtol=0, atol=0)

    buffer.copy_from(_batch(torch.arange(4, dtype=torch.int32, device=device), second))
    graph.replay()
    torch.testing.assert_close(work_q, expected_second[0], rtol=0, atol=0)
    torch.testing.assert_close(work_k, expected_second[1], rtol=0, atol=0)
    assert not torch.equal(expected_first[0], expected_second[0])


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
