from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.config import RotaryConfig
from freetoken.models.qwen4_exp.mrope import build_mrope_positions
from freetoken.models.qwen4_exp.mrope import Qwen4MRoPE


def test_mrope_position_builder_matches_transformers_reference():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    class Reference:
        config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        get_vision_position_ids = Qwen3VLModel.get_vision_position_ids
        get_rope_index = Qwen3VLModel.get_rope_index

    input_ids = torch.arange(9)
    token_types = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0, 0])
    grid = torch.tensor([[1, 4, 4]])
    actual, delta = build_mrope_positions(input_ids, token_types, grid, 2)
    expected, expected_delta = Reference().get_rope_index(
        input_ids.view(1, -1),
        token_types.view(1, -1),
        image_grid_thw=grid,
    )
    assert torch.equal(actual, expected[:, 0])
    assert delta == int(expected_delta[0, 0])


def test_mrope_position_builder_supports_two_pictures_and_rejects_video():
    input_ids = torch.arange(13)
    token_types = torch.tensor([0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    positions, delta = build_mrope_positions(input_ids, token_types, grid, 2)
    assert positions.shape == (3, 13)
    assert delta == int(positions.max() + 1 - input_ids.numel())

    with pytest.raises(NotImplementedError, match="video"):
        build_mrope_positions(input_ids[:3], torch.tensor([0, 2, 2]), grid[:1], 2)


@pytest.mark.parametrize(
    "tokens,types,grid,merge,error",
    [
        (torch.arange(3), torch.tensor([0, 1]), torch.tensor([[1, 2, 2]]), 2, "length"),
        (torch.arange(3), torch.tensor([0, 1, 1]), torch.empty((0, 3)), 2, "more image groups"),
        (torch.arange(3), torch.tensor([0, 0, 0]), torch.tensor([[1, 2, 2]]), 2, "more images"),
        (torch.arange(3), torch.tensor([0, 1, 1]), torch.tensor([[1, 3, 2]]), 2, "divide"),
    ],
)
def test_mrope_position_builder_rejects_inconsistent_metadata(tokens, types, grid, merge, error):
    with pytest.raises((ValueError, NotImplementedError), match=error):
        build_mrope_positions(tokens, types, grid, merge)


def _rotary() -> Qwen4MRoPE:
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
    return Qwen4MRoPE(config)


def _reference_rotate(
    tensor: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor, head_size: int
) -> torch.Tensor:
    output = tensor.clone()
    view = output.view(output.shape[0], -1, head_size)
    half = cache.shape[1] // 2
    pair = torch.arange(half, device=positions.device)
    axis = torch.zeros(half, dtype=torch.long, device=positions.device)
    axis[(pair % 3 == 1) & (pair < 33)] = 1
    axis[(pair % 3 == 2) & (pair < 30)] = 2
    selected = positions.transpose(0, 1)[:, axis]
    dim = pair.view(1, -1).expand_as(selected)
    cos = cache[:, :half][selected, dim]
    sin = cache[:, half:][selected, dim]
    first = view[..., :half].float().clone()
    second = view[..., half : 2 * half].float().clone()
    view[..., :half] = (first * cos[:, None] - second * sin[:, None]).to(view.dtype)
    view[..., half : 2 * half] = (second * cos[:, None] + first * sin[:, None]).to(view.dtype)
    return output


def test_mrope_equal_axes_match_scalar_rotation():
    torch.manual_seed(13)
    rotary = _rotary()
    scalar = torch.tensor([2, 5, 9, 11], dtype=torch.int64)
    positions = scalar.expand(3, -1)
    query = torch.randn(4, 2 * 256)
    key = torch.randn(4, 256)
    expected_query = _reference_rotate(query, positions, rotary._cos_sin_cache, 256)
    expected_key = _reference_rotate(key, positions, rotary._cos_sin_cache, 256)

    rotary.forward(positions, query, key)

    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(key, expected_key)


def test_mrope_cpu_rotation_matches_axis_reference():
    torch.manual_seed(17)
    rotary = _rotary()
    positions = torch.tensor(
        [[2, 3, 4, 9], [2, 3, 5, 9], [2, 4, 6, 9]], dtype=torch.int64
    )
    query = torch.randn(4, 2 * 256)
    key = torch.randn(4, 256)
    expected_q = _reference_rotate(query, positions, rotary._cos_sin_cache, 256)
    expected_k = _reference_rotate(key, positions, rotary._cos_sin_cache, 256)
    rotary.forward(positions, query, key)
    torch.testing.assert_close(query, expected_q)
    torch.testing.assert_close(key, expected_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mrope_rtx_kernel_matches_cpu_reference():
    torch.manual_seed(23)
    rotary = _rotary()
    positions = torch.randint(0, 128, (3, 37), dtype=torch.int64)
    query = torch.randn(37, 24 * 256, dtype=torch.bfloat16)
    key = torch.randn(37, 2 * 256, dtype=torch.bfloat16)
    expected_q = _reference_rotate(query, positions, rotary._cos_sin_cache, 256)
    expected_k = _reference_rotate(key, positions, rotary._cos_sin_cache, 256)
    rotary._cos_sin_cache = rotary._cos_sin_cache.cuda()
    query_gpu = query.cuda()
    key_gpu = key.cuda()
    rotary.forward(positions.cuda(), query_gpu, key_gpu)
    torch.testing.assert_close(query_gpu.cpu(), expected_q, rtol=0, atol=2e-3)
    torch.testing.assert_close(key_gpu.cpu(), expected_k, rtol=0, atol=2e-3)
