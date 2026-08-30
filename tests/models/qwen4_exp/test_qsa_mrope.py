from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.qsa_sparse import _first_mrope_positions


def test_cross_forward_group_reads_its_first_coordinate_from_the_pending_ring():
    # This forward carries logical positions 2 and 3. The group closing at 3 began at
    # logical position 0 in the prior forward, so all three rotary axes must come from ring[0].
    md = SimpleNamespace(
        positions=torch.tensor([2, 3], dtype=torch.int32),
        rope_positions=torch.tensor([[20, 21], [30, 31], [40, 41]], dtype=torch.int64),
        token_to_req=torch.tensor([0, 0], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        ring_slots=torch.tensor([1], dtype=torch.int32),
    )
    ring = torch.zeros((3, 4, 3), dtype=torch.int64)
    ring[1, 0] = torch.tensor([5, 7, 9])
    ring[1, 1] = torch.tensor([6, 8, 10])

    actual = _first_mrope_positions(md, ring, compress_ratio=4)

    assert actual.shape == (3, 2)
    assert torch.equal(actual[:, 0], torch.zeros(3, dtype=torch.int64))
    assert torch.equal(actual[:, 1], ring[1, 0])


def test_whole_group_reads_its_first_coordinate_from_the_current_forward():
    md = SimpleNamespace(
        positions=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        rope_positions=torch.tensor(
            [[2, 2, 2, 2], [5, 5, 6, 6], [8, 9, 8, 9]], dtype=torch.int64
        ),
        token_to_req=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        ring_slots=torch.tensor([0], dtype=torch.int32),
    )
    ring = torch.full((1, 4, 3), -1, dtype=torch.int64)

    actual = _first_mrope_positions(md, ring, compress_ratio=4)

    assert torch.equal(actual[:, 3], md.rope_positions[:, 0])


def _reference_norm_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    heads: int,
    section: tuple[int, int, int],
) -> torch.Tensor:
    rows, dim = x.shape
    rotary_half = cache.shape[1] // 2
    token = torch.arange(rows, device=x.device) // heads
    rotary_dim = 2 * rotary_half
    rotary_index = torch.arange(rotary_dim, device=x.device)
    pair = rotary_index % rotary_half
    axis = torch.zeros(rotary_dim, dtype=torch.long, device=x.device)
    axis[(pair % 3 == 1) & (pair < section[1] * 3)] = 1
    axis[(pair % 3 == 2) & (pair < section[2] * 3)] = 2
    selected = positions[axis, token[:, None]].squeeze(-1)
    cos = cache[selected, pair]
    sin = cache[selected, rotary_half + pair]

    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    normalized = normalized * (weight.float() + 1)
    partner = torch.where(
        rotary_index < rotary_half,
        rotary_index + rotary_half,
        rotary_index - rotary_half,
    )
    rotated = normalized.clone()
    rotated[:, :rotary_dim] = normalized[:, :rotary_dim] * cos + torch.where(
        rotary_index < rotary_half,
        -1.0,
        1.0,
    ) * normalized[:, partner] * sin
    return rotated.to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_split_group_picture_positions_remain_private_until_each_qsa_layer_consumes_them():
    from freetoken.kernel.triton.qsa import qsa_store_rows

    device = torch.device("cuda")
    md = SimpleNamespace(
        positions=torch.tensor([2, 3, 4, 5], dtype=torch.int32, device=device),
        rope_positions=torch.tensor(
            [[20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            dtype=torch.int64,
            device=device,
        ),
        token_to_req=torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=device),
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32, device=device),
        ring_slots=torch.tensor([1], dtype=torch.int32, device=device),
    )
    rings = torch.zeros((2, 3, 4, 3), dtype=torch.int64, device=device)
    prior = torch.tensor([5, 7, 9], dtype=torch.int64, device=device)
    rings[:, 1, 0].copy_(prior)
    ring_rows = torch.tensor([6, 7, 4, 5], dtype=torch.int32, device=device)

    # Layer 0 completes compression, then writes this forward's last coordinates. Its ring
    # wraps position 4 onto row 0, but layer 1 must still own the prior row 0.
    qsa_store_rows(rings[0], ring_rows, md.rope_positions.transpose(0, 1).contiguous())
    layer_zero = _first_mrope_positions(md, rings[0], compress_ratio=4)
    layer_one = _first_mrope_positions(md, rings[1], compress_ratio=4)

    assert not torch.equal(layer_zero[:, 1], prior)
    assert torch.equal(layer_one[:, 1], prior)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_picture_index_rotation_matches_axis_reference_and_text_equivalence():
    from freetoken.kernel.triton.qsa import qsa_index_norm_rope

    torch.manual_seed(41)
    rows, heads, dim = 7, 4, 128
    rotary_half = 32
    section = (11, 11, 10)
    x = torch.randn(rows * heads, dim, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(dim, dtype=torch.bfloat16, device="cuda") * 0.01
    position = torch.randint(0, 64, (rows,), dtype=torch.int64, device="cuda")
    positions = torch.randint(0, 64, (3, rows), dtype=torch.int64, device="cuda")
    frequency = torch.randn(64, rotary_half, dtype=torch.float32, device="cuda")
    cache = torch.cat((frequency.cos(), frequency.sin()), dim=-1).contiguous()

    actual = torch.empty_like(x)
    qsa_index_norm_rope(
        x,
        positions,
        cache,
        weight,
        1e-6,
        actual,
        heads=heads,
        mrope_section=section,
    )
    expected = _reference_norm_rope(x, positions, cache, weight, 1e-6, heads, section)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-3)

    scalar = torch.empty_like(x)
    broadcast = torch.empty_like(x)
    qsa_index_norm_rope(x, position, cache, weight, 1e-6, scalar, heads=heads)
    qsa_index_norm_rope(
        x,
        position.expand(3, -1),
        cache,
        weight,
        1e-6,
        broadcast,
        heads=heads,
        mrope_section=section,
    )
    torch.testing.assert_close(broadcast, scalar, rtol=0, atol=0)
