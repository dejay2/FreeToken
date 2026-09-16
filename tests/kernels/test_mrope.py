"""MRotaryEmbedding correctness: text degeneracy, section layouts, kernel-vs-fallback."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEAD = 256
ROT = 64
SECTION = (11, 11, 10)

# independent expectations for SECTION under each layout (half = 32 slots)
REFERENCE_TABLES = {
    "contiguous": [0] * 11 + [1] * 11 + [2] * 10,
    "interleaved": [1 if i % 3 == 1 and i < 33 else 2 if i % 3 == 2 and i < 30 else 0 for i in range(32)],
    "interleaved_glm": [0, 1, 2] * 10 + [0, 1],
}


def test_section_tables():
    from freetoken.layers.rotary import build_section_table

    for layout, expected in REFERENCE_TABLES.items():
        assert build_section_table(SECTION, layout).tolist() == expected, layout
    # GLM-V's [8,12,12]: T runs out first and H/W fill the tail
    assert build_section_table((8, 12, 12), "interleaved_glm").tolist() == [0, 1, 2] * 8 + [1, 1, 2, 1, 1, 2, 2, 2]
    with pytest.raises(ValueError, match="not representable"):
        build_section_table((8, 12, 12), "interleaved")
    with pytest.raises(ValueError, match="unknown mrope layout"):
        build_section_table(SECTION, "diagonal")


def _make(mrope: bool, layout: str = "interleaved"):
    from freetoken.layers.rotary import get_rope

    # the engine builds rope layers inside a cuda device context
    with torch.device("cuda"):
        return get_rope(
            head_dim=HEAD, rotary_dim=ROT, max_position=4096, base=1e7,
            mrope_section=SECTION if mrope else None, mrope_layout=layout,
        )


def _qk(n, heads=4, kv=2, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(n, heads * HEAD, generator=g, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(n, kv * HEAD, generator=g, dtype=torch.bfloat16, device="cuda")
    return q, k


@cuda
def test_text_positions_degenerate_to_1d_rope():
    n = 64
    pos1 = torch.arange(n, dtype=torch.int32, device="cuda")
    pos3 = pos1.unsqueeze(0).expand(3, -1).contiguous()
    q1, k1 = _qk(n)
    q3, k3 = q1.clone(), k1.clone()
    _make(False).forward(pos1, q1, k1)
    _make(True).forward(pos3, q3, k3)
    # 1-D path may use flashinfer while mrope uses the triton kernel: allclose, not bitwise
    assert torch.allclose(q1.float(), q3.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(k1.float(), k3.float(), atol=2e-2, rtol=2e-2)


def _torch_reference(pos3, q, k, table):
    """Per-axis freqs, per-slot axis selection, NeoX rotation, fp32 math."""
    half = ROT // 2
    inv = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, dtype=torch.float32, device="cuda") / ROT))
    sec = torch.tensor(table, dtype=torch.long, device="cuda")
    pos = pos3[sec, :].transpose(0, 1).float()          # [n, half]
    freqs = pos * inv.unsqueeze(0)                       # [n, half]
    cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)
    for t in (q, k):
        v = t.view(t.shape[0], -1, HEAD)
        lo, hi = v[..., :half].float(), v[..., half:ROT].float()
        v[..., :half] = (lo * cos - hi * sin).to(v.dtype)
        v[..., half:ROT] = (hi * cos + lo * sin).to(v.dtype)


@cuda
@pytest.mark.parametrize("layout", sorted(REFERENCE_TABLES))
def test_matches_reference_semantics(layout):
    n = 97
    g = torch.Generator(device="cuda").manual_seed(7)
    pos3 = torch.randint(0, 4000, (3, n), generator=g, dtype=torch.int32, device="cuda")
    q, k = _qk(n, seed=7)
    q_ref, k_ref = q.clone(), k.clone()
    _make(True, layout).forward(pos3, q, k)
    _torch_reference(pos3, q_ref, k_ref, REFERENCE_TABLES[layout])
    assert torch.allclose(q.float(), q_ref.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(k.float(), k_ref.float(), atol=2e-2, rtol=2e-2)


@cuda
def test_kernel_matches_torch_fallback():
    from freetoken.kernel.triton.rope import (
        apply_mrope_torch_fallback,
        apply_mrope_with_cos_sin_cache_inplace,
    )

    rope = _make(True)
    n = 33
    g = torch.Generator(device="cuda").manual_seed(3)
    pos3 = torch.randint(0, 4000, (3, n), generator=g, dtype=torch.int32, device="cuda")
    q, k = _qk(n, seed=3)
    q2, k2 = q.clone(), k.clone()
    cache = rope._cos_sin_cache
    sec = rope._section_table.cuda()
    apply_mrope_with_cos_sin_cache_inplace(pos3, q, k, HEAD, cache, sec)
    apply_mrope_torch_fallback(pos3, q2, k2, HEAD, cache, sec)
    assert torch.allclose(q.float(), q2.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(k.float(), k2.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_torch_fallback_rotates_both_halves_from_original_values(dtype):
    from freetoken.layers.rotary import _mrope_torch

    positions = torch.tensor([[0], [0], [0]], dtype=torch.int32)
    # A quarter turn: [a, b, c, d] must become [-c, -d, a, b].
    cache = torch.tensor([[0., 0., 1., 1.]])
    sections = torch.tensor([0, 1], dtype=torch.int32)
    q = torch.tensor([[1., 2., 3., 4., 9., 10.]], dtype=dtype)
    k = torch.tensor([[5., 6., 7., 8., 11., 12.]], dtype=dtype)
    _mrope_torch(positions, q, k, 6, cache, sections)
    torch.testing.assert_close(q, torch.tensor([[-3., -4., 1., 2., 9., 10.]], dtype=dtype))
    torch.testing.assert_close(k, torch.tensor([[-7., -8., 5., 6., 11., 12.]], dtype=dtype))
