"""EXL3 reconstruction reference and optional wheel-parity tests."""

from __future__ import annotations

import pytest
import torch


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _pack_trellis(states: torch.Tensor, k: int) -> torch.Tensor:
    """Pack 256 K-bit path values using the EXL3 trellis bit layout."""
    packed = torch.zeros((16 * k,), dtype=torch.uint16)
    for span in range(16):
        state_index = 16 * span
        packed_index = k * span
        bits_left = 32
        buffer = 0
        for offset in range(16):
            value = int(states[state_index + offset]) & ((1 << k) - 1)
            bits_left -= k
            buffer |= value << bits_left
            if bits_left <= 16:
                packed[packed_index] = (buffer >> 16) & 0xFFFF
                buffer = (buffer << 16) & 0xFFFFFFFF
                bits_left += 16
                packed_index += 1
    # ExLlamaV3's SWAP16 stores adjacent packed halfwords in the opposite order.
    swapped = packed.clone()
    swapped[0::2] = packed[1::2]
    swapped[1::2] = packed[0::2]
    return swapped


def _matrix_parts(in_features: int, out_features: int, k: int = 2):
    assert in_features % 16 == 0 and out_features % 16 == 0
    trellis = torch.zeros(
        (in_features // 16, out_features // 16, 16 * k), dtype=torch.int16
    )
    suh = torch.ones(in_features, dtype=torch.float16)
    svh = torch.ones(out_features, dtype=torch.float16)
    return trellis, suh, svh


def test_reference_zero_trellis_has_known_mul1_value_and_expected_shape():
    from freetoken.kernel.exl3 import reconstruct, reconstruct_reference

    trellis, suh, svh = _matrix_parts(128, 128)
    out = reconstruct_reference(trellis, suh, svh, k=2, codebook="mul1")
    with pytest.raises(ValueError, match="card-only|CUDA"):
        reconstruct(trellis, suh, svh, k=2, codebook="mul1")

    k_inv = torch.tensor([0x1EEE], dtype=torch.uint16).view(torch.float16)
    k_bias = torch.tensor([0xC931], dtype=torch.uint16).view(torch.float16)
    decoded_zero = (torch.tensor(1024.0, dtype=torch.float16) * k_inv + k_bias).half()
    expected_corner = (decoded_zero.float() * 128.0).squeeze()

    assert out.shape == (128, 128)
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    torch.testing.assert_close(out[0, 0].float(), expected_corner, rtol=1e-3, atol=1e-3)


def test_reference_handles_gate_up_and_down_orientations():
    from freetoken.kernel.exl3 import reconstruct_reference

    for in_features, out_features in ((128, 256), (256, 128)):
        trellis, suh, svh = _matrix_parts(in_features, out_features)
        out = reconstruct_reference(trellis, suh, svh, k=2, codebook="mul1")
        assert out.shape == (out_features, in_features)
        assert out.dtype == torch.bfloat16
        assert out.is_contiguous()


def test_reference_packs_and_decodes_sliding_trellis_states():
    from freetoken.kernel.exl3 import _decode_packed_states

    k = 2
    states = torch.arange(256, dtype=torch.int64) % (1 << k)
    trellis = _pack_trellis(states, k).view(1, 1, -1).view(torch.int16)
    decoded = _decode_packed_states(trellis, k)

    # Each extracted value is a 16-bit sliding window; only its low K bits are the
    # original trellis path state. The high bits are intentionally used by mul1.
    assert torch.equal(decoded[0] & ((1 << k) - 1), states)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda t, s, v: (t, s, v, 3, "mul1"), "last dimension"),
        (lambda t, s, v: (t, s, v, 2, "mcg"), "codebook"),
        (lambda t, s, v: (t, s[:64], v, 2, "mul1"), "suh shape"),
        (lambda t, s, v: (t, s, v.to(torch.float32), 2, "mul1"), "dtype"),
    ],
)
def test_reconstruction_rejects_bad_k_codebook_and_factor_shapes(mutator, match):
    from freetoken.kernel.exl3 import reconstruct_reference

    trellis, suh, svh = _matrix_parts(128, 128)
    args = mutator(trellis, suh, svh)
    with pytest.raises(ValueError, match=match):
        reconstruct_reference(args[0], args[1], args[2], k=args[3], codebook=args[4])


def test_reconstruction_rejects_non_128_divisible_dimensions_and_bad_buffers():
    from freetoken.kernel.exl3 import _validate_buffers, reconstruct, reconstruct_reference

    trellis = torch.zeros((1, 8, 32), dtype=torch.int16)
    suh = torch.ones(16, dtype=torch.float16)
    svh = torch.ones(128, dtype=torch.float16)
    with pytest.raises(ValueError, match="divisible by 128"):
        reconstruct_reference(trellis, suh, svh, k=2, codebook="mul1")

    with pytest.raises(ValueError, match="out shape"):
        _validate_buffers(
            torch.empty((128, 64), dtype=torch.bfloat16),
            None,
            device=torch.device("cpu"),
            in_features=128,
            out_features=128,
        )
    with pytest.raises(ValueError, match="work"):
        _validate_buffers(
            None,
            torch.empty((128, 128), dtype=torch.float32),
            device=torch.device("cpu"),
            in_features=128,
            out_features=128,
        )

    trellis, suh, svh = _matrix_parts(128, 128)
    with pytest.raises(ValueError, match="card-only|CUDA"):
        reconstruct(trellis, suh, svh, k=2, codebook="mul1")


@cuda
def test_extension_reconstruction_matches_reference_for_both_orientations():
    pytest.importorskip("exllamav3_ext")
    from freetoken.kernel.exl3 import reconstruct, reconstruct_reference

    device = torch.device("cuda")
    for in_features, out_features in ((128, 256), (256, 128)):
        trellis = torch.randint(
            -32768,
            32767,
            (in_features // 16, out_features // 16, 32),
            dtype=torch.int16,
            device=device,
        ).contiguous()
        suh = torch.randn(in_features, dtype=torch.float16, device=device).abs().contiguous()
        svh = torch.randn(out_features, dtype=torch.float16, device=device).abs().contiguous()
        out = torch.empty((out_features, in_features), dtype=torch.bfloat16, device=device)
        work = torch.empty((in_features, out_features), dtype=torch.float16, device=device)

        got = reconstruct(
            trellis,
            suh,
            svh,
            k=2,
            codebook="mul1",
            out=out,
            work=work,
        )
        expected = reconstruct_reference(trellis, suh, svh, k=2, codebook="mul1")
        torch.testing.assert_close(got.float().cpu(), expected.float(), rtol=2e-2, atol=0.08)


@cuda
def test_clamped_swiglu_order_is_gate_limit_sigmoid_then_clamped_up():
    from freetoken.layers import swiglu_clamp_and_mul

    limit = 10.0
    x = torch.tensor(
        [[-12.0, -10.0, -2.0, 0.0, 2.0, 10.0, 12.0,
          -12.0, -10.0, -2.0, 0.0, 2.0, 10.0, 12.0]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    got = swiglu_clamp_and_mul(x, alpha=1.0, limit=limit)
    gate = x[:, :7].float().clamp(max=limit)
    up = x[:, 7:].float().clamp(min=-limit, max=limit)
    expected = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
