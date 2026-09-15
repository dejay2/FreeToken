"""Validation for vision towers backed by the fork's dense Linear operators."""

import torch


def require_dense_vision_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Reject encoded weights before any dtype conversion can discard their format."""
    suffix = name.rsplit('.', 1)[-1]
    if (suffix not in {'weight', 'bias', 'position_embedding'} and
            any(part in suffix for part in ('scale', 'packed', 'trellis', 'suh', 'svh', 'qweight', 'qzeros', 'g_idx'))):
        raise NotImplementedError(f"Quantized vision weight {name!r} is unsupported by the dense vision encoder")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError(f"Quantized vision weight {name!r} ({tensor.dtype}) is unsupported by the dense vision encoder")
    return tensor
