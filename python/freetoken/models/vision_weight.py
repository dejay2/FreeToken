"""Validation for vision towers backed by the fork's dense Linear operators."""

import torch

# Component suffixes of an EXL3-packed linear (see kernel/exl3_linear.py _COMPONENTS plus bias).
_EXL3_COMPONENTS = ('trellis', 'suh', 'svh', 'mul1', 'bias')
# The Qwen3.8-Flash-Next EXL3 vision modules that are actually built as Exl3Linear (task 7):
# attn.proj and both MLP/merger projections. patch_embed and attn.qkv stay bf16 even on an
# EXL3 checkpoint, so their tensors still go through the ordinary dtype check below.
_EXL3_LINEAR_MODULES = (
    '.attn.proj.', '.mlp.linear_fc1.', '.mlp.linear_fc2.',
    '.merger.linear_fc1.', '.merger.linear_fc2.',
)


def require_dense_vision_weight(name: str, tensor: torch.Tensor, *, exl3: bool = False) -> torch.Tensor:
    """Reject encoded weights before any dtype conversion can discard their format.

    ``exl3=True`` lets the packed components of the named EXL3 vision linears (see
    ``_EXL3_LINEAR_MODULES``) through unchanged -- everything else (patch_embed, a non-EXL3
    checkpoint, or any other encoded suffix) still goes through the dense-only check.
    """
    suffix = name.rsplit('.', 1)[-1]
    if exl3 and suffix in _EXL3_COMPONENTS and any(module in f".{name}." for module in _EXL3_LINEAR_MODULES):
        return tensor
    # Gemma's std_scale is a dense standardisation parameter, not a quantizer
    # scale. It must still pass the dtype check below like every other tensor.
    if (suffix not in {'weight', 'bias', 'position_embedding', 'std_scale'} and
            any(part in suffix for part in ('scale', 'packed', 'trellis', 'suh', 'svh', 'qweight', 'qzeros', 'g_idx'))):
        raise NotImplementedError(f"Quantized vision weight {name!r} is unsupported by the dense vision encoder")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError(f"Quantized vision weight {name!r} ({tensor.dtype}) is unsupported by the dense vision encoder")
    return tensor
