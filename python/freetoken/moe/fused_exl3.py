"""Reconstruct-first EXL3 routed-expert operation.

This proof keeps the compressed EXL3 rows in the slot cache, rebuilds at most eight
selected experts into BF16 scratch banks, and reuses FreeToken's ordinary BF16 grouped
MoE operations. It is intentionally a correctness path, not the later fused EXL3 kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from freetoken.kernel import exl3 as _exl3_kernel
from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl


_MAX_TOP_K = 64
_MAX_RECONSTRUCT_EXPERTS = 8
_EXL3_K = 2
_EXL3_CODEBOOK = "mul1"


@dataclass
class Exl3Scratch:
    """Fixed-address buffers shared by the serial EXL3 layers on one card."""

    device: torch.device
    hidden_size: int
    intermediate_size: int
    max_tokens: int
    chunk_experts: int
    max_top_k: int
    gate_up: torch.Tensor
    down: torch.Tensor
    reconstruct_work: torch.Tensor
    input_buffer: torch.Tensor
    output_accumulator: torch.Tensor
    route_ids: torch.Tensor
    route_weights: torch.Tensor


def prepare_exl3_scratch(
    *,
    device,
    hidden_size,
    intermediate_size,
    max_tokens=8192,
    chunk_experts=_MAX_RECONSTRUCT_EXPERTS,
) -> Exl3Scratch:
    """Allocate the one reusable reconstruct-first workspace.

    The large buffers are allocated once, before graph setup. Route storage is one
    contiguous one-dimensional allocation so each call can take a contiguous ``[M, top_k]``
    view without allocating a new tensor when the request width changes.
    """
    device = torch.device(device)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    max_tokens = int(max_tokens)
    chunk_experts = int(chunk_experts)
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("EXL3 scratch dimensions must be positive")
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError(
            "EXL3 scratch dimensions must be divisible by 128, "
            f"got hidden_size={hidden_size}, intermediate_size={intermediate_size}"
        )
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if not 1 <= chunk_experts <= _MAX_RECONSTRUCT_EXPERTS:
        raise ValueError(
            "chunk_experts must be in 1.."
            f"{_MAX_RECONSTRUCT_EXPERTS}, got {chunk_experts}"
        )

    return Exl3Scratch(
        device=device,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_tokens=max_tokens,
        chunk_experts=chunk_experts,
        max_top_k=_MAX_TOP_K,
        gate_up=torch.empty(
            (chunk_experts, 2 * intermediate_size, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        down=torch.empty(
            (chunk_experts, hidden_size, intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        # Gate/up reconstruct as [H, I]; down reuses the same storage viewed as [I, H].
        reconstruct_work=torch.empty(
            (hidden_size, intermediate_size), dtype=torch.float16, device=device
        ),
        input_buffer=torch.empty(
            (max_tokens, hidden_size), dtype=torch.bfloat16, device=device
        ),
        output_accumulator=torch.empty(
            (max_tokens, hidden_size), dtype=torch.bfloat16, device=device
        ),
        route_ids=torch.empty(
            max_tokens * _MAX_TOP_K, dtype=torch.int32, device=device
        ),
        route_weights=torch.empty(
            max_tokens * _MAX_TOP_K, dtype=torch.float32, device=device
        ),
    )


def require_exl3_gpu_only(*, device, decode_target="gpu") -> None:
    """Reject CPU and hybrid EXL3 execution before any expert work starts."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            "EXL3 routed experts are card-only; a CUDA device is required "
            f"(got {device})"
        )
    if decode_target != "gpu":
        raise ValueError(
            "EXL3 routed experts support only GPU decode; "
            f"decode_target must be 'gpu' (got {decode_target!r})"
        )


def _validate_banks(
    banks: tuple[torch.Tensor, ...],
    *,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if len(banks) != 9:
        raise ValueError(
            "EXL3 expects nine banks in order "
            "gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh, "
            f"down_trellis, down_suh, down_svh (got {len(banks)})"
        )
    if not all(isinstance(bank, torch.Tensor) for bank in banks):
        raise ValueError("all EXL3 banks must be torch.Tensor values")
    banks = tuple(banks)
    if any(bank.device != device for bank in banks):
        raise ValueError("EXL3 banks must be on the same device as hidden_states")
    if any(not bank.is_contiguous() for bank in banks):
        raise ValueError("EXL3 banks must be contiguous")

    (
        gate_trellis,
        gate_suh,
        gate_svh,
        up_trellis,
        up_suh,
        up_svh,
        down_trellis,
        down_suh,
        down_svh,
    ) = banks
    expected = {
        "gate_trellis": (4, torch.int16),
        "gate_suh": (2, torch.float16),
        "gate_svh": (2, torch.float16),
        "up_trellis": (4, torch.int16),
        "up_suh": (2, torch.float16),
        "up_svh": (2, torch.float16),
        "down_trellis": (4, torch.int16),
        "down_suh": (2, torch.float16),
        "down_svh": (2, torch.float16),
    }
    for name, bank in zip(expected, banks):
        rank, dtype = expected[name]
        if bank.dim() != rank or bank.dtype != dtype:
            raise ValueError(
                f"EXL3 bank {name!r} must have rank {rank} and dtype {dtype}, "
                f"got shape {tuple(bank.shape)} and dtype {bank.dtype}"
            )

    slots = gate_trellis.shape[0]
    if any(bank.shape[0] != slots for bank in banks):
        raise ValueError("all EXL3 banks must have the same slot count")
    if gate_trellis.shape[1:] != (
        hidden_size // 16,
        intermediate_size // 16,
        16 * _EXL3_K,
    ):
        raise ValueError(
            "gate_trellis shape does not match the configured expert dimensions: "
            f"got {tuple(gate_trellis.shape)}"
        )
    if up_trellis.shape[1:] != gate_trellis.shape[1:]:
        raise ValueError("gate_trellis and up_trellis must have the same shape")
    if down_trellis.shape[1:] != (
        intermediate_size // 16,
        hidden_size // 16,
        16 * _EXL3_K,
    ):
        raise ValueError(
            "down_trellis shape does not match the configured expert dimensions: "
            f"got {tuple(down_trellis.shape)}"
        )
    for name, bank, width in (
        ("gate_suh", gate_suh, hidden_size),
        ("up_suh", up_suh, hidden_size),
        ("down_svh", down_svh, hidden_size),
        ("gate_svh", gate_svh, intermediate_size),
        ("up_svh", up_svh, intermediate_size),
        ("down_suh", down_suh, intermediate_size),
    ):
        if bank.shape[1:] != (width,):
            raise ValueError(
                f"EXL3 bank {name!r} must have row shape [{width}], got {tuple(bank.shape)}"
            )
    return banks


def _route_views(
    scratch: Exl3Scratch,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    top_k = topk_ids.shape[1]
    count = rows * top_k
    route_ids = scratch.route_ids[:count].view(rows, top_k)
    route_weights = scratch.route_weights[:count].view(rows, top_k)
    route_ids.zero_()
    route_weights.zero_()
    return route_weights, route_ids


def _unique_routed_slots(topk_ids: torch.Tensor, slot_count: int) -> list[int]:
    """Return sorted valid route ids; this host read is safe because proof graphs are off."""
    unique = torch.unique(topk_ids)
    values = [int(value) for value in unique[unique >= 0].detach().cpu().tolist()]
    invalid = [value for value in values if value >= slot_count]
    if invalid:
        raise ValueError(
            f"EXL3 route id {invalid[0]} is outside the available bank slots [0, {slot_count})"
        )
    return values


def _reconstruct_chunk(
    banks: tuple[torch.Tensor, ...],
    slots: list[int],
    scratch: Exl3Scratch,
) -> None:
    (
        gate_trellis,
        gate_suh,
        gate_svh,
        up_trellis,
        up_suh,
        up_svh,
        down_trellis,
        down_suh,
        down_svh,
    ) = banks
    hidden_size = scratch.hidden_size
    intermediate_size = scratch.intermediate_size
    work_gate_up = scratch.reconstruct_work
    work_down = scratch.reconstruct_work.view(intermediate_size, hidden_size)

    for local, slot in enumerate(slots):
        _exl3_kernel.reconstruct(
            gate_trellis[slot],
            gate_suh[slot],
            gate_svh[slot],
            k=_EXL3_K,
            codebook=_EXL3_CODEBOOK,
            out=scratch.gate_up[local, :intermediate_size, :],
            work=work_gate_up,
        )
        _exl3_kernel.reconstruct(
            up_trellis[slot],
            up_suh[slot],
            up_svh[slot],
            k=_EXL3_K,
            codebook=_EXL3_CODEBOOK,
            out=scratch.gate_up[local, intermediate_size:, :],
            work=work_gate_up,
        )
        _exl3_kernel.reconstruct(
            down_trellis[slot],
            down_suh[slot],
            down_svh[slot],
            k=_EXL3_K,
            codebook=_EXL3_CODEBOOK,
            out=scratch.down[local],
            work=work_down,
        )


def _run_bf16_experts(
    hidden_states: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    is_prefill: bool,
    activation: str,
    apply_router_weight_on_input: bool,
    hidden_act_alpha: float,
    swiglu_limit: float | None,
) -> torch.Tensor:
    fn = fused_experts_impl if is_prefill else fused_experts_decode_impl
    kwargs = {
        "activation": activation,
        "apply_router_weight_on_input": apply_router_weight_on_input,
    }
    if activation == "swiglu_clamp":
        kwargs["hidden_act_alpha"] = hidden_act_alpha
        kwargs["swiglu_limit"] = 10.0 if swiglu_limit is None else float(swiglu_limit)
    return fn(hidden_states, gate_up, down, topk_weights, topk_ids, **kwargs)


def fused_experts_exl3(
    hidden_states: torch.Tensor,
    banks,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    is_prefill: bool,
    activation: str,
    apply_router_weight_on_input: bool,
    swiglu_limit: float | None,
    hidden_act_alpha: float,
    scratch: Exl3Scratch,
) -> torch.Tensor:
    """Reconstruct routed EXL3 experts in chunks, then run the BF16 grouped path.

    CUDA graphs are deliberately disabled for this proof. The sorted expert list is read on
    the host and the ordinary BF16 operations still own their temporary activation buffers;
    the large EXL3 buffers themselves stay at fixed addresses and are reused layer to layer.
    """
    if not isinstance(scratch, Exl3Scratch):
        raise ValueError("fused_experts_exl3 requires an Exl3Scratch workspace")
    require_exl3_gpu_only(device=hidden_states.device)
    if hidden_states.device != scratch.device:
        raise ValueError(
            f"EXL3 scratch is on {scratch.device}, hidden_states are on {hidden_states.device}"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(
            f"EXL3 reconstruct-first experts require BF16 hidden states, got {hidden_states.dtype}"
        )
    if hidden_states.dim() != 2:
        raise ValueError(f"hidden_states must be rank 2, got rank {hidden_states.dim()}")
    rows, hidden_size = hidden_states.shape
    if hidden_size != scratch.hidden_size:
        raise ValueError(
            f"hidden size mismatch: hidden_states has {hidden_size}, "
            f"scratch expects {scratch.hidden_size}"
        )
    if rows > scratch.max_tokens:
        raise ValueError(
            f"EXL3 scratch supports at most {scratch.max_tokens} tokens, got {rows}"
        )
    if not isinstance(topk_weights, torch.Tensor) or not isinstance(topk_ids, torch.Tensor):
        raise ValueError("topk_weights and topk_ids must be torch.Tensor values")
    if topk_weights.shape != topk_ids.shape or topk_ids.dim() != 2:
        raise ValueError(
            f"topk routing shapes must match and be rank 2, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"topk_ids must have dtype torch.int32, got {topk_ids.dtype}")
    if topk_weights.dtype != torch.float32:
        raise ValueError(
            f"topk_weights must have dtype torch.float32, got {topk_weights.dtype}"
        )
    if topk_ids.device != hidden_states.device or topk_weights.device != hidden_states.device:
        raise ValueError("routing tensors must be on the same device as hidden_states")
    top_k = topk_ids.shape[1]
    if top_k <= 0 or top_k > scratch.max_top_k:
        raise ValueError(
            f"EXL3 scratch supports top_k in 1..{scratch.max_top_k}, got {top_k}"
        )

    banks = _validate_banks(
        tuple(banks),
        hidden_size=scratch.hidden_size,
        intermediate_size=scratch.intermediate_size,
        device=hidden_states.device,
    )
    slots = _unique_routed_slots(topk_ids, banks[0].shape[0])

    input_buffer = scratch.input_buffer[:rows]
    output = scratch.output_accumulator[:rows]
    output.zero_()
    # The BF16 prompt path overwrites its input. Copying once for decode too gives both
    # phases a contiguous fixed-address input and keeps the caller's tensor untouched.
    input_buffer.copy_(hidden_states)
    route_weights, route_ids = _route_views(
        scratch, topk_weights, topk_ids, rows=rows
    )

    for start in range(0, len(slots), scratch.chunk_experts):
        chunk = slots[start : start + scratch.chunk_experts]
        _reconstruct_chunk(banks, chunk, scratch)

        # Reuse the same contiguous route views for every chunk. Routes not in this
        # chunk keep weight zero and the valid placeholder id zero.
        route_ids.zero_()
        route_weights.zero_()
        for local, slot in enumerate(chunk):
            mask = topk_ids == slot
            route_ids[mask] = local
            route_weights[mask] = topk_weights[mask]

        if is_prefill:
            # ``fused_experts_impl`` writes its input in place; restore the original rows
            # before every chunk because the later shared expert still consumes them.
            input_buffer.copy_(hidden_states)
        chunk_out = _run_bf16_experts(
            input_buffer,
            scratch.gate_up[: len(chunk)],
            scratch.down[: len(chunk)],
            route_weights,
            route_ids,
            is_prefill=is_prefill,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            hidden_act_alpha=hidden_act_alpha,
            swiglu_limit=swiglu_limit,
        )
        output.add_(chunk_out)

    return output


__all__ = [
    "Exl3Scratch",
    "fused_experts_exl3",
    "prepare_exl3_scratch",
    "require_exl3_gpu_only",
]
