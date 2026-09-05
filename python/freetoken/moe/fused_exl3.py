"""EXL3 routed-expert operations.

The default reconstruct-first path keeps the compressed rows in the slot cache, rebuilds at most
8 selected experts into BF16 scratch banks, and reuses FreeToken's ordinary BF16 grouped
MoE operations.  The opt-in packed path calls ExLlamaV3's ``exl3_mgemm`` three times and falls
back to reconstruction when its compiled shape or route limits do not cover a call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from freetoken.kernel import exl3 as _exl3_kernel
from freetoken.kernel.exl3_mgemm import EXL3_MGEMM_MAX_INDICES
from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl
from freetoken.utils import init_logger


logger = init_logger(__name__)

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
    decode_max_tokens: int
    decode_top_k: int
    gate_up: torch.Tensor
    down: torch.Tensor
    reconstruct_work: torch.Tensor
    input_buffer: torch.Tensor
    output_accumulator: torch.Tensor
    route_ids: torch.Tensor
    route_weights: torch.Tensor
    slot_list: torch.Tensor
    slot_count: torch.Tensor
    slot_sort_ids: torch.Tensor
    slot_sort_indices: torch.Tensor
    slot_candidates: torch.Tensor
    slot_unique: torch.Tensor
    slot_valid: torch.Tensor
    slot_mask: torch.Tensor
    reconstruct_slots: torch.Tensor
    reconstruct_banks: tuple[torch.Tensor, ...]
    decode_intermediate_cache1: torch.Tensor
    decode_intermediate_cache2: torch.Tensor
    decode_intermediate_cache3: torch.Tensor
    decode_output: torch.Tensor
    # Packed EXL3 state is optional so the default reconstruct-first boot pays no extra VRAM.
    mgemm_decode_scratch: object | None = None
    mgemm_prefill_scratch: object | None = None
    # Values hold the source bank tensors, keeping pointer tables valid until the cache is
    # rebuilt.  The key is based on data pointers/shape, so each cache or overlap buffer gets one.
    mgemm_tables: dict[tuple, object] = field(default_factory=dict)
    # Static wheel/shape failures are decided during eager warm-up and skip packed attempts during
    # graph capture and later calls. Route-list capacity failures remain per-call fallbacks.
    mgemm_disabled: bool = False


def prepare_exl3_scratch(
    *,
    device,
    hidden_size,
    intermediate_size,
    max_tokens=8192,
    chunk_experts=_MAX_RECONSTRUCT_EXPERTS,
    decode_max_tokens=1,
    enable_mgemm=False,
) -> Exl3Scratch:
    """Allocate the one reusable EXL3 workspace.

    The large buffers are allocated once, before graph setup. Route storage is one
    contiguous one-dimensional allocation so each call can take a contiguous ``[M, top_k]``
    view without allocating a new tensor when the request width changes.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    max_tokens = int(max_tokens)
    chunk_experts = int(chunk_experts)
    decode_max_tokens = int(decode_max_tokens)
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("EXL3 scratch dimensions must be positive")
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError(
            "EXL3 scratch dimensions must be divisible by 128, "
            f"got hidden_size={hidden_size}, intermediate_size={intermediate_size}"
        )
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if decode_max_tokens <= 0:
        raise ValueError(f"decode_max_tokens must be positive, got {decode_max_tokens}")
    if not 1 <= chunk_experts <= _MAX_RECONSTRUCT_EXPERTS:
        raise ValueError(
            "chunk_experts must be in 1.."
            f"{_MAX_RECONSTRUCT_EXPERTS}, got {chunk_experts}"
        )

    trellis_shapes = (
        (chunk_experts, hidden_size // 16, intermediate_size // 16, 16 * _EXL3_K),
        (chunk_experts, hidden_size // 16, intermediate_size // 16, 16 * _EXL3_K),
        (chunk_experts, intermediate_size // 16, hidden_size // 16, 16 * _EXL3_K),
    )
    reconstruct_banks = (
        torch.empty(trellis_shapes[0], dtype=torch.int16, device=device),
        torch.empty((chunk_experts, hidden_size), dtype=torch.float16, device=device),
        torch.empty((chunk_experts, intermediate_size), dtype=torch.float16, device=device),
        torch.empty(trellis_shapes[1], dtype=torch.int16, device=device),
        torch.empty((chunk_experts, hidden_size), dtype=torch.float16, device=device),
        torch.empty((chunk_experts, intermediate_size), dtype=torch.float16, device=device),
        torch.empty(trellis_shapes[2], dtype=torch.int16, device=device),
        torch.empty((chunk_experts, intermediate_size), dtype=torch.float16, device=device),
        torch.empty((chunk_experts, hidden_size), dtype=torch.float16, device=device),
    )
    scratch = Exl3Scratch(
        device=device,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_tokens=max_tokens,
        chunk_experts=chunk_experts,
        max_top_k=_MAX_TOP_K,
        decode_max_tokens=decode_max_tokens,
        decode_top_k=chunk_experts,
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
        # Decode routing stays at fixed shape: sorting and compaction use sentinels rather
        # than a variable-length unique result, so graph replay never asks the allocator for
        # a new route tensor.
        slot_list=torch.empty(chunk_experts, dtype=torch.int32, device=device),
        slot_count=torch.empty(1, dtype=torch.int32, device=device),
        slot_sort_ids=torch.empty(chunk_experts, dtype=torch.int32, device=device),
        slot_sort_indices=torch.empty(chunk_experts, dtype=torch.int64, device=device),
        slot_candidates=torch.empty(chunk_experts, dtype=torch.int32, device=device),
        slot_unique=torch.empty(chunk_experts, dtype=torch.bool, device=device),
        slot_valid=torch.empty(chunk_experts, dtype=torch.bool, device=device),
        slot_mask=torch.empty(chunk_experts, dtype=torch.bool, device=device),
        reconstruct_slots=torch.empty(chunk_experts, dtype=torch.int32, device=device),
        reconstruct_banks=reconstruct_banks,
        decode_intermediate_cache1=torch.empty(
            (decode_max_tokens, chunk_experts, 2 * intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        decode_intermediate_cache2=torch.empty(
            (decode_max_tokens * chunk_experts, intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        decode_intermediate_cache3=torch.empty(
            (decode_max_tokens, chunk_experts, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        decode_output=torch.empty(
            (decode_max_tokens, hidden_size), dtype=torch.bfloat16, device=device
        ),
    )
    if enable_mgemm and device.type == "cuda":
        # The packed route form is capped at 128 indices.  Keep both phase arenas bounded at
        # that size; prompt groups tile here even though the kernel itself has no size-M ceiling.
        from freetoken.kernel.exl3_mgemm import prepare_exl3_mgemm_scratch

        max_features = max(hidden_size, intermediate_size)
        mgemm_rows = EXL3_MGEMM_MAX_INDICES
        scratch.mgemm_decode_scratch = prepare_exl3_mgemm_scratch(
            device=device,
            max_rows=mgemm_rows,
            max_features=max_features,
            preallocate_fused=True,
        )
        scratch.mgemm_prefill_scratch = prepare_exl3_mgemm_scratch(
            device=device,
            max_rows=mgemm_rows,
            max_features=max_features,
            preallocate_fused=True,
        )
    return scratch


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
    """Return sorted valid route ids for the eager, multi-chunk prompt path."""
    unique = torch.unique(topk_ids)
    values = [int(value) for value in unique[unique >= 0].detach().cpu().tolist()]
    invalid = [value for value in values if value >= slot_count]
    if invalid:
        raise ValueError(
            f"EXL3 route id {invalid[0]} is outside the available bank slots [0, {slot_count})"
        )
    return values


def _unique_routed_slots_device(
    topk_ids: torch.Tensor,
    scratch: Exl3Scratch,
    slot_capacity: int,
) -> None:
    """Fill the fixed decode slot list without reading routing data on the host.

    The decode contract is at most ``chunk_experts`` routes.  Sorting into a fixed-size
    candidate array, replacing duplicate/invalid entries with a sentinel, and sorting once
    more compacts the unique ids while keeping every tensor shape stable for graph capture.
    """
    route_count = topk_ids.numel()
    if not 1 <= route_count <= scratch.chunk_experts:
        raise ValueError(
            "EXL3 graph-safe decode needs rows * top_k in 1.."
            f"{scratch.chunk_experts}, got {route_count}"
        )
    flat_ids = topk_ids.reshape(-1)
    sorted_ids = scratch.slot_sort_ids[:route_count]
    sort_indices = scratch.slot_sort_indices[:route_count]
    torch.sort(flat_ids, dim=0, out=(sorted_ids, sort_indices))

    valid = scratch.slot_valid[:route_count]
    torch.ge(sorted_ids, 0, out=valid)
    capacity_mask = scratch.slot_mask[:route_count]
    torch.lt(sorted_ids, slot_capacity, out=capacity_mask)
    valid.logical_and_(capacity_mask)

    unique = scratch.slot_unique[:route_count]
    unique.zero_()
    unique[0].copy_(valid[0])
    if route_count > 1:
        torch.ne(sorted_ids[1:], sorted_ids[:-1], out=unique[1:])
        unique.logical_and_(valid)

    candidates = scratch.slot_candidates
    candidates.fill_(slot_capacity)
    candidates[:route_count].copy_(sorted_ids)
    capacity_mask.copy_(unique)
    capacity_mask.logical_not_()
    candidates[:route_count].masked_fill_(capacity_mask, slot_capacity)
    torch.sort(candidates, dim=0, out=(scratch.slot_list, scratch.slot_sort_indices))
    torch.sum(unique, 0, keepdim=True, dtype=torch.int32, out=scratch.slot_count)


def _route_views_device(
    scratch: Exl3Scratch,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    rows: int,
    slot_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw cache slots to compact reconstructed rows using fixed device buffers."""
    route_weights, route_ids = _route_views(scratch, topk_weights, topk_ids, rows=rows)
    route_count = rows * topk_ids.shape[1]
    valid = scratch.slot_valid[:route_count].view(rows, topk_ids.shape[1])
    torch.ge(topk_ids, 0, out=valid)
    route_mask = scratch.slot_mask[:route_count].view(rows, topk_ids.shape[1])
    torch.lt(topk_ids, slot_capacity, out=route_mask)
    valid.logical_and_(route_mask)

    route_weights.copy_(topk_weights)
    torch.logical_not(valid, out=route_mask)
    route_weights.masked_fill_(route_mask, 0.0)
    torch.searchsorted(
        scratch.slot_list,
        topk_ids,
        out_int32=True,
        out=route_ids,
    )
    route_ids.clamp_min_(0)
    route_ids.clamp_max_(scratch.chunk_experts - 1)
    return route_weights, route_ids


def _reconstruct_decode(
    banks: tuple[torch.Tensor, ...],
    scratch: Exl3Scratch,
) -> None:
    """Gather all fixed decode rows, then reconstruct them with stable tensor addresses."""
    slot_capacity = banks[0].shape[0]
    if slot_capacity <= 0:
        raise ValueError("EXL3 decode banks must contain at least one slot")
    torch.clamp(
        scratch.slot_list,
        max=slot_capacity - 1,
        out=scratch.reconstruct_slots,
    )
    for bank, staging in zip(banks, scratch.reconstruct_banks):
        torch.index_select(bank, 0, scratch.reconstruct_slots, out=staging)

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
    ) = scratch.reconstruct_banks
    work_gate_up = scratch.reconstruct_work
    work_down = scratch.reconstruct_work.view(scratch.intermediate_size, scratch.hidden_size)
    for local in range(scratch.chunk_experts):
        _exl3_kernel.reconstruct(
            gate_trellis[local],
            gate_suh[local],
            gate_svh[local],
            k=_EXL3_K,
            codebook=_EXL3_CODEBOOK,
            out=scratch.gate_up[local, : scratch.intermediate_size, :],
            work=work_gate_up,
        )
        _exl3_kernel.reconstruct(
            up_trellis[local],
            up_suh[local],
            up_svh[local],
            k=_EXL3_K,
            codebook=_EXL3_CODEBOOK,
            out=scratch.gate_up[local, scratch.intermediate_size :, :],
            work=work_gate_up,
        )
        _exl3_kernel.reconstruct(
            down_trellis[local],
            down_suh[local],
            down_svh[local],
            k=_EXL3_K,
            codebook=_EXL3_CODEBOOK,
            out=scratch.down[local],
            work=work_down,
        )


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
    workspace=None,
) -> torch.Tensor:
    fn = fused_experts_impl if is_prefill else fused_experts_decode_impl
    kwargs = {
        "activation": activation,
        "apply_router_weight_on_input": apply_router_weight_on_input,
    }
    if activation == "swiglu_clamp":
        kwargs["hidden_act_alpha"] = hidden_act_alpha
        kwargs["swiglu_limit"] = 10.0 if swiglu_limit is None else float(swiglu_limit)
    if not is_prefill and workspace is not None:
        kwargs["workspace"] = workspace
    return fn(hidden_states, gate_up, down, topk_weights, topk_ids, **kwargs)


_MGEMM_FALLBACKS: set[tuple] = set()


def _mgemm_tables_for_views(scratch: Exl3Scratch, banks: tuple[torch.Tensor, ...]):
    """Build one packed pointer table for each distinct cache/overlap bank allocation."""
    key = tuple(
        (int(bank.data_ptr()), tuple(bank.shape), tuple(bank.stride()), str(bank.dtype))
        for bank in banks
    )
    tables = scratch.mgemm_tables.get(key)
    if tables is None:
        from freetoken.kernel.exl3_mgemm import Exl3MgemmBanks

        tables = Exl3MgemmBanks.from_banks(banks)
        scratch.mgemm_tables[key] = tables
    return tables


def _log_mgemm_fallback(
    *, layer_id: int | None, is_prefill: bool, scratch: Exl3Scratch, reason: Exception
) -> None:
    """Report one packed-kernel fallback per layer/phase/compiled matrix shape."""
    key = (
        layer_id,
        "prefill" if is_prefill else "decode",
        scratch.hidden_size,
        scratch.intermediate_size,
    )
    if key in _MGEMM_FALLBACKS:
        return
    _MGEMM_FALLBACKS.add(key)
    logger.warning_rank0(
        "EXL3 packed expert op falling back to reconstruct-first for layer %s (%s, H=%d, I=%d): %s",
        layer_id,
        "prefill" if is_prefill else "decode",
        scratch.hidden_size,
        scratch.intermediate_size,
        reason,
    )


def _run_mgemm_prefill(
    hidden_states: torch.Tensor,
    tables,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    apply_router_weight_on_input: bool,
    hidden_act_alpha: float,
    swiglu_limit: float | None,
    scratch: Exl3Scratch,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run prompt routes grouped by expert, tiling each group to the fixed wrapper workspace."""
    from freetoken.kernel.exl3_mgemm import exl3_mgemm_projection

    packed_scratch = scratch.mgemm_prefill_scratch
    tile_rows = EXL3_MGEMM_MAX_INDICES
    if packed_scratch is not None:
        tile_rows = min(tile_rows, int(packed_scratch.max_rows))
    if tile_rows <= 0:
        raise ValueError("EXL3 packed prompt workspace must have positive row capacity")

    out.zero_()
    for expert in _unique_routed_slots(topk_ids, tables.slot_count):
        mask = topk_ids == expert
        token_indices, _route_indices = torch.where(mask)
        route_weights = topk_weights[mask]
        for start in range(0, token_indices.numel(), tile_rows):
            stop = min(start + tile_rows, token_indices.numel())
            indices = token_indices[start:stop]
            weights = route_weights[start:stop]
            group_hidden = hidden_states.index_select(0, indices)
            if packed_scratch is not None and packed_scratch.group_id_i64 is not None:
                group_id = packed_scratch.group_id_i64
                group_id.fill_(expert)
            else:
                group_id = torch.tensor(
                    [expert], dtype=torch.int64, device=hidden_states.device
                )
            gate = exl3_mgemm_projection(
                group_hidden,
                tables,
                group_id,
                projection="gate",
                scratch=packed_scratch,
            )
            up = exl3_mgemm_projection(
                group_hidden,
                tables,
                group_id,
                projection="up",
                scratch=packed_scratch,
            )
            gate_up = torch.cat((gate, up), dim=-1)
            if activation == "swiglu_clamp":
                from freetoken.layers import swiglu_clamp_and_mul

                activated = swiglu_clamp_and_mul(
                    gate_up,
                    alpha=float(hidden_act_alpha),
                    limit=10.0 if swiglu_limit is None else float(swiglu_limit),
                )
            elif activation == "silu":
                from freetoken.layers import silu_and_mul

                activated = silu_and_mul(gate_up)
            else:
                raise ValueError(f"unsupported EXL3 mgemm activation {activation!r}")

            if apply_router_weight_on_input:
                activated = activated * weights.to(dtype=activated.dtype).unsqueeze(1)
            down = exl3_mgemm_projection(
                activated,
                tables,
                group_id,
                projection="down",
                scratch=packed_scratch,
            )
            if not apply_router_weight_on_input:
                down = down * weights.to(dtype=down.dtype).unsqueeze(1)
            out.index_add_(0, indices, down)
    return out


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
    expert_op: str = "reconstruct",
    layer_id: int | None = None,
) -> torch.Tensor:
    """Run routed EXL3 experts with packed ``exl3_mgemm`` or reconstruct-first fallback.

    Decode uses one fixed-size route chunk, device-side routing compaction, stable bank staging,
    and a caller-owned BF16 activation workspace.  Prefill uses the packed grouped-by-expert
    form in bounded tiles; both phases fall back to reconstruction when the packed wheel cannot
    serve the requested compiled shape or route capacity.
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
    if expert_op not in {"reconstruct", "mgemm"}:
        raise ValueError(
            f"unsupported EXL3 expert operation {expert_op!r}; use 'reconstruct' or 'mgemm'"
        )
    output = scratch.output_accumulator[:rows]
    output.zero_()
    if expert_op == "mgemm" and not scratch.mgemm_disabled:
        from freetoken.kernel.exl3_mgemm import (
            Exl3MgemmLimitError,
            fused_experts_exl3_mgemm,
        )

        try:
            tables = _mgemm_tables_for_views(scratch, banks)
            packed_scratch = (
                scratch.mgemm_prefill_scratch if is_prefill else scratch.mgemm_decode_scratch
            )
            if is_prefill:
                return _run_mgemm_prefill(
                    hidden_states,
                    tables,
                    topk_weights,
                    topk_ids,
                    activation=activation,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    hidden_act_alpha=hidden_act_alpha,
                    swiglu_limit=swiglu_limit,
                    scratch=scratch,
                    out=output,
                )
            return fused_experts_exl3_mgemm(
                hidden_states,
                tables,
                topk_weights,
                topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                hidden_act_alpha=hidden_act_alpha,
                swiglu_limit=swiglu_limit,
                scratch=packed_scratch,
                out=output,
            )
        except Exl3MgemmLimitError as exc:
            # A compiled-shape miss is static for this model and must be decided during eager
            # warm-up; a route-list overflow is dynamic and remains a per-call fallback.
            if "has no compiled shape" in str(exc):
                scratch.mgemm_disabled = True
            _log_mgemm_fallback(
                layer_id=layer_id, is_prefill=is_prefill, scratch=scratch, reason=exc
            )
            output.zero_()
        except RuntimeError as exc:
            # A missing optional wheel is a normal static fallback; do not hide unrelated CUDA or
            # shape failures, which should still stop boot rather than silently change math.
            if "needs the ExLlamaV3" not in str(exc):
                raise
            scratch.mgemm_disabled = True
            _log_mgemm_fallback(
                layer_id=layer_id, is_prefill=is_prefill, scratch=scratch, reason=exc
            )
            output.zero_()

    input_buffer = scratch.input_buffer[:rows]
    # The BF16 prompt path overwrites its input. Copying once for decode too gives both
    # phases a contiguous fixed-address input and keeps the caller's tensor untouched.
    input_buffer.copy_(hidden_states)

    if not is_prefill and rows * top_k <= scratch.chunk_experts:
        _unique_routed_slots_device(
            topk_ids,
            scratch,
            banks[0].shape[0],
        )
        _reconstruct_decode(banks, scratch)
        route_weights, route_ids = _route_views_device(
            scratch,
            topk_weights,
            topk_ids,
            rows=rows,
            slot_capacity=banks[0].shape[0],
        )
        workspace = (
            scratch
            if top_k == scratch.decode_top_k and rows <= scratch.decode_max_tokens
            else None
        )
        chunk_out = _run_bf16_experts(
            input_buffer,
            scratch.gate_up,
            scratch.down,
            route_weights,
            route_ids,
            is_prefill=False,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            hidden_act_alpha=hidden_act_alpha,
            swiglu_limit=swiglu_limit,
            workspace=workspace,
        )
        output.copy_(chunk_out)
        return output

    slots = _unique_routed_slots(topk_ids, banks[0].shape[0])
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
            workspace=(
                scratch
                if not is_prefill
                and top_k == scratch.decode_top_k
                and rows <= scratch.decode_max_tokens
                else None
            ),
        )
        output.add_(chunk_out)

    return output


def decode_is_graph_safe(config) -> bool:
    """Whether the configured EXL3 decode can use the single fixed capture shape."""
    if getattr(config, "max_running_req", None) != 1:
        return False
    model_config = getattr(config, "model_config", None)
    top_k = getattr(model_config, "num_experts_per_tok", None)
    if top_k is None:
        top_k = getattr(model_config, "num_experts_per_token", None)
    if top_k is None:
        return False
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return False
    if not 1 <= top_k <= _MAX_RECONSTRUCT_EXPERTS:
        return False

    # This item captures only the one-row graph.  A wider requested graph would exceed the
    # fixed eight-expert reconstruction arena even though the eager prompt path is wider.
    graph_bs = getattr(config, "cuda_graph_bs", None)
    if graph_bs is not None and list(graph_bs) != [1]:
        return False
    graph_max_bs = getattr(config, "cuda_graph_max_bs", None)
    return graph_max_bs in (None, 0, 1)


__all__ = [
    "Exl3Scratch",
    "decode_is_graph_safe",
    "fused_experts_exl3",
    "prepare_exl3_scratch",
    "require_exl3_gpu_only",
]
