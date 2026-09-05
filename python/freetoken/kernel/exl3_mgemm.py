"""Packed EXL3 multi-expert GEMM helpers.

The pointer-table and multi-matrix call shape follow ExLlamaV3 v1.4.6.  FreeToken keeps
GLM-5.3's nine EXL3 banks in slot order, so this module builds one device pointer table
per bank and sends the packed trellis rows to ``exl3_mgemm`` without reconstructing them.
The wrapper deliberately does not use ExLlamaV3's one-launch ``exl3_moe`` operation:
GLM-5.3's clamped SwiGLU is different, so callers run gate, up, FreeToken's activation,
and down separately.

MIT License
Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch


# The wheel's exl3_gemm_kernel.cuh uses MAX_INDICES=128 for the pointer-index list.  This
# is a route-list capacity, not a size_m/token-row limit: the same kernel loops size_m in
# 16-row pieces, and exl3_gemm_shape_compat() does not inspect size_m.
EXL3_MGEMM_MAX_INDICES = 128
EXL3_MGEMM_K = 2
EXL3_MGEMM_CODEBOOK = "mul1"

_PROJECTION_BANKS = {
    "gate": (0, 1, 2),
    "up": (3, 4, 5),
    "down": (6, 7, 8),
}


class Exl3MgemmLimitError(ValueError):
    """Raised when a packed call is outside the extension's supported shape limits."""


@dataclass(frozen=True)
class Exl3MgemmLimits:
    """Limits callers can use before selecting the reconstruct-first fallback.

    ``max_tokens_per_expert`` is ``None`` because the v1.4.6 kernel has no fixed ``size_m``
    ceiling.  A route-list call is still capped at 128 indices; with top-k routing that is
    ``128 // top_k`` tokens per call.  Grouping one expert at a time uses a one-entry index
    list and can therefore use any row count accepted by the selected GEMM shape.
    """

    max_indices: int = EXL3_MGEMM_MAX_INDICES
    max_tokens_per_expert: int | None = None
    kernel_shapes: int | None = None

    def max_tokens_for_top_k(self, top_k: int) -> int:
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        return self.max_indices // top_k


EXL3_MGEMM_LIMITS = Exl3MgemmLimits()


@dataclass
class Exl3MgemmScratch:
    """Reusable flat FP16 workspaces for one packed call.

    The buffers are flat on purpose.  A route call views them as ``[routes, 1, width]``;
    a grouped one-expert call views them as ``[1, tokens, width]``.  Slicing a flat prefix
    keeps every tensor contiguous even when hidden and intermediate widths differ.
    Gate and up have separate output storage because a float16 cast is a no-op: sharing one
    output buffer would let the up launch overwrite gate before the activation reads it.
    """

    device: torch.device
    max_rows: int
    max_features: int
    input_fp16: torch.Tensor
    a_had: torch.Tensor
    output_fp16: torch.Tensor
    gate_output_fp16: torch.Tensor
    up_output_fp16: torch.Tensor
    # Optional fixed-shape buffers for the complete routed operation.  They are allocated only
    # for the graph-enabled EXL3 path; projection-only callers keep the smaller legacy scratch.
    route_ids_i64: torch.Tensor | None = None
    route_weights_fp16: torch.Tensor | None = None
    route_weights_bf16: torch.Tensor | None = None
    route_hidden_bf16: torch.Tensor | None = None
    gate_up_bf16: torch.Tensor | None = None
    activated_bf16: torch.Tensor | None = None
    down_output_bf16: torch.Tensor | None = None
    group_id_i64: torch.Tensor | None = None


def prepare_exl3_mgemm_scratch(
    *, device, max_rows: int, max_features: int, preallocate_fused: bool = False
) -> Exl3MgemmScratch:
    """Allocate reusable FP16 input, Hadamard, and output storage.

    ``preallocate_fused`` adds the fixed-shape route and BF16 activation buffers used by the
    graph-safe full MoE operation.  Projection-only callers leave it false to avoid paying for
    buffers they do not use.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    max_rows = int(max_rows)
    max_features = int(max_features)
    if device.type != "cuda":
        raise ValueError(f"EXL3 mgemm scratch is card-only (got {device})")
    if max_rows <= 0 or max_features <= 0:
        raise ValueError(
            f"EXL3 mgemm scratch sizes must be positive, got rows={max_rows}, features={max_features}"
        )
    size = max_rows * max_features
    fused_buffers = {}
    if preallocate_fused:
        # The engine's decode graph is BF16 and uses at most the wheel's 128 route entries.  Keep
        # the route/activation arena bounded by that contract instead of scaling with a prompt.
        fused_buffers = {
            "route_ids_i64": torch.empty(max_rows, dtype=torch.int64, device=device),
            "route_weights_fp16": torch.empty(max_rows, dtype=torch.float16, device=device),
            "route_weights_bf16": torch.empty(max_rows, dtype=torch.bfloat16, device=device),
            # Keep these as flat storage so a narrower model dimension can still be viewed as
            # contiguous (e.g. GLM-5.3 has H=4096 and I=2048).
            "route_hidden_bf16": torch.empty(
                max_rows * max_features, dtype=torch.bfloat16, device=device
            ),
            "gate_up_bf16": torch.empty(
                max_rows * 2 * max_features, dtype=torch.bfloat16, device=device
            ),
            "activated_bf16": torch.empty(
                max_rows * max_features, dtype=torch.bfloat16, device=device
            ),
            "down_output_bf16": torch.empty(
                max_rows * max_features, dtype=torch.bfloat16, device=device
            ),
            "group_id_i64": torch.empty(1, dtype=torch.int64, device=device),
        }
    return Exl3MgemmScratch(
        device=device,
        max_rows=max_rows,
        max_features=max_features,
        input_fp16=torch.empty(size, dtype=torch.float16, device=device),
        a_had=torch.empty(size, dtype=torch.float16, device=device),
        output_fp16=torch.empty(size, dtype=torch.float16, device=device),
        gate_output_fp16=torch.empty(size, dtype=torch.float16, device=device),
        up_output_fp16=torch.empty(size, dtype=torch.float16, device=device),
        **fused_buffers,
    )


class Exl3MgemmBanks:
    """Nine packed EXL3 banks plus their fixed device pointer tables.

    ``banks`` must be in the same order as ``models.exl3_banks.EXL3_BANK_NAMES``.  The
    first dimension is the local slot/expert index used by routed IDs.  Keeping the source
    tensors as fields also keeps the memory behind every pointer table alive.
    """

    __slots__ = ("banks", "ptrs", "slot_count", "device", "k", "_shape_support")

    def __init__(self, banks: tuple[torch.Tensor, ...], ptrs: tuple[torch.Tensor, ...], *, k: int):
        self.banks = banks
        self.ptrs = ptrs
        self.slot_count = int(banks[0].shape[0])
        self.device = banks[0].device
        self.k = int(k)
        # Shape compatibility is static for a bank allocation. Cache the extension query here so
        # decode does not ask the wheel about all compiled shapes for every projection call.
        self._shape_support: dict[str, bool] = {}

    def shape_supported(self, projection: Literal["gate", "up", "down"]) -> bool:
        supported = self._shape_support.get(projection)
        if supported is None:
            _ptr_trellis, _ptr_suh, _ptr_svh, input_features, output_features = self.projection(
                projection
            )
            supported = exl3_mgemm_shape_supported(
                input_features, output_features, k=self.k
            )
            self._shape_support[projection] = bool(supported)
        return supported

    @classmethod
    def from_banks(cls, banks: Sequence[torch.Tensor]) -> "Exl3MgemmBanks":
        banks = tuple(banks)
        if len(banks) != 9:
            raise ValueError(
                "EXL3 mgemm expects nine banks in order gate_trellis, gate_suh, gate_svh, "
                f"up_trellis, up_suh, up_svh, down_trellis, down_suh, down_svh (got {len(banks)})"
            )
        if not all(isinstance(bank, torch.Tensor) for bank in banks):
            raise ValueError("all EXL3 mgemm banks must be torch.Tensor values")
        if any(bank.device.type != "cuda" for bank in banks):
            raise ValueError("EXL3 mgemm banks are card-only and must be CUDA tensors")
        if any(not bank.is_contiguous() for bank in banks):
            raise ValueError("EXL3 mgemm banks must be contiguous")
        device = banks[0].device
        if any(bank.device != device for bank in banks):
            raise ValueError("EXL3 mgemm banks must be on one CUDA device")

        trellis_indices = (0, 3, 6)
        factor_indices = (1, 2, 4, 5, 7, 8)
        for index in trellis_indices:
            bank = banks[index]
            if bank.dim() != 4 or bank.dtype != torch.int16:
                raise ValueError(
                    f"EXL3 trellis bank {index} must be rank-4 int16, got "
                    f"shape={tuple(bank.shape)} dtype={bank.dtype}"
                )
            if bank.shape[-1] != 16 * EXL3_MGEMM_K:
                raise ValueError(
                    f"EXL3 trellis bank {index} has K={bank.shape[-1] // 16}; "
                    f"expected K={EXL3_MGEMM_K}"
                )
        for index in factor_indices:
            bank = banks[index]
            if bank.dim() != 2 or bank.dtype != torch.float16:
                raise ValueError(
                    f"EXL3 factor bank {index} must be rank-2 float16, got "
                    f"shape={tuple(bank.shape)} dtype={bank.dtype}"
                )
        slots = banks[0].shape[0]
        if slots <= 0 or any(bank.shape[0] != slots for bank in banks):
            raise ValueError("EXL3 mgemm banks must have one positive, shared slot count")

        # The extension reads these as device arrays of raw addresses.  Build them once when
        # the slot allocation is made; rebuilding them inside decode would add host work.
        ptrs = tuple(
            torch.tensor(
                [int(bank[row].data_ptr()) for row in range(slots)],
                dtype=torch.int64,
                device=device,
            ).contiguous()
            for bank in banks
        )
        return cls(banks, ptrs, k=EXL3_MGEMM_K)

    def projection(self, name: Literal["gate", "up", "down"]):
        try:
            bank_indices = _PROJECTION_BANKS[name]
        except KeyError as exc:
            raise ValueError(f"unsupported EXL3 mgemm projection {name!r}") from exc
        trellis_i, suh_i, svh_i = bank_indices
        trellis, suh, svh = (self.banks[i] for i in bank_indices)
        if trellis.shape[1] * 16 != suh.shape[1]:
            raise ValueError(f"{name} trellis/suh dimensions do not agree")
        if trellis.shape[2] * 16 != svh.shape[1]:
            raise ValueError(f"{name} trellis/svh dimensions do not agree")
        return (
            self.ptrs[trellis_i],
            self.ptrs[suh_i],
            self.ptrs[svh_i],
            int(trellis.shape[1]) * 16,
            int(trellis.shape[2]) * 16,
        )


def _load_extension():
    # Importing the optional wheel lazily keeps CPU-only schema tests usable without the
    # ExLlamaV3 DLLs, matching kernel/exl3.py's import boundary.
    try:
        import exllamav3_ext
    except ImportError as exc:  # pragma: no cover - depends on the optional proof wheel
        raise RuntimeError(
            "EXL3 mgemm needs the ExLlamaV3 v1.4.6 exllamav3_ext wheel"
        ) from exc
    return exllamav3_ext


def exl3_mgemm_limits() -> Exl3MgemmLimits:
    """Return the static route/index limit and the installed wheel's shape count."""
    try:
        extension = _load_extension()
    except RuntimeError:
        return EXL3_MGEMM_LIMITS
    return Exl3MgemmLimits(
        max_indices=EXL3_MGEMM_MAX_INDICES,
        max_tokens_per_expert=None,
        kernel_shapes=int(extension.exl3_gemm_num_kernel_shapes()),
    )


def exl3_mgemm_shape_supported(
    input_features: int, output_features: int, *, k: int = EXL3_MGEMM_K
) -> bool:
    """Check the wheel's compiled K/N shape table before launching a packed GEMM."""
    extension = _load_extension()
    input_features = int(input_features)
    output_features = int(output_features)
    k = int(k)
    if input_features <= 0 or output_features <= 0:
        return False
    return any(
        bool(extension.exl3_gemm_shape_compat(shape, 1, input_features, output_features, k))
        for shape in range(1, int(extension.exl3_gemm_num_kernel_shapes()) + 1)
    )


def _coerce_tables(banks: Exl3MgemmBanks | Sequence[torch.Tensor]) -> Exl3MgemmBanks:
    return banks if isinstance(banks, Exl3MgemmBanks) else Exl3MgemmBanks.from_banks(banks)


def _validate_inputs(
    inputs: torch.Tensor,
    ids: torch.Tensor,
    *,
    input_features: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(inputs, torch.Tensor) or inputs.dim() != 2:
        raise ValueError(f"EXL3 mgemm inputs must be rank-2, got {getattr(inputs, 'shape', None)}")
    if inputs.shape[1] != input_features:
        raise ValueError(
            f"EXL3 mgemm input width {inputs.shape[1]} does not match {input_features}"
        )
    if inputs.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"EXL3 mgemm inputs must be float16 or bfloat16, got {inputs.dtype}")
    if inputs.device != device:
        raise ValueError(f"EXL3 mgemm inputs are on {inputs.device}, expected {device}")
    if not inputs.is_contiguous():
        inputs = inputs.contiguous()
    if not isinstance(ids, torch.Tensor) or ids.dim() != 1:
        raise ValueError(f"EXL3 mgemm expert IDs must be rank-1, got {getattr(ids, 'shape', None)}")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"EXL3 mgemm expert IDs must be int32 or int64, got {ids.dtype}")
    if ids.device != device:
        raise ValueError(f"EXL3 mgemm expert IDs are on {ids.device}, expected {device}")
    if ids.numel() == 0:
        raise ValueError("EXL3 mgemm needs at least one expert ID")
    if not ids.is_contiguous():
        ids = ids.contiguous()
    return inputs, ids.to(dtype=torch.int64) if ids.dtype != torch.int64 else ids


def _workspace_view(
    scratch: Exl3MgemmScratch | None,
    *,
    rows: int,
    input_features: int,
    output_features: int,
    device: torch.device,
    output_storage: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scratch is None:
        return (
            torch.empty((rows, input_features), dtype=torch.float16, device=device),
            torch.empty((rows, input_features), dtype=torch.float16, device=device),
            torch.empty((rows, output_features), dtype=torch.float16, device=device),
        )
    if scratch.device != device:
        raise ValueError(f"EXL3 mgemm scratch is on {scratch.device}, expected {device}")
    if rows > scratch.max_rows or max(input_features, output_features) > scratch.max_features:
        raise ValueError(
            "EXL3 mgemm scratch is too small: "
            f"need rows={rows}, features={max(input_features, output_features)}; "
            f"have rows={scratch.max_rows}, features={scratch.max_features}"
        )
    input_buf = scratch.input_fp16[: rows * input_features].view(rows, input_features)
    had_buf = scratch.a_had[: rows * input_features].view(rows, input_features)
    if output_storage is None:
        output_storage = scratch.output_fp16
    output_buf = output_storage[: rows * output_features].view(rows, output_features)
    return input_buf, had_buf, output_buf


def _raw_mgemm(
    inputs: torch.Tensor,
    tables: Exl3MgemmBanks,
    projection: Literal["gate", "up", "down"],
    ids: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    num_tokens: int,
    broadcast_input: bool,
    scratch: Exl3MgemmScratch | None,
) -> torch.Tensor:
    ptr_trellis, ptr_suh, ptr_svh, input_features, output_features = tables.projection(projection)
    inputs, ids = _validate_inputs(
        inputs,
        ids,
        input_features=input_features,
        device=tables.device,
    )
    tokens = int(inputs.shape[0])
    if broadcast_input:
        if ids.numel() != 1:
            raise ValueError("broadcast EXL3 mgemm calls need exactly one expert ID")
        batch, rows_per_batch = 1, tokens
    else:
        if ids.numel() != tokens:
            raise ValueError(
                f"route EXL3 mgemm needs one expert ID per input row, got {ids.numel()} for {tokens} rows"
            )
        batch, rows_per_batch = tokens, 1
        if tokens > EXL3_MGEMM_MAX_INDICES:
            raise Exl3MgemmLimitError(
                f"EXL3 mgemm route list has {tokens} entries; the wheel limit is "
                f"{EXL3_MGEMM_MAX_INDICES}"
            )

    if weights is not None:
        if not isinstance(weights, torch.Tensor) or weights.dim() != 1:
            raise ValueError("EXL3 mgemm weights must be a rank-1 tensor")
        if weights.device != tables.device:
            raise ValueError(f"EXL3 mgemm weights are on {weights.device}, expected {tables.device}")
        if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"EXL3 mgemm weights have unsupported dtype {weights.dtype}")
        if weights.numel() not in (1, tokens):
            raise ValueError(
                f"EXL3 mgemm weights need one value or one per input row, got {weights.numel()}"
            )

    total_rows = batch * rows_per_batch
    output_storage = None
    if scratch is not None:
        if projection == "gate":
            output_storage = scratch.gate_output_fp16
        elif projection == "up":
            output_storage = scratch.up_output_fp16
    input_buf, had_buf, output_buf = _workspace_view(
        scratch,
        rows=total_rows,
        input_features=input_features,
        output_features=output_features,
        device=tables.device,
        output_storage=output_storage,
    )
    # copy_ performs the BF16->FP16 cast in the destination without creating a temporary tensor;
    # this is one of the per-call allocations that would otherwise break CUDA graph capture.
    input_buf.copy_(inputs)
    a = input_buf.view(batch, rows_per_batch, input_features)
    a_had = had_buf.view(batch, rows_per_batch, input_features)
    c = output_buf.view(batch, rows_per_batch, output_features)
    indices = ids.view(1, -1)

    extension = _load_extension()
    if not tables.shape_supported(projection):
        raise Exl3MgemmLimitError(
            f"EXL3 mgemm has no compiled shape for K={tables.k}, "
            f"input={input_features}, output={output_features}"
        )

    extension_weights = None
    if weights is not None and (broadcast_input and weights.numel() == 1):
        extension_weights = weights.to(dtype=torch.float16).view(1, 1).contiguous()
    elif weights is not None and not broadcast_input:
        extension_weights = weights.to(dtype=torch.float16).view(1, -1).contiguous()
    extension.exl3_mgemm(
        a,
        ptr_trellis,
        c,
        ptr_suh,
        a_had,
        ptr_svh,
        indices,
        extension_weights,
        tables.k,
        -1,
        0,
        1,
        -1,
        -1,
        0,
        int(num_tokens),
        None,
        None,
    )
    if broadcast_input:
        result = c[0]
    elif weights is not None:
        # With num_tokens > 1 the extension reduces each contiguous route group into
        # the first row of that group; the remaining rows are per-route scratch.
        result = c[: int(num_tokens), 0, :]
    else:
        result = c[:, 0, :]
    if weights is not None and extension_weights is None:
        result = result * weights.to(dtype=result.dtype).reshape(-1, 1)
    return result


def exl3_mgemm_projection(
    inputs: torch.Tensor,
    banks: Exl3MgemmBanks | Sequence[torch.Tensor],
    expert_ids: torch.Tensor,
    *,
    projection: Literal["gate", "up", "down"],
    weights: torch.Tensor | None = None,
    scratch: Exl3MgemmScratch | None = None,
) -> torch.Tensor:
    """Run one packed EXL3 projection without reconstructing its weights.

    ``inputs`` is ``[tokens, in_features]``.  A single ``expert_ids`` value broadcasts one
    packed row over every input token, which is the grouped-by-expert form and has no fixed
    token-row ceiling.  One ID per input row runs a route-list call and is capped at 128
    entries.  Per-row weights are applied after the projection; a one-value weight can be
    applied by the extension itself.
    """
    tables = _coerce_tables(banks)
    ptr_trellis, ptr_suh, ptr_svh, input_features, _output_features = tables.projection(projection)
    del ptr_trellis, ptr_suh, ptr_svh
    inputs, expert_ids = _validate_inputs(
        inputs,
        expert_ids,
        input_features=input_features,
        device=tables.device,
    )
    if expert_ids.numel() == 1:
        return _raw_mgemm(
            inputs,
            tables,
            projection,
            expert_ids,
            weights=weights,
            num_tokens=1,
            broadcast_input=True,
            scratch=scratch,
        ).to(dtype=inputs.dtype).clone()
    result = _raw_mgemm(
        inputs,
        tables,
        projection,
        expert_ids,
        weights=None,
        num_tokens=1,
        broadcast_input=False,
        scratch=scratch,
    )
    if weights is not None:
        if not isinstance(weights, torch.Tensor) or weights.dim() != 1 or weights.numel() != inputs.shape[0]:
            raise ValueError("route EXL3 mgemm weights need one value per input row")
        result = result * weights.to(dtype=result.dtype).reshape(-1, 1)
    return result.to(dtype=inputs.dtype).clone()


def _route_mgemm(
    inputs: torch.Tensor,
    tables: Exl3MgemmBanks,
    projection: Literal["gate", "up", "down"],
    expert_ids: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    num_tokens: int,
    scratch: Exl3MgemmScratch | None,
) -> torch.Tensor:
    """Run the route-list form; weighted calls reduce each token's route group."""
    return _raw_mgemm(
        inputs,
        tables,
        projection,
        expert_ids,
        weights=weights,
        num_tokens=num_tokens,
        broadcast_input=False,
        scratch=scratch,
    )


def fused_experts_exl3_mgemm(
    hidden_states: torch.Tensor,
    banks: Exl3MgemmBanks | Sequence[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "swiglu_clamp",
    apply_router_weight_on_input: bool = False,
    hidden_act_alpha: float = 1.0,
    swiglu_limit: float | None = 10.0,
    scratch: Exl3MgemmScratch | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run gate, up, FreeToken activation, and weighted down on packed EXL3 rows.

    The route list is flattened in token-major order.  The wheel's 128-index capacity means
    the caller must tile a larger prompt before calling this function; the error is explicit
    so the reconstruct-first path can take over.  This function intentionally leaves shared
    experts to the caller, just as the existing reconstruct-first operation does.
    """
    tables = _coerce_tables(banks)
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 2:
        raise ValueError("EXL3 mgemm hidden_states must be rank-2")
    if hidden_states.device != tables.device:
        raise ValueError(
            f"EXL3 mgemm hidden_states are on {hidden_states.device}, expected {tables.device}"
        )
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"EXL3 mgemm hidden_states need float16 or bfloat16, got {hidden_states.dtype}")
    if not isinstance(topk_ids, torch.Tensor) or topk_ids.dim() != 2:
        raise ValueError("EXL3 mgemm topk_ids must be rank-2")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"EXL3 mgemm topk_ids need int32 or int64, got {topk_ids.dtype}")
    if topk_ids.device != tables.device:
        raise ValueError(f"EXL3 mgemm topk_ids are on {topk_ids.device}, expected {tables.device}")
    if not isinstance(topk_weights, torch.Tensor) or topk_weights.shape != topk_ids.shape:
        raise ValueError("EXL3 mgemm topk_weights and topk_ids must have the same shape")
    if topk_weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"EXL3 mgemm topk_weights have unsupported dtype {topk_weights.dtype}")
    if topk_weights.device != tables.device:
        raise ValueError(
            f"EXL3 mgemm topk_weights are on {topk_weights.device}, expected {tables.device}"
        )

    rows, top_k = topk_ids.shape
    if rows <= 0 or top_k <= 0:
        raise ValueError(f"EXL3 mgemm routing shape must be non-empty, got {tuple(topk_ids.shape)}")
    route_count = int(rows * top_k)
    if route_count > EXL3_MGEMM_MAX_INDICES:
        raise Exl3MgemmLimitError(
            f"EXL3 mgemm route list has {route_count} entries; the wheel limit is "
            f"{EXL3_MGEMM_MAX_INDICES} (top_k={top_k} permits at most "
            f"{EXL3_MGEMM_MAX_INDICES // top_k} tokens per call)"
        )
    if scratch is not None and scratch.max_rows < route_count:
        raise ValueError(
            f"EXL3 mgemm scratch has {scratch.max_rows} rows, needs {route_count} route rows"
        )
    if out is not None:
        if not isinstance(out, torch.Tensor) or out.shape != (rows, hidden_states.shape[1]):
            raise ValueError(
                "EXL3 mgemm output must have shape "
                f"[{rows}, {hidden_states.shape[1]}], got {getattr(out, 'shape', None)}"
            )
        if out.device != tables.device or out.dtype != hidden_states.dtype or not out.is_contiguous():
            raise ValueError(
                "EXL3 mgemm output must be contiguous and match the hidden-state device/dtype"
            )

    fixed_fused = (
        scratch is not None
        and hidden_states.dtype == torch.bfloat16
        and scratch.route_ids_i64 is not None
        and scratch.route_weights_fp16 is not None
        and scratch.route_weights_bf16 is not None
        and scratch.route_hidden_bf16 is not None
        and scratch.gate_up_bf16 is not None
        and scratch.activated_bf16 is not None
        and scratch.down_output_bf16 is not None
    )
    if fixed_fused:
        route_ids = scratch.route_ids_i64[:route_count]
        route_ids.copy_(topk_ids.reshape(-1))
        route_weights = scratch.route_weights_fp16[:route_count]
        route_weights.copy_(topk_weights.reshape(-1))
        route_weights_bf16 = scratch.route_weights_bf16[:route_count]
        route_weights_bf16.copy_(route_weights)
        hidden_size = int(hidden_states.shape[1])
        route_hidden = scratch.route_hidden_bf16[: route_count * hidden_size].view(
            route_count, hidden_size
        )
        route_hidden.view(rows, top_k, hidden_size).copy_(hidden_states.unsqueeze(1))

        gate_fp16 = _route_mgemm(
            route_hidden,
            tables,
            "gate",
            route_ids,
            weights=None,
            num_tokens=1,
            scratch=scratch,
        )
        up_fp16 = _route_mgemm(
            route_hidden,
            tables,
            "up",
            route_ids,
            weights=None,
            num_tokens=1,
            scratch=scratch,
        )
        intermediate_size = int(gate_fp16.shape[-1])
        gate_up = scratch.gate_up_bf16[: route_count * 2 * intermediate_size].view(
            route_count, 2 * intermediate_size
        )
        gate_up[:, :intermediate_size].copy_(gate_fp16)
        gate_up[:, intermediate_size:].copy_(up_fp16)
        activated = scratch.activated_bf16[: route_count * intermediate_size].view(
            route_count, intermediate_size
        )
        if activation == "swiglu_clamp":
            from freetoken.layers import swiglu_clamp_and_mul

            swiglu_clamp_and_mul(
                gate_up,
                activated,
                alpha=float(hidden_act_alpha),
                limit=10.0 if swiglu_limit is None else float(swiglu_limit),
            )
        elif activation == "silu":
            from freetoken.layers import silu_and_mul

            silu_and_mul(gate_up, activated)
        else:
            raise ValueError(f"unsupported EXL3 mgemm activation {activation!r}")

        if apply_router_weight_on_input:
            # Without extension weights, the route-list call returns one down row per route;
            # reduce those rows ourselves after applying the router weights to the activation.
            activated.mul_(route_weights_bf16.view(-1, 1))
            down_fp16 = _route_mgemm(
                activated,
                tables,
                "down",
                route_ids,
                weights=None,
                num_tokens=1,
                scratch=scratch,
            )
            down_width = int(down_fp16.shape[-1])
            down = scratch.down_output_bf16[: route_count * down_width].view(
                route_count, down_width
            )
            down.copy_(down_fp16)
            result = out if out is not None else down.new_empty((rows, down_width))
            torch.sum(down.view(rows, top_k, -1), dim=1, out=result)
            return result

        # Passing weights to the extension asks it to reduce each contiguous top-k route group;
        # its result already has one row per token, not one row per route. Summing a route-shaped
        # view here would repeat the reduced row top-k times and include stale scratch rows.
        down_fp16 = _route_mgemm(
            activated,
            tables,
            "down",
            route_ids,
            weights=route_weights,
            num_tokens=rows,
            scratch=scratch,
        )
        down_width = int(down_fp16.shape[-1])
        down = scratch.down_output_bf16[: rows * down_width].view(rows, down_width)
        down.copy_(down_fp16)
        result = out if out is not None else down
        if result is not down:
            result.copy_(down)
        return result

    route_ids = topk_ids.reshape(-1).to(dtype=torch.int64).contiguous()
    route_weights = topk_weights.reshape(-1).to(dtype=torch.float16).contiguous()
    route_hidden = (
        hidden_states.unsqueeze(1)
        .expand(rows, top_k, hidden_states.shape[1])
        .reshape(route_count, hidden_states.shape[1])
        .contiguous()
    )

    gate = _route_mgemm(
        route_hidden,
        tables,
        "gate",
        route_ids,
        weights=None,
        num_tokens=1,
        scratch=scratch,
    ).to(dtype=hidden_states.dtype)
    up = _route_mgemm(
        route_hidden,
        tables,
        "up",
        route_ids,
        weights=None,
        num_tokens=1,
        scratch=scratch,
    ).to(dtype=hidden_states.dtype)
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
        activated = activated * route_weights.to(dtype=activated.dtype).unsqueeze(1)
        down = _route_mgemm(
            activated.reshape(route_count, -1).contiguous(),
            tables,
            "down",
            route_ids,
            weights=None,
            num_tokens=1,
            scratch=scratch,
        )
        result = down.view(rows, top_k, -1).sum(dim=1).to(dtype=hidden_states.dtype)
    else:
        down = _route_mgemm(
            activated.reshape(route_count, -1).contiguous(),
            tables,
            "down",
            route_ids,
            weights=route_weights,
            num_tokens=rows,
            scratch=scratch,
        )
        result = down.to(dtype=hidden_states.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result.clone()


__all__ = [
    "EXL3_MGEMM_CODEBOOK",
    "EXL3_MGEMM_K",
    "EXL3_MGEMM_LIMITS",
    "EXL3_MGEMM_MAX_INDICES",
    "Exl3MgemmBanks",
    "Exl3MgemmLimitError",
    "Exl3MgemmLimits",
    "Exl3MgemmScratch",
    "exl3_mgemm_limits",
    "exl3_mgemm_projection",
    "exl3_mgemm_shape_supported",
    "fused_experts_exl3_mgemm",
    "prepare_exl3_mgemm_scratch",
]
