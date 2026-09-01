"""Private Qwen3.8 MTP feasibility model.

This module is deliberately not exported by :mod:`freetoken.models.qwen4_exp` and is never
constructed by ordinary serving.  It models the checkpoint's one-layer native MTP sidecar while
keeping its stacked routed experts as an external execution seam.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator

import safetensors
import torch
from freetoken.layers import BaseOP, LinearReplicated, OPList
from freetoken.models.config import FullAttentionGroupConfig, ModelConfig

from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .model import Qwen4ExpDecoderLayer


# HBM the resident runner will spend gathering routed expert weights before it switches to the
# expert-major loop.
_ROUTED_GATHER_BUDGET_BYTES = 512 << 20
# ...and a floor the budget cannot argue with. Every call on the per-cycle speculative path is
# at most the widest step -- one row per draft step, and the accepted run (<= 1 + depth) when
# the head commits it -- so those must never reach the loop, whose per-expert host round trip
# is both a stall and capture-illegal. Their traffic is bounded by this row count either way.
_ROUTED_GATHER_MIN_TOKENS = 4

_EXPERT_MODEL_NAMES = frozenset(
    {
        "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.experts.down_proj",
    }
)

# model state name -> (raw checkpoint names, output-row alignment)
_MTP_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    "layers.0.self_attn.qkv_proj.weight": (
        (
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
        ),
        0,
    ),
    "layers.0.mlp.shared_expert.gate_up_proj.weight": (
        (
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        ),
        0,
    ),
    "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight": (
        (
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
            "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
        ),
        16,
    ),
    "layers.0.mlp_hyper_connection.input_mix_weight_down_block_inject.weight": (
        (
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
            "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
        ),
        16,
    ),
}


def derive_mtp_model_config(base: ModelConfig) -> ModelConfig:
    """Derive the native one-layer QSA MTP geometry without PLE, GDN, or vision."""

    full = [
        group
        for group in base.attention_groups
        if isinstance(group, FullAttentionGroupConfig)
    ]
    if len(full) != 1:
        raise ValueError(f"Qwen3.8 MTP needs one target full-attention group, got {len(full)}")
    full_group = replace(full[0], layer_ids=(0,), num_index_layers=1)
    qwen_args = replace(base.qwen4_args, ple_layer_ids=())
    return replace(
        base,
        num_layers=1,
        attention_groups=(full_group,),
        qwen4_args=qwen_args,
        slot_states=(),
        vision_config=None,
        image_token_id=None,
        # The checkpoint's MTP routed experts are exact BF16.  Private compression is an
        # external bank placement and must not change the dense layer's declared dtypes.
        expert_quant="none",
        attn_quant="none",
        dense_quant="none",
        moe_backend="fused",
    )


class MTPBF16ExpertBanks:
    """One exact, read-only-shaped BF16 MTP expert layer for ``CpuMoeExecutor``."""

    quant_format = "bf16"
    num_layers = 1
    decode_target = "cpu"
    cpu_executor = None

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        if gate_up.dtype is not torch.bfloat16 or down.dtype is not torch.bfloat16:
            raise ValueError(
                f"MTP exact expert banks must be BF16, got {gate_up.dtype}/{down.dtype}"
            )
        if gate_up.ndim != 3 or gate_up.shape[1] % 2:
            raise ValueError(f"invalid MTP gate_up shape {tuple(gate_up.shape)}")
        experts, twice_intermediate, hidden = gate_up.shape
        intermediate = twice_intermediate // 2
        if tuple(down.shape) != (experts, hidden, intermediate):
            raise ValueError(
                "MTP gate_up/down shapes disagree: "
                f"{tuple(gate_up.shape)} vs {tuple(down.shape)}"
            )
        if not gate_up.is_contiguous() or not down.is_contiguous():
            raise ValueError("MTP exact expert banks must be contiguous")
        if gate_up.device.type != "cpu" or down.device.type != "cpu":
            raise ValueError("MTP exact expert banks must remain on CPU")
        self.gate_up = gate_up
        self.down = down
        self.num_experts = int(experts)
        self.hidden_size = int(hidden)
        self.intermediate_size = int(intermediate)
        self.bank_sources = {"gate_up": [gate_up], "down": [down]}
        self.total_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in (gate_up, down)
        )
        self.bytes_per_expert = self.total_bytes // self.num_experts

    @classmethod
    def from_store(cls, store: "MTPWeightStore") -> "MTPBF16ExpertBanks":
        return cls(
            store.tensor("mtp.layers.0.mlp.experts.gate_up_proj"),
            store.tensor("mtp.layers.0.mlp.experts.down_proj"),
        )


_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def quantize_nvfp4_rows(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize CPU ``[N,K]`` rows to ModelOpt NVFP4 (low nibble = even K)."""

    if rows.ndim != 2:
        raise ValueError(f"NVFP4 input must be two-dimensional, got {tuple(rows.shape)}")
    if rows.device.type != "cpu":
        raise ValueError("private NVFP4 conversion runs on CPU")
    if rows.shape[1] % 16:
        raise ValueError(f"NVFP4 row width must be divisible by 16, got {rows.shape[1]}")
    values = rows.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("NVFP4 input rows must be finite")

    row_amax = values.abs().amax(dim=-1)
    raw_global = row_amax / (6.0 * 448.0)
    row_global = raw_global.to(torch.float16)
    # Rounding the row global downward could make the largest block scale exceed E4M3's
    # finite maximum. Move only those rounded-down rows to the next representable FP16.
    rounded_down = (row_global.float() < raw_global) & (row_amax > 0)
    if bool(rounded_down.any()):
        upward = torch.nextafter(
            row_global[rounded_down],
            torch.full_like(row_global[rounded_down], float("inf")),
        )
        row_global[rounded_down] = upward
    row_global[row_amax == 0] = 1.0

    blocks = values.view(values.shape[0], -1, 16)
    block_amax = blocks.abs().amax(dim=-1)
    target_scale = block_amax / (6.0 * row_global.float().unsqueeze(-1))
    target_scale.clamp_(min=0.0, max=448.0)
    block_scale = target_scale.to(torch.float8_e4m3fn)
    nonzero_underflow = (block_amax > 0) & (block_scale.float() == 0)
    if bool(nonzero_underflow.any()):
        minimum = torch.tensor(2.0**-9, dtype=torch.float32).to(torch.float8_e4m3fn)
        block_scale[nonzero_underflow] = minimum

    denominator = (
        row_global.float().unsqueeze(-1)
        * block_scale.float().repeat_interleave(16, dim=-1)
    )
    normalized = torch.where(denominator != 0, values / denominator, torch.zeros_like(values))
    normalized.clamp_(-6.0, 6.0)
    codes = (normalized.unsqueeze(-1) - _E2M1_VALUES).abs().argmin(dim=-1).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, block_scale.contiguous(), row_global.contiguous()


def dequantize_nvfp4_rows(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    row_global: torch.Tensor,
) -> torch.Tensor:
    """Independent CPU arithmetic for the native six-bank NVFP4 row layout."""

    if packed.ndim != 2 or packed.dtype is not torch.uint8:
        raise ValueError("packed NVFP4 rows must be two-dimensional uint8")
    rows, half_width = packed.shape
    width = 2 * half_width
    if tuple(block_scale.shape) != (rows, width // 16):
        raise ValueError("NVFP4 block-scale shape does not match packed rows")
    if tuple(row_global.shape) != (rows,):
        raise ValueError("NVFP4 row-global shape does not match packed rows")
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).view(rows, width)
    values = _E2M1_VALUES[codes.long()]
    return (
        values
        * block_scale.float().repeat_interleave(16, dim=-1)
        * row_global.float().unsqueeze(-1)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


class MTPNVFP4ExpertBanks:
    """Private read-only six-bank NVFP4 MTP expert layer."""

    quant_format = "nvfp4"
    num_layers = 1
    decode_target = "cpu"
    cpu_executor = None

    _NAMES = (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    )

    def __init__(
        self,
        gate_up_packed: torch.Tensor,
        gate_up_scale: torch.Tensor,
        gate_up_global: torch.Tensor,
        down_packed: torch.Tensor,
        down_scale: torch.Tensor,
        down_global: torch.Tensor,
    ) -> None:
        tensors = (
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            down_packed,
            down_scale,
            down_global,
        )
        if any(tensor.device.type != "cpu" or not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("MTP NVFP4 banks must be contiguous CPU tensors")
        if gate_up_packed.dtype is not torch.uint8 or down_packed.dtype is not torch.uint8:
            raise ValueError("MTP NVFP4 packed banks must be uint8")
        if gate_up_scale.element_size() != 1 or down_scale.element_size() != 1:
            raise ValueError("MTP NVFP4 block scales must be one-byte E4M3")
        if gate_up_global.dtype is not torch.float16 or down_global.dtype is not torch.float16:
            raise ValueError("MTP NVFP4 row globals must be float16")
        if gate_up_packed.ndim != 3:
            raise ValueError(f"invalid MTP NVFP4 gate_up shape {tuple(gate_up_packed.shape)}")
        experts, twice_intermediate, packed_hidden = gate_up_packed.shape
        if twice_intermediate % 2:
            raise ValueError("MTP NVFP4 gate_up output width must be even")
        hidden = 2 * packed_hidden
        intermediate = twice_intermediate // 2
        expected = {
            "gate_up_scale": (experts, 2 * intermediate, hidden // 16),
            "gate_up_global": (experts, 2 * intermediate),
            "down_packed": (experts, hidden, intermediate // 2),
            "down_scale": (experts, hidden, intermediate // 16),
            "down_global": (experts, hidden),
        }
        actual = {
            "gate_up_scale": tuple(gate_up_scale.shape),
            "gate_up_global": tuple(gate_up_global.shape),
            "down_packed": tuple(down_packed.shape),
            "down_scale": tuple(down_scale.shape),
            "down_global": tuple(down_global.shape),
        }
        if actual != expected:
            raise ValueError(f"MTP NVFP4 bank shapes disagree: {actual}, expected {expected}")
        for name, tensor in zip(self._NAMES, tensors):
            setattr(self, name, tensor)
        self.num_experts = int(experts)
        self.hidden_size = int(hidden)
        self.intermediate_size = int(intermediate)
        self.bank_sources = {name: [getattr(self, name)] for name in self._NAMES}
        self.total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        self.bytes_per_expert = self.total_bytes // self.num_experts

    @classmethod
    def from_manifest(
        cls, manifest_path: str | Path, *, validate_hashes: bool = True
    ) -> "MTPNVFP4ExpertBanks":
        manifest_path = Path(manifest_path).resolve()
        root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("format") != "nvfp4":
            raise ValueError("unsupported MTP NVFP4 manifest")
        records = manifest.get("banks", {})
        if set(records) != set(cls._NAMES):
            raise ValueError(f"MTP NVFP4 manifest bank set is invalid: {sorted(records)}")
        tensors: list[torch.Tensor] = []
        for name in cls._NAMES:
            record = records[name]
            path = (root / record["file"]).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"MTP NVFP4 bank escapes the private root: {path}") from exc
            shape = tuple(int(value) for value in record["shape"])
            dtype_name = str(record["dtype"])
            element_size = 2 if dtype_name == "float16" else 1
            numel = 1
            for value in shape:
                numel *= value
            expected_bytes = numel * element_size
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"MTP NVFP4 bank size mismatch for {name}")
            if int(record["nbytes"]) != expected_bytes:
                raise ValueError(f"MTP NVFP4 manifest size mismatch for {name}")
            if validate_hashes and _file_sha256(path) != record["sha256"]:
                raise ValueError(f"MTP NVFP4 bank hash mismatch for {name}")
            if dtype_name == "float16":
                tensor = torch.from_file(str(path), shared=False, size=numel, dtype=torch.float16)
            elif dtype_name == "float8_e4m3fn":
                tensor = torch.from_file(str(path), shared=False, size=numel, dtype=torch.uint8)
                tensor = tensor.view(torch.float8_e4m3fn)
            elif dtype_name == "uint8":
                tensor = torch.from_file(str(path), shared=False, size=numel, dtype=torch.uint8)
            else:
                raise ValueError(f"unsupported MTP NVFP4 dtype {dtype_name!r} for {name}")
            tensors.append(tensor.reshape(shape))
        result = cls(*tensors)
        result.manifest_path = manifest_path
        result.manifest = manifest
        return result


@dataclass
class MTPExpertStats:
    calls: int = 0
    tokens: int = 0
    logical_expert_bytes: int = 0
    explicit_pcie_bytes: int = 0


@dataclass(frozen=True)
class _MTPBankGeometry:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    bytes_per_expert: int


class MTPCPUExpertRunner:
    """Bounded MTP routed experts through FreeToken's existing CPU worker."""

    def __init__(
        self,
        banks,
        *,
        top_k: int,
        activation: str,
        renormalize: bool,
        max_tokens: int,
        num_threads: int,
        device: torch.device,
    ) -> None:
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        if device.type != "cuda":
            raise ValueError(f"MTP CPU expert runner needs a CUDA output device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if not 1 <= top_k <= banks.num_experts:
            raise ValueError(f"invalid MTP top_k={top_k} for {banks.num_experts} experts")
        if max_tokens < 1:
            raise ValueError(f"MTP max_tokens must be positive, got {max_tokens}")
        self.banks = banks
        self.top_k = int(top_k)
        self.renormalize = bool(renormalize)
        self.max_tokens = int(max_tokens)
        self.device = device
        self.stats = MTPExpertStats()
        self.executor: CpuMoeExecutor | None = CpuMoeExecutor(
            banks,
            top_k=top_k,
            activation=activation,
            apply_router_weight_on_input=False,
            num_threads=num_threads,
            max_tokens=max_tokens,
            device=self.device,
        )

    def route(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from freetoken.moe.fused import fused_topk

        if router_logits.shape != (hidden_states.shape[0], self.banks.num_experts):
            raise ValueError(
                "MTP router logits must be "
                f"[{hidden_states.shape[0]}, {self.banks.num_experts}], got "
                f"{tuple(router_logits.shape)}"
            )
        return fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )

    def run_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        executor = self.executor
        if executor is None:
            raise RuntimeError("MTP expert runner is closed")
        tokens = int(hidden_states.shape[0])
        if not 1 <= tokens <= self.max_tokens:
            raise ValueError(
                f"MTP expert calls support at most {self.max_tokens} tokens, got {tokens}"
            )
        if hidden_states.device != self.device or hidden_states.dtype is not torch.bfloat16:
            raise ValueError(
                f"MTP expert hidden states must be CUDA bfloat16 on {self.device}, got "
                f"{hidden_states.device}/{hidden_states.dtype}"
            )
        expected = (tokens, self.top_k)
        if tuple(topk_weights.shape) != expected or not topk_weights.dtype.is_floating_point:
            raise ValueError(f"MTP top-k weights must be floating {expected}")
        if tuple(topk_ids.shape) != expected or topk_ids.dtype is not torch.int32:
            raise ValueError(f"MTP top-k ids must be int32 {expected}")
        if topk_weights.device != self.device or topk_ids.device != self.device:
            raise ValueError("MTP routing tensors must use the runner CUDA device")

        output = executor.decode(0, hidden_states, topk_weights, topk_ids)
        self.stats.calls += 1
        self.stats.tokens += tokens
        self.stats.logical_expert_bytes += (
            tokens * self.top_k * self.banks.bytes_per_expert
        )
        # Explicit payload added by CpuMoeExecutor: x + ids + weights D2H, y H2D.
        self.stats.explicit_pcie_bytes += tokens * (
            2 * self.banks.hidden_size * torch.bfloat16.itemsize
            + self.top_k * (torch.int32.itemsize + torch.float32.itemsize)
        )
        return output

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        weights, ids = self.route(hidden_states, router_logits)
        return self.run_routed(hidden_states, weights, ids)

    def raise_if_unhealthy(self) -> None:
        if self.executor is not None:
            self.executor.raise_if_unhealthy()

    def close(self) -> None:
        executor = self.executor
        self.executor = None
        if executor is not None:
            # CpuMoeExecutor owns only Python/C++ resources; dropping the last reference is its
            # existing teardown contract.  The weak watchdog (when enabled) does not retain it.
            del executor
            gc.collect()


class MTPGPUExpertRunner:
    """Fully device-resident BF16 MTP routed experts; no per-call host transfer."""

    def __init__(
        self,
        banks,
        *,
        top_k: int,
        activation: str,
        renormalize: bool,
        max_tokens: int,
        num_threads: int,
        device: torch.device,
    ) -> None:
        if getattr(banks, "quant_format", None) != "bf16":
            raise ValueError(
                "MTP resident expert runner needs bf16 banks, got "
                f"{getattr(banks, 'quant_format', None)!r}"
            )
        if activation not in {"silu", "swish"}:
            raise ValueError(f"MTP resident experts support silu, got {activation!r}")
        if not 1 <= top_k <= banks.num_experts:
            raise ValueError(f"invalid MTP top_k={top_k} for {banks.num_experts} experts")
        if max_tokens < 1:
            raise ValueError(f"MTP max_tokens must be positive, got {max_tokens}")
        del num_threads  # the resident path owns no CPU worker pool
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.top_k = int(top_k)
        self.renormalize = bool(renormalize)
        self.max_tokens = int(max_tokens)
        self.device = device
        self.stats = MTPExpertStats()
        self.gate_up = banks.gate_up.to(device)
        self.down = banks.down.to(device)
        # Only the geometry outlives the host banks: holding the banks object here would
        # keep ~5 GB of committed CPU RAM alive for a handful of scalar reads.
        self.banks = _MTPBankGeometry(
            num_experts=int(banks.num_experts),
            hidden_size=int(banks.hidden_size),
            intermediate_size=int(banks.intermediate_size),
            bytes_per_expert=int(banks.bytes_per_expert),
        )
        self.resident_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.gate_up, self.down)
        )
        # Rows the routed gather may materialize before the expert-major loop takes over.
        # The gathered path costs tokens * top_k * bytes_per_expert of pure HBM traffic, which
        # is a rounding error at a speculative width and ~8 GB over a 128-row priming chunk.
        self.gather_max_tokens = max(
            _ROUTED_GATHER_MIN_TOKENS,
            _ROUTED_GATHER_BUDGET_BYTES
            // (self.top_k * max(self.banks.bytes_per_expert, 1)),
        )

    def route(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from freetoken.moe.fused import fused_topk

        if router_logits.shape != (hidden_states.shape[0], self.banks.num_experts):
            raise ValueError(
                "MTP router logits must be "
                f"[{hidden_states.shape[0]}, {self.banks.num_experts}], got "
                f"{tuple(router_logits.shape)}"
            )
        return fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )

    def run_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = int(hidden_states.shape[0])
        if not 1 <= tokens <= self.max_tokens:
            raise ValueError(
                f"MTP expert calls support at most {self.max_tokens} tokens, got {tokens}"
            )
        if hidden_states.device != self.device or hidden_states.dtype is not torch.bfloat16:
            raise ValueError(
                f"MTP expert hidden states must be bfloat16 on {self.device}, got "
                f"{hidden_states.device}/{hidden_states.dtype}"
            )
        expected = (tokens, self.top_k)
        if tuple(topk_weights.shape) != expected or not topk_weights.dtype.is_floating_point:
            raise ValueError(f"MTP top-k weights must be floating {expected}")
        if tuple(topk_ids.shape) != expected or topk_ids.dtype is not torch.int32:
            raise ValueError(f"MTP top-k ids must be int32 {expected}")
        if topk_weights.device != self.device or topk_ids.device != self.device:
            raise ValueError("MTP routing tensors must use the runner device")

        route_ids = topk_ids.reshape(-1).to(torch.int64)
        route_weights = topk_weights.reshape(-1).to(torch.float32)
        if tokens <= self.gather_max_tokens:
            result = self._run_gathered(hidden_states, route_ids, route_weights)
        else:
            result = self._run_expert_major(hidden_states, route_ids, route_weights)

        self.stats.calls += 1
        self.stats.tokens += tokens
        self.stats.logical_expert_bytes += (
            tokens * self.top_k * self.banks.bytes_per_expert
        )
        return result.to(hidden_states.dtype)

    def _run_gathered(
        self,
        hidden_states: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Route by GATHERING each (token, slot) pair's expert rows -- no host round trip.

        The expert-major loop below has to learn which experts were routed, and the only way to
        learn that on the host is ``route_ids.tolist()`` -- a device synchronization per expert
        call, which is both a per-cycle stall and outright illegal under CUDA-graph capture.
        Indexing the banks by ``route_ids`` instead keeps every decision on the device: the
        routing tensors are read only by the gather, and the whole call becomes one fixed
        sequence of kernels whose shapes depend on the token count alone.

        Numerically this is the same arithmetic in the same order per route -- one bf16 GEMM
        into fp32 accumulators, silu/multiply in fp32, a second bf16 GEMM, and an fp32 weighted
        sum over the ``top_k`` routes of a token. Only the batching changes: ``bmm`` over the
        pairs rather than one GEMM per expert over its rows.
        """
        tokens, hidden_size = hidden_states.shape
        # A router may mark a slot unrouted with a negative id. Clamp it to a real row so the
        # gather stays in bounds and zero its weight, which drops it from the sum -- the
        # branchless twin of the loop's ``if expert < 0: continue``.
        routed = route_ids >= 0
        safe_ids = torch.where(routed, route_ids, torch.zeros_like(route_ids))
        weights = torch.where(routed, route_weights, torch.zeros_like(route_weights))
        rows = hidden_states.repeat_interleave(self.top_k, dim=0).unsqueeze(1)
        gate_up = self.gate_up.index_select(0, safe_ids)
        projected = torch.bmm(rows, gate_up.transpose(1, 2)).squeeze(1)
        gate, up = projected.chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(
            hidden_states.dtype
        )
        down = self.down.index_select(0, safe_ids)
        value = torch.bmm(activated.unsqueeze(1), down.transpose(1, 2)).squeeze(1)
        return (value.float() * weights[:, None]).view(tokens, self.top_k, hidden_size).sum(1)

    def _run_expert_major(
        self,
        hidden_states: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Bulk fallback: one GEMM per routed expert, reading its bank row as a view.

        Beyond ``gather_max_tokens`` the gathered path's weight traffic stops being free, so
        prompt priming (up to 128 rows a chunk, one-off per request) still runs this. It
        synchronizes on ``route_ids`` and cannot be captured -- neither matters off the
        per-cycle speculative path.
        """
        tokens = int(hidden_states.shape[0])
        route_rows = torch.arange(
            tokens, device=self.device
        ).repeat_interleave(self.top_k)
        result = torch.zeros(
            hidden_states.shape, dtype=torch.float32, device=self.device
        )
        for expert in sorted({int(value) for value in route_ids.tolist()}):
            if expert < 0:
                continue
            selected = (route_ids == expert).nonzero(as_tuple=True)[0]
            rows = route_rows[selected]
            projected = hidden_states[rows] @ self.gate_up[expert].t()
            gate, up = projected.chunk(2, dim=-1)
            activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                hidden_states.dtype
            )
            value = activated @ self.down[expert].t()
            result.index_add_(0, rows, value.float() * route_weights[selected, None])
        return result

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        weights, ids = self.route(hidden_states, router_logits)
        return self.run_routed(hidden_states, weights, ids)

    def raise_if_unhealthy(self) -> None:
        return None

    def close(self) -> None:
        return None


class MTPExactExpertRunner(MTPCPUExpertRunner):
    """Exact file-backed BF16 placement."""


class MTPNVFP4ExpertRunner(MTPCPUExpertRunner):
    """Private six-bank NVFP4 placement."""


class MTPStackedExperts(BaseOP):
    """Checkpoint-shaped routed-expert tensors with execution supplied by a private runner."""

    def __init__(self, config: ModelConfig) -> None:
        self.gate_up_proj = torch.empty(
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        )
        self.down_proj = torch.empty(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )
        self._runner: MTPCPUExpertRunner | None = None

    def attach_runner(self, runner: MTPCPUExpertRunner) -> None:
        if runner.banks.num_experts != self.gate_up_proj.shape[0]:
            raise ValueError("MTP runner expert count does not match the model")
        if runner.banks.hidden_size != self.gate_up_proj.shape[2]:
            raise ValueError("MTP runner hidden size does not match the model")
        if runner.banks.intermediate_size * 2 != self.gate_up_proj.shape[1]:
            raise ValueError("MTP runner intermediate size does not match the model")
        self._runner = runner

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._runner is None:
            raise RuntimeError("MTP routed experts need an explicit private expert runner")
        if router_logits is None:
            raise ValueError("MTP routed experts need router logits")
        return self._runner.forward(hidden_states, router_logits)


def _resolve_state_owner(root: BaseOP, state_name: str):
    parts = state_name.split(".")
    owner = root
    for part in parts[:-1]:
        if part.isdigit() and hasattr(owner, "op_list"):
            owner = owner.op_list[int(part)]
        else:
            owner = getattr(owner, part)
    return owner, parts[-1]


@dataclass
class MTPStagingStats:
    stages: int = 0
    copied_bytes: int = 0
    copy_ms: float = 0.0
    peak_component_bytes: int = 0


class MTPStagedModelRunner:
    """Execute the real MTP model while only one bounded dense component is on CUDA."""

    _GROUP_PREFIXES = (
        ("pre_fc", ("pre_fc_norm_", "fc_")),
        ("attention_hc", ("layers.0.attn_hyper_connection.",)),
        ("attention", ("layers.0.self_attn.",)),
        ("mlp_hc", ("layers.0.mlp_hyper_connection.",)),
        ("mlp", ("layers.0.mlp.",)),
        ("final_hc", ("hyper_connection_mixer.",)),
    )

    def __init__(
        self,
        model: "Qwen4ExpMTPModel",
        cpu_weights: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
        resident: bool = False,
    ) -> None:
        self.model = model
        self.cpu_weights = dict(cpu_weights)
        self.device = torch.device(device)
        self.resident = bool(resident)
        self.stats = MTPStagingStats()
        self.groups: dict[str, tuple[str, ...]] = {}
        used: set[str] = set()
        for group, prefixes in self._GROUP_PREFIXES:
            names = tuple(
                name
                for name in sorted(self.cpu_weights)
                if any(name.startswith(prefix) for prefix in prefixes)
                and ".experts." not in name
            )
            self.groups[group] = names
            used.update(names)
        missing = set(self.cpu_weights) - used
        if missing:
            raise ValueError(f"unstaged MTP dense weights: {sorted(missing)}")
        if self.resident:
            for names in self.groups.values():
                for name in names:
                    owner, attr = _resolve_state_owner(self.model, name)
                    setattr(owner, attr, self.cpu_weights[name].to(self.device))

    @property
    def max_component_bytes(self) -> int:
        return max(
            sum(
                self.cpu_weights[name].numel() * self.cpu_weights[name].element_size()
                for name in names
            )
            for names in self.groups.values()
        )

    @property
    def resident_bytes(self) -> int:
        if not self.resident:
            return 0
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.cpu_weights.values()
        )

    @property
    def staging_bytes(self) -> int:
        """Device bytes a single forward stages on top of whatever is already resident."""

        return 0 if self.resident else self.max_component_bytes

    @contextmanager
    def stage(self, group: str):
        names = self.groups[group]
        if self.resident:
            self.stats.stages += 1
            yield
            return
        prior = []
        staged = []
        started = time.perf_counter()
        try:
            for name in names:
                owner, attr = _resolve_state_owner(self.model, name)
                prior.append((owner, attr, getattr(owner, attr)))
                value = self.cpu_weights[name].to(self.device, non_blocking=False)
                setattr(owner, attr, value)
                staged.append(value)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - started) * 1000.0
            copied = sum(value.numel() * value.element_size() for value in staged)
            self.stats.stages += 1
            self.stats.copied_bytes += copied
            self.stats.copy_ms += elapsed
            self.stats.peak_component_bytes = max(self.stats.peak_component_bytes, copied)
            yield
        finally:
            for owner, attr, value in prior:
                setattr(owner, attr, value)
            staged.clear()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.model.layers.op_list[0]
        with self.stage("pre_fc"):
            recursive = self.model.fuse_inputs(inputs_embeds, hidden_states)
        with self.stage("attention_hc"):
            block_input, inject = layer.attn_hyper_connection.mix(recursive)
        with self.stage("attention"):
            block_output = layer.self_attn.forward(block_input, batch)
        recursive = layer.attn_hyper_connection.combine(recursive, block_output, inject)
        with self.stage("mlp_hc"):
            block_input, inject = layer.mlp_hyper_connection.mix(recursive)
        with self.stage("mlp"):
            block_output = layer.mlp.forward(block_input)
        recursive = layer.mlp_hyper_connection.combine(recursive, block_output, inject)
        with self.stage("final_hc"):
            sample = self.model.hyper_connection_mixer.mix(recursive)[0]
        return sample, recursive


class Qwen4ExpMTPModel(BaseOP):
    """Native Qwen3.8 scheme-A MTP head, excluding the shared embedding and LM head."""

    def __init__(self, config: ModelConfig) -> None:
        if config.num_layers != 1 or config.is_linear_layer(0):
            raise ValueError("MTP model config must describe one full-attention layer")
        if config.qwen4_args.ple_layer_ids:
            raise ValueError("MTP model must not contain PLE")
        self.config = config
        width = config.qwen4_args.hc_count * config.hidden_size
        # These checkpoint norms are zero-centered (runtime multiplier 1+w).  One group means
        # pre_fc_norm_hidden normalizes the complete flattened HC stream, matching native MTP.
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(
            config.hidden_size, config.rms_norm_eps, num_groups=1
        )
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            width, config.rms_norm_eps, num_groups=1
        )
        self.fc_embedding = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.fc_hidden = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        layer = Qwen4ExpDecoderLayer(config, layer_id=0)
        # The normal fused layer declares per-expert GPU parameters.  Replace that child with
        # the checkpoint's two stacked BF16 banks before state-dict traversal/materialization.
        layer.mlp.experts = MTPStackedExperts(config)
        self.layers = OPList([layer])
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def fuse_inputs(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse following-token embeddings with target/recursive multi-stream hidden rows."""

        hidden_size = self.config.hidden_size
        hc_count = self.config.qwen4_args.hc_count
        if inputs_embeds.ndim != 2 or inputs_embeds.shape[-1] != hidden_size:
            raise ValueError(
                f"MTP embeddings must be [T, {hidden_size}], got {tuple(inputs_embeds.shape)}"
            )
        expected_hidden = hc_count * hidden_size
        if hidden_states.ndim != 2 or hidden_states.shape != (
            inputs_embeds.shape[0],
            expected_hidden,
        ):
            raise ValueError(
                f"MTP hidden must be [T, {expected_hidden}], got {tuple(hidden_states.shape)}"
            )

        embeddings = self.fc_embedding.forward(
            self.pre_fc_norm_embedding.forward(inputs_embeds)
        )
        hidden = self.pre_fc_norm_hidden.forward(hidden_states)
        hidden = hidden.view(-1, hc_count, hidden_size)
        hidden = self.fc_hidden.forward(hidden)
        return (hidden + embeddings.unsqueeze(-2)).flatten(-2)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(sample_hidden [T,H], recursive_multi [T,HC*H])``."""

        recursive = self.fuse_inputs(inputs_embeds, hidden_states)
        recursive = self.layers.op_list[0].forward(recursive, batch)
        sample, inject = self.hyper_connection_mixer.mix(recursive)
        assert inject is None
        return sample, recursive


class MTPDraftSampler:
    """Private sampler with its own generator; it never consumes target/global RNG state."""

    def __init__(self, *, seed: int, device: torch.device) -> None:
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def sample(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> int:
        if logits.ndim != 1:
            raise ValueError(f"MTP draft logits must be one-dimensional, got {tuple(logits.shape)}")
        if logits.device != self.device:
            raise ValueError(f"MTP draft logits must be on {self.device}, got {logits.device}")
        if temperature < 0:
            raise ValueError(f"MTP draft temperature cannot be negative, got {temperature}")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError(f"MTP draft top_p must be in (0,1], got {top_p}")
        vocab = logits.numel()
        if top_k is not None and top_k != -1 and top_k < 1:
            raise ValueError(f"MTP draft top_k must be -1 or positive, got {top_k}")
        if temperature == 0 or top_k == 1:
            return int(torch.argmax(logits))

        filtered = logits.float() / float(temperature)
        if top_k not in (None, -1) and top_k < vocab:
            keep = int(top_k)
            threshold = torch.topk(filtered, keep).values[-1]
            filtered = filtered.masked_fill(filtered < threshold, -float("inf"))
        probabilities = torch.softmax(filtered, dim=-1)
        if top_p is not None and top_p < 1:
            sorted_probabilities, order = torch.sort(probabilities, descending=True)
            remove = sorted_probabilities.cumsum(-1) - sorted_probabilities >= float(top_p)
            sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
            probabilities = torch.zeros_like(probabilities).scatter(
                0, order, sorted_probabilities
            )
            probabilities /= probabilities.sum()
        return int(torch.multinomial(probabilities, 1, generator=self.generator))


@dataclass(frozen=True)
class MTPProposalResult:
    tokens: torch.Tensor
    logits: torch.Tensor
    recursive_state: torch.Tensor
    qsa_blocks: torch.Tensor


class MTPProposalEngine:
    """Recursive shifted-token proposal chain with step-zero QSA selection reuse."""

    def __init__(self, sampler: MTPDraftSampler, *, max_depth: int = 8) -> None:
        if max_depth < 1:
            raise ValueError("MTP proposal max_depth must be positive")
        self.sampler = sampler
        self.max_depth = int(max_depth)

    def propose(
        self,
        *,
        initial_token: int,
        target_hidden: torch.Tensor,
        draft_tokens: int,
        embedding,
        step,
        lm_head,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> MTPProposalResult:
        if not 1 <= draft_tokens <= self.max_depth:
            raise ValueError(
                f"MTP proposal accepts at most {self.max_depth} draft tokens, got {draft_tokens}"
            )
        token = int(initial_token)
        recursive = target_hidden
        saved_qsa_blocks: torch.Tensor | None = None
        proposed: list[int] = []
        all_logits: list[torch.Tensor] = []
        for proposal_index in range(draft_tokens):
            # Native shifted alignment: hidden row i is paired with the actual following token.
            inputs_embeds = embedding(token)
            output = step(
                inputs_embeds,
                recursive,
                saved_qsa_blocks=saved_qsa_blocks,
            )
            if not isinstance(output, tuple) or len(output) != 3:
                raise ValueError("MTP proposal step must return sample, recursive, QSA blocks")
            sample_hidden, recursive, selected = output
            if proposal_index == 0:
                if not isinstance(selected, torch.Tensor):
                    raise ValueError("MTP proposal step zero must return selected QSA blocks")
                saved_qsa_blocks = selected.detach().clone()
            logits = lm_head(sample_hidden)
            if logits.ndim == 2 and logits.shape[0] == 1:
                logits = logits[0]
            if logits.ndim != 1:
                raise ValueError("MTP proposal LM head must produce one row of logits")
            token = self.sampler.sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            proposed.append(token)
            all_logits.append(logits.detach().clone())
        assert saved_qsa_blocks is not None
        return MTPProposalResult(
            tokens=torch.tensor(proposed, dtype=torch.int64, device=all_logits[0].device),
            logits=torch.stack(all_logits),
            recursive_state=recursive,
            qsa_blocks=saved_qsa_blocks,
        )


def accepted_draft_prefix(draft_tokens, target_tokens) -> int:
    accepted = 0
    for draft, target in zip(draft_tokens, target_tokens):
        if int(draft) != int(target):
            break
        accepted += 1
    return accepted


@dataclass
class MTPShadowMetrics:
    steps: int = 0
    proposal_ms: float = 0.0
    verification_ms: float = 0.0
    state_ms: float = 0.0
    draft_tokens: int = 0
    accepted_tokens: int = 0

    def record(
        self,
        *,
        proposal_ms: float,
        verification_ms: float,
        state_ms: float,
        draft: int,
        accepted: int,
    ) -> None:
        if min(proposal_ms, verification_ms, state_ms) < 0:
            raise ValueError("MTP shadow timings cannot be negative")
        if not 0 <= accepted <= draft:
            raise ValueError("MTP accepted tokens must be within the draft length")
        self.steps += 1
        self.proposal_ms += float(proposal_ms)
        self.verification_ms += float(verification_ms)
        self.state_ms += float(state_ms)
        self.draft_tokens += int(draft)
        self.accepted_tokens += int(accepted)

    def report(self) -> dict:
        if self.steps == 0:
            return {
                "steps": 0,
                "draft_tokens": 0,
                "accepted_tokens": 0,
                "projected_sequential_tokens_per_second": None,
            }
        proposal = self.proposal_ms / self.steps
        verification = self.verification_ms / self.steps
        state = self.state_ms / self.steps
        accepted = self.accepted_tokens / self.steps
        total_seconds = (proposal + verification + state) / 1000.0
        return {
            "steps": self.steps,
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "mean_accepted_per_step": accepted,
            "mean_proposal_ms": proposal,
            "mean_verification_ms": verification,
            "mean_state_ms": state,
            "projected_sequential_tokens_per_second": (
                (1.0 + accepted) / total_seconds if total_seconds > 0 else None
            ),
        }


_VERIFICATION_COMPONENTS = (
    "kv",
    "qsa_index",
    "qsa_ring",
    "recurrent",
    "ple",
    "page_table",
)


@dataclass
class MTPVerificationState:
    """All mutable target-side state for one private candidate verification."""

    components: dict[str, tuple[torch.Tensor, ...]]
    api_state: dict

    def __post_init__(self) -> None:
        for name in _VERIFICATION_COMPONENTS:
            if name not in self.components:
                raise ValueError(f"MTP verification state is missing {name}")
        normalized: dict[str, tuple[torch.Tensor, ...]] = {}
        for name, values in self.components.items():
            values = tuple(values)
            if not values or not all(isinstance(value, torch.Tensor) for value in values):
                raise ValueError(f"MTP verification component {name} must contain tensors")
            normalized[name] = values
        self.components = normalized

    def clone(self) -> "MTPVerificationState":
        return MTPVerificationState(
            components={
                name: tuple(value.detach().clone() for value in values)
                for name, values in self.components.items()
            },
            api_state=copy.deepcopy(self.api_state),
        )

    def digest(self) -> str:
        """Bit-level audit digest; intended for bounded active-state views, not whole caches."""

        digest = hashlib.sha256()
        for name in sorted(self.components):
            digest.update(name.encode("utf-8"))
            for tensor in self.components[name]:
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(str(tensor.dtype).encode("ascii"))
                raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                digest.update(raw)
        digest.update(
            json.dumps(self.api_state, sort_keys=True, separators=(",", ":"), default=repr).encode(
                "utf-8"
            )
        )
        return digest.hexdigest()


@dataclass(frozen=True)
class MTPVerificationResult:
    logits: torch.Tensor
    _prefix_states: tuple[MTPVerificationState, ...]

    @property
    def candidate_tokens(self) -> int:
        return len(self._prefix_states)

    def state_after(self, prefix_tokens: int) -> MTPVerificationState:
        if not 1 <= prefix_tokens <= len(self._prefix_states):
            raise IndexError(
                f"verified prefix must be in [1, {len(self._prefix_states)}], got {prefix_tokens}"
            )
        return self._prefix_states[prefix_tokens - 1].clone()


class MTPIsolatedTargetVerifier:
    """Run a bounded target candidate chain only on cloned active-state views."""

    def __init__(
        self, live_state: MTPVerificationState, *, max_candidate_tokens: int
    ) -> None:
        if max_candidate_tokens < 1:
            raise ValueError("max_candidate_tokens must be positive")
        self.live_state = live_state
        self.max_candidate_tokens = int(max_candidate_tokens)

    def verify(self, candidate_tokens, step) -> MTPVerificationResult:
        tokens = tuple(int(token) for token in candidate_tokens)
        if not tokens:
            raise ValueError("MTP verification needs at least one candidate token")
        if len(tokens) > self.max_candidate_tokens:
            raise ValueError(
                f"MTP verification accepts at most {self.max_candidate_tokens} candidates, "
                f"got {len(tokens)}"
            )
        before = self.live_state.digest()
        scratch = self.live_state.clone()
        logits: list[torch.Tensor] = []
        prefixes: list[MTPVerificationState] = []
        try:
            for token in tokens:
                result = step(token, scratch)
                if not isinstance(result, torch.Tensor) or result.ndim != 1:
                    raise ValueError("MTP target step must return one-dimensional logits")
                logits.append(result.detach().clone())
                prefixes.append(scratch.clone())
        finally:
            after = self.live_state.digest()
            if after != before:
                raise RuntimeError(
                    "MTP target verification changed live K/V, QSA, recurrent, PLE, "
                    "page-table, or API state"
                )
        return MTPVerificationResult(torch.stack(logits), tuple(prefixes))


@dataclass(frozen=True)
class MTPWeightEntry:
    model_name: str
    raw_names: tuple[str, ...]
    pad_to: int = 0
    expert: bool = False


@dataclass(frozen=True)
class MTPWeightPlan:
    entries: tuple[MTPWeightEntry, ...]

    @property
    def raw_names(self) -> tuple[str, ...]:
        return tuple(raw for entry in self.entries for raw in entry.raw_names)

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(entry.model_name for entry in self.entries)

    @property
    def expert_model_names(self) -> set[str]:
        return {entry.model_name for entry in self.entries if entry.expert}


def build_mtp_weight_plan(
    raw_names: Iterable[str], expected_model_names: Iterable[str]
) -> MTPWeightPlan:
    """Map every raw MTP tensor exactly once to the private model's state contract."""

    raw_list = list(raw_names)
    duplicate = sorted({name for name in raw_list if raw_list.count(name) > 1})
    if duplicate:
        raise ValueError(f"duplicate MTP source: {duplicate}")
    raw_set = set(raw_list)
    unexpected_prefix = sorted(name for name in raw_set if not name.startswith("mtp."))
    if unexpected_prefix:
        raise ValueError(f"unexpected MTP source: {unexpected_prefix}")

    entries: list[MTPWeightEntry] = []
    used: set[str] = set()
    model_list = list(expected_model_names)
    if len(model_list) != len(set(model_list)):
        raise ValueError("duplicate MTP model state name")
    for model_name in sorted(model_list):
        if model_name in _MTP_FUSIONS:
            sources, pad_to = _MTP_FUSIONS[model_name]
        else:
            sources, pad_to = (f"mtp.{model_name}",), 0
        missing = [name for name in sources if name not in raw_set]
        if missing:
            raise ValueError(f"missing MTP source for {model_name}: {missing}")
        overlap = used.intersection(sources)
        if overlap:
            raise ValueError(f"duplicate MTP source use: {sorted(overlap)}")
        used.update(sources)
        entries.append(
            MTPWeightEntry(
                model_name=model_name,
                raw_names=tuple(sources),
                pad_to=pad_to,
                expert=model_name in _EXPERT_MODEL_NAMES,
            )
        )

    extra = sorted(raw_set - used)
    if extra:
        raise ValueError(f"unexpected MTP source: {extra}")
    return MTPWeightPlan(tuple(entries))


class MTPWeightStore:
    """Read-only safetensor handles for the MTP sidecar and bounded fusion materialization."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).resolve()
        index_path = self.model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"MTP checkpoint index was not found: {index_path}")
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self._weight_map = {
            name: shard for name, shard in weight_map.items() if name.startswith("mtp.")
        }
        self.keys = tuple(sorted(self._weight_map))
        self._stack: ExitStack | None = None
        self._files: dict[str, object] = {}

    def __enter__(self) -> "MTPWeightStore":
        if self._stack is not None:
            raise RuntimeError("MTP weight store is already open")
        stack = ExitStack()
        try:
            for shard in sorted(set(self._weight_map.values())):
                path = self.model_path / shard
                self._files[shard] = stack.enter_context(
                    safetensors.safe_open(path, framework="pt", device="cpu")
                )
        except Exception:
            stack.close()
            self._files.clear()
            raise
        self._stack = stack
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._files.clear()

    def tensor(self, raw_name: str) -> torch.Tensor:
        if self._stack is None:
            raise RuntimeError("MTP weight store is not open")
        try:
            shard = self._weight_map[raw_name]
        except KeyError as exc:
            raise KeyError(f"unknown MTP tensor {raw_name!r}") from exc
        return self._files[shard].get_tensor(raw_name)  # type: ignore[union-attr]

    def materialize(
        self,
        entry: MTPWeightEntry,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        parts = [self.tensor(name).to(device=device) for name in entry.raw_names]
        if len(parts) == 1 and not entry.pad_to:
            return parts[0]
        pad = (-sum(part.shape[0] for part in parts)) % entry.pad_to if entry.pad_to else 0
        if pad:
            parts.append(
                torch.zeros(
                    pad,
                    *parts[0].shape[1:],
                    dtype=parts[0].dtype,
                    device=device,
                )
            )
        return torch.cat(parts, dim=0)

    def iter_materialized(
        self, plan: MTPWeightPlan, *, device: torch.device
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for entry in plan.entries:
            yield entry.model_name, self.materialize(entry, device=device)


__all__ = [
    "MTPBF16ExpertBanks",
    "MTPCPUExpertRunner",
    "MTPExactExpertRunner",
    "MTPGPUExpertRunner",
    "MTPNVFP4ExpertBanks",
    "MTPNVFP4ExpertRunner",
    "MTPDraftSampler",
    "MTPExpertStats",
    "MTPIsolatedTargetVerifier",
    "MTPProposalEngine",
    "MTPProposalResult",
    "MTPShadowMetrics",
    "MTPStagedModelRunner",
    "MTPStagingStats",
    "MTPStackedExperts",
    "MTPVerificationResult",
    "MTPVerificationState",
    "MTPWeightEntry",
    "MTPWeightPlan",
    "MTPWeightStore",
    "Qwen4ExpMTPModel",
    "accepted_draft_prefix",
    "build_mtp_weight_plan",
    "dequantize_nvfp4_rows",
    "derive_mtp_model_config",
    "quantize_nvfp4_rows",
]
