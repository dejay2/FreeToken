from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from flashlib.kernels.slot_cache import Stat

# The acceptance core lives in spec_sample so the integrated decode path can import it without
# this module's verifier / graph-runner / flashlib surface; re-exported here because the shadow
# observer and its tests import it from the verifier.
from freetoken.engine.spec_sample import (  # noqa: F401
    MTPAcceptanceResult,
    _sampling_probabilities_batch,
    _tensor_to_tuple,
    batched_speculative_accept,
)

# Survivable fixed-width capture now lives in spec_graph, shared with the integrated
# speculative step; re-exported here because the observer and its tests import it from the
# verifier and that surface has to keep working.
from freetoken.engine.spec_graph import (  # noqa: F401
    MTPGraphCaptureResult,
    _FixedWidthGraphRunner,
    _MTPVerifyGraphBuffer,
)


@dataclass
class MTPVerifyForwardResult:
    mode: str
    logits: torch.Tensor
    required_wall_ms: float
    core_cuda_ms: float
    required_synchronizations: int
    instrumentation_wall_ms: float
    instrumentation_synchronizations: int
    expert_movement: dict[str, int | bool]

    @property
    def wall_ms(self) -> float:
        return self.required_wall_ms

    @property
    def cuda_ms(self) -> float:
        return self.core_cuda_ms

    @property
    def synchronizations(self) -> int:
        return self.required_synchronizations + self.instrumentation_synchronizations


@dataclass(frozen=True)
class MTPVerifyComparison:
    rows: int
    matches: bool
    greedy_ids_match: bool
    max_abs_error: float
    rtol: float
    atol: float


def compare_verifier_logits(
    oracle: torch.Tensor,
    fast: torch.Tensor,
    *,
    rtol: float = 2e-2,
    atol: float = 2e-2,
) -> MTPVerifyComparison:
    if oracle.shape != fast.shape or oracle.ndim != 2:
        return MTPVerifyComparison(
            rows=int(fast.shape[0]) if fast.ndim else 0,
            matches=False,
            greedy_ids_match=False,
            max_abs_error=float("inf"),
            rtol=rtol,
            atol=atol,
        )
    difference = (oracle.float() - fast.float()).abs()
    max_abs_error = float(difference.max()) if difference.numel() else 0.0
    greedy_ids_match = bool(
        torch.equal(torch.argmax(oracle, dim=-1), torch.argmax(fast, dim=-1))
    )
    close = bool(torch.allclose(oracle.float(), fast.float(), rtol=rtol, atol=atol))
    return MTPVerifyComparison(
        rows=int(oracle.shape[0]),
        matches=close and greedy_ids_match,
        greedy_ids_match=greedy_ids_match,
        max_abs_error=max_abs_error,
        rtol=rtol,
        atol=atol,
    )


# Cross-kernel-path bf16 noise in logit units: the ulp at logit magnitude 16-32 is
# 0.125-0.25, and drift compounding through candidate-row KV reaches a few ulps on
# mass-bearing tokens. A structural error moves a mass-bearing logit by whole units.
_NOISE_TOLERANCE_LOGITS = 0.5


@dataclass(frozen=True)
class MTPVerifyDistributionComparison:
    rows: int
    matches: bool
    greedy_ids_match: bool
    max_abs_error: float
    worst_total_variation: float
    worst_greedy_logit_gap: float = 0.0
    worst_excess_total_variation: float = 0.0
    max_excess_total_variation: float = 0.05
    noise_tolerance: float = _NOISE_TOLERANCE_LOGITS


def compare_verifier_distributions(
    oracle: torch.Tensor,
    fast: torch.Tensor,
    *,
    noise_tolerance: float = _NOISE_TOLERANCE_LOGITS,
    max_excess_total_variation: float = 0.05,
) -> MTPVerifyDistributionComparison:
    """Compare two verifiers where their sampled output distributions actually differ.

    Raw-logit closeness cannot be bounded across two kernel paths: bf16 drift compounds per
    verify row through the earlier candidate rows' KV, so any fixed logit tolerance loses to
    depth. Plain total variation cannot be bounded either: on a knife-edge top pair, noise
    smaller than ``noise_tolerance`` legitimately moves TV with the pair's mass (observed
    live at 0.13), while the subtlest structural error known lands at 0.26 -- the ranges
    overlap. What separates them is logit space: noise moves logits at most a few bf16
    ulps, structure moves a mass-bearing logit by whole units. So the gate is the *excess*
    total variation -- between the fast distribution and the fast logits clamped into
    ``oracle +- noise_tolerance`` -- which is exactly the component of the divergence that
    beyond-noise logit moves are responsible for. Live noise scores <= 0.012 on it,
    structural errors 0.09-1.0. Plain TV and ``max_abs_error`` remain reported.

    An argmax mismatch is forgiven only as a near-tie: each path must score the other
    path's greedy pick within ``noise_tolerance`` of its own. Cross-path noise can collapse
    a two-ulp reference gap into an exact tie whose argmax breaks arbitrarily (observed
    live at 18.5/18.5); a structural flip fails the check from both sides by whole logits.
    """
    if oracle.shape != fast.shape or oracle.ndim != 2:
        return MTPVerifyDistributionComparison(
            rows=int(fast.shape[0]) if fast.ndim else 0,
            matches=False,
            greedy_ids_match=False,
            max_abs_error=float("inf"),
            worst_total_variation=float("inf"),
            worst_greedy_logit_gap=float("inf"),
            worst_excess_total_variation=float("inf"),
            max_excess_total_variation=max_excess_total_variation,
            noise_tolerance=noise_tolerance,
        )
    oracle_f = oracle.float()
    fast_f = fast.float()
    reference = torch.softmax(oracle_f, dim=-1)
    candidate = torch.softmax(fast_f, dim=-1)
    total_variation = 0.5 * (reference - candidate).abs().sum(-1)
    worst = float(total_variation.max()) if total_variation.numel() else 0.0
    difference = (oracle_f - fast_f).abs()
    max_abs_error = float(difference.max()) if difference.numel() else 0.0
    clamped = torch.softmax(
        torch.clamp(fast_f, oracle_f - noise_tolerance, oracle_f + noise_tolerance),
        dim=-1,
    )
    excess = 0.5 * (candidate - clamped).abs().sum(-1)
    worst_excess = float(excess.max()) if excess.numel() else 0.0
    oracle_ids = torch.argmax(oracle, dim=-1)
    fast_ids = torch.argmax(fast, dim=-1)
    mismatched = (oracle_ids != fast_ids).nonzero(as_tuple=True)[0]
    worst_greedy_gap = 0.0
    greedy_ids_match = True
    if mismatched.numel():
        oracle_gap = (
            oracle_f[mismatched, oracle_ids[mismatched]]
            - oracle_f[mismatched, fast_ids[mismatched]]
        )
        fast_gap = (
            fast_f[mismatched, fast_ids[mismatched]]
            - fast_f[mismatched, oracle_ids[mismatched]]
        )
        worst_greedy_gap = float(torch.maximum(oracle_gap, fast_gap).max())
        greedy_ids_match = worst_greedy_gap <= noise_tolerance
    return MTPVerifyDistributionComparison(
        rows=int(oracle.shape[0]),
        matches=greedy_ids_match and worst_excess <= max_excess_total_variation,
        greedy_ids_match=greedy_ids_match,
        max_abs_error=max_abs_error,
        worst_total_variation=worst,
        worst_greedy_logit_gap=worst_greedy_gap,
        worst_excess_total_variation=worst_excess,
        max_excess_total_variation=max_excess_total_variation,
        noise_tolerance=noise_tolerance,
    )


@dataclass(frozen=True)
class MTPProjectionSample:
    prompt_ms: float
    draft_ms: float
    verify_ms: float
    state_ms: float
    acceptance_ms: float
    emitted_tokens: int

    def __post_init__(self) -> None:
        times = (
            self.prompt_ms,
            self.draft_ms,
            self.verify_ms,
            self.state_ms,
            self.acceptance_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in times):
            raise ValueError("projection times must be finite and non-negative")
        if sum(times) <= 0:
            raise ValueError("total projection time must be positive")
        if self.emitted_tokens < 1:
            raise ValueError("a projection cycle must emit at least one token")

    @property
    def cycle_ms(self) -> float:
        return self.draft_ms + self.verify_ms + self.state_ms + self.acceptance_ms

    @property
    def total_ms(self) -> float:
        return self.prompt_ms + self.cycle_ms

    @property
    def tokens_per_second(self) -> float:
        return 1000.0 * self.emitted_tokens / self.total_ms

    @property
    def component_total_ms(self) -> float:
        return sum(
            (self.prompt_ms, self.draft_ms, self.verify_ms, self.state_ms,
             self.acceptance_ms)
        )

    @property
    def components(self) -> dict[str, float | int]:
        return {
            "P": self.prompt_ms,
            "D": self.draft_ms,
            "V": self.verify_ms,
            "S": self.state_ms,
            "A": self.acceptance_ms,
            "E": self.emitted_tokens,
        }


def sampling_support_order_matches(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> bool:
    if first_logits.shape != second_logits.shape or first_logits.ndim != 2:
        return False
    first = _sampling_probabilities_batch(
        first_logits, temperature=temperature, top_k=top_k, top_p=top_p
    )
    second = _sampling_probabilities_batch(
        second_logits, temperature=temperature, top_k=top_k, top_p=top_p
    )
    first_support = first > 0
    second_support = second > 0
    if not torch.equal(first_support, second_support):
        return False
    for row in range(first.shape[0]):
        count = int(first_support[row].sum())
        first_order = torch.argsort(first[row], descending=True)[:count]
        second_order = torch.argsort(second[row], descending=True)[:count]
        if not torch.equal(first_order, second_order):
            return False
    return True


def sampling_distribution_divergence(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    noise_tolerance: float | None = None,
) -> float:
    """Worst row's total variation between the two filtered sampling distributions.

    What a client can observe is the distribution the sampler draws from, not the rank order
    inside it: among bf16 near-ties that order is not a stable property across kernel paths.
    Greedy rows (``temperature <= 0`` or ``top_k == 1``) filter to one-hot, so this is 0.0 when
    the argmax agrees and 1.0 when it moves.

    With ``noise_tolerance`` set, the divergence is measured in excess terms: the second
    distribution against itself with its logits clamped into ``first +- noise_tolerance``.
    Temperature sharpening amplifies within-noise logit moves without bound (a greedy filter
    turns a knife-edge tie flip into divergence 1.0), so a fixed bound on the plain filtered
    divergence cannot separate path noise from structure -- the beyond-noise component can.
    """
    if first_logits.shape != second_logits.shape or first_logits.ndim != 2:
        return float("inf")
    second = _sampling_probabilities_batch(
        second_logits, temperature=temperature, top_k=top_k, top_p=top_p
    )
    if noise_tolerance is None:
        first = _sampling_probabilities_batch(
            first_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
    else:
        first_f = first_logits.float()
        first = _sampling_probabilities_batch(
            torch.clamp(
                second_logits.float(),
                first_f - noise_tolerance,
                first_f + noise_tolerance,
            ),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    divergence = 0.5 * (first - second).abs().sum(-1)
    return float(divergence.max()) if divergence.numel() else 0.0




class MTPFastVerifier:
    """Private eager target checker; the caller owns scratch-state isolation."""

    def __init__(
        self,
        target_ctx,
        target_model,
        moe_offload_cache,
        device: torch.device,
    ) -> None:
        self.target_ctx = target_ctx
        self.target_model = target_model
        self.moe_offload_cache = moe_offload_cache
        self.device = torch.device(device)

    @staticmethod
    def _bytes_per_expert(cache) -> int:
        banks = getattr(cache, "bank_caches", None)
        if not isinstance(banks, dict):
            return 0
        total = 0
        for tensor in banks.values():
            if isinstance(tensor, torch.Tensor) and tensor.ndim:
                total += tensor[0].numel() * tensor.element_size()
        return int(total)

    @staticmethod
    def _stats_snapshot(cache):
        if cache is None or not getattr(cache, "collect_stats", False):
            return None
        if getattr(cache, "decode_target", None) == "hybrid":
            names = ("stat_calls", "stat_active", "stat_missing", "stat_fetched")
            values = [getattr(cache, name, None) for name in names]
            if not all(isinstance(value, torch.Tensor) for value in values):
                return None
            return "hybrid", tuple(value.clone() for value in values)
        lru = getattr(cache, "lru_stats", None)
        fetched = getattr(cache, "stat_fetched", None)
        if not isinstance(lru, torch.Tensor) or not isinstance(fetched, torch.Tensor):
            return None
        return "lru", (lru.clone(), fetched.clone())

    @staticmethod
    def _movement_counters(cache, before):
        """Queue counter reductions before the verifier's single final synchronization."""
        if before is None:
            return None
        kind, prior = before
        if kind == "hybrid":
            names = ("stat_calls", "stat_active", "stat_missing", "stat_fetched")
            return tuple(getattr(cache, name) - old for name, old in zip(names, prior))
        prior_lru, prior_fetched = prior
        delta = getattr(cache, "lru_stats") - prior_lru
        active = delta[:, Stat.ACTIVE].sum()
        missing = delta[:, Stat.MISS].sum()
        calls = delta[:, Stat.CALLS].sum()
        fetched = getattr(cache, "stat_fetched") - prior_fetched
        return calls, active, missing, fetched

    @classmethod
    def _movement_result(cls, cache, counters) -> dict[str, int | bool]:
        bytes_per_expert = cls._bytes_per_expert(cache)
        if counters is None:
            return {
                "available": False,
                "layer_calls": 0,
                "active_experts": 0,
                "hit_experts": 0,
                "missing_experts": 0,
                "fetched_experts": 0,
                "cpu_experts": 0,
                "d2d_rows": 0,
                "bytes_per_expert": bytes_per_expert,
                # GPU-owned MoE layers never call ensure_experts, so they add nothing to
                # active/missing/fetched and the reconciliation invariant below is unchanged.
                # Reported so a reader knows why h2d_bytes covers fewer layers than the model
                # has (bytes_per_expert / actual_h2d_bytes are over bank_caches only).
                "gpu_owned_layers": len(getattr(cache, "gpu_owned_layer_ids", ()) or ()),
                "h2d_bytes": 0,
                "d2d_bytes": 0,
                "transfer_bytes": 0,
                "movement_reconciled": False,
            }
        calls, active, missing, fetched = (int(value) for value in counters)
        if min(calls, active, missing, fetched) < 0:
            raise RuntimeError("MTP specialist movement counters went backwards")
        if missing > active or fetched > missing:
            raise RuntimeError("MTP specialist movement counters do not reconcile")
        hybrid = getattr(cache, "decode_target", None) == "hybrid"
        cpu_experts = missing - fetched if hybrid else 0
        hit_experts = active - missing
        h2d_bytes = fetched * bytes_per_expert
        reconciled = (
            hit_experts + fetched + cpu_experts == active
            if hybrid
            else hit_experts + missing == active and fetched == missing
        )
        return {
            "available": True,
            "layer_calls": calls,
            "active_experts": active,
            "hit_experts": hit_experts,
            "missing_experts": missing,
            "fetched_experts": fetched,
            "cpu_experts": cpu_experts,
            "d2d_rows": 0,
            "bytes_per_expert": bytes_per_expert,
            # GPU-owned MoE layers never call ensure_experts, so they add nothing to
            # active/missing/fetched and the reconciliation invariant below is unchanged.
            # Reported so a reader knows why h2d_bytes covers fewer layers than the model
            # has (bytes_per_expert / actual_h2d_bytes are over bank_caches only).
            "gpu_owned_layers": len(getattr(cache, "gpu_owned_layer_ids", ()) or ()),
            "h2d_bytes": h2d_bytes,
            "d2d_bytes": 0,
            "transfer_bytes": h2d_bytes,
            "movement_reconciled": reconciled,
        }

    def forward_eager(self, batch) -> MTPVerifyForwardResult:
        rows = int(batch.input_ids.shape[0])
        if rows not in (2, 3, 4):
            raise ValueError("fast MTP verification requires exactly 2, 3, or 4 rows")
        if not batch.is_prefill or len(batch.reqs) != 1:
            raise ValueError("fast MTP verification requires one prefill request")
        if int(batch.reqs[0].extend_len) != rows:
            raise ValueError(
                f"fast MTP verification request extend_len {batch.reqs[0].extend_len} "
                f"!= {rows} token rows"
            )
        batch.mtp_verify = True
        use_cuda = self.device.type == "cuda"
        instrumentation_started = time.perf_counter()
        before = self._stats_snapshot(self.moe_offload_cache)
        instrumentation_synchronizations = 0
        if use_cuda and before is not None:
            torch.cuda.synchronize(self.device)
            instrumentation_synchronizations = 1
        instrumentation_wall_ms = (
            time.perf_counter() - instrumentation_started
        ) * 1000.0

        start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
        end_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
        started = time.perf_counter()
        if start_event is not None:
            start_event.record()
        with self.target_ctx.forward_batch(batch):
            hidden = self.target_model.model.forward(batch.input_ids, batch)
            logits = self.target_model.lm_head.forward_all(hidden)
        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize(self.device)
            core_cuda_ms = float(start_event.elapsed_time(end_event))
            required_synchronizations = 1
        else:
            core_cuda_ms = (time.perf_counter() - started) * 1000.0
            required_synchronizations = 0
        required_wall_ms = (time.perf_counter() - started) * 1000.0

        instrumentation_started = time.perf_counter()
        if logits.ndim != 2 or logits.shape[0] != rows:
            raise RuntimeError(
                f"fast MTP verifier returned {tuple(logits.shape)}, expected {rows} logit rows"
            )
        finite = torch.isfinite(logits).all()
        movement_counters = self._movement_counters(self.moe_offload_cache, before)
        if use_cuda:
            torch.cuda.synchronize(self.device)
            instrumentation_synchronizations += 1
        if not bool(finite):
            raise RuntimeError("fast MTP verifier returned non-finite logits")
        movement = self._movement_result(self.moe_offload_cache, movement_counters)
        instrumentation_wall_ms += (
            time.perf_counter() - instrumentation_started
        ) * 1000.0
        return MTPVerifyForwardResult(
            mode="fast-eager",
            logits=logits,
            required_wall_ms=required_wall_ms,
            core_cuda_ms=core_cuda_ms,
            required_synchronizations=required_synchronizations,
            instrumentation_wall_ms=instrumentation_wall_ms,
            instrumentation_synchronizations=instrumentation_synchronizations,
            expert_movement=movement,
        )


class MTPVerifyGraphRunner(_FixedWidthGraphRunner):
    """Private fixed-width CUDA graphs for the 2--4-row target checker.

    The survivable-capture machinery lives in ``spec_graph`` so the integrated speculative step
    shares it; what stays here is the observer's own surface -- MoE-movement instrumentation,
    per-replay timing, and the logits-only fp32 buffer.
    """

    widths = (2, 3, 4)

    def __init__(
        self,
        *,
        target_ctx,
        target_model,
        attn_backend,
        moe_cache,
        device: torch.device,
        vocab_size: int,
        guard_bytes: int,
    ) -> None:
        if int(vocab_size) <= 0:
            raise ValueError("MTP graph vocabulary size must be positive")
        super().__init__(
            target_ctx=target_ctx,
            target_model=target_model,
            attn_backend=attn_backend,
            device=device,
            guard_bytes=guard_bytes,
            widths=self.widths,
        )
        self.moe_cache = moe_cache
        self.vocab_size = int(vocab_size)

    def _new_buffer(self, width: int) -> _MTPVerifyGraphBuffer:
        return _MTPVerifyGraphBuffer.init(width, self.vocab_size, self.device)

    def _estimated_buffer_bytes(self, width: int) -> int:
        return (
            width * self.vocab_size * torch.float32.itemsize
            + width * (3 * torch.int32.itemsize + 3 * torch.int64.itemsize)
            + 64
        )

    def _forward(self, batch) -> torch.Tensor:
        hidden = self.target_model.model.forward(batch.input_ids, batch)
        return self.target_model.lm_head.forward_all(hidden)

    def _run(self, batch, buffer, *, allocate: bool) -> None:
        buffer.logits.copy_(self._forward(batch))

    def capture_forward(self, batch) -> MTPVerifyForwardResult:
        """Adapter used inside the observer's scratch-state transaction."""
        batch.mtp_verify = True
        instrumentation_started = time.perf_counter()
        before = MTPFastVerifier._stats_snapshot(self.moe_cache)
        instrumentation_synchronizations = 0
        if self.device.type == "cuda" and before is not None:
            torch.cuda.synchronize(self.device)
            instrumentation_synchronizations = 1
        instrumentation_wall_ms = (
            time.perf_counter() - instrumentation_started
        ) * 1000.0
        started = time.perf_counter()
        support = self.capture(batch)
        required_wall_ms = (time.perf_counter() - started) * 1000.0
        instrumentation_started = time.perf_counter()
        movement_counters = MTPFastVerifier._movement_counters(self.moe_cache, before)
        if support.status == "captured":
            logits = self._buffers[support.width].logits
        else:
            # the caller discards these rows and falls back to the eager checker; after a failed
            # capture the CUDA context may be unusable, so allocate nothing on the device
            logits = torch.empty((support.width, 0), dtype=torch.float32)
        if self.device.type == "cuda" and movement_counters is not None:
            torch.cuda.synchronize(self.device)
            instrumentation_synchronizations += 1
        movement = MTPFastVerifier._movement_result(
            self.moe_cache, movement_counters
        )
        instrumentation_wall_ms += (
            time.perf_counter() - instrumentation_started
        ) * 1000.0
        return MTPVerifyForwardResult(
            mode="graph-capture",
            logits=logits,
            required_wall_ms=required_wall_ms,
            core_cuda_ms=required_wall_ms,
            required_synchronizations=support.synchronizations,
            instrumentation_wall_ms=instrumentation_wall_ms,
            instrumentation_synchronizations=instrumentation_synchronizations,
            expert_movement=movement,
        )

    def replay(self, batch) -> MTPVerifyForwardResult:
        batch.mtp_verify = True
        width = self._validate_width(batch)
        graph = self._graphs.get(width)
        if graph is None:
            support = self._support.get(width)
            reason = support.reason if support is not None else "NOT_CAPTURED"
            raise RuntimeError(f"MTP graph width {width} is unavailable: {reason}")
        buffer = self._buffers[width]
        instrumentation_started = time.perf_counter()
        before = MTPFastVerifier._stats_snapshot(self.moe_cache)
        instrumentation_synchronizations = 0
        if before is not None:
            torch.cuda.synchronize(self.device)
            instrumentation_synchronizations = 1
        instrumentation_wall_ms = (
            time.perf_counter() - instrumentation_started
        ) * 1000.0
        start_event, end_event = self._events[width]
        started = time.perf_counter()
        start_event.record()
        self._replay_into_buffers(batch)
        owned_logits = buffer.logits.clone()
        end_event.record()
        torch.cuda.synchronize(self.device)
        core_cuda_ms = float(start_event.elapsed_time(end_event))
        required_wall_ms = (time.perf_counter() - started) * 1000.0

        instrumentation_started = time.perf_counter()
        finite = torch.isfinite(owned_logits).all()
        movement_counters = MTPFastVerifier._movement_counters(
            self.moe_cache, before
        )
        torch.cuda.synchronize(self.device)
        if not bool(finite):
            raise RuntimeError("fast MTP graph returned non-finite logits")
        movement = MTPFastVerifier._movement_result(
            self.moe_cache, movement_counters
        )
        instrumentation_wall_ms += (
            time.perf_counter() - instrumentation_started
        ) * 1000.0
        instrumentation_synchronizations += 1
        return MTPVerifyForwardResult(
            mode="fast-graph",
            logits=owned_logits,
            required_wall_ms=required_wall_ms,
            core_cuda_ms=core_cuda_ms,
            required_synchronizations=1,
            instrumentation_wall_ms=instrumentation_wall_ms,
            instrumentation_synchronizations=instrumentation_synchronizations,
            expert_movement=movement,
        )


__all__ = [
    "MTPAcceptanceResult",
    "MTPFastVerifier",
    "MTPGraphCaptureResult",
    "MTPProjectionSample",
    "MTPVerifyComparison",
    "MTPVerifyDistributionComparison",
    "MTPVerifyForwardResult",
    "MTPVerifyGraphRunner",
    "batched_speculative_accept",
    "compare_verifier_distributions",
    "compare_verifier_logits",
    "sampling_distribution_divergence",
    "sampling_support_order_matches",
]
