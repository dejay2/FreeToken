from __future__ import annotations

import gc
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch
from flashlib.kernels.slot_cache import Stat

_GRAPH_CAPTURE_RESERVE_FLOOR = 64 << 20
# the graph runner has no config handle, so the capture-failure traceback lands in the
# approved private evidence dir by absolute path; every write is best-effort
_CAPTURE_FAILURE_EVIDENCE_DIR = Path(
    r"D:\FreeToken-ple-mmap-vision\.local\mtp-spike\evidence"
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
class MTPAcceptanceResult:
    accepted_prefix: int
    corrected_token: int
    target_tokens: tuple[int, ...]
    acceptance_probabilities: tuple[float, ...]
    draft_probabilities: tuple[float, ...]
    target_probabilities: tuple[float, ...]
    greedy: bool
    required_wall_ms: float = field(compare=False)
    required_synchronizations: int = 0
    instrumentation_wall_ms: float = field(default=0.0, compare=False)
    instrumentation_synchronizations: int = 0

    @property
    def acceptance_ms(self) -> float:
        return self.required_wall_ms

    @property
    def synchronizations(self) -> int:
        return self.required_synchronizations + self.instrumentation_synchronizations


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


def _sampling_probabilities_batch(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("sampling logits must be a row matrix")
    if temperature <= 0 or top_k == 1:
        winners = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(1, winners, 1.0)
    filtered = logits.float() / float(temperature)
    if 1 <= top_k < filtered.shape[1]:
        threshold = torch.topk(filtered, top_k, dim=-1).values[:, -1:]
        filtered = filtered.masked_fill(filtered < threshold, -float("inf"))
    probabilities = torch.softmax(filtered, dim=-1)
    if top_p < 1:
        ordered, indices = probabilities.sort(dim=-1, descending=True)
        remove = ordered.cumsum(dim=-1) - ordered >= top_p
        ordered = ordered.masked_fill(remove, 0)
        probabilities = torch.zeros_like(probabilities).scatter(1, indices, ordered)
        probabilities /= probabilities.sum(dim=-1, keepdim=True)
    return probabilities


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


def _tensor_to_tuple(tensor: torch.Tensor, cast) -> tuple:
    return tuple(cast(value) for value in tensor.tolist())


def batched_speculative_accept(
    *,
    proposals: list[int],
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> MTPAcceptanceResult:
    depth = len(proposals)
    if depth not in (1, 2, 3):
        raise ValueError("batched MTP acceptance requires one to three proposals")
    if draft_logits.ndim != 2 or draft_logits.shape[0] != depth:
        raise ValueError("draft logits must have one row per proposal")
    if target_logits.ndim != 2 or target_logits.shape[0] != depth + 1:
        raise ValueError("target acceptance requires depth+1 logit rows")
    if draft_logits.shape[1] != target_logits.shape[1]:
        raise ValueError("draft and target logits must use the same vocabulary")
    if draft_logits.device != target_logits.device:
        raise ValueError("draft and target logits must use the same device")

    started = time.perf_counter()
    device = target_logits.device
    proposal_ids = torch.tensor(proposals, dtype=torch.int64, device=device)
    greedy = temperature <= 0 or top_k == 1
    q_rows = _sampling_probabilities_batch(
        draft_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    p_rows = _sampling_probabilities_batch(
        target_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    gather_ids = proposal_ids.unsqueeze(1)
    q_selected = q_rows.gather(1, gather_ids).squeeze(1)
    p_selected = p_rows[:depth].gather(1, gather_ids).squeeze(1)
    ratios = torch.where(
        q_selected <= 0,
        torch.ones_like(q_selected),
        (p_selected / q_selected).clamp(max=1.0),
    )
    target_tokens_device = torch.argmax(target_logits[:depth], dim=-1)

    if greedy:
        accepted_rows = proposal_ids == target_tokens_device
        accepted_prefix_device = accepted_rows.to(torch.int32).cumprod(0).sum()
        correction_options = torch.cat(
            (target_tokens_device, torch.argmax(target_logits[depth:depth + 1], dim=-1))
        )
        corrected_device = correction_options[accepted_prefix_device.to(torch.int64)]
    else:
        draws = torch.rand(depth, generator=generator, device=device)
        accepted_rows = draws <= ratios
        accepted_prefix_device = accepted_rows.to(torch.int32).cumprod(0).sum()
        residual = (p_rows[:depth] - q_rows).clamp_min(0)
        residual_sum = residual.sum(dim=-1, keepdim=True)
        residual = torch.where(residual_sum > 0, residual, p_rows[:depth])
        residual /= residual.sum(dim=-1, keepdim=True)
        correction_rows = torch.cat((residual, p_rows[depth:depth + 1]), dim=0)
        possible_corrections = torch.multinomial(
            correction_rows, 1, generator=generator
        ).squeeze(1)
        corrected_device = possible_corrections[
            accepted_prefix_device.to(torch.int64)
        ]

    required_synchronizations = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        required_synchronizations = 1
    required_wall_ms = (time.perf_counter() - started) * 1000.0

    instrumentation_started = time.perf_counter()
    accepted_prefix = int(accepted_prefix_device)
    corrected_token = int(corrected_device)
    target_tokens = _tensor_to_tuple(target_tokens_device, int)
    acceptance_probabilities = _tensor_to_tuple(ratios, float)
    draft_probabilities = _tensor_to_tuple(q_selected, float)
    target_probabilities = _tensor_to_tuple(p_selected, float)
    instrumentation_wall_ms = (
        time.perf_counter() - instrumentation_started
    ) * 1000.0
    result = MTPAcceptanceResult(
        accepted_prefix=accepted_prefix,
        corrected_token=corrected_token,
        target_tokens=target_tokens,
        acceptance_probabilities=acceptance_probabilities,
        draft_probabilities=draft_probabilities,
        target_probabilities=target_probabilities,
        greedy=greedy,
        required_wall_ms=required_wall_ms,
        required_synchronizations=required_synchronizations,
        instrumentation_wall_ms=instrumentation_wall_ms,
        instrumentation_synchronizations=0,
    )
    return result


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


@dataclass(frozen=True)
class MTPGraphCaptureResult:
    width: int
    status: str
    reason: str
    memory_bytes: int
    synchronizations: int = 0


@dataclass
class _MTPVerifyGraphBuffer:
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_loc: torch.Tensor
    rope_positions: torch.Tensor
    linear_table_idx: torch.Tensor
    fla_cu_seqlens: torch.Tensor
    fla_has_initial_state: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(
        cls, width: int, vocab_size: int, device: torch.device
    ) -> _MTPVerifyGraphBuffer:
        return cls(
            input_ids=torch.empty(width, dtype=torch.int32, device=device),
            positions=torch.empty(width, dtype=torch.int32, device=device),
            out_loc=torch.empty(width, dtype=torch.int32, device=device),
            rope_positions=torch.empty((3, width), dtype=torch.int64, device=device),
            linear_table_idx=torch.empty(1, dtype=torch.int32, device=device),
            # int64 so the GDN kernels' `.to(torch.int64)` is an identity no-op — the fla
            # chunk-index cache is keyed on tensor identity, and a per-call cast would miss
            # it inside capture and rebuild the indices via a pageable H2D copy
            fla_cu_seqlens=torch.tensor([0, width], dtype=torch.int64, device=device),
            fla_has_initial_state=torch.ones(1, dtype=torch.bool, device=device),
            logits=torch.empty((width, vocab_size), dtype=torch.float32, device=device),
        )

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in vars(self).values()
            if isinstance(tensor, torch.Tensor)
        )

    def copy_from(self, batch) -> None:
        width = self.input_ids.shape[0]
        if int(batch.input_ids.shape[0]) != width:
            raise ValueError("MTP graph replay width does not match capture width")
        if not batch.is_prefill or batch.size != 1 or batch.padded_size != 1:
            raise ValueError("MTP graph replay requires one prefill request")
        if not getattr(batch, "mtp_verify", False):
            raise ValueError("MTP graph replay requires the private verification marker")
        self.input_ids.copy_(batch.input_ids)
        self.positions.copy_(batch.positions)
        self.out_loc.copy_(batch.out_loc)
        rope_positions = getattr(batch, "rope_positions", None)
        if rope_positions is None:
            self.rope_positions.copy_(
                batch.positions.to(torch.int64).expand(3, -1)
            )
        else:
            self.rope_positions.copy_(rope_positions)
        self.linear_table_idx.copy_(batch.linear_table_idx[:1])
        fla = getattr(batch, "fla_metadata", None)
        has_initial_state = getattr(fla, "has_initial_state", None)
        if has_initial_state is not None:
            self.fla_has_initial_state.copy_(has_initial_state[:1])
        else:
            self.fla_has_initial_state.fill_(True)

    def bind(self, batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        batch.input_ids = self.input_ids
        batch.positions = self.positions
        batch.out_loc = self.out_loc
        batch.rope_positions = self.rope_positions
        batch.linear_table_idx = self.linear_table_idx
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens,
            cache_indices=self.linear_table_idx,
            has_initial_state=self.fla_has_initial_state,
            max_seq_len=int(self.input_ids.shape[0]),
        )
        batch.mtp_verify = True


class MTPVerifyGraphRunner:
    """Private fixed-width CUDA graphs for the 2--4-row target checker."""

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
        self.target_ctx = target_ctx
        self.target_model = target_model
        self.attn_backend = attn_backend
        self.moe_cache = moe_cache
        self.device = torch.device(device)
        self.vocab_size = int(vocab_size)
        self.guard_bytes = int(guard_bytes)
        if self.vocab_size <= 0:
            raise ValueError("MTP graph vocabulary size must be positive")
        if self.guard_bytes < 0:
            raise ValueError("MTP graph guard bytes must be non-negative")
        self._stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self._pool = None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._buffers: dict[int, _MTPVerifyGraphBuffer] = {}
        self._batches: dict[int, object] = {}
        self._events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._memory_bytes: dict[int, int] = {}
        self._support: dict[int, MTPGraphCaptureResult] = {}
        self._last_attempt: dict[int, MTPGraphCaptureResult] = {}
        self._fla_index_pins: dict[int, tuple[torch.Tensor, ...]] = {}
        self._destroyed = False
        self._graphs_disabled = False

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    @property
    def owned_buffer_bytes(self) -> int:
        attention_bytes = getattr(
            self.attn_backend, "mtp_verify_graph_bytes", lambda: 0
        )()
        return sum(buffer.nbytes for buffer in self._buffers.values()) + int(
            attention_bytes
        )

    @property
    def live_graph_memory_bytes(self) -> int:
        return sum(self._memory_bytes.values())

    def support(self, width: int) -> MTPGraphCaptureResult | None:
        return self._support.get(width)

    def last_attempt(self, width: int) -> MTPGraphCaptureResult | None:
        return self._last_attempt.get(width)

    def _validate_width(self, batch) -> int:
        if self._destroyed:
            raise RuntimeError("MTP graph runner was destroyed")
        width = int(batch.input_ids.shape[0])
        if width not in self.widths:
            raise ValueError("MTP graph width must be exactly 2, 3, or 4 token rows")
        if not batch.is_prefill or batch.size != 1 or batch.padded_size != 1:
            raise ValueError("MTP graph requires one prefill request")
        # the graph's fla_cu_seqlens is a fixed [0, width]; a request whose extend_len says
        # otherwise would replay against metadata that does not describe it
        if int(batch.reqs[0].extend_len) != width:
            raise ValueError(
                f"MTP graph request extend_len {batch.reqs[0].extend_len} != {width} token rows"
            )
        if not getattr(batch, "mtp_verify", False):
            raise ValueError("MTP graph requires the private verification marker")
        return width

    def _forward(self, batch) -> torch.Tensor:
        hidden = self.target_model.model.forward(batch.input_ids, batch)
        return self.target_model.lm_head.forward_all(hidden)

    def _discard_attention_width(self, width: int) -> None:
        discard = getattr(self.attn_backend, "discard_mtp_verify_graph", None)
        if discard is not None:
            discard(width)

    def _record_attempt(
        self,
        width: int,
        *,
        status: str,
        reason: str,
        memory_bytes: int = 0,
        synchronizations: int = 0,
    ) -> MTPGraphCaptureResult:
        if status not in {"captured", "retryable", "permanently-unsupported"}:
            raise ValueError(f"invalid MTP graph attempt status {status!r}")
        result = MTPGraphCaptureResult(
            width=width,
            status=status,
            reason=reason,
            memory_bytes=memory_bytes,
            synchronizations=synchronizations,
        )
        self._last_attempt[width] = result
        if status in {"captured", "permanently-unsupported"}:
            self._support[width] = result
        else:
            self._support.pop(width, None)
        return result

    def _cleanup_failed_attempt(self, width: int) -> None:
        self._discard_attention_width(width)
        self._graphs.pop(width, None)
        self._buffers.pop(width, None)
        self._batches.pop(width, None)
        self._events.pop(width, None)
        self._memory_bytes.pop(width, None)
        self._fla_index_pins.pop(width, None)
        if not self._graphs:
            # the shared pool belonged to a graph that is going away with this attempt
            self._pool = None
        gc.collect()

    def _reset_capture_context(self) -> bool:
        """Rebuild the private capture stream and probe the context after a failed capture.

        A ``capture_end`` that raises skips torch's stream restore, leaving the thread on the
        capture stream with the allocator still pointed at the dead graph pool; the probe reports
        whether the context survived at all.
        """
        self._pool = None
        try:
            self._stream = torch.cuda.Stream(device=self.device)
            torch.zeros(1, device=self.device)
            torch.cuda.synchronize(self.device)
        except Exception:
            return False
        return True

    def _disable_all_widths(self, reason: str) -> None:
        self._graphs_disabled = True
        for other in self.widths:
            self._record_attempt(
                other, status="permanently-unsupported", reason=reason
            )

    def capture(self, batch) -> MTPGraphCaptureResult:
        width = self._validate_width(batch)
        previous = self._support.get(width)
        if previous is not None:
            return previous
        if self._graphs_disabled:
            return self._record_attempt(
                width,
                status="permanently-unsupported",
                reason="CAPTURE_CONTEXT_LOST",
            )
        if self.device.type != "cuda":
            return self._record_attempt(
                width,
                status="permanently-unsupported",
                reason="CUDA_REQUIRED",
            )
        free_before = int(torch.cuda.mem_get_info(self.device)[0])
        owned_before = self.owned_buffer_bytes
        estimated_buffer_bytes = (
            width * self.vocab_size * torch.float32.itemsize
            + width * (3 * torch.int32.itemsize + 3 * torch.int64.itemsize)
            + 64
        )
        if free_before < self.guard_bytes + estimated_buffer_bytes:
            return self._record_attempt(
                width,
                status="retryable",
                reason="MEMORY_ADMISSION",
            )

        graph = torch.cuda.CUDAGraph()
        buffer = _MTPVerifyGraphBuffer.init(width, self.vocab_size, self.device)
        synchronizations = 0
        entered_capture = False
        body_error: Exception | None = None
        # a raising capture_end skips torch's own stream restore, so own the restore here
        entry_stream = torch.cuda.current_stream(self.device)
        try:
            buffer.copy_from(batch)
            buffer.bind(batch)
            # the fla chunk-index cache is a four-entry identity LRU and three widths need six
            # entries: prime it so capture takes no miss, and hold the tensors so a later
            # prefill cannot evict and free what this graph is about to bake an address for
            # (imported lazily: fla.utils initializes CUDA at import, before Engine allows it)
            from freetoken.kernel.fla.index import prime_chunk_index_cache

            self._fla_index_pins[width] = prime_chunk_index_cache(
                buffer.fla_cu_seqlens, width
            )
            prepare_attention = getattr(
                self.attn_backend, "prepare_mtp_verify_graph", None
            )
            if prepare_attention is not None:
                prepare_attention(batch)
            prepare_model = getattr(
                self.target_model, "prepare_cuda_graph_capture", None
            )
            with self.target_ctx.forward_batch(batch):
                if prepare_model is not None:
                    prepare_model(batch)
                # warm up on the stream that gets captured, so per-stream lazy resources
                # (cuBLAS workspaces, kernel modules) are already materialized there
                self._stream.wait_stream(entry_stream)
                with torch.cuda.stream(self._stream):
                    buffer.logits.copy_(self._forward(batch))
                entry_stream.wait_stream(self._stream)
                torch.cuda.synchronize(self.device)
                synchronizations += 1
                free_after_warm = int(torch.cuda.mem_get_info(self.device)[0])
                warm_reserve = max(
                    _GRAPH_CAPTURE_RESERVE_FLOOR,
                    free_before - free_after_warm,
                )
                if free_after_warm < self.guard_bytes + warm_reserve:
                    self._cleanup_failed_attempt(width)
                    return self._record_attempt(
                        width,
                        status="retryable",
                        reason="MEMORY_ADMISSION_AFTER_WARM",
                        synchronizations=synchronizations,
                    )
                entered_capture = True
                with torch.cuda.graph(
                    graph,
                    pool=self._pool,
                    stream=self._stream,
                    # background threads (ple-mmap staging, the CPU-MoE watchdog) must not
                    # invalidate this capture from outside
                    capture_error_mode="thread_local",
                ):
                    try:
                        buffer.logits.copy_(self._forward(batch))
                    except Exception as exc:
                        # capture_end raises next and would otherwise mask the real cause
                        body_error = exc
                        raise
            finish_attention = getattr(
                self.attn_backend, "finish_mtp_verify_graph_capture", None
            )
            if finish_attention is not None:
                finish_attention(width)
            captured_pool = graph.pool() if self._pool is None else self._pool
            torch.cuda.synchronize(self.device)
            synchronizations += 1
            free_after = int(torch.cuda.mem_get_info(self.device)[0])
            if free_after < self.guard_bytes:
                self._cleanup_failed_attempt(width)
                return self._record_attempt(
                    width,
                    status="retryable",
                    reason="MEMORY_GUARD_AFTER_CAPTURE",
                    synchronizations=synchronizations,
                )
        except Exception as exc:
            torch.cuda.set_stream(entry_stream)
            failure = body_error if body_error is not None else exc
            graph = None
            self._cleanup_failed_attempt(width)
            detail = " ".join(str(failure).split())[:300]
            reason = f"CAPTURE_FAILED:{type(failure).__name__}:{detail}"
            try:
                _CAPTURE_FAILURE_EVIDENCE_DIR.joinpath(
                    f"capture-failure-tb-width{width}.txt"
                ).write_text(
                    "".join(
                        traceback.format_exception(
                            type(failure), failure, failure.__traceback__
                        )
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
            if entered_capture:
                # retrying would re-execute the same capture-illegal op and re-poison the context
                if not self._reset_capture_context():
                    self._disable_all_widths("CAPTURE_CONTEXT_LOST")
                return self._record_attempt(
                    width,
                    status="permanently-unsupported",
                    reason=reason,
                    synchronizations=synchronizations,
                )
            status = (
                "permanently-unsupported"
                if isinstance(failure, NotImplementedError)
                else "retryable"
            )
            return self._record_attempt(
                width,
                status=status,
                reason=reason,
                synchronizations=synchronizations,
            )
        finally:
            torch.cuda.set_stream(entry_stream)

        try:
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self._graphs[width] = graph
            self._buffers[width] = buffer
            self._batches[width] = batch
            self._events[width] = events
            memory_bytes = max(
                self.owned_buffer_bytes - owned_before,
                free_before - free_after,
            )
            self._memory_bytes[width] = memory_bytes
            if self._pool is None:
                self._pool = captured_pool
        except Exception as exc:
            self._cleanup_failed_attempt(width)
            return self._record_attempt(
                width,
                status="retryable",
                reason=f"FINALIZE_FAILED:{type(exc).__name__}",
                synchronizations=synchronizations,
            )
        return self._record_attempt(
            width,
            status="captured",
            reason="",
            memory_bytes=memory_bytes,
            synchronizations=synchronizations,
        )

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
        static_batch = self._batches[width]
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
        buffer.copy_from(batch)
        stage_attention = getattr(
            self.attn_backend, "stage_mtp_verify_graph", None
        )
        if stage_attention is not None:
            stage_attention(batch, static_batch)
        prepare_model = getattr(self.target_model, "prepare_cuda_graph_replay", None)
        if prepare_model is not None:
            prepare_model(batch)
        graph.replay()
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

    def destroy(self) -> None:
        self._graphs = {}
        self._buffers = {}
        self._batches = {}
        self._events = {}
        self._memory_bytes = {}
        self._fla_index_pins = {}
        self._support = {}
        self._last_attempt = {}
        self._pool = None
        self._stream = None
        self._destroyed = True
        reset_attention = getattr(self.attn_backend, "reset_mtp_verify_graph", None)
        if reset_attention is not None:
            reset_attention()
        gc.collect()


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
