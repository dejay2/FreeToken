"""Speculative sampling: the acceptance core, and the integrated sampler built on it.

Design section 5.2. Deliberately dependency-light -- torch and nothing else at module scope, so
the decode path can import it without pulling in the shadow observer, the MTP weight store or
the flashlib slot cache, and without touching CUDA. ``mtp_fast_verify`` re-exports the
acceptance core it used to own, so the observer's import surface is unchanged.

WHAT A STEP RETURNS, AND WHY THE TWO NUMBERS DIFFER
---------------------------------------------------
A speculative step forwards ``w = 1 + k`` rows. Row ``i`` consumes the token at position
``cached_len + i`` and holds the target's distribution for position ``cached_len + i + 1``.
With ``j`` of the ``k`` drafts accepted the step emits::

    drafts[:j]  +  one token sampled from row j        ->  j + 1 tokens

and ``Scheduler._rollback_spec_tokens`` must be given ``j + 1`` ROWS, not ``j``: it settles
``cached_len += accepted``, and the bonus token from row ``j`` occupies a position of its own.
``SpecDecision.accepted_rows == len(SpecDecision.tokens)`` always; ``accepted_drafts`` is ``j``.
Even a wholly rejected step keeps one row, which is why ``accepted == 0`` is reachable only as
a deliberate full undo (the phase-2 state-digest gate) and never in production.

THE DISTRIBUTION CONTRACT
-------------------------
Acceptance is only correct against the distribution the server would itself have sampled from.
That distribution is defined by ``freetoken.engine.sample.sample_impl``: ``softmax(logits / T)``
then top-k then top-p ON THE TOP-K-RENORMALIZED PROBS, through ``flashinfer.sampling`` when
installed and ``freetoken.kernel.triton.sampling`` otherwise. ``filtered_probs`` reproduces it
from the same ``BatchSamplingArgs`` the batch carries, and
``tests/engine/test_spec_filter_backend.py`` pins the equality empirically against whichever
backend is installed. Phase 5 owes this module one further precondition the theorem needs: the
draft head must PROPOSE from ``filtered_probs(draft_logits, args)``, the same ``q`` acceptance
divides by.

RNG
---
The server has no reproducibility contract to preserve: ``SamplingParams`` carries no seed, the
triton sampler draws from a module-global generator that falls back to the default generator
under graph capture, and the default is seeded once at engine start. So the speculative path
owns its own stream instead -- one generator per (request, depth), seeded from
``FREETOKEN_MTP_DRAFT_SEED``, the shape the shadow observer already uses. Identical requests
therefore reproduce with speculation on, while the number of draws per emitted token differs
from plain decode (inherent: a cycle emits 1..k+1 tokens for a fixed 2k+1 draws). Nothing here
advances the default stream; ``guard_default_rng=True`` asserts it.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

if TYPE_CHECKING:
    from freetoken.engine.sample import BatchSamplingArgs

# Shared with the shadow observer's draft/acceptance streams (mtp_shadow.MTPShadowConfig.seed).
DEFAULT_SPEC_SEED = 1729
# purpose tags keep the acceptance stream disjoint from the draft head's, per request
_ACCEPTANCE_PURPOSE = 2


def resolve_spec_seed(env: Mapping[str, str] | None = None) -> int:
    """Base seed for the speculative streams (``FREETOKEN_MTP_DRAFT_SEED``)."""
    env = os.environ if env is None else env
    raw = env.get("FREETOKEN_MTP_DRAFT_SEED", str(DEFAULT_SPEC_SEED)).strip()
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"FREETOKEN_MTP_DRAFT_SEED must be an integer, got {raw!r}"
        ) from None


# --------------------------------------------------------------------------- the filter


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


def spec_filter_params(
    args: "BatchSamplingArgs", *, row: int = 0
) -> tuple[float, int, float]:
    """The ``(temperature, top_k, top_p)`` triple that ``BatchSamplingArgs`` row ``row`` means.

    ``Sampler.prepare`` (``engine/sample.py:58-73``) encodes "no filter" as an absent tensor:
    ``temperatures is None`` is the whole-batch greedy fast path, and a missing ``top_k`` /
    ``top_p`` means every request asked for the full vocabulary. Decoding that back is the only
    place the two samplers could drift apart on a request that never reaches the acceptance
    maths at all.
    """
    if args.temperatures is None:
        return 0.0, -1, 1.0
    temperature = float(args.temperatures[row])
    top_k = -1 if args.top_k is None else int(args.top_k[row])
    top_p = 1.0 if args.top_p is None else float(args.top_p[row])
    return temperature, top_k, top_p


def request_filter_params(params) -> tuple[float, int, float]:
    """The same triple, read off a request's ``SamplingParams`` instead of a prepared batch.

    The draft head has to choose its filter BEFORE the speculative batch (and so the batch's
    ``BatchSamplingArgs``) exists, and reading the raw params would silently disagree with
    acceptance: ``Sampler.prepare`` floors a non-greedy request's temperature at 1e-6, so a
    ``temperature=0, top_p<1`` request is SAMPLED by the server while its raw temperature says
    argmax. Speculation serves one request per step, which is exactly when ``prepare``'s
    whole-batch greedy fast path and its per-request path agree, so this reproduction is
    total; ``tests/engine/test_spec_draft.py`` pins it against the real ``Sampler``.
    """
    if params.is_greedy:
        return 0.0, -1, 1.0
    return (
        max(params.temperature, 1e-6),
        params.top_k if params.top_k >= 1 else -1,
        min(max(params.top_p, 1e-6), 1.0),
    )


def filtered_probs(
    logits: torch.Tensor, args: "BatchSamplingArgs", *, row: int = 0
) -> torch.Tensor:
    """The distribution the server would sample each row of ``logits`` from."""
    temperature, top_k, top_p = spec_filter_params(args, row=row)
    return _sampling_probabilities_batch(
        logits, temperature=temperature, top_k=top_k, top_p=top_p
    )


# --------------------------------------------------------------------------- acceptance


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
    # A three-way split of ``required_wall_ms``, for attributing a slow verify stage. Host-side
    # wall time of asynchronous launches: only ``sync_wall_ms`` is device work, the other two are
    # launch overhead. Timing only, so they stay out of the dataclass's equality like the rest.
    filter_wall_ms: float = field(default=0.0, compare=False)
    decide_wall_ms: float = field(default=0.0, compare=False)
    sync_wall_ms: float = field(default=0.0, compare=False)

    @property
    def acceptance_ms(self) -> float:
        return self.required_wall_ms

    @property
    def synchronizations(self) -> int:
        return self.required_synchronizations + self.instrumentation_synchronizations


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
    filter_done = time.perf_counter()

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

    decide_done = time.perf_counter()

    required_synchronizations = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        required_synchronizations = 1
    sync_done = time.perf_counter()
    required_wall_ms = (sync_done - started) * 1000.0
    filter_wall_ms = (filter_done - started) * 1000.0
    decide_wall_ms = (decide_done - filter_done) * 1000.0
    sync_wall_ms = (sync_done - decide_done) * 1000.0

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
        filter_wall_ms=filter_wall_ms,
        decide_wall_ms=decide_wall_ms,
        sync_wall_ms=sync_wall_ms,
    )
    return result


# ------------------------------------------------------------------------------- the RNG


@contextmanager
def default_rng_guard(device: torch.device | str | None = None):
    """Fail if the block advances the default CPU/CUDA RNG stream, restoring it first.

    Speculation must be invisible to the RNG plain decode draws from, so that "spec on" and
    "spec off" stay comparable and the flag-off boot stays byte-identical.
    """
    device = torch.device("cpu") if device is None else torch.device(device)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )
    yield
    unchanged = torch.equal(cpu_before, torch.get_rng_state())
    if unchanged and cuda_before is not None:
        unchanged = torch.equal(cuda_before, torch.cuda.get_rng_state(device))
    if not unchanged:
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state(cuda_before, device)
        raise RuntimeError("speculative sampling changed the default RNG stream")


# ---------------------------------------------------------------------------- the sampler


@dataclass(frozen=True)
class SpecDecision:
    """One speculative step's verdict, in the two shapes phase 5 consumes.

    ``tokens`` goes to the emission path (host append, one ``DetokenizeMsg``), ``accepted_rows``
    to ``Scheduler._rollback_spec_tokens``. They are the same number by construction; a stop
    condition may shorten the run, and then BOTH shrink together.
    """

    tokens: tuple[int, ...]
    accepted_rows: int
    accepted_drafts: int
    rejected_at: int | None
    greedy: bool
    acceptance_probabilities: tuple[float, ...]
    draft_probabilities: tuple[float, ...]
    target_probabilities: tuple[float, ...]
    acceptance_ms: float = field(default=0.0, compare=False)
    # ``acceptance_ms`` split three ways (MTPAcceptanceResult.filter/decide/sync_wall_ms).
    filter_ms: float = field(default=0.0, compare=False)
    decide_ms: float = field(default=0.0, compare=False)
    sync_ms: float = field(default=0.0, compare=False)

    @property
    def bonus_token(self) -> int:
        """The token sampled from the first unaccepted row -- always the last one emitted."""
        return self.tokens[-1]

    def truncated(self, keep: int) -> "SpecDecision":
        """The same verdict with the run cut to ``keep`` tokens (design 6.4: EOS, stop strings,
        the output budget). ``accepted_rows`` follows the run, which is what keeps the KV and
        the emitted text in step."""
        if not 1 <= keep <= len(self.tokens):
            raise ValueError(
                f"a truncated run keeps 1..{len(self.tokens)} tokens, got {keep}"
            )
        if keep == len(self.tokens):
            return self
        return SpecDecision(
            tokens=self.tokens[:keep],
            accepted_rows=keep,
            accepted_drafts=min(self.accepted_drafts, keep),
            rejected_at=self.rejected_at,
            greedy=self.greedy,
            acceptance_probabilities=self.acceptance_probabilities,
            draft_probabilities=self.draft_probabilities,
            target_probabilities=self.target_probabilities,
            acceptance_ms=self.acceptance_ms,
            filter_ms=self.filter_ms,
            decide_ms=self.decide_ms,
            sync_ms=self.sync_ms,
        )


class SpecSampler:
    """Draft + target logits in, the emitted run and its row count out.

    One request per step (the integrated path serves ``max_running_req == 1``); tensor shapes
    stay row-generic so a wider batch would only need the per-request loop.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        depth: int,
        seed: int = DEFAULT_SPEC_SEED,
        guard_default_rng: bool = False,
    ) -> None:
        if depth < 1:
            raise ValueError(f"speculative depth must be at least 1, got {depth}")
        self.device = torch.device(device)
        self.depth = int(depth)
        self.seed = int(seed)
        self.guard_default_rng = bool(guard_default_rng)
        self._uid: int | None = None
        self._generators: dict[int, torch.Generator] = {}
        self._steps = 0
        self._drafts_proposed = 0
        self._drafts_accepted = 0
        self._tokens_emitted = 0
        self._histogram = [0] * (self.depth + 1)
        self.reset_request(0)

    @classmethod
    def from_config(cls, config, device: torch.device | str, **kwargs) -> "SpecSampler":
        """Build from an ``EngineConfig.spec_decode``; the seed comes from the environment,
        which keeps it out of the config's equality (phase-2 tests compare those objects)."""
        return cls(
            device=device,
            depth=config.depth,
            seed=resolve_spec_seed(),
            **kwargs,
        )

    # ------------------------------------------------------------------------------ RNG

    def _seed_for(self, uid: int, depth: int) -> int:
        modulus = (1 << 63) - 1
        return int(
            (
                self.seed
                + int(uid) * 10_000_019
                + _ACCEPTANCE_PURPOSE * 1_000_003
                + depth * 10_009
            )
            % modulus
        )

    def reset_request(self, uid: int) -> None:
        """Re-seed every depth's stream for ``uid``. ``step`` calls this on its own whenever the
        request changes, so a repeated request replays exactly."""
        uid = int(uid)
        self._generators = {}
        for depth in range(1, self.depth + 1):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self._seed_for(uid, depth))
            self._generators[depth] = generator
        self._uid = uid

    def generator_state(self, *, depth: int) -> torch.Tensor:
        return self._generators[depth].get_state().clone()

    # ----------------------------------------------------------------------------- stats

    @property
    def stats(self) -> dict:
        return {
            "steps": self._steps,
            "drafts_proposed": self._drafts_proposed,
            "drafts_accepted": self._drafts_accepted,
            "tokens_emitted": self._tokens_emitted,
            "acceptance_histogram": tuple(self._histogram),
        }

    # ------------------------------------------------------------------------------ step

    def step(
        self,
        *,
        uid: int,
        draft_tokens: Sequence[int],
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        args: "BatchSamplingArgs",
    ) -> SpecDecision:
        """Accept a prefix of ``draft_tokens`` against the step's ``w = 1 + k`` target rows.

        ``target_logits[i]`` is the target's distribution for the token AFTER row ``i``'s input,
        so ``target_logits`` has one row more than ``draft_tokens``: the extra row supplies the
        bonus token when every draft is accepted. ``draft_logits`` carries the draft head's full
        rows, not just the proposed tokens' probabilities -- rejection sampling corrects from
        the residual ``(p - q)+``, which needs all of ``q``.
        """
        k = len(draft_tokens)
        if not 1 <= k <= self.depth:
            raise ValueError(
                f"a speculative step needs 1..{self.depth} drafts (the sampler's depth), got {k}"
            )
        if target_logits.ndim != 2 or target_logits.shape[0] != k + 1:
            raise ValueError(
                f"a {k}-draft step needs 1 + k = {k + 1} target rows, got "
                f"{tuple(target_logits.shape)}"
            )
        for name in ("temperatures", "top_k", "top_p"):
            tensor = getattr(args, name)
            assert tensor is None or tensor.numel() == 1, (
                f"the speculative path serves one request per step; {name} has "
                f"{tensor.numel()} rows"
            )

        if int(uid) != self._uid:
            self.reset_request(uid)
        temperature, top_k, top_p = spec_filter_params(args)

        if self.guard_default_rng:
            with default_rng_guard(self.device):
                acceptance = self._accept(
                    draft_tokens, draft_logits, target_logits, temperature, top_k, top_p, k
                )
        else:
            acceptance = self._accept(
                draft_tokens, draft_logits, target_logits, temperature, top_k, top_p, k
            )

        accepted = acceptance.accepted_prefix
        tokens = tuple(int(t) for t in draft_tokens[:accepted]) + (
            acceptance.corrected_token,
        )
        self._steps += 1
        self._drafts_proposed += k
        self._drafts_accepted += accepted
        self._tokens_emitted += len(tokens)
        self._histogram[accepted] += 1
        return SpecDecision(
            tokens=tokens,
            accepted_rows=len(tokens),
            accepted_drafts=accepted,
            rejected_at=None if accepted == k else accepted,
            greedy=acceptance.greedy,
            acceptance_probabilities=acceptance.acceptance_probabilities,
            draft_probabilities=acceptance.draft_probabilities,
            target_probabilities=acceptance.target_probabilities,
            acceptance_ms=acceptance.acceptance_ms,
            filter_ms=acceptance.filter_wall_ms,
            decide_ms=acceptance.decide_wall_ms,
            sync_ms=acceptance.sync_wall_ms,
        )

    def _accept(
        self, draft_tokens, draft_logits, target_logits, temperature, top_k, top_p, k
    ) -> MTPAcceptanceResult:
        return batched_speculative_accept(
            proposals=[int(t) for t in draft_tokens],
            draft_logits=draft_logits,
            target_logits=target_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=self._generators[k],
        )


__all__ = [
    "DEFAULT_SPEC_SEED",
    "MTPAcceptanceResult",
    "SpecDecision",
    "SpecSampler",
    "batched_speculative_accept",
    "default_rng_guard",
    "filtered_probs",
    "request_filter_params",
    "resolve_spec_seed",
    "spec_filter_params",
]
