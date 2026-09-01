from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List, Mapping

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

# w = 1 + depth must stay a capturable MTP verify width (mtp_fast_verify captures 2, 3, 4).
_MAX_SPEC_DEPTH = 3

# A cold request probes at least once every ``cooldown_cap`` plain steps. 4x rather than 8x
# because a re-cool now takes TWO consecutive failed probes, so each rung of the ladder is
# already twice the evidence it used to be.
_COOLDOWN_BACKOFF_CAP = 4


@dataclass(frozen=True)
class SpecDecodeConfig:
    """Resolved integrated speculative decode settings.

    ``num_speculative_tokens`` is the single number every sizing surface must agree on: the
    QSA pending ring is ``position % ring_capacity`` with no epoch tag, so a ring narrower
    than ``index_ratio + depth`` aliases a speculative row onto an open compression group's
    still-needed members -- silently wrong keys, no crash.
    """

    enabled: bool = False
    depth: int = _MAX_SPEC_DEPTH
    graph: bool = False
    # --- the adaptive fallback (see ``adaptive``) ---
    ema_alpha: float = 0.3
    min_emitted: float = 3.2
    # A probe is ONE integer sample from the distribution ``min_emitted`` bounds the mean of,
    # so it is judged against its own -- lower -- bar. Holding a single sample to the mean's
    # threshold rejects roughly half of content that is comfortably worth speculating on.
    probe_resume: float = 2.0
    cooldown: int = 16

    @property
    def adaptive(self) -> bool:
        """Whether the acceptance fallback is armed.

        A speculative cycle costs about ``min_emitted`` plain steps whatever the draft
        produces, so below that it is a net loss. ``min_emitted`` 0 means never fall back --
        the pre-adaptive always-speculate behaviour, exactly.
        """
        return self.enabled and self.min_emitted > 0.0

    @property
    def ema_seed(self) -> float:
        """A fresh request starts optimistic -- one full cycle's emission -- so early noise
        cannot lock speculation out before the request has evidence of its own."""
        return float(self.batch_width)

    @property
    def cooldown_cap(self) -> int:
        """The longest a cold request may go between probes. Content changes mid-stream; a
        request that stopped probing could never discover its draft went hot again."""
        return self.cooldown * _COOLDOWN_BACKOFF_CAP

    @property
    def graph_widths(self) -> tuple[int, ...]:
        """The widths worth capturing: ``1 .. batch_width``, or nothing when off.

        A step drafts ``min(depth, remain_len)`` tokens, so near a request's budget end it can
        narrow below the full width -- but never below 2, and never above ``1 + depth``. Each
        graph costs a warm-up forward and its own buffers, so capturing above that range would
        spend both on a width the configuration can never emit.

        Width 1 is not a verify width at all: it is the ORDINARY decode forward, recorded with
        the same fixed output slots. A spec-enabled boot cannot use the plain decode graph
        (logits only, and the draft head needs hidden + embeddings), so without this every
        fallback decode step ran eager at roughly half the graphed rate.
        """
        if not (self.enabled and self.graph):
            return ()
        return tuple(range(1, self.batch_width + 1))

    @property
    def num_speculative_tokens(self) -> int:
        """Extra rows a step may add beyond the confirmed one; 0 while speculation is off, so
        every pool and budget keeps its non-speculative geometry exactly."""
        return self.depth if self.enabled else 0

    @property
    def batch_width(self) -> int:
        """``w`` = the confirmed row plus the drafts."""
        return 1 + self.num_speculative_tokens


def resolve_spec_decode(env: Mapping[str, str] | None = None) -> SpecDecodeConfig:
    """Read ``FREETOKEN_MTP_SPECULATE`` / ``FREETOKEN_MTP_SPEC_DEPTH`` / ``..._SPEC_GRAPH``,
    plus the adaptive fallback's ``..._SPEC_EMA_ALPHA`` / ``..._SPEC_MIN_EMITTED`` /
    ``..._SPEC_PROBE_RESUME`` / ``..._SPEC_COOLDOWN``."""
    env = os.environ if env is None else env
    raw = env.get("FREETOKEN_MTP_SPECULATE", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError("FREETOKEN_MTP_SPECULATE must be 0 or 1")
    enabled = raw == "1"
    graph_raw = env.get("FREETOKEN_MTP_SPEC_GRAPH", "0").strip()
    if graph_raw not in ("0", "1"):
        raise ValueError("FREETOKEN_MTP_SPEC_GRAPH must be 0 or 1")
    depth_raw = env.get("FREETOKEN_MTP_SPEC_DEPTH", str(_MAX_SPEC_DEPTH)).strip()
    try:
        depth = int(depth_raw)
    except ValueError:
        depth = 0
    if not 1 <= depth <= _MAX_SPEC_DEPTH:
        raise ValueError(
            f"FREETOKEN_MTP_SPEC_DEPTH must be 1..{_MAX_SPEC_DEPTH}, got {depth_raw!r}"
        )
    if enabled and env.get("FREETOKEN_MTP_SHADOW", "0").strip() == "1":
        # The observer takes the eager capture path in Engine.forward_batch and would double
        # every verify forward.
        raise ValueError(
            "FREETOKEN_MTP_SPECULATE=1 is incompatible with FREETOKEN_MTP_SHADOW=1"
        )
    alpha = _float_env(env, "FREETOKEN_MTP_SPEC_EMA_ALPHA", "0.3")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(
            f"FREETOKEN_MTP_SPEC_EMA_ALPHA must be in (0, 1], got {alpha!r}"
        )
    # 3.6 is the measured end-to-end breakeven: a graphed depth-3 cycle costs ~3.7 plain
    # steps (verify replay is PCIe expert-fetch bound at w rows), and same-boot profiles
    # show prose LOSES at a 3.2 bar and recovers monotonically toward plain parity as the
    # bar rises to it. At shallow depths the ceiling below binds first, so the unset
    # default clamps to it.
    min_emitted = _float_env(
        env, "FREETOKEN_MTP_SPEC_MIN_EMITTED", str(min(3.6, float(1 + depth)))
    )
    if not 0.0 <= min_emitted <= 1 + depth:
        # Above the full width no cycle could ever clear the bar, so speculation would go
        # cold and never return -- a configuration that silently means "off".
        raise ValueError(
            f"FREETOKEN_MTP_SPEC_MIN_EMITTED must be 0..{1 + depth}, got {min_emitted!r}"
        )
    probe_resume = _float_env(env, "FREETOKEN_MTP_SPEC_PROBE_RESUME", "2.0")
    if not 0.0 < probe_resume <= 1 + depth:
        # Zero would resume on any emission at all, which is not a probe; above the full width
        # no probe could ever clear the bar and a cold request would never come back.
        raise ValueError(
            f"FREETOKEN_MTP_SPEC_PROBE_RESUME must be in (0, {1 + depth}], got "
            f"{probe_resume!r}"
        )
    cooldown_raw = env.get("FREETOKEN_MTP_SPEC_COOLDOWN", "16").strip()
    try:
        cooldown = int(cooldown_raw)
    except ValueError:
        cooldown = 0
    if cooldown < 1:
        raise ValueError(
            f"FREETOKEN_MTP_SPEC_COOLDOWN must be >= 1, got {cooldown_raw!r}"
        )
    return SpecDecodeConfig(
        enabled=enabled,
        depth=depth,
        graph=graph_raw == "1",
        ema_alpha=alpha,
        min_emitted=min_emitted,
        probe_resume=probe_resume,
        cooldown=cooldown,
    )


def _float_env(env: Mapping[str, str], name: str, default: str) -> float:
    """A finite float, or a ValueError naming the variable. NaN fails every comparison the
    caller then makes, so it has to be rejected here rather than silently pass a range test."""
    raw = env.get(name, default).strip()
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    return value


def require_speculation_supported(config) -> None:
    """Fail at boot if this configuration cannot support integrated speculation.

    The draft head keeps ONE running context, the state ladder ONE snapshot slot, and the
    sampler one acceptance stream per step. A batch that happens to carry two requests must
    not be where that is discovered.
    """
    if not config.spec_decode.enabled:
        return
    if config.max_running_req != 1:
        raise ValueError(
            "FREETOKEN_MTP_SPECULATE=1 requires --max-running-requests 1, got "
            f"{config.max_running_req}"
        )


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    nvfp4_backend: str = "triton"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    ple_backend: str = "pinned"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None

    @cached_property
    def spec_decode(self) -> SpecDecodeConfig:
        """Resolved once per config instance. The Engine and the Scheduler hold the SAME
        instance (SchedulerConfig extends EngineConfig), and it is the sole argument to both
        ``create_kv_pool`` and every pool family's ``kv_cost`` -- so the pool geometry and the
        boot budget cannot disagree about the speculative width."""
        return resolve_spec_decode()

    @property
    def num_speculative_tokens(self) -> int:
        return self.spec_decode.num_speculative_tokens

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        return parse_config(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
