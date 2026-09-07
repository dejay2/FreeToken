from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List, Mapping

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.moe.exl3_ops import DEFAULT_EXL3_EXPERT_OP
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

# The deepest chain an operator may ask for. ``w = 1 + depth`` is the verify width, and every
# width 2..6 is capturable by the width-generic ``SpecVerifyGraphRunner``: ``graph_widths``
# below derives 1..batch_width and ``engine/spec_graph.py`` builds one fixed-width graph per
# entry, with no per-width table anywhere. (The legacy ``mtp_fast_verify`` observer keeps its
# own 2..4 cap; it lives behind FREETOKEN_MTP_SHADOW, which ``resolve_spec_decode`` refuses to
# run alongside speculation, so the two ceilings never meet.)
#
# 5 rather than 3 is measured. With the confidence cut armed (``conf_cut``, default 0.8) only
# a confident chain ever reaches the deep rows: 40% of live cycles have every draft at >= 0.9
# confidence and 93% of those accept in full. An extra verified row costs ~6.5 ms (~5 unique
# experts per layer marginal, measured window-union cost) and pays for itself once the extra
# draft's acceptance clears ~0.63 -- against 0.73-0.77 measured in exactly those confident
# cycles. So the CEILING is 5; the DEFAULT stays 3 until a live sweep moves it.
_MAX_SPEC_DEPTH = 5

# What an unset ``FREETOKEN_MTP_SPEC_DEPTH`` means. 5 since the live paired sweep with the
# confidence cut AND the cost-aware bar (both required): depth 5 flat-bar regressed 8k context
# -11..-24%, but with the measured bar it beat depth 3 on numbers +7, code +4, and 8k +3
# tok/s while the bar priced dear cycles out. Without cost_aware, prefer depth 3.
_DEFAULT_SPEC_DEPTH = 5

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
    depth: int = _DEFAULT_SPEC_DEPTH
    graph: bool = False
    # --- the confidence cut (see ``SpecDraftHead.propose``) ---
    # Stop the draft chain before the first token whose RAW softmax top-1 probability falls
    # under this bar. Measured over 600 live cycles: accepted drafts average 0.91 confidence,
    # rejected 0.46; the <0.5 bucket is accepted 20% of the time, the >=0.95 bucket 81%. Cutting
    # at 0.8 takes the mean draft from 3.0 to 1.5 and acceptance-per-draft from 0.58 to 0.91,
    # which models out at 1.18x on prose because verify width is what a cycle actually costs
    # here (~6.5 ms/row of expert fetch). 0 disables the cut and restores always-full-depth.
    conf_cut: float = 0.8
    # WHEN the cut is applied, which is a pure cost question -- the proposal is the same
    # either way (``tests/engine/test_spec_draft.py`` pins the two modes token for token).
    #   "chain" (default): draft the full depth, read every row's confidence back in ONE
    #     device synchronization at the end, truncate to the prefix before the first doubtful
    #     row. The chain issues as one uninterrupted stream of launches -- and it is what
    #     makes the draft steps graph-capturable at all -- at the price of computing the rows
    #     after the cut and throwing them away.
    #   "step": the pre-existing early exit -- one small readback per drafted token (never the
    #     last), break at the first doubtful row. Saves those draft forwards; pays a queue
    #     drain per token, so the host cannot run ahead of the device inside a chain.
    # Which wins is measurable and not obvious: a cut chain averaged 1.5 drafts of a possible
    # 5, so "chain" spends ~3.5 extra draft forwards (~1.6-4.2 ms each here) to recover ~5
    # launch-ramp bubbles. This flag exists so that A/B can be run paired on one boot.
    draft_cut_mode: str = "chain"
    # --- the adaptive fallback (see ``adaptive``) ---
    ema_alpha: float = 0.3
    # With ``cost_aware`` on (the default) this is the FLOOR of a bar the request MEASURES,
    # not the bar itself: the policy divides its own speculative cycle's wall time by its own
    # plain decode step's and holds the emitted-EMA to that ratio, never dropping below this
    # value nor rising above the ``1 + depth`` a cycle could emit. With ``cost_aware`` off it
    # is the bar, flat, exactly as it was before the ratio existed.
    min_emitted: float = 3.2
    # Whether the bar tracks measured wall time (see ``_SpecAcceptance.bar``). A static bar
    # cannot be right at both ends of a context: measured live, a spec cycle costs ~28-45 ms
    # at short context and ~70-100 ms at 8-11k while a plain step goes ~15 -> ~20 ms, so the
    # true breakeven roughly DOUBLES over a long request. A bar fixed at the short-context
    # breakeven overspends at the long end -- measured -11..-24% at 8k on depth 5, ~-10% at
    # 11k on depth 3, against +11-13% at short context.
    cost_aware: bool = True
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

        What that cycle costs depends on ``conf_cut``: an always-full-width cycle verifies
        ``1 + depth`` rows, a cut one verifies far fewer, so the two arm different bars. See
        the unset default in ``resolve_spec_decode``.
        """
        return self.enabled and self.min_emitted > 0.0

    @property
    def ema_seed(self) -> float:
        """A fresh request starts optimistic -- one full cycle's emission -- so early noise
        cannot lock speculation out before the request has evidence of its own.

        Left at the full ``1 + depth`` even with ``conf_cut`` armed, where a cut cycle usually
        emits less than that: the seed is a head start, not a prediction, and the first few
        real cycles overwrite it at ``ema_alpha`` anyway."""
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
    the drafting cut's ``..._SPEC_CONF_CUT`` / ``..._SPEC_DRAFT_CUT_MODE``, plus the adaptive
    fallback's
    ``..._SPEC_EMA_ALPHA`` / ``..._SPEC_MIN_EMITTED`` / ``..._SPEC_PROBE_RESUME`` /
    ``..._SPEC_COOLDOWN`` / ``..._SPEC_COST_AWARE``."""
    env = os.environ if env is None else env
    raw = env.get("FREETOKEN_MTP_SPECULATE", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError("FREETOKEN_MTP_SPECULATE must be 0 or 1")
    enabled = raw == "1"
    graph_raw = env.get("FREETOKEN_MTP_SPEC_GRAPH", "0").strip()
    if graph_raw not in ("0", "1"):
        raise ValueError("FREETOKEN_MTP_SPEC_GRAPH must be 0 or 1")
    depth_raw = env.get("FREETOKEN_MTP_SPEC_DEPTH", str(_DEFAULT_SPEC_DEPTH)).strip()
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
    conf_cut = _float_env(env, "FREETOKEN_MTP_SPEC_CONF_CUT", "0.8")
    cut_mode = (env.get("FREETOKEN_MTP_SPEC_DRAFT_CUT_MODE", "") or "chain").strip().lower()
    if cut_mode not in ("chain", "step"):
        raise ValueError(
            "FREETOKEN_MTP_SPEC_DRAFT_CUT_MODE must be chain or step, got "
            f"{env.get('FREETOKEN_MTP_SPEC_DRAFT_CUT_MODE')!r}"
        )
    if not 0.0 <= conf_cut <= 1.0:
        # It is a softmax probability, so anything outside 0..1 is either a typo or a bar no
        # row could ever clear -- which would silently mean "always draft exactly 1 token".
        raise ValueError(
            f"FREETOKEN_MTP_SPEC_CONF_CUT must be 0..1, got {conf_cut!r}"
        )
    # The bar the adaptive fallback holds a cycle to is a function of what a cycle COSTS, and
    # the cut changes that:
    #   cut off (conf_cut == 0): every cycle verifies the full 1 + depth rows. 3.6 is the
    #     measured end-to-end breakeven -- a graphed depth-3 cycle costs ~3.7 plain steps
    #     (verify replay is PCIe expert-fetch bound at w rows), and same-boot profiles show
    #     prose LOSES at a 3.2 bar and recovers monotonically toward plain parity as the bar
    #     rises to it.
    #   cut armed (conf_cut > 0): the doomed rows never get verified, so the mean cycle is
    #     narrower and cheaper; the same cost model puts breakeven at ~1.9-2.0. The default is
    #     2.4 rather than 2.0 -- deliberately conservative, since the cut's own saving is what
    #     the bar is now being asked to trust.
    # An explicit FREETOKEN_MTP_SPEC_MIN_EMITTED always wins over both. At shallow depths the
    # 1 + depth ceiling binds first, so either default clamps to it.
    breakeven = 3.6 if conf_cut == 0.0 else 2.4
    min_emitted = _float_env(
        env, "FREETOKEN_MTP_SPEC_MIN_EMITTED", str(min(breakeven, float(1 + depth)))
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
    # On by default: the static bar is measurably wrong at one end of a long request whichever
    # end it was tuned for. "0" restores the flat bar byte for byte -- the policy then never
    # reads a clock at all -- which is what a like-for-like A/B of the two is run with.
    cost_raw = env.get("FREETOKEN_MTP_SPEC_COST_AWARE", "1").strip()
    if cost_raw not in ("0", "1"):
        raise ValueError("FREETOKEN_MTP_SPEC_COST_AWARE must be 0 or 1")
    return SpecDecodeConfig(
        enabled=enabled,
        depth=depth,
        graph=graph_raw == "1",
        conf_cut=conf_cut,
        draft_cut_mode=cut_mode,
        ema_alpha=alpha,
        min_emitted=min_emitted,
        probe_resume=probe_resume,
        cooldown=cooldown,
        cost_aware=cost_raw == "1",
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
    # The draft head is the qwen4_exp MTP block (``derive_mtp_model_config`` rebuilds it from
    # ``qwen4_args``); no other model class has one. Without this check the flag survives
    # boot and dies inside that helper with ``dataclasses.replace(None)``.
    model_type = getattr(getattr(config, "model_config", None), "model_type", None)
    if model_type != "qwen4_exp":
        raise ValueError(
            "FREETOKEN_MTP_SPECULATE=1 needs the qwen4_exp MTP draft head; this checkpoint's "
            f"model type is {model_type!r}"
        )
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
    # Main QSA K/V storage. The compressed QSA index and every non-QSA pool keep ``dtype``.
    kv_dtype: str = "bf16"
    # Complete QSA + GDN/PLE prefixes can leave VRAM between turns. Off constructs nothing;
    # enabled modes are bounded independently because RAM owns full entries while SSD owns only
    # two staging windows plus files under its disk LRU.
    kv_park: str = "off"
    kv_park_idle_ms: int = 0
    kv_park_min_tokens: int = 8192
    kv_park_ram_gib: float = 2.0
    kv_park_ssd_dir: str = "~/.cache/freetoken/kv-park"
    kv_park_ssd_gib: float = 32.0
    kv_park_window_mib: int = 256
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    nvfp4_backend: str = "triton"
    # EXL3 routed-expert operation. The packed mgemm path is the accepted default: R4e measured
    # decode 136.26 -> 35.43 ms/token and cold prefill 112.72 -> 238.83 prompt tok/s
    # (2026-09-05); reconstruct remains the fallback (--exl3-expert-op). The literal lives in
    # moe/exl3_ops.py so every site that reads the field by getattr shares one default.
    exl3_expert_op: str = DEFAULT_EXL3_EXPERT_OP
    # PLE table backend: "disk" reads rows from the checkpoint files per fill via the Linux
    # io_uring row store (Linux-only); "mmap" demand-pages the safetensors table (the Windows
    # option); "pinned" preloads the whole table into page-locked host RAM.
    ple_backend: str = "disk" if sys.platform == "linux" else "pinned"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
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
    # Per-(layer, expert) decode routing histogram for cache-skew analysis, exposed at
    # GET /v1/cache/routing. Implies moe_collect_stats. Like it, the accumulation is a
    # device-side scatter_add_ that has to be armed BEFORE graph capture to be replayed,
    # so this is boot-time only (--moe-collect-decode-freq /
    # FREETOKEN_MOE_COLLECT_DECODE_FREQ=1), never a runtime toggle.
    moe_collect_decode_freq: bool = False
    # Learn the MoE layer ranking from use (--disable-moe-learn-routing /
    # FREETOKEN_MOE_LEARN_ROUTING=0): arms the same decode routing histogram, flushes it to
    # freetoken-routing-stats.json beside the checkpoint every minute and at shutdown, and
    # lets --moe-gpu-owned-layers auto[:N] rank layers from that file on the next boot. With no
    # file (or too few routes) the boot is exactly the fixed-list boot. See moe/learned_routing.py.
    moe_learn_routing: bool = True
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # GPU-owned MoE layers (--moe-backend offload only): layers whose full expert set is
    # permanently VRAM-resident and which allocate NO host bank at all. Spec is the
    # --moe-cpu-layers grammar (explicit ids "0,1,2", a count "6", a fraction "0.125")
    # plus "auto" (the measured six hungriest layers) and "auto:N". None = off.
    # Each owned layer costs num_experts LRU slots of VRAM and returns one host bank of RAM.
    moe_gpu_owned_layers: str | None = None
    # Write a contiguous on-disk expert copy in the background after banks load (--moe-disk-copy).
    # None = auto: default on for --moe-backend offload on NVFP4 checkpoints.
    moe_disk_copy: bool | None = None
    # Directory to write the expert disk copy (--moe-disk-copy-dir; default <model>/freetoken-expert-cache).
    moe_disk_copy_dir: str | None = None
    # VRAM the MoE cache must NOT spend because something allocated AFTER it was sized
    # already owns those bytes: the integrated MTP resident draft head (2.17 GiB measured),
    # the decode/spec/draft CUDA-graph pools, and the vision layer-stream workspace. Joins
    # fixed_cache_size before the MoE-vs-KV split, exactly like the GDN state pool, so both
    # --moe-cache-auto and an explicit --moe-cache-size respect it.
    # -1 (the default) = auto: cache_budget.auto_vram_reserve_bytes composes it from the
    # features this boot actually switched on (the 2.25 GiB draft-head half only when
    # speculation is on); >= 0 is taken as typed, and 0 reserves nothing.
    # (--moe-vram-reserve-bytes; see cache_budget.DEFAULT_MOE_VRAM_RESERVE_BYTES.)
    moe_vram_reserve_bytes: int = -1
    # Free-VRAM headroom the MoE cache must leave after every known reservation. An explicit
    # --moe-cache-size that leaves less fails loudly at boot naming the largest slot count
    # that fits (--moe-cache-headroom-bytes; 1.5 GiB default).
    moe_cache_headroom_bytes: int = 3 << 29
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
