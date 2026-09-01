from __future__ import annotations

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
    """Read ``FREETOKEN_MTP_SPECULATE`` / ``FREETOKEN_MTP_SPEC_DEPTH``."""
    env = os.environ if env is None else env
    raw = env.get("FREETOKEN_MTP_SPECULATE", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError("FREETOKEN_MTP_SPECULATE must be 0 or 1")
    enabled = raw == "1"
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
    return SpecDecodeConfig(enabled=enabled, depth=depth)


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
