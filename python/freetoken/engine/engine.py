from __future__ import annotations

import gc
import math
import os
import sys
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, NamedTuple, Sequence, Tuple

import torch
from freetoken import diag
from freetoken.attention import AttnType, attention_backend_info, create_attention_backend
from freetoken.core import Batch, Context, Req, set_global_ctx
from freetoken.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from freetoken.gpu_select import gpu_identity
from freetoken.layers import set_rope_device
from freetoken.models import create_model, load_weight
from freetoken.moe import create_moe_backend, is_offload_moe_backend
from freetoken.moe.expert_banks import load_expert_banks
from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache
from freetoken.utils import align_ceil, init_logger, is_sm90_family, is_sm100_family, mem_GB, torch_dtype

from .config import EngineConfig, require_speculation_supported
from .graph import GraphRunner, get_free_memory

if TYPE_CHECKING:
    from .spec_sample import SpecDecision
from .sample import BatchSamplingArgs, Sampler
from freetoken.kvcache import create_kv_pool, resolve_pool_class
from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.kvcache.cache_status import _supports_swa_ratio
from freetoken.kvcache.linear_state_pool import (
    _linear_pool_min_slots, _linear_pool_num_slots, state_pool_bytes,
)

logger = init_logger(__name__)

# Free VRAM a speculative verify graph refuses to capture below. Same floor the observer's
# private verify graphs use: a capture that leaves the card with no headroom trades a decode
# stall for an allocator failure on the next prefill.
_SPEC_GRAPH_GUARD_BYTES = 128 << 20

# The context length the boot-time speculative captures pretend the dummy request already has.
# Small on purpose, and >0 on purpose: a verify batch continues an existing sequence
# (has_initial_state), and every length-dependent value the step reads -- positions, out_loc,
# the page/ring metadata -- is refilled per replay, so nothing about this number is baked.
_SPEC_BOOT_CAPTURE_BASE_LEN = 8

# The armed graph runner's replay split, as (its key, the probe's stage name).
_SPEC_REPLAY_TIMINGS = (
    ("copy_ms", "replay.copy"),
    ("attn_ms", "replay.attn"),
    ("model_ms", "replay.model"),
    ("launch_ms", "replay.launch"),
    ("gpu_ms", "replay.gpu"),
)


def _require_offload_cache_size(cache_size: int, num_experts: int) -> None:
    """The offload MoE cache needs at least one slot per expert per layer. A too-small size
    (e.g. a bare offload run with moe_cache_size unset and auto disabled) must fail loudly."""
    if cache_size < num_experts:
        raise ValueError(
            f"moe_cache_size={cache_size} is too small: need at least num_experts={num_experts} "
            f"slots. Pass --moe-cache-size/--moe-cache-rate, or use --moe-cache-auto "
            f"(the default for offload/hybrid backends when no cache-sizing flag is given; "
            f"--moe-backend cpu always sizes its own fixed two-layer buffer and ignores "
            f"cache-sizing flags)."
        )


def _flashinfer_available() -> bool:
    from freetoken.kernel.backend import is_flashinfer_installed

    return is_flashinfer_installed()


def _sgl_flash_attn_available() -> bool:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401
    except Exception as exc:
        detail = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
        logger.warning_rank0(
            "sgl_kernel.flash_attn is unavailable; auto attention backend falls back to fi "
            f"({type(exc).__name__}: {detail})"
        )
        return False
    return True


def _startup_kv_budget(memory_ratio: float, init_free_memory: int, new_free_memory: int) -> int:
    """Bytes available to the KV pool at startup: ratio-scaled pre-load free memory minus
    what the resident model consumed. Kept as a pure function so the composition with the
    pool families' ``solve_num_pages`` stays CPU-testable."""
    return int(memory_ratio * init_free_memory) - (init_free_memory - new_free_memory)


def _cpu_moe_executor_tokens(config) -> int:
    """Rows one CPU MoE submit may carry -- the executor's ``max_tokens`` sizing.

    Decode batches never exceed ``max_running_req``, but CUDA-graph padding can round a
    batch up to the largest captured size, so both bound it. A speculative verify step
    adds a third: it submits the WHOLE ``w = 1 + depth`` row block in one forward, and
    speculation pins ``max_running_req`` to 1 (``require_speculation_supported``), so
    neither of the other two covers it. The C++ scratch and the pinned IO buffers are
    cut to ``max_tokens`` once, ahead of graph capture -- a wider submit would run past
    them. ``batch_width`` is 1 while speculation is off, so this is inert then.
    """
    return max(
        config.max_running_req,
        config.cuda_graph_max_bs or 0,
        config.spec_decode.batch_width,
        1,
    )


def _page_table_width(max_seq_len: int, page_size: int) -> int:
    """Column count for the page table. ``_write_page_table`` writes WHOLE trailing pages, so the
    highest column touched is ``align_ceil(max_seq_len, page_size) - 1`` -- which the 32-alignment
    alone does not cover once page_size > 32 (an unaligned --max-seq-len-override on DSV4's P=128
    or trtllm's forced 64 would index past the row)."""
    return align_ceil(align_ceil(max_seq_len, page_size), 32)


def _required_attn_types(model_config) -> frozenset[AttnType]:
    """Backend-driving attention types of this model, from the group-spec walk
    (single source shared with the pool factory and the KV cost model). getattr
    fallbacks: duck-typed test configs may not implement the spec walk; for those,
    dsv4_args marks DSV4 (the real config declares a DSV4 attention group)."""
    specs_fn = getattr(model_config, "kv_cache_group_specs", None)
    if specs_fn is None:
        if getattr(model_config, "dsv4_args", None) is not None:
            return frozenset({AttnType.DSV4})
        return frozenset({AttnType.FULL})
    types = frozenset(
        spec.attn_type for spec in specs_fn() if spec.attn_type.backend_driven
    )
    return types or frozenset({AttnType.FULL})


def _backend_parts_serve(name: str, required: frozenset[AttnType]) -> bool:
    return all(
        required <= attention_backend_info(part).supported_types
        for part in name.split(",")
    )


def _backend_requirements_met(name: str) -> bool:
    # flashinfer first across ALL parts: the sgl probe logs a "falls back to fi" warning,
    # which would mislead when the candidate is about to fail on flashinfer anyway.
    infos = [attention_backend_info(part) for part in name.split(",")]
    if any(i.requires_flashinfer for i in infos) and not _flashinfer_available():
        return False
    if any(i.requires_sgl_kernel for i in infos) and not _sgl_flash_attn_available():
        return False
    if any(i.requires_sm100 for i in infos) and not is_sm100_family():
        return False
    return True


def _resolve_auto_attention_backend(required: frozenset[AttnType]) -> str:
    """First candidate (in per-type priority order) whose arch condition holds,
    whose packages are installed, and whose every comma part serves ALL required
    types. Reproduces the historical hardware tree for FULL-only models:
    sm_100 -> trtllm, sm_90+sgl_kernel -> "fa,fi", flashinfer -> fi, else triton."""
    candidates: list[tuple[str, bool]] = []
    if AttnType.DSV4 in required:
        candidates.append(("dsv4_sparse", True))
    if required & {AttnType.MLA, AttnType.DSA}:
        candidates.append(("dsa", True))
    if AttnType.BSA in required:
        candidates.append(("m3_sparse", True))
    if AttnType.QSA in required:
        candidates.append(("qsa_sparse", True))
    if AttnType.SWA in required:
        candidates.append(("triton", True))
    if AttnType.FULL in required:
        candidates += [
            ("trtllm", is_sm100_family()),
            ("fa,fi", is_sm90_family()),
            ("fi", True),
            ("triton", True),
        ]
    for name, arch_ok in candidates:
        if not arch_ok:
            continue
        if not _backend_parts_serve(name, required):
            continue
        if not _backend_requirements_met(name):
            continue
        return name
    raise RuntimeError(
        "No attention backend can serve attention types "
        f"{sorted(t.value for t in required)} on this machine."
    )


def _validate_attention_backend_choice(config, override, required: frozenset[AttnType]) -> None:
    """Config-time type x backend capability check for the resolved (or explicit)
    backend string: every comma part must serve every required type and have its
    packages/arch available. Replaces the per-model gates; in particular this is
    where a DSV4 or MLA checkpoint rejects a generic backend before weights load,
    and where a generic model rejects dsa/dsv4_sparse."""
    from freetoken.attention import validate_attn_backend

    # Name membership first (ArgumentTypeError listing the supported names): the CLI already
    # ran this, but the programmatic EngineConfig path reaches here unvalidated and would
    # otherwise die on a bare KeyError from the info lookup below.
    validate_attn_backend(config.attention_backend, allow_auto=False)

    model_config = config.model_config
    backend_parts = [p.strip() for p in config.attention_backend.split(",")]
    for part in backend_parts:
        info = attention_backend_info(part)
        missing = required - info.supported_types
        if missing:
            valid = [
                name
                for name in (
                    "fa", "fi", "trtllm", "triton", "dsa", "dsv4_sparse", "m3_sparse",
                    "qsa_sparse",
                )
                if required <= attention_backend_info(name).supported_types
            ]
            missing_names = "/".join(sorted(t.value for t in missing))
            raise ValueError(
                f"{getattr(model_config, 'model_type', 'model')} uses {missing_names} "
                f"attention, which backend {part!r} does not support; valid backends: "
                f"{', '.join(valid)} (or auto), got {config.attention_backend!r}."
            )
        if AttnType.SWA in required and not info.consumes_attn_spec:
            # SWA models drive window/sinks/sm_scale through the per-call AttentionSpec;
            # a backend that drops it would attend with the wrong window silently.
            raise ValueError(
                f"backend {part!r} does not consume the per-call AttentionSpec that "
                f"SWA models require, got {config.attention_backend!r}."
            )

    # An explicitly-selected backend may require a package that isn't installed. Auto
    # never resolves to one of these when its package is missing, so this only fires for
    # explicit --attention-backend choices.
    for part in backend_parts:
        info = attention_backend_info(part)
        if info.requires_flashinfer and not _flashinfer_available():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires flashinfer, which is "
                "not installed. Install it with `pip install 'freetoken[fi]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sgl_kernel and not _sgl_flash_attn_available():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires sgl_kernel, which is "
                "not installed. Install it with `pip install 'freetoken[sgl]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sm100 and not is_sm100_family():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires a compute capability "
                "10.x GPU: flashinfer's trtllm-gen kernels ship sm_100a/103a cubins only. "
                "Use --attention-backend fi (or triton) instead."
            )

    if required & {AttnType.MLA, AttnType.DSA}:
        # Plain MLA/DSA runs on page_size 1; the kpool indexer layout needs 64.
        _kpool_ratio = max(
            (s.index_ratio for s in model_config.kv_cache_group_specs() if s.mla),
            default=1,
        )
        want_page = 64 if _kpool_ratio > 1 else 1
        if config.page_size != want_page:
            logger.warning_rank0(
                f"Page size {config.page_size} is auto-adjusted to {want_page} "
                f"for latent-KV attention."
            )
            override("page_size", want_page)

    for part in backend_parts:
        info = attention_backend_info(part)
        if info.page_sizes is not None and config.page_size not in info.page_sizes:
            override("page_size", info.page_sizes[-1])
            logger.warning_rank0(
                f"Page size is overridden to {info.page_sizes[-1]} for the {part} backend"
            )


def _make_dummy_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    for key, param in model_state.items():
        if param.dtype in fp8_dtypes:
            # torch.randn is not implemented for fp8; fill via a uint8 view with small
            # codes (avoid NaN/inf fp8 encodings). Lets dummy-weight startup work for
            # block-fp8 models (the dense fp8 linears are fp8 regardless of moe_backend).
            t = torch.empty(param.shape, dtype=param.dtype, device=device)
            t.view(torch.uint8).random_(0, 16)
            state_dict[key] = t
        elif param.dtype.is_floating_point or param.dtype.is_complex:
            state_dict[key] = torch.randn(param.shape, dtype=param.dtype, device=device)
        elif param.dtype == torch.uint8 and key.endswith("weight_scale_inv"):
            # MXFP8 e8m0 exponent codes: 127 encodes scale 1.0; zeros would collapse
            # every scale to 2^-127 and zero the model. Scoped BY NAME: other uint8
            # buffers are packed payloads whose bytes mean something else entirely
            # (GGUF qweight blocks embed fp16 scales -- 0x7F7F is fp16 NaN), so they
            # keep the benign all-zeros fill below.
            state_dict[key] = torch.full(param.shape, 127, dtype=param.dtype, device=device)
        else:
            state_dict[key] = torch.zeros(param.shape, dtype=param.dtype, device=device)
    return state_dict


def _materialize_loaded_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    weights: Iterable[Tuple[str, torch.Tensor]],
    *,
    device: torch.device,
    device_for_key: Callable[[str, torch.device], torch.device] | None = None,
) -> Dict[str, torch.Tensor]:
    """Materialize checkpoint values on their model-owned persistent devices.

    Models without ``device_for_key`` retain the historical one-device behavior. The
    callback is deliberately key-local so optional CPU-resident components do not leak
    model-specific naming into the engine.
    """
    state_dict: Dict[str, torch.Tensor] = {}
    for key, weight in weights:
        expected = model_state.get(key)
        target = device_for_key(key, device) if device_for_key is not None else device
        if expected is None:
            state_dict[key] = weight.to(device=target)
        else:
            state_dict[key] = weight.to(device=target, dtype=expected.dtype)
    return state_dict


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class SpecForwardOutput(NamedTuple):
    """One speculative step's verdict, plus what the next cycle needs.

    ``next_tokens_gpu`` carries only the emitted run (``len(decision.tokens)`` ids), because
    the rejected rows' sampled tokens must never reach the token pool. ``hidden`` is all ``w``
    rows of the target's pre-mix stream: the draft head consumes the accepted prefix of it,
    and which prefix that is is not known until the scheduler's stop scan has run.
    """

    decision: "SpecDecision"
    next_tokens_gpu: torch.Tensor
    hidden: torch.Tensor


class Engine:
    # MoE layer ids resolved from --moe-gpu-owned-layers. A CLASS default so the budget
    # helpers read a sane empty set on an Engine.__new__(Engine) stub (the unit tests build
    # one to exercise _resolve_auto_moe_cache_size without a GPU); __init__ rebinds it.
    _gpu_owned_layer_ids: frozenset = frozenset()

    def __init__(self, config: EngineConfig):
        from .mtp_shadow import MTPShadowConfig

        self._mtp_shadow_config = MTPShadowConfig.from_env(config)
        require_speculation_supported(config)
        self.mtp_shadow_observer = None
        # Integrated speculation's three pieces. None -- and costing nothing, byte-identically
        # -- while FREETOKEN_MTP_SPECULATE is off. The ladder is sized beside the linear state
        # pool below; the head and the sampler are built last, after graphs and warmup.
        self.spec_draft = None
        self.spec_sampler = None
        self.spec_state_ladder = None
        self.spec_graph_runner = None
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _ensure_expandable_segments()  # before the first CUDA allocation below

        from freetoken.gpu_select import bind_assigned_gpu

        self.device = bind_assigned_gpu(config.tp_info.rank)
        _adjust_config(config)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.config = config  # retained for runtime cache rebuild (rebuild_runtime_cache)
        # KV pool family fixed at construction from the model config: its classmethods own the
        # page-token geometry and cost arithmetic the engine needs BEFORE the pool exists
        # (num_pages sizing, --moe-cache-auto); the instance owns rebuild/validation after.
        self._pool_cls = resolve_pool_class(config.model_config)
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        free_min, free_max = self._sync_get_memory()
        init_free_memory = free_max  # startup KV sizing keeps cross-rank MAX (unchanged)
        self._baseline_free = free_min  # rebuild baseline: cross-rank MIN, deterministic across ranks
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self._install_model_weights(config)
        post_weights_free = self._sync_get_memory()[0]
        self._weights_bytes = self._baseline_free - post_weights_free
        # Pool-budget baseline for the desktop cache sliders: free VRAM after the weights are
        # resident but before ANY runtime cache pool (MoE expert cache below, KV pages, GDN
        # state) is allocated. This is the stable "if all free VRAM went to one pool" budget —
        # unlike a query-time mem_get_info it doesn't drift with allocator caching, CUDA
        # graphs, or other processes. Cross-rank MIN, deterministic across ranks.
        self._post_weights_free = post_weights_free
        self.moe_offload_cache = None
        # MoE layer ids resolved from --moe-gpu-owned-layers; read by the budget helpers
        # below and by the cache build. Empty until _init_offload_moe_cache resolves it.
        self._gpu_owned_layer_ids: frozenset = frozenset()
        self.cpu_moe_executor = None
        # Host-side auxiliary stores (qwen4_exp's pinned PLE table): after the weights so a
        # load failure is not masked, before the MoE offload cache so the bank residency
        # planning sees the pin quota the table already spent.
        self._host_tables_bytes = 0
        if hasattr(self.model, "load_host_tables"):
            self._host_tables_bytes = int(self.model.load_host_tables(config) or 0)
        if is_offload_moe_backend(config.moe_backend):
            self._init_offload_moe_cache(config)
        if hasattr(self.model, "prepare_for_runtime"):
            self.model.prepare_for_runtime()

        # ======================= KV cache initialization ========================
        new_free = self._sync_get_memory()[1]
        # The engine measures the budget and settles the sibling GDN state pool's bytes
        # off it; the KV pool family owns every geometry-specific formula behind the rest.
        available_memory = _startup_kv_budget(config.memory_ratio, init_free_memory, new_free)
        available_memory -= state_pool_bytes(config)
        self.num_pages = self._pool_cls.solve_num_pages(config, available_memory)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kv_pool(
            config, self.num_pages, device=self.device, dtype=self.dtype
        )

        # ======================= Linear (GatedDeltaNet) state initialization ========================
        linear_group = config.model_config.linear_attention_group()
        if linear_group is not None:
            from freetoken.kvcache.linear_state_pool import LinearStatePool

            self.linear_state_pool = LinearStatePool(
                group=linear_group,
                num_slots=_linear_pool_num_slots(config),
                dtype=self.dtype,
                device=self.device,
                tp_size=config.tp_info.size,
                slot_states=config.model_config.slot_states,
            )
            self.ctx.linear_state_pool = self.linear_state_pool
        else:
            self.linear_state_pool = None

        # Integrated speculation's linear-state rollback (design section 4, Strategy R). Sized
        # once here so the spare pool slot and the activation arena are charged at boot rather
        # than at the first speculative cycle; None -- and costing nothing -- when the flag is off.
        if config.spec_decode.enabled and self.linear_state_pool is not None:
            from freetoken.engine.spec_state_ladder import SpecStateLadder

            self.spec_state_ladder = SpecStateLadder(
                self.linear_state_pool, config.spec_decode.batch_width
            )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        # Pools routed by the shared table but deriving reads through their own mappings (DSV4)
        # re-point here (and again on any table realloc). The graph-input snapshot that reads
        # through them belongs to the attention backend, built later in init_capture_graph.
        self.kv_cache.attach_page_table(self.page_table)

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # padded/dummy rows index the GDN padding slot (0) so gather/scatter hits scratch.
        if self.linear_state_pool is not None:
            self.dummy_req.linear_slot_idx = self.linear_state_pool.padding_slot
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            moe_offload_cache=self.moe_offload_cache,
        )
        if config.attention_backend.split(",")[0] == "triton":
            # Prefill runs on the first comma part; warm its autotune cache.
            self._warmup_prefill()
        # Heavy private state is deliberately last: target weights, trusted pools, graphs, and
        # warmup are complete before the opt-in observer gets any memory.
        if self._mtp_shadow_config.enabled:
            from .mtp_shadow import MTPShadowObserver

            self.mtp_shadow_observer = MTPShadowObserver(self, self._mtp_shadow_config)
        if config.spec_decode.enabled:
            # Same placement as the observer's, and for the same reason: the target's weights,
            # pools, graphs and warmup are all settled before the draft head takes memory.
            from .spec_draft import SpecDraftHead
            from .spec_sample import SpecSampler

            self.spec_draft = SpecDraftHead(self, config.spec_decode)
            self.spec_sampler = SpecSampler.from_config(config.spec_decode, self.device)
            logger.info_rank0(
                "Integrated MTP speculation enabled: depth "
                f"{config.spec_decode.depth}, draft head "
                f"{mem_GB(self.spec_draft.resident_bytes)} resident "
                f"(experts {self.spec_draft.expert_placement}, "
                f"lm_head {self.spec_draft.lmhead_placement})"
            )
        if config.spec_decode.graph_widths:
            # With FREETOKEN_MTP_SPEC_GRAPH unset the runner does not exist and the step is eager.
            from .spec_graph import SpecVerifyGraphRunner

            self.spec_graph_runner = SpecVerifyGraphRunner(
                target_ctx=self.ctx,
                target_model=self.model,
                attn_backend=self.attn_backend,
                device=self.device,
                widths=config.spec_decode.graph_widths,
                guard_bytes=_SPEC_GRAPH_GUARD_BYTES,
            )
            logger.info_rank0(
                "Integrated MTP graphs armed for widths "
                f"{config.spec_decode.graph_widths} (1 = the capture-decode step)"
            )
        if config.spec_decode.enabled or config.spec_decode.graph_widths:
            # ...and captured HERE, beside the decode graphs, while boot memory is still fresh:
            # the verify widths, then the DRAFT head's own chain and commit graphs, then the
            # ladder's replay rungs (which need the verify warm-ups' stash to have run).
            self._capture_spec_graphs_at_boot()

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _install_model_weights(self, config: EngineConfig) -> None:
        """Load the weights, let the model adopt the weight sources it built while loading,
        then log where everything landed -- in that order.

        ``adopt_weight_sources`` exists for state the loader creates but the state dict
        cannot carry: qwen4_exp's memory-mapped picture-weight holder is built inside
        ``iter_weights`` and has to reach the vision tower, which is what lets it prefetch
        its extent. It has to happen before the placement report, because the report
        describes what those sources decided; adopting later (it used to ride along in
        ``load_host_tables``) made every mapped boot log ``backing=ram`` while the mapping
        was live.
        """
        self.model.load_state_dict(self._load_weight_state_dict(config))
        adopt = getattr(self.model, "adopt_weight_sources", None)
        if callable(adopt):
            adopt(config)
        placement_report = getattr(self.model, "weight_placement_report", None)
        if callable(placement_report):
            report = placement_report()
            if report:
                logger.info_rank0(report)

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        model_state = self.model.state_dict()
        if config.use_dummy_weight:
            return _make_dummy_weight_state_dict(model_state, device=self.device)
        # _materialize casts each loaded tensor to its model-param dtype (model_state), so
        # models declaring per-tensor dtypes (e.g. DSV4's mixed fp8/fp32/bf16) are preserved;
        # offload models exclude experts (served from the offload cache, not dense weights).
        return _materialize_loaded_weight_state_dict(
            model_state,
            load_weight(
                config.model_path,
                self.device,
                include_moe_experts=not is_offload_moe_backend(config.moe_backend),
            ),
            device=self.device,
            device_for_key=getattr(self.model, "weight_device_for_key", None),
        )

    def _resolve_auto_moe_cache_size(self, config: EngineConfig, banks) -> tuple[int, int, bool]:
        """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

        Pure glue over the Phase-1 budget policy; isolated here so it is unit-testable
        without a GPU. Reused by the Phase-2 runtime rebuild.
        """
        from freetoken.engine.cache_budget import (
            expert_bytes_per_slot,
            gpu_owned_reservation_bytes,
            resolve_moe_cache_auto,
        )

        cache_per_page, fixed_cache_size, page_tokens, min_reserve = self._pool_cls.kv_cost(config)
        fixed_cache_size += state_pool_bytes(config)  # sibling GDN state pool, engine-summed
        num_experts = config.model_config.num_experts
        owned = self._gpu_owned_layer_ids
        per_expert_bytes = expert_bytes_per_slot(banks.sources, owned)
        # GPU-owned layers hold a full expert layer of VRAM forever and leave the slot cache
        # entirely, exactly as state_pool_bytes accounts for the GDN pool.
        fixed_cache_size += gpu_owned_reservation_bytes(len(owned), num_experts, per_expert_bytes)
        total_experts = (config.model_config.num_moe_layers - len(owned)) * num_experts
        moe_cache_size, num_pages, overlap = resolve_moe_cache_auto(
            baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes,
            memory_ratio=config.memory_ratio,
            cache_per_page=cache_per_page,
            fixed_cache_size=fixed_cache_size,
            per_expert_bytes=per_expert_bytes,
            num_experts=num_experts,
            total_experts=total_experts,
            prefill_overlap=config.moe_prefill_overlap,
            kv_reserve_tokens=max(config.kv_reserve_tokens, min_reserve),
            page_size=page_tokens,
            quant_format=banks.quant_format,
        )
        # Page 0 is the pool's dummy. Additional pages beyond the model's usable context
        # cannot serve a request, so do not spend residual auto-budget on unreachable KV.
        max_context_pages = -(-config.max_seq_len // page_tokens) + 1
        return moe_cache_size, min(num_pages, max_context_pages), overlap

    def _charge_gpu_owned_layers_to_cache_size(
        self, config: EngineConfig, owned: "frozenset[int]"
    ) -> None:
        """Take the owned layers' resident slots OUT of an explicit ``--moe-cache-size``.

        ``--moe-cache-size`` is the total expert-slot budget on the card, so switching
        ``--moe-gpu-owned-layers`` on trades LRU slots for resident layers rather than adding
        7.9 GiB of VRAM on top of them. Called exactly once, from the single
        :meth:`_init_offload_moe_cache` call site, before anything reads the size --
        ``--moe-cache-auto`` charges the same bytes through ``fixed_cache_size`` instead.
        """
        if not owned or config.moe_cache_auto or not config.moe_cache_size:
            return
        total = config.moe_cache_size
        lru = _gpu_owned_lru_slots(config, len(owned))
        if lru == total:
            return
        object.__setattr__(config, "moe_cache_size", lru)
        logger.info_rank0(
            f"--moe-cache-size {total} is the total MoE expert-slot budget: "
            f"{len(owned)} GPU-owned layer(s) hold {total - lru} of those slots, "
            f"leaving {lru} for the streaming-layer LRU"
        )

    def _check_gpu_owned_cache_fits(self, config: EngineConfig, banks) -> None:
        """Explicit --moe-cache-size + the owned reservation must fit the same budget the
        auto path solves against. Fails loudly (never shrinks) -- see spec section 6."""
        from freetoken.engine.cache_budget import (
            check_explicit_moe_cache_fits,
            expert_bytes_per_slot,
            net_cache_budget_bytes,
        )
        from freetoken.utils import div_ceil

        owned = self._gpu_owned_layer_ids
        if not owned or config.moe_cache_auto:
            return
        cache_per_page, fixed_cache_size, page_tokens, min_reserve = self._pool_cls.kv_cost(config)
        fixed_cache_size += state_pool_bytes(config)
        budget = net_cache_budget_bytes(
            config.memory_ratio, self._baseline_free, self._weights_bytes, fixed_cache_size
        )
        kv_reserve_pages = (
            div_ceil(max(config.kv_reserve_tokens, min_reserve), page_tokens) + 1
        )
        check_explicit_moe_cache_fits(
            moe_cache_size=config.moe_cache_size,
            per_expert_bytes=expert_bytes_per_slot(banks.sources, owned),
            budget_bytes=budget - kv_reserve_pages * cache_per_page,
            owned_layers=len(owned),
            num_experts=config.model_config.num_experts,
        )

    def _init_offload_moe_cache(self, config: EngineConfig) -> OffloadMoeCache:
        # A model may fully own cache construction via make_offload_moe_cache.
        # Otherwise load_expert_banks gives the model module a setup hook first, then
        # falls back to per-quant providers, and the engine wires the banks into cache.
        cache_factory = getattr(self.model, "make_offload_moe_cache", None)
        if cache_factory is not None and config.moe_cache_auto:
            raise ValueError(
                "--moe-cache-auto is not supported for models with a custom "
                "make_offload_moe_cache; pass --moe-cache-size explicitly."
            )
        # decode_target picks the bank layout + the per-decode mechanism:
        #   "hybrid" -> GPU-cache + CPU-overflow co-compute, every layer (--moe-backend hybrid);
        #   "cpu"    -> CPU executor for the cpu_layer_ids set (all layers under --moe-backend
        #               cpu, the --moe-cpu-layers subset under offload);
        #   "gpu"    -> plain GPU offload.
        # cpu/hybrid both read experts on the CPU, so banks load in the native (CPU-readable)
        # layout; the GPU slot-cache GEMM reads those same native rows. decode_target also
        # gates the CPU executor build below.
        cpu_layer_ids = _resolve_cpu_layers(config, config.model_config.num_moe_layers)
        # Resolved (and fully validated) here, not just in _adjust_config: the backend may
        # still have been 'auto' at parse time. Stored on the engine because the budget
        # helpers read it.
        gpu_owned_layer_ids = _validate_gpu_owned_layers(
            config, config.model_config.num_moe_layers
        )
        self._gpu_owned_layer_ids = gpu_owned_layer_ids
        self._charge_gpu_owned_layers_to_cache_size(config, gpu_owned_layer_ids)
        if (
            not cpu_layer_ids
            and config.moe_cpu_layers is None
            and config.moe_backend in ("offload", "hybrid")
            and _pin_budget_bytes(self._host_tables_bytes) is not None
        ):
            cpu_layer_ids = _auto_cpu_layers(
                config,
                config.model_config.num_moe_layers,
                reserved=self._host_tables_bytes,
                gpu_owned=len(gpu_owned_layer_ids),
            ) - gpu_owned_layer_ids  # an owned layer has no host bank to lock
        if config.moe_backend == "hybrid":
            decode_target = "hybrid"
        elif cpu_layer_ids:
            decode_target = "cpu"
        else:
            decode_target = "gpu"
        # split residency: where pinning is quota-capped (_pin_budget_bytes), pin only the GPU layers' banks and mlock the CPU layers'
        # uncapped hosts keep every bank pinned (CPU decode reads them the same; overlap prefill stays on)
        # not applied to plain --moe-backend cpu; all-locked under a cap = --moe-backend offload --moe-cpu-layers 1.0
        split_residency = (
            bool(cpu_layer_ids)
            and config.moe_backend in ("offload", "hybrid")
            and _pin_budget_bytes(self._host_tables_bytes) is not None
        )
        if config.moe_backend == "cpu" and not split_residency:
            # cpu mode pins every bank for the prefill double buffer; over the pin cap that dies in cudaHostRegister, so lock everything instead
            from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

            budget = _pin_budget_bytes(self._host_tables_bytes)
            bank_bytes = None
            if budget is not None:
                bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(
                    config.model_config, gpu_owned=len(gpu_owned_layer_ids)
                )
            if bank_bytes and bank_bytes > budget:
                split_residency = True
                logger.info_rank0(
                    f"--moe-backend cpu: banks {bank_bytes / 2**30:.2f} GiB exceed the "
                    f"pin budget; OS-locking all layers instead of pinning"
                )
        if split_residency and config.moe_prefill_overlap:
            # locked (unregistered) layers cannot feed the async pinned H2D double buffer; their prefill is a synchronous pageable copy via materialize
            logger.info_rank0(
                "--moe-cpu-layers split residency: disabling MoE prefill overlap "
                "(locked layers prefill via synchronous pageable copies)"
            )
            object.__setattr__(config, "moe_prefill_overlap", False)
        if cache_factory is None:
            # Fast path: an FTW checkpoint loads its repacked banks directly.
            # Slow path: load_expert_banks auto-picks parallel vs serial baseline by
            # expert-tensor granularity. Both pin-after-fill.
            # --expert-load: serial/parallel force the read; auto (None) lets load_expert_banks
            # pick (parallel for scattered experts, with a low-RAM fallback to serial).
            expert_parallel = {"serial": False, "parallel": True}.get(config.expert_load, None)
            requested_residency = None
            if split_residency or gpu_owned_layer_ids:
                from freetoken.moe.host_banks import HostResidency

                requested_residency = [
                    HostResidency.GPU_OWNED.value if i in gpu_owned_layer_ids
                    else HostResidency.LOCKED.value
                    if (split_residency and i in cpu_layer_ids)
                    else HostResidency.PINNED.value
                    for i in range(config.model_config.num_moe_layers)
                ]
            banks = load_expert_banks(
                config.model_path,
                config.model_config,
                device=self.device,
                dtype=self.dtype,
                dummy=config.use_dummy_weight,
                parallel=expert_parallel,
                decode_target=("cpu" if decode_target in ("cpu", "hybrid") else "gpu"),
                layer_residency=requested_residency,
            )
            if config.moe_cache_auto:
                size, pages, overlap = self._resolve_auto_moe_cache_size(config, banks)
                object.__setattr__(config, "moe_cache_size", size)
                object.__setattr__(config, "moe_prefill_overlap", overlap)
                if config.num_page_override is None:
                    # Honor the plan's KV half too: MoE slots and KV pages were solved
                    # against ONE budget (ratio x baseline - weights), so both must come
                    # from it. Re-solving pages later from a fresh free-memory reading
                    # double-counts everything allocated since the weights measurement
                    # (this expert cache, the CPU-executor GPU buffers, allocator
                    # slack) and goes negative whenever the expert fill is exact --
                    # a greedy fill leaves no headroom for the measurement delta.
                    object.__setattr__(config, "num_page_override", pages)
                logger.info_rank0(
                    f"--moe-cache-auto resolved moe_cache_size={size} "
                    f"num_pages={pages} (prefill_overlap={overlap})"
                )
            else:
                self._check_gpu_owned_cache_fits(config, banks)
            _require_offload_cache_size(config.moe_cache_size, config.model_config.num_experts)
            cache = OffloadMoeCache(
                # Models with leading dense layers (GLM-4) only have experts on the MoE
                # layers; num_moe_layers == num_layers when first_k_dense_replace == 0.
                num_layers=config.model_config.num_moe_layers,
                num_experts=config.model_config.num_experts,
                cache_size=config.moe_cache_size,
                device=self.device,
                cache_policy=config.moe_cache_policy,
                prefill_overlap=config.moe_prefill_overlap,
                prefill_hit_d2d=config.moe_prefill_hit_d2d,
                quant_format=banks.quant_format,
                decode_target=decode_target,
                hybrid_max_fetch=config.moe_hybrid_max_fetch,
            )
            # before set_bank_sources: the residency validation and the copy plan's skip of non-pinned layers key on the CPU-layer set; the GPU-owned set is validated against the residency labels the loader honored
            cache.cpu_layer_ids = cpu_layer_ids
            cache.gpu_owned_layer_ids = gpu_owned_layer_ids
            cache.set_bank_sources(
                banks.sources,
                layer_residency=banks.layer_residency,
                gpu_owned_layers=gpu_owned_layer_ids,
            )
            cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
            if gpu_owned_layer_ids:
                from freetoken.engine.cache_budget import expert_bytes_per_slot

                logger.info_rank0(
                    _gpu_owned_boot_line(
                        gpu_owned_layer_ids,
                        config.model_config.num_moe_layers,
                        config.model_config.num_experts,
                        expert_bytes_per_slot(banks.sources, gpu_owned_layer_ids),
                        config.moe_cache_size,
                    )
                )
        else:
            cache = cache_factory(config, self.device)
            cache.decode_target = decode_target
            cache.hybrid_max_fetch = config.moe_hybrid_max_fetch
            cache.cpu_layer_ids = cpu_layer_ids
        if decode_target == "hybrid":
            self._resolve_hybrid_fetch(config, cache)
        # Must be set before CUDA graph capture so the (device-side) accumulation ops are
        # captured and re-run on every decode replay.
        collect_decode_freq = config.moe_collect_decode_freq or _env_flag(
            "FREETOKEN_MOE_COLLECT_DECODE_FREQ"
        )
        cache.collect_stats = config.moe_collect_stats or collect_decode_freq
        # The routing histogram is a per-layer scatter_add_ over device tensors, so a
        # captured decode graph replays it like any other decode op -- but only if it is
        # armed here, before capture. Arming it later leaves the graph without the scatter.
        cache.collect_decode_freq = collect_decode_freq
        if collect_decode_freq:
            logger.info_rank0(
                "MoE decode routing histogram armed (GET /v1/cache/routing); the capture "
                "warm-up contributes a handful of counts before the first real token"
            )
        # attach_offload_moe_cache walks for OffloadMoELayers, or defers to a model's
        # _iter_offload_moe_layers() hook when its MoE blocks are bespoke nn.Modules (DSV4).
        layers = attach_offload_moe_cache(self.model, cache)
        assert len(layers) == config.model_config.num_moe_layers
        if cache.decode_target in ("cpu", "hybrid"):
            self._init_cpu_moe_executor(config, cache, layers)
        self.ctx.moe_offload_cache = cache
        self.moe_offload_cache = cache
        return cache

    def _resolve_hybrid_fetch(self, config: EngineConfig, cache) -> None:
        """Resolve --moe-hybrid-max-fetch -1 (auto) into a bandwidth-matched fetch fraction.

        Perfect fetch/compute overlap wants fetched : cpu-computed misses = pcie_bw :
        (cpu_bw - pcie_bw), i.e. fetching a pcie_bw / cpu_bw fraction of each decode
        step's misses -- both sides then finish together instead of one idling. The
        achieved bandwidths come from the cached `ft bench bw` profile (the same one the
        auto backend pick reads); without a usable profile the old fixed cap of 1 applies.
        """
        if config.moe_hybrid_max_fetch >= 0:
            return  # explicit fixed cap
        from freetoken.moe.bench_profile import load_hybrid_fetch_fraction

        gpu_name, gpu_uuid = _profile_gpu(self.device.index)
        fraction = load_hybrid_fetch_fraction(
            cache.quant_format, gpu_name=gpu_name, gpu_uuid=gpu_uuid
        )
        if fraction is None:
            cache.hybrid_max_fetch = 1
            logger.warning_rank0(
                "--moe-hybrid-max-fetch auto: no usable `ft bench bw` profile for "
                f"{cache.quant_format!r} experts; using a fixed fetch cap of 1"
            )
            return
        cache.hybrid_max_fetch = cache.num_experts  # inert: the fraction is the cap
        cache.hybrid_fetch_fraction = fraction
        logger.info_rank0(
            f"--moe-hybrid-max-fetch auto: fetching {fraction:.1%} of each decode step's "
            "expert misses over PCIe (benched PCIe/CPU bandwidth ratio), the rest on the CPU"
        )

    def _init_cpu_moe_executor(self, config: EngineConfig, cache, layers) -> None:
        """Build the persistent CPU MoE executor (decode-time expert compute).

        Must run before CUDA graph capture: the worker pool has to be live for the
        eager warmup forward, and the pinned IO buffers / host-func task pointers
        must be stable for the captured nodes. Buffers/tasks themselves are
        allocated lazily on the first (eager) forward at each batch size.
        """
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        sample = layers[0]
        required = ("top_k", "activation", "apply_router_weight_on_input")
        if not all(hasattr(sample, attr) for attr in required):
            raise NotImplementedError(
                "CPU MoE backend is not yet supported for this model architecture "
                f"(MoE layer {type(sample).__name__} is missing {required})."
            )
        max_tokens = _cpu_moe_executor_tokens(config)
        # gpt-oss mxfp4 carries clamped-swiglu scalars; other formats use the defaults.
        executor = CpuMoeExecutor(
            cache,
            top_k=sample.top_k,
            activation=sample.activation,
            apply_router_weight_on_input=sample.apply_router_weight_on_input,
            num_threads=config.moe_cpu_threads,
            max_tokens=max_tokens,
            device=self.device,
            swiglu_alpha=getattr(sample, "hidden_act_alpha", 1.702),
            swiglu_limit=getattr(sample, "swiglu_limit", None),
        )
        cache.set_cpu_executor(executor)
        self.cpu_moe_executor = executor

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def _target_moe_and_expert_bytes(self, moe_cache_size: int | None) -> tuple[int, int]:
        from freetoken.engine.cache_budget import expert_bytes_per_slot

        target_moe = (
            moe_cache_size
            if moe_cache_size is not None
            else (self.moe_offload_cache.cache_size if self.moe_offload_cache else 0)
        )
        per_expert_bytes = (
            expert_bytes_per_slot(
                self.moe_offload_cache.bank_sources, self._gpu_owned_layer_ids
            )
            if self.moe_offload_cache is not None else 0
        )
        return target_moe, per_expert_bytes

    def _resize_kv_pool(self, config, num_pages: int, num_swa_pages: int | None) -> None:
        # IN-PLACE, identity-preserving: the CacheManager's swa_pool reference, ctx.kv_cache and
        # the model's per-access pool property all keep pointing at THIS pool, which frees its old
        # buffers before allocating the new ones. mark_for_rebind re-binds per-bind scratch on the
        # next forward (graph re-capture); the prefix tree + page bookkeeping reset is the
        # scheduler's generic cache_manager.rebuild.
        if self.kv_cache.needs_rebind_on_rebuild:
            self.model.mark_for_rebind()
        self.kv_cache.rebuild_from_config(config, num_pages, num_swa_pages=num_swa_pages)
        self.num_pages = num_pages

    def _refresh_seq_state(self, config) -> None:
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        if aligned_max_seq_len != self.page_table.shape[1]:
            # max_seq_len changed (e.g. KV grew past the startup token budget); the page table
            # columns must track it or new requests would index out of bounds. The scheduler
            # re-points its managers to engine.page_table on a num_pages rebuild.
            self.ctx.page_table = self.page_table = torch.zeros(
                (config.max_running_req + 1, aligned_max_seq_len),
                dtype=torch.int32,
                device=self.device,
            )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)
        self.kv_cache.attach_page_table(self.page_table)

    @torch.inference_mode()
    def rebuild_runtime_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only in-place resize of the MoE slot cache, KV page pool, GDN (mamba) state pool,
        and/or the window pool (num_swa_pages: an absolute pinned window), followed by CUDA-graph
        re-capture. Does NOT reload weights or host expert banks. The caller (scheduler) must
        guarantee no in-flight prefill/decode.
        """
        config = self.config
        if (moe_cache_size is None and num_pages is None and num_mamba_slots is None
                and num_swa_pages is None):
            return

        # 0a. Geometry prevalidation BEFORE any destructive free. An invalid target (moe
        #     slots on a model with no offload cache, moe below num_experts / above the
        #     marlin cap, non-positive pages, or too few GDN slots to run) must reject
        #     recoverably with the old cache intact -- NOT after teardown, which would
        #     leave the server unable to serve. These checks are model-agnostic.
        if moe_cache_size is not None:
            if self.moe_offload_cache is None:
                raise CacheRebuildRejected(
                    "moe_cache_size requested but this model has no MoE offload cache"
                )
            try:
                self.moe_offload_cache.validate_rebuild(moe_cache_size)
            except ValueError as e:
                raise CacheRebuildRejected(str(e)) from e
        if num_pages is not None and num_pages <= 0:
            raise CacheRebuildRejected(f"num_pages must be positive, got {num_pages}")
        if num_mamba_slots is not None:
            if self.linear_state_pool is None:
                raise CacheRebuildRejected(
                    "num_mamba_slots requested but this model has no GDN state pool"
                )
            # num_mamba_slots is the USABLE slot count (what the user sets and the status bar
            # shows); the pool also reserves a padding sink (slot 0), so the physical pool is
            # num_mamba_slots + 1. _linear_pool_min_slots is the physical floor -> usable - 1.
            min_usable = _linear_pool_min_slots(config) - 1
            if num_mamba_slots < min_usable:
                raise CacheRebuildRejected(
                    f"num_mamba_slots {num_mamba_slots} is below the minimum {min_usable} "
                    f"(non-evictable working set for max_running_req={config.max_running_req}) "
                    f"needed to run; admission would deadlock"
                )
        if num_swa_pages is not None:
            # An absolute window pin for the radix-SWA window pool (Gemma) or the DSV4 window tier;
            # meaningless for dense/MHA models and the naive SWA path (concurrency x window).
            if not _supports_swa_ratio(config):
                raise CacheRebuildRejected(
                    "num_swa_pages requested but this model has no window pool "
                    "(needs DSV4 or a sliding-window model with --cache-type radix)"
                )
            if num_swa_pages <= 0:
                raise CacheRebuildRejected(
                    f"num_swa_pages must be positive, got {num_swa_pages}"
                )

        # 0b. Pool-family budget fit-check BEFORE any destructive free: an unfit geometry
        #     must reject (recoverable) so the old caches stay intact and serving continues,
        #     rather than freeing and then OOMing into permanent failure. The engine supplies
        #     the memory account; the pool answers whether its target geometry fits.
        target_moe, per_expert_bytes = self._target_moe_and_expert_bytes(moe_cache_size)
        # Price the sibling GDN state pool at ITS target (physical slots = usable + padding
        # sink) and hand the bytes in -- the KV pool only budgets its own tiers.
        target_mamba = (
            num_mamba_slots + 1
            if num_mamba_slots is not None
            else (self.linear_state_pool.num_slots if self.linear_state_pool is not None else None)
        )
        self.kv_cache.validate_rebuild(
            config, num_pages=num_pages,
            num_swa_pages=num_swa_pages, target_moe=target_moe,
            per_expert_bytes=per_expert_bytes, baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes, current_num_pages=self.num_pages,
            extra_fixed_bytes=(
                state_pool_bytes(config, target_mamba) if target_mamba is not None else 0
            ),
            extra_note=(
                f", mamba={target_mamba - 1} slots" if target_mamba is not None else ""
            ),
        )

        torch.cuda.synchronize(self.device)
        # Preserve the CUDA-graph batch-size set resolved at startup. The auto heuristic keys
        # off free memory, which is far smaller now that the caches are resident (post-cache
        # free << startup pre-load free), so re-deriving it here would silently drop large
        # batch sizes after the first rebuild. Reusing the already-resolved list keeps the
        # captured coverage identical (the fit-check above guarantees the graph headroom fits).
        prior_graph_bs = self.graph_runner.graph_bs_list
        # Point of no return for the scheduler's rollback logic: from here the live graphs and
        # pools start being freed. A failure BEFORE this flag flips leaves the engine serving
        # untouched (no rollback needed); after it, only a rebuild restores service.
        self.rebuild_teardown_started = True
        # 1. Tear down CUDA graphs + backend capture scratch (free-before-alloc).
        # The speculative verify graphs go first: they bake KV-pool and page-table addresses,
        # and reset_capture drops the QSA verify metadata they replay against.
        spec_graph_widths = ()
        if self.spec_graph_runner is not None:
            spec_graph_widths = self.spec_graph_runner.widths
            self.spec_graph_runner.destroy()
            self.spec_graph_runner = None
        self.attn_backend.reset_capture()
        self.graph_runner.destroy_cuda_graphs()
        # 2. Resize caches in place (each frees its old GPU tensors before allocating).
        # Pin the new window first (validated above) so any KV-pool rebuild below sizes the window
        # to it (_dsv4_pool_sizes / _swa_paged_num_tokens read config.swa_num_pages_override).
        # frozen EngineConfig — mutate in place like the moe_cache_size path; `config.x = y` raises
        # FrozenInstanceError, which here aborts the rebuild after the CUDA graphs are gone (→ 503).
        if num_swa_pages is not None:
            object.__setattr__(config, "swa_num_pages_override", num_swa_pages)
        if moe_cache_size is not None:
            assert self.moe_offload_cache is not None, "no MoE offload cache to resize"
            self.moe_offload_cache.rebuild(moe_cache_size)
        if num_pages is not None:
            # sets self.num_pages (rebuilds KV + window)
            self._resize_kv_pool(config, num_pages, num_swa_pages)
        elif num_swa_pages is not None:
            # Window-only change: no page-count change, but re-derive the window pool at the new
            # pin against the CURRENT page count. This re-allocs the same-size full pool and
            # the resized window, both inside the pool's own rebuild_from_config.
            self._resize_kv_pool(config, self.num_pages, num_swa_pages)
        if num_mamba_slots is not None:
            # Reallocate the GDN state pool (frees old tensors first). Must sit between graph
            # teardown and re-capture so the recaptured graphs bind the new state tensors.
            # +1 for the reserved padding sink: num_mamba_slots is the usable count.
            self.linear_state_pool.rebuild(num_mamba_slots + 1)
            if self.spec_state_ladder is not None:
                # rebuild resets the free list, so the ladder's snapshot slot would otherwise
                # be handed out to a live request as well.
                self.spec_state_ladder.rebind()
        # 3. Refresh max_seq_len (+ generic page table) for the new token budget.
        self._refresh_seq_state(config)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        # 4. Re-capture CUDA graphs against the new tensors (reset_capture above re-armed
        #    the backend; _sync_get_memory empties the cache so freed memory is reclaimed).
        gc.collect()
        free_min = self._sync_get_memory()[0]
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=prior_graph_bs,  # reuse the startup-resolved set (see above)
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=free_min,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            moe_offload_cache=self.moe_offload_cache,
        )
        if spec_graph_widths:
            # re-armed, not re-captured: the widths capture lazily on their next live step,
            # against the tensors this rebuild just allocated
            from .spec_graph import SpecVerifyGraphRunner

            self.spec_graph_runner = SpecVerifyGraphRunner(
                target_ctx=self.ctx,
                target_model=self.model,
                attn_backend=self.attn_backend,
                device=self.device,
                widths=spec_graph_widths,
                guard_bytes=_SPEC_GRAPH_GUARD_BYTES,
            )

    def _capture_spec_graphs_at_boot(self) -> None:
        """Capture every armed speculative width NOW, while boot memory is still fresh.

        The decode graphs capture at boot for a reason, and the speculative ones need the same
        reason applied: capture admission needs free VRAM, and the first request's prefill
        activation spike is what takes it away. Capturing LAZILY on each width's first live step
        put every capture strictly AFTER a prefill -- so a long-context boot logged
        MEMORY_ADMISSION per width, exhausted the retry budget, and then served every step
        eagerly at roughly half the graphed rate, silently.

        Nothing here is load-bearing. A width that cannot capture now is left exactly as
        capturable at its first live step as it was before this method existed (see
        ``SpecVerifyGraphRunner.refund_attempt``), the boot never aborts on a capture failure,
        and every side effect of the warm-up forwards -- the dummy request's page-table row, the
        MoE offload cache's expert residency, the GDN slot the warm-up advanced -- is undone.

        ``FREETOKEN_MTP_SPEC_BOOT_CAPTURE=0`` restores the old lazy behaviour for debugging.
        """
        if os.environ.get("FREETOKEN_MTP_SPEC_BOOT_CAPTURE", "1").strip() == "0":
            logger.info_rank0(
                "MTP spec graph boot capture disabled (FREETOKEN_MTP_SPEC_BOOT_CAPTURE=0): "
                "every width captures lazily on its first live step"
            )
            return
        if self.device.type != "cuda":
            return
        try:
            self._capture_spec_verify_graphs_at_boot()
        finally:
            # The draft head's graphs and the ladder's rungs do not depend on the verify
            # widths having captured, only on the boot memory this method runs in -- and the
            # ladder's rungs DO depend on a verify warm-up having stashed the per-layer replay
            # parameters, which is why they come last.
            self._capture_draft_graphs_at_boot()
            self._capture_ladder_replays_at_boot()

    def _capture_spec_verify_graphs_at_boot(self) -> None:
        """The ``w``-row target forward, one graph per armed width."""
        runner = self.spec_graph_runner
        if runner is None:
            return
        widths = sorted(runner.widths)
        needed = _SPEC_BOOT_CAPTURE_BASE_LEN + max(widths)
        if self.max_seq_len < needed:
            logger.info_rank0(
                f"MTP spec graph boot capture skipped: max_seq_len {self.max_seq_len} is "
                f"below the {needed} dummy tokens the widest capture needs"
            )
            return

        dummy_row = self.page_table[self.dummy_req.table_idx]
        dummy_slot = int(dummy_row[0].item())
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record(self.stream)
        captured = 0
        try:
            for width in widths:
                if not runner.capture_pending(width):
                    continue
                try:
                    batch = self._spec_boot_batch(
                        width, _SPEC_BOOT_CAPTURE_BASE_LEN, dummy_row
                    )
                    if batch is None:
                        continue
                    result = self._capture_spec_width(runner, batch)
                except Exception as exc:  # noqa: BLE001 -- a bonus capture, never a boot gate
                    # the runner classifies its OWN failures; anything that escapes it came from
                    # building the batch, and the width simply stays lazily capturable
                    logger.warning_rank0(
                        f"MTP spec graph width {width}: boot capture raised "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                finally:
                    # the verify shapes repoint the row at the KV slots their rows write
                    dummy_row.fill_(dummy_slot)
                if result.status == "captured":
                    captured += 1
                elif result.status == "retryable":
                    runner.refund_attempt(width)
        finally:
            dummy_row.fill_(dummy_slot)
            if self.moe_offload_cache is not None:
                # the warm-up forwards moved routed experts into the cache; leave it exactly as
                # GraphRunner and _warmup_prefill leave it (their finally does the same)
                self.moe_offload_cache.reset()
        ended.record(self.stream)
        torch.cuda.synchronize(self.device)
        logger.info_rank0(
            f"MTP spec graphs captured at boot: {captured}/{len(widths)} "
            f"(widths {tuple(widths)}) in {started.elapsed_time(ended) / 1000.0:.3f} s"
        )

    def _capture_draft_graphs_at_boot(self) -> None:
        """The DRAFT head's chain and commit graphs, for the same reason the verify ones.

        ~11 ms of a 25 ms cycle is the host walking the chain's 843 launches and ~120 more for
        the commit; both are fixed sequences over the head's own private KV, so both record.
        Never a boot gate: a failure inside ``SpecDraftHead`` is already classified and logged
        by its own runner, and anything that escapes leaves the head running eager.
        """
        head = self.spec_draft
        if head is None or not getattr(head, "graphs_enabled", False):
            return
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record(self.stream)
        try:
            results = head.capture_graphs_at_boot()
        except Exception as exc:  # noqa: BLE001 -- a bonus capture, never a boot gate
            logger.warning_rank0(
                f"MTP draft graph boot capture raised {type(exc).__name__}: {exc}; "
                "the draft chain and commit run eager"
            )
            return
        finally:
            if self.moe_offload_cache is not None:
                # the draft head owns its own expert banks, but a warm-up that fell through to
                # the target's MoE would have moved experts into the shared cache
                self.moe_offload_cache.reset()
        ended.record(self.stream)
        torch.cuda.synchronize(self.device)
        captured = sum(1 for status in results.values() if status == "captured")
        logger.info_rank0(
            f"MTP draft graphs captured at boot: {captured}/{len(results)} "
            f"in {started.elapsed_time(ended) / 1000.0:.3f} s"
        )

    def _capture_ladder_replays_at_boot(self) -> None:
        """The state ladder's per-accepted-depth recurrent replays, up front.

        Captured lazily they cost ~300 ms each on the settle that first needs them, so the
        first ~30 cycles of a boot pay 1.8 s between them -- inside the window a benchmark
        measures. The ladder needs a live slot and a populated stash to record against; the
        dummy request supplies the slot and the verify captures above supplied the stash.
        """
        ladder = self.spec_state_ladder
        if ladder is None:
            return
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record(self.stream)
        try:
            results = ladder.capture_replays(self.dummy_req)
        except Exception as exc:  # noqa: BLE001 -- a bonus capture, never a boot gate
            logger.warning_rank0(
                f"MTP spec ladder boot capture raised {type(exc).__name__}: {exc}; "
                "the rungs stay lazily captured"
            )
            return
        if not results:
            return
        ended.record(self.stream)
        torch.cuda.synchronize(self.device)
        captured = sum(1 for ok in results.values() if ok)
        logger.info_rank0(
            f"MTP spec ladder replays captured at boot: {captured}/{len(results)} "
            f"(steps {tuple(sorted(results))}) in "
            f"{started.elapsed_time(ended) / 1000.0:.3f} s"
        )

    def _capture_spec_width(self, runner, batch: Batch):
        """One boot capture, ARMED and settled exactly the way a live speculative step is.

        The ladder is not optional decoration on a verify width. Capture RECORDS the forward,
        and the forward's per-layer stash writes are part of what gets recorded: ``begin`` sets
        ``batch.spec_capture``, which is what makes ``gdn.forward`` and ``ple.forward`` copy each
        row's replay inputs into the ladder's arena (``stash_gdn`` / ``stash_ple``). A graph
        recorded WITHOUT them replays a step that can never be rolled back -- every later settle
        dies in ``SpecStateLadder.rollback`` with "GDN layer index 0 never stashed", because the
        replay runs no Python and the recorded pass wrote nothing into the arena. The lazy path
        got this for free: the scheduler calls ``begin`` before the step whose forward triggers
        the capture. Boot has to arm it deliberately.

        ``begin`` takes its own snapshot -- the one ``restore_snapshot`` winds back to -- so it
        SUPERSEDES ``borrow_snapshot`` rather than composing with it (borrowing under a live
        step is refused outright), and ``rollback(req, 0)``, the complete undo, is the settle:
        it restores that snapshot again and ends the step with no accepted rows to replay. What
        ``begin`` leaves behind is host-side and per-step -- ``_live``, ``_width``, the n-gram
        context row -- and the next live ``begin`` overwrites all of it. The one thing that
        outlives the capture is ``_params``, which the ladder deliberately carries across steps
        for exactly this reason: a graphed step never re-runs the hooks' Python.

        WIDTH 1 STAYS UNARMED. It is decode-shaped, and ``gdn.forward`` stashes only on the
        prefill branch -- but ``ple.forward`` stashes on the mere presence of ``spec_capture``,
        so arming it would bake a PLE stash write into a graph replayed by ordinary decode
        steps, which never roll back and must never touch the arena. Its warm-up still advances
        the slot, so it keeps the plain borrow (the same undo ``_capture_decode_graph`` takes).
        """
        ladder = self.spec_state_ladder
        if ladder is None:
            return runner.capture(batch)
        if not batch.mtp_verify:
            with ladder.borrow_snapshot(self.dummy_req) as restore_state:
                return runner.capture(batch, restore_state=restore_state)
        req = batch.reqs[0]
        ladder.begin(req, batch)
        try:
            return runner.capture(batch, restore_state=ladder.restore_snapshot)
        finally:
            ladder.rollback(req, 0)

    def _spec_boot_batch(self, width: int, base: int, dummy_row: torch.Tensor) -> "Batch | None":
        """One capturable batch at ``width``, built on the dummy request's page-table row.

        The shape contract is ``spec_graph._check_batch``: width 1 is DECODE-shaped (the
        capture-decode graph records the ordinary forward) and every other width is the
        ``mtp_verify`` prefill batch. The source of truth for the verify shape is
        ``Scheduler._prepare_spec_batch`` (positions from ``extend_len``, ``out_loc`` read back
        from the page-table row, ``emit_width``, the ``mtp_verify`` marker, the GDN slot map)
        and for the decode shape ``GraphRunner._capture_graphs``; both are replicated minimally
        here rather than imported, because the scheduler's builders need a cache manager, live
        pages and a token pool that boot does not have -- and because a boot batch must redirect
        to the dummy row instead of a real request's.

        Returns ``None`` when this width has no capturable shape at boot.
        """
        pool = self.linear_state_pool
        slot = (
            self.dummy_req.linear_slot_idx
            if self.dummy_req.linear_slot_idx is not None
            else self.dummy_req.table_idx
        )
        if width == 1:
            batch = Batch(reqs=[self.dummy_req], phase="decode")
            batch.padded_reqs = batch.reqs
            if not self.graph_runner.can_use_cuda_graph(batch):
                # the width-1 graph stages its addressing through the backend's persistent
                # decode buffers, which exist only when the plain decode graphs were armed
                return None
            batch.input_ids = torch.zeros(1, dtype=torch.int32, device=self.device)
            batch.positions = torch.tensor(
                [self.dummy_req.cached_len], dtype=torch.int32, device=self.device
            )
            batch.out_loc = dummy_row[:1]  # the dedicated dummy KV slot, as padded replay uses
            batch.rope_positions = batch.positions.to(torch.int64).expand(3, -1).contiguous()
            if pool is not None:
                batch.linear_table_idx = torch.tensor(
                    [slot], dtype=torch.int32, device=self.device
                )
            # attn_metadata comes from the backend's own decode-capture seam, which the runner
            # calls for this width (prepare_for_capture) -- the same one GraphRunner uses.
            return batch

        warm_req = Req(
            input_ids=torch.zeros(base + width, dtype=torch.int32, device="cpu"),
            table_idx=self.dummy_req.table_idx,
            cached_len=base,  # extend_len == width, which the runner validates
            output_len=width,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=None,  # type: ignore[arg-type]
        )
        warm_req.linear_slot_idx = self.dummy_req.linear_slot_idx
        # Point the dummy row at the KV slots these rows write, exactly as _warmup_prefill does;
        # the caller restores the row (and with it padded decode replay's dummy slot) after.
        dummy_row[: base + width] = torch.arange(
            base + width, dtype=torch.int32, device=self.device
        )
        batch = Batch(reqs=[warm_req], phase="prefill")
        batch.padded_reqs = batch.reqs
        batch.mtp_verify = True
        batch.emit_width = width
        batch.input_ids = torch.zeros(width, dtype=torch.int32, device=self.device)
        batch.positions = torch.arange(
            base, base + width, dtype=torch.int32, device=self.device
        )
        batch.out_loc = dummy_row[base : base + width]
        # rope_positions stays None, as it does for a text request: the graph buffer expands
        # the scalar positions into its own three-axis slot.
        # The verify buffer refills its slot map every replay and so needs one even on a model
        # with no GDN pool, where the value is inert.
        batch.linear_table_idx = torch.tensor([slot], dtype=torch.int32, device=self.device)
        if pool is not None:
            from freetoken.attention.linear import build_fla_metadata

            batch.fla_metadata = build_fla_metadata(batch, self.device)
        self.attn_backend.prepare_metadata(batch)
        return batch

    def _capture_decode_graph(self, batch: Batch):
        """Replay the width-1 spec graph for an ordinary decode step, or None to run eagerly.

        Eligibility is deliberately "a batch the plain path would have graphed": that shape is
        the only one whose attention addressing the backend stages into the persistent decode
        buffers a capture can bake (``init_capture_graph``), and it is exactly the step the
        acceptance fallback made slow.
        """
        runner = self.spec_graph_runner
        if (
            runner is None
            or self.spec_draft is None
            or not batch.is_decode
            or batch.size != 1
            or batch.padded_size != 1
            or not self.graph_runner.can_use_cuda_graph(batch)
        ):
            return None
        ladder = self.spec_state_ladder
        if ladder is None or not runner.capture_pending(1):
            return runner.forward_decode(batch)
        # capture's warm-up EXECUTES this decode and advances the request's GDN slot; a plain
        # decode step takes no ladder snapshot of its own, so one is borrowed for the capture.
        with ladder.borrow_snapshot(batch.reqs[0]) as restore_state:
            return runner.forward_decode(batch, restore_state=restore_state)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        # Integrated speculation feeds its draft head the same (hidden, embeddings) pair the
        # observer captures, so a spec-enabled boot takes the capture path too -- but only for
        # the ordinary forwards it still runs (prefill, and the decode fallback); the
        # speculative step itself goes through speculative_decode_batch.
        wants_capture = self.mtp_shadow_observer is not None or self.spec_draft is not None
        mtp_capture = self._capture_decode_graph(batch) if wants_capture else None
        if mtp_capture is not None:
            logits = mtp_capture[0]
        else:
            use_graph = (not wants_capture) and self.graph_runner.can_use_cuda_graph(batch)
            with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, use_graph):
                if wants_capture:
                    # Capture is private; normal serving keeps its original graph path.
                    mtp_capture = self.model.forward_mtp_capture(all_row_logits=False)
                    logits = mtp_capture[0]
                elif use_graph:
                    logits = self.graph_runner.replay(batch)
                else:
                    logits = self.model.forward()
        if self.cpu_moe_executor is not None:
            # One pinned read: surfaces a fired flag-handshake watchdog (dead coordinator
            # -> stale expert outputs) as a loud error instead of silent corruption.
            self.cpu_moe_executor.raise_if_unhealthy()

        observer_capture = (
            self.mtp_shadow_observer.prepare_capture(batch, mtp_capture)
            if self.mtp_shadow_observer is not None
            else None
        )
        for req in batch.reqs:
            req.complete_one()

        batch_logits = logits[: batch.size]
        # Nested inside diag.prefill_forward, so prefill_forward's self time loses exactly
        # what these two take (freetoken/diag.py; default-off).
        with diag.region("diag.prefill_sample" if batch.is_prefill else None):
            next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)
        if self.mtp_shadow_observer is not None:
            self.mtp_shadow_observer.observe(observer_capture, next_tokens_gpu[0])
        if self.spec_draft is not None:
            # The MTP draft head's prompt priming: once per prompt (per chunk), eager.
            with diag.region("diag.prefill_prime_draft" if batch.is_prefill else None):
                self.spec_draft.observe_forward(batch, mtp_capture, next_tokens_gpu[0])
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def speculative_decode_batch(
        self,
        batch: Batch,
        args: BatchSamplingArgs,
        *,
        draft_tokens: Sequence[int],
        draft_logits: torch.Tensor,
        probe=None,
    ) -> SpecForwardOutput:
        """Forward one ``w = 1 + k`` row speculative batch and accept a prefix of the drafts.

        Three things separate this from ``forward_batch``, and each is load-bearing:

        * **all-row logits.** Acceptance needs row ``i``'s distribution for every draft, and
          the bonus token comes from row ``k``. The ordinary LM head slices a prefill batch to
          its last row (``layers/embedding.py``), so the step goes through
          ``forward_mtp_capture(all_row_logits=True)`` -- which also hands back the hidden
          rows the draft head consumes next cycle, in one pass.
        * **no ``complete_one``.** ``Scheduler._rollback_spec_tokens`` is the single authority
          on this step's lengths; advancing here would double-count every rejected row.
        * **the speculative sampler.** ``SpecSampler`` divides the target's filtered ``p`` by
          the draft's filtered ``q``; the plain sampler would just argmax row 0.

        ``probe`` is the scheduler's ``_SpecTimingProbe`` (anything with ``.mark(name)``), which
        subdivides its own "verify+accept" stage. It device-syncs at every mark, so it is a
        diagnosis tool only; ``None`` -- production -- must cost nothing. A probe also arms the
        graph runner's replay split, which needs ``.add_ms(name, ms)``: the marks time the
        forward as a whole, and only the runner can separate its host staging from the graph's
        device duration. A probe without ``add_ms`` simply gets no split.
        """
        assert self.device.type != "cuda" or torch.cuda.current_stream() == self.stream
        if self.spec_sampler is None:
            raise RuntimeError("speculative decode is not enabled on this engine")
        if not batch.mtp_verify or batch.size != 1:
            raise RuntimeError("a speculative step forwards one mtp_verify batch")
        captured = None
        graph_runner = self.spec_graph_runner
        armed = probe is not None and graph_runner is not None
        with diag.region("diag.spec_verify_replay"):
            if graph_runner is not None:
                if armed:
                    # the runner's own split of the replay, which the probe's marks cannot see:
                    # its host-side staging costs versus the graph's device duration
                    graph_runner.replay_timings = {}
                ladder = self.spec_state_ladder
                captured = graph_runner.forward(
                    batch,
                    # capture's warm-up EXECUTES this forward against the live GDN slot; the
                    # ladder's own pre-step snapshot is the wind-back (spec_graph,
                    # spec_state_ladder)
                    restore_state=None if ladder is None else ladder.restore_snapshot,
                )
            if captured is None:
                with self.ctx.forward_batch(batch):
                    logits, hidden, _ = self.model.forward_mtp_capture(all_row_logits=True)
            else:
                logits, hidden = captured
        probe and probe.mark("verify.forward")
        if armed:
            timings = graph_runner.replay_timings or {}
            # production stays disarmed even if the probe is dropped mid-run
            graph_runner.replay_timings = None
            add_ms = getattr(probe, "add_ms", None)
            if add_ms is not None:
                for key, name in _SPEC_REPLAY_TIMINGS:
                    value = timings.get(key)
                    if value is not None:
                        add_ms(name, value)
        if self.cpu_moe_executor is not None:
            self.cpu_moe_executor.raise_if_unhealthy()
        with diag.region("diag.spec_accept"):
            decision = self.spec_sampler.step(
                uid=batch.reqs[0].uid,
                draft_tokens=draft_tokens,
                draft_logits=draft_logits,
                target_logits=logits[: batch.emit_width],
                args=args,
            )
        probe and probe.mark("verify.accept")
        # Acceptance already cut the run out of tensors it held on the device, so the ids that
        # were produced there never leave and come back: the H2D this used to pay is a cat of
        # device slices instead. ``tokens_gpu`` is None only for a decision acceptance did not
        # build (a stubbed sampler in a test), and then the H2D is still the right answer.
        next_tokens_gpu = getattr(decision, "tokens_gpu", None)
        if next_tokens_gpu is None or next_tokens_gpu.device != logits.device:
            next_tokens_gpu = torch.tensor(
                decision.tokens, dtype=torch.int32, device=logits.device
            )
        probe and probe.mark("verify.pack")
        return SpecForwardOutput(decision, next_tokens_gpu, hidden)

    @torch.inference_mode()
    def _warmup_prefill(self) -> None:
        """Compile the Triton prefill path before the first real request.

        Decode CUDA graph capture warms the decode path, but the first prefill
        can still pay Triton/cublas setup costs. Use the dummy request row and
        restore it afterwards so padded decode graph replay keeps using the
        dedicated dummy KV slot.
        """
        if self.max_seq_len < 2:
            return

        warmup_lens = [min(80, self.max_seq_len)]
        if self.max_seq_len >= 128:
            warmup_lens.append(128)
        warmup_lens = sorted({length for length in warmup_lens if length >= 2})
        if not warmup_lens:
            return

        dummy_row = self.page_table[self.dummy_req.table_idx]
        dummy_slot = int(dummy_row[0].item())
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record(self.stream)
        try:
            for length in warmup_lens:
                dummy_row[:length] = torch.arange(
                    length, dtype=torch.int32, device=self.device
                )
                warm_req = Req(
                    input_ids=torch.zeros(length, dtype=torch.int32, device="cpu"),
                    table_idx=self.dummy_req.table_idx,
                    cached_len=0,
                    output_len=1,
                    uid=-1,
                    sampling_params=None,  # type: ignore[arg-type]
                    cache_handle=None,  # type: ignore[arg-type]
                )
                batch = Batch(reqs=[warm_req], phase="prefill")
                batch.padded_reqs = batch.reqs
                batch.input_ids = torch.zeros(length, dtype=torch.int32, device=self.device)
                batch.positions = torch.arange(length, dtype=torch.int32, device=self.device)
                batch.out_loc = dummy_row[:length]
                self.attn_backend.prepare_metadata(batch)
                with self.ctx.forward_batch(batch):
                    self.model.forward()
        finally:
            dummy_row.fill_(dummy_slot)
            if self.moe_offload_cache is not None:
                self.moe_offload_cache.reset()
        ended.record(self.stream)
        torch.cuda.synchronize(self.device)
        logger.info_rank0(
            f"Prefill warmup complete for lengths {warmup_lens} "
            f"in {started.elapsed_time(ended) / 1000.0:.3f} s"
        )

    def shutdown(self) -> None:
        if self.mtp_shadow_observer is not None:
            self.mtp_shadow_observer.close()
            self.mtp_shadow_observer = None
        if self.spec_draft is not None:
            self.spec_draft.close()
            self.spec_draft = None
        if self.spec_graph_runner is not None:
            self.spec_graph_runner.destroy()
            self.spec_graph_runner = None
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _profile_gpu(index: "int | None" = None) -> Tuple[str | None, str | None]:
    """(name, uuid) of visible device ``index`` (default: the current, i.e. bound, device); (None, None) without CUDA."""
    if not torch.cuda.is_available():
        return None, None
    ident = gpu_identity(torch.cuda.current_device() if index is None else index)
    return ident["name"], ident["uuid"]


def _ensure_expandable_segments() -> None:
    """Default the CUDA allocator to expandable segments.

    The motivating case is the offload prefill, which repeatedly dequantizes
    variable-sized NVFP4 expert blocks to BF16 (a different size per layer as the
    active-expert count varies). Under that alloc/free churn the default caching
    allocator fragments badly -- reserved memory can balloon far past the actual peak
    allocation (observed ~78GiB reserved for a <30GiB working set).
    ``expandable_segments`` lets freed regions of any size be reused, keeping
    reserved ~= allocated, so it is applied to every run, not just offload ones.

    Env vars are parsed once at import and ignored if set afterwards, so we apply the
    setting via the runtime API instead. Must run before the first CUDA allocation (the
    caller guarantees CUDA is not yet initialized). Any user-provided allocator config
    is respected and left untouched.
    """
    if os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except Exception as exc:  # pragma: no cover - depends on torch build
        logger.info_rank0(f"Could not enable expandable_segments ({exc}); continuing")
        return
    logger.info_rank0("Enabled expandable_segments (override via PYTORCH_ALLOC_CONF)")


def _resolve_cache_type(has_linear_attention: bool, requested: str) -> str:
    # Hybrid GDN models default to the HybridRadixCache (snapshots GDN state at chunk
    # boundaries -> cross-request prefix reuse). An explicit ``--cache-type naive`` opts out
    # to the old no-reuse path (debugging / parity baseline / lower GDN-state memory).
    if has_linear_attention:
        return "naive" if requested == "naive" else "hybrid_radix"
    return requested


def _adjust_dsv4_config(config: EngineConfig, override) -> None:
    """DSV4 engine-config reconciliation at config-resolution time (before the pool exists).
    Syncs the resolved runtime config into the opaque ``dsv4_args`` payload, sets
    page_size to the window page P, forces single-chunk prefill, and clamps cuda_graph_bs/max_bs to
    the DSV4 decode batch size.
    """
    model_config = config.model_config
    model_config.dsv4_args.max_seq_len = config.max_seq_len
    model_config.dsv4_args.max_batch_size = config.max_running_req + 1  # +1 dummy
    # config.swa_full_tokens_ratio is the DSV4 window/full ratio directly (default sizing);
    # a runtime rebuild pins an absolute window via swa_num_pages_override instead.
    # DSV4's KV page IS the P-token window page (window == radix reuse granularity == lcm of
    # the compress ratios), so max_num_tokens = num_pages * page_size holds like every model.
    P = model_config.dsv4_args.window_size
    override("page_size", P)
    logger.info_rank0(f"DSV4 KV pages are {P}-token window pages; page_size set to {P}")
    # The generic CacheManager materializes DSV4 'radix' as the shared SWARadixCache (is_swa);
    # 'naive' stays naive with the pool's swa currency riding swa_paged.
    if getattr(config, "cache_type", "radix") != "naive":
        override("cache_type", "swa_radix")
    # 'radix' (SWARadixCache on the full-loc currency, carry-aware re-prefill) is the default and is
    # honored, as is an explicit 'naive'. Don't let max_extend_tokens force a second chunk within
    # one prompt (the pool's prefill_chunk_budget still chunks prompts larger than the window
    # pool); prefill batches ragged (bs>=1), each segment resuming from its own cached_len.
    if getattr(config, "max_extend_tokens", 0) < config.max_seq_len:
        override("max_extend_tokens", config.max_seq_len)

    # DSV4 decode batches at most max_running_req rows; its full-loc snapshot is sized to that,
    # so a graph bs above it would exceed the backend's captured snapshot rows. Clamp any
    # oversized explicit list / max_bs here (before GraphRunner ever sees it).
    mr = config.max_running_req
    if config.cuda_graph_max_bs is not None and config.cuda_graph_max_bs > mr:
        logger.warning_rank0(
            f"cuda_graph_max_bs {config.cuda_graph_max_bs} exceeds DSV4 max_running_req {mr}; "
            "clamping to max_running_req (larger decode batches never occur)."
        )
        override("cuda_graph_max_bs", mr)
    if config.cuda_graph_bs is not None:
        kept = [bs for bs in config.cuda_graph_bs if bs <= mr]
        if kept != list(config.cuda_graph_bs):
            dropped = [bs for bs in config.cuda_graph_bs if bs > mr]
            logger.warning_rank0(
                f"dropping cuda_graph_bs entries {dropped} above DSV4 max_running_req {mr} "
                "(larger decode batches never occur)."
            )
            override("cuda_graph_bs", kept)


def _parse_cpu_layers_spec(spec: str, num_moe_layers: int) -> frozenset[int]:
    """Parse ``--moe-cpu-layers``: an explicit MoE-layer id list (``"3,7,11"``), a count
    (``"8"`` -> 8 layers evenly strided across depth), or a fraction (``"0.5"``). Ids are
    indices into the MoE layers, ``[0, num_moe_layers)``."""
    s = spec.strip()
    if not s:
        return frozenset()
    if "," in s:
        ids = {int(x) for x in s.split(",") if x.strip()}
        for i in ids:
            if not 0 <= i < num_moe_layers:
                raise ValueError(
                    f"--moe-cpu-layers id {i} out of range [0, {num_moe_layers})"
                )
        return frozenset(ids)
    if "." in s:
        frac = float(s)
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"--moe-cpu-layers fraction {frac} must be in [0, 1]")
        k = round(frac * num_moe_layers)
    else:
        k = int(s)
        if not 0 <= k <= num_moe_layers:
            raise ValueError(f"--moe-cpu-layers count {k} must be in [0, {num_moe_layers}]")
    # k layers spread evenly across depth (frozenset dedups any rounding collisions;
    # k == 0 yields an empty range, hence an empty set).
    return frozenset(round(i * num_moe_layers / k) for i in range(k))


def _resolve_cpu_layers(config: EngineConfig, num_moe_layers: int) -> frozenset[int]:
    """MoE layer ids whose decode runs on the CPU executor.

    ``--moe-backend cpu`` -> every layer. ``--moe-backend offload`` + ``--moe-cpu-layers``
    -> the parsed subset (the rest stay on the GPU offload/PCIe path). Otherwise none.
    """
    if config.moe_backend == "cpu":
        return frozenset(range(num_moe_layers))
    spec = config.moe_cpu_layers
    if not spec or not is_offload_moe_backend(config.moe_backend):
        return frozenset()
    return _parse_cpu_layers_spec(spec, num_moe_layers)


# MoE layers ranked by measured decode miss rate, hungriest first: the mean of
# per_layer[].miss_rate over the four cache_size=6750 decode captures in
# docs/research/routing-skew-2026-09-02/{code,prose,chat8k,toolcall}.json. The order is
# identical under mean missing_per_step and under the pooled union.json. A fixed constant,
# never a runtime heuristic; _auto_cpu_layers' U-shaped head+tail guess is NOT supported by
# this data (the tail 39-47 is mid-pack, the minimum is layer 31) and must not be reused.
GPU_OWNED_LAYER_RANK = (
    1, 6, 0, 2, 7, 22, 10, 13, 5, 18, 21, 38, 8, 12, 34, 11,
    24, 26, 29, 14, 28, 17, 19, 4, 35, 9, 37, 30, 33, 20, 3, 23,
    27, 25, 45, 36, 42, 47, 41, 46, 44, 40, 43, 39, 16, 32, 15, 31,
)


def _parse_gpu_owned_layers_spec(spec: str, num_moe_layers: int) -> frozenset[int]:
    """Parse ``--moe-gpu-owned-layers``: the ``--moe-cpu-layers`` grammar (explicit id list
    ``"0,1,2"``, count ``"6"``, fraction ``"0.125"``) plus ``"auto"`` (the six hungriest
    layers of :data:`GPU_OWNED_LAYER_RANK`) and ``"auto:N"`` (its first N)."""
    s = spec.strip()
    if s == "auto":
        s = "auto:6"
    if s.startswith("auto:"):
        try:
            n = int(s[len("auto:"):])
        except ValueError as exc:
            raise ValueError(f"--moe-gpu-owned-layers {spec!r}: 'auto:N' needs an integer N") from exc
        if not 0 <= n <= num_moe_layers:
            raise ValueError(
                f"--moe-gpu-owned-layers auto:{n} must be in [0, {num_moe_layers}]"
            )
        ranked = [i for i in GPU_OWNED_LAYER_RANK if i < num_moe_layers]
        return frozenset(ranked[:n])
    try:
        return _parse_cpu_layers_spec(s, num_moe_layers)
    except ValueError as exc:
        # reuse the grammar, not its error text: the operator typed a different flag
        raise ValueError(str(exc).replace("--moe-cpu-layers", "--moe-gpu-owned-layers")) from None


def _resolve_gpu_owned_layers(config: EngineConfig, num_moe_layers: int) -> frozenset[int]:
    """MoE layer ids whose experts are permanently VRAM-resident (no host bank).

    Only ``--moe-backend offload`` supports it: ``cpu``/``hybrid`` read every expert on the
    CPU (a VRAM-resident layer has no host bank to read), and ``fused`` keeps every expert
    resident already. Validation of the resolved set lives in
    :func:`_validate_gpu_owned_layers`.
    """
    spec = config.moe_gpu_owned_layers
    if not spec or config.moe_backend != "offload":
        return frozenset()
    return _parse_gpu_owned_layers_spec(spec, num_moe_layers)


def _validate_gpu_owned_layers(config: EngineConfig, num_moe_layers: int) -> frozenset[int]:
    """Resolve and fully validate the owned set, or raise. Returns the empty set when off."""
    from freetoken.checkpoint.ftw import is_ftw_checkpoint

    spec = config.moe_gpu_owned_layers
    if not spec:
        return frozenset()
    if config.moe_backend != "offload":
        raise ValueError(
            "--moe-gpu-owned-layers requires --moe-backend offload (got "
            f"{config.moe_backend!r}): a VRAM-resident layer has no host bank for the CPU "
            "executor to read, and 'fused' keeps every expert resident already"
        )
    owned = _parse_gpu_owned_layers_spec(spec, num_moe_layers)
    if not owned:
        return owned
    cpu_layer_ids = _resolve_cpu_layers(config, num_moe_layers)
    clash = sorted(owned & cpu_layer_ids)
    if clash:
        raise ValueError(
            f"--moe-gpu-owned-layers and --moe-cpu-layers name layers that are "
            f"both GPU-owned and CPU layers: {clash}"
        )
    if config.model_path and is_ftw_checkpoint(config.model_path):
        raise ValueError(
            "--moe-gpu-owned-layers is not supported on an FTW packed checkpoint "
            "(load_ftw_banks always allocates one host bank per layer); serve the original "
            "checkpoint or drop the flag"
        )
    # --moe-cache-size is the TOTAL expert-slot budget: the owned layers are charged to it
    # (see _charge_gpu_owned_layers_to_cache_size), so what has to clear the prefill-overlap
    # floor is what is LEFT for the streaming layers, not the number the operator typed.
    if config.moe_cache_size and not getattr(config, "moe_cache_auto", False):
        _gpu_owned_lru_slots(config, len(owned))
    return owned


def _gpu_owned_lru_slots(config: EngineConfig, owned_layers: int) -> int:
    """LRU slots left once ``owned_layers`` are charged to an explicit ``--moe-cache-size``."""
    from freetoken.engine.cache_budget import lru_slots_after_owned_charge

    num_experts = config.model_config.num_experts
    return lru_slots_after_owned_charge(
        moe_cache_size=config.moe_cache_size,
        owned_layers=owned_layers,
        num_experts=num_experts,
        floor=2 * num_experts if config.moe_prefill_overlap else num_experts,
    )


def _gpu_owned_boot_line(
    owned: frozenset[int],
    num_moe_layers: int,
    num_experts: int,
    per_expert_bytes: int,
    cache_size: int,
) -> str:
    """The single boot line reporting what --moe-gpu-owned-layers actually did (spec 3)."""
    layer_gib = num_experts * per_expert_bytes / 2**30
    return (
        f"MoE GPU-owned layers: {sorted(owned)} ({len(owned)} x {layer_gib:.2f} GiB "
        f"resident, no host bank); LRU cache {cache_size} slots for "
        f"{num_moe_layers - len(owned)} streaming layers"
    )


# expert activations the CPU MoE executor supports (csrc ActKind)
_CPU_MOE_ACTS = (
    "silu", "swish", "gelu", "gelu_tanh", "gelu_pytorch_tanh", "swigluoai",
    "swiglu_clamp",
)


def _cpu_moe_executor_viable(model_config) -> bool:
    """Whether an automatic CPU-decode decision may target the CPU MoE executor.

    A default boot must degrade to GPU offload instead of crashing in CpuMoeExecutor after the whole load; explicit cpu/hybrid/--moe-cpu-layers picks still fail loudly."""
    from freetoken.moe.cpu_executor import _WFMT_IDS, compiled_extension_supports

    try:
        from freetoken.kernel import _cpu_moe  # noqa: F401
    except ImportError:
        return False
    act = getattr(model_config, "hidden_act", "silu")
    moe_wfmt = getattr(model_config, "moe_weight_format", None)
    if act not in _CPU_MOE_ACTS and moe_wfmt != "mxfp4":
        return False
    if moe_wfmt != "mxfp4" and not compiled_extension_supports(act):
        return False
    expert_quant = getattr(model_config, "expert_quant", "none")
    fmt = expert_quant if expert_quant != "none" else (moe_wfmt or "bf16")
    return fmt == "mxfp4" or fmt in _WFMT_IDS


def _env_flag(name: str) -> bool:
    """A FREETOKEN_* boolean knob, in the spelling the rest of the MoE path accepts."""
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _pin_budget_bytes(reserved: int = 0) -> int | None:
    """Bytes this process can still safely cudaHostRegister, or None when the platform does not cap pinning (plain Linux).

    WSL's WDDM-backed CUDA caps pinning near half of RAM, shared across processes -- budget 40%. FREETOKEN_PIN_BUDGET_GB overrides anywhere. ``reserved`` subtracts host bytes already pinned outside the expert banks (qwen4_exp's PLE table)."""
    if env := os.environ.get("FREETOKEN_PIN_BUDGET_GB"):
        cap = int(float(env) * 2**30)
    elif not hasattr(os, "uname") or "microsoft" not in os.uname().release.lower():  # WSL kernel tag
        return None
    else:
        cap = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.4)
    return max(0, cap - reserved)


def _auto_cpu_layers(
    config: EngineConfig, num_moe_layers: int, reserved: int = 0, gpu_owned: int = 0
) -> frozenset[int]:
    """Pick CPU (locked) MoE layers automatically when the banks exceed the pin budget.

    Locks just enough head+tail layers: per-layer decode miss rates are U-shaped, so the ends are the cheapest to move off the slot cache."""
    from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

    bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(
        config.model_config, gpu_owned=gpu_owned
    )
    if not bank_bytes:
        return frozenset()
    budget = _pin_budget_bytes(reserved)
    if budget is None or bank_bytes <= budget:
        return frozenset()
    if not _cpu_moe_executor_viable(config.model_config):
        logger.info_rank0(
            f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB exceed the "
            f"pin budget {budget / 2**30:.2f} GiB, but the CPU MoE executor cannot "
            f"serve this model; keeping every layer pinned on the GPU offload path"
        )
        return frozenset()
    n = min(num_moe_layers, math.ceil(num_moe_layers * (1 - budget / bank_bytes)))
    head = (n + 1) // 2
    ids = frozenset(range(head)) | frozenset(range(num_moe_layers - (n - head), num_moe_layers))
    logger.info_rank0(
        f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB > pin budget "
        f"{budget / 2**30:.2f} GiB; locking {n} head+tail MoE layers for CPU decode "
        f"({sorted(ids)})"
    )
    return ids


# MoE-only knobs and the value each resolves to on a dense model. moe_backend is handled
# separately (its dense value is 'fused', but 'auto' resolves there without a warning).
_DENSE_MOE_SETTINGS = {
    "moe_cache_size": 0,
    "moe_cache_rate": None,
    "moe_cache_auto": False,
    "moe_cpu_layers": None,
    "moe_gpu_owned_layers": None,
    "moe_cpu_threads": 0,
    "moe_hybrid_max_fetch": -1,
    "moe_prefill_overlap": True,
    "moe_prefill_hit_d2d": False,
    "moe_collect_decode_freq": False,
    "expert_load": "auto",
}


def _validate_ple_backend(config: EngineConfig, model_config) -> None:
    ple_backend = getattr(config, "ple_backend", "pinned")
    if ple_backend not in ("pinned", "mmap", "disk"):
        raise ValueError(f"ple_backend must be 'pinned', 'mmap' or 'disk', got {ple_backend!r}")
    if ple_backend == "pinned":
        return

    qwen4_args = getattr(model_config, "qwen4_args", None)
    has_ple_layers = qwen4_args is not None and bool(getattr(qwen4_args, "ple_layer_ids", ()))

    if ple_backend == "mmap" and not has_ple_layers:
        raise ValueError(
            "--ple-backend mmap is currently supported only for "
            "Qwen3.8-Flash-Next checkpoints with a PLE table"
        )
    if ple_backend == "disk" and sys.platform != "linux" and has_ple_layers:
        raise ValueError(
            "--ple-backend disk needs the Linux io_uring row store; "
            "use --ple-backend mmap on Windows"
        )

    if ple_backend == "disk" and (
        os.environ.get("FREETOKEN_MTP_SHADOW", "0") == "1"
        or os.environ.get("FREETOKEN_MTP_SPECULATE", "0") == "1"
    ):
        raise ValueError(
            "the MTP capture path does not enter forward_host_ctx; "
            "use --ple-backend mmap or pinned with MTP"
        )


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    model_config = config.model_config
    single_stream_only = getattr(model_config, "single_stream_only", False)
    is_dsv4 = getattr(model_config, "dsv4_args", None) is not None
    has_swa_attention = getattr(model_config, "has_swa_attention", False)
    has_linear_attention = getattr(model_config, "has_linear_attention", False)
    is_moe = getattr(model_config, "is_moe", False)
    expert_quant = getattr(model_config, "expert_quant", "none")

    if not is_moe:
        # A dense model has no routed experts: the MoE knobs are inert, and the offload family
        # is worse than inert -- engine init would build an expert cache for a model that has
        # none and abort startup (weights already resident) on an unrelated expert-source
        # error. Drop them at this one choke point, which the CLI and the programmatic
        # LLM(...) path both pass through. 'auto'/'fused' is the silent dense resolution;
        # anything else was asked for explicitly, so report what is being ignored.
        dropped = [
            f"{name}={getattr(config, name)!r}"
            for name, dense_value in _DENSE_MOE_SETTINGS.items()
            if getattr(config, name, dense_value) != dense_value
        ]
        if config.moe_backend not in ("auto", "fused"):
            dropped.insert(0, f"moe_backend={config.moe_backend!r}")
        override("moe_backend", "fused")
        for name, dense_value in _DENSE_MOE_SETTINGS.items():
            override(name, dense_value)
        if dropped:
            logger.warning_rank0(
                f"{getattr(model_config, 'model_type', 'model')} is a dense model (no routed "
                f"experts); ignoring MoE settings: {', '.join(dropped)}"
            )

    if single_stream_only:
        # The model runs one sequence at a time: it collapses the batch to one row and the
        # decode CUDA graph is captured at bs=1. Force the runtime knobs so the KV pool, page
        # table and graph capture all stay bs=1.
        if config.max_running_req != 1:
            override("max_running_req", 1)
        if config.cuda_graph_max_bs is None or config.cuda_graph_max_bs >= 1:
            override("cuda_graph_bs", [1])
            override("cuda_graph_max_bs", 1)

    _validate_ple_backend(config, model_config)
    if config.cuda_graph_max_bs is None:
        override("cuda_graph_max_bs", config.max_running_req)

    if is_dsv4:
        _adjust_dsv4_config(config, override)

    if has_swa_attention:
        # Both SWA cache paths use the global-paged swa pool (page_size==1 only for now).
        if config.page_size != 1:
            raise ValueError(
                f"SWA models currently support only page_size=1, got {config.page_size}."
            )
        # naive keeps cache_type='naive' (NaivePrefixCache, no reuse) on the paged pool (==
        # sglang SWAChunkCache); radix materializes as swa_radix (SWARadixCache, cross-request
        # reuse == sglang SWARadixCache). Both allocate from the same swa pool + free out-of-window.
        if getattr(config, "cache_type", "radix") != "naive":
            if not 0.0 < config.swa_full_tokens_ratio <= 1.0:
                raise ValueError(
                    f"swa_full_tokens_ratio must be in (0, 1], got {config.swa_full_tokens_ratio}"
                )
            override("cache_type", "swa_radix")

    if has_linear_attention:
        override(
            "cache_type",
            _resolve_cache_type(True, getattr(config, "cache_type", "radix")),
        )

    # Type x backend capability matrix: resolve auto from the per-type priority
    # lists, then validate whatever is now selected (explicit or auto) -- every
    # comma part must serve every required type, with packages/arch available.
    required_attn_types = _required_attn_types(model_config)
    _dtype = getattr(config, "dtype", None)  # duck-typed test configs omit it
    if (
        required_attn_types & {AttnType.BSA, AttnType.QSA}
        and _dtype is not None
        and _dtype.itemsize != 2
    ):
        # Reject at config time: the BSA/QSA pool's own assert only fires after the
        # model is resident (and not at all under `python -O`).
        raise ValueError(
            f"--dtype {config.dtype}: block-sparse attention serves 16-bit "
            "compute only (the index slab budgets 2 bytes/token); use bfloat16 "
            "or float16."
        )
    if _dtype == torch.float16 and "mxfp8" in (
        getattr(model_config, "attn_quant", "none"),
        getattr(model_config, "dense_quant", "none"),
    ):
        # The MXFP8 GEMV folds the pow2-descaled fp8 weight into the activation
        # dtype; fp16's narrow exponent can overflow/flush what bf16 represents
        # exactly, and the combination was never numerically validated.
        raise ValueError(
            "--dtype float16 with MXFP8 resident weights is unsupported (the "
            "W8A16 fold is only validated exact in bfloat16); use bfloat16."
        )
    if _dtype == torch.float16 and "int8" in (
        getattr(model_config, "attn_quant", "none"),
        getattr(model_config, "dense_quant", "none"),
        getattr(model_config, "lm_head_quant", "none"),
    ):
        # The int8 W8A16 GEMM feeds tl.dot bf16 operands whatever the model dtype, so an
        # fp16 prefill (M > 1) rounds its activations to bf16 while the M == 1 GEMV stays
        # exact in fp32 -- prefill and decode disagree -- and the fp32-tiny scale floor in
        # quantize_int8_rows underflows to 0.0 in fp16.
        raise ValueError(
            "--dtype float16 with int8 dense weights (FREETOKEN_DENSE_QUANT=int8) is "
            "unsupported: the W8A16 kernel computes in bfloat16; use bfloat16."
        )
    if config.attention_backend == "auto":
        override(
            "attention_backend",
            _resolve_auto_attention_backend(required_attn_types),
        )
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")
    _validate_attention_backend_choice(config, override, required_attn_types)

    if config.moe_cache_rate is not None:
        total_experts = config.model_config.num_moe_layers * config.model_config.num_experts
        override("moe_cache_size", math.ceil(total_experts * config.moe_cache_rate))

    # The CPU MoE executor supports the silu/gelu family plus the clamped
    # swigluoai (csrc ActKind; "gpt_oss_swiglu" rides inside the mxfp4 kernel and
    # swigluoai the generic GEMV epilogue). A model with any other expert
    # activation cannot decode on the CPU: reject an explicit cpu/hybrid pick at
    # config time, and keep auto from upgrading offload -> hybrid off the profile.
    # hidden_act (the dense activation) stands proxy for the expert activation --
    # true for every in-tree model. mxfp4 experts pass regardless: their act runs
    # inside the mxfp4 kernel, not the generic epilogue.
    _cpu_moe_act_ok = getattr(model_config, "hidden_act", "silu") in _CPU_MOE_ACTS or (
        getattr(model_config, "moe_weight_format", None) == "mxfp4"
    )
    if (
        is_moe
        and not _cpu_moe_act_ok
        and (config.moe_backend in ("cpu", "hybrid") or config.moe_cpu_layers)
    ):
        asked = (
            f"--moe-cpu-layers={config.moe_cpu_layers!r}"
            if config.moe_backend not in ("cpu", "hybrid")
            else f"--moe-backend {config.moe_backend!r}"
        )
        raise ValueError(
            f"{asked}: the CPU MoE executor does not support this model's expert "
            f"activation {getattr(model_config, 'hidden_act', None)!r}; drop the flag "
            "and let every layer decode on the GPU offload path instead."
        )

    if is_moe and config.moe_backend == "auto":
        # A MoE model always defaults to the offload family: experts stream from pinned host
        # banks into an auto-sized GPU slot cache, which is the only default that serves a model
        # bigger than the GPU. The resident 'fused' path (bf16 / block-fp8 experts, the two
        # formats MoELayer can allocate) is still reachable, but only when asked for explicitly
        # -- auto never picks it, because nothing here knows whether the experts would fit in
        # HBM and a wrong guess is a weight-load OOM rather than a slower-but-working run.
        default_backend = "offload"
        # Hardware-adaptive config: a cached `ft bench bw` profile can upgrade
        # the offload default to hybrid when this machine's CPU MoE bandwidth clears its PCIe
        # gather bandwidth by the bench threshold (default 2x). hybrid is VRAM-equivalent to
        # offload -- same auto-sized GPU slot cache (_resolve_auto_moe_cache_size), plus a
        # host-RAM CPU executor -- so this never raises the OOM risk; with no profile (or one
        # from different hardware) it stays offload. offload remains the always-safe fallback.
        # Key the lookup on the real expert format: mxfp4/q4_0 live in moe_weight_format when
        # expert_quant is "none", and "none" with no weight format means plain bf16 experts.
        moe_wfmt = getattr(model_config, "moe_weight_format", None)
        bench_fmt = expert_quant if expert_quant != "none" else (moe_wfmt or "bf16")
        from freetoken.moe.bench_profile import load_backend_recommendation

        gpu_name, gpu_uuid = _profile_gpu()
        if load_backend_recommendation(bench_fmt, gpu_name=gpu_name, gpu_uuid=gpu_uuid) == "hybrid":
            from freetoken.moe.cpu_executor import compiled_extension_supports

            _act = getattr(model_config, "hidden_act", "silu")
            if not _cpu_moe_act_ok:
                logger.info_rank0(
                    f"benchbw profile recommends hybrid, but the CPU MoE executor does not "
                    f"support this model's expert activation "
                    f"{getattr(model_config, 'hidden_act', None)!r}; staying on offload"
                )
            elif moe_wfmt != "mxfp4" and not compiled_extension_supports(_act):
                # Stale prebuilt _cpu_moe.so: an explicit cpu/hybrid pick still
                # hard-fails in the executor, but a default must not turn into a
                # post-load crash -- degrade to offload.
                logger.info_rank0(
                    f"benchbw profile recommends hybrid, but the compiled _cpu_moe "
                    f"extension predates activation {_act!r} (rebuild with "
                    f"`python setup.py build_ext --inplace`); staying on offload"
                )
            else:
                default_backend = "hybrid"
                logger.info_rank0(
                    f"benchbw profile recommends hybrid for {bench_fmt!r} experts on this GPU"
                )
        override("moe_backend", default_backend)
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

        if (
            is_offload_moe_backend(config.moe_backend)
            and config.moe_cache_size <= 0
            and config.moe_cache_rate is None
            and not getattr(config, "moe_cache_auto", False)
        ):
            # args.py's "no sizing flag -> default --moe-cache-auto" only fires when the
            # backend is already offload-family at *parse* time. A bare `ft serve <FTW MoE
            # checkpoint>` (no --moe-backend, no cache flags) still has moe_backend=="auto" at
            # parse time -- the auto -> offload/cpu/hybrid resolution above is the first point
            # the concrete backend is known, so mirror the same default here: no sizing flag
            # was given, so let the scheduler resolve the cache size from free VRAM instead of
            # failing the _require_offload_cache_size guard with size=0.
            override("moe_cache_auto", True)
            logger.info_rank0(
                "No MoE cache sizing flag given; defaulting to --moe-cache-auto for "
                f"auto-selected backend {config.moe_backend!r}"
            )

    if is_moe and config.moe_backend == "fused":
        # An explicit 'fused' keeps the experts resident, so there is no slot cache to size. The
        # sizing flags no longer redirect the backend, so ignore them here and say so -- the
        # geometry the user asked for is what runs. Report the flag actually passed: --moe-cache-
        # rate was already folded into moe_cache_size above, and the three are mutually exclusive.
        if config.moe_cache_rate:
            inert = f"--moe-cache-rate={config.moe_cache_rate}"
        elif config.moe_cache_size:
            inert = f"--moe-cache-size={config.moe_cache_size}"
        elif getattr(config, "moe_cache_auto", False):
            inert = "--moe-cache-auto"
        else:
            inert = None
        if inert:
            logger.warning_rank0(
                f"MoE backend 'fused' keeps its experts resident; ignoring {inert} "
                "(use --moe-backend offload to serve experts from a slot cache)"
            )
            override("moe_cache_size", 0)
            override("moe_cache_rate", None)
            override("moe_cache_auto", False)

    if is_moe and config.moe_backend == "cpu":
        # CPU-compute decode keeps experts in host RAM and computes them on the CPU;
        # the GPU only holds the two-layer prefill double buffer. So the slot cache is
        # fixed at exactly two expert layers (prefill overlap requires >= 2*num_experts)
        # and --moe-cache-size / --moe-cache-auto / --moe-cache-rate do not apply.
        num_experts = config.model_config.num_experts
        if getattr(config, "moe_cache_auto", False):
            override("moe_cache_auto", False)
        override("moe_cache_size", 2 * num_experts)
        override("moe_prefill_overlap", True)
        logger.info_rank0(
            f"MoE backend 'cpu': decode computes experts on CPU; GPU keeps a "
            f"two-layer prefill buffer (moe_cache_size={2 * num_experts})"
        )

    if (
        is_moe
        and expert_quant not in ("none", "fp8_block")
        and not is_offload_moe_backend(config.moe_backend)
    ):
        raise ValueError(
            f"{expert_quant} experts require --moe-backend offload or cpu, "
            f"got {config.moe_backend!r}"
        )

    if is_moe and config.moe_cpu_layers and config.moe_backend not in ("offload", "hybrid"):
        # the layer split needs the offload host banks + slot cache; 'cpu' already runs every layer on CPU, 'fused' keeps experts resident on the GPU (no host banks)
        raise ValueError(
            "--moe-cpu-layers requires --moe-backend offload or hybrid (got "
            f"{config.moe_backend!r}); use --moe-backend cpu to run all layers on CPU"
        )

    if is_moe and getattr(config, "moe_gpu_owned_layers", None):
        # resolved once here so a bad spec fails before any weight is read; the engine
        # re-resolves at cache build (the backend may still be 'auto' at parse time).
        # Gated on the spec (which _validate_gpu_owned_layers checks first anyway) so a
        # partial stub model_config without num_moe_layers is untouched when the flag is off.
        _validate_gpu_owned_layers(config, model_config.num_moe_layers)

    if is_moe:
        object.__setattr__(model_config, "moe_backend", config.moe_backend)
    object.__setattr__(model_config, "nvfp4_backend", config.nvfp4_backend)

    # Must stay LAST: page_size is only final here (_adjust_dsv4_config sets P=128, the
    # TRTLLM block sets 64). Also covers the programmatic LLM(...) path that bypasses parse_args.
    if config.num_token_override is not None:
        if config.num_page_override is not None:
            raise ValueError("--num-tokens and --num-pages are mutually exclusive")
        if config.num_token_override % config.page_size != 0:
            raise ValueError(
                f"--num-tokens {config.num_token_override} is not a multiple of the resolved "
                f"page size {config.page_size}; nearest valid values: "
                f"{config.num_token_override // config.page_size * config.page_size} or "
                f"{(config.num_token_override // config.page_size + 1) * config.page_size}"
            )
        override("num_page_override", config.num_token_override // config.page_size)

    # The rope cos/sin table is baked to rotary_config.max_position, and neither rope kernel
    # bounds-checks the position it gathers with -- a longer ceiling reads past the table.
    # DSV4 is exempt: it sizes its own table from the resolved max_seq_len (_adjust_dsv4_config).
    rotary = getattr(model_config, "rotary_config", None)
    seq_override = getattr(config, "max_seq_len_override", None)
    if seq_override is not None and rotary is not None and not is_dsv4:
        if seq_override > rotary.max_position:
            raise ValueError(
                f"--max-seq-len-override {seq_override} exceeds the model's "
                f"rope table ({rotary.max_position} positions). Serving past it would read "
                "out of bounds; extend the checkpoint's rope_scaling / "
                "max_position_embeddings in config.json instead."
            )

    # The startup ServerArgs dump is the *requested* config, printed in the frontend process
    # before any of the resolution above ran -- so "moe_backend='auto'" is all it can say. This
    # is the one line that reports what actually runs, for every path (explicit backends never
    # hit an "Auto-selected ..." log at all).
    resolved = [
        f"attention_backend={config.attention_backend!r}",
        f"cache_type={getattr(config, 'cache_type', 'radix')!r}",
        f"page_size={config.page_size}",
    ]
    if is_moe:
        resolved.insert(0, f"moe_backend={config.moe_backend!r}")
    logger.info_rank0(f"Resolved config: {', '.join(resolved)}")
