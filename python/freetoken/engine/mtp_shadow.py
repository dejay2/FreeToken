from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from freetoken.attention.qsa_sparse import QSASparseAttnBackend
from freetoken.core import Batch, Context, Req
from freetoken.kvcache import create_kvcache_pool, resolve_pool_class
from freetoken.engine.mtp_fast_verify import (
    MTPFastVerifier,
    MTPGraphCaptureResult,
    MTPProjectionSample,
    MTPVerifyForwardResult,
    MTPVerifyGraphRunner,
    batched_speculative_accept,
    compare_verifier_distributions,
    compare_verifier_logits,
    sampling_distribution_divergence,
    sampling_support_order_matches,
)
from freetoken.models.qwen4_exp.mtp_spike import (
    MTPBF16ExpertBanks,
    MTPDraftSampler,
    MTPExactExpertRunner,
    MTPGPUExpertRunner,
    MTPNVFP4ExpertBanks,
    MTPNVFP4ExpertRunner,
    MTPStagedModelRunner,
    MTPWeightStore,
    Qwen4ExpMTPModel,
    build_mtp_weight_plan,
    derive_mtp_model_config,
)
from freetoken.engine.spec_draft import build_shifted_pairs, build_shifted_rope_positions
from freetoken.utils.torch_utils import torch_dtype

_GUARD_BYTES = 128 << 20
# Excess filtered (temperature/top-k/top-p) TV between the oracle and fast checker:
# only the component beyond-noise logit moves are responsible for (see
# sampling_distribution_divergence's noise_tolerance). Plain filtered TV is unusable
# here — sharpening amplifies within-noise moves without bound (a greedy filter turns
# a knife-edge tie flip into 1.0). Observed live noise scores <= 0.02 in excess terms
# while the subtlest structural error lands at 0.08 and a wrong token at 1.0.
_MAX_SAMPLING_DIVERGENCE = 0.05


@dataclass(frozen=True)
class MTPShadowConfig:
    enabled: bool
    placement: str = "bf16"
    private_root: Path | None = None
    depth: int = 3
    seed: int = 1729
    cpu_threads: int = 8
    verify_mode: str = "oracle"
    resident: bool = False
    target_pages: int = 4097
    target_experts: int = 4063

    @classmethod
    def from_env(cls, engine_config) -> "MTPShadowConfig":
        raw = os.getenv("FREETOKEN_MTP_SHADOW", "0").strip()
        if raw not in {"0", "1"}:
            raise ValueError("FREETOKEN_MTP_SHADOW must be 0 or 1")
        if raw == "0":
            return cls(enabled=False)
        verify_mode = os.getenv("FREETOKEN_MTP_VERIFY_MODE", "oracle").strip().lower()
        if verify_mode not in {"oracle", "compare", "fast-eager", "fast-graph"}:
            raise ValueError(
                "FREETOKEN_MTP_VERIFY_MODE must be oracle, compare, fast-eager, or "
                "fast-graph"
            )
        placement = os.getenv("FREETOKEN_MTP_EXPERT_FORMAT", "bf16").strip().lower()
        if placement not in {"bf16", "nvfp4"}:
            raise ValueError("FREETOKEN_MTP_EXPERT_FORMAT must be bf16 or nvfp4")
        raw_root = os.getenv("FREETOKEN_MTP_PRIVATE_ROOT", "").strip()
        if not raw_root:
            raise ValueError(
                "FREETOKEN_MTP_PRIVATE_ROOT must point at the private MTP spike root "
                "(evidence/, prototypes/, weights/)"
            )
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise ValueError(f"MTP private root {root} is not a directory")
        depth = int(os.getenv("FREETOKEN_MTP_DEPTH", "3"))
        if depth not in {1, 2, 3}:
            raise ValueError("FREETOKEN_MTP_DEPTH must be 1, 2, or 3")
        cpu_threads = int(os.getenv("FREETOKEN_MTP_CPU_THREADS", "8"))
        if cpu_threads < 1:
            raise ValueError("FREETOKEN_MTP_CPU_THREADS must be positive")
        raw_resident = os.getenv("FREETOKEN_MTP_RESIDENT", "0").strip()
        if raw_resident not in {"0", "1"}:
            raise ValueError("FREETOKEN_MTP_RESIDENT must be 0 or 1")
        target_pages = int(os.getenv("FREETOKEN_MTP_TARGET_PAGES", "4097"))
        target_experts = int(os.getenv("FREETOKEN_MTP_TARGET_EXPERTS", "4063"))
        if target_pages < 1:
            raise ValueError("FREETOKEN_MTP_TARGET_PAGES must be positive")
        if target_experts < 1:
            raise ValueError("FREETOKEN_MTP_TARGET_EXPERTS must be positive")
        if engine_config.model_config.model_type != "qwen4_exp":
            raise ValueError("MTP shadow supports only the private Qwen3.8 target")
        if engine_config.max_running_req != 1:
            raise ValueError("MTP shadow requires exactly one active request")
        # qsa_sparse resolves the generic CLI page size to 64 later in Engine startup;
        # the observer validates the fully-resolved page geometry before allocation.
        if engine_config.max_seq_len != 262_144:
            raise ValueError("MTP shadow requires exactly 262144 target tokens")
        return cls(
            enabled=True,
            placement=placement,
            private_root=root,
            depth=depth,
            seed=int(os.getenv("FREETOKEN_MTP_DRAFT_SEED", "1729")),
            cpu_threads=cpu_threads,
            verify_mode=verify_mode,
            resident=raw_resident == "1",
            target_pages=target_pages,
            target_experts=target_experts,
        )


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def sampling_probabilities(
    logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    if temperature <= 0 or top_k == 1:
        probabilities = torch.zeros_like(logits, dtype=torch.float32)
        probabilities[int(torch.argmax(logits))] = 1.0
        return probabilities
    filtered = logits.float() / float(temperature)
    if 1 <= top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, -float("inf"))
    probabilities = torch.softmax(filtered, dim=-1)
    if top_p < 1:
        ordered, indices = probabilities.sort(descending=True)
        remove = ordered.cumsum(0) - ordered >= top_p
        ordered.masked_fill_(remove, 0)
        probabilities = torch.zeros_like(probabilities).scatter(0, indices, ordered)
        probabilities /= probabilities.sum()
    return probabilities


def speculative_accept(
    draft_token: int,
    draft_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[bool, int, float]:
    q = float(draft_probabilities[draft_token])
    p = float(target_probabilities[draft_token])
    ratio = 1.0 if q <= 0 else min(1.0, p / q)
    draw = float(torch.rand((), generator=generator, device=draft_probabilities.device))
    if draw <= ratio:
        return True, int(draft_token), ratio
    residual = (target_probabilities - draft_probabilities).clamp_min(0)
    if float(residual.sum()) == 0:
        residual = target_probabilities
    residual = residual / residual.sum()
    corrected = int(torch.multinomial(residual, 1, generator=generator))
    return False, corrected, ratio


def greedy_acceptance(
    proposals: list[int], target_logits: torch.Tensor
) -> tuple[int, int]:
    depth = len(proposals)
    if target_logits.ndim != 2 or target_logits.shape[0] != depth + 1:
        raise ValueError("target verifier must return depth+1 logit rows")
    targets = torch.argmax(target_logits[:depth], dim=-1).tolist()
    accepted = 0
    for draft, target in zip(proposals, targets):
        if draft != target:
            break
        accepted += 1
    corrected = (
        int(torch.argmax(target_logits[depth]))
        if accepted == depth
        else int(targets[accepted])
    )
    return accepted, corrected


# The shifted-pair builders live in ``spec_draft`` -- the integrated decode path needs them
# and must not import the observer to get them. Re-exported here so the observer's own
# import surface is unchanged (the same shape ``mtp_fast_verify`` uses for the acceptance
# core it handed to ``spec_sample``).


@dataclass
class MTPTargetCapture:
    uid: int
    is_chunked: bool
    cached_len: int
    device_len: int
    table_idx: int
    linear_slot_idx: int
    protected_linear_slots: tuple[int, ...]
    input_ids_cpu: torch.Tensor
    multi_stream_cpu: torch.Tensor
    inputs_embeds_cpu: torch.Tensor
    rope_positions_cpu: torch.Tensor | None
    mrope_position_delta: int
    temperature: float
    top_k: int
    top_p: float


class MTPShadowObserver:
    """In-process, append-only native MTP observer. It never changes target/client output."""

    def __init__(self, engine, config: MTPShadowConfig) -> None:
        self.engine = engine
        self.config = config
        assert config.private_root is not None, "private_root must be set when enabled=True"
        self.device = engine.device
        self.target_ctx = engine.ctx
        self.target_model = engine.model
        self.mtp_config = derive_mtp_model_config(engine.config.model_config)
        self.trace_path = config.private_root / "evidence" / f"live-{config.placement}.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._weight_store: MTPWeightStore | None = None
        self._expert_banks = None
        self._pending_hidden_cpu: torch.Tensor | None = None
        self._pending_rope_cpu: torch.Tensor | None = None
        self._uid: int | None = None
        self.committed_len = 0
        self.cache_manager = None
        self.graph_verifier: MTPVerifyGraphRunner | None = None
        if engine.moe_offload_cache is not None:
            # movement reconciliation in the verify trace needs the offload-cache counters, and
            # --moe-collect-stats defaults off; the observer's own forwards are eager or captured
            # into its private graph after this, so they pick the accumulation ops up
            engine.moe_offload_cache.collect_stats = True
        self.fast_verifier = MTPFastVerifier(
            target_ctx=self.target_ctx,
            target_model=self.target_model,
            moe_offload_cache=engine.moe_offload_cache,
            device=self.device,
        )
        self._init_private_rng()
        self._init_target_expert_temperature()

        # The page count and expert-cache size follow the boot's KV reservation; the QSA page
        # size is a model property and stays pinned.
        if engine.num_pages != config.target_pages or engine.config.page_size != 64:
            raise RuntimeError("MTP shadow refuses target KV geometry drift")
        cache_size = getattr(engine.moe_offload_cache, "cache_size", None)
        if cache_size != config.target_experts:
            raise RuntimeError(
                f"MTP shadow requires {config.target_experts} target experts, "
                f"got {cache_size}"
            )
        slots = self._usable_linear_slots(engine.linear_state_pool)
        if slots != 8:
            raise RuntimeError(f"MTP shadow requires eight target recurrent slots, got {slots}")

        with torch.device("meta"), torch_dtype(torch.bfloat16):
            self.model = Qwen4ExpMTPModel(self.mtp_config)
        model_state = self.model.state_dict()
        with MTPWeightStore(engine.config.model_path) as store:
            plan = build_mtp_weight_plan(store.keys, model_state.keys())
            cpu_weights = {
                entry.model_name: store.materialize(entry, device=torch.device("cpu"))
                for entry in plan.entries
                if not entry.expert
            }
        predicted_pool = self._predicted_pool_bytes()
        # Read the budget before anything resident lands on the device, so the plan and the
        # measurement describe the same starting point.
        free = torch.cuda.mem_get_info(self.device)[0]
        self.staged_model = MTPStagedModelRunner(
            self.model, cpu_weights, device=self.device, resident=config.resident
        )
        resident_bytes = 0
        if config.resident:
            self._expert_banks = self._load_expert_banks()
            resident_bytes = self.staged_model.resident_bytes + int(
                self._expert_banks.total_bytes
            )
        self.memory_plan = self._build_memory_plan(
            free=free,
            qsa_bytes=predicted_pool,
            staging_bytes=self.staged_model.staging_bytes,
            resident_bytes=resident_bytes,
        )
        required = self.memory_plan["required_bytes"]
        if free < required:
            raise RuntimeError(
                f"MTP shadow needs {required} free bytes including guard, only {free} available"
            )

        self._init_private_state()
        self._init_experts()
        self.model.layers.op_list[0].mlp.experts.attach_runner(self.expert_runner)
        self._write_event({"event": "observer_ready", "memory_plan": self.memory_plan})

    @staticmethod
    def _usable_linear_slots(pool) -> int:
        if pool is None:
            return 0
        for name in ("num_slots", "_num_slots"):
            value = getattr(pool, name, None)
            if value is not None:
                value = int(value() if callable(value) else value)
                return value - 1
        state = getattr(pool, "state", None)
        if isinstance(state, torch.Tensor):
            return int(state.shape[0]) - 1
        return -1

    def _predicted_pool_bytes(self) -> int:
        pool_cls = resolve_pool_class(self.mtp_config)
        fake = SimpleNamespace(
            model_config=self.mtp_config,
            page_size=64,
            max_running_req=1,
            tp_info=self.engine.config.tp_info,
            dtype=torch.bfloat16,
        )
        per_page, fixed, _, _ = pool_cls.kv_cost(fake)
        ratio = self.mtp_config.qwen4_args.index_ratio
        default_ring = 4
        private_ring = ratio * math.ceil((ratio + self.config.depth) / ratio)
        extra_rows = 2 * (private_ring - default_ring)
        extra = extra_rows * (
            self.mtp_config.qwen4_args.index_head_dim * torch.bfloat16.itemsize
            + 3 * torch.int64.itemsize
        )
        return 4097 * per_page + fixed + extra

    def _init_private_state(self) -> None:
        width = 4097 * 64
        self.page_table = torch.zeros((2, width), dtype=torch.int32, device=self.device)
        self.page_table[0] = torch.arange(width, dtype=torch.int32, device=self.device)
        self.page_table[1].fill_(width - 64)
        self.kv_cache = create_kvcache_pool(
            model_config=self.mtp_config,
            num_pages=4097,
            page_size=64,
            dtype=torch.bfloat16,
            device=self.device,
            num_req_slots=2,
            num_speculative_tokens=self.config.depth,
        )
        self.kv_cache.attach_page_table(self.page_table)
        with self._private_context_fields():
            self.attn_backend = QSASparseAttnBackend(self.mtp_config)

    @staticmethod
    def _build_memory_plan(
        *, free: int, qsa_bytes: int, staging_bytes: int, resident_bytes: int
    ) -> dict:
        required = qsa_bytes + staging_bytes + resident_bytes + _GUARD_BYTES
        plan = {
            "free_before": int(free),
            "mtp_qsa_bytes": int(qsa_bytes),
            "max_dense_stage_bytes": int(staging_bytes),
            "guard_bytes": _GUARD_BYTES,
            "required_bytes": int(required),
        }
        if resident_bytes:
            plan["resident_bytes"] = int(resident_bytes)
        return plan

    def _load_expert_banks(self):
        if self.config.placement == "bf16":
            if self._weight_store is None:
                self._weight_store = MTPWeightStore(self.engine.config.model_path)
                self._weight_store.__enter__()
            return MTPBF16ExpertBanks.from_store(self._weight_store)
        if self.config.resident:
            raise ValueError(
                "FREETOKEN_MTP_RESIDENT supports only the bf16 expert placement"
            )
        manifest = self.config.private_root / "weights" / "mtp-experts-nvfp4" / "manifest.json"
        return MTPNVFP4ExpertBanks.from_manifest(manifest, validate_hashes=True)

    def _expert_runner_type(self):
        if self.config.resident:
            return MTPGPUExpertRunner
        if self.config.placement == "bf16":
            return MTPExactExpertRunner
        return MTPNVFP4ExpertRunner

    def _init_experts(self) -> None:
        banks = self._expert_banks
        if banks is None:
            banks = self._load_expert_banks()
        runner_type = self._expert_runner_type()
        self.expert_runner = runner_type(
            banks,
            top_k=self.mtp_config.num_experts_per_tok,
            activation=self.mtp_config.hidden_act,
            renormalize=self.mtp_config.norm_topk_prob,
            max_tokens=128,
            num_threads=self.config.cpu_threads,
            device=self.device,
        )
        if self.config.resident:
            # The runner copied the banks to the device and kept only their geometry;
            # this reference is the last thing pinning the ~5 GB host copy.
            self._expert_banks = None

    @contextmanager
    def _private_context_fields(self):
        ctx = self.target_ctx
        old = (
            getattr(ctx, "kv_cache", None),
            getattr(ctx, "page_table", None),
            getattr(ctx, "attn_backend", None),
        )
        try:
            if hasattr(self, "kv_cache"):
                ctx.kv_cache = self.kv_cache
            if hasattr(self, "page_table"):
                ctx.page_table = self.page_table
            yield
        finally:
            ctx.kv_cache, ctx.page_table, ctx.attn_backend = old

    @contextmanager
    def _private_forward(self, batch):
        ctx = self.target_ctx
        old = (ctx.kv_cache, ctx.page_table, ctx.attn_backend)
        assert ctx._batch is None, "target context is still in a forward"
        try:
            ctx.kv_cache = self.kv_cache
            ctx.page_table = self.page_table
            ctx.attn_backend = self.attn_backend
            with ctx.forward_batch(batch):
                yield
        finally:
            ctx.kv_cache, ctx.page_table, ctx.attn_backend = old

    def prepare_capture(self, batch, capture) -> MTPTargetCapture:
        if len(batch.reqs) != 1:
            raise RuntimeError("MTP shadow observed more than one target request")
        req = batch.reqs[0]
        if self._uid is None and req.cached_len != 0:
            raise RuntimeError("MTP shadow requests must start at target cached length zero")
        params = req.sampling_params
        # Spill promptly so an 8192-row target capture does not remain resident beside staging.
        multi = capture[1].detach().to("cpu")
        inputs = capture[2].detach().to("cpu")
        if req.linear_slot_idx is None:
            raise RuntimeError("MTP verifier requires a live target recurrent slot")
        return MTPTargetCapture(
            uid=int(req.uid),
            is_chunked=type(req).__name__ == "ChunkedReq",
            cached_len=int(req.cached_len),
            device_len=int(req.device_len),
            table_idx=int(req.table_idx),
            linear_slot_idx=int(req.linear_slot_idx),
            protected_linear_slots=(
                int(req.linear_slot_idx),
                *(int(slot) for slot in (req.mamba_ping_pong or ())),
            ),
            input_ids_cpu=self._complete_token_snapshot(batch, req),
            multi_stream_cpu=multi,
            inputs_embeds_cpu=inputs,
            rope_positions_cpu=(
                None
                if getattr(batch, "rope_positions", None) is None
                else batch.rope_positions.detach().to("cpu")
            ),
            mrope_position_delta=int(getattr(req, "mrope_position_delta", 0)),
            temperature=float(getattr(params, "temperature", 0.0)),
            top_k=int(getattr(params, "top_k", -1)),
            top_p=float(getattr(params, "top_p", 1.0)),
        )

    @staticmethod
    def _complete_token_snapshot(batch, req) -> torch.Tensor:
        """The ``device_len`` tokens behind this forward, as CPU ids.

        Under overlap scheduling the engine calls ``prepare_capture`` before ``complete_one``,
        and decode batch k launches before ``append_host`` lands batch k-1's token, so from the
        first decode on ``req.input_ids`` is short of ``req.device_len`` by exactly the tokens
        this batch is feeding. Those are ``batch.input_ids``; take them back to CPU rather than
        letting a later slice truncate the snapshot silently.
        """
        ids = req.input_ids.detach().clone()
        missing = int(req.device_len) - int(ids.numel())
        if missing > 0:
            tail = batch.input_ids[:missing].detach().to("cpu", dtype=ids.dtype)
            ids = torch.cat((ids, tail))
        if ids.numel() != int(req.device_len):
            raise RuntimeError(
                f"MTP capture token snapshot holds {ids.numel()} ids for device_len "
                f"{int(req.device_len)}"
            )
        return ids

    def bind_cache_manager(self, cache_manager) -> None:
        if self.cache_manager is not None:
            raise RuntimeError("MTP observer cache manager is already bound")
        self.cache_manager = cache_manager
        if self.config.verify_mode in {"compare", "fast-graph"}:
            self.graph_verifier = MTPVerifyGraphRunner(
                target_ctx=self.target_ctx,
                target_model=self.target_model,
                attn_backend=self.engine.attn_backend,
                moe_cache=self.engine.moe_offload_cache,
                device=self.device,
                vocab_size=self.engine.config.model_config.vocab_size,
                guard_bytes=_GUARD_BYTES,
            )

    def _init_private_rng(self) -> None:
        self._draft_sampler: MTPDraftSampler | None = None
        self._acceptance_generators: dict[int, torch.Generator] = {}
        self._rng_request_uid: int | None = None
        self._draft_draw_count = 0
        self._acceptance_draw_counts: dict[int, int] = {}
        self._rng_cycle_index = 0
        self._acceptance_cycle_indices: dict[int, int] = {}

    def _private_rng_seed(self, uid: int, *, purpose: int, depth: int = 0) -> int:
        modulus = (1 << 63) - 1
        return int(
            (
                int(self.config.seed)
                + int(uid) * 10_000_019
                + purpose * 1_000_003
                + depth * 10_009
            )
            % modulus
        )

    def _reset_private_rng(self, uid: int) -> None:
        uid = int(uid)
        self._draft_sampler = MTPDraftSampler(
            seed=self._private_rng_seed(uid, purpose=1),
            device=self.device,
        )
        self._acceptance_generators = {}
        for depth in range(1, int(self.config.depth) + 1):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(
                self._private_rng_seed(uid, purpose=2, depth=depth)
            )
            self._acceptance_generators[depth] = generator
        self._rng_request_uid = uid
        self._draft_draw_count = 0
        self._acceptance_draw_counts = {
            depth: 0 for depth in self._acceptance_generators
        }
        self._rng_cycle_index = 0
        self._acceptance_cycle_indices = {
            depth: 0 for depth in self._acceptance_generators
        }

    def _default_rng_states(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        cpu = torch.get_rng_state().clone()
        cuda = (
            torch.cuda.get_rng_state(self.device).clone()
            if self.device.type == "cuda"
            else None
        )
        return cpu, cuda

    def _restore_default_rng_states(
        self,
        states: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        cpu, cuda = states
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, self.device)

    def _assert_default_rng_unchanged(
        self,
        before: tuple[torch.Tensor, torch.Tensor | None],
    ) -> dict:
        after = self._default_rng_states()
        cpu_same = torch.equal(before[0], after[0])
        cuda_same = (
            before[1] is None
            if after[1] is None
            else before[1] is not None and torch.equal(before[1], after[1])
        )
        if not cpu_same or not cuda_same:
            self._restore_default_rng_states(before)
            raise RuntimeError("MTP checker changed target/global RNG state")
        return {
            "default_rng_unchanged": True,
            "default_cpu_rng_sha256_before": tensor_sha256(before[0]),
            "default_cpu_rng_sha256_after": tensor_sha256(after[0]),
            "default_cuda_rng_sha256_before": (
                None if before[1] is None else tensor_sha256(before[1])
            ),
            "default_cuda_rng_sha256_after": (
                None if after[1] is None else tensor_sha256(after[1])
            ),
        }

    def _rng_stream_snapshot(self, stream: str, *, depth: int | None = None) -> dict:
        if stream == "draft":
            if depth is not None or self._draft_sampler is None:
                raise ValueError("draft RNG snapshot does not accept a depth")
            generator = self._draft_sampler.generator
            draw_count = self._draft_draw_count
            cycle_index = self._rng_cycle_index
            stream_id = "draft"
        elif stream == "acceptance":
            if depth not in self._acceptance_generators:
                raise ValueError(f"unknown MTP acceptance RNG depth {depth}")
            generator = self._acceptance_generators[depth]
            draw_count = self._acceptance_draw_counts[depth]
            cycle_index = self._acceptance_cycle_indices[depth]
            stream_id = f"acceptance-depth-{depth}"
        else:
            raise ValueError(f"unknown private RNG stream {stream!r}")
        return {
            "stream": stream_id,
            "request_uid": self._rng_request_uid,
            "cycle_index": int(cycle_index),
            "draw_count": int(draw_count),
            "state_sha256": tensor_sha256(generator.get_state()),
        }

    @staticmethod
    def _rng_transition(before: dict, after: dict, *, default_rng: dict) -> dict:
        if before["stream"] != after["stream"]:
            raise RuntimeError("private RNG transition changed streams")
        if before["request_uid"] != after["request_uid"]:
            raise RuntimeError("private RNG transition crossed a request boundary")
        if before["cycle_index"] != after["cycle_index"]:
            raise RuntimeError("private RNG transition crossed a checker cycle")
        return {
            "stream": before["stream"],
            "request_uid": before["request_uid"],
            "cycle_index": before["cycle_index"],
            "draw_count_before": before["draw_count"],
            "draw_count_after": after["draw_count"],
            "draws": after["draw_count"] - before["draw_count"],
            "state_sha256_before": before["state_sha256"],
            "state_sha256_after": after["state_sha256"],
            **default_rng,
        }

    def _sample_draft(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> tuple[int, dict]:
        if self._rng_request_uid is None or self._draft_sampler is None:
            raise RuntimeError("MTP private RNG has no active request")
        instrumentation_started = time.perf_counter()
        before = self._rng_stream_snapshot("draft")
        private_state_before = self._draft_sampler.generator.get_state()
        default_before = self._default_rng_states()
        instrumentation_wall_ms = (
            time.perf_counter() - instrumentation_started
        ) * 1000.0
        required_started = time.perf_counter()
        try:
            token = self._draft_sampler.sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            required_wall_ms = (
                time.perf_counter() - required_started
            ) * 1000.0
            instrumentation_started = time.perf_counter()
            default_rng = self._assert_default_rng_unchanged(default_before)
        except BaseException:
            self._draft_sampler.generator.set_state(private_state_before)
            self._assert_default_rng_unchanged(default_before)
            raise
        # EVERY draft costs exactly one uniform now, greedy included. ``MTPDraftSampler`` is
        # branch-free on the host so the draft chain can be a CUDA graph (see its docstring):
        # the greedy id is still bit-exact ``argmax``, but it is SELECTED from the same fixed
        # sequence of kernels the sampled draw runs, and that sequence contains the draw. The
        # accounting therefore counts calls rather than kinds, and the invariant this observer
        # enforces is the one that survived: a draft always advances its private stream, and
        # never the default one.
        greedy = temperature == 0 or top_k == 1
        self._draft_draw_count += 1
        after = self._rng_stream_snapshot("draft")
        if before["state_sha256"] == after["state_sha256"]:
            raise RuntimeError("MTP draft did not advance its private RNG")
        transition = self._rng_transition(
            before,
            after,
            default_rng=default_rng,
        )
        transition["greedy"] = greedy
        instrumentation_wall_ms += (
            time.perf_counter() - instrumentation_started
        ) * 1000.0
        self._last_draft_timing = {
            "required_wall_ms": required_wall_ms,
            "required_synchronizations": 0,
            "instrumentation_wall_ms": instrumentation_wall_ms,
            "instrumentation_synchronizations": 0,
        }
        return token, transition

    @staticmethod
    def _proposal_component_times(
        *,
        request_cycle_index: int,
        prompt_setup_ms: float,
        update_ms: float,
        draft_step_ms: list[float],
        recursive_step_ms: list[float],
        cleanup_ms: float,
        graph_capture_setup_ms: float,
        depth: int,
        setup_ms: float = 0.0,
    ) -> dict[str, float]:
        if request_cycle_index < 0:
            raise ValueError("request cycle index must be non-negative")
        if depth < 1 or depth > len(draft_step_ms):
            raise ValueError("proposal timing depth is out of range")
        if len(recursive_step_ms) < depth - 1:
            raise ValueError("proposal timing is missing recursive steps")
        values = [
            prompt_setup_ms,
            update_ms,
            cleanup_ms,
            graph_capture_setup_ms,
            setup_ms,
            *draft_step_ms[:depth],
            *recursive_step_ms[: depth - 1],
        ]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("proposal timing values must be finite and non-negative")
        if request_cycle_index and graph_capture_setup_ms:
            raise ValueError("graph capture setup appeared on a later request cycle")
        prompt_ms = (
            prompt_setup_ms + update_ms + graph_capture_setup_ms
            if request_cycle_index == 0
            else 0.0
        )
        draft_ms = setup_ms + sum(draft_step_ms[:depth]) + sum(
            recursive_step_ms[: depth - 1]
        ) + cleanup_ms
        if request_cycle_index:
            draft_ms += update_ms
        return {"P": prompt_ms, "D": draft_ms}

    def _init_target_expert_temperature(self) -> None:
        self._target_expert_request_index = 0
        self._target_expert_temperature: dict | None = None
        self._target_expert_request_observed = False

    def _assign_target_expert_temperature(self, uid: int) -> dict:
        uid = int(uid)
        current = self._target_expert_temperature
        if current is not None:
            if current["request_uid"] == uid:
                return dict(current)
            if not self._target_expert_request_observed:
                if current["state"] == "cold":
                    raise RuntimeError(
                        "cold request must be followed by warm checker preparation"
                    )
                raise RuntimeError(
                    "target expert temperature request ended without a checker sample"
                )
        request_index = self._target_expert_request_index
        assigned = {
            "state": "cold" if request_index % 2 == 0 else "warm",
            "pair_index": request_index // 2,
            "request_uid": uid,
        }
        self._target_expert_request_index += 1
        self._target_expert_temperature = assigned
        self._target_expert_request_observed = False
        return dict(assigned)

    def _prepare_target_expert_temperature(self, uid: int) -> dict:
        current = self._target_expert_temperature
        if current is None or current["request_uid"] != int(uid):
            raise RuntimeError("target expert temperature request order is invalid")
        started = time.perf_counter()
        residency_reset = current["state"] == "cold"
        if residency_reset:
            self.engine.moe_offload_cache.reset()
        reset_setup_ms = (time.perf_counter() - started) * 1000.0
        self._target_expert_request_observed = True
        return {
            **current,
            "residency_reset": residency_reset,
            "reset_setup_ms": reset_setup_ms,
        }

    def _reset_request(self, uid: int) -> None:
        self._uid = uid
        self._assign_target_expert_temperature(uid)
        self._reset_private_rng(uid)
        self._pending_hidden_cpu = None
        self._pending_rope_cpu = None
        self.committed_len = 0
        self.kv_cache._kv_buffer.zero_()
        self.kv_cache._cmp_k_buffer.zero_()
        self.kv_cache._pending_ring.zero_()
        self.kv_cache._pending_position_ring.zero_()
        self._request_prompt_setup_ms = 0.0

    def _batch(self, rows: int, *, rope_positions=None, capture=None, saved=None):
        start = self.committed_len
        req = SimpleNamespace(
            table_idx=0, cached_len=start, device_len=start + rows, extend_len=rows
        )
        positions = torch.arange(start, start + rows, dtype=torch.int32, device=self.device)
        batch = SimpleNamespace(
            reqs=[req], padded_reqs=[req], phase="decode" if rows == 1 else "prefill",
            size=1, padded_size=1, is_prefill=rows != 1, is_decode=rows == 1,
            positions=positions, rope_positions=rope_positions,
            out_loc=self.page_table[0, start : start + rows].contiguous(),
            attn_metadata=None,
            active_table_idx=torch.zeros(1, dtype=torch.int32, device=self.device),
        )
        if capture is not None:
            batch.mtp_qsa_capture_blocks = capture
        if saved is not None:
            batch.mtp_qsa_saved_blocks = saved
        self.attn_backend.prepare_metadata(batch)
        return batch

    def _run_rows(
        self, embeddings_cpu, hidden_cpu, *, rope_positions_cpu=None, capture=None, saved=None
    ):
        outputs = None
        for lo in range(0, embeddings_cpu.shape[0], 128):
            hi = min(lo + 128, embeddings_cpu.shape[0])
            is_last = hi == embeddings_cpu.shape[0]
            local_capture = capture if is_last else None
            rope_positions = (
                None
                if rope_positions_cpu is None
                else rope_positions_cpu[:, lo:hi].to(self.device)
            )
            batch = self._batch(
                hi - lo,
                rope_positions=rope_positions,
                capture=local_capture,
                saved=saved,
            )
            embeddings = embeddings_cpu[lo:hi].to(self.device)
            hidden = hidden_cpu[lo:hi].to(self.device)
            with self._private_forward(batch):
                outputs = self.staged_model.forward(embeddings, hidden, batch)
            self.committed_len += hi - lo
        assert outputs is not None
        return outputs[0][-1:], outputs[1][-1:]

    @staticmethod
    def _target_page_index(
        physical_token_base: int,
        *,
        page_size: int,
        num_pages: int,
    ) -> int:
        physical_token_base = int(physical_token_base)
        page_size = int(page_size)
        num_pages = int(num_pages)
        if page_size < 1:
            raise ValueError("target K/V page size must be positive")
        if physical_token_base < 0 or physical_token_base % page_size:
            raise ValueError(
                "target K/V physical token base must be non-negative and page-aligned"
            )
        page_index = physical_token_base // page_size
        if page_index >= num_pages:
            raise IndexError("target K/V physical token base is out of range")
        return page_index

    @classmethod
    def _target_kv_page(
        cls,
        tensor: torch.Tensor,
        physical_token_base: int,
        *,
        page_size: int,
    ) -> torch.Tensor:
        if tensor.ndim < 2 or int(tensor.shape[1]) != int(page_size):
            raise ValueError(
                "target K/V tensor must use [pages, page_size, ...] layout"
            )
        page_index = cls._target_page_index(
            physical_token_base,
            page_size=page_size,
            num_pages=int(tensor.shape[0]),
        )
        return tensor[page_index]

    @staticmethod
    def _state_tensor_digest(tensor: torch.Tensor) -> str:
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _combine_state_family_digests(families: dict[str, str]) -> str:
        digest = hashlib.sha256()
        for name in sorted(families):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(families[name]))
        return digest.hexdigest()

    def _target_state_family_digests(
        self, captured: MTPTargetCapture
    ) -> dict[str, str]:
        tensor_digest = self._state_tensor_digest
        base = captured.device_len
        families = {
            "request-metadata": hashlib.sha256(
                json.dumps(
                    {
                        "uid": captured.uid,
                        "cached_len": captured.cached_len,
                        "device_len": captured.device_len,
                        "table_idx": captured.table_idx,
                        "linear_slot_idx": captured.linear_slot_idx,
                        "protected_linear_slots": captured.protected_linear_slots,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "page-table-prefix": tensor_digest(
                self.engine.page_table[captured.table_idx, :base]
            ),
        }
        pool = self.engine.linear_state_pool
        families["linear-conv"] = tensor_digest(
            pool.conv_states[:, captured.linear_slot_idx]
        )
        families["linear-recurrent"] = tensor_digest(
            pool.recurrent_states[:, captured.linear_slot_idx]
        )
        for name in sorted(pool.slot_states):
            families[f"linear-extra:{name}"] = tensor_digest(
                pool.slot_states[name][:, captured.linear_slot_idx]
            )
        target_kv = self.engine.kv_cache
        backend = self.engine.attn_backend
        for slot in sorted(set(backend._idx_slot.values())):
            families[f"qsa-ring:{slot}"] = tensor_digest(
                target_kv.pending_ring(slot)[captured.table_idx]
            )
            families[f"qsa-position-ring:{slot}"] = tensor_digest(
                target_kv.pending_position_ring(slot)[captured.table_idx]
            )
        if base:
            page_size = int(self.cache_manager.page_size)
            ratio = int(backend.ratio)
            if page_size % ratio:
                raise ValueError("target K/V page size must be divisible by QSA ratio")
            page_start = (base - 1) // page_size * page_size
            physical = int(self.engine.page_table[captured.table_idx, page_start].item())
            rows_per_page = page_size // ratio
            for layer_id in sorted(backend._idx_slot):
                k_page = self._target_kv_page(
                    target_kv.k_cache(layer_id),
                    physical,
                    page_size=page_size,
                )
                v_page = self._target_kv_page(
                    target_kv.v_cache(layer_id),
                    physical,
                    page_size=page_size,
                )
                families[f"target-k:{layer_id}"] = tensor_digest(k_page)
                families[f"target-v:{layer_id}"] = tensor_digest(v_page)
                slot = backend._idx_slot[layer_id]
                page_index = physical // page_size
                compressed_start = page_index * rows_per_page
                compressed_end = compressed_start + rows_per_page
                compressed = target_kv.cmp_k_cache(slot)
                if compressed_end > int(compressed.shape[0]):
                    raise IndexError("target QSA compressed page is out of range")
                families[f"qsa-compressed:{slot}"] = tensor_digest(
                    compressed[compressed_start:compressed_end]
                )
        return families

    def _target_state_digest(self, captured: MTPTargetCapture) -> str:
        return self._combine_state_family_digests(
            self._target_state_family_digests(captured)
        )

    @staticmethod
    def _linear_slot_snapshot(pool, slot: int) -> dict[str, torch.Tensor]:
        snapshot = {
            "conv": pool.conv_states[:, slot].detach().cpu().clone(),
            "recurrent": pool.recurrent_states[:, slot].detach().cpu().clone(),
        }
        for name, tensor in pool.slot_states.items():
            snapshot[f"extra:{name}"] = tensor[:, slot].detach().cpu().clone()
        return snapshot

    @staticmethod
    def _linear_snapshot_digest(snapshot: dict[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        names = ["conv", "recurrent"] + sorted(
            name for name in snapshot if name.startswith("extra:")
        )
        for name in names:
            digest.update(snapshot[name].contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _linear_slot_digest(pool, slot: int) -> str:
        digest = hashlib.sha256()
        tensors = [pool.conv_states[:, slot], pool.recurrent_states[:, slot]]
        tensors.extend(pool.slot_states[name][:, slot] for name in sorted(pool.slot_states))
        for tensor in tensors:
            digest.update(
                tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
        return digest.hexdigest()

    @staticmethod
    def _restore_linear_slot(pool, slot: int, snapshot: dict[str, torch.Tensor]) -> None:
        pool.conv_states[:, slot].copy_(snapshot["conv"])
        pool.recurrent_states[:, slot].copy_(snapshot["recurrent"])
        for name, tensor in pool.slot_states.items():
            tensor[:, slot].copy_(snapshot[f"extra:{name}"])

    def _copy_partial_target_page(self, src: int, dst: int) -> None:
        target_kv = self.engine.kv_cache
        backend = self.engine.attn_backend
        page_size = int(self.cache_manager.page_size)
        ratio = int(backend.ratio)
        if page_size % ratio:
            raise ValueError("target K/V page size must be divisible by QSA ratio")
        rows_per_page = page_size // ratio
        for layer_id in sorted(backend._idx_slot):
            src_k = self._target_kv_page(
                target_kv.k_cache(layer_id), src, page_size=page_size
            )
            dst_k = self._target_kv_page(
                target_kv.k_cache(layer_id), dst, page_size=page_size
            )
            src_v = self._target_kv_page(
                target_kv.v_cache(layer_id), src, page_size=page_size
            )
            dst_v = self._target_kv_page(
                target_kv.v_cache(layer_id), dst, page_size=page_size
            )
            dst_k.copy_(src_k)
            dst_v.copy_(src_v)

            slot = backend._idx_slot[layer_id]
            compressed = target_kv.cmp_k_cache(slot)
            src_start = (src // page_size) * rows_per_page
            dst_start = (dst // page_size) * rows_per_page
            src_end = src_start + rows_per_page
            dst_end = dst_start + rows_per_page
            if max(src_end, dst_end) > int(compressed.shape[0]):
                raise IndexError("target QSA compressed page is out of range")
            compressed[dst_start:dst_end].copy_(compressed[src_start:src_end])

    def _target_verify_batch(
        self,
        captured: MTPTargetCapture,
        candidate_ids: list[int],
        *,
        dummy_table_idx: int,
        shadow_slot: int,
    ) -> Batch:
        full_ids = torch.cat(
            (
                captured.input_ids_cpu[: captured.device_len],
                torch.tensor(candidate_ids, dtype=torch.int32),
            )
        )
        req = Req(
            input_ids=full_ids,
            table_idx=dummy_table_idx,
            cached_len=captured.device_len,
            output_len=0,
            uid=-captured.uid - 10_000,
            sampling_params=SimpleNamespace(),
            cache_handle=None,
        )
        if req.extend_len != len(candidate_ids):
            raise RuntimeError(
                f"MTP verify request extend_len {req.extend_len} != {len(candidate_ids)} "
                "candidate rows; the captured token snapshot is short"
            )
        req.linear_slot_idx = shadow_slot
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        batch.input_ids = torch.tensor(candidate_ids, dtype=torch.int32, device=self.device)
        batch.positions = torch.arange(
            captured.device_len,
            captured.device_len + len(candidate_ids),
            dtype=torch.int32,
            device=self.device,
        )
        batch.rope_positions = (
            None
            if captured.rope_positions_cpu is None
            else (
                batch.positions.to(torch.int64) + captured.mrope_position_delta
            ).expand(3, -1)
        )
        batch.out_loc = self.engine.page_table[
            dummy_table_idx,
            captured.device_len : captured.device_len + len(candidate_ids),
        ].contiguous()
        batch.mm_embeds = None
        batch.linear_table_idx = torch.tensor(
            [shadow_slot], dtype=torch.int32, device=self.device
        )
        from freetoken.attention.linear import build_fla_metadata

        batch.fla_metadata = build_fla_metadata(batch, self.device)
        self.engine.attn_backend.prepare_metadata(batch)
        return batch

    def _forward_target_oracle(self, batch) -> MTPVerifyForwardResult:
        use_cuda = self.device.type == "cuda"
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
        if logits.ndim != 2 or logits.shape[0] != batch.input_ids.shape[0]:
            raise RuntimeError("MTP oracle returned the wrong number of logit rows")
        finite = torch.isfinite(logits).all()
        instrumentation_synchronizations = 0
        if use_cuda:
            torch.cuda.synchronize(self.device)
            instrumentation_synchronizations = 1
        if not bool(finite):
            raise RuntimeError("MTP oracle returned non-finite logits")
        instrumentation_wall_ms = (
            time.perf_counter() - instrumentation_started
        ) * 1000.0
        return MTPVerifyForwardResult(
            mode="oracle",
            logits=logits,
            required_wall_ms=required_wall_ms,
            core_cuda_ms=core_cuda_ms,
            required_synchronizations=required_synchronizations,
            instrumentation_wall_ms=instrumentation_wall_ms,
            instrumentation_synchronizations=instrumentation_synchronizations,
            expert_movement={
                "available": False,
                "layer_calls": 0,
                "active_experts": 0,
                "hit_experts": 0,
                "missing_experts": 0,
                "fetched_experts": 0,
                "cpu_experts": 0,
                "d2d_rows": 0,
                "bytes_per_expert": 0,
                "h2d_bytes": 0,
                "d2d_bytes": 0,
                "transfer_bytes": 0,
                "movement_reconciled": False,
            },
        )

    @staticmethod
    def _assert_out_loc_owned_by_lease(
        out_loc: torch.Tensor,
        leased_pages: torch.Tensor,
        *,
        page_size: int,
    ) -> None:
        if out_loc.ndim != 1 or leased_pages.ndim != 1:
            raise ValueError("MTP scratch ownership expects one-dimensional tensors")
        page_size = int(page_size)
        if page_size < 1:
            raise ValueError("MTP scratch ownership page size must be positive")
        out_pages = torch.div(
            out_loc.to(torch.int64), page_size, rounding_mode="floor"
        ) * page_size
        owned = (out_pages[:, None] == leased_pages.to(torch.int64)[None, :]).any(dim=1)
        if not bool(owned.all().item()):
            raise RuntimeError("MTP candidate out_loc is outside its scratch-page lease")

    def _transaction_fault(self, stage: str) -> None:
        hook = getattr(self, "_transaction_fault_hook", None)
        if hook is not None:
            hook(stage)

    def _run_target_transaction(
        self,
        captured: MTPTargetCapture,
        candidate_ids: list[int],
        *,
        forward,
        mode: str,
    ) -> dict:
        if self.cache_manager is None:
            raise RuntimeError("MTP verifier is not bound to CacheManager")
        if len(candidate_ids) not in (2, 3, 4):
            raise ValueError("MTP target transaction requires 2, 3, or 4 rows")

        page_size = int(self.cache_manager.page_size)
        base = captured.device_len
        end = base + len(candidate_ids)
        first_page = base // page_size
        last_page = (end + page_size - 1) // page_size
        needed_pages = last_page - first_page
        dummy = self.engine.dummy_req.table_idx
        if dummy == captured.table_idx:
            raise RuntimeError(
                "MTP verifier cannot use the live request page-table row as scratch"
            )
        page_start = first_page * page_size
        row_end = last_page * page_size
        target_kv = self.engine.kv_cache
        backend = self.engine.attn_backend
        ratio = int(backend.ratio)
        if page_size % ratio:
            raise ValueError("target K/V page size must be divisible by QSA ratio")
        rows_per_page = page_size // ratio

        state_instrumentation_started = time.perf_counter()
        state_families_before = self._target_state_family_digests(captured)
        digest_before = self._combine_state_family_digests(state_families_before)
        default_rng_before = self._default_rng_states()
        state_instrumentation_ms = (
            time.perf_counter() - state_instrumentation_started
        ) * 1000.0
        state_instrumentation_synchronizations = 0
        state_required_synchronizations = 0
        default_rng_evidence = None

        preparation_started = time.perf_counter()
        page_row_backup = self.engine.page_table[dummy, :row_end].clone()
        ring_backup = target_kv._pending_ring[dummy].clone()
        position_backup = target_kv._pending_position_ring[dummy].clone()
        scratch_backup = target_kv._cmp_k_buffer[
            :, target_kv.cmp_scratch_base + dummy
        ].clone()
        free_page_order_before = self.cache_manager.free_slots.clone()
        pool = self.engine.linear_state_pool
        free_slots_before = pool.num_free_slots
        linear_free_order_before = (
            tuple(pool._free_slots) if hasattr(pool, "_free_slots") else None
        )
        live_prefix_pages = self.engine.page_table[
            captured.table_idx, :base:page_size
        ].clone()
        shadow_slot = None
        owns_shadow_slot = False
        shadow_snapshot = None
        shadow_digest_before = None
        scratch_page_backups: list[tuple[torch.Tensor, torch.Tensor]] = []
        state_prepare_ms = 0.0
        cleanup_started = None
        forward_result = None
        copied_partial_pages = 0
        candidate_out_loc_owned = False
        try:
            if pool.num_free_slots:
                shadow_slot = pool.alloc(1)[0]
                owns_shadow_slot = True
            else:
                candidates = [
                    slot
                    for slot in range(1, pool.num_slots)
                    if slot not in captured.protected_linear_slots
                ]
                if not candidates:
                    raise RuntimeError("MTP verifier has no inactive recurrent slot to borrow")
                shadow_slot = candidates[0]
            shadow_snapshot = self._linear_slot_snapshot(pool, shadow_slot)
            shadow_digest_before = self._linear_snapshot_digest(shadow_snapshot)

            with self.cache_manager.temporary_page_lease(
                needed_pages,
                forbidden_pages=live_prefix_pages,
            ) as leased:
                leased_list = [int(value) for value in leased.cpu().tolist()]
                for layer_id in sorted(backend._idx_slot):
                    for physical in leased_list:
                        for tensor in (
                            target_kv.k_cache(layer_id),
                            target_kv.v_cache(layer_id),
                        ):
                            page = self._target_kv_page(
                                tensor, physical, page_size=page_size
                            )
                            scratch_page_backups.append((page, page.clone()))
                for slot in sorted(set(backend._idx_slot.values())):
                    compressed = target_kv.cmp_k_cache(slot)
                    for physical in leased_list:
                        page_index = physical // page_size
                        compressed_start = page_index * rows_per_page
                        compressed_end = compressed_start + rows_per_page
                        if compressed_end > int(compressed.shape[0]):
                            raise IndexError("target QSA compressed page is out of range")
                        page = compressed[compressed_start:compressed_end]
                        scratch_page_backups.append((page, page.clone()))

                self.engine.page_table[dummy, :page_start].copy_(
                    self.engine.page_table[captured.table_idx, :page_start]
                )
                for offset, physical in enumerate(leased_list):
                    logical = page_start + offset * page_size
                    self.engine.page_table[
                        dummy, logical : logical + page_size
                    ] = torch.arange(
                        physical,
                        physical + page_size,
                        dtype=torch.int32,
                        device=self.device,
                    )
                self._transaction_fault("after-page-table-rewrite")

                if base % page_size:
                    source = int(
                        self.engine.page_table[captured.table_idx, page_start].item()
                    )
                    self._copy_partial_target_page(source, leased_list[0])
                    copied_partial_pages = 1
                self._transaction_fault("after-kv-copy")

                pool.copy_from(captured.linear_slot_idx, shadow_slot)
                self._transaction_fault("after-recurrent-copy")
                for slot in sorted(set(backend._idx_slot.values())):
                    target_kv.pending_ring(slot)[dummy].copy_(
                        target_kv.pending_ring(slot)[captured.table_idx]
                    )
                    target_kv.pending_position_ring(slot)[dummy].copy_(
                        target_kv.pending_position_ring(slot)[captured.table_idx]
                    )
                batch = self._target_verify_batch(
                    captured,
                    candidate_ids,
                    dummy_table_idx=dummy,
                    shadow_slot=shadow_slot,
                )
                self._assert_out_loc_owned_by_lease(
                    batch.out_loc,
                    leased,
                    page_size=page_size,
                )
                candidate_out_loc_owned = True
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    state_required_synchronizations += 1
                state_prepare_ms = (
                    time.perf_counter() - preparation_started
                ) * 1000.0
                forward_result = forward(batch)
                cleanup_started = time.perf_counter()
                self._transaction_fault("after-target-forward")
        finally:
            if cleanup_started is None:
                cleanup_started = time.perf_counter()
            for page, backup in reversed(scratch_page_backups):
                page.copy_(backup)
            self.engine.page_table[dummy, :row_end].copy_(page_row_backup)
            target_kv._pending_ring[dummy].copy_(ring_backup)
            target_kv._pending_position_ring[dummy].copy_(position_backup)
            target_kv._cmp_k_buffer[:, target_kv.cmp_scratch_base + dummy].copy_(
                scratch_backup
            )
            if shadow_slot is not None and shadow_snapshot is not None:
                self._restore_linear_slot(pool, shadow_slot, shadow_snapshot)
                if owns_shadow_slot:
                    pool.free(shadow_slot)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                state_required_synchronizations += 1
            state_cleanup_ms = (
                time.perf_counter() - cleanup_started
            ) * 1000.0
            default_rng_instrumentation_started = time.perf_counter()
            default_rng_evidence = self._assert_default_rng_unchanged(
                default_rng_before
            )
            state_instrumentation_ms += (
                time.perf_counter() - default_rng_instrumentation_started
            ) * 1000.0

        post_state_instrumentation_started = time.perf_counter()
        if not torch.equal(self.cache_manager.free_slots, free_page_order_before):
            raise RuntimeError("MTP verifier changed target KV free-list order")
        if pool.num_free_slots != free_slots_before:
            raise RuntimeError("MTP verifier leaked target recurrent slots")
        if linear_free_order_before is not None and tuple(pool._free_slots) != (
            linear_free_order_before
        ):
            raise RuntimeError("MTP verifier changed recurrent free-list order")
        if shadow_snapshot is not None:
            shadow_digest_after = self._linear_slot_digest(pool, shadow_slot)
            if shadow_digest_after != shadow_digest_before:
                raise RuntimeError("MTP verifier changed its recurrent scratch snapshot")
        scratch_state_restored = all(
            torch.equal(page, backup) for page, backup in scratch_page_backups
        )
        if not scratch_state_restored:
            raise RuntimeError("MTP verifier changed leased scratch-page state")
        if not torch.equal(
            self.engine.page_table[dummy, :row_end], page_row_backup
        ):
            raise RuntimeError("MTP verifier changed its scratch page-table row")
        state_families_after = self._target_state_family_digests(captured)
        digest_after = self._combine_state_family_digests(state_families_after)
        if digest_after != digest_before:
            changed_families = sorted(
                name
                for name in set(state_families_before) | set(state_families_after)
                if state_families_before.get(name) != state_families_after.get(name)
            )
            raise RuntimeError(
                "MTP verifier changed live target state families: "
                + ",".join(changed_families)
            )
        if not isinstance(forward_result, MTPVerifyForwardResult):
            raise TypeError("MTP target forward returned an invalid result")
        if forward_result.mode != mode:
            raise RuntimeError(
                f"MTP target forward mode {forward_result.mode!r} != {mode!r}"
            )
        if forward_result.logits.shape[0] != len(candidate_ids):
            raise RuntimeError("MTP target transaction returned the wrong logit-row count")
        state_instrumentation_ms += (
            time.perf_counter() - post_state_instrumentation_started
        ) * 1000.0
        return {
            "mode": mode,
            "logits": forward_result.logits,
            "target_required_wall_ms": forward_result.required_wall_ms,
            "target_core_cuda_ms": forward_result.core_cuda_ms,
            "target_required_synchronizations": (
                forward_result.required_synchronizations
            ),
            "target_instrumentation_wall_ms": (
                forward_result.instrumentation_wall_ms
            ),
            "target_instrumentation_synchronizations": (
                forward_result.instrumentation_synchronizations
            ),
            "target_forward_ms": forward_result.required_wall_ms,
            "target_cuda_ms": forward_result.core_cuda_ms,
            "target_synchronizations": forward_result.synchronizations,
            "expert_movement": forward_result.expert_movement,
            "state_required_scope": "scratch-backup-through-cleanup",
            "state_required_synchronizations": state_required_synchronizations,
            "state_prepare_ms": state_prepare_ms,
            "state_cleanup_ms": state_cleanup_ms,
            "state_ms": state_prepare_ms + state_cleanup_ms,
            "state_instrumentation_ms": state_instrumentation_ms,
            "state_instrumentation_synchronizations": (
                state_instrumentation_synchronizations
            ),
            "state_digest_unchanged": True,
            "default_rng": default_rng_evidence,
            "scratch_live_disjoint": True,
            "candidate_out_loc_lease_owned": candidate_out_loc_owned,
            "copied_partial_pages": copied_partial_pages,
            "scratch_state_restored": scratch_state_restored,
            "pages_conserved": True,
            "free_list_order_restored": True,
            "recurrent_slots_conserved": True,
            "recurrent_state_restored": True,
            "borrowed_recurrent_snapshot": not owns_shadow_slot,
            "borrowed_snapshot_digest_unchanged": True,
        }

    def _accept_target_logits(
        self,
        captured: MTPTargetCapture,
        proposals: list[int],
        draft_logits: list[torch.Tensor],
        logits: torch.Tensor,
    ) -> dict:
        if self._rng_request_uid != captured.uid:
            raise RuntimeError("MTP acceptance RNG is not bound to the active request")
        depth = len(proposals)
        generator = self._acceptance_generators.get(depth)
        if generator is None:
            raise RuntimeError(f"MTP acceptance RNG depth {depth} is unavailable")
        before = self._rng_stream_snapshot("acceptance", depth=depth)
        private_state_before = generator.get_state()
        default_before = self._default_rng_states()
        try:
            acceptance = batched_speculative_accept(
                proposals=proposals,
                draft_logits=torch.stack(draft_logits),
                target_logits=logits,
                temperature=captured.temperature,
                top_k=captured.top_k,
                top_p=captured.top_p,
                generator=generator,
            )
            acceptance_evidence_started = time.perf_counter()
            default_rng = self._assert_default_rng_unchanged(default_before)
        except BaseException:
            generator.set_state(private_state_before)
            self._assert_default_rng_unchanged(default_before)
            raise
        draws = 0 if acceptance.greedy else 2 * depth + 1
        self._acceptance_draw_counts[depth] += draws
        after = self._rng_stream_snapshot("acceptance", depth=depth)
        if draws == 0 and before["state_sha256"] != after["state_sha256"]:
            raise RuntimeError("greedy MTP acceptance advanced its private RNG")
        if draws and before["state_sha256"] == after["state_sha256"]:
            raise RuntimeError("sampled MTP acceptance did not advance its private RNG")
        rng_transition = self._rng_transition(
            before,
            after,
            default_rng=default_rng,
        )
        rng_transition["greedy"] = acceptance.greedy
        self._acceptance_cycle_indices[depth] += 1
        selected = [
            {
                "draft": int(draft),
                "p": acceptance.target_probabilities[index],
                "q": acceptance.draft_probabilities[index],
                "acceptance_probability": acceptance.acceptance_probabilities[index],
            }
            for index, draft in enumerate(proposals)
        ]
        target_logits_sha256 = tensor_sha256(logits)
        acceptance_evidence_ms = (
            time.perf_counter() - acceptance_evidence_started
        ) * 1000.0
        return {
            "verified_rows": depth + 1,
            "target_logits_sha256": target_logits_sha256,
            "target_tokens": list(acceptance.target_tokens),
            "accepted_tokens": acceptance.accepted_prefix,
            "corrected_token": acceptance.corrected_token,
            "acceptance_ms": acceptance.required_wall_ms,
            "acceptance_required_wall_ms": acceptance.required_wall_ms,
            "acceptance_required_synchronizations": (
                acceptance.required_synchronizations
            ),
            "acceptance_instrumentation_wall_ms": (
                acceptance.instrumentation_wall_ms + acceptance_evidence_ms
            ),
            "acceptance_instrumentation_synchronizations": (
                acceptance.instrumentation_synchronizations
            ),
            "acceptance_synchronizations": acceptance.synchronizations,
            "acceptance_rng": rng_transition,
            "selected_probabilities": selected,
        }

    def _ensure_target_graph(
        self, captured: MTPTargetCapture, candidate_ids: list[int]
    ) -> tuple[MTPGraphCaptureResult, dict | None]:
        graph_verifier = getattr(self, "graph_verifier", None)
        width = len(candidate_ids)
        if graph_verifier is None:
            return (
                MTPGraphCaptureResult(
                    width=width,
                    status="permanently-unsupported",
                    reason="GRAPH_RUNNER_NOT_BOUND",
                    memory_bytes=0,
                ),
                None,
            )
        support = graph_verifier.support(width)
        capture_record = None
        if support is None:
            capture_record = self._run_target_transaction(
                captured,
                candidate_ids,
                forward=graph_verifier.capture_forward,
                mode="graph-capture",
            )
            capture_record.pop("logits")
            support = graph_verifier.support(width)
            if support is None:
                support = graph_verifier.last_attempt(width)
        if support is None:
            raise RuntimeError("MTP graph capture did not record an attempt result")
        return support, capture_record

    def verify_candidates(
        self,
        captured: MTPTargetCapture,
        confirmed_token: int,
        proposals: list[int],
        draft_logits: list[torch.Tensor],
        *,
        mode: str,
    ) -> dict:
        if mode not in {"oracle", "compare", "fast-eager", "fast-graph"}:
            raise ValueError(f"unsupported MTP verifier mode {mode!r}")
        if len(proposals) not in (1, 2, 3) or len(draft_logits) != len(proposals):
            raise ValueError("MTP verifier requires one to three proposals and matching logits")
        candidate_ids = [confirmed_token, *proposals]
        comparison_record = None
        oracle_record = None
        graph_record = None
        graph_capture_record = None
        graph_support = None
        graph_comparison_record = None
        if mode == "oracle":
            selected = self._run_target_transaction(
                captured,
                candidate_ids,
                forward=self._forward_target_oracle,
                mode="oracle",
            )
        else:
            eager_record = None
            if mode in {"fast-eager", "compare"}:
                eager_record = self._run_target_transaction(
                    captured,
                    candidate_ids,
                    forward=self.fast_verifier.forward_eager,
                    mode="fast-eager",
                )
            if mode in {"fast-graph", "compare"}:
                graph_support, graph_capture_record = self._ensure_target_graph(
                    captured, candidate_ids
                )
                graph_verifier = getattr(self, "graph_verifier", None)
                if graph_support.status == "captured" and graph_verifier is not None:
                    graph_record = self._run_target_transaction(
                        captured,
                        candidate_ids,
                        forward=graph_verifier.replay,
                        mode="fast-graph",
                    )
                if mode == "fast-graph" and graph_record is None:
                    eager_record = self._run_target_transaction(
                        captured,
                        candidate_ids,
                        forward=self.fast_verifier.forward_eager,
                        mode="fast-eager",
                    )
            selected = graph_record if mode == "fast-graph" and graph_record else eager_record
            assert selected is not None
            if mode == "compare":
                if graph_record is not None:
                    graph_comparison = compare_verifier_logits(
                        selected["logits"], graph_record["logits"]
                    )
                    graph_comparison_record = vars(graph_comparison)
                    graph_comparison_record["normal_support_order_matches"] = (
                        sampling_support_order_matches(
                            selected["logits"],
                            graph_record["logits"],
                            temperature=getattr(captured, "temperature", 0.0),
                            top_k=getattr(captured, "top_k", 1),
                            top_p=getattr(captured, "top_p", 1.0),
                        )
                    )
                    if not graph_comparison.matches or not graph_comparison_record[
                        "normal_support_order_matches"
                    ]:
                        self._dump_comparison_failure(
                            "graph-vs-eager",
                            graph_comparison_record,
                            candidate_ids,
                            reference=selected["logits"],
                            fast=graph_record["logits"],
                        )
                        raise RuntimeError(
                            "graph MTP target logits did not match the eager fast checker"
                        )
                oracle_record = self._run_target_transaction(
                    captured,
                    candidate_ids,
                    forward=self._forward_target_oracle,
                    mode="oracle",
                )
                # the oracle runs the prefill MoE/kernel path and the fast checker the
                # decode-movement path; that bf16 drift compounds per verify row, because each
                # candidate row attends the previous rows' KV, so no fixed logit tolerance
                # survives depth. Judge faithfulness where it is consumed instead -- the sampled
                # output distribution -- but gate on its *excess* component only: on knife-edge
                # rows, within-noise logit moves shift plain TV with the tied pair's mass
                # (observed live at 0.13 against a subtlest-structural 0.26 -- the ranges
                # overlap), while excess TV separates them by an order of magnitude. The
                # greedy-argmax check stays mandatory, with a near-tie escape: cross-path noise
                # can collapse a two-ulp reference gap into an exact bf16 tie whose argmax
                # breaks arbitrarily (observed live at logit 18.5/18.5).
                comparison = compare_verifier_distributions(
                    oracle_record["logits"], selected["logits"]
                )
                comparison_record = vars(comparison)
                # exact rank order among a bf16 near-tie is not a stable property across the two
                # kernel paths -- at top_k=40 over a 248k vocab, boundary churn and rank flips are
                # certain. Bound the client-visible quantity instead -- the filtered distribution
                # the sampler draws from -- in excess terms (noise_tolerance), because sharpening
                # amplifies within-noise moves without bound. The record key keeps its name
                # because the live-gate harness requires it.
                divergence = sampling_distribution_divergence(
                    oracle_record["logits"],
                    selected["logits"],
                    temperature=getattr(captured, "temperature", 0.0),
                    top_k=getattr(captured, "top_k", 1),
                    top_p=getattr(captured, "top_p", 1.0),
                    noise_tolerance=comparison.noise_tolerance,
                )
                comparison_record["normal_sampling_divergence"] = divergence
                comparison_record["normal_support_order_matches"] = (
                    divergence <= _MAX_SAMPLING_DIVERGENCE
                )
                if not comparison.matches or not comparison_record[
                    "normal_support_order_matches"
                ]:
                    self._dump_comparison_failure(
                        "oracle-vs-eager",
                        comparison_record,
                        candidate_ids,
                        reference=oracle_record["logits"],
                        fast=selected["logits"],
                    )
                    raise RuntimeError(
                        "fast MTP target logits did not match the prompt-style oracle"
                    )
        graph_logits_hash = (
            tensor_sha256(graph_record["logits"])
            if graph_record is not None
            else None
        )
        logits = selected.pop("logits")
        result = {
            **self._accept_target_logits(captured, proposals, draft_logits, logits),
            **selected,
        }
        result["verification_ms"] = (
            result["target_forward_ms"] + result["acceptance_ms"]
        )
        result["projection_eligible"] = mode == selected["mode"] and mode in {
            "fast-eager",
            "fast-graph",
        }
        if graph_support is not None:
            result["graph"] = {
                "support": vars(graph_support),
                "capture": graph_capture_record,
                "projection_eligible": result["projection_eligible"]
                and selected["mode"] == "fast-graph",
            }
            if graph_record is not None:
                if graph_record is not selected:
                    graph_record.pop("logits")
                result["graph"].update(
                    {
                        "logits_sha256": graph_logits_hash,
                        "replay": graph_record,
                        "eager_comparison": graph_comparison_record,
                    }
                )
        if comparison_record is not None and oracle_record is not None:
            oracle_logits = oracle_record.pop("logits")
            result["comparison"] = {
                **comparison_record,
                "oracle_logits_sha256": tensor_sha256(oracle_logits),
                "fast_logits_sha256": result["target_logits_sha256"],
                "oracle_target_forward_ms": oracle_record["target_forward_ms"],
                "oracle_state_ms": oracle_record["state_ms"],
                "projection_eligible": False,
            }
        return result

    def _verify_target(
        self,
        captured: MTPTargetCapture,
        confirmed_token: int,
        proposals: list[int],
        draft_logits: list[torch.Tensor],
    ) -> dict:
        """Retained prompt-style correctness oracle for the completed feasibility spike."""
        return self.verify_candidates(
            captured,
            confirmed_token,
            proposals,
            draft_logits,
            mode="oracle",
        )

    def observe(self, captured: MTPTargetCapture, next_token: torch.Tensor) -> None:
        default_rng_before = self._default_rng_states()
        try:
            self._observe_impl(captured, next_token)
        finally:
            self._assert_default_rng_unchanged(default_rng_before)

    def _observe_impl(self, captured: MTPTargetCapture, next_token: torch.Tensor) -> None:
        if self._uid != captured.uid:
            self._reset_request(captured.uid)
        if captured.is_chunked:
            target_expert_temperature = {
                **self._target_expert_temperature,
                "residency_reset": False,
                "reset_setup_ms": 0.0,
            }
        else:
            target_expert_temperature = self._prepare_target_expert_temperature(
                captured.uid
            )
        update_started = time.perf_counter()
        hidden = captured.multi_stream_cpu
        embeds = captured.inputs_embeds_cpu
        target_embedding = None
        if not captured.is_chunked:
            target_embedding = self.target_model.model.embed_tokens.forward(
                next_token.to(self.device).view(1)
            ).to("cpu")
        had_pending = self._pending_hidden_cpu is not None
        pending_rope = self._pending_rope_cpu
        paired_hidden, paired_embeds, self._pending_hidden_cpu = build_shifted_pairs(
            self._pending_hidden_cpu,
            hidden,
            embeds,
            next_embedding=target_embedding,
        )
        paired_rope, self._pending_rope_cpu = build_shifted_rope_positions(
            pending_rope,
            captured.rope_positions_cpu,
            had_pending_hidden=had_pending,
            final_chunk=target_embedding is not None,
        )
        if paired_hidden.shape[0] == 0:
            if captured.is_chunked:
                self._request_prompt_setup_ms += (
                    time.perf_counter() - update_started
                ) * 1000.0
            return
        if paired_rope is not None and paired_rope.shape[1] != paired_hidden.shape[0]:
            raise RuntimeError("MTP shifted picture positions do not match paired rows")
        capture_blocks: dict[int, torch.Tensor] = {}
        sample, recursive = self._run_rows(
            paired_embeds,
            paired_hidden,
            rope_positions_cpu=paired_rope,
            capture=capture_blocks,
        )
        torch.cuda.synchronize(self.device)
        update_ms = (time.perf_counter() - update_started) * 1000.0
        if captured.is_chunked:
            self._request_prompt_setup_ms += update_ms
            self._write_event(
                {"event": "prompt_chunk", "uid": captured.uid, "rows": len(paired_hidden),
                 "committed_len": self.committed_len, "mtp_ms": update_ms,
                 "prompt_setup_accumulated_ms": self._request_prompt_setup_ms,
                 "target_expert_temperature": target_expert_temperature}
            )
            return

        proposal_setup_started = time.perf_counter()
        slot = next(iter(capture_blocks))
        saved = {slot: capture_blocks[slot][-1:].clone()}
        ring = self.kv_cache._pending_ring.clone()
        position_ring = self.kv_cache._pending_position_ring.clone()
        base_len = self.committed_len
        proposal_setup_synchronizations = 0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            proposal_setup_synchronizations = 1
        proposal_setup_ms = (
            time.perf_counter() - proposal_setup_started
        ) * 1000.0
        proposals = []
        draft_logits: list[torch.Tensor] = []
        draft_rng_steps: list[dict] = []
        draft_step_ms: list[float] = []
        recursive_step_ms: list[float] = []
        draft_instrumentation_step_ms: list[float] = []
        proposal_required_synchronizations = 1 + proposal_setup_synchronizations
        cleanup_ms = 0.0
        try:
            for index in range(self.config.depth):
                lm_head_started = time.perf_counter()
                logits = self.target_model.lm_head.forward_all(sample)[0]
                torch.cuda.synchronize(self.device)
                proposal_required_synchronizations += 1
                lm_head_ms = (time.perf_counter() - lm_head_started) * 1000.0

                token, draft_rng = self._sample_draft(
                    logits,
                    temperature=captured.temperature,
                    top_k=captured.top_k,
                    top_p=captured.top_p,
                )
                draft_timing = self._last_draft_timing
                draft_instrumentation_step_ms.append(
                    draft_timing["instrumentation_wall_ms"]
                )
                draft_rng_steps.append(draft_rng)

                clone_started = time.perf_counter()
                draft_logits.append(logits.detach().clone())
                torch.cuda.synchronize(self.device)
                proposal_required_synchronizations += 1
                clone_ms = (time.perf_counter() - clone_started) * 1000.0
                draft_step_ms.append(
                    lm_head_ms + draft_timing["required_wall_ms"] + clone_ms
                )
                proposals.append(token)
                if index + 1 == self.config.depth:
                    break

                recursive_started = time.perf_counter()
                embedding = self.target_model.model.embed_tokens.forward(
                    torch.tensor([token], dtype=torch.int32, device=self.device)
                ).to("cpu")
                self.committed_len = base_len + index
                recursive_rope = (
                    None
                    if captured.rope_positions_cpu is None
                    else torch.full(
                        (3, 1),
                        self.committed_len + captured.mrope_position_delta,
                        dtype=torch.int64,
                    )
                )
                sample, recursive = self._run_rows(
                    embedding,
                    recursive.to("cpu"),
                    rope_positions_cpu=recursive_rope,
                    saved=saved,
                )
                torch.cuda.synchronize(self.device)
                proposal_required_synchronizations += 1
                recursive_step_ms.append(
                    (time.perf_counter() - recursive_started) * 1000.0
                )
        finally:
            cleanup_started = time.perf_counter()
            self.kv_cache._pending_ring.copy_(ring)
            self.kv_cache._pending_position_ring.copy_(position_ring)
            self.committed_len = base_len
            torch.cuda.synchronize(self.device)
            proposal_required_synchronizations += 1
            cleanup_ms = (time.perf_counter() - cleanup_started) * 1000.0
        depth_results = []
        request_cycle_index = self._rng_cycle_index
        prompt_setup_ms = (
            self._request_prompt_setup_ms if request_cycle_index == 0 else 0.0
        )
        for depth in range(1, self.config.depth + 1):
            verification = self.verify_candidates(
                captured,
                int(next_token),
                proposals[:depth],
                draft_logits[:depth],
                mode=self.config.verify_mode,
            )
            graph_capture = (
                verification.get("graph", {}).get("capture")
                if isinstance(verification.get("graph"), dict)
                else None
            )
            graph_capture_setup_ms = 0.0
            graph_capture_required_synchronizations = 0
            if graph_capture is not None:
                graph_capture_setup_ms = (
                    graph_capture["target_required_wall_ms"]
                    + graph_capture["state_ms"]
                )
                graph_capture_required_synchronizations = (
                    graph_capture["target_required_synchronizations"]
                    + graph_capture["state_required_synchronizations"]
                )
            proposal_components = self._proposal_component_times(
                request_cycle_index=request_cycle_index,
                prompt_setup_ms=prompt_setup_ms,
                update_ms=update_ms,
                draft_step_ms=draft_step_ms,
                recursive_step_ms=recursive_step_ms,
                cleanup_ms=cleanup_ms,
                graph_capture_setup_ms=graph_capture_setup_ms,
                depth=depth,
                setup_ms=proposal_setup_ms,
            )
            projection = MTPProjectionSample(
                prompt_ms=proposal_components["P"],
                draft_ms=proposal_components["D"],
                verify_ms=verification["target_required_wall_ms"],
                state_ms=verification["state_ms"],
                acceptance_ms=verification["acceptance_required_wall_ms"],
                emitted_tokens=1 + verification["accepted_tokens"],
            )
            component_total_ms = sum(
                projection.components[name] for name in ("P", "D", "V", "S", "A")
            )
            if abs(component_total_ms - projection.total_ms) > 1e-6:
                raise RuntimeError("MTP projection components do not conserve total time")
            depth_results.append(
                {
                    "depth": depth,
                    "proposal_ms": proposal_components["P"]
                    + proposal_components["D"],
                    "proposal_required_synchronizations": (
                        3 + 2 * depth + (depth - 1)
                        + graph_capture_required_synchronizations
                    ),
                    "proposal_instrumentation_ms": sum(
                        draft_instrumentation_step_ms[:depth]
                    ),
                    "verification_ms": verification["verification_ms"],
                    "state_ms": verification["state_ms"],
                    "accepted_tokens": verification["accepted_tokens"],
                    "components": projection.components,
                    "component_total_ms": component_total_ms,
                    "component_reconciled": True,
                    "projection_eligible": verification["projection_eligible"],
                    "projected_sequential_tokens_per_second": (
                        projection.tokens_per_second
                    ),
                    "verification": verification,
                }
            )
        self._request_prompt_setup_ms = 0.0
        proposal_evidence_started = time.perf_counter()
        draft_logits_sha256 = [tensor_sha256(logits) for logits in draft_logits]
        proposal_evidence_ms = (
            time.perf_counter() - proposal_evidence_started
        ) * 1000.0
        self._write_event(
            {
                "event": "proposal",
                "uid": captured.uid,
                "rng_cycle_index": self._rng_cycle_index,
                "target_expert_temperature": target_expert_temperature,
                "target_cached_len": captured.cached_len,
                "committed_len": self.committed_len,
                "depth": self.config.depth,
                "draft_tokens": proposals,
                "draft_logits_sha256": draft_logits_sha256,
                "draft_rng_steps": draft_rng_steps,
                "prompt_setup_ms": prompt_setup_ms,
                "mtp_update_ms": update_ms,
                "proposal_setup_ms": proposal_setup_ms,
                "draft_step_ms": draft_step_ms,
                "recursive_step_ms": recursive_step_ms,
                "proposal_cleanup_ms": cleanup_ms,
                "proposal_required_synchronizations": (
                    proposal_required_synchronizations
                ),
                "proposal_instrumentation_ms": (
                    sum(draft_instrumentation_step_ms) + proposal_evidence_ms
                ),
                "proposal_instrumentation_synchronizations": 0,
                "proposal_ms": depth_results[-1]["proposal_ms"],
                "depth_results": depth_results,
                "verification": depth_results[-1]["verification"],
                "staging": vars(self.staged_model.stats),
                "expert": vars(self.expert_runner.stats),
            }
        )
        self._rng_cycle_index += 1

    def _dump_comparison_failure(
        self,
        kind: str,
        comparison_record: dict,
        candidate_ids: list[int],
        *,
        reference: torch.Tensor,
        fast: torch.Tensor,
    ) -> None:
        """Persist the mismatching verifier logits so the stop is diagnosable offline."""
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dump_path = (
            self.config.private_root
            / "evidence"
            / f"comparison-failure-{kind}-{stamp}.pt"
        )
        try:
            torch.save(
                {
                    "kind": kind,
                    "comparison": comparison_record,
                    "candidate_ids": list(candidate_ids),
                    "reference_logits": reference.detach().float().cpu(),
                    "fast_logits": fast.detach().float().cpu(),
                },
                dump_path,
            )
            dumped = str(dump_path)
        except Exception:
            dumped = None
        self._write_event(
            {
                "event": "comparison_failure",
                "kind": kind,
                "comparison": {
                    key: value
                    for key, value in comparison_record.items()
                    if isinstance(value, (bool, int, float, str))
                },
                "candidate_count": len(candidate_ids),
                "dump": dumped,
            }
        )

    def _write_event(self, event: dict) -> None:
        event = {"time_unix": time.time(), "placement": self.config.placement, **event}
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self.graph_verifier is not None:
            self.graph_verifier.destroy()
            self.graph_verifier = None
        self.expert_runner.close()
        if self._weight_store is not None:
            self._weight_store.close()
            self._weight_store = None


__all__ = [
    "MTPShadowConfig",
    "MTPShadowObserver",
    "MTPTargetCapture",
    "build_shifted_pairs",
    "build_shifted_rope_positions",
    "greedy_acceptance",
    "sampling_probabilities",
    "speculative_accept",
    "tensor_sha256",
]
