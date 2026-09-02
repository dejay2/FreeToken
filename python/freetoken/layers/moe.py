import atexit
import json
import os
from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_backend
from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl, fused_topk
from freetoken.moe.offload_cache import OffloadMoeCache

# Imported as a module, not `from ... import PREFETCH`: the config is read through the
# module attribute at every use so a test (or a future runtime re-arm) has ONE place to
# swap it, shared with OffloadMoeCache._init_prefetch.
from freetoken.moe import prefetch as _prefetch
from freetoken.utils import div_even, init_logger

from .base import BaseOP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router decision (topk_weights[float32], topk_ids[int32]) for models whose router
# is computed outside the MoE layer. Such models call ``routed_forward`` (offload) or
# ``_run_experts`` (dense) with a precomputed routing instead of going through the
# generic softmax+top-k path.
TopK = Tuple[torch.Tensor, torch.Tensor]

# The widest MTP verification block one forward may carry: ``w = 1 + depth`` rows. A MIRROR of
# ``1 + freetoken.engine.config._MAX_SPEC_DEPTH``, which is the source of truth; importing the
# engine config here would drag the model registry and the HF config loader into every MoE
# layer. ``tests/moe/test_mtp_fast_verify_moe.py`` imports both and pins them equal.
_MAX_MTP_VERIFY_ROWS = 6

# The offload PREFILL path streams every expert of every layer (48 x 1.32 GiB ~ 1.25 s over
# PCIe) whatever the prompt size, so a 26-token chat turn -- or a prefix-cache hit with three
# new rows -- pays the whole 63 GiB bank. FREETOKEN_MOE_SMALL_PREFILL_ROWS=N sends a prefill
# batch of at most N rows (summed over every request in it) through DECODE movement instead:
# the LRU lookup plus a fetch of only the routed experts. Default 0 = off, byte-identical to
# before. Read once at import (this is per-layer, per-forward) and referenced through the
# module global, so a test swaps it with monkeypatch.setattr like _prefetch.PREFETCH.
_SMALL_PREFILL_ROWS_ENV = "FREETOKEN_MOE_SMALL_PREFILL_ROWS"
# Hard ceiling regardless of the env. The GPU decode GEMVs have no M limit of their own (the
# marlin grid is (M*top_k, cdiv(N, BLOCK_N)), one program per route, masked epilogue), but
# they are GEMV-shaped: past a few dozen rows the streamed grouped GEMM wins anyway, and the
# cpu/hybrid decode target has its own width bound (checked separately, per cache).
_MAX_SMALL_PREFILL_ROWS = 64


def _read_small_prefill_rows() -> int:
    raw = (os.getenv(_SMALL_PREFILL_ROWS_ENV) or "").strip()
    if not raw:
        return 0
    try:
        return max(0, min(int(raw), _MAX_SMALL_PREFILL_ROWS))
    except ValueError:
        return 0


_SMALL_PREFILL_ROWS = _read_small_prefill_rows()

# ``_expert_gemm``'s ``is_prefill`` flag picks the KERNEL, not just the movement, and for most
# formats the two kernels are not the same arithmetic:
#
#   fp8_block   prefill is W8A8 (activations quantized to fp8 per-128-K group, fp8 tensor
#               cores), decode is W8A16 (activations stay bf16). Measured on a real cache,
#               the two agree with an fp32 reference to 4.9e-3 and 5.0e-4 respectively and
#               differ from EACH OTHER by ~4% of the layer output -- at every row count,
#               M=1 included. Over 48 layers that changes the prefill's hidden states, its
#               KV, its GDN/PLE recurrent state and the MTP draft priming, so a greedy
#               continuation near a decision boundary flips.
#   bf16        grouped GEMM vs per-route GEMVs: a different reduction order (small, but
#               not zero -- ~6e-3 absolute on a ~1.0 scale).
#   mxfp4_triton / ds_fp4   separate prefill and decode kernels likewise.
#
# Only formats whose dispatch lands on ONE kernel for both phases can take decode movement
# without also changing the model's math: there the movement decides which rows the banks
# hold and ``topk_ids``/``alphas`` are remapped to match, and nothing else moves.
_MOVEMENT_ONLY_FORMATS = frozenset({"nvfp4_marlin", "nvfp4_b12x"})
_small_prefill_refused: set[str] = set()

logger = init_logger(__name__)


def _refuse_small_prefill(fmt: str | None) -> bool:
    """Warn once per format, then decline. A wrong answer is worse than a slow one."""
    key = str(fmt)
    if key not in _small_prefill_refused:
        _small_prefill_refused.add(key)
        logger.warning(
            "%s is set, but %r expert kernels are different arithmetic for prefill and "
            "decode (not just different data movement), so routing a small prefill through "
            "decode movement would change the model's output. Ignoring it for this format; "
            "small prefills keep streaming the expert bank.",
            _SMALL_PREFILL_ROWS_ENV,
            fmt,
        )
    return False

# Hybrid decode overlaps the CPU overflow GEMV behind the GPU PCIe fetch + GEMM by
# default. Set FREETOKEN_HYBRID_OVERLAP=0 to force the serial path (CPU sync before the
# GPU work) -- a measurement-only escape hatch to A/B the overlap benefit.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"

# ----------------------------------------------------------------------
# Throwaway routing diagnostics (expert-prefetch feasibility study).
# Enabled only when FREETOKEN_MOE_ROUTE_LOG names a directory; the single
# cached module-level check below is the entire cost when it is unset.
# ----------------------------------------------------------------------
_ROUTE_LOG_DIR = os.getenv("FREETOKEN_MOE_ROUTE_LOG") or None
_route_log_records: list = []
_ROUTE_LOG_FLUSH_EVERY = 200
_route_log_path: str | None = None


def _route_log_flush() -> None:
    global _route_log_path
    if not _route_log_records:
        return
    if _route_log_path is None:
        assert _ROUTE_LOG_DIR is not None
        os.makedirs(_ROUTE_LOG_DIR, exist_ok=True)
        _route_log_path = os.path.join(_ROUTE_LOG_DIR, f"route-log-{os.getpid()}.jsonl")
    batch, _route_log_records[:] = list(_route_log_records), []
    with open(_route_log_path, "a", encoding="utf-8") as fh:
        for rec in batch:
            fh.write(json.dumps(rec) + "\n")


def _route_log(layer_id, topk_ids: torch.Tensor) -> None:
    """Record one MoE forward's routing decision. Best-effort: never raises."""
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        batch = get_global_ctx().batch
        experts = topk_ids.detach().reshape(topk_ids.shape[0], -1).cpu().tolist()
        tokens = batch.input_ids.detach().reshape(-1).cpu().tolist()
        positions = getattr(batch, "positions", None)
        pos = [] if positions is None else positions.detach().reshape(-1).cpu().tolist()
        _route_log_records.append(
            {
                "layer": layer_id,
                "phase": batch.phase,
                "tokens": tokens,
                "positions": pos,
                "experts": experts,
            }
        )
        if len(_route_log_records) >= _ROUTE_LOG_FLUSH_EVERY:
            _route_log_flush()
    except Exception:  # diagnostics must never break a forward
        pass


def _route_log_flush_atexit() -> None:
    try:
        _route_log_flush()
    except Exception:
        pass


if _ROUTE_LOG_DIR is not None:
    atexit.register(_route_log_flush_atexit)


# ----------------------------------------------------------------------
# Throwaway routing-PREDICTOR diagnostics (expert-prefetch feasibility, part two).
#
# The question: can layer L+1's routed experts be read off the hidden state that is
# already available at layer L? If they can, layer L+1's PCIe fetch can be started
# while layer L is still computing, and the serialized fetch stops being serialized.
#
# What this records is the CEILING of the cheapest possible predictor: layer L+1's OWN
# router, scored on layer L's router input. It is not a proposal for a mechanism -- it
# is the measurement that says whether any mechanism could work.
#
# Enabled only when FREETOKEN_MOE_PREDICT_LOG names a directory; the single cached
# module-level check below is the entire cost when it is unset. Prefill/eager only:
# the capture guard below self-disables the whole thing under CUDA graph capture, and
# a captured decode never re-enters this code.
# ----------------------------------------------------------------------
_PREDICT_LOG_DIR = os.getenv("FREETOKEN_MOE_PREDICT_LOG") or None
_PREDICT_DUMP = os.getenv("FREETOKEN_MOE_PREDICT_DUMP") == "1"
_PREDICT_TOPK = 20  # how deep the prediction is scored (recall@10/@15/@20 offline)
_PREDICT_DUMP_STRIDE = 4  # keep every 4th token's router input for offline training
_PREDICT_FLUSH_EVERY = 50
_predict_routers: "dict[int, object]" = {}
_predict_records: list = []
_predict_path: str | None = None
_predict_dump: "dict[int, dict]" = {}
_predict_dump_last_layer = -1
_predict_dump_chunk = 0


def register_predict_router(layer_id, gate) -> None:
    """Register layer ``layer_id``'s router (gate) module for the prediction study.

    A registry populated at model construction rather than a walk back up to the parent
    model: an MoE layer has no handle on its siblings and nothing in serving wants one.
    It holds the MODULE (never a weight copy), so ``_predict_topk`` scores with the real
    router; and it is populated only while the study is armed, so an unarmed process
    keeps an empty dict and one ``is None`` test per model layer at construction.

    The same registry now backs the PRODUCTION layer-ahead prefetch
    (``FREETOKEN_MOE_PREFETCH=1``): the mechanism the study measured needs exactly the same
    handle on the next layer's router, so it reads the same dict rather than growing a
    second one that could drift out of sync with it.
    """
    if layer_id is None or (_PREDICT_LOG_DIR is None and not _prefetch.PREFETCH.enabled):
        return
    _predict_routers[int(layer_id)] = gate


def _predict_flush() -> None:
    global _predict_path
    if not _predict_records:
        return
    if _predict_path is None:
        assert _PREDICT_LOG_DIR is not None
        os.makedirs(_PREDICT_LOG_DIR, exist_ok=True)
        _predict_path = os.path.join(_PREDICT_LOG_DIR, f"predict-log-{os.getpid()}.jsonl")
    batch, _predict_records[:] = list(_predict_records), []
    with open(_predict_path, "a", encoding="utf-8") as fh:
        for rec in batch:
            fh.write(json.dumps(rec) + "\n")


def _predict_topk(layer_id: int, x: torch.Tensor, renormalize: bool):
    """Top-``_PREDICT_TOPK`` experts of layer ``layer_id``'s router, scored on ``x``.

    Deliberately the real path: the registered gate module, then ``fused_topk`` with the
    calling layer's own ``renormalize`` -- the same scoring function the live router runs,
    not a re-implementation of it. Only the ids are kept.
    """
    gate = _predict_routers.get(layer_id)
    if gate is None:
        return None
    logits = gate.forward(x)
    topk = min(_PREDICT_TOPK, int(logits.shape[-1]))
    _, ids = fused_topk(
        hidden_states=x, gating_output=logits, topk=topk, renormalize=renormalize
    )
    return ids


def _predict_ids(ids) -> list:
    if ids is None:
        return []
    return ids.detach().reshape(ids.shape[0], -1).cpu().tolist()


def _predict_dump_flush() -> None:
    """Write the pending forward's per-layer training dumps and drop them.

    Stitching is why the dump is buffered at all: layer L's record wants the ACTUAL top-10
    of layers L+1 and L+2, which are only routed later in the same forward. Exactly one
    forward's worth is ever held (``_predict_dump_note`` flushes when a new forward starts),
    so memory stays bounded by one prefill chunk.
    """
    global _predict_dump_chunk
    pending = dict(_predict_dump)
    _predict_dump.clear()
    if not pending:
        return
    assert _PREDICT_LOG_DIR is not None
    os.makedirs(_PREDICT_LOG_DIR, exist_ok=True)
    chunk = _predict_dump_chunk
    _predict_dump_chunk += 1
    for layer_id, entry in sorted(pending.items()):
        nxt = pending.get(layer_id + 1)
        nxt2 = pending.get(layer_id + 2)
        if nxt is None or nxt2 is None:
            continue  # the top two layers have nothing to predict
        rows = entry["x"].shape[0]
        if nxt["top10"].shape[0] != rows or nxt2["top10"].shape[0] != rows:
            continue  # different forward shapes: not the same tokens
        path = os.path.join(_PREDICT_LOG_DIR, f"x_layer{layer_id:02d}_{chunk}.pt")
        torch.save(
            {
                "layer": layer_id,
                "num_experts": entry["num_experts"],
                "positions": entry["positions"],
                "x": entry["x"],
                "top10_l": entry["top10"],
                "top10_l1": nxt["top10"],
                "top10_l2": nxt2["top10"],
            },
            path,
        )


def _predict_dump_note(layer_id: int, x: torch.Tensor, topk_ids, positions, experts) -> None:
    """Buffer every ``_PREDICT_DUMP_STRIDE``-th token's router input for this layer."""
    global _predict_dump_last_layer
    if layer_id <= _predict_dump_last_layer:
        _predict_dump_flush()  # a new forward started; the previous one is complete
    _predict_dump_last_layer = layer_id
    step = _PREDICT_DUMP_STRIDE
    _predict_dump[layer_id] = {
        "num_experts": int(experts),
        "positions": list(positions[::step]),
        "x": x.detach()[::step].to(torch.float16).cpu(),
        "top10": topk_ids.detach()[::step].to(torch.int32).cpu(),
    }


def _predict_log(layer, x: torch.Tensor, topk_ids: torch.Tensor) -> None:
    """Record one MoE forward's routing decision beside the next two layers' predictions.

    ``x`` must be the tensor the router itself consumed, and ``topk_ids`` this layer's real
    decision -- so both hooks sit before ``ensure_experts``' in-place rewrite. Best effort:
    never raises.
    """
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        layer_id = int(layer.layer_id)
        pred_next = _predict_topk(layer_id + 1, x, layer.renormalize)
        pred_next2 = _predict_topk(layer_id + 2, x, layer.renormalize)
        batch = get_global_ctx().batch
        tokens = batch.input_ids.detach().reshape(-1).cpu().tolist()
        raw_positions = getattr(batch, "positions", None)
        positions = (
            [] if raw_positions is None else raw_positions.detach().reshape(-1).cpu().tolist()
        )
        actual = _predict_ids(topk_ids)
        if not positions:
            positions = list(range(len(actual)))
        _predict_records.append(
            {
                "layer": layer_id,
                "phase": batch.phase,
                "positions": positions,
                "tokens": tokens,
                "actual_top10": actual,
                "pred_next_top20": _predict_ids(pred_next),
                "pred_next2_top20": _predict_ids(pred_next2),
            }
        )
        if _PREDICT_DUMP:
            _predict_dump_note(layer_id, x, topk_ids, positions, layer.num_experts)
        if len(_predict_records) >= _PREDICT_FLUSH_EVERY:
            _predict_flush()
    except Exception:  # diagnostics must never break a forward
        pass


def _predict_flush_atexit() -> None:
    try:
        if _PREDICT_DUMP:
            _predict_dump_flush()
    except Exception:
        pass
    try:
        _predict_flush()
    except Exception:
        pass


if _PREDICT_LOG_DIR is not None:
    atexit.register(_predict_flush_atexit)


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        allocate_experts: bool = True,
        weight_format: str = "bf16",
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.weight_format = weight_format
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        if allocate_experts:
            self._alloc_resident_experts(intermediate_size_per_partition)

    def _alloc_resident_experts(self, intermediate_size_per_partition: int) -> None:
        """Allocate the resident (in-GPU) expert weights for ``self.weight_format``.

        The resident sibling of the offload bank schemas: each format owns its
        tensor layout here and its kernel branch in ``_resident_gemm``.
        """
        if self.weight_format == "fp8_block":
            # Stacked block-fp8 experts + bf16 per-128x128-block inverse scales.
            # Full (unpartitioned) intermediate size: this layout is TP=1-only.
            from freetoken.kernel.triton.fp8_block_linear import FP8

            blk = 128
            n, i, h = self.num_experts, self.intermediate_size, self.hidden_size
            self.gate_up_proj = torch.empty(n, 2 * i, h, dtype=FP8)
            self.gate_up_scale_inv = torch.empty(
                n, 2 * i // blk, h // blk, dtype=torch.bfloat16
            )
            self.down_proj = torch.empty(n, h, i, dtype=FP8)
            self.down_scale_inv = torch.empty(n, h // blk, i // blk, dtype=torch.bfloat16)
            return
        assert self.weight_format == "bf16", (
            f"no resident expert allocation for weight_format {self.weight_format!r}"
        )
        self.gate_up_proj = torch.empty(
            self.num_experts,
            2 * intermediate_size_per_partition,
            self.hidden_size,
        )
        self.down_proj = torch.empty(
            self.num_experts,
            self.hidden_size,
            intermediate_size_per_partition,
        )

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            return self._comm.all_reduce(hidden_states)
        return hidden_states

    def _run_experts(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Dense (in-GPU) expert compute for a precomputed routing decision."""
        return fused_experts_impl(
            hidden_states,
            self.gate_up_proj,
            self.down_proj,
            topk_weights,
            topk_ids,
            self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

    def _resident_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Kernel dispatch on ``self.weight_format`` -- the resident mirror of
        ``OffloadMoELayer._expert_gemm``'s ``cache.quant_format`` dispatch."""
        if self.weight_format == "fp8_block":
            # Prefill dequantizes the layer's experts to bf16 and runs the bf16
            # grouped GEMM; decode dequantizes only the routed rows.
            from freetoken.moe.fused_fp8_block import (
                fused_experts_decode_fp8_block,
                fused_experts_fp8_block,
            )

            if get_global_ctx().batch.is_prefill:
                return fused_experts_fp8_block(
                    hidden_states, self.gate_up_proj, self.gate_up_scale_inv,
                    self.down_proj, self.down_scale_inv,
                    topk_weights, topk_ids, self.num_experts,
                )
            return fused_experts_decode_fp8_block(
                hidden_states, self.gate_up_proj, self.gate_up_scale_inv,
                self.down_proj, self.down_scale_inv, topk_weights, topk_ids,
            )
        assert self.weight_format == "bf16", (
            f"no resident expert kernel for weight_format {self.weight_format!r}"
        )
        return self._run_experts(hidden_states, topk_weights, topk_ids)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        Same name and shape as ``OffloadMoELayer.routed_forward`` so a model with
        its own router calls ``experts.routed_forward(...)`` without knowing whether
        the experts are resident or offloaded. The shared contract is the offload
        one: ``topk_ids`` must be safe to mutate in place (the offload decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        The resident path does not mutate it today, but callers must not rely on
        that.
        """
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        if self.weight_format != "bf16":
            # Quantized resident experts: generic softmax router + format kernel.
            # The bf16 path below stays on ctx.moe_backend byte-for-byte.
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
            return self._maybe_all_reduce(
                self._resident_gemm(hidden_states, topk_weights, topk_ids)
            )
        ctx = get_global_ctx()
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        return self._maybe_all_reduce(final_hidden_states)


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            allocate_experts=False,
        )
        self.layer_id = layer_id
        self.offload_cache: OffloadMoeCache | None = None

    def _use_decode_movement(self, hidden_states: torch.Tensor) -> bool:
        """Choose expert movement without changing the batch's causal phase."""
        batch = get_global_ctx().batch
        if not getattr(batch, "mtp_verify", False):
            return batch.is_decode or self._small_prefill_moves_like_decode(
                hidden_states.shape[0]
            )
        if (
            not batch.is_prefill
            or len(batch.reqs) != 1
            or not 2 <= hidden_states.shape[0] <= _MAX_MTP_VERIFY_ROWS
        ):
            raise ValueError(
                "private MTP verification requires one prefill request and 2 to "
                f"{_MAX_MTP_VERIFY_ROWS} rows"
            )
        return True

    def _small_prefill_moves_like_decode(self, rows: int) -> bool:
        """Is this prefill batch narrow enough to fetch its routed experts instead of the bank?

        ``rows`` is the batch's TOTAL row count (every request's extend concatenated), which is
        exactly what the prefill streaming cost is independent of. Opt-in through
        ``FREETOKEN_MOE_SMALL_PREFILL_ROWS``; at the default 0 this returns False before
        touching anything, so the movement choice is byte-identical to before.

        The hard precondition is ``_MOVEMENT_ONLY_FORMATS``: for most quant formats the decode
        path is not merely a different way of getting the experts onto the GPU, it is different
        ARITHMETIC (see that constant), and swapping it under a prefill changes what the model
        computes. Formats outside the set are refused with a warning rather than silently
        answered differently.

        Given a format where the two paths are one kernel, everything else the decode path
        needs is already true of a prefill batch: it reads nothing off ``batch`` past this
        point, it only needs ``[M, H]`` hidden states with an ``[M, top_k]`` int32 ``topk_ids``
        it may rewrite in place, and it touches none of the prefill double-buffer bookkeeping
        (``begin_prefill`` / ``prefetch_prefill_layer`` / ``_invalidate_prefill_buffer`` /
        ``release_prefill_layer``). Skipping that bookkeeping is safe in both directions:
        nothing is claimed, so nothing is left unreleased, and the next streaming prefill
        re-establishes the whole thing at its layer 0. What it does do is evict LRU residents
        like any decode step -- which is the trade being made.
        """
        limit = _SMALL_PREFILL_ROWS
        if limit <= 0 or not 1 <= rows <= limit:
            return False
        cache = self.offload_cache
        if cache is None:
            return False
        fmt = getattr(cache, "quant_format", None)
        if fmt not in _MOVEMENT_ONLY_FORMATS:
            return _refuse_small_prefill(fmt)
        # The cpu/hybrid decode target sizes its C++ scratch and pinned IO sets once, from
        # max(max_running_req, cuda_graph_max_bs, spec batch_width); a wider submit runs past
        # them. The GPU target has no such bound.
        executor = getattr(cache, "cpu_executor", None)
        if executor is not None and rows > int(getattr(executor, "max_tokens", 0)):
            return False
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        if self._use_decode_movement(hidden_states):
            final_hidden_states = self.decode_forward(hidden_states, router_logits)
        else:
            final_hidden_states = self.prefill_forward(hidden_states, router_logits)
        return self._maybe_all_reduce(final_hidden_states)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        The entry point for models whose router does not fit ``fused_topk`` (sigmoid
        scores, selection bias, group-limited top-k, ...); identical to ``forward``
        past the router. ``topk_ids`` must be safe to mutate in place (decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        """
        if _ROUTE_LOG_DIR is not None:
            _route_log(self.layer_id, topk_ids)  # before ensure_experts' in-place rewrite
        if _PREDICT_LOG_DIR is not None:
            _predict_log(self, hidden_states, topk_ids)
        if self._use_decode_movement(hidden_states):
            out = self._decode_routed(hidden_states, topk_weights, topk_ids)
        else:
            out = self._prefill_routed(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        if _ROUTE_LOG_DIR is not None:
            _route_log(self.layer_id, topk_ids)  # before ensure_experts' in-place rewrite
        if _PREDICT_LOG_DIR is not None:
            _predict_log(self, hidden_states, topk_ids)
        return self._decode_routed(hidden_states, topk_weights, topk_ids)

    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        if _ROUTE_LOG_DIR is not None:
            _route_log(self.layer_id, topk_ids)
        if _PREDICT_LOG_DIR is not None:
            _predict_log(self, hidden_states, topk_ids)
        return self._prefill_routed(hidden_states, topk_weights, topk_ids)

    # ------------------------------------------------------------------
    # Data movement -- one decision tree for every quant format (the banks
    # registry makes the cache machinery bank-count agnostic). Decode loads
    # on demand; prefill streams whole layers, double-buffered when overlap
    # is enabled. The kernels only ever see bank views plus row indices;
    # which kernel runs is decided afterwards, in ``_expert_gemm``.
    # ------------------------------------------------------------------

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot
        ids in place (loading missing experts), then the GEMM reads the full slot
        cache. All device-side with fixed shapes, so the decode call is CUDA-graph
        capturable.

        For ``decode_target == "cpu"`` the experts are instead computed on the CPU
        (high RAM bandwidth) straight from the host banks: ship hidden/routing to
        pinned host memory, run the GEMV on the worker pool via host nodes, ship the
        result back. The GPU slot cache is untouched (topk_ids keep their raw expert
        ids), so no ``ensure_experts``/``copy_missing`` here.

        For a GPU-owned layer (``--moe-gpu-owned-layers``) the experts are already resident
        at position == expert id, so the ids pass through unmapped and no LRU state is
        touched; ``alphas_for_layer`` is the matching (position == expert id) scale lookup."""
        cache = self.offload_cache
        assert cache is not None
        if cache.is_cpu_layer(self.layer_id):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            return executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
        if cache.is_gpu_owned_layer(self.layer_id):
            # Every expert of this layer is already in VRAM at position == expert id, so
            # there is nothing to predict, fetch, evict or remap: hand the kernel the RAW
            # topk_ids and the resident banks. Fixed shapes over fixed addresses -- strictly
            # simpler than the streaming path, so CUDA-graph capture is unaffected.
            cache._note_decode_routing(self.layer_id, topk_ids)
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.resident_views(self.layer_id),
                n=None,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=False,
            )
        if cache.decode_target == "hybrid":
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids)
        # Layer-ahead expert prefetch. Join the previous layer's prefetch BEFORE this
        # layer's ensure: after that edge the prefetched rows are ordinary resident slots,
        # so ensure/copy_missing need no notion of an in-flight fill. One module-attribute
        # read is the entire cost while unarmed -- and it keeps the whole feature off the
        # slot-cache stand-ins other decode tests substitute here.
        armed = _prefetch.PREFETCH.enabled
        if armed and cache.prefetch_wait(self.layer_id):
            cache.prefetch_note_actual(self.layer_id, topk_ids)  # raw ids, pre-rewrite
        cache.ensure_experts(self.layer_id, topk_ids)
        cache.copy_missing()
        if armed:
            self._prefetch_next_layer(cache, hidden_states)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )

    def _prefetch_next_layer(
        self, cache: OffloadMoeCache, hidden_states: torch.Tensor
    ) -> None:
        """Start layer ``L+1``'s expert fetch now, on the side stream (prefetch armed only).

        The predictor is layer ``L+1``'s OWN router scored on layer ``L``'s router input --
        the cheapest thing that could work, and the one the offline study measured at
        recall@10 = 0.62. Deliberately the real path: the registered gate module, then
        ``fused_topk`` with this layer's ``renormalize``, so the prediction is produced by
        the same scoring function the live router will run one layer later.

        ``hidden_states`` must still be the router input, which is why this sits before
        ``_expert_gemm`` (the fused MoE kernels may write it in place). Everything here is
        fixed-shape and device-side, so it captures.
        """
        target = self.layer_id + 1
        if not cache.prefetch_ready(target):
            return
        gate = _predict_routers.get(target)
        if gate is None:
            return  # e.g. a dense layer sits at L+1, or the model registers no routers
        logits = gate.forward(hidden_states)
        k = min(_prefetch.PREFETCH.topk, int(logits.shape[-1]))
        _, pred_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=logits,
            topk=k,
            renormalize=self.renormalize,
        )
        cache.prefetch_experts(target, pred_ids)

    def _decode_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid decode: GPU computes cache hits + <=K freshly-fetched experts, the CPU
        computes the overflow misses, overlapped, then the partials merge.

        The CPU pool is kicked off (``decode_submit``) before the GPU PCIe fetch + GEMM so
        the CPU overflow GEMV runs concurrently with the GPU work. Capture-safe: the
        routing split is device-side elementwise and the CPU submit/sync are host nodes.
        Each route is computed exactly once -- the GPU weights are zeroed for CPU-assigned
        routes and the CPU ids are -1 for GPU-assigned routes (the C++ kernel skips id<0).
        """
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        raw = topk_ids.clone()  # raw expert ids for the CPU partial
        cache.ensure_experts_hybrid(self.layer_id, topk_ids)  # -> slot (hit/fetched) or -1
        on_gpu = topk_ids >= 0

        cpu_ids = torch.where(on_gpu, raw.new_full((), -1), raw).contiguous()
        pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)

        # Measurement knob: FREETOKEN_HYBRID_OVERLAP=0 syncs the CPU pool *before* the
        # PCIe fetch + GPU GEMM, serializing the two so an A/B isolates the overlap win.
        cpu_routed_early = (
            executor.decode_sync(pending) if not _HYBRID_OVERLAP else None
        )

        cache.copy_missing()
        gpu_slots = topk_ids.clamp_min(0)  # -1 -> slot 0 (zero-weighted below)
        gpu_w = torch.where(on_gpu, topk_weights, topk_weights.new_zeros(())).contiguous()
        gpu_routed = self._expert_gemm(
            cache,
            hidden_states,
            gpu_w,
            gpu_slots,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        cpu_routed = cpu_routed_early if not _HYBRID_OVERLAP else executor.decode_sync(pending)
        return gpu_routed + cpu_routed

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill movement: stream whole layers -- double-buffered behind the
        previous layer's GEMMs when ``prefill_overlap`` is on, else a synchronous
        ``materialize_layer``. In both, position == expert id, so the routing ids
        pass through unmapped."""
        cache = self.offload_cache
        assert cache is not None
        if cache.is_gpu_owned_layer(self.layer_id):
            # No overlap buffer, no materialize, no release: the layer is already resident.
            # Keep the double-buffer pipeline moving anyway -- the next streaming layer's
            # copy still has to start one layer early, and prefetch_prefill_layer is a quiet
            # no-op for an owned target.
            if cache.prefill_overlap:
                if self.layer_id == 0:
                    cache.begin_prefill()
                cache.prefetch_prefill_layer(self.layer_id + 1)
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.resident_views(self.layer_id),
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
        if cache.prefill_overlap:
            views = self._wait_prefill_overlap(cache)
            out = self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            cache.release_prefill_layer(self.layer_id)
            return out
        cache.materialize_layer(self.layer_id)
        cache.copy_missing()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(self.num_experts),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )

    def _wait_prefill_overlap(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, ...]:
        """Double-buffer choreography for this layer's overlap prefill: kick off the
        next layer's full-layer H2D copy, then return this layer's bank views (in
        bank registration order; buffer position == expert id, so routing ids pass
        through unmapped). The caller runs ``release_prefill_layer`` after its GEMMs.
        """
        if self.layer_id == 0:
            cache.begin_prefill()
        cache.prefetch_prefill_layer(self.layer_id)
        cache.prefetch_prefill_layer(self.layer_id + 1)
        return cache.wait_prefill_layer(self.layer_id)

    # ------------------------------------------------------------------
    # Kernel dispatch -- pure routing on the cache's quant format. ``views``
    # are the bank tensors the movement step produced (in bank registration
    # order) and ``topk_ids`` already index their rows.
    # ------------------------------------------------------------------

    def _expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        views: tuple[torch.Tensor, ...],
        n: int | None,
        alphas: tuple[torch.Tensor, torch.Tensor] | None,
        is_prefill: bool,
    ) -> torch.Tensor:
        fmt = cache.quant_format
        if fmt in ("nvfp4_marlin", "nvfp4_b12x"):
            # Borrowed W4A16 fused MoE -- Marlin (vLLM, sm_80-99) or b12x
            # (flashinfer, sm_120) over their pre-tiled banks; one kernel serves
            # prefill and decode, with the movement-matched per-row global scales.
            from freetoken.moe.nvfp4_backends import b12x_fused_experts, marlin_fused_experts

            assert alphas is not None
            gate_up_packed, gate_up_scale, down_packed, down_scale = views
            fused = marlin_fused_experts if fmt == "nvfp4_marlin" else b12x_fused_experts
            return fused(
                hidden_states,
                gate_up_packed,
                gate_up_scale,
                alphas[0],
                down_packed,
                down_scale,
                alphas[1],
                topk_weights,
                topk_ids,
                self.activation,
                self.apply_router_weight_on_input,
            )
        if fmt == "nvfp4":
            # FreeToken's Triton inline-dequant kernels over the native ModelOpt
            # rows: the FP4 banks are read directly in the GEMM, no BF16 copy of
            # the experts is ever materialized. The swigluoai scalars (MiniMax-M3)
            # live on the layer via make_moe_layer's extra_attrs (gpt-oss precedent)
            # and are ignored by the plain *_and_mul activations.
            act_alpha = getattr(self, "hidden_act_alpha", 1.702)
            act_limit = getattr(self, "swiglu_limit", None)
            # None == "no clamp" everywhere else in the repo (mxfp4 maps it to +inf).
            act_limit = float("inf") if act_limit is None else act_limit
            if is_prefill:
                from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

                return fused_experts_nvfp4(
                    hidden_states,
                    *views,
                    topk_weights,
                    topk_ids,
                    n,
                    self.activation,
                    self.apply_router_weight_on_input,
                    act_alpha,
                    act_limit,
                )
            # Marlin-style int32 wide-load GEMV (arithmetic dequant, no HW cvt).
            # Bit-identical to the byte-at-a-time path; lifts gate/up BW ~43%->51%
            # (I=512), ~41%->53% (I=768), 65%->72% (I=1536). CUDA-graph safe (fixed
            # shapes, no host sync).
            from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

            return fused_experts_decode_nvfp4_marlin(
                hidden_states,
                *views,
                topk_weights,
                topk_ids,
                self.activation,
                self.apply_router_weight_on_input,
                act_alpha,
                act_limit,
            )
        if fmt == "fp8_block":
            # Block-fp8 experts: fused inline-dequant grouped GEMM reads the routed fp8 rows
            # directly (fp8 banks halve host/cache bytes; no bf16 materialization).
            from freetoken.moe.fused_fp8_block import (
                fused_experts_decode_fp8_block,
                fused_experts_fp8_block,
            )

            gate_up, gate_up_scale, down, down_scale = views
            if is_prefill:
                return fused_experts_fp8_block(
                    hidden_states, gate_up, gate_up_scale, down, down_scale,
                    topk_weights, topk_ids, n, self.activation,
                    self.apply_router_weight_on_input,
                )
            return fused_experts_decode_fp8_block(
                hidden_states, gate_up, gate_up_scale, down, down_scale,
                topk_weights, topk_ids, self.activation, self.apply_router_weight_on_input,
            )
        if fmt == "q4_0":
            # Native GGUF Q4_0 experts: dequant-in-kernel grouped GEMV (MMVQ) over the
            # streamed packed banks; topk_ids already index the cache slots / layer.
            from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

            gate_up, down = views
            return fused_experts_gguf_q4_0(
                hidden_states, gate_up, down, topk_weights, topk_ids, self.activation
            )
        if fmt == "mxfp4_triton":
            # gpt-oss MXFP4 experts (biased, clamped swiglu): transposed split-K GEMV
            # decode + grouped `_t` prefill. The swiglu scalars live on the layer
            # (set at construction), not in the base signature.
            from freetoken.moe.fused_mxfp4 import (
                run_mxfp4_prefill_experts_t,
                run_mxfp4_splitk_decode_experts,
            )

            gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias = views
            run = run_mxfp4_prefill_experts_t if is_prefill else run_mxfp4_splitk_decode_experts
            return run(
                hidden_states, topk_weights, topk_ids,
                gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias,
                top_k=self.top_k,
                hidden_act_alpha=self.hidden_act_alpha,
                swiglu_limit=self.swiglu_limit,
            )
        if fmt == "ds_fp4":
            # DeepSeek-V4 FP4 experts: grouped inline-dequant GEMM for streaming
            # prefill chunks (n = bank rows to sort over); per-route dequant GEMV
            # for decode and the sparse small-chunk slot path (n is None there,
            # and sorting over the full slot cache would drown in padding).
            gate_up_packed, gate_up_scale, down_packed, down_scale = views
            if is_prefill and n is not None:
                from freetoken.moe.fused_ds_fp4 import routed_experts_fp4_prefill

                return routed_experts_fp4_prefill(
                    hidden_states, topk_ids, topk_weights,
                    gate_up_packed, gate_up_scale, down_packed, down_scale,
                    self.swiglu_limit, n,
                )
            from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

            return routed_experts_fp4(
                hidden_states, topk_ids, topk_weights,
                gate_up_packed, gate_up_scale, down_packed, down_scale,
                self.swiglu_limit,
            )
        assert fmt == "bf16", f"unknown quant_format {fmt!r}"
        gate_up, down = views
        impl = fused_experts_impl if is_prefill else fused_experts_decode_impl
        return impl(
            hidden_states,
            gate_up,
            down,
            topk_weights,
            topk_ids,
            self.activation,
            self.apply_router_weight_on_input,
        )


def make_moe_layer(
    config: "ModelConfig",
    *,
    layer_id: int | None = None,
    activation: str = "silu",
    weight_format: str = "bf16",
    renormalize: bool | None = None,
    apply_router_weight_on_input: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    resident_cls: type[MoELayer] | None = None,
    offload_cls: "type[OffloadMoELayer] | None" = None,
    extra_attrs: dict | None = None,
) -> MoELayer:
    """Build the experts layer for ``config.moe_backend`` -- the one construction
    seam between a model and the MoE strategy.

    Picks ``OffloadMoELayer`` for the offload family (offload/cpu/hybrid, which
    ignore ``weight_format``: the quant format comes from the offload cache) and
    ``MoELayer`` otherwise. Geometry defaults come from ``config``; pass overrides
    for models whose fields deviate. ``extra_attrs`` become instance attributes --
    the seam for per-format scalars the base signature does not carry (e.g.
    ``hidden_act_alpha``/``swiglu_limit``, read back via ``getattr`` by the engine
    and format kernels). ``resident_cls``/``offload_cls`` keep model-specific
    subclasses constructible through the same seam.
    """
    offload = is_offload_moe_backend(config.moe_backend)
    layer_cls = (offload_cls or OffloadMoELayer) if offload else (resident_cls or MoELayer)
    kwargs = dict(
        num_experts=num_experts if num_experts is not None else config.num_experts,
        top_k=top_k if top_k is not None else config.num_experts_per_tok,
        hidden_size=hidden_size if hidden_size is not None else config.hidden_size,
        intermediate_size=(
            intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        ),
        renormalize=renormalize if renormalize is not None else config.norm_topk_prob,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    if offload:
        assert layer_id is not None, "offload MoE backends need the layer_id"
        kwargs["layer_id"] = layer_id
    else:
        kwargs["weight_format"] = weight_format
    layer = layer_cls(**kwargs)
    for name, value in (extra_attrs or {}).items():
        setattr(layer, name, value)
    return layer
