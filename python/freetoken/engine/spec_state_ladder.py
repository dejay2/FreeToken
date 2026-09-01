"""Speculative linear-state rollback -- design section 4, Strategy R.

A speculative step forwards ``w = 1 + drafts`` rows through the request's REAL GDN slot, so
every linear state comes back holding all ``w`` rows while only the accepted prefix survives.
There is no state-only entry point into the model and the chunked prefill kernel exposes no
per-row transition operator, so the slot cannot be wound back from itself. It can be REBUILT:

  * snapshot the whole slot into a spare pool slot before the forward (one ``copy_from``,
    conv + recurrent + every declared sibling state);
  * during the forward, stash each linear layer's per-row inputs -- the post-conv q|k|v, the
    raw gates ``a``/``b``, and the raw conv/PLE conv inputs. Every one of these depends only on
    rows <= its own, so the accepted rows' values are exactly what plain decode would have
    produced. They are COPIED into a preallocated arena: the forward reuses its activation
    buffers, and a reference would go stale a layer later;
  * on settle, restore the snapshot and re-advance the accepted rows through
    ``fused_sigmoid_gating_delta_rule_update`` -- the same kernel plain decode uses, at
    ``T = accepted`` -- while the conv and PLE states, which are shift registers of raw inputs,
    are rewritten by pure indexing.

Every rung of that ladder is bit-exact (``tests/models/qwen4_exp/test_mtp_state_ladder.py``):
the varlen conv reproduces successive decode convs, the recurrent kernel at ``T = n`` reproduces
``n`` launches at ``T = 1`` because the state round-trips through fp32 memory, and the shift
registers are plain copies.

FULL ACCEPTANCE REPLAYS TOO. It is tempting to keep the forward's own state when every row is
accepted, but that state came from the CHUNKED kernel -- a different (equally valid)
approximation of the same recurrence, measurably off the decode ladder. Full acceptance is the
common case under speculation, so skipping the replay would put nearly every state advance on a
different ladder from plain decode and spend the greedy-equivalence gate for ~0.5 % of a cycle.

``rollback`` is the ``state_rollback`` callable ``Scheduler._rollback_spec_tokens`` takes; it
runs after the lengths are final, because a stop condition can still shrink the accepted run.

THE SETTLE IS LAUNCH-BOUND, NOT BANDWIDTH-BOUND. Every tensor it touches is a few hundred KB;
what it costs is HOST LAUNCHES, one per GDN layer in the recurrent replay (36 on Qwen3.8) plus
the whole-slot copies around them. Two things cut that:

  * the replay's inputs all live at addresses the ladder allocated once and never moves (the
    stash arena, the state pool, the layer's own ``A_log``/``dt_bias``), and the only per-step
    variable is WHICH slot to advance -- so the ``steps``-row replay records into a CUDA graph
    keyed on ``steps`` alone, and a settle issues one replay instead of 36 launches. Off with
    ``FREETOKEN_MTP_SPEC_LADDER_GRAPH=0``, which restores the eager per-layer loop exactly;
  * a settle that accepts anything REWRITES the conv and PLE shift registers whole, so
    restoring their snapshot first is three strided copies whose every byte is overwritten a
    moment later. ``rollback`` restores only the recurrent state (the replay's own starting
    point) and reads the shift registers' pre-step halves straight out of the snapshot slot.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Mapping

import torch

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.kvcache.linear_state_pool import LinearStatePool

logger = init_logger(__name__)

LADDER_GRAPH_ENV = "FREETOKEN_MTP_SPEC_LADDER_GRAPH"


def ladder_graph_enabled(env: "Mapping[str, str] | None" = None) -> bool:
    """Whether the recurrent replay may be recorded into a CUDA graph (default: yes)."""
    env = os.environ if env is None else env
    raw = (env.get(LADDER_GRAPH_ENV, "1") or "1").strip()
    if raw not in {"0", "1"}:
        raise ValueError(f"{LADDER_GRAPH_ENV} must be 0 or 1, got {raw!r}")
    return raw == "1"


class SpecStateLadder:
    """Snapshot / stash / replay for one request's linear state across a speculative step.

    Sized once, at speculation-enable time: one spare pool slot plus a per-layer activation
    arena for ``max_width`` rows. Nothing on the settle path allocates.
    """

    def __init__(self, pool: "LinearStatePool", max_width: int) -> None:
        from freetoken.models.qwen4_exp.config import PLE_CONV_STATE, PLE_NGRAM_STATE

        if max_width < 1:
            raise ValueError(f"max_width must be at least 1, got {max_width}")
        if pool.recurrent_states.dtype is not torch.float32:
            # The replay runs T = accepted steps in ONE launch, holding the state in fp32
            # registers throughout; plain decode runs one launch per step and rounds to the
            # pool dtype in between. The two coincide only when that round trip is lossless.
            raise ValueError(
                "SpecStateLadder needs an fp32 recurrent state to replay bit-exactly, got "
                f"{pool.recurrent_states.dtype} (FREETOKEN_MAMBA_SSM_DTYPE)"
            )
        unknown = sorted(set(pool.slot_states) - {PLE_CONV_STATE, PLE_NGRAM_STATE})
        if unknown:
            # A sibling state rides the same slot and would be left holding the rejected rows.
            raise ValueError(
                f"SpecStateLadder cannot roll back slot states {unknown}; teach it their "
                "shift-register shape before speculating on this model"
            )
        self.pool = pool
        self.max_width = max_width
        self._ple_conv_name = PLE_CONV_STATE
        self._ngram_name = PLE_NGRAM_STATE

        device = pool.device
        n_layers, num_slots, conv_dim, self._km1 = pool.conv_states.shape
        _, _, v_heads, key_dim, value_dim = pool.recurrent_states.shape
        k_heads, rem = divmod(conv_dim - v_heads * value_dim, 2 * key_dim)
        assert rem == 0 and k_heads > 0, (
            f"conv dim {conv_dim} is not 2*k_heads*{key_dim} + {v_heads}*{value_dim}"
        )
        self._splits = (k_heads * key_dim, 2 * k_heads * key_dim)
        self._k_heads, self._v_heads = k_heads, v_heads
        self._key_dim, self._value_dim = key_dim, value_dim

        dtype = pool.conv_states.dtype
        # conv history = [restored state | this step's raw conv inputs]; the state after row j
        # is the (kernel-1)-wide window ending at j, so both halves live in one buffer.
        self._conv_hist = torch.empty(
            (n_layers, conv_dim, self._km1 + max_width), dtype=dtype, device=device
        )
        self._mixed = torch.empty((n_layers, max_width, conv_dim), dtype=dtype, device=device)
        self._a = torch.empty((n_layers, max_width, v_heads), dtype=dtype, device=device)
        self._b = torch.empty_like(self._a)
        self._params: list[tuple[torch.Tensor, torch.Tensor, float] | None] = [None] * n_layers
        # cu_seqlens for every replay length, so a settle needs no host->device copy
        self._cu = torch.stack(
            [
                torch.tensor([0, t], dtype=torch.int64, device=device)
                for t in range(max_width + 1)
            ]
        )
        self._slot_ids = torch.arange(num_slots, dtype=torch.int32, device=device)
        # The graphed replay's ONE moving input. A graph bakes the address of its index tensor,
        # not the slot id inside it, so the replay reads this fixed cell and a settle refills it
        # with a single-element device copy off ``_slot_ids``.
        self._graph_slot = torch.empty(1, dtype=torch.int32, device=device)
        self._graph_enabled = device.type == "cuda" and ladder_graph_enabled()
        self._graphs: dict[int, "torch.cuda.CUDAGraph"] = {}
        # the kernel's per-call output tensor is allocated from the graph's private pool and
        # written on every replay, so the capture's copy has to stay alive with the graph
        self._graph_outputs: dict[int, list[torch.Tensor]] = {}
        self._graph_stream = None

        self._ple_rows: dict[int, int] = {}
        self._ple_hist = None
        if pool.has_slot_state(PLE_CONV_STATE):
            states = pool.slot_states[PLE_CONV_STATE]
            n_ple, _, width, self._ple_state_len = states.shape
            self._ple_rows = {
                layer_id: row
                for row, layer_id in enumerate(pool.slot_state_layer_ids(PLE_CONV_STATE))
            }
            self._ple_hist = torch.empty(
                (n_ple, width, self._ple_state_len + max_width),
                dtype=states.dtype,
                device=device,
            )
        self._ngram_hist = None
        if pool.has_slot_state(PLE_NGRAM_STATE):
            states = pool.slot_states[PLE_NGRAM_STATE]
            self._ngram_len = states.shape[-1]
            self._ngram_hist = torch.empty(
                self._ngram_len + max_width, dtype=states.dtype, device=device
            )

        self.slot = pool.alloc(1)[0]
        self._live: int | None = None
        self._width = 0

    def rebind(self) -> None:
        """Take a fresh snapshot slot after ``LinearStatePool.rebuild`` reset the free list.
        Idle-only, like the rebuild itself; the ladder holds no state between steps."""
        self._slot_ids = torch.arange(
            self.pool.num_slots, dtype=torch.int32, device=self.pool.device
        )
        # ``rebuild`` REPLACED the pool's state tensors, so every recorded replay is pointing at
        # freed storage. Drop them; the next settle re-records against the new addresses.
        self._drop_graphs()
        self.slot = self.pool.alloc(1)[0]
        self._live = None
        self._width = 0

    # ------------------------------------------------------------------ before the forward

    def begin(self, req: "Req", batch: "Batch") -> None:
        """Snapshot the request's live slot and arm the per-layer capture on ``batch``."""
        width = batch.emit_width
        if not 1 <= width <= self.max_width:
            raise ValueError(
                f"speculative width {width} outside the ladder's arena (1..{self.max_width})"
            )
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        self.pool.copy_from(slot, self.slot)
        self._live = slot
        self._width = width
        # ``_params`` deliberately survives the step. A captured verify graph bakes the stash
        # hooks' COPIES but never re-runs their Python, so a replay step records nothing here;
        # what they record is the layer's own ``A_log``/``dt_bias`` tensors and a constant
        # scale, which never change, so carrying them forward is both necessary and correct.
        # ``rollback``'s per-layer assertion still catches a hook that never ran at all.
        if self._ngram_hist is not None:
            ids = batch.input_ids
            assert ids is not None and ids.numel() == width, (
                f"a {width}-row speculative batch must carry {width} input ids"
            )
            self._ngram_hist[self._ngram_len : self._ngram_len + width].copy_(ids)
        batch.spec_capture = self

    @contextmanager
    def borrow_snapshot(self, req: "Req"):
        """Snapshot ``req``'s slot for a capture that is NOT a speculative step, and yield the
        wind-back.

        ``begin`` is the speculative step's snapshot and also arms the per-layer stash, sized
        to that step's width. The width-1 capture-decode graph needs only the undo: its warm-up
        pass advances the slot by one row, and the recorded pass -- which is where the step's
        real values come from -- must start where the warm-up did. Nothing is stashed and no
        rollback follows, so the borrow ends when the capture does.
        """
        if self._live is not None:
            raise RuntimeError("a speculative step is already in flight on this ladder")
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        self.pool.copy_from(slot, self.slot)
        self._live = slot
        self._width = 0
        try:
            yield self.restore_snapshot
        finally:
            self._live = None
            self._width = 0

    # ------------------------------------------------------------------ during the forward

    def stash_gdn(
        self,
        layer_id: int,
        *,
        conv_in: torch.Tensor,
        mixed: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
    ) -> None:
        """Copy one GDN layer's per-row replay inputs out of the forward's activations."""
        width = self._width
        assert conv_in.shape[0] == width, (
            f"speculation forwards one request; got {conv_in.shape[0]} rows for width {width}"
        )
        li = self.pool.local_index(layer_id)
        self._conv_hist[li, :, self._km1 : self._km1 + width].copy_(conv_in.transpose(0, 1))
        self._mixed[li, :width].copy_(mixed)
        self._a[li, :width].copy_(a)
        self._b[li, :width].copy_(b)
        self._params[li] = (A_log, dt_bias, scale)

    def stash_ple(self, layer_id: int, x: torch.Tensor) -> None:
        """Copy one PLE layer's per-row conv inputs (the state is their shift register)."""
        if self._ple_hist is None:
            return
        width = self._width
        row = self._ple_rows[layer_id]
        self._ple_hist[row, :, self._ple_state_len : self._ple_state_len + width].copy_(
            x.transpose(0, 1)
        )

    # ------------------------------------------------------------------------- on settle

    def restore_snapshot(self) -> None:
        """Put the live slot back where ``begin`` left it, WITHOUT ending the step.

        This is the seam CUDA-graph capture needs. Capture runs the forward twice: a warm-up
        pass that EXECUTES (and so advances this slot by ``w`` rows) and a recorded pass that
        executes nothing at all. Between them the slot has to go back, or the replay that
        actually produces the step's values would start one whole speculative width late.

        Nothing else the forward writes needs this: the KV store, the compressed slab and the
        pending rings are all position-addressed and re-derive from the same inputs.
        """
        if self._live is None:
            raise RuntimeError("no speculative step is in flight on this ladder")
        self.pool.copy_from(self.slot, self._live)

    def _restore_recurrent(self) -> None:
        """The recurrent half of ``restore_snapshot`` -- the replay's starting state, and the
        only half a settle that accepts anything has to put back (see the module docstring)."""
        if self._live is None:
            raise RuntimeError("no speculative step is in flight on this ladder")
        self.pool.recurrent_states[:, self._live].copy_(
            self.pool.recurrent_states[:, self.slot]
        )

    def rollback(self, req: "Req", accepted: int) -> None:
        """Leave the slot holding exactly ``accepted`` of the step's rows.

        The resting invariant is the one plain decode keeps: between steps the linear state has
        consumed positions ``0 .. cached_len - 1``. The step's row ``i`` consumed position
        ``cached_len_before + i``, and settling leaves ``cached_len = cached_len_before +
        accepted``, so the state must end as if rows ``0 .. accepted - 1`` had been consumed --
        no more, no fewer.
        """
        if self._live is None:
            raise RuntimeError("no speculative step is in flight on this ladder")
        if not 0 <= accepted <= self._width:
            raise ValueError(f"accepted must be 0..{self._width} rows, got {accepted}")
        slot = self._live
        if accepted:
            # Only the recurrent state is restored: ``_replay_conv`` / ``_replay_ple`` rewrite
            # the conv and PLE shift registers WHOLE out of the snapshot slot, so restoring
            # those first would be three strided copies nothing ever reads.
            self._restore_recurrent()
            self._replay_recurrent(slot, accepted)
            self._replay_conv(slot, accepted)
            self._replay_ple(slot, accepted)
        else:
            # The full undo keeps ``copy_from``'s exact semantics -- boot capture's
            # ``rollback(req, 0)`` is the wind-back for a pass that produced nothing.
            self.restore_snapshot()
        self._live = None
        self._width = 0

    # -------------------------------------------------------------- the recurrent replay

    def _launch_recurrent(
        self, indices: torch.Tensor, steps: int
    ) -> list[torch.Tensor]:
        """One ``fused_sigmoid_gating_delta_rule_update`` per GDN layer, advancing ``steps``
        rows of the slot named by ``indices``. Every other input is at a fixed address, which
        is what makes the whole loop capturable."""
        from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

        cu_seqlens = self._cu[steps]
        q_end, k_end = self._splits
        outputs = []
        for li, params in enumerate(self._params):
            assert params is not None, (
                f"GDN layer index {li} never stashed; the capture hook did not run"
            )
            A_log, dt_bias, scale = params
            mixed = self._mixed[li, :steps]
            outputs.append(
                fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    a=self._a[li, :steps],
                    dt_bias=dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=mixed[:, :q_end].view(1, steps, self._k_heads, self._key_dim),
                    k=mixed[:, q_end:k_end].view(1, steps, self._k_heads, self._key_dim),
                    v=mixed[:, k_end:].view(1, steps, self._v_heads, self._value_dim),
                    b=self._b[li, :steps],
                    initial_state_source=self.pool.recurrent_states[li],
                    initial_state_indices=indices,
                    scale=scale,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                )
            )
        return outputs

    def _replay_recurrent(self, slot: int, steps: int) -> None:
        graph = self._graphs.get(steps)
        if graph is None and self._graph_enabled:
            graph = self._capture_replay(steps)
        if graph is None:
            self._launch_recurrent(self._slot_ids[slot : slot + 1], steps)
            return
        self._graph_slot.copy_(self._slot_ids[slot : slot + 1])
        graph.replay()

    def _drop_graphs(self) -> None:
        self._graphs.clear()
        self._graph_outputs.clear()

    def _capture_replay(self, steps: int):
        """Record the ``steps``-row replay once, or fall back to the eager loop for good.

        Capture's warm-up EXECUTES the replay against the live slot, so the recurrent state is
        wound back to the snapshot between the two passes -- exactly where ``rollback`` handed
        it over -- and the recorded pass runs nothing. An in-capture failure poisons the CUDA
        context for retries, so nothing is retried: graphing switches off and the ladder is the
        pre-graph ladder, which is also what ``FREETOKEN_MTP_SPEC_LADDER_GRAPH=0`` selects.
        """
        device = self.pool.device
        if torch.cuda.is_current_stream_capturing():
            # a capture is already recording (the verify graph); nesting is illegal
            return None
        entry_stream = torch.cuda.current_stream(device)
        if self._graph_stream is None:
            self._graph_stream = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        warmed = False
        try:
            self._graph_slot.copy_(self._slot_ids[self._live : self._live + 1])
            # warm up ON the stream that gets captured, so its lazy per-stream resources
            # (the triton module, its launch metadata) exist before recording
            self._graph_stream.wait_stream(entry_stream)
            warmed = True
            with torch.cuda.stream(self._graph_stream):
                self._launch_recurrent(self._graph_slot, steps)
            entry_stream.wait_stream(self._graph_stream)
            torch.cuda.synchronize(device)
            self._restore_recurrent()
            # a private pool per graph: these replay in whatever order acceptance dictates,
            # which is exactly what a shared pool does not allow
            with torch.cuda.graph(
                graph,
                stream=self._graph_stream,
                # background threads (ple-mmap staging, the CPU-MoE watchdog) must not
                # invalidate this capture from outside
                capture_error_mode="thread_local",
            ):
                outputs = self._launch_recurrent(self._graph_slot, steps)
            torch.cuda.synchronize(device)
        except Exception as exc:  # pragma: no cover - needs a live CUDA context
            torch.cuda.set_stream(entry_stream)
            self._graph_enabled = False
            self._drop_graphs()
            if warmed:
                # the warm-up may have advanced the live state before it fell over
                self._restore_recurrent()
            logger.warning(
                "spec ladder replay capture failed at %d step(s) (%s: %s); "
                "falling back to the eager per-layer replay",
                steps, type(exc).__name__, " ".join(str(exc).split())[:200],
            )
            return None
        self._graphs[steps] = graph
        self._graph_outputs[steps] = outputs
        return graph

    def _replay_conv(self, slot: int, steps: int) -> None:
        km1 = self._km1
        # the pre-step window comes from the SNAPSHOT, so the live slot needs no restore first
        self._conv_hist[:, :, :km1].copy_(self.pool.conv_states[:, self.slot])
        self.pool.conv_states[:, slot].copy_(self._conv_hist[:, :, steps : steps + km1])

    def _replay_ple(self, slot: int, steps: int) -> None:
        if self._ple_hist is not None:
            state_len = self._ple_state_len
            states = self.pool.slot_states[self._ple_conv_name]
            self._ple_hist[:, :, :state_len].copy_(states[:, self.slot])
            states[:, slot].copy_(self._ple_hist[:, :, steps : steps + state_len])
        if self._ngram_hist is not None:
            ctx_len = self._ngram_len
            states = self.pool.slot_states[self._ngram_name]
            self._ngram_hist[:ctx_len].copy_(states[0, self.slot])
            states[0, slot].copy_(self._ngram_hist[steps : steps + ctx_len])


__all__ = ["LADDER_GRAPH_ENV", "SpecStateLadder", "ladder_graph_enabled"]
