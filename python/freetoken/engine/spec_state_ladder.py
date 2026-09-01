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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.kvcache.linear_state_pool import LinearStatePool


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
        self.restore_snapshot()
        if accepted:
            self._replay_recurrent(slot, accepted)
            self._replay_conv(slot, accepted)
            self._replay_ple(slot, accepted)
        self._live = None
        self._width = 0

    def _replay_recurrent(self, slot: int, steps: int) -> None:
        from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

        indices = self._slot_ids[slot : slot + 1]
        cu_seqlens = self._cu[steps]
        q_end, k_end = self._splits
        for li, params in enumerate(self._params):
            assert params is not None, (
                f"GDN layer index {li} never stashed; the capture hook did not run"
            )
            A_log, dt_bias, scale = params
            mixed = self._mixed[li, :steps]
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

    def _replay_conv(self, slot: int, steps: int) -> None:
        km1 = self._km1
        self._conv_hist[:, :, :km1].copy_(self.pool.conv_states[:, slot])
        self.pool.conv_states[:, slot].copy_(self._conv_hist[:, :, steps : steps + km1])

    def _replay_ple(self, slot: int, steps: int) -> None:
        if self._ple_hist is not None:
            state_len = self._ple_state_len
            states = self.pool.slot_states[self._ple_conv_name]
            self._ple_hist[:, :, :state_len].copy_(states[:, slot])
            states[:, slot].copy_(self._ple_hist[:, :, steps : steps + state_len])
        if self._ngram_hist is not None:
            ctx_len = self._ngram_len
            states = self.pool.slot_states[self._ngram_name]
            self._ngram_hist[:ctx_len].copy_(states[0, slot])
            states[0, slot].copy_(self._ngram_hist[steps : steps + ctx_len])


__all__ = ["SpecStateLadder"]
