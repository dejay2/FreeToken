"""Fixed-width CUDA graphs for the ``w``-row speculative target forward.

Two callers need the same graph and disagree about almost nothing:

* the shadow observer's private checker (``MTPVerifyGraphRunner``), which forwards a leased
  scratch page range through a shadow GDN slot and wants logits back;
* the integrated speculative step (``SpecVerifyGraphRunner``), which forwards the request's
  REAL page-table row through its REAL linear slot and wants logits AND the ``w`` hidden rows,
  because the draft head consumes them next cycle.

What they share is the hard part, and it is all in ``_FixedWidthGraphRunner``: capture that
survives its own failure. Torch 2.11's ``graph.__exit__`` skips the stream restore when
``capture_end`` raises, leaving the thread on a dead capture stream with the allocator pointed
at a dead pool -- so the entry stream is restored by hand, the body's exception is preserved
before ``capture_end`` masks it, capture runs ``thread_local`` so the ple-mmap staging thread
cannot invalidate it from outside, the warm-up runs ON the capture stream so per-stream lazy
resources exist before recording, an in-capture failure is classified permanently-unsupported
rather than retried into a re-poisoned context, and the context is probed afterwards.

WHAT THE INTEGRATED RUNNER ADDS
-------------------------------
**A hidden-state slot.** The historical blocker: the observer's buffer had nowhere to put the
pre-mix stream, so the integrated path had no way to feed the draft head from a graph. Both
output buffers are allocated from the warm-up pass's own tensors, so their dtypes are the
model's rather than a guess -- which is what lets the offline gate compare replay against eager
with ``torch.equal``.

**A restore seam.** Capture's warm-up pass EXECUTES the forward; the recorded pass does not
execute at all. Against a live request that means the warm-up advances the GDN slot (and the
PLE conv / n-gram context riding it) by ``w`` rows before a single real value has been
produced. ``capture(..., restore_state=...)`` calls back between the two passes -- the state
ladder's snapshot is already sitting there -- and again on any failure, so the eager fallback
also starts from the right state. Everything else the step writes is position-addressed and
idempotent: the KV store, the compressed slab, and the pending rings all re-derive from the
same inputs.

Sampling stays outside: the graph's outputs are fixed buffers the caller reads.
"""

from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

_GRAPH_CAPTURE_RESERVE_FLOOR = 64 << 20
# the graph runner has no config handle, so the capture-failure traceback lands in the
# approved private evidence dir by absolute path; every write is best-effort
_CAPTURE_FAILURE_EVIDENCE_DIR = Path(
    r"D:\FreeToken-ple-mmap-vision\.local\mtp-spike\evidence"
)
# a retryable capture (memory admission, a transient allocator failure) is worth another go on
# a later cycle, but not every cycle: a doomed width would otherwise pay a warm-up forward per
# step forever.
_MAX_RETRYABLE_ATTEMPTS = 3


@dataclass(frozen=True)
class MTPGraphCaptureResult:
    width: int
    status: str
    reason: str
    memory_bytes: int = 0
    synchronizations: int = 0


def _render_widths(widths: Sequence[int]) -> str:
    values = [str(width) for width in widths]
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])}, or {values[-1]}"


class _GraphInputBuffer:
    """The per-replay inputs, at addresses the graph bakes once.

    Everything here is refilled every replay, because in integrated mode every one of them
    moves: the ids are this cycle's confirmed row plus its drafts, ``positions`` and
    ``rope_positions`` follow ``cached_len``, ``out_loc`` is re-read from the live page table
    (the step may have straddled a page boundary), and ``linear_table_idx`` is whatever slot
    the request holds now.
    """

    def __init__(self, width: int, device: torch.device) -> None:
        self.input_ids = torch.empty(width, dtype=torch.int32, device=device)
        self.positions = torch.empty(width, dtype=torch.int32, device=device)
        self.out_loc = torch.empty(width, dtype=torch.int32, device=device)
        self.rope_positions = torch.empty((3, width), dtype=torch.int64, device=device)
        self.linear_table_idx = torch.empty(1, dtype=torch.int32, device=device)
        # int64 so the GDN kernels' `.to(torch.int64)` is an identity no-op -- the fla
        # chunk-index cache is keyed on tensor identity, and a per-call cast would miss
        # it inside capture and rebuild the indices via a pageable H2D copy
        self.fla_cu_seqlens = torch.tensor(
            [0, width], dtype=torch.int64, device=device
        )
        self.fla_has_initial_state = torch.ones(1, dtype=torch.bool, device=device)

    @property
    def width(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in vars(self).values()
            if isinstance(tensor, torch.Tensor)
        )

    def copy_from(self, batch) -> None:
        width = self.width
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
            self.rope_positions.copy_(batch.positions.to(torch.int64).expand(3, -1))
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
        # fresh_state_indices / track_* stay None on purpose: a speculative step always
        # continues an existing sequence, and a w <= 4 row forward never reaches a
        # chunk-aligned track boundary (attention/linear.py, c = (extend_len - 1) // 64).
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens,
            cache_indices=self.linear_table_idx,
            has_initial_state=self.fla_has_initial_state,
            max_seq_len=self.width,
        )
        batch.mtp_verify = True


class _MTPVerifyGraphBuffer(_GraphInputBuffer):
    """The observer's buffer: inputs plus one fp32 all-row logit slab."""

    def __init__(self, width: int, vocab_size: int, device: torch.device) -> None:
        super().__init__(width, device)
        self.logits = torch.empty(
            (width, vocab_size), dtype=torch.float32, device=device
        )

    @classmethod
    def init(
        cls, width: int, vocab_size: int, device: torch.device
    ) -> _MTPVerifyGraphBuffer:
        return cls(width, vocab_size, device)


class _SpecVerifyGraphBuffer(_GraphInputBuffer):
    """The integrated buffer: inputs plus BOTH the logit rows and the hidden rows.

    The outputs are shaped from the warm-up pass rather than declared up front, so the graph's
    tensors carry the model's own dtypes and the offline gate can compare replay to eager with
    ``torch.equal`` instead of a tolerance.
    """

    def __init__(self, width: int, device: torch.device) -> None:
        super().__init__(width, device)
        self.logits: torch.Tensor | None = None
        self.hidden: torch.Tensor | None = None

    @classmethod
    def init(cls, width: int, device: torch.device) -> _SpecVerifyGraphBuffer:
        return cls(width, device)

    def alloc_outputs(self, logits: torch.Tensor, hidden: torch.Tensor) -> None:
        width = self.width
        for name, tensor in (("logits", logits), ("hidden", hidden)):
            if tensor.ndim != 2 or int(tensor.shape[0]) != width:
                raise RuntimeError(
                    f"a {width}-row speculative forward returned {name} "
                    f"{tuple(tensor.shape)}"
                )
        if self.logits is None:
            self.logits = torch.empty_like(logits)
            self.hidden = torch.empty_like(hidden)
        elif self.logits.shape != logits.shape or self.hidden.shape != hidden.shape:
            raise RuntimeError("the speculative forward changed its output shapes")

    def store_outputs(self, logits: torch.Tensor, hidden: torch.Tensor) -> None:
        assert self.logits is not None and self.hidden is not None
        self.logits.copy_(logits)
        self.hidden.copy_(hidden)


class _FixedWidthGraphRunner:
    """Survivable per-width capture, shared by the observer and the integrated step."""

    def __init__(
        self,
        *,
        target_ctx,
        target_model,
        attn_backend,
        device: torch.device,
        guard_bytes: int = 0,
        widths: Sequence[int] = (2, 3, 4),
    ) -> None:
        self.target_ctx = target_ctx
        self.target_model = target_model
        self.attn_backend = attn_backend
        self.device = torch.device(device)
        self.guard_bytes = int(guard_bytes)
        self.widths = tuple(int(width) for width in widths)
        if not self.widths or min(self.widths) < 2:
            raise ValueError(f"graph widths must all be >= 2, got {self.widths}")
        if self.guard_bytes < 0:
            raise ValueError("MTP graph guard bytes must be non-negative")
        self._stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self._pool = None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._buffers: dict[int, _GraphInputBuffer] = {}
        self._batches: dict[int, object] = {}
        self._events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._memory_bytes: dict[int, int] = {}
        self._support: dict[int, MTPGraphCaptureResult] = {}
        self._last_attempt: dict[int, MTPGraphCaptureResult] = {}
        self._attempts: dict[int, int] = {}
        self._fla_index_pins: dict[int, tuple[torch.Tensor, ...]] = {}
        self._destroyed = False
        self._graphs_disabled = False

    # ------------------------------------------------------------------ subclass contract

    def _new_buffer(self, width: int) -> _GraphInputBuffer:
        raise NotImplementedError

    def _run(self, batch, buffer, *, allocate: bool) -> None:
        """Run one forward and land its outputs in ``buffer``'s fixed slots."""
        raise NotImplementedError

    def _estimated_buffer_bytes(self, width: int) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------------ inspection

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

    # -------------------------------------------------------------------------- internals

    def _validate_width(self, batch) -> int:
        if self._destroyed:
            raise RuntimeError("MTP graph runner was destroyed")
        width = int(batch.input_ids.shape[0])
        if width not in self.widths:
            raise ValueError(
                f"MTP graph width must be exactly {_render_widths(self.widths)} token rows"
            )
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

    def capture(
        self, batch, *, restore_state: Callable[[], None] | None = None
    ) -> MTPGraphCaptureResult:
        """Capture ``batch``'s width, calling ``restore_state`` after the executing warm-up.

        The recorded pass runs no kernels, so the ONLY execution this method performs is the
        warm-up -- and against a live request that execution has already advanced the request's
        linear state. ``restore_state`` is the caller's undo for exactly that, and it is called
        on every path that ran the warm-up, success or failure.
        """
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
        self._attempts[width] = self._attempts.get(width, 0) + 1
        free_before = int(torch.cuda.mem_get_info(self.device)[0])
        owned_before = self.owned_buffer_bytes
        if free_before < self.guard_bytes + self._estimated_buffer_bytes(width):
            return self._record_attempt(
                width,
                status="retryable",
                reason="MEMORY_ADMISSION",
            )

        graph = torch.cuda.CUDAGraph()
        buffer = self._new_buffer(width)
        synchronizations = 0
        entered_capture = False
        warmed = False
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
                warmed = True
                with torch.cuda.stream(self._stream):
                    self._run(batch, buffer, allocate=True)
                entry_stream.wait_stream(self._stream)
                torch.cuda.synchronize(self.device)
                synchronizations += 1
                if restore_state is not None:
                    restore_state()
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
                        self._run(batch, buffer, allocate=False)
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
            if warmed and restore_state is not None:
                # the warm-up may have advanced the live state before it fell over
                restore_state()
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

    def _replay_into_buffers(self, batch) -> _GraphInputBuffer:
        """Refill this replay's inputs, restage the attention metadata, and run the graph."""
        width = self._validate_width(batch)
        graph = self._graphs.get(width)
        if graph is None:
            support = self._support.get(width)
            reason = support.reason if support is not None else "NOT_CAPTURED"
            raise RuntimeError(f"MTP graph width {width} is unavailable: {reason}")
        buffer = self._buffers[width]
        buffer.copy_from(batch)
        stage_attention = getattr(self.attn_backend, "stage_mtp_verify_graph", None)
        if stage_attention is not None:
            stage_attention(batch, self._batches[width])
        prepare_model = getattr(self.target_model, "prepare_cuda_graph_replay", None)
        if prepare_model is not None:
            prepare_model(batch)
        graph.replay()
        return buffer

    def destroy(self) -> None:
        self._graphs = {}
        self._buffers = {}
        self._batches = {}
        self._events = {}
        self._memory_bytes = {}
        self._fla_index_pins = {}
        self._support = {}
        self._last_attempt = {}
        self._attempts = {}
        self._pool = None
        self._stream = None
        self._destroyed = True
        reset_attention = getattr(self.attn_backend, "reset_mtp_verify_graph", None)
        if reset_attention is not None:
            reset_attention()
        gc.collect()


class SpecVerifyGraphRunner(_FixedWidthGraphRunner):
    """The integrated step's graph: real addresses in, logits AND hidden rows out."""

    def _new_buffer(self, width: int) -> _SpecVerifyGraphBuffer:
        return _SpecVerifyGraphBuffer.init(width, self.device)

    def _estimated_buffer_bytes(self, width: int) -> int:
        # the output slabs are sized from the warm-up, so admission can only bound the inputs;
        # the post-warm-up guard below is what actually protects the capture.
        return width * (3 * torch.int32.itemsize + 3 * torch.int64.itemsize) + 64

    def _run(self, batch, buffer, *, allocate: bool) -> None:
        logits, hidden, _ = self.target_model.forward_mtp_capture(all_row_logits=True)
        if allocate:
            buffer.alloc_outputs(logits, hidden)
        buffer.store_outputs(logits, hidden)

    def replay(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """One captured step: refill, replay, and hand back OWNED logit and hidden rows.

        Owned because the caller holds both across the cycle's accept / emit / rollback tail
        while the buffers belong to the next cycle's replay -- and because the eager fallback
        returns fresh tensors, so the two paths must be indistinguishable to their consumer.
        """
        buffer = self._replay_into_buffers(batch)
        return buffer.logits.clone(), buffer.hidden.clone()

    def forward(
        self, batch, *, restore_state: Callable[[], None] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Replay this step through its width's graph, capturing it first if need be.

        Returns ``None`` when the width has no graph and cannot get one, which is the caller's
        signal to run the step eagerly -- so a capture failure costs correctness nothing.
        """
        if self._destroyed or self.device.type != "cuda":
            return None
        width = int(batch.input_ids.shape[0])
        if width not in self.widths:
            return None
        if width not in self._graphs:
            support = self._support.get(width)
            if support is not None and support.status == "permanently-unsupported":
                return None
            if self._attempts.get(width, 0) >= _MAX_RETRYABLE_ATTEMPTS:
                return None
            if self.capture(batch, restore_state=restore_state).status != "captured":
                return None
            # capture's own warm-up produced values from a state the restore has since wound
            # back, and the recorded pass ran nothing at all: this replay is the step.
        return self.replay(batch)


__all__ = [
    "MTPGraphCaptureResult",
    "SpecVerifyGraphRunner",
    "_MTPVerifyGraphBuffer",
    "_SpecVerifyGraphBuffer",
]
