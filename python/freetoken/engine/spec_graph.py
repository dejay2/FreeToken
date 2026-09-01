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

**WIDTH 1: THE CAPTURE-DECODE GRAPH.** A spec-enabled boot cannot serve its ORDINARY forwards
from the plain decode graph, which returns logits and nothing else while the draft head needs
the hidden rows and the input embeddings too -- so every one of them ran eager, and once the
adaptive policy falls back to plain decode that is every step, at roughly half the graphed
rate. Width 1 records exactly the forward the eager path runs
(``forward_mtp_capture(all_row_logits=False)``) into fixed logit, hidden and embedding slots.

It is DECODE-shaped, not a one-row verify batch: same phase, same kernels and same numbers as
plain decode, which is what the draft head has always been fed. Its restore seam is the same
one, but an ordinary decode step takes no ladder snapshot of its own, so the engine borrows one
around the capture (``SpecStateLadder.borrow_snapshot``).

Sampling stays outside: the graph's outputs are fixed buffers the caller reads.
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

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

    #: whether this shape drives the CHUNKED GDN kernel, and so needs the fla chunk-index
    #: cache primed before capture. Decode does not: it runs the fused recurrent path.
    primes_chunk_indices = True

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


class _SpecDecodeGraphBuffer(_GraphInputBuffer):
    """The width-1 buffer: one ORDINARY decode row in, all three capture outputs out.

    Decode-shaped, deliberately. A one-row verify batch would be prefill-phase and take the
    chunked GDN kernel and the ragged QSA path; the draft head has been fed from the
    decode-phase forward since it existed, and a graph that swapped the kernels underneath it
    would silently change what a fallback step computes. So this mirrors
    ``graph.GraphCaptureBuffer`` at ``bs = 1`` -- the same int32 arange indptr, the same
    state-slot map -- and adds the hidden and embedding slots the draft head consumes.
    """

    primes_chunk_indices = False

    def __init__(self, device: torch.device) -> None:
        super().__init__(1, device)
        # decode GDN metadata is the plain runner's: an int32 arange indptr and no
        # continuation flag, because a decode step always continues its sequence
        self.fla_cu_seqlens = torch.arange(2, dtype=torch.int32, device=device)
        del self.fla_has_initial_state
        self.logits: torch.Tensor | None = None
        self.hidden: torch.Tensor | None = None
        self.embeddings: torch.Tensor | None = None

    @classmethod
    def init(cls, device: torch.device) -> _SpecDecodeGraphBuffer:
        return cls(device)

    def copy_from(self, batch) -> None:
        if int(batch.input_ids.shape[0]) != 1:
            raise ValueError("the width-1 spec graph replays exactly one token row")
        _check_decode_batch(batch)
        self.input_ids.copy_(batch.input_ids)
        self.positions.copy_(batch.positions)
        self.out_loc.copy_(batch.out_loc)
        rope_positions = getattr(batch, "rope_positions", None)
        if rope_positions is None:
            self.rope_positions.copy_(batch.positions.to(torch.int64).expand(3, -1))
        else:
            self.rope_positions.copy_(rope_positions)
        if batch.linear_table_idx is not None:
            self.linear_table_idx.copy_(batch.linear_table_idx[:1])

    def bind(self, batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        batch.input_ids = self.input_ids
        batch.positions = self.positions
        batch.out_loc = self.out_loc
        batch.rope_positions = self.rope_positions
        if batch.linear_table_idx is not None:
            batch.linear_table_idx = self.linear_table_idx
            batch.fla_metadata = FLAMetadata(
                cu_seqlens=self.fla_cu_seqlens, cache_indices=self.linear_table_idx
            )

    def alloc_outputs(self, logits, hidden, embeddings) -> None:
        rows = {"logits": logits, "hidden": hidden, "embeddings": embeddings}
        for name, tensor in rows.items():
            if tensor.ndim != 2 or int(tensor.shape[0]) != 1:
                raise RuntimeError(
                    f"a one-row capture-decode forward returned {name} {tuple(tensor.shape)}"
                )
        if self.logits is None:
            self.logits = torch.empty_like(logits)
            self.hidden = torch.empty_like(hidden)
            self.embeddings = torch.empty_like(embeddings)
        elif any(
            getattr(self, name).shape != tensor.shape for name, tensor in rows.items()
        ):
            raise RuntimeError("the capture-decode forward changed its output shapes")

    def store_outputs(self, logits, hidden, embeddings) -> None:
        assert self.logits is not None
        self.logits.copy_(logits)
        self.hidden.copy_(hidden)
        self.embeddings.copy_(embeddings)


def _check_decode_batch(batch) -> None:
    if not batch.is_decode or batch.size != 1 or batch.padded_size != 1:
        raise ValueError("the width-1 spec graph requires one decode request")
    if getattr(batch, "mtp_verify", False):
        raise ValueError(
            "the width-1 spec graph records the ordinary decode forward, not a verify step"
        )


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
        if not self.widths or min(self.widths) < 1:
            raise ValueError(f"graph widths must all be >= 1, got {self.widths}")
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
        #: diagnosis seam: set to a dict to have every replay fill it with its host-side
        #: staging split and the graph's own GPU duration. None -- production -- costs nothing:
        #: no clock reads, no timing events, and no synchronization.
        self.replay_timings: dict[str, float] | None = None
        self._replay_events_pair: tuple[torch.cuda.Event, torch.cuda.Event] | None = None

    # ------------------------------------------------------------------ subclass contract

    def _new_buffer(self, width: int) -> _GraphInputBuffer:
        raise NotImplementedError

    def _run(self, batch, buffer, *, allocate: bool) -> None:
        """Run one forward and land its outputs in ``buffer``'s fixed slots."""
        raise NotImplementedError

    def _estimated_buffer_bytes(self, width: int) -> int:
        raise NotImplementedError

    def _check_batch(self, batch, width: int) -> None:
        """The shape contract this width's graph replays against. Verify-shaped by default;
        width 1 of the integrated runner overrides it with the decode shape."""
        if not batch.is_prefill or batch.size != 1 or batch.padded_size != 1:
            raise ValueError("MTP graph requires one prefill request")
        if not getattr(batch, "mtp_verify", False):
            raise ValueError("MTP graph requires the private verification marker")

    # Attention staging, as three seams rather than three inline getattrs: the verify widths
    # bind private QSA metadata, and the decode width goes through the backend's ordinary
    # decode-graph buffers instead.

    def _prepare_attention(self, batch, width: int) -> None:
        prepare = getattr(self.attn_backend, "prepare_mtp_verify_graph", None)
        if prepare is not None:
            prepare(batch)

    def _finish_attention(self, width: int) -> None:
        finish = getattr(self.attn_backend, "finish_mtp_verify_graph_capture", None)
        if finish is not None:
            finish(width)

    def _stage_attention(self, batch, static_batch, width: int) -> None:
        stage = getattr(self.attn_backend, "stage_mtp_verify_graph", None)
        if stage is not None:
            stage(batch, static_batch)

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

    def capture_pending(self, width: int) -> bool:
        """Whether the NEXT live step at this width would attempt a capture.

        The caller needs this before the step, not after: capture's warm-up executes the
        forward, so the state it advances has to be snapshotted first -- and an ordinary decode
        step, unlike a speculative one, has no snapshot of its own to reuse.
        """
        if self._destroyed or self.device.type != "cuda" or width not in self.widths:
            return False
        if width in self._graphs:
            return False
        support = self._support.get(width)
        if support is not None and support.status == "permanently-unsupported":
            return False
        return self._attempts.get(width, 0) < _MAX_RETRYABLE_ATTEMPTS

    # -------------------------------------------------------------------------- internals

    def _validate_width(self, batch) -> int:
        if self._destroyed:
            raise RuntimeError("MTP graph runner was destroyed")
        width = int(batch.input_ids.shape[0])
        if width not in self.widths:
            raise ValueError(
                f"MTP graph width must be exactly {_render_widths(self.widths)} token rows"
            )
        self._check_batch(batch, width)
        # the graph's fla_cu_seqlens is a fixed [0, width]; a request whose extend_len says
        # otherwise would replay against metadata that does not describe it
        if int(batch.reqs[0].extend_len) != width:
            raise ValueError(
                f"MTP graph request extend_len {batch.reqs[0].extend_len} != {width} token rows"
            )
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
        # capture outcomes decide whether whole serving paths run graphed or eager, and a
        # failure is otherwise survivable-and-silent -- always say what happened
        logger.info(
            f"MTP spec graph width {width}: {status}"
            + (f" ({reason})" if reason else "")
        )
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
            if buffer.primes_chunk_indices:
                from freetoken.kernel.fla.index import prime_chunk_index_cache

                self._fla_index_pins[width] = prime_chunk_index_cache(
                    buffer.fla_cu_seqlens, width
                )
            self._prepare_attention(batch, width)
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
            self._finish_attention(width)
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
        timings = self.replay_timings
        if timings is not None:
            return self._replay_timed(batch, graph, buffer, width, timings)
        buffer.copy_from(batch)
        self._stage_attention(batch, self._batches[width], width)
        prepare_model = getattr(self.target_model, "prepare_cuda_graph_replay", None)
        if prepare_model is not None:
            prepare_model(batch)
        graph.replay()
        return buffer

    def _replay_timed(
        self, batch, graph, buffer, width: int, timings: dict[str, float]
    ) -> _GraphInputBuffer:
        """``_replay_into_buffers``, armed: the same sequence, split into wall-ms per step.

        ``launch_ms`` is host time only -- ``graph.replay()`` returns once the graph is
        launched -- so the graph's own duration needs the device clock, which is what the event
        pair reads. The sync is why this path is a diagnosis tool and not the default one.
        """
        clock = time.perf_counter
        t0 = clock()
        buffer.copy_from(batch)
        t1 = clock()
        self._stage_attention(batch, self._batches[width], width)
        t2 = clock()
        prepare_model = getattr(self.target_model, "prepare_cuda_graph_replay", None)
        if prepare_model is not None:
            prepare_model(batch)
        t3 = clock()
        start, end = self._timing_events()
        if start is not None:
            start.record()
        graph.replay()
        t4 = clock()
        timings["copy_ms"] = 1e3 * (t1 - t0)
        timings["attn_ms"] = 1e3 * (t2 - t1)
        timings["model_ms"] = 1e3 * (t3 - t2)
        timings["launch_ms"] = 1e3 * (t4 - t3)
        if end is None:
            timings["gpu_ms"] = 0.0
            return buffer
        end.record()
        end.synchronize()
        timings["gpu_ms"] = float(start.elapsed_time(end))
        return buffer

    def _timing_events(self):
        """The armed path's device clock: one pair for the runner's life, built on first use."""
        if self._replay_events_pair is None and self.device.type == "cuda":
            self._replay_events_pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        if self._replay_events_pair is None:
            return None, None
        return self._replay_events_pair

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
    """The integrated step's graphs: the ``w``-row verify forward, and the 1-row decode one.

    Two shapes, one runner -- one graph pool, one memory guard, one survivable-capture
    classifier, and one object for the engine to hold and tear down. Width 1 is never a verify
    batch (a step drafts at least one token, so ``w >= 2`` always), which is what lets the
    width alone select the shape.
    """

    def _is_decode_width(self, width: int) -> bool:
        return width == 1

    def _new_buffer(self, width: int) -> _GraphInputBuffer:
        if self._is_decode_width(width):
            return _SpecDecodeGraphBuffer.init(self.device)
        return _SpecVerifyGraphBuffer.init(width, self.device)

    def _estimated_buffer_bytes(self, width: int) -> int:
        # the output slabs are sized from the warm-up, so admission can only bound the inputs;
        # the post-warm-up guard below is what actually protects the capture.
        return width * (3 * torch.int32.itemsize + 3 * torch.int64.itemsize) + 64

    def _check_batch(self, batch, width: int) -> None:
        if self._is_decode_width(width):
            _check_decode_batch(batch)
            return
        super()._check_batch(batch, width)

    def _prepare_attention(self, batch, width: int) -> None:
        if not self._is_decode_width(width):
            return super()._prepare_attention(batch, width)
        # the ordinary decode-graph seam: it stages this step's block table, lengths and ring
        # slots into the backend's persistent buffers, which is what the capture bakes
        prepare = getattr(self.attn_backend, "prepare_for_capture", None)
        if prepare is not None:
            prepare(batch)

    def _finish_attention(self, width: int) -> None:
        if not self._is_decode_width(width):
            super()._finish_attention(width)

    def _discard_attention_width(self, width: int) -> None:
        # the decode width owns no private QSA metadata; its addressing lives in the backend's
        # ordinary decode-graph buffers, which this runner must never drop
        if not self._is_decode_width(width):
            super()._discard_attention_width(width)

    def _stage_attention(self, batch, static_batch, width: int) -> None:
        if not self._is_decode_width(width):
            return super()._stage_attention(batch, static_batch, width)
        stage = getattr(self.attn_backend, "prepare_for_replay", None)
        if stage is not None:
            stage(batch)

    def _run(self, batch, buffer, *, allocate: bool) -> None:
        if isinstance(buffer, _SpecDecodeGraphBuffer):
            # all_row_logits=False: the same call Engine.forward_batch runs eagerly, so the
            # graphed step's logits, hidden rows and embeddings are the eager ones exactly
            outputs = self.target_model.forward_mtp_capture(all_row_logits=False)
            if allocate:
                buffer.alloc_outputs(*outputs)
            buffer.store_outputs(*outputs)
            return
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
        width = int(batch.input_ids.shape[0])
        if width in self._graphs:
            return self.replay(batch)
        if not self.capture_pending(width):
            return None
        if self.capture(batch, restore_state=restore_state).status != "captured":
            return None
        # capture's own warm-up produced values from a state the restore has since wound
        # back, and the recorded pass ran nothing at all: this replay is the step.
        return self.replay(batch)

    def forward_decode(
        self, batch, *, restore_state: Callable[[], None] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """One graphed capture-decode step: logits, hidden rows and input embeddings.

        ``None`` means "run this decode eagerly" -- no width-1 graph, and none to be had. The
        three tensors are OWNED copies, because the buffers they came from belong to the next
        step's replay while ``observe_forward`` still holds these.
        """
        if not self._is_decode_width(int(batch.input_ids.shape[0])):
            return None
        if 1 not in self._graphs:
            if not self.capture_pending(1):
                return None
            if self.capture(batch, restore_state=restore_state).status != "captured":
                return None
        buffer = self._replay_into_buffers(batch)
        return buffer.logits.clone(), buffer.hidden.clone(), buffer.embeddings.clone()


__all__ = [
    "MTPGraphCaptureResult",
    "SpecVerifyGraphRunner",
    "_MTPVerifyGraphBuffer",
    "_SpecDecodeGraphBuffer",
    "_SpecVerifyGraphBuffer",
]
