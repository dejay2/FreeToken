"""CUDA graphs for the DRAFT side of a speculative cycle.

WHAT THIS IS FOR
----------------
The kernel profile of a depth-5 cycle on this box says the draft chain
(``SpecDraftHead.propose`` in ``chain`` mode) is 18 ms of wall clock over 7.2 ms of GPU
kernels -- 843 launches, ~170 per step -- and the commit forward that follows it is 7 ms of
wall clock over 1 ms of kernels. Nearly two thirds of a cycle's host time is therefore the
CPU walking Python and pushing launch packets, for a sequence of kernels whose shapes never
change. That is precisely what a CUDA graph deletes.

The previous session cleared the three documented blockers to capturing the chain (per-step
host ints, per-step pinned metadata, per-step readback). What remained were the values that
genuinely MOVE from cycle to cycle, and the whole of this module plus
:meth:`SpecDraftHead._stage_device_positions` exists to turn each of them into a device
input written before the replay rather than a host scalar baked into the record:

===========================  =====================================================
what moves per cycle          how it becomes a device input
===========================  =====================================================
``committed_len``             ``_graph_base``, an int32 ``[1]`` cell; every
                              position, ``out_loc`` gather and ``device_len`` in
                              the graph is derived from it by device arithmetic
the seed hidden row           ``_graph_in_sample`` -- copied in before the replay
the recursive multi-stream    ``_graph_in_recursive`` -- likewise
step 0's QSA block choice     ``_graph_in_blocks`` -- likewise
the picture rope origin       ``_graph_rope_base``, an int64 ``[1]`` cell, so a
                              step's ``(3, 1)`` coordinate is ``base + delta + i``
the commit's hidden/embeds    ``_commit_in_*`` -- copied in before the replay
the QSA context length        ``QSASparseAttnBackend.step_seq_len_source``: the
                              backend's ``fill_`` bakes a host scalar, so an armed
                              chain hands it a cell to copy from instead
the request's temperature,    ``MTPDraftSampler``'s three cells, staged by
top-k and top-p               ``_stage_chain_inputs`` -- see below
the draft RNG's philox state  registered on the graph, so the replay bumps it
the private KV / ring         already at fixed addresses (the head owns its pool)
===========================  =====================================================

SAMPLED REQUESTS ARE GRAPHED TOO
--------------------------------
This chain first shipped greedy-only, because ``MTPDraftSampler`` branched on HOST values
(``temperature == 0 or top_k == 1`` took ``argmax``; the other arms divided by a host
temperature and sliced a host ``top_k``) and because the draft's generator was rebuilt per
request, so a record could hold neither. The operator's real traffic runs at the model's
temperature, which is exactly where the 11 ms was being left on the table. Both blockers are
gone:

* ``MTPDraftSampler.draw`` is branch-free -- one scale, one stable sort, a top-k THRESHOLD
  mask, a top-p prefix-sum mask and an inverse-CDF draw, always, with the greedy id selected
  by a ``torch.where`` on the temperature cell. Greedy stays bit-exact ``argmax``; the sampled
  distribution stays ``filtered_probs``, pinned by a chi-square/total-variation test rather
  than by matching ``multinomial``'s draw sequence;
* the head owns ONE generator for its life (``SpecDraftHead.reset_request`` explains what the
  per-request stream did and did not guarantee), and ``capture(..., generators=...)`` registers
  its state so replays advance the philox offset exactly as eager chains would.

So the key is ``chain:mrope={0,1}`` and nothing about the request enters it.

THE CAPTURE DISCIPLINE
----------------------
Copied from ``spec_graph._FixedWidthGraphRunner``, which is the machinery this repo has
already proven against torch 2.11: warm up ON the capture stream so per-stream lazy resources
exist before recording, capture ``thread_local`` so the ple-mmap staging thread cannot
invalidate it from outside, restore the entry stream by hand because a raising ``capture_end``
skips torch's own restore, preserve the body's exception before ``capture_end`` masks it,
classify an in-capture failure as permanently unsupported rather than retrying into a
re-poisoned context, and refund a retryable boot attempt so the live path is no worse off.

``restore`` is called after the warm-up, after the recorded pass and on every failure. The
warm-up EXECUTES the draft forwards -- it advances ``committed_len`` and mutates the two
pending rings -- while the recorded pass executes no kernels at all, so the state the replay
starts from has to be wound back between them. The recorded pass still runs the head's
Python, so the host-side half of the wind-back is needed after it too.

ONE MEMORY POOL. Every graph here shares one private pool. Sharing is safe because every
tensor a replay allocates from it is written before it is read within that same replay: the
graphs are pure feed-forward passes over fixed input buffers, so no replay depends on a
value another graph's replay left in the pool, and they may run in any order. What that buys
is that the pool sizes to the WIDEST shape (a 6-row commit's expert gather) rather than to
the sum of eight of them.

``FREETOKEN_MTP_SPEC_DRAFT_GRAPH=0`` turns the whole module off; the head then runs exactly
the eager chain and commit it ran before this existed.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

DRAFT_GRAPH_ENV = "FREETOKEN_MTP_SPEC_DRAFT_GRAPH"

#: Free VRAM a capture refuses to dip below. The draft graphs are captured at boot beside the
#: verify ones and share their guard, so a card that cannot afford them says so instead of
#: taking memory the target's expert cache needs.
DRAFT_GRAPH_GUARD_BYTES = 128 << 20
#: A retryable outcome (memory admission) is worth another go on a later cycle, but not every
#: cycle: a doomed key would otherwise pay a warm-up forward per cycle forever.
_MAX_RETRYABLE_ATTEMPTS = 3
_CAPTURE_RESERVE_FLOOR = 64 << 20


def draft_graph_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the draft chain and commit forward may be recorded (default: yes)."""
    env = os.environ if environ is None else environ
    raw = (env.get(DRAFT_GRAPH_ENV, "1") or "1").strip()
    if raw not in {"0", "1"}:
        raise ValueError(f"{DRAFT_GRAPH_ENV} must be 0 or 1, got {raw!r}")
    return raw == "1"


@dataclass(frozen=True)
class DraftGraphCaptureResult:
    """One capture attempt's verdict, in the shape ``MTPGraphCaptureResult`` uses."""

    key: str
    status: str
    reason: str
    memory_bytes: int = 0
    synchronizations: int = 0


class SpecDraftGraphRunner:
    """Survivable capture and replay of the draft head's fixed-shape forwards.

    Keys are opaque strings the head builds (``"chain:greedy"``, ``"commit:3"``, ...). The
    runner owns nothing about the model: the caller hands it a ``run`` callable that performs
    the sequence into its own fixed buffers, and a ``restore`` callable that undoes whatever
    executing it changed.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        guard_bytes: int = DRAFT_GRAPH_GUARD_BYTES,
    ) -> None:
        self.device = torch.device(device)
        self.guard_bytes = int(guard_bytes)
        if self.guard_bytes < 0:
            raise ValueError("draft graph guard bytes must be non-negative")
        self._stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self._pool = None
        self._graphs: dict[str, "torch.cuda.CUDAGraph"] = {}
        self._support: dict[str, DraftGraphCaptureResult] = {}
        self._last_attempt: dict[str, DraftGraphCaptureResult] = {}
        self._attempts: dict[str, int] = {}
        self._memory_bytes: dict[str, int] = {}
        self._disabled = False
        self._destroyed = False

    # ------------------------------------------------------------------------- inspection

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    @property
    def live_graph_memory_bytes(self) -> int:
        return sum(self._memory_bytes.values())

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._graphs))

    def available(self, key: str) -> bool:
        return key in self._graphs

    def support(self, key: str) -> DraftGraphCaptureResult | None:
        return self._support.get(key)

    def last_attempt(self, key: str) -> DraftGraphCaptureResult | None:
        return self._last_attempt.get(key)

    def capture_pending(self, key: str) -> bool:
        """Whether the NEXT use of ``key`` would attempt a capture."""
        if self._destroyed or self._disabled or self.device.type != "cuda":
            return False
        if key in self._graphs:
            return False
        support = self._support.get(key)
        if support is not None and support.status == "permanently-unsupported":
            return False
        return self._attempts.get(key, 0) < _MAX_RETRYABLE_ATTEMPTS

    def refund_attempt(self, key: str) -> None:
        """Give one retryable attempt back, so a boot-time try costs the live path nothing."""
        if self._attempts.get(key, 0) > 0:
            self._attempts[key] -= 1

    # ---------------------------------------------------------------------------- capture

    def _record_attempt(
        self,
        key: str,
        *,
        status: str,
        reason: str,
        memory_bytes: int = 0,
        synchronizations: int = 0,
    ) -> DraftGraphCaptureResult:
        if status not in {"captured", "retryable", "permanently-unsupported"}:
            raise ValueError(f"invalid draft graph attempt status {status!r}")
        result = DraftGraphCaptureResult(
            key=key,
            status=status,
            reason=reason,
            memory_bytes=memory_bytes,
            synchronizations=synchronizations,
        )
        self._last_attempt[key] = result
        if status in {"captured", "permanently-unsupported"}:
            self._support[key] = result
        else:
            self._support.pop(key, None)
        # a capture outcome decides whether a whole serving path runs graphed or eager, and a
        # failure is otherwise survivable-and-silent -- always say what happened
        logger.info(
            f"MTP draft graph {key}: {status}" + (f" ({reason})" if reason else "")
        )
        return result

    def _reset_capture_context(self) -> bool:
        """Rebuild the private stream and probe the context after a failed capture."""
        self._pool = None
        try:
            self._stream = torch.cuda.Stream(device=self.device)
            torch.zeros(1, device=self.device)
            torch.cuda.synchronize(self.device)
        except Exception:
            return False
        return True

    def capture(
        self,
        key: str,
        run: Callable[[], None],
        *,
        restore: Callable[[], None] | None = None,
        generators: "Sequence[torch.Generator]" = (),
    ) -> DraftGraphCaptureResult:
        """Record ``run`` under ``key``, winding the head back with ``restore`` around it.

        ``run`` is called twice: once as an executing warm-up (on the capture stream, so the
        per-stream lazy resources it needs exist before recording) and once under capture,
        where it executes nothing. ``restore`` follows each call and every failure.

        ``generators`` are the RNG streams the body draws from. Each is registered on the graph
        before recording (``CUDAGraph.register_generator_state``, torch 2.11) so that its philox
        seed and offset are read from device tensors the replay bumps, rather than baked as the
        constants they were at capture. Without it a sampled draft chain would propose the
        capture's tokens on every replay for ever; with it, N replays draw exactly what N eager
        chains from the same state draw.
        """
        previous = self._support.get(key)
        if previous is not None:
            return previous
        if self._destroyed:
            return self._record_attempt(
                key, status="permanently-unsupported", reason="RUNNER_DESTROYED"
            )
        if self.device.type != "cuda":
            return self._record_attempt(
                key, status="permanently-unsupported", reason="CUDA_REQUIRED"
            )
        if self._disabled:
            return self._record_attempt(
                key, status="permanently-unsupported", reason="CAPTURE_CONTEXT_LOST"
            )
        self._attempts[key] = self._attempts.get(key, 0) + 1
        free_before = int(torch.cuda.mem_get_info(self.device)[0])
        if free_before < self.guard_bytes:
            return self._record_attempt(key, status="retryable", reason="MEMORY_ADMISSION")

        graph = torch.cuda.CUDAGraph()
        synchronizations = 0
        entered_capture = False
        warmed = False
        body_error: Exception | None = None
        # a raising capture_end skips torch's own stream restore, so own the restore here
        entry_stream = torch.cuda.current_stream(self.device)
        try:
            for generator in generators:
                # before the warm-up, so a torch that refuses the registration fails on the
                # cheap side of the attempt rather than half-way through a recording
                graph.register_generator_state(generator)
            self._stream.wait_stream(entry_stream)
            warmed = True
            with torch.cuda.stream(self._stream):
                run()
            entry_stream.wait_stream(self._stream)
            torch.cuda.synchronize(self.device)
            synchronizations += 1
            if restore is not None:
                restore()
            free_after_warm = int(torch.cuda.mem_get_info(self.device)[0])
            warm_reserve = max(_CAPTURE_RESERVE_FLOOR, free_before - free_after_warm)
            if free_after_warm < self.guard_bytes + warm_reserve:
                gc.collect()
                return self._record_attempt(
                    key,
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
                    run()
                except Exception as exc:
                    # capture_end raises next and would otherwise mask the real cause
                    body_error = exc
                    raise
            captured_pool = graph.pool() if self._pool is None else self._pool
            torch.cuda.synchronize(self.device)
            synchronizations += 1
            if restore is not None:
                # the recorded pass ran no kernels, but it DID run the head's Python, so the
                # host-side half of the wind-back (``committed_len``) is still owed
                restore()
            free_after = int(torch.cuda.mem_get_info(self.device)[0])
            if free_after < self.guard_bytes:
                gc.collect()
                return self._record_attempt(
                    key,
                    status="retryable",
                    reason="MEMORY_GUARD_AFTER_CAPTURE",
                    synchronizations=synchronizations,
                )
        except Exception as exc:  # pragma: no cover - needs a live CUDA context
            torch.cuda.set_stream(entry_stream)
            if warmed and restore is not None:
                try:
                    restore()
                except Exception:
                    pass
            failure = body_error if body_error is not None else exc
            graph = None
            gc.collect()
            detail = " ".join(str(failure).split())[:300]
            reason = f"CAPTURE_FAILED:{type(failure).__name__}:{detail}"
            if entered_capture:
                # retrying would re-execute the same capture-illegal op and re-poison the
                # context, so this key -- and, if the context did not survive, every key --
                # is done for the life of the process
                if not self._reset_capture_context():
                    self._disabled = True
                    self._graphs.clear()
                return self._record_attempt(
                    key,
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
                key, status=status, reason=reason, synchronizations=synchronizations
            )
        finally:
            torch.cuda.set_stream(entry_stream)

        self._graphs[key] = graph
        self._memory_bytes[key] = max(0, free_before - free_after)
        if self._pool is None:
            self._pool = captured_pool
        return self._record_attempt(
            key,
            status="captured",
            reason="",
            memory_bytes=self._memory_bytes[key],
            synchronizations=synchronizations,
        )

    def replay(self, key: str) -> None:
        graph = self._graphs.get(key)
        if graph is None:
            support = self._support.get(key)
            reason = support.reason if support is not None else "NOT_CAPTURED"
            raise RuntimeError(f"MTP draft graph {key} is unavailable: {reason}")
        graph.replay()

    def destroy(self) -> None:
        self._graphs = {}
        self._support = {}
        self._last_attempt = {}
        self._attempts = {}
        self._memory_bytes = {}
        self._pool = None
        self._stream = None
        self._destroyed = True
        gc.collect()


__all__ = [
    "DRAFT_GRAPH_ENV",
    "DRAFT_GRAPH_GUARD_BYTES",
    "DraftGraphCaptureResult",
    "SpecDraftGraphRunner",
    "draft_graph_enabled",
]
