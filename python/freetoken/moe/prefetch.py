"""Layer-ahead expert prefetch for the offloaded-MoE decode path.

WHAT THIS IS
------------
In decode, every MoE layer's PCIe fetch is strictly serialized with its own compute::

    router -> ensure_experts(L) -> copy_missing(L) -> expert GEMV

The copy and the GEMV are on one stream, so PCIe idles during the GEMV and the SMs idle
during the copy. Roughly a quarter of a plain decode step (3.6 of 15 ms) and nearly half
of a 6-row MTP verify step (17 of 39 ms) is that serialized copy.

The fix is a predictor plus a second stream. Layer ``L+1``'s router, applied to layer
``L``'s router input, recovers 62% of layer ``L+1``'s real top-10 (recall@10, measured on
48k tokens; most middle layers 0.65-0.75, layers 0/22/38 ~0.3-0.4 and skipped by default).
That is enough to start layer ``L+1``'s copy one layer early, on a side stream, hidden
under layer ``L``'s GEMV. Offline simulation at 5000 slots: synchronous misses per token
101 -> 59 (5.5 -> 3.2 ms the step waits on) for +60% bus bytes, which fit inside the step.

Everything here is off unless ``FREETOKEN_MOE_PREFETCH=1``; with it unset the decode path
is byte-identical to today's.

THE ORDERING ARGUMENT
---------------------
Per MoE layer ``L``, on the compute stream, in this order:

1. ``prefetch_wait(L)``     -- compute stream waits on the done-event of the prefetch that
                               layer ``L-1`` issued *for* ``L`` (if any). After this edge
                               the prefetched bytes have landed, so ``ensure_experts(L)``
                               may treat those slots as plainly resident: there is no
                               "present but still copying" state visible to it.
2. ``prefetch_note_actual`` -- stats only; reads the RAW ids before ensure rewrites them.
3. ``ensure_experts(L)``    -- bumps the global ``step`` to ``S`` and stamps ``usage[slot] =
                               S`` for every slot this layer hit or evicted into.
4. ``copy_missing(L)``      -- the synchronous fetch, compute stream.
5. ``prefetch_experts(L+1, pred)`` -- the Triton ensure variant below, compute stream. It
                               does NOT advance ``step``; it reads ``S`` and protects every
                               slot stamped ``S``.
6. fork: record ``fork_event`` on the compute stream, the side stream waits on it, runs
   ``fast_index_copy_multi_jit`` over the SEPARATE prefetch plan with layer ``L+1``'s source
   pointers, and records ``done_event``.
7. expert GEMV(L), compute stream.

Why each hazard is covered (``step`` advances once per ``ensure_experts`` call, i.e. once
per layer per forward -- see flashlib's ``lru_ensure``, which does ``step = step + 1`` at
the top of every call -- so ``usage == S`` is exactly "touched by layer ``L`` this step"):

* **GEMV(L) reads a slot the prefetch copy overwrites.** GEMV(L) reads only slots routed by
  layer ``L``, all stamped ``S`` at step 3. The victim scan skips ``usage == S``. Excluded.
* **copy_missing(L) and the prefetch copy write the same slot.** copy_missing(L) writes the
  slots ``ensure_experts(L)`` evicted into, also stamped ``S``. Excluded by the same rule.
  (The fork event is recorded *after* copy_missing is enqueued, so the two copies do not
  even contend for PCIe: the prefetch copy starts once the blocking fetch has drained.)
* **The prefetch copy is still in flight when layer L+1 evicts into the same slot.** It
  cannot be: step 1 of layer ``L+1`` waits on ``done_event`` before ``ensure_experts(L+1)``
  runs at all.
* **Two prefetch copies race each other.** They are issued on one side stream, so they are
  serialized; and the earlier one is joined to the compute stream before the later one is
  planned.
* **A prefetch evicts a slot an earlier prefetch is copying into.** The earlier prefetch's
  copy was joined at step 1 of this very layer, which precedes step 5 on the compute
  stream. Nothing is in flight when the victim scan runs.
* **A prefetch evicts a slot it just admitted.** Admitted slots are stamped ``S`` as they
  are installed and removed from the in-register candidate set, so neither this call nor a
  later one in the same step can pick them.

The protection is therefore exactly "``usage == step``", read (never written) from the
shared ``step`` counter. If the counter were per-token rather than per-layer the same
predicate would protect the union of all layers touched this token -- strictly more
conservative and still correct, just wasteful. It is per-layer, so the set protected is
the minimum sound one.

The one thing the prefetch is NOT allowed to do is change numerics: it only ever moves
bytes into slots nothing is reading and rewrites index entries for a layer that has not run
yet. ``topk_ids`` is untouched. Prefetch on vs off must produce identical outputs.

A SMALL CACHE CANNOT BREAK IT
-----------------------------
Protection shrinks the victim pool, so on a cache barely larger than one layer's working
set there may be fewer unprotected slots than predictions to admit. The kernel counts the
evictable slots and clamps ``num_indices`` to them rather than assuming a size: an
over-subscribed step simply prefetches less (the bandwidth guard, arrived at from the other
direction). It never falls back to evicting a live slot, which is the one thing that would
corrupt the step in flight. On the production geometry (5000 slots against ~130 protected)
the clamp never binds.

(One corollary: on a completely cold cache every ``usage`` is 0, so at ``step == 0``
everything reads as protected and the clamp admits nothing. The live path never sees it --
``prefetch_experts`` always runs after an ``ensure_experts`` that has already advanced the
clock to at least 1 -- and the fallback is a normal synchronous miss, not a wrong eviction.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl

_TRUE = {"1", "true", "yes", "on"}

# Column layout of the prefetch stats accumulator; callers never hardcode it.
STAT_CALLS = 0  # prefetch_experts launches
STAT_PRED = 1  # distinct predicted experts
STAT_HIT = 2  # predicted and already resident (no bytes moved)
STAT_FETCH = 3  # predicted, missing, and admitted (bytes moved on the side stream)
STAT_MISS = 4  # predicted and missing, BEFORE the max-misses cap
STAT_USED = 5  # experts the target layer really routed to that were predicted
STAT_USED_FETCH = 6  # ...of those, the ones the prefetch actually fetched
STAT_ACTUAL = 7  # distinct experts the target layer really routed to
N_PREFETCH_STATS = 8

# Sentinel for "not predicted" in the per-expert priority vector. int32, so it can never
# collide with a real priority (rank * rows + row, bounded by kprime * rows). Wrapped in
# tl.constexpr because a @triton.jit body may only read globals declared that way.
_BIG = tl.constexpr(0x7FFFFFFF)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_layers(name: str, default: str) -> frozenset[int]:
    raw = os.getenv(name)
    raw = default if raw is None else raw
    return frozenset(int(p) for p in raw.replace(" ", "").split(",") if p)


@dataclass(frozen=True)
class PrefetchConfig:
    """Env-derived knobs for the layer-ahead expert prefetch.

    ``skip_layers`` names TARGET layers whose prediction is not worth the bus bytes: the
    default ``0,22,38`` are the three measured layers whose recall@10 sits at 0.3-0.4 while
    every other layer clears 0.6. A skipped target means layer ``target-1`` issues no
    prefetch and layer ``target`` waits on nothing -- the fork/join stay balanced.
    """

    enabled: bool = False
    topk: int = 10
    max_misses: int = 8
    skip_layers: frozenset = field(default_factory=frozenset)
    log: bool = False
    log_every: int = 200

    @staticmethod
    def from_env() -> "PrefetchConfig":
        return PrefetchConfig(
            enabled=os.getenv("FREETOKEN_MOE_PREFETCH", "0").strip().lower() in _TRUE,
            # k' > 10 is bus-bound (top-20 costs 808 MiB/token vs 428 at top-10, against a
            # ~15 ms step); the default is the measured knee, not a guess.
            topk=_env_int("FREETOKEN_MOE_PREFETCH_TOPK", 10),
            max_misses=_env_int("FREETOKEN_MOE_PREFETCH_MAX_MISSES", 8),
            skip_layers=_env_layers("FREETOKEN_MOE_PREFETCH_SKIP_LAYERS", "0,22,38"),
            log=os.getenv("FREETOKEN_MOE_PREFETCH_LOG", "0").strip().lower() in _TRUE,
            log_every=_env_int("FREETOKEN_MOE_PREFETCH_LOG_EVERY", 200),
        )


# Read once at import, like every other FREETOKEN_* knob in the MoE path: the whole cost of
# the feature being off is one attribute read per decode layer.
PREFETCH = PrefetchConfig.from_env()


# ----------------------------------------------------------------------
# The ensure variant
# ----------------------------------------------------------------------


@triton.jit(do_not_specialize=["layer_id", "rows", "kprime", "max_misses"])
def _prefetch_ensure_kernel(
    pred_ids_ptr,  # [rows, kprime] int32: predicted expert ids of the TARGET layer, rank-major
    slot_for_id_ptr,  # [num_layers, num_experts] int32, in/out
    id_of_slot_ptr,  # [cache_size] int32, in/out
    usage_ptr,  # [cache_size] int64, in/out
    step_ptr,  # () int64, READ ONLY -- this kernel never advances the clock
    mark_ptr,  # [num_layers, num_experts] int64, out: 2*step (predicted) | 2*step+1 (fetched)
    evict_slots_ptr,  # [max_misses] int32, out: prefetch copy plan, dst slots
    src_indices_ptr,  # [max_misses] int32, out: prefetch copy plan, layer-local src rows
    num_indices_ptr,  # [1] int64, out: valid length of the plan
    stats_ptr,  # [N_PREFETCH_STATS] int64, optional accumulator
    layer_id,
    rows,
    kprime,
    max_misses,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
):
    """Timestamp-LRU admission for a PREDICTED expert set, protecting the current step.

    Differs from the real ensure (flashlib's ``lru_ensure``, and its in-repo twin
    ``_ensure_experts_hybrid_kernel``) in exactly four ways, each forced by the fact that
    the layer whose experts these are has not run yet:

    1. It does not advance ``step``. The clock belongs to the real ensure; this call has to
       be able to *name* the current step in order to protect it.
    2. Victim selection excludes every slot with ``usage == step`` -- the experts the layer
       that is running right now is about to read, plus the rows its own ``copy_missing``
       is writing. The real ensure only has to exclude the ids in its own query.
    3. It admits at most ``max_misses`` experts -- and never more than the protection left
       evictable -- in descending predicted score (the topk column order, tie-broken across
       rows by rank then row), and writes its plan into SEPARATE
       ``evict_slots``/``src_indices``/``num_indices`` buffers so the real ensure's pending
       plan is untouched. Capped predictions stay non-resident; nothing is half-inserted.
    4. It rewrites nothing: ``pred_ids`` is read-only and no caller maps through it. The
       target layer's own ensure will do the id -> slot rewrite, and will simply find these
       ids resident.

    ``mark`` records, per (layer, expert), the step at which the prefetch touched it --
    ``2*step`` for a prediction that was already resident, ``2*step+1`` for one it fetched.
    :func:`_prefetch_score_kernel` reads it at the target layer (while the clock still holds
    the same ``step``) to turn predictions into a measured precision/recall.
    """
    step = tl.load(step_ptr)
    base = layer_id * num_experts
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts

    # ---- Phase 1: dedup the prediction, keeping each expert's BEST rank ----
    # Priority index r * rows + t: rank-major, so with a multi-row (MTP verify) batch every
    # row's top-1 outranks any row's top-2. First write wins, and the loop runs in
    # ascending priority, so `prio` ends up holding each expert's best occurrence. The
    # values are distinct across experts, which makes the argmin below tie-free.
    prio = tl.full((BLOCK_E,), _BIG, tl.int32)
    for r in tl.range(kprime):
        for t in tl.range(rows):
            e = tl.load(pred_ids_ptr + t * kprime + r)
            prio = tl.where((off_e == e) & (e >= 0) & (prio == _BIG), r * rows + t, prio)

    is_pred = (prio < _BIG) & e_mask
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
    is_hit = is_pred & (slot >= 0)
    is_miss = is_pred & (slot == -1)
    num_miss = tl.sum(is_miss.to(tl.int32))
    num_fetch = tl.minimum(num_miss, max_misses)
    # A predicted hit is a real LRU touch: it is the freshest possible reference (the layer
    # that will read it runs next). Bumping it to `step` also protects it from this very
    # call's victim scan, which is what stops a prefetch from evicting its own hits.
    tl.store(usage_ptr + slot, step, mask=is_hit)
    mark_pred = tl.zeros((BLOCK_E,), dtype=tl.int64) + 2 * step
    tl.store(mark_ptr + base + off_e, mark_pred, mask=is_pred)

    # ---- Phase 2: admit the top `num_fetch` misses over LRU victims ----
    score = tl.where(is_miss, prio, _BIG)
    num_admit = num_fetch
    if num_fetch > 0:
        # REQUIRED (same hazard flashlib's kernel documents): the hit bump above is a
        # scatter and the load below is a bulk reload of the same array. Without a CTA-scope
        # fence the reload can observe a pre-bump value and evict a slot the target layer is
        # about to read -- or, worse here, one this step's GEMV is reading.
        tl.debug_barrier()
        off_c = tl.arange(0, BLOCK_C)
        c_mask = off_c < cache_size
        u = tl.load(usage_ptr + off_c, mask=c_mask, other=9223372036854775807).to(tl.int64)
        # THE protection: everything stamped with the current step is live -- layer L's
        # routed experts (GEMV about to read them) and the rows layer L's copy_missing is
        # writing right now. See the module docstring for the full ordering argument.
        u = tl.where((u == step) | (~c_mask), 9223372036854775807, u)
        # Never admit more than there are unprotected slots. On a cache barely wider than
        # one layer's working set that clamp binds; on the production geometry it never
        # does. It is what lets the loop below assume every argmin is a real victim.
        num_admit = tl.minimum(
            num_fetch, tl.sum((u != 9223372036854775807).to(tl.int32))
        )
        for i in tl.range(num_admit):
            victim = tl.argmin(u, axis=0).to(tl.int32)
            e = tl.argmin(score, axis=0).to(tl.int32)
            # Scalar load: victims are distinct (each is masked out of `u` once taken), so
            # no earlier iteration wrote the slot being read.
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_for_id_ptr + old, -1)
            tl.store(id_of_slot_ptr + victim, base + e)
            tl.store(slot_for_id_ptr + base + e, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(mark_ptr + base + e, 2 * step + 1)
            tl.store(evict_slots_ptr + i, victim)
            tl.store(src_indices_ptr + i, e)  # layer-local row, resolved against layer_id
            score = tl.where(off_e == e, _BIG, score)
            u = tl.where(off_c == victim, 9223372036854775807, u)

    tl.store(num_indices_ptr, num_admit.to(tl.int64))
    if COLLECT_STATS:
        si = tl.arange(0, 8)
        v = tl.where(
            si == 0,
            1,
            tl.where(
                si == 1,
                tl.sum(is_pred.to(tl.int32)),
                tl.where(
                    si == 2,
                    tl.sum(is_hit.to(tl.int32)),
                    tl.where(si == 3, num_admit, tl.where(si == 4, num_miss, 0)),
                ),
            ),
        )
        tl.atomic_add(stats_ptr + si, v.to(tl.int64), mask=si < 5)


@triton.jit(do_not_specialize=["layer_id", "num_active"])
def _prefetch_score_kernel(
    expert_ids_ptr,  # [num_active] int32: this layer's RAW routed ids (pre-rewrite)
    mark_ptr,  # [num_layers, num_experts] int64
    step_ptr,  # () int64, read only
    stats_ptr,  # [N_PREFETCH_STATS] int64
    layer_id,
    num_active,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Score the prediction that was made for this layer against what it really routed to.

    Runs at layer ``L`` before ``ensure_experts(L)``, so (a) the ids are still raw expert
    ids and (b) the clock still reads the ``step`` that layer ``L-1``'s ensure set -- the
    same value the prefetch stamped into ``mark``. That makes ``mark >> 1 == step`` mean
    exactly "predicted for this layer, this step", with no extra state to keep in sync.
    """
    step = tl.load(step_ptr)
    base = layer_id * num_experts
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    is_active = tl.zeros((BLOCK_E,), dtype=tl.int1)
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        is_active = is_active | ((off_e == e) & (e >= 0))
    is_active = is_active & e_mask
    mark = tl.load(mark_ptr + base + off_e, mask=e_mask, other=-1).to(tl.int64)
    used = is_active & ((mark >> 1) == step)
    used_fetch = is_active & (mark == 2 * step + 1)
    si = tl.arange(0, 8)
    v = tl.where(
        si == 5,
        tl.sum(used.to(tl.int32)),
        tl.where(si == 6, tl.sum(used_fetch.to(tl.int32)), tl.sum(is_active.to(tl.int32))),
    )
    tl.atomic_add(stats_ptr + si, v.to(tl.int64), mask=(si >= 5) & (si < 8))


def prefetch_ensure(cache, layer_id: int, pred_ids: torch.Tensor, max_misses: int) -> None:
    """Launch :func:`_prefetch_ensure_kernel` for ``cache`` (one program, no host sync)."""
    rows, kprime = pred_ids.shape
    _prefetch_ensure_kernel[(1,)](
        pred_ids,
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.prefetch_mark,
        cache.prefetch_evict_slots,
        cache.prefetch_src_indices,
        cache.prefetch_num_indices,
        cache.prefetch_stats,
        layer_id,
        rows,
        kprime,
        int(max_misses),
        cache.num_experts,
        cache.cache_size,
        BLOCK_E=triton.next_power_of_2(cache.num_experts),
        BLOCK_C=triton.next_power_of_2(cache.cache_size),
        COLLECT_STATS=cache.collect_stats,
        num_warps=8 if cache.cache_size >= 2048 else 4,
    )


def prefetch_score(cache, layer_id: int, expert_ids: torch.Tensor) -> None:
    """Launch :func:`_prefetch_score_kernel` for ``cache`` (stats only)."""
    _prefetch_score_kernel[(1,)](
        expert_ids,
        cache.prefetch_mark,
        cache.step,
        cache.prefetch_stats,
        layer_id,
        expert_ids.numel(),
        cache.num_experts,
        BLOCK_E=triton.next_power_of_2(cache.num_experts),
    )


# ----------------------------------------------------------------------
# CPU reference -- the oracle the Triton kernel is tested against
# ----------------------------------------------------------------------


def prefetch_ensure_reference(
    *,
    pred_ids: torch.Tensor,
    slot_for_id: torch.Tensor,
    id_of_slot: torch.Tensor,
    usage: torch.Tensor,
    step: int,
    layer_id: int,
    num_experts: int,
    cache_size: int,
    max_misses: int,
) -> tuple[list[int], list[int]]:
    """Pure-python mirror of :func:`_prefetch_ensure_kernel`; returns ``(dst, src)``.

    Same decisions, same order, same tie-breaks: dedup keeping the best (rank, row)
    priority; hits bumped to ``step``; victims by ``min(usage, slot)`` over slots whose
    usage is not ``step``; at most ``max_misses`` admissions in descending predicted score.
    Mutates the tensors it is given, exactly as the kernel does.
    """
    rows, kprime = pred_ids.shape
    order: list[int] = []
    seen: set[int] = set()
    for r in range(kprime):
        for t in range(rows):
            e = int(pred_ids[t, r])
            if e >= 0 and e not in seen:
                seen.add(e)
                order.append(e)

    for e in order:
        s = int(slot_for_id[layer_id, e])
        if s >= 0:
            usage[s] = step

    missing = [e for e in order if int(slot_for_id[layer_id, e]) == -1]
    live = usage.tolist()
    evictable = sum(1 for s in range(cache_size) if live[s] != step)
    num_fetch = min(len(missing), int(max_misses), evictable)
    dst: list[int] = []
    src: list[int] = []
    for i in range(num_fetch):
        e = missing[i]
        victim = min(
            (s for s in range(cache_size) if live[s] != step),
            key=lambda s: (live[s], s),
        )
        old = int(id_of_slot[victim])
        if old >= 0:
            slot_for_id.view(-1)[old] = -1
        id_of_slot[victim] = layer_id * num_experts + e
        slot_for_id[layer_id, e] = victim
        usage[victim] = step
        live[victim] = step
        dst.append(victim)
        src.append(e)
    return dst, src
