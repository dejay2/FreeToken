from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from freetoken.core import Req
from freetoken.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from freetoken.utils import align_down, div_ceil, init_logger

if TYPE_CHECKING:
    from .utils import PendingReq

logger = init_logger(__name__)

# Proactive out-of-window free_swa runs every `interval` forwards (== sglang SWA_EVICTION_INTERVAL).
def _swa_eviction_interval() -> int:
    raw = os.environ.get("FREETOKEN_SWA_EVICTION_INTERVAL", "128")
    try:
        return max(1, int(raw))
    except ValueError:
        raise ValueError(f"FREETOKEN_SWA_EVICTION_INTERVAL must be an integer, got {raw!r}")


_SWA_EVICTION_INTERVAL = _swa_eviction_interval()

# Finish-time retention keeps [P - window - gap, P) swa-live for the next turn's cut near the
# prompt end. The gap covers templates whose generation prompt injects tokens that vanish when
# the client drops reasoning (Qwen's "<think>\n": the re-render diverges 2 tokens BEFORE P).
_SWA_RETAIN_GAP = 16


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str,
                 linear_state_pool=None, swa_pool=None, sliding_window_size=None, park_store=None,
                 park_consensus=None):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        # Hybrid GDN models drive a second currency (GDN state snapshots in LinearStatePool)
        # through a HybridRadixCache; SWA models drive a second currency (swa-pool KV slots in
        # the HybridSWAKVCache global-paged mode) through a SWARadixCache; non-hybrid models keep
        # the plain naive/radix path.
        self.linear_state_pool = linear_state_pool
        self.swa_pool = swa_pool
        self.sliding_window_size = sliding_window_size
        self.is_hybrid = type == "hybrid_radix"
        self.is_swa = type == "swa_radix"
        # swa_paged: this SWA model drives the global-paged swa pool -- true for BOTH the naive
        # (NaivePrefixCache, no reuse) and radix (SWARadixCache) paths. Gates the swa slot
        # lifecycle (alloc_swa / out-of-window free / free-on-finish). is_swa gates only the extra
        # SWARadixCache reuse machinery (tree match/insert/evict_swa/swa_uuid lock).
        self.swa_paged = swa_pool is not None and getattr(swa_pool, "swa_paged", False)
        # Owned-pool capability pickup: a plugged-in swa pool may cap the prefill chunk (DSV4:
        # ~half the window working set). Instance attrs shadow the class defaults; absent
        # attributes leave the defaults untouched (Gemma4).
        if swa_pool is not None:
            self.prefill_chunk_budget = getattr(swa_pool, "prefill_chunk_budget", None)
        self.prefix_cache = self._make_prefix_cache(device, page_size, type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.cache_type = type
        if park_store is not None and not self.is_hybrid:
            raise ValueError("KV parking is supported only by the hybrid radix cache")
        self.park_store = park_store
        self.park_consensus = park_consensus
        self._park_generation = int(getattr(park_store, "generation", 0))
        self._pending_parks = []
        self._temporary_lease_depth = 0
        if park_store is not None:
            bind_change_callback = getattr(park_store, "bind_change_callback", None)
            if bind_change_callback is not None:
                bind_change_callback(self._bump_park_generation)

    @property
    def park_generation(self) -> int:
        return self._park_generation

    def _bump_park_generation(self) -> None:
        self._park_generation += 1

    # ----- capability hooks (defaults; plugged-in pools may narrow them) -----
    supports_runtime_rebuild = True
    prefill_chunk_budget = None  # generic shared page pool: no per-model prefill chunk cap

    @property
    def prefill_chunk_align(self) -> int:
        """Granularity a non-final prefill chunk should end on. A hybrid snapshot is donated only
        at a page-aligned boundary, so at page_size>1 one unaligned chunk end costs every reuse
        point for the rest of the prompt. 1 (no-op) everywhere else."""
        return self.page_size if self.is_hybrid else 1

    def page_usage(self) -> tuple[int, int]:
        """(used_pages, total_pages): allocated, non-evictable pages over the pool total
        (active requests + protected prefix; evictable prefix-cache pages are excluded)."""
        total = self.num_pages
        evictable = (self.prefix_cache.full_evictable_size if (self.is_hybrid or self.is_swa)
                     else self.prefix_cache.size_info.evictable_size)
        return total - len(self.free_slots) - evictable // self.page_size, total

    def _make_prefix_cache(self, device, page_size, type):
        if type == "hybrid_radix":
            from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
            return HybridRadixCache(device, page_size)
        if type == "swa_radix":
            from freetoken.kvcache.swa_radix_cache import SWARadixCache
            return SWARadixCache(device, page_size, self.sliding_window_size)
        return create_prefix_cache(device=device, type=type, page_size=page_size)

    def _lookup_parked(self, input_ids: torch.Tensor, live_match, pending_req=None):
        if self.park_store is None or len(input_ids) < self.park_store.min_tokens:
            return None
        generation = self.park_generation
        if (
            pending_req is not None
            and getattr(pending_req, "park_probe_generation", None) == generation
        ):
            return getattr(pending_req, "park_entry", None)

        live_cached_len = live_match.cached_len
        # Restoring must buy enough extra prefill to repay the measured transfer: 4,096 tokens
        # from SSD, one page from pinned RAM. This also prevents an on-card match from being
        # pointlessly replaced by the same immutable entry retained in the park store.
        margin = 4096 if self.park_store.mode == "ssd" else self.page_size
        # Keep one page for the current turn's uncached tail and ask the store for the longest
        # entry that can fit. A too-large parked prefix must not hide a shorter usable one.
        max_len = max(live_cached_len, self.available_size - self.page_size)
        keys = None
        if pending_req is not None:
            keys = getattr(pending_req, "park_keys", None)
            if keys is None:
                from freetoken.kvcache.park_store import rolling_page_keys

                keys = rolling_page_keys(
                    input_ids,
                    self.page_size,
                    getattr(self.park_store, "fingerprint", ""),
                )
                pending_req.park_keys = keys
        try:
            entry = self.park_store.lookup(
                input_ids,
                min_len=live_cached_len + margin,
                max_len=max_len,
                **({"keys": keys} if keys is not None else {}),
            )
        except TypeError as exc:
            # Keep small test/dummy stores compatible while production ParkStore supports the
            # precomputed-key fast path. A real store's TypeError must not be hidden.
            if keys is None or "keys" not in str(exc):
                raise
            entry = self.park_store.lookup(
                input_ids,
                min_len=live_cached_len + margin,
                max_len=max_len,
            )
        if pending_req is not None:
            pending_req.park_entry = entry
            pending_req.park_probe_generation = self.park_generation
        return entry

    def _restore_parked(self, input_ids: torch.Tensor, live_match, entry):
        needs_restore = entry is not None
        live_cached_len = live_match.cached_len
        pages = None
        slot = None
        suffix_indices = None
        # Allocation may park or evict unlocked leaves. Protect the live match while the missing
        # suffix is materialized so a failed restore can safely return the original handle.
        if needs_restore:
            self.prefix_cache.inc_lock(live_match.node)
        try:
            ready = True
            if needs_restore:
                self.ensure_mamba_slots(1)
                ready = self.linear_state_pool.num_free_slots >= 1
            agreed = (
                self.park_consensus(f"ready:{int(ready)}".encode())
                if self.park_consensus is not None
                else ready
            )
            if not ready or not agreed:
                return None, self.park_consensus is None

            restored = True
            if needs_restore:
                try:
                    live_pages = live_cached_len // self.page_size
                    suffix_pages = entry.token_count // self.page_size - live_pages
                    pages = self._allocate(suffix_pages)
                    slot = self.linear_state_pool.alloc(1)[0]
                    suffix_indices = self._page_to_token(pages)
                    self.park_store.restore(
                        entry, pages, slot, page_offset=live_pages
                    )
                except Exception as exc:
                    # A restore that quietly becomes a cold prefill is invisible from outside:
                    # ParkStore.restore has already dropped the entry, so only the counters move.
                    logger.warning(f"KV park restore failed, using a cold prefill: {exc!r}")
                    self.park_store.note_error(f"restore failed: {exc!r}")
                    restored = False
            agreed = (
                self.park_consensus(f"restore:{int(restored)}".encode())
                if self.park_consensus is not None
                else restored
            )
            if not restored or not agreed:
                if slot is not None:
                    self.linear_state_pool.free(slot)
                if suffix_indices is not None:
                    self._free(suffix_indices)
                return None, self.park_consensus is None
            if not needs_restore:
                return live_match, True

            indices = torch.cat([live_match.kv_indices, suffix_indices])
            prefix_len, mamba_exist = self.prefix_cache.insert(
                entry.token_ids, indices, slot
            )
            self._free(indices[live_cached_len:prefix_len])
            if mamba_exist:
                self.linear_state_pool.free(slot)
            return self.prefix_cache.match_prefix(input_ids), True
        finally:
            if needs_restore:
                self.prefix_cache.dec_lock(live_match.node)

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        # Multimodal requests must not reuse a shared prefix: image-placeholder tokens
        # have identical ids across images but carry different content (and KV), so a
        # match would serve the wrong image's KV. Match against the empty prefix.
        ids = (
            req.input_ids[:0]
            if getattr(req, "cache_private", False) or req.mm_embeds is not None
            else req.input_ids[: input_len - 1]
        )
        if self.is_swa:
            from freetoken.kvcache.swa_radix_cache import SWACacheHandle
            m = self.prefix_cache.match_prefix(ids)
            return MatchResult(SWACacheHandle(m.cached_len, m.node, m.kv_indices))
        if self.is_hybrid:
            from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle
            m = self.prefix_cache.match_prefix(ids)
            entry = self._lookup_parked(ids, m, req)
            target_len = entry.token_count if entry is not None else m.cached_len
            target_agreed = True
            if self.park_consensus is not None:
                # The request tokens are already identical across TP ranks. Compare only the best
                # reusable length: persistent entry keys include the rank-specific layout digest.
                target_agreed = self.park_consensus(f"target:{target_len}".encode())
                if not target_agreed:
                    m = self.prefix_cache.match_prefix(ids[:0])
                    entry = None
            if target_agreed and (
                entry is not None
                or (self.park_consensus is not None and target_len > 0)
            ):
                restored, restore_agreed = self._restore_parked(ids, m, entry)
                if not restore_agreed:
                    m = self.prefix_cache.match_prefix(ids[:0])
                elif restored is not None and restored.cached_len > m.cached_len:
                    m = restored
            return MatchResult(
                HybridCacheHandle(m.cached_len, m.node, m.kv_indices), mamba_value=m.mamba_value)
        return self.prefix_cache.match_prefix(ids)

    @property
    def available_size(self) -> int:
        evictable = (self.prefix_cache.full_evictable_size if (self.is_hybrid or self.is_swa)
                     else self.prefix_cache.size_info.evictable_size)
        return evictable + len(self.free_slots) * self.page_size

    @property
    def mamba_available_size(self) -> int:
        """Hybrid only: free GDN state slots + evictable (unlocked) tree snapshots."""
        return self.linear_state_pool.num_free_slots + self.prefix_cache.mamba_evictable_size

    def _park_candidates(self):
        """Return only snapshots from a completed turn, never an intermediate prefill leaf.

        The 2026-09-08 live 190k run produced 42 SSD entries while ``kv_park_idle_ms=0`` because
        allocation pressure treated every snapshot-bearing leaf as parkable. A finish marker lets
        the idle path keep one store entry per turn while short-pool pressure still falls through
        to ordinary eviction for unfinished intermediate state.
        """
        return [
            candidate
            for candidate in self.prefix_cache.park_candidates()
            # Direct cache fixtures predate the finish marker; preserve their old park behavior.
            # Scheduler-created intermediate snapshots are explicitly marked False below.
            if getattr(candidate.node, "park_finished", True)
        ]

    def _park_candidate(self, candidate, *, synchronous: bool = False) -> bool:
        if self.park_store is None or self._temporary_lease_depth:
            return False
        if not getattr(candidate.node, "park_finished", True):
            return False
        page_bases = candidate.kv_indices[:: self.page_size]
        if synchronous:
            if not self.park_store.save(
                candidate.input_ids, page_bases, candidate.mamba_slot
            ):
                return False
            pending = None
        else:
            pending = self.park_store.offer(
                candidate.input_ids, page_bases, candidate.mamba_slot
            )
            if pending is None:
                return False
        evicted = self.prefix_cache.detach_parked(candidate)
        if not evicted.mamba_slots:
            return False
        self._bump_park_generation()
        if pending is None:
            if evicted.lock_node is not None:
                self.prefix_cache.dec_lock(evicted.lock_node)
            self.linear_state_pool.free(evicted.mamba_slots)
            self._free(evicted.kv_indices)
        else:
            # The radix node is gone, but its physical pages/state remain exclusively owned here
            # until the worker's D2H completion signal makes them safe to reissue.
            self._pending_parks.append((pending, evicted))
            self.drain_pending_parks()
        return True

    def drain_pending_parks(self, *, wait: bool = False) -> int:
        """Return copied sources to their allocators; never mutate free lists inside an MTP lease."""
        if self._temporary_lease_depth:
            return 0
        kept = []
        drained = 0
        for pending, evicted in self._pending_parks:
            if wait:
                pending.wait_copied()
            if not pending.copy_done.is_set():
                kept.append((pending, evicted))
                continue
            if evicted.lock_node is not None:
                self.prefix_cache.dec_lock(evicted.lock_node)
            self.linear_state_pool.free(evicted.mamba_slots)
            self._free(evicted.kv_indices)
            drained += 1
        self._pending_parks = kept
        return drained

    def _park_lru(self, *, synchronous: bool = False) -> bool:
        if self.park_store is None or self._temporary_lease_depth:
            return False
        for candidate in self._park_candidates():
            if self._park_candidate(candidate, synchronous=synchronous):
                return True
        return False

    def park_idle(self, *, now_ns: int | None = None) -> int:
        """Park completed-turn leaves older than the configured delay; idle scheduler only."""
        if self.park_store is None or self._temporary_lease_depth:
            return 0
        if now_ns is None:
            import time

            now_ns = time.monotonic_ns()
        threshold = self.park_store.idle_ms * 1_000_000
        parked = 0
        for candidate in self._park_candidates():
            if now_ns - candidate.timestamp < threshold:
                continue
            if self._park_candidate(candidate):
                parked += 1
        return parked

    def next_park_delay_ms(self, *, now_ns: int | None = None) -> int | None:
        """Milliseconds until idle parking or a pending-copy poll; ``None`` can block forever."""
        if self.park_store is None:
            return None
        if self._pending_parks:
            return 10
        candidates = self._park_candidates()
        timestamp = min((candidate.timestamp for candidate in candidates), default=None)
        if timestamp is None:
            return None
        if now_ns is None:
            import time

            now_ns = time.monotonic_ns()
        remaining = self.park_store.idle_ms * 1_000_000 - (now_ns - timestamp)
        return max(1, (remaining + 999_999) // 1_000_000)

    def park_status(self) -> dict[str, int | float | str | bool]:
        if self.park_store is None:
            return {
                "mode": "off",
                "parked_count": 0,
                "parked_bytes": 0,
                "hits": 0,
                "misses": 0,
                "last_restore_ms": 0.0,
                "last_restore_breakdown_ms": {},
                "disabled": False,
                "last_error": None,
            }
        return self.park_store.status()

    @property
    def swa_available_size(self) -> int:
        """SWA only: free swa-pool slots + (radix) evictable unlocked live tree swa tokens.
        Naive has no tree, so only the free-list counts."""
        tree = self.prefix_cache.swa_evictable_size if self.is_swa else 0
        return self.swa_pool.swa_available_size() + tree

    def ensure_swa_slots(self, n: int) -> None:
        """Free swa-pool slots until >= ``n`` are available by tombstoning LRU tree swa nodes
        (evict_swa, internal -> tombstone in place / leaf -> free both pools), returning their swa
        slots to the pool and any deleted-leaf full KV to free_slots."""
        while self.swa_pool.swa_available_size() < n:
            ev = self.prefix_cache.evict_swa(n - self.swa_pool.swa_available_size())
            if ev.swa_indices.numel() == 0:
                break
            self.swa_pool.free_swa(ev.swa_indices)
            if ev.kv_indices.numel():
                self._free(ev.kv_indices)

    def ensure_mamba_slots(self, n: int) -> None:
        """Free GDN state slots until >= ``n`` are available by tombstoning LRU tree snapshots
        (evict_mamba), returning their slots + any freed KV to the pools."""
        self.drain_pending_parks()
        while self.linear_state_pool.num_free_slots < n:
            if self._park_lru():
                self.drain_pending_parks()
                continue
            er = self.prefix_cache.evict_mamba(n - self.linear_state_pool.num_free_slots)
            if er.mamba_slots:
                self.linear_state_pool.free(er.mamba_slots)
                self._free(er.kv_indices)
                continue
            if self._pending_parks:
                self.drain_pending_parks(wait=True)
                continue
            break

    def snapshot_toolcall_anchor(self, reqs: List[Req]) -> None:
        """Freeze each decoding request's GDN state at its tool-call anchor, into the ping-pong
        slot that is idle during decode (the kernel-side ×CHUNK track only runs on prefill
        extends). Must run on the engine stream before the current step's kernels: cached_len
        equals the anchor exactly when every enqueued step up to the anchor-consuming one has
        been issued and the next (current) one has not, so the copy lands between them in
        stream order. Reuses ``mamba_last_track_seqlen`` as the pending-donate mark -- the
        prefill track's own pending freeze was consumed by the prefill-commit ``cache_req``
        before any decode drain could set an anchor."""
        if not self.is_hybrid:
            return
        pool = self.linear_state_pool
        for r in reqs:
            a = r.toolcall_anchor_len
            if (
                a is None
                or r.mamba_ping_pong is None
                or r.mamba_last_track_seqlen is not None
                or r.cached_len != a
                or align_down(a, self.page_size) != a
            ):
                continue
            dst = r.mamba_ping_pong[r.mamba_next_track_idx]
            pool.copy_from(r.linear_slot_idx, dst)
            r.mamba_last_track_seqlen = a
            r.mamba_next_track_idx = 1 - r.mamba_next_track_idx

    def maybe_free_swa_out_of_window(self, reqs: List[Req], *, forward_iter: int) -> None:
        """Proactively free each decoding request's now-out-of-window SWA slots, bounding its swa
        footprint to ~one window so a smaller-than-full swa pool (swa_full_tokens_ratio<1) stays
        viable. Mirrors sglang ``ScheduleBatch.maybe_evict_swa`` / ``free_swa_out_of_window_slots``:
        evict every ``interval`` forwards; skip a request's first decode step (its extend forward
        may still be in-flight under overlap); floor the frontier at the request's protected
        (reused) prefix so it only frees its OWN slots, never the tree-shared prefix's swa; and keep
        a ``window + page_size`` margin so the freed slots are out-of-window for every in-flight
        forward."""
        if not self.swa_paged or forward_iter % _SWA_EVICTION_INTERVAL != 0:
            return
        window = self.sliding_window_size
        for req in reqs:
            if req.decode_batch_idx < 1:
                continue                       # overlap guard: extend forward may still be running
            floor = req.cache_handle.cached_len   # reused prefix -> its swa is tree-owned, not ours
            threshold = (req.device_len - 1) - window - self.page_size
            if req.toolcall_anchor_len is not None:
                # Keep the window ending at the anchor resumable: a client-side rewrite of the
                # echoed tool call forks after the anchor, and a resume there needs
                # [anchor - window, anchor) live. The finish-insert then adopts (rather than
                # tombstones) these never-evicted slots; they stay unlocked, so real pool
                # pressure can still reclaim them (same soft retention as the prompt-end pin).
                cap = req.toolcall_anchor_len - window - _SWA_RETAIN_GAP
                if threshold - cap > window + _SWA_RETAIN_GAP:
                    # The decode ran on far past the anchor (a tool call is normally within
                    # tens of tokens of the end). Holding the cap would grow this request's
                    # live swa without bound ("SWA pool exhausted" is unhandled) -- drop the
                    # anchor and let normal eviction resume. This bound is what the
                    # anchor-retention term in _swa_per_req_swa_floor sizes the pool for.
                    req.toolcall_anchor_len = None
                else:
                    threshold = min(threshold, cap)
            new_evicted = align_down(threshold, self.page_size)
            start = max(req.swa_evicted_seqlen, floor)
            if new_evicted > start:
                self._free_swa(self.page_table[req.table_idx, start:new_evicted])
                req.swa_evicted_seqlen = new_evicted

    def free_swa_out_of_window_extend(self, reqs: List[Req]) -> None:
        """Prefill sibling of ``maybe_free_swa_out_of_window``: before allocating a chunk, return
        each request's now-out-of-window SWA slots so a chunked prompt's live swa stays ~one window
        regardless of prompt length (else a prompt longer than the swa pool exhausts alloc_swa).
        Runs on EVERY prefill batch -- no eviction-interval cadence, since a long prompt would
        overflow the pool before a cadence fires. The frontier is based on ``cached_len`` (the
        pre-chunk, already-forwarded length; the chunk ``[cached_len, device_len)`` is allocated by
        the following ``allocate_paged``, not here), so only positions prior chunks consumed are
        freed, floored at the tree-owned reused prefix. Overlap-safe by the same scheduler stream
        gate + ``window + page_size`` margin the decode driver relies on; ``free_swa`` is idempotent
        over the sentinel, so re-freeing an earlier chunk's range is a no-op. The pool is always
        sized > one window (see the swa-pool floor), so a chunk can always make forward progress."""
        if not self.swa_paged:
            return
        window = self.sliding_window_size
        for req in reqs:
            floor = req.cache_handle.cached_len   # reused prefix -> its swa is tree-owned, not ours
            new_evicted = align_down(req.cached_len - window - self.page_size, self.page_size)
            start = max(req.swa_evicted_seqlen, floor)
            if new_evicted > start:
                self._free_swa(self.page_table[req.table_idx, start:new_evicted])
                req.swa_evicted_seqlen = new_evicted

    def lock(self, handle: BaseCacheHandle) -> None:
        if self.is_swa:
            # records the window boundary on the (frozen) handle for unlock/dec_lock.
            object.__setattr__(handle, "swa_uuid", self.prefix_cache.inc_lock(handle.node))
        elif self.is_hybrid:
            self.prefix_cache.inc_lock(handle.node)
        else:
            self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        if self.is_swa:
            self.prefix_cache.dec_lock(handle.node, handle.swa_uuid)
        elif self.is_hybrid:
            self.prefix_cache.dec_lock(handle.node)
        else:
            self.prefix_cache.lock_handle(handle, unlock=True)

    def _free_swa(self, indices: torch.Tensor) -> None:
        """Free the swa-pool slots backing ``indices`` (full-pool slots). Idempotent over the
        0 sentinel, so safe to call on any slots being returned to free_slots."""
        if self.swa_pool is not None and len(indices) > 0:
            self.swa_pool.free_swa(indices)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            if self.swa_paged:
                # Each newly-allocated full token needs a swa-pool slot (where its SWA-layer KV is
                # written; read back via the full->swa mapping). radix reuses the existing prefix's
                # (live, mapped) swa slots and evicts tree swa if the pool is short; naive has no
                # tree (the pool is sized concurrency x window so it always fits).
                if self.is_swa:
                    self.ensure_swa_slots(len(allocated))
                self.swa_pool.alloc_swa(allocated)
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # Design rule: a speculative step must never reach a commit. Every commit path below
        # inserts into the prefix cache, frees the deduped pages and RE-POINTS this request's
        # page-table row at the tree's canonical pages (and the hybrid one also donates its GDN
        # state slot) -- none of which a device_len rewind can undo. Speculation lives inside
        # the decode phase, where commits are unreachable by construction; assert it anyway.
        if getattr(req, "spec_inflight", None) is not None:
            raise RuntimeError(
                f"request {req.uid} has {req.spec_inflight.width} unsettled speculative rows; "
                "a prefix commit re-points its page-table row and cannot be rolled back "
                "(settle the step with Scheduler._rollback_spec_tokens first)"
            )
        if self.is_swa:
            return self._cache_req_swa(req, finished=finished)
        if self.is_hybrid:
            return self._cache_req_hybrid(req, finished=finished)
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        # Multimodal requests are never inserted into the shared prefix cache (see
        # ``match_req``). Their KV pages stay owned by the active request and are freed
        # on completion; nothing is exposed for cross-request reuse.
        if getattr(req, "cache_private", False) or req.mm_embeds is not None:
            # A private request keeps its admission handle for its whole lifetime. The
            # prefill-to-decode boundary is deliberately a no-op: no shared insert, no
            # unlock, and no page donation. Completion/abort releases everything once.
            if not finished:
                return
            self.unlock(old_handle)
            tail = self._padded_tail(req, old_handle.cached_len)
            if self.swa_paged:
                self._free_swa(tail)
            self._free(tail)
            return
        insert_ids = req.input_ids[: req.cached_len]
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it. A naive-SWA request (swa_paged, no
        # reuse) also returns the swa slots backing every full slot it frees; the out-of-window
        # ones were already freed by the decode driver (free_swa is idempotent over the sentinel).
        if self.swa_paged:
            self._free_swa(page_indices[old_handle.cached_len : cached_len])
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            tail = self._padded_tail(req, new_handle.cached_len)
            if self.swa_paged:
                self._free_swa(tail)
            self._free(tail)
        else:  # keep the tail part, update the handle
            # Re-point the deduped span at the tree's canonical pages: the request's own pages
            # for [old_handle.cached_len, cached_len) went back on the free list above, but the
            # attention backends read this row every step and the next allocation hands those
            # pages to someone else. [0, old_handle.cached_len) needs no rewrite -- it has been
            # locked since admission, so the row already equals canonical there.
            if cached_len > old_handle.cached_len:
                canonical = new_handle.get_matched_indices()
                self.page_table[req.table_idx, old_handle.cached_len : cached_len].copy_(
                    canonical[old_handle.cached_len : cached_len])
            req.cache_handle = new_handle
            self.lock(new_handle)

    def _mark_finished_park_node(self, input_ids: torch.Tensor) -> None:
        # Parking off pays nothing: the marker only steers park candidates, and the walk below
        # needs the C++ radix compare (absent on the GPU-less devbox, where these paths must
        # still run in tests). With parking on it is one fast_compare_key over the prefix per
        # commit, well under a millisecond at 190k tokens (R1 follow-up: have insert() return
        # the node and drop the walk).
        if self.park_store is None or len(input_ids) == 0:
            return
        match = self.prefix_cache.match_prefix(input_ids)
        if match.mamba_value is not None:
            # ``park_candidates`` can contain snapshots from chunked prefill. Mark only the final
            # completed-turn boundary; intermediate snapshots remain ordinary eviction victims.
            match.node.park_finished = True

    def _mark_unfinished_park_node(self, input_ids: torch.Tensor) -> None:
        if self.park_store is None or len(input_ids) == 0:
            return
        match = self.prefix_cache.match_prefix(input_ids)
        if match.mamba_value is not None:
            match.node.park_finished = False

    def _eager_checkpoint_wanted(self, req: Req) -> bool:
        """SSD parking only, and never for an aborted request (an aborted chunked prefill
        would otherwise promote a half-built prompt into a saved checkpoint)."""
        return (
            self.park_store is not None
            and getattr(self.park_store, "mode", None) == "ssd"
            and not self._temporary_lease_depth
            and not getattr(req, "aborted", False)
        )

    def _checkpoint_match(self, match, L: int) -> bool:
        """``match`` is the exact, page-aligned, state-bearing tree node at ``L``."""
        store = self.park_store
        return (
            store is not None
            and match.mamba_value is not None
            and not match.node.is_root()
            and match.cached_len == L
            and L >= store.min_tokens
            and L % self.page_size == 0
        )

    def _save_prompt_checkpoint(self, match, input_ids: torch.Tensor, L: int) -> bool:
        """Persist the final-prefill checkpoint at ``L`` as its own SSD segment, synchronously
        and without detaching anything from the tree.

        On the live 5090 (2026-09-10, v4 parent links) real agent traffic linked almost no
        saves and hit 1 restore in 18: the harness sends one conversation prefix two or three
        times per turn with different tails, and the next real turn diverged 44-148 tokens
        before the end of the previous *finish* save, whose state encodes the side request's
        tail. The prefix every later request really shares is this prompt checkpoint, and it
        must be saved now: the engine has 12 state slots, so an internal snapshot may be
        tombstoned long before it becomes a parkable leaf. The save reads the canonical tree
        pages and the frozen tree slot (never the request's advancing live slot) while the
        request's lock protects them, and returns before the caller releases anything, so
        page/state ownership is unchanged. This is a deliberate, bounded synchronous choice:
        it may delay the final-prefill reply while the disk write completes (J4 records that
        latency) but adds no second GPU-ownership mode or extra state slot. A first save of a
        new conversation is the expensive one; later checkpoints add one state plus changed
        pages. Failure or refusal leaves inference untouched and surfaces in park status.
        """
        store = self.park_store
        if store is None:
            return False
        try:
            return bool(
                store.save(input_ids[:L], match.kv_indices[:: self.page_size], match.mamba_value)
            )
        except Exception as exc:
            logger.warning(f"KV park prompt checkpoint save failed: {exc!r}")
            note_error = getattr(store, "note_error", None)
            if note_error is not None:
                note_error(f"prompt checkpoint save failed: {exc!r}")
            return False

    def _cache_req_hybrid(self, req: Req, *, finished: bool) -> None:
        """Hybrid (GDN) cache_req: commit KV like radix AND manage the GDN state snapshot.
        Prefill chunk commit: DONATE the frozen ping-pong slot (the snapshot the forward wrote
        at the tracked ×64 boundary mamba_last_track_seqlen) into the tree; replace it with a
        fresh slot if the tree took it (dedup keeps it for reuse). Finish: donate the live slot
        (final full-sequence state, zero-copy since the req is done) and free the req's slots."""
        from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle

        pool = self.linear_state_pool
        old_handle = req.cache_handle
        page_indices = self.page_table[req.table_idx, : req.cached_len]

        if getattr(req, "cache_private", False) or req.mm_embeds is not None:
            if not finished:
                return
            self.unlock(old_handle)
            self._free(page_indices[old_handle.cached_len :])
            self._free_req_slots(req)
            return

        if finished:
            # A pending freeze (the tool-call anchor, or a prefill ×64 track the request
            # finished too early to chunk-commit) is a strictly shorter prefix than the live
            # donate below: insert it first and advance the dedup-free floor to its boundary
            # -- [prefix_len, L) is now tree-owned by the donated node, so only [old, prefix_len)
            # is this request's dup to free. The frozen slot is consumed either way (taken by
            # the tree or freed here) and both ping-pong refs are dropped before
            # _free_req_slots so nothing double-frees.
            free_upto = old_handle.cached_len
            L = req.mamba_last_track_seqlen
            if (
                L is not None
                and 0 < L <= req.cached_len
                and align_down(L, self.page_size) == L
                and req.mamba_ping_pong is not None
            ):
                frozen_idx = 1 - req.mamba_next_track_idx
                frozen = req.mamba_ping_pong[frozen_idx]
                prefix_len, mamba_exist = self.prefix_cache.insert(
                    req.input_ids[:L], page_indices[:L], frozen)
                self._bump_park_generation()
                checkpointed = False
                if self._eager_checkpoint_wanted(req):
                    # The frozen checkpoint at L is a strictly shorter prefix than the finish
                    # donate below and would otherwise be freed or evicted before it could
                    # ever be parked. Save it now from the canonical pages and tree slot,
                    # under a temporary lock of its own node (old_handle covers only the
                    # admission prefix), exactly as the final-prefill commit does.
                    m = self.prefix_cache.match_prefix(req.input_ids[:L])
                    if self._checkpoint_match(m, L):
                        self.prefix_cache.inc_lock(m.node)
                        try:
                            self._save_prompt_checkpoint(m, req.input_ids, L)
                        finally:
                            self.prefix_cache.dec_lock(m.node)
                        m.node.park_finished = True
                        checkpointed = True
                if not mamba_exist and not checkpointed:
                    self._mark_unfinished_park_node(req.input_ids[:L])
                pool.free([s for s in req.mamba_ping_pong if mamba_exist or s != frozen])
                req.mamba_ping_pong = None
                self._free(page_indices[free_upto : max(free_upto, prefix_len)])
                free_upto = max(free_upto, L)
            # Donate the live slot (final full-sequence state). The live state is at cached_len;
            # only attach it when cached_len is itself the page-aligned node boundary (always for
            # page_size==1). For page_size>1 a non-aligned cached_len would attach an over-advanced
            # state to a shorter prefix node -> skip the finish-donate (the ×64 prefill snapshots
            # remain as reuse points).
            insert_len = align_down(req.cached_len, self.page_size)
            keep_live = False
            if insert_len == req.cached_len and insert_len > 0:
                prefix_len, mamba_exist = self.prefix_cache.insert(
                    req.input_ids[:insert_len], page_indices[:insert_len], req.linear_slot_idx)
                self._bump_park_generation()
                self.unlock(old_handle)
                self._free(page_indices[free_upto : max(free_upto, prefix_len)])
                keep_live = not mamba_exist           # tree now owns linear_slot_idx
            else:
                self.unlock(old_handle)
                self._free(page_indices[free_upto :])
            # The turn is over: whichever snapshot node is deepest on this request's path is
            # the one worth parking -- the donated full-sequence state when cached_len was
            # page-aligned, otherwise the last x64 prefill/decode snapshot (match_prefix walks
            # up to the deepest LIVE snapshot). Marking only inside the donate branch left every
            # real chat unparkable: the 2026-09-08 22:49 live run (190,004-token prompt + 48
            # answer tokens, cached_len 190,052, not a multiple of 64) parked nothing in 90 s
            # and re-read all 190k tokens on turn two (141 s).
            self._mark_finished_park_node(req.input_ids[:insert_len])
            self._free_req_slots(req, keep_live=keep_live)
            return

        # Prefill chunk commit: donate the frozen snapshot at the tracked ×64 boundary.
        L = req.mamba_last_track_seqlen
        if L is None:
            return  # no ×64 boundary crossed this chunk; req keeps its pages (committed later)
        if align_down(L, self.page_size) != L:
            # page_size>1 only: insert would align the key down, attaching a state that encodes
            # L tokens to a SHORTER node -- a future hit would COW-restore an over-advanced
            # state. Skip; the next aligned boundary (or the finish-donate) commits instead.
            req.mamba_last_track_seqlen = None
            return
        frozen_idx = 1 - req.mamba_next_track_idx          # the slot the forward just wrote
        frozen = req.mamba_ping_pong[frozen_idx]
        prefix_len, mamba_exist = self.prefix_cache.insert(
            req.input_ids[:L], page_indices[:L], frozen)
        self._bump_park_generation()
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len : prefix_len])
        # Lock the committed snapshot node FIRST: the replacement-slot alloc below can trigger
        # evict_mamba (via ensure_mamba_slots), which would otherwise reclaim this still-unlocked
        # just-donated node -- freeing its KV pages under the still-decoding request.
        m = self.prefix_cache.match_prefix(req.input_ids[:L])
        # Same re-point as the generic path: the dedup free above returned this request's own
        # pages for [old_handle.cached_len, prefix_len) while its row still named them.
        if prefix_len > old_handle.cached_len:
            self.page_table[req.table_idx, old_handle.cached_len : prefix_len].copy_(
                m.kv_indices[old_handle.cached_len : prefix_len])
        req.cache_handle = HybridCacheHandle(m.cached_len, m.node, m.kv_indices)
        self.lock(req.cache_handle)
        # The scheduler commits only the final prefill (intermediate chunks never reach
        # cache_req), so this node is the request's prompt checkpoint: the shared prefix every
        # later side request and next turn starts from. With SSD parking save it now, before
        # replacement-slot pressure below can tombstone it, and mark it a completed checkpoint
        # so that once its children are gone the ordinary leaf path finds a duplicate key
        # (no rewrite) and releases it. RAM mode and parking-off keep the previous behavior:
        # an unfinished intermediate snapshot that ordinary eviction reclaims.
        if self._eager_checkpoint_wanted(req) and self._checkpoint_match(m, L):
            self._save_prompt_checkpoint(m, req.input_ids, L)
            m.node.park_finished = True
        elif not mamba_exist:
            m.node.park_finished = False
        if not mamba_exist:                                # tree took `frozen`; replace it
            self.ensure_mamba_slots(1)
            pp = list(req.mamba_ping_pong)
            pp[frozen_idx] = pool.alloc(1)[0]
            req.mamba_ping_pong = tuple(pp)
        req.mamba_last_track_seqlen = None

    def _cache_req_swa(self, req: Req, *, finished: bool) -> None:
        """SWA cache_req: commit the request's full KV prefix into the SWARadixCache (node.value =
        the canonical full-pool page indices; the swa KV rides along via the full->swa mapping).
        Tokens < req.swa_evicted_seqlen are marked tombstone on insert. No donate/COW (the swa KV
        is in the pool already). On any dup/tail free, free both pools (full slot + its swa slot)."""
        from freetoken.kvcache.swa_radix_cache import SWACacheHandle

        old_handle = req.cache_handle
        page_indices = self.page_table[req.table_idx, : req.cached_len]

        if getattr(req, "cache_private", False) or req.mm_embeds is not None:
            if not finished:
                return
            self.unlock(old_handle)
            tail = self._padded_tail(req, old_handle.cached_len)
            self._free_swa(tail)
            self._free(tail)
            return

        insert_len = align_down(req.cached_len, self.page_size)
        freed = page_indices[:0]
        if insert_len > 0:
            # insert reconciles tombstones (revives the in-window ones by ADOPTING the request's
            # live-swa slots into node.value) and returns every full slot to reclaim: the displaced
            # old tree slots + the request's non-adopted dups. swa_evicted_seqlen is the request's
            # own extend/decode free frontier: insert must tombstone [.., swa_evicted_seqlen) rather
            # than adopt those (now-sentinel) swa slots. This holds for BOTH finished and unfinished
            # commits -- the extend driver frees out-of-window swa during chunked prefill too, so an
            # unfinished chunk's frontier is already > 0 and must be honored (else insert adopts
            # sentinel slots -> the request's later SWA gathers read slot 0 -> corruption).
            _, freed = self.prefix_cache.insert(
                req.input_ids[:insert_len], page_indices[:insert_len],
                swa_evicted_seqlen=req.swa_evicted_seqlen,
                update_kv_after_len=old_handle.cached_len)
        self.unlock(old_handle)
        self._free_swa(freed)   # idempotent: revived/out-of-window slots are already sentinel -> no-op
        self._free(freed)
        if finished:
            # Page-unaligned tail (page_size>1) not inserted. The padded slice reaches to the
            # page-ceil bound: allocate_paged charged a swa slot for EVERY token of the last
            # partial page (whole-page alloc_swa), so the padding slots must return with it or
            # they leak (-cached_len mod page_size slots per request, permanently).
            tail = self._padded_tail(req, insert_len)
            self._free_swa(tail)
            self._free(tail)
            # Soft-pin the prompt-end window: decode never re-stamps the prompt path, so after
            # the unlock above it is the stalest LRU entry and the first evict_swa victim. A
            # follow-up turn diverges at the prompt end when the client drops reasoning; a cut
            # there only needs the trailing window live, so eagerly reclaim the head's swa
            # (full KV stays for the full-attn layers) and re-stamp the retained tail -- still
            # unlocked, so it remains reclaimable under real pressure.
            prompt_len = align_down(req.max_device_len - req.output_len, self.page_size)
            if prompt_len > 0:
                keep_from = align_down(
                    max(prompt_len - self.sliding_window_size - _SWA_RETAIN_GAP, 0),
                    self.page_size,
                )
                if keep_from > 0:
                    self._free_swa(
                        self.prefix_cache.trim_head_swa(req.input_ids[:prompt_len], keep_from))
                self.prefix_cache.match_prefix(req.input_ids[:prompt_len])
        else:
            # inc_lock is node-granular, and the suffix insert just made this chunk's whole
            # extend one node: locking it would pin the entire chunk's swa for all of decode,
            # though the request reads only its trailing window from here on. Force a node
            # boundary a window back (match_prefix splits) so the lock lands on that window
            # alone. The head stays live and unlocked -- still reusable while the pool is
            # roomy, evictable the moment it is not.
            keep_from = align_down(
                max(insert_len - self.sliding_window_size - _SWA_RETAIN_GAP, 0), self.page_size)
            if keep_from > 0:
                self.prefix_cache.match_prefix(req.input_ids[:keep_from])
            m = self.prefix_cache.match_prefix(req.input_ids[:insert_len])
            # Re-point the page table to the tree's live slots for the committed region. Any dup
            # slots insert reclaimed had their full->swa mapping reset to the 0 sentinel; unlike the
            # full pool (KV survives in place until realloc), a stale swa mapping would make the
            # request's subsequent SWA gathers read the sentinel -> corruption. The reconcile revived
            # the in-window tombstones, so the re-matched slots are live.
            if m.cached_len > 0:
                self.page_table[req.table_idx, : m.cached_len].copy_(m.kv_indices)
            req.cache_handle = SWACacheHandle(m.cached_len, m.node, m.kv_indices)
            self.lock(req.cache_handle)

    def _padded_tail(self, req: Req, start: int) -> torch.Tensor:
        """The request's OWN slice [start, page_ceil(cached_len)) of the page table. A finish
        frees through the page-CEIL bound, not cached_len: allocate_paged allocates (and, when
        swa_paged, charges swa for) whole pages, so the padding [cached_len, page_ceil) belongs
        to the finishing request. ``start`` is page-aligned (a match/insert boundary), so the
        full-pool page bases derived via ``[::page_size]`` are identical to the unpadded slice."""
        end = div_ceil(req.cached_len, self.page_size) * self.page_size
        return self.page_table[req.table_idx, start:end]

    def _free_req_slots(self, req: Req, keep_live: bool = False) -> None:
        """Return a finished request's GDN pool slots: both ping-pong slots, plus the live slot
        unless it was donated to the tree. Idempotent -- clears the refs so a re-entry frees
        nothing (defense-in-depth against the abort/finish double-free, see _free_req_resources)."""
        slots = list(req.mamba_ping_pong) if req.mamba_ping_pong is not None else []
        if not keep_live and req.linear_slot_idx is not None:
            slots.append(req.linear_slot_idx)
        if slots:
            self.linear_state_pool.free(slots)
        req.mamba_ping_pong = None
        req.linear_slot_idx = None

    def check_integrity(self) -> None:
        pending_pages = sum(
            len(evicted.kv_indices) // self.page_size
            for _pending, evicted in self._pending_parks
        )
        if self.is_hybrid:
            pc = self.prefix_cache
            pc.check_integrity()  # structural: every snapshot node owns a slot, refs >= 0
            cache_pages = (pc.full_evictable + pc.full_protected) // self.page_size
            # GDN-slot conservation upper bound: free slots + tree-held snapshots can never
            # exceed the (non-padding) pool capacity; the remainder is held by running requests.
            pool = self.linear_state_pool
            tree_slots = pc.mamba_evictable_size + pc.mamba_protected
            pending_slots = sum(
                len(evicted.mamba_slots) for _pending, evicted in self._pending_parks
            )
            assert pool.num_free_slots + tree_slots + pending_slots <= pool.num_slots - 1, (
                f"GDN-slot leak: free({pool.num_free_slots}) + tree({tree_slots}) + "
                f"pending({pending_slots}) > capacity({pool.num_slots - 1})"
            )
        elif self.is_swa:
            pc = self.prefix_cache
            pc.check_integrity()  # full>=swa refs, tombstone => no swa lock
            cache_pages = (pc.full_evictable + pc.full_protected) // self.page_size
            # swa-slot conservation upper bound: free swa slots + tree-held live swa tokens can
            # never exceed the (non-sentinel) swa-pool capacity; the rest is held by running reqs.
            tree_swa = pc.swa_evictable + pc.swa_protected
            cap = self.swa_pool.swa_num_tokens - 1  # slot 0 is the reserved sentinel
            # check_integrity is idle-only (like the exact full-pool check below), so no request
            # holds a swa slot: free + tree must equal cap exactly. `==` (not `<=`) so a LEAK
            # (free + tree < cap) is caught, not just a double-free (> cap).
            assert self.swa_pool.swa_available_size() + tree_swa == cap, (
                f"SWA-slot leak/double-free: free({self.swa_pool.swa_available_size()}) + "
                f"tree({tree_swa}) != capacity({cap})"
            )
        else:
            self.prefix_cache.check_integrity()
            cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages + pending_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) + pending_pages({pending_pages})"
                f" != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    def close(self) -> None:
        if self.park_store is None:
            return
        self.drain_pending_parks(wait=True)
        self.park_store.close()

    def prepare_rebuild(self) -> None:
        """Park every eligible hybrid prefix before the engine tears down its pools."""
        if self.park_store is not None:
            self.drain_pending_parks(wait=True)
            self.park_store.flush()
            while self._park_lru(synchronous=True):
                pass

    def rebuild(self, num_pages: int, page_table: torch.Tensor) -> None:
        """Re-point the page table and reset page accounting + prefix cache IN PLACE.

        Idle-only: assumes no request holds a live handle. Builds a brand-new prefix
        cache (RadixPrefixCache.reset() is an unimplemented stub) rather than mutating
        the old one. Call prepare_rebuild() before the engine destroys an old parked pool.
        """
        device = page_table.device
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * self.page_size
        self.prefix_cache = self._make_prefix_cache(device, self.page_size, self.cache_type)
        # The discarded hybrid tree owned donated GDN-snapshot slots; rebuild is idle-only, so
        # reclaim the whole LinearStatePool free-list (else those slots leak -> admission hangs).
        if self.is_hybrid:
            self.linear_state_pool.reclaim_all_slots()
        self._bump_park_generation()

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            # clone: callers pass page-table VIEWS, and the deferred concat below only reads them
            # when the region exits. A commit that re-points the row in between (the dedup
            # re-point, the SWA one) would otherwise rewrite the pending free list underneath us
            # and return the tree's canonical pages instead of the request's duplicates.
            lazy_free_list.append(indices[:: self.page_size].clone())

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            if lazy_free_list:
                self.free_slots = torch.cat([self.free_slots] + lazy_free_list)
                self._bump_park_generation()

    @contextmanager
    def temporary_page_lease(
        self,
        needed_pages: int,
        *,
        forbidden_pages: torch.Tensor | None = None,
    ):
        """Lease immediately free pages without eviction and restore exact allocation order."""
        if needed_pages < 1:
            raise ValueError("temporary page lease must request at least one page")
        if needed_pages > len(self.free_slots):
            raise RuntimeError(
                f"temporary page lease needs {needed_pages} free pages, "
                f"only {len(self.free_slots)} are immediately free"
            )

        before = self.free_slots.clone()
        pages = before[:needed_pages].clone()
        page_values = [int(value) for value in pages.detach().cpu().tolist()]
        if len(set(page_values)) != len(page_values):
            raise ValueError("temporary page lease pages must be unique")
        if any(value % self.page_size for value in page_values):
            raise ValueError("temporary page lease pages must be page-aligned")
        limit = self.num_pages * self.page_size
        if any(value < 0 or value >= limit for value in page_values):
            raise ValueError("temporary page lease page is out of range")
        forbidden_values = (
            set()
            if forbidden_pages is None
            else {
                int(value)
                for value in forbidden_pages.detach().reshape(-1).cpu().tolist()
            }
        )
        if forbidden_values.intersection(page_values):
            raise RuntimeError("temporary page lease intersects a forbidden live page")

        self.free_slots = self.free_slots[needed_pages:]
        expected_remainder = before[needed_pages:]
        self._temporary_lease_depth += 1
        try:
            yield pages
        finally:
            try:
                if not torch.equal(self.free_slots, expected_remainder):
                    raise RuntimeError(
                        "MTP temporary page lease observed allocator mutation"
                    )
                self.free_slots = before
            finally:
                self._temporary_lease_depth -= 1

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if self.park_store is not None:
            self.drain_pending_parks()
        if (
            needed_pages > len(self.free_slots)
            and self.is_hybrid
            and self.park_store is not None
        ):
            # Queue cold leaves while the worker has room. Their source pages remain pending,
            # so ordinary eviction below may reclaim a different leaf rather than wait for SSD.
            while needed_pages > len(self.free_slots) and self._park_lru():
                pass
            self.drain_pending_parks()
        if needed_pages > len(self.free_slots):
            need = (needed_pages - len(self.free_slots)) * self.page_size
            if self.is_swa:
                # Evicting KV leaf nodes drops their swa slots too -> return both pools.
                ev = self.prefix_cache.evict_full(need)
                evicted = ev.kv_indices
                self._free_swa(ev.swa_indices)
            elif self.is_hybrid:
                # Evicting KV leaf nodes drops their GDN snapshots too -> return both pools.
                er = self.prefix_cache.evict_full(need)
                evicted = er.kv_indices
                if er.mamba_slots:
                    self.linear_state_pool.free(er.mamba_slots)
            else:
                evicted = self.prefix_cache.evict(need)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
        if needed_pages > len(self.free_slots) and self._pending_parks:
            # Only wait when no ordinary evictable leaf can satisfy the request. The wait ends at
            # D2H completion, never at the later SSD manifest update.
            self.drain_pending_parks(wait=True)
        assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])
            self._bump_park_generation()

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    # Pinned only when there is a device to copy to asynchronously; CPU-only runs (unit tests,
    # a CPU CI runner) would otherwise raise instead of just doing a plain host allocation.
    pin = torch.cuda.is_available()
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=pin)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=pin)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    assert allocated.dtype == page_table.dtype, (
        f"allocated dtype {allocated.dtype} != page_table dtype {page_table.dtype}"
    )
    page_table[table_idxs, offsets] = allocated
