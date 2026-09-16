from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Tuple

import torch
from freetoken import diag
from freetoken.core import Batch, Req
from freetoken.utils import align_down, div_ceil, init_logger

from .mm import mm_chunk_end, mm_rows_after
from .utils import PendingReq

if TYPE_CHECKING:
    from freetoken.kvcache import BaseCacheHandle
    from freetoken.message import UserMsg
    from freetoken.mm.encoder_cache import EncoderCache

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager
    from .prefixes import PrefixCoordinator

logger = init_logger(__name__)


def _maybe_pinned(t: torch.Tensor) -> torch.Tensor:
    """Pinning only buys the async H2D copy below; without a device it just raises."""
    return t.pin_memory() if torch.cuda.is_available() else t


class ChunkedReq(Req):
    def _alloc_ids_buf(self) -> None:
        pass  # never sampled; keep input_ids a view of the pending prompt

    def append_host(self, tokens: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    encoder_cache: EncoderCache | None = None
    # end a chunk before an image it would cut; only models whose image spans attend in both directions need it
    keep_images_whole: bool = False
    # the whole budget of this pass; token_budget shrinks as requests are admitted
    pass_budget: int = 0
    # SWA-pool tokens charged to reqs admitted so far this pass. Mirrors reserved_size: swa is
    # allocated only in allocate_paged (after the pass), so swa_available_size does not decrement
    # across the admission loop -- without this, successive admits all see the full pool.
    reserved_swa: int = 0
    blocked_by_cache: bool = field(default=False, init=False)
    # Dynamic KV pool hooks (scheduler/kv_dynamic.py). Both None = today's behaviour.
    on_reserved: Callable[[int], None] | None = None
    capacity_blocked: List[PendingReq] = field(default_factory=list)

    def _note_capacity_blocked(self, req: PendingReq) -> None:
        """Record a never-started request this pass could not seat, with the room the requests
        that DID hold the pool were using. The dynamic KV controller needs that figure to size a
        grow the blocked request can run BESIDE (spec rule 2, concurrent target): by the time it
        escalates at the idle point those requests have finished and reserved_size is 0, so the
        number has to be captured here, at the moment the block happened."""
        req.blocked_reserved_tokens = self.reserved_size
        self.capacity_blocked.append(req)

    def __post_init__(self) -> None:
        if not self.pass_budget:
            self.pass_budget = self.token_budget


    def _try_allocate_one(self, req: PendingReq):
        if self.table_manager.available_size == 0:
            return None

        # Retry gate (R1, 2026-09-08): a never-admitted request is re-probed only when something
        # that can change the outcome moved -- the parking generation (pages freed/inserted/
        # parked/rebuilt) or the free KV / GDN-slot counts. Without this, the 190k live run
        # re-hashed 2,970 pages and re-probed the park store on EVERY scheduler iteration while a
        # governor rebuild waited. Continuation chunks of an already-admitted request bypass it:
        # their pages are held by the prior chunk and nothing needs to change for them to run.
        generation = (
            self.cache_manager.park_generation,
            self.cache_manager.available_size,
            self.cache_manager.linear_state_pool.num_free_slots
            if self.cache_manager.is_hybrid else 0,
        )
        if req.chunked_req is None and getattr(req, "admission_generation", None) == generation:
            return None
        req.admission_generation = generation
        req.admission_error = None

        # TODO: consider host cache match case
        mr = getattr(req, "preparation_match", None)
        if mr is None:
            mr = self.cache_manager.match_req(req)
        else:
            del req.preparation_match
        handle = mr.cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len
        # Reject only what can NEVER fit, even with an empty cache and no other request running.
        # reserved_size (other requests' in-flight decode) is deliberately not subtracted: that
        # shrinks as they finish, so a request blocked by it must wait, not be refused (R1).
        empty_cache_limit = self.cache_manager.num_pages * self.cache_manager.page_size
        if estimated_len > empty_cache_limit:
            reason = (
                f"KV admission gate rejected request {req.uid}: needs {estimated_len} tokens "
                f"(prompt {extend_len} + output budget {req.output_len}), but the KV pool holds "
                f"{empty_cache_limit} tokens even when empty"
            )
            req.admission_error = reason
            logger.warning_rank0(reason)
            return None

        from .cache import admission_fits

        if not admission_fits(need_now=estimated_len, reserved=self.reserved_size,
                              available=self.cache_manager.available_size, protect_tokens=0):
            self.blocked_by_cache = True
            if req.chunked_req is None:
                self._note_capacity_blocked(req)
            return None
        self.cache_manager.lock(handle)
        if not admission_fits(need_now=estimated_len, reserved=self.reserved_size,
                              available=self.cache_manager.available_size, protect_tokens=0):
            self.blocked_by_cache = True
            if req.chunked_req is None:
                self._note_capacity_blocked(req)
            return self.cache_manager.unlock(handle)

        # Second currency (hybrid GDN): reserve 1 live + 2 ping-pong state slots; evict tree
        # snapshots if the pool is short, fail admission if still short (mirrors the KV gate).
        if self.cache_manager.is_hybrid:
            pool = self.cache_manager.linear_state_pool
            if pool.num_free_slots < 3:
                self.cache_manager.ensure_mamba_slots(3)
            if pool.num_free_slots < 3:
                self.blocked_by_cache = True
                return self.cache_manager.unlock(handle)

        # Third currency (SWA): refuse admission unless the swa pool can seat this request's first
        # chunk / one window (the per-chunk charge is in _add_one_req; the reclaim -- radix
        # evict_swa -- happens in allocate_paged, so no ensure here; swa_available_size already
        # folds the evictable tree). For naive (no tree) this can only refuse, which is correct.
        if self.cache_manager.swa_paged:
            ps = self.cache_manager.page_size
            # swa is charged per WHOLE page (allocate_paged -> alloc_swa), so the seat check is
            # in page units too; identical at page_size==1.
            need_swa = div_ceil(
                min(max(extend_len, 1), self.cache_manager.sliding_window_size) + 1, ps
            ) * ps
            if self.cache_manager.swa_available_size - self.reserved_swa < need_swa:
                return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            device_ids.copy_(_maybe_pinned(req.input_ids[:cached_len]), non_blocking=True)
            # Write the matched indices into the TAIL of the page_entry: a cache may return
            # fewer matched indices than cached_len, in which case only the trailing n slots are
            # known-live. Today both the generic radix and the SWA radix match a prefix whose
            # full-loc row is entirely live (n == cached_len), so the tail IS the whole prefix.
            # (DSV4 reads this table too: its pool's full_loc_map is attached to it.)
            matched = handle.get_matched_indices()
            n = int(matched.numel())
            self.table_manager.page_table[table_idx][cached_len - n : cached_len].copy_(matched)

        linear_slot_idx = ping_pong = None
        if self.cache_manager.is_hybrid:
            pool = self.cache_manager.linear_state_pool
            linear_slot_idx = pool.alloc(1)[0]
            ping_pong = tuple(pool.alloc(2))

        return handle, table_idx, linear_slot_idx, ping_pong, mr.mamba_value

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
        linear_slot_idx: int | None = None,
        ping_pong: tuple | None = None,
        next_track_idx: int = 0,
        restore_src: int | None = None,
        swa_evicted_seqlen: int = 0,
    ) -> Req | None:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        if self.cache_manager.swa_paged:
            # Cap this chunk by the swa the pool can back this pass. swa is allocated per token in
            # allocate_paged, and token_budget (max_extend_tokens, default 8192) won't chunk a
            # shorter prompt -- so this cap is what forces a prompt whose swa footprint exceeds the
            # pool to chunk. Credit the slots THIS request's own extend-free (in _prepare_batch,
            # which runs AFTER this sizing) will release this batch, else a continuation sees a
            # drained pool and stalls at chunk_size 0.
            cm = self.cache_manager
            window, ps = cm.sliding_window_size, cm.page_size
            floor = cache_handle.cached_len
            new_evicted = align_down(cached_len - window - ps, ps)
            self_reclaim = max(0, new_evicted - max(swa_evicted_seqlen, floor))
            swa_budget = cm.swa_available_size + self_reclaim - self.reserved_swa
            # swa is charged per WHOLE page: cap the chunk so its PAGE-SPAN cost fits the budget
            # (the extend [cached_len, cached_len+chunk) pulls div_ceil(end,ps)-div_ceil(start,ps)
            # fresh pages -- the partial head page was charged by the previous chunk), and reserve
            # that cost, not the raw token count. Degenerates to the token math at page_size==1.
            max_end = (div_ceil(cached_len, ps) + max(swa_budget, 0) // ps) * ps
            chunk_size = min(chunk_size, max(max_end - cached_len, 0))
            # A continuation resumes the compressor carry at its boundary, which must be
            # page-aligned; the token_budget leftover (unlike max_end) is not. Align the end
            # down when the chunk mints a continuation; no whole page -> retry next pass.
            # 0 <: a chunk the swa cap collapsed to 0 must NOT bail (undersized pool --
            # bailing would livelock; the floor tests pin the loud failure).
            if 0 < chunk_size < remain_len:
                aligned = align_down(cached_len + chunk_size, ps) - cached_len
                if aligned <= 0:
                    return None
                chunk_size = aligned
        align = self.cache_manager.prefill_chunk_align
        if align > 1 and 0 < chunk_size < remain_len:
            # An unaligned chunk end is correct, it just loses this prompt's snapshot boundaries --
            # so keep it when the leftover budget cannot fill one whole unit instead of stalling
            # the request until it gets a bigger turn.
            aligned = align_down(cached_len + chunk_size, align) - cached_len
            chunk_size = aligned if aligned > 0 else chunk_size
        if self.keep_images_whole and pending_req.mm_items and chunk_size < remain_len:
            # a cut image would attend within only the part already in the cache: end the chunk before it, decided last because the caps above only move the end earlier and would undo it
            unit = math.lcm(self.cache_manager.page_size if self.cache_manager.swa_paged else 1, align if align > 1 else 1)
            end = mm_chunk_end(pending_req.mm_items, cached_len, cached_len + chunk_size, unit)
            cut = next((hi for item in pending_req.mm_items for lo, hi in item.offsets if lo < end < hi), None)
            if cut is not None:
                # A new request was checked against the full pass/pool before allocation.
                # Temporary contention must wait; never run approximate bidirectional attention.
                return None
            chunk_size = end - cached_len
        if self.cache_manager.swa_paged:
            ps = self.cache_manager.page_size
            self.reserved_swa += (div_ceil(cached_len + chunk_size, ps) - div_ceil(cached_len, ps)) * ps
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        if self.on_reserved is not None:
            self.on_reserved(pending_req.uid)
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(_maybe_pinned(pending_req.input_ids[_slice]), non_blocking=True)
        req = CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
            mm_embeds=pending_req.mm_embeds,
            cache_private=pending_req.cache_private,
            mrope_position_ids=pending_req.mrope_position_ids,
            mrope_position_delta=pending_req.mrope_position_delta,
            prefill_only=pending_req.prefill_only,
            prefix_key=pending_req.prefix_key,
        )
        req.mm_items = pending_req.mm_items
        req.mrope_positions_full = pending_req.mrope_positions_full
        req.mrope_delta = pending_req.mrope_delta
        # Hybrid GDN per-request state slots (None for non-hybrid). On a fresh admit these are
        # freshly allocated; on a chunked continuation they are inherited from the prior chunk.
        req.linear_slot_idx = linear_slot_idx
        req.mamba_ping_pong = ping_pong
        req.mamba_next_track_idx = next_track_idx
        req.mamba_restore_src = restore_src
        req.swa_evicted_seqlen = swa_evicted_seqlen  # carry the extend-free watermark across chunks
        return req

    def _whole_image_admission_error(self, pending_req: PendingReq) -> str | None:
        """Reject permanent image constraints before taking table, KV, or state resources."""
        if not self.keep_images_whole or not pending_req.mm_items:
            return None
        cm = self.cache_manager
        ps = cm.page_size if cm.swa_paged else 1
        unit = math.lcm(ps, cm.prefill_chunk_align or 1)
        groups: list[list[int]] = []
        for lo, hi in sorted(span for item in pending_req.mm_items for span in item.offsets):
            start = align_down(lo, unit)
            # Intermediate chunks must end aligned. The final prompt chunk may end
            # at its actual length, including any short text tail after the image.
            end = min(pending_req.input_len, div_ceil(hi, unit) * unit)
            if groups and start < groups[-1][1]:
                # No legal aligned boundary exists between these image spans. Treat
                # their union as indivisible, matching mm_chunk_end's backward walk.
                groups[-1][1] = max(groups[-1][1], end)
            else:
                groups.append([start, end])
        for start, end in groups:
            need = end - start
            reason = None
            if need > self.pass_budget:
                reason = f"needs {need} prefill tokens including alignment, but the pass budget is {self.pass_budget}"
            elif cm.swa_paged:
                # Match extend-free's conservative retention rule: a window plus an
                # extra page remains before this chunk. Include page padding after
                # the image too; slot zero is the pool's non-allocatable sentinel.
                retained_start = max(0, align_down(start - cm.sliding_window_size - ps, ps))
                footprint = div_ceil(end, ps) * ps - retained_start
                capacity = cm.swa_pool.swa_num_tokens - 1
                if footprint > capacity:
                    reason = f"needs {footprint} sliding-window pool tokens for retained context and whole image pages, but the pool holds {capacity}"
            if reason is not None:
                return (
                    f"Bidirectional image admission rejected request {pending_req.uid}: "
                    f"aligned image group [{start}, {end}) {reason}; reduce image tokens or increase "
                    "the prefill/sliding-window pool budget"
                )
        return None

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        self.blocked_by_cache = False
        if self.token_budget <= 0:
            return None

        if pending_req.chunked_req is None:
            if reason := self._whole_image_admission_error(pending_req):
                pending_req.admission_error = reason
                logger.warning_rank0(reason)
                return None

        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
                linear_slot_idx=chunked_req.linear_slot_idx,
                ping_pong=chunked_req.mamba_ping_pong,
                next_track_idx=chunked_req.mamba_next_track_idx,
                restore_src=None,  # continuation chunk already has live state
                swa_evicted_seqlen=chunked_req.swa_evicted_seqlen,  # extend-free watermark so far
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx, linear_slot_idx, ping_pong, restore_src = resource
            req = self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
                linear_slot_idx=linear_slot_idx,
                ping_pong=ping_pong,
                next_track_idx=0,
                restore_src=restore_src,
            )
            if req is None:
                # no aligned chunk this pass: undo the admission (a continuation keeps its
                # resources -- they belong to the prior chunk's Req). The match did not fail, so
                # permit another sizing attempt in this same park generation without re-probing.
                pending_req.admission_generation = None
                self.cache_manager.unlock(cache_handle)
                self.table_manager.free(table_idx)
                if linear_slot_idx is not None:
                    self.cache_manager.linear_state_pool.free([linear_slot_idx, *ping_pong])
            return req

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    encoder_cache: EncoderCache | None = None
    keep_images_whole: bool = False
    pending_list: List[PendingReq] = field(default_factory=list)
    rejected: List[Tuple[int, str]] = field(default_factory=list)
    prefix_coordinator: PrefixCoordinator | None = None
    # Dynamic KV pool hooks (scheduler/kv_dynamic.py). Both None = today's behaviour.
    on_reserved: Callable[[int], None] | None = None
    on_capacity_blocked: Callable[[PendingReq], None] | None = None
    _capacity_blocked: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        pending = PendingReq(
            req.uid,
            req.input_ids,
            req.sampling_params,
            mm_embeds=req.mm_embeds,
            # Content pad IDs retain only 30 hash bits. Keep image KV private;
            # encoder embedding dedup uses the full content hash independently.
            cache_private=(
                bool(req.mm_items)
                or req.mm_embeds is not None
                or req.mm_pixel_values is not None
                or req.mm_image_grid_thw is not None
                or req.mm_token_type_ids is not None
            ),
            mrope_position_ids=(req.mrope_position_ids if req.mrope_position_ids is not None else req.mrope_positions),
            mrope_position_delta=(req.mrope_position_delta if req.mrope_position_ids is not None else req.mrope_delta),
            mm_items=req.mm_items,
            mrope_positions_full=req.mrope_positions,
            mrope_delta=req.mrope_delta,
        )
        # Keep the parked-probe state on the request without widening PendingReq's shared wire
        # shape; the ids never change while a request waits for a cache-generation change.
        pending.park_keys = None
        pending.park_probe_generation = None
        # Set by _note_capacity_blocked; declared here so every pending request carries the
        # attribute whether or not an admission pass has ever blocked it.
        pending.blocked_reserved_tokens = 0
        if not pending.mm_items and self.prefix_coordinator is not None and self.prefix_coordinator.route(pending):
            # The shared preparation now owns runnable admission. Drop the dynamic
            # controller's provisional charge for this cold prompt so later followers
            # can join it. Each private tail is reserved when it returns to admission.
            if self.on_reserved is not None:
                self.on_reserved(pending.uid)
        else:
            self.pending_list.append(pending)

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None  # BEFORE the range: an idle scheduler must not look like traffic
        with diag.region("diag.prefill_admit"):
            return self._admit_next_batch(prefill_budget)

    def _admit_next_batch(self, prefill_budget: int) -> Batch | None:
        """Prefix match, admission gates, page/GDN allocation and the prompt's H2D copies."""
        # estimated offset due to in-flight decode
        # Hand back the pages of parks whose device->host copy has completed BEFORE the size
        # gate sees available_size. Draining used to happen only at idle (run_when_idle) or
        # inside the allocation paths; a pending request that fails the size gate keeps the
        # scheduler out of idle and never reaches allocation, so a 190k prefix parked at the
        # end of turn one held its 2.6 GB of pages hostage and turn two could never be admitted
        # (live 2026-09-08 23:18-23:36: one parked entry, main thread spinning in overlap_loop,
        # request pending 18 min). The old per-iteration re-probe reached _allocate by
        # accident; the once-per-generation gate does not, so drain here on purpose.
        if getattr(self.cache_manager, "park_store", None) is not None:
            self.cache_manager.drain_pending_parks()
        adder = PrefillAdder(
            token_budget=prefill_budget,
            # Preparation yields between chunks. Keep its unallocated future pages
            # reserved while another request is admitted, or both can overcommit the pool.
            reserved_size=self.decode_manager.inflight_tokens + sum(
                p.input_len - p.chunked_req.cached_len
                for p in self.pending_list if p.prefill_only and p.chunked_req is not None
            ),
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            on_reserved=self.on_reserved,
            encoder_cache=self.encoder_cache,
            keep_images_whole=self.keep_images_whole,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        prompt_admissions: List[Tuple[int, int, int]] = []
        # Snapshot here, before the forward's complete_one() advances cached_len: the tokens
        # forwarded this batch (extend_len) and the prefix-cache hit. SGLang counts the hit
        # once at admission, so continuation chunks (already-chunked reqs) contribute 0.
        log_new_tokens = 0
        log_cached_tokens = 0
        remaining: List[PendingReq] = []
        pending_list, self.pending_list = self.pending_list, []
        for index, pending_req in enumerate(pending_list):
            # Preparation never shares a batch with user generation. New waiters released
            # by a restored hit land on self.pending_list and are considered next pass.
            if reqs and (pending_req.prefill_only or reqs[0].prefill_only):
                remaining.extend(pending_list[index:])
                break
            coordinator = self.prefix_coordinator
            if coordinator is not None and coordinator.before_admit(pending_req):
                continue
            is_continuation = pending_req.chunked_req is not None
            req = adder.try_add_one(pending_req)
            if (req is None and adder.blocked_by_cache and coordinator is not None
                    and coordinator.release_preferences()):
                # Preference leases must never make a request wait forever for capacity.
                pending_req.admission_generation = None
                if coordinator.before_admit(pending_req):
                    continue
                req = adder.try_add_one(pending_req)
            if req is not None:
                if req.prefill_only and coordinator is not None:
                    coordinator.prepared_chunk(req)
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
                if not is_continuation and not req.prefill_only:
                    if coordinator is not None:
                        coordinator.admitted(req.uid)
                    # Record the COMPLETE prompt length and the prefix-cache hit on the
                    # first chunk. The scheduler publishes them only after _prepare_batch
                    # succeeds; continuation chunks must never publish them again.
                    prompt_admissions.append(
                        (req.uid, pending_req.input_len, req.cache_handle.cached_len)
                    )
                    if pending_req.mm_items and self.encoder_cache is not None:
                        # claim the rows every chunk of this request will gather; the entry outlives the chunks
                        for item in pending_req.mm_items:
                            self.encoder_cache.register(
                                item.hash, req.uid, mm_rows_after(item, req.cache_handle.cached_len)
                            )
                log_new_tokens += req.extend_len
                if not is_continuation:
                    log_cached_tokens += req.cache_handle.cached_len
            else:
                reason = getattr(pending_req, "admission_error", None)
                if reason is not None:
                    self.rejected.append((pending_req.uid, reason))
                    continue
                # A yielding preparation already owns resources needed to finish. A
                # blocked new arrival must not prevent that continuation from running.
                if not reqs and any(p.prefill_only and p.chunked_req is not None
                                    for p in pending_list[index + 1:]):
                    remaining.append(pending_req)
                    continue
                remaining.extend(pending_list[index:])
                break  # We cannot add more requests
        ordinary_chunks = [p for p in chunked_list if not p.prefill_only]
        preparation_chunks = [p for p in chunked_list if p.prefill_only]
        self.pending_list = ordinary_chunks + remaining + preparation_chunks + self.pending_list
        if self.on_capacity_blocked is not None and adder.capacity_blocked:
            # Membership by uid, never by `in`: PendingReq is a dataclass whose __eq__ compares
            # its input_ids TENSOR, so `pending in list` raises "Boolean value of Tensor ... is
            # ambiguous" as soon as the identity shortcut misses (any other entry ahead of it).
            # uids are unique within the pending list, so this is the same test.
            waiting = {pending.uid for pending in self.pending_list}
            known = {pending.uid for pending in self._capacity_blocked}
            for pending in adder.capacity_blocked:
                if pending.uid in waiting and pending.uid not in known:
                    self._capacity_blocked.append(pending)
                    known.add(pending.uid)
                    self.on_capacity_blocked(pending)
        if len(reqs) == 0:
            return None
        batch = Batch(reqs=reqs, phase="prefill")
        batch.log_new_tokens = log_new_tokens
        batch.log_cached_tokens = log_cached_tokens
        batch.prompt_admissions = prompt_admissions
        return batch

    def pop_rejections(self) -> List[Tuple[int, str]]:
        rejected, self.rejected = self.rejected, []
        return rejected

    def pop_capacity_blocked(self) -> List[PendingReq]:
        """Never-started requests still waiting for capacity; removed from pending_list so the
        dynamic controller can hold them (rule 2(b)). Empty unless a hook is installed.

        Filtered at POP time, not only where the entries are collected: a request blocked on one
        admission pass is very often admitted on the next, and handing it back afterwards would
        let the controller hold a RUNNING request and re-append it to the pending list -- a
        second full response for one request. Anything that has left the pending list (admitted,
        rejected or aborted) is dropped silently; it is no longer this list's business.
        """
        blocked, self._capacity_blocked = self._capacity_blocked, []
        # By uid, not by `in`/`remove`: see the note in _admit_next_batch -- comparing
        # PendingReq objects compares their input_ids tensors.
        waiting = {pending.uid for pending in self.pending_list}
        still_waiting: List[PendingReq] = [p for p in blocked if p.uid in waiting]
        if still_waiting:
            taken = {pending.uid for pending in still_waiting}
            self.pending_list = [p for p in self.pending_list if p.uid not in taken]
        return still_waiting

    def abort_req(self, uid: int) -> Req | None:
        if self.prefix_coordinator is not None:
            self.prefix_coordinator.cancel_waiter(uid)
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                chunked_req = req.chunked_req
                # A pending picture continuation owns the complete GPU feature tensor and
                # CPU MRoPE table. Drop both deterministically on cancellation instead of
                # waiting for the PendingReq object to be garbage-collected.
                req.mm_embeds = None
                req.mrope_position_ids = None
                return chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
