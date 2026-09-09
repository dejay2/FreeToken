from __future__ import annotations

import time
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    NamedTuple,
    NoReturn,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
)

import torch
from freetoken import diag
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req, SpecInflight
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.moe.learned_routing import FLUSH_INTERVAL_S
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheParkStatusMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    CacheResidencyBackendMsg,
    CacheResidencyResultMsg,
    CacheStepBackendMsg,
    CacheStepResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    RoutingStatsBackendMsg,
    RoutingStatsResultMsg,
    UserMsg,
)
from freetoken.utils import (
    div_ceil,
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig, pin_kv_park_model_path
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        config = pin_kv_park_model_path(config)
        self.engine = Engine(config)
        self._routing_flush_at = time.monotonic()

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        park_store = None
        if config.kv_park != "off":
            from freetoken.kvcache.park_store import ParkStore

            park_store = ParkStore.from_config(
                config, self.engine.kv_cache, self.engine.linear_state_pool
            )
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
            park_store=park_store,
            park_consensus=(
                self._park_consensus
                if park_store is not None and config.tp_info.size > 1
                else None
            ),
        )
        if self.engine.mtp_shadow_observer is not None:
            self.engine.mtp_shadow_observer.bind_cache_manager(self.cache_manager)
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self._last_park_status = None
        self._idle_wait_logged = False
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
            debug_log=logger.debug_rank0,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def _park_consensus(self, payload: bytes) -> bool:
        """Return true only when every TP rank supplied the same parking decision."""
        if len(payload) > 63:
            raise ValueError("KV parking consensus payload exceeds 63 bytes")
        local = torch.zeros(64, dtype=torch.uint8)
        local[0] = len(payload)
        if payload:
            local[1 : 1 + len(payload)] = torch.tensor(tuple(payload), dtype=torch.uint8)
        primary = local.clone()
        self.tp_cpu_group.broadcast(primary, root=0).wait()
        agreed = torch.tensor([int(torch.equal(local, primary))], dtype=torch.int32)
        torch.distributed.all_reduce(
            agreed,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_cpu_group,
        )
        return bool(agreed.item())

    def _send_park_status(self) -> None:
        if self.cache_manager.park_store is None:
            return
        status = self.cache_manager.park_status()
        if status != self._last_park_status:
            self.send_result([CacheParkStatusMsg(status=status)])
            self._last_park_status = status

    def idle_poll_timeout_ms(self) -> int | None:
        return self.cache_manager.next_park_delay_ms()

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        if not self._idle_wait_logged:
            logger.info_rank0("Scheduler is idle, waiting for new reqs...")
            self._idle_wait_logged = True
        self.cache_manager.drain_pending_parks()
        self.cache_manager.park_idle()
        self.cache_manager.drain_pending_parks()
        self.cache_manager.check_integrity()
        self._send_park_status()

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
        layer_moves: list[tuple[int, str]] | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        is_moe_only = (
            num_pages is None and num_mamba_slots is None and num_swa_pages is None
        )
        assert not self._prefill_has_chunked_continuation(), "rebuild requires no in-flight prefill"
        if not is_moe_only:
            assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        # The old pools are the only source for parking. Snapshot eligible prefixes before the
        # engine reallocates them, then rebuild the radix tree against the new page table below.
        if (
            getattr(self.cache_manager, "park_store", None) is not None
            and (num_pages is not None or num_mamba_slots is not None)
        ):
            self.cache_manager.prepare_rebuild()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size,
            num_pages=num_pages,
            num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
            layer_moves=layer_moves,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            # Per-message so an idle poll never counts as traffic (freetoken/diag.py arms on
            # the first opened range). Online this covers validation, the vision tower and the
            # queue insert -- the text tokenizer itself runs in the tokenizer PROCESS and is
            # invisible here; offline it also covers ``LLM._tokenize_one``.
            with diag.region("diag.prefill_tokenize"):
                self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        # A MoE-only rebuild (slot cache and/or layer residency; no KV, GDN or window change)
        # may run at a decode step boundary with requests in flight: their state lives in the
        # KV pages and the GDN state pool, both untouched, and last_data is None means the
        # previous batch is drained (rebuild_cache still host-syncs). Never mid prefill chunk.
        if self._pending_rebuild is not None and last_data is None and self._rebuild_can_run():
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)

        if self._spec_dispatch_ready():
            # Design 6.2: a speculative cycle is serial (draft -> verify -> accept), so it
            # runs SYNCHRONOUSLY -- drain the previous batch first, in this iteration, which
            # is what makes the host ids caught up and removes the input_ids-lags-device_len
            # skew entirely. Prefill and the non-speculative fallback keep the overlap below.
            self._process_last_data(last_data)
            self._flush_abort_acks()
            self._last_data = last_data = None
            spec_req = self._spec_candidate()
            if spec_req is not None:
                with self.engine_stream_ctx:
                    self.engine.stream.wait_stream(self.stream)
                    self._speculative_decode_step(spec_req)
                self.stream.wait_stream(self.engine.stream)
                return None

        if (
            self._pending_rebuild is not None
            and self._is_moe_only_rebuild(self._pending_rebuild)
            and not self.prefill_manager.runnable
        ):
            # Drain toward the MoE-only rebuild: launch nothing this iteration so the next one
            # starts with last_data None and executes it above, then decode resumes. Without
            # this the overlap loop always has a batch in flight while a request decodes and
            # the between-step path is unreachable (the step waits for the request to end).
            forward_input = None
        else:
            forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                with diag.region(
                    "diag.prefill_restore_state"
                    if forward_input.batch.is_prefill
                    else None
                ):
                    self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            # Per-message so an idle poll never counts as traffic (freetoken/diag.py arms on
            # the first opened range). Online this covers validation, the vision tower and the
            # queue insert -- the text tokenizer itself runs in the tokenizer PROCESS and is
            # invisible here; offline it also covers ``LLM._tokenize_one``.
            with diag.region("diag.prefill_tokenize"):
                self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        # A MoE-only rebuild may execute between decode steps (see overlap_loop).
        if self._pending_rebuild is not None and self._rebuild_can_run():
            self._execute_pending_rebuild()

        # Non-overlap mode already drains what it launches, so the speculative step needs no
        # early drain here -- only the same dispatch.
        if self._spec_dispatch_ready():
            spec_req = self._spec_candidate()
            if spec_req is not None:
                self._speculative_decode_step(spec_req)
                self._flush_abort_acks()
                return

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        # ``diag.profile_step`` is the kernel profiler's iteration boundary (default-off; see
        # freetoken/diag.py). It sits HERE and not inside the two loops because this is the one
        # point both of them pass through exactly once per iteration -- and because boot-time
        # graph capture is finished by construction: it ran in the Engine constructor, long
        # before run_forever, so the recorder can never catch a capture.
        diag.profile_loop_begin()
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
                    diag.profile_step()
                    self._maybe_flush_routing_stats()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)
                diag.profile_step()
                self._maybe_flush_routing_stats()

    def _maybe_flush_routing_stats(self) -> None:
        """Once per FLUSH_INTERVAL_S, merge the routing histogram into the stats file. A
        monotonic-clock compare per iteration; the flush itself syncs the device once."""
        if self.engine.routing_recorder is None:
            return
        now = time.monotonic()
        if now - self._routing_flush_at < FLUSH_INTERVAL_S:
            return
        self._routing_flush_at = now
        self.engine.flush_routing_stats()

    def shutdown(self) -> None:
        self.cache_manager.close()
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        # The step is host-settled exactly here, on a sync this path already owned.
        self._spec_record_plain(batch)
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                # One row per request, whatever the step's width (a scalar row for decode).
                with diag.region("diag.prefill_emit" if batch.is_prefill else None):
                    msg = self._emit_step_tokens(req, next_tokens_cpu[i].reshape(-1))
                finished = msg.finished
                reply.append(msg)

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    with diag.region("diag.prefill_cache_commit"):
                        self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        with diag.region("diag.prefill_emit" if batch.is_prefill else None):
            self._ship_replies(
                batch,
                reply,
                # One token per scheduled request, as ever, plus whatever a wider step added.
                generated_tokens=len(batch.reqs)
                + sum(len(m.next_tokens) - 1 for m in reply),
            )

    def _ship_replies(
        self, batch: Batch, reply: List[DetokenizeMsg], *, generated_tokens: int
    ) -> None:
        """Stamp, report and send one step's replies. Shared by the drain and the
        speculative step, which produces the same one-message-per-uid shape."""
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
            generated_tokens=generated_tokens,
        )
        self.send_result(reply)
        send_park_status = getattr(self, "_send_park_status", None)
        if send_park_status is not None:
            send_park_status()

    def _emit_step_tokens(
        self, req: Req, tokens: torch.Tensor, *, settled: bool = False
    ) -> DetokenizeMsg:
        """Append this step's sampled tokens to the request and build its single reply.

        The run is processed in order and the first stop condition truncates the rest --
        the remaining tokens are neither appended nor shipped -- so the emitted ids and the
        finish reason equal what the same tokens emitted one per step would have produced.
        The output budget is a property of the step (the device already advanced past it),
        so it terminates the run's last surviving token; EOS and stop strings win over it.

        ``settled`` says the device length does NOT describe this run. A speculative step
        leaves ``device_len`` at the last DRAFT row -- past every rejected token -- so
        ``can_decode`` would answer about a length the request will never keep, and the run's
        own exhaustion of the host budget is the only honest reading. Plain decode leaves it
        False and keeps reading ``can_decode``, which under overlap counts the token already
        in flight; that is the behaviour every existing client sees, and it is untouched.
        """
        budget = req.max_device_len - req.input_ids.numel()
        n = min(tokens.numel(), budget)
        assert n >= 1
        hit_length = n == budget if settled else not req.can_decode
        emitted: List[int] = []
        hit_eos = False
        matched_stop: str | None = None
        for i in range(n):
            req.append_host(tokens[i : i + 1])
            token = int(tokens[i].item())
            emitted.append(token)
            hit_eos = not req.sampling_params.ignore_eos and token in self.eos_token_ids
            matched_stop = (
                self._match_stop_str(req)
                if not hit_eos and req.sampling_params.stop_strs
                else None
            )
            stopped = hit_eos or matched_stop is not None
            terminal = stopped or (i == n - 1 and hit_length)
            if (
                token == self.toolcall_anchor_id
                and req.toolcall_anchor_len is None
                and not terminal
            ):
                req.toolcall_anchor_len = req.input_ids.numel()
            if stopped:
                break
        # EOS / stop-string -> "stop", output budget exhausted -> "length";
        # EOS and stop strings win over length.
        finished = hit_length or hit_eos or matched_stop is not None
        finish_reason = (
            ("stop" if (hit_eos or matched_stop is not None) else "length")
            if finished
            else None
        )
        return DetokenizeMsg(
            uid=req.uid,
            next_tokens=tuple(emitted),
            finished=finished,
            finish_reason=finish_reason,
            matched_stop=matched_stop,
            stop_strs=req.sampling_params.stop_strs or None,
        )

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    @torch.inference_mode()
    def _prepare_multimodal_request(self, msg: UserMsg) -> None:
        """Encode one tokenizer-prepared still picture and derive Qwen rotary positions."""
        pixels = msg.mm_pixel_values
        grid = msg.mm_image_grid_thw
        token_types = msg.mm_token_type_ids
        try:
            if pixels is None or grid is None or token_types is None:
                raise ValueError(
                    "Qwen picture input needs pixel_values, image_grid_thw, and "
                    "mm_token_type_ids"
                )
            model = self.engine.model
            if not hasattr(model, "encode_images"):
                raise ValueError(f"{type(model).__name__} does not support picture inputs")
            vision_config = self.config.model_config.vision_config
            if vision_config is None:
                raise ValueError("model picture configuration is not loaded")
            # Earliest in-process signal that a picture is coming, and the encode weights
            # may be a mapped, non-resident 856 MiB extent. The prefetch syscall returns
            # while the reads continue, so issuing it here overlaps the read with the
            # encode's own GPU work. Optional hook, and a failure only costs latency.
            prefetch = getattr(model, "prefetch_picture_weights", None)
            if prefetch is not None:
                try:
                    prefetch()
                except Exception as exc:  # noqa: BLE001 - never fail a request over a hint
                    logger.warning_rank0(
                        "Picture weight prefetch failed for request %d: %s", msg.uid, exc
                    )

            from freetoken.models.qwen4_exp.mrope import build_mrope_positions

            positions, delta = build_mrope_positions(
                msg.input_ids,
                token_types,
                grid,
                int(vision_config.spatial_merge_size),
            )
            encode_started = time.perf_counter()
            try:
                # The model owns placement: layer-stream keeps these transport tensors on
                # CPU, while the backward-compatible GPU path moves them inside encode_images.
                features = model.encode_images(pixels, grid)
                if features.device.type == "cuda":
                    torch.cuda.synchronize(features.device)
            finally:
                logger.info_rank0(
                    "Picture encoder request %d: %.3f seconds",
                    msg.uid,
                    time.perf_counter() - encode_started,
                )
            image_token_id = self.config.model_config.image_token_id
            if image_token_id is None:
                raise ValueError("model configuration has no picture token id")
            placeholders = int((msg.input_ids == image_token_id).sum().item())
            if features.ndim != 2 or features.shape[0] != placeholders:
                raise ValueError(
                    f"picture-token slots ({placeholders}) do not match picture features "
                    f"({features.shape[0] if features.ndim else 0})"
                )
            msg.mm_embeds = features
            msg.mrope_position_ids = positions
            msg.mrope_position_delta = delta
        finally:
            # These CPU tensors dominate request transport memory and are never needed after
            # the picture reader has produced soft tokens (including on a rejected request).
            msg.mm_pixel_values = None
            msg.mm_image_grid_thw = None
            msg.mm_token_type_ids = None

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        self._idle_wait_logged = False
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            has_raw_picture = any(
                value is not None
                for value in (
                    msg.mm_pixel_values,
                    msg.mm_image_grid_thw,
                    msg.mm_token_type_ids,
                )
            )
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                if has_raw_picture:
                    msg.mm_pixel_values = None
                    msg.mm_image_grid_thw = None
                    msg.mm_token_type_ids = None
                return
            if has_raw_picture:
                try:
                    self._prepare_multimodal_request(msg)
                except Exception as exc:  # noqa: BLE001 - reject one request, keep serving
                    logger.warning_rank0(
                        "Picture processing failed for request %d: %s", msg.uid, exc
                    )
                    self.send_result(
                        [ErrorReplyMsg(uid=msg.uid, error=f"could not encode picture: {exc}")]
                    )
                    return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            is_moe_only = (
                msg.num_pages is None and msg.num_mamba_slots is None and msg.num_swa_pages is None
            )
            is_busy = (
                False
                if is_moe_only
                else (self._prefill_has_chunked_continuation() or self.decode_manager.runnable)
            )
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif is_busy:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        elif isinstance(msg, CacheStepBackendMsg):
            # Never run a step inline here: in overlap mode the batch launched last iteration
            # may still be executing on the GPU (self._last_data), and step_memory rebuilds
            # pools and graphs. Queue it for the safe point like a rebuild.
            if self.config.tp_info.size > 1:
                self._reply_step(msg.request_id, "unsupported", error="step unsupported under TP > 1")
            elif not self.cache_manager.supports_runtime_rebuild:
                self._reply_step(msg.request_id, "unsupported", error="this model's cache does not support runtime rebuild")
            elif self._pending_rebuild is not None:
                self._reply_step(msg.request_id, "busy", error="another rebuild or step is queued")
            else:
                # Authoritative no-op preflight: residency alone can prove a step has nothing
                # to do (every layer already pinned for "ram up"), and that answer needs no
                # safe point, no CUDA synchronize and no rebuild. The live 5090 boot of
                # 2026-09-09 answered 847 of 861 governor steps this way, each one queued to
                # the safe point and paid for with a device sync while the API gate was shut.
                noop = self._step_noop_reply(msg)
                if noop is not None:
                    self._reply_step(msg.request_id, "ok", noop)
                else:
                    self._pending_rebuild = msg
        elif isinstance(msg, CacheResidencyBackendMsg):
            try:
                rep = self.engine.residency_report()
                self._reply_residency(msg.request_id, "ok", rep)
            except Exception as e:
                logger.error(f"residency_report failed: {e!r}")
                self._reply_residency(msg.request_id, "failed", error=str(e))
        elif isinstance(msg, RoutingStatsBackendMsg):
            # A read of counters the decode path already maintains: answer inline, whether or
            # not the scheduler is idle. Only rank 0 owns the reply link.
            self._reply_routing_stats(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _step_noop_reply(self, msg: CacheStepBackendMsg) -> dict | None:
        """The engine's residency-only verdict that ``msg`` cannot change anything, or None."""
        preflight = getattr(self.engine, "step_memory_noop", None)
        if preflight is None:
            return None
        try:
            return preflight(msg.axis, msg.direction)
        except Exception as e:  # noqa: BLE001 - a preflight error is not a reason to skip the step
            logger.warning(f"cache step preflight failed, queueing the step instead: {e!r}")
            return None

    def _reply_step(
        self,
        request_id: str,
        status: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        res = result or {}
        report = self.engine.residency_report() if status == "ok" else {}
        self.send_result(
            [
                CacheStepResultMsg(
                    request_id=request_id,
                    status=status,
                    applied=res.get("applied"),
                    layer=res.get("layer"),
                    at_floor=bool(res.get("at_floor", False)),
                    moe_cache_size=int(res.get("moe_cache_size", 0) or 0),
                    layers=({**{k: report.get(k, 0) for k in ("owned", "pinned", "disk")},
                             "parked": len(report.get("ram_parked") or [])} if report else None),
                    vram_free_bytes=int(res.get("vram_free_bytes", 0) or 0),
                    error=error or res.get("reason"),
                    exhausted=bool(res.get("exhausted", False)),
                )
            ]
        )

    def _reply_residency(
        self,
        request_id: str,
        status: str,
        report: dict | None = None,
        error: str | None = None,
    ) -> None:
        self.send_result(
            [
                CacheResidencyResultMsg(
                    request_id=request_id,
                    status=status,
                    residency=report or {},
                    error=error,
                )
            ]
        )

    @staticmethod
    def _routing_per_layer_rows(cache) -> list:
        """Per-layer decode rows; without collect_stats the step-derived columns are null.

        A streaming layer reported as ``steps: 0, miss_rate: 0.0`` would read as perfectly
        cacheable (the sentinel offload_cache's docstring forbids), so the columns follow the
        owned-layer convention and go null when the counters were never armed (R2 N1)."""
        rows = cache.decode_miss_stats_per_layer()["per_layer"]
        if cache.collect_stats:
            return rows
        for row in rows:
            for key in ("active_per_step", "missing_per_step", "miss_rate", "fetched_per_step"):
                row[key] = None
        return rows

    def _reply_routing_stats(self, msg: RoutingStatsBackendMsg) -> None:
        """Answer GET /v1/cache/routing from the live offload cache's decode counters."""
        cache = getattr(self.engine, "moe_offload_cache", None)
        stats: dict = {}
        error: str | None = None
        if cache is None:
            error = "this model has no MoE offload cache"
        elif not (
            getattr(cache, "collect_stats", False) or getattr(cache, "collect_decode_freq", False)
        ):
            # Routing learning arms the histogram without the miss counters; the route still
            # serves it (the per_layer miss columns then read as null, see below).
            error = (
                "decode counters are off; boot with --moe-collect-decode-freq "
                "(or FREETOKEN_MOE_COLLECT_DECODE_FREQ=1)"
            )
        else:
            try:
                stats = {
                    "collect_decode_freq": bool(cache.collect_decode_freq),
                    "num_layers": cache.num_layers,
                    "num_experts": cache.num_experts,
                    "cache_size": cache.cache_size,
                    # MoE layers served from resident VRAM banks: their per_layer rows carry
                    # resident: true / miss_rate: null, and they are excluded from the
                    # streaming-cache summary.
                    "gpu_owned_layers": sorted(getattr(cache, "gpu_owned_layer_ids", ()) or ()),
                    "summary": cache.decode_routing_stats(),
                    "per_layer": Scheduler._routing_per_layer_rows(cache),
                    # False when only routing learning armed the histogram: the miss columns
                    # above are then null, not zero.
                    "counters": bool(cache.collect_stats),
                    # raw [layers, experts] histogram: the input every offline skew study
                    # (static hot set sizing, overlap across workloads) actually needs.
                    "decode_freq": (
                        cache.decode_freq.tolist() if cache.collect_decode_freq else []
                    ),
                }
                if msg.reset:
                    # Window the next workload: zero the same counters decode_routing_stats
                    # and decode_miss_stats_per_layer read.
                    cache.reset_stats()
            except Exception as e:  # noqa: BLE001
                error = f"failed to read routing stats: {e!r}"
        self.send_result(
            [RoutingStatsResultMsg(request_id=msg.request_id, stats=stats, error=error)]
        )

    @staticmethod
    def _is_moe_only_rebuild(msg) -> bool:
        """True when the queued work touches only the MoE slot cache / layer residency.

        A governor step counts as MoE-only: its KV rung is idle-only by construction
        (step_memory gets is_idle at execution time), so queuing it never needs an idle wait.
        """
        if isinstance(msg, CacheStepBackendMsg):
            return True
        if not isinstance(msg, CacheRebuildBackendMsg):
            return False  # unknown work waits for idle, as every rebuild did before J1
        return msg.num_pages is None and msg.num_mamba_slots is None and msg.num_swa_pages is None

    def _prefill_has_chunked_continuation(self) -> bool:
        pending = getattr(self.prefill_manager, "pending_list", None)
        if pending is None:
            # Keep lightweight scheduler shells used by maintenance tests on the old boolean seam.
            return bool(getattr(self.prefill_manager, "runnable", False))
        return any(getattr(req, "chunked_req", None) is not None for req in pending)

    def _rebuild_can_run(self) -> bool:
        # A never-admitted PendingReq owns no pages or GDN slots, so it is harmless. Only a
        # chunked continuation is in-flight on the prefill side; the 190k live run deadlocked
        # because this gate treated every pending request as a live chunk.
        if self._prefill_has_chunked_continuation():
            return False
        if self._is_moe_only_rebuild(self._pending_rebuild):
            return True
        return not self.decode_manager.runnable

    def _execute_pending_step(self, msg: CacheStepBackendMsg) -> None:
        """Run a governor step at the safe point through rebuild_cache (not the engine directly)."""
        from freetoken.engine.engine import CacheRebuildRejected

        is_idle = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        self.engine.rebuild_teardown_started = False
        try:
            res = self.engine.step_memory(
                axis=msg.axis,
                direction=msg.direction,
                ram_tight=msg.ram_tight,
                is_idle=is_idle,
                rebuild=self.rebuild_cache,
            )
        except CacheRebuildRejected as e:
            logger.warning(f"cache step rejected: {e}")
            self._reply_step(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                logger.error(f"cache step failed before teardown: {e!r} — old cache intact")
                self._reply_step(msg.request_id, "rejected", error=repr(e))
                return
            # No geometry rollback for a step (which pool it touched is inside step_memory);
            # the frontend latches failed on this status. Follow-up: roll a failed rung back.
            logger.error(f"cache step failed after teardown: {e!r} — latching failed")
            self._reply_step(msg.request_id, "failed", error=repr(e))
            return
        if res.get("applied"):
            self._log_cache_geometry(f"Cache stepped ({res['applied']})")
        self._reply_step(msg.request_id, "ok", res)

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        if isinstance(msg, CacheStepBackendMsg):
            self._execute_pending_step(msg)
            return
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
            "layer_moves": getattr(msg, "layer_moves", None),
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None and k in snapshot}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        batch.rope_positions = _make_rope_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _prepare_spec_batch(self, req: Req, draft_tokens: Sequence[int]) -> ForwardInput:
        """Build one request's ``w = 1 + len(draft_tokens)`` row speculative verify batch.

        The decode-batch builder above, generalized from one row to ``w``: row 0 is the last
        accepted token (already in the token pool at ``cached_len``) and rows 1..k are the
        drafts, staged into the token pool at the positions they would occupy if every one
        were accepted. Positions, page allocation and ``out_loc`` then fall out of the same
        ``extend_len``-driven helpers plain decode uses.

        Unlike ``mtp_shadow._target_verify_batch`` this redirects NOTHING -- the request's own
        ``table_idx``, its own page-table row, its own pages and its own GDN slot. The dummy
        table / page lease / shadow slot live outside the ``mtp_verify`` flag, and dropping
        them is exactly what makes the integrated forward cheap.

        The step's length advance is NOT ``complete_many``: ``_rollback_spec_tokens`` is the
        single authority that settles ``cached_len`` / ``device_len`` once the accepted run is
        known, because a rejected row must leave no trace.
        """
        spec = self.config.spec_decode
        if not spec.enabled:
            raise RuntimeError(
                "speculative decode is off; set FREETOKEN_MTP_SPECULATE=1 to enable it"
            )
        k = len(draft_tokens)
        if not 1 <= k <= spec.depth:
            raise ValueError(
                f"a speculative step needs 1..{spec.depth} draft tokens, got {k}"
            )
        if req.spec_inflight is not None:
            raise RuntimeError(
                f"request {req.uid} already has a speculative step in flight"
            )
        if req.extend_len != 1:
            raise RuntimeError(
                f"speculation needs a request in decode shape (extend_len 1), got "
                f"{req.extend_len}"
            )
        if req.input_ids.numel() != req.device_len:
            # Spec steps drain synchronously (design 6.2), so the host ids are caught up. Under
            # overlap they lag by the in-flight tokens and the drafts would be staged over a
            # position whose accepted token has not been appended yet.
            raise RuntimeError(
                f"request {req.uid} host ids ({req.input_ids.numel()}) lag device_len "
                f"({req.device_len}); a speculative step must drain synchronously"
            )
        w = 1 + k
        if req.remain_len < w - 1:
            # device_len must stay <= max_device_len; the last row's sampled token may still
            # fall past the budget and take the write mapping's -1 discard slot.
            raise RuntimeError(
                f"request {req.uid} has {req.remain_len} tokens of output budget left, "
                f"short of the {w}-row speculative step"
            )

        page_size = self.cache_manager.page_size
        base = req.cached_len                     # row i reads position base + i
        first_page = div_ceil(base, page_size)
        device_len_before = req.device_len
        req.device_len = base + w                 # extend_len = w drives every helper below
        last_page = div_ceil(req.device_len, page_size)
        page_row = self.engine.page_table[
            req.table_idx, first_page * page_size : last_page * page_size
        ].clone()
        self.cache_manager.allocate_paged([req])
        req.spec_inflight = SpecInflight(
            width=w,
            cached_len=base,
            device_len=device_len_before,
            first_page=first_page,
            last_page=last_page,
            # Read back from the row rather than from the free list: this is what the step
            # actually got, eviction included.
            pages=self.engine.page_table[
                req.table_idx, first_page * page_size : last_page * page_size : page_size
            ].clone(),
            page_row=page_row,
        )
        self.token_pool[req.table_idx, base + 1 : base + w] = torch.tensor(
            list(draft_tokens), dtype=self.token_pool.dtype, device=self.token_pool.device
        )

        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        # Causal prefill semantics over w rows, with the offloaded MoE still reading routed
        # experts through its decode cache (layers/moe.py _use_decode_movement).
        batch.mtp_verify = True
        batch.emit_width = w
        batch.positions = _make_positions(batch, self.device)
        batch.rope_positions = _make_rope_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_spec_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            pool = self.engine.linear_state_pool
            slot = req.linear_slot_idx if req.linear_slot_idx is not None else pool.padding_slot
            batch.linear_table_idx = torch.tensor(
                [slot], dtype=torch.int32, device=self.device
            )
            # No track checkpoint can ride a speculative step, whatever cached_len is: the
            # hybrid-radix snapshot is scheduled per FORWARD, at c = (extend_len - 1) // 64
            # chunks into it (attention/linear.py:123-127), so a w <= 6 row batch gives c == 0
            # and is skipped -- absolute x64 alignment never enters it. Were one scheduled it
            # would freeze rejected rows into a donatable prefix-cache slot.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _rollback_spec_tokens(
        self,
        req: Req,
        accepted: int,
        *,
        state_rollback: "Callable[[Req, int], None] | None" = None,
    ) -> None:
        """Settle a speculative step, keeping ``accepted`` of its ``width`` forwarded rows.

        Row ``i`` samples the token for position ``cached_len + i + 1``, so keeping ``accepted``
        rows emits exactly ``accepted`` tokens -- ``j`` accepted drafts plus the bonus token
        from row ``j``. Production therefore always keeps at least one row; ``accepted == 0``
        is the complete undo, which is what makes the state-digest gate meaningful.

        Leaves the request exactly where ``accepted`` plain decode steps would have: lengths
        back in decode shape, the over-allocated pages returned to the head of the free list in
        allocation order (so a full rollback is byte-identical to never having run the step,
        which ``_free``'s tail-append could not give), and their page-table cells restored.

        Rejected QSA rows need no KV rewind: the K/V write is position-addressed and the
        compressed-slab scorer clamps visible blocks to ``sequence_length // index_ratio``, so
        the length rewind alone makes them unreachable.

        Scope is KV / page / length bookkeeping. The GDN conv + recurrent state (and the PLE
        conv / n-gram context riding its slot) still hold all ``width`` rows: rolling those
        back is Phase 3 (Strategy R -- restore the pre-step snapshot and replay the accepted
        prefix through the recurrent decode kernel). ``state_rollback`` is that seam; it is
        called with ``(req, accepted)`` only after the lengths are final, because the accepted
        run can still shrink under a stop condition.
        """
        spec = req.spec_inflight
        if spec is None:
            raise RuntimeError(f"request {req.uid} has no speculative step to settle")
        if not 0 <= accepted <= spec.width:
            raise ValueError(
                f"accepted must be 0..{spec.width} rows, got {accepted}"
            )
        page_size = self.cache_manager.page_size
        req.cached_len = spec.cached_len + accepted
        req.device_len = req.cached_len + 1
        keep = max(0, min(div_ceil(req.cached_len, page_size), spec.last_page) - spec.first_page)
        returned = spec.pages[keep:]
        if returned.numel():
            # NOT per-cycle churn, despite the fresh free-list tensor: ``spec.pages`` spans
            # only [first_page, last_page), and a w <= 6 row step straddles a page boundary
            # rarely (never, at page_size 64, unless cached_len lands within w of one) and
            # LOSES that page rarer still -- it must also have rejected back across it. The
            # free list is a suffix VIEW consumed from the front and appended at the back, and
            # ``_allocate`` hands its callers views of the very prefix an in-place prepend
            # would overwrite, so making this in-place is a free-list refactor, not a patch.
            self.cache_manager.free_slots = torch.cat(
                [returned, self.cache_manager.free_slots]
            )
            start = (spec.first_page + keep) * page_size
            self.engine.page_table[req.table_idx, start : spec.last_page * page_size] = (
                spec.page_row[keep * page_size :]
            )
        req.spec_inflight = None
        if state_rollback is not None:
            state_rollback(req, accepted)

    # -------------------------------------------------------------- integrated speculation

    def _spec_timing_probe(self) -> "_SpecTimingProbe | None":
        """Per-stage cycle timing behind FREETOKEN_MTP_SPEC_TIMING=1. The probe device-syncs
        at every mark, so it distorts absolute rates -- diagnosis only, never benchmarks."""
        probe = getattr(self, "_spec_probe", False)
        if probe is False:
            import os

            probe = (
                _SpecTimingProbe(self.device)
                if os.getenv("FREETOKEN_MTP_SPEC_TIMING") == "1"
                else None
            )
            self._spec_probe = probe
        if probe is not None:
            probe.start_cycle()
        return probe

    def _spec_conf_log(self) -> "_SpecConfLog | None":
        """The per-cycle (confidence, acceptance) log behind FREETOKEN_MTP_SPEC_CONF_LOG=<dir>.

        Resolved once per process, the same lazy shape as ``_spec_timing_probe``. Unset it is a
        single ``getattr`` per cycle and no device work whatsoever; set, it costs a readback of
        the draft's own confidences and a buffered line of JSON -- a diagnosis flag, never a
        serving one.
        """
        log = getattr(self, "_spec_conf", False)
        if log is False:
            import os

            directory = (os.getenv("FREETOKEN_MTP_SPEC_CONF_LOG", "") or "").strip()
            log = _SpecConfLog(directory) if directory else None
            self._spec_conf = log
        return log

    def _spec_policy(self, req: Req) -> "_SpecAcceptance | None":
        """This request's acceptance policy, or None while the fallback is disabled.

        Fresh per REQUEST: content changes at request boundaries, and a served request's own
        early cycles are the only honest predictor of its later ones. With
        ``FREETOKEN_MTP_SPEC_MIN_EMITTED=0`` nothing is constructed at all, so the dispatch
        path is byte for byte the pre-adaptive one.
        """
        spec = self.config.spec_decode
        if not spec.adaptive:
            return None
        states = getattr(self, "_spec_policies", None)
        if states is None:
            states = self._spec_policies = {}
        policy = states.get(req.uid)
        if policy is None:
            live = {r.uid for r in self.decode_manager.running_reqs}
            for uid in [u for u in states if u not in live]:
                del states[uid]
            policy = states[req.uid] = _SpecAcceptance(spec)
        return policy

    def _spec_record(self, req: Req, emitted: int) -> None:
        """Feed one cycle's emitted-token count back into the request's policy."""
        policy = self._spec_policy(req)
        if policy is not None:
            policy.record(emitted)

    def _spec_live_policy(self, uid: int) -> "_SpecAcceptance | None":
        """This uid's policy IF it already has one -- never constructing, never pruning.

        The timing hooks run at points where the request may already have finished (the drain
        frees it; the cycle's tail can too), and ``_spec_policy`` would both resurrect a state
        for a dead uid and evict live ones against a ``running_reqs`` that no longer holds it.
        """
        if not self.config.spec_decode.cost_aware:
            return None
        states = getattr(self, "_spec_policies", None)
        return None if not states else states.get(uid)

    def _spec_record_plain(self, batch: Batch) -> None:
        """Time one PLAIN decode step for the cost-aware bar, at the drain that settles it.

        Under overlap a decode step spans a whole loop iteration -- batch N is launched in the
        iteration that drains batch N-1 -- so the step's wall time is the INTERVAL between
        consecutive drains, not the duration of one. Hence a mark rather than a span, dropped
        (``_spec_drop_plain_mark``) whenever anything that is not a plain decode step lands
        between two drains: a prefill, a wider batch, or a speculative cycle. An interval that
        straddled one of those would price a plain step at something no plain step costs.

        Sync-free by placement: ``_process_last_data`` has already run ``copy_done``'s wait by
        the time this is called, so the timestamp is taken on a host that is by construction
        caught up with the batch it is timing -- two ``perf_counter`` calls per step and not
        one byte of device work added.
        """
        spec = self.config.spec_decode
        if not (spec.enabled and spec.adaptive and spec.cost_aware):
            return
        now = time.perf_counter()
        mark = getattr(self, "_plain_step_mark", None)
        self._plain_step_mark = now
        if not (batch.is_decode and len(batch.reqs) == 1):
            self._plain_step_mark = None
            return
        policy = self._spec_live_policy(batch.reqs[0].uid)
        if policy is not None and mark is not None:
            policy.record_plain_ms(1e3 * (now - mark))

    def _spec_drop_plain_mark(self) -> None:
        """Forget the last drain's timestamp: the next interval would not be a plain step."""
        self._plain_step_mark = None

    def _spec_dispatch_ready(self) -> bool:
        """Whether this iteration should try a speculative step instead of a plain decode.

        Deliberately request-independent: what it gates is draining the previous batch EARLY,
        and the per-request predicate can only be evaluated after that drain has caught the
        host ids up. Prefill wins, exactly as ``_schedule_next_batch`` has always ordered it.
        """
        return (
            self.config.spec_decode.enabled
            and self.engine.spec_draft is not None
            and not self.prefill_manager.runnable
            and self.decode_manager.runnable
        )

    def _spec_candidate(self) -> Req | None:
        """The running request a speculative step can serve, or None to decode plainly.

        The predicate is the REQUEST's shape, never a batch's phase: the speculative batch is
        itself prefill-phase (that is what gives the ``w`` rows causal semantics and the MoE
        its decode-movement path), so ``batch.is_decode`` would be false for every step.

        Every miss below falls back to an ordinary one-row decode for this step, never to an
        error: an unprimed head (a request admitted before speculation, or one still walking
        through chunked prefill) and a request with no room for a multi-token run are normal.
        """
        running = self.decode_manager.running_reqs
        if len(running) != 1:
            return None
        req = next(iter(running))
        if req.aborted or req.table_idx == -1 or req in self.finished_reqs:
            return None
        if req.extend_len != 1:
            return None
        if req.input_ids.numel() != req.device_len:
            # the previous batch has not drained; a draft would be staged over a position
            # whose accepted token is not on the host yet
            return None
        if req.remain_len < 2:
            return None
        if not self.engine.spec_draft.is_ready(req):
            return None
        policy = self._spec_policy(req)
        if policy is not None and not policy.should_speculate():
            # A cold draft loses to plain decode; the cooldown is counted in the steps this
            # request actually decodes plainly, which is why the policy is consulted LAST --
            # a structural miss above is not a step the policy chose to spend.
            return None
        return req

    def _speculative_decode_step(self, req: Req) -> None:
        """One draft -> verify -> accept -> emit -> settle cycle, start to finish.

        Synchronous by construction (design 6.2): the cycle is a serial chain -- the draft for
        the next cycle consumes the target hidden state of the row this one just accepted, and
        acceptance already forces a device sync -- so nothing is left in flight. That is what
        keeps ``_prepare_spec_batch``'s host-ids guard true on the next iteration.

        The order of the tail is the whole of design 6.4: emit (which truncates the run at the
        first stop condition), then shrink the verdict to what was emitted, then settle the
        lengths / pages / linear state to that, then free. Settling before the free is what
        keeps the finish path legal -- ``_free_req_resources`` commits the prefix, and the
        radix guard refuses a commit under in-flight speculative rows.
        """
        engine = self.engine
        # One pair of perf_counter calls around the whole cycle, and no sync of its own: the
        # cycle is host-synced start to end by construction (above), so the closing timestamp
        # already describes finished device work. The probe's per-stage marks DO sync, which
        # is why this span is taken here rather than read back off the probe.
        started = time.perf_counter()
        probe = self._spec_timing_probe()
        conf_log = self._spec_conf_log()
        # read before the cycle settles it: what a confidence cut has to be judged against is
        # the context length the drafts were made AT
        cached_len = int(req.cached_len) if conf_log is not None else 0
        depth = min(self.config.spec_decode.depth, req.remain_len)
        with diag.region("diag.spec_draft"):
            proposal = engine.spec_draft.propose(req, depth, probe=probe)
        probe and probe.mark("draft")

        forward_input = self._prepare_spec_batch(req, proposal.tokens)
        batch, sample_args, input_mapping, write_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        ladder = engine.spec_state_ladder
        if ladder is not None:
            ladder.begin(req, batch)
        probe and probe.mark("prepare")
        output = engine.speculative_decode_batch(
            batch,
            sample_args,
            draft_tokens=proposal.tokens,
            draft_logits=proposal.logits,
            probe=probe,
        )
        probe and probe.mark("verify+accept")

        with diag.region("diag.spec_emit"):
            msg = self._emit_step_tokens(
                req, torch.tensor(output.decision.tokens, dtype=torch.int32), settled=True
            )
            emitted = len(msg.next_tokens)
            decision = output.decision.truncated(emitted)
            # Only the emitted run reaches the token pool; a rejected row's sampled token would
            # become the next step's row 0.
            self.token_pool[write_mapping[0][:emitted], write_mapping[1][:emitted]] = (
                output.next_tokens_gpu[:emitted]
            )
        # The tail's three sub-stages, in the same dotted shape the engine subdivides
        # "verify+accept" with: each is timed against the previous sub-mark, so the coarse
        # stages they sit inside keep spanning exactly what they spanned before. They mirror
        # the diag regions above one for one, so a probe run and a diag trace name the same
        # spans. "tail.commit" is the draft head's forward -- measured here, owned elsewhere.
        probe and probe.mark("tail.emit")
        with diag.region("diag.spec_rollback"):
            self._rollback_spec_tokens(
                req,
                decision.accepted_rows,
                state_rollback=None if ladder is None else ladder.rollback,
            )
        probe and probe.mark("tail.rollback")
        probe and probe.mark("emit+rollback")

        if msg.finished:
            engine.spec_draft.reset_request(req.uid)
        else:
            # The draft's context is the shifted pairs (target hidden i, embedding of token
            # i+1); this cycle contributed exactly `emitted` of them.
            with diag.region("diag.spec_commit"):
                engine.spec_draft.commit(
                    req,
                    hidden=output.hidden,
                    token_ids=msg.next_tokens,
                    probe=probe,
                )
        probe and probe.mark("tail.commit")
        self._spec_record(req, emitted)
        probe and probe.finish_cycle(
            emitted=emitted,
            accepted=decision.accepted_rows,
            policy=self._spec_policy(req),
            # the untruncated verdict: the acceptance timings are the step's, not the run's
            decision=output.decision,
        )
        conf_log and conf_log.record(
            uid=req.uid,
            cached_len=cached_len,
            proposal=proposal,
            # untruncated for the same reason the probe takes it untruncated: what was accepted
            # is a property of the step, not of where a stop condition cut the run
            decision=output.decision,
            emitted=emitted,
        )
        self.decode_manager.filter_reqs([req])

        finished_now: Set[Req] = set()
        if msg.finished and req not in self.finished_reqs:
            with self.cache_manager.lazy_free_region():
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
            finished_now.add(req)
        self.finished_reqs = finished_now
        self._ship_replies(batch, [msg], generated_tokens=emitted)
        # After the ship, so the span is the whole of what a cycle displaces from the loop --
        # the same thing the plain-step interval measures. The next cycle's `record` is what
        # reads it; a cycle cannot be judged against its own not-yet-known cost.
        self._spec_drop_plain_mark()
        policy = self._spec_live_policy(req.uid)
        if policy is not None:
            policy.record_cycle_ms(1e3 * (time.perf_counter() - started))

    def _gather_multimodal(self, batch: Batch) -> None:
        """Gather only the picture feature rows used by this prefill step.

        ``PendingReq`` owns the complete feature tensor across continuation steps. Each
        scheduled ``Req`` temporarily receives that tensor, then releases its reference
        after this method copies the rows whose image placeholders fall in
        ``[cached_len, device_len)``. ``cache_private`` remains authoritative after the
        reference is cleared, so no picture KV can enter the shared prefix cache.
        """
        image_token_id = self.config.model_config.image_token_id
        parts = []
        for req in batch.reqs:
            features = req.mm_embeds
            if features is None:
                continue
            try:
                if image_token_id is None:
                    raise ValueError("picture features were supplied without a picture token id")
                ids = req.input_ids
                feature_start = int((ids[: req.cached_len] == image_token_id).sum().item())
                feature_stop = int((ids[: req.device_len] == image_token_id).sum().item())
                if feature_stop > features.shape[0]:
                    raise ValueError(
                        f"picture feature slice ends at {feature_stop}, but only "
                        f"{features.shape[0]} rows exist"
                    )
                if feature_stop > feature_start:
                    parts.append(features[feature_start:feature_stop])
            finally:
                req.mm_embeds = None
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        pop_rejections = getattr(self.prefill_manager, "pop_rejections", None)
        if pop_rejections is not None:
            rejections = pop_rejections()
            if rejections:
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=uid,
                            error=reason,
                            code="context_length_exceeded",
                        )
                        for uid, reason in rejections
                    ]
                )
        if batch is None:
            return None
        with diag.region("diag.prefill_batch" if batch.is_prefill else None):
            forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        # The whole non-speculative forward: a prefill batch, or a plain decode step
        # (graph replay including its PLE staging, or the width-1 spec-graph fallback).
        with diag.region(
            "diag.prefill_forward" if batch.is_prefill else "diag.plain_decode_step"
        ):
            forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_rope_positions(batch: Batch, device: torch.device) -> torch.Tensor | None:
    """Build packed three-axis rotary positions when a batch contains Qwen picture input."""
    if not any(req.mrope_position_ids is not None for req in batch.padded_reqs):
        return None
    needed_size = sum(req.extend_len for req in batch.padded_reqs)
    host = torch.empty(
        (3, needed_size),
        dtype=torch.int64,
        pin_memory=device.type == "cuda",
    )
    offset = 0
    for req in batch.padded_reqs:
        start, end = int(req.cached_len), int(req.device_len)
        length = end - start
        if not length:
            continue
        prompt_positions = req.mrope_position_ids
        prompt_len = 0 if prompt_positions is None else int(prompt_positions.shape[1])
        prompt_stop = min(end, prompt_len)
        copied = max(prompt_stop - start, 0)
        if copied:
            host[:, offset : offset + copied].copy_(prompt_positions[:, start:prompt_stop])
        generated_start = start + copied
        if generated_start < end:
            generated = torch.arange(generated_start, end, dtype=torch.int64).add_(
                int(req.mrope_position_delta)
            )
            host[:, offset + copied : offset + length].copy_(generated.expand(3, -1))
        offset += length
    return host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """Destination of each token this step samples, flattened request-major.

    ``emit_width`` slots per request (one for plain decode). A slot past the request's
    output budget writes to the token pool's -1 discard column, as it always has.
    """
    mapping_list = [req.table_idx for req in batch.reqs for _ in range(batch.emit_width)]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [
        (req.device_len + i if req.remain_len > i else -1)
        for req in batch.reqs
        for i in range(batch.emit_width)
    ]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)


class _SpecAcceptance:
    """One request's speculative acceptance, and the fallback that acts on it.

    A cycle emits ``1 + accepted`` tokens for a fixed cost of roughly ``min_emitted`` plain
    steps, so the EMA of emitted-per-cycle IS the profit signal: above the threshold the
    request is ahead, below it every cycle loses time. Cold is not permanent -- after a
    cooldown of plain steps one probe cycle re-measures, because content changes mid-stream.

    The probe is judged on its OWN emission, not on the EMA it inherited: a sluggish average
    built while the draft was useless would veto a probe that just emitted a full run. So a
    good probe replaces the EMA outright -- the pre-cold history describes content that has
    since changed.

    AND IT IS JUDGED TWICE, AGAINST ITS OWN THRESHOLD. ``min_emitted`` bounds a MEAN; a probe
    is one integer sample from the distribution that mean describes. Content whose cycles
    alternate a full run with a single token sits comfortably above the bar on average while
    every other sample lands below it, so a single-sample verdict re-cooled roughly half the
    time -- with doubling backoff behind it, that is how a request that should speculate
    throughout ended up decoding 16, then 32, then 64 steps plainly. Hence ``probe_resume``
    (a sample's bar, not a mean's) and a pair of consecutive probes before any re-cool.

    AND THE BAR ITSELF IS MEASURED, not fixed (``cost_aware``). "A cycle costs about
    ``min_emitted`` plain steps" is true only at the context it was tuned at: live, a cycle
    costs ~28-45 ms at short context and ~70-100 ms at 8-11k -- verify and draft both scale
    with the KV it reads -- while a plain step only goes ~15 -> ~20 ms. The breakeven
    therefore roughly DOUBLES over one long request, and a bar fixed at the short-context
    number keeps spending at the long end where cycles are dear (measured -11..-24% at 8k on
    depth 5). So the policy times both sides of that ratio on the request itself and holds
    the EMA to ``cycle_ms / plain_ms``: exactly the number of plain steps this cycle is
    costing, right now, at this context. ``min_emitted`` becomes its FLOOR and ``1 + depth``
    -- what a cycle could at most emit -- its ceiling.
    """

    #: consecutive failing probes before the request cools down again
    PROBE_CYCLES = 2

    #: wall-time samples of EACH kind before the measured bar is trusted over the static one
    MIN_TIMING_SAMPLES = 3

    def __init__(self, spec) -> None:  # SpecDecodeConfig
        self.ema = spec.ema_seed
        self.alpha = spec.ema_alpha
        self.min_emitted = spec.min_emitted
        self.probe_resume = spec.probe_resume
        self.base_cooldown = spec.cooldown
        self.cooldown_cap = spec.cooldown_cap
        self.next_cooldown = spec.cooldown
        self.remaining = 0        # plain steps still owed before the next probe
        self.probes_left = 0      # probe cycles still owed before a re-cool
        self.cycles = 0
        self.plain_steps = 0
        self.cost_aware = spec.cost_aware
        # The ceiling: no cycle can emit more than its own width, so a bar above it would mean
        # "never speculate" rather than "speculate only when it pays".
        self.max_bar = float(spec.batch_width)
        # Same alpha as the emission EMA on purpose: content and context shift TOGETHER
        # mid-request, so a bar that tracked slower than the emission it judges would be
        # comparing this paragraph's acceptance against the last one's cost.
        self.plain_ms = 0.0
        self.cycle_ms = 0.0
        self.plain_samples = 0
        self.cycle_samples = 0

    @property
    def bar(self) -> float:
        """The emitted-per-cycle this request must clear for a cycle to pay for itself.

        Falls back to the static ``min_emitted`` until BOTH sides of the ratio have real
        samples -- a request that speculated from its very first step has no plain step to
        divide by, and one whose first cycles are still warming caches would divide by a
        number it will never see again. The static value is the floor even when the measured
        ratio comes in under it: the ratio times only wall clock, while ``min_emitted``
        carries what a cycle costs the REST of the batch too.
        """
        if not self.cost_aware:
            return self.min_emitted
        warm = min(self.plain_samples, self.cycle_samples) >= self.MIN_TIMING_SAMPLES
        if not warm or self.plain_ms <= 0.0:
            return self.min_emitted
        return max(self.min_emitted, min(self.cycle_ms / self.plain_ms, self.max_bar))

    def should_speculate(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
            self.plain_steps += 1
            return False
        return True

    def record(self, emitted: int) -> None:
        self.cycles += 1
        if self.probes_left:
            self.probes_left -= 1
            if emitted >= self.probe_resume:
                self.probes_left = 0
                self.ema = float(emitted)
                self.next_cooldown = self.base_cooldown
            elif not self.probes_left:
                self._cool_down()
            # else: the next iteration is the second probe, with no cooldown between them
            return
        self.ema += self.alpha * (emitted - self.ema)
        # The mean's bar is read HERE and only here, so a bar that moved while the request was
        # cooling down (plain steps keep arriving, and their cost keeps growing with context)
        # is applied to the next cycle the request actually spends -- never retroactively to
        # a probe, which answers to ``probe_resume`` as a single sample always has.
        if self.ema < self.bar:
            self._cool_down()

    def record_plain_ms(self, ms: float) -> None:
        """One plain decode step's wall time. Fed during cooldowns too -- that is how the bar
        keeps tracking a growing context while speculation is switched off."""
        self.plain_ms = ms if not self.plain_samples else self.plain_ms + self.alpha * (
            ms - self.plain_ms
        )
        self.plain_samples += 1

    def record_cycle_ms(self, ms: float) -> None:
        """One speculative cycle's wall time, draft through ship."""
        self.cycle_ms = ms if not self.cycle_samples else self.cycle_ms + self.alpha * (
            ms - self.cycle_ms
        )
        self.cycle_samples += 1

    def _cool_down(self) -> None:
        self.remaining = self.next_cooldown
        self.probes_left = self.PROBE_CYCLES
        self.next_cooldown = min(2 * self.next_cooldown, self.cooldown_cap)


class _SpecTimingProbe:
    """Accumulates per-stage wall time across speculative cycles; logs every 32 cycles.

    Stage names are free-form and aggregated as they arrive, so a callee handed the probe can
    subdivide the caller's stage. A DOTTED name ("verify.forward") is such a sub-stage: it is
    timed against the last sub-mark instead of the last coarse mark, which leaves the enclosing
    stage's span exactly what it was before the subdivision existed.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.stages: dict[str, float] = {}
        self.cycles = 0
        self.emitted = 0
        self.accepted_rows = 0
        self.acceptance: dict[str, float] = {}
        self.extra: dict[str, float] = {}
        self._t0 = 0.0
        self._sub = 0.0

    def _now(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def start_cycle(self) -> None:
        self._t0 = self._sub = self._now()

    def mark(self, stage: str) -> None:
        now = self._now()
        if "." in stage:
            self.stages[stage] = self.stages.get(stage, 0.0) + (now - self._sub)
        else:
            self.stages[stage] = self.stages.get(stage, 0.0) + (now - self._t0)
            self._t0 = now
        self._sub = now

    def add_ms(self, name: str, ms: float) -> None:
        """A measurement the timed code took itself, accumulated beside the marked stages.

        Not a mark: these spans are measured inside a callee (the graph replay's host/GPU
        split), so they neither consume nor subdivide any stage this probe clocks.
        """
        self.extra[name] = self.extra.get(name, 0.0) + float(ms)

    def finish_cycle(
        self,
        *,
        emitted: int,
        accepted: int,
        policy: "_SpecAcceptance | None" = None,
        decision=None,
    ) -> None:
        self.mark("tail")
        self.cycles += 1
        self.emitted += emitted
        self.accepted_rows += accepted
        for name in ("filter_ms", "decide_ms", "sync_ms"):
            value = getattr(decision, name, None)
            if value is not None:
                self.acceptance[name] = self.acceptance.get(name, 0.0) + float(value)
        if self.cycles % 32 == 0:
            per = {k: f"{1e3 * v / self.cycles:.1f}" for k, v in self.stages.items()}
            # The acceptance split attributes "verify.accept": only sync_ms is device work, the
            # other two are the host-side launch cost of the filter and the decision.
            accept = {
                k: f"{v / self.cycles:.1f}" for k, v in self.acceptance.items()
            }
            # The replay split attributes "verify.forward": everything before the launch is
            # host-side staging, and only "replay.gpu" is the graph's own device duration.
            replay = {k: f"{v / self.cycles:.1f}" for k, v in self.extra.items()}
            # The EMA and the spec/plain split are what a live tuning pass of
            # FREETOKEN_MTP_SPEC_MIN_EMITTED / _COOLDOWN reads. The bar and the two wall-time
            # EMAs beside it are what says WHY the fallback fired: a bar that has climbed away
            # from min_emitted is a request whose cycles got dear as its context grew, not a
            # draft that got worse.
            adaptive = (
                "off"
                if policy is None
                else (
                    f"{policy.ema:.2f} | bar {policy.bar:.2f} "
                    f"(cycle {policy.cycle_ms:.1f}ms / plain {policy.plain_ms:.1f}ms) "
                    f"| spec/plain {policy.cycles}/{policy.plain_steps}"
                )
            )
            logger.info(
                "spec timing over %d cycles: ms/cycle %s | accept %s | replay %s | "
                "emitted/cycle %.2f | ema %s",
                self.cycles, per, accept, replay, self.emitted / self.cycles, adaptive,
            )


class _SpecConfLog:
    """One JSON line per speculative cycle: what the draft believed, and what was accepted.

    The evidence for CONFIDENCE-CUT DRAFTING. The head drafts a fixed ``depth`` every cycle and
    the target verifies all of them, so a draft that was never going to be accepted still costs
    its row in the verify forward -- the expensive direction on an expert-offload box. The
    question this log answers is whether the draft's OWN confidence says which drafts those
    are, before the target is asked. Analysis is offline (see the companion script): here we
    only record, never decide, and nothing in the serving path reads a line back.

    Two confidences are written because they answer it at different temperatures.
    ``draft_q`` is the acceptance test's own ``q`` -- the drafted token's probability under the
    REQUEST's sampling filter -- which is exactly what a sampled request's ratio divides by,
    but is degenerate at temperature 0 (the filter collapses onto the argmax, so ``q`` is 1.0
    for every draft whether or not the target agrees). ``draft_top1`` / ``draft_top1_gap`` are
    the head's RAW softmax top-1 and its top1-top2 margin, which stay informative under a
    greedy filter; they are None when the draft head did not compute them.

    Buffered and appended, because a cycle is ~20 ms and an fsync per cycle would be a
    measurable share of it. The file is opened per flush rather than held open so that a
    killed server leaves everything up to the last flush readable.
    """

    FLUSH_EVERY = 100

    def __init__(self, directory: str) -> None:
        import atexit
        import os

        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"conf-log-{os.getpid()}.jsonl")
        self.cycles = 0
        self._lines: List[str] = []
        self._broken = False
        atexit.register(self.flush)

    def record(self, *, uid: int, cached_len: int, proposal, decision, emitted: int) -> None:
        import json

        top1 = getattr(proposal, "draft_top1", None)
        gap = getattr(proposal, "draft_top1_gap", None)
        self._lines.append(
            json.dumps(
                {
                    "uid": int(uid),
                    "cached_len": int(cached_len),
                    "draft_q": [float(v) for v in decision.draft_probabilities],
                    "draft_top1": None if top1 is None else [float(v) for v in top1],
                    "draft_top1_gap": None if gap is None else [float(v) for v in gap],
                    "accepted_drafts": int(decision.accepted_drafts),
                    "emitted": int(emitted),
                    "greedy": bool(decision.greedy),
                }
            )
        )
        self.cycles += 1
        if len(self._lines) >= self.FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        """Append the buffer, best effort -- a diagnosis log never takes the server down."""
        lines, self._lines = self._lines, []
        if not lines:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write("".join(line + "\n" for line in lines))
        except OSError as error:  # pragma: no cover - a full or vanished log directory
            if not self._broken:
                self._broken = True
                logger.warning("spec confidence log %s is not writable: %s", self.path, error)


def _make_spec_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """Destination of each row of a speculative step's sampled tokens.

    Row ``i`` reads position ``cached_len + i`` and samples the token for ``cached_len + i +
    1``, so the run's write base is one past row 0 -- not ``device_len``, which the batch has
    already advanced to the last speculative row. (For plain decode the two coincide, which is
    why ``_make_write_tuple`` can key off ``device_len``.) A slot past the request's output
    budget writes to the token pool's -1 discard column, as it always has.
    """
    mapping_list: List[int] = []
    write_list: List[int] = []
    for req in batch.reqs:
        for i in range(batch.emit_width):
            position = req.cached_len + 1 + i
            mapping_list.append(req.table_idx)
            write_list.append(position if position < req.max_device_len else -1)
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
