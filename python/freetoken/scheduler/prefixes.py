"""One scheduler's named, exact token prefixes and coalesced preparation jobs.

The radix tree owns device pages and recurrent snapshots. This registry owns only CPU
source tokens, waiting requests, and optional tree references. It never stores tensor
addresses across a pool rebuild, nor serves a snapshot from a different prefix boundary.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from freetoken.core import SamplingParams
from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle

from .utils import PendingReq


@dataclass
class PrefixEntry:
    key: str
    tokens: torch.Tensor
    ttl: float
    # Requested lengths belong to aliases; aligned source tokens belong to the entry.
    names: dict[str, int] = field(default_factory=dict)
    waiters: dict[int, PendingReq] = field(default_factory=dict)
    handoffs: set[int] = field(default_factory=set)
    job_uid: int | None = None
    started: bool = False
    ever_ready: bool = False
    lease: HybridCacheHandle | None = None
    expires_at: float = 0.0
    error: str | None = None
    preparations: int = 0
    restores: int = 0
    restored_tokens: int = 0
    forwarded_tokens: int = 0
    followers: int = 0


class PrefixCoordinator:
    MAX_ALIASES = 64
    MAX_SOURCE_BYTES = 16 << 20

    def __init__(self, cache_manager, enqueue: Callable[[PendingReq], None], *,
                 max_seq_len: int, kv_bytes_per_token: int, state_bytes: int,
                 supported: bool = True, clock=time.monotonic):
        self.cm = cache_manager
        self.enqueue = enqueue
        self.max_seq_len = max_seq_len
        self.kv_bytes_per_token = kv_bytes_per_token
        self.state_bytes = state_bytes
        self.supported = supported and cache_manager.is_hybrid
        self.clock = clock
        pool_bytes = cache_manager.num_pages * cache_manager.page_size * kv_bytes_per_token
        if self.supported:
            pool_bytes += (cache_manager.linear_state_pool.num_slots - 1) * state_bytes
        self.max_retained_bytes = min(1 << 30, pool_bytes // 2)
        self.entries: dict[str, PrefixEntry] = {}
        self.names: dict[str, str] = {}
        self.jobs: dict[int, str] = {}
        self.next_uid = -1
        if self.supported:
            cache_manager.release_prefix_preferences = self.release_preferences

    @staticmethod
    def _reply(status="ok", result=None, error=None):
        return {"status": status, "result": result or {}, "error": error}

    def command(self, msg) -> dict:
        if not self.supported:
            return self._reply("unsupported", error="Named prefixes require text QSA/GDN hybrid radix on one GPU (TP=1)")
        self.expire()
        try:
            if msg.action == "register":
                return self._register(msg)
            if msg.action == "list":
                pages, states = self._physical_cost([e.lease for e in self.entries.values() if e.lease])
                resident = [self._resident(e) for e in self.entries.values()]
                gpu_pages, gpu_states = self._physical_cost([m for m in resident if m is not None])
                return self._reply(result={
                    "prefixes": [self._describe(self.entries[key], name) for name, key in self.names.items()],
                    "source_bytes": self.source_bytes,
                    "retained_bytes": pages * self.cm.page_size * self.kv_bytes_per_token + states * self.state_bytes,
                    "retained_pages": pages, "retained_snapshots": states,
                    "gpu_unique_pages": gpu_pages, "gpu_unique_snapshots": gpu_states,
                    "gpu_attention_bytes": gpu_pages * self.cm.page_size * self.kv_bytes_per_token,
                    "gpu_snapshot_bytes": gpu_states * self.state_bytes,
                    "private_state_bytes_per_request": 3 * self.state_bytes,
                    "max_retained_bytes": self.max_retained_bytes,
                    "max_aliases": self.MAX_ALIASES, "max_source_bytes": self.MAX_SOURCE_BYTES,
                })
            if msg.action == "configure":
                value = msg.max_retained_bytes
                if type(value) is not int or value < 0:
                    raise ValueError("max_retained_bytes must be a non-negative integer")
                self.max_retained_bytes = value
                self.release_preferences()
                return self._reply(result={"max_retained_bytes": value})
            key = self.names.get(msg.name)
            if key is None:
                return self._reply("not_found", error="Unknown prefix name")
            entry = self.entries[key]
            if msg.action == "get":
                return self._reply(result=self._describe(entry, msg.name))
            if msg.action == "warm":
                if self._resident(entry):
                    self._retain(entry)
                    return self._reply(result=self._describe(entry, msg.name))
                self._queue(entry)
                return self._reply("warming", self._describe(entry, msg.name))
            if msg.action == "delete":
                del self.names[msg.name]
                del entry.names[msg.name]
                if not entry.names:
                    entry.expires_at = 0
                    self._release_if_unused(entry)
                    self._collect(entry)
                return self._reply(result={"name": msg.name, "deleted": True})
            raise ValueError("Unknown prefix action")
        except (ValueError, TypeError) as exc:
            return self._reply("invalid", error=str(exc))

    @property
    def source_bytes(self):
        return sum(e.tokens.numel() * e.tokens.element_size() for e in self.entries.values())

    def _register(self, msg):
        if not isinstance(msg.name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", msg.name):
            raise ValueError("name must be 1–128 letters, digits, dots, colons, underscores or hyphens")
        ttl = msg.ttl_seconds
        if isinstance(ttl, bool) or not isinstance(ttl, (float, int)) or not math.isfinite(ttl) or not 0 <= ttl <= 86400:
            raise ValueError("ttl_seconds must be finite and between 0 and 86400")
        ids = msg.input_ids
        if not isinstance(ids, torch.Tensor) or not ids.is_cpu or ids.ndim != 1 or ids.dtype != torch.int32:
            raise ValueError("registration requires CPU int32 token IDs from the active tokenizer")
        requested = len(ids) if msg.prefix_tokens is None else msg.prefix_tokens
        if type(requested) is not int or not 0 < requested <= len(ids):
            raise ValueError("prefix_tokens must select a non-empty prefix of the rendered request")
        # Generation needs a non-empty prompt extension to produce its first logits.
        # If the whole rendered prompt is aligned, keep its final page private;
        # the coalesced checkpoint must be consumable by that same request.
        length = min(requested, len(ids) - 1) // self.cm.page_size * self.cm.page_size
        if length < self.cm.page_size:
            raise ValueError("registration must leave a prompt tail after at least one whole cache page")
        if not self.cm.page_size <= length < self.max_seq_len:
            raise ValueError("aligned prefix must be at least one page and leave context for a request tail")
        if length + self.cm.page_size > self.cm.num_pages * self.cm.page_size:
            raise ValueError("prefix must fit the KV pool with at least one page for a request tail")
        tokens = ids[:length].contiguous()
        key = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
        if msg.name in self.names and self.names[msg.name] != key:
            raise ValueError("name already describes different tokens; delete it or use a new name")
        if msg.name not in self.names and len(self.names) >= self.MAX_ALIASES:
            return self._reply("busy", error="Prefix alias limit reached")
        entry = self.entries.get(key)
        if entry is not None and not torch.equal(entry.tokens, tokens):
            raise ValueError("Prefix identity collision")
        if entry is None:
            if len(self.entries) >= self.MAX_ALIASES:
                return self._reply("busy", error="Prefix preparation limit reached; let orphaned jobs finish")
            if self.source_bytes + length * 4 > self.MAX_SOURCE_BYTES:
                return self._reply("busy", error="Prefix source-token budget reached")
            entry = PrefixEntry(key, tokens.clone(), float(ttl))
            self.entries[key] = entry
        entry.ttl = float(ttl)
        entry.names[msg.name] = requested
        self.names[msg.name] = key
        return self._reply(result=self._describe(entry, msg.name))

    def _resident(self, entry):
        match = self.cm.prefix_cache.match_prefix(entry.tokens, touch=False)
        return match if match.cached_len == len(entry.tokens) else None

    def _describe(self, entry, name):
        match = self.cm.prefix_cache.match_prefix(entry.tokens, touch=False)
        state = ("warming" if entry.job_uid is not None else "ready" if match.cached_len == len(entry.tokens)
                 else "error" if entry.error else "evicted" if entry.ever_ready else "registered")
        parked = None
        if self.cm.park_store is not None:
            # Inspection must report stored bytes independently of GPU capacity or the
            # restore profitability gate. A store hit is not a residency guarantee.
            parked = self.cm.park_store.lookup(entry.tokens, min_len=len(entry.tokens),
                                               max_len=len(entry.tokens), touch=False)
        return {"name": name, "id": entry.key, "prefix_tokens": len(entry.tokens),
                "requested_tokens": entry.names[name], "alignment": self.cm.page_size,
                "state": state, "gpu_cached_tokens": match.cached_len,
                "retained": entry.lease is not None, "ttl_seconds": entry.ttl,
                "waiting_requests": len(entry.waiters), "preparations": entry.preparations,
                "restores": entry.restores, "followers": entry.followers, "error": entry.error,
                "restored_tokens": entry.restored_tokens, "forwarded_tokens": entry.forwarded_tokens,
                "parked_tokens": parked.token_count if parked is not None else 0,
                "parking_backend": getattr(self.cm.park_store, "mode", "off")}

    def route(self, pending: PendingReq) -> bool:
        if not self.supported or pending.cache_private or pending.mm_embeds is not None:
            return False
        self.expire()
        candidates = (e for e in self.entries.values() if e.names and len(e.tokens) < pending.input_len)
        for entry in sorted(candidates, key=lambda e: len(e.tokens), reverse=True):
            if not torch.equal(entry.tokens, pending.input_ids[:len(entry.tokens)]):
                continue
            if entry.job_uid is None and self._resident(entry):
                self._retain(entry)
                return False
            entry.waiters[pending.uid] = pending
            entry.followers += 1
            self._queue(entry)
            return True
        return False

    def _queue(self, entry):
        if entry.job_uid is not None:
            return
        uid = self.next_uid
        self.next_uid -= 1
        entry.job_uid, entry.started, entry.error = uid, False, None
        self.jobs[uid] = entry.key
        self.enqueue(PendingReq(uid, entry.tokens, SamplingParams(max_tokens=0),
                                prefill_only=True, prefix_key=entry.key))

    def before_admit(self, pending):
        """An exact restored hit needs no forward (Req requires a non-empty extension)."""
        if not pending.prefill_only or pending.chunked_req is not None:
            return False
        generation = (self.cm.park_generation, self.cm.available_size,
                      self.cm.linear_state_pool.num_free_slots)
        if getattr(pending, "admission_generation", None) == generation:
            return False
        entry = self.entries[pending.prefix_key]
        before = self.cm.prefix_cache.match_prefix(entry.tokens).cached_len
        match = self.cm.match_req(pending)
        restored_tokens = max(0, match.cuda_handle.cached_len - before)
        entry.restores += int(restored_tokens > 0)
        entry.restored_tokens += restored_tokens
        if match.cuda_handle.cached_len == pending.input_len:
            self._ready(entry)
            return True
        # Consume this very match in the adder: a second park lookup could turn a
        # failed restore into a full hit and leave it trying to forward zero tokens.
        pending.preparation_match = match
        return False

    def prepared_chunk(self, req):
        entry = self.entries[req.prefix_key]
        if not entry.started:
            entry.preparations += 1
            entry.started = True
        req.prefix_chunk_tokens = req.extend_len

    def drained_chunk(self, req):
        self.entries[req.prefix_key].forwarded_tokens += req.prefix_chunk_tokens
        req.prefix_chunk_tokens = 0

    def completed(self, req):
        entry = self.entries[req.prefix_key]
        match = self._resident(entry)
        if match is None:
            self.failed(req.uid, "Exact prefix checkpoint was not published")
            return
        # The existing synchronous save returns before this temporary tree lock drops.
        if self.cm._checkpoint_match(match, len(entry.tokens)):
            self.cm.prefix_cache.inc_lock(match.node)
            try:
                self.cm._save_prompt_checkpoint(match, entry.tokens, len(entry.tokens))
            finally:
                self.cm.prefix_cache.dec_lock(match.node)
        self._ready(entry)

    def _ready(self, entry):
        self.jobs.pop(entry.job_uid, None)
        entry.job_uid, entry.error, entry.ever_ready = None, None, True
        entry.handoffs.update(entry.waiters)
        self._retain(entry)
        self._release_waiters(entry)
        self._collect(entry)

    def _release_waiters(self, entry):
        for pending in entry.waiters.values():
            self.enqueue(pending)  # direct admission; never re-route failures into another job
        entry.waiters.clear()

    def failed(self, uid, error):
        key = self.jobs.pop(uid, None)
        if key is None:
            return False
        entry = self.entries[key]
        entry.job_uid, entry.error = None, error
        self._release_waiters(entry)
        self._collect(entry)
        return True

    def cancel_waiter(self, uid):
        for entry in list(self.entries.values()):
            entry.waiters.pop(uid, None)
            entry.handoffs.discard(uid)
            self._release_if_unused(entry)
            self._collect(entry)
        self._trim_preferences()

    def admitted(self, uid):
        self.cancel_waiter(uid)  # active request now owns its own radix reference

    def _physical_cost(self, handles):
        # Management path only: count physical pages once, including parent/child aliases.
        nodes, pages, snapshots = set(), 0, set()
        for handle in handles:
            node = handle.node
            while not node.is_root():
                if id(node) not in nodes:
                    nodes.add(id(node))
                    pages += node.length // self.cm.page_size
                    if node.mamba_value is not None:
                        snapshots.add(node.mamba_value)
                node = node.parent
        return pages, len(snapshots)

    def _retain(self, entry):
        match = self._resident(entry)
        if match is None:
            return
        if entry.lease is None:
            entry.lease = HybridCacheHandle(match.cached_len, match.node, match.kv_indices)
            self.cm.lock(entry.lease)
        entry.expires_at = self.clock() + entry.ttl if entry.names else 0
        pages, states = self._physical_cost([e.lease for e in self.entries.values() if e.lease])
        if pages * self.cm.page_size * self.kv_bytes_per_token + states * self.state_bytes > self.max_retained_bytes:
            entry.expires_at = 0
        self._release_if_unused(entry)
        self._trim_preferences()

    def _trim_preferences(self):
        # Inserting a shorter checkpoint may add a protected ancestor snapshot to an
        # already-held child. Dropping just the new entry's lease cannot fix that union.
        while True:
            pages, states = self._physical_cost([e.lease for e in self.entries.values() if e.lease])
            if pages * self.cm.page_size * self.kv_bytes_per_token + states * self.state_bytes <= self.max_retained_bytes:
                return
            candidates = [e for e in self.entries.values() if e.lease and not e.handoffs]
            if not candidates:
                return  # transient handoffs yield at the admission pressure gate instead
            victim = min(candidates, key=lambda e: e.expires_at)
            self.cm.unlock(victim.lease)
            victim.lease, victim.expires_at = None, 0

    def _release_if_unused(self, entry):
        if entry.lease is not None and not entry.handoffs and entry.expires_at <= self.clock():
            self.cm.unlock(entry.lease)
            entry.lease = None

    def _collect(self, entry):
        if not entry.names and not entry.waiters and not entry.handoffs and entry.job_uid is None:
            self._release_if_unused(entry)
            self.entries.pop(entry.key, None)

    def expire(self):
        for entry in list(self.entries.values()):
            self._release_if_unused(entry)
            self._collect(entry)
        self._trim_preferences()

    def release_preferences(self):
        """Under pressure even handoff leases yield; active request refs remain intact."""
        released = False
        for entry in self.entries.values():
            entry.expires_at = 0
            if entry.lease is not None:
                self.cm.unlock(entry.lease)
                entry.lease = None
                released = True
        return released

    def before_rebuild(self):
        self.release_preferences()

    def engine_failed(self, reason):
        """Detach waiters without touching pools that a failed teardown invalidated."""
        waiters = []
        for entry in self.entries.values():
            waiters.extend(entry.waiters.values())
            entry.waiters.clear()
            entry.handoffs.clear()
            entry.lease, entry.expires_at = None, 0
            entry.job_uid, entry.error = None, reason
        self.jobs.clear()
        return waiters

    def next_delay_ms(self):
        deadlines = [e.expires_at for e in self.entries.values() if e.lease and not e.handoffs]
        return max(1, math.ceil((min(deadlines) - self.clock()) * 1000)) if deadlines else None
