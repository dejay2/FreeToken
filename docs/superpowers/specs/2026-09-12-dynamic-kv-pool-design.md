# Dynamic KV pool: grow the conversation memory on demand, shrink it back on a timer

Date: 2026-09-12. Status: approved design (Jay), written by the Fable lead; revised the same day after two external review rounds (commit 8262e14: seven required changes; commit 702543b: four more; commit 70d79c7: two admission corrections), each verified against the code before being applied. Target: Qwen3.8-Flash-Next-NVFP4 in the `vllm` WSL distro on the RTX 5090, branch `mtp-upstream-merge`. Linux/WSL launch only; the old Windows launcher is out of scope.

## Goal

Boot with a small KV pool and spend the saved VRAM on MoE expert slots. When a request arrives that
could outgrow the pool, trade slots for KV pages before admitting it. When the card has served no
request for a while, shrink the pool back and return the slots. Every trade is byte-neutral, runs
only at an idle safe point through the existing `rebuild_cache` path, and loses no conversation:
KV parking to host RAM (`--kv-park ram`, live on the box) already holds a copy of every finished
prefix, so a resized pool is refilled from RAM on the next turn instead of re-read.

Decisions fixed by Jay (do not relitigate):

- Our own implementation inside the scheduler/engine, not upstream PR #300 (grow-only, one request
  at a time, no shrink). Its pure rung arithmetic is borrowed.
- The helper's memory governor does not drive it: it cannot see what an arriving request needs.
  The engine's ladder is the driver; the governor stays the authority on the free-VRAM cushion.
- Two timers, Anthropic-style: Timer 1 shrinks the card pool after a quiet spell (default
  10 min); Timer 2 drops a parked prefix from host RAM after a long quiet spell (default 5 h).
- No cap on concurrent requests beyond the existing `MaxRunningRequests` dial.
- Step dial: default 32,768 tokens, page minimum 8,192.
- Settings live on the web page (Model & chats tab); defaults as listed below.
- The cold slot cache after a resize (1-3 s of slower decode) is accepted for this batch. The
  named follow-up is the never-move-the-pools design (CUDA virtual-memory reservation), not a
  copy-across: the card cannot hold the old and new slot pools at once (about 17 GB against
  1.6 GB free), and neither can host RAM.

## Measured facts this design rests on

| Fact | Value | Source |
|---|---|---|
| Live boot flags | `--kv-reserve-tokens 262144 --num-tokens 262208 --kv-dtype fp8 --moe-cache-auto --moe-cache-headroom-bytes 1610612736 --max-running-requests 2 --kv-park ram --kv-park-idle-ms 0 --kv-park-min-tokens 8192 --kv-park-ram-gib 8.0` | `logs/server-2020.log` on the box, 2026-09-12 |
| KV bytes per token, fp8 QSA geometry | 13,248 B (`kvcache/base.py:spec_kv_bytes_per_token`, 12 QSA layers, FP8 K/V + 2-byte index + FP32 scales); the live pool at 262,208 tokens is 4,097 pages, 3.24 GiB | code; `logs/server-2020.log` |
| One nvfp4 expert slot | 2,772,480 B across the three banks; 1,000 slots = 2.58 GiB | `docs/research/memory-audit-qwen38-rtx5090.md` |
| Slots at boot with the full pool | about 6,260 | commit 4ff86b3 message; `kv-ram-conversation-switching-2026-09-10.md` (5,972 / 6,262) |
| Decode speed per 1,000 slots | about 6 % (76.0 vs 71.2 tok/s for 6,144 vs 5,405 slots, pi 6.8k prompt) | memory `project-crash-watchdog`, pi A/B 2026-09-08 |
| Cache step duration on this box, graphs for bs [1, 2] included | under 1 s (three steps at 01:13:33/40/47 each start and finish inside one log second) | `logs/server-2020.log` 2026-09-12 |
| Upstream first visit to a new geometry | 3-4 s, mostly graph capture; 0.67-0.99 s warm | PR #300 description |
| RAM park restore of a 200k prefix | 1.5-1.9 s, byte-exact, eight of eight | `kv-ram-conversation-switching-2026-09-10.md` |
| Page exhaustion mid-request | assertion `Eviction did not free enough space` (`scheduler/cache.py:1158`), a crash | code |

Arithmetic (13,248 B/token, 2,772,480 B/slot): one 32,768-token step is 0.404 GiB = 157 slots;
growing from 65,536 to 262,144 usable tokens costs 2.426 GiB = about 940 slots. The daily boot
would start at about 7,200 slots (6,260 + 940, a calculation, not a measured boot) and fall back to
about 6,260 when a full-context request is live.
Expected decode gain for short chats after a shrink: 5-8 %, a benchmark target, not a verified result; the net win also depends on how much of the day's traffic stays under the floor against the resize and restore delays it pays. Not 33 %: PR #300 measured a bf16 pool
on a steeper part of the slot-count curve.

## Behaviour

Definitions. `need(req) = len(input_ids) + max_tokens` after the existing clip to `max_seq_len`
(this is `estimated_len` in `PrefillManager`, `prefill.py:79`). `pool` = `engine.num_pages *
page_size`. `floor`, `ceiling`, `step` are the dials below. `idle` = no runnable prefill, no
running decode, no chunked continuation, no held request being admitted.

1. **Boot.** The pool is sized to `floor` tokens (plus the dummy page, as `--num-tokens` is today).
   `--moe-cache-auto` plans the slots against that pool with the same headroom, owned-layer and MTP
   reserves it uses now. An explicit `MoECacheSize` is respected as today; the controller then
   trades from it.
2. **Grow on admission.** For every arriving `UserMsg` (after the abort-tombstone check, before
   `prefill_manager.add_one_req`), first clip against the **ceiling**, not the current pool: a
   prompt longer than `ceiling` is refused with the existing `context_length_exceeded` reply, and
   `max_tokens` is clipped to `ceiling - prompt`. Two sizes are computed from the clipped value:
   - `need_total = len(input_ids) + max_tokens`: the room the request needs in an **empty** pool.
     This sizes a rebuilt pool (a rebuild wipes the tree; the RAM restore re-allocates the whole
     prefix), so it is the right figure for a growth target.
   - `need_now = extend_len + max_tokens` with `extend_len = len(input_ids) - resident_cached_len`:
     the extra room the request needs **right now** given prefixes already resident on the card.
     `match_req` must NOT be used for this: it looks up parked prefixes and can call
     `_restore_parked`, which allocates pages and a GDN slot and copies data back
     (`scheduler/cache.py:114-205`), so it is a mutating operation whatever locking is skipped.
     Instead a genuinely non-mutating `CacheManager.probe_admission(input_ids, output_len)` walks
     the radix tree only (no park lookup, no restore, no lock) and returns
     `(need_now, protect_tokens, fits_empty, fits_now)`. **Every quantity is in tokens**:
     `available_size` is tokens (`cache.py:274-277`, evictable tokens plus free pages times
     `page_size`), `reserved_size` is tokens, and `protect_tokens = protect_pages * page_size`
     where `protect_pages` is the matched prefix's pages that are currently evictable and would
     become locked on real admission. The real path checks capacity before and after `lock`
     (`prefill.py:96-99`) precisely because locking removes those pages from `available_size`; the
     probe reproduces the post-lock answer as
     `fits_now = need_now + reserved_size + uncommitted_tokens <= available_size - protect_tokens`
     (`uncommitted_tokens` is defined under "Same-batch arrivals" below). A parked-but-not-resident prefix
     counts as not cached (its restore would allocate the whole prefix again). Both the probe and
     `_try_admit` call one pure function `admission_fits(need_now, reserved, available, to_protect)`
     so there is a single admission formula. A request sharing a large locked prefix must not
     trigger a grow that the existing path would have admitted.
   Then, in order:
   - If the held FIFO is non-empty: **join it** (arrival order). This is the drain barrier: once
     any request waits for growth, later arrivals queue behind it instead of overtaking it, the
     running requests finish, the pool grows once, and the FIFO drains in order. Without it a
     stream of small requests could keep the scheduler busy and starve the held one forever.
   - Else if `need_total > pool`: hold. At the next idle point the controller rebuilds to
     `target = min(ceiling, max(pool + step, round_up(need_total + 1, step)))`, funded by slots,
     then drains the FIFO: each entry is re-planned against the geometry its predecessor left;
     entries that fit are admitted with no further rebuild.
   - Else if `fits_empty and not fits_now and pool < ceiling` (the probe's post-lock verdict, not a
     separate inequality: the decision must consume the same `fits_now` the probe computed): hold
     with a growth target of
     `round_up(sum(need_total of running and pending requests) + need_total + 1, step)`, capped at
     `ceiling`. The shared-prefix overestimate here only rounds the target up, never triggers a
     hold on its own. The request waits for the running requests to finish, as it would have waited
     on pages today, then the pool grows so the two run side by side next time.
   - Else: admitted at once, no rebuild.
   **Same-batch arrivals.** The scheduler drains every received message before it schedules, and
   `PrefillManager.add_one_req` only appends to the pending list (`prefill.py:278`); reservations
   are charged later, in the admission pass (`prefill.py:200`). Two requests arriving in one batch
   would each see the full pool, both be appended, and then serialise in prefill with no grow.
   Two rules close this. (a) The controller keeps `uncommitted_tokens`: the sum of `need_now` for
   requests it passed to `add_one_req` whose admission pass has not yet run; the probe adds it (see
   `fits_now` above). The entry for a uid is removed the moment `_try_admit` charges the real
   reservation for it, or on abort, so nothing is charged twice. (b) Escalation: when `_try_admit`
   returns `None` for capacity on a never-started pending request (no chunk running, no pages
   held) while `pool < ceiling`, the prefill manager hands it back to the controller, which moves it
   into the held FIFO with the concurrent growth target and the drain barrier engages. Rule (a)
   catches the common case at arrival; rule (b) is the safety net for anything that slipped past
   the estimate. Regression test: two distinct-prefix requests of `need_total` 39,936 delivered in
   one batch into a 65,536 pool, no third arrival; the second is held (or escalated) and the
   controller plans 98,304, instead of both running one after the other with no grow.
   The controller never rebuilds while a request is active and never rebuilds twice for one
   arrival. Latency trade-off, stated plainly: a request that today runs alongside another in the
   fixed 262k pool may, while the dynamic pool is small, have to wait behind it once before the
   pool grows. The live run logs every hold with its duration.
3. **Funding.** One authoritative byte budget, owned by the engine and read by both the planner
   and the validator, so they can never disagree:
   `pool_budget_bytes = moe_slots * slot_bytes + num_pages * kv_bytes_per_page`, snapshotted
   after boot and re-snapshotted after every rebuild the controller did not issue (a governor
   `/v1/cache/step`, a manual `/v1/cache/rebuild`). A governor slot step (-512) lowers it by 512
   slots' bytes; a governor KV shrink to the floor with slots untouched lowers it by the KV bytes
   it freed; a later grow is then funded from slots, never by asking the card for the cushion
   back. The governor stays the authority on the total; the controller only moves bytes between
   the two pools inside it.
   A plan for `target_pages` sets
   `target_slots = max(slot_floor, (pool_budget_bytes - target_pages * kv_bytes_per_page) // slot_bytes)`,
   where `slot_floor` is `2 * num_experts` with prefill overlap, else `num_experts` (the floor
   `step_memory` uses). Floor division means every geometry the controller produces satisfies
   `target_slots * slot_bytes + target_pages * kv_bytes_per_page <= pool_budget_bytes` exactly,
   with unused slack below one slot (2.77 MB) that never accumulates. Byte-neutrality is defined
   against this budget, not per transition: one 32,768-token step is 156.6 slots, so a grow that
   releases 156 slots leaves the resident total about 1.5 MiB higher than before while still inside
   the budget, and a shrink back to the floor can likewise land a few hundred KiB above the
   pre-grow total. The policy test asserts the budget inequality on every reachable geometry
   (all page-aligned pools from floor to ceiling) and that `target_slots` is monotone
   non-increasing in `target_pages`; it does not assert `bytes_before >= bytes_after` per step.
   `slot_baseline` in the rest of this document means `target_slots(floor_pages)`.
   If the budget cannot fund `target_pages` above `slot_floor`, the pool grows only as far as
   `slot_floor` funds (`capped=True`, one warning log line) and the request is admitted against
   that pool: the ordinary admission path then clips or refuses as it does today.
4. **Timer 1: shrink.** When idle and `now - last_request_finished >= shrink_idle_s` and
   `pool > floor`: `prepare_rebuild` (drain pending parks, park every eligible prefix
   synchronously), then rebuild to `floor_pages` with `slots = slots_for_pages(floor_pages)`. Prefixes shorter than
   `--kv-park-min-tokens` are lost and re-read; they are cheap. The timer is checked in
   `run_when_idle`, which the I/O mixin calls on every idle poll timeout (`io.py:_wait_from_queue`);
   `idle_poll_timeout_ms` returns `min(next_park_delay_ms, next_park_expiry_ms, time to the next
   controller deadline)` so a quiet server wakes for the shrink. Today `next_park_delay_ms`
   returns `None` once nothing is pending or parkable (`cache.py:379-388`) and the loop then blocks
   on the queue with no timer at all; both new deadlines must feed the same minimum. A shrink found due in `run_when_idle` executes there:
   the scheduler is idle by construction inside the blocking wait, so it sets `_pending_rebuild`
   and calls `_execute_pending_rebuild()` directly. If a request arrives while the timer is overdue
   and that request itself needs a grow, the shrink is skipped and one plan goes straight to the
   grow target (never two rebuilds for one arrival). More generally a shrink is never planned while
   a never-admitted request sits in the pending list (`plan_idle(allow_shrink=False)`).
5. **Timer 2: RAM TTL.** `ParkStore` gains `ttl_s`; `park_idle()` (already called from
   `run_when_idle`) sweeps entries whose family `max(last_used_ns)` is older than `ttl_s` and
   evicts the family with the existing eviction path (`_evict_to_fit`'s family logic), releasing
   pinned RAM. Applies to both `ram` and `ssd` modes; `0` disables. Independent of the dynamic
   pool switch, including its wake-up: `ParkStore.next_expiry_delay_ms() -> int | None`
   (oldest family's `last_used_ns + ttl` minus `time.time_ns()`, clamped at 0; `None` when empty
   or `ttl_s == 0`) is folded into `CacheManager.next_park_delay_ms`, so an idle scheduler with
   everything already parked still wakes to expire it. Clock contract, stated because the two
   sides differ today: `ParkedEntry.last_used_ns` and the SSD index use the wall clock
   (`time.time_ns()`, `st_mtime_ns`), while `park_idle` and `next_park_delay_ms` take
   `time.monotonic_ns()`. The store therefore does all age arithmetic against its own wall clock
   and hands the scheduler a **relative** delay; the scheduler never passes its monotonic `now_ns`
   into the store. Test with two fake clocks whose origins differ by years: a single fake clock
   starting at zero would hide the bug. Cases: no incoming messages, dynamic pool on and off,
   expiry fires after 5 h and not after 4 h.
6. **Governor interplay.** With the dynamic pool on, `step_memory`'s KV rung changes meaning:
   VRAM down, rung 3, idle-only: shrink to `floor_pages` (not `-25 % of initial`), slots
   unchanged: the freed bytes are the cushion the governor asked for, and the controller's
   re-snapshotted `pool_budget_bytes` drops by the same bytes so the next grow is funded from slots.
   VRAM up: the KV rung is a no-op (`step_memory_noop` reports it as exhausted; the controller
   grows on demand). `_initial_num_pages` is set to `floor_pages` so the existing "KV restore
   pending" logic in `step_memory_noop` never fires.
7. **Aborts.** An `AbortBackendMsg` for a held request removes it from the FIFO (PR #300 logic).
8. **Failure.** The rebuild goes through `_execute_pending_operation`, which has three outcomes
   and must return them (today it returns `None` and only replies on the wire):
   - `"ok"`: admit the held FIFO against the new geometry.
   - `"rejected"` (pre-teardown rejection, or a teardown failure rolled back to the prior
     geometry): the old engine is intact; admit the FIFO against the retained pool and let the
     ordinary path clip or refuse.
   - After any non-`ok` outcome of an automatic plan the controller re-arms Timer 1 a full
     `shrink_idle_s` away and counts a consecutive failure; after three consecutive failures no
     further automatic shrink is planned until a request finishes or a plan succeeds (final review
     of 9ad8f69: without this a shrink refused by the card was retried every millisecond).
   - `"failed"` (teardown failed and the rollback also failed, or tp > 1): the engine is not known
     to be usable and the frontend latches `failed`. **Do not admit.** Every held request gets an
     `ErrorReplyMsg` (`code="server_error"`, text "cache rebuild failed; server needs a restart"),
     the FIFO is cleared, the controller disables itself for the life of the process, and the
     crash watchdog's existing `failed`-state handling restarts the server. Rejection is not only an
     external VRAM grab: `validate_rebuild` also enforces geometry limits and the boot memory
     budget (see the engine section for the allowance a byte-neutral swap needs).
9. **Front door and maintenance lifecycle.** Today only `POST /v1/cache/rebuild` and
   `/v1/cache/step` open a maintenance operation (`api_server._open_maintenance`: state
   `rebuilding`, correlated record, `rebuild_done` cleared), and `_note_progress` drops progress
   for an unknown operation. A rebuild the scheduler starts on its own would therefore run with the
   API showing `serving`: arriving requests would still be safe (the scheduler is single-threaded,
   they queue on the socket), but the status page, the 120 s wait gate and the maintenance
   watchdog would not see it, and a hung idle shrink with no request in flight would go unnoticed.
   Required: the scheduler sends `MaintenanceBeginMsg(request_id, kind="auto-kv", detail)` before
   an automatic rebuild; the frontend opens the operation through the same `_open_maintenance`
   path, receives the existing progress messages, and closes it on the `CacheRebuildReply` it
   already handles, so `new_user` waits on `rebuild_done` exactly as for a manual rebuild.
   Ownership: today the frontend keeps ONE record, `maintenance_op`, written by `_open_maintenance`
   at dispatch time, before the message reaches the scheduler (`api_server.py:167-185, 912, 1033`),
   and cleared by the completion handlers. Reusing it as-is races: the frontend opens manual
   operation M and sends it; the scheduler starts automatic A before it has received M; `Begin(A)`
   overwrites M's record; A completes and the gate reopens with M still outstanding and its
   progress ignored. Required: `maintenance_ops: dict[request_id, record]`. A manual dispatch adds
   its record at dispatch; an automatic `Begin` adds its own; each completion removes only its own
   id; `maintenance_state` is `rebuilding` while the dict is non-empty and `rebuild_done` is set
   only when it empties; `_note_progress` looks the operation up by id; the maintenance watchdog
   times the oldest record. Automatic ids are unique, not descriptive:
   `auto-kv:{instance_id}:{seq}` with a scheduler-side counter, and the reason and target go in
   `detail`. Scheduler side: one operation executes at a time from `_pending_rebuild`; if a manual
   rebuild or governor step is already queued when the controller wants to plan, the controller
   yields (plans on the next idle point, after `note_external_geometry`); a step arriving while the
   controller's rebuild executes queues behind it as today. Regression test: dispatch a manual
   rebuild, delay its delivery to the scheduler, let an automatic begin and completion arrive, and
   assert the frontend still reports `rebuilding` with M outstanding and later resolves M normally.

### Worked example (the daily boot, floor 65,536, ceiling 262,208, step 32,768)

| Event | Rule | Result |
|---|---|---|
| Boot | 1 | pool 65,536; about 7,200 slots (calculated) |
| pi chat 7k prompt, max_tokens 32k | need 39k <= pool | admitted, no rebuild |
| Claude Code A, 60k prompt | need_total 92k > pool, idle | rebuild to 98,304, slots -157, admit; about 1 s |
| A's turn at 95k prompt | need 127k > 98,304, idle between turns | A's 95k already parked (idle 0 ms); rebuild to 131,072; A restores from RAM (about 1.5 s) and reads 5k new tokens |
| B arrives, 40k, while A decodes at 110k | need_now 72k (no shared prefix) fits empty pool, not available; pool < ceiling | B held with target round_up(142k + 72k + 1) = 229,376; when A's turn ends: rebuild, admit B; A's next turn fits alongside. Small requests arriving meanwhile queue behind B. |
| Both stop 12:30; 12:40 | Timer 1 | A and B in RAM (about 2.7 GiB of 8); rebuild to 65,536; slots back to about 7,200; the wake-up comes from the idle poll deadline, not from a new message |
| A resumes at 14:00, 135k | need 167k > pool | rebuild to 196,608, restore 130k from RAM, first token in about 10 s |
| 300k prompt | need > ceiling, prompt > ceiling | refused `prompt is too long`, unchanged |
| 240k prompt | need 272k > ceiling | pool grows to the ceiling, max_tokens clipped to 22k |
| 19:00 | Timer 2 | B dropped from RAM 5 h after its last use; A later |

## Components

### `scheduler/kv_dynamic.py` (new, pure, CUDA-free)

```python
@dataclass(frozen=True)
class KVDynamicPolicy:
    floor_pages: int; ceiling_pages: int; step_pages: int
    page_size: int; kv_bytes_per_page: int; slot_bytes: int; slot_floor: int
    # pool_budget_bytes is passed per call (it changes with external rebuilds), not stored here

@dataclass(frozen=True)
class KVPlan:
    target_pages: int; target_slots: int; capped: bool; reason: str  # "grow" | "grow-concurrent" | "shrink"

def slots_for_pages(policy, pool_budget_bytes, pages) -> int
def plan_grow(policy, *, current_pages, pool_budget_bytes, need_tokens, running_need_tokens=0) -> KVPlan | None
def plan_shrink(policy, *, current_pages, pool_budget_bytes) -> KVPlan | None
```

`__post_init__` validates every field positive, `floor_pages <= ceiling_pages`,
`step_pages >= 8192 // page_size`. `plan_grow` returns `None` when the pool already holds
`need_tokens` (or the concurrent sum) or is at the ceiling. The step rounding follows PR #300: at
least one step, and the `+1` token of breathing room when `need` lands exactly on a rung. Tests:
`tests/scheduler/test_kv_dynamic_policy.py`, table-driven, including the capped case, the ceiling
clamp, the concurrent target, and the budget inequality of rule 3 on every page-aligned pool from floor to ceiling (per-step
byte monotonicity is deliberately not asserted; see rule 3).

### Scheduler (`scheduler/scheduler.py`)

- `self._kv_dynamic: KVDynamicController | None`, built in `__init__` when
  `config.kv_dynamic` is true, from `type(engine.kv_cache).kv_cost(config)[0]` (bytes per page),
  `expert_bytes_per_slot(moe.bank_sources, owned)`, `engine.pool_budget_bytes`, the slot floor from the same expression `step_memory` uses, and the three dials.
  Refuses (ValueError at boot) without an offload MoE cache, without `supports_runtime_rebuild`,
  and on DSV4 (owned KV tiers). TP > 1 is refused for this batch.
- `KVDynamicController` (same module as the policy, or a sibling; no torch import) holds the held
  FIFO, `last_request_finished`, the last-read `pool_budget_bytes`, an operation sequence counter,
  and a `pending_plan`. Methods:
  `on_admission(msg, need, pool_tokens, available_tokens, reserved_tokens, running_need) -> "admit" | "held"`,
  `on_idle(now) -> KVPlan | None` (held head first, then Timer 1), `on_request_finished(now)`,
  `note_external_geometry(slots, pages)` (re-reads `pool_budget_bytes` from the engine), `on_abort(uid)`,
  `next_deadline_ms(now) -> int | None`, `status() -> dict`.
- `_process_one_msg`: the `UserMsg` branch becomes `if not self._queue_for_kv_dynamic(msg):
  self._admit_user_msg(msg)`, with `_admit_user_msg` factored out exactly as PR #300 does.
  `need_total` uses the ceiling clip; `need_now`, `protect_tokens`, `fits_empty` and `fits_now`
  come from the non-mutating `CacheManager.probe_admission(input_ids, output_len)` of rule 2
  (tree-only walk, no park lookup, no restore, no lock), which shares the pure
  `admission_fits(need_now, reserved, available, protect_tokens)` with `_try_admit`; the
  controller adds its `uncommitted_tokens` to `reserved`. `_try_admit` gains the escalation hook
  of rule 2(b) and clears the uid's `uncommitted_tokens` entry when it charges the real
  reservation.
- Idle point: in both `overlap_loop` and `normal_loop`, where `_execute_pending_rebuild` is
  gated today, add: when idle and `_pending_rebuild is None`, ask the controller for a plan; if
  one comes back, set `_pending_rebuild = CacheRebuildBackendMsg(request_id=f"auto-kv:{instance_id}:{seq}", moe_cache_size=target_slots, num_pages=target_pages)` (unique per rule 9; `reason` and `target_pages` travel in `MaintenanceBeginMsg.detail` and the log line)
  and call `_execute_pending_rebuild()`; then act on the returned outcome as rule 8 says (admit
  on `ok` or `rejected`, error-reply and stop on `failed`). `_execute_pending_operation`,
  `_execute_pending_step` and `_execute_pending_rebuild` change contract to return the outcome
  string; the wire replies stay as they are. Before executing, send `MaintenanceBeginMsg` (rule 9).
  `blocking` for `receive_msg` must also be false while the FIFO is non-empty.
- `_execute_pending_operation` and `_execute_pending_step`: after any successful rebuild or step
  whose request_id does not start with `auto-kv:`, the engine re-snapshots `pool_budget_bytes`
  and the scheduler calls `controller.note_external_geometry(...)` so the planner reads it.
- `rebuild_cache`: the parking snapshot (`prepare_rebuild`) already runs for any `num_pages`
  change when `park_store` is set. No change.
- Finished requests: both completion paths must report. `_process_batch_result` assigns
  `self.finished_reqs = new_finished_reqs` (`scheduler.py:549`) and the MTP path
  `_speculative_decode_step` assigns `self.finished_reqs = finished_now` separately
  (`scheduler.py:1877`); aborts free requests on a third path. Centralise: one
  `_note_request_finished(reqs)` helper called from all three, which calls
  `controller.on_request_finished(now)`. A stale timestamp after an MTP request would delay or
  skip Timer 1. `running_need_tokens` for the concurrent rule is
  `sum(len(r.input_ids) + r.output_len)` over the prefill manager's pending list and the decode
  manager's running requests.
- Logging: one `INFO` line per plan (`KV pool grow 65536 -> 98304 tokens, slots 7200 -> 7043
  (held request 42, need 92160)`), one per shrink, one `WARNING` per capped plan.
- `/v1/cache/status`: add a `kv_dynamic` block: `enabled, floor_tokens, ceiling_tokens,
  step_tokens, pool_tokens, pool_budget_bytes, held, shrink_in_s, last_plan, operation_id`. The settings page status
  bar reads it.

### Engine (`engine/engine.py`)

- `rebuild_runtime_cache`: order pools so shrinks free before grows allocate. Today the MoE cache
  rebuilds before the KV pool, which is right for grow-KV/shrink-MoE and wrong for the Timer 1
  shrink-KV/grow-MoE. New order: if `num_pages is not None and num_pages < self.num_pages`, run
  `_resize_kv_pool` before `moe_offload_cache.rebuild`; otherwise keep today's order. The
  `validate_rebuild` fit check compares final totals against the boot budget; with this order the
  transient peak is `max(before, after)`. The 4ff86b3 allowance ("no larger than the resident
  total") is per-transition and would refuse legitimate controller moves on an over-budget card
  (131,072 -> 163,840 raises the resident total by 1.6 MB; the Timer 1 return to the floor by
  0.7 MB), so it is NOT the check to widen. Instead the validator and the planner share rule 3's
  `pool_budget_bytes`: the engine exposes it, and `validate_rebuild` accepts a MoE+KV target whose
  final total is `<= pool_budget_bytes` when every shrinking pool is resized before any growing
  one, in addition to the existing boot-budget check. `pool_budget_bytes` is never raised by the
  controller; only an external rebuild that itself passed validation re-snapshots it. Geometry
  limits (`num_experts` floor, marlin cap, GDN minimum, positive pages) still apply. Regression test
  on an over-budget fake account: both transitions above pass; a target above `pool_budget_bytes`
  is refused; a grow-MoE/shrink-KV request is refused unless ordered shrink-first. Test: a CPU test on a recording fake that asserts call order for
  the four sign combinations (`tests/engine/test_cache_rebuild.py`).
- `_resolve_auto_moe_cache_size`: unchanged; the launcher passes `--num-tokens floor` and
  `--kv-reserve-tokens floor` when the dynamic pool is on, so the planner already prices the boot
  pool correctly (the override-as-floor fix in 4ff86b3).
- `step_memory`: when `config.kv_dynamic`, rung 3 shrinks to `kv_dynamic_floor_pages` and the up
  KV rung is skipped; `step_memory_noop` treats the KV restore as never pending. `_initial_num_pages
  = floor_pages` at boot. Tests extend `tests/engine/test_cache_rebuild.py`'s step fakes.
- `EngineConfig`/`ServerArgs` (`server/args.py`): `kv_dynamic: bool = False`,
  `kv_floor_tokens: int = 65536`, `kv_step_tokens: int = 32768`, `kv_shrink_idle_s: int = 600`,
  `kv_park_ttl_s: int = 18000`. Flags `--kv-dynamic`, `--kv-floor-tokens`, `--kv-step-tokens`,
  `--kv-shrink-idle-s`, `--kv-park-ttl-s`. Validation in `args.py`: `--kv-dynamic` requires an
  offload-family MoE backend and `--num-tokens` (the ceiling) `>= --kv-floor-tokens`;
  `--kv-step-tokens >= 8192`; `--kv-park-ttl-s >= 0`. When `--kv-dynamic` is set, `args.py`
  records the ceiling as `kv_ceiling_tokens = num_token_override` (when `--num-tokens` is absent
  or 0, the ceiling is `max_seq_len + page_size`, never the auto card-filling size) and rewrites
  `num_token_override = kv_floor_tokens + page_size` (dummy page) and `kv_reserve_tokens =
  kv_floor_tokens` for the boot plan; `max_seq_len` still follows the ceiling so a long prompt is
  clipped against the ceiling, not the floor. The scheduler's controller reads `kv_ceiling_tokens`.

### Park store (`kvcache/park_store.py`)

- `ParkStore(..., ttl_s: int = 0)`; `sweep_expired(now_ns) -> int` evicts every family whose newest
  `last_used_ns` is older than `ttl_s`, using the family eviction already used by `_evict_to_fit`
  (queued-save reservations and pinned buffers stay charged until released, as today).
  `CacheManager.park_idle()` calls it once per idle tick, and `next_expiry_delay_ms()` (relative,
  wall-clock arithmetic inside the store, no scheduler monotonic timestamp passed in) feeds
  `CacheManager.next_park_delay_ms` so the tick happens (rule 5). `status()` gains `ttl_s`,
  `expired_evictions` and `next_expiry_s`. Test: `tests/kvcache/test_park_store.py` with a fake clock.

### Settings helper (`daemon/settings/`)

Dials (Model & chats group), all mapped in `linux_launch.build_launch`:

| Dial | Type | Default | Engine flag | Notes |
|---|---|---|---|---|
| `KVDynamic` | toggle | on | `--kv-dynamic` | plain "Dynamic KV memory"; off = today's fixed pool |
| `KVFloorTokens` | number | 65536 | `--kv-floor-tokens <N>` | slider 16384..262144 step 8192; must be `<= KVCacheTokens` |
| `KVCacheTokens` | existing | 262144 (262208 on the box) | `--num-tokens <N>` | help text gains: "with Dynamic KV memory on, this is the largest size the pool may grow to" |
| `KVStepTokens` | number | 32768 | `--kv-step-tokens <N>` | minimum 8192, slider 8192..131072 step 8192 |
| `KVShrinkIdleMin` | number | 10 | `--kv-shrink-idle-s <N*60>` | Timer 1, minutes, slider 1..120 |
| `KVParkTTLHours` | number | 5 | `--kv-park-ttl-s <N*3600>` | Timer 2, hours, slider 0..48, 0 = never; shown whenever `KVPark != off` |

`validate_settings` refuses `KVFloorTokens > KVCacheTokens` and `KVStepTokens < 8192` (422 with the
rule in the message, like the owned-layer slot floor). `memory_fit.estimate_settings` prices the
boot at `KVFloorTokens` when `KVDynamic` is on and adds a warning when
`(KVCacheTokens - KVFloorTokens) * bytes_per_token > (slots_at_boot - slot_floor) * slot_bytes`
("the largest KV size cannot be reached: slots would fall below N"). The status bar shows
`KV pool 65,536 of 262,208 (dynamic)` from the new status block. Tests: `tests/settings/
test_dials_metadata.py`, `test_linux_launch.py`, `test_memory_fit.py`.

Helper version bump and `docs/cli.md` rows for the five flags. README fork section: one paragraph.

## Out of scope

Growing while a request is mid-decode (needs the between-step rebuild under a live request, the
GPU-fault suspect). Preserving slot-cache contents across a resize (blocked by transient memory;
follow-up is CUDA virtual-memory reservation for both pools so nothing moves and graphs never
recapture). Funding growth from GPU-owned layers (`layer_moves`); the daily boot owns none.
TP > 1. DSV4. The Windows launcher. Sending to upstream.

## Risks

1. **Rebuild under an external VRAM grab.** Byte-neutral trades ask for nothing extra, but a game
   grabbing memory between the plan and the allocation can still fail the grow. Handled by the
   existing rollback; the request is admitted against the old pool.
2. **Held-request wait.** A request that needs more than the pool waits for every running request
   to finish. This is no worse than today for requests that would have waited on pages, and the
   `grow-concurrent` rule makes the wait happen at most once per pair of sessions. Watch the log
   for `held` durations in the live run.
3. **Cold slot cache after each resize.** 1-3 s of slower decode; measured in the live run as
   tok/s over the first 100 tokens after a grow.
4. **Timer 1 and parking budget.** A shrink parks everything eligible; if the 8 GiB RAM budget is
   full, the oldest family is evicted and that prefix is re-read later. Log the count.
5. **Graph recapture path.** Unchanged code, idle-only, the mode that has never faulted.
6. **MTP after a resize.** `_rearm_spec_graphs` re-arms the verify widths for lazy capture; it
   does not capture them. Capture after a large prefill can fail memory admission and leave
   decode eager (see the boot capture comments). The live run must confirm "width N: captured"
   in the log after a resize and MTP decode speed within noise of the pre-resize figure, as the
   19f061b check did.

## Live acceptance (on the box, when Jay is not using it)

1. Boot with `KVDynamic` on; ledger shows pool 65,536 and slots about 940 above the previous boot (calculated; record the real number).
2. Short chat (pi, about 7k prompt, 200 tokens): tok/s vs the previous boot's 200-token bench.
3. Open a 150k `/v1/messages` conversation: log shows one `auto-kv:grow` rebuild, first token
   delay under 5 s beyond prefill, `cached_tokens` non-zero on the next turn (RAM restore hit).
4. Second conversation while the first decodes: `grow-concurrent` plan logged; both run together
   afterwards.
5. Stop both; after 10 min the log shows `auto-kv:shrink` and the status block shows the floor;
   `nvidia-smi` free memory unchanged (byte-neutral), slots back to baseline.
6. Resume the first conversation: restore hit, first token within about 10 s.
7. Zero failed requests through the whole run; `/health` never leaves `serving` except for the
   sub-second `rebuilding` windows.
8. Governor squeeze (grab 4 GB on the card) during a grown pool: slots step down,
   `pool_budget_bytes` follows, a later shrink does not undo the governor's step.
9. Starvation: a 150k request arrives while 20 short requests keep coming; the short ones queue
   behind it, the grow happens, all complete in arrival order.
10. Failure outcomes (CPU tests with the recording fake engine, plus one live forced rejection via
    an impossible target): rejected -> admitted against the old pool; rolled back -> same; failed
    -> held requests get error replies, nothing is scheduled, frontend `failed`.
11. Automatic rebuild overlapping a governor step and a manual `/v1/cache/rebuild`: one operation
    at a time, operation ids unique and `/v1/cache/status` consistent, `pool_budget_bytes`
    re-snapshotted; plus the delayed-delivery race test of rule 9.
16. Admission probe: zero restores and zero page allocations across a run of probes over locked,
    evictable, partially shared and parked prefixes; the probe's verdict equals `_try_admit`'s on
    each, including the 98,304-token fully-evictable prefix with 16,384 tokens of demand that the
    pre-lock check would wrongly accept; all quantities in tokens (a page-count regression is
    asserted against).
19. Same-batch arrivals test of rule 2 (two 39,936-token requests in one batch, 65,536 pool).
17. Over-budget validator: the two transitions named in the engine section pass; a target above the
    budget is refused.
18. Two-clock TTL test as in rule 5.
12. RAM TTL with no traffic at all (dynamic on, then off): the family expires and pinned RAM drops.
13. Shared-prefix pair: two requests sharing a 100k prefix admitted without a grow when the
    increment fits.
14. MTP on: after a resize the log shows the verify widths captured and MTP decode speed within
    noise; Timer 1 fires after an MTP-completed request.
15. Performance is reported as three numbers per case, not one: time to first token, total
    request time, warm decode tok/s; short-chat tok/s alone does not establish the win.
