# FreeToken Sleep: design

Date: 2026-09-25. Branch `feat/freetoken-sleep` (from `mtp-upstream-merge` at fedf802).
Author: Claude (Opus 5.5), on Jay's delegation ("free the graphics card for games, but keep
FreeToken in memory so it wakes up faster than a full load"). Jay delegated every choice below.

## 1. Goal

**In words Jay would use:** a **Sleep** button on the control panel. Sleep empties the graphics
card so a game can use it. The model stays loaded in the PC's memory. The next chat wakes it
in about half a minute. A cold load takes about two and a half minutes.

**Measurable:**

| | today (cold boot) | Sleep phase 1 (this plan) | Sleep phase 2 (later) |
|---|---|---|---|
| time to answer again | 149-161.5 s [a] | wake ≤ 45 s, aim ≤ 30 s | wake ≤ 50 s |
| card used while "put away" | 1.8-2.4 GB after Unload [b] | ≤ desktop baseline + 7.5 GiB (about 9.5 GB whole card) | ≤ baseline + 1.5 GiB |
| Windows RAM held while away | 0 after Unload | unchanged, about 58-61 GB (the point of sleeping) | unchanged |
| output after wake | n/a | greedy tokens identical to before sleep | identical |

[a] `docs/research/control-panel-a-acceptance-2026-09-25.md:19` ("loaded in 149 s");
`docs/research/own-switcher-acceptance-2026-09-24.md:16` ("answered in 161.5 s").
[b] `own-switcher-acceptance-2026-09-24.md:15` ("card back to 1.8 GB"), control-panel acceptance row 9 ("2.4 GB").

## 2. Research answers

### 2.1 What is on the card in a serving FreeToken process, and what can be released in-process

The Qwen3.8-Flash-Next NVFP4 boot on the 5090 (int8 dense, 262k fp8 KV, `auto:8` owned layers,
MTP optional) holds these on the card:

| Item | Size (measured) | Where it is made | Freeable in-process? | Rebuilt by |
|---|---|---|---|---|
| Dense weights (attention, GDN, hyper-connections, shared expert, router, `lm_head`; int8) | 5.28 GiB ledger "weights" [c] | `engine.py:436` `_install_model_weights`; loaded straight to CUDA with no host copy (`memory-audit-qwen38-rtx5090.md` §2, `qwen4_exp/weight.py:180-183`) | Yes in principle (phase 2): nothing else holds them. A host or SSD copy has to be made first. | phase 2: read back into the same tensors |
| GPU-owned expert layers | 1.32 GiB each (8 layers = 10.6 GiB) | `offload_cache.resident_banks`; owned layers have **no host bank** (`engine.py:1450-1623` docstring; run2 doc line 144: "no host bank") | Yes: `Engine._move_layer(l, "disk")` exists (`engine.py:1495-1519`) and needs only the SSD expert copy | `_move_layer(l, "gpu_owned")` from disk (`engine.py:1560-1585`) |
| Shared expert slot cache | 2,772,480 B per slot [d]; 5-7k slots = 13-19 GiB | `offload_cache.py:763-776` (`bank_caches`) | Yes, but `rebuild()` has a floor of `num_experts` slots (`offload_cache.py:736`) and 1,024 with prefill overlap | new `release_slots()`, then `rebuild(size)` |
| MoE bookkeeping (`slot_for_id`, `usage`, stats, `decode_freq`) | a few MiB | `offload_cache.py:188-262` | Not worth it | none |
| KV pool | 3.24 GiB at 262,208 fp8 tokens [e] | `engine.py:516` | Yes: `_resize_kv_pool` (`engine.py:1423`); minimum is 1 page (`kvcache/base.py:94-96`) | `_resize_kv_pool(config, pages)` |
| GDN linear-state pool | 0.86-0.97 GiB [c][f] | `engine.py:532`, `linear_state_pool.py:66-89` | Yes: `LinearStatePool.rebuild(n)` (`linear_state_pool.py:144-176`) | `rebuild(n)` |
| Page table | small | `engine.py:558`, re-made by `_refresh_seq_state` (`engine.py:1434`) | follows KV | `_refresh_seq_state` |
| Decode CUDA graphs, their private pool and capture buffers | part of "unaccounted" 3.03 GiB [c] | `graph.py:215` | Yes: `destroy_cuda_graphs` (`graph.py:258-267`) | a new `GraphRunner` (`engine.py:2430-2462`) |
| MTP draft head (when MTP is on) | 2.17-2.56 GiB resident [g] | `engine.py:635` `SpecDraftHead` | Yes: drop the object (`spec_draft.py:1425` `close()` frees graphs; the weights go with the reference) | `SpecDraftHead(engine, config.spec_decode)` + `_capture_spec_graphs_at_boot` |
| Spec verify graphs, ladder replays | small | `spec_graph.py:815`, `spec_state_ladder.py:175` | Yes: `destroy()` / `rebind()` | `_rearm_spec_graphs`, boot capture |
| Allocator cache | varies | torch caching allocator with expandable segments (`engine.py:3067`) | Yes: `torch.cuda.empty_cache()` (called by `_sync_get_memory`, `engine.py:1347`) | n/a |
| cuBLAS / Triton / flashinfer workspaces | tens to hundreds of MiB | library-owned | Partly (`torch._C._cuda_clearCublasWorkspaces`) | on first use |
| **CUDA context + loaded kernel modules** | about 0.5-1 GiB | the process | **No, not without exiting the process** | n/a |

[c] boot C ledger, `measurements-gpu-owned-followups-live-2026-09-02.md:122-131`.
[d] `memory-audit-qwen38-rtx5090.md` table row 1; CLAUDE.md: "1,000 slots ≈ 2.58 GiB".
[e] memory `project-262k-context-settings`: "KV 3.24 GiB" at 262,208 tokens.
[f] audit: 8 slots × 115,642,376 B.
[g] `measurements-gpu-owned-followups-live-2026-09-02.md:63` (2.17 GiB), memory
`project-262k-context-settings` (2.56 GiB).

The PLE table (47.7 GiB) is memory-mapped on the host (`qwen4_exp/weight.py:426-441`). The token
embedding is host-only (`FREETOKEN_EMBED_HOST=1`). Vision weights are pageable CPU and streamed
through a transient workspace that `vision.py:286-309` empties. None of these are on the card
at rest.

**Conclusion:** everything on the card except the dense weights and the CUDA context can be
released and rebuilt with code that already exists and is used live by the memory governor.
The dense weights (5.28 GiB int8, about 9 GiB bf16) are the only large item that needs new
machinery.

### 2.2 Existing mechanisms, and how close they are to Sleep

| Mechanism | Where | What it gives Sleep | Gap |
|---|---|---|---|
| Layer demotion `gpu_owned -> disk` / `disk -> gpu_owned` | `engine.py:1450-1623` | Frees 1.32 GiB per owned layer with **no host RAM cost**, and restores it byte-identically (`tests/moe/test_disk_banks.py:434`) | Needs the SSD expert copy complete. It is on by default for NVFP4 (`engine.py:1242-1276`) and written once in 213 s at the first boot (`memory-governor-live-2026-09-07.md:34`) |
| `gpu_owned -> pinned` (the VRAM ladder) | same | Would free VRAM too | Costs 1.32 GiB of **Windows RAM** per layer. With 8 layers that is +10.6 GB, which this box does not have: Windows free fell to 2.7 GB during a boot (control-panel acceptance line 49). **Rejected** |
| Slot cache `rebuild(size)` | `offload_cache.py:745-812` | Frees and re-makes the slot cache in place, keeping object identity | Floor `num_experts` (1.32 GiB), or 2.64 GiB with overlap. Sleep adds `release_slots()` (size 0) |
| KV / GDN pool rebuild | `engine.py:2156-2215` | In place, identity-preserving | The scheduler must park prefixes first and re-thread its page managers after (`scheduler.py:318-390`) |
| KV parking | `scheduler/cache.py:1146-1152` `prepare_rebuild` | Parks every eligible conversation to RAM before the KV pool is freed, so **chats survive a sleep** and restore on their next turn | Bounded by the park store budget (8 GiB on the box) |
| `rebuild_runtime_cache` | `engine.py:2217-2468` | The whole teardown → move → resize → recapture sequence, with progress reports and rollback | Always recaptures graphs at the end. Its budget check refuses tiny targets. Sleep reuses its parts, not the whole function |
| Graph teardown + recapture | `engine.py:2395-2468`, 1-4 s measured [h] | Exactly what wake needs | The capture branch is inlined; Task 2 extracts it unchanged |
| Maintenance gate ("rebuilding") with progress-per-unit stuck clock | `api_server.py:77-96`, `173-212` | Requests wait instead of failing; the watchdog sees progress | Needs a "sleeping" state that is not "serving" to the helper and is ready to llama-swap |
| Dynamic KV pool controller | `scheduler/kv_dynamic.py`, `scheduler.py:1162-1246` | n/a | Must be paused while asleep, or it would try to grow the pool |
| `prepare_stop` / helper stop | `server/api_server.py`, `daemon/settings/process_manager.py:334-386` | Full unload stays one click | none |

[h] `dynamic-kv-pool-live-2026-09-12.md:37` ("Grow rebuild: 1 s (warm geometry) to 4 s").

"Demote every owned layer to the SSD + slot cache to 0 + KV to 1 page + GDN to its padding slot +
drop graphs + drop the MTP head + empty the cache" **is** a sleep that keeps the dense weights on
the card (phase 1). The reverse, in the order below, is a wake. The only new engine primitives are
`OffloadMoeCache.release_slots()` and a small `engine/sleep.py` that composes the existing steps
in a safe order and remembers what to restore.

### 2.3 Alternative: process-level sleep (a "bank keeper" that outlives the server)

The idea is that a keeper process holds the 53 GiB of expert banks in shared memory. The server
exits to free the card, and a new server attaches to the banks.

- **CUDA IPC cannot carry them.** Legacy CUDA IPC shares *device* allocations. On WSL it works
  only from driver R510 ([NVIDIA CUDA on WSL guide, "Known Limitations"](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)),
  and never for host memory. The banks are pinned *host* memory (`cudaHostAlloc` with
  `BANK_CUDA_ALLOC=1`, or `mmap` + `cudaHostRegister`, `host_banks.py:108-142`). The new process
  would have to map a POSIX shm segment and `cudaHostRegister` 53 GiB again. The same guide
  lists pinned system memory as "limited" on WSL.
- **Most of the boot still happens.** A new process pays CUDA init, the dense load (about
  9.2 GiB bf16 read, then int8 quantize: audit F8), PLE setup, KV allocation, graph capture,
  the MTP head, warmup and prompt-cache start-up. On native Windows the expert load was 43 s of
  a 73 s boot (`measurements-unbuffered-boot-2026-09-02.md`, headline table). On WSL the boot is
  149-161 s, so a restart that skips only the expert read would still take roughly 90-120 s.
  That is 3-4× slower than an in-process wake.
- **Risk.** A crash leaves a 53 GiB orphan segment on a box that already runs Windows down to
  2.7 GB free. On 2026-09-12 Windows memory pressure took the WSL disk down
  (memory `project-wsl-disk-quota-outage`). It would also need a new supervised daemon.
- vLLM's sleep mode is the reference design, and it is in-process: "level 1" offloads weights to
  CPU and discards KV; "level 2" discards both ([vLLM docs, Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)).
  Its CuMem allocator keeps virtual addresses so graphs survive. We do not need that here,
  because recapture costs 1-4 s [h].

**Decision: in-process sleep.** The process-level design is rejected. It would be 3-4× slower
to wake, cost more to build, and add a new way to crash the box.

### 2.4 Expected wake time and VRAM after sleep

**Where a cold boot goes.** The only phase breakdown on record is native Windows: 73.1 s to
serving, of which 43 s was the expert-bank read and pin (unbuffered-boot doc). WSL boots take
149-161.5 s, and the phases have not been broken down there. Task 12 records a baseline from the
server log timestamps. Sleep keeps every boot phase except the ones in the wake list below.

**Wake estimate (phase 1):**

| Step | Estimate | Evidence |
|---|---|---|
| Wake request → safe point | < 1 s | idle by construction |
| GDN + KV + slot cache allocation | < 1 s | allocations only |
| 8 owned layers SSD → card, 10.6 GiB | 5-15 s | 64-row synchronous chunked reads (`engine.py:1569-1583`). Throughput on the box is **unmeasured**; `tests/moe/test_disk_banks.py:538` is the bench. Assumed 1-2 GB/s from the NVMe vhdx |
| Decode graph capture | 1-3 s | runtime captures 0-3 s (`api_server.py:86`); boot capture 9 s |
| MTP head rebuild + spec/draft graphs (MTP on only) | 5-13 s | graphs 2.5 + 0.65 s (`measurements-gpu-owned-layers-run2-2026-09-02.md:144`); the head read from the private root is unmeasured (est. 3-10 s) |
| Wake self-check (two short prefills) | < 1 s | `engine.py:2958` `_warmup_prefill` |
| **Total** | **10-20 s MTP off, 15-33 s MTP on** | criterion ≤ 45 s |

The first chat after a wake also restores its parked conversation, as it does today after a
dynamic-pool shrink. The longest measured restore was 26.1 s for a 66k-token conversation
(`api_server.py:86`). That time is the chat's own, not the wake's.

**Sleep time:** parking eligible prefixes takes up to about 10 s ("shrink with six parked
prefixes: 10 s", dynamic-kv-pool-live line 37). Freeing everything takes about 1-2 s.
Criterion: ≤ 20 s.

**VRAM left after phase 1:** dense 5.28 GiB + CUDA context and modules about 0.5-1 GiB +
residue, so 6-6.5 GiB for FreeToken. The whole card would read about 8-9 GB against 1.8-2.4 GB
with FreeToken stopped. That leaves **about 23 GB of the 32 GB card for a game**. Phase 2
takes FreeToken down to about 1 GiB (context floor).

### 2.5 How Sleep fits the server, helper, switcher and panel

- **Server API:** `POST /v1/sleep` and `POST /v1/wake` (under `/v1` like `/v1/cache/*`; vLLM
  uses `/sleep` and `/wake_up`). `/health`, `/ready` and `/v1/cache/status` report
  `"sleeping"`. **A chat request to a sleeping server wakes it** and waits (up to 300 s), the
  way requests already wait out a rebuild (`api_server.py:470-497`).
- **llama-swap sees a sleeping FreeToken as loaded ("ready").** Nothing in the frozen switcher
  changes: no new patch. Chats proxy to :2020 and wake it. Loading any other model makes
  llama-swap unload FreeToken fully (SIGTERM → `freetoken.sh` `on_term` → helper Stop). That is
  Jay's rule, "switching models while FreeToken sleeps: unload it fully first". `ninfer.sh`
  already stops any FreeToken whose state is not `unreachable` (`engines/adapters/ninfer.sh:35-47`),
  so the RAM the P2 memory gate waits for is really returned. The idle TTL ("Unload when idle")
  still fully unloads a sleeping model.
- **Helper:** `/api/server/sleep` and `/api/server/wake` proxy to the server. The crash
  watchdog treats `sleeping` as alive (`watchdog.py:158-177`). The governor does not step the
  card while asleep. It runs only the RAM axis's down rung, which lands on the SSD because the
  slot cache is empty (`step_memory` RAM ladder, engine.py:1660 onward). A game that squeezes
  Windows memory therefore still gets expert layers out of its way.
- **Control panel:** a FreeToken row that is loaded shows **Sleep** (or **Wake** when asleep)
  next to **Unload**. The status word becomes "Asleep (graphics card free)".
- **Adapter:** `freetoken.sh` adopts a sleeping server as "already running this model"
  instead of rebooting it.

## 3. Design

### 3.1 Decisions

| # | Decision | Why |
|---|---|---|
| D1 | In-process sleep | §2.3 |
| D2 | Owned (and RAM-parked) layers go to the **SSD copy**, not pinned RAM | +10.6 GB Windows RAM is not available; the SSD copy already exists and the move is proven live |
| D3 | KV: park, then shrink to the pool's minimum (1 page); GDN: shrink to the padding slot (+1 for the MTP ladder) | Conversations survive via parking; the budget check is bypassed because we only shrink |
| D4 | MTP draft head dropped, rebuilt on wake | 2.2-2.6 GiB; no host copy exists; the constructor is the tested path |
| D5 | Dense weights stay on the card in phase 1 | Freeing them needs a pointer-stability audit (phase 2); phase 1 alone frees about 23 GB |
| D6 | Auto-wake on a chat; wake refuses **before touching anything** when the card lacks room (a game is running), with a plain message; the server stays asleep | Jay never sees a crash from a busy card |
| D7 | Wake failure after allocation starts → release back to the sleep geometry → "rejected, still asleep". Only if that also fails → latch failed (watchdog restarts) | Same rule-8 shape as rebuilds (`scheduler.py:1846-1927`) |
| D8 | Governor while asleep: RAM-down only (SSD spills); never VRAM, never recall | The card belongs to the game; Windows RAM pressure has killed the WSL disk before |
| D9 | No new llama-swap patch | §2.5 |
| D10 | Phase 1 has manual Sleep/Wake plus auto-wake; auto-sleep is a follow-up | Keep scope bounded; measure first |
| D11 | Sleep is refused while a chat runs ("busy") and while the MTP shadow observer runs | Idle-only like KV rebuilds |
| D12 | Models without a complete SSD copy (EXL3, a first boot still writing it) refuse sleep with a plain reason | No silent RAM spike |

### 3.2 Sleep order (engine, at the scheduler's idle safe point)

Scheduler first: refuse if anything is pending, running or held. Then run the prefix
coordinator's `before_rebuild()` and `cache_manager.prepare_rebuild()` (park every eligible
prefix, `cache.py:1146`).

Engine (`engine/sleep.py: sleep_engine`):

1. **Preflight, nothing freed:** refuse if already asleep (answer ok), if the shadow observer
   is on, or if any owned layer lacks a complete SSD copy. Measure `free_before`.
2. Remember the decode graph sizes (`_graph_bs_for_recapture`, extracted from
   `rebuild_runtime_cache` step 0). Set `rebuild_teardown_started = True`.
3. Destroy the spec verify graphs, `attn_backend.reset_capture()`, and `graph_runner.destroy_cuda_graphs()`.
4. Close and drop the MTP draft head.
5. Each owned layer: `_move_layer(l, "disk")`. This suspends prefill overlap, drops the device
   banks and appends to `_ram_spilled_layers`.
6. `moe_offload_cache.release_slots()`.
7. KV → minimum pages; GDN → 1 slot (+1 with the ladder, then `ladder.rebind()`);
   `_refresh_seq_state`.
8. `gc.collect()`, clear the cuBLAS workspaces, `empty_cache` (inside `_sync_get_memory`).
   Measure `free_after`. Store a `SleepSnapshot` (sizes, owned list, RAM-parked list, graph
   sizes, had-draft, free before/after).

Scheduler after: re-thread `cache_manager` / `table_manager` / `token_pool` against the 1-page
pool (as `rebuild_cache` does, `scheduler.py:367-381`), pause the dynamic KV controller, parks
and integrity checks, and reply.

### 3.3 Wake order

1. **Preflight, nothing allocated:** `free_now ≥ released_bytes + 512 MiB`, else
   `SleepRefused("the graphics card has X GB free and waking needs Y GB …")`.
2. GDN → snapshot slots (+ ladder rebind); KV → snapshot pages; `_refresh_seq_state`.
3. Slot cache `rebuild(snapshot size)` **before** layers come home. Resuming prefill overlap
   needs the slot cache (`engine.py:367-379`, `offload_cache.py:1012-1033`).
4. Each snapshot owned layer still on disk: `_move_layer(l, "gpu_owned")`. Then restore
   `_ram_parked_layers` (a move removes them, `engine.py:1620-1623`).
5. `_recapture_graphs(config, graph sizes, free)`. If a layer was spilled to the SSD during
   sleep (D8), capture is deferred exactly as today, and the governor recalls it later.
6. MTP on: rebuild `SpecDraftHead`, `_rearm_spec_graphs()`, `_capture_spec_graphs_at_boot()`.
7. Self-check: `_warmup_prefill()` (two short prefills). A sticky CUDA fault surfaces here as
   a wake failure, not in Jay's chat.
8. `snapshot_pool_budget()`; clear the snapshot. The scheduler re-threads the managers, refreshes
   the dynamic-KV policy, and admits held chats.

Failure handling follows D7.

### 3.4 Surfaces

- **Messages:** `CacheSleepMsg` (api → tokenizer), `CacheSleepBackendMsg` (→ scheduler),
  `CacheSleepResultMsg` (scheduler → detokenizer) and `CacheSleepReply` (→ api). Fields:
  `request_id, action ("sleep"|"wake"), status ("ok"|"rejected"|"busy"|"unsupported"|"failed"),
  asleep, released_bytes, vram_free_bytes, elapsed_s, error`.
- **Scheduler:** queues through the existing one-operation slot (`_pending_rebuild`,
  `_queue_maintenance`). A `UserMsg` that reaches a sleeping scheduler (a race) is **held**,
  and an auto-wake (`auto-wake:<n>`, `MaintenanceBeginMsg kind="wake"`) is queued. Held chats
  are admitted after the wake, or error-replied if the wake is refused. An abort removes a held
  chat. Prefix-cache commands and manual rebuilds are refused while asleep. A governor step is
  accepted only as `ram/down`, executed with a sleep-safe `rebuild` that only spills to the SSD.
- **API:** a new `FrontendManager.asleep` flag. The public state is
  `"sleeping" if asleep and maintenance_state == "serving"`. `ensure_awake()` runs one shared
  wake task, and every chat route already goes through `wait_until_serving` / `new_user`.
  `/ready` counts `sleeping` as ready.
- **Helper:** `ProcessManager.sleep_server(action)`, routes `/api/server/sleep|wake`. The
  watchdog adopts `sleeping`. The governor handles the asleep tick (D8). The reclaim controller
  runs while sleeping.
- **Panel:** `models()` rows gain `"sleep": "asleep"|"awake"|null` for the loaded FreeToken row.
  New routes `/api/panel/models/{id}/sleep|wake` and buttons.
- **Adapter:** `freetoken.sh` adopts `sleeping`.

## 4. Phases

- **Phase 1 (this plan):** §3 in full. Dense weights stay on the card.
- **Phase 2 (a separate plan, gated on phase 1 numbers):** dense weights → an SSD sleep file.
  For every CUDA tensor in `model.state_dict()`, deduplicated by storage: write the bytes, then
  `untyped_storage().resize_(0)`. Wake does `resize_(nbytes)`, reads the file back through
  pinned staging, and copies H2D. Graphs are recaptured anyway. **Gate:** an audit of every
  place that caches a raw weight pointer outside a CUDA graph (int8 linear descriptors, the
  spec `lm_head`, flashinfer plans, PLE dense), plus a GPU greedy test. It goes to the SSD, not
  RAM, for the same reason as D2. It adds about 4 s to a wake (5.3 GiB at about 1.5 GB/s).
- **Follow-ups (not planned):** "Sleep instead of unload when idle" per model; "sleep when a
  game starts" (nvidia-smi process watch); pipelined multi-layer SSD reads on wake.

## 5. Risks

| Risk | Mitigation |
|---|---|
| **Stale pointers / invalid graphs** after pools move | Every graph is destroyed and recaptured. Fused-copy tables are rebuilt by `rebuild()`, overlap views torn down, ladder replays dropped by `rebind()`, `attach_page_table` re-points. EXL3 (packed pointer tables) cannot sleep (no SSD copy, D12). The greedy-equality GPU test pins it |
| **The long-context GPU fault class** (MMU NACK after teardown + recapture, memory `project-long-context-gpu-fault`) | Wake runs at idle only, at human frequency (not the governor's 6,614 steps a day), and ends with a self-check prefill. Acceptance watches the Windows System log for nvlddmkm event 153 |
| **Governor interplay** | One operation slot. The helper sends only ram/down while asleep and the engine refuses anything else. Sleep and wake re-read geometry (`snapshot_pool_budget`) |
| **MTP** | Head rebuilt with the boot constructor; ladder rebind; spec graphs captured at wake like at boot (a width that fails stays lazy, today's behaviour). Acceptance runs both MTP on and off |
| **A game takes VRAM between the wake check and the allocation** | D7: release back to asleep, reply "rejected" with a plain reason |
| **Windows RAM while a game runs** (FreeToken keeps about 58-61 GB) | D8 SSD spills; wake does not recall them (the governor does later) |
| **Freed VRAM not returned to Windows by WSL/WDDM** | Measured in acceptance on the Windows side (Task Manager / nvidia-smi.exe) |
| **Stop while a game holds the card**: `stop_servers` waits for whole-card VRAM < 3 GiB for up to 120 s (`long-context-gpu-fault-2026-09-08.md:75-77`) | Known, unchanged; noted in acceptance |
| **Park budget**: conversations beyond the park store are recomputed on their next turn | Unchanged behaviour of a KV shrink; noted on the page's help text |

## 6. Test plan

- **Devbox (CPU, run by implementers):** `release_slots` unit tests. Engine sleep/wake on the
  existing `FakeDiskEngine` harness (real `OffloadMoeCache`, real `ExpertDiskCopy`, real
  `_move_layer`): byte-identical owned banks after a round trip, pools restored, parked lists
  restored, refusal before teardown, rollback on a failed wake, SSD spill while asleep.
  Also: message wire round-trips, scheduler shells (busy, held chats, auto-wake, abort, refused
  commands), API state/gate/auto-wake, helper routes, watchdog, governor asleep tick, adapter
  adopt, panel routes and page words.
- **Box GPU tests (controller):** `tests/engine/test_sleep_gpu.py`. It checks that CUDA
  `release_slots` returns the bytes to `mem_get_info`, and that a round trip keeps a CUDA
  `OffloadMoeCache` usable (`needs CUDA`). Also the full existing suite `-m "not slow"`.
- **Live acceptance on the box (controller, up to 3 FreeToken boots, each announced, Windows
  free RAM watched):** `scripts/sleep_bench.py` runs 3 cycles each with MTP off and MTP on:
  greedy output before == after, sleep s, wake s, card used asleep. It also covers auto-wake by
  chat, wake refused while a synthetic VRAM hog holds the card, a SSD spill while asleep under
  a Windows RAM grab, switching to QUASAR while asleep (FreeToken fully unloaded), and panel
  screenshots (awake, asleep, waking) with `~/.npm-global/bin/chrome-devtools-axi`.
  Results go to `docs/research/freetoken-sleep-acceptance-<date>.md`.
