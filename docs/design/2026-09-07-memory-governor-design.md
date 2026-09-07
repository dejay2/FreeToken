# Memory governor: step expert layers down and up so the server never dies

Date: 2026-09-07. Status: approved job card (Jay), design by the Fable lead; builders Gemini Flash 3.8, reviews Fable 5.1.

## Goal

The Qwen3.8-Flash-Next server in WSL keeps a cushion of free memory on the card and in Windows
main memory. When a game or another program eats into a cushion, the server steps MoE expert
layers down a ladder and shrinks its pools; when memory comes back it steps them up again. It
never stops serving and never answers an error during a move: requests wait a few seconds in a
queue while a rebuild runs. At the bottom of the ladder the model decodes at roughly one to two
tokens a second with every expert coming from the SSD. Feature off = today's code path.

Decisions fixed by the job card (do not relitigate here): automatic, no switch; cushion defaults
1.5 GB card / 4 GB RAM, editable on the settings page; card floor about 6 GB (dense int8 weights
+ minimum slot shelf + small context), below the floor the model keeps going; no pausing, queue
instead of 503; automatic step-up with a hold-off; Qwen in WSL only.

## What was measured (2026-09-07, serving box, daily 262k boot)

| Item | Command | Result |
|---|---|---|
| Card use with the daily boot | `nvidia-smi --query-gpu=memory.used,memory.total,memory.free` | 31,591 / 32,607 MiB used, **597 MiB free**. The daily boot already violates a 1.5 GB cushion: `MoECacheSize 0` (auto) fills the card. |
| Windows main memory | `Get-CimInstance Win32_OperatingSystem` | 100.2 GB total, **8.1 GB free** with the server up. `.wslconfig` caps the VM at 88 GB. |
| VM memory | `/proc/meminfo` | 90.6 GB total, 16.9 GB available. |
| SSD holding `~/models` | `df -h`; `findmnt` | `/dev/sdd` ext4, 1 TB, **569 GB free**. The 63 GB expert copy fits. |
| SSD sequential read | `dd if=<shard> bs=8M count=128 iflag=direct` | 1 GiB in 0.256 s = **4.2 GB/s** (O_DIRECT, page cache bypassed). |
| Card-grab behaviour (does a native Windows allocation crash the server?) | not run | Jay was using the server. Runs in L1 before the governor defaults are frozen. |
| Pinned bank free returns memory to Windows? | not run | Same; needs a CUDA context and the card had 597 MiB free. Runs in L1. |
| Decode speed | timed 200-token chat | 5.9 tok/s, **not a baseline**: the card was busy with Jay's own chat (3D engine 93 % on the WSL VM process). Prior page-booted number on this box: 83.5 tok/s 8k-chat (2026-09-05). |

Arithmetic used below (from `docs/research/memory-audit-qwen38-rtx5090.md`): 48 MoE layers,
512 experts, top-10; one nvfp4 expert row = 2.77 MB across the three banks, so one layer =
1.33 GiB and 512 slots = 1.33 GiB of VRAM. Dense int8 always-on weights about 4.1 GiB.

## The ladder

Per MoE layer, one of three rungs (`HostResidency`, `host_banks.py`):

| Rung | Where the 512 experts live | VRAM | Host RAM | How decode reads it |
|---|---|---|---|---|
| `gpu_owned` | device tensors, `resident_banks[layer]` | 1.33 GiB | 0 | raw ids, `resident_views` (exists) |
| `pinned` | born-pinned host bank (`cudaHostAlloc`) | 0 (rows pass through the LRU slot cache) | 1.33 GiB | slot cache + fused copy (exists) |
| `disk` (new) | the on-disk expert copy | 28 MB scratch | 177 MB shared staging | per-step gather from disk, no slot cache (new) |

Pool rungs on the card, independent of layers: the LRU slot cache (512 slots per step,
floor 1024 with prefill overlap on, 512 with it off), and the KV page pool (idle-only, last rung).

### Why a DISK layer bypasses the slot cache

The fused copy kernel indexes a layer's host bank by expert id (`_copy_src_ptrs[layer] +
expert_id * feat_bytes`, `offload_cache.py:~675`). A disk layer has no such bank, and grafting a
per-row staging address into that table would mean a second LRU on the host. Instead a DISK
layer decodes exactly like a GPU-owned layer but on a tiny device buffer: gather the unique routed
rows (at most `top_k x batch`, 10 for the daily single-chat boot) from the disk copy into pinned
staging, copy them to a `[top_k x max_bs, ...]` device buffer per bank (28 MB), remap `topk_ids`
to 0..n-1, and call `_expert_gemm(views=...)` with position == remapped id (the owned-layer
branch at `layers/moe.py:729`). No LRU, no pointer tables, no slot accounting.

Cost per step per DISK layer: 10 rows x 2.77 MB = 28 MB. At the measured 4.2 GB/s sequential
that is 7 ms; ten random 2.77 MB reads through a small thread pool are budgeted at 15 ms. With
all 48 layers on disk: about 0.7 s per token, 1.4 tok/s. With 12 layers on disk: 0.18 s plus
the eager decode of the rest, roughly 4 tok/s. This is the "1 tok/s or whatever" Jay accepted.

Decode runs **eager, no CUDA graph, while any layer is DISK** (`GraphRunner.can_use_cuda_graph`
returns False when `cache.has_disk_layers`): the per-layer disk gather needs a host sync
(`topk_ids` D2H) that a captured graph cannot contain. The CPU executor's host-node trick
(`cpu_executor.py`) would keep graphs, but hybrid decode hangs under WSL (measured 2026-09-05),
so it is not used here.

Prefill of a DISK layer reuses the existing whole-layer branch for unpinned layers
(`copy_missing`'s pageable path, position == expert id in slots `[0, 512)`): a
`disk_materialize_layer` fills those slots in 64-row chunks through the staging buffer
(1.33 GiB per layer per chunk, about 0.4 s at 3+ GB/s). Prefill overlap is disabled while any
layer is DISK, exactly as it is for LOCKED/PAGEABLE layers today.

### The disk copy

Written once, in the background, after the banks are loaded: one raw file per layer per bank,
`<model>/freetoken-expert-cache/<expert_quant>-h<hidden>-i<inter>/L<layer>.<bank>.bin`, rows in
bank order, row length = `feat_bytes` of that bank. `manifest.json` records the checkpoint
identity (shard names, sizes, mtimes), the bank shapes and dtypes, and a per-layer `complete`
flag. A layer may be spilled only when its files are complete and the manifest matches. Owned
layers are written from their device tensors in 64-row D2H chunks; pinned layers straight from
the bank. 63 GB at the SSD's write speed is one to two minutes of low-priority I/O per fresh
checkpoint; nothing is written when the manifest already matches.

The bank layout is the loader's repacked row layout, not the checkpoint's, so a row is a plain
`pread` at `expert_id * feat_bytes`. Reads use a small thread pool with `os.pread` (O_DIRECT
optional; the page cache would otherwise compete with the VM's memory, which is the thing we are
trying to give back).

### Moves

All moves run inside `Engine.rebuild_runtime_cache` (`engine.py:1248`), which already tears
down CUDA graphs and the spec graphs, resizes the slot cache in place, and recaptures. A new
argument `layer_moves: list[tuple[int, str]]` applies before the slot cache rebuild:

| Move | Work | Bytes moved | Expected time |
|---|---|---|---|
| gpu_owned -> pinned | alloc host bank for the layer, D2H copy, free device tensors, rebind | 1.33 GiB D2H | 0.1 s |
| gpu_owned -> disk | check manifest complete, free device tensors, rebind | 0 | ms |
| pinned -> disk | drop the layer's `HostBank`s (cudaFreeHost through the extension deleter), rebind | 0 | ms |
| disk -> pinned | alloc host bank, read from disk copy, rebind | 1.33 GiB read | 0.4 s |
| pinned -> gpu_owned | alloc device tensors, H2D from bank, free bank, rebind | 1.33 GiB H2D | 0.1 s |
| disk -> gpu_owned | alloc device tensors, read through staging | 1.33 GiB read | 0.5 s |

"Rebind" = `OffloadCache.rebind_layer(layer, residency, banks)`: updates `bank_sources`,
`layer_residency`, `gpu_owned_layer_ids`, `resident_banks`, `_unpinned_layers`,
`_first_streaming_layer`, the fused-copy pointer tables (`_init_fused_copy`), and the prefill
overlap flag. Slot accounting: the slot total charged for owned layers changes by 512 per layer
moved (`cache_budget.lru_slots_after_owned_charge`); the governor decides whether the freed 512
slots are returned to the LRU or given back to the card (a VRAM step-down keeps the LRU size and
frees VRAM; a RAM step-down keeps VRAM constant).

Plus the graph recapture the existing rebuild already pays (a few seconds). That is the hiccup
Jay accepted.

### Rebuild between decode steps, and the wait queue

Today a rebuild waits for the scheduler to be idle (`scheduler.py:222`, asserts no runnable
prefill or decode) and the API answers 503 to new generation while `maintenance_state ==
"rebuilding"` (`api_server.py:262`). Two changes:

1. A rebuild that changes only the slot cache and layer residency (no KV, mamba or window
   change) may execute at a **decode step boundary** with requests in flight: their state lives
   in the KV pages and the GDN state pool, both untouched. `_execute_pending_rebuild` runs when
   `last_data is None` (the previous step is drained) even if `decode_manager.runnable`; it does
   not run mid prefill chunk. Prefill-overlap buffers alias the slot cache and are rebuilt anyway.
   KV-pool changes stay idle-only (there is no request retraction in this scheduler; adding one is
   out of scope), so the KV rung is the last one and is skipped while a request is active.
2. `FrontendManager.new_user()` waits on an `asyncio.Event` while the state is `rebuilding`
   (cap 120 s, then 503 as today) instead of raising at once. `loading`, `failed` and `stopping`
   still 503. Streaming clients see a delayed first token, never an error.

### The governor

Runs in the settings helper (`freetoken.daemon.settings`, torch-free, already probes VRAM with
`nvidia-smi` in `memory_fit._read_vram_snapshot`). Every 2 s while the server is up:

- free VRAM: `nvidia-smi --query-gpu=memory.free` (inside WSL it reports the whole card, matching
  the Windows view; verified 31,591 MiB both sides).
- free Windows RAM: `powershell.exe -NoProfile -Command "(Get-CimInstance
  Win32_OperatingSystem).FreePhysicalMemory"` (about 0.5 s, cached, 3 s timeout); fallback
  `/proc/meminfo MemAvailable` when interop is unavailable. Windows free is the number that
  matters: the VM's own free pages are only useful to Windows once they are returned (L1
  measures the return time).
- policy: below a cushion -> `POST /v1/cache/step {"axis": "vram"|"ram", "direction": "down"}`.
  The engine picks the rung and replies with what it did and whether it is at the floor. One step
  per 5 s per axis. Step-up: free above cushion + one rung (1.33 GiB) + 0.5 GiB margin for 60 s
  continuously -> `direction: "up"`; a step-up that immediately trips the cushion again
  doubles the hold-off (cap 10 min) so it cannot flap.
- boot: the card cushion is also written to the boot as `-MoEVramReserveBytes` so the auto slot
  sizing leaves it free from the start (today's auto fill leaves 597 MiB). The fit check counts
  it.

Rung order in the engine (`engine/memory_ladder.py`):

- VRAM down: (1) `gpu_owned` layer -> `disk` if the RAM axis is also below cushion, else ->
  `pinned` (LRU not grown, VRAM freed); (2) slot cache -512 while above the floor; (3) KV pool
  -25 % when idle; (4) at floor: reply `at_floor`, log once. Up: reverse.
- RAM down: `pinned` layer -> `disk`, choosing the layer with the fewest routes in the learned
  routing histogram (`moe/learned_routing.py`) when one exists, else the highest layer id. Up:
  `disk` -> `pinned`, reverse order.
- Owned layers were VRAM-neutral trades in the daily boot; with the governor the daily profile is
  expected to own 6 layers (`auto:6`, 70 tok/s measured 2026-09-05) so the VRAM ladder has
  layer rungs before it has to touch the slot cache.

### Settings page

Three dials in `dials.py`: `MemoryGovernor` (toggle, default on), `GovernorVRAMFreeGB`
(number, 1.5), `GovernorRAMFreeGB` (number, 4). Written to the boot file like every other dial,
read by the helper's governor loop and mapped by `linux_launch.build_launch` onto
`--moe-vram-reserve-bytes`. The status bar shows layers per rung from `GET /v1/cache/residency`.

## Risks, in order

1. **Native card grabs may kill the server faster than a step.** Unmeasured. Under WSL the
   server dies on the next allocation once the card is oversubscribed (dmesg `dxgkio_make_resident
   -12`, 2026-09-07 memory note). Mitigation: the boot leaves the cushion free; the cushion is a
   page setting; L1 measures the real behaviour with 2, 4, 8 GB native grabs before defaults are
   frozen. If a native app cannot allocate while the server holds the card (allocation refused
   rather than the server dying), the governor still works, just reactively.
2. **Freed pinned memory may not reach Windows quickly.** WSL2 returns free VM pages to Windows
   via the balloon; `drop_caches` was seen to return memory (2026-09-05). L1 measures the delay
   for a cudaFreeHost'd bank. If slow, the RAM axis reacts with a lag, not a failure.
3. **Rebuild between decode steps** touches the scheduler's most delicate invariant. Tests must
   cover a rebuild landing between steps of a running request and mid-prefill-chunk deferral.
   Fallback if the builder cannot make it safe: idle-only, which delays a step by one answer.
4. **Eager decode speed** with DISK layers is unmeasured; the slow rung has no speed target
   beyond "does not fail".

## Out of scope

GLM/EXL3, the old Windows launcher, request retraction for KV shrink, moving the dense weights
off the card, zero-hiccup moves, MTP-specific rungs (MTP is off in the daily boot; the resident
draft head can be added as a rung later).
