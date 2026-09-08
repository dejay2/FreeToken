# Two scheduler crashes at long context on the RTX 5090 WSL box, 2026-09-07

Branch `memory-governor` (fa4c7c3 at the time), daily 262k profile (KV fp8, `qsa_sparse`, page size 64,
dense int8, vision layer-stream, MaxRunningRequests 1, CUDA graph bs 1), PLE backend `disk` (io_uring,
O_DIRECT, wait-sync), MTP off in both crashed runs (the helper picks `disk` only when MTP is off).

## What happened

| Time | Context at the fault | Where it was | Traffic |
|---|---|---|---|
| 15:40:02 | 117,504 tokens, ~250 tokens into decode | graph-replayed decode step | `/v1/chat/completions`, 116k cached prefix + 239 new tokens |
| 23:32:08 | 153,068 tokens, first decode step after an 18x8192 + 5612 chunked prefill | graph-replayed decode step | `/v1/messages?beta=true` (Claude Code via `ccq`), no cached prefix |

Both times `freetoken-TP0-scheduler` died with `torch.AcceleratorError: CUDA error: an illegal memory
access was encountered`, surfacing at `ple_disk.py:206` (`_readback_event.synchronize()` inside the
deferred PLE fill) because that is the first host-side sync after the decode graph launch. The sync is
where the sticky error shows up, not where it originates.

The Windows System log (`nvlddmkm`, device `\Device\Video8`) carries the same four events at both
timestamps, with identical exception registers:

```
Graphics Exception: ESR 0x505730=0xb010020 0x505734=0x4 0x505728=0x1c81fb60 0x50572c=0x1174
Graphics SM Global Exception on (GPC 0, TPC 0, SM 0): Multiple Warp Errors
Graphics SM Warp Exception on (GPC 0, TPC 0, SM 0): MMU NACK Errors
Error occurred on GPUID: 100   (event 153)
```

`MMU NACK` = a warp touched an address with no mapping: an out-of-range read/write inside a compute
kernel, the same instruction both times (identical ESR words). It is not a TDR/timeout (no event 4101,
TdrDelay unset), not host memory pressure, and not a governor move: no cache step happened in the
15:00 run at all, and the last step in the 20:59 run was 47 minutes before the fault.

The 5.5-6 minute gap between the fault and the supervisor's "backend worker exited" line is the crash
dump: `core_pattern` is `|/wsl-capture-crash` and the service has `LimitCORE=infinity`, so the abort
pipes a ~70 GB process image before the process is gone.

## Exposure

Counted from `logs/server-2020.log` (`Decode batch ... #token:` lines):

| PLE backend / MTP | decode steps > 65k | > 100k | > 131k | crashes |
|---|---|---|---|---|
| mmap + MTP on (runs before 2026-09-07 14:44) | 0 | 0 | 0 | 0 |
| disk + MTP off (14:44 to 23:38) | 2,798 | 1,193 | 210 | 2 |

So every long-context decode step in the log ran on the disk/MTP-off setup; the log cannot separate
"long context" from "disk backend" or "MTP off".

## Reproduction attempts (2026-09-08 04:25-05:10, all passed, server healthy afterwards)

| Setup | Test | Result |
|---|---|---|
| mmap + MTP on | one 151,822-token prompt, 48 then 200 output tokens | OK, 5 logged decode steps at 152k |
| mmap + MTP on | the same three rounds as below (152k prompt, 12-turn 124k chat, 8-turn `/v1/messages` at 155k) | OK, 05:20-05:30 |
| disk + MTP off | same 151,822-token prompt, 200 output tokens | OK, 54.8 s |
| disk + MTP off | 12-turn chat on 124k tokens of repo source, 120 tokens per turn, cached prefix | OK |
| disk + MTP off | 8-turn `/v1/messages` chat with thinking + a tool at 155k tokens, up to 400 tokens per turn | OK |

The fault is intermittent and content-dependent; ~8,000 synthetic decode steps at 124k-155k across both setups did not hit it.

## Open

- Which kernel: unknown. Same faulting instruction both times. Next occurrence: rerun the offending
  request with `CUDA_LAUNCH_BLOCKING=1` (the Python traceback then names the launch), or under
  `compute-sanitizer` if the prompt can be captured.
- ~~The server stays dead after a scheduler crash~~ Fixed the same morning (a14dcec, helper 1.4.0):
  the helper's crash watchdog restarts a server that stops answering unasked (three misses ten
  seconds apart, at most three restarts an hour), the service unit now has `LimitCORE=0` so the
  5-6 minute crash dump is gone, and a `Diagnostic mode` dial boots with `CUDA_LAUNCH_BLOCKING=1`
  and every graph off so the next fault names its kernel.
- Follow-ups found on the way: `POST /api/server/start` against a server that is already serving
  boots a twin that loads 60 GB of banks before failing to bind the port (05:33, squeezed Windows to
  2.5 GB free and spilled a live layer); the page never does this but the API allows it. And
  `stop_servers` waits for whole-card VRAM below 3 GiB, which a game or QUASAR can hold up for the
  full 120 s timeout; the watchdog sidesteps it by using a plain start when no server process is
  left, a page Restart does not.

## Code audit of the long-context decode path (2026-09-08, read-only)

Every index in the `qsa_sparse` select/score/attend/store path, the fp8 KV store, the compressed
index slab, the block top-k split slab, the graph-runner buffers and the scheduler's page/position
math is masked or bounded at the shipping geometry (4,097 pages, 65,536 compressed columns, top-k 512);
nothing there can go out of range at 117k or 153k tokens. Latent (not live) defects worth a clamp:
`kernel/triton/qsa/compress.py:179-189` (destination row has no upper bound; only bites on a corrupt
page table), `kernel/triton/qsa/score.py:50,87,112` (int32 row product, and a clamp where the attend
kernel invalidates instead). No test or document in the repo records a decode run past 131,072 tokens
of real context; the widest end-to-end fixture is 32k.

Because the attention path is bounded and the fault is content-dependent, the stronger candidate is
the MoE offload cache: at 262k the KV pool leaves 5.9-7.0k expert slots, so eviction pressure and the
per-step host-to-device expert copies rise with context and with what the prompt routes to. Bisect
levers that need no code change: `FREETOKEN_QSA_TORCH_TOPK=1` (swap the Triton top-k),
`--cuda-graph-max-bs 0` (graphs off, so the faulting launch is attributable), bf16 KV instead of fp8,
`CUDA_LAUNCH_BLOCKING=1`, or `compute-sanitizer --tool memcheck` on one long request.

## Memory facts measured while recovering from the twin boot (06:00-06:10)

- Windows `FreePhysicalMemory` is the truth; the WSL VM's own `MemFree` (15 GB at the time) is
  not spare memory. `hv_balloon` runs "cold memory discard hint" (order 9): the guest reports free
  pages, Windows drops their backing, the guest still counts them free. Pinning a layer inside the
  VM took ~1.3 GB off Windows free and left VM `MemFree` unchanged; an 8 GB grab on Windows left
  VM `MemFree` unchanged too (Windows compressed its own pages instead: free 4.9 -> 0.4 GB, then
  9.9 GB after release). `drop_caches` and `compact_memory` inside the VM changed nothing.
- So the governor's RAM axis reads the right number. What looked like "WSL holding 17 GB" was
  Windows genuinely short: server ~65 GB + Windows ~9 GB + compressed/standby ~3 GB.
- A Windows-side memory grab has a lasting side effect: Windows trims its standby/compressed
  pages and stays at the higher free figure afterwards (5 -> 10 GB free).
- `POST /v1/cache/step` answers 504 after 60 s when a long eager prefill is running (8192-token
  chunks at ~10 s each with a disk layer); the step still applies when the prefill ends. While it
  waits, `maintenance` is "rebuilding" and the front door holds new requests up to 120 s, so a
  recall queued behind a 3-minute prefill can bounce a new request with 503. Worth a longer front
  door wait or letting the MoE-only rebuild run between prefill chunks.
- Jay set `GovernorRAMRungsBeforeUp` to 1 (from 2) at 06:0x; the recall bar is now
  cushion + 1 rung + margin = 5.82 GiB of Windows free.

## Card-first RAM ladder (2026-09-08, Jay's suggestion)

Measured on this card: a layer on the SSD costs the whole model its CUDA graphs (8-11 tok/s);
a layer parked on the card costs 512 shared expert slots, ~3 tok/s by the 2026-09-02 slot sweep
(5.8 tok/s per 1,000 slots, 8k chat), and on 2026-09-07 13:51 the server ran 8 parked layers with
1,024 shared slots at ~45 tok/s. So the RAM ladder now parks first: shrink the slot cache by one
layer's slots (card-neutral), promote the busiest pinned layer in the learned routing, repeat while
the shared slots stay at or above the streaming floor (1,024 with prefill overlap, the same rule
the settings page's fit check enforces), and only then spill to the SSD. Recovery brings SSD layers
home first, then unparks (last parked first) and grows the slots back toward the boot size. With
today's 5,917 slots that is up to 9 parked layers, ~12 GB of host RAM, before the SSD is touched.
Cost to note: a park or unpark is two rebuilds (shrink, then promote; or demote, then grow), each a
full graph teardown and re-capture, inside the same 60 s step window that already answers 504
behind a long eager prefill; the card needs 512 MiB free before a park is attempted, otherwise the
rung is skipped and the SSD rung runs as before.

### Live proof of the card-first ladder (07:02-07:08, branch at c0f4780, governor ON)

Boot dip: Windows free 3.6 GB right after serving; the governor parked 4 layers (slots 5917 -> 3869),
nothing to the SSD, graphs kept. Squeeze: `hammer.py` 270 s of short chats plus `grab.py` taking
4 then 8 GB of Windows RAM (hold 60 s). Governor moves from the server log:

| Time | Move |
|---|---|
| 07:03:06-07:03:27 | four more parks (8 parked, slots 1821) |
| 07:03:34 | one pinned->disk (the 9th rung: card margin / floor reached) |
| 07:04:35 | disk->pinned first, once Windows read 10 GB free during the hold |
| 07:04:41-07:05:19 | six unparks, one every 7 s, slots growing back 512 at a time |

Every `POST /v1/cache/step` answered 200 (no 504s with short chats). Hammer: 64 requests,
0 failed, p50 15.8 tok/s, min 8.8 tok/s, max first-token wait 7.3 s. Yesterday's same squeeze on
the SSD-first ladder saw 6-11 disk layers and 2.6-4.6 tok/s at the worst point. Windows free did
not move per park (it went up: Windows trims its own standby pages under the grab), so the proof
that a park frees host RAM is the recovery direction: each unpark cost ~1.3 GB of Windows free.
After release one layer stayed parked at Windows free 4.8 GB (recall bar 6.25 GB); ~3 tok/s.

## Guess-ahead (MTP) cost, isolated from the PLE reader (07:10-07:27)

Jay turned `FREETOKEN_MTP_SPECULATE` on at 21:54 on 2026-09-07 through the new MTP tab. Same
daily 262k profile, `bench.py` (three 200-token completions, median decode tok/s), all 48 layers
pinned, nothing parked, server quiet:

| PLE reader | MTP | slots | decode tok/s |
|---|---|---|---|
| disk (io_uring) | off | 6429 | 55.8 (2026-09-07 19:48) |
| mmap | on (depth 5, cost-aware, min emitted 2.4) | 5917 | 19.5-20.6 (three runs) |
| mmap | off | 6429 | 50.3 |

So the guessing itself costs ~2.5x on this build and the memory-mapped reader it forces costs
~10%; the mmap reader took zero major page faults during a 600-token run (scheduler `majflt`
unchanged), so it is not stalling on disk. The cost-aware / min-emitted catches did not stop the
speculation from losing. MTP also holds 2.56 GiB of the card for its draft head (512 fewer slots).
Left: MTP off, `PleBackend` back to auto (disk at the next start). A dynamic on/off worth its name
means fixing the cost-aware gate inside the spike (`engine/spec_*.py`), a separate job.
