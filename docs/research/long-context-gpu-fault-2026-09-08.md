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
- The server stays dead after a scheduler crash ("Backend worker is gone and cannot be restarted");
  the settings helper does not restart it. That is what left 2020 unreachable from 23:38 to 04:20.

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
