# Live run: GPU-owned MoE layers — the feature does not boot

Measured 2026-09-02 on the live box: Windows 11 Pro 26200, RTX 5090 32 GB (32,607 MiB),
95.6 GiB system RAM, model `D:\Models\Qwen3.8-Flash-Next-NVFP4`.
Branch `gpu-owned-layers` @ `ae81873`, worktree `D:\FreeToken-gpu-owned-layers`.
Baseline branch `mtp-upstream-merge` @ `8591caf`, tree `D:\FreeToken`.

## Verdict

**`--moe-gpu-owned-layers` cannot load this checkpoint on this build.** Every boot attempt
failed, all for one root cause, and none of the nine remaining checks in the plan's live table
could be reached — no serving candidate ever existed, so there are no RAM, speed, TTFT,
answer-identity, picture or routing numbers for the feature.

The single defect, with two faces:

1. **`GpuOwnedStagingPool.flush` copies into an inference tensor from a non-inference thread
   and always raises.** The engine loads weights inside `torch.inference_mode()`, and inference
   mode is *thread-local*. `alloc_layer_banks` therefore creates each GPU-owned layer's device
   bank as an **inference tensor** on the loading thread, but the flush that fills it runs on
   the `PinPipeline` drain thread, which is not in inference mode:

   ```
   File "python\freetoken\moe\host_banks.py", line 288, in flush
       dst.copy_(staging[name].tensor, non_blocking=True)
   RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
   ```

2. **With the default (`auto`, six layers) that error is swallowed and the boot hangs
   forever instead of failing.** `PinPipeline._run` catches every exception into `self._exc`
   and then *drains the queue without running anything* (`host_banks.py:499`), so no later
   flush executes and no staging slot is ever returned. The staging pool's cap is 2, so the
   loader blocks in `GpuOwnedStagingPool._acquire_locked`'s `self._cv.wait()` the moment it
   touches a **third** owned layer, and `self._exc` is only re-raised by `PinPipeline.wait()`
   / `__exit__`, which the blocked placement loop never reaches. The result is a silent,
   CPU-idle, I/O-idle hang partway through the expert-bank load.

Both faces are reproduced standalone in the scratchpad (see *Reproduction*, below); the second
is confirmed on the live process with a `py-spy dump`.

## What was run

One server at a time throughout; `nvidia-smi` < 3 GB and a 45–60 s settle between every boot;
only `python.exe` processes carrying `freetoken.cli serve` (and their python children) were
ever killed; `D:\Models` and both source trees untouched; nothing committed or pushed.

| # | build | flags | outcome |
|---|---|---|---|
| 0 | `D:\FreeToken` (baseline, 3 h 42 m uptime) | `-MoECacheSize 6750` | measured, then stopped |
| 1 | worktree | `-GpuOwnedLayers auto -MoECacheSize 4400` (parallel expert load) | **HANG** in expert load, killed 15 min in |
| 2 | worktree | same + `-ExpertLoad serial` | **HANG** at the same point, killed 10 min in |
| 3 | worktree | same as #1, re-run to capture stacks | **HANG**, `py-spy dump` taken |
| 4 | worktree | `-GpuOwnedLayers auto:2 -MoECacheSize 5750` | **CRASH** — the `RuntimeError` above, surfaced |
| 5 | `D:\FreeToken` (restore) | `-MoECacheSize 6750` | serving, left running |

Boot #4 is the informative one: with only two owned layers the loader never asks for a third
staging slot, so the placement loop runs to the end, `PinPipeline.__exit__` re-raises the
stored exception and the boot dies with a stack instead of hanging. Boots #1–#3 (six owned
layers) hang instead — and both #1 and #2 stopped after exactly **two** owned layers, the
cap-2 signature.

Boot #1's log tail, `nvidia-smi` and process counters at the hang: GPU 14,055 MiB, engine
private 73.61 GiB, working set 8.49 GiB, **CPU 0.016 s per 10 s wall and zero disk reads** —
fully blocked, not slow.

### The live stack (boot #3)

`py-spy dump --pid <engine>` on the hung scheduler process:

```
Thread (idle): "MainThread"
    wait (threading.py:355)
    _acquire_locked (freetoken\moe\host_banks.py:269)
    fill_view (freetoken\moe\host_banks.py:263)
    fill (freetoken\moe\host_banks.py:321)
    _load (freetoken\models\nvfp4_banks.py:331)
    load_nvfp4_expert_source_banks_parallel (freetoken\models\nvfp4_banks.py:351)
    ... _init_offload_moe_cache (freetoken\engine\engine.py:801)
Thread (idle): "Thread-1 (_run)"
    _run (freetoken\moe\host_banks.py:494)        <- self._q.get(): the drain thread is IDLE
```

The drain thread is parked on an **empty** queue: no layer ever completed, so no flush was
ever queued after the first one failed. Full dump: `pyspy-hang-parallel.txt` in the scratchpad.

## Reproduction (no server, seconds, on this box)

`repro_pipeline_real.py` builds the production chain — `requested_residency` →
`alloc_layer_banks` → `LayerCompletionTracker` → `PinPipeline` → `GpuOwnedStagingPool` — with
Qwen3.8's real NVFP4 bank shapes (E=512, H=2560, I=640) and the real `E*6` writes per layer:

| owned layers | `torch.inference_mode()` | result |
|---|---|---|
| 3 | no | PASS |
| 4 | no | PASS |
| 2 | **yes** | `RuntimeError: Inplace update to inference tensor outside InferenceMode` |
| 3 | **yes** | `HANG: completed layers [0, 1]; blocked while filling layer 2` |

`repro_loader.py` runs the same through the **real** NVFP4 shard loader against the real
checkpoint, clipped to the first N MoE layers (serial and parallel, mixed owned/streaming
sets): it passes outside inference mode, which is why the CPU suite never caught this.

`repro_staging_deadlock.py` isolates face 2 on its own: cap 2, a single-threaded placement
loop and three owned layers deadlock even with no error at all, because the placement loop is
the only thread that can ever complete a layer. The NVFP4 placement loop *is* single-threaded
in both readers (`iter_expert_tensors_parallel` parallelises the byte reads and yields to one
consumer), so the cap-2 back-pressure in spec §4.2 has no second producer to fall back on.
Even with the inference-mode bug fixed, `auto`'s six layers will still deadlock the parallel
reader unless the cap is raised to cover the interleaving or the flush is made synchronous.

## Before / after table

| quantity | baseline (`D:\FreeToken`, 6,750 slots) | candidate (`auto`, 4,400 slots) |
|---|---|---|
| commit | `8591caf` | `ae81873` |
| booted to `serving` | yes, 53 s (fresh) | **never** |
| whole-system commit, idle | 217.87 GiB (uptime 3 h 42 m) | — |
| whole-system physical in use, idle | 87.55 GiB | — |
| server-attributable physical (minus 22.43 GiB empty ref) | 65.12 GiB | — |
| scheduler private bytes | 99.66 GiB | — |
| scheduler working set | ~0 (Windows had trimmed the idle process; 2.29 GiB after traffic) | — |
| boot peak commit / min available | not sampled for the legacy boot | 181.8 GiB / 64.2 GiB at the hang |
| free VRAM after init | 4.89 GiB | — |
| `nvidia-smi` idle / during decode | 30,593 / 29,150 MiB | 14,055 MiB, stuck mid-load |
| 8k-chat decode (sweep method) | 62.1 tok/s | — |
| 8k-greedy decode | 59.0 tok/s | — |
| 8k-chat, temperature 0, thinking off | 75.0 cold / 70.5–72.2 warm tok/s | — |
| TTFT cold-7k / warm turn | 6.54 s / 1.63 s | — |
| TTFT on the temp-0 8k prompt | 8.93 s cold / 1.29–1.39 s warm | — |
| picture request (`-VisionWeights mmap`) | 8.71 s cold, 5.27 s warm, answered correctly | — |
| answer identical to baseline | n/a | — |

The baseline column is the **long-uptime** server the run started against (booted 09:48, measured
13:35–13:45). Its 8k-chat 62.1 tok/s is ~15 % below the fresh-boot 73.1 tok/s of
`measurements-moe-cache-sweep-2026-09-02.md` at the same 6,750 slots, consistent with (and
larger than) the ~8 % uptime decay that sweep documented. A fresh-boot baseline column is in
*Fresh baseline*, below.

## Plan check table

| check | verdict | evidence |
|---|---|---|
| boot log shows owned set, LRU size, MTP graphs 6/6 + 7/7 | **FAIL** | boot never reaches `_gpu_owned_boot_line`; the load hangs (or crashes) inside `_init_offload_moe_cache`. The launcher banner `GPU-owned MoE layers: auto` and `moe_gpu_owned_layers='auto'` in `ServerArgs` are the only owned-layer output produced. |
| scheduler private bytes and whole-system commit −7.9 GiB | **NOT MEASURABLE** | no serving candidate |
| whole-system physical in-use −7.9 GiB | **NOT MEASURABLE** | no serving candidate |
| boot peak host RAM ≤ baseline + 1.5 GiB | **NOT MEASURABLE** | the boot never completes; at the hang it sat at 181.8 GiB commit / 64.2 GiB available, below the baseline boot peak, but that is a partial load |
| 8k-chat decode tok/s | **NOT MEASURABLE** | no serving candidate |
| TTFT | **NOT MEASURABLE** | no serving candidate |
| answers at temperature 0 identical to baseline | **NOT DECIDABLE on this box** | the *baseline* is not reproducible against itself: five 512-token temperature-0 runs of the same prompt on the same server produced five different answers, first divergence at 247–1,775 bytes. A 48-token version of the same request *is* stable (3/3 identical, md5 `29f0e74d…`), so a short-answer identity probe is the only usable form of this check. |
| picture request works | **NOT MEASURABLE** for the candidate; baseline 8.71 s cold / 5.27 s warm, correct answer |
| `/v1/cache/routing` owned rows `resident: true` | **NOT MEASURABLE** | no serving candidate. Note the endpoint needs `--moe-collect-decode-freq` (launcher `-CollectRoutingStats`) at boot; without it it answers 409, as the baseline server did. The plan's boot command does not include it, so this check can never pass as written. |
| owned-layer rows byte-identical to a host-bank load | **NOT RUN** | it needs the model loaded, and the model cannot load |

## Items only a live run could decide (status doc §"Only a live GPU run can decide this")

| # | item | outcome |
|---|---|---|
| 1 | `cudaHostAlloc` staging + real H2D + the CUDA event gating staging reuse | **FAILS.** The H2D never executes: `dst.copy_` raises before any event is recorded. The cap-2 back-pressure is separately unsound against a single-threaded placement loop. |
| 2 | resident VRAM rows byte-identical to host-bank rows | not run — no load |
| 3 | boot peak host RAM with cap-2 staging live | not measured — no complete boot |
| 4 | decode / MTP graphs still capture | not reached |
| 5 | `_build_fused_copy_plan` 0-placeholder on a CUDA device | not reached |
| 6 | speed cost of 6,750 → ~4,400 slots | not measured |

## Fresh baseline (restore boot, for the record)

The step-7 restore boot of `D:\FreeToken` on port 2020 was also measured, so the branch has a
*fresh-boot* baseline to be compared against when it can boot. Identical flags, 6,750 slots.

| quantity | legacy 2020 (3 h 42 m uptime) | fresh 2020 (restore boot) |
|---|---|---|
| boot to `API server is ready` | 53 s | 65 s |
| free VRAM after initialization | 4.89 GiB | 4.64 GiB |
| MTP spec / draft graphs at boot | 6/6, 7/7 | 6/6 (2.271 s), 7/7 (0.663 s) |
| whole-system commit, idle | 217.87 GiB | 216.90 GiB |
| whole-system physical in use, idle | 87.55 GiB | 91.96 GiB |
| scheduler private bytes | 99.66 GiB | 98.35 GiB |
| scheduler working set | ~0 (trimmed) | 67.05 GiB |
| `nvidia-smi` idle / after benchmark | 30,593 / 29,150 MiB | 29,247 / 31,313 MiB |
| 8k-chat (sweep method, median of 2) | 62.1 tok/s | **72.1 tok/s** |
| 8k-greedy | 59.0 tok/s | 70.7 tok/s |
| cold-7k TTFT / warm-turn TTFT | 6.54 s / 1.63 s | 5.89 s / 1.59 s |
| 8k-chat, temperature 0, thinking off | 75.0 / 70.5 tok/s | 68.2 / 69.2 tok/s |
| picture cold / warm | 8.71 s / 5.27 s | 6.89 s / 4.86 s |

The fresh 72.1 tok/s reproduces the cache sweep's 73.1 tok/s at 6,750 slots to within 1.4 %,
which is the methodology check for these numbers. The legacy server's 62.1 tok/s is the same
uptime decay the sweep saw, here ~14 %.

**Cross-server answer identity works at 48 tokens.** The 48-token temperature-0 chat answer to
the 8k prompt is byte-identical across the two *different server processes* — md5
`29f0e74dfed744b538f856f38553e1ae` on all five samples (3 legacy + 2 fresh). That is the probe
a future candidate boot should be compared against; the 512-token form is not reproducible even
against itself (see the check table).

## Anything unexpected in the logs

- No warning, error or fallback appears in either hung boot before the stall: the last line is
  `expert banks: slow path (parallel build)` / `(serial build)`, then nothing. The failure is
  entirely silent, which is its worst property.
- `PinPipeline._run`'s post-failure drain (`if self._exc is not None: continue`) converts *any*
  settle or flush failure during a GPU-owned load into a hang rather than an error. That is a
  hazard independent of this bug: a `cudaHostRegister` failure on a streaming layer would do
  the same thing once an owned layer is waiting for a staging slot.
- `nvidia-smi --query-compute-apps` on this box lists ~27 pids with `[N/A]` used memory, so it
  is not usable for identifying the server's processes; `Win32_Process` command-line matching
  is.

## Suggested fixes (not applied — nothing was changed in either tree)

1. Do the flush inside inference mode, or allocate the owned banks outside it. The narrow
   change is to wrap the copy: `with torch.inference_mode(): dst.copy_(...)` inside
   `GpuOwnedStagingPool.flush` — but note the inference/normal tensor split is exactly the sort
   of thing that will bite again at `resident_views`, so allocating the device banks as normal
   tensors (`torch.empty` outside inference mode, e.g. under `torch.inference_mode(False)`) is
   the more durable option.
2. Make a drain-thread failure fail fast: have `PinPipeline` record the exception *and* wake
   every waiter (e.g. a pool `abort()` that notifies `_cv` and makes `_acquire_locked` raise),
   so a broken flush surfaces as an error rather than a hang.
3. Re-examine the staging cap. With a single-threaded placement loop, `cap` must be ≥ the
   number of owned layers the reader can have in flight at once; `auto` is six, and the
   parallel reader interleaves freely. Either raise the cap to `len(owned)`, flush
   synchronously on the placement thread, or make back-pressure impossible by construction.
4. Add a CPU test that runs the loader chain inside `torch.inference_mode()` — that one
   context manager is the whole difference between the green suite and this run.

## Scratchpad

Scripts, logs, dumps and raw snapshots:
`C:\Users\jay\AppData\Local\Temp\claude\D--FreeToken\14eaa0eb-ee91-4e4f-a9c5-2096e88229bd\scratchpad\gpu-owned-live`
— `boot-2030.ps1`, `stop-server.ps1`, `snap.ps1`, `chat8k.py`, `snaps.jsonl`,
`hang-cand*-*.log`, `pyspy-hang-parallel.txt`, `repro_staging_deadlock.py`,
`repro_pipeline_real.py`, `repro_loader.py`.
