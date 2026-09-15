# Host RAM release and governor headroom repair

## Observed failure

The live Qwen3.8-Flash-Next-NVFP4 worker had about 71 GiB RSS while reporting
13 GPU-owned, 30 pinned and 5 disk layers. Its memory map still contained all
40 startup host allocations (about 53 GiB), plus about ten recalled layer
allocations (13 GiB). Conversation parking was RAM, with 0.89 GiB stored.

`GpuOwnedLayers=auto:8` made `_ResidencyPlan.has_unpinned` true, overriding
`FREETOKEN_BANK_CUDA_ALLOC=1` for the other layers. They therefore used
registered anonymous mmap allocations. `_LIVE_BUFFERS` retained those mappings
for the process lifetime. Startup owners were absent from the engine's runtime
`_host_banks`, and mmap `free()` was itself a no-op while pinned. Governor
moves changed logical placement without releasing the original allocations;
recalls allocated additional banks.

The governor also used Windows free RAM without respecting the WSL ceiling.
A recall was observed around 6 GiB host headroom with only approximately
1.2 GiB available inside WSL. The settings geometry retained boot-time
owned IDs and slot counts after later governor moves.

## Repair

- GPU-owned layers no longer veto the requested CUDA host allocation policy;
  pageable/OS-locked host layers still do.
- Registered mmap banks transfer explicit ownership from loader to engine.
  Claims handle interior FTW views and reject an allocation shared across
  independently movable layers.
- At the synchronized rebuild safe point, sources and pointer tables are
  rebound before old banks are unregistered and released. CPU tensor aliases
  keep the underlying buffer valid until their last reference disappears.
  Unregister failures retain ownership and are retried before another move,
  including no-op retries and recalls.
- Native Linux checks MemAvailable; native Windows checks physical host RAM;
  WSL requires both and uses the smaller value. Missing required probes block
  moves and reset continuous-high-memory holds. A VRAM move is followed by
  fresh readings before the RAM axis decides, avoiding stale recall approval.
- Successful governor replies carry measured pool geometry through the
  scheduler/tokenizer/frontend path. Correlated replies update the settings
  geometry and invalidate the old residency snapshot.

## Verification

CPU regression tests first reproduced allocation-policy, retained-buffer,
unregister, startup-owner, failure-retry, governor-headroom and stale-geometry
failures. The combined targeted CPU suite passed 151 tests, with 10 GPU skips.
Independent read-only reviews found one failed-unregister retry issue; it was
reproduced, repaired and retested. Final source review found no remaining
important findings in the host, governor or geometry changes.

Broader CPU checks encountered pre-existing failures: one unavailable
FlashInfer backend and three settings expectations. Those same failures were
reproduced on the unchanged starting revision. Older disk-engine test fixtures
were updated to call current engine helpers and attach their GPU cache.

The native pinned-memory extension must be rebuilt with this source change
because it adds `host_unregister`. Live deployment and measurements are
recorded below after GPU validation.

GPU validation on the RTX 5090: rebuilt the packaged native extension in an
isolated checkout; 55 tests passed and one skipped. This includes real CUDA
copy/registration/release tests for both mmap and CUDA allocations, repeated
three times each, surviving CPU aliases, native disk-fed decode, and GPU-owned
bank loading. The first remote invocation resolved the old editable checkout;
imports were then pinned explicitly to the validation checkout before testing.
