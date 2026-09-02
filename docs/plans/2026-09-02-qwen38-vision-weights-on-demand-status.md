# Status: picture weights served from the SSD on demand

Implements option (a) of `docs/design/2026-09-02-qwen38-vision-weights-on-demand-design.md`
on branch `vision-on-demand` (from `mtp-upstream-merge` at `750d83d`), worktree
`D:\FreeToken-vision-on-demand`. Nothing pushed, no PR, the GPU untouched, the server never
launched, `D:\Models` never written.

New mode `FREETOKEN_VISION_WEIGHTS=mmap` (launcher `-VisionWeights mmap`), default `ram`
(today's behaviour, byte for byte). In `mmap` mode the 333 picture tensors are not read into
process memory at boot: their 897,862,112-byte extent of `model-bf16-00001.safetensors` is
mapped copy-on-write and each tensor is installed as a zero-copy `torch.frombuffer` view,
with one `PrefetchVirtualMemory` over the extent issued when a picture is admitted.

## What was implemented

### `python/freetoken/models/config.py`

- `vision_weights_backing()` — parses `FREETOKEN_VISION_WEIGHTS` (`ram` default, `mmap`),
  rejects an unknown value, and rejects `mmap` unless `FREETOKEN_LOAD_VISION=1` and
  `FREETOKEN_VISION_EXECUTION=layer-stream`. Shaped like `vision_execution_mode()`.

### `python/freetoken/models/qwen4_exp/weight.py`

- `VisionTensorSpec` / `VisionShardLayout` / `VisionLayout` — the validated on-disk layout.
- `_vision_shard_files(folder)` — shards carrying `model.visual.*`, from
  `model.safetensors.index.json` when present, else a shard scan.
- `_vision_layout(model_path)` — parses every picture tensor's byte range, shape and dtype
  out of the safetensors headers. Fatal, named errors for: no picture tensors, a dtype no
  reader maps, a header whose byte count disagrees with its own shape and dtype, and a
  mixed-dtype extent. Contiguity is never assumed; the window per shard is
  `[min offset, max end)`.
- `MappedVisionWindow` — `(path, aligned file offset, span, virtual address)` per mapping.
- `MmapVisionWeights` — one `mmap.mmap(..., ACCESS_COPY)` per shard, starting at the extent
  rounded down to `mmap.ALLOCATIONGRANULARITY` and only as long as the extent needs (~857 MiB
  of commit charge, not the file's 1.27 GiB); `torch.frombuffer` views per tensor;
  `tensor()`, `names`, `nbytes`, `mapped_bytes`, `windows`, `contains()`, `prefetch()`,
  `release_prefetch()`, `close()`. Holds `_files` and `_maps` so the mapping outlives every
  view built over it.
- `visual.pos_embed.weight` carve-out: read into an ordinary aligned heap tensor
  (`_read_resident`), because it is the one picture tensor the streamed encode uses as a CPU
  `F.embedding` operand rather than a memcpy source.
- `prefetch()` — one `PrefetchVirtualMemory` covering every window, or one
  `madvise(MADV_WILLNEED)` per window on POSIX. At most one outstanding prefetch per
  picture; `release_prefetch()` clears it. Never raises.
- `open_mmap_vision_weights` / `mmap_vision_weights` / `close_mmap_vision_weights` — a
  process-scoped holder per checkpoint folder, so the loader and the model share one mapping.
- `iter_weights` — in `mmap` mode yields `mapped_vision.tensor(name)` for `visual.*` keys and
  skips opening the second CPU safetensors handle entirely; otherwise unchanged.

### `python/freetoken/models/qwen4_exp/vision.py`

- `Qwen4VisionModel._weight_source` (`_`-prefixed, so `BaseOP.state_dict` skips it),
  `attach_weight_source()`, `weight_backing()`, `prefetch_weights()`,
  `release_weight_prefetch()`.
- `forward_layer_streamed` — prefetches as its first act, before the validation guards, and
  releases in an outer `try/finally`. The existing workspace/`empty_cache` `finally` is
  unchanged.

### `python/freetoken/models/qwen4_exp/model.py`

- `_attach_picture_weight_source(engine_config)`, called from `load_host_tables`: hands the
  tower the holder the loader already built (a lookup, never a second mapping).
- `prefetch_picture_weights()` — the scheduler's optional hook, delegating to the tower.
- `weight_placement_report()` — now reports `backing=mmap|ram`.

### `python/freetoken/scheduler/scheduler.py`

- `_prepare_multimodal_request` calls the optional `prefetch_picture_weights` hook via
  `getattr` after validating `pixels`/`grid`/`token_types` and before
  `build_mrope_positions`, inside a `try/except` that logs and continues.

### `python/freetoken/models/weight.py`

- `load_weight`'s FTW branch warns once that `mmap` does not apply (an FTW checkpoint replays
  post-`iter_weights` tensors and never reaches the per-model reader) and serves them
  resident.

### `scripts/start-qwen38-flash-next-mmap-windows.ps1`

- `-VisionWeights <ram|mmap>`, default `ram`, set only inside the `-EnableVision` branch,
  echoed in the banner as "Picture weights". `mmap` with `-VisionExecution gpu` is rejected
  before the torchvision package probe. A text-only boot pins the variable to `ram`.

### Tests

| file | added |
| --- | --- |
| `tests/models/qwen4_exp/test_config.py` | 6 flag-plumbing tests |
| `tests/models/qwen4_exp/test_weight.py` | 24 layout / view / prefetch / fallback / `iter_weights` tests over a synthetic checkpoint with an odd data base |
| `tests/models/qwen4_exp/test_vision_weights_ckpt.py` | 9 tests against the real checkpoint (new file) |
| `tests/models/qwen4_exp/test_vision.py` | 8 prefetch-handshake and mapped-copy tests (1 CUDA-gated) |
| `tests/engine/test_vision_weight_placement.py` | 3 backing-report / hook tests, 1 updated |
| `tests/scheduler/test_multimodal_admission.py` | 4 prefetch-hook tests |
| `tests/models/qwen4_exp/common.py` | `write_safetensors_shard` (shared) |

## Commits

| commit | subject |
| --- | --- |
| `4e444ce` | feat(vision): add the FREETOKEN_VISION_WEIGHTS backing gate |
| `637432e` | feat(vision): map the picture-weight extent instead of reading it into RAM |
| `a9841bf` | feat(vision): prefetch the mapped picture weights at picture admission |
| `3b417e8` | feat(launcher): add -VisionWeights <ram\|mmap> |
| `1a6cca0` | test(vision): prove a misaligned mapped view is a valid copy source |

Branch point: `750d83d`. Every message ends with the required `Co-Authored-By` and
`Claude-Session` trailers.

## Test commands and results

All runs on this box with `CUDA_VISIBLE_DEVICES=""`, so no GPU was touched. From the
worktree, in PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:PYTHONPATH = "D:\FreeToken-vision-on-demand\scripts\windows-ple-mmap;" +
                  "D:\FreeToken-vision-on-demand\python;" +
                  "C:\Users\jay\AppData\Local\Temp\claude\D--FreeToken\14eaa0eb-ee91-4e4f-a9c5-2096e88229bd\scratchpad\pytest-site"
$py = "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe"
```

| # | command (after `-m pytest`) | result |
| --- | --- | --- |
| 1 | `tests\models\qwen4_exp\test_weight.py tests\models\qwen4_exp\test_vision.py tests\models\qwen4_exp\test_config.py tests\models\qwen4_exp\test_vision_weights_ckpt.py tests\scheduler\test_multimodal_admission.py tests\engine\test_vision_weight_placement.py -q -p no:cacheprovider --timeout=900` | **159 passed**, 0 failed |
| 2 | `tests\models\qwen4_exp\test_vision_weights_ckpt.py -q -p no:cacheprovider --timeout=900` | **9 passed** (real checkpoint) |
| 3 | `tests\models\qwen4_exp -q -p no:cacheprovider --timeout=900` | 275 passed, 52 skipped, **90 failed — all pre-existing** |
| 4 | `tests\models\qwen4_exp tests\scheduler tests\engine -q -p no:cacheprovider --timeout=900` | 1282 passed, 60 skipped, **102 failed + 8 errors — all pre-existing** |

The same command 4 was run against a pristine `git archive` of the branch point `750d83d`
(extracted to a temp directory, so neither worktree was disturbed): **102 failed, 1230
passed, 60 skipped, 8 errors**. The sorted list of failing and erroring test ids is
**byte-identical** between `750d83d` and this branch — nothing regressed, and the delta is
+52 passing tests. The pre-existing failures are all GPU/`flashinfer` absence on this box
(`AssertionError: Invalid device id`, `Attention backend 'fi' requires flashinfer`,
`SimpleNamespace has no attribute forward_host_ctx`), which the brief lists as known.

### What the real-checkpoint tests prove (off-GPU)

- One shard, 333 tensors, one unbroken 897,862,112-byte extent ending at the last byte of
  `model-bf16-00001.safetensors`, all BF16, every tensor at an odd file offset.
- Per-component budget matches the design exactly: 324 block tensors / 822,933,216 B,
  6 merger / 66,079,232 B, 2 patch_embed / 3,541,248 B, 1 pos_embed / 5,308,416 B.
- Every one of the 333 mapped views is **byte-identical** to
  `safetensors.safe_open(..., device="cpu").get_tensor(...)`.
- Every view except `visual.pos_embed.weight` has its `data_ptr()` inside the mapped window;
  `pos_embed` is outside it and 2-byte aligned.
- One window, aligned down to 64 KiB, spanning less than the extent + 64 KiB.
- The real `PrefetchVirtualMemory` over the real 856 MiB returns success, and a second call
  for the same picture issues no syscall.
- `target.copy_(view)` from every misaligned mapped view lands the exact bytes, and no view
  moved out of the mapping — i.e. copying did not write through the copy-on-write view. This
  also faults the whole extent in, so it doubles as the off-GPU read smoke test.

## Live-verification checklist for the operator

Needs the GPU, or a live server. In this order, once the GPU is free.

**A. The two tests that skipped here.** With a GPU visible:

```powershell
& $py -m pytest tests\models\qwen4_exp\test_vision.py -q -p no:cacheprovider --timeout=900
```

- `test_streamed_encode_from_mapped_sources_prefetches_exactly_once` — one prefetch, one
  release per encode.
- `test_layer_stream_matches_gpu_reference_uses_all_blocks_once_and_cleans_up` — the existing
  streamed-encode reference, unchanged by this work.

**B. `_copy_component_state_` H2D from mapped sources.** The CPU analogue is proven above;
what is unverified is the pageable host-to-device copy whose *source* is a misaligned mapped
page. Boot with `-EnableVision -VisionExecution layer-stream -VisionWeights mmap` and encode
one picture. Open question 2 of the design: if that H2D is measurably slower than from heap
pages, option (a2) — one extra memcpy through a 63 MiB pinned buffer — is the fix.

**C. RAM saving (acceptance criterion 3).** Boot `ram`, record the engine PID's
`PrivateMemorySize64` and `WorkingSet64` at idle; restart in `mmap`, record the same. Expect
the private working set at least 800 MiB lower. Also record the boot high-water mark.

**D. Placement report (criterion 1).** The boot log must read
`Picture weights: mode=layer-stream, backing=mmap, tensors=333, bytes=897862112, devices=cpu`,
plus the new INFO line `Picture weights: mapped 333 tensors, 897862112 bytes in 1 window(s)
of model-bf16-00001.safetensors`.

**E. Post-encode containment (criterion 2, live).** After two picture requests, confirm every
`visual.*` tensor except `pos_embed` still has its `data_ptr()` inside the mapped window. A
tensor that has left it means something wrote through a view and the saving is gone.

**F. Screenshot latency budget (criterion 7).** Cold page cache immediately after boot, then
four warm ones back to back. Record every `Picture encoder request N: X seconds`. Expect
~1.9 s cold and ~0.8 s warm against the 6 s budget (`ram` records: 1.519 s / 0.724 s).
Encodes 2-5 within 20% of encode 2 (criterion 9).

**G. Text throughput (criterion 6).** The deterministic 512-input/512-output benchmark three
times before any picture and three times after two pictures, in both `ram` and `mmap`.
Expect ≥ 50 output tok/s in every run (`ram` records: 50.58 before / 52.10 after).

**H. Sizing unchanged (criteria 4, 5).** `/v1/cache/status` reports 4,097 pages of 64 tokens
(262,208 allocated, 262,144 usable) and automatic sizing restores 4,063 cached experts.

**I. Failure paths (criteria 10, 11).** A malformed picture must produce one terminal request
error followed by a passing direct text-health request. `-VisionWeights ram` must reproduce
today's behaviour exactly. A text-only boot must build no tower and open no mapping.

**J. Regressions.** `benchmarks/run_qwen38_vision_smoke.py` (33K chunked picture), and one
normal pi screenshot plus a following pi text request with normal context, skills, templates
and extensions.

**K. Open question 1 — standby list.** Sample `PrivateMemorySize64` and the standby-list size
at boot, after the cold picture, and 60 s after the last picture, to find out whether
`PrefetchVirtualMemory` charges the extent to the working set and whether that depresses the
standby pages the 47.7 GiB PLE table needs. If it hurts, the knob is `EmptyWorkingSet` after
an encode, at the price of a cold next picture.

## Deviations from the design draft

1. **Separate prefetch one-shot flags.** The design said to reuse the PLE
   `_prefetch_failed`. This uses its own `_vision_prefetch_failed` /
   `_vision_advise_failed`, reusing the resolver and the entry-array technique but not the
   flag: a `FALSE` return for a 112,000-entry PLE row prefetch is a working-set or quota
   refusal that says nothing about a single-entry request over one extent, so one subsystem
   must not silence the other.
2. **How the holder reaches the model.** The design specified `Qwen4VisionModel._weight_source`
   but not how it gets there, and `iter_weights` is a free function that never sees the
   model. A process-scoped registry keyed on the checkpoint folder
   (`open_mmap_vision_weights` / `mmap_vision_weights`) lets the loader build the mapping and
   `load_host_tables` adopt the same object, with no second mapping and no second 857 MiB of
   commit charge.
3. **Dtype validation is against the header, not a passed-in model dtype.** `iter_weights`'
   contract carries no model dtype. The layout instead requires that every picture tensor's
   dtype be one the reader maps, that all of them agree, and that each header's byte count
   match its own shape and dtype — which is what prevents a garbage view. See risk 1 below
   for the residual.
4. **The prefetch is issued before the encode's validation guards**, not after, wrapped in an
   outer `try/finally` that releases it. Earliest issue wins the most overlap, and it makes
   the handshake testable without a GPU.
5. **At most one outstanding prefetch per picture.** The design called the second (defensive)
   call "idempotent and cheap"; the handshake makes it free instead of 40 ms.
6. **One extra test file.** The design's test list is implemented as written; the
   real-checkpoint tests needed their own module (`test_vision_weights_ckpt.py`) because
   `test_weight_ckpt.py` is gated on `FREETOKEN_QWEN4EXP_MODEL` and loads the 47.7 GiB PLE
   table, which must not become a default-on test.
7. **The FTW warning lives in `models/weight.py`**, not in the design's file list, because
   `load_weight`'s FTW branch is the code that bypasses the per-model reader.

## Open risks

1. **A vision dtype that differs from the model dtype would silently un-map, not fail.**
   `BaseOP.load_state_dict` asserts `param.dtype == item.dtype`, but
   `_materialize_loaded_weight_state_dict` runs first and does
   `weight.to(device=target, dtype=expected.dtype)`, which for a mismatched dtype returns a
   *converted private copy* — correct output, no mapping, no saving, no error. The verified
   checkpoint is BF16 throughout and the meta model is BF16, so `.to()` is a no-op today, and
   the placement report's `backing=mmap` plus check E would expose it. Making it loud would
   need the model dtype threaded into `iter_weights`, which the loader contract does not carry.
2. **`EXCEPTION_IN_PAGE_ERROR` is a new crash mode.** If the checkpoint file is deleted,
   moved, or the drive drops while serving, a fault at copy time is unrecoverable. Identical
   exposure to the already-accepted 47.7 GiB PLE mapping, which faults on every decoded
   token. Documented, not caught. The standing rule stays: the model directory is immutable
   while the server is up.
3. **Writing through a view would silently negate the saving.** Copy-on-write makes the page
   private and dirty with no error. Guarded by tests at the tensor, component and real-file
   level, but only check E proves it on the live tower after a real encode.
4. **~857 MiB of commit charge** for the copy-on-write mapping, in exchange for the private
   working set. Not free on a machine measured at ~8 GiB free with the server up.
5. **Cold latency is a bound, not a measurement.** 0.30-0.45 s expected, 0.90 s pessimistic,
   from proxy slices of a different shard. Check F is the real number.
6. **The mapping is never closed on the serving path.** The registry holds it for the process
   lifetime, exactly as `self._ple_table` holds the PLE one; `close_mmap_vision_weights()`
   exists for tests and explicit teardown.
7. **File order is lexicographic, execution order is numeric.** Blocks are consumed in an
   order the sequential read does not follow, so some copies wait on pages the read has not
   reached. Estimated tens of milliseconds over one 856 MiB extent at 5-7 GB/s; not worth
   reordering for unless check F says otherwise.
