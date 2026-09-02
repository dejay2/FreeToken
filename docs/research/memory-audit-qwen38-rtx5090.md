# FreeToken memory audit: Qwen3.8-Flash-Next-NVFP4 on RTX 5090 / Windows 11

Read-only source trace at commit `8591caf` (branch `mtp-upstream-merge`), 2026-09-02, plus the checkpoint's safetensors headers at `D:\Models\Qwen3.8-Flash-Next-NVFP4`. No server launched, no GPU touched.

Verified geometry (`config.json`): 48 layers, 12 `full_attention` (QSA) + 36 `linear_attention` (GDN), `hidden_size=2560`, `moe_intermediate_size=640`, `num_experts=512`, `num_experts_per_tok=10`, `vocab_size=248320`, `tie_word_embeddings=false`.

Boot flags assumed: `-ContextTokens 65536 -KVCacheTokens 65536 -MoECacheSize 6750 -DenseQuant int8 -EmbedHost -EnableVision -VisionExecution layer-stream --ple-backend mmap`. Measured: GPU ~31.5 GB; host 4 python processes, working set 67 GiB, commit 89 GiB.

## (a) Summary table, steady state after boot

Host column is the scheduler process only (`freetoken-TP0-scheduler`).

| Weight class | Bytes | Host location | GPU location | Duplicated? | Freeable? |
|---|---|---|---|---|---|
| Routed NVFP4 experts (24,576 x 2,772,480 B) | 63.457 GiB | Pinned (anonymous mmap + `cudaHostRegister`), 288 banks | 6,750 slot copies = 17.43 GiB | YES, byte-identical copy | Host: no mechanism. GPU: yes (`rebuild`) |
| of which packed e2m1 codes | 56.25 GiB | pinned | in the 17.43 GiB | yes | no |
| of which fp8 block scales | 7.03 GiB | pinned | in the 17.43 GiB | yes | no |
| of which fp16 per-row globals | 180 MiB | pinned | ~49 MiB | yes + broadcast-redundant (144 KiB of distinct values) | no |
| PLE n-gram table (320,001,536 x 160 FP8) | 47.684 GiB | mmap `ACCESS_COPY` file pages only (page-cache backed, reclaimable) | none | NO | n/a |
| PLE host row cache | ~0.28 GiB | anonymous pageable | none | YES (rows already in page cache) | `FREETOKEN_PLE_ROW_CACHE=0` |
| PLE pinned staging ring | ~30 MiB + torch pinned-cache retention | pinned | tiny | no | by design |
| Token embedding (248,320 x 2560 bf16) | 1.184 GiB | Pinned only (`cudaHostAlloc`) | none, gathered per row over UVA | NO | no |
| `lm_head` (untied, int8) | ~0.592 GiB | none | GPU-only | NO | no |
| QSA attention (int8) | ~0.575 GiB | none | GPU-only | no | no |
| GDN (int8) | ~1.95 GiB | none | GPU-only | no | no |
| Hyper-connections (int8) | ~0.60 GiB | none | GPU-only | no | no |
| Shared expert (int8) | ~0.220 GiB | none | GPU-only | no | no |
| Router gate + norms + PLE dense (bf16) | 0.178 GiB | none | GPU-only | no | no |
| Vision tower (333 tensors) | 0.836 GiB | pageable CPU | transient workspace only | NO | `FREETOKEN_LOAD_VISION=0` |
| MTP head | 4.856 GiB | not loaded | not loaded | n/a | n/a |
| KV pool @ 65,536 tokens x 25,344 B | 1.547 GiB | none | GPU | no | yes |
| GDN linear-state pool (8 slots x 115,642,376 B) | 0.862 GiB | none | GPU | no | partly (F6) |

Totals. Host pinned: ~64.7 GiB. Host pageable: ~1.1 GiB. Host file-mapped (reclaimable): 0 to 47.68 GiB demand-driven. GPU accounted: 17.43 + ~4.05 + 1.55 + 0.86 = ~23.9 GiB against 31.5 GB observed.

## (b) Per-class evidence

### 1. Routed NVFP4 experts
- Bank shapes: `python/freetoken/models/nvfp4_banks.py:63-78`. Sum 1353.75 MiB/layer x 48 = 63.457 GiB; 2,772,480 B per expert, which matches `unit_bytes.moe_per_expert` in `results/benchmark-262144.json` and `expert_bytes_per_slot` (`engine/cache_budget.py:17-28`). GPU slot and host row are the same bytes in the same layout.
- Allocation and pinning: `moe/host_banks.py:108-112` (`mmap.mmap(-1, size)`, appended to module-global `_LIVE_BUFFERS`, lines 53-54 and 109); pin after fill `host_banks.py:126-142` via `kernel/pinned.py:48-50`.
- GPU cache is a separate allocation: `moe/offload_cache.py:544-548`; rows copied by `fast_index_copy_multi_jit` from the host bank's device alias (`offload_cache.py:604`, `pinned.py:59-68`).
- No conversion: `moe_intermediate_size=640 < 1024` selects the `triton` backend (`moe/nvfp4_backends.py:258-268`), `quant_format="nvfp4"` (`expert_banks.py:213-215`); marlin/b12x repacks are in-place (`nvfp4_backends.py:333-344`, `641-654`).
- No release path: `rebuild` keeps bank sources (`offload_cache.py:647-655`); `cudaHostUnregister` appears nowhere; `HostBank.release()` is a no-op for pinned banks (`host_banks.py:144-150`).

### 2. Dense weights
- Loaded straight to CUDA: `safe_open(file, framework="pt", device=str(device))` (`models/qwen4_exp/weight.py:180-183`); vision uses a second CPU handle (`weight.py:190-198`).
- State dict drained by `pop` + `setattr` (`layers/base.py:32-50`); empty after `engine.py:403`.
- No safetensors handle survives boot except the ten PLE shards (`weight.py:426-441`).
- int8 pops bf16, quantizes in chunks, drops it (`kernel/triton/int8_linear.py:80-114`, `685-693`).
- Embedding host-only with `FREETOKEN_EMBED_HOST=1`: `model.py:250-253`, `313-326`; `layers/embedding.py:133-135`, `164-184`.
- `lm_head` untied, separate int8 tensor (`model.py:222-230`, `qwen4_exp/config.py:263-264`; comment at `model.py:207-209`).
- bf16 sizes from shard headers: GDN 3.886 GiB, HC 1.193, lm_head 1.184, embedding 1.184, QSA 1.150, vision 0.836, shared expert 0.440, router 0.117, PLE dense 0.061, MTP 4.856. int8 turns 7.853 GiB into ~3.93 GiB, matching the launcher's "3.9 GiB of VRAM back".

### 3. PLE table (`--ple-backend mmap`)
- `mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_COPY)` + zero-copy `torch.frombuffer` (`weight.py:426-441`); reads only (`weight.py:632-641`), so no private copies. The pinned path (`weight.py:749-778`) is not taken; launcher routes to `load_mmap_ple_table` (`model.py:355-366`).
- Row cache: 1,048,576 rows x 160 B = 160 MiB slab + ~118 MiB bookkeeping (`weight.py:286-290`, `315-405`).
- Pinned staging ring ~30 MiB (`ple.py:47-50`); prefill sizes above 4096 rows allocate fresh pinned buffers (`ple.py:295-296`) that torch's pinned allocator retains.

### 4. Vision tower
Pageable CPU (`model.py:246-248`, `weight.py:190-198`); `forward_layer_streamed` builds one component at a time and calls `empty_cache()` in `finally` (`vision.py:286-309`). No steady-state duplicate.

### 5. MTP head
Dropped at `weight.py:110-111` before any `get_tensor`; expert regex anchored to exclude `mtp.` (`weight.py:46-49`). Spike paths gated on `FREETOKEN_MTP_SPECULATE` / `FREETOKEN_MTP_SHADOW`, default off (`engine/config.py:168`, `engine/mtp_shadow.py:71`).

### 6. Multiprocess
Spawn forced on every platform (`server/launch.py:164`); three `mp.Process` sites (`launch.py:173`, `183`, `202`) plus `daemon/serve_manager.py:139`. No shared memory. Only the scheduler holds weights (`engine.py:373`). Tokenizer loaded independently in up to three processes (`scheduler/scheduler.py:130`, `tokenizer/server.py:374`, `server/api_server.py:212`), ~0.1-0.6 GiB total.

## (c) Ranked findings

- **F1. 17.43 GiB duplicated: GPU expert slots vs pinned host bank, no release path.** 6,750 of 24,576 experts held twice. Inherent to a demand-paged LRU cache. Removing it needs hot-set pinning, host row release (never called today), hotness-sorted re-layout because banks are contiguous per (bank, layer), and handling of holes in `copy_missing` / `_build_fused_copy_plan` (`offload_cache.py:567-610`). High risk; costs the ability to resize the slot cache down.
- **F2. Boot page cache from the 63.3 GiB expert shards is never dropped on Windows.** `drop_page_cache` (`models/loader.py:56-66`) calls `posix_fadvise`, which `scripts/windows-ple-mmap/sitecustomize.py:52-56` stubs to a no-op; called around all 192 shards (`nvfp4_banks.py:115-116`, `150`, `199`). Explains the 87-91 GiB whole-system peak. Fix: Windows `drop_page_cache` or a direct-I/O reader (`FILE_FLAG_NO_BUFFERING`); the parallel reader is gated off on Windows (`expert_banks.py:31`); bounce-buffer logic exists (`host_banks.py:446-459`).
- **F3. 0.836 GiB vision weights in RAM for text-only sessions.** Drop `-EnableVision`.
- **F4. ~280 MiB PLE row cache duplicates page-cache rows.** Shrink via `FREETOKEN_PLE_ROW_CACHE` (e.g. 256 Ki rows ~70 MiB).
- **F5. ~180 MiB host + ~49 MiB GPU of broadcast redundancy in `*_global` banks.** `down_global` `[E, H]` holds one value per expert. Collapsing needs Triton kernel addressing and `_BANK_SCHEMAS["nvfp4"]` row-byte changes (`offload_cache.py:560-565` alignment rule).
- **F6. 0.35-0.46 GiB GPU in GDN snapshot slots at `--max-running-requests 1`.** `_linear_pool_num_slots` (`kvcache/linear_state_pool.py:284-293`); `linear_state_cache_ratio` converts to ~130-170 expert slots.
- **F7. Three HF tokenizers.** Not worth attacking.
- **F8. Transient boot peaks.** Experts never pass through bf16. Dense state dict materialized whole on GPU as bf16 (~9.2 GiB) before int8 drains it (`engine.py:315-336`, `857`). Embedding briefly has three copies (`model.py:325`); adding it to the CPU-handle branch in `weight.py:190-198` would skip the CUDA round trip. Host peak: banks pinning while shard pages sit on the standby list.

## (d) Needs live measurement
1. The ~20 GiB working-set-to-commit gap (suspect WDDM backing store for evictable GPU allocations; also torch pinned allocator retention, CUDA arenas). Measure with VMMap on the scheduler PID and Task Manager "Shared GPU memory".
2. Whether 67 / 89 GiB are per-process sums or whole-system counters (`docs/windows-qwen38-flash-next-mmap.md:191-194` says whole-system).
3. Actual PLE page residency under load (RAMMap, per-file).
4. Whether `--nvfp4-backend` is `auto` -> `triton` (startup log line `NVFP4 expert backend:`, `expert_banks.py:210`).
5. Steady-state resident expert count and hit rate (`--enable-cache-report`, `decode_miss_stats`), which decides whether F1 is worth attacking.
6. Identity of the 4th process (daemon vs tokenizer worker).
7. Torch pinned-host allocator retention for prefill PLE staging.

## Verdict on "FreeToken loads the whole model into RAM and experts into the GPU, some duplicated"
Partly right, and right about the part that matters: the 63.457 GiB of routed experts are whole in pinned RAM and 17.43 GiB of them are simultaneously on the GPU as an identical copy. Everything else is not duplicated by design: PLE is memory-mapped, dense weights go disk-to-GPU with no host copy, embedding is host-only, `lm_head` GPU-only, vision has one CPU copy, MTP is never read. Weights load in exactly one process. The most actionable item is F2, the no-op page-cache drop on Windows.
