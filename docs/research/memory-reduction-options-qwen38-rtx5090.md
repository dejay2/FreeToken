# Reducing memory use in FreeToken on RTX 5090 / Windows 11 / 96 GiB

Research report, 2026-09-02. Builds on `memory-audit-qwen38-rtx5090.md`. Read-only: code verified, external facts from the sources at the end.

## 0. Two reframings

**(a) On Windows, VRAM costs host RAM too.** Microsoft's WDDM documentation: "Every graphics allocation in the WDDM model has a backing store," a committed memory buffer holding the allocation's contents when it is not resident in video memory. That is almost certainly the unexplained ~20 GiB commit gap (17.43 GiB slot cache + ~4 GiB int8 dense + ~1.5 GiB KV = ~23 GiB). If it holds, the audit's F1 is a triplication (pinned host copy, VRAM copy, pagefile-backed backing store), and shrinking `cache_size` saves host RAM roughly 1:1 as well as VRAM. Confirm this first (section 5).

**(b) Don't try to free host rows; try not to allocate them.** Every release mechanism is blocked on Windows. `cudaHostUnregister` takes only the base address given to `cudaHostRegister`, so sub-range unregistration is unsupported. The banks are `mmap.mmap(-1, size)`, which on CPython/Windows is a pagefile-backed section; decommitting pages inside a section view is not supported. `DiscardVirtualMemory` requires `PAGE_READWRITE` and does not decommit; `MEM_RESET` + `VirtualUnlock` cannot drop pages the NVIDIA driver has locked. `HostBank.release()` already encodes this (`host_banks.py:147-150`). Any host-RAM win must come from never allocating the row, decided at load time.

Useful accident: the four large NVFP4 banks have per-expert rows that are exact page multiples (`gate_up_packed` 1,638,400 B, `gate_up_scale` 204,800 B, `down_packed` 819,200 B, `down_scale` 102,400 B), 99.72% of each expert. Per-expert decommit would need no re-layout on Linux (`madvise(MADV_DONTNEED)` on an unregistered range). Not usable on Windows.

## 1. Strategies

### A. Static hot set with host rows never allocated (recommended)
Reserve K slots per layer of the GPU slot cache as permanently resident for the K hottest experts of that layer, and allocate that layer's host banks at `[512-K, ...]`. Hot rows stream from the safetensors shards straight into the GPU slot pool at load and never occupy a host row.

Savings: K x 48 x 2,772,480 B. K=42/layer (2,016 slots) -> 5.2 GiB host RAM; K=70 (3,360 slots) -> 8.7 GiB; K=94 (4,500) -> 11.6 GiB. VRAM unchanged.

Cost: LRU capacity falls from ~140.6 to (6750 - 48K)/48 slots per layer. Whether this costs tok/s depends on routing skew, measurable today via `decode_routing_stats()`. Literature: Mixtral-class routers put >50% of assignments in the top-2 of 8 experts per layer; a 25% expert budget captures 37-53% of activations; MoE-Infinity uses LFU over LRU and reports 3.1-16.7x over LRU-style baselines; fMoE warns hot sets are prompt-dependent. Qwen3-Next's top-10-of-512 router is far finer-grained and likely flatter, so Mixtral skew may not transfer. This one number decides the strategy.

Files: `models/nvfp4_banks.py:63-78` (`_alloc_nvfp4_host_banks`, allocate E-K rows); `load_nvfp4_expert_source_banks` (route hot experts to a device staging path); `moe/offload_cache.py:544-548` (`set_bank_sources`, drop the `source.size(0) == self.num_experts` assertion, carry K); `moe/offload_kernels.py:19-40` (`ensure_experts`, pass a `slot_base`, pre-seed `slot_for_id` for hot ids); `flashlib.kernels.slot_cache.lru_ensure` is external and is the real risk: it already takes `id_base`, a symmetric `slot_base` is small, but vendor the single Triton kernel if the repo can't be patched.

Avoid a per-row indirection by permuting the router projection's output rows at load time so expert ids are hotness-sorted per layer; hot experts are ids [0,K), cold experts map to host rows by constant subtraction.

Confidence: high on RAM saved; medium on tok/s (skew-dependent); medium effort dominated by the flashlib dependency.

### B. NVMe cold tier (park it)
Top-10 x 48 layers = 480 expert rows per decode token = 1.33 GB. At 50 tok/s (20 ms/step), serving all from disk needs 66 GB/s. A 990 PRO does ~7 GB/s sequential (~5.4 GB/s measured at 8 MB blocks; 4K QD1 latency ~131 us). A 20 ms step absorbs at most ~140 MB from SSD fully overlapped: ~50 of 480 rows, a 10.4% miss-to-disk ceiling, realistically <=5%. The pinned tier must still serve >=95% of routed rows; with fine-grained routing that plausibly needs 70-90% of experts resident, so host RAM 63.5 -> ~45-57 GiB. A 6-18 GiB saving for a large risky build with a speed cliff A doesn't have.

Windows cost: the upstream disk PLE store is Linux-only (`ple_store_ext.cpp` includes `dlfcn.h`, `unistd.h`, `fcntl.h`, io_uring guarded on `__linux__`). A Windows backend needs `CreateFile(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)`, sector-aligned buffers, IOCP or thread-pool `ReadFile`, plus graph-safe completion. GPUDirect Storage is not supported on Windows.

### C. Pageable banks (rule out)
The fused UVA gather needs a device alias (`kernel/pinned.py:59-68`) that unregistered memory lacks; `offload_cache.py:1232-1243` raises for decode on pageable. And it saves no RAM: the bank is a dirty pagefile-backed section, resident until Windows pages it out, at which point decode collapses (llama.cpp reports ~25-27 t/s falling to 7.55 t/s once expert pages hit swap). `HostBank.lock()` cannot help on Windows: `_os_lock` imports `resource` and calls libc `mlock` (`host_banks.py:174-197`).

### D. Managed / unified memory (rule out, high confidence)
CUDA Programming Guide for platforms without `concurrentManagedAccess` (Windows/WDDM): cannot allocate more managed memory than physical GPU memory; no fine-grained on-demand movement to GPU; page faulting only from the CPU side; all managed memory generally transferred to GPU on every kernel launch. A 63 GiB managed table on a 32 GiB card will not allocate.

### E. Fix F2, boot page-cache bloat (recommended, cheap insurance)
`drop_page_cache` (`models/loader.py:56-66`) calls `os.posix_fadvise`, stubbed to a no-op by `scripts/windows-ple-mmap/sitecustomize.py:52-56`. The parallel direct-I/O reader is gated by `_PARALLEL_READER_SUPPORTED = hasattr(os, "O_DIRECT") and hasattr(os, "preadv")` (`moe/expert_banks.py:31`), both false on Windows.

Right fix: never cache rather than evict after the fact. `FILE_FLAG_NO_BUFFERING` is an open-mode flag; there is no reliable retroactive purge. Add a Windows reader beside `read_file_into` / `read_range_into` (`host_banks.py:380-471`): `CreateFile` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, thread-pool `ReadFile` at absolute offsets. The bounce-buffer logic (`host_banks.py:446-459`) already handles alignment; only a `preadv` shim is needed.

Honest accounting: standby pages are reclaimable, so this mostly does not reduce steady-state usable RAM. It removes the 87-91-of-95.6 GiB boot peak (the thing most likely to make a pin fail), speeds boot, and reduces reclaim pressure on the PLE's 47.68 GiB mapping.

### F. Small wins
| Item | Saves | Cost | Where |
|---|---|---|---|
| Vision off | 0.84 GiB RAM | none if unused | launch flag `-EnableVision` |
| PLE row cache 1 Mi -> 256 Ki rows | ~0.21 GiB RAM | more mmap faults per PLE gather | `FREETOKEN_PLE_ROW_CACHE`, `weight.py:286-290` |
| `*_global` bank collapse | 0.18 GiB RAM + 0.05 GiB VRAM | kernel edit in `fused_nvfp4.py` | `_BANK_SCHEMAS`, `offload_cache.py:50-58` |
| `linear_state_cache_ratio` 2.0 -> 1.0 | ~0.2 GiB VRAM (~75 slots) | fewer concurrent GDN snapshots | `engine/config.py:358` |

The `*_global` collapse is also a performance cleanup: rows are 2,560 B and 5,120 B, below the 256 KiB threshold where `cudaMemcpyBatchAsync` degrades to a synchronous copy (`offload_cache.py:19-27`, documented -22% e2e on gpt-oss 2048tok). Caveat: the broadcast-redundancy claim was inherited from the audit and not verified numerically.

### G. Just lower `--moe-cache-size` (zero-code experiment)
If 0(a) holds, every 1,000 slots dropped frees 2.58 GiB VRAM and ~2.58 GiB host commit, for one flag. Also the calibration experiment that yields the tok/s-per-GiB exchange rate needed to judge A and B. Do this first.

## 2. Ranked recommendation for this box

| # | Action | Host RAM | VRAM | Risk | Effort |
|---|---|---|---|---|---|
| 0 | Measure (section 5): cache-size sweep + routing stats + VMMap | - | - | none | hours |
| 1 | Vision off; PLE row cache to 256 Ki | -1.05 GiB | 0 | ~none | flags only |
| 2 | Windows direct-I/O bank reader (E) | boot peak -~20 GiB | 0 | low | ~1-2 days |
| 3 | Static hot set, allocation-time (A) | -5 to -12 GiB | 0 | medium (skew) | ~1 week + flashlib |
| 4 | `*_global` collapse + `linear_state_cache_ratio` | -0.18 GiB | -0.25 GiB | low | ~1 day |
| 5 | NVMe cold tier (B) | -6 to -18 GiB | 0 | high | weeks |
| - | Pageable banks (C), managed memory (D) | rule out | | | |

Do not set `FREETOKEN_BANK_CUDA_ALLOC=1`: the ~50%-of-RAM Windows ceiling applies to `cudaHostAlloc`, whereas this box already registers 63.5 GiB (66%) via `cudaHostRegister` after fill. Keep `born_pinned_default()` False (`host_banks.py:64-71`). `_pin_budget_bytes` (`engine.py:1769-1780`) returns None on native Windows, so there is no automatic guardrail; set `FREETOKEN_PIN_BUDGET_GB` explicitly if wanted.

## 3. Implementation sketch, A (static hot set)
1. Profile: add `--moe-collect-decode-freq` (today `cache.collect_decode_freq` is set programmatically, `offload_cache.py:241`). Run representative prompts with CUDA graphs disabled (the histogram is a host-side `scatter_add_` in `ensure_experts`, `offload_cache.py:1031-1035`, skipped by graph replay). Dump `cache.decode_freq` `[48, 512]` next to the checkpoint.
2. Choose K per layer from the histogram; `decode_routing_stats()` reports `experts_for_90pct` per layer; per-layer miss rates are U-shaped so head/tail layers want larger K.
3. Permute the router projection rows into hotness order at load; store the permutation in a sidecar.
4. `_alloc_nvfp4_host_banks` takes E-K rows. Hot experts go via a small pinned staging buffer to `slot_pool[2E + L*K + e]`; cold experts to host row e-K.
5. `set_bank_sources` accepts `num_host_experts = E-K`; `_build_fused_copy_plan` unchanged; `ensure_experts` pre-seeds `slot_for_id` for hot ids and passes `slot_base = 2E + 48K` to `lru_ensure`.
6. Gate behind `FREETOKEN_MOE_HOT_SET`; steps 1-2 alone are a useful measurement tool.

## 4. Implementation sketch, E (Windows direct-I/O reader)
Add `python/freetoken/moe/win_io.py` exposing `preadv_into(handle, buffers, offset)` via ctypes `ReadFile` + `OVERLAPPED` over a handle opened `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN`. Select the backend at import in `host_banks.py`; make `drop_cache=True` a no-op on Windows; relax `_PARALLEL_READER_SUPPORTED` to `hasattr(os, "O_DIRECT") or os.name == "nt"`. Alignment: offset, length, and buffer address sector-aligned; `read_range_into`'s bounce path covers unaligned head/tail; 4096 covers any 990 PRO sector size.

## 5. Measure first
1. `--moe-cache-size` sweep at fixed context (6750 / 5750 / 4750 / 3750): tok/s, Windows Commit Charge, scheduler private commit. Gives the tok/s-per-slot curve and tests the WDDM backing-store hypothesis (commit drops ~2.58 GiB per 1,000 slots if true).
2. `decode_routing_stats()` with `collect_decode_freq=True`, graphs off, on 3-4 different workloads: `oracle_hit_at_slots`, `experts_for_90pct`, `working_set_mean/max`, `norm_entropy`, and top-K overlap across workloads (<60% overlap means A degrades and B's cold tail is thicker).
3. `--moe-collect-stats` -> `decode_miss_stats_per_layer()` on a real session; compare to `oracle_hit_at_slots`; read `prefetch_stats_summary()`.
4. VMMap on the scheduler PID: Private / Shareable / Mapped-file / Page-table.
5. RAMMap: Standby vs Active vs Modified during and after boot.

Least certain: the WDDM backing-store hypothesis for CUDA allocations under the NVIDIA driver; whether Qwen3-Next's router is skewed enough for A; the `*_global` redundancy claim.

## Sources
- [CUDA Runtime API, Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [CUDA Programming Guide, Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
- [Sharing the Backing Store with KMD](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/sharing-backing-store-with-kmd)
- [WDDM Residency Overview](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/residency-overview)
- [DiscardVirtualMemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-discardvirtualmemory)
- [Windows Memory Mapped File IO](https://www.jeremyong.com/winapi/io/2024/11/03/windows-memory-mapped-file-io/)
- [File Buffering](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)
- [File Caching](https://learn.microsoft.com/en-us/windows/win32/fileio/file-caching)
- [NVIDIA forums: 50% cudaHostAlloc limit on Windows](https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235)
- [GPUDirect Storage release notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html)
- [Fast Inference of MoE Language Models with Offloading (arXiv:2312.17238)](https://arxiv.org/pdf/2312.17238)
- [MoE-Infinity (arXiv:2401.14361)](https://arxiv.org/pdf/2401.14361)
- [fMoE (arXiv:2502.05370)](https://arxiv.org/html/2502.05370v1)
- [In-depth Analysis on Caching and Pre-fetching in MoE Offloading (arXiv:2511.05814)](https://arxiv.org/pdf/2511.05814)
- [llama.cpp #26110](https://github.com/ggml-org/llama.cpp/issues/26110)
- [llama.cpp #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- [StorageReview, Samsung 990 PRO 2TB](https://www.storagereview.com/review/samsung-990-pro-ssd-review-2tb)
- [TechPowerUp, Samsung 990 Pro 2TB](https://www.techpowerup.com/review/samsung-990-pro-2-tb/4.html)

## Addendum (same day)

1. **Strategy A's flashlib risk largely dissolves.** `flashlib==0.3.0` (pyproject.toml:41) is pure Python + Triton source at `%LOCALAPPDATA%\FreeToken\venv\Lib\site-packages\flashlib\kernels\slot_cache\triton\lru_ensure.py`. Victim selection goes through one helper, `_packed_keys` (lines 101-119): evictable slots pack to `(usage << SLOT_BITS) | slot`, non-evictable to INT64_MAX; today only `u == step` is non-evictable. A permanent pin is a one-line widening of that predicate (`& (c >= num_static)`), shared by both strategies. The `id_base` docstring already blesses split backing stores. Effort confidence: medium-high.

2. **Zero-patch prototype for A.** Seed `cache.usage[static_slots]` with a large sentinel (e.g. 2**40) and pre-seed `slot_for_id` for hot ids; those slots sort last in every victim scan. Caveat: a hit stores `step` into `lru_usage[s]` (lru_ensure.py:97), so this is a strong preference, not a guaranteed pin. Good enough to measure the hit-rate effect before patching.

3. **`serve-pr279-windows-safe.cmd` probably saves no RAM.** It passes `--moe-cpu-layers 0.5` (and `--kv-reserve-tokens 50000`). The residency split that would save RAM is gated on `split_residency`, which needs `_pin_budget_bytes(...)` non-None (engine.py:679-686), and that returns None on native Windows (engine.py:1774-1776, `if not hasattr(os, "uname")`). So `pin_banks` pins all 63.5 GiB anyway (host_banks.py:270-281), half the MoE layers run on the slower CPU executor, and prefill overlap is disabled. Confirm in the boot log (residency line, `prefill_overlap`). Either drop the flag or set `FREETOKEN_PIN_BUDGET_GB` explicitly; even then the "locked" half is pageable on Windows (`_os_lock` needs `resource`/`mlock`), i.e. strategy C.

   **Verified 2026-09-02 (code read, no boot).** Confirmed, with one correction.

   - `_pin_budget_bytes` (engine.py:1774-1782) returns `None` on native Windows: with
     `FREETOKEN_PIN_BUDGET_GB` unset the `elif not hasattr(os, "uname") or ...` arm is taken
     and returns `None`. CPython on Windows has no `os.uname`.
   - `split_residency` (engine.py:682-686) requires `_pin_budget_bytes(...) is not None`, so it
     is unconditionally `False` there. `requested_residency` therefore stays `None`
     (engine.py:716-723), `load_expert_banks(layer_residency=None)` reaches `pin_banks` with no
     plan, and `pin_banks` (host_banks.py:277-285) takes the `plan is None` branch: **every**
     layer bank is `PINNED`. No RAM is saved. Confirmed.
   - `--moe-cpu-layers 0.5` still resolves a real CPU layer set (`_resolve_cpu_layers`,
     engine.py:1735-1746) and sets `decode_target = "cpu"` (engine.py:674-679), so half the MoE
     layers do move onto the slower CPU executor. Confirmed.
   - **Correction: prefill overlap is NOT disabled.** The only site that clears
     `moe_prefill_overlap` for this reason is engine.py:701-707, and it is gated on
     `split_residency` — which is `False` on Windows. Grep confirms no other disable path
     (`moe_prefill_overlap` appears at config.py:330, engine.py:635/701/707/737/760/1821/2110,
     args.py:596). So the flag costs the CPU-executor slowdown and buys nothing, but it does not
     also cost overlap.
   - Setting `FREETOKEN_PIN_BUDGET_GB` would not rescue it: the "locked" half goes through
     `_os_lock` (host_banks.py:184-207), which does `import resource` — absent on Windows — so
     the `except (OSError, ImportError)` at host_banks.py:173-176 leaves those banks *pageable*,
     not locked. That is strategy C, with all of its paging risk.
   - No boot log for this launcher exists in any scratchpad on this box (searched for
     `moe_cpu_layers='0.5'` and the "split residency" log line); the verdict is code-only.
   - Also note the file's `PYTHONPATH` points at `D:\FreeToken-ple-mmap`, which does not exist
     on this box — the launcher could not have run as written.

   Action taken: `--moe-cpu-layers 0.5` removed from `serve-pr279-windows-safe.cmd` with a
   comment recording why.
