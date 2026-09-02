# Qwen3.8 KV host tier: keep the per-token K/V in pinned host RAM and gather the QSA selection over PCIe

## Status

Design draft, 2026-09-02. Nothing implemented. Written read-only against `mtp-upstream-merge` at `8591caf` while another agent owns the GPU for server reboots and a third edits code in a separate worktree. Source claims carry `file:line`. The PCIe numbers were measured on this box by a microbenchmark run against the live serving GPU (script, raw JSON, and the exact conditions are recorded below and in the session scratchpad); everything else comes from `/v1/cache/status`, `results/benchmark-262144.json`, or `docs/research/measurements-moe-cache-sweep-2026-09-02.md`.

**Read the "Verdict" section first.** The measurements say the mechanism is sound and cheap; whether it is *worth it* turns almost entirely on one unmeasured quantity — how much the QSA block selection overlaps between consecutive decode steps — and on whether this box has the host RAM to hold the tier at the context where it pays. Both are addressed below with published comparanda and a concrete first experiment.

## Purpose

At 262,144 tokens the KV pool is **6.19 GiB** of a 32 GB card, and it is taken directly out of the MoE expert slot cache: `plan_cache_budget` splits one budget between the two (`python/freetoken/engine/cache_budget.py:47-92`), and the measured consequence is that `moe_cache_size` falls from **6,750** at 65,536 tokens to **4,063** at 262,144 (`/v1/cache/status` vs `results/benchmark-262144.json:cache_status_before.geometry`), which the sweep prices at roughly **14 tok/s of decode** (`docs/research/measurements-moe-cache-sweep-2026-09-02.md`).

But QSA does not read most of that cache. Per query row, per layer, the indexer scores one compressed key per 4-token block and attends to the top `index_budget // index_ratio = 2048 // 4 = 512` blocks — **2,048 tokens**, regardless of how long the context is (`python/freetoken/attention/qsa_sparse.py:186-190`). So at 262k, a decode step touches **0.78%** of the K/V it is paying 6 GiB of VRAM to hold.

The proposal is the InfiniGen shape: keep the per-token K/V slab in **pinned host RAM**, keep on the card only the compressed index slab (which *is* scanned in full every step), the rings, a small write window, and a staging buffer for the selection; and gather the selected rows over PCIe each layer with the same UVA kernel the MoE expert cache already uses.

## Non-goals

- Changing numerics. A host-tier decode must produce bit-identical logits to a device-resident decode.
- Touching the GDN layers, the linear state pool, the MoE cache, the PLE path, or the prefix cache. Prefix parking is the companion design (`2026-09-02-qwen38-kv-prefix-parking-design.md`) and is **not** a prerequisite for this one.
- Moving the **compressed index slab** off the card. It is read in full every step by `qsa_mqa_paged` and is only 768 B/token (192 MiB at 262k); streaming it would be exactly the wrong trade.
- Concurrency. `max_running_req` is 1 under MTP (`python/freetoken/engine/config.py:299-303`).
- Quantizing the KV cache. That is a rival lever, priced in the options section, not part of this design.
- Non-QSA models. `MHAKVCache`, `DSAKVCache`, `DSV4PagedPool`, and the SWA pools keep their current behavior unchanged.

## Current behavior

### What the QSA layer actually does, per forward

`QSASparseAttnBackend.qsa_forward` (`attention/qsa_sparse.py:431-470`), for each of the 12 QSA layers:

1. `self.kvcache.store_kv(k, v, batch.out_loc, layer_id)` — writes this forward's K/V into the paged slab at absolute token slots (`kvcache/mha_pool.py:115-133` → `kernel/store.store_cache`, which takes the slab as a flat `[num_tokens, feature]` view plus an index vector).
2. `_plan_index_writes` / `_update_index_cache` — compress each closing 4-token group into the index slab and refresh the pending ring.
3. `_select` — score, top-k, expand (below).
4. `qsa_sparse_paged_attention(q, k_cache(layer_id), v_cache(layer_id), indices, md.block_table, md.token_to_req, out)`.

### The selection, verified

This is the crux, so it is worth being exact.

- **`_select`** (`qsa_sparse.py:547-607`) runs `qsa_mqa_paged` over the compressed slab viewed as `[pages, page_size//ratio, 1, index_head_dim]` = `[pages, 16, 1, 128]` (`qsa_sparse.py:229-234`), producing `logits` of shape `[rows, columns]` where `columns = block_table.shape[1] * 16` — one score per **4-token block**, in **logical block index space** for the request.
- The score is `sum_h relu(<q_h, k_bar_b>) / sqrt(index_head_dim)` — **summed over the indexer heads** (module docstring, `qsa_sparse.py:22-24`). So the per-head scores are collapsed before the top-k.
- `_top_blocks` (`qsa_sparse.py:638-666`) takes the top `block_topk = 512` per **row** into `blocks` of shape `[rows, 512]`.
- `expand_qsa_block_indices` (`kernel/triton/qsa/expand.py:85-115`) turns those into `indices` of shape `[rows, select_width]` where `select_width = token_topk + ratio - 1 = 2051` — logical **token** indices, plus the causal tail of the currently-open group.

**Therefore: the top-k is per query row (per token), shared across all 16 query heads and both KV heads, and computed independently per layer.** The attend kernel confirms it: `logical_token = tl.load(indices_ptr + row * stride_indices_row + columns)` has no head term (`kernel/triton/qsa/attend.py:76-81`), while `kv_head` is `tl.program_id(1)`. There is exactly **one 2,048-token set per (layer, query row)** — 12 sets per decode step at batch 1.

### How the attend kernel addresses memory — and why this design is possible

`_qsa_sparse_paged_gqa_splitk_kernel` (`kernel/triton/qsa/attend.py:76-118`) does, for each selected logical token:

```
logical_page   = safe_token // PAGE_SIZE                     # PAGE_SIZE is a constexpr from k_cache.shape[1]
page_offset    = safe_token %  PAGE_SIZE
physical_page  = block_table[safe_request, logical_page]     # int32 indirection
keys           = k_cache_ptr + physical_page * stride_k_block
                             + page_offset  * stride_k_token
                             + kv_head      * stride_k_head + dims
```

Every address is derived from **`k_cache`'s own strides and shape** and from a **caller-supplied indirection table**. `PAGE_SIZE` and `PAGE_TABLE_WIDTH` are `tl.constexpr` taken from `k_cache.shape[1]` and `block_table.shape[1]` (`attend.py:317-320`). The kernel has no idea what the slab is, how big it is, or where it lives.

**So a staged slab of the selected blocks plus a remapped indirection table is enough, with no change to the attend kernel at all.** That is the single most important finding in this document.

The same is *not* true of the score kernel: `qsa_mqa_paged` reads the **compressed** slab through the **same `md.block_table`** (`qsa_sparse.py:576-591`, `kernel/triton/qsa/score.py:151-178`). Since the compressed slab stays device-resident with its real physical page ids, the two kernels need **two different tables** after this change. That is a metadata change, not a kernel change.

### Where the block table comes from, and its graph discipline

`md.block_table[req, p] = page_table[req, p * 64] // 64` (`qsa_sparse.py:372-380`). In decode it is staged into a **static** graph buffer by `_stage_decode` (`qsa_sparse.py:382-394`) so the captured graph reads one fixed address, and `prepare_for_replay` refreshes it every step (`qsa_sparse.py:749-753`). Host-side page-table writes happen in `_write_page_table` (`scheduler/cache.py:718-743`) during scheduling, outside any capture. Any new table this design introduces must follow the same pattern: **a fixed-address static buffer, filled by a device kernel during replay.**

### What the pool costs today

`/v1/cache/status`, live: `kv_per_token = 25,344`, `page_size = 64`. Decomposed against `kvcache/base.spec_kv_bytes_per_token` (`kvcache/base.py:19-37`) and `kvcache/qsa_pool.py`:

| tier | shape | B/token | B/token/layer | scanned per step |
|---|---|---|---|---|
| paged K/V | `[2, 12, P, 64, 2, 256]` bf16 (`mha_pool.py:44-52`) | 24,576 | 2,048 | **2,048 tokens (0.78% at 262k)** |
| compressed index | `[12, P*16 + slots, 128]` bf16 (`qsa_pool.py:128-135`) | 768 | 64 | **all of them** |
| pending rings + scratch | fixed, `num_req_slots`-sized (`qsa_pool.py:136-149`) | 0 | — | all |

Fixed cost of the rings, from `QSAKVCache.kv_cost` (`qsa_pool.py:186-208`) at `num_req_slots = 2`, `ring_capacity = 4`: `2 × (128×12×2) × 5 + 2×12×4×3×8` = **33,024 B**. Negligible.

At 262,144 tokens: K/V **6.00 GiB**, index **192 MiB**, total 6.19 GiB — matching `4097 pages × 64 × 25,344 = 6.19 GiB`.

## Measurements

### The microbenchmark

Run 2026-09-02 against the live GPU while `/v1/cache/status` reported `state: "serving"` (`num_pages` 1024, `moe_cache_size` 6750) and `nvidia-smi` showed 4,003 MiB free. Desktop venv, `PYTHONPATH="D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python"`. Device footprint kept under 128 MiB; host source is a 256 MiB `cudaHostAlloc` pinned buffer (`kernel/pinned.alloc_pinned_tensor`). Medians of 100 CUDA-event-timed iterations after 10 warm-ups. Script `bench_kv_host_gather.py`, raw output `kv_gather_results.json`, both in the session scratchpad.

Two gather paths were compared at three granularities, each moving one QSA layer's worth of selection:

- **`uva`** — the production gather, `kernel/fast_index_copy.fast_index_copy_multi_jit`, reading the pinned host buffer's **device alias** (`kernel/pinned.device_ptr`). This is byte-for-byte the mechanism `moe/offload_cache.copy_missing` uses for expert rows (`offload_cache.py:1244-1259`). (Note: on this box `device_ptr(host) == host.data_ptr()`, i.e. UVA pointer identity holds; the Windows/WDDM remap path documented at `pinned.py:59-68` is not exercised here.)
- **`cpuidx`** — the naive baseline: `torch.index_select` on the pinned host tensor into pinned staging, then one `non_blocking` H2D.

| granularity | rows | row bytes | bytes moved | `uva` µs/layer | `uva` GB/s | `uva` µs/step (×12) | `cpuidx` µs/layer | `cpuidx` GB/s |
|---|---|---|---|---|---|---|---|---|
| **token** (1 tok, K+V) | 2,048 | 2,048 | 4 MiB | **107.7** | 38.9 | **1,293** | 179.9 | 23.3 |
| **block** (`index_ratio` = 4 tok) | 512 | 8,192 | 4 MiB | **90.6** | 46.3 | **1,088** | 177.6 | 23.6 |
| **page** (`page_size` = 64 tok) | 128 | 131,072 | 16 MiB | **372.3** | 45.1 | **4,467** | 883.3 | 19.0 |

Contiguous reference copies, same run:

| copy | bytes | µs | GB/s |
|---|---|---|---|
| H2D memcpy | 4 MiB | 78.6 | 53.4 |
| H2D memcpy | 48 MiB | 894.2 | 56.3 |

And, from a second script (`bench_d2h.py`, medians of 100), the link in both directions:

| size | D2H µs | D2H GB/s | H2D µs | H2D GB/s |
|---|---|---|---|---|
| 4 MiB | 74.4 | 56.4 | 74.8 | 56.0 |
| 48 MiB | 878.4 | 57.3 | 871.6 | 57.7 |
| 128 MiB | 2,343.1 | 57.3 | 2,321.3 | 57.8 |

A repeat of the gather benchmark taken later, when the GPU had gone idle (30,991 MiB free — the server had been stopped), came out 10-25% *slower* (block granularity 113.1 µs/layer, 37.1 GB/s), which is the idle-clock penalty, not contention. **The serving-GPU numbers in the table above are the ones to design against**; the idle run is recorded in the scratchpad as a variance check.

### What the microbenchmark says

1. **The UVA gather reaches 82-87% of contiguous memcpy bandwidth** at 8 KiB rows, and the scattered gather of 512 blocks (90.6 µs) is barely slower than a contiguous 4 MiB memcpy (78.6 µs). Scatter is essentially free on this link.
2. **8 KiB rows are already at the knee.** The 128 KiB page rows achieve 45.1 GB/s — *no better* than 8 KiB rows' 46.3 GB/s. So **there is no bandwidth argument for fetching whole 64-token pages**, and every argument against it (see below). The fetch unit should be the 4-token index block.
3. **Token granularity costs 19%** relative to block granularity (107.7 vs 90.6 µs). Since the selection is inherently block-aligned, that penalty is avoidable for free.
4. **The naive CPU-gather path is 2x worse** and would also burn a CPU core per layer per step. The UVA kernel is the right mechanism, and it already exists.
5. **A full-miss step costs 1.09 ms** across 12 layers at block granularity — against a decode step of 16.9 ms at 262k (`results/benchmark-262144.json:summary.context_check.mean_tpot_milliseconds` = 18.1 ms; `summary.long` = 17.1 ms), i.e. **+6.4%**.

### Why page granularity is not an option

512 selected 4-token blocks, drawn from `P` pages, touch an expected `P·(1 − (1 − 1/P)^512)` distinct pages under a uniform model: **403 of 1,024 pages at 65k**, **481 of 4,096 at 262k**. At 64 tokens × 2,048 B that is **50-60 MiB per layer**, **600-720 MiB per step**, ~11-13 ms of PCIe. Real selections are clustered, not uniform, so the true figure is lower — but not by the order of magnitude that would be needed. Page-granular fetching is off the table unless an on-card page cache absorbs nearly all of it, and the 8 KiB-row measurement removes the only reason to want it.

## Prior art, and the one number this design turns on

A web survey was run alongside this design (2026-09-02; the full report is in the session transcript, sources linked inline below). Three findings matter enough to change the design.

### 1. The upstream tiering split is exactly this one

The Transformers `qwen4_exp` documentation states: *"With cache offloading, **GatedDeltaNet, PLE, and QSA indexer states remain on device** while attention key/value states are offloaded."* ([HF docs](https://huggingface.co/docs/transformers/main/en/model_doc/qwen4_exp)) That is precisely the split proposed here — index slab and GDN state resident, K/V offloaded — which is reassuring about the shape and means the model authors expect it to be viable. The same page also warns that *"cache cropping is not supported when PLE or QSA is enabled"*, which is a constraint the companion parking design has to check, not this one.

It also confirms the model constants independently of the repo: `indexer_n_heads` 4, `indexer_kv_heads` 1, `indexer_head_dim` 128, `indexer_budget` 2048, `indexer_compress_ratio` 4 — matching `args.index_n_heads`, `args.index_budget`, and `spec.index_ratio` as this repo names them, and confirming *"sparse attention touches at most 2051 positions"*, which is `select_width` exactly.

### 2. The top-k really is shared across heads, and that is a structural advantage

DeepSeek's NSA paper makes the point that decides fetch amplification: *"For models employing GQA or MQA where key-value caches are shared across query heads, **consistent block selection across these heads has to be ensured** to minimize KV cache loading during decoding"* ([arXiv:2502.11089](https://arxiv.org/pdf/2502.11089) §3.3.2), and criticizes Quest for selecting per query head, where *"the memory access volume corresponds to the **union** of selections from all query heads"*. SGLang's QSA writeup gives the aggregated form for this model: `s_{t,b} = 1/√128 · Σ_{h=1..4} ReLU(⟨q^I_{t,h}, k̄^I_b⟩)` ([LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)) — one score per block, one top-512 per row.

That matches the source reading above and means **one 4 MiB gather serves all 24 query heads of a layer**. ShadowKV selects per KV head; Quest per query head. QSA is on the favourable side of this, and only 12 of 48 layers have a growing KV cache at all.

### 3. Consecutive-step selection overlap: the crux, and 4-token blocks are the unfavourable end

No one has published an overlap figure for 4-token-block selection, and no one has published anything for QSA. What is published, sorted by granularity:

| source | granularity | consecutive-step overlap |
|---|---|---|
| Levy, [arXiv:2603.13430](https://arxiv.org/pdf/2603.13430) | **token**-level top-k (64-256) | **~45%** (new lookups 0.55 of top-k, P95 0.90) |
| Guess-Verify-Refine, [arXiv:2604.22312](https://arxiv.org/html/2604.22312v1) | token-level, DeepSeek-V3.2 DSA | **35-50%** for layers 20-60; **1-2%** for layers 0-1 |
| ShadowKV, [arXiv:2410.21465](https://arxiv.org/abs/2410.21465) §3.2, §5.3 | **8-token** chunks | **~60%** ("chunk hit rate … remains around 60%") |
| PRR, [arXiv:2606.30389](https://arxiv.org/html/2606.30389) | 16-64-token blocks (Quest, InfLLM-V2) | **63.6-69.7%**, mean ~68% |
| NOSA, [arXiv:2510.13602](https://arxiv.org/pdf/2510.13602) | 64-token blocks (InfLLMv2) | **ρ ≥ 0.8 for most layers**, emergent, untrained |

**Overlap rises monotonically with block size, and QSA's 4-token blocks sit below every published measurement.** The honest planning assumption is therefore **h ≈ 0.5-0.6**, ShadowKV's 8-token figure or a little worse — not the 0.8 that the block-sparse papers report at 64 tokens.

Two counterweights. InfiniGen (OSDI '24, [arXiv:2406.19707](https://arxiv.org/abs/2406.19707)) explicitly argues the *opposite* — *"the tokens deemed unimportant in the current iteration could become important in subsequent iterations"* — and builds a cross-*layer* predictor instead (cosine similarity 0.89-0.97 between consecutive layers' block inputs on OPT/Llama-2). But Levy measured **inter-layer overlap of only 0.36** on modern indexer-based sparse attention and concludes *"the previous layer is a very poor predictor of the next layer, with less than 50% overlap"*. **So cross-layer prefetch is not available to this design**, which independently confirms the "the gather cannot be overlapped" conclusion reached from the source above. Note also that InfiniGen's 1.6-33x speedups are on **PCIe Gen3** against baselines whose step time was 92-97% data transfer; on Gen5 with 0.78% of the cache selected, this design starts from a far better place.

### 4. Size the on-card cache *above* the per-step budget, not at it

This is the finding that changes the recommendation, and it comes from three independent systems:

- **HiSparse** ([arXiv:2608.07009](https://arxiv.org/html/2608.07009)) keeps the full KV in host DRAM and an **LRU-managed bounded working set** of size `B` on the GPU. Measured on LongBenchV2 at top-k 2048: **13.4% miss at `B` = 2k (i.e. `B = k`)**, **87% hit at `B` = 4k (`B = 2k`)**, versus FIFO 17.2%, random 16.1%, and **naive top-k-only staging 30% miss**. LRU *"tracks the trend of the offline Bélády optimum."*
- **ArkVale** (NeurIPS 2024, [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/cd4b49379efac6e84186a3ffce108c37-Paper-Conference.pdf)) Fig. 4b: average page recalls per decoding step falls **17.2 → 6.5 → 4.1 → 2.6 → 2.0 → 1.5 → 1.2 → 1.0** as GPU cache capacity goes 4 → 6 → 8 → 12 → 16 → 32 → 64 → 128 pages. The knee is at ~2x the budget; past that it is flat.
- **Levy** measured the **working set of a 50-token generation burst at 5.15x top-k** (P95 7.2x, σ 1.02) — a direct sizing constant.

Together: a cache of **2-5x the per-layer budget** captures nearly all the available reuse, and more is wasted. For QSA that is `2048 tokens × 2,048 B × {2..5} × 12 layers` = **96 to 240 MiB**, which at 2,772,480 B per expert slot costs only **35 to 91 slots** — 0.2 to 0.5 tok/s. **The cache is essentially free, and the earlier instinct to size it in gigabytes was wrong.**

ShadowKV's own cache, read out of its source, is instructive as the *minimum* viable design: it has no separate cache and no eviction policy at all — the "cache" is last step's 256 chunks, and each step builds a shared-memory hash map of the previous step's chunk ids, reorders hits to the front, and PCIe-fetches the rest. That is `B = k` with a 1-step history, and it gets 60%. HiSparse's LRU at `B = 2k` gets 87%. The gap between those two is what a real LRU buys.

### 5. Fine-grained transfers are the known failure mode — and the measurement here says we are clear of it

NOSA reports that *"element-wise communication degrades sharply under high locality, whereas **block-wise** communication sustains high throughput"*, and that their custom Triton + UVA gather reaches **83% of peak PCIe** on 16 KiB blocks. The microbenchmark above reaches **82% of peak** (46.3 of 56.3 GB/s) on **8 KiB** blocks with the kernel this repo already ships. That is an unusually clean independent corroboration, and it says the fetch mechanism is not where this design will fail.

KVSwap ([arXiv:2511.11907](https://arxiv.org/pdf/2511.11907)) gives the corresponding SSD number: **1.8 GB/s NVMe random read**, with *"bandwidth utilization improv[ing] substantially with larger block sizes"* because of NAND read amplification. That is the quantitative reason option (c) below is rejected.

### The VRAM-for-PCIe exchange rate

This is the whole economics of the design, and it is measurable from data already committed to the repo.

Moving the K/V slab off the card frees `T × 24,576` bytes (the index slab stays). Each MoE expert slot is `2,772,480` B (`/v1/cache/status:unit_bytes.moe_per_expert`), so:

> **slots gained = T × 24,576 / 2,772,480 = T × 0.008865**

The sweep (`measurements-moe-cache-sweep-2026-09-02.md`, 8k-chat column) gives the decode value of a slot: 2750→49.9, 3750→56.5, 4750→64.9, 5750→68.4, 6750→73.1 tok/s. Two independent checks that this curve transfers: the 262k benchmark ran at `moe_cache_size` 4,063 and measured **59.1 tok/s** on an 8,204-token prompt (`summary.long`), while linear interpolation of the sweep between 3750 and 4750 predicts **59.1**. The curve is trustworthy.

A 240 MiB on-card LRU block cache (5x the per-layer budget, per the sizing evidence above) costs 91 slots and scales the gather to `1.09 × (1 − h)` ms.

| context `T` | KV VRAM freed | net slots gained | slots before → after | tok/s (sweep) before → after | step ms before → after | net tok/s, `h`=0 | `h`=0.6 | `h`=0.87 | best-case vs. today |
|---|---|---|---|---|---|---|---|---|---|
| 65,536 | 1.50 GiB | 490 | 6,750 → 7,240 | 73.1 → ~75.9 | 13.68 → 13.17 | 70.1 | 73.5 | 75.1 | **+2.7%** |
| 131,072 | 3.10 GiB | 1,071 | ~5,855 → 6,926 | ~68.9 → ~74.1 | 14.51 → 13.50 | 68.5 | 71.7 | 73.3 | **+6.4%** |
| 262,144 | 6.00 GiB | 2,233 | 4,063 → 6,296 | 59.1 → ~71.0 | 16.92 → 14.09 | 65.9 | 68.8 | 70.3 | **+18.9%** |

(The 65,536 and 131,072 "after" points extrapolate past the sweep's top measured point at +5.8 tok/s per 1,000 slots, the sweep's own mean. `h`=0.6 is the honest planning assumption from the locality survey; `h`=0.87 is HiSparse's measured LRU figure at `B` = 2k, at a coarser granularity than ours.)

**Break-even, as a function of the hit rate.** At 73.1 tok/s a step is 13.68 ms, and buying back `1.09 × (1 − h)` ms requires `Δ` tok/s such that `1000/(73.1 + Δ) = 13.68 − 1.09(1 − h)`, with `Δ = (T × 0.008865 − 91) × 5.8 / 1000`:

| `h` | gather ms/step | break-even context |
|---|---|---|
| 0 (no cache) | 1.09 | **~133,000 tokens** |
| 0.5 | 0.55 | **~70,000 tokens** |
| 0.6 | 0.44 | **~59,000 tokens** |
| 0.87 | 0.14 | **~24,000 tokens** |

**So the hit rate is the design.** With no cache it only pays past 133k, where this box has no RAM for it. At the honest `h` ≈ 0.6 it pays from about 59k — i.e. from today's boot context — and returns +19% at 262k. Measuring `h` is therefore the first thing to do, before any of this is built (measurement plan step 0).

### The host RAM problem, stated plainly

From `docs/research/memory-audit-qwen38-rtx5090.md` and the sweep:

- Host resident sits at **82-95 GiB of 95.6 GiB** and does not move with `--moe-cache-size`. The 63.46 GiB pinned expert bank is fixed.
- The engine's own free-host-memory estimate with the server up is **~8 GiB** (`models/qwen4_exp/weight.py:266-270`), and that memory is not idle: it is the standby list backing the **47.68 GiB memory-mapped PLE n-gram table** that every decoded token gathers rows from.
- The sweep's central finding: on Windows/WDDM **every GPU byte costs one byte of system commit and ~zero bytes of resident physical RAM**.

So this design **converts 6.00 GiB of commit-only VRAM into 6.00 GiB of resident, pinned host RAM** — and then hands the freed VRAM to 2,324 expert slots, which re-add 6.00 GiB of commit. Net commit: unchanged. Net resident: **+6.00 GiB, on a box that has ~8 GiB of headroom and needs that headroom for PLE.**

| context | host tier (pinned) | vs. ~8 GiB headroom |
|---|---|---|
| 65,536 | 1.50 GiB | comfortable — but the design *loses* speed here |
| 131,072 | 3.10 GiB | tight, PLE paging likely rises |
| 262,144 | 6.00 GiB | **infeasible** — and this is where the +19% lives |

That is the central tension of this design on this machine.

## Verdict

**The mechanism works and the kernel change is smaller than expected — the attend kernel needs no edit at all. The PCIe cost is 1.09 ms/step uncached, measured, and a ~240 MiB on-card LRU cache that costs 91 expert slots should cut that by half or more.** At an assumed 60% hit rate the design pays from about 59,000 tokens of context and returns **+19% decode at 262k**.

Two things stand between that and a build:

1. **The hit rate is unmeasured, and QSA's 4-token blocks sit below every published overlap figure.** Everything above is a prediction keyed on `h`. The experiment that settles it is a shadow counter costing one Triton kernel and one boot — do it first.
2. **At 262k, where the win is largest, this box does not have the host RAM.** The tier is 6.00 GiB of *resident, pinned* host memory against ~8 GiB of headroom that the memory-mapped PLE table needs. Below ~131k it fits; at 262k it does not, and freeing the VRAM to expert slots re-adds the commit it removed.

So: the design is sound and probably worth building, but **the prefix-parking design should still be built first** — it wins on this box today, at today's context, with none of this one's kernel or graph risk (see the comparison at the end). And **FP8 KV quantization should be tried before either**: it halves the tier, halves the gather, and improves every row of the exchange-rate table.

## Options considered

### (a) Full host tier, block-granular gather, no on-card cache — the fallback

Host holds all `T × 24,576` bytes. Every step, every layer gathers its 512 selected blocks (4 MiB) from host into a per-layer staging slab and attends against it with a remapped table. Fixed, predictable **+1.09 ms/step**, no cache to tune, no hit-rate risk, and it frees the maximum VRAM.

This is the simplest thing that works, and it is the right *first* implementation because it is the correctness baseline the cached version must match bit-for-bit. But it only pays past ~133k tokens, so it is not the shipping configuration.

### (b) Host tier + a *small* on-card LRU block cache — RECOMMENDED

Add a `flashlib` `lru_ensure`-managed slot cache over 4-token blocks, per layer, exactly the shape of `moe/offload_cache.py` (`ensure_experts` → `copy_missing`, `moe/offload_kernels.py:19-40`). Gather only the missing blocks.

The sizing evidence above (HiSparse's `B = 2k` knee, ArkVale's Fig. 4b flattening past ~2x, Levy's 5.15x working set) says **2 to 5 times the per-layer budget** and no more:

| cache size | per layer | tokens/layer | expert slots cost | tok/s cost | needs `h` > |
|---|---|---|---|---|---|
| 96 MiB (2x) | 8 MiB | 4,096 | 35 | ~0.2 | 3% |
| 240 MiB (5x) | 20 MiB | 10,240 | 91 | ~0.5 | 8% |
| 1 GiB (21x) | 85 MiB | 43,700 | 387 | ~2.2 | 33% |

A `C`-byte cache costs `C / 2,772,480` slots; near 6,300 slots the sweep's marginal value is ~4.7 tok/s per 1,000 slots, and at a 14.1 ms step 1 tok/s is ~0.2 ms. So the cache pays iff `h × 1.09 ms > C_GiB × 0.32 ms`. **At 240 MiB the bar is h > 8%, which every published system clears by a wide margin.** The earlier instinct to spend gigabytes here was wrong: past ~5x the budget the extra VRAM is worth more as expert slots than as cache.

The cache must be **per layer** — each layer selects independently — and its `ensure` must run *after* `_select` and *before* the gather, rewriting the block ids to slot ids in place, which is exactly what `lru_ensure` does for expert ids today (`moe/offload_kernels.py:28-40`). It also removes the need for a separate address table: see "The block cache is the block table" below.

**Still: do not skip the overlap measurement.** If `h` comes back near ShadowKV's 60% the design ships; if it comes back near Levy's token-level 45% it still ships (the bar is 8%); if it comes back near GVR's 1-2% for the first layers, those layers should be exempted from the cache — Quest skips its first two layers for the same reason.

### (c) SSD-backed cold tier with a pinned host window — REJECTED for the decode path

The obvious answer to the RAM problem, and the one the PLE table already uses (`weight.py:426-441` mmap + `weight.py:279-300` `PrefetchVirtualMemory` + `ple.py:47-50` pinned ring). It does not survive contact with the decode step:

- The UVA gather needs a **device alias**, which requires `cudaHostRegister`ed (pinned, resident) memory. mmap'd file pages must be read into the pinned window first — a two-hop with the SSD in the critical path.
- 512 scattered 8 KiB reads per layer at QD1 is the worst possible SSD access pattern. The memory-reduction study measured ~131 µs 4K QD1 latency on this drive (`docs/research/memory-reduction-options-qwen38-rtx5090.md:29`); even at high queue depth, 4 MiB of scattered 8 KiB reads per layer, 12 layers, every 15 ms, is a sustained ~3.2 GB/s of *random* I/O. The drive does ~5.4 GB/s at 8 MB blocks and far less at 8 KiB. KVSwap measured **1.8 GB/s NVMe random read** for exactly this access pattern and reports that *"bandwidth utilization improves substantially with larger block sizes"* because of NAND read amplification ([arXiv:2511.11907](https://arxiv.org/pdf/2511.11907)).
- And the SSD is already serving the PLE table at a measured 3.1-3.3 GB/s peak (`results/benchmark-262144.json`).

SSD is right for *cold, whole-prefix* movement (that is the parking design) and wrong for *per-step, scattered* movement.

### (d) FP8 KV quantization instead — the rival lever, and probably the better one

Halve `24,576 → 12,288` B/token with an FP8 K/V slab. At 262k that frees **3.0 GiB of VRAM** (+1,162 slots, ~+4.7 tok/s by the sweep) for **zero PCIe traffic, zero host RAM, no kernel-addressing change, no graph risk**, and it makes every row of the exchange-rate table above better if this design is later built on top. Its cost is an accuracy question, not an engineering one, and the literature puts FP8 KV loss near zero.

It is not the same feature — it does not raise the ceiling on context the way a host tier does — but for the specific goal "get expert slots back at 262k on this box", it is cheaper, safer, and available now. **A design doc that did not say so would be dishonest.**

### (f) Compute the sparse attention on the CPU instead of moving K/V — noted, not recommended

Fluxion ([arXiv:2605.07719](https://arxiv.org/pdf/2605.07719)) inverts the problem: keep K/V in host DRAM and compute the sparse attention *on the CPU*, returning only the attention **outputs** across PCIe. Nothing but a `[24, 256]` bf16 output per layer crosses the bus — 12 KiB per layer instead of 4 MiB. Reported 1.9-3.7x TPOT over the best fixed sparse hybrid on an A100 + PCIe 4.0 + 24-core EPYC with 20 cores dedicated.

Not recommended here for two reasons specific to this box: this repo's CPU cores are already the hybrid MoE backend's escape hatch (`moe_cpu_layers`, `moe_hybrid_max_fetch`, `engine/config.py:341-350`), so the cores are spoken for; and PCIe Gen5 at 57 GB/s makes the 4 MiB gather cheap enough (90.6 µs) that the CPU-compute alternative has much less to beat than it did at Gen4. Worth revisiting only if the gather turns out to contend badly with the MoE path for the link.

### (e) Lower the context instead — the honest baseline

The server is booted at 65,536 tokens today and gets 73.1 tok/s. The 262k configuration is a capability, not a default. If long context is rare, none of this is worth building.

## Recommended design

**Option (b): host tier + a per-layer LRU block cache sized at 2-5x the per-step budget.** Build option (a) first as the correctness baseline (`FREETOKEN_KV_HOST_CACHE_MIB=0`), then turn the cache on; the two must produce bit-identical output.

### Flag surface

| flag | default | meaning |
|---|---|---|
| `FREETOKEN_KV_HOST_TIER` | `0` | master switch; `0` is byte-identical to today |
| `FREETOKEN_KV_HOST_CACHE_MIB` | `240` (**288 when MTP is on**) | on-card LRU block cache, all layers (5x the per-step budget); `0` = option (a), uncached. The MTP floor is forced by `cache_size >= _MAX_MTP_VERIFY_WIDTH x block_topk` — see failure modes. |
| `FREETOKEN_KV_HOST_CACHE_SKIP_LAYERS` | `""` | QSA layer ids exempted from the cache (see GVR's 1-2% overlap for the first layers; Quest skips its first two) |
| `FREETOKEN_KV_HOST_STAT` | `0` | shadow counters: selected blocks, distinct blocks, inter-step overlap, cache hit rate, `-1` remap reads |

### Memory map after the change

At `T` = 262,144 tokens:

| what | where | bytes |
|---|---|---|
| paged K/V slab `[2, 12, P, 64, 2, 256]` bf16 | **pinned host**, `HostBank` (`moe/host_banks.py:84-125`) | 6.00 GiB |
| compressed index slab | device, unchanged | 192 MiB |
| pending rings + scratch | device, unchanged | 33 KB |
| **write window**: 2 pages/layer, K+V — 32 *reserved slots inside the cache*, not a separate buffer | device (new) | (3.0 MiB of the row below) |
| **LRU block cache** `[num_slots, 4, 2, 256]` bf16 × 2 (K,V) × 12 layers — this is also the staging slab | device (new) | 240 MiB |
| **`slot_for_id`** — doubles as the attend kernel's block table | device (new) | 6.0 MiB |
| `id_of_slot` / `usage` / miss plan | device (new) | ~0.2 MiB |

On-card total: **~444 MiB** with the 240 MiB cache (492 MiB at the 288 MiB MTP floor), ~250 MiB uncached — down from 6.19 GiB. Freed: **5.75 GiB** (cached) or **5.95 GiB** (uncached).

The staging slab and the cache are the *same* buffer: `lru_ensure` rewrites the selected block ids to cache slot ids in place and `copy_missing` fills only the misses — exactly how `ensure_experts` → `copy_missing` → GEMV composes on the MoE path (`moe/offload_kernels.py:19-40`, `moe/offload_cache.py:1228-1272`).

### The block cache is the block table

The block cache is shaped `[num_slots, index_ratio, kv_heads, head_dim]` = `[num_slots, 4, 2, 256]`, one per layer, `num_slots` = 512 (uncached) or 10,240 (240 MiB cache). Passing it as `k_cache` makes `PAGE_SIZE` a `constexpr` **4** inside the attend kernel (it is taken from `k_cache.shape[1]`, `attend.py:317`), so the kernel's `logical_token // PAGE_SIZE` becomes exactly "which 4-token index block" — which is the unit the selection already speaks in. No kernel edit.

**There is no separate address table.** `lru_ensure` already maintains exactly the map the attend kernel wants: `slot_for_id[id] = slot or -1`, over a flat id space (`moe/offload_kernels.py:22-40`). Set `id_base = (layer * num_req_slots + table_idx) * (T / index_ratio)` and the row `slot_for_id[layer, table_idx]` **is** the block table — logical block -> cache slot, width `T/4`, int32, maintained by the ensure kernel itself. It replaces `md.block_table` **for the attend call only** (the score kernel keeps the real one). Size at 262k with `num_req_slots` = 2: `12 x 2 x 65,536 x 4 B` = **6 MiB**. That removes a kernel, a generation counter, and a whole class of staleness bug: the residency map and the address map are one object by construction.

Per (layer, step), on the compute stream, inside the captured graph:

1. `_select` runs unchanged against the device-resident compressed slab and the **original** `md.block_table`. Output: `blocks` `[rows, 512]` (logical block ids) and `indices` `[rows, 2051]` (logical token ids).
2. **New:** `lru_ensure(blocks, slot_for_id, id_of_slot, usage, step, blocks, src_indices, evict_slots, num_indices, id_base=layer_base)` — the unmodified flashlib call, with block ids where expert ids go. It rewrites `blocks` in place to **cache slot ids** and stages the miss list. Uncached mode replaces this with `arange(512)` and a full miss list.
3. **New:** the gather. `fast_index_copy_multi_jit` with two banks (K and V), `src_ptrs` = the host slab's device alias for this layer (fixed for the pool's lifetime), `src_indices` / `dst_indices` / `num_indices` = the miss plan from step 2. This is `copy_missing`'s exact call shape (`offload_cache.py:1251-1258`). Measured cost: **90.6 µs** for a full 512-block miss, `(1 − h)` of that in steady state.
4. `qsa_sparse_paged_attention(q, k_cache_slab, v_cache_slab, indices, slot_for_id[layer], token_to_req, out)` — **unchanged kernel**.

Two hazards inherited from the expert path, both already solved there:

- **Evicting a slot this step still needs.** `lru_ensure` stamps every hit and every admitted slot with the current `step` and its victim scan skips `usage == step` (`moe/prefetch.py:47-95` documents the argument in full). All 512 of this step's blocks are stamped before any eviction, so none can be chosen as a victim. This is sound only if `cache_size >= block_topk` per layer — assert it (512 slots minimum, 10,240 at the recommended size).
- **The uncached mode must take the same path.** With `FREETOKEN_KV_HOST_CACHE_MIB=0` the cache is exactly `block_topk` slots per layer, so every step evicts everything and `slot_for_id` degenerates to "this step's 512 blocks". Same code, no branch — which is why (a) and (b) can be required to produce bit-identical output.

The causal tail of the open group (the last up-to-3 tokens, which `expand_qsa_block_indices` appends) is always in the current page, which is always in the device **write window** — so those blocks must be resolved against the window, not the host. Simplest correct rule: **the last two pages of every request are pinned resident in the cache** (32 reserved slots, 256 KiB), stamped so `lru_ensure` can never evict them, written directly by `store_kv` and never gathered. `_plan_index_writes` already knows the page boundary arithmetic (`qsa_sparse.py:473-495`).

### Writes: `store_kv` to a host slab

Decode appends one token per step. Prefill appends up to `max_extend_tokens` = 8,192 (`scheduler/prefill.py:127-130`).

**Do not write to host from a kernel.** Instead:

- `store_kv` writes into the device **write window** (2 pages per layer, rotating), exactly as it writes into the pool today — same `store_cache` call, different base pointer and a modulo on the slot (`mha_pool.py:115-133`).
- When a page completes (`out_loc % 64 == 63`, which `_plan_index_writes` already computes an analogue of at `qsa_sparse.py:487`), enqueue a **D2H of that page on a side stream** into the host slab, and record an event. The page becomes gatherable once the event fires.
- Because the last two pages are always resident in the window, a page in flight is never a gather source.

Prefill cost: a full chunk is `8,192 × 2,048 = 16 MiB` per layer, **192 MiB per chunk** across 12 layers, **3.4 ms** at 57 GB/s D2H. A chunk takes ~4.7 s of compute at the measured ~1,740 tok/s prefill rate. **The writeback is 0.07% of prefill and fully hideable on a side stream.** Prefill is a non-issue.

### CUDA graph compatibility

| element | capturable? | evidence |
|---|---|---|
| UVA gather (`fast_index_copy_multi_jit`) | **yes** | it is inside the decode graphs today; `_build_fused_copy_plan` exists precisely to make the pointer descriptors fixed-address and graph-safe (`offload_cache.py:567-574`) |
| `lru_ensure` slot cache | **yes** | the expert cache runs it inside captured decode (`moe/offload_kernels.py:19-40`) |
| `slot_for_id` used as the attend block table | **yes** | a fixed-address device int32 tensor, mutated only by the ensure kernel |
| host slab base pointer | **yes** | one `device_ptr` per layer, fixed for the pool's lifetime; store it in a device int64 tensor like `_copy_src_ptrs` |
| **host-side page-table remap** | **no** | it does not occur inside the graph — `_write_page_table` runs during scheduling (`scheduler/cache.py:718-743`) and `_stage_decode` only *copies* into a static buffer (`qsa_sparse.py:382-394`) |
| **page-completion D2H on a side stream** | **must be outside the graph** | fork/join with events, the pattern `moe/prefetch.py` documents at length (`prefetch.py:20-45`) |

The last row is the only real constraint: the decode graph must not contain the writeback fork. Either issue the writeback from the eager wrapper around the graph replay (it only fires every 64 steps), or keep a device-side counter and let a tiny always-launched kernel no-op 63 times out of 64. The former is simpler.

### MTP speculative verify

`prepare_mtp_verify_graph` captures verify graphs of width up to `1 + _MAX_SPEC_DEPTH` = 6 (`qsa_sparse.py:55-58`, `755-838`). A verify step runs `_select` over `w` rows, so up to `6 × 512 = 3,072` block selections per layer.

- **Worst case** (no dedup): 24 MiB/layer, 288 MiB/step, **5.1 ms** at 57 GB/s. Against the ~39 ms 6-row verify step that `moe/prefetch.py:12-14` documents, that is **+13%** — tolerable but not free.
- **Real case**: the 6 rows are consecutive positions in one sequence, so their selections overlap heavily, and the LRU cache absorbs the overlap for free — `lru_ensure` over the concatenated `6 x 512` ids hits on every repeat. What it does *not* absorb is the eviction pressure: 3,072 ids against a 10,240-slot cache is fine, against a 512-slot one is not. The `w` rows share one `slot_for_id` row, which the attend kernel already supports (it indexes `block_table[safe_request, ...]`, one row per *request*, not per query row). A dedup pass before the ensure (a bitmap over `T/4` blocks — 65,536 bits = 8 KiB at 262k) is a cheap optional optimization, not a correctness requirement.
- The draft chain's `mtp_qsa_saved_blocks` path (`qsa_sparse.py:453-457`, `_expand_selected_blocks`) reuses step-zero block choices at later positions, so the draft steps gather **nothing new** — a genuine advantage of this workload.

This is the highest-risk part of the design and should be built last, behind the same flag, with the verify path falling back to a device-resident pool if `w x block_topk` exceeds the cache.

### State machine: one 4-token block

```
            store_kv into the write window
   ABSENT ─────────────────────────────────► WINDOW-RESIDENT (device, last 2 pages of the request)
                                                  │ page completes; D2H on the side stream
                                                  ▼
                                             IN-FLIGHT (not gatherable; event pending)
                                                  │ event fires
                                                  ▼
                                             HOST-RESIDENT (pinned; the canonical copy)
                                                  │ selected this step
                                                  ▼
                                             CACHED (a slot in this layer's LRU cache;
                                                     slot_for_id[layer,req,block] = slot)
                                                  │ evicted by lru_ensure (usage != step)
                                                  └────────► HOST-RESIDENT
```

Invariants:

- **HOST-RESIDENT is canonical.** The cache is read-only scratch; nothing is ever written back from it.
- **The last two pages of every live request are WINDOW-RESIDENT and never gathered.** The causal tail depends on it.
- **A block is never gathered while IN-FLIGHT.** Guaranteed by the two-page window: a page enters IN-FLIGHT only after two newer pages exist, and a selected block is either in the window or has completed.
- **Every block named by `blocks[row]` is CACHED before the attend kernel runs.** If one is not, `slot_for_id` reads `-1`, the kernel masks the load, and it contributes zero — a *silent* numerical error. `lru_ensure` guarantees it by construction (that is its whole job), provided `cache_size >= block_topk`. This is the design's sharpest edge; see failure modes.

## Failure modes and fallbacks

| failure | why it is dangerous | detection / response |
|---|---|---|
| **A selected block reads `slot_for_id == -1`** | the attend kernel masks it and contributes 0 — no error, wrong logits | Structurally impossible if `lru_ensure` ran with `cache_size >= block_topk`, but this is the one failure that is silent, so verify it: a debug-mode kernel that counts `-1` reads and asserts zero, gated on `FREETOKEN_KV_HOST_STAT`. **Never ship without having run it.** |
| **`lru_ensure` evicts a block this step still needs** | same silent-wrong-answer class | The `usage == step` protection (`moe/prefetch.py:47-95`). Assert `cache_size >= block_topk` at construction; the argument fails below that. |
| **Two requests' logical block ids collide** | one request reads the other's K/V | `id_base` includes `table_idx`, not just the layer. Assert the id space is `12 × num_req_slots × (T/index_ratio)`. |
| Host slab is not device-mapped (pin quota) | gather reads garbage or faults | `HostBank.pin` raises on `cudaHostRegister` failure (`host_banks.py:136-152`); refuse to boot with the tier on. On Windows the pin quota is roughly half of RAM and the expert bank already holds 63.5 GiB — this is a live risk. |
| Page D2H still in flight when its block is selected | reads uninitialized host memory | The two-page window makes it unreachable; assert `page_id < completed_pages` in debug mode. |
| MTP verify's `w x block_topk` exceeds the cache | `lru_ensure`'s `usage == step` protection has no unprotected victim left; it clamps `num_indices` and some blocks stay non-resident -> silent wrong answer | Require `cache_size >= _MAX_MTP_VERIFY_WIDTH x block_topk` = 3,072 slots per layer (24 MiB/layer, 288 MiB total) whenever speculation is enabled. That is above the 240 MiB default, so **the default cache size must be raised to 288 MiB when MTP is on**, or verify must fall back to a device-resident pool. Assert at construction. |
| Host RAM exhaustion → PLE thrash | decode slows for a *different* reason and the cause looks like this design | Watch `peak_disk_read_bytes_per_second` in every benchmark; a rise means PLE pages are being re-faulted. |
| Cache rebuild (`--kv-cache-tokens` change) | the host slab must be reallocated and re-pinned; `QSAKVCache.rebuild` currently frees device buffers and re-derives the index tiers (`qsa_pool.py:151-172`) | The host slab reallocation must follow the same "free the index tiers first, null everything on failure" discipline. Rebuild is idle-only, so no in-flight copies exist. |
| Non-QSA model boots with the flag on | the addressing argument only holds for QSA | Refuse: the tier is a `QSAKVCache` subclass, and the factory only builds it for `AttnType.QSA`. |

## Acceptance criteria

1. **Bit-identical logits, three ways.** Greedy decode of a fixed prompt at 4k / 32k / 200k tokens produces token-identical output for (tier off) == (tier on, cache 0) == (tier on, cache 240 MiB). This is the gate, and the three-way form is what catches a cache bug.
2. **Zero `-1` reads.** The debug counter reports zero masked-out selected blocks over a 10,000-token decode at 262k context.
3. **Off is free.** `FREETOKEN_KV_HOST_TIER=0` leaves the decode path byte-identical; 8k-chat tok/s within noise of the current baseline.
4. **Step cost matches the model.** Measured TPOT increase with the tier on and cache 0 is within 20% of the predicted 1.09 ms/step at batch 1, and falls by at least `h` when the cache is on.
5. **VRAM actually freed.** `/v1/cache/status` at 262k shows `num_pages` unchanged, `kv_per_token` fallen to ~768, and `moe_cache_size` risen to ≥6,200.
6. **Net throughput at 262k.** 8k-chat decode with the tier and cache on at 262k context beats the tier off by ≥12% (predicted +19% at `h`=0.87, +16% at `h`=0.6).
7. **Prefill unaffected.** TTFT at 32,780 tokens within 2% of the 18.87 s baseline.
8. **MTP verify correct and bounded.** Speculative decode with the tier on produces identical accepted-token streams, and the verify step's added time is ≤6 ms.
9. **Host RAM bounded and declared.** Resident host RAM rises by no more than `T × 24,576` + 64 MiB, and the boot refuses if that would exceed a configured fraction of free host memory.

## Measurement plan

**Step 0, and it can be done before any of this is built: measure the inter-step block overlap and the simulated LRU hit rate.** Add a shadow counter behind `FREETOKEN_KV_HOST_STAT` that, per layer, keeps the previous step's `blocks` row plus a simulated LRU of `B` blocks, and reports:

- `|B_t ∩ B_{t−1}| / 512` — the 1-step overlap, directly comparable to ShadowKV's 60% and PRR's 68%;
- the simulated LRU hit rate at `B` ∈ {512, 1024, 2048, 4096, 10240} blocks per layer — directly comparable to HiSparse's 87% at `B` = 2k and directly the `h` in the break-even table;
- the per-layer breakdown, to see whether the first QSA layers behave like GVR's layers 0-1 (1-2% correlation) and should be exempted.

Run at 8k, 64k, and 262k context on real prompts. **This decides whether the design ships and at what cache size**, and it costs one Triton kernel and one boot. It should be the first thing done, before a line of the tier is written.

Then, in order:

1. **Baseline**, tier off, fresh boot, at 65,536 and 262,144 tokens: 8k-chat tok/s, TPOT, TTFT at 0.5k / 8k / 32k, `/v1/cache/status` geometry, and the host-memory columns from the MoE sweep.
2. **Gather cost in situ**: with the tier on, CUDA-event the gather per layer inside the graph; compare to the 90.6 µs/layer microbenchmark. A large gap means the in-graph launch is behaving differently from the standalone one.
3. **Exchange rate, verified**: with the tier on at 262k, sweep `--moe-cache-size` across the freed range and confirm the sweep curve still predicts throughput. If it does not, the whole economic argument needs redoing.
4. **Net throughput** at 65k / 131k / 262k, tier on vs off — the table in "The VRAM-for-PCIe exchange rate" is a prediction and this is the test of it. Publish the measured table next to the predicted one.
5. **LRU cache sweep**: `FREETOKEN_KV_HOST_CACHE_MIB` ∈ {0, 48, 96, 240, 480, 1024} at 262k — i.e. 1x to 21x the per-layer budget; report hit rate, gather µs/layer, expert slots, and net tok/s. Confirm the `h > C_GiB × 0.32` break-even empirically and locate the knee that ArkVale and HiSparse both put near 2x.
6. **MTP verify**: accepted-token stream identity, verify-step ms, and the union size distribution across the 6 rows.
7. **Prefill**: TTFT at 8k / 32k / 200k with the writeback on, plus the side-stream occupancy, to confirm the 0.07% prediction.
8. **Host memory and PLE interaction**: the sweep's memory columns plus `peak_disk_read_bytes_per_second`, at each context. This is the measurement most likely to kill the design at 262k.

## Files that would change

| file | change |
|---|---|
| `python/freetoken/kvcache/qsa_host_pool.py` | **new.** `QSAHostKVCache(QSAKVCache)`: host slab + device alias, device write window, page-completion D2H on a side stream, and the per-layer `lru_ensure` block cache (slab, `slot_for_id`, `id_of_slot`, `usage`, miss plan). Overrides `store_kv`, `k_cache`/`v_cache`, `unit_bytes`, `kv_cost`, `rebuild`. |
| `python/freetoken/attention/qsa_sparse.py` | `qsa_forward` gains a branch: when the pool is a host pool, run `ensure_blocks` + `gather_blocks` and pass the cache slab + `slot_for_id` row to `qsa_sparse_paged_attention` instead of the pool slab + `md.block_table`. `init_capture_graph` / `_ensure_mtp_verify_scratch` allocate the new static buffers. **No change to `_select`, `_update_index_cache`, or `_plan_index_writes`.** |
| `python/freetoken/kernel/triton/qsa/union.py` | **new.** The MTP verify row-union/dedup over the `w` rows' block sets. (There is no remap kernel: `slot_for_id` is the block table.) |
| `python/freetoken/kernel/triton/qsa/attend.py` | **none.** This is the finding that makes the design viable. |
| `python/freetoken/kernel/triton/qsa/score.py` | **none.** The compressed slab stays device-resident with real page ids. |
| `python/freetoken/kvcache/__init__.py` | factory routes `AttnType.QSA` + the flag to the host pool. |
| `python/freetoken/engine/cache_budget.py` | the KV per-page term must reflect the on-card residual (768 B/token, not 25,344) so `plan_cache_budget` gives the freed bytes to the MoE cache. |
| `python/freetoken/env.py`, `engine/config.py`, `launch.py` | the three flags. |
| `tests/kvcache/test_qsa_host_pool.py` | **new.** Bit-identity against the device pool on a synthetic model, at cache sizes 1x and 5x the budget; no `-1` reads; write-window boundary cases; the `cache_size >= w x block_topk` assert. |

Shared with the prefix-parking design: the **pinned staging ring + side-stream event discipline**, and a **`page_byte_view`-style accessor** on `QSAKVCache` so neither feature reaches into `_kv_buffer` directly.

## Open questions

1. **The inter-step block overlap for 4-token blocks is unmeasured, on this model and on any model.** Nothing else in this document is as load-bearing. The published ladder — ~45% at token level (Levy), ~60% at 8-token chunks (ShadowKV), ~68% at 16-64 tokens (PRR), ≥80% at 64 tokens (NOSA) — is monotonic in block size and QSA sits below all of it. Measure it first (measurement plan step 0). A related question: **does a coarser *fetch* unit than the selection unit help?** Fetching the 16-token neighbourhood of each selected block would 4x the bytes but might more than 4x the hit rate, since the published locality improves with granularity. The microbenchmark says 32 KiB rows would still run at ~45 GB/s, so it is affordable to test.
2. **Does the sweep's slot→throughput curve hold above 6,750 slots?** Both the 65k and 131k rows of the exchange-rate table extrapolate past the sweep's top measured point. If the curve flattens (routing hit rate saturating), the design is worse than modelled at short context and unchanged at 262k.
3. **Is 8 KiB really the knee, or is it the kernel's default `worker_threads`?** `fast_index_copy` picks 8/16/32 worker threads by feature size (`kernel/fast_index_copy.py:60-66`); at 8,192 B it takes 32. A tuning pass might find a better point for this row size, which would move the 1.09 ms directly.
4. **Could the gather overlap with anything?** Within a layer it cannot: the selection depends on that layer's own query, so `score → topk → ensure → gather → attend` is a hard chain. Across layers it also cannot — InfiniGen's cross-layer predictor relies on 0.89-0.97 cosine similarity between consecutive layers' block inputs on OPT/Llama-2, but Levy measured **0.36 inter-layer overlap** on modern indexer-based sparse attention and states that *"the previous layer is a very poor predictor of the next layer"* ([arXiv:2603.13430](https://arxiv.org/pdf/2603.13430)). The only overlap available is with the *MoE* layers between QSA layers — the model is 48 layers with 12 QSA — and the MoE path already owns the PCIe link during those. Whether the two can share the link without either regressing is an open experiment, and if they can, the gather becomes partly free. This is also the risk that the MoE prefetch (`moe/prefetch.py`) and this gather contend and *both* regress.
5. **Would per-head selection change any of this?** No — verified above that the top-k is per query row, collapsed over indexer heads. But if a future model version made it per-head, the traffic multiplies by the head count and this design dies. Pin the assumption in a test.
6. **Windows pin quota headroom.** 63.46 GiB is already pinned. The observed quota behavior on WDDM (roughly half of RAM per `moe/host_banks.py:49-56`) suggests ~48 GiB, which is already exceeded — so the quota is evidently not binding the way the comment expects. Measure the actual remaining `cudaHostRegister` headroom before sizing anything.
7. **Interaction with the vision-weights and prefix-parking designs**, both of which want the same ~8 GiB of host headroom. If more than one ships, the budgets must be reconciled explicitly.

---

## Which of the two designs to build first

**Build the prefix-parking design first.** Four reasons, in order of weight:

1. **Parking's win is certain and this one's is conditional.** Parking turns a 19-38 s cold prefill into a 0.03-0.5 s restore at any context, using arithmetic that is fully measured. The host tier's payoff is +3% to +19% depending on a hit rate nobody has measured for 4-token blocks, and its largest payoff is at 262k, where this box cannot hold the tier in RAM.
2. **It is far lower risk.** Parking touches no kernel, no CUDA graph, and no attention path; every copy happens on a side stream during scheduling. The host tier changes how the attend kernel is addressed, puts a `lru_ensure` and a PCIe gather on the hot path inside a captured graph, introduces a silent-wrong-answer failure mode, and has to be made safe under MTP verify capture.
3. **Its central unknown is smaller and cheaper to resolve.** Parking's open question is SSD write bandwidth — a half-hour measurement. The host tier's is inter-step selection overlap, which requires a Triton kernel and a boot, and which decides its cache size and whether it ships at all.
4. **Something cheaper stands in front of the host tier.** FP8 KV quantization (option d) buys 3.0 GiB at 262k for none of the risk, and it halves the host tier and the gather if the tier is built afterwards.

**But run the host tier's step 0 immediately, in parallel.** It is a shadow counter, it touches nothing, and it can be measured on the same server boots the parking work uses. If it comes back near 87%, the host tier's economics change enough (+19% at 262k, +3% even at 65k) to reorder everything after step (2).

**Parking is not a prerequisite for the host tier**, and neither depends on the other's data structures. But three pieces should be built once and shared:

- **The pinned staging ring + side-stream fork/join discipline** (`moe/prefetch.py:20-95` is the reference implementation of the hazard argument). Both designs need a D2H that must not stall decode and whose completion gates a resource release.
- **A `page_byte_view(page_id)` accessor on `QSAKVCache`** returning the K/V rows and the 16 compressed index rows of one page. Parking needs it to serialize a page; the host tier needs it to write back a completed page. Neither should reach into `_kv_buffer` / `_cmp_k_buffer` directly.
- **The model fingerprint** (checkpoint hash + `page_size`, `index_ratio`, `dtype`, `tp_size`) — parking needs it to reject stale files, and the host tier's rebuild path needs the same geometry check.

Build order: **(0) the inter-step overlap shadow counter** — cheap, parallel, and it gates everything below; **(1) FP8 KV quantization** if accuracy allows; **(2) prefix parking**; **(3) the host tier**, uncached first for correctness, then the LRU cache, gated on what (0) says and on host RAM at the target context.
