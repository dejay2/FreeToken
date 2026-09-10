# Qwen3.8 KV prefix parking: keep evicted prefixes in host RAM / on the SSD instead of discarding them

## Status

Historical preimplementation design draft, 2026-09-02, written against `mtp-upstream-merge` at `8591caf`. It is not the current behavior contract; see [README: KV prefix parking](../../README.md#kv-prefix-parking) for user-facing behavior and [the RAM conversation-switching report](../research/kv-ram-conversation-switching-2026-09-10.md) for the implemented RAM checkpoint design and evidence. The source locations and measurements below record the design-time state.

## Purpose

Today, when the KV page allocator runs short, `CacheManager._allocate` (`python/freetoken/scheduler/cache.py:684-704`) asks the prefix cache to evict, gets back a list of page indices and a list of GDN state slot ids, and puts both straight back on their free lists. The bytes are then overwritten by the next request. **Nothing is saved.** If the same conversation comes back — the next turn of a chat, the next step of an agent loop, a resumed session after a server restart — its prefix is recomputed from scratch.

Recomputation is the expensive thing here. On this box a 32,780-token prompt takes **18.87 s** to first token (`results/benchmark-262144.json`, `summary.context_check`), i.e. roughly **1,740 tokens/s** of chunked prefill. A 65,536-token prefix is therefore about **38 s** of prefill; a 262,144-token prefix is about **2.5 minutes**. Against that, moving the same prefix's bytes across PCIe costs milliseconds: the KV+index footprint of 65,536 tokens is 1.547 GiB, and this box measures **57.3 GB/s** device-to-host and **57.7 GB/s** host-to-device on pinned memory (microbenchmark below), so the whole thing moves in **~29 ms**. Off the Samsung 990 PRO at the ~3.1 GB/s the PLE table already sustains (`results/benchmark-262144.json`, `summary.long.peak_disk_read_bytes_per_second` = 3,144,928,517), it is **~0.5 s**.

So the ratio is roughly **1,300x** in favour of parking-to-RAM and **75x** in favour of parking-to-SSD, versus recomputing. That is the whole argument.

The workload this is for is Jay's: one user, `max_running_requests` effectively 1 under MTP (`python/freetoken/engine/config.py:299-303`), long system prompts and long histories re-sent every turn, several conversations alive at once, and a server that gets restarted several times a day. In that shape the in-VRAM radix cache already handles the *common* case well (warm TTFT is 1.38-1.62 s across the whole `--moe-cache-size` sweep, `docs/research/measurements-moe-cache-sweep-2026-09-02.md`). What it does *not* handle is:

1. **Conversation switching.** Two 30k-token conversations do not both fit in the 65,536-token pool the server is booted with today (`/v1/cache/status`: `num_pages` 1024, `page_size` 64). Alternating between them evicts one to admit the other, and every switch pays a full cold prefill.
2. **Server restarts.** `CacheManager.rebuild` throws the whole tree away (`scheduler/cache.py:601-618`) and a process restart obviously does. Every restart costs the next turn of every live conversation a cold prefill.
3. **Long-context work at 262k.** At `-ContextTokens 262144` the pool is 4,097 pages and the MoE slot cache falls to 4,063 (`results/benchmark-262144.json`, `cache_status_before.geometry`), so KV pressure is at its worst exactly where recomputation is most expensive.

## Non-goals

- Changing any numerics. A restored prefix must produce bit-identical K/V, index keys, and GDN state to the ones that were parked. This is a pure byte-preservation feature.
- Changing the attention path, the QSA kernels, the CUDA graphs, the MoE expert cache, or the PLE path. Nothing in `python/freetoken/attention/` or `python/freetoken/kernel/triton/qsa/` is touched.
- Moving the *live* KV of a running request off the card. That is the companion design (`2026-09-02-qwen38-kv-host-tier-sparse-attention-design.md`); the two are independent.
- Concurrency. This design assumes one running request (`max_running_req` 1 under MTP). Multi-tenant fairness, sharing a parked entry between two simultaneous requests, and distributed KV stores are out of scope.
- Multimodal requests. They already bypass the prefix cache entirely (`scheduler/cache.py:93-105`, `cache.py:318-330`: `cache_private` or `mm_embeds` forces a match against the empty prefix and a private page lifetime) and they stay bypassed. Image-placeholder token ids are identical across different images, so a token-id-keyed park would serve the wrong image's KV.
- Compressing or quantizing the parked bytes. FP8/INT8 KV is a separate, orthogonal lever.
- Cross-machine or networked KV stores (Mooncake/NIXL shapes). Local host RAM and the local SSD only.

## Current behavior

### What a cache entry is today

The hybrid model routes to `HybridRadixCache` (`scheduler/cache.py:84-91`, `kvcache/hybrid_radix_cache.py`). Each `RadixTreeNode` (`kvcache/radix_cache.py:17-49`) carries:

- `_key`: the node's own slice of the token-id sequence (int32/int64 device tensor).
- `_value`: the matching KV **page-table slot indices** — one entry per token, values are absolute token slots into the pool (`radix_cache.py:52-56`, and `cache_req` passes `self.page_table[req.table_idx, :req.cached_len]`, `scheduler/cache.py:320`).
- `mamba_value`: an optional `LinearStatePool` slot id, attached **only at page-aligned chunk boundaries** (`hybrid_radix_cache.py:88-114`, and the `CHUNK_SIZE % page_size == 0` assertion at `hybrid_radix_cache.py:59-63`; `CHUNK_SIZE` is 64, `kernel/fla/chunk.py:28`).
- `ref_count` / `mamba_ref_count`: the dual lock, with the invariant `full_ref >= mamba_ref` enforced in `inc_lock` (`hybrid_radix_cache.py:120-136`).
- `timestamp`: LRU key, refreshed on every walk (`radix_cache.py:27`, `hybrid_radix_cache.py:278-292`).

Node keys are page-granular: `_get_key_fn(page_size)` returns `lambda x: tuple(x[:page_size].tolist())` (`radix_cache.py:262-265`), and both `insert` and `_walk` `align_down` to the page size (`hybrid_radix_cache.py:89`, `286`). So **every reusable boundary in the tree is a multiple of 64 tokens**, which is exactly the granularity a parked entry wants.

### What "the KV of a prefix" physically is

`/v1/cache/status` on the live server reports `unit_bytes.kv_per_token = 25344`. That decomposes (verified against `kvcache/qsa_pool.py` and `kvcache/base.spec_kv_bytes_per_token`, `kvcache/base.py:19-37`) as:

| tier | shape | bytes/token | file |
|---|---|---|---|
| paged K/V slab | `[2, 12, num_pages, 64, 2, 256]` bf16 | 24,576 | `kvcache/mha_pool.py:44-52` |
| compressed index slab `_cmp_k_buffer` | `[12, num_pages*64/4 + num_req_slots, 128]` bf16 | 768 | `kvcache/qsa_pool.py:128-135` |
| **total** | | **25,344** | |

Two facts make the index slab free to park alongside the K/V:

- It is a strict **1/4 shadow** of the K/V slab, addressed by `slot // index_ratio` with `index_ratio = 4` and `page_size % index_ratio == 0` (`qsa_pool.py:1-20`, `qsa_pool.py:70-75`). One 64-token KV page is exactly 16 contiguous compressed rows at `page_id * 16`, in each of the 12 index layers. So "park page *p*" is a fixed, contiguous, page-id-derived byte range in both slabs.
- The `pending_ring` / `pending_position_ring` / scratch rows are **not** per-token and do **not** need parking. The pool's own docstring gives the argument: "a new tenant of a `table_idx` starts at a group boundary (`cached_len` is 0 or a page multiple), so its first closing group takes every member from its own forward" (`qsa_pool.py:22-28`). A restored prefix always resumes on a page boundary, so the ring is irrelevant to it.

Per page (64 tokens): `64 * 25,344 = 1,622,016 B = 1.547 MiB`.

### What "the GDN state of a prefix" physically is

`/v1/cache/status` reports `unit_bytes.mamba_per_slot = 115642376` = **110.28 MiB**, and `num_mamba_slots = 8`. That is `conv_states` + `recurrent_states` for the 36 GDN layers (`kvcache/linear_state_pool.py:66-79`), fp32 recurrent by default (`linear_state_pool.py:19-21`). It does **not** grow with context: it is a fixed per-slot cost.

The critical structural fact is in `HybridRadixCache.match_prefix` (`hybrid_radix_cache.py:70-83`): a matched KV prefix is **truncated back to the deepest ancestor that still owns a live snapshot**, because a continuation can only resume the GDN recurrence from a checkpointed boundary. So a parked entry without its GDN snapshot is worth *nothing* — the KV would be matched and then truncated to 0. **The snapshot is not optional baggage; it is half the entry.**

### Where entries die today

Three places, all of which simply return bytes to free lists:

1. `CacheManager._allocate` (`scheduler/cache.py:684-704`): under page pressure, `prefix_cache.evict_full(need)` walks unlocked LRU **leaves**, and for each returns its KV page indices and frees its GDN slot (`hybrid_radix_cache.py:160-179`). `_allocate` concatenates the pages onto `free_slots` and calls `linear_state_pool.free(er.mamba_slots)`. Nothing is copied anywhere.
2. `CacheManager.ensure_mamba_slots` (`scheduler/cache.py:145-153`): under GDN-slot pressure, `evict_mamba` tombstones the LRU snapshot-bearing node — internal nodes keep their KV and lose only the snapshot; leaves lose both (`hybrid_radix_cache.py:181-205`). A tombstoned internal node's KV survives but is now unreachable as a resume point, so it is dead weight until a descendant is matched.
3. `CacheManager.rebuild` (`scheduler/cache.py:601-618`): builds a brand-new prefix cache and calls `linear_state_pool.reclaim_all_slots()`. Everything is gone.

Only 8 GDN slots exist, and the running request holds a ping-pong pair (`scheduler/cache.py:433-455`), so **path 2 fires constantly**: with `linear_state_cache_ratio` 2.0 (`engine/config.py:355-357`) there is room for a handful of parked snapshots at most. Snapshot pressure, not page pressure, is the binding constraint on how many conversations the tree can hold resumable.

### Where a restore would plug in

`CacheManager.match_req` (`scheduler/cache.py:93-113`) is the single entry point: it calls `prefix_cache.match_prefix(ids)` and wraps the result in a `HybridCacheHandle` plus `MatchResult.mamba_value`. The `# TODO: support HiCache` comment sits on `MatchResult` itself (`kvcache/base.py:197`). Downstream, `PrefillAdder` reads only `.cached_len` and `.get_matched_indices()` (`kvcache/hybrid_radix_cache.py:33-45`), so **a restore that has already materialized pages and a GDN slot before `match_req` returns is invisible to every caller**. That is the seam.

## Measurements

All on this box (Windows 11 Pro 26200, RTX 5090 32 GB, 95.6 GiB RAM, PCIe Gen5, Samsung 990 PRO), 2026-09-02.

### PCIe, measured here

Median of 100 `cudaMemcpyAsync` calls on exact-size `cudaHostAlloc` pinned memory (`kernel/pinned.alloc_pinned_tensor`), CUDA-event timed. Script and raw JSON in the session scratchpad (`bench_d2h.py`, `d2h_results.json`).

| size | D2H µs | D2H GB/s | H2D µs | H2D GB/s |
|---|---|---|---|---|
| 4 MiB | 74.4 | 56.4 | 74.8 | 56.0 |
| 48 MiB | 878.4 | 57.3 | 871.6 | 57.7 |
| 128 MiB | 2343.1 | 57.3 | 2321.3 | 57.8 |

The link is symmetric to within 1%. Use **57 GB/s** in both directions.

### Derived park/restore costs

| prefix length | KV+index bytes | park (D2H, 57 GB/s) | restore (H2D) | SSD write/read @3.1 GB/s | recompute (prefill @1,740 tok/s) |
|---|---|---|---|---|---|
| 4,096 tok (64 pages) | 99.0 MiB | 1.8 ms | 1.8 ms | 33 ms | 2.4 s |
| 16,384 tok | 396 MiB | 7.3 ms | 7.3 ms | 0.13 s | 9.4 s |
| 32,768 tok | 792 MiB | 14.6 ms | 14.6 ms | 0.26 s | 18.8 s (measured) |
| 65,536 tok (today's whole pool) | 1.547 GiB | 29.1 ms | 29.1 ms | 0.51 s | 37.7 s |
| 262,144 tok | 6.19 GiB | 116 ms | 116 ms | 2.05 s | 151 s |

Plus, in every row, **one GDN snapshot: 110.28 MiB → 2.0 ms over PCIe, 36 ms off the SSD.**

The 3.1 GB/s SSD figure is the *observed peak read rate of the live server* during the 262k benchmark (the PLE table demand-paging), not a synthetic drive benchmark; a 990 PRO does ~7 GB/s sequential, so 3.1 GB/s is a conservative, in-situ number. Writes are not measured — labelled below as an open question.

### The shape of the entry

The GDN snapshot's fixed 110.28 MiB is the awkward part of the arithmetic:

| prefix length | KV+index | GDN snapshot | snapshot share |
|---|---|---|---|
| 1,024 tok | 24.8 MiB | 110.3 MiB | **82%** |
| 4,096 tok | 99.0 MiB | 110.3 MiB | **53%** |
| 16,384 tok | 396 MiB | 110.3 MiB | 22% |
| 65,536 tok | 1,584 MiB | 110.3 MiB | 6.5% |

**Parking short prefixes is not worth it** and the design should refuse them: below roughly 8,192 tokens the snapshot dominates the entry and the recompute it saves is only ~5 s. A floor of one prefill chunk (`max_extend_tokens`, default 8,192 — `scheduler/prefill.py:127-130`) is the natural cut.

**Exactly one snapshot per parked entry.** The tree can hold a snapshot at every ×64 boundary; parking them all would be 110 MiB per 64 tokens, which is absurd. Park only the snapshot at the entry's **end boundary** — i.e. the one on the leaf node being evicted, which is precisely what `evict_full` already hands back (`hybrid_radix_cache.py:160-179`, `_free_node_mamba`). A restored entry is then resumable at exactly one point, its full length. That is the right trade: the alternative (multiple resume points) costs 110 MiB each to buy the ability to fork a conversation mid-history, which this workload does not do.

### Host RAM headroom on this box

This is the binding constraint and it is severe. From `docs/research/measurements-moe-cache-sweep-2026-09-02.md` and `docs/research/memory-audit-qwen38-rtx5090.md`:

- Host **resident** is 82-95 GiB of 95.6 GiB across the whole sweep, and it is **flat in `--moe-cache-size`**: the 63.46 GiB pinned expert bank does not shrink, and every GPU byte costs commit but ~zero resident bytes.
- The engine's own estimate of free host memory with the server up is ~8 GiB (`models/qwen4_exp/weight.py:266-270`).
- That 8 GiB is not spare: it is the standby list feeding the 47.68 GiB memory-mapped PLE n-gram table that every decoded token gathers rows from. Taking 4 GiB of it as pinned KV park would evict PLE pages and slow decode.

So: **a pinned host park tier on this box must be small (≤2 GiB) or absent.** The SSD tier is the one that can actually be large.

## Prior art

A web survey was run alongside this design (2026-09-02; sources linked inline). Four findings change decisions here.

### 1. Everyone keys by a prefix-chained hash, and the two biggest implementations key it wrong

**SGLang HiCache** ([design doc](https://docs.sglang.io/advanced_features/hicache_design.html)) has three tiers — L1 GPU, L2 host DRAM, L3 storage — with *"L1 and L2 private to a single inference instance; only L3 can be shared."* Its page hash is `SHA256(prev_page_digest ‖ page_token_bytes)`, prefix-chained exactly as proposed above, and the storage key appends a config suffix of `_{model_name}_{tp_rank}_{tp_size}_{pp_size}_{pp_rank}`. **`dtype`, KV quantization, and `page_size` are not in the key** — two servers differing only in `--kv-cache-dtype` collide on the same file.

**LMCache** keys `model_name @ world_size @ worker_id @ chunk_hash @ dtype_str` (`lmcache/utils.py`, `CacheEngineKey`) — better, dtype is in — but its rolling hash falls back to Python's builtin `hash()` when vLLM is absent (the config default is literally `pre_caching_hash_algorithm: "builtin"`), so cross-process reuse silently depends on `PYTHONHASHSEED`. And its disk backend writes one flat `.pt` per chunk **with no header, no magic, and no version**, with metadata held only in an in-memory dict that is **never rebuilt from the directory on startup** — so LMCache's disk files survive a restart as dead bytes: never hit, never reclaimed.

**This validates two choices above and sharpens one.** The rolling-hash-over-page chain is the industry-standard structure. Putting `dtype`, `page_size`, `index_ratio`, and `tp_size` in the fingerprint fixes a latent corruption bug that both reference implementations have. And the on-disk **header plus manifest, with the header re-validated per entry at read time**, is what stops the LMCache orphaning failure — worth keeping even though it costs a few lines.

Two HiCache policy constants are worth stealing directly: `write_through_threshold = 1` (back up on the first hit) vs `2` for `write_through_selective`, and `prefetch_threshold = 256` tokens — below which a prefix is never prefetched from storage. The `FREETOKEN_KV_PARK_MIN_TOKENS` floor above is the same idea at a much larger scale, because the GDN snapshot makes short entries far more expensive here than in a pure-attention model.

Published HiCache results ([LMSYS](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)): up to 80% TTFT reduction and 6x throughput; Novita AI on Qwen3-Coder-480B took hit rate 40% → 80% and TTFT −56%; Ant Group on DeepSeek-R1 −84% TTFT. The nearest peer-reviewed number is **Strata** ([arXiv:2508.18572](https://arxiv.org/abs/2508.18572)): up to 5x lower TTFT than vLLM+LMCache. vLLM's own KV-offload blog reports 2x-22x TTFT depending on prompt size.

### 2. Marconi settles the "how many GDN snapshots per entry" question

Marconi (MLSys 2025, [arXiv:2411.19379](https://arxiv.org/abs/2411.19379)) measured the thing that decides this design's economics: at block size 32, **25.0% of KV token blocks are reused by future requests but only 0.4% of SSM states are — a 65.3x gap.** Its conclusion is that states should be admitted only at **branch points** and at the last decoded token, and that input-only prefixes should be admitted only on their *second* occurrence; the result is *"typically 2"* admitted states per sequence instead of thousands. It reports 4.5-34.4x higher token hit rate than vLLM and P95 TTFT reductions of 36.1-71.1%.

vLLM converged on the same shape independently: its hybrid prefix caching ([#26201](https://github.com/vllm-project/vllm/issues/26201)) defaults `mamba_cache_mode` to **`align`** — snapshot only at scheduler-step ends and block boundaries — which the tracking issue itself calls "Marconi-style". SGLang's `MambaRadixCache` allows a state at any node but tombstones aggressively.

**So "exactly one snapshot per parked entry, at the end boundary" is not a shortcut; it is what the field converged on**, and the 65.3x reuse gap is the number that justifies it.

Marconi also states the constraint this design has to live with, more bluntly than the source does: *"in-place state updates for recurrent layers preclude rolling back cache entries for partial sequence overlaps, and instead mandate only exact-match cache hits."*

### 3. Nobody has shipped recurrent-state offload to disk, and the RAM-tier attempts are visibly immature

- **SGLang `HiMambaRadixCache`** has `mamba_host_value` / `protect_host_mamba()` wired in and a merged "Support HiCache for MambaRadixCache" commit, but carries open bugs [#20495](https://github.com/sgl-project/sglang/issues/20495) (crash on first prefill), [#24121](https://github.com/sgl-project/sglang/issues/24121), and [#33713](https://github.com/sgl-project/sglang/issues/33713) (*the unified tree prunes MAMBA nodes instead of downgrading them, breaking host-tier loadback*). The PyTorch blog lists HiCache for linear-attention layers as **future work**. **No evidence of anyone writing recurrent states to an L3 disk backend.**
- **LMCache** has the cleanest statement of the idea — it *"reinterprets that [linear-attention] state as an opaque page at registration time"* ([docs](https://docs.lmcache.ai/mp/hybrid_models.html)) — but publishes the caveat that **results are not bit-exact** between cached and fresh runs on the GDN backend. Acceptance criterion 1 above rejects that outcome outright, and it should stay rejected.
- **llama.cpp** context checkpoints were broken for recurrent models until recently ([#22384](https://github.com/ggml-org/llama.cpp/issues/22384): *"for recurrent models `pos_min` always equals the full sequence length, so this check always fails and no checkpoint is ever restored"*). The fix took a Qwen3.6-27B second turn from ~12,146 re-processed tokens (~11 s) to **31 tokens (115 ms)** — the same shape of win this design predicts.

**Being first here is a real risk and should be priced in.** The mitigation is acceptance criterion 2: a byte-identity unit test on the snapshot round trip, run before anything else.

### 4. What llama.cpp's slot save/restore teaches about the file format

`POST /slots/:id?action=save|restore|erase`, gated on `--slot-save-path`. The format (`src/llama-context.cpp`) is a magic `'ggsq'` + a version + a token count + tokens, then per stream a cell table and per-layer raw K then V rows. What matters is the **load-time compatibility checks**: `"mismatched key type"`, `"mismatched key row size"`, `"mismatched value element size"`, `"mismatched GQA embedding size"`, `"mismatched layer count"`, and `"incompatible V transposition"`. Every one of those is a field the FreeToken fingerprint must cover, and the V-transposition check is a reminder that a *layout* change (there, flash attention) is as invalidating as a dtype change.

Its sharpest trap is worth avoiding by name: on SWA/iSWA models, save silently writes **only the unmasked window** rather than the prompt, unless `--swa-full` is passed ([discussion #18244](https://github.com/ggml-org/llama.cpp/discussions/18244)). The FreeToken analogue is the QSA compressed index slab: parking the K/V pages without the 16 matching compressed rows per page would produce an entry that restores, matches, and then attends to the wrong blocks. The page-granular `page_byte_view` accessor proposed below exists to make that impossible to get wrong.

### 5. A published cross-check on the GDN snapshot size

The survey's independent computation of this model's recurrent state from the published config (36 GDN layers, 16 QK heads and 48 V heads at dim 128, conv kernel 4, `conv_dim = 2·16·128 + 48·128 = 10,240`) gives `36 × (10,240 × 3 × 2 B + 48 × 128 × 128 × 4 B)` = **115,458,048 B**. The live server reports `mamba_per_slot = 115,642,376` — the difference is the declared sibling `slot_states` (`kvcache/linear_state_pool.py:100-110`). The 110.28 MiB figure this design is built on is confirmed from outside the repo.

One caveat from the upstream docs to check against the restore path: the Transformers `qwen4_exp` page states *"cache cropping is not supported when PLE or QSA is enabled"* ([HF docs](https://huggingface.co/docs/transformers/main/en/model_doc/qwen4_exp)). Restoring a parked entry never crops — it installs a whole page-aligned prefix and resumes at its end boundary — but `PARK_RESTORE_MARGIN` and the `align_down` in `HybridRadixCache.insert` must be verified to never produce a partial-page install.

## Options considered

### (a) Pinned host ring only, 2-4 GiB — REJECTED as the primary tier

A fixed pinned buffer, allocated at boot via `moe/host_banks.HostBank` (`moe/host_banks.py:84-125`), holding whole parked entries. 2 GiB holds one 65,536-token conversation and its snapshot, or about 20 conversations of 4,096 tokens. Park and restore are single `cudaMemcpyAsync`es at 57 GB/s: 29 ms for the largest entry.

Fast and simple, and it composes with everything already in the repo. But 2 GiB is one conversation at today's context and a *fifth* of one at 262k, it competes directly with the PLE page cache for the 8 GiB of headroom, and it evaporates on restart — which is one of the three problems this design exists to fix. As a *window* in front of an SSD tier it is excellent; as the whole tier it does not earn its RAM.

### (b) SSD file tier only — RECOMMENDED as the primary tier

One file per parked entry under a park directory, keyed by a content hash. 20-60 GiB is nothing on this drive and it survives restarts, which is the highest-value property. The cost is latency: 0.51 s to restore a 65,536-token entry at 3.1 GB/s, against a 1.38-1.62 s warm TTFT today and a 37.7 s cold prefill. So a restore is **still a 74x win over recompute** and lands inside the latency envelope a user already tolerates for a cold turn.

The write side is the risk: parking a 1.547 GiB entry writes 1.5 GiB to the SSD, and eviction happens *during* a request's page allocation. That must not be synchronous (see the recommended design).

### (c) Two tiers: small pinned window + SSD backing — RECOMMENDED

(a) as an L1 in front of (b) as an L2. The most-recently-parked entry (or two) stays in the pinned ring and restores in 29 ms; older entries live only on the SSD and restore in ~0.5 s. This is the SGLang HiCache / LMCache shape, and it matches how the PLE table is already served on this box (mmap + `PrefetchVirtualMemory` + a pinned staging ring, `models/qwen4_exp/weight.py:279-300`, `426-441`, `ple.py:47-50`).

The pinned window should be **sized in pages, not gigabytes**, and default small (1 GiB) so it does not fight the PLE standby list. It can also be **omitted entirely** (window size 0) with no code path change: the SSD tier is the correctness-bearing one.

### (d) Park to pageable host memory instead of pinned — REJECTED

Avoids CUDA pin quota, but the D2H copy from device to *pageable* memory cannot be async and cannot overlap with decode; it stalls the compute stream for the whole 29 ms. And on Windows the pages would then be ordinary anonymous memory competing with the PLE standby list *and* the pagefile. Pinned-staging-then-write-to-file (which is (c)) gets the same RAM discipline with an async copy.

### (e) Keep only the GDN snapshots and recompute KV — REJECTED

Tempting, because the snapshot is what `match_prefix` truncates on: park 110 MiB and the tree's *surviving in-VRAM* KV becomes resumable again. But the two are evicted together in `evict_full` (a leaf loses both), so there is no surviving KV to resume against in the case that matters. It only helps the `evict_mamba` tombstone path (`hybrid_radix_cache.py:181-205`), where KV survives and only the snapshot is dropped. That is a real but narrow win — see "Open questions".

### (f) Do nothing — the honest case

The in-VRAM radix cache already gives 1.38-1.62 s warm TTFT for the single-conversation case, which is most of Jay's usage. This design buys nothing for a user who works in one conversation and never restarts the server. Its value is entirely in conversation switching, agent loops that interleave contexts, and restart recovery. If those are rare, (f) wins.

## Recommended design

**Option (c): a page-granular, content-keyed, two-tier park store, written asynchronously at eviction and read synchronously at match.**

### Flag surface

Everything off by default. New environment reads, following the existing `freetoken.env.ENV` convention:

| flag | default | meaning |
|---|---|---|
| `FREETOKEN_KV_PARK` | `0` | master switch |
| `FREETOKEN_KV_PARK_DIR` | `<model_dir>/../.freetoken-park` | SSD tier root; `""` disables the SSD tier |
| `FREETOKEN_KV_PARK_DISK_GIB` | `32` | SSD tier cap |
| `FREETOKEN_KV_PARK_RAM_MIB` | `1024` | pinned window cap; `0` disables the window |
| `FREETOKEN_KV_PARK_MIN_TOKENS` | `8192` | do not park a prefix shorter than this |

### What one parked entry is

A **page-aligned token prefix** of length `L` (a multiple of 64), consisting of:

1. `L` token ids (int32), stored verbatim — needed to rebuild the radix key and to verify a hash match.
2. `L/64` KV pages, each 1,622,016 B: for each of the 12 QSA layers, the K rows and V rows of the page's 64 slots, plus the 16 compressed index rows at `page_id * 16` in each of the 12 index layers.
3. **One** GDN snapshot, 115,642,376 B, the one that sat on the evicted leaf.
4. A header: model identity, geometry, and the key.

Entry size: `L/64 * 1,622,016 + 115,642,376` bytes. A 32,768-token entry is 902 MiB.

### The key, and why it must be a chain

`key = BLAKE2b-128( model_fingerprint || parent_key || token_ids[i*64 : (i+1)*64] )`, folded page by page from a fixed root — i.e. a **rolling hash over 64-token pages**, one hash per page boundary, with the entry keyed by the hash at its end boundary.

Three properties fall out:

- A prefix of a parked entry has a key that is a *prefix of the chain*, so a shorter match is findable without storing every length.
- It is content-addressed, so it survives restarts and `CacheManager.rebuild` for free.
- `model_fingerprint` (checkpoint path + the safetensors index file's own hash + `page_size`, `index_ratio`, `dtype`, `tp_size`) makes a stale park directory from a different checkpoint or a different TP layout un-matchable rather than silently wrong. **This is the single most dangerous failure mode in the design and the fingerprint is the only defence.** It must include everything `spec_kv_bytes_per_token` depends on (`kvcache/base.py:19-37`).

The index is a small on-disk manifest (`park.json`, rewritten atomically) mapping key → (file path, `L`, byte size, last-use time). It is rebuilt by scanning headers if missing or corrupt.

### The cold tree

Do **not** add parked entries to `HybridRadixCache`. Its nodes carry device tensors and its eviction walks assume every value is a live page index; teaching it a second residency class would touch every method and every invariant (`full_ref >= mamba_ref`, the tombstone cascade at `hybrid_radix_cache.py:236-250`). Instead:

**A separate `ParkStore` object, owned by `CacheManager`, holding a flat `dict[key, ParkedEntry]`** plus the two tiers. It is consulted only from `match_req`, and it never sees a device tensor except during the copy itself. The radix tree is unchanged.

Lookup is a walk of the rolling hash over the request's own token ids, longest first: hash forward page by page, remember the deepest page boundary whose key is in the store, stop at the first miss that has no deeper hit (the chain means a miss at page *p* cannot be followed by a hit at *p+1* for the same conversation). At `max_running_req` 1 and 64-token pages this is at most 4,096 BLAKE2b updates over 64 int32s each for a 262k prompt — sub-millisecond, on the host, off the critical path of any kernel.

### Park: the write path

Hook `CacheManager._allocate` (`scheduler/cache.py:684-704`), at the point where `evict_full` has returned but before the pages hit `free_slots`:

```
er = self.prefix_cache.evict_full(need)          # unchanged
if self.park is not None:
    self.park.offer(er)                           # NEW: D2H on the park stream, records an event
self.free_slots = torch.cat([...])                # unchanged
if er.mamba_slots: self.linear_state_pool.free(er.mamba_slots)   # DEFERRED (see below)
```

`offer` must not stall decode. The mechanism:

1. A **dedicated park CUDA stream** and a small pinned **staging ring** (the same pattern as `moe/prefetch.py`'s side stream and `ple.py:47-50`'s ring). Ring size = `FREETOKEN_KV_PARK_RAM_MIB`, minimum one page plus one snapshot (112 MiB).
2. For each evicted leaf worth parking (`length >= FREETOKEN_KV_PARK_MIN_TOKENS`, and it has a `mamba_value`): enqueue on the park stream a gather of its pages' KV+index rows into the staging ring, and a copy of its GDN slot, then record an event.
3. **The freed pages and the freed GDN slot must not be reissued until that event fires.** This is the one real correctness hazard in the design. The clean way is the mechanism the repo already uses for exactly this shape: `CacheManager.lazy_free_region` (`scheduler/cache.py:619-635`) defers page returns to the end of a region. Extend it to a *park-pending* list: pages and mamba slots handed to `offer` go on a pending list, and are returned to their free lists only when the park stream's event has completed (checked cheaply at the top of the next `_allocate`). If the pool is so tight that the pending pages are needed *now*, `offer` synchronizes on the event and takes the stall — bounded by one entry's D2H (29 ms worst case at today's context), and it degrades to today's behavior if parking is disabled.
4. A **host thread** drains the staging ring to the SSD tier (`open(..., 'wb')`, sequential write) and updates the manifest. The pinned window keeps the most recent entries in RAM in addition; when it overflows, the oldest entry is dropped from RAM (it is already on disk).

The GDN slot deserves a note: 110.28 MiB per park, and only 8 slots exist. Holding one back for the duration of a D2H (2.0 ms) is negligible; holding it for an SSD write (36 ms) is not, so **the snapshot must be staged through pinned RAM, not written to disk from the device**. The staging ring's minimum size is therefore one snapshot.

### Restore: the read path

Hook `CacheManager.match_req` (`scheduler/cache.py:93-113`), after the in-VRAM match and only when the store can beat it:

```
m = self.prefix_cache.match_prefix(ids)
if self.park is not None:
    hit = self.park.lookup(ids, min_len=m.cached_len + PARK_RESTORE_MARGIN)
    if hit is not None:
        m = self.park.restore(hit, self)          # allocates pages + a mamba slot, inserts into the tree
```

`restore`:

1. Allocates `hit.L / 64` pages through the ordinary `_allocate` (so it evicts, and can itself trigger parking — bounded by refusing to re-park an entry that is already in the store, which the content key makes trivial).
2. Allocates one GDN slot through `ensure_mamba_slots(1)` + `linear_state_pool.alloc(1)`.
3. Issues the H2D: from the pinned window if resident (29 ms for the largest entry), else read the file into the staging ring and H2D chunk by chunk, overlapping read and copy — the PLE prefetch pattern (`weight.py:515-560`).
4. Calls `prefix_cache.insert(token_ids[:L], page_indices, mamba_value=slot)` (`hybrid_radix_cache.py:85-118`) so the tree owns it exactly as if it had been computed, and returns a normal `HybridMatch`. Everything downstream — `PrefillAdder`, the lock, `cache_req` — is unchanged.

`PARK_RESTORE_MARGIN` exists because a restore is not free: restoring a 65,536-token entry costs ~0.5 s off SSD, so it must beat the prefill it displaces. Prefill runs at ~1,740 tok/s, so the extra tokens a restore buys are worth `extra_tokens / 1740` seconds against a cost of `entry_bytes / 3.1e9` seconds. Break-even is around **2,600 extra tokens off SSD** and **~150 off RAM**. Set the margin to one page-aligned prefill chunk (4,096 tokens) off SSD and one page (64) off RAM; both are conservative.

### State machine and lifetime

An entry is in exactly one of:

```
                     evict_full offers it
   LIVE (in tree) ───────────────────────► PARKING  (D2H in flight; pages+slot pending, unreissuable)
        ▲                                     │ event complete
        │                                     ▼
        │                                  RAM-RESIDENT  (in the pinned window; pages+slot released)
        │                                     │ writer thread drains
        │                                     ▼
        │                                  DISK-RESIDENT (in the park dir + manifest)
        │                                     │ window overflow drops the RAM copy
        └─────────── restore ─────────────────┘
                (H2D, then insert into the tree)
                                              │ disk tier over cap / manifest LRU
                                              ▼
                                            DROPPED
```

Invariants:

- **A parked entry always carries exactly one GDN snapshot.** A KV-only entry is unusable (`match_prefix` truncates to 0) and must never be created; `offer` skips leaves whose `mamba_value` is `None`.
- **Pages and GDN slots in PARKING are owned by neither the tree nor the free list.** They are on the pending list, and only the event completion moves them.
- **RAM-RESIDENT ⊇ nothing; DISK-RESIDENT is the source of truth.** The window is a pure cache over the disk tier, so dropping it is always safe. Consequence: an entry becomes matchable only once it is DISK-RESIDENT *or* RAM-RESIDENT — both are fine, and a PARKING entry is not matchable (it is also still in the tree's evicted set, so nothing is lost).
- **Restoring does not remove the entry from the store.** It stays on disk; a later eviction of the restored copy is a no-op re-park (same key already present).

### Behavior across rebuilds and restarts

- **`CacheManager.rebuild`** (`scheduler/cache.py:601-618`) discards the tree. With parking on, `rebuild` should first `offer` every unlocked snapshot-bearing leaf, then proceed unchanged. It is idle-only, so the D2H can be synchronous there. This turns a `--kv-cache-tokens` resize from "lose everything" into "lose nothing but the resize latency".
- **Process restart.** The disk tier and its manifest survive. On boot, `ParkStore` reads the manifest, validates each header's `model_fingerprint` against the live geometry, and discards non-matching entries (they are from another checkpoint or another `page_size`/`index_ratio`/dtype). This is the property option (b) exists for.
- **Geometry change.** A different `page_size`, `index_ratio`, `dtype`, or TP size changes the byte layout of both slabs. The fingerprint covers all of them, so such entries are dropped, not misread. A change in `num_pages` does **not** invalidate anything: parked entries store *contents*, not page ids, and restore into whatever pages `_allocate` hands them.

### CUDA graph interaction

None. Every copy here happens on a side stream during scheduling (`_allocate`, `match_req`), outside any captured region. The decode graphs never see the park store. This is the main reason this design is the lower-risk of the two.

### MTP interaction

`cache_req` already refuses to commit a request with unsettled speculative rows (`scheduler/cache.py:288-298`), and parking hangs off eviction and matching, not commit. But `_allocate` *can* run inside a speculative step's page allocation, so the park stream's event and the `temporary_page_lease` invariant check (`scheduler/cache.py:637-682`, which asserts `free_slots` is untouched across the lease) must not collide: **`offer` must never mutate `free_slots`**, only the pending list, and the pending drain must not run inside a lease. Assert both.

## Failure modes and fallbacks

| failure | detection | response |
|---|---|---|
| Park directory unwritable / full | `OSError` on write | Log once, disable the SSD tier for the process, keep the RAM window. Never fail a request. |
| Manifest corrupt or truncated | JSON parse error / header mismatch | Rebuild by scanning entry headers; drop unreadable files. |
| Stale entries from another checkpoint | `model_fingerprint` mismatch in the header | Drop the file. **Must be checked per entry at read time, not only at boot** — a park dir can be shared by two model versions. |
| Hash collision (BLAKE2b-128) | not detectable by hash | The entry stores its token ids; `restore` compares them against the request's before inserting. Mismatch → drop the entry and fall through to prefill. This makes a collision a performance event, not a correctness one. |
| Park D2H still in flight when the pages are needed | pending list non-empty and `free_slots` short | Synchronize on the park event and stall (bounded by one entry, ≤116 ms at 262k). |
| Pinned staging allocation fails (CUDA pin quota) | `cudaHostAlloc` failure in `HostBank.pin` (`moe/host_banks.py:136-152`) | Disable parking for the process, log. On Windows/WDDM the quota is roughly half of RAM and the expert bank already holds 63.5 GiB, so this is a live risk. |
| Restore H2D fails mid-way | CUDA error | The pages hold garbage but were never inserted into the tree; free them, drop the entry, fall through to prefill. **`insert` must be the last step.** |
| GDN slot exhaustion during restore | `LinearStatePool.alloc` raises (`linear_state_pool.py:130-137`) | `ensure_mamba_slots(1)` first; if it still fails, abandon the restore and prefill. |
| SSD write thread falls behind | pending write queue over a bound | Drop the oldest un-written entry (it is still just an eviction that did not get parked). Never block the scheduler. |
| Parking a multimodal prefix | `cache_private` / `mm_embeds` | Those requests never reach the tree (`scheduler/cache.py:97-101`), so they never reach `evict_full`. Assert it anyway in `offer`. |

## Acceptance criteria

1. **Numerics.** With parking on, a conversation that is evicted, parked, and restored produces **token-identical** output to the same conversation with parking off, at temperature 0, for at least: a 16k-token prompt, a 65k-token prompt, and a multi-turn chat of 5 turns. This is the gate; nothing ships without it.
2. **Bytes.** A unit test parks a synthetic prefix, zeroes the pool, restores it, and asserts the K/V slab rows, the compressed index rows, and the GDN `conv_states`/`recurrent_states` are bitwise equal to the originals.
3. **No regression when off.** With `FREETOKEN_KV_PARK=0` the decode path executes the same instruction sequence — verified by the existing benchmark suite showing no change in 8k-chat tok/s beyond noise.
4. **No stall.** With parking on, TPOT during a decode that triggers an eviction+park is within 5% of TPOT with parking off. (A synchronous 29 ms D2H inside a 17 ms step would be a 170% spike; this criterion is what forces the side stream.)
5. **Restart survival.** Park a 32k conversation, restart the server, re-send the same prompt, and observe a restore rather than a cold prefill: TTFT ≤ 1.5 s instead of ~19 s.
6. **Conversation switching.** Two 30k conversations alternating, five turns each: with parking on, every turn after the first is a restore; measured TTFT ≤ 1.5 s versus ~17 s cold.
7. **Poisoned store.** Point the park directory at entries produced by a different checkpoint; the server must boot, drop them, and serve correctly.
8. **Bounded footprint.** Resident host RAM with parking on and a 1 GiB window is within 1.2 GiB of the same run with parking off (verified with the same instrumentation as `measurements-moe-cache-sweep-2026-09-02.md`).

## Measurement plan

Run against the existing harness, one variable at a time, on a fresh boot each time (the sweep showed a long-running server drifts ~8% in throughput).

1. **Baseline**, parking off, at `-KVCacheTokens 65536` and `262144`: cold TTFT and warm TTFT for 4k / 16k / 32k prompts; 8k-chat tok/s. This re-establishes the `measurements-moe-cache-sweep` baseline on the current commit.
2. **Park/restore latency, isolated**: instrument `offer` and `restore` with CUDA events and wall clock; report park D2H ms, SSD write MB/s, SSD read MB/s, restore H2D ms, and total restore wall time, for entry sizes 8k / 32k / 65k tokens. The predicted values are in the table above; the SSD *write* rate is the one number that is currently unmeasured.
3. **Conversation-switch benchmark** (new scenario): N conversations of L tokens each, round-robin, M turns. Report mean TTFT per turn with parking off / RAM window only / SSD only / both, at (N=2, L=30k) and (N=4, L=8k). This is the headline result.
4. **Restart recovery**: park, restart, replay; report TTFT of the first turn after restart.
5. **Decode interference**: 8k-chat tok/s and TPOT p99 with parking on under a workload that forces an eviction every few turns, versus off. Acceptance criterion 4.
6. **Host memory**: the same `physical used / sched private / sched WS / system commit` columns as the MoE sweep, with the window at 0 / 512 MiB / 1 GiB / 2 GiB, to confirm the window does not displace the PLE standby list. Watch `peak_disk_read_bytes_per_second` — a rise means PLE pages are being re-faulted.
7. **Store hygiene**: park 40 GiB of entries against a 32 GiB cap; confirm LRU eviction, a consistent manifest, and no unbounded growth.

## Files that would change

| file | change |
|---|---|
| `python/freetoken/kvcache/park_store.py` | **new.** `ParkStore`, `ParkedEntry`, the rolling key, the manifest, the two tiers, the staging ring and park stream. |
| `python/freetoken/scheduler/cache.py` | `CacheManager.__init__` builds the store; `_allocate` calls `offer` and drains the pending list; `match_req` calls `lookup`/`restore`; `rebuild` parks before discarding; the pending list interacts with `lazy_free_region` and `temporary_page_lease`. |
| `python/freetoken/kvcache/qsa_pool.py` | **read-only additions**: a `page_byte_view(page_id)` helper returning the K/V rows and the 16 compressed index rows of one page, so `ParkStore` does not reach into `_kv_buffer`/`_cmp_k_buffer` directly. No behavior change. |
| `python/freetoken/kvcache/linear_state_pool.py` | a `slot_byte_views(slot)` helper (conv + recurrent + declared `slot_states`) for the snapshot copy. No behavior change. |
| `python/freetoken/kvcache/hybrid_radix_cache.py` | none required. `insert` already takes a donated `mamba_value` (`hybrid_radix_cache.py:85-118`). |
| `python/freetoken/env.py` | the five new flags. |
| `python/freetoken/engine/config.py` | plumb the flags into `EngineConfig`; CLI in `python/freetoken/launch.py`. |
| `python/freetoken/kvcache/base.py` | replace the `# TODO: support HiCache` comment (`base.py:197`) with a reference to the store. |
| `tests/kvcache/test_park_store.py` | **new.** Byte-identity, key chain, fingerprint rejection, collision handling, manifest rebuild. |

Shared with the companion host-tier design (build once, use twice): the **pinned staging ring + side-stream copy scaffolding**, the **`page_byte_view` accessor on `QSAKVCache`**, and the **model fingerprint**.

## Open questions

1. **SSD write bandwidth under load, unmeasured.** Everything here assumes ~3.1 GB/s reads (observed) and comparable writes. A 990 PRO's sustained write past its SLC cache is much lower. If sustained writes are ~1 GB/s, a 1.5 GiB park takes 1.5 s of background I/O — fine for a background thread, but it changes the sizing of the write queue.
2. **Does the `evict_mamba` tombstone path deserve its own park?** When a snapshot is tombstoned but the KV survives (`hybrid_radix_cache.py:181-205`), parking just the 110 MiB snapshot would make the surviving in-VRAM KV resumable again at ~2 ms restore cost. With only 8 GDN slots this path fires often. It is a strictly smaller feature than the main design and might be worth building *first*.
3. **Should the number of GDN slots go up instead?** 110.28 MiB per slot against a 4,063-slot MoE cache worth 10.7 GiB at 262k — buying 8 more snapshot slots costs 0.86 GiB of VRAM, i.e. ~340 expert slots, i.e. ~2 tok/s by the sweep's exchange rate. That may be a cheaper fix for conversation switching than any of this, for the case where both conversations fit in the KV pool. It does not help restarts.
4. **Interaction with the vision-weights-on-demand design.** Both want the ~8 GiB of host headroom. If both ship, their budgets have to be reconciled explicitly.
5. **Is `PARK_RESTORE_MARGIN` the right shape?** It assumes restore cost is linear in entry size and prefill cost is linear in token count. Prefill is actually super-linear in attention at long context (though QSA's sparsity flattens it), so the margin is conservative in the direction that matters.
6. **Should the store be keyed by *chunk* rather than by *entry*?** LMCache-style per-chunk keying (256 tokens there; 64-token pages here) would let two conversations share a parked system prompt, and it is what both reference implementations do. With one user and one running request the sharing benefit is small and per-entry keying is much simpler — but the rolling hash already computes every page's key, so promoting to per-page entries later is a storage-layout change, not a redesign. Revisit if the workload grows a second user.

7. **Should the parked GDN snapshot be stored quantized?** SGLang ships `--enable-int8-mamba-checkpoint`, reporting *~2x cached-prefix capacity at fixed GPU memory*, for radix-cached states. int8 would take a parked snapshot from 110.28 MiB to ~28 MiB and would make short entries worth parking. But it breaks acceptance criterion 1 (bit-identical restore), so it can only be an opt-in accuracy trade, never the default. A safer half-step: store the fp32 `recurrent_states` as bf16 (110.28 → 56 MiB) — still not bit-identical, still an opt-in.

8. **Is `write_through` or `write_back` right here?** HiCache defaults to `write_through` with a threshold of one hit — back an entry up as soon as it is reused once, rather than waiting for eviction. This design is pure write-back (park only at eviction), which minimizes I/O but loses everything to a crash or a kill. Given that restart recovery is one of the three stated motivations, a `write_through` mode — park a locked prefix in the background when the request finishes, before it is ever evicted — may be worth more than it costs. It does not change any of the machinery, only when `offer` is called.
