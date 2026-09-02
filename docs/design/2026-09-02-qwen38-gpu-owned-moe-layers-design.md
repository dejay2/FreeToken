# GPU-owned MoE layers ("stuck-down layers") for Qwen3.8-Flash-Next-NVFP4

Status: approved design, 2026-09-02. Branch base: `mtp-upstream-merge` @ 6229077.
Companion research: `docs/research/memory-audit-qwen38-rtx5090.md`,
`docs/research/routing-skew-2026-09-02.md`, `docs/research/measurements-moe-cache-sweep-2026-09-02.md`.

## 1. Goal

Let a chosen subset of the 48 MoE layers keep **all 512 experts permanently resident in VRAM**
and allocate **no pinned host bank at all** for those layers. Every other layer keeps today's
behaviour: pinned host bank + global LRU GPU slot cache (`--moe-cache-size`).

Why: host RAM is the binding constraint on the target box (95.6 GiB, ~89 GiB commit). The only
duplicated bytes are the LRU slot copies of host rows (17.43 GiB at 6,750 slots), and the host
bank cannot be partially released on Windows (`cudaHostUnregister` is base-only; pagefile-backed
sections cannot decommit; `HostBank.release()` is a no-op for pinned banks, `host_banks.py:154-160`).
So the only way to give RAM back is to **never allocate** a layer's host bank. Per-expert static
pinning was measured and rejected (routing skew report: static top-K catches 27-66 % vs LRU 72-90 %).
Whole-layer residency is the surviving idea: a fully resident layer never misses regardless of routing.

Geometry (per `models/nvfp4_banks.py:96-103`): one expert row = 2,772,480 B across 6 banks
(`gate_up_packed` 1,638,400, `gate_up_scale` 204,800, `gate_up_global` 5,120, `down_packed` 819,200,
`down_scale` 102,400, `down_global` 2,560). One layer = 512 rows = **1.322 GiB**. Each owned layer
saves 1.322 GiB host RAM and costs 1.322 GiB VRAM (~512 LRU slots).

Target configuration for the first live test ("50/50 of the expert room"): 6 owned layers
(7.93 GiB) + LRU of about 4,400 slots (today's 6,750 minus ~3,072 slots for the owned layers plus
~760 slots from the ~2 GiB of VRAM that is unused after boot today). Expected: 7.9 GiB host RAM
back; speed somewhere between −5 % and −13 % of 73 tok/s. Speed is to be measured, not assumed;
the operator decides afterwards whether to keep 6, fewer, or none.

## 2. Layer choice

Owned layers are chosen by the operator, or by `auto`, which uses a **built-in list** derived from
the four decode captures in `docs/research/routing-skew-2026-09-02/{code,prose,chat8k,toolcall}.json`
(`per_layer[].miss_rate`, cache_size 6,750). Hungriest by mean miss rate, and identical under
mean `missing_per_step` and the pooled `union.json`:

| rank | layer | mean miss rate |
|---|---|---|
| 1 | 1 | .217 |
| 2 | 6 | .213 |
| 3 | 0 | .212 |
| 4 | 2 | .207 |
| 5 | 7 | .207 |
| 6 | 22 | .204 |
| 7 | 10 | .199 |
| 8 | 13 | .197 |

`auto` = `{0, 1, 2, 6, 7, 22}`; `auto:N` takes the first N of the ranked list (ranked order
1, 6, 0, 2, 7, 22, 10, 13, 5, 18, ...). The list is a module constant with a comment naming its
source; no runtime heuristic. The existing "U-shaped, head+tail" heuristic of `_auto_cpu_layers`
(`engine.py:1816-1843`) is **not** supported by the data (the tail 39-47 is mid-pack; the minimum is
layer 31) and must not be reused.

## 3. Interface

- CLI: `--moe-gpu-owned-layers <spec>` beside `--moe-cpu-layers` (`server/args.py:565-578`).
  Grammar is the existing `_parse_cpu_layers_spec` grammar (`engine.py:1727-1753`: explicit id list
  `"0,1,2"`, count `"6"`, fraction `"0.125"`) plus `auto` and `auto:N`. Default `None` = off.
- Config: `EngineConfig.moe_gpu_owned_layers: str | None` next to `moe_cpu_layers`
  (`engine/config.py:350`); inert entry in `_DENSE_MOE_SETTINGS` (`engine.py:1846-1859`).
- Env: `FREETOKEN_MOE_GPU_OWNED_LAYERS` read by the launcher only (mirrors `FREETOKEN_VISION_WEIGHTS`).
- Launcher `scripts/start-qwen38-flash-next-mmap-windows.ps1`: `-GpuOwnedLayers <spec>` (default
  off), passed through as `--moe-gpu-owned-layers`. Boot banner prints the spec.
- Validation (`_adjust_config`, near `engine.py:2161`): requires `--moe-backend offload`; rejects any
  overlap with the resolved `cpu_layer_ids`; rejects the FTW packed checkpoint path with a clear
  error (out of scope, see §10) rather than silently ignoring the flag; rejects ids outside
  `[0, num_moe_layers)`; rejects an owned set that leaves fewer than `2*num_experts` LRU slots when
  prefill overlap is on (existing floor, `offload_cache.py:159-163`).
- Boot log, one line: `MoE GPU-owned layers: [0, 1, 2, 6, 7, 22] (6 x 1.32 GiB resident, no host
  bank); LRU cache 4400 slots for 42 streaming layers`.
- `/v1/cache/status` geometry (`server/api_server.py:820-833`, `cache_report.py:159-164`) and
  `/v1/cache/routing` (`scheduler/scheduler.py:860-875`) expose `gpu_owned_layers`.

## 4. Loading (no host bank for owned layers)

Precedent to mirror end-to-end: `--moe-cpu-layers` (config → `_parse_cpu_layers_spec` →
`_resolve_cpu_layers` `engine.py:1756-1767` → residency label vector `engine.py:732-740` →
`load_expert_banks(..., layer_residency=...)` `engine.py:741-750` → ambient `_ResidencyPlan`
`host_banks.py:227-265` → consulted in `pin_banks` / `PinPipeline` → `ExpertBanks.layer_residency`
→ `cache.cpu_layer_ids` set before `set_bank_sources` `engine.py:783-785`).

New residency label `GPU_OWNED` in the per-layer vector. For owned layers:

1. `_alloc_nvfp4_host_banks` (`nvfp4_banks.py:88-103`) / `alloc_layer_banks`
   (`host_banks.py:215-224`) do **not** allocate a `HostBank`. The per-layer list keeps length
   `num_layers`; owned indices hold the **device tensor** `[512, *row_shape]` for that bank kind
   (allocated with `torch.empty(..., device="cuda")` before the fill), so every consumer still sees
   one entry per layer and `size(0) == num_experts`.
2. **AMENDED 2026-09-02 after live run 1 — the staging design below was built, failed on the
   box and has been removed (`8a63977`).** What ships instead: an owned layer's `.fill` **is**
   its device tensor, so each `fill[expert] = row` is a synchronous pageable H2D copy issued
   by the placement thread itself. No staging, no cap, no back-pressure, no CUDA event. Two
   reasons, both observed live (`docs/research/measurements-gpu-owned-layers-2026-09-02.md`):
   the engine loads weights inside `torch.inference_mode()`, which is **thread-local**, so the
   device banks are inference tensors that only the loading thread may write — the drain-thread
   flush raised `Inplace update to inference tensor outside InferenceMode is not allowed` every
   time; and the NVFP4 placement loop is single-threaded, so choice (a)'s bounded staging
   deadlocks by construction (the thread waiting for a slot is the only thread that could free
   one). Cost: 7.9 GiB of pageable H2D at boot for six owned layers, a few seconds. The
   original text follows for the record.

   ~~Fill goes through **one reusable pinned staging layer**~~: a `HostBank` per bank kind shaped for
   one layer, allocated once, `pin()`ed once, reused for each owned layer in turn. The existing
   assignment loops (`nvfp4_banks.py:200-221` serial, `326-341` parallel) write into the staging
   bank instead of the layer bank; at the layer-completion sink (`LayerCompletionTracker`,
   `host_banks.py:359-387`, which fires at `E*6` notes) the pipeline issues
   `device[k].copy_(staging[k], non_blocking=True)` for the 6 bank kinds, records a CUDA event, and
   the staging bank is not reused until that event has completed. Peak extra host RAM during boot:
   1.32 GiB, released after the last owned layer. The serialized `PinPipeline` drain thread already
   sets the device (`host_banks.py:306-312`).
   Note: the shard readers may deliver an owned layer's rows interleaved with other layers' rows
   (parallel reader). The staging design must therefore either (a) hold one staging layer per
   *in-flight* owned layer, bounded by a small cap (e.g. 2) with back-pressure, or (b) sort/route so
   that only one owned layer is in flight. Pick (a) with cap 2 unless measurement shows (b) is
   needed; document the choice.
3. `placed` asserts (`nvfp4_banks.py:233-234`, `352-353`) stay `num_layers * E * 6`: owned layers
   are still fully placed, just into device memory.
4. `bank_bytes_estimate` / `ftw_bank_bytes` (`expert_banks.py:390-419`) subtract owned layers so the
   pin budget and boot banner are honest.
5. `_echo_residency` (`expert_banks.py:510-534`) echoes `GPU_OWNED` for the owned layers; a
   downgrade is impossible by construction (no lock to fail).
6. FTW path (`checkpoint/ftw.py:414-462`): refuse the flag with a clear error (§3).
7. `moe/cpu_executor.py:196, 325-345` (`_resolve_banks` / `_make_table` builds a raw `data_ptr()`
   table for C++): must raise if any per-layer source is a CUDA tensor. This path is only reached for
   `cpu`/`hybrid`, which §3 already rejects, but the guard prevents a silent wrong-memory read.

## 5. Cache and forward path (identity mapping)

The NVFP4 decode kernel only ever indexes row `topk_ids[m,k]` of whatever tensors it is handed
(`moe/fused_nvfp4.py:168-232`; dispatch `layers/moe.py:910-949`). It has no notion of slots. So an
owned layer needs **no kernel change and no LRU bookkeeping**.

`OffloadMoeCache` (`moe/offload_cache.py`):

- `set_bank_sources` (`494-556`): accept `gpu_owned_layers: frozenset[int]`; for owned indices skip
  the host-shape assertions and store `self.resident_banks[layer_id] = tuple(device tensors in
  schema order)`; head-shape comparison (`542-546`) uses the first **streaming** layer, not index 0.
- New predicates beside `is_cpu_layer` / `is_unpinned_layer` (`756-763`): `is_gpu_owned_layer`.
  New `resident_views(layer_id)` beside `bank_views` (`787-793`).
- `_build_fused_copy_plan` (`571-632`): owned layers get the existing `0` pointer placeholder
  (`600-604`). **Never** call `device_ptr` on an owned layer's tensor: `kernel/pinned.py:59-68`
  returns `data_ptr()` for CUDA tensors, which would produce a plausible but wrong copy source.
- `prefetch_prefill_layer` (`857-890`): early return for owned target layers (the caller
  `_wait_prefill_overlap`, `layers/moe.py:857-867`, pre-issues `layer_id + 1`, so the guard must be
  keyed on the *target*). `release_prefill_layer` already returns early on mismatch (`1025-1026`).
  `prefetch_ready` (`331-345`) excludes owned layers.
- `copy_missing` / `ensure_experts` / `materialize_layer` must never be called for owned layers;
  add asserts so a wiring bug fails loudly (mirror
  `test_locked_layer_copy_missing_rejects_ensure_experts_staging`).
- `rebuild` (`651-723`): slot caches are reallocated from the first streaming layer's shape;
  `resident_banks` survive untouched; `_build_copy_plan()` refresh keeps skipping owned layers.

`OffloadMoELayer` (`layers/moe.py`):

- `_decode_routed` (`709-738`): if owned, skip `prefetch_wait` / `ensure_experts` / `copy_missing`
  and call `_expert_gemm(cache, hidden, topk_weights, topk_ids, views=cache.resident_views(id),
  n=None, alphas=cache.alphas_for_layer(id), is_prefill=False)` with **raw** `topk_ids`.
  (`alphas_for_layer`, `offload_cache.py:777-785`, is already the position == expert id variant;
  `None` for triton nvfp4.)
- `_prefill_routed` (`818-855`) and `_wait_prefill_overlap` (`857-867`): owned layers use
  `resident_views` directly, no overlap buffer, no wait/release.
- The small-prefill movement path (`FREETOKEN_MOE_SMALL_PREFILL_ROWS`, `layers/moe.py:596-615`)
  routes narrow prefills through the decode path; the owned branch must be correct in both.
- CUDA graphs and the integrated MTP speculation graphs: the owned path is fixed-shape reads of
  fixed-address tensors, strictly simpler than today's; no capture changes expected. Verify by the
  existing boot capture (bs=1, MTP widths 1-6) succeeding.

Decode-frequency histogram: `ensure_experts` currently does the `collect_decode_freq` scatter
(`offload_cache.py:1035-1039`). Lift the scatter into a helper called from both the streaming and
the owned branch so `/v1/cache/routing` `decode_freq` still counts owned layers.

## 6. VRAM budget

- `engine.py:639-642`: add `len(owned) * num_experts * per_expert_bytes` to `fixed_cache_size`
  (exactly how `state_pool_bytes` accounts for the GDN pool) and shrink `total_experts` to
  `(num_moe_layers - len(owned)) * num_experts`. `cache_budget.py` stays pure.
- `expert_bytes_per_slot` (`cache_budget.py:17-28`) reads `t[0][0]` (layer 0). Change it to read the
  first streaming layer (layer 0 is in the default owned set).
- `--moe-cache-size N` remains explicit for the streaming layers. If the explicit N plus the owned
  reservation plus KV/GDN pools exceeds `memory_ratio * free`, boot fails with a message stating how
  many owned layers or slots would fit (operator decision: fail loudly, never silently shrink).
- `_target_moe_and_expert_bytes` (`engine.py:904-916`) and `validate_rebuild` inherit the change.
- `decode_routing_stats` `slots_per_layer` (`offload_cache.py:1216`) divides by the streaming layer
  count. `cache_report.py:88, 115-117, 159-164` and `server/model_meta.py:120-152` derive
  `moe_experts` from `num_experts * num_moe_layers`; subtract owned layers.

## 7. Reporting honesty

- `decode_miss_stats_per_layer` (`offload_cache.py:1171-1197`): owned layers report
  `resident: true` and `miss_rate: null` (not `0.0`), so future heuristics cannot mistake them for
  perfect streaming layers.
- `weight_placement_report` / boot log list the owned set and the resident bytes.
- `engine/mtp_fast_verify.py:345-409` movement reconciliation: check `movement_reconciled` still
  holds when owned layers contribute active-but-never-fetched counts; adjust the invariant to count
  only streaming layers if needed, with a test.
- `bytes_per_expert_row` / `actual_h2d_bytes` (`1137-1146`) are over `bank_caches` only and now
  under-report total expert bytes; document, do not change.

## 8. Tests (CPU only, `CUDA_VISIBLE_DEVICES=""`)

Mirror the existing suites:

- `tests/engine/test_moe_cpu_layers.py`: parser (`list` / count / fraction / `auto` / `auto:N` /
  empty / out of range), resolver, disjointness with `cpu_layer_ids`, backend rejection, FTW
  rejection, dense-model inertness.
- `tests/moe/test_offload.py` (`_make_split_cache` fixture pattern, `788-864`): owned layer stored as
  resident views; `ensure_experts` / `copy_missing` / `materialize_layer` / `prefetch_prefill_layer`
  refuse owned layers; copy plan holds the `0` placeholder for owned layers; `rebuild` preserves
  `resident_banks`; decode forward for an owned layer passes raw ids and resident views (mirror
  `test_offload_moe_layer_decode_forward_uses_remapped_slot_ids`, `327-378`); prefill overlap skips
  owned layers and still alternates buffers correctly for `L-1 -> L+1` when `L` is owned.
- `tests/engine/test_cache_budget.py`: owned reservation reduces the auto slot count by exactly
  `len(owned) * 512 * 2,772,480 B`; `expert_bytes_per_slot` ignores an owned layer 0; explicit
  `--moe-cache-size` overflow fails with the sizing message.
- `tests/moe/test_routing_stats.py`: per-layer rows carry `resident: true, miss_rate: null` for
  owned layers; `decode_freq` still counts them.
- Loader: with the real checkpoint present (skip otherwise), owned-layer device tensors (on CPU in
  the test via a device override) are byte-identical to the host-bank rows of a normal load for the
  same layer; ~~staging bank reuse does not corrupt rows when two owned layers are in flight~~.
  AMENDED (§4.2): the shipped tests run the real NVFP4 loaders over a *synthetic* checkpoint
  (no `D:\Models` needed), inside `torch.inference_mode()`, serial and with the parallel
  reader's rows of three owned layers interleaved round-robin.
- Runner: `uv pip install --target <scratch>\pytest-site pytest pytest-timeout` once, then
  `PYTHONPATH="<tree>\scripts\windows-ple-mmap;<tree>\python;<scratch>\pytest-site"` and
  `%LOCALAPPDATA%\FreeToken\venv\Scripts\python.exe -m pytest ... -p no:cacheprovider`.

## 9. Live verification (operator, one server at a time)

Boot script: `-GpuOwnedLayers auto -MoECacheSize <N>` with N computed from the boot log's free
memory so that ~1 GiB stays free after MTP graphs (first guess 4,400; the launcher/engine prints
the fit). Measure against the current baseline (6,750 slots, no owned layers), fresh boots both:

| check | pass |
|---|---|
| boot log shows owned set, LRU size, MTP graphs 6/6 + 7/7 captured | yes |
| scheduler private bytes and whole-system commit | −7.9 GiB ± 0.3 |
| whole-system physical in-use | −7.9 GiB ± 0.5 |
| boot peak host RAM | ≤ baseline + 1.5 GiB |
| 8k-chat decode tok/s (same prompt as the sweep) | recorded; operator decides |
| TTFT on the same prompt | recorded |
| answers at temperature 0 | identical to baseline |
| picture request (`-VisionWeights mmap`) | works, latency recorded |
| `/v1/cache/routing` | owned rows `resident: true`; streaming rows sane |
| owned-layer rows on device | byte-identical to a host-bank load (one-off probe script) |

## 10. Out of scope

- Automatic layer choice from live statistics (the built-in list is enough for now).
- FTW packed checkpoints (flag refuses loudly).
- Any change to the LRU policy, per-layer slot budgets, or `--moe-cpu-layers` semantics.
- Releasing host rows at run time (impossible on Windows; see §1).

## 11. Risks

- Speed: at a fixed VRAM budget the LRU drops from 6,750 to ~3,700 slots for 42 layers. The
  `experts_for_90pct` measurement (code workload: 139 mean) sits right at today's 140.6 slots/layer,
  so a noticeable slowdown is possible. Measured, then decided.
- `device_ptr` on CUDA tensors succeeds silently (`kernel/pinned.py:59-68`); every place that
  builds pointer tables must exclude owned layers or raise.
- Layer 0 is special-cased in three places (`cache_budget.py:28`, `offload_cache.py:542-546`,
  `offload_cache.py:680`); all three must use "first streaming layer".
- ~~Parallel reader interleaving requires bounded staging (§4.2).~~ RESOLVED the other way:
  bounded staging *is* the deadlock against a single-threaded placement loop; owned layers
  fill in place instead (§4.2 amendment).
- The loader path only works from a thread that is in the same `torch.inference_mode()` as
  the one that allocated the device banks. Any future attempt to move owned-layer filling
  onto a helper thread must allocate those banks outside inference mode first.
- Two bank-allocation sites exist (raw + FTW); the FTW site must refuse, not ignore.
