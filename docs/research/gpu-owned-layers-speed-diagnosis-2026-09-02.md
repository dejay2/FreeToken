# Why GPU-owned MoE layers halved decode — and the one-line reason it was not the LRU

Live run 3, 2026-09-02, same box as runs 1 and 2: Windows 11 Pro 26200, RTX 5090 32 GB
(32,607 MiB), 95.6 GiB RAM, `D:\Models\Qwen3.8-Flash-Next-NVFP4`, worktree
`D:\FreeToken-gpu-owned-layers`, port 2030; accepted server on 2020 from `D:\FreeToken`.

Run 2 ([`measurements-gpu-owned-layers-run2-2026-09-02.md`](measurements-gpu-owned-layers-run2-2026-09-02.md))
measured `-GpuOwnedLayers auto -MoECacheSize 4400` as functionally perfect and twice as slow:
34.6 tok/s against 70.4, cold-7k TTFT 5.86 → 15.57 s, first picture 6.62 → 76.91 s. It
attributed the loss to the smaller LRU. **That attribution was wrong.**

## Root cause, in two sentences

`--moe-gpu-owned-layers` **added** 7.93 GiB of resident expert banks *on top of* the
`--moe-cache-size` the operator asked for instead of charging them to it, so the run-2
candidate held 7,472 slot-equivalents of MoE weights where the accepted baseline holds 6,750
— +1.85 GiB of VRAM, leaving the card 569 MiB free at decode peak. At that occupancy the
Windows video-memory manager starts demoting live allocations to system memory, which is what
halved decode, tripled cold TTFT and made the first (allocation-heavy) vision encode take
76.9 s — the expert traffic itself was never the problem: the run-2 candidate moved **11 %
fewer** expert rows per decode step than the baseline it lost to.

## Hypotheses, and how each died

| # | hypothesis | verdict | killed by |
|---|---|---|---|
| H1 | integrated MTP speculation silently degraded/disabled with owned layers (movement reconciliation, ladder replay) | **rejected** | `MTPFastVerifier`/`MTPVerifyGraphRunner` — the only `collect_stats`-gated `cuda.synchronize` calls in the MTP tree — are reachable only from `mtp_shadow.py`, and the boot script clears `FREETOKEN_MTP_SHADOW`. Boot logs identical on both builds: spec graphs 6/6, draft graphs 7/7, ladder replays 6/6. E1 below reproduces baseline speed on the same branch with speculation on. |
| H2 | the owned decode branch breaks CUDA-graph capture / replays eager | **rejected** | `_decode_routed`'s owned branch is strictly *fewer* ops than the streaming one (no `ensure_experts`, no `copy_missing`, no overlap buffer, no slot remap) and lands on the same `fused_experts_decode_nvfp4_marlin` kernel; `resident_views`/`alphas_for_layer` are pure device-side lookups (a dict get and a contiguous slice). Capture succeeded in every boot. E2 runs the *same* owned branch at 63 tok/s. |
| H3 | per-call Python/allocation overhead in the owned branch | **rejected** | E2 and the post-fix boot run the identical owned code path on 6 layers at 63–65 tok/s. Also arithmetically impossible: 6 owned layers would have to cost as much as all 48 layers combined. |
| H4 | cold-start triton autotune/JIT for the new `[512, …]` bank shapes explains the cold TTFT and the 77 s picture | **rejected as the cause** | E1 (no owned layers, same branch) first picture 4.91 s; E2 (6 owned layers, same shapes, VRAM parity) first picture 4.76 s; post-fix 4.27 s. New bank shapes are present in E2 and post-fix and cost nothing. |
| H5 | `-CollectRoutingStats` (which the candidate had and **no baseline number ever did**) | **rejected** | E1 ran the candidate branch *with* `-CollectRoutingStats` at 6750 slots and reproduced baseline speed. Worth recording as a real methodology confound that had never been controlled before this run. |
| H6 | the LRU drop 6750 → 4400 | **rejected — the evidence runs the other way** | `/v1/cache/routing` says the run-2 candidate did **91.8** expert fetches per decode step against the 6750 baseline's **103.5**; E2 at 3678 slots does **106.2** fetches per step and is *1.8× faster* than the candidate. Shrinking the cache made it faster. |
| **H7** | **total MoE device residency exceeded what the card can serve without paging** | **CONFIRMED** | E2/E1/post-fix vs run 2, below. |

### The measurement that made H6 untenable

Baseline per-layer decode miss rates were already on disk from the routing-skew study
(`routing-skew-2026-09-02/chat8k.json`, 6750 slots, 48 streaming layers) and were never
compared against run 2's `routing-cand2.json`. They should have been:

```
baseline 6750 : miss_rate min 0.097 / med 0.154 / max 0.250   -> 103.45 fetches per step
run2  4400+6  : miss_rate min 0.120 / med 0.189 / max 0.234   ->  91.82 fetches per step
```

The owned set `auto` picks (`0, 1, 2, 6, 7, 22`) is exactly the six worst-miss layers in that
baseline capture (0.250, 0.233, 0.217, 0.207, 0.196, 0.195) — 18.45 of the baseline's 103.45
fetches per step, removed entirely. Per-step PCIe traffic went **down 11 %** while throughput
went down 51 %. Whatever the candidate was losing, it was not spent moving experts.

Caveat on that pair: the 103.45 comes from a chat8k-only capture and the 91.82 from run 2's
mixed window (chat + short + picture requests), so the −11 % is indicative rather than
controlled. The comparison that is properly controlled is **run 2 (91.8) against E2/E3 (106.2 /
105.8)** — three mixed windows over near-identical request mixes on the same box, where the
configuration doing 15 % *more* fetches per step runs 1.8× faster. Either way the sign is the
one that kills H6.

## The experiments

Four boots, one server at a time, `nvidia-smi` < 3 GB and a 60 s settle before each; only
`python.exe` carrying `freetoken.cli serve` and their `python.exe` descendants killed;
`ft.exe daemon` never touched; `D:\Models` and `D:\FreeToken` untouched; nothing pushed.

Metric: `chat8k.py` — the run-2 8k-chat prompt, `temperature 0`, thinking off, 512 tokens,
warm prefix (TTFT ~1.2–1.4 s), decode tok/s from the streamed usage block. Identity check:
the same prompt at `max_tokens 48`, md5 `29f0e74dfed744b538f856f38553e1ae`.

| # | build | owned | LRU slots | total MoE slot-equivalents | free VRAM after init | 8k-chat 512-tok decode (warm) | first picture | fetches/step | 48-tok md5 |
|---|---|---|---|---|---|---|---|---|---|
| — | baseline 2020 (`D:\FreeToken`), start of session | — | 6750 | 6750 | 4.58 GiB [run 2] | **69.0 / 73.7** | 6.62 s [run 2] | 103.5 [skew study] | match |
| — | run 2 candidate `2470573` | 6 | 4400 | **7472** | **3.27 GiB** | **34.6** | **76.91 s** | 91.8 | match |
| E1 | `2470573`, no `-GpuOwnedLayers`, `-CollectRoutingStats` | — | 6750 | 6750 | 4.41 GiB | **72.9 / 66.8 / 67.6** | 4.91 s | 81.4 | match (2/2) |
| E2 | `2470573`, `auto`, `-MoECacheSize 3678` (hand-computed VRAM parity) | 6 | 3678 | 6750 | 5.14 GiB | **63.3 / 62.2 / 63.7** | 4.76 s | 106.2 | match (2/2) |
| E3 | **`19286e1` (the fix)**, `auto`, `-MoECacheSize 6750` | 6 | **3678 (charged)** | 6750 | 5.20 GiB | **64.8 / 64.7 / 64.7** | 4.27 s | 105.8 | match (3/3) |
| — | baseline 2020 restored, end of session | — | 6750 | 6750 | — | **58.3 / 61.2 / 61.8 / 62.1 / 62.6 / 63.3** | — | — | match |

Reading the table:

* **E1 exonerates everything that is not the owned reservation.** Same branch, same
  `-CollectRoutingStats`, same LRU as the accepted server: baseline speed, baseline picture
  latency, identical answer. So the branch, the routing histogram and the 6750-slot LRU are
  all innocent.
* **E2 is the decisive one.** Same six owned layers, same owned forward path, same new bank
  shapes — only the LRU is *smaller* (3678 vs 4400) and total residency is back at 6750
  slot-equivalents. It runs **1.8× faster than run 2 while doing 16 % more expert fetches per
  step**. Making the cache smaller made the server faster; that is only possible if the
  bottleneck was memory pressure, not cache misses.
* The 76.9 s first picture is gone in every configuration that is not oversubscribed
  (4.3–4.9 s), including the two that carry the owned layers and their `[512, …]` banks. H4 is
  dead: it was the same paging, hitting the encode's transient allocations hardest.

### The arithmetic

One expert slot on this model is 2,772,480 B (`/v1/cache/status` `unit_bytes.moe_per_expert`),
and one owned layer is 512 of them = 1.322 GiB.

```
accepted baseline : 6750 slots                      = 17.43 GiB
run 2 candidate   : 4400 slots + 6 x 512 owned      = 19.29 GiB   (+1.85 GiB)
E2 / post-fix     : 3678 slots + 6 x 512 owned      = 17.43 GiB   (parity, to the byte)
```

`_check_gpu_owned_cache_fits` accepted 4400 because it is arithmetically correct against the
budget it is given: `cache_budget_bytes` is 23,564,753,305 B (21.95 GiB), of which KV reserves
1.55 GiB, leaving ~20.4 GiB for MoE. The accepted server spends only 17.43 GiB of that. **The
~3 GiB it leaves on the table is not slack — it is where the 2.17 GiB resident MTP draft head
goes**, and the draft head is loaded *after* the MoE cache is sized, so no budget term knows
about it. The owned reservation was free to spend that headroom, and did.

The result at the device: `Free memory after initialization` 3.27 GiB (vs 4.41–5.20 GiB for
every healthy configuration), then −2.17 GiB for the draft head, and run 2's continuous
`nvidia-smi` trace peaks at **32,038 of 32,607 MiB — 569 MiB free** during decode. Every
healthy configuration measured here peaks around 31,150–31,300 MiB (~1.3–1.4 GiB free).

The docs walked the operator into it: `docs/windows-qwen38-flash-next-mmap.md` said "lower
`-MoECacheSize` by roughly 512 slots per owned layer" and then gave `-MoECacheSize 4400` as the
worked example. 6750 − 6×512 is 3678, not 4400.

## The fix

**`19286e1` `fix(moe): charge GPU-owned layers to --moe-cache-size, not on top of it`.**

`--moe-cache-size` is the *total* expert-slot budget on the card. The GPU-owned layers hold one
full expert layer each and are now charged to it, so switching `--moe-gpu-owned-layers` on
trades LRU slots for resident layers and can never raise total MoE residency above the number
the operator asked for. `--moe-cache-auto` is untouched: it already charges the same bytes
through `fixed_cache_size` before the MoE-vs-KV split.

* `engine/cache_budget.py`: `gpu_owned_reservation_slots()` (the slot twin of the existing
  `gpu_owned_reservation_bytes()`) and `lru_slots_after_owned_charge()`, which does the split
  and raises — naming the size that works — when the remainder falls under the prefill-overlap
  floor.
* `engine/engine.py`: `Engine._charge_gpu_owned_layers_to_cache_size()`, called once from
  `_init_offload_moe_cache` before anything reads the size; `_validate_gpu_owned_layers`'s
  floor check now measures the LRU *left after the charge*, not the number typed.
* Docs corrected in `docs/cli.md` and `docs/windows-qwen38-flash-next-mmap.md`, including the
  worked example that caused this.
* Tests (CPU-only, `CUDA_VISIBLE_DEVICES=-1`): 8 new cases across
  `tests/engine/test_cache_budget.py` and `tests/engine/test_moe_gpu_owned_layers.py`,
  including the neutrality property itself — for every owned count, `lru * per_expert +
  owned_bytes` is invariant. Suite before the change: 33 failed / 321 passed; after: the same
  33 failures (pre-existing: flashinfer not installed, and the `test_small_prefill_movement`
  group) / 329 passed.

Boot output after the fix:

```
INFO --moe-cache-size 6750 is the total MoE expert-slot budget: 6 GPU-owned layer(s) hold
     3072 of those slots, leaving 3678 for the streaming-layer LRU
INFO MoE GPU-owned layers: [0, 1, 2, 6, 7, 22] (6 x 1.32 GiB resident, no host bank);
     LRU cache 3678 slots for 42 streaming layers
```

## Post-fix numbers

`-GpuOwnedLayers auto -MoECacheSize 6750 -CollectRoutingStats`, boot E3, `19286e1`:

| quantity | run 2 candidate | **post-fix** | accepted baseline |
|---|---|---|---|
| boot to `serving` | 64.2 s | **59.5 s** | ~71 s |
| 8k-chat 512-tok decode, warm | 34.6 tok/s | **64.7 tok/s** (64.65 / 64.65 / 64.77) | 69.0–73.7 (session start) / 61.2–63.3 (session end) |
| warm TTFT, same prompt | 1.51 s | **1.16 s** | 1.30–1.38 s |
| cold TTFT, same prompt | 10.01 s | **7.64 s** | 9.26 s (fresh boot, same window) |
| first picture of the boot | 76.91 s | **4.27 s** | 6.62 s [run 2] |
| warm picture | 5.19 / 6.74 s | **3.01 / 3.03 s** | 4.68 s [run 2] |
| picture answer | `738214` | **`738214`** | `738214` |
| 48-token temp-0 md5 | match | **match, 3/3** | `29f0e74dfed744b538f856f38553e1ae` |
| free VRAM after initialization | 3.27 GiB | **5.20 GiB** | 4.41 GiB (E1) / 4.58 GiB [run 2] |
| streaming-layer LRU | 4400 | 3678 | 6750 |
| expert fetches per decode step | 91.8 | 105.8 | 103.5 [skew study] |
| whole-system physical in use, idle | 80.33 GiB | **81.55 GiB** | 88.11 GiB |
| scheduler working set | 59.13 GiB | **60.70 GiB** | 67.78 GiB |
| whole-system commit, idle | 209.37 GiB | **208.37 GiB** | 216.80 GiB |
| `nvidia-smi` idle | 31,566 MiB | **29,971 MiB** | 30,876 MiB |

**RAM saving, measured against the baseline restored in the same session, adjacent boots:**
whole-system physical in use **−6.56 GiB**, scheduler working set **−7.08 GiB**, whole-system
commit **−8.43 GiB**, against the predicted 6 × 1.322 = 7.93 GiB of host banks never
allocated. (Run 2's figures for the same quantities were −7.50 / −8.02 / −4.37 GiB; physical
in-use is the noisiest of the three because the rest of the box moves under it.) Scheduler
private bytes barely move (100.21 → 99.26 GiB) — as run 2 established, the host expert banks
are mapped/pinned territory, not private commit, so that criterion is measuring the wrong
counter.

**Speed, honestly stated.** The box drifted ~8 % slower over the session: the accepted server
measured 69.0/73.7 tok/s at the start and 58.3–63.3 tok/s when restored at the end, on the same
prompt with the same script. Against the run-2 baseline of 70.4 the post-fix figure is **−8 %**;
against the baseline measured minutes later in the same window it is **at parity or slightly
above**. The E1 control (same branch, no owned layers, 67.6 tok/s median) sits between the two.
The defensible reading: **the feature now costs somewhere between 0 and 9 % of decode, not
51 %**, and that residual is the honest price of trading 3,072 LRU slots for 6 resident layers
— the streaming layers' median miss rate goes 0.143 → 0.229 and fetches per step 81 → 106.

So the trade is now the one the design intended: **~7 GiB of host RAM back for ≤9 % of decode,
at unchanged VRAM, with byte-identical answers.**

## What remains

1. **`net_cache_budget_bytes` does not know about anything allocated after the MoE cache.**
   The resident MTP draft head is 2.17 GiB and lands after sizing; the vision encoder's
   transient workspace lands later still. The accepted 6750-slot configuration only works
   because it happens to leave ~3 GiB of the budget unspent. This is a **pre-existing,
   general** bug, not an owned-layers one, and it is still live: `--moe-cache-auto` on this box
   would resolve to ~7,900 slots (20.4 GiB) and walk straight into the wall run 2 hit — with or
   without `--moe-gpu-owned-layers`. The fix here makes the owned feature VRAM-neutral so it
   can no longer *trigger* the bug; it does not fix the bug. A proper fix subtracts a
   measured/estimated post-cache resident term from the budget.
2. **The paging mechanism is inferred, not instrumented.** The evidence is behavioural and
   consistent (569 MiB free at peak; 2× decode, 3× cold prefill, 11× first vision encode; all
   three cured by returning the bytes) but nothing here read a WDDM eviction counter. If it
   matters, `nvidia-smi --query-gpu=memory.used` sampled at 250 ms alongside a
   `Get-Counter '\GPU Local Adapter Memory(*)\*'` trace would settle it.
3. **`rebuild_runtime_cache(moe_cache_size=…)` still takes a raw LRU size**, not a total
   budget. It is a separate API on an already-built cache, so it cannot reproduce this failure
   from the flags — but its argument now means something different from `--moe-cache-size`.
4. **`-MoECacheSize` below `4096` with `auto`** (6×512 charge + a 1024-slot overlap floor) now
   refuses to boot, naming 4096. That is intended, but it is a behaviour change for anyone who
   had scripted the old "lower it yourself" advice.
5. **No baseline routing capture at 3678 slots without owned layers**, so the ≤9 % residual is
   attributed to the LRU by inference from the sweep and the miss-rate shift, not measured
   against a same-size no-owned control. One more boot would close it.
6. **`--moe-cache-auto` + `--moe-gpu-owned-layers` has never been run live.** It charges the
   reservation through a different path (`fixed_cache_size`) and is untested on the box.

## Scratchpad

`C:\Users\jay\AppData\Local\Temp\claude\D--FreeToken\14eaa0eb-ee91-4e4f-a9c5-2096e88229bd\scratchpad\gpu-owned-live\run3`
— `boot-2030.ps1`, `stop-server.ps1`, `snap.ps1`, `chat8k.py`, `pic2.py`,
`server-{e1,e2,fix}-2030.{out,err}.log`, `bootmem-{e1,e2,fix}-2030.csv`,
`chat8k-{base3,e1,e2,fix,rest3,rest3b}*.txt`, `answer-*.txt`, `pic-{e1,e2,fix}-*.json`,
`routing-{e1,e2,fix}.json`, `status-{e1,fix}.json`, `snaps.jsonl`.
Run 2's evidence is one directory up in `run2/`; the baseline routing capture this diagnosis
turns on is `docs/research/routing-skew-2026-09-02/chat8k.json`.
