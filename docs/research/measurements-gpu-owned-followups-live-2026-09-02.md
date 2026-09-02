# GPU-owned MoE follow-ups: live run (run 4), 2026-09-02

Operator verification of the seven follow-up fixes on `gpu-owned-followups` @ `9421384`
(worktree `D:\FreeToken-gpu-owned-followups`). Same box as runs 1-3: Windows 11 Pro 26200,
RTX 5090 32 GB (32,607 MiB), 95.6 GiB RAM, `D:\Models\Qwen3.8-Flash-Next-NVFP4`, candidate
port 2030, accepted server on 2020 from `D:\FreeToken` @ `0964145`.

Five candidate boots, one server at a time, `nvidia-smi` < 3 GB and a 60 s settle before
each. Only `python.exe` carrying `freetoken.cli serve` and their python descendants (plus
one real `spawn_main` orphan) were killed; `ft.exe daemon --port 1900` was never touched and
was confirmed alive and listening after every stop. `D:\Models` untouched, no source file
modified, nothing committed or pushed.

Evidence: `<scratch>\gpu-owned-live\run4\` (`run3\` was left intact — it is the evidence the
2026-09-02 speed diagnosis cites, so this run wrote to `run4\` instead).

## Headline

1. **The corrected budget refuses the configuration the checklist was written around.**
   `-GpuOwnedLayers auto -MoECacheSize 6750` — the flags now in `boot-2020.ps1`, the flags
   run 3 measured at 64.7 tok/s — no longer boots on this box. The 3.00 GiB post-cache
   reserve plus the 1.50 GiB headroom do not fit beside 6750 slot-equivalents. This is the
   fix working as designed (open risk 3), but it means L1/L2/L3/L6/L7/L11/L12 had to be run
   at the largest size that *does* fit.
2. **The refusal names a size that cannot be typed.** It reports the *post-charge LRU* count
   under the name `--moe-cache-size`, which is the *total* budget. An operator who follows
   the message verbatim gets a second refusal. Two message defects, detail below. **L4 fails.**
3. **The refusal is substantively right.** The configuration it refuses (reproduced as boot E
   with both knobs zeroed) peaks at **1,112 MiB free VRAM** during the 8k decode — under the
   1.5 GiB headroom the check demands — and runs **55.2 tok/s**, while the configuration the
   check allows (boot C, a *smaller* cache) peaks at **3,029 MiB free** and runs **60.0
   tok/s**. Smaller cache, more free VRAM, 9 % faster: the same signature the run-3 diagnosis
   used to convict memory pressure.
4. The stop script (L9, L10) worked exactly as specified against real processes, including a
   real ZMQ-port orphan that occurred on its own.

## Boots

| # | Tag | Flags (beyond the common boot-2020 set) | Result |
|---|---|---|---|
| 1 | A | `-GpuOwnedLayers auto -MoECacheSize 6750 -CollectRoutingStats` | **REFUSED** after the expert load, `check_explicit_moe_cache_fits` |
| 2 | B | `... -MoECacheSize 2709` (the size boot A named) | **REFUSED** at config time, `lru_slots_after_owned_charge` floor |
| 3 | C | `... -MoECacheSize 5781` (= 2709 + the 3072-slot owned charge) | **SERVING** in 60.6 s — the full checklist ran here |
| 4 | D | `-MoECacheSize 0` (auto) `-MoEVramReserveBytes 0 -MoECacheHeadroomBytes 0` | **SERVING** in 60.5 s — L5, ledger read, no decode load |
| 5 | E | `-MoECacheSize 6750 -MoEVramReserveBytes 0 -MoECacheHeadroomBytes 0` | **SERVING** in 58.1 s — run-3 E3 reproduction |

Common to all: `-ContextTokens 65536 -KVCacheTokens 65536 -MaxRunningRequests 1 -DenseQuant
int8 -EmbedHost -EnableVision -VisionExecution layer-stream -VisionWeights mmap
-EnableCacheReport`, integrated MTP speculation with the resident draft head,
`-CollectRoutingStats` on every candidate.

The baseline on 2020 as found at the start of the session was the **pre-feature** build
(no `moe_gpu_owned_layers` in its `ServerArgs`, no `GPU-owned MoE layers:` banner line, no
`gpu_owned_*` in its cache geometry, `moe_cache_size` 6750): it was booted before
`-GpuOwnedLayers auto -MoECacheSize 6750` was added to `boot-2020.ps1`. It is therefore a
clean no-owned-layers control.

## L1-L12

| # | Verdict | Evidence |
|---|---|---|
| L1 | **PASS** | Boot C log 17:37:20: exactly one `VRAM ledger` block, emitted immediately after `Allocating 65536 tokens for KV cache, K + V = 1.55 GiB`; the KV row reads **1.55 GiB**, not 0.00. Every named row present. |
| L2 | **PASS** | Ledger `GPU-owned MoE layers 7.93 GiB` = geometry `gpu_owned_reserved_bytes` 8,517,058,560 B; `MoE LRU cache 6.99 GiB (2709 slots)` = geometry `moe_cache_size` 2709 x `unit_bytes.moe_per_expert` 2,772,480; `KV cache 1.55 GiB` = `num_pages` 1024 x `page_size` 64 x 25,344 B. `unaccounted` **+3.03 GiB**, positive, and exactly `(1 - memory_ratio 0.9) x 30.25 GiB` — the allocator/activation remainder, not slack the plan lost. `nvidia-smi` at `serving` 27,125 MiB ~= 22.72 GiB of named allocations + 2.17 GiB draft head + ~1.6 GiB desktop. |
| L3 | **PASS (boot C)** | `nvidia-smi` sampled at 250 ms across the 8k-chat decode (193 samples): peak 29,578 of 32,607 MiB -> **3,029 MiB (2.96 GiB) free at decode peak**, well over 1.5 GiB, no throughput cliff. The counter-case is boot E (reserve and headroom zeroed): peak 31,495 MiB -> **1,112 MiB free**, i.e. the check's 1.5 GiB headroom is not a theoretical margin on this box. |
| L4 | **FAIL** | The refusal fires (boot A) and names a size, but **that size does not boot** (boot B). Two defects, below. |
| L5 | **PASS** | Boot D: `--moe-cache-auto resolved moe_cache_size=4452 num_pages=1025`; 4452 LRU + 3072 owned = **7,524 slot-equivalents** = 20,860,139,520 B, which is the entire net MoE budget (20,861,318,737 B) to within one slot — the pre-2026-09 sizing exactly. Ledger `post-cache reserve 0.00 GiB` / `headroom 0.00 GiB` as typed. `Free memory after initialization: 3.14 GiB` (vs 8.12 GiB for boot C), which after the 2.17 GiB draft head is the ~1 GiB run 2 crashed into. (The diagnosis doc's "~7,900 slots" estimate is 7,524 in fact.) |
| L6 | **PASS** | With `-GpuOwnedLayers auto`: `gpu_owned_reserved_bytes: 8517058560`, `gpu_owned_layers: [0,1,2,6,7,22]`, `limits.moe_experts.max: 5427`. On the pre-feature baseline the field is **absent** and `limits.moe_experts.max` is **8499**. |
| L7 | **PASS** | Boot C log, one block, verbatim below; contains `Expert placement: ... gpu_owned_layers=[0, 1, 2, 6, 7, 22], cpu_layers=[], streaming_layers=42, lru_slots=2709` and `PLE table: backend=mmap, mapped_bytes=51200245760, layers=1`. The dedicated `MoE GPU-owned layers: [0, 1, 2, 6, 7, 22] (6 x 1.32 GiB resident, no host bank); LRU cache 2709 slots for 42 streaming layers` line is still present, one line above. The re-homed `Token embedding: host-resident, bytes=1271398400, device=cpu` still reads sensibly (open risk 6). |
| L8 | **NOT TESTABLE** | `D:\Models` contains exactly one checkpoint (`Qwen3.8-Flash-Next-NVFP4`, NVFP4). Nothing was downloaded. |
| L9 | **PASS** | `-DryRun -Port 2020` listed exactly four processes under "FreeToken server processes": pids 50096 (launcher python), 25160 (serve), 20416 and 24704 (`spawn_main` children); "(none)" under orphans. `ft.exe` (pid 59556) and its two python processes (32800, 75392, port 1900) were **not** listed, and neither was the unrelated `omp-python-runner` python pair. The real run killed those four and reported `VRAM in use: 1618 MB`, `Ports 2020-2029: free`, `Settled`, exit 0; `ft.exe` was still `LISTENING` on 127.0.0.1:1900 afterwards. |
| L10 | **PASS (real orphan, not synthesised)** | Boot A's refusal left `python.exe` pid 89892 (`spawn_main(parent_pid=49804 ...)`) alive with parent 49804 gone, **LISTENING on 127.0.0.1:2033**. `-DryRun -Port 2030` printed "(none running)" under servers and pid 89892 under "Orphaned multiprocessing children"; the real run killed it and reported settled. The next boot (B) started with no `ZMQError: Address in use`. Reproduced a second time after boot B (pid 95480), same outcome. |
| L11 | **PASS with a caveat** | See the speed table. At the largest size the corrected budget allows (boot C, 2709 LRU) the candidate runs **60.0 tok/s median warm** against the same-session no-owned baseline's **58.5 tok/s median** — parity. The run-3 target of "~63-65 at 6750" is not reachable: 6750 is refused, and forcing it with zeroed knobs (boot E) gives 55.5 tok/s on a box running ~15 % slower today than during run 3. |
| L12 | **INCONCLUSIVE (short of the band)** | Idle-to-idle, boot C vs the no-owned baseline: scheduler **working set -4.98 GiB** (64.13 -> 59.15), whole-system **physical in use -5.49 GiB** (85.45 -> 79.96), whole-system **commit -13.11 GiB** (216.75 -> 203.64). The criterion is -7.9 +/- 0.5 on working set. The shortfall is in the baseline, not the candidate: the candidate's absolute figures (59.1-61.0 GiB WS, 79.2-81.8 GiB physical) match run 3's post-fix figures (60.70 / 81.55) almost exactly, while today's baseline read 3.6 GiB *lower* than run 3's baseline (64.13 vs 67.78) — it had been serving for 75 minutes and Windows had trimmed its working set. Not re-measurable this session without a sixth boot. |

## L4 in detail: the two message defects

Boot A, `-GpuOwnedLayers auto -MoECacheSize 6750`, refused after the 40 s expert load:

```
ValueError: --moe-cache-size 3678 plus 6 GPU-owned MoE layers (8517058560 B resident) plus
4831838208 B of post-cache reservations (--moe-vram-reserve-bytes + --moe-cache-headroom-bytes)
needs 23546078208 B of the 20861318737 B MoE budget. Either lower --moe-cache-size to 2709
slots, or own at most 4 layer(s) at this cache size.
```

Two lines above it, the same boot logged:

```
--moe-cache-size 6750 is the total MoE expert-slot budget: 6 GPU-owned layer(s) hold 3072 of
those slots, leaving 3678 for the streaming-layer LRU
```

1. **It quotes a size the operator never typed.** `_charge_gpu_owned_layers_to_cache_size`
   (`engine.py:873`) rewrites `config.moe_cache_size` from 6750 to the post-charge LRU 3678;
   `_check_explicit_cache_fits` (`engine.py:969`) then reads that rewritten value and prints
   it as "`--moe-cache-size 3678`". The operator typed 6750.
2. **The size it names cannot be typed.** `fits_slots` (2709) is likewise an LRU count, but
   the message says "lower `--moe-cache-size` to 2709 slots" — and `--moe-cache-size` is the
   *total*. Boot B typed exactly that and was refused by the other check:

   ```
   ValueError: --moe-cache-size 2709 is the TOTAL expert-slot budget, and 6 GPU-owned MoE
   layer(s) charge 3072 slots of it (6 x 512 experts), leaving -363 for the LRU -- but the
   streaming layers need at least 1024. Raise --moe-cache-size (launcher: -MoECacheSize) to
   at least 4096, or own fewer layers.
   ```

   The size that actually works is 2709 + 3072 = **5781**, which is what boot C used.

Suggested fix: `_check_explicit_cache_fits` should carry the operator's original total (stash
it in `_charge_gpu_owned_layers_to_cache_size`) and report both the typed size and
`fits_slots + gpu_owned_reservation_slots(...)` as the size to type. The `own at most 4
layer(s)` clause is correct as written.

A third, smaller point: the check runs **after** the ~40 s parallel expert load (boot A spent
44 s before refusing) even though every term in it is known at config time. Boot B, which is
refused in `_adjust_config`, failed in under 15 s.

## The ledger, verbatim (boot C)

```
VRAM ledger (30.25 GiB on the card):
  weights               5.28 GiB
  KV cache              1.55 GiB
  GDN state pool        0.97 GiB
  GPU-owned MoE layers  7.93 GiB (6 layers [0, 1, 2, 6, 7, 22])
  MoE LRU cache         6.99 GiB (2709 slots)
  post-cache reserve    3.00 GiB (MTP draft head, graphs, vision)
  headroom              1.50 GiB
  unaccounted           3.03 GiB
  named total           27.22 GiB
```

Boot E, the same card with both knobs zeroed:

```
VRAM ledger (30.25 GiB on the card):
  weights               5.28 GiB
  KV cache              1.55 GiB
  GDN state pool        0.97 GiB
  GPU-owned MoE layers  7.93 GiB (6 layers [0, 1, 2, 6, 7, 22])
  MoE LRU cache         9.50 GiB (3678 slots)
  post-cache reserve    0.00 GiB (MTP draft head, graphs, vision)
  headroom              0.00 GiB
  unaccounted           5.03 GiB
  named total           25.23 GiB
```

Boot D, auto with both knobs zeroed — `MoE LRU cache 11.50 GiB (4452 slots)`, reserve and
headroom 0.00, `named total 27.23 GiB`, `Free memory after initialization: 3.14 GiB`.

## The placement block, verbatim (boot C)

```
PLE table: backend=mmap, mapped_bytes=51200245760, layers=1
Token embedding: host-resident, bytes=1271398400, device=cpu
Picture weights: mode=layer-stream, backing=mmap, tensors=333, bytes=897862112, devices=cpu
Dense weights: quant=int8
Expert placement: backend=offload, moe_layers=48, experts=512,
  gpu_owned_layers=[0, 1, 2, 6, 7, 22], cpu_layers=[], streaming_layers=42, lru_slots=2709
```

One block, one log call. (The PLE line sorts first because it is the only line carrying the
logger prefix; the other four are continuation lines of the same record.)

## Speed and VRAM

8k-chat probe (`chat8k.py`, 6,968-token prompt, 512 tokens, `temperature 0`, thinking off,
decode tok/s from the streamed usage block). Rep 1 of each boot is cold.

| config | LRU | total slot-equiv | free after init | free at decode peak | cold TTFT | warm TTFT | decode tok/s | 48-tok md5 |
|---|---|---|---|---|---|---|---|---|
| baseline 2020 (pre-feature, no owned) | 6750 | 6750 | 4.71 GiB | not sampled | — (warm) | 1.32-1.56 s | 66.75 / 51.46 / **58.53** | match |
| **boot C** `5781`, default reserve | 2709 | 5781 | **8.12 GiB** | **3,029 MiB** | 7.63 s | 1.15-1.19 s | 60.66 / 59.28 / **59.98** | match |
| boot E `6750`, reserve+headroom 0 | 3678 | 6750 | 5.13 GiB | **1,112 MiB** | 8.48 s | 1.19-1.20 s | 54.61 / 55.30 / **55.73** | match |
| boot D auto, reserve+headroom 0 | 4452 | 7524 | 3.14 GiB | not run (would page) | — | — | — | — |
| restored 2020 (`auto` + 6750 -> 3678 LRU) | 3678 | 6750 | — | — | 8.36 s | 1.17-1.22 s | 60.12 / 60.42 | match (warm) |

Vision, `verify-shot.png`, same prompt, answer `738214` in every case:

| config | first picture | warm picture |
|---|---|---|
| boot C | 4.57 s | 3.11 s |
| boot E | 4.70 s | 3.39 s |

Routing (`/v1/cache/routing`, `-CollectRoutingStats`):

| config | resident rows | streaming rows | streaming miss_rate min/med/max | fetches per decode step |
|---|---|---|---|---|
| boot C (2709 LRU) | 6, all `miss_rate: null` | 42 | 0.193 / 0.296 / 0.390 | 140.23 |
| boot E (3678 LRU) | 6, all `miss_rate: null` | 42 | 0.149 / 0.221 / 0.289 | 103.44 |
| run-3 E3 (3678 LRU) | 6 | 42 | — / 0.229 / — | 105.8 |

Boot E reproduces run-3 E3 to within noise on every structural quantity (LRU 3678, free after
init 5.13 vs 5.20 GiB, fetches/step 103.4 vs 105.8, median miss 0.221 vs 0.229) but decodes at
55.5 rather than 64.7 tok/s — the box is running ~15 % slower this session (today's no-owned
baseline reads 58.5 tok/s median against run 3's 69.0-73.7 at session start).

**The comparison that matters for the fix:** boot C holds *969 fewer* LRU slots than boot E
and does *36 % more* expert fetches per decode step, yet runs **8 % faster** (60.0 vs 55.5)
with 1.9 GiB more free VRAM at peak. That is the H7 signature again, and it is direct
evidence that the 1.5 GiB headroom the new check enforces is real on this box rather than
conservative.

## RAM

`snap.ps1` (whole-system committed/available from `\Memory\*`, physical in use from
`Win32_OperatingSystem`, engine process picked by private bytes).

| snapshot | commit | physical in use | engine private | engine working set | nvidia-smi |
|---|---|---|---|---|---|
| no server | 48.75 GiB | 18.92 GiB | — | — | 1,735 MiB |
| baseline 2020, idle (no owned) | 216.75 GiB | 85.45 GiB | 100.21 GiB | 64.13 GiB | 31,354 MiB |
| boot C, idle | 203.64 GiB | 79.96 GiB | 95.64 GiB | 59.15 GiB | 27,141 MiB |
| boot C, after load | 205.20 GiB | 80.50 GiB | 96.68 GiB | 60.55 GiB | 27,870 MiB |
| boot E, idle | 206.40 GiB | 79.18 GiB | 98.14 GiB | 59.14 GiB | 29,749 MiB |
| boot E, after load | 208.01 GiB | 81.79 GiB | 99.22 GiB | 60.97 GiB | 29,869 MiB |
| restored 2020 (owned), after load | 208.41 GiB | 79.43 GiB | 100.16 GiB | 59.66 GiB | 31,383 MiB |

Boot C against the no-owned baseline, idle to idle: **working set -4.98 GiB**, **physical in
use -5.49 GiB**, **commit -13.11 GiB**, against the 7.93 GiB of host expert banks the six
owned layers never allocate. Short of the -7.9 +/- 0.5 criterion, but see L12: the candidate's
absolute figures match run 3's post-fix boot almost exactly and the baseline is the term that
moved. Scheduler private bytes 100.21 -> 95.64 GiB, again not the counter to read.

## Anything else worth recording

* **The first (cold) request of a boot can produce a different `temperature 0` answer.** The
  restored 2020's cold 48-token probe hashed `95f5c65811bdc16355213511b2e7afdf`; the next two
  warm probes on the same server both hashed the reference
  `29f0e74dfed744b538f856f38553e1ae`. All the candidate md5s in this document were taken warm,
  as in run 3. Worth knowing before someone reads a cold mismatch as a correctness failure.
* **`unaccounted` in the ledger is not slack.** It is exactly `(1 - memory_ratio) x baseline
  free` (0.1 x 30.25 = 3.03 GiB) plus whatever the reserve rows over- or under-state; it was
  3.03 GiB on both boots that spend the whole budget. A reader looking for "allocator slack"
  in that row will misread it.
* **The check refuses the *baseline* configuration too.** `_check_explicit_cache_fits` runs
  for every explicit `--moe-cache-size` with or without owned layers, and 6750 slots
  (17.43 GiB) + 4.5 GiB of reserve and headroom exceeds the 19.43 GiB net MoE budget on this
  box by 2.5 GiB. Anyone on this branch who boots the long-standing `-MoECacheSize 6750` with no
  owned layers at all will be refused. That is open risk 3 landing, and on this box it lands
  on the default.
* **No hang, no `cudaHostRegister` failure, no `ZMQError`** across five candidate boots and
  the restore.

## Final state

The 2020 server is back up from `D:\FreeToken` @ `0964145` via the unmodified
`boot-2020.ps1`, `state == serving`, answering correctly (60.12 / 60.42 tok/s warm, reference
md5 on both warm probes). Note that `boot-2020.ps1` now carries `-GpuOwnedLayers auto
-MoECacheSize 6750`, so the restored server is **not** the configuration that was running at
the start of the session: it now owns layers `[0, 1, 2, 6, 7, 22]` with a 3678-slot LRU. It
boots because `0964145` predates the post-cache reserve check.
