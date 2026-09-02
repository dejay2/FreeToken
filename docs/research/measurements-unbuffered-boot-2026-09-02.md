# Windows unbuffered expert-shard reads: live boot verification (2026-09-02)

Live A/B on the operator box (Windows 11 Pro 26200, RTX 5090 32 GB, 95.56 GiB RAM,
`D:\Models\Qwen3.8-Flash-Next-NVFP4`, 192 expert shards / 63.3 GiB).

* **fix** = `leakfix-verify` @ `750d83d` ("perf(windows): read expert shards unbuffered so
  the boot load never caches"), booted from the worktree
  `D:\FreeToken\.claude\worktrees\leakfix-verify`.
* **old** = the accepted tree `D:\FreeToken` @ `748bd97` (= `750d83d^`).

Both boots used the identical launcher (`scripts/start-qwen38-flash-next-mmap-windows.ps1`,
byte-identical in the two trees), identical flags and identical env, on port 2020, back to
back, each after the previous server was stopped, the GPU drained below 3 GB and ~60-80 s of
settle. Order: [accepted server already up] -> reference capture -> stop -> **fix boot** ->
correctness + benchmark -> stop -> **old boot** -> correctness + benchmark. The accepted
server is running again at the end.

## Headline

The fix does exactly what it claims: **the 63.3 GiB expert load no longer puts anything on
the Windows standby (file-cache) list.** Peak standby during the load falls from 42.25 GiB to
11.79 GiB, and the standby delta *across the expert load* falls from **+30.1 GiB to +0.0 GiB**.
Boot to `serving` is **9.7 s faster** (82.8 s -> 73.1 s); the expert-load phase alone is **15 s
faster** (58 s -> 43 s), because the cache manager no longer has to build and then forcibly
evict 30 GiB of pages it will never use. No correctness or throughput regression.

## Side by side

| | old (`748bd97`, `D:\FreeToken`) | fix (`750d83d`, worktree) |
|---|---|---|
| launch -> `/health` 200 | 8 s | 8 s |
| launch -> `state == "serving"` | **82.8 s** | **73.1 s** (-11.7 %) |
| expert-load phase (`expert banks: slow path` -> `NVFP4 expert backend`) | 07:14:50 -> 07:15:48 = **58 s** | 07:09:45 -> 07:10:28 = **43 s** (-26 %) |
| reader path taken | serial safetensors/`mmap` (buffered) | serial via `DirectShard` / `win_io` (`FILE_FLAG_NO_BUFFERING`) |
| standby entering the expert load | 12.17 GiB | 11.79 GiB |
| **peak standby during boot** | **42.25 GiB** | **11.79 GiB** |
| **standby delta across the expert load** | **+30.08 GiB, then force-trimmed back to 11.42** | **+0.00 GiB (flat)** |
| peak physical used (`TotalVisible - FreePhysical`) | 84.13 GiB | 85.38 GiB |
| min Available MBytes during boot | 11.42 GiB | 10.25 GiB |
| peak Committed Bytes | 207.87 GiB | 207.88 GiB |
| steady physical used (serving + 30 s) | 83.30 GiB | 83.99 GiB |
| steady standby (serving + 30 s) | 12.24 GiB | 11.50 GiB |
| GPU at serving | 28.5 GiB | 28.5 GiB |
| warnings / fallbacks in the logs | none | none (no `unbuffered`, `fallback`, `win_io`, `Traceback` hits) |
| `/v1/cache/status` geometry | identical | identical |

Monitor CSVs (3 s sampling): [`measurements-unbuffered-boot-2026-09-02-old.csv`](measurements-unbuffered-boot-2026-09-02-old.csv),
[`measurements-unbuffered-boot-2026-09-02-fix.csv`](measurements-unbuffered-boot-2026-09-02-fix.csv).

## The trace that settles it

`sb` = standby total (normal + core + reserve), `phys` = physical used, GiB.

**old** — the expert load starts at 07:14:50 and standby climbs ~1.2 GiB/s until the machine
runs out of room at 07:15:20, after which every further GiB of pinned bank has to be paid for
by evicting a standby page:

```
07:14:51  sb=12.17  phys=20.04   <- expert banks: slow path (serial build)
07:14:59  sb=20.74  phys=29.18
07:15:08  sb=30.64  phys=39.25
07:15:16  sb=39.87  phys=48.83
07:15:20  sb=42.25  phys=53.62   <- peak standby; reclaim takes over here
07:15:29  sb=32.83  phys=62.95
07:15:37  sb=23.70  phys=72.15
07:15:45  sb=14.16  phys=81.19
07:15:49  sb=11.42  phys=84.13   <- NVFP4 expert backend: triton (07:15:48)
```

The 42.25 GiB ceiling is set by RAM, not by the workload: the shards are 63.3 GiB, so on a
larger box the standby list would have grown further.

**fix** — same phase, standby does not move at all while 63.5 GiB of banks are pinned:

```
07:09:47  sb=11.79  phys=20.73   <- expert banks: slow path (serial build) at 07:09:45
07:09:55  sb=11.76  phys=28.06
07:10:04  sb=11.78  phys=42.51
07:10:12  sb=11.79  phys=57.77
07:10:21  sb=11.79  phys=72.32
07:10:25  sb=11.79  phys=80.08
07:10:29  sb=10.41  phys=84.74   <- NVFP4 expert backend: triton (07:10:28)
```

The ~11.8 GiB of standby present in both runs is *not* expert shards — it is the dense/int8
weights, the PLE mmap tables, the token embedding (1.27 GiB host-resident), the picture
weights (0.90 GiB) and the Python/torch/vision imports, which are still read through the
normal cached paths. In the fix run that ~9.3 GiB was created between 07:09:36 and 07:09:47
(before/at the very start of the expert load) and then stayed flat; in the old run it was
already resident from the preceding boot. Both runs therefore enter the expert load from the
same ~12 GiB baseline, which makes the +30.1 vs +0.0 delta a clean like-for-like.

### On "peak whole-system RAM should fall from 87-91 GiB"

Not reproduced as stated. In this pair the peak physical-used was 84.13 GiB (old) and
85.38 GiB (fix) — the fix was 1.25 GiB *higher*, not lower. That is expected once you look at
what the counter measures: `TotalVisible - FreePhysical` counts standby pages as used, and in
the old run the memory manager simply **trims standby to stay under the same ceiling**. Both
runs end the load pressed against the wall (min available 11.42 vs 10.25 GiB). What differs is
the *quality* of the headroom: at the mid-load point the old run's 42 GiB of "available" was
almost entirely standby that had to be reclaimed, whereas the fix run's headroom was genuinely
free (e.g. at phys=64.9 GiB the fix had 30.8 GiB available of which only 11.8 GiB was standby,
i.e. ~19 GiB actually free). So the correct claim is **"the boot no longer generates 30+ GiB of
throw-away file-cache pressure"**, not "peak RAM drops". Any earlier 87-91 GiB peak was not
observed in this session on either build.

## Reader path actually exercised

The launcher hard-codes `--expert-load serial` (line 151 of
`scripts/start-qwen38-flash-next-mmap-windows.ps1`), so **the parallel expert reader that the
fix newly enables on Windows was not used in either boot**. Both boots logged
`expert banks: slow path (serial build)`. The win in this measurement therefore comes entirely
from the `nvfp4_banks.py` serial loader's `safe_open` -> `DirectShard`/`win_io` change, not
from `_PARALLEL_READER_SUPPORTED`. The parallel path is untested by this run; it would need
`--expert-load parallel` (or a launcher change) to exercise.

No fallback fired: greps for `unbuffered`, `buffered`, `fallback`, `win_io`, `Traceback` over
`server-fix.out.log` / `server-fix.err.log` returned nothing, i.e. `win_io.read_file_into`
never raised `OSError` on the `D:` volume and the warn-once buffered fallback in
`read_shard_direct` was never reached. The only stderr noise is the usual pre-existing
`qwen4_exp` transformers warning, the `expandable_segments` warning and the c10d socket
message — identical on both builds.

## Correctness

`/v1/cache/status` geometry is **byte-identical** across old (before), fix, and old
(restored): `num_pages=1024`, `page_size=64`, `moe_cache_size=6750`, `num_experts=512`,
`num_moe_layers=48`, `moe_per_expert=2772480`, `cache_budget_bytes=23564753305`, same limits
and reasoning gears.

Short chat (`enable_thinking=false`, "What is the capital of France? Answer in one word.")
returned exactly `Paris` on every server and every run (old x3, fix x2, old-restored x1).

**The engine is not deterministic at `temperature=0`, on either build.** Three consecutive
greedy 200-token completions of the same prompt on the *unchanged* accepted server produced
three different texts (common prefixes of 242 / 592 / 242 characters before diverging on a
near-tie token). So a byte-for-byte old-vs-fix comparison is impossible, and its failure would
not have been evidence of a fix bug. What was compared instead:

* Five short greedy probes (24 tokens), two runs each, on both builds. **4 of 5 produced
  byte-identical text across old and fix**; the fifth ("Roses are red, violets are") diverged,
  but the old server itself produced two different answers for two of the five probes, so this
  is inside the observed nondeterminism envelope.
* The 200-token bicycle explanation: fix run B opens with a 242-character prefix identical to
  old run A and is semantically the same explanation (gyroscopic effect is not the reason;
  trail / caster geometry is); every variant on both builds gives the same physics.

Verdict: **no correctness difference detectable above the engine's own run-to-run noise.**
Anyone wanting a byte-exact regression gate here needs to fix the nondeterminism first (it
predates this commit).

## Speed

`ab_send.py`, 2 reps, decode tok/s (median) and TTFT seconds. `old (restored)` was measured in
the same session immediately after the fix, so it is the like-for-like column; the sweep column
is the earlier fresh-6750 boot from `measurements-moe-cache-sweep-2026-09-02.md`.

| | sweep fresh-6750 | old (restored, same session) | fix | fix vs old |
|---|---|---|---|---|
| numbers | 142.0 | 138.6 | 138.5 | -0.1 % |
| essay | 76.8 | 77.7 | 75.0 | -3.5 % |
| code | 96.3 | 97.2 | 95.1 | -2.2 % |
| 8k-chat | 73.1 | 70.8 | 74.4 | **+5.1 %** |
| 8k-greedy | 74.6 | 72.4 | 71.6 | -1.1 % |
| cold-7k TTFT | 5.51 s | 5.66 s | 5.66 s | 0.0 % |
| warm-turn TTFT | 1.45 s | 1.37 s | 1.39 s | +1.5 % |

Spread is +-3 % with no consistent direction and the two largest movers point opposite ways,
so this is run-to-run noise: **no steady-state throughput change**, as expected — the fix only
touches boot-time IO.

## What failed / caveats

1. **The parallel reader was never exercised** (launcher forces `--expert-load serial`). The
   `_PARALLEL_READER_SUPPORTED` half of the commit is unverified live.
2. **The "87-91 GiB peak RAM" claim did not reproduce**; peak physical used was ~84-85 GiB on
   both builds and was 1.25 GiB *higher* on the fix. See the section above for why that counter
   is the wrong instrument for this change; peak standby is the right one and it moved 42.25 ->
   11.79 GiB.
3. **Byte-for-byte output comparison was impossible** — the engine is nondeterministic at
   temperature 0 on the unchanged build (documented above).
4. Only one boot per build. Boot times vary; the 9.7 s difference is consistent with the 15 s
   difference in the isolated expert-load phase, but a single pair is a single pair.
5. The standby list could not be purged between boots (no admin rights on this session), so
   each boot inherits the previous one's ~12 GiB of non-expert page cache. Both runs happened
   to enter the expert load at the same ~12 GiB, so this did not bias the comparison, but it is
   not a controlled zero.
6. `Committed Bytes` is dominated by a ~168 GiB system-wide baseline and carried no signal
   (207.87 vs 207.88 GiB).

## Exact commands

Scratchpad:
`C:\Users\jay\AppData\Local\Temp\claude\D--FreeToken\14eaa0eb-ee91-4e4f-a9c5-2096e88229bd\scratchpad`

```powershell
# boot-verify.ps1 = the accepted boot-2020.ps1 with the private-root/manifest/vision paths
# hard-coded under D:\FreeToken\.local (so -Root may be a worktree) and readiness polled on
# /v1/cache/status state=="serving" instead of /health. Every server flag and env var is
# unchanged from the accepted boot.

# memory monitor (3 s: standby normal/core/reserve, Available MBytes, Committed Bytes,
# Cache Bytes, TotalVisibleMemorySize-FreePhysicalMemory, nvidia-smi memory.used)
Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
  '-File','.\monitor.ps1','-Out','.\mon-fix.csv','-IntervalSec','3','-MaxMinutes','30' -WindowStyle Hidden

# stop only "freetoken.cli serve" python processes + their python descendants
.\stop-server.ps1            # -WhatIfOnly to preview targets

# fix
.\boot-verify.ps1 -Port 2020 -Root 'D:\FreeToken\.claude\worktrees\leakfix-verify' -Tag 'fix'
# old
.\boot-verify.ps1 -Port 2020 -Root 'D:\FreeToken' -Tag 'old'

# correctness + geometry
python refcap.py 2020 <tag>          # chat(enable_thinking=false) + greedy-200 + cache/status
python probe.py  2020 <tag>          # 5 x 24-token greedy probes, 2 runs each
python cmp.py old-a old-b old-c      # divergence report

# speed
python ab_send.py 2020 fix 2
python ab_send.py 2020 old 2

# trace summary
python csvstat.py .\mon-fix.csv .\mon-old.csv
```

Boot logs: `server-fix.out.log` / `server-fix.err.log`, `server-old.out.log` /
`server-old.err.log` in the same scratchpad.

## Verdict

**Ship it.** The fix eliminates the boot-time file-cache growth on Windows completely (+30.1
GiB -> +0.0 GiB of standby across the expert load), makes the expert load 26 % faster and boot
to `serving` 11.7 % faster, needs no fallback on this volume, and shows no correctness or
throughput regression. The one claim in the commit message that this run does not support is
the drop in peak whole-system RAM — that counter includes standby and the memory manager was
already trimming standby to hold the same ceiling; the change is real but it is a reduction in
wasted cache churn, not in peak footprint. The newly-enabled parallel Windows reader remains
untested because the launcher forces `--expert-load serial`.
