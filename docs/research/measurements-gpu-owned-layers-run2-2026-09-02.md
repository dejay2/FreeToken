# Live run 2: GPU-owned MoE layers — it boots, it is byte-correct, it costs half the decode

Measured 2026-09-02 on the live box: Windows 11 Pro 26200, RTX 5090 32 GB (32,607 MiB),
95.6 GiB system RAM, model `D:\Models\Qwen3.8-Flash-Next-NVFP4`.

- Candidate: branch `gpu-owned-layers` @ `2470573`, worktree `D:\FreeToken-gpu-owned-layers`,
  port 2030, `-GpuOwnedLayers auto -MoECacheSize 4400 -CollectRoutingStats`.
- Baseline: branch `mtp-upstream-merge` @ `cc085cf`, tree `D:\FreeToken`, port 2020,
  `-MoECacheSize 6750`, no `-GpuOwnedLayers`. (Run 1 recorded the baseline as `8591caf`;
  `cc085cf` is the docs-only plan commit on top of it — no code difference.)
- Everything else identical on both: 65,536 context / KV tokens, `-MaxRunningRequests 1`,
  `-DenseQuant int8 -EmbedHost -EnableVision -VisionExecution layer-stream -VisionWeights mmap
  -EnableCacheReport`, MTP speculation + spec graphs + resident draft head, private root and
  vision packages under `D:\FreeToken\.local\`.

## Verdict

**The loader fix (`8a63977`) works.** The candidate booted on the first genuine attempt, in
64.2 s to `state == serving` — *faster* than the baseline, not slower — and the six GPU-owned
layers are visible in the boot log, in `/v1/cache/status` and in `/v1/cache/routing`. The
48-token temperature-0 answer is **byte-identical to the baseline** (md5
`29f0e74dfed744b538f856f38553e1ae`, 3/3 samples), which is the strongest available evidence
that the resident VRAM rows carry the same weights as the host-bank path.

**The RAM saving is real and lands where the design predicted**: −7.50 GiB whole-system
physical in use, −8.02 GiB scheduler working set — against a predicted 6 × 1.32 = 7.92 GiB.

**The speed cost is the problem.** Dropping the LRU from 6,750 to 4,400 slots over 42 streaming
layers costs **half the decode throughput**: 34.6 tok/s against the baseline's 70.4 tok/s on the
same 8k-chat prompt in the same session, and the cold 7k TTFT goes 5.86 s → 15.57 s. This is
exactly spec section 11's open risk, now answered with a number. `/v1/cache/routing` shows why:
the 42 streaming layers run an 18.8 % median miss rate (~2.2 expert fetches per layer per decode
step, ~92 fetches per step overall).

So: the feature is correct, it saves the RAM it promised, and at `auto` on this box it is not
worth paying for. The owned set is not the cheap win; the LRU it has to give up is the cost.

## What was run

One server at a time throughout; `nvidia-smi` < 3 GB and a 60 s settle between every boot; only
`python.exe` processes carrying `freetoken.cli serve` (and their `python.exe` descendants) were
killed; `ft.exe daemon` never touched; `D:\Models` and both source trees untouched; nothing
committed or pushed.

| # | time | build | outcome |
|---|---|---|---|
| 0 | 15:17 | 2020 baseline, 1 h 03 m uptime | idle counters + 48-token identity answer taken, then stopped |
| — | 15:21 | (empty) | commit 51.76 GiB, available 74.71 GiB, phys in use 20.84 GiB, GPU 1,317 MiB |
| 1 | 15:19 | worktree | **aborted before the model loaded** — `zmq.error.ZMQError: Address in use (tcp://127.0.0.1:2033)`. Not a feature failure: a *run-1* orphan (pid 99124, started 14:08) still held the port. See *The port collision*. |
| 2 | 15:22 | worktree, `auto` / 4,400 / routing stats | **SERVING after 64.2 s**, measured |
| — | 15:33 | (empty) | commit 47.13 GiB, available 76.56 GiB, phys in use 18.99 GiB, GPU 759 MiB |
| 3 | 15:33 | worktree, CPU/GPU test | the CUDA-gated `test_copy_plan_holds_a_zero_placeholder_for_gpu_owned_layers` run **with a device visible**: PASSED |
| 4 | 15:34 | `D:\FreeToken` restore | serving after 71 s, benchmarked as a same-session baseline, **left running** |

### The port collision

Boot #1 died 28 s in because the detokenizer could not bind `tcp://127.0.0.1:2033`. The holder
was pid 99124, a `spawn_main` worker left over from **run 1's** last candidate boot (started
14:08:19). It survived run 1's cleanup because `stop-server.ps1` walks down from a live
`freetoken.cli serve` root, and that worker's root had already exited — an orphan with no
reachable parent. It did not disturb the run-1 restore (port 2020 uses different ZMQ ports), so
it went unnoticed for over an hour.

Boot #1 also left its own orphan: pid 94596, already at 70.95 GiB private and still loading
after the launcher had exited. Both were killed by explicit pid after verifying `Name ==
'python.exe'`, a `spawn_main` command line, and no `ft.exe` marker. **Operational note for the
next run: after stopping a server, check `netstat -ano | findstr :203` and for stray
`spawn_main` python processes, not just `nvidia-smi`.**

## Before / after table

Baseline column = the **restore boot in this same session** (15:34), which reproduces run 1's
fresh baseline to within 2.4 % on every speed number — that is the methodology check for this
table. Run 1's fresh-baseline figures are given in brackets where they differ.

| quantity | baseline 2020 (6,750 slots) | candidate 2030 (`auto`, 4,400 slots) | delta |
|---|---|---|---|
| commit | `cc085cf` | `2470573` | — |
| booted to `serving` | 71 s [run 1: 65 s] | **64.2 s** | −6.8 s |
| expert-bank load phase | 43 s (15:34:20 → 15:35:03) | **38 s** (15:22:21 → 15:22:59) | −5 s |
| boot peak whole-system commit | 215.45 GiB | 209.16 GiB | **−6.29 GiB** |
| boot minimum available | 6.72 GiB | 14.33 GiB | +7.61 GiB |
| free VRAM after initialization | 4.58 GiB [run 1: 4.64] | 3.27 GiB | −1.31 GiB |
| CUDA graph bs=1 | captured | captured (3.29 GiB avail) | same |
| MTP spec graphs at boot | 6/6 in 2.488 s | 6/6 in 2.485 s | same |
| MTP draft graphs at boot | 7/7 in 0.695 s | 7/7 in 0.649 s | same |
| whole-system commit, idle | 213.74 GiB | 209.37 GiB | **−4.37 GiB** |
| whole-system physical in use, idle | 87.83 GiB | 80.33 GiB | **−7.50 GiB** |
| server-attributable physical (minus empty ref) | 68.84 GiB | 59.49 GiB | −9.35 GiB |
| scheduler private bytes | 98.48 GiB [run 1: 98.35] | 100.00 GiB | **+1.52 GiB** |
| scheduler working set | 67.15 GiB [run 1: 67.05] | 59.13 GiB | **−8.02 GiB** |
| `nvidia-smi` idle | 29,113 MiB | 31,566 MiB | +2,453 MiB |
| `nvidia-smi` during decode | 31,313 MiB [run 1] | 30,436–32,038 MiB (typ. 31,523) | +~0.2 GiB |
| 8k-chat decode, sweep method, median of 2 | **70.4 tok/s** [run 1: 72.1] | **34.6 tok/s** | **−50.9 %** |
| 8k-greedy decode | 71.0 tok/s [run 1: 70.7] | 34.6 tok/s | −51.3 % |
| cold-7k TTFT | 5.86 s [run 1: 5.89] | 15.57 s | **+9.71 s** |
| warm-turn TTFT | 1.64 s [run 1: 1.59] | 1.87 s | +0.23 s |
| 8k-chat 512 tok, temp 0, thinking off | 73.29 cold / 67.31 warm tok/s | 42.24 cold / 28.42 warm tok/s | −42 % / −58 % |
| — its TTFT | 6.68 s cold / 1.57 s warm | 10.01 s cold / 1.51 s warm | +3.33 s / −0.06 s |
| 48-token temp-0 answer md5 | `29f0e74dfed744b538f856f38553e1ae` | `29f0e74dfed744b538f856f38553e1ae` (3/3) | **identical** |
| picture request, first of the boot | 6.62 s | **76.91 s** | +70.3 s (see below) |
| picture request, warm | 4.68 s | 5.19 s / 6.74 s | comparable |
| picture answer | correct (`738214`) | correct (`738214`) | same |

Server-attributable physical uses the nearest empty reference to each measurement
(18.99 GiB before the restore boot, 20.84 GiB before the candidate boot).

### Where the RAM saving shows up — and where it does not

The predicted saving is 6 owned layers × 1.32 GiB = **7.92 GiB** of host expert banks that are
never allocated.

- **Scheduler working set: −8.02 GiB.** Dead on.
- **Whole-system physical in use: −7.50 GiB.** Within the ±0.5 GiB criterion.
- **Whole-system commit: −4.37 GiB.** Only about half the predicted figure.
- **Scheduler private bytes: +1.52 GiB.** No saving at all — it went *up*.

That split is consistent and worth recording: the host expert banks are not private commit.
The server's private bytes (~103 GiB across all four processes) account for far less than its
~167 GiB of attributable commit; the ~64 GiB difference is the mapped/pinned expert-bank
territory (the loader reads 63.3 GiB of experts through the mmap path). Removing six layers'
host banks therefore removes *mapped* pages, which is why the physical and working-set numbers
move by the full 7.9 GiB while private bytes do not move at all. The +1.52 GiB on private bytes
is unexplained by the design and is small enough to be host-allocator variance, but it means
**the "scheduler private bytes −7.9 GiB" criterion as written cannot be met by this design** —
the criterion is measuring the wrong counter.

### The 76.9 s first picture

The candidate's first picture request took 76.91 s; its second and third took 5.19 s and 6.74 s.
The baseline, controlled for the same condition (first picture of a fresh boot, same image, same
prompt, `prompt_tokens` 2104 on both), took 6.62 s then 4.68 s. So this is a candidate-only
anomaly of roughly 11×, confined to the first request, and it is in the same direction as the
cold-7k TTFT regression (5.86 → 15.57 s) but far larger than it. Two plausible contributors,
neither isolated here: the 897 MB of `mmap`-backed vision weights paging in from disk after the
63.3 GiB expert re-read evicted the standby cache, and a 2,104-token vision prefill thrashing a
25 %-smaller expert LRU. Worth a targeted probe before the next design iteration; it is not
explained by the owned-layer path itself, which is strictly simpler at inference time.

## Plan check table

| check | criterion | verdict | evidence |
|---|---|---|---|
| boot log shows owned set, LRU size, MTP graphs 6/6 + 7/7 | yes | **PASS** | `MoE GPU-owned layers: [0, 1, 2, 6, 7, 22] (6 x 1.32 GiB resident, no host bank); LRU cache 4400 slots for 42 streaming layers`; `Free memory after initialization: 3.27 GiB`; `Start capturing CUDA graphs with sizes: [1]`; `MTP spec graphs captured at boot: 6/6 ... in 2.485 s`; `MTP draft graphs captured at boot: 7/7 in 0.649 s` |
| scheduler private bytes and whole-system commit | −7.9 GiB ± 0.3 | **FAIL** | private +1.52 GiB, commit −4.37 GiB. The saving is not in these counters — see *Where the RAM saving shows up*. The criterion is mis-specified rather than the feature failing. |
| whole-system physical in use | −7.9 GiB ± 0.5 | **PASS** | 87.83 → 80.33 GiB = **−7.50 GiB** |
| boot peak host RAM | ≤ baseline + 1.5 GiB | **PASS** | 209.16 GiB vs baseline 215.45 GiB = **−6.29 GiB**; minimum available 14.33 vs 6.72 GiB |
| 8k-chat decode tok/s | recorded; operator decides | **PASS (recorded) / regression** | 34.6 vs 70.4 tok/s, **−50.9 %** |
| TTFT on the same prompt | recorded | **PASS (recorded) / regression** | cold-7k 15.57 s vs 5.86 s; warm turn 1.87 s vs 1.64 s |
| answers at temperature 0, `max_tokens` 48 | md5 `29f0e74dfed744b538f856f38553e1ae` | **PASS** | 3/3 candidate samples match; the baseline reproduced the same md5 on its warm request this session |
| picture request (`-VisionWeights mmap`) | works, latency recorded | **PASS with a caveat** | correct answer (`738214`) on all three; 76.91 s first / 5.19 s / 6.74 s vs baseline control 6.62 s / 4.68 s |
| `/v1/cache/routing` owned rows | owned `resident: true`, `miss_rate: null`; streaming rows sane; summary excludes owned | **PASS** | all six owned rows `resident: true, miss_rate: null, steps: 0`; no non-owned row is resident; 42 streaming rows, all `steps: 2688`, miss rate 0.120–0.234 (median 0.188); `summary.slots_per_layer = 104.76 = 4400 / 42` |
| `/v1/cache/status` geometry | shows `gpu_owned_layers` and the LRU size | **PASS** | `gpu_owned_layers: [0,1,2,6,7,22]`, `moe_cache_size: 4400` (baseline has no `gpu_owned_layers` key and `moe_cache_size: 6750`); every other geometry field identical |
| owned-layer rows byte-identical to a host-bank load | one-off probe script | **NOT RUN — needs a second model process** | no such probe script exists in the tree, and comparing device rows to host-bank rows for the same real layer requires loading that layer both ways, i.e. a second model process. The 48-token md5 identity above is the indirect live evidence; `tests/moe/test_gpu_owned_banks.py` pins byte-identity on a synthetic checkpoint. |

## Items only a live run could decide (status doc §"Only a live GPU run can decide this")

| # | item | outcome |
|---|---|---|
| 1 | real pageable H2D of the owned layers against a CUDA device, and its boot cost | **WORKS, and it is free.** The expert-bank load phase was 38 s against the baseline's 43 s — the 7.9 GiB of synchronous pageable H2D copies cost *less* than building the same six host banks. The predicted "a few seconds of boot" overhead did not materialise. |
| 2 | resident VRAM rows byte-identical to host-bank rows | **not run** (needs a second model process). Indirect: the 48-token greedy answer is byte-identical to the baseline. |
| 3 | boot peak host RAM | **lower**, as predicted: 209.16 vs 215.45 GiB peak commit, 14.33 vs 6.72 GiB minimum available |
| 4 | decode and MTP graphs still capture (bs=1, widths 1–6) | **PASS**, no change: bs=1 captured, spec 6/6 in 2.485 s, draft 7/7 in 0.649 s, ladder replays 6/6 |
| 5 | `_build_fused_copy_plan` 0-placeholder assertion on a CUDA device | **PASS** — `tests/moe/test_offload.py::test_copy_plan_holds_a_zero_placeholder_for_gpu_owned_layers` ran with the device visible (in the empty window between servers) and passed; the whole `-k "gpu_owned or fused_copy_plan"` selection is 12 passed, 0 skipped |
| 6 | speed cost of 6,750 → ~4,400 slots | **−50.9 % decode, +9.71 s cold TTFT.** This is the finding of the run. |

## Routing detail (candidate)

```
gpu_owned_layers : [0, 1, 2, 6, 7, 22]
cache_size       : 4400          num_layers: 48   num_experts: 512
summary          : slots_per_layer 104.76 (= 4400 / 42, owned layers excluded)
                   working_set_mean 448.74, working_set_max 496, experts_for_90pct 196
                   oracle_hit_at_slots 0.7212, norm_entropy 0.8469
owned rows (6)   : resident true, steps 0, miss_rate null, fetched_per_step 0.0
streaming (42)   : steps 2688 on every row
                   active_per_step  ~11.4–11.9
                   missing_per_step ~1.96–2.76
                   miss_rate        0.120 min / 0.188 median / 0.234 max
```

Roughly 92 expert fetches per decode step across the 42 streaming layers, against a mean working
set of 449 experts per layer and only 104.76 slots. The oracle hit rate at that slot count is
0.721, so the LRU (0.812 measured hit rate) is not the problem — the slot count is. There is no
baseline routing histogram to compare against: the baseline boot script does not pass
`-CollectRoutingStats`, and step 7 required running it unchanged.

## Warnings, fallbacks and anything unexpected in the logs

- **No new warnings from the feature.** The candidate's log carries only the same three benign
  entries the baseline does: the `transformers` `qwen4_exp` architecture notice, the
  `torch.cuda._set_allocator_settings is deprecated` FutureWarning, and
  `expandable_segments not supported on this platform` (so the `Enabled expandable_segments`
  line at 15:22:16 is immediately contradicted — true on the baseline too).
- `[c10d] The client socket has failed to connect to [DESKTOP-2TOO4JN]:2031 (system error:
  10049)` appears once on both builds during startup and is harmless.
- `Page size is overridden to 64 for the qsa_sparse backend` — expected, on both.
- No expert-load fallback, no refusal, no owned-layer diagnostic beyond the single summary line.
  The one thing the log does **not** say is how long the owned-layer copies themselves took; the
  38 s figure is the whole expert phase. A per-phase timing line would have made item 1 above a
  measurement rather than an inference.
- `Free memory after initialization` fell from 4.58 to 3.27 GiB, i.e. `-MoECacheSize 4400` leaves
  1.31 GiB *less* VRAM headroom than the baseline, not the ~1 GiB more the plan aimed for.
  Arithmetic: 2,350 slots returned × 2,772,480 B = 6.07 GiB freed, 7.92 GiB of owned banks added,
  net +1.85 GiB of VRAM demand. It fit, and the CUDA and MTP graphs captured with 3.18 GiB still
  free, but a size of ~3,900 would have been the true like-for-like.
- `cache_budget_bytes` is identical (23,564,753,305) on both servers even though one of them
  spends 7.9 GiB of it on resident banks. The reported budget is the total, not the LRU's share,
  so the owned reservation is invisible in that field.

## Scratchpad

Scripts, logs and raw snapshots for this run:
`C:\Users\jay\AppData\Local\Temp\claude\D--FreeToken\14eaa0eb-ee91-4e4f-a9c5-2096e88229bd\scratchpad\gpu-owned-live\run2`
— `boot-2030.ps1`, `pic2.py`, `snaps.jsonl`, `server-run2-2030.{out,err}.log`,
`server-portclash-2030.{out,err}.log`, `bootmem-run2-2030.csv`, `bootmem-restore-2020.csv`,
`nvsmi-cand.csv`, `ab-cand2.txt`, `ab-restore2020.txt`, `chat8k-*.txt`, `answer-*.txt`,
`routing-cand2.json`, `status-cand2.json`, `pic-*.json`.
Run 1's scripts (`snap.ps1`, `stop-server.ps1`, `chat8k.py`) and the shared `ab_send.py` /
`vision/verify-shot.png` are in the parent scratchpad directories.
