# MoE decode routing skew on Qwen3.8-Flash-Next-NVFP4 (2026-09-02)

The number that decides whether **strategy A** of
[`memory-reduction-options-qwen38-rtx5090.md`](memory-reduction-options-qwen38-rtx5090.md)
— a *static* per-layer hot set of experts, resident on the GPU, with **no host rows at all**
— is worth building.

**Verdict: no. Do not build strategy A for this model.** Routing is far too broad and far too
workload-dependent. A static hot set sized at the whole per-layer slot budget would still miss
most of the traffic, and one sized at the 30-67 % that strategy A's memory saving requires
misses two thirds of it.

## How the data was taken

One live boot on the operator box (Windows 11 Pro 26200, RTX 5090 32 GB, 95.56 GiB RAM,
`D:\Models\Qwen3.8-Flash-Next-NVFP4`), booted from this worktree with
`-ExpertLoad parallel -CollectRoutingStats`, i.e. the new `--moe-collect-decode-freq`. Every
other flag and env var identical to the accepted server: `-MoECacheSize 6750`,
`-ContextTokens 65536`, `-KVCacheTokens 65536`, `-MaxRunningRequests 1`, `-DenseQuant int8`,
`-EmbedHost`, vision on, MTP speculation on with spec graphs.

Geometry: **48 MoE layers x 512 experts**, `moe_cache_size = 6750` slots, i.e. **140.6 slots
per layer** — 27.5 % of a layer's experts.

Four workloads, `temperature 0`, `enable_thinking false`, **520 generated tokens each**:

| id | prompt |
|---|---|
| `code` | write a complete thread-safe LRU cache with TTL, tests included, code only |
| `prose` | a continuous-prose essay on the history of marine navigation |
| `chat8k` | ~8 k-token Python source (`engine/spec_graph.py`) + "explain the graph-replay guards" |
| `toolcall` | a 15-tool JSON-array travel-planning tool call, >= 20 calls |

`GET /v1/cache/routing?reset=true` was called before and after each workload, so each raw
`[48, 512]` histogram covers exactly that workload. A fifth pass ran all four back to back
without an intervening reset: `union`.

Raw data and the two scripts are in [`routing-skew-2026-09-02/`](routing-skew-2026-09-02/)
(`code|prose|chat8k|toolcall|union.json` = the server's reply verbatim, including
`decode_freq`; `analysis.json` = the derived numbers below; `collect.py`, `analyze.py`;
`boot-parallel-memory.csv` = the 3 s standby/RAM/GPU trace of this boot).

Two caveats on every number here. The histogram is counted in `ensure_experts`, which the
MTP draft/verify path also drives, so **rejected speculative tokens are counted too**; and
CUDA-graph capture contributes a few hundred warm-up counts before the first real token
(~0.1 % at these volumes). Neither changes any conclusion — both make routing look *more*
concentrated than it is, not less.

## Per-workload concentration

Per layer, over the 48 layers with traffic:

| | `code` | `prose` | `chat8k` | `toolcall` | `union` |
|---|---|---|---|---|---|
| routing events | 284,640 | 279,840 | 279,840 | 261,120 | 996,480 |
| **experts for 90 % of mass** (mean) | 139.3 | 128.4 | **151.6** | **163.7** | **242.2** |
| experts for 90 % (min / max layer) | 50 / 275 | 49 / 202 | 76 / 281 | 88 / 242 | — / 330 |
| **working set** (distinct experts, mean) | 307.2 | 274.6 | 309.5 | 327.4 | **461.3** |
| working set (max layer) | 428 | 384 | 435 | 389 | **497** |
| **normalized entropy** (mean) | 0.776 | 0.776 | 0.808 | 0.813 | **0.893** |
| normalized entropy (min / max layer) | 0.626 / 0.921 | 0.605 / 0.870 | 0.706 / 0.919 | 0.702 / 0.889 | — |
| oracle hit @ 140.6 slots (server) | 0.894 | 0.913 | 0.874 | 0.861 | 0.716 |

Read the first row against the slot budget. **140.6 slots per layer, and a single workload
needs 128-164 experts per layer just to cover 90 % of its own traffic.** The cache is
correctly sized for one workload at a time and nothing more. Over the union of all four it
takes 242 — 1.7x the budget.

The working-set row is the harsher one: in 520 tokens a layer touches 275-327 *distinct*
experts of 512, and over the union 461 of 512. **Ninety per cent of this model's experts get
used within four short conversations.** Normalized entropy 0.78-0.81 per workload (0.89 for
the union, where 1.0 is uniform) says the same thing in one number: routing here is mildly
skewed, not sparse.

The skew that does exist is depth-shaped and matches the U-curve the engine's `_auto_cpu_layers`
already assumes. Layers 0-2 are the broadest (`code`: 275/242/202 experts for 90 %), layers
15, 31 and 39 the narrowest (50, 62, 88). Full per-layer profiles are in `analysis.json`
(`per_layer_e90`, `per_layer_ws`, `per_layer_entropy`).

## Do the workloads agree on *which* experts are hot?

Pairwise Jaccard of each layer's top-K expert set, averaged over the 48 layers:

| pair | top-70 | top-140 |
|---|---|---|
| `code` / `prose` | 0.077 | 0.199 |
| `code` / `chat8k` | 0.275 | 0.411 |
| `code` / `toolcall` | 0.140 | 0.269 |
| `prose` / `chat8k` | 0.168 | 0.320 |
| `prose` / `toolcall` | 0.060 | 0.163 |
| `chat8k` / `toolcall` | 0.085 | 0.195 |
| **mean** | **0.134** | **0.259** |

For calibration, two *independent uniform* draws of 70 experts from 512 would score
70x70/512 / (140 - 70x70/512) = **0.074**, and of 140, **0.158**. So the observed agreement is
only **~1.7x chance** at top-70 and **~1.6x chance** at top-140. There is a shared backbone,
but it is thin: the two most similar workloads (`code` and `chat8k`, both Python) reach 0.28
/ 0.41, and the two least similar (`prose` and `toolcall`) 0.06 / 0.16 — barely
distinguishable from picking at random.

Per-layer min/max are in `analysis.json`; even the *best* layer of the best pair only reaches
0.44 at top-70.

## What a static top-K would actually capture

Fit a hot set of K experts per layer to a reference distribution, then measure the fraction
of each workload's routing events that land inside it. K = 42 / 70 / 94 is 30 % / 50 % / 67 %
of the 140.6-slot budget — the range in which strategy A's memory saving lives.

**Hot set fit to the union of all four workloads** (the realistic construction):

| K | `code` | `prose` | `chat8k` | `toolcall` | worst layer |
|---|---|---|---|---|---|
| 42 | 0.451 | 0.326 | 0.344 | 0.271 | 0.148 |
| 70 | 0.578 | 0.458 | 0.477 | 0.397 | 0.261 |
| 94 | 0.660 | 0.556 | 0.574 | 0.479 | 0.323 |

**Hot set fit to the very workload being measured** — an oracle no static design can reach,
included only as an upper bound:

| K | `code` | `prose` | `chat8k` | `toolcall` | worst layer |
|---|---|---|---|---|---|
| 42 | 0.605 | 0.578 | 0.516 | 0.527 | 0.282 |
| 70 | 0.732 | 0.726 | 0.666 | 0.667 | 0.398 |
| 94 | 0.804 | 0.811 | 0.757 | 0.751 | 0.484 |

**Hot set fit to `code` alone**, to show what a domain-mismatched hot set costs:

| K | `code` | `prose` | `chat8k` | `toolcall` |
|---|---|---|---|---|
| 42 | 0.605 | **0.065** | 0.281 | 0.135 |
| 70 | 0.732 | **0.129** | 0.405 | 0.221 |
| 94 | 0.804 | **0.185** | 0.488 | 0.286 |

Now the comparison that matters. The **dynamic LRU already running on those same 140.6
slots** was measured in the same windows (`per_layer` in each JSON):

| | `code` | `prose` | `chat8k` | `toolcall` |
|---|---|---|---|---|
| decode steps | 219 | 441 | 372 | 102 |
| per-layer miss rate (min-max) | 0.093-0.180 | 0.055-0.138 | 0.097-0.250 | 0.193-0.360 |
| **mean hit rate** | **0.862** | **0.899** | **0.839** | **0.715** |

So the live cache delivers a **72-90 % hit rate** on the full slot budget, while a static
hot set at K=94 — giving up a third of that budget, which is the *point* of strategy A —
captures 48-66 %, and at K=42 only 27-45 %. Even the oracle static set, fit to the workload
it is then scored on, tops out at 75-81 %: below what the dynamic cache already achieves
without knowing the workload in advance.

And under strategy A those are not slow misses but **unservable** ones: the design removes
the host rows the miss path would stream from.

## Verdict

Strategy A is not worth building for Qwen3.8-Flash-Next-NVFP4.

1. **The working set is not sparse.** 275-327 distinct experts per layer per 520-token
   workload, 461 of 512 over four of them. There is no small hot set to find.
2. **The 90 % mass does not fit.** 128-164 experts per layer for one workload against a
   140.6-slot budget; 242 across workloads. A static set sized *below* the budget is
   structurally too small.
3. **The workloads disagree about which experts are hot.** Mean top-70 Jaccard 0.134, only
   1.7x chance. A hot set tuned on code serves prose at a 6-19 % hit rate.
4. **It is strictly worse than what already runs.** Static K=94: 48-66 %. Dynamic LRU on the
   full budget: 72-90 %. Even an oracle static set fit to the workload it is scored on
   (75-81 %) loses to the dynamic cache. Giving up adaptivity buys nothing here.

What the data *does* support, in rough order of expected value:

- **Keep the dynamic cache; the remaining headroom is in policy, not residency.** Realized
  hit is 0.86 / 0.90 / 0.84 / 0.72 against a stationary-oracle ceiling of 0.89 / 0.91 / 0.87
  / 0.86 at the same slot count. Three of four workloads are already within 3 points of the
  ceiling; only `toolcall` leaves ~14 points on the table, and that is a policy question
  (LFU/2Q, admission control), not a reason to change where experts live.
- **Layer-selective, not expert-selective.** The concentration is strongly depth-shaped
  (experts for 90 %: 275 at layer 0 down to 50 at layer 15). Layers 15, 31 and 39 are ~3x
  narrower than layers 0-2, so a *per-layer* slot budget that follows the measured curve —
  rather than the uniform 140.6 everywhere — is cheap and is supported by this data.
  Note this is the same U-shape `_auto_cpu_layers` already exploits.
- **A hybrid pin, if anything.** If a permanent pin is built at all, pin only the narrow
  layers and leave the broad ones fully dynamic; do not remove host rows anywhere.

## Reproducing

```powershell
# boot with the counter armed
.\scripts\start-qwen38-flash-next-mmap-windows.ps1 -ModelPath D:\Models\... -CollectRoutingStats
# collect (writes one JSON per workload)
python docs\research\routing-skew-2026-09-02\collect.py 2020 out\
python docs\research\routing-skew-2026-09-02\analyze.py out\
```

`--moe-collect-decode-freq` is boot-time only and needs no `-CudaGraphMaxBS 0`: the
accumulation is a `scatter_add_` over device tensors, so a captured decode graph replays it,
provided the flag is armed before capture — which is exactly why it is a boot flag. This run
had CUDA graphs and MTP spec graphs fully armed, and the counters tracked correctly
(996,480 events over 2,080 generated tokens).
