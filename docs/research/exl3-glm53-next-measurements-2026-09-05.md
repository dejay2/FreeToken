# GLM-5.3-Flash EXL3 2.05bpw: grouped prompt path and learned layer ranking (2026-09-05)

Follow-on to the packed `exl3_mgemm` path (commits 6e9a149 and 67fc492: decode 136.26 -> 35.43
ms/token, cold prefill 112.72 -> 238.83 prompt tok/s). Same box (Windows 11, RTX 5090 32 GB, 95.6 GB RAM), same server flags
(`--moe-cache-size 2221 --moe-gpu-owned-layers auto:5 --cuda-graph-max-bs 1 --num-tokens 131072
--kv-reserve-tokens 131072 --exl3-expert-op mgemm`, overlap on), same three-prompt bench
(non-streaming chat, `reasoning_effort=low`, `max_tokens=800`, `temperature=0`, each prompt twice,
second run reported; the first long-ctx run is the cold prefill). Every row is a fresh boot.

## Prompt path: one sort instead of a mask per expert (0af8521)

The packed prompt path scanned all 288 experts with a boolean mask and `torch.where`, one host
sync each (R4c N13: ~24,200 per 42-layer chunk). It now sorts the flattened routes once and
walks the sorted segments: four syncs per layer (R1 measured with `set_sync_debug_mode`).

| boot | cold long-ctx prompt tok/s | warm repeat | short | long-ctx | gen-300w tok/s |
|---|---:|---:|---:|---:|---:|
| packed path, before (R4e, fresh) | 238.83 | 589.34 | 16.18 | 9.62 | 24.47 |
| grouped prompt path (`after-item1`) | **319.09** | 690.77 | 17.37 | 11.28 | 25.21 |

Cold prompt reading +34 %. The "warm repeat" long-ctx row is dominated by its 39 decode tokens
(prefix cache hit), so treat it as a long-context decode number, not a prefill one.

## Learned layer ranking (5fd2e1e, ee97c78)

The decode routing histogram is saved beside the checkpoint every 60 s and at shutdown; the next
boot ranks MoE layers by "experts needed to cover 90 % of the layer's routes" and `auto:5` takes
the first five. Two boots, three prompts each:

- Boot 1 (no file): `auto:5 -> [0, 1, 2, 6, 7] from the fixed measured order`; file written 65 s
  after serving with 5,192 routes per layer.
- Boot 2: `auto:5 -> [0, 1, 2, 3, 7] from the learned order (15096 routes per layer over 1 boots)`.
  GLM's own order differs from the Qwen-measured list in one of five layers (3 for 6). After
  ~1,600 decode steps the deeper layers still show 35-79 never-routed experts each out of 288;
  layers 0-2 have none.

Cost of collecting (owned layers named per row; the lead's histogram-only row ran on the
learned set, so the like-for-like pair is the reviewer's two boots at 391ace6):

| boot | owned layers | short | long-ctx | gen-300w tok/s | cold long-ctx prompt tok/s |
|---|---|---:|---:|---:|---:|
| learning off (`--disable-moe-learn-routing`) | `[0, 1, 2, 6, 7]` | 17.79 | 11.22 | 25.02 | 308.57 |
| learning on, 5fd2e1e (histogram + `collect_stats`) | `[0, 1, 2, 6, 7]` | 17.49 | **10.57** | 25.08 | 281.63 |
| learning on, ee97c78 (histogram only) | `[0, 1, 2, 3, 7]` learned | 17.60 | 11.19 | 25.43 | 321.77 |
| R3 reviewer, 391ace6, learning off | `[0, 1, 2, 6, 7]` | 17.79 | 11.22 | 25.02 | -- |
| R3 reviewer, 391ace6, learning on (default) | `[0, 1, 2, 3, 7]` learned | 17.72 | 11.22 | 25.26 | 316.02 |

5fd2e1e armed the LRU miss counters, prefetch scoring and copy-row counters with the histogram
(the way `--moe-collect-decode-freq` does): -6 % long-context decode. ee97c78 arms the histogram
alone and the cost is gone: the reviewer's fresh boots put learning on and off at the same
11.22 tok/s. The learned layer choice itself made no measurable difference on these three
prompts (the two sets share four of five layers; layer 3 out-ranks layer 6 on breadth 211 vs
188 of 288). Cold prefill varies 282-322 across boots on the same code, so single-boot cold
numbers carry about +-10 %.

## Ranking metric versus the measured Qwen order

On the pooled histogram of the four `routing-skew-2026-09-02` captures, "experts for 90 % of
routes" agrees with the miss-rate-derived `GPU_OWNED_LAYER_RANK` on five of the top six (layer 5,
ninth measured, replaces layer 22, sixth), seven of the top eight, Spearman 0.98 over 48 layers.
Averaging the score per workload instead reproduces the top six exactly. Entropy, working-set
size and "oracle miss at the slot budget" scored no better; this one needs no slot budget and
works for card-owned layers, whose realized miss rate is null.
