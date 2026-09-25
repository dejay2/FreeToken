# Qwen3.8-Flash-Next EXL3 3.05bpw on the RTX 5090: first live boots and the freeze (2026-09-25)

Checkpoint: turboderp `Qwen3.8-Flash-Next-exl3`, branch `3.05bpw_h5_ng5` (80 GiB on disk; routed
experts K=3, attention / GDN / shared expert / `lm_head` / vision K=5, indexer K=3, MTP experts
K=4). Every linear stays EXL3 at runtime (exllamav3 1.4.6 `exllamav3_ext`); the n-gram PLE table
was converted once to FreeToken's fp8 layout on the SSD. Box: Windows 11, RTX 5090 32 GB, WSL
distro `vllm` (74 GiB), branch `feat/exl3-qwen-flash`. One chat at a time
(`--max-running-requests 1 --cuda-graph-max-bs 1`), `--moe-gpu-owned-layers auto:8`,
`--moe-cache-auto`, fp8 KV. Prompts: 8 fixed short questions, `temperature 0`,
`max_tokens 128`, thinking off, streamed.

## The freeze: PLE disk wait-sync deadlocks with EXL3 decode graphs

The first boot (15:31) served nothing: the first request never made progress, the scheduler
could not exit, its ~38 GB of pinned expert banks stayed held, WSL stopped answering and Windows
froze at the login screen (fixed only by a remote `Restart-Computer -Force`).

Bisection that evening, one boot at a time, with a stall guard (py-spy dump, then SIGKILL of the
server group, when a request is in flight for 120 s):

| boot | PLE backend / sync | other differences | result |
|---|---|---|---|
| b4 | mmap | overlap off, KV 32k, EXL3 trace + strict | 1/1 correct; 22,888 eager EXL3 launches, all on one stream |
| b5 | mmap | overlap on, KV 32k | 8/8 correct, 55-61 tok/s |
| b6 | disk, **wait-sync** (auto) | the 15:31 boot's exact shape minus vision/MTP | **hung on the first request**; guard killed it cleanly |
| b7 | disk, launch-gating (`FREETOKEN_PLE_SYNC=gate`) | same as b6 | 8/8 correct, 57-63 tok/s |
| b8 | disk, launch-gating chosen automatically (59a74f7) | + vision | 8/8 correct, 58-63 tok/s, picture OK |

b6's py-spy stack put the scheduler's main thread inside `torch.cuda.graphs.replay`
(`engine/graph.py:246`) with the GPU at desktop idle (~10 %), i.e. not a spinning kernel. Under
wait-sync the captured decode graph WAITs on a pinned flag that the host raises only after the
replay call returns (`forward_host_ctx`'s deferred fill). With ExLlamaV3 kernels in the graph the
replay call never returned, so the flag was never raised. NVFP4 serves with the same wait-sync
path every day, so the blocking is specific to the EXL3 kernels in the graph; which launch
blocks the host was not isolated. Fix: EXL3 checkpoints never use wait-sync (`auto` becomes
launch-gating, an explicit `FREETOKEN_PLE_SYNC=wait` is refused).

The cross-stream EXL3 serialisation added earlier the same day (5c051de) was not the cause (0
stream switches in the text path) and stays as a guard.

## Results (b5, b7, b8: one chat, no MTP)

| measure | value |
|---|---|
| boot to ready | 56-85 s |
| card used | 31.5 GiB (8 GPU-owned layers 7.10 GiB, LRU 6,959 slots 12.07 GiB) |
| WSL server RSS | ~41 GiB (40 streaming layers x 512 x 1.86 MB pinned banks); WSL available 34.6 GiB |
| decode, short prompts | 55-63 tok/s |
| TTFT, short prompts | 1.0-2.3 s |
| picture (256 px red disc on white) | "Flag" in 4.3 s; text request after it normal |
| answers | 8/8 correct and coherent on every boot |

For reference, NVFP4 Qwen3.8-Flash-Next on the same box measured a warm short-chat median of
51.9 tok/s (docs/research/dynamic-kv-pool-live-2026-09-12.md, 262k pool, 7,242 slots). That is
not a same-day A/B: the EXL3 boots used a smaller KV pool on b4/b5 and the NVFP4 figure is two
weeks older, so read it as "EXL3 is not slower on short chats", not as a ratio.

## MTP (b9)

Depth 2, spec graphs on, PLE mmap: boot 276 s (spec graph capture 141 s; widths 2 and 3 stayed
`retryable (MEMORY_ADMISSION)`), 8/8 correct, decode 33-40 tok/s against 58-63 without MTP. MTP
works on EXL3 but is a net loss at this memory fit; leave it off.

## PLE table conversion (13:48, once)

`exl3_ngram_trellis` K=5 rows -> fp8 E4M3, 128 shards x 2,500,012 rows, scale 2.0007e-4.
Precision gate on sampled rows: median relative RMS 0.0265, p99 0.0308, max 0.0360.

## Control panel

`model_detect` on the box: engine freetoken, `EXL3 (3-bit experts)`, 1,862,400 bytes per expert
(matches the loader), vision / PLE / MTP found, `ramNeedGB` 43 against ~41 GiB measured RSS.

## Not measured

- Same-day NVFP4 A/B and the long workloads (8k chat, cold 7k TTFT, warm-turn TTFT).
- Two chats at once, and the 262k context at depth.
- Starting and stopping through the settings page and llama-swap (after merge).
- Which EXL3 kernel makes the graph launch block the host under wait-sync.
