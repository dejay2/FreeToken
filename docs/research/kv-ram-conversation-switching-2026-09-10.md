# RAM conversation checkpoints and branch consolidation

## Scope and branch map

The delivery branch is `mtp-upstream-merge` on **dejay2/FreeToken**. Neither upstream nor
main is a push target. Its previous tip `046d328` was 66 commits behind `28c9249`.
The history already joins all of the relevant work:

```
046d328 mtp-upstream-merge
  -> adf67b0 memory-governor
     -> 91589fb fix/governor-nochange-steps --+
     -> 6c42613 perf/kv-park-incremental ----+-> 82249a9 live/governor-kvpark
                                               -> 28c9249 perf/kv-park-branch
```

The consolidated feature branch fast-forwards this history; it does not cherry-pick or
reimplement the governor, MTP, vision, or SSD fixes. Old branch names remain available as
historical references; deleting them is unnecessary to establish one working branch.

## Research and choice

This hybrid model has full-attention QSA pages, QSA index data, Gated DeltaNet recurrent
and convolution state, and PLE sibling state. A resumable checkpoint needs all of them at
the **same exact token boundary**. A later recurrent state cannot be truncated to an earlier
prefix just because its attention pages cover that prefix.

This follows the constraints described by [Marconi](https://arxiv.org/abs/2411.19379),
[SGLang's hybrid cache design](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/),
and [LMCache's hybrid-model integration](https://docs.lmcache.ai/mp/hybrid_models.html).
LMCache transfers linear state as an opaque page and separates cache groups; its documentation
also cautions that cached versus fresh GDN computation is not necessarily bit-exact.
We keep the existing native pool views and verify byte-exact transfers, then compare real
model answers against cold reference runs. Approximate state reconstruction is unnecessary.

Two problems explain why merely parking finished answers was insufficient:

1. A side request or subsequent turn can diverge before the saved answer's endpoint. There
   is no recurrent state at that divergence unless an earlier prompt checkpoint was saved.
2. The prompt checkpoint fix in `964874a` applied only to SSD. RAM still allowed a useful
   internal state to be evicted before saving it.

RAM now eagerly saves the canonical final-prefill checkpoint before state-slot pressure can
reclaim it. Finish saves and shorter historical checkpoints each retain their own exact state.
Like SSD, RAM uses immutable parent-linked segments: a continuation retains only its new
pages and state, and a shorter checkpoint can borrow all its pages from a longer donor.
Borrowed bytes are compared against the live source in a bounded staging buffer. Changed
bytes force an independent root; token equality alone does not justify sharing.

Family eviction and queued-save reservations account for shared ownership. Invalidated RAM
buffers stay charged while a queued writer pins them and are released before a replacement
can spend their budget. `copy_done` releases GPU sources; publication may finish afterward.
Tests that inspect published entries therefore wait for publication as well.

## CUDA change recovered

The serving checkout had an uncommitted change in `kernel/fla/chunk_delta_h.py`, selecting
`SGLANG_GDN_CHUNK_H_NUM_WARPS=2` instead of four. The consolidated code preserves two warps
by default on compute capability 12.0 (the tested RTX 5090); other devices retain four, and
the environment override remains authoritative. This is a recovered launch workaround,
not proof that the original illegal-access root cause has been identified.

The separate committed repair `19f061b` re-arms MTP verifier graphs after cache rebuilds;
it is also retained by ancestry.

## Validation record

Hardware: RTX 5090 32 GiB, Windows/WSL Linux; Qwen3.8-Flash-Next-NVFP4, FP8 KV,
262,208 cache tokens, page size 64, 12 state slots, int8 dense projections, MTP off.

- Initial selected baseline: 106 CPU cache/store tests passed.
- New regression: RAM final-prefill checkpoint was absent before the repair.
- New RAM segment and budget tests failed before their corresponding changes.
- Selected final cache/store/scheduler run: 147 passed, two CUDA tests skipped on the devbox.
- Both CUDA RAM ownership tests passed on the RTX 5090 (BF16 and FP8).
- All six existing GDN reference/chunk/decode tests passed on the RTX 5090 with two warps.
- Broad CPU suite baseline: 112 failed, 964 passed, eight skipped. The changed tree initially
  added nine missing machine-local boot-file fixture failures; the worktree now provides the
  same ignored boot fixture. No other additional failed test names were observed. Broad
  suite green status is not claimed; many existing tests require CUDA/pinned allocation.

The reproducible live test is `scripts/bench/kv_ram_switch_live.py`. Its `record` phase uses
parking off and alternates A/B across three turns each, with independent facts near the
beginning, middle and end of each 200k-token archive. Its `verify` phase requires RAM hits,
matching answers, no instance restart, at least 199,936 restored tokens on every revisit,
and at most 512 prompt tokens left to process. It also revisits shorter historical turns.

The six cold reference requests all passed:

| Conversation / turn | Prompt tokens | Seconds | Restored tokens |
|---|---:|---:|---:|
| A / 1 | 200,060 | 71.218 | 0 |
| B / 1 | 200,060 | 58.935 | 0 |
| A / 2 | 200,114 | 58.994 | 0 |
| B / 2 | 200,112 | 58.950 | 0 |
| A / 3 | 200,168 | 59.016 | 0 |
| B / 3 | 200,164 | 59.012 | 0 |

## RAM live acceptance — passed

Ten requests completed with two initial cold fills and eight subsequent full RAM restores.
Every answer exactly matched its cold reference, all eight revisits transferred the entire
saved attention prefix and 115,642,376 state bytes, and the server instance stayed unchanged.
`page_offset` was zero on every restore; the byte counts therefore exclude a false win from
retained GPU pages. Successful restore sequence numbers increased from one through eight.

| Request | Restored tokens | Tail processed | Restore seconds | Full response seconds |
|---|---:|---:|---:|---:|

| A / 2 (next turn) | 200,000 | 114 | 1.747 | 9.783 |
| B / 2 (next turn) | 200,000 | 112 | 1.711 | 7.992 |
| A / 3 (next turn) | 200,064 | 104 | 1.683 | 6.202 |
| B / 3 (next turn) | 200,064 | 100 | 1.473 | 6.121 |
| A / 1 (revisit) | 200,000 | 60 | 1.517 | 4.465 |
| B / 1 (revisit) | 200,000 | 60 | 1.858 | 3.561 |
| A / 2 (revisit) | 200,064 | 50 | 1.695 | 3.647 |
| B / 2 (revisit) | 200,064 | 48 | 1.480 | 4.085 |

The two initial RAM fills took 132.662 and 66.236 seconds. Automatic expert-cache sizing
selected 5,972 slots in this boot versus 6,262 in the cold reference boot, so these response
latencies are observations, not an isolated causal speedup measurement. The restored-token
and transfer-byte evidence directly establishes avoided prefill independently of that difference.

Six checkpoints occupied **6,001,247,280 bytes (5.59 GiB)** within the 8 GiB budget. After the
two roots, each new prompt checkpoint added about 117.3 MB (new KV pages, a full 115.6 MB state,
and token IDs), rather than another 2.77 GB copy. Historical revisits added no stored bytes.
The two initial misses stayed at two; every subsequent request hit and completed a restore.

The GPU cache/scheduler run produced 827 passes and two failures from old geometry expectations
missing the governor's GPU-owned-layer fields. Those two expectations were corrected; their
entire 16-test file then passed. The final targeted CPU run passed 167 tests, with the two
CUDA ownership tests skipped there and separately passing on the RTX 5090. The CUDA test also
queues unfinished work on a producer stream distinct from the store's constructing stream.

### Reproduce

Use the same model, 262,208-token GPU cache, FP8 KV, page size 64, and cache reporting enabled.
With parking off, run:

```bash
python scripts/bench/kv_ram_switch_live.py record --model-path /path/to/model --record /tmp/kv-cold.json
```

Restart with `--kv-park ram --kv-park-ram-gib 8 --kv-park-idle-ms 0 --enable-cache-report`,
then run:

```bash
python scripts/bench/kv_ram_switch_live.py verify --model-path /path/to/model --record /tmp/kv-cold.json
```

The script fails on missing RAM transfers, partial GPU retention, excessive tail prefill,
wrong answers, cache errors, or an instance restart. Raw numeric observations are in
`benchmarks/kv-ram-two-conversations-2026-09-10.json`.

### Limits

This validates text conversations on the stated Qwen/RTX configuration with MTP off; it is not
new live acceptance of SSD persistence, multimodal caching, tensor parallelism, or MTP.
The existing SSD tests remain covered. RAM is volatile and bounded: restart, budget eviction,
a changed model/layout, or a prompt that diverges before every saved checkpoint can require
cold prefill. Each retained checkpoint pays for a complete recurrent state, so a long-lived
session with many checkpoints can eventually reach the budget. Page alignment leaves a short
tail to process; the test proves that the 200k-token saved prefix itself was not reprocessed.

Delivery verification and final branch SHA are recorded after the publication gate.
