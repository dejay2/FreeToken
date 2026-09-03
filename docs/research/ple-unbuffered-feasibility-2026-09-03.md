# PLE unbuffered-read feasibility — 2026-09-03

## Scope

This is a live measurement of the Qwen3.8-Flash-Next-NVFP4 checkpoint on the RTX 5090 / Windows 11 host. The server used `--ple-backend mmap`, `--max-running-requests 1`, `--kv-reserve-tokens 65536`, `--moe-cache-size 5781`, and parallel expert loading. The PLE switch was not added: this report is the required measure-first decision.

## Verdict

**Do not add a PLE unbuffered reader. It is not worth the likely decode cost.**

The table's virtual mapping is large, but its actively resident working set is negligible. The current mmap path leaves those pages reclaimable in the Windows file cache. An unbuffered path would turn nearly every 160-byte random row into a separate 4 KiB read and would give up useful warm file-cache hits. It would not recover a meaningful amount of active RAM.

## A. Resident memory

### VMMap

VMMap was run against the scheduler process (PID 35948) after boot and several real text requests. The official VMMap capture showed:

- Ten `model-plefp8-*.safetensors` mappings.
- The mappings reserve/commit about 51.8 GiB of file address space in total (the data payload is 47.68 GiB; the mapped files include headers and other file extent details).
- Each PLE per-file row had no `Total WS`, `Private WS`, or `Shareable WS` value: VMMap therefore reports **0 active working-set pages for each PLE mapping** at the capture point.
- The VMMap `Mapped File` aggregate had 25,832 KB of total working set, but the PLE rows did not contribute to that active value. The full-size number shown on the PLE rows is the copy-on-write mapping's committed/private reservation, not resident RAM.

The scheduler itself had about 63.7 GB of working set, overwhelmingly the intentionally pinned expert banks. The PLE mapping did not account for that resident amount.

### RAMMap and system counters

RAMMap was opened for the requested Active/Standby/Modified and file views. Its window runs elevated, so this non-elevated measurement session could not export the per-file table rows. The Windows counters captured at steady serving were:

| Counter | Bytes / value |
|---|---:|
| Available memory | 13,013 MB |
| System cache resident | 681.0 MiB |
| Modified page list | 242.2 MiB |
| Standby cache reserve | 919 MiB |
| Standby cache, normal priority | 11.25 GiB |
| Standby cache, core | 137.8 MiB |

These counters match the VMMap result: the machine has a large reclaimable standby pool, while the PLE mappings themselves are not an active working-set burden. The per-file Active result is the VMMap result above; a per-file Standby split was not available from RAMMap without crossing its elevation boundary.

## B. Actual PLE access pattern

The model configuration has `ngram_size=3`, `heads_per_ngram=8`, and `ple_embed_dim=2560`, so each token produces:

- 16 hash rows (`(3 - 1) * 8`)
- 160 bytes per row (`2560 / 16`)
- 2,560 bytes of logical row data per token

The table contains 320,001,536 rows. `MmapPleStorage.gather` receives the hash results, checks the 1,048,576-row host cache, and faults the misses. The source's measured design note says the IDs are near-uniform over the 320 million rows; adjacent rows are vanishingly uncommon, so it deliberately does not coalesce them. The 160-byte rows also do not line up with 4 KiB pages in a useful sequential pattern.

A live request generated 1,024 completion tokens from a 61-token prompt in 18.93 seconds. That is about 54.1 completion tokens/second, or about 866 decode rows/second at 16 rows/token. Including the prompt, the full forward work was about 17,360 rows, or about 917 rows/second over the measured wall time.

The 1,048,576-row cache is only about 0.33% of the 320 million-row table. With near-uniform IDs it cannot turn this into a locality-friendly workload; in a short cold run, almost every row is a miss. There is no useful next-row sequence for read-ahead: the next hash result depends on the next token and the per-head hash constants, not on the prior row's location.

## C. 4 KiB unbuffered-read cost

A standalone read benchmark used 8,192 deterministic random table rows from the real checkpoint. The same rows touched 8,523 distinct 4 KiB pages, or **0.96 rows per page touched**. This is the expected scattered pattern: although a page can hold about 25 rows, uniformly random row IDs almost never land on the same page in a small decode batch.

| Read method | Time for 8,192 rows | Rows/second |
|---|---:|---:|
| mmap, first random pass (cold-ish) | 0.880 s | 9,306 |
| mmap, second pass (warm cache) | 0.00277 s | 2,956,227 |
| Windows unbuffered, one 4 KiB QD1 read per 160-byte row | 0.544 s | 15,046 |

The direct path transferred an estimated 32 MiB from the SSD for only 1.31 MiB of logical row bytes. Per row that was about 66.5 microseconds in this run. The already-measured 990 PRO QD1 figure used for this box is about 131 microseconds, so 66.5 microseconds is a favorable lower result, not a safe upper bound.

The current mmap implementation also batches the miss addresses through `PrefetchVirtualMemory` on Windows before doing a serial copy. Existing same-box measurements in `models/qwen4_exp/weight.py` record about 0.44 ms for a 16-row cold batch (about 27.5 microseconds per row). Even using the favorable new 66.5 microseconds per direct row, one-row-at-a-time unbuffered reads would be about 1.06 ms for those 16 rows — roughly 2.4x the measured batched mmap miss cost. Against a warm cache hit, the direct path is roughly two orders of magnitude slower.

For the 1,024-token request, a simple worst-case decode-only estimate is 16 x 1,024 x 66.5 microseconds = 1.09 seconds of serialized row-read time; at the 131-microsecond box figure it is 2.15 seconds. The estimate is conservative about overlap and ignores the cost of opening/closing extra handles. It still represents an avoidable several-percent-to-low-teens addition to an 18.93-second request, while providing no active-RAM saving.

## Decision

The mmap table already keeps the 47.68 GiB table out of active working memory; Windows can reclaim its file pages from the standby list. The access pattern is scattered, but the existing batched prefetch path is materially better than a separate 4 KiB QD1 read for every row, and warm file-cache hits are dramatically cheaper. Read-ahead for future rows would not help because the hash IDs are effectively random.

Therefore Step 2 is intentionally not performed:

- no `FREETOKEN_PLE_UNBUFFERED` switch,
- no second Windows `ReadFile` layer,
- no PLE unbuffered unit tests,
- no read-ahead implementation.

The useful memory knob remains `FREETOKEN_PLE_ROW_CACHE`; reducing it is the targeted way to lower the small anonymous row-cache cost when needed.
