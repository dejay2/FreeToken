# Live measurements: full-context, four-request serving, KV parking and FP8 KV

Date: 2026-09-03. Box: Windows 11 Pro, RTX 5090 (32,607 MiB), 95.56 GiB host RAM.
Commit: `4d534ba` on `mtp-upstream-merge`. Model: Qwen3.8-Flash-Next-NVFP4, port 2020,
booted through the machine-local helper that calls
`scripts/start-qwen38-flash-next-mmap-windows.ps1`.

Every number below is measured on this box. The estimates in the sizing note
(`prompts/` job folder, not tracked) are quoted only where they are compared against.

## Configurations

| boot | kv_dtype | kv_park | context | kv cache | max running | moe total slots | gpu owned layers | streaming LRU | mtp | ready state | boot s |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| A (default) | bf16 | off | 262,144 | 262,144 | 4 | 4,188 | 6 (`0,1,2,6,7,22`) | 1,116 | off | serving | 83.1 / 73.3 |
| B (park ssd) | bf16 | ssd | 262,144 | 262,144 | 4 | 4,188 | 6 | 1,116 | off | serving | 73.4 / 74.4 |
| B2 (park ram) | bf16 | ram | 262,144 | 262,144 | 4 | 4,188 | 6 | 1,116 | off | serving | 73.4 |
| C (fp8) | fp8 | off | 262,144 | 262,144 | 4 | 5,332 | 6 | 2,260 | off | serving | 73.4 |

`/v1/cache/status` geometry, profile A: `num_pages` 4,096, `page_size` 64,
`unit_bytes.kv_per_token` 25,344, `moe_per_expert` 2,772,480, `mamba_per_slot` 115,642,376,
`num_mamba_slots` 24, `gpu_owned_reserved_bytes` 8,517,058,560,
`cache_budget_bytes` 23,564,753,305. Profile C differs only in
`kv_per_token` 13,248 and `moe_cache_size` 2,260.

`/v1/models` reports `max_model_len` 262144 and `context_length` 262144.

The sizing note predicted 4,097 physical pages and 25 mamba slots; the live pool allocates
4,096 pages (262,144 tokens exactly) and 24 mamba slots. It predicted 1,116 streaming-LRU
slots for BF16 and 2,260 for FP8 at 5,332 total — both are exactly what booted.

## Memory

| boot | VRAM idle | VRAM peak (4 requests) | host RAM idle | host RAM peak |
|---|---:|---:|---:|---:|
| A | 28,388 MiB | 30,412 MiB | 91.36 GiB | 86.45 GiB |
| C | — | 30,387 MiB | 86.63 GiB | — |

Host RAM is the tight resource on this box, not VRAM: the expert host banks plus the mapped
PLE table leave under 10 GiB free at idle. Anything that pins several more GiB of host RAM
(a large `--kv-park-ram-gib`, for instance) has very little room here.

## Long chat, one request, >= 250,000-token prompt

| run | boot | prompt_tokens | completion_tokens | cached_tokens | HTTP | answer | answer_sha256 | identity | wall s | completion tok/s | total tok/s | replay source |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|
| first | A | 250,018 | 47 | 0 | 200 | `maple-7319\|cobalt-4826` | `2c713885…ff48` | recorded | 69.87 | 0.67 | 3,579 | cold |
| displacement | A | 250,022 | 1 | 0 | 200 | `amber` | `b1601f69…91c7` | — | 66.28 | 0.02 | 3,772 | cold |
| replay | A | 250,018 | 47 | 0 | 200 | `maple-7319\|cobalt-4826` | `2c713885…ff48` | matches first | 67.86 | 0.69 | 3,685 | cold |
| first | C (fp8) | 250,018 | 47 | 0 | 200 | `maple-7319\|cobalt-4826` | `f2113d26…a917` | mismatch vs A | ~66 | 0.71 | ~3,750 | cold |
| replay | C (fp8) | 250,018 | 47 | 0 | 200 | `maple-7319\|cobalt-4826` | `f2113d26…a917` | matches its own first | ~66 | 0.71 | ~3,750 | cold |

The prompt carries a code at the very beginning and a different code at the very end. Every
run returned both, in order, so a 250,018-token prompt is genuinely read end to end.

**Time to first token**, cold 250,006-token prompt the server had never seen, streamed:
**66.65 s**, total 67.41 s. Prefill therefore runs at about **3,751 prompt tok/s** and
dominates the request; decode of 47 tokens costs under a second. The same run answered a
fresh code pair (`quartz-5150|indigo-7742`) correctly.

The server reports **47** completion tokens for a 48-token answer: the emitted `<|im_end|>`
is not counted in `usage`. The two P0 drivers asserted exactly 48 and were corrected to
accept 47-48 while still requiring the two runs to agree.

## Four requests at once, 8,192-token prompts, 48 tokens each

| request | prompt | completion | total | HTTP | answer | elapsed s | completion tok/s |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 8,205 | 47 | 8,252 | 200 | `request-0-marker-7000` | 4.047 | 11.61 |
| 1 | 8,205 | 47 | 8,252 | 200 | `request-1-marker-7001` | 4.047 | 11.61 |
| 2 | 8,205 | 47 | 8,252 | 200 | `request-2-marker-7002` | 4.047 | 11.61 |
| 3 | 8,205 | 47 | 8,252 | 200 | `request-3-marker-7003` | 4.047 | 11.61 |

Aggregate, profile A fully warm: wall **4.047 s**, 188 completion tokens,
**46.45 completion tok/s**, 33,008 total tokens, **8,155 total tok/s**, peak active
requests 4. Across 11 runs the aggregate ranged 25.4-49.5 completion tok/s purely by cache
warmth. R3's warm re-runs land at 31.4-43.5 tok/s, inside that spread; a **fully cold**
run, in which all four prompts are new to the server, aggregates **15.50 completion tok/s**.
The honest range over cold and warm together is therefore **15.5-49.5 completion tok/s**:
15.5 cold, 25.4-49.5 warm, with the highest figures needing a fully warm prefix cache.

Profile C (FP8) fully warm: wall **3.93 s**, **47.81 completion tok/s**, **8,395 total
tok/s** — **+2.9%** over BF16. The sizing note's interpolated sweep predicted 60.2 tok/s
(BF16) and 67.0 tok/s (FP8), a +11% gap; the measured gap is much smaller, because at this
prompt size the run is not expert-streaming bound.

## KV prefix parking

Two-turn chat, 65,540-token first turn, second turn asks for the first and last list item.
The baseline is the same two turns with parking off.

| mode | parked entries | parked bytes | hits | last restore ms | second-turn answer identical to parking-off baseline | verdict |
|---|---:|---:|---:|---:|---|---|
| off | 0 | 0 | 0 | 0.0 | (baseline) `maple syrup\|cobalt paint` | — |
| ram | 1 | 1,778,471,176 | 1 | **683.47** | yes, sha `67369f51…eb21` | **works** |
| ssd | 1 | 1,778,479,112 | 1 | **1,882.45** | yes, sha `67369f51…eb21` | **works** (after `2ce6403`) |

With parking off every counter in the `/v1/cache/status` `parking` block stayed at zero
through both turns, which is the live confirmation that the off path builds no store.

**RAM mode works end to end.** It parked 1,778,471,176 bytes for a 65,600-token prefix (the
sizing note predicted 1,776,586,760 bytes at 65,536 tokens) and restored it in **683 ms**,
and the restored turn produced a byte-identical answer. The bench prototype measured 31 ms
for the RAM copy at 65,536 tokens; the live restore is 22x that, because the live path also
does the lookup, page and state allocation, radix insert and tensor-parallel consensus that
the prototype did not. It is still well inside the 1,883 ms gate.

**SSD mode works end to end since `2ce6403`.** As first measured here it never parked
anything: `ParkStore.__init__` allocates its two pinned windows on the scheduler thread, which
runs inside `torch.inference_mode()`, so they are inference tensors, while the background save
worker ran on a plain thread with no such context and its first in-place write into a window
raised `RuntimeError('Inplace update to inference tensor outside InferenceMode is not
allowed.')`. Parking then disabled itself fail-safe: manifest directory created, no payload
written, answers still correct. RAM mode escaped it because its per-entry host buffer is
allocated on the worker thread itself.

`2ce6403` makes the save worker re-enter the constructing thread's context — device, ambient
CUDA stream and inference mode — before draining the queue, and records failures in
`last_error` on `/v1/cache/status` instead of one log line. Re-measured live on profile B:
**1,778,479,112 bytes** parked for a 65,600-token prefix, one restore in **1,882.45 ms**,
second turn 3.92 s against 18.27 s cold, and the answer byte-identical to the parking-off
baseline (sha `67369f51…eb21`); `disabled` false and `last_error` null throughout.

The restore passes P0's 1,883 ms gate by **0.6 ms**. It is also ~6x the bench prototype's
317 ms for SSD at 65,536 tokens — the same prototype-to-live gap RAM shows (31 ms against
683 ms), so the cost is the surrounding lookup, page and state allocation, radix insert and
tensor-parallel consensus, not the disk read. A longer parked prefix will exceed the gate, so
SSD restore wants tuning (or the gate wants raising on evidence) before it is defaulted on.

## FP8 KV gate

The gate is: the first 48 temperature-0 tokens identical to the BF16 reference on the same
prompt, and a coherent 250k+ answer that uses both ends of the prompt.

| check | result |
|---|---|
| coherent >= 250,000-token answer using both ends | **PASS** — `maple-7319\|cobalt-4826`, identical to BF16 |
| FP8 deterministic within its own boot | **PASS** — first and replay share sha `f2113d26…a917` |
| first 48 tokens identical to BF16 | **FAIL** — first **20** tokens identical, then divergence |
| **gate overall** | **FAIL** |

The 20 shared tokens are `maple-7319|cobalt-4826<|im_end|>\n<|endoftext|><|im_start|>\n`.
Divergence begins only after `<|im_end|>`, that is, after the model has already ended its
turn; the tokens that differ exist solely because the request sets `ignore_eos` and forces
generation to continue past the end of the answer. BF16 continues with
`<|im_start|><|im_end|>\n…` and FP8 with `<|im_end|>\n<|endoftext|>…`; BF16 emitted 47
tokens and FP8 45.

Stated plainly: **the answer itself is identical to BF16** (`maple-7319|cobalt-4826`, both
ends of a 250k-token prompt, in order) and FP8 is **+2.9%** faster. What fails is the gate's
48-token window, which reaches past `<|im_end|>` into padding that only exists because the
request sets `ignore_eos`. That tail is the most numerically fragile part of the output and is
not stable even within BF16 — a cache-hit BF16 run and a cold BF16 run produce different tails
on the same prompt — so the gate is comparing the one part of the output that carries no
information. **Recommendation: redefine the gate as "identical up to and including
`<|im_end|>`"** and re-run; the existing evidence already passes on that reading, and such a
gate would still catch a real regression in the answer. Until that decision is made the
recorded verdict stays FAIL and **the FP8 KV switch stays off by default**.

FP8 also costs accuracy that the offline work already predicted: measured aggregate
relative-L2 of the FP8 attention output against BF16 sits at 3.3-4.2% (median 3.6%) over 24
draws of the kernel test's shape, against the ~2.4% E4M3 arithmetic prediction.

## What is left running

Profile A (BF16, parking off, four requests, 262,144-token context). The FP8 gate failed and
FP8 was only 2.9% faster, so the default boot stays.

## Follow-ups

1. **Fixed by `2ce6403`** — the parking save worker now re-enters its caller's inference
   mode, device and stream, and both `ram` and `ssd` park and restore live. `ram` remains
   limited on this box by host RAM, not by the setting; see follow-up 5 for `ssd`.
2. The two live drivers assert exactly 48 completion tokens; the server reports 47. Corrected
   in the drivers, but the same assumption may exist elsewhere.
3. `tests/models/qwen4_exp/test_qsa_fp8_kernels.py::test_fp8_sparse_attention_dequantizes_scales_with_masking_and_split_k`
   fails about one run in three: it seeds a generator for Q/K/V but selects its columns with
   the global CUDA RNG, and its 4% aggregate ceiling sits inside the natural 3.3-4.2% spread.
4. **Withdrawn — not a routing defect.** One early four-request run appeared to carry another
   request's marker text. D1 traced it: the probe's four secrets differed by a single digit,
   which is below the model's discrimination threshold once a mixed prefill batch perturbs the
   numbers, and incrementing that digit is the model's own strongest continuation. Nothing
   crossed between requests — every solo re-issue against the same warm cache answered
   correctly. With distinct-word secrets the failure disappears: **0 mismatches in 200 normal
   runs, 0 in 60 distinct-word runs, 0 in R3's 20 further runs**. Profile A also runs none of
   today's new code (parking and FP8 are both gated off). The corrected probe is
   `scripts/bench/kv_concurrency_distinct.py`.
5. **SSD restore tuning.** 1,882.45 ms against an 1,883 ms gate is no margin; the cost is the
   live lookup/allocate/radix-insert/consensus path, not the read.
6. **FP8 gate redefinition.** Cut the gate at `<|im_end|>` instead of 48 forced tokens, then
   re-run before FP8 is judged.
7. **Packed-batch logit perturbation.** D1 measured that a mixed batch changes the token run
   for essentially every request in it (15/15 with distinct-word prompts) while solo answers
   are stable. Worth an experiment to bound how far greedy decoding can move with batch shape.
8. **Settings web page with restart**, so parking mode, FP8 KV and the other switches can be
   changed without hand-editing a launch command. Owner-approved as a separate job.
