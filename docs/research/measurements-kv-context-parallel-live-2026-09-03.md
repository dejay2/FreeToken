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
warmth; the cold first run is the low end.

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
| ssd | 0 | 0 | 0 | 0.0 | yes (parking never engaged) | **fails to engage** |

With parking off every counter in the `/v1/cache/status` `parking` block stayed at zero
through both turns, which is the live confirmation that the off path builds no store.

**RAM mode works end to end.** It parked 1,778,471,176 bytes for a 65,600-token prefix (the
sizing note predicted 1,776,586,760 bytes at 65,536 tokens) and restored it in **683 ms**,
and the restored turn produced a byte-identical answer. The bench prototype measured 31 ms
for the RAM copy at 65,536 tokens; the live restore is 22x that, because the live path also
does the lookup, page and state allocation, radix insert and tensor-parallel consensus that
the prototype did not. It is still well inside the 1,883 ms gate.

**SSD mode never parks anything on this build.** The first save fails and parking disables
itself:

```text
KV parking disabled after ssd save failed: RuntimeError('Inplace update to inference tensor
outside InferenceMode is not allowed. You can make a clone to get a normal tensor before
doing inplace update.')
```

Reproduced on two separate boots. `ParkStore.__init__` allocates the two pinned windows
inside the scheduler's inference-mode setup, so they are inference tensors, while the
background save runs on a plain thread with no inference-mode context and its first in-place
write into a window raises. RAM mode escapes this because its per-entry host buffer is
allocated on the worker thread itself. The failure is fail-safe: parking switches off, the
manifest directory is created but no payload file is written, and serving continues to
produce correct answers. Fixing it means entering `torch.inference_mode()` in the parking
worker loop, or allocating the windows outside inference mode; that repair is not part of
this measurement run.

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

Two things follow. First, by the letter of the gate FP8 fails, and that is the verdict
recorded here. Second, the gate as written measures the wrong thing: the post-turn junk is
the most numerically fragile part of the output, and it is not stable even within BF16 — a
cache-hit BF16 run and a cold BF16 run produce different tails on the same prompt. A gate
that compared the answer up to the end-of-turn token would have passed FP8, and would still
have caught a real regression in the answer itself. Whoever revisits FP8 should redefine the
gate before re-running it.

FP8 also costs accuracy that the offline work already predicted: measured aggregate
relative-L2 of the FP8 attention output against BF16 sits at 3.3-4.2% (median 3.6%) over 24
draws of the kernel test's shape, against the ~2.4% E4M3 arithmetic prediction.

## What is left running

Profile A (BF16, parking off, four requests, 262,144-token context). The FP8 gate failed and
FP8 was only 2.9% faster, so the default boot stays.

## Follow-ups

1. SSD parking cannot park: the parking worker thread needs the inference-mode context its
   pinned windows were allocated under. Until then only `ram` is usable, and `ram` is limited
   on this box by host RAM, not by the setting.
2. The two live drivers assert exactly 48 completion tokens; the server reports 47. Corrected
   in the drivers, but the same assumption may exist elsewhere.
3. `tests/models/qwen4_exp/test_qsa_fp8_kernels.py::test_fp8_sparse_attention_dequantizes_scales_with_masking_and_split_k`
   fails about one run in three: it seeds a generator for Q/K/V but selects its columns with
   the global CUDA RNG, and its 4% aggregate ceiling sits inside the natural 3.3-4.2% spread.
4. Once, one of four concurrent replies carried another request's marker text (HTTP 200,
   correct token count). It did not recur in 40 further concurrent requests and no cause was
   established. Worth watching if four-request serving is used in anger.
