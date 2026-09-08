# MTP on the WSL box: where a guess cycle's time goes (2026-09-08)

Operator report: "guess-ahead used to make things faster; now it makes them slower, by too
much to be the VRAM it takes". Measured on the serving PC (RTX 5090, WSL distro `vllm`,
Qwen3.8-Flash-Next-NVFP4, int8 dense, 6429 auto slots, KV 262208 fp8, one request at a time).

## Verdict

1. The "much slower" was a bug, not the speculation: after any SSD spill + recall the verify
   graph runner was gone and every MTP-on step ran eager (20 tok/s). Fixed in 19f061b and
   proven live today: layer 47 -> disk -> pinned, then `MTP spec graph width 1: captured`
   re-appears and the bench stays at 56-62 tok/s; the per-cycle log keeps growing (300 cycles
   over a 1,500-token run after the trip).
2. With the fix in, MTP on this box is a modest win, not a loss: prose/code +5-10 %,
   repetitive content (counting) +20-25 %. It is not the 09-02 Windows-native headline
   (139 tok/s on a cache-friendly numbers prompt) because a cycle here costs ~2.5 plain steps
   and pays back 2.8-4 tokens.

## Numbers

Decode tok/s, `specbench.py` (streamed, 400 tokens, temperature 0, decode-only):

| build | numbers | essay | code |
|---|---|---|---|
| MTP off, `disk` PLE reader (07:2x, short prompt) | 55.8 | – | – |
| MTP off, `mmap` PLE reader (07:2x, short prompt) | 50.3 | – | – |
| MTP on depth 3, timing probe ON (distorts) | 41.6 / 60.0 | 50.2 / 59.3 | 49.2 / 54.4 |
| MTP on depth 5, no probe, fresh boot | 56.7 / 60.7 | 54.7 / 58.9 | 54.2 / 54.3 |
| MTP on depth 5, after SSD round trip | 56.4 / 59.1 | 60.6 / 59.9 | 56.9 / 55.7 |
| MTP on depth 5, 1,500-token counting run | 69.7 | | |

Cycle anatomy, `FREETOKEN_MTP_SPEC_TIMING=1` (device-syncs at every mark, so absolute rates
are pessimistic), depth 3, ms per cycle over 32-160 cycles:

| stage | ms |
|---|---|
| draft (stage 1.8, replay 3.0, readback 0.1) | 6 |
| prepare | 1 |
| verify.forward | 36-39 |
| of which replay.gpu (the verify graph's own device time) | 25-27 |
| of which replay.model (`prepare_cuda_graph_replay` = PLE row staging) | 9-10 |
| verify.accept | 3 |
| tail (rollback 0.7, commit replay 1.2) | 1.2 |
| whole cycle (policy EMA) | 38-58 |
| one plain step (policy EMA) | 14-20 |

Acceptance from `FREETOKEN_MTP_SPEC_CONF_LOG`: depth 3 short prompts 2.77 emitted/cycle
(42 % of cycles take all 3 drafts); depth 5 short prompts 3.98 emitted/cycle (48 % take
all 5); depth 5 counting run 3.47. Cost-aware bar sits at 2.4-3.05, so on prose the policy
keeps speculation off most of the time (spec/plain 40/144 steps) and the request runs plain
with the ~5 % capture-path tax.

## The 10 ms PLE staging is a probe artefact, not a Linux regression

The timed replay path charged 9-10 ms per cycle to the mmap PLE gather. Measured directly:

- raw cold random 160-byte rows from the 47.7 GiB table (`plebench.py`, 15 trials, median):
  madvise(WILLNEED)+serial copy 0.7 ms / 16 rows, 2.1 ms / 64 rows; thread-fanned faults
  3.6 / 14.0; pread x4 1.2 / 3.6. The current Linux path (advise then serial copy) is the
  fastest shape; ACCESS_COPY vs ACCESS_READ makes no difference (0.96 vs 1.17 ms).
- the real `MmapPleStorage.gather` on the real layout: 64 cold rows 3.2 ms with the row
  cache, 0.55 ms warm.
- py-spy on the live scheduler during three prompts: `prepare_cuda_graph_replay` is 1.2 % of
  samples; 71 % is `_process_last_data > synchronize` (waiting for the card).

So the mmap reader costs ~2 ms per plain step (50.3 vs 55.8 with the io_uring `disk`
reader) and ~2-3 ms per verify, not 10. The remaining cost of a cycle is the verify graph's
device time (expert fetch for the union of the rows' experts), which is structural.

## What would move the needle

- The `disk` (io_uring) PLE reader is refused with MTP because the spec capture path never
  enters `forward_host_ctx`; teaching it to stage the verify rows would recover the ~10 %
  the mmap reader costs every plain step of an MTP-on boot. Larger job (ple_disk.py memops).
- Nothing cheap on the verify side: 25-27 ms of device time per cycle is the H2D expert
  fetch, and slots are the only dial (docs/research/memory-audit-qwen38-rtx5090.md).

Boot file left at depth 5 (the 09-02 default; measured no worse than 3 here), MTP on.
