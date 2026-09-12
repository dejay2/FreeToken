# Dynamic KV pool: live acceptance on the RTX 5090 WSL box, 2026-09-12

Branch `feat/dynamic-kv-pool` at 22596bf (PR #5), settings helper 1.5.0, Qwen3.8-Flash-Next-NVFP4,
fp8 KV, page 64, `MaxRunningRequests 2`, `KVPark ram` (idle 0 ms, 8 GiB), memory governor ON
(cushions 1.5 GB card / 4 GB RAM), MTP off, no GPU-owned layers at boot. Dials for the run:
`KVDynamic` on, floor 65,536, step 32,768, Timer 1 10 min, Timer 2 5 h, `ContextTokens 262144`,
`KVCacheTokens 262208` (the box had been left at 52,224 / 122,880 by an earlier fit-check
suggestion; backup of the boot file at `~/boot-2020.before-dynamic-kv.ps1`). Client:
`~/kvdyn.py` on the box (streamed `/v1/chat/completions`, filler prompts, usage read back);
`scripts/squeeze-test/hammer.py` and `bench.py`. Spec items: `docs/superpowers/specs/
2026-09-12-dynamic-kv-pool-design.md`, "Live acceptance". Times are the box's log clock.

## Results

| # | Item | Result |
|---|---|---|
| 1 | Boot at the floor | PASS. 220 s to healthy. Ledger: `Dynamic KV pool on: floor 65536, ceiling 262144, step 32768; budget 19.40 GiB`; pool 1,025 pages (65,536 usable), **7,201 slots** (the spec's calculation said about 7,200), 3.5 GiB free on the card. |
| 2 | Short-chat speed | INCONCLUSIVE. 3 x 200 tokens right after boot: 32.8 / 38.3 / 52.5 tok/s (cold slot cache warming). Warm run at 15:24 (pool 163,840, 7,242 slots): median 51.9 tok/s. No clean same-day A/B: the governor moved layers (including one layer to disk = eager decode, 15:10-15:11) during the run, and QUASAR shares the card. A controlled A/B is a follow-up. |
| 3 | 150k conversation | PASS. A: 135,856-token prompt; held (`need 135920 > pool 65536`), `KV pool grow: 65536 -> 163840, slots -> 6731`, `Cache rebuilt` **1 s** later, admitted. TTFT 62.9 s = the prefill. Turn 2: `cached_tokens 135808`, 4.8 s. |
| 4 | Second conversation while the first decodes | PASS. B (36,283 tokens) arrived while A decoded 300 tokens: held (`need 36331`, pool held by A), `KV pool grow-concurrent: 163840 -> 196608, slots -> 6574` when A's turn ended, rebuild 4 s (first visit to that geometry), B admitted. Next turn A and B ran side by side (6.5 s each, both cache hits). |
| 5 | Timer 1 shrink | PASS. Last request 15:14:45; `KV pool shrink: 229376 -> 65536, slots -> 7711` at 15:24:07, `Cache rebuilt` 15:24:17 (10 s: parks every eligible prefix first, then the rebuild). Card free memory 2,205 -> 2,389 MiB (byte-neutral, small gain from fragmentation). `/health` serving. Slots landed at 7,711 rather than 7,201 because the governor had recalled layers to host RAM meanwhile and the budget followed (22.25 GB). |
| 6 | Resume after the shrink | PASS. A turn 5: held, grow to 163,840 in 2 s, `cached_tokens 135872` (RAM restore, parking hits 2 -> 3), whole turn 8.9 s. Without parking this would have been a 63 s re-read. |
| 7 | Zero failures | PASS. Every chat request 200; hammer 8 + 6 requests, 0 failed. No WARNING/ERROR lines this boot except the deliberate rejection of item 10. |
| 8 | Governor interplay | PASS (natural squeeze, no grab script). The RAM axis stepped layers throughout (15:10 gpu_owned->disk then three pinned->gpu_owned, slots 6574 -> 5038; 15:13-15:20 recalls back to pinned, slots up to 6929). `pool_budget_bytes` followed every step (20.83 -> 16.57 -> 19.41 -> 22.25 GB) and the later shrink used the governor-adjusted budget instead of undoing it. |
| 9 | Starvation | PASS on the second try. First try: C (181k) fitted the grown pool, no hold. Second: D 217,344 tokens with `hammer.py` running: `holds request 19 ... 1 held`, the next short request `holds request 20 ... 2 held` (drain barrier), `grow-concurrent 196608 -> 229376` in 3 s, D prefilled 93 s, the 6 short requests completed after it (max first-token wait 88 s), 0 failures. |
| 10 | Failure outcomes | PASS for the live rejection: `POST /v1/cache/rebuild {"moe_cache_size": 20000}` -> `rejected ... needs 54.47 GiB > budget 20.91 GiB; old cache kept, still serving`; a chat right after hit its cache. Rolled-back and failed outcomes are CPU-tested only (`tests/scheduler/test_kv_dynamic_scheduler.py`). |
| 11 | Overlap with governor/manual operations | PASS by observation: governor steps and automatic rebuilds interleaved all run long, `open_operations` always `[]` between them, ids unique; the delayed-delivery race is CPU-tested (`tests/server/test_maintenance_ownership.py`). |
| 12 | RAM TTL with no traffic | NOT RUN live (needs a boot with a minutes-long TTL; the dial is in hours). CPU-tested with two clocks. |
| 13 | Shared-prefix pair | NOT RUN as a separate case; every follow-up turn above was admitted on its resident prefix without a grow (CPU-tested in `test_kv_dynamic_probe.py`). |
| 14 | MTP after a resize | NOT RUN: MTP is off on this box. |

Counts: 5 automatic rebuilds in 20 minutes (4 grows, 1 shrink), all `ok`, none capped; every
rebuild 1-4 s except the shrink (10 s, dominated by the synchronous park of six prefixes).

## Numbers worth keeping

- Grow rebuild: 1 s (warm geometry) to 4 s (first visit); shrink with six parked prefixes: 10 s.
- 136k prefill 62.9 s; 217k prefill 93.6 s (the 150k Claude Code case the spec describes costs a
  minute on the first turn whatever the pool does; the pool adds about a second).
- RAM restore after a shrink: `cached_tokens 135872`, 8.9 s for the whole turn.
- Slots: 7,201 at the floor with the boot budget; 6,731 at 163,840; 6,574 at 196,608; 5,393 at
  229,376 under a governor-reduced budget; 7,711 at the floor after the governor recalled layers.

## Watch items and follow-ups

- Short-chat speed gain is unmeasured cleanly (item 2): run `bench.py` at 65,536 / 7,200 slots and
  at 262,208 / 6,260 slots on a quiet card, same day, governor idle.
- The box's Windows RAM sat at 2.2-4.2 GiB free during the run; the governor's RAM ladder was busy.
  Parking now holds 7 entries; Timer 2 (5 h) will clear them tonight.
- `last_rebuild` in `/v1/cache/status` is overwritten by automatic replies (known).
- Boot took 220 s (the 130-215 s range seen before).
