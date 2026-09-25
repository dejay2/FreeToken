# FreeToken sleep: live acceptance (2026-09-25)

Box: Windows 11 + WSL `vllm`, RTX 5090 (32 GB), `feat/freetoken-sleep` at a17d2a9, model
`qwen3.8-flash` (Qwen3.8-Flash-Next NVFP4), daily profile (MTP off). The memory cushion was 5 GB
for this session with Jay's OK (Bambu Studio open); it goes back to 6 GB afterwards.
One FreeToken boot. Screenshots: `img/sleep-*.png`. Bench JSON: `sleep-bench-mtp-off-2026-09-25.json`.

## Numbers

| | value |
|---|---|
| Card used, nothing loaded (BASE) | 1,635 MiB |
| Cold boot (panel load through the switcher) | 114 s; Windows free RAM minimum 4.9 GB |
| Card used awake | 27.97-29.98 GiB |
| Card used asleep (3 cycles) | 7.28-7.29 GiB (≈ BASE + 5.7 GiB; dense weights stay on the card in phase 1) |
| Released per sleep | 25.2 GB (`released_bytes` 25,228,759,040) |
| Sleep time | 0.86-1.15 s |
| Wake time (`POST /v1/wake`) | 7.92 / 8.09 / 8.79 s — 13x faster than the 114 s cold boot |
| Chat to a sleeping model (auto-wake) | 12.4-13.2 s to the answer (wake 8.3-9.1 s inside it) |
| Greedy output after wake | identical to before sleep in all 3 cycles |
| Windows free RAM asleep vs awake | 9.4 vs 9.4 GB (host banks stay in RAM, as designed) |

## Checks

| Check | Result | Evidence |
|---|---|---|
| Bench, 3 cycles, MTP off | PASS | every cycle `same_output`, `wake_s` ≤ 8.8 (limit 45, target 30), asleep ≤ BASE + 7.5 GiB |
| Panel Sleep / Wake | PASS | awake row with Sleep; after Sleep "Asleep (graphics card free)", Wake button, card 7.7 GB on the Right-now strip, notice "…is asleep. The graphics card is free; a chat wakes it."; "Waking… about half a minute"; awake again at 29.7 GB — `sleep-a-awake.png`, `sleep-b-asleep.png`, `sleep-c-waking.png`, `sleep-d-awake-again.png` |
| Auto-wake over the tailnet (`100.106.5.124:12020`, the one address) | PASS | answered in 11.3 s from asleep |
| Card busy (another process holding 25 GB) | PASS, one message fix | chat got an immediate plain 503 "…could not wake: … close the game or program using it, then try again"; `/health` stayed `ok`/`sleeping`; after the hog exited a chat woke the model and answered in 13 s. The message quoted "17.7 GB free" while ~1.9 GB was really free — fixed in the review round |
| RAM squeeze while asleep (grab.py, 8 GB for 60 s) | PASS | governor status: `ram down -> pinned->disk (free RAM 3.9 GiB)`, no card step; after the wake (16 s with a layer on the SSD) the answer was correct and the governor recalled layers (`ram up -> disk->pinned`) |
| Switch while asleep (panel/P6 load of `quasar-27b`) | PASS | FreeToken stopped fully (no process, `/health` gone), QUASAR ready; 49 s end to end; card 29.4 GB (QUASAR), Windows free 49.6 GB |
| GPU tests on the box | PASS | 68 passed incl. `test_sleep_gpu.py` 2/2 (not skipped) |
| Fault watch | PASS | no `nvlddmkm` events in the System log over the last 4 hours |

MTP-on round (plan step 6) skipped: the box's daily shape is MTP off.
