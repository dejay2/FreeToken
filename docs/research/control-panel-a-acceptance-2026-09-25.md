# Control panel stage A: live acceptance, 2026-09-25

Box: Windows 11, RTX 5090 (34,190,917,632 bytes as nvidia-smi reports it), WSL `vllm`.
Branch `feat/control-panel`, checked out in `~/FreeToken`. llama-swap was rebuilt from
`engines/llama-swap` (`frozen-v257-freetoken (e30d29f)`). The helper was restarted, and
restarted again after the fit fix (e3c54ef). Backups taken:
`~/llama-swap/config.yaml.bak-before-panel-20260925-065629` and
`config.yaml.bak-before-registry`.

| Step | Check | Result |
|---|---|---|
| 2 | `--check-config` on today's live config | pass: `config is valid: 5 model(s)`, exit 0. `/api/config/hash` equals the file's sha256 (5475b72a…). Registry `missing` |
| 3 | NInfer `--help` catalogue tests on the box | pass: 11 passed, none skipped |
| 4 | First-run import | pass: 0 warnings. `swap_config compare` gives `same`, exit 0. `/v1/models` lists all five ids. `registry.json` is `-rw-------`. The switcher's hash follows the new file |
| 5 | Save Fable while QUASAR is loaded (P5) | pass: QUASAR pid 29169 is the same before and after. `/running` shows quasar-27b ready. Log: `reload: kept [quasar-27b …], stopped [fable-27b], rebuilt [fable-27b]`. Resetting it returned the registry to its original revision. A chat through tailnet 12020 answered |
| 6 | Restart now, then next time | pass. Saving without a choice returns 409 `choose_restart` listing QUASAR. **Restart now:** runs in the background; about 20 s later a new pid had `--max-concurrency 5` and lastRestart showed ok. **Next time:** same pid, still 5, and `held: [quasar-27b]`. After unload the config lost `max-concurrency 5` within about 1 s, and the reload used 4 |
| 7 | Presets | pass with max-concurrency 5. Preset "Fast agents" added. Reset reloaded 4. Picking the preset reloaded 5 (view: `from preset`). "No preset" reloaded 4. The preset was then deleted. Each restart took about 20 s |
| 8 | NInfer fit | **found a gap, fixed it, pass after the fix.** See below |
| 9 | One FreeToken boot through its profile | pass: loaded in 149 s. The memory gate waited (57.2 → 62.9 GB free), then let it through. Helper log shows `PUT /api/profiles/model-qwen3.8-flash` and `activate`; `activeProfile` was `model-qwen3.8-flash`; `/ready?model=…` returned 200. A chat through 12020 answered. It unloaded cleanly and the card went back to 2.4 GB |
| 10 | Damaged registry, then restore | pass: the page reports corrupt, with 20 backups; `/models` returns 409. Restore of the newest backup succeeded. The config hash was unchanged throughout (52b04e6d…). The damaged file was kept as `.corrupt-*`, outside the list |

## Step 8: the fit check and NInfer's up-front reservation

The first test raised QUASAR to max-concurrency 6, and it failed to start, although the fit check said "fits":

`FATAL server failed during startup | requested Engine runtime reservation requires 13177821184 bytes, but only 13111561216 bytes are available for runtime capacity`

The fit check estimated the memory NInfer actually uses, and that estimate was correct. At mc4 it predicted 31.91 GB against 31.97 GB measured. The per-chat measurements were:

| max-concurrency | card used | NInfer "runtime" reservation |
|---|---|---|
| 3 | 30,077 MiB | – |
| 4 | 30,484 MiB | 10.6 GiB |
| 5 | 30,892 MiB | 11.4 GiB |
| 6 | refused | 12.27 GiB needed, 12.21 available |

Two more readings came out the same. At max-context 100k instead of 150k, the reservation was unchanged. With fp8 at mc4 it was 10.4 GiB.

Actual use grows by about 0.4 GiB per chat, but the reservation grows by about 0.85 GiB. NInfer checks the reservation before it starts. Fixes e132f72 and e3c54ef model that reservation from the frozen planner code, and it reproduces all six points to within 0.03 GiB. The page now says "won't fit" at mc6 and gives the reason in plain words. Live after the fix:

- mc4 tight: 10.6 of 12.0 GB
- mc5 tight: 11.4 of 12.0 GB
- mc6 won't fit: 12.3 of 12.0 GB

The room shown live (11.97 GiB) is about 0.24 GiB below the room on the refusal day. The check uses the larger of 2.6 GiB and the live card reading, so it errs on the safe side.

## Other findings

- **Windows free RAM during the FreeToken boot:** the minimum was **2.7 GB**, below the 6 GB cushion. The boot cost about 61 GB on the Windows side (64 → 2.7). `ramNeedGB: 58` is about 3 GB too low. Suggestion: raise both FreeToken models to 61–62 on the page.
- **Restore wording:** "Restore newest" returns the list as it was just before the last save, which is correct but not obvious. It brought back a preset that had just been deleted.
- **Not covered live:** Fable and Twin reservation constants. They reuse QUASAR's workspace terms and had 3.5 GB spare at today's settings. They need one measurement with `--log-level debug` before relying on edge cases.
