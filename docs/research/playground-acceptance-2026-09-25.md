# Test tab (playground): live acceptance (2026-09-25)

Box: Windows 11 + WSL `vllm`, RTX 5090, helper 2.2.0 on `feat/playground`, llama-swap rebuilt from
the branch (patch P7 `?ifIdle=1`). Driven through the real page over the tailnet
(chrome-devtools-axi, 1440x900 unless noted). Screenshots in `img/playground-*`.

| # | Step | Result | Numbers | Shot |
|---|---|---|---|---|
| 3 | Empty tab | pass | loaded model (QUASAR) preselected for A, B off; phone 390x844 has no sideways scroll (`scrollWidth <= innerWidth` true) | `playground-empty.png`, `playground-empty-phone.png` |
| 4 | QUASAR, saved settings | pass | first word 64 ms, writing 204.8 tok/s (engine figure, no `~`), guesses kept 30 % (163/546); llama-swap Activity for the same request: 204.757 tok/s, 546 draft / 163 accepted (exact match) | `playground-quasar-one.png` |
| 5 | QUASAR vs Fable | pass | Fable load 17.1 s (not counted); A 210.3 tok/s, 41 ms first word; B 129.6 tok/s, 83 ms, guesses kept 42 % vs 32 %; put-back reloaded QUASAR (19.0 s), `/running` quasar-27b ready | `playground-compare-running.png`, `playground-compare-done.png` |
| 6 | Fable saved vs preset "PG test 3" (draft 3) | pass | during B `ps` showed `--draft-tokens 3`; Right-now `test` = Fable on "PG test 3"; a panel save during the test got 409 `test_running` in plain words; afterwards registry.json and config.yaml sha256 unchanged, no marker, QUASAR back. Saved (draft 4) 131.2 tok/s / 43 % kept vs preset (draft 3) 139.6 tok/s / 50 % kept | `playground-preset-running.png`, `playground-preset-compare.png` |
| 7 | Stop during "Load Twin" | **fail, fixed, pass** | first run: put-back read the test's own still-`starting` Twin load as another app's model and left nothing loaded. Fixed (runner tracks every model it asked to load; `test_…stop…` cases). Re-run: load ✗ "Stopped." (5.0 s), put-back reloaded QUASAR (18.6 s), sha unchanged, no marker | `playground-stopped.png` |
| 8 | Helper restart mid-answer | pass | marker present and engine on `--draft-tokens 3` during A's answer; `systemctl --user restart freetoken-settings`; 12 s later: no marker, Fable put away, config.yaml back to the registry render, `/runs/current` idle, no test / leftover on the strip. (An 8000-token limit was refused first: "Longest answer must be between 1 and 4,096 tokens.") | `playground-after-restart.png` |
| 9 | In-use refusal | pass | a 6000-token stream on QUASAR from another shell: Run → "QUASAR … is answering something right now. Try again when it's done."; nothing unloaded. After it ended: plan warned "QUASAR … was used 3 seconds ago. The test will put it away." | `playground-in-use.png`, `playground-in-use-warning.png` |
| 10 | FreeToken (qwen3.8-flash) | pass (after the memory gate) | first two tries: the P2 gate refused after 5 min (62.7 / 66.1 GB free vs 62 + 6 cushion; Bambu Studio / Blender open) and the page said so, loaded QUASAR back, and (after a fix) showed "did not load" instead of "already loaded". With Jay's OK the cushion was lowered to 5 GB for the session: load 2 min 3 s (not counted), warm-up 12.4 s, first word 5.1 s, writing ~11.4 tok/s (measured, `~`), guesses kept "not reported by this engine", put back "Nothing was loaded before the test." | `playground-freetoken.png` |
| 11 | History, Copy | pass | every run listed newest first with Show / Copy; Copy produced the Markdown table (checked by capturing `navigator.clipboard.writeText`); times now read as local time ("25 Sep, 09:06") | `playground-history.png` |

Notes
- After an NInfer model is put away, Windows gets its memory back from WSL slowly (64.2 → 67.5 GB
  over ~20 s, `drop_caches` made no difference); the gate's 5-minute wait covers that.
- FreeToken's ~11 tok/s on the first answer after a cold boot at a 5 GB cushion is an engine /
  memory observation, not a Test-tab problem; follow up separately.
