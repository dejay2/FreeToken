# Test playground (own model system, part 3)

Date: 2026-09-25. Status: design. Jay left every choice to us ("a better playground: speed
numbers, and comparing settings side by side"); the decisions below are ours, with reasons.
Follows part 1 (own switcher, c665011) and part 2 Stage A (control panel, fedf802).

## Why

Jay tries models and settings, then wants to know two things:

- **How fast is it?** For every answer: how long until the first word, how fast it writes,
  how long the whole answer took, how big the prompt and the answer were, and, where the
  engine guesses words ahead (MTP, DFlash2), how many guesses it kept.
- **Which setup is better?** The same prompt on setup A and setup B, side by side. A and B
  can be two models, or one model with two presets, or one model with two answer settings
  (temperature and so on).

There is one graphics card, so A and B cannot be loaded together. Switching takes time:
FreeToken loads in about 2.5 min, NInfer in about 20 s. The page has to run A, then switch,
then run B. It shows each switch as its own step with its own time, so loading never looks
like a slow answer. Afterwards it puts back what was loaded before and says so.

## What already exists (checked 2026-09-25)

- **llama-swap's Chat playground** (`engines/llama-swap/ui`, frozen v257) already shows
  per-answer stats under each reply. `lib/generationStats.ts` + `StatsBreakdown.svelte` give
  time to first token, prompt and generation speed, token counts, cached tokens and draft
  acceptance. They read llama.cpp's `timings` block and OpenAI `usage`. For NInfer this
  already includes guess-ahead acceptance; for FreeToken it shows counts and browser-measured
  timing. It has no compare, and it knows nothing about presets.
- **What each engine sends on a streamed `/v1/chat/completions`** with
  `stream_options.include_usage`:

  | | FreeToken (`server/openai_api.py`) | NInfer, both copies (`src/serve/openai_chat_response.cpp`) |
  |---|---|---|
  | `usage.prompt_tokens`, `completion_tokens` | yes, last chunk | yes, last chunk |
  | cached prompt tokens | `usage.prompt_tokens_details.cached_tokens`, only when nonzero | `timings.cache_n` |
  | engine timing (`timings.prompt_ms`, `predicted_ms`, `predicted_per_second`) | no | yes, last chunk (every chunk with `timings_per_token`) |
  | guess-ahead counts (`timings.draft_n`, `draft_n_accepted`) | **no.** MTP acceptance is only in engine logs, not per request | yes, when speculation ran |
  | `reasoning_content` deltas | yes | yes |
  | rejects unknown fields | no (`extra="allow"`) | no (`timings_per_token` is read) |

  FreeToken's `/v1/stats` has only a 5 s sliding throughput, and `/v1/requests` has no token
  or speculation fields. So guess-ahead numbers for FreeToken are **not cheaply available**.
  The page says "not reported by this engine". A follow-up could add
  `timings.draft_n/draft_n_accepted` to FreeToken's usage chunk.
- **What we compute ourselves:**
  - first word after = first content, thinking or tool-call chunk minus request sent;
  - thinking time = first answer chunk minus first chunk;
  - whole answer = stream end minus request sent;
  - measured writing speed = (answer tokens - 1) / (last token time - first token time).
  The engine's `predicted_per_second` wins when present and is labelled "engine";
  otherwise the value is "measured" and shown with a `~`.
- **The control panel** (`daemon/settings/panel.py`) owns the registry, presets, the switcher
  file, holds ("old settings until the next load"), the P6 load and the unload. It restarts a
  model on new settings safely: it unloads, writes, waits for the switcher's config hash (P5),
  then loads.

## Decisions

| # | Topic | Decision | Why |
|---|---|---|---|
| 1 | Where it lives | A new **Test** tab on the control panel page (helper :2031). llama-swap's frozen UI is **not** changed. Its Chat tab stays for free-form chat and already shows per-answer stats. | Comparing presets needs the registry and the panel's safe restart path, which live in the helper. Putting settings back must happen in the helper anyway, because a browser tab can close mid-test. Jay opens the panel over the tailnet at `https://5090.tail45ff04.ts.net`, where the page cannot call llama-swap's plain-http port (mixed content, CORS). The panel already speaks plain words. Changing the Svelte UI would mean a new frozen patch, a Node build and cross-origin calls into the panel, for chat rendering we can do without. |
| 2 | Who runs the test | The **helper** runs it as one background job: switch, answer, time, put back. The page starts it and polls it every 0.5 s. | Timing is taken next to the switcher, so tailnet latency never enters "first word after". The job survives a page reload or a closed tab, and put-back always runs. |
| 3 | What a setup is | A model, plus **saved settings or one of its presets**, plus **answer settings** sent with the request: temperature, top-p, top-k, longest answer. | One setup type covers all three comparisons: model vs model, preset vs preset, and answer settings vs answer settings. Answer settings need no restart. |
| 4 | How a preset is tried | A **test overlay** in the panel, held in memory. While it is set, the switcher file and FreeToken's effective settings show the test model on the chosen preset alone (no per-model changes). The **registry file is never written**, so no saved setting can be lost. A marker file `~/.config/freetoken/playground-test.json` records the overlay, so a restarted helper can find a model left on test settings. | Writing the registry and writing it back is the unsafe path: a crash, a stale revision or a concurrent edit could leave Jay's settings changed. An overlay gives nothing on disk to restore. The helper's start-up sync already rewrites the file from the registry. |
| 5 | A preset that equals the saved settings | No restart. The plan compares the rendered switcher entry and FreeToken profile settings. | Avoids a pointless 2.5 min FreeToken reload. |
| 6 | Order | Setups run in the order shown (A, then B). A setup already loaded on the right settings is not reloaded. | Predictable, and it saves a load when A is what's loaded. |
| 7 | Load time | The model is loaded through P6 (`POST /api/models/load/{model}`), which answers when ready. The load is its own step with its own time ("Loading (not counted)"). The timed answer starts only after that. | Load time never leaks into answer numbers. |
| 8 | Warm-up | On by default: one short untimed answer ("Reply with the single word OK.", 8 tokens max) after each load. | The first request after a boot pays one-off costs (graph capture, caches). Without a warm-up the first setup looks slower than it is. |
| 9 | Putting back | Always runs, after success, failure, Stop or another app taking the card. It clears the overlay. If the test model is still loaded on test settings and idle, it is put away. If something was loaded before, that model is loaded again on its saved settings ("Load it again afterwards" is on by default). **It never evicts a model another app is using.** When it cannot do something, it says so on the job and on the Right-now strip. | Safe and honest. Nothing is ever left on the wrong settings without a message. |
| 10 | Someone else using the card | Before starting, the test is refused if the loaded model is answering (llama-swap in-flight list). It asks first if the model was used in the last 120 s (llama-swap activity). Mid-test, if another app asks for a model (P1 "latest wins" cancels our load), or the loaded model is no longer ours, the test stops as "Stopped to let another app through" and does not load anything back over that app. | Jay's agents use these models. "Never evict a model Jay is using" is the hard rule. |
| 11 | Panel changes during a test | Saves, import, restore, Load and Unload on the panel are refused with "A test is running on the Test tab…" (409 `test_running`). The FreeToken adapter's `effective` read still works (and returns the test settings). | One writer at a time. The overlay and a save cannot race. |
| 12 | Stop | Stop ends the current step: it closes the answer stream, or unloads the model being loaded (P1 cancels a half-finished swap on unload). Then put-back runs. | A 2.5 min FreeToken load must not make Stop wait. |
| 13 | Speed words | Loading (not counted) · First word after · Writing speed (tokens a second) · Whole answer · Prompt size (with "reused") · Answer size · Guesses kept · Stopped because. The better of A and B gets a small "better" tag when they differ by at least 3%. | Plain words for Jay. Technical names only in the tooltip. |
| 14 | History | The last **20** tests are kept in the browser (`localStorage` key `ft-test-history-v1`). Each answer is capped at 20,000 characters. Buttons: **Copy** (Markdown: prompt, a speed table, both answers), **Export all** (JSON download), **Clear**. When storage is full, the oldest are dropped. | Asked for. It needs no server state, and nothing private leaves the browser. |
| 15 | Answer rendering | Plain text (`white-space: pre-wrap`), thinking in a closed "Thinking" box. No markdown. | Enough for comparing. Markdown and code highlighting are what llama-swap's Chat tab is for. |
| 16 | One or two setups | B has an on/off switch. With B off, the tab is a one-prompt playground with speed numbers. | "Speed numbers for every answer" without a compare. |

## Out of scope

- Changing llama-swap's frozen UI. No new patch; `FROZEN.md` is unchanged. If a later part
  touches it: mark each change `// FreeToken patch Pn`, list it in `FROZEN.md`, build with
  `scripts/engines/build.sh` (Node pinned).
- Guess-ahead numbers for FreeToken (the engine does not report them per request). Follow-up:
  add `timings.draft_n/draft_n_accepted` to FreeToken's usage chunk.
- Repeating a run N times, averages, charts. The Performance page in llama-swap has history.
- Multi-turn chats, images, tools. A test is one system message plus one prompt.
- Editing settings from the Test tab. Presets are made on the model's settings screen.
- Markdown rendering; three or more setups; running two models at once.

## Architecture

```
browser (tailnet) ── Test tab (index.html + static/playground.js)
     │  POST /api/playground/plan · POST /api/playground/runs · GET …/runs/current (0.5 s) · POST …/stop
     ▼
helper :2031  daemon/settings/playground.py   PlaygroundRunner (one job at a time, own thread)
     │              │ uses                       │ uses
     │              ▼                            ▼
     │   daemon/settings/panel.py         daemon/settings/playground_speed.py
     │   PanelService: test overlay,      AnswerTracker + answer_stats (pure)
     │   guards, marker, wait_for_switcher
     ▼
llama-swap :2040  /running · POST /api/models/load/{m} (P6) · POST /api/models/unload/{m}
                  /api/config/hash (P5) · /api/events (in-flight snapshot) · /api/metrics/activity
                  POST /v1/chat/completions (stream) ─► FreeToken :2020 or NInfer :8090
```

- `playground_speed.py`: pure. It folds SSE lines into an `AnswerTracker` and turns that into
  the stats dict.
- `playground.py`:
  - `SwitcherChat` streams one chat and can be aborted from another thread;
  - `SwitcherProbe` reads the in-flight snapshot and the last activity time;
  - `PlaygroundRunner` does plan, start, run, stop, put back and recover;
  - `create_playground_router` holds the routes.
- `panel.py` gains:
  - the test overlay (`set_test_settings`, `_for_switcher` used by `_plan_rewrite` and
    `effective`);
  - `test_preset_key`;
  - `begin_test`/`end_test` and the `test_running` guard;
  - `wait_for_switcher`, factored out of `_reload_after_write`;
  - the marker file and `note_test_leftover`;
  - `test` and `testLeftover` in `now()`.
- `app.py` builds the runner, includes the router and serves `/playground.js`. `server.py`
  calls `playground.recover()` before `panel.sync_config()`.

## UI sketch (in words)

A new tab **Test** after "FreeToken defaults". The Right-now strip stays on top.

1. **Prompt card.** A big text box "What should the model answer?", and a closed "System
   message (optional)" box.
2. **Setups.** Two cards side by side (stacked on a phone), headed **Setup A** and
   **Setup B**. Setup B has a switch "Compare with a second setup". Each card has:
   - **Model**: every model in the registry, with "(loaded)" after the one on the card;
   - **Settings**: "Saved settings", then each preset of that model;
   - a closed box **Answer settings** with "Creativity (temperature)", "Top-p", "Top-k"
     (blank = the model's own) and "Longest answer" (tokens, default 512).
3. **Options:** "Warm up after loading (not counted)" (on), and "Load what's loaded now
   again afterwards" (on).
4. **Run** first shows the plan in a box: the numbered steps with a time guess each,
   for example "1. Put away QUASAR · about 5 s  2. Use preset “Fast” for Fable ·
   3. Load Fable · about 20 s …", then "About 1 min in all". Any warning ("QUASAR was used
   40 seconds ago. The test will put it away.") shows in amber. Buttons: **Start** and
   **Cancel**.
5. **Progress:** the step list with ○ waiting / ◐ running (live seconds) / ✓ done (time) /
   ✗ failed / – skipped, and a **Stop** button while it runs.
6. **Results:** two columns, "A · Fable 27B · preset Fast" and
   "B · Fable 27B · Saved settings". Each has:
   - the speed table (eight rows, with the "better" tag);
   - the answer text, which fills in live;
   - a closed "Thinking" box.
   Below them, the put-back line, for example "QUASAR is loaded again on its saved settings
   (21 s)".
7. **History:** the last 20 tests, newest first. Each row shows the date, the start of the
   prompt and each setup's writing speed. Buttons: **Show** (fills Results from the stored
   record), **Copy**, and at the top **Export all** and **Clear history**.

The Right-now strip shows "Test running: Fable on preset “Fast”" during a test. If a model
was left on test settings, it shows "QUASAR is still on test settings (preset “Fast”)
because an app was using it. It goes back to its saved settings at its next load."

## Data flow

1. The page sends `POST /api/playground/plan` with `{prompt, system, sides:[{model, preset,
   temperature, top_p, top_k, maxTokens}], warmup, putBack}`. The runner:
   - validates;
   - reads `/running` (refuses when unknown, down, or a model is `starting`);
   - reads the in-flight snapshot (refuses when the loaded model is answering for someone
     else);
   - reads the last activity for the loaded model (warns when under 120 s, ignoring the
     test's own requests);
   - builds the steps.
   It answers `{before, held, steps:[{kind, side, model, preset, label, guessS}], estimateS,
   warnings}`.
2. The page shows the plan. **Start** sends `POST /api/playground/runs` with the same body
   plus `expectBefore` (the plan's `before`) and `confirm: true`. The runner plans again. If
   `before` changed, it answers 409 `changed` with the new plan. Then `panel.begin_test()`,
   a new job, and a thread.
3. The thread walks the steps:
   - `unload`: refused if another app is using that model now: the in-flight read first, then
     llama-swap patch P7 (`POST /api/models/unload/{model}?ifIdle=1`, 409 `busy`), whose idle
     check and stop are one run-loop step, so a request that arrives after the read is never
     killed (status `yielded`);
   - `settings`: `panel.set_test_settings(model, preset)`, or clear it for saved settings,
     then `panel.wait_for_switcher(text)`;
   - `load`: `/running` is read again first (another app's model starting or ready →
     `yielded`), then P6, timed into the setup's `loadMs`. A load that Stop's unload missed (it
     arrived before P6 registered the load) is put away again;
   - `warmup`: an untimed short answer;
   - `answer`: checks `/running` says the model is `ready`, streams with
     `X-Session-ID: ft-test-<id>` and `stream_options.include_usage`, and publishes partial
     text and live stats on every chunk.
4. `restore` always runs last:
   - if the overlay's model is loaded and idle, it is put away; if busy, it is left and noted;
   - the overlay is cleared and the file rewritten (a busy NInfer test model gets a hold
     through the existing `_plan_rewrite` rule);
   - the before-model is loaded again when asked and when no other app's model is on the
     card, read again right before the load (a P1 409 there: "… was not loaded again, because
     another app took the graphics card.");
   - when `/running` cannot be read, the test model is noted on the strip and the saved file
     is still waited for; if clearing the overlay fails, the model is noted and the marker is
     left for `recover()`;
   - `panel.end_test()`.
5. The page polls `GET /api/playground/runs/current` every 0.5 s while the job is active.
   When the job ends it saves the job to history once (keyed by job id).
6. At helper start, `recover()` reads the marker. If that model is loaded and idle, it is
   put away; otherwise it is noted. The marker is removed. Then `sync_config()` writes the
   file from the registry as usual.

## Error handling

| Situation | What happens | Words |
|---|---|---|
| Switcher state unknown / down | Plan refused (503) | "Can't tell which model is loaded right now. Try again in a moment." / "The model switcher is not running." |
| A model is loading or unloading | Plan refused (409 `loading`) | "A model is loading or unloading right now. Try again when it has finished." |
| Loaded model is answering for another app | Plan refused (409 `in_use`) | "QUASAR is answering something right now. Try again when it's done." |
| Loaded model used within 120 s | Warning in the plan; Start needs `confirm` | "QUASAR was used 40 seconds ago. The test will put it away." |
| State changed between plan and Start | 409 `changed` with the new plan | "Something changed since the plan was shown. Check the new plan and press Start again." |
| A test is already running | 409 `test_running` | "A test is already running." |
| A panel restart (save with "restart now") or a panel load/unload is still running | Start refused (409 `busy`) | "The control panel is restarting a model with its new settings. Try again when it has finished." / "The control panel is loading or unloading a model right now. Try again when it has finished." |
| Panel save / load / unload during a test | 409 `test_running` | "A test is running on the Test tab. Wait for it to finish, or stop it there." |
| Switcher never picks up test settings (hash) | Step fails, nothing loads, put-back runs | "The switcher didn't pick up the test settings, so nothing was loaded." |
| Load refused: not enough memory (P2 503) | Step fails, put-back runs | "Loading Fable failed: …" (switcher's message) |
| Another app asks for a model mid-test (P1 409 `model_superseded`, or `/running` no longer shows ours) | Status `yielded`; put-back does not load over that app | "Another app asked for a different model, so the test stopped to let it through." |
| Answer error / cut off (no finish reason and no `[DONE]`) | Status `failed`; put-back runs | "Fable could not answer: …" / "Fable could not answer: the answer was cut off before it finished." |
| Stop | Current stream closed or current load cancelled; status `stopped`; put-back runs | "Stopped." |
| Put-back cannot unload a busy test model | Left loaded, hold for NInfer, strip note | "QUASAR is still on test settings because an app is using it…" |
| Helper restarts mid-test | `recover()` at start (see data flow 6) | Strip note when it had to leave the model |
| Browser storage full or blocked | Oldest history dropped; history off if storage throws | (silent; the page still works) |

## Testing

- **Unit (devbox, no GPU):**
  - `tests/settings/test_playground_speed.py`: FreeToken-shaped and NInfer-shaped streams,
    thinking, no usage, one-token answer, junk lines.
  - `tests/settings/test_playground_io.py`: `SwitcherChat`/`SwitcherProbe` against a real
    local HTTP server, including abort and error bodies.
  - `tests/settings/test_playground_panel.py`: overlay, effective, guards, marker, now.
  - `tests/settings/test_playground_runner.py`: plan, run, put-back, stop, yield, recover,
    with a fake switcher, fake chat, fake probe and fake clock.
  - `tests/settings/test_playground_routes.py`: the routes.
  - `tests/settings/test_playground_page.py`: node tests of the page helpers and the page
    contract.
- **Full settings suite:**
  `PYTHONPATH=python .venv/bin/python -m pytest tests/settings tests/daemon -q`.
- **Live acceptance** on the box, run by the controller only, with Jay's OK and at most one
  FreeToken boot. Screenshots are taken with `~/.npm-global/bin/chrome-devtools-axi` against
  `https://5090.tail45ff04.ts.net` (desktop 1440x900 and phone 390x844). The run covers:
  - a one-setup run on QUASAR (guesses kept shown);
  - QUASAR vs Fable (put-back reloads QUASAR);
  - Fable saved vs a temporary preset (registry and config hash unchanged afterwards, `ps`
    shows the saved flags);
  - Stop during a load;
  - the helper restarted mid-test (recover);
  - the in-use refusal (a long curl stream);
  - one FreeToken setup (FreeToken speed, "not reported" guesses);
  - history after a reload, Copy, Export.
  Results go in `docs/research/playground-acceptance-<date>.md`, with screenshots in
  `docs/research/img/`.
