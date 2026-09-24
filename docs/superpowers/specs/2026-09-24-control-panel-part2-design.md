# Control panel (own model system, part 2)

Date: 2026-09-24. Status: design, awaiting Jay's review.
Follows part 1 (`2026-09-24-own-model-switcher-part1-design.md`, merged c665011).

## Why

Part 1 gave one address for every local model. Its settings still live in a hand-edited
`~/llama-swap/config.yaml`, and FreeToken's settings page knows only FreeToken. Jay wants one
page where he can:

- fully configure each engine and each model;
- keep presets;
- see and control what is running;
- add and remove models.

## What Jay decided (2026-09-24)

| Topic | Decision |
|---|---|
| Must-haves | Change model settings; add/remove models; see and control what's running; presets. A settings page per engine covering every NInfer option and every FreeToken setting we have |
| Where | Grow FreeToken's settings page (approach A). llama-swap's own page stays for the playground and logs (part 3) |
| Settings levels | Three: whole system → engine defaults → each model (only its differences, plus presets) |
| Change while loaded | Ask each time: "restart now / next time" |
| FreeToken models | Each FreeToken model keeps its own settings and presets |
| Adding models | Both: browse to a file on the PC, or paste a HuggingFace link to download |
| Home screen | Layout A: a list, with a "Right now" strip on top (mock-up approved) |
| Look | Same style as today's settings page (mock-up approved) |

## Out of scope

- Playground and speed comparison (part 3).
- Sleep mode for FreeToken.
- Engines other than FreeToken and NInfer. The design makes adding one a catalogue plus an
  adapter.
- A login or password. The page stays tailnet-only, as today.

## Build order

- **Stage A:**
  - registry and import;
  - config generator;
  - llama-swap patches P5 and P6;
  - the System, NInfer defaults and FreeToken defaults tabs;
  - per-model screens with presets;
  - load/unload;
  - the restart prompt;
  - fit checks;
  - the Models list and the "Right now" strip.
- **Stage B:**
  - add a model (browse, download, detect the engine);
  - remove a model;
  - keep Pi's model list in step.

Each stage gets its own plan, review rounds and live acceptance.

## Architecture

```
browser ── settings page (helper :2031, FreeToken/daemon/settings)
              │ reads/writes
              ▼
        registry.json  ──generate──▶  ~/llama-swap/config.yaml ──watch──▶ llama-swap :2040
              │                                                            │ runs cmd
              └─ FreeToken models: effective settings ─▶ helper profile ◀──┘ engines/adapters/*.sh
```

### 1. Registry: the single source of truth

- **File:** `~/.config/freetoken/registry.json` on the box. It is machine-specific and never
  tracked.
- **Backups:** every save keeps the previous file as `registry.json.bak-<timestamp>`, and the
  newest 20 are kept.
- **Code:** `daemon/settings/registry.py` loads, validates and saves the file, and a JSON schema
  lives next to it.
- **Shape:**

```json
{
  "version": 1,
  "system": {"floorGB": 6, "waitSeconds": 300, "latestWins": true, "defaultIdleMinutes": 0,
             "helperURL": "http://127.0.0.1:2031"},
  "engines": {
    "ninfer":    {"defaults": {"max-context": 150000, "kv-capacity": 200000, "...": "..."}},
    "freetoken": {"defaults": {"KVCacheTokens": 32768, "...": "..."}}
  },
  "models": [
    {"id": "quasar-27b", "name": "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)", "engine": "ninfer",
     "runtime": "ninfer", "artifact": "~/ninfer-work/models/quasar_27b_nvfp4.ninfer",
     "ramNeedGB": 18, "idleMinutes": 0, "aliases": [],
     "overrides": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 7},
     "presets": {"Fast agents": {"max-concurrency": 6}}, "activePreset": "Fast agents"}
  ]
}
```

- **Effective settings for a model** = engine defaults, then the active preset, then the model's
  overrides; later layers win.
- A preset stores only the values it changes. "Save as new preset" stores the model's current
  differences from engine defaults.
- **First-run import:** read today's live `~/llama-swap/config.yaml` and the helper's current
  boot file. Build the registry from them: memoryGate becomes `system`; each model's cmd flags
  and filters become that model's overrides against the new engine defaults; the helper's boot
  file becomes FreeToken defaults. Then back up the old config as `config.yaml.bak-before-registry`.

### 2. Engine catalogues

Each engine has one catalogue: a list of settings in the same `Dial` shape today's page renders
(name, plain label, blurb, info, type, range or choices, default, advanced flag, effects).

- **FreeToken** reuses `daemon/settings/dials.py` as it is: all 54 dials, their validation and
  `memory_fit.py`.
- **NInfer** gets a new `daemon/settings/ninfer_dials.py`. It covers every `ninfer-serve` option
  from the frozen runtimes' `--help`, grouped into tabs:
  - **Model & chats:** `max-context`, `kv-capacity` (number or auto), `max-concurrency` (1-8),
    `max-pending-requests`, `pending-timeout-ms`, `prefill-chunk`, `default-max-tokens`.
  - **Memory:** `kv-dtype` (bf16, int8, fp8, nvfp4, k8v4), `host-kv-mib`, `host-state-slots`,
    `device-state-slots`, `max-request-mib`, `max-private-continuations`,
    `max-shared-prefixes`, `max-long-anchors-per-continuation`.
  - **Guess-ahead:** `spec` (off, mtp, dflash, dflash2), `draft-tokens` (1-15), `lm-head-draft`.
  - **Answer style:** `temperature` (0-2), `top-p` (0-1), `top-k` (0-20), `min-p` (0-1),
    `presence-penalty` (-2 to 2), `frequency-penalty` (-2 to 2), `seed`, `greedy`, `no-thinking`,
    `default-thinking-budget`, `preserve-thinking`.
  - **Pictures:** `vision`, `media-cache-mib`, `media-live-mib`, `media-preprocess-threads` (0-64).
  - **Advanced:** `no-cuda-graph`, `no-prefix-reuse`, `log-stats-interval-ms`,
    `response-store-max-records`, `response-store-max-mib`, `request-log-jsonl`, `log-level`,
    `cors`, `context-cost-presets`, `api-key`, `device`.
  - Host, port and model id are fixed by the generator and never shown.
- **Runtime support:** a test runs each frozen runtime's `--help` (`engines/ninfer*/build/apps/ninfer-serve`)
  and checks every catalogue flag against it. A flag that only one runtime supports is marked
  with that runtime, and the page hides it for models on the other runtime.
- **Answer-style limits:** the catalogue ranges also feed the generated `clampParams`, so an
  app's out-of-range value is clamped instead of rejected.

### 3. Config generator

`daemon/settings/swap_config.py` turns the registry into `~/llama-swap/config.yaml`.

- **Output:** the file starts with a header line: "generated by the control panel from
  registry.json; edits here are overwritten". It contains:
  - `memoryGate` and `latestWins` from `system`;
  - the adapter macros, with paths under `~/FreeToken/engines`;
  - one model entry per registry model: cmd, proxy, checkEndpoint, `ramNeedGB`, ttl (idle
    minutes × 60), unloadTimeout, aliases, and `useModelName` plus `clampParams` for NInfer.
- **NInfer cmd:** `engines/adapters/ninfer.sh <runtime ninfer-serve> <artifact> <model-id>`,
  followed by the effective settings rendered as flags. `host`/`port` are fixed to 127.0.0.1:8090.
- **FreeToken cmd:** `engines/adapters/freetoken.sh --profile <profile-id> <model folder>`
  (adapter change below).
- **Safe write:**
  1. Write to `config.yaml.new`.
  2. Validate it by running the llama-swap binary's config check
     (`llama-swap --config <file> --check-config`, added as part of P5).
  3. Swap it in atomically.
  4. On a failed check, keep the old file and show the error on the page.
- **Stable output:** the same registry always produces a byte-identical file, so a no-op save
  triggers no reload.

### 4. llama-swap patches (our frozen copy)

- **P5, selective reload.** Today a config reload builds a new server and shuts the old one
  down, which stops every loaded model (`llama-swap.go` reload). Change it so that:
  - a running process whose model entry is byte-identical in the new config keeps running and is
    adopted by the new router;
  - models whose entry changed or was removed are stopped;
  - everything else reloads as before.

  Also add `--check-config` (load, validate, print errors, exit 0/1). Go tests cover: unchanged
  model survives a reload, changed model is stopped, removed model is stopped, invalid config
  keeps the old server.

  This is the riskiest patch. If adopting processes across routers proves unsafe in review, the
  fallback is to keep the single router and apply config changes to it in place for changed
  models only. The plan decides after reading the router code, and the spec's behaviour holds
  either way.
- **P6, load endpoint.** `POST /api/models/load/{model}` starts a swap to that model, reusing
  the normal request path so P1/P2 apply, and returns once it is ready or has failed. It uses
  the same error envelopes (409 superseded, 503 not enough memory).

### 5. Adapters

`engines/adapters/freetoken.sh` gains `--profile <id>`. When given, it does two things instead
of `PUT /api/settings {ModelPath}`:

1. `PUT /api/profiles/<id>` with the model's effective settings, which include ModelPath;
2. `POST /api/profiles/<id>/activate`.

Then it starts as today.

- **Adopt check:** the current model is adopted only if it is serving the same folder *and* the
  active profile is `<id>`.
- **Profile ids:** `model-<registry id>`, created by the page on first save.
- **Stays as today:** ninfer.sh is unchanged. The existing adapter contract tests gain the
  `--profile` path.

### 6. The page

The page keeps its style and code layout. Today's settings UI becomes a set of tabs:
**Models · System · NInfer defaults · FreeToken defaults**. A per-model settings view opens from
the Models list.

- **Right now strip:** what's loaded (llama-swap `/running`), graphics card used/total, Windows
  free RAM, and the cushion. It refreshes every 5 s.
- **Models list:** each row shows status, name and id, engine (with runtime), active preset,
  `ramNeedGB`, idle unload, Load/Unload (P6 and `POST /api/models/unload/{model}`), and Settings.
- **Model settings view:**
  - the engine's catalogue tabs, rendered by the existing dial renderer;
  - each dial shows "from <engine> defaults" or "changed for this model · reset";
  - a preset picker, "Save as new preset", and delete/rename preset;
  - identity fields: name in apps, aliases, memory need, idle unload;
  - the fit check, then Save.
- **Engine defaults tabs:** the same renderer, editing `engines.<engine>.defaults`. Saving shows
  which models inherit the change.
- **System tab:** cushion, memory wait, latest wins, default idle unload.
- **Restart prompt:** when a save changes a loaded model's effective settings, the page asks
  **Restart now** / **Next time**.
  - Restart now unloads the model and loads it again through P6.
  - Next time only saves. With P5, the loaded model keeps running on its old settings until its
    next load.
- **Fit check:**
  - FreeToken models use today's `memory_fit.py` estimate with the model's effective settings.
  - NInfer models get a new estimate: artifact bytes + KV bytes/token (by `kv-dtype`) ×
    `kv-capacity` + a fixed cost for drafter/vision/CUDA graphs. It is calibrated against the
    measured QUASAR boot (29.7 GB used at kv-capacity 200000, int8, DFlash2, vision).
  - The check shows "fits / tight / won't fit" with numbers. It blocks Save only when the
    estimate exceeds the card, and a "save anyway" override is offered, as the FreeToken fit
    check does today.
  - It also shows the host RAM need against the cushion.
- **What happens to the old parts:** the old "Profile" strip and the Start/Stop/Restart buttons
  go. Load/unload happens through the Models list, and FreeToken's profiles become each model's
  presets under the hood. The boot-log viewer stays under FreeToken defaults → Advanced.

### 7. Stage B: add and remove

- **Add model** is a small wizard:
  1. **Source:** browse the PC (existing `browse.py`), or paste a HuggingFace repo link. The link
     is downloaded with the existing `download.py` into `~/models/<name>` or
     `~/ninfer-work/models/`, with progress shown and the checksum verified where the repo
     publishes one.
  2. **Detection:**
     - A `.ninfer` file means the NInfer engine. The runtime is chosen by reading the artifact
       header: v2 goes to `ninfer` (QUASAR fork), v3 to `ninfer-upstream`.
     - A folder with `config.json` whose architecture FreeToken's model registry supports means
       the FreeToken engine.
     - Anything else gets a clear "not supported by your engines".
  3. **Identity:** id (suggested from the file name, must be unique), name, and memory need
     (suggested from file size, editable).
  4. Save regenerates the config. The model appears in the list, and in Pi (below).
- **Remove model:** confirm first, with a checkbox "also delete the model files" (off by
  default). Removing a loaded model unloads it first. Its FreeToken profile is deleted too.
- **Pi sync:**
  - The page keeps the Windows Pi file `C:\Users\jay\.pi\agent\models.json` (from WSL,
    `/mnt/c/Users/jay/.pi/agent/models.json`) in step. It updates only the `freetoken-local`
    provider's `models` list and adds or removes `freetoken-local/<id>` in `settings.json`
    `enabledModels`.
  - It writes a timestamped backup of both files before every change, and never touches other
    providers.
  - New entries copy the context window, max tokens and thinking map from a same-engine
    neighbour.
  - If the Windows path is unreachable, the page shows "Pi not updated" and carries on.

## Error handling

| Situation | What Jay sees |
|---|---|
| A value is out of range or impossible | The field is marked with a plain message, and Save stays off |
| The generated config fails `--check-config` | "The switcher refused these settings: <reason>"; the old config stays live |
| The registry file is missing or corrupt | The page offers "restore the last good backup" and lists the backups |
| llama-swap is down | "Right now" shows "switcher not running". Settings still save and apply on its next start |
| A download fails or the checksum mismatches | The wizard stops, partial files are deleted, and the reason is shown |
| Pi's file can't be reached | "Pi not updated", and everything else carries on |

## Testing

- **Unit tests:**
  - registry (layering, presets, validation, backups, import from today's config);
  - NInfer catalogue (every flag known to each runtime's `--help`; ranges);
  - generator (golden files for the five current models, byte-stable output, clampParams from
    catalogue);
  - NInfer fit estimate (within 10% of the measured QUASAR boot);
  - `freetoken.sh --profile` (adapter contract tests).
- **Go tests:** P5 (the four reload cases) and P6 (load, superseded, not enough memory), with
  upstream's suite passing.
- **Page tests:** the existing helper page test style for the new API routes (`tests/settings`).
- **Live acceptance on the box**, asking Jay first, with at most one FreeToken boot:
  - import today's config and confirm the generated config matches today's behaviour;
  - change a setting on an unloaded model while QUASAR is loaded, and QUASAR keeps running (P5);
  - change a setting on QUASAR, then "Restart now", and it comes back with the new value;
  - presets switch;
  - the NInfer fit estimate is within 10%;
  - a FreeToken model loads through its own profile (the one boot);
  - Stage B: add a small NInfer model from a link, it appears in Pi, then remove it.
- **Reviews:** two rounds before each stage merges.
