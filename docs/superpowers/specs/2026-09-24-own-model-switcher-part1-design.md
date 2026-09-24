# Own model switcher, part 1: frozen engines, one address, our rules

Date: 2026-09-24. Status: design, awaiting Jay's review.

## Why

Jay wants one system he owns that serves every local model on the RTX 5090 PC through one
address. It should take the best parts of existing tools, stay unaffected by other projects'
updates, and let engines be added or removed. Today (PR #13, 839f586) llama-swap v257 runs
as a downloaded binary in front of FreeToken and NInfer. It works, but it is someone else's
program. Its switching rules have already annoyed Jay: a wrong pick waits out a two-minute
FreeToken boot, and the playground sends `top_k` values that NInfer rejects.

The whole system is three sub-projects, each with its own spec, plan and build:

1. **This spec:** freeze the engines into this repo, one build, one service, our own
   switching rules, engine plug-ins.
2. Control panel: settings screens for every model and engine, presets, fit checks.
3. Test bench: playground upgrades (speed numbers, side-by-side settings).

## What Jay decided (2026-09-24)

| Topic | Decision |
|---|---|
| Where it lives | Inside this FreeToken repo |
| How engines are kept | Plain frozen copies (approach A): their files copied into folders with a note of the source version. Nothing updates unless we change it |
| Wrong model picked | A new pick cancels a half-finished load straight away |
| Idle models | Set per model: keep loaded, unload after X minutes, or (later, own spec) sleep after X minutes |
| Switching while a model sleeps | The sleeping model is fully unloaded first (applies once Sleep exists) |
| PC restart | Everything comes up ready, with no model loaded |
| Not enough free memory | Wait until there is room, then load; give a clear error after a time limit |
| Engines in version 1 | FreeToken (Qwen3.8 Flash NVFP4 and ABLITERATED) and NInfer (QUASAR, Fable, Twin). Others later as plug-ins |
| Address | Unchanged: tailnet `12020` (and `127.0.0.1:2040` on the PC); Pi and other apps unchanged |

## Out of scope for part 1

- **Sleep for FreeToken** (free the card, keep host banks in RAM, wake fast). This is a new
  engine feature and gets its own design.
- **Settings screens and presets** (part 2). In part 1, settings still live in the config file
  and FreeToken's existing settings page.
- **Playground upgrades** (part 3). llama-swap's existing playground keeps working.
- **Other engines** (llama.cpp, SGLang, the GLM EXL3 setup).

## Layout

```
engines/
  README.md                  what each folder is, source + version, how to update on purpose
  llama-swap/                frozen copy of mostlygeek/llama-swap v257 (f00d375), MIT
    FROZEN.md                source URL, commit, date copied, list of our patches
    ...                      Go source incl. its web UI; LICENSE kept
  ninfer/                    frozen copy of NInfer, Apache-2.0 (see "One NInfer or two")
    FROZEN.md
  adapters/                  one small start/stop/ready adapter per engine
    freetoken.sh             today's scripts/llama-swap/freetoken.sh, moved
    ninfer.sh                replaces other-engine.sh plus the inline ninfer-serve flags
    CONTRACT.md              what every adapter must do
  config/
    config.example.yaml      template; the live config stays outside the repo
scripts/engines/
  build.sh                   builds llama-swap (Go) and NInfer (CMake/CUDA) from engines/
  install-service.sh         writes the systemd --user unit, points it at the built binary
```

- The live config stays machine-specific at `~/llama-swap/config.yaml` on the box. It holds
  paths and model files and is not tracked.
- `scripts/llama-swap/` is removed once `engines/adapters/` replaces it.

**Frozen copy rules.** Each `FROZEN.md` records the upstream URL, commit and copy date, and
lists every patch we made, one line per patch with the file it touches. Updating from upstream
is a deliberate job: diff the new upstream against the recorded commit, bring over only what
we want, and re-apply our patch list. Nothing is fetched automatically.

**Repo size.** llama-swap is about 7 MB and NInfer about 24 MB, both mostly source. Build
outputs (`build/`, Go binaries, `node_modules`, the UI bundle) are git-ignored. If the NInfer
tree carries large test fixtures or vendored third-party blobs, those are left out and fetched
by `build.sh` at a pinned version, with the pin recorded in `FROZEN.md`.

**One NInfer or two.** QUASAR needs the MirkoCovizzi/ninfer-rtx5090-mobile runtime
(d4bc75db), which includes upstream through ce7dee50. Fable and Twin run today on upstream
Neroued/ninfer f76e19c0.
- **First build step:** start Fable and Twin on the mobile runtime once each, send one chat,
  and compare decode speed with today's runtime.
- **If both work and speed is within about 5%:** freeze only the mobile runtime as `engines/ninfer`.
- **Otherwise:** freeze both, as `engines/ninfer` (upstream) and `engines/ninfer-quasar` (mobile),
  and record the reason in `engines/README.md`.

## Engine plug-ins (adapters)

**The contract** (in `engines/adapters/CONTRACT.md`). An adapter is an executable that
llama-swap runs as a model's `cmd`. It must:

1. **Make room.** Stop anything else that holds the card through that engine's own proper stop
   path; FreeToken is stopped through its settings helper, which also disarms its watchdog.
   If the card cannot be freed, refuse (exit non-zero).
2. **Start the engine** on a fixed local port and stay in the foreground while it runs, either
   by exec'ing the engine or by monitoring it.
3. **Report readiness only when chats will be answered.** The model's `checkEndpoint` returns
   200 only then. FreeToken uses `/ready?model=<folder>` and NInfer uses `/health`, which it
   opens only after "engine ready".
4. **Stop fully on SIGTERM:** exit only when the card is released, and exit non-zero if the
   stop failed.

`freetoken.sh` already meets this: it is PR #13's two-round-reviewed script, moved.
`ninfer.sh` takes `<runtime> <artifact> <model-id> [flags...]`. It does the make-room step
now done by `other-engine.sh`, then execs the runtime. The shared flags move into the
config's macros.

**Adding an engine later** means writing one adapter against this contract, adding its build
step to `build.sh` if it needs building, and adding config entries. Removing one means deleting
its adapter and entries.

## Our patches to llama-swap

Each patch is small, has Go unit tests next to the upstream tests it touches, and is listed
in `engines/llama-swap/FROZEN.md`.

### P1: a new pick cancels a half-finished load ("latest wins")

- **Where:** `internal/router/scheduler/fifo.go`, `OnRequest` step (4). Today a request that
  collides with an in-flight swap is queued until that swap finishes.
- **Change:** when the colliding in-flight swap is still loading its target (process state
  `starting`) and the new request is for a different model, cancel that swap:
  1. Stop its target. The process layer already supports cancelling a start: a Stop during
     `starting` ends it. Measured 2026-09-24: an unload during a NInfer start stopped it in
     0.25 s and freed the card.
  2. Answer that swap's waiters with a clear error: HTTP 409, "cancelled: model X was
     requested instead".
  3. Start the swap for the new model.
- **Kept as today:** a swap whose target is already `ready` is never interrupted.
- **Setting:** config `latestWins: true` (default true), so the old behaviour can be restored.
- **FreeToken:** cancelling its load sends SIGTERM to `freetoken.sh`. Its handler turns that
  into a helper Stop, which cancels the start job (verified in the PR #13 review).

### P2: wait for memory before loading

- **Where:** in the swap goroutine, before the target process starts.
- **Per-model config:** `ramNeedGB`, for example about 60 for FreeToken Flash and about 18 for a
  NInfer 27B with its host KV. Global config: `ramFloorGB`, default 6, and `ramWaitSeconds`,
  default 300.
- **Rule:** start only when Windows free RAM − `ramNeedGB` ≥ `ramFloorGB`. Otherwise re-check
  every 5 s until `ramWaitSeconds`, then fail the swap's waiters with HTTP 503: "not enough
  free memory to load X (need N GB, Windows has M GB free); close something and try again".
- **Measuring Windows free RAM:** read the same way as the settings helper's governor:
  `powershell.exe ... (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory`, cached for
  2 s.
  - The swap stops the evicted model first, then measures. The memory the old model held is
    therefore already free (or returning) when the check runs.
  - If the reading fails, log it and do not block the load, so a broken probe never makes
    models unusable.
- **The number is conservative.** Memory WSL has freed but not yet returned to Windows counts
  as used.

### P3: per-model idle unload

llama-swap already has `ttl` per model. P3 only makes sure that expiry goes through the
adapter's SIGTERM path, so FreeToken is stopped through its helper. It also documents `ttl` in
the example config: 0 means keep loaded, N means unload after N seconds idle. Sleep is left for
its own spec. No scheduler change is expected. If testing shows the ttl path skips `cmdStop` or
the unload timeout, that gets fixed here.

### P4: engine setting limits

- **What exists now:** the live config strips `top_k` for NInfer models, because NInfer rejects
  top_k > 20.
- **Change:** a per-model `clampParams` filter, e.g. `top_k: [0, 20]`, `temperature: [0, 2]`,
  `top_p: [0, 1]`, in the same filter code path as `stripParams`. A value outside the range is
  pulled to the nearest limit instead of being dropped, so the client's intent survives.
- **NInfer entries** get the ranges NInfer's server enforces (`src/serve`, recorded in the
  example config).

## Build and service

`scripts/engines/build.sh` runs on the box (WSL `vllm`). It:
- builds `engines/llama-swap` with Go, installed at a pinned version if missing, into
  `~/.local/share/freetoken-engines/bin/llama-swap`;
- builds `engines/ninfer` with CMake/Ninja for `sm_120a`, using CUDA 13.x as today;
- builds the web UI from its frozen source (Node pinned; the build output is git-ignored).

`install-service.sh` rewrites the `llama-swap` systemd --user unit to run the built binary:
same flags, `--listen 127.0.0.1:2040 --watch-config`, and start-on-boot with no preload.
`freetoken-settings` stays as it is. Tailscale serve (12020 → 2040) is unchanged.

**Switch-over.** Keep the downloaded v257 binary and `~/ninfer-work` builds until the frozen
builds pass the acceptance list below. Then point the unit at the new binary, and the config
macros at the new NInfer builds. Rollback is the reverse.

## Error handling summary

| Situation | What the caller sees |
|---|---|
| Wrong model superseded by P1 | 409 "cancelled: model X was requested instead" |
| Memory wait timed out (P2) | 503 with the need and free numbers |
| Memory probe broken (P2) | Load goes ahead; warning logged |
| Adapter could not free the card | The model fails to start; llama-swap returns its start error; adapter log says why |
| FreeToken boot fails | `freetoken.sh` stops what is left and exits 1 (PR #13 behaviour) |
| Parameter out of range | Clamped (P4), never a 400 from the engine |

## Testing

- **Go unit tests** for P1 (cancel a starting swap, keep a ready swap, `latestWins: false` keeps
  old behaviour, waiters get 409), P2 (wait then start, timeout 503, probe failure proceeds,
  measurement taken only after the evicted model has stopped), and P4 (clamp both ends, untouched when in
  range, absent param untouched). Upstream's own test suite must still pass after the patches.
- **Adapter checks** on the devbox with a fake helper and a fake engine: make-room, refuse,
  SIGTERM stop, exit codes.
- **Live acceptance on the box.** Ask Jay first. At most one FreeToken boot, and Windows free
  RAM is sampled throughout.
  1. All five models are listed; QUASAR loads and answers through 12020.
  2. Pick FreeToken, then QUASAR within 10 s. The FreeToken load is cancelled, QUASAR answers,
     and the first caller gets 409.
  3. QUASAR → Fable → QUASAR swaps work on the frozen NInfer build(s); decode speed is within
     5% of today's (QUASAR DFlash2 K7 about 293-320 tok/s code, 199-209 chat).
  4. With `ramNeedGB` set artificially high, a request waits, then gets the 503 message.
  5. A playground request with `top_k: 40` succeeds (clamped to 20).
  6. `ttl: 60` on Fable unloads it after a minute; the card is back to baseline.
  7. After `wsl --shutdown` and restart (only with Jay's OK), the service comes up with no
     model loaded.
- **Review:** two review rounds on the PR before merge, as with PR #13. Merge only when both
  pass and the live list passes.

## Open points settled during the build

- **One or two NInfer copies** is decided by the first build step above.
- **`ramNeedGB` defaults** are measured from one boot of each model (host RSS plus pinned
  memory) and written into the example config.
