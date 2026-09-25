# llama-swap (frozen copy)

- Source: https://github.com/mostlygeek/llama-swap
- Version: v257, commit f00d375a927f72d72e74ca86012be210f155c67b (2026-09-22)
- Copied: 2026-09-24, without .git and ui/node_modules
- License: MIT (LICENSE.md, kept unchanged)

Nothing here updates by itself. To take something from upstream: diff upstream against the
commit above, copy only the wanted change, re-run the tests, and add a line below.

## Known upstream test state

(none failing)

## Known upstream test changes

- P1: eight upstream `TestFIFO_*` tests in internal/router/scheduler/fifo_test.go expect a request
  for another model to queue behind a not-ready in-flight swap. Latest wins cancels that swap
  instead, so each of those tests now builds its FIFO with `config.FifoConfig{LatestWins: boolPtr(false)}`
  and nothing else in them changed: QueueOnEvictionCollision, QueueDrainPromotesMultiple,
  QueueCollation, OnUnload_DropsQueuedRequests, PriorityQueueOrder, OnCancel_QueuedRequest,
  ConcurrencyLimit_QueuedWaitersReserveCapacity,
  ConcurrencyLimit_CancelledQueuedWaiterReleasesReservation.
  The same applies to two router tests, which now set
  `conf.Routing.Scheduler.Settings.Fifo.LatestWins` to false:
  TestGroup_SameGroupSwapSerialises (internal/router/group_test.go) and
  TestMatrix_IncompatibleQueues (internal/router/matrix_test.go).
- P1: the router test fake `fakeProcess.EnsureReady` (internal/router/helpers_test.go) now returns
  ctx.Err() without starting when its ctx is already cancelled at the start decision, as
  ProcessCommand's run loop now does. It also gained optional `ensureGate`/`ensureExit` hooks, which
  are nil in upstream tests.
- P5, P6: none; upstream router and server tests pass unchanged.

## Our patches

Every changed spot carries a `// FreeToken patch Pn:` comment.

| Patch | What | Files |
|---|---|---|
| P4 | clampParams filter: clamp numeric params into [min,max] | internal/config/filters.go, internal/server/filters.go, config-schema.json |
| P1 | latest wins: a new pick cancels a colliding not-ready swap (409 model_superseded) | internal/config/config.go, internal/router/scheduler/{scheduler.go,fifo.go}, internal/router/base.go, internal/process/process_command.go, internal/swaputil/superseded.go, config-schema.json |
| P2 | memory gate: wait for Windows free RAM - ramNeedGB >= floorGB before loading (503 not_enough_memory); optional `memoryGate.helperURL` bypass when the settings helper runs a FreeToken llama-swap did not start; probe cmd has WaitDelay; probe failure logs Warn, a cancelled probe returns ctx.Err() | internal/memgate/*, internal/config/{config.go,model_config.go}, internal/router/base.go, config-schema.json |
| P5 | selective reload, part 1: the group router reconfigures in place (`PrepareReconfigure` -> `ReconfigPlan.Commit/Abort`); unchanged loaded models keep their process and in-flight requests, changed and removed models are stopped through `OnUnload`; each process gets its own child context of procCtx; swaps work from the table captured at `StartSwap`; matrix router has no planner factory and keeps upstream's full rebuild | internal/router/{reconfigure.go,base.go,group.go}, internal/router/scheduler/{scheduler.go,fifo.go} |
| P5 | selective reload, part 2: a config reload reconfigures the local router in place (unchanged entries keep their process and requests; changed/removed ones are stopped via OnUnload; matrix or router-kind changes rebuild as upstream) through `server.Rebuild`; the retired Server shuts down everything but the kept router (`ShutdownExceptLocal`); a stale plan is refused (`ErrStaleReconfigure`); reloads coalesce instead of being dropped; `--check-config` (= `-validate`); `GET /api/config/hash` | llama-swap.go, freetoken_reload.go, internal/router/{base.go,reconfigure.go}, internal/server/{server.go,freetoken_api.go} |
| P6 | `POST /api/models/load/{model}`: load through the scheduler (P1/P2 apply), answer when ready or failed (200 `{"model","state":"ready"}`, 409 model_superseded, 503 not_enough_memory, 404 unknown or not local) | internal/router/load.go, internal/server/{server.go,freetoken_api.go} |

Notes (final review fixes, 2026-09-24):

- P1/P2: `FIFO.OnUnload` (internal/router/scheduler/fifo.go) now calls `CancelSwap` for every
  unloaded in-flight swap. Upstream left the swap goroutine running; with the P2 gate that let a
  swap parked in the memory wait boot its model after the unload, even beside a newer pick in the
  same exclusive group. Covered by TestBase_MemGate_UnloadDuringWaitNeverStarts and
  TestBase_MemGate_UnloadThenPickOtherOnlyOtherRuns (internal/router/memgate_test.go).
- P1: `supersede` carries a comment that a victim turning ready before its stop is still stopped.
- P2: `memoryGate.helperURL` parsing is covered by internal/config/memgate_config_test.go (new file);
  the bypass, the Warn-level probe failure, the cancelled-probe ctx.Err() and the probe WaitDelay by
  internal/memgate/memgate_test.go.

Notes (P5 part 1, 2026-09-24):

- P5: `Scheduler` gained `OnReconfigure(conf, planner)`; `FIFO` swaps its config, planner and
  concurrency limits and drains the queue, keeping `active`, `queued`, `reserved` and `inFlight`,
  so a reload never lets a pick evict a kept model mid-answer
  (TestReconfigure_RequestInFlightOnKeptModelFinishes, internal/router/reconfigure_test.go).
- P5: `baseRouter.config`, `processes` and `memGate` are replaced whole by the run loop under
  `stateMu`; readers off the run loop (`Handles`, `ProcessLogger`, `RunningModels`, `Unload`,
  `ServeHTTP`, the timeout helpers) use `snapshot()`. The P2 gate construction moved into
  `newMemGate` unchanged. No upstream test changed.

Notes (P5 part 2 and P6, 2026-09-24):

- P5: `server.New` is split; everything after building the local router moved unchanged into
  `newWithLocal`, which `Rebuild` calls with the kept router. `Shutdown` skips the local router
  when `keepLocal` is set (`ShutdownExceptLocal`).
- P5: `baseRouter.generation` counts committed reconfigures. `PrepareReconfigure` records it;
  `applyReconfig` refuses a plan from an older generation (the plan's new processes are
  cancelled, `ReconfigPlan.Commit` returns `ErrStaleReconfigure`, the table is untouched)
  instead of dropping the earlier commit's processes
  (TestReconfigure_StalePlanIsRefused, internal/router/reconfigure_test.go). `Commit` now
  returns an error (also when the router has shut down); `Rebuild` then returns it and the
  caller keeps the old Server.
- P5: upstream's reload guard in llama-swap.go dropped a reload asked for while one ran.
  `reloadCoalescer` (freetoken_reload.go) folds any number of such requests into exactly one
  more reload after the running one (TestReloadCoalescer_RequestDuringAReloadRunsOnceMore), so
  a config save landing mid-reload is not lost. The reload is the only caller of
  `server.Rebuild`, so prepare+commit pairs never overlap; `--check-config` never builds a
  Server. The config hash is read before the config is loaded, so a write after the read
  shows up as a stale hash plus another reload, never as a new hash on an old config.
- P5 (fix round 1): a SIGTERM during a reload could orphan kept model processes: shutdown
  read the old Server, the reload then retired it with `ShutdownExceptLocal` (winning
  Shutdown's once-guard), and nobody shut the shared local router down; processes use Setpgid
  without Pdeathsig, so they outlived llama-swap (reviewer reproduced it, SIGTERM 2.5 s into a
  reload). The SIGINT/SIGTERM path now calls `reloadCoalescer.stop()` (refuse later reloads, drop
  a pending one, wait for the running one) before reading the active Server, and exits through
  `Server.ShutdownWithLocal`, which also stops a local router the Server had handed on
  (TestReloadCoalescer_StopWaitsDropsAndRefuses, TestServer_ShutdownWithLocal).
- P6 (fix round 1): `Load` has no "already ready" shortcut; a ready model goes through the
  scheduler's own fast path like a chat request, so P1 supersede applies identically.
