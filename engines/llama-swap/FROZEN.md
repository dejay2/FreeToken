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
- P5, P6, P7, P8: none; upstream router and server tests pass unchanged.

## Our patches

Every changed spot carries a `// FreeToken patch Pn:` comment.

| Patch | What | Files |
|---|---|---|
| P4 | clampParams filter: clamp numeric params into [min,max]; config load refuses entries it would drop | internal/config/{filters.go,load.go,peer.go}, internal/server/filters.go, config-schema.json |
| P1 | latest wins: a new pick cancels a colliding not-ready swap (409 model_superseded) | internal/config/config.go, internal/router/scheduler/{scheduler.go,fifo.go}, internal/router/base.go, internal/process/process_command.go, internal/swaputil/superseded.go, config-schema.json |
| P2 | memory gate: wait for Windows free RAM - ramNeedGB >= floorGB before loading (503 not_enough_memory); optional `memoryGate.helperURL` bypass when the settings helper runs a FreeToken llama-swap did not start; probe cmd has WaitDelay; probe failure logs Warn, a cancelled probe returns ctx.Err() | internal/memgate/*, internal/config/{config.go,model_config.go,load.go}, internal/router/{base.go,reconfigure.go}, config-schema.json |
| P5 | selective reload, part 1: the group router reconfigures in place (`PrepareReconfigure` -> `ReconfigPlan.Commit/Abort`); unchanged loaded models keep their process and in-flight requests, changed and removed models are stopped through `OnUnload`; each process gets its own child context of procCtx; swaps work from the table captured at `StartSwap`; matrix router has no planner factory and keeps upstream's full rebuild | internal/router/{reconfigure.go,base.go,group.go}, internal/router/scheduler/{scheduler.go,fifo.go} |
| P5 | selective reload, part 2: a config reload reconfigures the local router in place (unchanged entries keep their process and requests; changed/removed ones are stopped via OnUnload; matrix or router-kind changes rebuild as upstream) through `server.Rebuild`; the retired Server shuts down everything but the kept router (`ShutdownExceptLocal`); a stale plan is refused (`ErrStaleReconfigure`); reloads coalesce instead of being dropped; `--check-config` (= `-validate`); `GET /api/config/hash` | llama-swap.go, freetoken_reload.go, internal/router/{base.go,reconfigure.go}, internal/server/{server.go,freetoken_api.go} |
| P6 | `POST /api/models/load/{model}`: load through the scheduler (P1/P2 apply), answer when ready or failed (200 `{"model","state":"ready"}`, 409 model_superseded, 503 not_enough_memory, 404 unknown or not local) | internal/router/load.go, internal/server/{server.go,freetoken_api.go} |
| P7 | unload only if idle: `POST /api/models/unload/{model}?ifIdle=1` stops the model only when the scheduler holds no request for it (in flight, queued, waiting on a swap, or a swap to it running), else 409 code `busy` and nothing stops; the check and the stop run in one run-loop step, atomic with admission; a request still before the run loop is not seen (it reloads the model after the stop instead of being killed); a router without the check answers 501; plain unload unchanged | internal/router/{unload_idle.go,base.go}, internal/router/scheduler/fifo.go, internal/server/apigroup.go |
| P8 | `POST /api/models/load/{model}?ifFree=1`: load only when nothing else is on the card or on its way there; refused at admission on the run loop (409 code `busy`, nothing admitted or cancelled) when another model is running, being swapped in (a swap parked in the memory gate counts, though it has no process state yet), queued or holding requests; the target itself does not count; a router without it answers 501. P7's `Busy` no longer counts a swap whose waiters have all gone (a P6 load whose caller cancelled), so an if-idle unload can stop it. Round 2: a P6 load's cancel aborts its swap (gate wait included) and stops its process when no other waiter is left (`HandlerReq.AbortSwapIfLast`); a chat request with header `X-FreeToken-If-Free: 1` gets the same if-free admission (409 `busy` instead of superseding); the header is removed before the request reaches the engine | internal/router/{load.go,base.go}, internal/router/scheduler/{scheduler.go,fifo.go,if_free.go}, internal/server/freetoken_api.go |

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

Notes (tidy of deferred review minors, 2026-09-25):

- P4: `clampParams` entries with an empty or protected key, not exactly two numbers, a NaN
  bound (YAML `.nan`: every comparison with NaN is false, so it clamped nothing) or min > max
  are now a config load error (`Filters.ValidateClampParams`, called from load.go), so
  `--check-config` and the control panel see them instead of the filter dropping them silently.
  `.inf`/`-.inf` bounds are allowed and mean no limit on that side. Keys are not checked
  against a list of known request parameters: like setParams/stripParams a key is any gjson
  path, and ours are generated from the panel's NInfer dial table. A JSON value cannot be
  NaN; an out-of-range literal (`1e999`) parses to +-Inf and is clamped to the finite bound;
  the int conversion skips infinite bounds. `SanitizedClampParams` drops NaN bounds too.
  Tests: TestLoadConfig_ClampParamsValidation, TestFilters_SanitizedClampParams,
  TestApplyFilters_ClampParamsInfAndNaN.
- P2: `memoryGate.floorGB` is now `*float64`: unset = 6 (`config.DefaultFloorGB`), an explicit
  0 = no cushion (it used to become 6 silently), negative or NaN is a load error
  (TestLoadConfig_MemoryGateFloorGB).
- P2: TestBase_MemGate_ReservationReleased (internal/router/memgate_test.go) covers the FIFO
  concurrency reservation being handed back after a gate 503, a start failure after the gate,
  and a client giving up while parked in the gate (concurrencyLimit 1, so a leak shows up
  as a 429 on the next request; checked by removing the releases, all three fail).
- P1: after latest-wins cancels a swap, `OnRequest` now drains the queue once the new request
  is placed. Cancelled swaps send no SwapDone, so a request queued only behind the victim (a
  shared eviction target) waited for some unrelated later event
  (TestFIFO_LatestWins_SupersedeDrainsQueuePromptly; fails without the drain).
- P1, left as is: `supersede` stops its victims synchronously on the run loop, so a slow stop
  holds up other scheduling for up to the victim's unloadTimeout (NInfer 30 s, FreeToken
  180 s; the FreeToken adapter's stop goes through the helper). Not made asynchronous
  because the synchronous stop is what orders it before any later decision: with a
  background stop, a quick re-pick of the victim (A -> B -> A) starts a fresh swap for A
  whose EnsureReady can then be killed by the stale stop, and the next pick's memory gate
  would probe while the victim still holds its RAM. Victims are loads that have not become
  ready, and several are stopped in parallel (`StopProcesses`), so the cost is one stop.
  Making it asynchronous needs a per-process stop generation; not worth it for this box.
- P5: a reload whose config differs only in comments keeps every model: kept/changed is
  `reflect.DeepEqual` on the parsed `ModelConfig`, which carries nothing from comments, and the
  hash served by `/api/config/hash` only tells the panel which file is live. Could not
  reproduce the reported "header-only change stopped all models" with the generated config
  (TestReconfigure_HeaderOnlyChangeKeepsEveryModel, loaded from YAML through NewGroup and
  PrepareReconfigure). The stale example-config comment saying a save stops the loaded model
  (pre-P5 behaviour) is corrected.
- P5: `${PORT}` is allocated from startPort in sorted model-ID order at every load, so an
  unchanged model set gives every model the same port and a `${PORT}` model is kept. Adding
  or removing a model that sorts before it moves its port; that entry then really differs
  (cmd and proxy name another port, and its old port goes to another model), so it is
  restarted. Our generated config uses fixed ports (2020, 8090) and no `${PORT}`
  (TestReconfigure_PortMacroUnchangedModelIsKept).
- P5, left as is: a SIGTERM while a reload is blocked for longer than the backstop
  (shutdownTimeout + 5 s = 35 s) ends in `os.Exit(1)` before the router stops its processes.
  On the box the unit (scripts/engines/install-service.sh) has `KillMode=mixed`, so when the
  main process exits systemd SIGKILLs everything left in the unit's cgroup: NInfer
  (`ninfer-serve` is exec'd by the adapter) goes with it. FreeToken's server runs in the
  `freetoken-settings` unit, not here, so it can outlive a killed `freetoken.sh`; the next
  FreeToken load adopts it (same model and profile) or stops it through the helper, and the
  next NInfer load stops it through the helper and kills a stray `ninfer-serve` by name.
  Outside systemd the same adapters converge on the next load. Not fixed in Go: the forced
  exit would have to find and kill each process group itself.

Notes (P7, 2026-09-25):

- P7: added for the control panel's Test tab, which puts models away between its steps. Upstream
  Unload kills in-flight requests (base.go Unload comment); a read of the in-flight list followed
  by a plain unload left a window in which another app's new request could be admitted and then
  killed. `FIFO.Busy` counts `inFlight`, `reserved` and `active`; the unload request carries
  `ifIdle` and the run loop checks it just before `OnUnload`. Covered by
  internal/router/unload_idle_test.go and internal/server/freetoken_unload_idle_test.go.

Notes (P8 round 2, 2026-09-25, Codex review of PR #18):

- P8: the Test tab's Stop cancels its own if-free load. A load parked in the memory gate has no
  process state, so it is absent from /running; upstream `OnCancel` removed the waiter but kept
  the swap, which then refused every later if-free load (the Test tab's own put-back) and booted
  its model whenever room appeared. `FIFO.OnCancel` now aborts a swap left with no waiters when
  the cancelled request was a load (`AbortSwapIfLast`, set only by `baseRouter.load`): the swap
  is removed from `active`, `CancelSwap` ends the gate wait or `EnsureReady`, and the process is
  stopped (blocking, as `supersede` does). A chat request keeps upstream's behaviour (the swap
  completes on its own). Covered by TestLoad_CancelledWhileGatedIsAborted and
  TestLoad_CancelKeepsASwapAnotherRequestJoined (internal/router/if_free_round2_test.go).
- P8: the Test tab's warm-up and measured chats went through ordinary routing and could
  supersede (P1) another app's load. `ServeHTTP` now reads `X-FreeToken-If-Free: 1` into
  `HandlerReq.IfFree`, so the refusal is decided on the run loop at admission exactly as for
  `?ifFree=1`. Covered by TestChatIfFree_* in the same file.
