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

## Our patches

Every changed spot carries a `// FreeToken patch Pn:` comment.

| Patch | What | Files |
|---|---|---|
| P4 | clampParams filter: clamp numeric params into [min,max] | internal/config/filters.go, internal/server/filters.go, config-schema.json |
| P1 | latest wins: a new pick cancels a colliding not-ready swap (409 model_superseded) | internal/config/config.go, internal/router/scheduler/{scheduler.go,fifo.go}, internal/router/base.go, internal/process/process_command.go, internal/swaputil/superseded.go, config-schema.json |
| P2 | memory gate: wait for Windows free RAM - ramNeedGB >= floorGB before loading (503 not_enough_memory) | internal/memgate/*, internal/config/{config.go,model_config.go}, internal/router/base.go, config-schema.json |
