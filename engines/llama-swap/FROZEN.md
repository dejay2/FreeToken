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

- P1: nine upstream `TestFIFO_*` tests in internal/router/scheduler/fifo_test.go expect a request
  for another model to queue behind a not-ready in-flight swap. Latest wins cancels that swap
  instead, so each of those tests now builds its FIFO with `config.FifoConfig{LatestWins: boolPtr(false)}`
  and nothing else in them changed: QueueOnEvictionCollision, OverlappingEvictSetsDoNotRunInParallel,
  QueueDrainPromotesMultiple, QueueCollation, OnUnload_DropsQueuedRequests, PriorityQueueOrder,
  OnCancel_QueuedRequest, ConcurrencyLimit_QueuedWaitersReserveCapacity,
  ConcurrencyLimit_CancelledQueuedWaiterReleasesReservation.
  The same applies to two router tests, which now set
  `conf.Routing.Scheduler.Settings.Fifo.LatestWins` to false:
  TestGroup_SameGroupSwapSerialises (internal/router/group_test.go) and
  TestMatrix_IncompatibleQueues (internal/router/matrix_test.go).

## Our patches

Every changed spot carries a `// FreeToken patch Pn:` comment.

| Patch | What | Files |
|---|---|---|
| P4 | clampParams filter: clamp numeric params into [min,max] | internal/config/filters.go, internal/server/filters.go, config-schema.json |
| P1 | latest wins: a new pick cancels a colliding not-ready swap (409 model_superseded) | internal/config/config.go, internal/router/scheduler/{scheduler.go,fifo.go}, internal/router/base.go, internal/swaputil/superseded.go, config-schema.json |
