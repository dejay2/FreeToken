# Local Agent Prefix Cache Implementation Plan

> **For agentic workers:** Use test-driven development and scoped code review for each implementation boundary. Track execution here; keep changes in the isolated feature worktree.

**Goal:** Let agents on one local FreeToken server prepare one exact common prefix and share its attention pages while keeping mutable state private.

**Architecture:** A scheduler-owned preset registry coordinates one internal prefill-only job per exact registered prefix. Its final, settled endpoint donates a complete hybrid checkpoint to the existing radix cache and local parking store; ordinary requests then use normal cache matching and admission. An additive control API registers canonical prompts, inspects status, requests warming, and deletes presets. Preparation yields between chunks and reserves its unfinished capacity while unrelated work runs.

**Tech stack:** Existing Python/PyTorch scheduler, FastAPI, dataclass message queues, pytest.

**Spec:** `docs/research/shared-agent-prefix-cache-2026-09-14.md`, steps 1–3. Jay approved proceeding after fixing the permanent local-only scope.

## Global constraints

- One local server; no distributed storage or multi-server phase.
- Initial support: text-only QSA/GDN hybrid radix, TP1. Return an explicit unsupported response for other pool families/TP; preserve their generation behavior.
- Existing clients send complete requests. No prompt reordering, fabricated chat turns, approximate state reuse or required client protocol change.
- Prefix identity uses exact rendered token IDs; aliases never bypass validation. Runtime registries belong to the current engine lifetime.
- An internal preparation produces no generated tokens and never enters decode or client usage accounting.
- Intermediate chunks remain unpublished. Publish only after the final internal job has drained, preserving overlap safety.
- Keep existing RAM/SSD behavior and checksums. New durable aliases/combined tiers are outside this implementation.
- Existing cache rebuild releases preference leases before replacing pools; active request ownership is unchanged.
- Registry limits: 64 aliases and 16 MiB of unique int32 token storage. GPU preference retention is bounded and releasable under admission pressure; its configurable limit is returned by the control API.
- No deployment or restart of the serving engine as part of local development. Run CPU tests and provide a reproducible live acceptance benchmark.

## File boundaries and interfaces

`scheduler/prefixes.py` owns `PrefixCoordinator`, exact-token `PrefixEntry` records, aliases, in-flight preparation ownership, waiting requests, retention and metrics. `PrefillManager` owns pending runnable work; `Scheduler` owns final checkpoint publication and control-message dispatch. `CacheManager` remains the only authority for page/state allocation, matching and locks.

`message/{backend,tokenizer,frontend}.py`, `tokenizer/server.py`, `server/prefix_api.py` and narrow `api_server.py` hooks implement transport and HTTP. The HTTP/tokenizer implementation does not allocate GPU memory or own prefix leases.

Wire contract:

```python
PrefixCacheMsg(request_id: str, action: str, name: str = "",
               text: str | list[dict] | None = None,
               tools: list[dict] | None = None,
               chat_template_kwargs: dict | None = None,
               preserve_system_order: bool = False,
               prefix_tokens: int | None = None, ttl_seconds: float = 300.0,
               max_retained_bytes: int | None = None)
PrefixCacheBackendMsg(request_id: str, action: str, name: str = "",
                      input_ids: torch.Tensor | None = None,
                      prefix_tokens: int | None = None, ttl_seconds: float = 300.0,
                      max_retained_bytes: int | None = None)
PrefixCacheResultMsg(request_id: str, status: str, result: dict,
                     error: str | None = None)
PrefixCacheReply(request_id: str, status: str, result: dict,
                 error: str | None = None)
```

Actions: `register`, `list`, `get`, `warm`, `delete`, `configure`. Status: `ok`, `warming`, `invalid`, `not_found`, `unsupported`, `busy`, `failed`. Warm returns asynchronously; get/list returns current readiness, physical residency, counters and last failure. No readiness claim comes from an alias alone.

## Task 1: Reproduce cold fanout and build registry ownership

Files: create `tests/scheduler/test_prefix_coordinator.py`, `python/freetoken/scheduler/prefixes.py`; extend existing CPU hybrid cache fixture patterns.

- [x] Write tests that register a 16-token base with four-token pages, offer four 17+-token requests, and assert only one preparation is runnable while all four wait. Verify a different token prefix proceeds, alias dedup uses one entry, and a private request bypasses sharing.
- [x] Run the new tests and verify the missing coordinator fails.
- [x] Implement exact int32 token storage/hash verification, page-aligned registration, alias limits, token-memory limit, idempotent registration, name conflict checks, cancellation of individual waiters, bounded preferred retention, expiry and failure fallback. One preparation UID is allocated from an internal negative range; reject external negative generation UIDs before any collision is possible.
- [x] Check duplicate aliases and repeated warm/delete/expire sequences do not leak entries or leases. Use the real CPU radix and state pool when checking byte/page ownership.

Interfaces used by scheduling:

```python
coordinator.route(pending: PendingReq) -> bool  # True means held; may enqueue one preparation
coordinator.command(msg: PrefixCacheBackendMsg) -> dict
coordinator.before_admit(pending: PendingReq) -> bool  # complete an exact resident/restored warm hit without a forward
coordinator.completed(req: Req) -> None  # called only after the final forward drains and resources are committed
coordinator.failed(uid: int, error: str) -> bool  # recognizes internal jobs and releases ordinary waiters
coordinator.cancel_waiter(uid: int) -> None
coordinator.release_preferences() -> bool  # whether a retry can benefit
coordinator.before_rebuild() -> None
```

## Task 2: Add a real prefill-only scheduler path

Files: `core.py`, `scheduler/{utils,prefill,scheduler,cache}.py`, `engine/engine.py`; tests in `tests/scheduler/test_prefix_coordinator.py` and `tests/scheduler/test_kv_park_checkpoint.py`.

- [x] Tests: an internal request advances its cached length without extending output; intermediate internal chunks retain ownership and never publish; a final internal request never emits `DetokenizeMsg` or `PromptAdmittedMsg`; sampling/MTP observation is not run for an internal-only batch.
- [x] Add `prefill_only` and `prefix_key` metadata to pending/runtime requests. Internal batches are homogeneous so the engine can skip sampling and token-pool output writes without changing ordinary sampling behavior.
- [x] Integrate registry routing at new-request admission. Keep held followers outside the runnable list so unrelated requests can progress; internal preparation chunks run through existing chunk scheduling. Detect an exact restored hit before constructing a runtime request with zero remaining input.
- [x] At final drain, discard the unused intermediate track mark, donate the exact live state through the existing finish path, retain a locked handle, save that exact checkpoint through the existing store when enabled, then return followers to ordinary admission.
- [x] On admission pressure release optional preference leases and retry once; preserve active locks. On cancellation release only the cancelled follower. Preparation failure routes its followers through ordinary cold admission once instead of retry-looping.
- [x] Before an idle cache rebuild, release registry GPU handles so parking can save eligible leaves; invalidate readiness and allow ordinary restore/re-preparation afterward.
- [x] Verify four requests use the same common page indices and different live state slots, and an unrelated request can make progress while a prefix is being prepared.

## Task 3: Add preset management and transport

Files: `message/{backend,tokenizer,frontend,__init__}.py`, `tokenizer/server.py`, `server/{prefix_api,api_server}.py`; tests in `tests/server/test_prefix_api.py`, `tests/tokenizer/test_prefix_control_messages.py`.

- [x] Add failing tests for register/list/get/warm/delete/configure; correlated response/error/timeout/cancellation cleanup; unsupported model status; and tokenizer failure reporting.
- [x] Register API under `/v1/cache/prefixes`. Registration body contains `name`, `format` (`openai` or `anthropic`), `request` (existing canonical request schema), `prefix_tokens`, `ttl_seconds`. Convert using existing protocol-to-GenSpec helpers; reject non-text inputs before tokenization and send the ordinary rendering inputs to a tokenizer worker.
- [x] Tokenizer worker runs `TokenizeManager.tokenize` on registration input and sends the resulting tensor in the backend message. All other actions are passthrough. Conversion failures return correlated errors instead of hanging.
- [x] Add one frontend future map with dispatch timeout/finally cleanup and backend-death resolution. Warm answers when queued; callers poll get for completion. Preserve normal readiness gates and access behavior and generation behavior.
- [x] Bind scheduler control dispatch to the coordinator, returning explicit invalid/not-found/unsupported/capacity errors without crashing the scheduler.
- [x] Verify dataclass encode/decode round trips and the normal Anthropic/OpenAI rendering path, including later system messages and tool schemas.

## Task 4: Record physical sharing, add live benchmark, document usage

Files: `scripts/bench/agent_prefix_live.py`, `README.md`; tests in the preceding suites.

- [x] Extend status with registered/aligned length, actual GPU/RAM residency, unique physical attention bytes, private-state cost, retained bytes, preparations, restored tokens, and follower counts. Do not infer physical savings from API logical usage.
- [x] Write a benchmark that registers a prefix, launches 1/2/4/8 ordinary agent-shaped requests and records TTFT, actual forwarded work, cache hits and server identity. Support a cold baseline and warm run without silently changing server settings or clearing an active server's cache.
- [x] Document exact registration/warm/status commands, page-boundary leftovers, limits, invalidation, the process-local registry, supported model path, and the existing RAM flags. Keep local durability follow-up separate.
- [x] Run focused CPU cache/scheduler/transport suites plus relevant existing park/rebuild/abort/accounting tests; compare failures against baseline.
- [x] Request an independent review of ownership, overlap, cancellation and protocol compatibility; resolve substantive findings, rerun affected tests, and record the live-GPU validation limit candidly.

## Execution record

- Isolated branch: `feat/local-agent-prefix-cache`, based on `b793433`.
- Existing dynamic-KV worktree is independent and untouched.
- User approval: “carry on then” after the local-only scope correction.


### Implementation and review results

- Implemented exact CPU token identities, bounded aliases/source storage, one job per cold prefix,
  final-drain checkpoint publication, GPU reference sharing, RAM/SSD restore, cancellation,
  pressure/TTL release, and rebuild invalidation. Preparation never samples or emits tokens.
- API/worker work uses the normal OpenAI/Anthropic converters and tokenizer. Timeout is HTTP 504;
  pending futures resolve or clean up on reply, timeout, cancellation, send failure and backend death.
  Nested images in Anthropic tool results are rejected before conversion.
- The fairness review led to preparation quanta yielding to queued ordinary requests and active
  decoding. Unfinished preparation capacity stays reserved, and blocked new arrivals cannot
  prevent an owning continuation from finishing.
- Status inspection leaves GPU eviction timestamps and parked-store hit counters unchanged.
  Retention accounts for the union of held parent/child pages and snapshots; newly inserted
  ancestor state cannot silently exceed the preference budget. Table/token-budget failures do
  not clear TTLs. Actual KV/state allocation can revoke preferences acquired after admission.
- Two bounded independent code reviews covered scheduler ownership/overlap and API/transport.
  Concrete findings were reproduced and fixed. Follow-up scheduler inspection found the
  fairness, pressure-reason and non-mutating-inspection concerns addressed.
- New scheduler/engine behavior is consolidated in `tests/scheduler/test_prefix_coordinator.py`;
  real CPU RAM/SSD byte restoration extends `tests/scheduler/test_kv_park_checkpoint.py`.
  The review regressions include preparation-first/unrelated-second, decode progress, blocked
  request tables, unfinished-capacity reservation, retained ancestor growth and allocation-time
  preference release. The existing diagnostic request stub gained the new default request flag.

### Verification environment and limits

Use the existing CPU environment without installing or changing dependencies:

```bash
PATH=/home/jay/projects/FreeToken/.venv/bin:$PATH PYTHONPATH="$PWD/python" \
  /home/jay/projects/FreeToken/.venv/bin/python -m pytest \
  tests/scheduler tests/server tests/tokenizer tests/kvcache -q --tb=short
```

- Focused cache/scheduler/parking/radix validation: **206 passed, 1 skipped**.
- Final feature broad suite: **1,431 passed, 108 failed, 13 skipped**. The failure test IDs
  are identical to baseline: **zero additional failures**. The 54 added tests pass.
- The unchanged `b793433` checkout's broad suite: **1,377 passed, 108 failed, 13 skipped**.
  Of these failures, 107 require an unavailable CUDA pinned-memory allocator; the remaining
  existing CLI test expects `ple_backend=pinned` while this Linux environment selects `disk`.
- Engine MTP/shadow/draft/rebuild validation: **110 passed, 6 failed, 13 skipped**, with exactly
  the same six pinned-memory failures reproduced on the unchanged checkout.
- The benchmark's command-line help, canonical converter, common-prefix helper and Python
  compilation were checked locally. No GPU server benchmark was run. No live server settings,
  cache contents, or process lifecycle were changed. The live acceptance command and its
  explicit incomplete-report checks are documented in `docs/local-agent-prefix-cache.md`.
- `git diff --check` and Python compilation pass. Ruff is absent from the existing environment;
  no lint result is claimed.

The implementation is reviewable on the isolated feature branch. GPU kernel correctness,
real-model output parity, TTFT improvements and throughput remain live acceptance work. This
CPU-validated implementation does not claim those measurements or restart the user's engine.


### Draft PR integration with the current target branch

Integrated `origin/mtp-upstream-merge` at `8d3b5c3` (dynamic KV pool) before publishing the
feature branch. Both features retain their admission hooks, idle timers and rebuild guards.
Three integration regressions were reproduced before their fixes:

- A failed rebuild now answers prefix-held agents exactly once, drops CPU ownership without
  unlocking torn-down device pools, and rejects late prefix-control messages.
- Successful pool resizing refreshes the registry's current context limit. Registration still
  requires the prefix to fit today's pool; a warm command does not request dynamic growth.
- Moving a cold agent into the shared preparation's waiters clears its provisional dynamic
  admission charge. Later agents can join that preparation, and private tails are reserved on
  actual admission. The regression seats two agents together where duplicate cold estimates
  would otherwise put the second into the dynamic controller's held FIFO.

Focused prefix/API/transport and dynamic-pool tests: **143 passed**. There are now **57 added
feature tests**, including the three integration regressions. Python compilation and
`git diff --check` pass. GPU acceptance remains outstanding; the server has not been deployed
or restarted.

A broad comparison also exposed a timing-sensitive existing parking test:
`test_idle_park_frees_only_after_the_copy_and_restore_is_byte_identical`. Its assertion assumes
that the copy has not completed before `park_idle` returns, although `_park_candidate` drains
completed copies immediately. It passed in isolation on both branches. Forcing the copy to
complete before `ParkStore.offer` returns reproduced the same assertion on unchanged `8d3b5c3`.
The test was left unchanged; this timing sensitivity is disclosed in the PR.

Final merged-head comparison (same CPU environment and commands):

```bash
PATH=/home/jay/projects/FreeToken/.venv/bin:$PATH PYTHONPATH="$PWD/python" \
  /home/jay/projects/FreeToken/.venv/bin/python -m pytest -q \
  tests/scheduler tests/server tests/tokenizer tests/kvcache \
  tests/engine/test_kv_dynamic_engine.py tests/engine/test_spec_rearm_after_rebuild.py
```

- Feature: **1,550 passed, 108 failed, 13 skipped**.
- Unchanged target `8d3b5c3`: **1,493 passed, 108 failed, 13 skipped**.
- Exact failure-ID sets match: **zero additional failures**, **57 additional passes**.
  The 108 failures comprise the same 107 CPU pinned-memory failures and existing Linux CLI
  default expectation described above. The initial parking timing failure did not recur in
  either complete rerun; it is still disclosed rather than hidden by the passing reruns.


### PR #6 review corrections

Three review findings were reproduced against `1191a69` before changing production code:

- Full, page-aligned rendered prompts now retain a non-empty generation tail. Registration
  clamps the requested cut before page alignment, leaving the final page private when needed.
  A prompt that cannot leave both a cached page and a tail is rejected clearly. Regression
  cases cover both omitted and explicit full-length cuts, one preparation for four matching
  agents, shared attention pages and separate recurrent state.
- The original requested length is stored per alias. Register/get/list, updating one alias,
  deleting and re-registering it all preserve the other alias's metadata and one shared source.
- Expected `ValueError` and Jinja `TemplateError` exceptions return `invalid` (HTTP 400).
  Unexpected encoder failures remain `failed` (HTTP 500). A real Transformers/Jinja renderer
  exercises the HTTP route with rejected role ordering, alongside correlated worker replies.

The regressions failed for the reviewed causes before the fixes and pass afterward. The
focused prefix/API/transport and dynamic-pool suite now has **150 passing tests**; the feature
adds **64 tests** compared with target `8d3b5c3`. Compilation and `git diff --check` pass.
Live Qwen3.8 Flash correctness and performance on the RTX 5090 remain a merge requirement;
this update does not deploy or restart the server.

Final review-update validation, using the same broad command above:
**1,557 passed, 108 failed, 13 skipped**. The failure-ID set exactly matches the recorded
unchanged `8d3b5c3` baseline (**1,493 passed, 108 failed, 13 skipped**): zero additional
failures and 64 additional passes. The focused suite has 150 passes and the checkpoint suite
has 19 passes. The RAM/SSD fixtures now include explicit prompt tails while preserving their
original checkpoint sizes and all byte-restoration assertions. Independent inspection found
no further defects in the three fixes.
