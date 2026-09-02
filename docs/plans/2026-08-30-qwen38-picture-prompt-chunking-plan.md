# Qwen3.8 Chunked Picture-Bearing Prompt Implementation Plan

**Goal:** Let normal pi and direct picture-bearing prompts span repeated prompt-loading steps of at most 8,192 tokens, up to the model's available combined context, without shrinking pictures or weakening picture/cache correctness.

**Design source:** `docs/design/2026-08-30-qwen38-picture-prompt-chunking-design.md`

**Completion state:** Scheduler implementation is accepted at `56a34ec`; direct 33,243-token and user-run normal pi picture checks passed. The second tracked acceptance checkpoint is being recorded separately. The final pi text-after-picture and restored language-speed gates are intentionally carried into `docs/plans/2026-08-30-qwen38-vision-layer-streaming-plan.md` after the all-GPU picture placement measured only 7.00 output tokens/second.

**Context sources:** `CONTEXT.md`, `CONTEXT-MAP.md`

**ADR sources:** `docs/adr/0001-desktop-assisted-native-windows-fork.md`

**Architecture:** Prepare each request's complete picture features and Qwen three-axis positions once. Retain them only on the private pending request. Reuse the existing long-text continuation machinery, and for each scheduled token range select exactly the picture-feature rows and three-axis positions belonging to that range. Keep 8,192 as the per-step work budget; remove it as a whole-request rejection limit.

**Global constraints:**

- Work only in `D:\FreeToken-ple-mmap-vision` on local branch `windows-ple-mmap-vision`; keep the rollback worktree `D:\FreeToken-ple-mmap` at `14ee7b0`.
- Do not edit `C:\Users\jay\AppData\Local\FreeToken` or `D:\Models\Qwen3.8-Flash-Next-NVFP4`.
- Do not push, publish, open/edit upstream items, or expose local evidence.
- Keep `127.0.0.1:2020`, one active request, `--ple-backend mmap`, `--moe-backend offload`, serial expert loading, automatic expert-cache sizing, and `--kv-reserve-tokens 262144`.
- Keep each prompt-loading step at or below the live `max_extend_tokens` budget, currently 8,192.
- Keep picture requests private from admission through final cleanup, even after a step releases its local picture-feature reference.
- Preserve exactly 262,144 usable KV tokens after the internal 64-token page.
- Keep pi's default provider, model, and thinking level unchanged.
- If any live graphics-card, server, text-regression, or normal-pi acceptance check fails, restore pi's text-only declaration and start the known-good text server from the rollback worktree.

**Out of scope:**

- Raising one processing step to 262,144 tokens.
- Splitting the picture encoder's pixel work, automatically resizing/cropping pictures, or changing processor pixel limits.
- Video, Anthropic picture input, a shared/public server, dependency changes, Desktop changes, model-file changes, publication, or upstream work.

## Execution setup

Use the existing Desktop Python with checkout-local sources and picture packages:

```powershell
$VisionRoot = 'D:\FreeToken-ple-mmap-vision'
$KnownGood = 'D:\FreeToken-ple-mmap'
$DesktopPython = 'C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe'
$VisionSite = "$VisionRoot\.local\vision-packages"
$ModelPath = 'D:\Models\Qwen3.8-Flash-Next-NVFP4'
$env:PYTHONPATH = "$VisionRoot\scripts\windows-ple-mmap;$VisionSite;$VisionRoot\python"
```

For CPU-only work, set `$env:CUDA_VISIBLE_DEVICES = '-1'` and leave the accepted server running. Remove that value only after CPU checks pass and the current server has been stopped for the graphics-card stage.

## Codebase map

### Product files

- Modify: `python/freetoken/scheduler/scheduler.py`
  - `Scheduler._process_one_msg`: remove the whole-request 8,192-token picture rejection while preserving combined-context checks and one-request error isolation.
  - `Scheduler._gather_multimodal`: derive the current token range's feature-row slice by Qwen picture-placeholder order, concatenate only non-empty current slices, and release each scheduled request object's full-tensor reference.
- Modify: `python/freetoken/scheduler/prefill.py`
  - `PrefillAdder._add_one_req`: allow a private picture request to produce `ChunkedReq` continuations.
  - `PrefillManager.abort_req`: explicitly clear the popped pending request's complete picture features and position table before returning an in-flight continuation for safe draining.
- Modify: `python/freetoken/scheduler/cache.py`
  - `CacheManager.cache_req`, `_cache_req_hybrid`, and `_cache_req_swa`: make unfinished private commits no-ops, keeping the admission handle locked until completion/abort, then unlock and free exactly once without inserting or donating anything.

The following existing state needs no new product field:

- `python/freetoken/scheduler/utils.py::PendingReq` already retains the complete GPU `mm_embeds`, complete CPU `mrope_position_ids`, continuation request, and authoritative `cache_private` marker.
- `python/freetoken/core.py::Req` already carries per-step references plus persistent privacy; `Batch` already carries only the current batch's concatenated picture features.
- `python/freetoken/scheduler/scheduler.py::_make_rope_positions` already selects `[cached_len, device_len)` from complete three-axis prompt positions.
- `python/freetoken/models/qwen4_exp/model.py::_merge_multimodal` already checks current-batch picture-placeholder count against current-batch feature rows.
- QSA pending rings, GatedDeltaNet continuation state, PLE context, and page-table continuation paths already support repeated text steps and have picture-position boundary tests.

### Tests and evidence

- Modify: `tests/scheduler/test_multimodal_admission.py`
  - Replace the old 8,193-token rejection expectation with successful picture preparation/admission up to the combined model context.
  - Retain feature/placeholder mismatch and raw-tensor cleanup checks.
- Create: `tests/scheduler/test_multimodal_chunked_prefill.py`
  - Drive real `PrefillManager` continuation scheduling plus real multimodal gathering over controlled token ranges.
  - Check feature slicing before/across/after boundaries, two-picture row order, empty-feature steps, exact-once prompt accounting, final release, and private radix lifecycle.
- Modify: `tests/scheduler/test_mrope_positions.py`
  - Concatenate positions produced for successive token ranges and compare them with the independently chosen complete three-axis position table.
- Modify: `tests/scheduler/test_abort_inflight_prefill.py`
  - Add picture-feature ownership checks before first scheduling, while a private continuation is in flight, and between steps; retain existing exact-once page/GDN cleanup checks.
- Modify: `tests/scheduler/test_hybrid_cache_manager.py`
  - Prove unfinished private requests do not donate KV/GDN state and completion frees request-owned resources without creating a reusable hit.
- Modify: `tests/scheduler/test_swa_pagesize.py`
  - Prove a multi-step private picture lifecycle frees full/SWA pages and creates no reusable picture prefix.
- Modify: `benchmarks/run_qwen38_vision_smoke.py`
  - Add a direct picture-bearing prompt above 32,768 combined tokens, zero cached-token check, answer check, timing, memory samples, and post-request health.
  - Retain prior sources, errors, two-picture, cross-picture, 32K text, and geometry evidence.
- Create locally/ignored: `.local/evidence/pi-normal-picture-chunking.*`
  - Copy or regenerate the normal 1920×1280 terminal screenshot, run pi with ordinary context/tools enabled, and retain stdout/stderr/exit status without session contents.
- Update after passing evidence only: `docs/design/2026-08-30-qwen38-picture-prompt-chunking-design.md` and this plan's status/checks in the internal rollback worktree.

## Testing strategy

### Highest test seam

The focused behavior seam is the real scheduler path: tokenizer-prepared `UserMsg` admission, `PrefillManager.schedule_next_batch`, `Scheduler._gather_multimodal`, real cache-manager lifecycle, and existing Qwen model batch contract. Expectations come from explicit token positions and independently numbered feature rows, not from the new slicing code's own output.

The public acceptance seam remains `POST /v1/chat/completions` and a real pi `@picture` request.

### Focused CPU command

```powershell
Set-Location $VisionRoot
$env:CUDA_VISIBLE_DEVICES = '-1'
& $DesktopPython -m pytest -q `
  tests/scheduler/test_multimodal_admission.py `
  tests/scheduler/test_multimodal_chunked_prefill.py `
  tests/scheduler/test_mrope_positions.py `
  tests/scheduler/test_abort_inflight_prefill.py `
  tests/scheduler/test_scheduler_chunked_prefill.py `
  tests/scheduler/test_hybrid_cache_manager.py `
  tests/scheduler/test_swa_pagesize.py `
  tests/scheduler/test_cost_accounting_core.py
```

### Relevant CPU regression command

```powershell
Set-Location $VisionRoot
$env:CUDA_VISIBLE_DEVICES = '-1'
& $DesktopPython -m pytest -q `
  tests/server/test_openai_api.py `
  tests/server/test_message_wire.py `
  tests/tokenizer/test_multimodal_input.py `
  tests/scheduler `
  tests/engine/test_graph_mrope.py `
  tests/kvcache/test_qsa_pool.py
& $DesktopPython -m pytest -q tests/models/qwen4_exp `
  -k 'not load_ple_table_concatenates and not load_ple_table_rejects and not read_range_into'
```

The six Linux-only direct-I/O cases remain excluded on Windows for the same reason as the accepted first milestone. GPU cases must report skips during CPU work, not be counted as passing.

### Graphics-card regression command

After stopping the current server and removing `CUDA_VISIBLE_DEVICES`:

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
Set-Location $VisionRoot
& $DesktopPython -m pytest -q `
  tests/models/qwen4_exp/test_mrope.py `
  tests/models/qwen4_exp/test_qsa_kernels.py `
  tests/models/qwen4_exp/test_qsa_backend.py `
  tests/models/qwen4_exp/test_qsa_hf.py `
  tests/models/qwen4_exp/test_qsa_mrope.py `
  tests/engine/test_graph_mrope.py `
  tests/kvcache/test_qsa_pool.py
```

Every required tiled MRoPE, split QSA pending-ring, QSA norm/rotation, and two-value graph-replay test must run rather than skip.

### Live commands

Start the edited branch only after CPU and graphics-card checks pass:

```powershell
& "$VisionRoot\scripts\start-qwen38-flash-next-mmap-windows.ps1" `
  -ModelPath $ModelPath `
  -Port 2020 `
  -ContextTokens 262144 `
  -MaxRunningRequests 1 `
  -EnableVision `
  -EnableCacheReport
```

Then run:

```powershell
& $DesktopPython "$VisionRoot\benchmarks\run_qwen38_vision_smoke.py" `
  --base-url http://127.0.0.1:2020/v1 `
  --model Qwen3.8-Flash-Next-NVFP4
& $DesktopPython "$VisionRoot\benchmarks\run_qwen38_mmap_smoke.py" `
  --base-url http://127.0.0.1:2020/v1 `
  --model Qwen3.8-Flash-Next-NVFP4
```

## Task breakdown

### Task 1: Process private picture prompts through repeated scheduler steps

**Outcome:** A picture-bearing request longer than one 8,192-token step is admitted, picture features and three-axis positions are aligned to each current step, private resources survive continuations, and completion/cancellation releases them without shared reuse.

**Blocked by:** None.

**Files:**

- Modify: `python/freetoken/scheduler/scheduler.py` - remove one-step admission rejection and slice picture features in `_gather_multimodal`.
- Modify: `python/freetoken/scheduler/prefill.py` - allow private `ChunkedReq` creation and explicitly release pending picture state on abort.
- Modify: `python/freetoken/scheduler/cache.py` - defer private-handle unlock/free until terminal cleanup and prohibit all private insertion/donation.
- Modify: `tests/scheduler/test_multimodal_admission.py` - long picture admission and existing failure cleanup.
- Create: `tests/scheduler/test_multimodal_chunked_prefill.py` - repeated-step picture alignment, ordering, accounting, privacy, and final release.
- Modify: `tests/scheduler/test_mrope_positions.py` - complete-table equivalence across successive slices.
- Modify: `tests/scheduler/test_abort_inflight_prefill.py` - picture ownership and exact-once cancellation cleanup.
- Modify: `tests/scheduler/test_hybrid_cache_manager.py` - private hybrid no-donation lifecycle.
- Modify: `tests/scheduler/test_swa_pagesize.py` - private multi-step SWA conservation.

**Interfaces:**

- Consumes: tokenizer-prepared `UserMsg` with complete `input_ids`, raw picture tensors, and sampling allowance; scheduler-produced complete `mm_embeds` and `mrope_position_ids`; existing `PendingReq.chunked_req` continuation.
- Produces: one `ChunkedReq` or final `Req` per scheduled range, with current-batch `Batch.mm_embeds` containing only that range's feature rows and `Batch.rope_positions` containing only that range's three-axis values.
- Preserves: complete prompt usage reported once, `cache_private=True` through cleanup, no shared match/insert/donation, and unchanged text-only requests.

**Acceptance criteria:**

- [ ] An 8,193-token picture-bearing request is prepared and queued rather than rejected; a prompt at or above the combined model context still receives `context_length_exceeded`.
- [ ] For every range `[cached_len, device_len)`, the feature slice begins after the number of picture placeholders before `cached_len` and contains exactly the placeholders in the range.
- [ ] Concatenating feature slices from all steps exactly reproduces independently numbered complete picture features, including a picture range crossing a boundary.
- [ ] A step with no picture placeholders receives `Batch.mm_embeds is None`.
- [ ] Two pictures retain their processor feature-row order when their placeholders occupy different and shared steps.
- [ ] Concatenating all step position slices exactly reproduces the complete three-axis prompt position table.
- [ ] Only the first step reports complete prompt usage; continuations report no duplicate admission or cached prompt tokens.
- [ ] Ordinary, hybrid/GDN, and sliding-window private lifecycles produce no reusable prefix and return their pages/state slots exactly once.
- [ ] Cancellation before scheduling, during an in-flight intermediate step, and between steps releases complete picture features and existing scheduler resources.
- [ ] Existing text chunking, malformed-picture isolation, feature/placeholder validation, and text-only cache reuse remain unchanged.

- [ ] **Step 1: Write the failing tests**
  - Replace `test_scheduler_rejects_picture_prompt_above_one_prefill_batch_before_encoding` with a test that supplies 8,193 valid tokens/placeholders and expects one picture encoding plus queue admission.
  - In `test_multimodal_chunked_prefill.py`, use token ID `99` as the picture placeholder and feature rows containing their own absolute row number. Schedule controlled 8-token steps so expected slices are literal row ranges independent of implementation.
  - Cover placeholder layouts entirely before a boundary, crossing a boundary, entirely after a boundary, and two ordered picture spans.
  - Drive real `PrefillManager` and real cache managers to completion; assert empty cache growth for private requests and exact page/state conservation.
  - Extend abort tests with weak references or explicit field assertions proving the complete GPU feature tensor is no longer owned after each cancellation point.
  - Expected current failures: admission returns the one-step error, `_add_one_req` raises `NotImplementedError`, `_gather_multimodal` supplies all rows to the first step then clears them, and unfinished private cache commits unlock too early.

- [ ] **Step 2: Run the focused tests before implementation**
  - Run the focused CPU command above.
  - Expected: only the newly changed/created chunked-picture cases fail for the four missing behaviors; all existing text, abort, MRoPE, cache, and accounting cases remain green.

- [ ] **Step 3: Implement the smallest complete scheduler change**
  - In `_process_one_msg`, delete only the one-step picture-limit branch; retain the full model-context check before picture encoding and retain request-level picture error replies.
  - In `_add_one_req`, remove only the `cache_private` chunk prohibition. Continue to copy full pending feature/position references into each temporary request.
  - In `_gather_multimodal`, use the configured Qwen `image_token_id`, count placeholders in the complete request prefix and current range, select the corresponding contiguous feature rows, omit zero-row slices, concatenate current slices in request order, and clear every scheduled request object's full `mm_embeds` reference after selection. Keep the pending continuation's complete reference untouched.
  - In `abort_req`, clear the popped pending request's complete `mm_embeds` and `mrope_position_ids` before returning its in-flight continuation.
  - In all three cache modes, return immediately for unfinished private commits without unlocking or donating; on terminal cleanup unlock once, free request-owned pages/state, and insert nothing.
  - Preserve text-only cache commits, picture admission's complete feature-count validation, existing `_make_rope_positions`, QSA rings, GDN continuation fields, PLE context, and prompt accounting.

- [ ] **Step 4: Run the focused tests again**
  - Run the focused CPU command above.
  - Expected: every new boundary/order/privacy/cancellation case and every existing focused case passes; CUDA-dependent tests remain explicitly skipped.

- [ ] **Step 5: Run relevant CPU checks and create the checkpoint**
  - Run the relevant CPU regression commands above, `python -m py_compile` on the three modified product files and new test file, and `git diff --check`.
  - Inspect exact staged paths and the complete staged diff. `.local`, planning files, pi files, logs, generated kernels, and caches must not be staged.
  - Commit only Task 1 tracked files as `feat(scheduler): chunk picture-bearing prompts` after all CPU checks pass.

### Task 2: Prove long picture prompts on the RTX 5090 and in normal pi

**Outcome:** The edited local server retains exact context/mmap geometry, serves a direct picture-bearing prompt above 32K through repeated 8,192-token steps, and handles the previously rejected normal pi screenshot without stripping pi's normal instructions or tools.

**Blocked by:** Task 1 and its CPU checkpoint.

**Files:**

- Modify: `benchmarks/run_qwen38_vision_smoke.py` - direct long-picture acceptance, evidence, timings, and health.
- Create locally/ignored: `.local/evidence/pi-normal-picture-chunking.*` - normal pi command output and exit status.
- Update locally after evidence: `.local/qwen38-vision-acceptance.json` - new long-picture, step, memory, pi, branch, and commit facts.
- Update internally after success: `D:\FreeToken-ple-mmap\docs\design\2026-08-30-qwen38-picture-prompt-chunking-design.md` and this plan's completion state.

**Interfaces:**

- Consumes: Task 1's chunkable private scheduler request, existing vision launcher, OpenAI-compatible chat route, cache-status route, and pi's `@picture` request path.
- Produces: local evidence that direct and ordinary pi picture prompts larger than 8,192 work while text, tools, reasoning, privacy, context, mmap PLE, and rollback remain healthy.

**Acceptance criteria:**

- [ ] Required RTX MRoPE, QSA split-ring, norm/rotation, graph replay, and cache-pool tests run and pass rather than skip.
- [ ] Startup reports picture input enabled, `ple_backend='mmap'`, serial NVFP4 expert loading, all picture weights, 3,727-or-auto-resolved cached experts, and no missing/unexpected key.
- [ ] `/v1/cache/status` reports 4,097 pages of 64 tokens, 262,208 total allocated tokens, and 262,144 usable tokens.
- [ ] A direct picture-bearing request above 32,768 combined tokens succeeds, returns its fixture code, reports zero cached prompt tokens, and produces at least five prompt-loading log entries when counting the final partial step.
- [ ] The previous source, two-picture, cross-picture, malformed-input recovery, 32K text, reasoning, streaming, tool, usage, and final-health checks pass.
- [ ] Peak whole-GPU and whole-system memory during the long picture request are recorded and the server remains healthy.
- [ ] A real pi request using the normal loaded context, skills, prompt templates, extensions, and tools reads the 1920×1280 terminal screenshot successfully; the command does not use `--no-context-files`, `--no-skills`, `--no-prompt-templates`, or `--no-tools`.
- [ ] A subsequent pi text request succeeds.
- [ ] Pi still lists the model as accepting text and pictures, and its default provider/model/thinking values remain `openai-codex`, `gpt-5.6-luna`, and `xhigh`.
- [ ] The rollback worktree remains at `14ee7b0`; no picture code/evidence is pushed or published.

- [ ] **Step 1: Extend the failing live acceptance helper**
  - Add a long picture payload with at least 32,768 repeated text tokens plus the existing deterministic `424242` fixture.
  - Assert HTTP 200, answer contains `424242`, reported prompt tokens exceed 32,768, cached prompt tokens are zero, and `/v1/models` plus a short text request remain healthy afterward.
  - Store elapsed time and memory samples under a distinct `long_picture` evidence key.
  - Run `py_compile` and the helper's non-server fixture construction/import path; do not claim the live case passes against the still-running old server.

- [ ] **Step 2: Stop the accepted old process and run required graphics-card tests**
  - Confirm the listener PID on `127.0.0.1:2020`, stop that process and its worker children, and verify the port and RTX 5090 are released.
  - Run the exact graphics-card regression command above.
  - Expected: every required case runs and passes. If any fails, restart the known-good text launcher immediately, keep pi text-only, and stop this task.

- [ ] **Step 3: Start the edited picture server and verify geometry**
  - Start the vision launcher with the exact live command above as a supervised task.
  - Wait for ready or terminal error. Verify launcher arguments, all picture weights, mmap PLE, serial experts, automatic expert count, 4,097 pages, and 262,144 usable tokens from logs and health routes.
  - Do not lower context, enlarge the per-step budget, disable graphs, disable mmap, or force an expert count to make startup pass.
  - On failure, stop every picture worker and execute the rollback recipe.

- [ ] **Step 4: Run direct long-picture and regression acceptance**
  - Run the updated vision helper followed by the existing mmap text helper with the exact live commands above.
  - Inspect JSON, server logs, health routes, and process state independently. Require the long picture request's answer, zero cache hit, repeated step logs, memory facts, and post-request text health.
  - On any failure, restore pi's text-only declaration and the known-good text server; do not proceed to pi acceptance.

- [ ] **Step 5: Run normal pi acceptance, record evidence, and checkpoint**
  - Save current pi model/settings backups under ignored `.local/evidence` and verify defaults before the request.
  - Use the user's retained terminal screenshot when available; otherwise generate a 1920×1280 terminal-style fixture with independently known visible text. Run pi with the explicit FreeToken provider/model and `--no-session`, but leave normal context files, skills, prompt templates, extensions, and tools enabled.
  - Because nested pi commands are automatically handed off by the harness, launch the exact pi command through a detached local `.cmd`, redirect stdout/stderr, and record the real exit code as in the accepted first milestone.
  - Require a correct description containing independently visible terminal details and exit code 0, then run a second pi text request requiring `PI_TEXT_HEALTH_OK`.
  - Confirm model listing still says pictures `yes` and defaults remain unchanged.
  - Update ignored acceptance evidence with the normal pi command/result and final local commit IDs. Update internal design/plan completion status.
  - Run `git diff --check`, inspect exact staged paths, and commit only `benchmarks/run_qwen38_vision_smoke.py` as `test(windows): prove chunked picture prompts`. Do not push.

## Rollback recipe

On any live failure:

```powershell
# Stop edited picture workers and verify port 2020 is free first.
Set-Location 'D:\FreeToken-ple-mmap'
& 'D:\FreeToken-ple-mmap\scripts\start-qwen38-flash-next-mmap-windows.ps1' `
  -ModelPath 'D:\Models\Qwen3.8-Flash-Next-NVFP4' `
  -Port 2020 `
  -ContextTokens 262144 `
  -MaxRunningRequests 1
```

Restore `C:\Users\jay\.pi\agent\models.json` from the ignored pre-change backup so the FreeToken entry is text-only, then run `benchmarks/run_qwen38_mmap_smoke.py`. The required rollback evidence is branch `windows-ple-mmap` at `14ee7b0`, loopback serving, mmap PLE, exactly 262,144 usable tokens, and passing text smoke.

## Checkpoint commits

1. `feat(scheduler): chunk picture-bearing prompts`
2. `test(windows): prove chunked picture prompts`

Both commits remain local on `windows-ple-mmap-vision`. Before each commit, stage exact paths, inspect `git diff --cached --stat` and the complete staged diff, and verify no `.local`, planning, pi, model, Desktop, log, generated-kernel, or cache path is staged.

## Final evidence record

Ignored `.local\qwen38-vision-acceptance.json` must add:

- implementation branch and both new local commit IDs;
- per-step limit and complete long-picture prompt token count;
- long-picture answer, elapsed time, cached-token count, and observed prompt-step log count;
- whole-GPU and whole-system memory samples/peaks during long-picture work;
- exact cache geometry, expert count, mmap confirmation, and server health;
- normal pi command flags proving normal context/tools were enabled, screenshot identity, stdout/stderr/exit code, and subsequent text result;
- unchanged pi defaults;
- rollback status and unpublished status.

## Self-review

- **Coverage:** Every approved requirement maps to Task 1 scheduler/resource behavior or Task 2 live/direct/pi proof. The old hard rejection is removed, while model-context, pixel, source, and per-step limits remain.
- **Evidence:** Product symbols, tests, launcher, Python path, and commands come from the current `66c425c` checkout, completed first-milestone plan, `tests/README.md`, and accepted Windows execution record.
- **Interfaces:** The pending request keeps complete features/positions; each scheduled request receives a temporary reference; gathering selects current rows; the model receives matching current placeholders/features; privacy remains authoritative after references clear.
- **Dependencies:** Task 1 is the only code slice. Task 2 starts only after its CPU checkpoint and sequences GPU release, tests, startup, direct acceptance, then normal pi acceptance. No cycle exists.
- **Seams:** CPU tests cross real scheduling/cache seams with independently numbered rows and positions. Live tests cross OpenAI and pi boundaries.
- **Scope:** No encoder splitting, resizing, video, Anthropic input, new dependency, larger per-step budget, public action, Desktop/model edit, or unrelated cleanup is included.
- **Placeholders:** The plan contains no unresolved marker or deferred load-bearing choice.
- **Fresh-context readiness:** Each task names exact files, symbols, commands, expected initial failures, smallest changes, acceptance results, rollback, staging boundaries, and commit message.
