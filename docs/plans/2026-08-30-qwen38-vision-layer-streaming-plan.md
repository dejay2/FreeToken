# Qwen3.8 Layer-Streamed Picture Reader Implementation Plan

**Goal:** Restore at least 50 output tokens/second for normal Qwen3.8 language generation while retaining still-picture input, exactly 262,144 usable context tokens, and no more than six seconds of picture-encoder time for the retained 1920×1280 screenshot.

**Design source:** `docs/design/2026-08-30-qwen38-vision-layer-streaming-design.md`

**Completion:** Implemented and accepted locally on 2026-08-30 through `4375d71`, `728db40`, and final layer-stream commit `c5876d5`. All 333 picture tensors remain CPU-resident; automatic sizing restored 4,063 experts and 262,144 usable tokens. The exact 512-token benchmark averaged 50.58 tokens/second, the final post-picture benchmark averaged 52.10, the retained screenshot encoded under six seconds, the complete 33K picture regression passed, and Jay's normal pi picture/text checks passed. Evidence remains ignored under `D:\FreeToken-ple-mmap-vision\.local`; nothing was pushed or published.

**Context sources:** `CONTEXT.md`, `CONTEXT-MAP.md`

**ADR sources:** `docs/adr/0001-desktop-assisted-native-windows-fork.md`

**Architecture:** Keep every `visual.*` checkpoint tensor in pageable CPU memory. Let Qwen expose a model-owned destination for each state key so the generic engine keeps current behavior for every other model and key. During picture encoding, stage the patch projection, one reusable transformer block, and the final merger into bounded RTX 5090 workspaces; keep picture hidden states on the GPU and return final features on the GPU. Preserve the existing all-GPU mode as an explicit fallback.

**Global constraints:**

- Work only in `D:\FreeToken-ple-mmap-vision` on local branch `windows-ple-mmap-vision`.
- Keep `D:\FreeToken-ple-mmap` on `windows-ple-mmap` at `14ee7b0` as the rollback checkout.
- Do not edit FreeToken Desktop, the model files, pi defaults, or the published branch.
- Do not push, publish, open/edit upstream items, or expose local evidence.
- Keep loopback `127.0.0.1:2020`, one active request, mmap PLE, MoE offload, serial expert loading, automatic expert sizing, and `--kv-reserve-tokens 262144`.
- Keep each prompt-loading step at or below 8,192 tokens and preserve the accepted chunked-picture path at `56a34ec`.
- Store all 897,862,112 picture-weight bytes in ordinary CPU memory in `layer-stream` mode; do not pin the complete picture reader.
- Do not permanently reserve the complete picture reader or a complete second tower on the GPU.
- Require at least 50 output tokens/second on the exact deterministic 512-input/512-output benchmark and at most six seconds of logged picture-encoder time for the retained screenshot.
- The local server is stopped at planning time. Restart it only in the ordered GPU/live stages below or at Jay's request.

**Out of scope:**

- Full-CPU production picture execution, picture-weight quantization, two-layer transfer overlap, or automatic placement changes.
- Lowering context, enlarging one prefill step, moving language/KV/GDN/PLE state, changing picture resolution policy, or adding dependencies.
- Video, Anthropic picture input, shared/public serving, publication, or upstream work.

## Execution setup

Use the Desktop Python with checkout-local source and picture packages:

```powershell
$VisionRoot = 'D:\FreeToken-ple-mmap-vision'
$KnownGood = 'D:\FreeToken-ple-mmap'
$DesktopPython = 'C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe'
$VisionSite = "$VisionRoot\.local\vision-packages"
$ModelPath = 'D:\Models\Qwen3.8-Flash-Next-NVFP4'
$env:PYTHONPATH = "$VisionRoot\scripts\windows-ple-mmap;$VisionSite;$VisionRoot\python"
```

Use `$env:CUDA_VISIBLE_DEVICES = '-1'` for CPU-only checks. Remove it only for the required RTX tests and live server. The server must remain stopped during implementation until the GPU stage.

## Codebase map

### Prior accepted chunking checkpoint

- Modify/commit first: `benchmarks/run_qwen38_vision_smoke.py`
  - Current only uncommitted tracked change.
  - Adds a 33K+ picture-bearing prompt using the deterministic `424242` fixture and records zero cache reuse, timing, memory, and post-request health.
  - Already passed live at 33,243 prompt tokens through `8,192 + 8,192 + 8,192 + 8,192 + 475` scheduled tokens.

### Placement and loading

- Modify: `python/freetoken/models/config.py`
  - Add one validated picture execution-mode reader with `gpu` as the backward-compatible default and `layer-stream` as the selected local mode.
  - Keep `FREETOKEN_LOAD_VISION` as the independent on/off gate.
- Modify: `python/freetoken/engine/engine.py`
  - Extend `_materialize_loaded_weight_state_dict` with an optional model-owned destination callback.
  - Ask the constructed model for a key destination when it provides the hook; otherwise preserve the single engine-device behavior.
  - Log an optional model-provided picture-weight placement summary after strict loading.
- Modify: `python/freetoken/models/qwen4_exp/weight.py`
  - In `layer-stream`, read `model.visual.*`/`visual.*` tensors directly through a CPU safetensors handle while reading language tensors through the existing engine-device handle.
  - Preserve current fusion, dropping, all-GPU, expert, and PLE behavior.
- Modify: `python/freetoken/models/qwen4_exp/model.py`
  - Expose `weight_device_for_key(key, engine_device)` for Qwen.
  - Return CPU only for `visual.*` in `layer-stream`; return the engine device for every language key and in `gpu` mode.
  - Report persistent picture tensor count, bytes, and devices for startup evidence.
- Modify: `scripts/start-qwen38-flash-next-mmap-windows.ps1`
  - Add a validated `-VisionExecution` choice (`layer-stream` or `gpu`).
  - Select `layer-stream` by default only when `-EnableVision` is used, set the execution setting deterministically, and print it.
  - Preserve all text-only defaults and current local package validation.

### Streamed execution

- Modify: `python/freetoken/models/qwen4_exp/vision.py`
  - Add in-place state copying from a CPU source component into an already allocated GPU component.
  - Add a streamed path that stages patch projection, reuses one GPU block for all 27 blocks, computes position data without moving the full position embedding permanently, stages the final merger, returns GPU features, and releases staged objects in `finally`.
  - Keep `Qwen4VisionModel.forward` unchanged as the all-GPU reference.
  - Reject non-empty `deepstack_visual_indexes` clearly in `layer-stream` until that configured path has an independently tested bounded workspace; the target checkpoint has none.
- Modify: `python/freetoken/models/qwen4_exp/model.py`
  - Route `encode_images` to all-GPU or streamed execution.
  - Derive the language device from a real loaded language tensor, not from a hard-coded CUDA index.
  - Keep final features on the language device.
- Modify: `python/freetoken/scheduler/scheduler.py`
  - Pass CPU pixels and grids into `model.encode_images`; let the model own placement.
  - Log picture-encoder elapsed seconds per request for the six-second acceptance gate.
  - Preserve row-count validation, MRoPE construction, request-local errors, and raw CPU tensor cleanup.

### Tests

- Modify: `tests/models/qwen4_exp/test_config.py`
  - Picture execution default, selected mode, invalid value, and text-only isolation.
- Modify: `tests/models/qwen4_exp/test_weight.py`
  - Tiny synthetic checkpoint proof that `layer-stream` reads visual tensors on CPU and language tensors on the requested device, while `gpu` retains current placement.
- Create: `tests/engine/test_vision_weight_placement.py`
  - Generic `_materialize_loaded_weight_state_dict` callback seam, no-hook compatibility, dtype preservation, and Qwen key destination behavior.
- Modify: `tests/models/qwen4_exp/test_vision.py`
  - Small full-GPU versus streamed BF16 output, exact block order, one workspace identity, two-call reuse, failure cleanup, output device, and deep-stack rejection.
- Modify: `tests/scheduler/test_multimodal_admission.py`
  - Scheduler passes CPU tensors to model-owned encoding, records timing without changing request behavior, and releases raw tensors on injected streamed failures.
- Retest unchanged: chunked picture, MRoPE, QSA, graph replay, cache privacy, cancellation, hybrid/GDN, SWA, source/message, and text suites.

### Live evidence

- Reuse: `benchmarks/bench_qwen38_mmap_windows.py`
  - Exact published benchmark generator and streaming timing calculation.
- Reuse: `benchmarks/run_qwen38_vision_smoke.py`
  - Picture sources, two-picture ordering, privacy, 33K chunking, errors, memory, and text health.
- Reuse: `benchmarks/run_qwen38_mmap_smoke.py`
  - Text, streamed reasoning, tool call, context, and health.
- Update ignored: `.local/qwen38-vision-acceptance.json` and `.local/evidence/*`
  - Placement, startup, benchmark, picture time, memory, user pi result, and commit IDs.
- Update internally after passing evidence: the approved design, this plan, the completed first-picture design/plan, and chunking design/plan under `D:\FreeToken-ple-mmap`.

No dependency or model-file change is required.

## Important interfaces

### Execution mode

`vision_execution_mode()` consumes `FREETOKEN_VISION_EXECUTION` and returns exactly `gpu` or `layer-stream`. An absent setting returns `gpu`. An unsupported non-empty setting raises a startup `ValueError` naming the accepted values. `FREETOKEN_LOAD_VISION=0` still prevents picture model construction entirely.

### Weight destination

The generic materializer accepts a callback shaped as:

```python
(key: str, engine_device: torch.device) -> torch.device
```

When no callback exists, every tensor follows the current `engine_device` path. Qwen's callback returns CPU only when all three are true: picture loading is enabled, execution mode is `layer-stream`, and `key.startswith("visual.")`.

The Qwen safetensors reader must also read those raw visual values directly on CPU. The destination hook alone is insufficient because reading them on CUDA first would leave allocator reservations and corrupt automatic expert sizing.

### Reusable component copy

The component-copy seam consumes two `BaseOP` state dictionaries with identical names, shapes, and dtypes. It copies values into existing destination tensors in place and returns no replacement tensors. Any name/shape/dtype mismatch raises before forward computation. The block workspace's destination tensor addresses remain unchanged for all 27 source blocks.

### Picture encoding

`Qwen4ExpForCausalLM.encode_images(pixel_values, image_grid_thw)` continues to return `[picture_tokens, hidden_size]` on the language device. Callers do not choose placement. In `gpu`, behavior remains current. In `layer-stream`, inputs may be CPU and the model stages them. Scheduler and prompt-chunking consumers remain unchanged after the returned feature tensor.

## Testing strategy

### Highest seams

- Loader seam: real tiny safetensors checkpoint -> Qwen iterator -> generic materializer -> mixed-device model state.
- Execution seam: CPU-resident small Qwen picture reader -> repeated streamed GPU components -> final features compared with the existing all-GPU forward using the same weights and inputs.
- Public seam: `/v1/chat/completions`, `/v1/cache/status`, the exact deterministic benchmark, and normal pi `@picture`.

Expected values come from independently numbered test weights, stable destination tensor addresses, existing all-GPU output, published text benchmark settings, and known picture text (`424242`, `gpt-5.6-luna`).

### Focused CPU command

```powershell
Set-Location $VisionRoot
$env:CUDA_VISIBLE_DEVICES = '-1'
& $DesktopPython -m pytest -q `
  tests/engine/test_vision_weight_placement.py `
  tests/models/qwen4_exp/test_config.py `
  tests/models/qwen4_exp/test_weight.py `
  tests/models/qwen4_exp/test_vision.py `
  tests/scheduler/test_multimodal_admission.py `
  tests/scheduler/test_multimodal_chunked_prefill.py
```

CUDA-required streamed-output cases must report skips here and must not be counted as passing.

### Relevant CPU regression commands

```powershell
Set-Location $VisionRoot
$env:CUDA_VISIBLE_DEVICES = '-1'
& $DesktopPython -m pytest -q `
  tests/server/test_openai_api.py `
  tests/server/test_message_wire.py `
  tests/tokenizer/test_multimodal_input.py `
  tests/scheduler `
  tests/engine/test_cache_budget.py `
  tests/engine/test_graph_mrope.py `
  tests/kvcache/test_qsa_pool.py
& $DesktopPython -m pytest -q tests/models/qwen4_exp `
  -k 'not load_ple_table_concatenates and not load_ple_table_rejects and not read_range_into'
```

The existing Windows-only direct-I/O exclusions retain their accepted meaning. GPU cases remain skipped, not passed.

### Required RTX command

After confirming port 2020 is free:

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
Set-Location $VisionRoot
& $DesktopPython -m pytest -q `
  tests/engine/test_vision_weight_placement.py `
  tests/models/qwen4_exp/test_vision.py `
  tests/models/qwen4_exp/test_mrope.py `
  tests/models/qwen4_exp/test_qsa_kernels.py `
  tests/models/qwen4_exp/test_qsa_backend.py `
  tests/models/qwen4_exp/test_qsa_hf.py `
  tests/models/qwen4_exp/test_qsa_mrope.py `
  tests/engine/test_graph_mrope.py `
  tests/kvcache/test_qsa_pool.py
```

Every required placement, streamed-output, workspace-reuse, tiled MRoPE, split QSA ring, norm/rotation, and graph-replay case must run. Optional FlashInfer/Qwen-HF comparisons may skip only for the same previously recorded unavailable packages.

### Live commands

```powershell
& "$VisionRoot\scripts\start-qwen38-flash-next-mmap-windows.ps1" `
  -ModelPath $ModelPath `
  -Port 2020 `
  -ContextTokens 262144 `
  -MaxRunningRequests 1 `
  -EnableVision `
  -VisionExecution layer-stream `
  -EnableCacheReport
```

```powershell
& $DesktopPython "$VisionRoot\benchmarks\bench_qwen38_mmap_windows.py" `
  --base-url http://127.0.0.1:2020/v1 `
  --model Qwen3.8-Flash-Next-NVFP4 `
  --tokenizer $ModelPath `
  --context-label 262144 `
  --output "$VisionRoot\.local\evidence\layer-stream-text-benchmark.json"
& $DesktopPython "$VisionRoot\benchmarks\run_qwen38_vision_smoke.py" `
  --base-url http://127.0.0.1:2020/v1 `
  --model Qwen3.8-Flash-Next-NVFP4
& $DesktopPython "$VisionRoot\benchmarks\run_qwen38_mmap_smoke.py" `
  --base-url http://127.0.0.1:2020/v1 `
  --model Qwen3.8-Flash-Next-NVFP4 `
  --expected-context 262144
```

Normal pi acceptance uses the retained screenshot and explicit local model while leaving normal context, skills, templates, extensions, and tools enabled:

```powershell
pi --provider freetoken-local --model Qwen3.8-Flash-Next-NVFP4 --thinking low --no-session `
  -p '@D:\FreeToken-ple-mmap-vision\.local\vision-fixtures\normal-pi-screenshot.png' `
  'Read this screenshot. Reply with exactly the selected model name shown in the header, and nothing else.'
```

Expected visible answer: `gpt-5.6-luna`, followed by a separate pi text request returning `PI_TEXT_HEALTH_OK`.

## Task breakdown

### Task 1: Close the accepted picture-chunking checkpoint

**Outcome:** Preserve the already-passed 33K picture request in its own local commit before placement code begins, leaving a clean tracked baseline.

**Blocked by:** None.

**Files:**

- Modify: `benchmarks/run_qwen38_vision_smoke.py` - current uncommitted 33K picture acceptance and `--skip-long-picture` switch.
- Update ignored: `.local/qwen38-vision-acceptance.json` - direct 33,243-token result, five step sizes, memory, and Jay's successful normal pi screenshot result.
- Update internal planning docs only: completed chunking design/plan in `D:\FreeToken-ple-mmap`.

**Interfaces:**

- Consumes: accepted scheduler commit `56a34ec`, deterministic `424242` fixture, live evidence already recorded before the server was stopped.
- Produces: clean placement starting point and local commit `test(windows): prove chunked picture prompts`.

**Acceptance criteria:**

- [ ] Helper source contains the 32,768 repeated-token picture payload, `prompt_tokens > 32_768`, correct answer, and zero cached-token assertion through `_assert_picture`.
- [ ] Existing evidence records 33,243 prompt tokens, answer `424242`, zero cache reuse, 13.71 seconds, steps `8192,8192,8192,8192,475`, peak 31,830 MiB whole-GPU use, and 92,448,464,896 peak whole-system bytes.
- [ ] Jay's user-run normal pi result `gpt-5.6-luna` and post-picture direct text result `PI_TEXT_HEALTH_OK` are recorded as separate evidence sources without inventing a pi exit file.
- [ ] Only `benchmarks/run_qwen38_vision_smoke.py` is staged; `.local`, docs, pi files, logs, and prototype files remain unstaged/untracked.

- [ ] **Step 1: Verify the completed acceptance change**
  - Inspect the exact current diff and the ignored JSON evidence.
  - Expected: the helper change is limited to the long-picture case and one skip switch; the evidence has `passed: true` and the results above.

- [ ] **Step 2: Run static checks**
  - Run: `& $DesktopPython -m py_compile benchmarks/run_qwen38_vision_smoke.py`; `git diff --check`.
  - Expected: both exit 0. Do not restart the server merely to repeat already captured acceptance.

- [ ] **Step 3: Update internal completion records**
  - Mark the approved chunking design implemented/accepted and its plan checks complete in `D:\FreeToken-ple-mmap`.
  - Preserve the original first-picture plan as historical evidence and add a clear pointer that its one-step limit was superseded by chunking.

- [ ] **Step 4: Stage and inspect exactly one path**
  - Run: `git add -- benchmarks/run_qwen38_vision_smoke.py`; `git diff --cached --check`; inspect `git diff --cached --stat`, names, and complete diff.
  - Expected: exactly one staged path and no ignored/internal content.

- [ ] **Step 5: Commit the checkpoint**
  - Commit: `test(windows): prove chunked picture prompts`.
  - Expected: clean tracked worktree before Task 2 and no push.

### Task 2: Keep picture weights on CPU without changing any language key

**Outcome:** `layer-stream` constructs the same Qwen picture reader but loads every persistent picture tensor directly into CPU memory, allowing GPU weight accounting and automatic expert sizing to reclaim that space. All-GPU and text-only modes remain unchanged.

**Blocked by:** Task 1.

**Files:**

- Modify: `python/freetoken/models/config.py` - `vision_execution_mode` validation/default.
- Modify: `python/freetoken/engine/engine.py` - `_materialize_loaded_weight_state_dict` destination callback and optional placement report.
- Modify: `python/freetoken/models/qwen4_exp/weight.py` - direct CPU visual reads in `layer-stream`.
- Modify: `python/freetoken/models/qwen4_exp/model.py` - Qwen destination hook and picture-weight report.
- Modify: `scripts/start-qwen38-flash-next-mmap-windows.ps1` - `-VisionExecution` selection and output.
- Modify: `tests/models/qwen4_exp/test_config.py` - mode contract.
- Modify: `tests/models/qwen4_exp/test_weight.py` - source device contract.
- Create: `tests/engine/test_vision_weight_placement.py` - generic materialization and Qwen destination seam.

**Interfaces:**

- Consumes: `FREETOKEN_LOAD_VISION`, new `FREETOKEN_VISION_EXECUTION`, raw checkpoint key names, constructed model state, engine CUDA device.
- Produces: a strict mixed-device model state where only `visual.*` is CPU in `layer-stream`, plus a startup placement summary.
- Preserves: current `gpu` picture placement and no-picture behavior for all model families.

**Acceptance criteria:**

- [ ] Missing execution setting resolves to `gpu`; `layer-stream` and `gpu` are accepted; every other value raises a named startup error.
- [ ] Picture loading off still constructs no picture reader regardless of the execution setting.
- [ ] The generic materializer calls the model destination hook per key and preserves dtype; no-hook models retain the prior one-device result.
- [ ] Qwen routes only `visual.*` to CPU in `layer-stream` and every key to the engine device in `gpu`.
- [ ] The Qwen checkpoint reader fetches visual values directly on CPU rather than first allocating them on CUDA.
- [ ] Tiny-checkpoint strict loading has no missing/unexpected key and reports exact visual count/bytes/device.
- [ ] Launcher text-only path sets picture loading off deterministically; vision path prints and sets the chosen execution mode.

- [ ] **Step 1: Write failing placement tests**
  - Add literal mode cases to `test_config.py`.
  - Extend the synthetic checkpoint with enough visual tensors to assert source device and destination key behavior independently.
  - Add `test_materializer_uses_model_key_destination_without_changing_default` and `test_qwen_layer_stream_routes_only_visual_keys_to_cpu` in the new engine test.
  - Expected current failures: no execution-mode reader, no destination callback, Qwen iterator uses one device handle, and the launcher has no execution choice.

- [ ] **Step 2: Run focused tests before implementation**
  - Run the focused CPU command, allowing pre-existing CUDA skips.
  - Expected: only new mode/placement cases fail; existing config, weight, vision, admission, and chunking cases pass.

- [ ] **Step 3: Implement the smallest placement slice**
  - Add validated mode reading without adding a dependency.
  - Pass a callback into `_materialize_loaded_weight_state_dict`; use `getattr` so other models need no edit.
  - Give Qwen the destination hook and direct CPU visual safetensors handle.
  - Ensure CPU visual reads do not create transient CUDA values.
  - Add the launcher choice with backward-compatible all-GPU mode.
  - Do not add streamed forward behavior yet; `layer-stream` may fail clearly if `encode_images` is called before Task 3.

- [ ] **Step 4: Run focused tests again**
  - Run the focused CPU command and the PowerShell parser against the launcher.
  - Expected: placement tests pass, existing focused tests pass, CUDA cases remain explicit skips, and the server stays stopped.

- [ ] **Step 5: Run relevant checks and checkpoint**
  - Run the relevant CPU regression commands, `py_compile` on modified Python files, and `git diff --check`.
  - Stage only Task 2 product/tests/launcher paths; inspect complete staged diff.
  - Commit: `feat(qwen4-exp): keep picture weights CPU-resident`.
  - Expected: no benchmark, `.local`, docs, model, Desktop, or pi file is staged; no push.

### Task 3: Encode pictures through one reusable GPU layer

**Outcome:** A CPU-resident Qwen picture reader stages bounded components, reuses one GPU transformer block across all 27 layers, returns features on the language GPU, and cleans up deterministically.

**Blocked by:** Task 2.

**Files:**

- Modify: `python/freetoken/models/qwen4_exp/vision.py` - component copy, reusable block, streamed forward, cleanup, and deep-stack guard.
- Modify: `python/freetoken/models/qwen4_exp/model.py` - execution routing and language-device derivation.
- Modify: `python/freetoken/scheduler/scheduler.py` - model-owned input placement and encoder timing log.
- Modify: `tests/models/qwen4_exp/test_vision.py` - streamed correctness, reuse, ordering, cleanup, and boundaries.
- Modify: `tests/scheduler/test_multimodal_admission.py` - CPU input ownership, timing, and failures.

**Interfaces:**

- Consumes: Task 2's CPU `visual.*` state, CPU pixel/grid tensors, one engine CUDA device, and current Qwen picture geometry.
- Produces: the unchanged GPU `[picture_tokens, 2560]` feature contract for scheduler slicing and language placeholder replacement.
- Preserves: all-GPU `forward`, picture row order, MRoPE, private cache, chunking, cancellation, and text-only behavior.

**Acceptance criteria:**

- [ ] Component copy validates identical names/shapes/dtypes and preserves every destination tensor address.
- [ ] One block workspace handles every source block in order; a test with independently numbered block weights proves no skip, repeat, or stale block.
- [ ] Streamed small-model output matches the same model's all-GPU BF16 output within the test's fixed tolerance and stays on the requested GPU.
- [ ] Two sequential encodes use bounded memory and leave no staged source/component reference after success.
- [ ] Injected copy, block, and merger failures execute cleanup and allow a subsequent encode/text path.
- [ ] Empty deep-stack indexes work; non-empty indexes fail before partial execution with a clear unsupported message.
- [ ] Scheduler sends CPU pixels/grid to the model in `layer-stream`, still validates feature rows, and releases raw tensors on success/failure.
- [ ] All-GPU mode continues to accept CPU scheduler inputs because `encode_images` owns the move.
- [ ] Encoder timing log names request UID and finite elapsed seconds.

- [ ] **Step 1: Write failing streamed-execution tests**
  - Use a small deterministic picture config and the existing all-GPU model as the reference.
  - Spy on destination state tensor `data_ptr()` values across at least three independently weighted source blocks.
  - Call twice and inject failures at copy/block/merger seams; assert cleanup state and output device.
  - Update admission fake model to assert it receives CPU tensors and controls its own output device.
  - Expected current failures: no streamed forward, scheduler moves inputs itself, no timing evidence, and no workspace lifecycle.

- [ ] **Step 2: Run focused tests before implementation**
  - Run the focused CPU command.
  - Expected: CPU validation/error cases fail only for missing streamed behavior; CUDA output/reuse cases skip until the RTX stage.

- [ ] **Step 3: Implement the smallest streamed path**
  - Copy source state into existing destination tensors with `copy_`; never call `load_state_dict` per layer because it replaces tensor objects.
  - Create patch workspace, one reusable block workspace, and merger workspace within one encode.
  - Keep hidden/rotary values on GPU, compute segment lengths from CPU grid, and return GPU features.
  - Delete staged objects in one `finally`; do not `empty_cache` between blocks. Release allocator cache once after encode only if measurements prove it necessary.
  - Route modes in `encode_images`; move all-GPU inputs there.
  - Add scheduler timing and remove unconditional scheduler-side `.to(self.device)`.
  - Do not resize/evict language experts or KV while encoding.

- [ ] **Step 4: Run CPU and RTX focused tests**
  - Run the focused CPU command, then the required RTX command with the server still stopped.
  - Expected: all required streamed correctness/reuse cases run and pass; only the documented optional FlashInfer/Qwen-HF cases may skip.

- [ ] **Step 5: Run relevant checks and checkpoint**
  - Run relevant CPU suites again, `py_compile`, and `git diff --check`.
  - Stage only Task 3 paths and inspect the complete staged diff.
  - Commit: `feat(qwen4-exp): stream picture layers through GPU`.
  - Expected: no evidence/docs/pi/Desktop/model path staged and no push.

### Task 4: Prove speed, picture latency, full context, and normal pi

**Outcome:** The live layer-stream server restores coding speed, meets picture time, preserves exact context and all accepted picture/text behavior, and records a private rollback-ready result.

**Blocked by:** Task 3 and every required RTX test.

**Files:**

- Reuse/test: `benchmarks/bench_qwen38_mmap_windows.py` - exact language speed gate.
- Reuse/test: `benchmarks/run_qwen38_vision_smoke.py` - complete direct picture/chunking regression.
- Reuse/test: `benchmarks/run_qwen38_mmap_smoke.py` - text/reasoning/tool health.
- Update ignored: `.local/qwen38-vision-acceptance.json`, `.local/evidence/*` - live records only.
- Update internal: approved design/plan and superseded first-picture/chunking records in `D:\FreeToken-ple-mmap`.

**Interfaces:**

- Consumes: Task 3's server, launcher, status/log output, OpenAI route, exact benchmark, and pi model declaration already advertising pictures.
- Produces: measured acceptance or immediate rollback; no public artifact.

**Acceptance criteria:**

- [ ] Startup logs `layer-stream`, 333 CPU picture tensors, exactly 897,862,112 CPU picture bytes, no persistent CUDA picture tensor, mmap PLE, serial experts, and automatic expert count.
- [ ] Cache status reports 4,097 pages, page size 64, 262,144 usable tokens, 8 Mamba slots, and serving state.
- [ ] Automatic expert count returns to 4,063 or another value that still yields at least 50 output tokens/second without changing context.
- [ ] Exact deterministic benchmark's three short runs average at least 50 output tokens/second; result validates with its built-in validator.
- [ ] Retained screenshot encoder log is no more than six seconds and its answer is `gpt-5.6-luna` or the independently expected visible detail requested.
- [ ] Two sequential screenshot requests do not grow steady memory or alter expert count; post-picture text speed remains at least 50 on a repeated short benchmark sample.
- [ ] Full vision smoke passes sources, two pictures, cross-picture zero reuse, 33K picture chunking, bounded errors, memory, 32K text, and final health.
- [ ] Text smoke passes text, streamed reasoning, tool parsing, 262,144 context, and final health.
- [ ] Normal pi with ordinary context/skills/templates/extensions/tools returns `gpt-5.6-luna`; subsequent pi text returns `PI_TEXT_HEALTH_OK`.
- [ ] Pi defaults remain `openai-codex`, `gpt-5.6-luna`, `xhigh`; local model remains text+image.
- [ ] Rollback checkout remains `14ee7b0`; picture branch and evidence remain local and unpublished.

- [ ] **Step 1: Start the edited server and verify placement before requests**
  - Run the exact live launcher command.
  - Wait for ready or terminal error; inspect logs and health.
  - Expected: selected mode, CPU picture summary, mmap, 4,097 pages, 262,144 usable tokens, and auto expert count. On any mismatch, stop all workers and execute rollback; do not benchmark.

- [ ] **Step 2: Run the exact language speed gate first**
  - Run `bench_qwen38_mmap_windows.py` with the exact live command and validate its JSON using `--validate`.
  - Expected: three `short` runs, each 512 completion tokens, mean at least 50 output tokens/second.
  - If below 50, stop placement acceptance and return to design/profiling; do not relax the threshold or context.

- [ ] **Step 3: Run picture latency and memory gates**
  - Send the retained screenshot twice through the direct API and inspect per-request encoder logs.
  - Expected: each encoder time is at most six seconds, correct answer, second request has no larger steady-state allocation, and cache geometry/expert count are unchanged.
  - Run one short exact benchmark sample afterward; require at least 50 output tokens/second.

- [ ] **Step 4: Run full direct regression**
  - Run the vision helper, mmap smoke, health routes, and inspect server logs independently.
  - Expected: all helper assertions pass, the 33K request has repeated 8,192-token steps plus a partial step, no picture cache reuse, no worker error, and final text health.
  - Record whole-system/GPU peaks and picture encoder times in ignored evidence.

- [ ] **Step 5: Run normal pi, finalize evidence, and close the plan**
  - Ask Jay to run the exact normal pi picture command directly in Windows Terminal, avoiding nested background wrappers; require visible `gpt-5.6-luna` and no 8,192-token error.
  - Run/ask for the subsequent pi text command and require `PI_TEXT_HEALTH_OK`.
  - Verify pi defaults/model capability from existing local files without changing them.
  - Update ignored evidence and internal design/plan statuses with exact commits/results.
  - Run `git status --short`, `git diff --check`, and inspect that no uncommitted tracked placement change remains.
  - Do not push. Leave the accepted layer-stream server running only if Jay wants it; otherwise stop it cleanly.

## Checkpoint commits

1. `test(windows): prove chunked picture prompts`
2. `feat(qwen4-exp): keep picture weights CPU-resident`
3. `feat(qwen4-exp): stream picture layers through GPU`

Every commit is local on `windows-ple-mmap-vision`. Stage exact paths, inspect the complete staged diff, and keep `.local`, planning files, pi files, model/Desktop files, logs, generated kernels, and caches out of git.

## Rollback recipe

On any startup, CUDA, memory, speed, picture, text, or pi failure:

```powershell
# Stop every edited worker and verify 127.0.0.1:2020 is free first.
Set-Location 'D:\FreeToken-ple-mmap'
& 'D:\FreeToken-ple-mmap\scripts\start-qwen38-flash-next-mmap-windows.ps1' `
  -ModelPath 'D:\Models\Qwen3.8-Flash-Next-NVFP4' `
  -Port 2020 `
  -ContextTokens 262144 `
  -MaxRunningRequests 1
```

Restore the local FreeToken pi entry from the ignored pre-picture backup only if picture reliability is lost. Run `benchmarks/run_qwen38_mmap_smoke.py` and require mmap PLE, loopback binding, 4,097 total pages/262,144 usable tokens, and healthy text. Keep the failed placement local.

## Final evidence record

Ignored `.local\qwen38-vision-acceptance.json` must contain:

- branch and all three new local commit IDs;
- execution mode and exact persistent picture tensor count/bytes/device set;
- expert count, cache geometry, mmap confirmation, and startup timing;
- exact benchmark per-run and mean output speed;
- exact screenshot encoder times, API times, answers, and repeated-request memory;
- 33K picture prompt tokens, step boundaries, answer, and cache reuse;
- whole-GPU/system memory peaks;
- direct regression and post-error text results;
- user-provided normal pi picture and text results, clearly labeled by source;
- unchanged pi defaults/capability;
- rollback head/status and unpublished status.

## Self-review

- **Coverage:** Every design requirement maps to Task 1's clean baseline, Task 2's persistent placement, Task 3's bounded computation, or Task 4's live gates.
- **Evidence:** Paths, symbols, commands, fixture values, benchmark settings, and initial failures come from the current checkout, approved design, accepted chunking evidence, and user-approved placement probes.
- **Interfaces:** Raw picture tensors remain CPU until model-owned encoding; CPU source weights feed one stable GPU component; final features retain the existing GPU scheduler contract.
- **Dependencies:** Tasks form the acyclic order 1 -> 2 -> 3 -> 4. Task 2 does not claim runnable picture execution; Task 3 depends on its mixed-device state.
- **Seams:** Tests cross real checkpoint-reader/materializer, model execution, scheduler, OpenAI, benchmark, and pi boundaries with independent expectations.
- **Scope:** No full-CPU production path, quantization, double buffering, context reduction, dependency, resolution, video, publication, Desktop, model-file, or pi-default work was added.
- **Placeholders:** The plan contains no unresolved marker or deferred load-bearing decision.
- **Fresh-context readiness:** Each task names exact files, symbols, expected initial failures, commands, pass conditions, staging boundaries, and rollback behavior.
