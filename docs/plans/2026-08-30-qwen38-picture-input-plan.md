# Qwen3.8 Picture Input Implementation Plan

**Goal:** Add reliable still-picture input to the local Qwen3.8 Flash Next Windows fork while retaining SSD-backed PLE, 262,144 usable text tokens, the newer QSA/text engine, and an immediate rollback to the proven text server.

**Design source:** `docs/design/2026-08-30-qwen38-picture-input-design.md`

**Context sources:** `CONTEXT.md`, `CONTEXT-MAP.md`

**ADR sources:** `docs/adr/0001-desktop-assisted-native-windows-fork.md`

**Completion:** Implemented and accepted locally on 2026-08-30 at first-milestone head `66c425c`. Direct acceptance evidence is under `D:\FreeToken-ple-mmap-vision\.local`; the picture branch remains unpublished, and the rollback worktree remains at `14ee7b0`. The historical one-prefill-step limit in this plan was superseded by accepted chunking at `56a34ec`; permanent all-GPU picture placement was superseded by the accepted CPU-resident layer-stream implementation at `c5876d5`.

**Reusable prior art:** closed FreeToken PR #232 at `ad752c9970e0dc3f1b09aeec38235332149336ed`. Port only its picture-specific behavior; do not cherry-pick or copy its older Qwen text engine.

## Architecture

Keep `D:\FreeToken-ple-mmap` on the known-good `windows-ple-mmap` branch at `14ee7b0`. Create a second git worktree at `D:\FreeToken-ple-mmap-vision` on a new local branch, `windows-ple-mmap-vision`. The currently running text workers continue reading the original worktree while all picture changes and CPU-only checks happen in the second worktree.

The request path is:

1. OpenAI structured message content keeps picture parts intact.
2. The tokenizer worker reads `data:`, HTTP(S), `file://`, and direct Windows-path sources under the approved local-only policy.
3. The checkpoint's `Qwen3VLProcessor` produces text token IDs, picture patch values, picture grid values, and text/picture markers.
4. Optional CPU tensors cross the existing tokenizer-to-scheduler message wire.
5. The GPU scheduler validates the prepared request, rejects picture prompts over one prefill batch, runs the Qwen vision model, and releases the CPU picture tensors.
6. Picture features replace only Qwen's picture-placeholder token embeddings.
7. Qwen three-axis MRoPE positions flow through prompt processing, generated text, CUDA graph replay, main QSA attention, and QSA's compressed-key pending groups.
8. Picture requests keep private prompt memory and never match or enter the shared token-only prefix cache.

Use two position values deliberately:

- `Batch.positions` remains the ordinary one-dimensional logical token position. Cache addressing, causal visibility, QSA block selection, and PLE keep using it unchanged.
- `Batch.rope_positions` is optional `[3, tokens]` temporal/height/width data used only for Qwen rotation. Generated text uses the request's Qwen position delta on all three axes.

QSA compresses four raw index keys into one key rotated at the first member's position. Add a small **per-QSA-layer** three-axis pending-position ring beside the existing per-layer raw-key ring. Per-layer storage is required: one QSA layer must not overwrite old ring coordinates before later QSA layers consume them during a long prefill.

Picture support remains opt-in through `FREETOKEN_LOAD_VISION=1`, set only by the Windows launcher when `-EnableVision` is supplied. Without that switch, configuration, weight loading, launcher requirements, memory use, and text behavior stay on the current path.

## Fixed decisions and guardrails

- Do not edit `C:\Users\jay\AppData\Local\FreeToken` or `D:\Models\Qwen3.8-Flash-Next-NVFP4`.
- Do not alter, merge into, or move the known-good `windows-ple-mmap` branch.
- Do not push the picture branch or publish picture code, results, paths, or instructions.
- Do not open or edit a pull request, issue, or upstream comment.
- Keep the service at `127.0.0.1:2020` with one active request.
- Keep `--ple-backend mmap`, `--moe-backend offload`, automatic expert-cache sizing, serial expert loading, and `--kv-reserve-tokens 262144`.
- Keep picture prompts at or below the live `max_extend_tokens`/prefill budget of 8,192 tokens; reject larger picture prompts before vision execution or scheduler admission.
- Keep ordinary text prompts chunkable through the full 262,144-token usable allocation.
- Accept any readable local path and any HTTP(S) address only because the server is loopback-only. Do not describe this as safe for a shared server.
- Limit every carried, local, or downloaded source to 64 MiB; use a 30-second network timeout and at most five redirects.
- Convert readable still pictures to RGB. Treat an animated file as one still frame.
- Implement still pictures only. Do not add video fields, frame sampling, timestamps, or pi video work.
- Leave Anthropic picture input outside this work.
- Keep pi's default provider, model, and thinking level unchanged.
- Change pi's FreeToken entry to text-and-picture only after every direct acceptance check passes.
- On any GPU, server, regression, or pi acceptance failure, stop picture workers, restore pi's text-only declaration, and restart the launcher from `D:\FreeToken-ple-mmap`.

## Work isolation and execution setup

Run only after this plan is approved:

```powershell
$KnownGood = 'D:\FreeToken-ple-mmap'
$VisionRoot = 'D:\FreeToken-ple-mmap-vision'
$DesktopPython = 'C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe'
$ModelPath = 'D:\Models\Qwen3.8-Flash-Next-NVFP4'

Set-Location $KnownGood
git status --short
git worktree add -b windows-ple-mmap-vision $VisionRoot 14ee7b0dc8c8d97e9fd18776fe71e6a80b0dd1bb
Set-Location $VisionRoot
git branch --show-current
git rev-parse HEAD
```

Expected setup evidence:

- the original worktree remains on `windows-ple-mmap` at `14ee7b0`;
- the second worktree is on `windows-ple-mmap-vision` at the same starting commit;
- the existing text server remains serving from the original worktree;
- no untracked internal files from the original worktree are copied into the picture worktree.

Use this test setup after Task 1 installs the local overlay:

```powershell
$VisionSite = Join-Path $VisionRoot '.local\vision-packages'
$env:PYTHONPATH = "$VisionRoot\scripts\windows-ple-mmap;$VisionSite;$VisionRoot\python"
$env:CUDA_VISIBLE_DEVICES = '-1'
& $DesktopPython -m pytest --version
```

`CUDA_VISIBLE_DEVICES=-1` keeps Tasks 1-3 and 5 on the CPU while the known-good server owns the RTX 5090. Remove it only in the live GPU stage.

## Codebase map

### Request and local dependency path

- Modify: `.gitignore` — ignore `/.local/` so installed picture packages, fixtures, evidence, and generated caches cannot be committed.
- Modify: `pyproject.toml` — add a bounded `vision` extra for Pillow 11-12 and TorchVision 0.26, with TorchVision routed to the existing CUDA 13 PyTorch package source.
- Create: `scripts/install-qwen38-vision-deps-windows.ps1` — install exact Windows picture packages into `.local\vision-packages`, never Desktop.
- Modify: `scripts/start-qwen38-flash-next-mmap-windows.ps1` — add opt-in `-EnableVision`, local-package validation, and optional cache-reporting without changing the default text launch.
- Modify: `python/freetoken/server/generation.py` — preserve structured picture parts while continuing to flatten text-only part lists.
- Modify: `python/freetoken/tokenizer/server.py` — bounded source reading, Pillow decode, Qwen processing, per-request error isolation, and prepared tensor handoff.
- Modify: `python/freetoken/message/backend.py` — optional online picture tensors.
- Modify: `python/freetoken/message/utils.py` — arbitrary-shape and BF16 tensor round trips while accepting old one-dimensional wire records.

### Qwen picture model and weights

- Modify: `python/freetoken/models/config.py` — include Qwen's `visual.` prefix in opt-in vision filtering.
- Modify: `python/freetoken/models/qwen4_exp/config.py` — parse the checkpoint's vision geometry and MRoPE settings only when vision loading is enabled.
- Create: `python/freetoken/models/qwen4_exp/vision.py` — adapted Qwen image-reading network from PR #232.
- Modify: `python/freetoken/models/qwen4_exp/model.py` — instantiate `visual`, encode pictures, and replace picture placeholders before hyper-connection expansion.
- Modify: `python/freetoken/models/qwen4_exp/weight.py` — retain and map `model.visual.*`/`visual.*` only in picture mode while preserving all current text, NVFP4 expert, PLE, and mmap mappings.

### Qwen positions and current QSA

- Create: `python/freetoken/models/qwen4_exp/mrope.py` — independently checkable Qwen position builder and rotation wrapper.
- Modify: `python/freetoken/models/qwen4_exp/attention.py` — rotate main Q/K with scalar or three-axis positions without changing QSA projection geometry.
- Modify: `python/freetoken/kernel/triton/rope.py` — add the fused interleaved three-axis Qwen path to the current tiled kernel.
- Modify: `python/freetoken/core.py` — request MRoPE state and optional batch rotation positions.
- Modify: `python/freetoken/scheduler/utils.py`, `python/freetoken/scheduler/prefill.py`, and `python/freetoken/scheduler/scheduler.py` — retain request position state and build prompt/decode rotation positions.
- Modify: `python/freetoken/engine/graph.py` — persist `[3, batch]` positions through decode graph capture and replay; ordinary text fills all axes with the scalar position.
- Modify: `python/freetoken/kvcache/qsa_pool.py` — per-layer pending three-axis coordinate ring and byte accounting.
- Modify: `python/freetoken/kernel/triton/qsa/compress.py` and `python/freetoken/kernel/triton/qsa/__init__.py` — produce first-member MRoPE coordinates and apply fused QSA norm plus scalar/three-axis rotation.
- Modify: `python/freetoken/attention/qsa_sparse.py` — keep scalar positions for cache/causal work and use three-axis positions only for index query/compressed-key rotation.

`python/freetoken/kernel/triton/qsa/score.py`, `expand.py`, and the token/block top-k logic should not need picture-specific changes: they operate in logical token order, not model-space rotation coordinates.

### Scheduler safety and acceptance

- Modify: `python/freetoken/scheduler/scheduler.py` — validate and encode prepared pictures, enforce one-batch picture prompts, release CPU tensors, and return a request error instead of stopping a worker.
- Use/modify only if tests expose a gap: `python/freetoken/scheduler/cache.py` — current `mm_embeds` guards already disable shared prefix matching, insertion, and donation for picture requests.
- Create: `benchmarks/run_qwen38_vision_smoke.py` — generated fixtures, all source forms, multiple pictures, cache-confusion check, invalid input recovery, short text speed, long text check, memory samples, and JSON evidence.
- Modify only after direct success: `C:\Users\jay\.pi\agent\models.json` — change the existing FreeToken model's `input` from `['text']` to `['text', 'image']`.
- Never modify for this feature: `C:\Users\jay\.pi\agent\settings.json` defaults.

## Testing strategy

Use independent expectations at the seam where each failure could hide:

1. OpenAI request tests verify structured picture parts reach `TokenizeMsg`; they do not test an internal helper's own output against itself.
2. Source tests use real temporary files and a loopback HTTP server for data/local/web forms, limits, and redirects.
3. Message tests use encoder/decoder round trips and exact shape/dtype/value comparisons.
4. Qwen picture-network tests compare a small CPU model to Transformers' `Qwen3VLVisionModel` with the same state dictionary.
5. MRoPE position tests compare to Transformers' `Qwen3VLModel.get_rope_index`; rotation tests compare to a plain PyTorch axis reference.
6. QSA tests compare compression, rotation, selection, chunk boundaries, and graph replay to CPU/eager references.
7. Cache tests submit identical token IDs with different picture features and verify an empty shared-prefix match in ordinary, hybrid, and sliding-window modes.
8. Live tests cross `/v1/chat/completions`, `/v1/models`, and `/v1/cache/status`, then cross pi's documented `@picture.png` command path.

### Focused CPU command

```powershell
Set-Location $VisionRoot
$env:PYTHONPATH = "$VisionRoot\scripts\windows-ple-mmap;$VisionSite;$VisionRoot\python"
$env:CUDA_VISIBLE_DEVICES = '-1'
& $DesktopPython -m pytest `
  tests/server/test_openai_api.py `
  tests/server/test_message_wire.py `
  tests/tokenizer/test_multimodal_input.py `
  tests/models/qwen4_exp/test_vision.py `
  tests/models/qwen4_exp/test_mrope.py `
  tests/scheduler/test_mrope_positions.py `
  tests/scheduler/test_multimodal_admission.py `
  tests/scheduler/test_multimodal_cache_safety.py
```

### Full CPU-visible regression command

```powershell
Set-Location $VisionRoot
$env:CUDA_VISIBLE_DEVICES = '-1'
& $DesktopPython -m pytest tests -m 'not slow'
```

GPU-dependent tests should report skips here, not passes. Their required execution is in Task 6 after the known-good server is stopped.

## Task breakdown

### Task 1: Carry bounded OpenAI picture input to the scheduler wire

**Outcome:** A loopback OpenAI chat request can preserve picture parts, read every approved source form, run the real Qwen processor, and deliver validated CPU tensors over the existing message wire without changing text-only requests.

**Blocked by:** Approved plan and worktree setup.

**Files:**

- Modify: `.gitignore`.
- Modify: `pyproject.toml`.
- Create: `scripts/install-qwen38-vision-deps-windows.ps1`.
- Modify: `scripts/start-qwen38-flash-next-mmap-windows.ps1`.
- Modify: `python/freetoken/server/generation.py`.
- Modify: `python/freetoken/tokenizer/server.py`.
- Modify: `python/freetoken/message/backend.py`.
- Modify: `python/freetoken/message/utils.py`.
- Modify: `tests/server/test_openai_api.py`.
- Modify: `tests/server/test_message_wire.py`.
- Create: `tests/tokenizer/test_multimodal_input.py`.

**Interfaces:**

- Consume OpenAI `image_url` parts whose source is either a string or `{ "url": ... }`; retain PR #232's equivalent `image` spelling when unambiguous.
- Produce `UserMsg.mm_pixel_values`, `mm_image_grid_thw`, and `mm_token_type_ids` only for picture requests.
- Keep `TokenizeMsg` and `UserMsg` text-only behavior byte-for-byte compatible at the field level.
- Install Pillow `12.3.0`, TorchVision `0.26.0+cu130`, and optional pytest tools under `.local\vision-packages`.

**Acceptance criteria:**

- [x] Text-only content lists still flatten to one string and existing OpenAI tests pass.
- [x] Picture-bearing content stays structured through `chat_request_to_genspec` and reaches the tokenizer.
- [x] Valid base64 data, percent-encoded data, HTTP(S), `file://`, drive-letter paths, forward-slash Windows paths, and UNC paths route to the intended reader.
- [x] Invalid base64, unsupported schemes, unreadable files, invalid pictures, a sixth redirect, a timeout, and more than 64 MiB produce one terminal request error.
- [x] Responses, files, and Pillow images are closed in success and error cases.
- [x] An animated picture contributes only its first still frame after RGB conversion.
- [x] The real local `Qwen3VLProcessor` produces one-dimensional `int32` input IDs, BF16 picture patches, `int64` grid rows shaped `[pictures,3]`, and `int32` text/picture markers matching token count.
- [x] Multidimensional BF16 tensors survive a backend-message round trip exactly; old shape-less one-dimensional tensor records still decode.
- [x] Without `-EnableVision`, the launcher sets picture loading off and does not require Pillow or TorchVision.
- [x] With `-EnableVision`, the launcher validates the local package overlay, confirms TorchVision's compiled operations load against Desktop Torch 2.11/CUDA 13, sets `FREETOKEN_LOAD_VISION=1`, and still binds only to `127.0.0.1`.
- [x] No package is written under FreeToken Desktop.

- [x] **Step 1: Write the failing tests**
  - Add the OpenAI structured-content assertion, backend tensor round trip, and source/processor tests.
  - Add a launcher parse/contract check that expects `-EnableVision` and the local package path.
  - Independent expectations come from the OpenAI content shape, exact temporary-file bytes, loopback HTTP responses, the checkpoint processor's required keys, and tensor round trips.

- [x] **Step 2: Run the focused tests before implementation**
  - Run the OpenAI picture test, wire test, and tokenizer tests with Desktop Python and source `PYTHONPATH`.
  - Expected current failures: `generation.py` rejects `image_url` as text-only; `UserMsg` lacks prepared picture fields; the tensor serializer rejects dimensions other than one; picture source helpers and local picture packages are absent.

- [x] **Step 3: Implement the smallest complete request slice**
  - Add `/.local/` to `.gitignore` before installing anything.
  - Add the bounded `vision` extra and exact Windows overlay installer.
  - Preserve picture parts only when a message contains a picture; keep text-only flattening unchanged.
  - Decode/read sources in chunks, reject a known oversized file/response before allocating when possible, and still enforce the limit while reading.
  - Use a redirect handler with a five-hop counter and a 30-second open/read timeout.
  - Load `AutoProcessor.from_pretrained(model_path)` lazily in each tokenizer worker; do not import Pillow/TorchVision on the text path.
  - Convert processor outputs to the exact CPU dtypes before serialization.
  - Add wire shape plus dtype metadata and byte-view encoding for BF16.
  - Add `-EnableVision`, `-VisionPackagesPath`, and `-EnableCacheReport` as optional launcher switches; preserve all existing default arguments.

- [x] **Step 4: Run the focused tests again**
  - Parse both PowerShell scripts without executing the server.
  - Run the exact overlay installer with `-IncludeTestTools`.
  - Import `torch`, `torchvision`, `PIL`, and `pytest` through the overlay; assert no `torch` package exists inside the overlay and TorchVision compiled operations are available.
  - Run the OpenAI, message-wire, and tokenizer picture tests with CUDA hidden.
  - Expected: all focused checks pass and the known-good text server still answers `/v1/models`.

- [x] **Step 5: Run relevant surrounding checks**
  - Run all current `tests/server/test_openai_api.py`, `tests/server/test_message_wire.py`, and `tests/tokenizer/` tests.
  - Run one real local checkpoint processor check by setting its test model path to `D:\Models\Qwen3.8-Flash-Next-NVFP4`.
  - Inspect `git status --short`; `.local` must not appear and only intended Task 1 files may be staged.
  - Commit locally as `feat(server): carry bounded picture inputs` only after these checks pass.

### Task 2: Load Qwen's picture reader and merge picture features

**Outcome:** Picture mode constructs and loads the checkpoint's approximately 0.84 GiB Qwen vision stack, produces features matching Transformers on a small reference, and replaces exactly the picture-token embeddings while text mode still omits all picture weights.

**Blocked by:** Task 1.

**Files:**

- Modify: `python/freetoken/models/config.py`.
- Modify: `python/freetoken/models/qwen4_exp/config.py`.
- Create: `python/freetoken/models/qwen4_exp/vision.py`.
- Modify: `python/freetoken/models/qwen4_exp/model.py`.
- Modify: `python/freetoken/models/qwen4_exp/weight.py`.
- Modify: `tests/models/qwen4_exp/test_config.py`.
- Modify: `tests/models/qwen4_exp/test_weight.py`.
- Create: `tests/models/qwen4_exp/test_vision.py`.

**Interfaces:**

- `FREETOKEN_LOAD_VISION=0` or unset produces `ModelConfig.vision_config is None` and drops `visual.` weights.
- `FREETOKEN_LOAD_VISION=1` produces a Qwen vision configuration from `config.json`, constructs `Qwen4VisionModel`, retains `model.visual.*`/`visual.*`, and exposes `Qwen4ExpForCausalLM.encode_images(pixel_values, image_grid_thw)`.
- `Qwen4ExpModel.forward` scatters picture features before repeating embeddings across hyper-connection streams.

**Acceptance criteria:**

- [x] Parsed geometry matches the checkpoint: patch size, temporal patch size, spatial merge size, channels, depth, heads, hidden widths, position embeddings, output width, activation, and deep-stack indexes.
- [x] Picture mode remains an explicit opt-in; ordinary text configuration tests see no behavior change.
- [x] A small CPU `Qwen4VisionModel` loaded from Transformers' state dictionary matches `Qwen3VLVisionModel(...).pooler_output` within the existing PR #232 tolerance.
- [x] Meta-device construction followed by CPU loading derives non-weight rotation values correctly.
- [x] Picture features are two-dimensional `[picture_tokens, text_hidden]` and their row count must equal picture-placeholder slots.
- [x] Zero, missing, or extra picture features fail with a named mismatch before text layers run.
- [x] Picture-mode key mapping retains all 333 checkpoint vision tensors; text mode retains none.
- [x] Existing Qwen text, fused projection, NVFP4 expert, PLE, and mmap table key tests remain unchanged and pass.
- [x] Generic FTW filtering recognizes `visual.` without changing Gemma's `vision_tower.`/`embed_vision.` handling.

- [x] **Step 1: Write the failing tests**
  - Port the small PR #232 vision-reference tests into the current Qwen test directory.
  - Add environment-gated config tests and raw-name mapping tests beside current Qwen config/weight tests.
  - Add a focused embedding replacement test with known placeholder locations and independently chosen feature rows.

- [x] **Step 2: Run the focused tests before implementation**
  - Run the three Qwen config/weight/vision test files with CUDA hidden.
  - Expected current failures: Qwen config forces `vision_config=None`, `vision.py` is absent, raw visual names are discarded, and the model has no Qwen `encode_images` path.

- [x] **Step 3: Implement the smallest model slice**
  - Adapt only PR #232's `Qwen4VisionConfig` data and `vision.py` network.
  - Gate Qwen picture configuration with the existing `vision_load_enabled()` function.
  - Add `visual` construction and `encode_images` without moving or replacing current text layers, QSA, GDN, HC, MoE, PLE, mmap, or LM-head code.
  - Scatter features into the base token embeddings before `repeat(1, hc_count)`.
  - Make Qwen weight renaming accept an explicit `include_vision` decision derived from parsed config; keep text mode's drop behavior.

- [x] **Step 4: Run the focused tests again**
  - Run the Qwen config, weight, and vision tests with CUDA hidden.
  - Expected: Transformers reference, meta-load, key filtering, and embedding replacement checks pass.

- [x] **Step 5: Run relevant surrounding checks**
  - Run every test under `tests/models/qwen4_exp/` with CUDA hidden.
  - Enumerate checkpoint index keys without loading the full model; assert the picture-mode rename accepts exactly the 333 measured vision tensors and no `mtp.*` tensor.
  - Start no second server yet; confirm the original server remains healthy.
  - Commit locally as `feat(qwen4-exp): load the still-picture encoder` only after these checks pass.

### Task 3: Carry Qwen three-axis positions through prompt, decode, and graph buffers

**Outcome:** Qwen computes Transformers-equivalent picture positions, rotates main attention on the correct temporal/height/width axes, and continues generated text at the correct position through eager and captured decode. Ordinary text positions remain equivalent to the current scalar path.

**Blocked by:** Task 2.

**Files:**

- Modify: `python/freetoken/models/qwen4_exp/config.py`.
- Create: `python/freetoken/models/qwen4_exp/mrope.py`.
- Modify: `python/freetoken/models/qwen4_exp/attention.py`.
- Modify: `python/freetoken/kernel/triton/rope.py`.
- Modify: `python/freetoken/core.py`.
- Modify: `python/freetoken/scheduler/utils.py`.
- Modify: `python/freetoken/scheduler/prefill.py`.
- Modify: `python/freetoken/scheduler/scheduler.py`.
- Modify: `python/freetoken/engine/graph.py`.
- Create: `tests/models/qwen4_exp/test_mrope.py`.
- Create: `tests/scheduler/test_mrope_positions.py`.
- Create: `tests/engine/test_graph_mrope.py`.

**Interfaces:**

- `build_mrope_positions(input_ids, mm_token_type_ids, image_grid_thw, spatial_merge_size)` returns CPU `[3,prompt_tokens]` coordinates plus Qwen's scalar continuation delta.
- `Req`/`PendingReq` carry optional prompt coordinates and the delta.
- `Batch.rope_positions` is `None` for ordinary eager text batches, `[3,tokens]` for picture batches, and a persistent `[3,batch]` graph tensor during captured decode.
- Main Qwen attention accepts scalar or three-axis positions; no other model family changes its rotary call.

**Acceptance criteria:**

- [x] Position building matches Transformers `Qwen3VLModel.get_rope_index` for text-picture-text, two pictures, and generated continuation fixtures.
- [x] Marker/token length mismatches, extra/missing grids, invalid spatial divisibility, and video markers fail before GPU work.
- [x] CPU rotation matches a plain PyTorch per-axis reference for sections `(11,11,10)`.
- [x] The current tiled Triton kernel matches the CPU reference on the RTX 5090 for BF16 Q and K.
- [x] Scalar text rotation remains equal to three equal axes.
- [x] Scheduler packing handles prompt slices, generated rows, more than one request, and graph padding without shifting request boundaries.
- [x] Graph replay copies new three-axis values into stable buffers; a second replay cannot reuse the first replay's positions.
- [x] Existing text graph and rotary tests pass.

- [x] **Step 1: Write the failing tests**
  - Port/adapt PR #232's Transformers position and plain PyTorch rotation tests.
  - Add graph-buffer tests that copy two different MRoPE batches through the same persistent buffer.
  - Add invalid marker/grid cases from the approved still-picture-only boundary.

- [x] **Step 2: Run the focused tests before implementation**
  - Run the new MRoPE, scheduler-position, and graph tests with CUDA hidden.
  - Expected current failures: no `mrope.py`, no request position fields, no `rope_positions`, and graph buffers hold only scalar positions.

- [x] **Step 3: Implement the smallest position slice**
  - Parse `mrope_section=(11,11,10)` and `mrope_interleaved=true` into current `Qwen4ExpArgs`.
  - Adapt PR #232's position builder and rotation axis mapping, but integrate them into current separated `attention.py` and tiled RoPE kernel.
  - Keep logical `positions` untouched; select `rope_positions` only at Qwen rotary calls.
  - Propagate prompt coordinates and delta through pending/chunk request types even though picture prompts are later prohibited from chunking.
  - Give `GraphCaptureBuffer` a persistent `[3,max_batch]` tensor. During text replay, copy scalar positions to all axes so the captured Qwen branch is valid for both text and picture requests.

- [x] **Step 4: Run the focused tests again**
  - Run the CPU MRoPE, scheduler, and graph tests.
  - Expected: all CPU/reference cases pass; the one CUDA kernel case remains skipped because the known-good server still owns the GPU.

- [x] **Step 5: Run relevant surrounding checks**
  - Run existing kernel rotary, Qwen attention, scheduler chunked-prefill, and engine graph-related tests with CUDA hidden.
  - Confirm a normal text request to the still-running known-good server remains healthy; do not treat that old process as evidence for the edited branch.
  - Commit locally as `feat(qwen4-exp): add picture-aware MRoPE positions` after CPU checks pass.

### Task 4: Make the newer QSA path position-correct for pictures

**Outcome:** Current compressed QSA rotates picture queries and compressed keys with the correct three-axis position, including a four-token group split across prompt/decode forwards and repeated CUDA graph replay, without changing causal selection or current text output.

**Blocked by:** Task 3.

**Files:**

- Modify: `python/freetoken/kvcache/qsa_pool.py`.
- Modify: `python/freetoken/kernel/triton/qsa/compress.py`.
- Modify: `python/freetoken/kernel/triton/qsa/__init__.py`.
- Modify: `python/freetoken/attention/qsa_sparse.py`.
- Modify: `tests/kvcache/test_qsa_pool.py`.
- Modify: `tests/models/qwen4_exp/test_qsa_kernels.py`.
- Modify: `tests/models/qwen4_exp/test_qsa_backend.py`.
- Modify if needed for an independent model reference: `tests/models/qwen4_exp/test_qsa_hf.py`.

**Interfaces:**

- QSA metadata keeps `positions` as one-dimensional logical coordinates and adds/accesses current `rope_positions` separately.
- Each QSA slot receives its own pending coordinate ring shaped `[request_slots, ring_capacity, 3]` alongside its raw-key ring.
- Compression emits both the pooled raw key and the first member's three-axis coordinate.
- QSA norm/rotation accepts scalar positions for ordinary eager text and `[3,rows]` positions for picture/captured decode.
- Score, top-k, expansion, and sparse attention continue to consume logical positions.

**Acceptance criteria:**

- [x] Pending coordinate-ring shape, allocation, cleanup, rebuild, and byte accounting are correct for every QSA layer and request slot.
- [x] A complete group in one forward uses its first current row's three-axis coordinate.
- [x] A group split across forwards reads its first coordinate from the same layer's pending ring before overwriting it.
- [x] A long prefill cannot let layer 0 overwrite coordinates that a later layer still needs.
- [x] QSA fused norm plus MRoPE matches a plain PyTorch reference for query heads and pooled keys.
- [x] Scalar/equal-axis QSA output remains equal to the existing text reference.
- [x] One-shot and unaligned chunked text prefill remain equal.
- [x] Picture prompt followed by decode closes a pending group with the right first picture/text coordinate.
- [x] Two CUDA graph replays with different MRoPE values each match eager execution; no stale coordinate survives.
- [x] Existing top-k, causal visibility, page-table, pending-key, expert-offload, and mmap PLE tests do not require behavior changes.

- [x] **Step 1: Write the failing tests**
  - Extend the QSA pool test with per-layer coordinate rings.
  - Extend compression tests with independently chosen temporal/height/width rows and a split group.
  - Extend backend graph replay with two distinct position tensors.
  - Keep logical QSA selection expectations from the current CPU mirrors.

- [x] **Step 2: Run the focused tests before implementation**
  - Run QSA pool and CPU-collectable cases with CUDA hidden.
  - Expected failures: no coordinate ring, compression returns only scalar first positions, and QSA fused rotation accepts only one-dimensional positions.
  - CUDA kernel/backend cases should be recorded as `NOT RUN`/skipped until Task 6, not counted as passing.

- [x] **Step 3: Implement the smallest QSA slice**
  - Allocate one coordinate ring per QSA layer, not one globally shared ring.
  - Extend the current graph-capturable compression kernel to select the first coordinate from current rows or that layer's pending ring using the same source decision as raw keys.
  - Store current coordinates after compression, with the same ring-row mask as raw keys.
  - Add graph scratch for first coordinates and route three-axis positions only into QSA norm/rotation.
  - Leave `qsa_mqa_paged`, block top-k, `expand_qsa_block_indices`, cache slot addressing, and logical positions unchanged.

- [x] **Step 4: Run the focused tests again**
  - Run QSA pool plus CPU/reference tests with CUDA hidden.
  - Expected: pool/accounting and all CPU checks pass; GPU checks remain explicitly skipped.

- [x] **Step 5: Run relevant surrounding checks**
  - Run all Qwen/QSA tests that do not require a visible GPU.
  - Inspect changed QSA call sites to confirm every causal/cache function still receives one-dimensional `positions`.
  - Commit locally as `feat(qwen4-exp): carry MRoPE through sparse QSA` after CPU checks pass.

### Task 5: Admit picture requests safely and keep picture prompts private

**Outcome:** The scheduler turns prepared CPU tensors into Qwen picture features, rejects unsafe/too-long picture requests as request errors, carries MRoPE into generation, and never reuses one picture's prompt state for another picture.

**Blocked by:** Tasks 1-4.

**Files:**

- Modify: `python/freetoken/scheduler/scheduler.py`.
- Modify: `python/freetoken/scheduler/prefill.py` and `python/freetoken/scheduler/utils.py` only for any remaining field propagation from Task 3.
- Modify only if a failing test proves a gap: `python/freetoken/scheduler/cache.py`.
- Create: `tests/scheduler/test_multimodal_admission.py`.
- Create: `tests/scheduler/test_multimodal_cache_safety.py`.
- Modify: `tests/scheduler/test_scheduler_chunked_prefill.py`.
- Modify: `tests/scheduler/test_hybrid_cache_manager.py` if its fixtures need the new optional fields.

**Interfaces:**

- `_process_one_msg(UserMsg)` checks picture prompt length and tensor contracts before `model.encode_images` and before `PrefillManager.add_one_req`.
- Successful preparation fills `msg.mm_embeds`, `msg.mrope_position_ids`, and `msg.mrope_position_delta`, then clears all raw CPU picture tensors.
- Failed preparation sends one `ErrorReplyMsg` and returns without queueing or stopping the scheduler.
- Existing `mm_embeds` cache guards are the source of truth for private picture requests.

**Acceptance criteria:**

- [x] An 8,192-token picture prompt is accepted when the live prefill budget is 8,192; 8,193 is rejected with a clear context-length error before `encode_images` is called.
- [x] Pixel rows, grid shape/count, marker count/type, picture-token count, spatial merge, feature rank, feature width, and placeholder/feature rows are validated.
- [x] Missing picture configuration or a text-only model returns one request error.
- [x] CPU picture tensors are released on successful encode and every error path.
- [x] A bad picture request followed by a valid text request leaves the scheduler able to queue and serve the text request.
- [x] Two pictures in one request preserve grid order and feature order.
- [x] Picture requests match a zero-token shared prefix, are not inserted/donated into the shared tree, retain private active pages, and free those pages at completion.
- [x] The same token IDs with two different picture features cannot report a prefix hit in ordinary radix, hybrid/GDN, or sliding-window cache modes.
- [x] Text-only prefix matching, chunking, tool-call anchors, GDN state, sliding-window release, and finished-request donation remain unchanged.
- [x] Oversized picture prompts never reach the current `NotImplementedError` in chunked prefill.

- [x] **Step 1: Write the failing tests**
  - Build a CPU fake model with counted `encode_images` calls and deterministic features.
  - Drive the current `_process_one_msg` seam with valid, invalid, and over-budget `UserMsg` values.
  - Parameterize cache safety across the live cache-manager modes with identical token IDs and independently chosen picture features.
  - Add a prefill regression asserting oversized picture input is rejected before `_add_one_req` can attempt chunking.

- [x] **Step 2: Run the focused tests before implementation**
  - Run the two new scheduler files and affected prefill/cache files with CUDA hidden.
  - Expected failures: online picture fields are not prepared, length is not checked before the current chunking error, and MRoPE data is not populated by the scheduler.
  - Existing cache-private behavior may already pass; preserve it rather than rewriting it.

- [x] **Step 3: Implement the smallest admission slice**
  - Validate input length first, then prepared tensor contracts, then call the model, then build MRoPE, then clear CPU inputs in a `finally`-safe path.
  - Convert all input-driven failures to `ErrorReplyMsg`; do not catch process-control exceptions outside the request preparation boundary.
  - Pass resulting fields into `PendingReq`/`Req` and retain `mm_embeds` until request cleanup so cache guards remain active.
  - Change `cache.py` only if a parameterized cache test finds a current path that ignores its existing `mm_embeds` guard.

- [x] **Step 4: Run the focused tests again**
  - Run admission, cache safety, chunked prefill, hybrid cache, abort, and message error-recovery tests with CUDA hidden.
  - Expected: all focused checks pass and raw tensor references are absent after both success and failure.

- [x] **Step 5: Run relevant surrounding checks**
  - Run the full CPU-visible `pytest tests -m 'not slow'` command with CUDA hidden.
  - Record GPU cases as skipped and list them for Task 6; do not call the suite fully passing if a required CPU test fails.
  - Confirm the known-good live server still passes its existing public smoke test from the original worktree.
  - Commit locally as `feat(scheduler): admit picture requests safely` only after CPU regressions pass.

### Task 6: Prove the RTX 5090 server, all picture sources, rollback, and pi

**Outcome:** The local picture branch starts at full text context with SSD-backed PLE, passes required GPU/reference and public-API checks, records actual memory/expert-cache impact, and completes one real pi picture request. Any failure returns the machine to the proven text server.

**Blocked by:** Tasks 1-5 and their CPU checks.

**Files:**

- Create: `benchmarks/run_qwen38_vision_smoke.py`.
- Create locally/ignored: `.local\vision-fixtures\*`.
- Create locally/ignored: `.local\qwen38-vision-acceptance.json`.
- Modify only after direct checks pass: `C:\Users\jay\.pi\agent\models.json`.
- Update after evidence: `docs/design/2026-08-30-qwen38-picture-input-design.md` status and this plan's checkboxes; keep these internal and uncommitted.

**Interfaces:**

- Launch: `scripts/start-qwen38-flash-next-mmap-windows.ps1 -EnableVision -EnableCacheReport`.
- Direct checks: `/v1/models`, `/v1/cache/status`, and `/v1/chat/completions` on `127.0.0.1:2020`.
- Pi check: documented print mode with `@picture.png` and `freetoken-local/Qwen3.8-Flash-Next-NVFP4`.
- Rollback: original worktree's launcher without `-EnableVision`.

**Acceptance criteria:**

- [x] Required RTX tests for tiled MRoPE, QSA norm/MRoPE, split pending groups, and two-value CUDA graph replay run and pass after the old server releases the GPU.
- [x] Startup reaches serving with `FREETOKEN_LOAD_VISION=1`, all 333 picture tensors loaded, and no missing/unexpected key.
- [x] Startup still reports `ple_backend='mmap'`/mmap PLE and serial NVFP4 expert loading.
- [x] `/v1/cache/status` reports 262,208 total KV tokens and 262,144 usable tokens.
- [x] Actual GPU-cached expert count is recorded; a reduction from 4,063 is acceptable.
- [x] Startup time, peak whole-GPU memory, peak whole-system memory, short text speed, and picture-request timings are recorded from live data.
- [x] Embedded data, direct Windows path, `file://`, and loopback HTTP picture sources each produce the fixture's exact large code text.
- [x] One message with two distinct pictures identifies both.
- [x] Two otherwise identical prompts with different pictures identify the second picture correctly and each report zero cached prompt tokens.
- [x] Malformed base64, invalid picture bytes, a source over 64 MiB, a redirect overflow, and an unreadable file each return a clear error; `/v1/models` and a text request still succeed afterward.
- [x] Existing text, sequential cache comparison, reasoning stream, usage, parsed tool call, and final health smoke checks pass.
- [x] A 32,768-token ordinary text check succeeds, proving text chunking remains active beyond the picture limit.
- [x] Pi lists the model as text-and-picture, reads one generated fixture, and returns its exact code text.
- [x] Pi's prior default provider/model/thinking values remain unchanged.
- [x] The original worktree remains at `14ee7b0` and can restart the known-good text server.
- [x] No picture commit is pushed and no public/upstream action occurs.

- [x] **Step 1: Write the failing live smoke helper**
  - Generate deterministic PNG fixtures with large code words and distinct colored shapes.
  - Start an in-process loopback HTTP fixture server for normal, redirect-chain, and oversized routes.
  - Send data, direct path, `file://`, HTTP, multiple-picture, cross-picture, invalid, oversized, sequential text, short speed, and 32K text requests.
  - Poll model/cache health after expected request failures.
  - Sample `nvidia-smi` whole-GPU use and Windows whole-system memory during requests; write raw samples and summaries to the ignored JSON path.
  - Expected before implementation: helper is absent and the current text server rejects the first `image_url` request.

- [x] **Step 2: Run all required GPU-focused tests before full startup**
  - Stop the known-good supervised server only now.
  - Remove `CUDA_VISIBLE_DEVICES`.
  - Run:

    ```powershell
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    Set-Location $VisionRoot
    & $DesktopPython -m pytest `
      tests/models/qwen4_exp/test_mrope.py `
      tests/models/qwen4_exp/test_qsa_kernels.py `
      tests/models/qwen4_exp/test_qsa_backend.py `
      tests/models/qwen4_exp/test_qsa_hf.py
    ```

  - Expected: every required RTX case runs rather than skips and matches its independent CPU/eager/Transformers reference.
  - If any required test fails, do not start picture workers; restart the known-good launcher immediately and record the failure.

- [x] **Step 3: Start the smallest complete live picture server**
  - Parse/compile the launcher, shim, and smoke helpers.
  - Start as a supervised task:

    ```powershell
    & 'D:\FreeToken-ple-mmap-vision\scripts\start-qwen38-flash-next-mmap-windows.ps1' `
      -ModelPath 'D:\Models\Qwen3.8-Flash-Next-NVFP4' `
      -Port 2020 `
      -ContextTokens 262144 `
      -MaxRunningRequests 1 `
      -EnableVision `
      -EnableCacheReport
    ```

  - Wait for readiness or a terminal worker error. Do not lower context, disable mmap, disable graphs, or manually force an expert count to make startup pass.
  - Capture startup time and the actual loaded picture/PLE/expert/cache facts from logs and health endpoints.
  - On startup failure or out-of-memory, stop every picture worker and restart the original launcher's text mode; retain the picture branch for diagnosis.

- [x] **Step 4: Run direct acceptance and text regressions**
  - Run the new vision smoke helper and write `.local\qwen38-vision-acceptance.json`.
  - Run the existing public text smoke helper from the picture worktree:

    ```powershell
    & $DesktopPython "$VisionRoot\benchmarks\run_qwen38_mmap_smoke.py" `
      --base-url http://127.0.0.1:2020/v1 `
      --model Qwen3.8-Flash-Next-NVFP4
    ```

  - Independently inspect acceptance JSON, `/v1/cache/status`, process health, and the server log for all criteria above.
  - If any direct check fails, stop picture mode and restore the original text server. Do not enable picture input in pi.

- [x] **Step 5: Enable and prove pi picture input last**
  - Save a local backup of `C:\Users\jay\.pi\agent\models.json` under ignored `.local` evidence.
  - Change only the existing model's `input` list to `["text", "image"]`.
  - Parse the JSON and assert the existing default provider, default model, and default thinking level in `settings.json` are unchanged.
  - From the fixture directory, run pi's documented picture command:

    ```powershell
    Set-Location "$VisionRoot\.local\vision-fixtures"
    pi --provider freetoken-local `
      --model Qwen3.8-Flash-Next-NVFP4 `
      --thinking low `
      --no-session --no-context-files --no-skills --no-prompt-templates --no-tools `
      -p @pi-picture.png 'Read the large code in this picture. Reply with only that code.'
    ```

  - Expected: pi sends a picture-bearing OpenAI request and the answer contains only the fixture's exact code.
  - Run `pi --list-models freetoken-local` and one final direct text health request.
  - If pi fails, restore the backed-up text-only model declaration and restart the known-good text server as required by the approved design.
  - If all checks pass, leave the picture server available locally, commit the tracked smoke helper as `test(windows): add Qwen3.8 picture acceptance`, and do not push.

## Rollback recipe

Use after any Task 6 failure:

```powershell
# Stop the supervised picture task and any remaining child workers first.
Set-Location 'D:\FreeToken-ple-mmap'
git branch --show-current
git rev-parse HEAD
& 'D:\FreeToken-ple-mmap\scripts\start-qwen38-flash-next-mmap-windows.ps1' `
  -ModelPath 'D:\Models\Qwen3.8-Flash-Next-NVFP4' `
  -Port 2020 `
  -ContextTokens 262144 `
  -MaxRunningRequests 1
```

Then run the existing public text smoke. Expected rollback evidence is branch `windows-ple-mmap`, commit `14ee7b0dc8c8d97e9fd18776fe71e6a80b0dd1bb`, 262,144 usable tokens, mmap PLE, and every text smoke section passing.

The picture worktree and local branch may remain for diagnosis. Removing them is not part of rollback and must not be done while a process uses them.

## Checkpoint commits

Create commits only in `windows-ple-mmap-vision`, only after each task's checks pass:

1. `feat(server): carry bounded picture inputs`
2. `feat(qwen4-exp): load the still-picture encoder`
3. `feat(qwen4-exp): add picture-aware MRoPE positions`
4. `feat(qwen4-exp): carry MRoPE through sparse QSA`
5. `feat(scheduler): admit picture requests safely`
6. `test(windows): add Qwen3.8 picture acceptance`

Before each commit, stage exact paths and inspect `git diff --cached --stat` plus `git diff --cached`. Never stage `.local`, model files, logs, pi files, context/planning files, installed package trees, generated kernels, or caches. Do not push any checkpoint.

## Final evidence record

`.local\qwen38-vision-acceptance.json` must record facts rather than estimates:

- branch and commit IDs;
- Desktop Torch/Transformers/Pillow/TorchVision versions;
- picture tensor count and bytes;
- launcher arguments and loopback address;
- startup seconds;
- total and usable KV tokens;
- GPU-cached expert count;
- mmap PLE confirmation;
- each source-form result and timing;
- two-picture and cross-picture outputs plus cached-token reports;
- invalid-input statuses and post-error health;
- text smoke summary, short decode speed, and 32K text check;
- whole-GPU and whole-system memory samples/peaks;
- pi command result without session contents or unrelated settings;
- rollback status (`available`, `used-and-passed`, or `not-needed`).

Do not commit this machine-local evidence.

## Self-review

- **Coverage:** Every approved picture source, multiple-picture case, cache-safety rule, one-batch limit, dependency boundary, picture model step, MRoPE step, live memory fact, text regression, full-context fact, mmap requirement, and pi check maps to an explicit task and acceptance check.
- **Evidence:** Current paths and test commands come from the checkout and `tests/README.md`; picture model/position behavior comes from PR #232 and Transformers 5.15.1; QSA changes are based on the current `qsa_sparse.py`, compressed pending ring, and CUDA graph path; pi's image command comes from its installed README and model configuration documentation.
- **Interfaces:** Task 1 produces the exact wire fields Task 5 consumes; Task 2 produces `encode_images`; Task 3 produces `rope_positions`; Task 4 makes current QSA consume them safely; Task 5 joins the request; Task 6 crosses the real HTTP and pi boundaries.
- **Dependencies:** Tasks form an acyclic chain. The known-good server stays running through CPU work and is stopped only for required GPU tests and full model startup.
- **Rollback:** A separate worktree keeps product files for `14ee7b0` untouched. Every live failure path names the same immediate restoration procedure.
- **Scope:** No video, Anthropic picture support, public release, Desktop edit, model edit, network exposure, upstream action, older text-engine replacement, reduced context, or non-mmap PLE workaround is included.
- **Test quality:** Position/network/model/kernel expectations use Transformers, PyTorch, exact byte fixtures, round trips, eager execution, or live API results instead of restating implementation branches.
- **Fresh-context readiness:** Every task names files, interfaces, blockers, failing evidence, smallest implementation, focused rerun, surrounding checks, expected results, commit boundary, and failure action. No load-bearing design choice remains open.
