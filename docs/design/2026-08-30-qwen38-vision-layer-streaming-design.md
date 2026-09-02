# Qwen3.8 CPU-resident, layer-streamed picture reader on Windows

## Status

Implemented and accepted locally on 2026-08-30. Final implementation head: `c5876d5` on `windows-ple-mmap-vision`. After acceptance, Jay separately authorized a fast-forward publication to `dejay2/FreeToken:windows-ple-mmap`; this supersedes the design's original no-publication boundary but does not make private evidence public. Startup placed all 333 picture tensors (897,862,112 bytes) on CPU, restored 4,063 cached experts, and retained exactly 262,144 usable tokens. The exact three-run 512-token benchmark averaged 50.58 output tokens/second before picture work and 52.10 after two final picture requests. The retained screenshot encoded in 1.519 and 0.724 seconds live, and Jay's final normal pi checks returned `gpt-5.6-luna` and `PI_TEXT_HEALTH_OK`.

## Purpose

Restore normal Qwen3.8 language-generation speed while retaining still-picture input and exactly 262,144 usable context tokens. The complete Qwen picture reader currently occupies about 0.836 GiB of RTX 5090 memory, reducing the GPU-resident language-expert cache from 4,063 to 3,727 entries. The exact pre-picture 512-token benchmark averaged 54.50 output tokens/second; the same benchmark on the accepted all-GPU picture server averaged 7.00 output tokens/second.

Keep every picture weight resident in ordinary system memory. During picture encoding, copy only one picture component or transformer block at a time into one reusable RTX 5090 workspace. The language model, KV cache, recurrent state, PLE staging, CUDA graphs, and expert cache remain on their established devices.

## Users

- Jay uses `freetoken-local/Qwen3.8-Flash-Next-NVFP4` for normal coding and occasional still-picture prompts in pi.
- Direct loopback OpenAI-compatible callers use the same picture path.

Normal language work is the priority. The selected acceptance targets are at least 50 output tokens/second on the existing deterministic 512-token benchmark and no more than six seconds to encode the retained 1920×1280 screenshot.

## Evidence baseline

- Known-good text rollback: `D:\FreeToken-ple-mmap`, `windows-ple-mmap`, commit `14ee7b0`.
- Active private picture checkout: `D:\FreeToken-ple-mmap-vision`, `windows-ple-mmap-vision`, current scheduler checkpoint `56a34ec`.
- Picture weights: 333 tensors, 897,862,112 bytes (about 0.836 GiB), checkpoint BF16.
- Current engine behavior is not CPU placement:
  - `Engine._materialize_loaded_weight_state_dict` copies every dense state tensor, including `visual.*`, to the engine CUDA device.
  - `Scheduler._prepare_multimodal_request` copies picture pixels and grids to that CUDA device before `encode_images`.
  - Matched ~2,480-token probes used 99-100% GPU activity. Text averaged 1.51 seconds and picture averaged 1.95 seconds, so the picture reader added about 0.44 seconds while running on the RTX 5090.
- Exact language regression:
  - picture-disabled/full-context published baseline: 4,063 cached experts, 54.50 output tokens/second;
  - current all-GPU picture server: 3,727 cached experts, 7.00 output tokens/second, with first-token time changing only from about 2.11 to 2.32 seconds.
- User-approved throwaway placement probes used the retained screenshot, 9,600 input patches, and 2,400 output picture rows:
  - complete CPU execution: 16.24 seconds;
  - CPU-resident weights with one GPU layer at a time: 4.15 seconds;
  - streamed probe Torch peak: 429,316,608 allocated bytes and 648,019,968 reserved bytes;
  - output comparison: cosine similarity 0.999537, mean absolute difference 0.000543, all values finite.
- The local server is intentionally stopped at the start of planning and must not be restarted except by the implementation/validation sequence or Jay's request.

## Scope

### Included

- Keep all `visual.*` checkpoint tensors permanently in ordinary CPU memory.
- Keep all language-model tensors on the RTX 5090 exactly as before.
- Add an explicit layer-streamed picture execution mode.
- Reuse one GPU transformer-block workspace across all 27 picture blocks.
- Stage the patch projection, position embedding work, final merger, and any configured deep-stack mergers without retaining their complete checkpoint weights on the GPU.
- Keep intermediate picture activations on the GPU while advancing through the streamed blocks.
- Return final picture features on the GPU for existing placeholder replacement and chunked prompt loading.
- Retain the existing all-GPU picture mode as an explicit fallback/comparison mode.
- Preserve picture prompt chunking, three-axis positions, QSA correctness, private caching, source bounds, and still-picture-only scope.
- Measure language speed, picture time, CPU/GPU memory, expert count, and output agreement.

### Excluded

- Full-CPU production picture execution.
- Quantizing picture weights.
- Keeping two GPU block workspaces or overlapping transfer of the next block with current computation in the first implementation.
- Moving language layers, routed experts, KV state, GatedDeltaNet state, PLE tables, or CUDA graphs to new devices.
- Lowering the 262,144-token usable context to fund picture memory.
- Increasing one prompt-loading step above 8,192 tokens.
- Picture resizing/cropping changes, video, Anthropic picture input, public serving, publication, or upstream work.

## Success criteria

- All 333 picture tensors are loaded into CPU memory and no persistent `visual.*` tensor is CUDA-resident after startup.
- Automatic sizing restores 4,063 cached experts, or another automatically resolved count that still passes the required speed target without reducing context.
- `/v1/cache/status` reports 4,097 pages of 64 tokens: 262,208 total allocated and exactly 262,144 usable tokens.
- The exact existing 512-input/512-output deterministic benchmark averages at least 50 output tokens/second across its three recorded runs.
- The retained 1920×1280 screenshot's picture-encoder call completes in at most six seconds.
- Streamed full-checkpoint picture features are finite and agree with the accepted all-GPU path closely enough to preserve the independently known screenshot and deterministic fixture answers.
- Direct screenshot, normal pi screenshot, 33K picture prompt, multiple pictures, cache privacy, streaming, reasoning, tool calls, malformed-picture recovery, and post-picture text health all pass.
- Repeated pictures reuse bounded temporary GPU memory; they do not grow the allocator, retain a staged layer, or reduce the expert cache after completion.
- System memory remains within the current machine's capacity and the server remains healthy.
- No Desktop package, model file, pi default, rollback checkout, or public branch is changed.

## Constraints

- Native Windows, RTX 5090 32 GB, Ryzen 9 9950X3D, about 96 GB RAM.
- Keep `127.0.0.1:2020`, one active request, mmap PLE, MoE offload, serial expert loading, automatic expert sizing, and `--kv-reserve-tokens 262144`.
- Do not pin the complete 0.836 GiB picture state by default. Existing pinned expert banks already consume substantial host resources; ordinary pageable CPU tensors are the safe baseline.
- The temporary GPU workspace is created only for picture encoding and must fit after all normal runtime pools and CUDA graphs exist.
- Do not reserve permanent GPU memory equal to the whole picture reader; that would recreate the expert-cache regression.
- Do not catch a CUDA failure and silently continue on a different placement. Return one picture error where the CUDA state remains reliable; otherwise execute the rollback procedure.
- Keep the picture branch and all evidence local and unpublished.

## Chosen approach

### Explicit execution mode

Add an execution setting with at least these values:

- `gpu`: current behavior, all picture weights resident and executed on the engine CUDA device;
- `layer-stream`: selected behavior, all picture weights resident on CPU and one component/block staged at a time.

The Windows vision launcher selects `layer-stream` for this local setup. Picture loading remains gated by `FREETOKEN_LOAD_VISION=1`. Text-only startup continues to construct no picture reader and require no picture packages.

### Key-aware weight placement

The generic engine currently materializes every loaded state tensor onto one device. Add a narrow model-owned placement seam: before materialization, the engine asks the model (when it provides the hook) for the destination of a state key. Qwen returns CPU only for `visual.*` in `layer-stream` mode and returns the engine device for every other key. Models without the hook retain current behavior.

The model's state loader then installs CPU `visual.*` tensors into the picture reader while language tensors remain CUDA tensors. GPU-weight accounting and automatic expert-cache sizing naturally see only actual GPU-resident dense weights.

Do not hard-code Qwen `visual.*` policy into unrelated model families.

### One reusable GPU block

All 27 Qwen picture transformer blocks have the same shape. Allocate one GPU block workspace for a picture request. For each CPU block in order:

1. Copy that block's CPU tensors into the existing GPU workspace tensors in place.
2. Run the block on the current GPU hidden state.
3. Overwrite the same workspace with the next block.

Do not construct, garbage-collect, and empty the CUDA allocator for every layer as the throwaway probe did. Reuse must remove that probe overhead and keep allocation bounded.

Patch projection, position preparation, final merger, and optional deep-stack mergers use bounded staging objects appropriate to their different shapes. Their checkpoint tensors remain CPU-resident outside the active encode. Qwen3.8 currently has no deep-stack merger indexes, but the implementation must either support configured indexes correctly or reject the unsupported configuration clearly; it must not silently omit features.

### Data movement

- Tokenizer picture tensors arrive on CPU as they do now.
- In `layer-stream` mode, keep pixels and grid values on CPU at scheduler admission.
- The picture reader stages pixels, necessary position values, and component weights to the engine CUDA device.
- Picture hidden states remain on CUDA through the 27 blocks.
- Final merged picture features remain CUDA tensors and enter the unchanged per-step picture-feature slicing path.
- Raw CPU picture tensors are released on success and failure as before.
- Complete final picture features remain request-private until chunked prefill finishes or is cancelled.

### Workspace lifetime and cleanup

The server allows one active request, so one request-local workspace is sufficient and no concurrent picture encodes may overlap. Use explicit `try/finally` cleanup. Drop all staged-component references after encode or error. Release allocator cache only at the end of the whole picture encode when needed; never between identical blocks.

Record allocated/reserved memory before and after two sequential picture requests. The second request may reuse allocator storage but must not increase the steady-state high-water mark or leave persistent component weights that reduce language speed.

## Important interfaces and seams

- `python/freetoken/engine/engine.py`
  - materialize state tensors using an optional model-owned destination hook;
  - keep GPU weight accounting based on actual CUDA allocations.
- `python/freetoken/models/qwen4_exp/model.py`
  - expose Qwen picture-key placement for `layer-stream`;
  - route `encode_images` to current all-GPU or streamed execution;
  - keep returned features on the engine device.
- `python/freetoken/models/qwen4_exp/vision.py`
  - add reusable in-place state copying and the layer-stream execution path;
  - retain the current all-GPU forward as the independent reference/fallback.
- `python/freetoken/scheduler/scheduler.py`
  - stop unconditionally moving raw pixels/grids to the language device;
  - let the model's execution mode own source and destination placement;
  - preserve request isolation and raw-tensor cleanup.
- `python/freetoken/models/qwen4_exp/config.py` or a narrow runtime configuration seam
  - parse and validate the explicit execution mode without changing checkpoint geometry.
- `scripts/start-qwen38-flash-next-mmap-windows.ps1`
  - select and validate `layer-stream` only with `-EnableVision`;
  - preserve the existing all-GPU option and all text defaults.
- Existing feature slicing, MRoPE, QSA, cache privacy, PLE, GDN, expert-offload, and graph paths receive no placement-specific semantics.

## Error and cancellation behavior

- Unsupported execution values fail at startup with a clear message.
- A CPU/GPU state-key mismatch fails during strict model loading.
- A staged component shape/dtype/device mismatch fails before its computation.
- Picture execution errors remain one terminal request error and release raw pixels, staged objects, temporary activations, complete picture features, and pending position state.
- Cancellation after picture features exist follows the accepted private chunked-request cleanup.
- Do not free or resize language expert/KV pools during picture encoding.
- A failed picture must be followed by a direct text-health request during validation.

## Security and privacy

The existing local-only picture-source policy is unchanged. The server remains bound to loopback. CPU picture weights are immutable checkpoint data. Picture pixels, intermediate activations, final features, KV pages, and recurrent state remain request-private and are released under the existing lifecycle. No new network or file access is introduced.

## Testing strategy

### Focused tests without the full checkpoint

- Key-aware materialization sends only Qwen `visual.*` tensors to CPU in `layer-stream`; every language key and every model without a placement hook follows the current device.
- Strict state loading accepts mixed CPU/CUDA state only in the selected mode.
- A small picture model's streamed output matches its all-GPU output within an independently chosen BF16 tolerance.
- Numbered block weights prove the same GPU workspace is overwritten in exact layer order.
- Two successive encodes reuse one workspace and clear all staged references on success and injected failures.
- Pixels remain CPU at scheduler admission in `layer-stream`; all-GPU mode retains current movement.
- Text-only configuration constructs no picture workspace and runs existing tests unchanged.
- Unsupported deep-stack geometry is handled explicitly.

### RTX 5090 checks

1. Run all current MRoPE, QSA, graph, cache, and picture GPU tests.
2. Start `layer-stream` with unchanged context/mmap/offload settings.
3. Verify every persistent picture tensor is CPU-resident, every language tensor has its expected device, expert count is restored, and exact context geometry remains.
4. Run the deterministic old 512-token benchmark and require at least 50 output tokens/second.
5. Time picture encoding separately for the retained screenshot and require at most six seconds.
6. Compare streamed and all-GPU feature/output references.
7. Send two sequential screenshots and inspect memory high-water behavior.
8. Run direct picture acceptance, the 33K chunked picture, source/error/privacy tests, text/reasoning/streaming/tool smoke, and post-error health.
9. Run a normal pi screenshot and subsequent pi text request without disabling normal context, skills, templates, or extensions.
10. Record all local evidence and keep the branch unpublished.

## Alternatives considered

### Keep the complete picture reader on GPU

Rejected for this machine's full-context setup. It is fast for pictures but reduced the expert cache to 3,727 and the exact language benchmark to 7.00 tokens/second.

### Run the complete picture reader on CPU

Rejected as the primary path. It produced equivalent features but required 16.24 seconds for the retained screenshot, exceeding the selected six-second target.

### Quantize picture weights

Deferred. It changes numerical behavior and checkpoint handling while providing less certain memory recovery than CPU residency.

### Two workspaces with transfer/compute overlap

Deferred. It increases temporary VRAM and complexity. Implement one reusable workspace first; optimize only if the six-second target is missed.

## Rollback

The known-good text checkout remains `D:\FreeToken-ple-mmap` at `14ee7b0`. On startup, memory, speed, correctness, or normal-pi failure:

1. stop every edited picture worker and verify port 2020 is free;
2. restore pi's text-only declaration if picture reliability is lost;
3. start the known-good text launcher from the rollback checkout;
4. require its text smoke and 262,144-token geometry;
5. keep all placement work local and publish nothing.
