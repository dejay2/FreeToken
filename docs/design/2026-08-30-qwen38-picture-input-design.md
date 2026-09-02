# Qwen3.8 picture input on Desktop-assisted Windows

## Status

Implemented and accepted locally on 2026-08-30 at first-milestone head `66c425c`. Direct OpenAI picture, long-text, error-recovery, cache-privacy, text-regression, and real pi picture checks passed. Its original one-step picture ceiling was superseded by chunking at `56a34ec`, and its all-GPU picture placement was superseded by the accepted CPU-resident layer-stream path at `c5876d5`. Final ignored evidence: `D:\FreeToken-ple-mmap-vision\.local\qwen38-vision-acceptance.json`.

## Purpose

Add still-picture input to `RadixArk/Qwen3.8-Flash-Next-NVFP4` on the local `windows-ple-mmap` fork while preserving the known-good text server, SSD-backed PLE, native Windows operation, and 262,144 usable text tokens.

## Users

- Jay uses pictures with the existing `freetoken-local` model in pi.
- Jay may send direct requests to FreeToken's OpenAI-compatible local address for source forms that pi does not create.
- This is a local experiment, not an upstream or public-fork release.

## Evidence baseline

- Known-good local and public branch: `windows-ple-mmap` at `14ee7b0`.
- SSD-backed PLE base: FreeToken PR #279 at `feaeaa3`.
- Reusable picture implementation: closed FreeToken PR #232 at `ad752c9`.
- PR #232 passed real Qwen3.8 image requests on RTX 3090, RTX 4090, and RTX 5090 systems.
- PR #232 was closed as superseded by text-only PR #257; its maintainer said vision work would follow.
- The local checkpoint contains 333 vision tensors totaling 897,862,112 bytes, about 0.84 GiB.
- The Desktop Python currently lacks Pillow and TorchVision. It has Transformers 5.15.1 and `Qwen3VLProcessor`.
- pi's model messages support text and pictures, not video.

## Scope

### Included in the picture path

- OpenAI-compatible `/v1/chat/completions` picture parts.
- One or more still pictures in one user message.
- Picture bytes carried in a `data:` URL.
- `http://` and `https://` sources.
- `file://` sources and direct Windows paths supplied as the picture source value.
- Common still-picture formats that Pillow can decode and convert to RGB.
- Qwen picture preprocessing, vision-weight loading, vision execution, picture-token replacement, and picture-aware position values.
- Direct local-address tests and one real pi picture test.
- Existing text, reasoning, streaming, tools, health, context, and PLE checks.
- Full-context allocation from the first live picture test.

### Excluded from the picture path

- Video input.
- Native video messages in pi.
- Animated-image timing; an animated file is treated as one still picture.
- Anthropic-compatible image content.
- Authentication or safe exposure on a shared network.
- A public push, pull request, issue, or upstream comment.
- Replacing the current Qwen text implementation with the older PR #232 implementation.
- Multimodal prompts longer than one configured prefill batch.

## Required behavior

### Picture source forms

Accept the PR #232 content shapes:

```json
{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
```

```json
{"type":"image_url","image_url":{"url":"https://example.test/picture.png"}}
```

Also accept `file://` URLs and direct Windows paths in the same `url` value. Preserve support for the equivalent `image` field handled by PR #232 when this does not create an ambiguous source.

Use the existing 64 MiB source-byte limit from PR #232 for every carried, downloaded, or local picture. Apply a finite network timeout and a bounded redirect count. Let the Qwen processor enforce the checkpoint's configured processed-picture range: 65,536 through 16,777,216 pixels.

Any readable local path and any HTTP(S) address are intentionally allowed. This is the user-approved broad picture source access policy. It is acceptable only because the launcher binds to `127.0.0.1` and allows one active request. Do not represent this policy as safe for a public or shared server.

### Picture preparation

Use the checkpoint's `Qwen3VLProcessor` and its checked-in `preprocessor_config.json`. Produce and validate:

- text token IDs;
- picture pixels;
- picture grid shape;
- token-type markers identifying text and picture positions.

Convert pictures to RGB and close all files and network responses promptly. Return one request error for invalid base64, unsupported schemes, oversized sources, network timeout, unreadable local files, invalid pictures, processor failure, or mismatched output shapes. A bad picture must not stop a worker or the server.

### Worker handoff

Carry the prepared picture values from the tokenizer worker to the GPU scheduler using explicit optional fields based on PR #232. Text-only messages leave those fields absent and follow the current path unchanged.

Validate tensor sizes and types before GPU work. Release CPU picture tensors after the scheduler has produced the model's picture features.

### Qwen model integration

Port only the vision-specific behavior from PR #232:

- retain `vision_config` instead of forcing it to `None` for Qwen3.8;
- instantiate the Qwen vision model;
- stop discarding `model.visual.*` and `visual.*` tensors;
- map the checkpoint's vision names to the current Qwen model without changing current text, PLE, attention, MoE, or output-layer mappings;
- run the vision model during prompt processing;
- replace only the checkpoint's picture-token positions with the resulting picture features;
- reject any mismatch between picture-token slots and produced picture features;
- use Qwen's three-axis picture position calculation from PR #232, while continuing the correct position value during generated text.

Do not cherry-pick or copy PR #232 wholesale. Its Qwen text engine was superseded by PR #257 and would replace newer native-context, CUDA-graph, hybrid-prefix-cache, and PLE behavior.

### Prompt reuse correctness

Different pictures can have identical picture-placeholder token IDs. Reusing prompt memory by token IDs alone could answer from the wrong picture.

Port PR #232's safe behavior across the current hybrid, sliding-window, and ordinary cache paths:

- picture requests match an empty reusable prefix;
- picture requests are not inserted into the shared prefix cache;
- their pages remain private for the active request and are freed at completion;
- text-only prefix reuse remains unchanged.

Add a focused check using two different pictures with otherwise identical text. The second request must process its own picture rather than reuse the first picture's saved prompt state.

### Picture prompt length

A picture request must fit in one configured prompt-processing batch. For the current server, `max_extend_tokens` defaults to 8,192.

After picture preparation, reject a request above that limit with a clear request error before scheduler admission. Do not let the existing scheduler `NotImplementedError` stop a worker. Ordinary text requests retain the full 262,144-token usable allocation and existing chunked processing.

### Memory and speed

Keep these launch requirements:

- `--ple-backend mmap`;
- `--moe-backend offload`;
- automatic expert-cache sizing;
- `--kv-reserve-tokens 262144`;
- one active request;
- `127.0.0.1` only.

Load about 0.84 GiB of vision weights before final automatic expert-cache sizing, or explicitly reserve equivalent GPU room if the current sizing order makes that necessary. Record the resulting expert-cache count and peak GPU/system memory.

A lower expert-cache count and slower text generation are acceptable. Reduced usable context, PLE leaving mmap mode, worker crashes, or unstable sequential requests are not acceptable.

### Dependency delivery

Do not install or modify packages inside `C:\Users\jay\AppData\Local\FreeToken`.

Provide Pillow 11–12 and TorchVision 0.26 for the Desktop Python through a checkout-local package directory or another project-local overlay. TorchVision must match the Desktop runtime's Torch 2.11 and CUDA 13 build. The launcher must add the local package location before starting FreeToken and fail clearly when the picture packages are absent or incompatible.

Do not add downloaded wheels, installed package trees, or generated caches to git.

### pi integration

Keep pi's existing default provider, model, and reasoning setting unchanged.

After every direct FreeToken acceptance check passes, change only the existing `freetoken-local/Qwen3.8-Flash-Next-NVFP4` entry from text-only input to text-and-picture input. Confirm a real pi picture request. If that test fails, restore the text-only pi declaration and the known-good server.

## Important seams to preserve

- `python/freetoken/tokenizer/server.py`: structured content parsing and picture preparation.
- Message classes between tokenizer and scheduler: optional prepared picture values.
- `python/freetoken/scheduler/`: vision execution, private picture-request caching, and one-batch length enforcement.
- `python/freetoken/models/qwen4_exp/`: vision configuration, model, positions, and weight mappings.
- `scripts/start-qwen38-flash-next-mmap-windows.ps1`: project-local picture libraries and unchanged Windows/PLE launch behavior.
- `scripts/windows-ple-mmap/sitecustomize.py`: Windows bridge; change only if new picture libraries expose a proven Windows incompatibility.
- `C:\Users\jay\.pi\agent\models.json`: picture capability after server acceptance only.

## Work isolation and rollback

1. Keep `windows-ple-mmap` and public commit `14ee7b0` unchanged.
2. Create a new local branch for picture work.
3. Keep the known-good server serving while code and non-GPU checks run.
4. Stop it only when the picture branch is ready for a live test.
5. If startup or acceptance fails, stop picture workers, return to `windows-ple-mmap`, and restart the known-good launcher.
6. Do not push the picture branch unless Jay gives separate approval after all evidence is reviewed.

## Testing strategy

### Checks without the full model

- Extract and decode a valid picture data URL.
- Reject malformed base64 and a source over 64 MiB.
- Read `file://` and direct Windows paths.
- Fetch an HTTP(S) picture with size, timeout, and redirect bounds.
- Reject an invalid picture without stopping the tokenizer.
- Preserve structured picture parts until the picture processor sees them.
- Verify processor output fields, types, and matching lengths with a small fixture or controlled substitute.
- Verify picture position calculations against Transformers or PR #232 fixtures.
- Verify picture weight names are retained while unrelated and text names keep current mappings.
- Verify picture requests skip shared prefix matching and insertion in every cache mode.
- Reject picture prompts over `max_extend_tokens` before scheduler admission.
- Confirm existing text-only focused tests still pass.

### Live RTX 5090 checks

Use one generated still picture containing large exact text and distinct colored shapes. Reuse it through each source path so failures identify transport rather than model quality.

1. Start with 262,144 usable tokens and SSD-backed PLE.
2. Read the exact large text from an embedded data picture.
3. Read it from a local file.
4. Read it from an HTTP address.
5. Identify two different pictures in one message.
6. Send two same-prompt requests with different pictures and verify the second picture is not confused with the first.
7. Send invalid and oversized picture requests; verify clear errors and continued health.
8. Run the existing text, reasoning, streaming, tool, context, and health smoke checks.
9. Confirm `/v1/models` and `/v1/cache/status` still report 262,144 usable tokens.
10. Confirm startup output still reports mmap PLE.
11. Record startup time, cached experts, peak GPU memory, peak system memory, and one short text speed check.
12. Enable picture input in pi and complete one real pi picture request.

## Acceptance criteria

The picture path is complete only when:

- all three picture source families work;
- multiple pictures work;
- pi sends and receives one real picture request;
- different-picture prompt reuse is correct;
- invalid picture input returns an error without stopping the server;
- the existing smoke suite passes;
- 262,144 usable text tokens remain allocated;
- SSD-backed PLE remains active;
- FreeToken Desktop and model files remain unchanged;
- rollback to `14ee7b0` is proven or remains immediately available;
- no picture code or local evidence has been published.

## Later video stage

Begin a separate project-start and design only after the picture acceptance criteria pass. That design must cover the checkpoint's video processor, frame sampling and timestamps, video position values, request limits, memory, direct local-address message shape, and the fact that pi currently has no native video message type.
