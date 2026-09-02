# Qwen3.8 chunked picture-bearing prompts on Desktop-assisted Windows

## Status

Implemented and accepted locally on 2026-08-30 at scheduler commit `56a34ec`. Direct picture input passed at 33,243 prompt tokens through five bounded steps, and Jay's normal pi screenshot request returned `gpt-5.6-luna` without the old 8,192-token rejection. A separate placement regression then showed language generation falling to 7.00 tokens/second because the all-GPU picture reader reduced the expert cache; the approved follow-up is `docs/design/2026-08-30-qwen38-vision-layer-streaming-design.md`. Final normal-pi text-after-picture qualification is carried by that follow-up.

## Purpose

Remove the local FreeToken picture path's 8,192-token hard rejection while retaining 8,192 tokens as the maximum work done in one prompt-loading step. A picture-bearing prompt may span repeated steps up to the model's available combined prompt-and-reply context, just like ordinary long text.

This corrects a gap exposed by a normal pi screenshot request: the complete prompt contained 9,878 tokens after pi's instructions, tools, project context, text, and picture placeholders were combined. The Qwen checkpoint did not impose that 8,192-token ceiling; the first local picture implementation did.

## Users

- Jay uses the local `freetoken-local/Qwen3.8-Flash-Next-NVFP4` model in an ordinary pi session with its normal instructions, tools, skills, and extensions loaded.
- Direct local OpenAI-compatible callers may send still pictures with long surrounding text.

## Scope

### Included

- Picture-bearing prompts split into repeated prompt-loading steps of at most the live `max_extend_tokens` budget, currently 8,192 tokens.
- Combined picture, instruction, tool, project-context, conversation, and text tokens up to the engine's available context after reserving the requested reply allowance.
- One or more still pictures whose placeholder ranges begin, end, or continue across a step boundary.
- Existing Qwen three-axis picture positions across every step and generated continuation.
- Existing private-cache behavior across every step, cancellation, completion, and failure.
- Direct local and normal pi acceptance using a prompt larger than 8,192 tokens.

### Excluded

- Raising one processing step to the full 262,144-token context.
- Splitting the Qwen picture encoder itself; each request's picture pixels are still encoded once before language-model prompt loading.
- Automatic picture shrinking, cropping, or quality reduction.
- Removing the checkpoint processor's own picture-size bounds.
- Video, Anthropic picture messages, shared-network exposure, publication, or upstream work.

## Success criteria

- The previously rejected normal pi screenshot request, measured at 9,878 combined prompt tokens, succeeds without disabling pi instructions, tools, skills, or context files.
- A synthetic picture-placeholder range crossing an 8,192-token boundary receives the exact matching feature rows in both steps.
- Two pictures distributed across multiple steps retain feature order and produce the expected answer.
- A picture-bearing prompt above 32,768 combined tokens succeeds, proving at least five scheduled prompt steps including the final partial step.
- Picture requests continue to report zero shared-prefix reuse and never donate private picture state to the shared tree.
- Cancellation during any intermediate step releases request pages, state slots, picture features, and pending references exactly once.
- Text-only chunking, reasoning, streaming, tool calls, malformed-picture recovery, and final text health remain correct.
- The live server still reports 4,097 total KV pages, 262,208 allocated tokens, and exactly 262,144 usable tokens after the reserved page.
- SSD-backed PLE, serial expert loading, one active request, loopback-only binding, and automatic expert-cache sizing remain active.
- Peak graphics-card and system memory are measured for the long picture-bearing request and fit the current machine without changing the context allocation.

## Constraints

- Keep each prompt-loading step at or below the scheduler's live budget, currently 8,192 tokens. This bounds temporary graphics-card work on the 32 GB RTX 5090.
- The model's context is a combined prompt-and-reply allowance. Existing behavior may shorten the requested reply allowance when the prompt consumes more of that context.
- Keep picture features private even after the current step has released its local reference.
- Keep the complete picture feature tensor only for the lifetime of its pending request. Do not re-run the picture encoder for each step.
- Do not edit FreeToken Desktop, model files, the published text branch, pi defaults, or the rollback worktree.
- Do not push or publish this local picture work.

## Chosen approach

### Prepare once

The tokenizer continues to create the complete token list, picture patches, picture grid, and text/picture markers. Scheduler admission validates the complete request, runs the picture encoder once, checks that every picture placeholder has one feature row, and computes the complete Qwen three-axis position table once.

The scheduler no longer rejects a picture-bearing prompt merely because its complete token count exceeds one prompt-loading step. It rejects only the existing model-context and input-contract failures.

### Retain complete private request state

The pending request retains:

- the complete CPU token list;
- the complete GPU picture-feature tensor;
- the complete CPU three-axis position table;
- the generated-token position offset;
- an authoritative `cache_private` marker.

The authoritative privacy marker remains true after per-step feature references are cleared. Different pictures therefore cannot share or donate token-only prefix state.

### Slice feature rows by placeholder order

For each scheduled step covering logical token range `[cached_len, device_len)`:

1. Count picture placeholders before `cached_len` to find the first feature row for this step.
2. Count picture placeholders inside the current range to find how many rows it needs.
3. Select exactly that contiguous feature-row slice.
4. Concatenate slices in request order when a batch contains more than one request.
5. Give the model `None` when the step has no picture placeholders.

This uses Qwen's established rule that picture feature rows and picture placeholders have the same order. Admission's complete-request row-count check remains the independent guard against missing or extra features.

The temporary request object for a completed step releases its reference after the batch slice is built. A pending continuation keeps the complete tensor until the final step is scheduled or the request is cancelled.

### Slice positions through the existing path

The existing three-axis position builder already creates complete prompt coordinates. The existing batch position packer selects `[cached_len, device_len)` for each step. This behavior becomes required for picture-bearing continuation steps and is tested at boundaries rather than replaced.

Logical one-dimensional positions remain authoritative for cache addresses, causal visibility, PLE, and QSA block selection. Three-axis values remain limited to Qwen rotation.

### Reuse the proven long-text continuation machinery

Remove the picture-private prohibition from the prompt splitter. The existing continuation object, page table, QSA pending rings, GatedDeltaNet state, PLE context, prompt accounting, and final sampling path remain responsible for advancing one request through repeated steps.

Intermediate private steps are never inserted into the shared prefix tree. The final picture-bearing request also remains private and its pages are freed at completion.

## Important interfaces and seams

- `python/freetoken/scheduler/scheduler.py`
  - remove the one-step picture rejection;
  - build only the current step's feature slice in multimodal gathering;
  - preserve cleanup and request-level error isolation.
- `python/freetoken/scheduler/prefill.py`
  - allow `cache_private` continuation requests;
  - retain the complete feature tensor in `PendingReq` while each `Req` receives a temporary reference.
- `python/freetoken/scheduler/utils.py` and `python/freetoken/core.py`
  - retain complete pending picture state and per-step request state without changing text-only defaults.
- `python/freetoken/models/qwen4_exp/model.py`
  - keep the exact per-batch placeholder-to-feature count check; no model-wide full-prompt assumption.
- `python/freetoken/scheduler/cache.py`
  - preserve empty shared-prefix matching, no insertion/donation, and completion cleanup for every private continuation.
- QSA/MRoPE/GDN/PLE paths
  - preserve their existing long-text continuation behavior and prove picture coordinates/state survive boundaries.

## Error and cancellation behavior

- Invalid sources, invalid pixels, processor failures, feature/placeholder mismatches, and prompts exceeding the model's combined context remain one terminal request error and do not stop workers.
- A per-step feature-slice mismatch is an internal correctness failure. Focused tests must prevent it; if encountered live, the request must terminate without serving an answer from misaligned picture features.
- Cancelling before the first step releases prepared picture state.
- Cancelling while a step is running waits for that work to drain, then releases pages and state once.
- Cancelling between continuation steps removes the pending continuation and releases the complete feature tensor.
- A failed or cancelled picture request must be followed by a healthy text request in acceptance testing.

## Security and privacy

The existing broad local picture-source policy is unchanged and remains acceptable only because the server binds to `127.0.0.1`. Picture feature rows, KV pages, recurrent state, and prompt prefixes remain request-private and are never made reusable by another request.

## Testing strategy

### Focused tests without the full model

- Replace the old 8,193-token rejection test with admission of a long picture-bearing request.
- Drive a deterministic 8,193+ token prompt through at least two scheduler steps and verify each step receives only its own feature rows.
- Place picture placeholders before, across, and after a step boundary.
- Put two pictures in separate and shared steps and verify row order.
- Verify a step containing no picture placeholder receives no picture features.
- Verify complete MRoPE coordinates are sliced exactly at each boundary.
- Verify prompt accounting is emitted once for the complete prompt, not once per step.
- Verify cache-private requests match zero shared tokens, are never inserted during intermediate/final steps, and free pages/state at completion.
- Verify cancellation before, during, and between steps releases the complete feature tensor and scheduler resources.
- Re-run current long-text, hybrid-cache, QSA split-group, graph replay, picture admission, source/error, and model suites.

### RTX 5090 and live checks

1. Keep the existing server running during CPU-only work; stop it only for required graphics-card tests and the final restart.
2. Run picture-aware QSA/MRoPE boundary tests on the RTX 5090.
3. Start the picture branch with the unchanged launcher, 262,144 usable tokens, mmap PLE, serial experts, one active request, and loopback binding.
4. Send a direct picture-bearing prompt above 32,768 combined tokens and verify multiple 8,192-token steps in server logs.
5. Run the prior picture-source, two-picture, cross-picture privacy, malformed-input recovery, text, reasoning, tool, streaming, and 32K text checks.
6. Run a real ordinary pi screenshot request with normal context files, skills, prompt templates, extensions, and tools available; do not use the stripped acceptance command from the first picture milestone.
7. Verify a subsequent pi text request remains healthy.
8. Record timings, memory peaks, step boundaries, cache geometry, answer, and local commit in ignored evidence.

## Rollback

The published text-only worktree remains `D:\FreeToken-ple-mmap` at `14ee7b0`. On any live failure, stop all picture workers, restore pi's text-only declaration if picture capability is no longer reliable, start the known-good launcher, and run its text smoke check. Keep failed picture work local for diagnosis and publish nothing.
