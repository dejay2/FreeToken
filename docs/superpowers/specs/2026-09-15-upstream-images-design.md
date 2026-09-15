# Upstream image integration

## Intent and authorization
The user accepted the recommendation to adopt upstream image API and model integration while preserving the fork's memory-mapped Qwen vision weights. This is an architectural port into an isolated branch based on 1ad07e7. Existing serving installations are not switched during implementation.

## Design
Port the image-specific changes from upstream commits 08d728d, 8ff0cce, db64879, 84d236c, 3faef36, 63d6471, and 68a81ff. Adapt these to the fork's existing quantization interfaces rather than importing the unrelated quantization refactor. Use upstream MMItem, content-based image identity, encoder ownership, model registry and processor contracts. Preserve local text scheduling, dynamic KV, prompt-cache, MTP, host-memory and EXL3 features.

Qwen4 retains its existing vision tower and mmap loader. Provide the upstream multimodal encoder interface through this tower, retaining prefetch, streaming, cleanup and truthful memory reporting. Existing FREETOKEN_LOAD_VISION / FREETOKEN_VISION_EXECUTION / FREETOKEN_VISION_WEIGHTS settings retain their meaning. New model image paths follow upstream contracts, adapted to legacy unquantized vision layers. Do not silently ignore quantized vision formats.

All three APIs preserve images, including tool-result images carried into a following user turn. Text-only servers reject unsupported images clearly. Agent launch configuration declares image capability from server stats. Model coverage targets Qwen VL, Qwen3.5/3.6, Qwen4, Gemma4, GLM5-next, Muse-Glimmer and MiniMax-M3.

## Alternatives
A full upstream merge pulls in a wide quantization refactor and risks unrelated performance regressions. A protocol-only patch is smaller but misses reusable processing and model coverage. The selected port has more adaptation work but keeps the upgrade focused.

## Validation
Baseline targeted suite: 120 passed, 5 skipped on CPU. Extend upstream regression tests for fork-specific mmap and protocol paths. Run API/tokenizer/MM/model/scheduler/engine CPU suites and available CLI tests. CUDA-dependent checks must be reported as skipped when no GPU is available; never claim live model quality, speed or VRAM acceptance from CPU tests. Review the combined diff for preservation of local features.

## Review decisions
- Image requests retain private KV. Upstream's int32 content pad stores only 30 hash bits; it cannot authorize collision-safe cross-request KV reuse. Encoder embedding sharing still keys the complete hash and lasts only while requests own unconsumed rows.
- Legacy precomputed image batches may mix with new MMItem batches; both embedding sources use explicit scatter rows.
- Image cache ownership is released at the same drain point as in-flight KV resources.
- Bidirectional image spans must fit a legal whole-image chunk. Impossible aligned spans (including connected spans with no legal split) are rejected before allocating request resources. Temporary prefill/window contention yields until the complete span can run.
