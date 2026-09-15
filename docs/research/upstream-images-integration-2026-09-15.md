# Upstream image integration validation

Worktree: `.worktrees/upstream-images`, branch `feat/upstream-images`, base `1ad07e7`.
Source image commits: `08d728d`, `8ff0cce`, `db64879`, `84d236c`, `3faef36`, `63d6471`, `68a81ff`.

## Scope
All three serving APIs retain image content and tool images. Shared upstream MM processing, image encoder ownership, model registry and supported image families are adapted onto the fork's current quantization and scheduling code. Qwen4 keeps its memory-mapped encoder, prefetch, MTP and placement controls. Existing services have not been restarted or switched.

Important deviations: image KV remains private, because the upstream 30-bit content token cannot safely identify an image for shared KV reuse. Complete hashes still identify encoder-cache entries. Unsupported quantized vision weights are rejected before conversion, rather than importing the broader quantization refactor.

## Environment and baseline
Linux CPU-only PyTorch 2.11.0+cpu; Python 3.13. Shared project virtualenv. Baseline API/vision check: 120 passed, 5 skipped.

Eight broader runtime failures reproduced on the unchanged original checkout: two cache-budget tests assume optional FlashInfer is installed, and six speculative-sampler tests unconditionally allocate CUDA-pinned memory on CPU PyTorch. The existing server PLE-default test also fails on the original checkout (expects pinned, current Linux default is disk). These are excluded from final acceptance runs, not counted as passing.

## Review repairs
- Preserve legacy and new image embeddings in mixed batches.
- Defer image-cache claim release until in-flight request drain.
- Preserve private KV even when two images have equal compact pad IDs.
- Fix float32 rotary fallback view aliasing.
- Keep Gemma legacy encode_images and offline precomputed embedding support.
- Ensure Qwen4's legacy opt-in and explicit FREETOKEN_LOAD_VISION=0 agree between processors and engine.

## Remaining hardware acceptance
No live checkpoint generation or CUDA kernel execution has been claimed. Test screenshot recognition, ordered multiple images, tool-result images, long chunked prompts, abort, text-only follow-up, MTP image continuation, mmap vs RAM working set, and peak VRAM on the target GPU before switching the service.

Oversized bidirectional image spans are rejected before allocation with advice to reduce image tokens or increase the chunk/window budget. Spans without a legal aligned split are checked together. Temporary contention yields rather than splitting an image. Muse preprocessing with torchvision 0.26 uses the upstream HF bicubic approximation for Lanczos; exact checkpoint resize parity is not established.

Three additional unchanged-baseline failures were reproduced: two host-embedding tests require CUDA NVTX, and one QSA workspace prefill case unconditionally pins CPU memory. A parking-idle test raced once during the broad run and passed alone; neither its code nor cache manager was changed.

## Reproduce the combined check

Run from the integration worktree with the project virtualenv on PATH.

```bash
PYTHONPATH=python OMP_NUM_THREADS=2 python -m pytest \
  tests/engine \
  tests/kernels/test_mrope.py \
  tests/llm/test_multimodal_compat.py \
  tests/mm \
  tests/models/qwen4_exp \
  tests/models/test_gemma4_vision.py \
  tests/models/test_glm5_next_config.py \
  tests/models/test_glm5_next_vision.py \
  tests/models/test_image_family_dense_contract.py \
  tests/models/test_minimax_m3.py \
  tests/models/test_minimax_m3_vision.py \
  tests/models/test_muse_glimmer.py \
  tests/models/test_muse_glimmer_vision.py \
  tests/models/test_qwen3_vl.py \
  tests/models/test_qwen3_vl_vision.py \
  tests/scheduler \
  tests/server \
  tests/tokenizer \
  -q \
  --disable-warnings \
  --tb=short \
  --maxfail=5 \
  --deselect \
  tests/server/test_parser_auto_selection.py::test_ple_backend_is_exposed_by_the_server_cli \
  --deselect \
  tests/engine/test_cache_budget.py::test_adjust_config_resolves_num_tokens_generic \
  --deselect \
  tests/engine/test_cache_budget.py::test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend \
  --deselect \
  tests/engine/test_spec_draft.py::test_the_drafts_filter_is_the_one_the_server_would_have_prepared \
  --deselect \
  tests/models/qwen4_exp/test_embedding_host.py::test_forward_matches_dense_lookup_and_flattens \
  --deselect \
  tests/models/qwen4_exp/test_embedding_host.py::test_forward_path_never_syncs_the_host \
  --deselect \
  'tests/models/qwen4_exp/test_qsa_step_workspace.py::test_armed_staging_reproduces_the_allocating_paths_addressing[3-prefill]'
```

## Final result

Combined command above: **2,433 passed, 319 skipped, 14 deselected, 4 warnings in 134.25s**. Deselections include parameterized cases in the baseline-affected groups, including two cases that pass independently. The parking-idle test passed in this final combined run. Compilation and full staged whitespace checks passed. Real HF preprocessing tests cover six image families with both processor defaults and token limits (12 cases); torch remains 2.11.0+cpu with torchvision 0.26.0+cpu and transformers 5.16.1. Independent final review found no remaining confirmed P0/P1/P2 blockers.
