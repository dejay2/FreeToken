# Upstream Image Integration Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development. Track task ownership and validation here.

**Goal:** Integrate upstream image serving while preserving the fork's memory-mapped Qwen vision implementation.
**Architecture:** Port image-specific upstream patches by subsystem. Adapt model interfaces to the existing quantization stack. Join through upstream MMItem and SupportsMultimodal contracts.
**Tech Stack:** Python, PyTorch, transformers, pytest.
**Spec:** ../specs/2026-09-15-upstream-images-design.md

## Global Constraints
- Work only in .worktrees/upstream-images; base 1ad07e7.
- Preserve mmap vision, MTP, dynamic KV, prefix-cache, EXL3 and memory-governor behavior.
- No blanket quantization refactor; do not silently accept unsupported quantized vision.
- Do not change a running service or push.
- CPU tests cannot establish live GPU performance or answer quality.

## Task 1: Protocol and processor integration
- [x] Port upstream MM types/config/processors, protocol conversion, tokenizer, CLI, launcher and stats changes; preserve chronological system messages and local image options.
- [x] Run new upstream protocol tests against baseline to establish missing behavior, then against port.
- [x] Add regressions for tool images, disabled image errors and existing local input forms.
Ownership: API agent. Files: mm/, server/, tokenizer/, message/, launch.py, matching tests, docs/cli.md.
Interfaces: upstream MMItem, MultimodalConfig, get_mm_processor, model.input_modalities.

## Task 2: Qwen models and encoder contracts
- [x] Port upstream Qwen model interfaces, registry, shared model config/blocks/weight handling and Qwen VL tower.
- [x] Preserve Qwen4 mmap loader/tower via encode/place_encoder_weights adapter.
- [x] Run model construction, loader and mapped-weight regressions.
Ownership: Qwen agent. Files: models/qwen*/, models/{register,config,blocks,weight,weight_stream}.py, llm/llm.py, matching tests.
Interfaces: SupportsMultimodal.encode(items), place_encoder_weights, EncoderSpec, model_is_mrope.

## Task 3: Other image families
- [x] Port image patches for Gemma4, GLM5-next, Muse-Glimmer, MiniMax-M3 and necessary layer helpers.
- [x] Adapt constructor/loader changes to current quantization interfaces.
- [x] Run family model and vision unit tests.
Ownership: family agent. Files: those four model directories, layers/{activation,norm,embedding}.py, matching tests.
Shared registry and processors are coordinated with Tasks 1/2.

Validation: family image/config/loader and local EXL3/GGUF regression selection: 59 passed, 18 skipped on CPU. Added CPU dense-constructor, unified Gemma embedder and quantized-vision rejection checks; GPU tower/streaming/checkpoint checks remain skipped.

## Task 4: Runtime integration
- [x] Port core, engine, graph, scheduler, attention, rotary/kernel and KV interfaces.
- [x] Retain local request admission, speculative paths and memory budgeting; use image content IDs to avoid cache aliasing.
- [x] Run MM encoder, scheduler, cache and engine CPU regressions.
Ownership: root.

## Task 5: Combined review and acceptance
- [x] Run compile/import and combined CPU suites; repair integration issues.
- [x] Review all local feature preservation and unsupported-format errors.
- [x] Record exact checks and remaining GPU acceptance in docs.
- [x] Deliver reviewable branch and concise limitations; do not deploy or push.

## Final verification

Combined CPU suite: 2,433 passed, 319 skipped, 14 deselected in 134.25s. Deselections are cases in the independently reproduced baseline failure groups listed in the validation report. Independent combined review found no remaining confirmed blocking issues. GPU inference/performance/peak-memory acceptance remains outstanding; branch retained without changing the running service.
