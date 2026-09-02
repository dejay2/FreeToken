# Assessment of upstream issue #335 against today's fork work (2026-09-02)

Issue: https://github.com/FlashML-org/FreeToken/issues/335. Reporter: Linux (vast.ai, `pytorch:2.11.0-cuda13.0` container), sm_120-class GPU, Qwen3.8-Flash-Next-NVFP4, commit `4b94bdc`, `--nvfp4-backend flashinfer` (b12x path). Two defects: (1) CUDA graph capture aborts at bs=8 with an unjoined stream fork inside `flashinfer/.../moe_w4a16_route_pack.py`; (2) a real 2-request decode batch raises a tile-config ValueError for fc2 with N/K = 2560/640.

## Verdict

Unrelated to, and not fixed by, today's fork work (750d83d unbuffered Windows shard reader; memory audit; cache-size sweep). One part is explained by the audit: b12x is opt-in and `auto` never selects it for this checkpoint.

## Evidence

- `4b94bdc` is an ancestor of `mtp-upstream-merge`; `git diff --stat 4b94bdc HEAD -- python/freetoken/moe/nvfp4_backends.py` is empty. Nothing in 750d83d (`moe/win_io.py`, `DirectShard`, `read_shard_direct`) is on the NVFP4 GEMM path.
- `engine/config.py:315`: `nvfp4_backend = "triton"` by default. `moe/nvfp4_backends.py:157-180` `_b12x_min_intermediate()` = 1024; `:253-268` `auto` falls to triton below it. This checkpoint has `moe_intermediate_size=640`, `hidden_size=2560` (audit `memory-audit-qwen38-rtx5090.md:5,42`), which is exactly the N/K=2560/640 in the reporter's fc2 error (down projection). All fork measurements ran `NVFP4 expert backend: triton` (`measurements-unbuffered-boot-2026-09-02.md:67,82`).
- Defect 1: at `4b94bdc`, `graph.py:175` is the eager warm-up forward and `:179` the forward inside `torch.cuda.graph(...)`; the failure at 179 means the warm-up completed, so this is not the workspace-resolution hazard the b12x docstring anticipates (`nvfp4_backends.py:726-729`) but a real unjoined fork inside flashinfer. FreeToken's decode path forks no side stream at that commit (`offload_cache.py` has only `prefill_copy_stream`).
- Defect 2: FreeToken passes no tile knobs (`force_tile_config|moe_block_size|tile_n|tile_k`: zero hits). Selection is entirely flashinfer's.
- Reporter's note that shapes JIT per request with `--cuda-graph-max-bs 0`: the only decode warm-up is inside `_capture_graphs` (`graph.py:175`); `_warmup_prefill` is prefill-only and gated on a triton attention backend (`engine.py:424-426`).
- Default ladder to 160: `graph.py:81` uses 160 whenever free memory <= 80 GB; the ladder is not clamped to `--max-running-requests` except for DSV4 (`engine.py:1677-1695` fork / `1079` upstream) although `_cpu_moe_executor_tokens` states decode batches never exceed it (`engine.py:101` fork / `692` upstream).
- New observation: the bs=8 eager warm-up passed without the tile error but a real 2-request batch failed. The capture batch is `[dummy_req] * bs` with all-zero ids/positions (`GraphCaptureBuffer.init`, `graph.py:36-43`), so every row routes to the same 10 experts; real requests scatter routes. The tile / `moe_block_size` choice appears to track the routing distribution, not batch size.

## Would ask the reporter

1. flashinfer version (both tracebacks are in flashinfer internals).
2. Does `--cuda-graph-max-bs 1` capture? Capture is largest-first and aborted at 8, so 1/2/4 were never tried.
3. Does it reproduce with `--moe-backend hybrid`?
4. Does b12x capture on a wide-I NVFP4 checkpoint (e.g. MiniMax-M2, I=1536)?
5. Exact `moe_block_size` / route distribution at failure.
6. Whether the tile error ever appeared during boot warm-up or prefill (the 28-token prefill passed).

## Draft comment (not posted)

Read through this against the tree at `4b94bdc`. Independent code reading, no repro (no sm_120 + CUDA 13 Linux box here). A few things that may narrow it down.

b12x is opt-in, and `auto` would never have chosen it for this checkpoint. `EngineConfig.nvfp4_backend` defaults to `"triton"` (`engine/config.py:25`), and even under `auto` the pick is gated on `moe_intermediate_size >= _b12x_min_intermediate()` = 1024 (`moe/nvfp4_backends.py:157-180`, `253-268`). Qwen3.8-Flash-Next has `hidden_size=2560`, `moe_intermediate_size=640`, `num_experts=512`, `top_k=10`, which is exactly the N/K=2560/640 in your fc2 tile error (the down projection). So no default deployment is affected. The users who are affected are the ones following the `_b12x_min_intermediate` docstring, which says a throughput deployment of a small-I model "should force it with `--nvfp4-backend flashinfer`". That advice is unreachable for this shape, which looks like the concrete FreeToken-side fix: a load-time guard (or at minimum a docstring correction) rather than a crash on the first multi-request batch.

On defect 1, your traceback says the warm-up worked. At `4b94bdc`, `graph.py:175` is the eager forward and `:179` is the one inside `torch.cuda.graph(...)`. You failed at 179, so the eager forward at bs=8 completed. Most of that 2m13s was CuTe JIT, and the workspace-resolution-during-capture hazard that `b12x_fused_experts`' docstring anticipates was already handled. That leaves a genuine unjoined fork inside flashinfer's route pack. FreeToken forks no side stream on the decode path at this commit (`offload_cache.py` only has `prefill_copy_stream` and its events), so the fork isn't ours.

A detail from your two traces that may be the useful clue: the bs=8 eager warm-up did not raise the tile ValueError, but a real 2-request batch did. The capture batch is `[dummy_req] * bs` with `input_ids` and `positions` all zeros (`GraphCaptureBuffer.init`), so all 8 rows route to the same 10 experts, whereas two real requests scatter ~20 routes across up to 20 experts. Combined with "M=1 decode fine, 28-token prefill fine, M>=2 decode dies", that suggests the tile / `moe_block_size` choice tracks the routing distribution rather than batch size, and the broken configuration is a narrow band rather than the shape being unsupported outright. FreeToken passes no tile hints at all (`force_tile_config` / `moe_block_size` / `tile_n` / `tile_k`: zero hits in the tree), so the selection is entirely inside flashinfer.

Two things that would help: (a) the flashinfer version; (b) whether `--cuda-graph-max-bs 1` captures cleanly. Capture runs largest-first and aborted at bs=8, so bs=1/2/4 were never attempted. If bs=1 captures, batch-1 at graph speed is available now instead of eager. `--moe-backend hybrid` instead of `offload` would separate the slot-cache views from the kernel, and a wide-I NVFP4 checkpoint (MiniMax-M2, I=1536, the shape `auto` actually selects b12x for) would tell whether the route pack is ever capture-safe or only breaks on this shape.

On your note 1, the mechanism is what you suspected: the only decode-path warm-up lives inside `_capture_graphs` (`graph.py:175`), so `--cuda-graph-max-bs 0` removes it entirely and every new shape JITs on a live request. `_warmup_prefill` is prefill-only and gated on the resolved attention backend starting with `triton` (`engine.py:424`).

One unrelated thing your log surfaces: with `--max-running-requests 8` the default ladder still captures up to 160 (`graph.py:81`, since `free_memory_gb <= 80` on any 32 GB card), even though decode batches never exceed `max_running_req` (`engine.py:692`). There is already a clamp for DSV4 (`engine.py:1079`); generalizing it would cut capture time and pool memory for everyone, independently of this bug.
