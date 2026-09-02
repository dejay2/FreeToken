# Qwen3.8 picture weights served from the SSD instead of RAM

## Status

Design draft, 2026-09-02. Nothing implemented. Written read-only against `mtp-upstream-merge` at `8591caf` while another agent owns the GPU and a third edits code in a separate worktree. Every measurement below was taken with the server stopped, on the real checkpoint, without touching CUDA.

## Purpose

The 333 picture-reader tensors are 897,862,112 bytes (856.27 MiB, 0.836 GiB) of BF16. Today `-EnableVision -VisionExecution layer-stream` reads them into ordinary pageable CPU memory at boot and holds them for the life of the engine process, whether or not a picture ever arrives. Jay's ask: hold them on the Samsung 990 PRO the same way the PLE n-gram table is held there, and pull them in only when a request carrying a picture arrives.

The goal is not to reclaim 856 MiB of a 95.6 GiB machine in the abstract. It is to return 856 MiB to the Windows standby list, which is what feeds the 47.7 GiB memory-mapped PLE n-gram table that every decoded token gathers rows from. With the server up, this box has been measured at roughly 8 GiB of free host memory (`python/freetoken/models/qwen4_exp/weight.py:266-270`), so the picture reader is about a tenth of the working headroom, and it is headroom that the language path — not the picture path — is starved for.

## Non-goals

- Changing anything the language model does. No GPU placement, KV, GDN state, expert cache, CUDA graph, PLE, or context change.
- Changing picture numerics. The streamed encode must produce bit-identical component weights; only where those bytes live changes.
- Quantizing, resizing, or re-laying-out the picture weights.
- Making picture input the default. It stays gated behind `FREETOKEN_LOAD_VISION=1`, and the new behavior stays behind its own flag, default off.
- Video, Anthropic picture input, non-Windows tuning, or public serving.
- Depending on the FILE_FLAG_NO_BUFFERING reader another agent is adding under `python/freetoken/moe/`. It is referenced below as a future opportunity only.

## Current behavior

### Where the CPU copy is created

`iter_weights` (`python/freetoken/models/qwen4_exp/weight.py:154-216`) gates on two environment reads (`weight.py:180-181`, from `python/freetoken/models/config.py:15-34`). In `layer-stream` mode, when a shard contains `model.visual.*` keys and the engine device is CUDA, it opens a **second** safetensors handle bound to `device="cpu"` (`weight.py:187-202`) and pulls every picture tensor through it with `source.get_tensor(raw_name)` (`weight.py:203-214`). Each `get_tensor` allocates a fresh pageable CPU tensor and copies the shard bytes into it. That allocation is the resident copy.

### Who holds the reference

1. `Engine.__init__` builds the model under `torch.device("meta")` with `config.dtype` (BF16) and immediately calls `load_state_dict` on the materialized dict (`python/freetoken/engine/engine.py:401-403`).
2. `_load_weight_state_dict` (`engine.py:596-612`) runs `_materialize_loaded_weight_state_dict` (`engine.py:315-336`), which for each key does `weight.to(device=target, dtype=expected.dtype)` where `target` comes from the model hook.
3. `Qwen4ExpForCausalLM.weight_device_for_key` (`python/freetoken/models/qwen4_exp/model.py:245-255`) returns `torch.device("cpu")` for every `visual.*` key in `layer-stream`.
4. `BaseOP.load_state_dict` (`python/freetoken/layers/base.py:32-53`) does `setattr(self, name, item)` — **the module attribute becomes the loaded tensor object itself**, with only a shape and dtype assert in between.

So the only strong reference to the 856 MiB is the set of `Qwen4VisionModel` attributes. Nothing else in the engine holds a second copy.

Two consequences that matter for the design:

- `Tensor.to(device, dtype)` returns `self` unchanged when the tensor is already CPU BF16. The checkpoint dtype is BF16 (verified: all 333 headers say `BF16`) and the meta model is BF16, so **anything `iter_weights` yields as a CPU BF16 tensor reaches the module untouched**. No engine change is required to install a different kind of tensor.
- `BaseOP.state_dict` skips attribute names beginning with `_` (`layers/base.py:19-30`), so a private attribute holding a memory mapping is invisible to `state_dict`, `load_state_dict`, and `weight_placement_report`.

### Can the forward path accept a lazily-loaded source?

Yes, with one exception.

`forward_layer_streamed` (`python/freetoken/models/qwen4_exp/vision.py:270-309`) asserts every persistent tensor is on CPU, builds one GPU workspace component at a time in `_forward_layer_streamed_impl` (`vision.py:311-355`), and fills it with `_copy_component_state_` (`vision.py:15-41`), whose only real operation is `target_state[key].copy_(source_tensor, non_blocking=False)` — a host-to-device memcpy. A memcpy source does not care whether the bytes are anonymous heap or a mapped file page; it faults them in.

The exception is `_position_data` (`vision.py:219-249`). It calls `self.pos_embed.forward(indices)` and multiplies the result by interpolation weights **on the CPU**, on the persistent tensor, before moving the result to the device. `visual.pos_embed.weight` is the one picture tensor that is a CPU compute operand rather than a copy source. It is 5,308,416 bytes (5.06 MiB).

Everything else — patch projection, 27 blocks, merger — is copy-only.

### What touches picture weights outside a picture request

Nothing.

- `weight_placement_report` (`model.py:257-278`) walks `self.visual.state_dict()` but reads only `numel()`, `element_size()`, and `device.type`. It faults no data pages.
- Warm-up (`engine.py:525`, `_warmup_prefill` at `engine.py:1536`) and CUDA graph capture contain no vision reference; `grep -n "visual\|vision" python/freetoken/engine/*.py` returns one unrelated comment in `spec_draft.py:1146`.
- CUDA graph capture happens inside the `Engine` constructor, long before `run_forever` (noted at `python/freetoken/scheduler/scheduler.py:338-352`). Picture encoding happens in the scheduler's message-drain phase. The two can never overlap.
- Health checks are text requests.

### When the picture encode runs

`_process_one_msg` (`scheduler.py:650`) detects raw picture tensors at `scheduler.py:665`, calls `_prepare_multimodal_request` (`scheduler.py:592-648`) **synchronously, inline in the scheduler loop**, and only then queues the request with `prefill_manager.add_one_req(msg)` (`scheduler.py:719`). The encode therefore completes **before any prefill chunk of that request is scheduled** — there is no preceding text prefill of the same request to hide a load behind. `msg.mm_embeds` is then sliced per 8,192-token chunk at `scheduler.py:1480-1499`, and multimodal requests match against the empty prefix so they never reuse cached KV (`python/freetoken/scheduler/cache.py:96-103`).

The drain runs at the top of `overlap_loop` (`scheduler.py:229-236`), before `_schedule_next_batch`. A picture encode already stalls the loop for its whole duration (1.519 s cold / 0.724 s warm on the retained screenshot). Any disk read added to the encode extends that same, already-accepted stall.

### On-disk layout of the picture weights (verified)

Read from the safetensors headers with the Desktop venv Python, read-only:

| fact | value |
| --- | --- |
| shards containing `model.visual.*` | exactly one: `model-bf16-00001.safetensors` |
| vision tensors in it | 333 of its 385 (`model-bf16-00010/11/12` have none) |
| dtype | `BF16`, all 333 |
| bytes | 897,862,112 |
| data region | file offsets 375,252,349 .. 1,273,114,461 |
| contiguity | **one unbroken extent**: span == bytes, 0 gaps, 0 non-vision tensors interleaved |
| ends at | the last byte of the file |
| header size | 47,581 B, so the data base is 47,589 — an **odd** number |

Per-component sizes:

| component | tensors | bytes | MiB |
| --- | --- | --- | --- |
| 27 transformer blocks | 324 | 822,933,216 | 784.81 |
| one block | 12 | 30,479,008 | 29.07 |
| merger | 6 | 66,079,232 | 63.02 |
| pos_embed | 1 | 5,308,416 | 5.06 |
| patch_embed | 2 | 3,541,248 | 3.38 |

File order is lexicographic (`blocks.0`, `blocks.1`, `blocks.10`, … `blocks.19`, `blocks.2`, …, then `merger`, `patch_embed`, `pos_embed`), which is not the execution order. Because the whole extent is prefetched in one call this only affects which faults land early, not total I/O.

The odd data base means **every BF16 tensor in this shard starts at an odd address**. `torch.frombuffer` accepts an odd offset and `uint8_view.view(torch.bfloat16)` succeeds (verified on torch 2.11.0+cu130), producing a correctly-shaped but 2-byte-misaligned CPU tensor. That is safe as a `cudaMemcpy` source; it is the reason `pos_embed` is treated specially below.

### The PLE n-gram mechanism, and which parts transfer

Jay's "the same way the n-gram table is served from SSD" refers to `--ple-backend mmap`: `MmapPleStorage` (`weight.py:408-450`) maps each PLE shard with `mmap.mmap(fd, length=0, access=mmap.ACCESS_COPY)`, builds `torch.frombuffer` views over the header-derived byte ranges, and gathers rows either after one batched `PrefetchVirtualMemory` (`weight.py:515-559`) or, on POSIX, one `madvise(MADV_WILLNEED)` per row (`weight.py:561-603`), falling back to a 4-worker fault fan-out. `MmapStagedTable` (`python/freetoken/models/qwen4_exp/ple.py:249-430`) then stages the gathered rows through a pinned ring into CUDA-graph-resident buffers.

**Transfers directly to vision:**

- Parsing byte ranges out of the safetensors header (`_safetensors_header` at `weight.py:684-687`, `_ple_layout` at `weight.py:701-753`) instead of calling `get_tensor`.
- `mmap.ACCESS_COPY` rather than `ACCESS_READ`. Verified: `torch.frombuffer` over an `ACCESS_READ` map warns "The given buffer is not writable, and PyTorch does not support non-writable tensors" on every call and yields a tensor PyTorch believes is writable; `ACCESS_COPY` is silent and correct. Clean copy-on-write pages are still file-backed and still reclaimable.
- `torch.frombuffer` views instead of copies.
- Batched `PrefetchVirtualMemory` with the one-shot disable on `FALSE` (`weight.py:541-559`), and the `madvise(MADV_WILLNEED)` twin for POSIX.
- The lifetime discipline: the mapping object must outlive every tensor built over it, or the tensors dangle. `MmapPleStorage` keeps `self._maps`; the vision holder must do the same.

**Does not transfer:**

- `_PleRowCache` (`weight.py:320-405`). PLE gathers ~16-1120 near-uniform random rows out of 320 M per step, so a bounded FIFO row cache buys hits. Vision reads 100% of its 333 tensors, in the same order, once per picture. The only cache it needs is the OS page cache.
- The 4-worker gather fan-out (`_ple_gather_pool`, `weight.py:309-317`, `_fault` at `weight.py:605-634`). That exists to spread *random* faults across NVMe queue slots. Vision's working set is one 856 MiB sequential extent, where a single prefetch beats any fan-out (measured below).
- `MmapStagedTable`'s pinned staging ring, CUDA-graph staging buffers, and `is_current_stream_capturing` guards (`ple.py:249-430`). PLE runs per decoded token inside CUDA graphs; vision runs once per request, eagerly, outside any graph.
- The per-token gather kernel and FP8 dequant. `_copy_component_state_` copies whole typed tensors into a workspace of matching dtype.
- `load_ple_table`'s O_DIRECT path (`weight.py:756-782` via `read_range_into` at `python/freetoken/moe/host_banks.py:425-471`). It is `os.O_DIRECT` + `os.posix_fadvise`, POSIX-only. This is exactly why this box runs `--ple-backend mmap`, and it is the reason option (c) below is deferred.

## Measurements

Taken 2026-09-02 with the server stopped (71 GiB free), on `D:\Models\Qwen3.8-Flash-Next-NVFP4`, using the Desktop venv Python. Each run maps the file `ACCESS_COPY`, issues one `PrefetchVirtualMemory` over the whole 897,862,112-byte extent, then copies it into a heap buffer.

| run | prefetch call | copy | total | effective |
| --- | --- | --- | --- | --- |
| `model-bf16-00001` vision extent, run 1 (cache state unknown) | 136.7 ms | 71.8 ms | **208.5 ms** | 4.31 GB/s |
| same, run 2 (warm) | 40.2 ms | 66.7 ms | 106.9 ms | 8.40 GB/s |
| same, warm, no prefetch (fault on copy) | — | 66.7 ms | 66.7 ms | 13.46 GB/s |
| same, warm, plain buffered `readinto` into a fresh buffer | — | — | 208.0 ms | 4.32 GB/s |

Cold proxy: three 897,862,112-byte extents from `model-bf16-00011.safetensors` (10.7 GB, untouched this session), same method:

| slice | prefetch call | copy | total |
| --- | --- | --- | --- |
| offset 1,000,000,001 | 84.4 ms | 820.2 ms | 904.6 ms |
| offset 3,000,000,001 | 136.6 ms | 259.4 ms | 396.0 ms |
| offset 5,000,000,001 | 156.3 ms | 270.3 ms | 426.6 ms |

Three things fall out of this:

1. **`PrefetchVirtualMemory` is asynchronous here.** The cold slices returned from the syscall in 84-156 ms while 260-820 ms of faulting work remained. It queues the reads and returns; the consumer absorbs the tail. That is the property the design leans on.
2. **Cold cost of the whole picture reader is 0.40-0.45 s** (the 905 ms first slice includes drive/queue warm-up and is treated as the pessimistic bound, not the expectation).
3. **Warm cost is 0.07-0.11 s** — i.e. a burst of screenshots pays essentially nothing after the first.

For context, the accepted record for the retained 1920×1280 screenshot is 1.519 s cold and 0.724 s warm against a 6 s budget.

## Options considered

### (a) Map the vision extent and let the streamed encode copy from mapped pages — RECOMMENDED

Map `model-bf16-00001.safetensors`' 897,862,112-byte vision extent `ACCESS_COPY`, build `torch.frombuffer` views for the 333 tensors, install those views as the module attributes, and issue one `PrefetchVirtualMemory` over the whole extent when a picture request is admitted.

- Steady-state private RAM: **−856.27 MiB**, permanently, whether or not a picture ever arrives.
- First picture after a cold page cache: **+0.30-0.45 s** (0.90 s pessimistic bound). Screenshot encode goes 1.519 s → ~1.9 s against a 6 s budget.
- Warm/repeat picture: **+0.04-0.11 s**. Screenshot 0.724 s → ~0.8 s.
- Lifetime state machine: **none**. Windows owns residency; clean file-backed pages are reclaimed under pressure with no pagefile write, and re-faulted on the next picture.
- Boot: one shard header parse and one `mmap` replace 333 `get_tensor` calls, so boot gets marginally *faster* and its host high-water mark drops by 856 MiB.
- Files touched: `weight.py`, `vision.py`, `model.py`, `config.py`, the launcher, tests. **No engine change** — `.to()` is a no-op on an already-CPU-BF16 tensor.

Costs and risks: misaligned BF16 views (mitigated below), a new `EXCEPTION_IN_PAGE_ERROR` crash mode if the checkpoint file is removed or the drive drops mid-serve, and ~857 MiB of commit charge for the copy-on-write mapping.

### (b) Load on demand into RAM with an explicit lifetime (TTL / drop on next text request)

Read the 856 MiB into fresh CPU tensors when a picture is admitted, keep them for N seconds or until the next text-only request, then drop.

- Steady-state private RAM: −856 MiB **only between pictures**. While loaded — which includes the whole prefill and decode of the picture request, the part where text throughput is measured — it is exactly today's footprint.
- Every cold picture pays the full read *plus* an 856 MiB allocation the OS must zero. Measured proxy: plain buffered `readinto` into a fresh buffer took 208 ms warm and would be 0.4-0.9 s cold, on top of allocation.
- A burst of screenshots must not reload each time, so it needs a TTL, and the TTL must not overlap the text benchmark, so it needs tuning against two contradictory targets.
- Needs new state: a load lock (the scheduler loop is single-threaded but a background prefetch thread would not be), a drop policy, a "who is using it right now" guard so a drop cannot race an in-flight encode, and error paths for a load that fails halfway.
- Strictly worse than (a) on RAM, on cold latency, on warm latency, and on complexity. Its only advantage is avoiding mmap's fault-time failure mode.

**Rejected.**

### (c) Direct disk → pinned staging → GPU workspace, per component

Never materialize a CPU tensor. For each component, read its byte range with unbuffered I/O into a reusable pinned buffer (max needed: 63.02 MiB for the merger, 29.07 MiB per block) and copy pinned→device.

- Upside is real: a pinned H2D is several times a pageable one, and skipping the page cache means the read never competes with the PLE table's standby residency.
- But the reader does not exist on Windows. `read_range_into` (`host_banks.py:425-471`) is `os.O_DIRECT`, and `drop_page_cache` (`python/freetoken/models/loader.py:56-65`) is `os.posix_fadvise` — stubbed to a no-op by `windows-shim/sitecustomize.py:50-53`. A plain buffered replacement measured 4.32 GB/s *warm* (208 ms), slower than mmap's warm 107 ms, and double-buffers through the page cache anyway.
- 27 sequential 29 MiB reads with a compute gap between them lose the read-ahead that one 856 MiB extent gets for free; the 6.57 GB/s figure in run 1 above is the whole-extent number and should not be assumed for chunked reads.
- Adds a pinned buffer, a read/compute pipeline, and a Windows-specific I/O path to a code path that today has none.

**Deferred**, not rejected. When the concurrent `python/freetoken/moe/` FILE_FLAG_NO_BUFFERING reader lands, (c) becomes a drop-in variant of (a): keep (a)'s layout parsing and lifetime, swap the fault-driven copy for an explicit unbuffered read into pinned staging. Do not build a second Windows unbuffered reader for this.

### (a2) Map, prefetch, then stage each component through a pinned buffer

A middle path: keep (a)'s mapping and prefetch, but instead of copying mapped→device directly, copy mapped→pinned (aligned) →device. Costs one extra CPU memcpy of 856 MiB (measured 66.7 ms warm) and 63 MiB of pinned memory; buys aligned H2D and removes the misalignment question entirely.

Recommended as a **follow-up only if** measurement shows the pageable H2D from mapped pages is materially slower than today's pageable H2D from heap pages. It should not be, since both are pageable.

### (d) Keep as is — the honest case

856.27 MiB is 0.87% of 95.6 GiB. The vision tower is already opt-in and off by default (`config.py:30-34`), so a text-only server pays nothing today. On that framing this is not worth doing.

The framing that changes the answer: with the server up this box has ~8 GiB free (`weight.py:266-270`), the NVFP4 expert source banks alone are 67,987,279,488 B (63.3 GiB) on disk, and the 47.7 GiB PLE table is served entirely out of whatever standby pages remain — every PLE row that misses is an NVMe fault on the decode critical path. 856 MiB is roughly a tenth of that headroom, and this change hands it to the exact consumer that is starved for it. It also removes 856 MiB from the boot-time host high-water mark on a machine where boot host memory is already the tightest resource.

Verdict: **worth doing, as an efficiency change, behind a default-off flag.** It is not a capability change; it does not enable anything that does not work today; and if the measurement in step 3 of the plan shows no text-throughput movement, the flag simply stays off and the code costs nothing.

## Recommended design

### Flag surface

New environment variable `FREETOKEN_VISION_WEIGHTS`, parsed beside the existing gates in `python/freetoken/models/config.py`:

- `ram` (default) — today's behavior, byte for byte.
- `mmap` — the mapped extent.

Validation, all at startup, all fatal with a named error:

- `mmap` requires `FREETOKEN_LOAD_VISION=1`.
- `mmap` requires `FREETOKEN_VISION_EXECUTION=layer-stream`. In `gpu` mode every picture tensor is copied to CUDA at load, so a mapping would be faulted once and then be dead weight; reject rather than silently accept.
- An unrecognized value raises, matching `vision_execution_mode`'s shape (`config.py:15-27`).

Launcher: `-VisionWeights <ram|mmap>` in `scripts/start-qwen38-flash-next-mmap-windows.ps1`, defaulting to `ram`, set only inside the `if ($EnableVision)` branch (`start-qwen38-flash-next-mmap-windows.ps1:91-114`) and echoed in the startup banner beside "Picture execution".

### Layout and mapping

A `VisionLayout` / `MmapVisionWeights` pair in `weight.py`, shaped after `PleLayout` / `MmapPleStorage`:

1. Find the shards carrying `model.visual.*` from `model.safetensors.index.json` when present (the same trick `_ple_table_files` uses at `weight.py:690-698`), else scan the bf16 shards. Here that resolves to one file.
2. Parse its header with `_safetensors_header` (`weight.py:684-687`). Validate every vision tensor's dtype against the model's expected dtype and fail loudly on a mismatch rather than producing a garbage view.
3. Per shard, compute the minimum and maximum file offset of its vision tensors, round the start **down to `mmap.ALLOCATIONGRANULARITY` (65,536)** — for this checkpoint, 375,252,349 → 375,193,600, a delta of 58,749 — and `mmap.mmap(fd, length=span, access=mmap.ACCESS_COPY, offset=aligned_start)`. Mapping the window rather than the whole file keeps commit charge at ~857 MiB instead of 1.27 GiB.
4. For each tensor, `torch.frombuffer(mapping, dtype=torch.uint8, count=nbytes, offset=rel).view(model_dtype).reshape(shape)`.
5. Keep `self._files` and `self._maps` alive, and expose `prefetch()` and `close()`.

`prefetch()` is one `WIN32_MEMORY_RANGE_ENTRY` per mapped window — reuse `_resolve_prefetch_virtual_memory` and the `int64 [n, 2]` entry-array trick already in `weight.py:279-306` and `weight.py:490-513`, and the one-shot `_prefetch_failed` disable at `weight.py:541-559`. On POSIX, `mapping.madvise(mmap.MADV_WILLNEED, 0, span)` — one call for the whole window, not the per-row loop `_advise_rows` needs.

### Installing the views

In `iter_weights`, when `FREETOKEN_VISION_WEIGHTS=mmap` and `stream_vision` is true, replace the second CPU safetensors handle (`weight.py:187-202`) and the `visual.*` branch of `source.get_tensor` (`weight.py:207-208`) with a lookup into the mapping holder. Two carve-outs:

- **`visual.pos_embed.weight` is materialized as an ordinary aligned CPU tensor** (5,308,416 B, 0.6% of the extent). It is the only picture tensor the streamed path uses as a CPU compute operand rather than a memcpy source (`vision.py:232`), and the mapped view would be 2-byte-misaligned. Copying it costs 5 MiB of RAM and removes an entire class of question about misaligned CPU kernels.
- **FTW checkpoints fall back to `ram`.** `load_weight` replays FTW dense shards through `iter_ftw_weights` (`python/freetoken/models/weight.py:237-247`) and never reaches the per-model reader. Log once at startup and continue rather than failing a boot over an optimization.

The holder is stored as a private attribute (`self._weight_source`) on `Qwen4VisionModel`, so `BaseOP.state_dict` skips it (`layers/base.py:22-24`). `weight_placement_report` (`model.py:257-278`) gains a `backing=mmap|ram` field so the boot log states the mode without a new probe.

### Trigger and prefetch placement

The load must happen **inside the engine process** — the picture weights exist nowhere else, and the tokenizer runs in a separate process (noted at `scheduler.py:231-235`).

The earliest in-process signal that a picture is coming is `_process_one_msg`'s `has_raw_picture` check (`scheduler.py:665-671`). There is no text prefill of the same request to hide behind: `_prepare_multimodal_request` runs to completion before `add_one_req` (`scheduler.py:719`).

So:

1. `_prepare_multimodal_request` (`scheduler.py:592`), immediately after validating `pixels`/`grid`/`token_types` and before `build_mrope_positions`, calls a new optional model hook `prefetch_picture_weights()` inside a `try/except` that logs and continues. `getattr(model, ..., None)` keeps the scheduler free of Qwen-specific naming, matching how `weight_device_for_key` is wired (`engine.py:611`).
2. `Qwen4ExpForCausalLM.prefetch_picture_weights` delegates to `Qwen4VisionModel.prefetch_weights`, which is a no-op in `ram` mode.
3. `forward_layer_streamed` (`vision.py:270`) calls the same prefetch defensively at entry, so a direct `encode_images` caller and every test path get the same behavior without the scheduler's cooperation. The call is idempotent and cheap when pages are already resident (40.2 ms warm for the whole extent).

**Why this is enough overlap.** The syscall returns in 84-156 ms while the reads continue in the background. The encode that follows takes ~1.5 s of GPU work. The 27 block copies consume the extent while the drive is still filling it, so the visible cost is the difference between the read rate and the consume rate, not the full read. Executing blocks in numeric order while the file stores them lexicographically means some copies wait on pages the sequential read has not reached yet; at 5-7 GB/s over one 856 MiB extent this is tens of milliseconds and is not worth reordering for.

**Stalling the running request.** The encode already blocks the scheduler loop for its entire duration and has done since the layer-stream design was accepted; this adds 0.3-0.45 s cold to an existing 1.5 s stall. With `--max-running-requests 1` there is at most one other request that could be decoding. If that ever matters, the cheap mitigation is to issue the prefetch on a one-thread executor (the syscall is thread-safe and the pages are process-wide) and let `build_mrope_positions` run concurrently — but that buys only the few milliseconds MRoPE takes, so it is not in the first implementation.

### CUDA graphs

Unaffected, and confirmable without running anything: graph capture happens in the `Engine` constructor before `run_forever` (`scheduler.py:344-350`), picture encoding happens in the scheduler's message-drain phase (`scheduler.py:229-236`), and `forward_layer_streamed` runs eager throughout. Vision is not in any captured graph and never sees `is_current_stream_capturing`. The `torch.cuda.empty_cache()` in `forward_layer_streamed`'s `finally` (`vision.py:302-309`) is unchanged and still runs after every encode — it is what keeps the ~631 MiB workspace segment from depressing text decode.

### Chunking

Unaffected. One encode per picture, at admission, before the first prefill chunk; `mm_embeds` is then sliced across the 8,192-token steps (`scheduler.py:1480-1499`). The disk read happens once per picture request, never per chunk. Multimodal requests still bypass the prefix cache (`cache.py:96-103`), so a repeated screenshot re-encodes — and that is precisely the case the warm 0.07-0.11 s number covers.

### Lifetime and state machine

Deliberately trivial, and this is the main reason (a) beats (b):

| state | entered by | held | exited by |
| --- | --- | --- | --- |
| `unmapped` | `FREETOKEN_VISION_WEIGHTS=ram`, FTW checkpoint, or a mapping failure | 856 MiB private | never |
| `mapped, cold` | boot in `mmap` mode | 0 private, ~857 MiB address space + commit | picture admission |
| `mapped, resident` | prefetch or first fault | file-backed pages in the working set / standby list | Windows trims under pressure — no code path |
| `closed` | engine shutdown | — | process exit |

There is no eviction timer, no reference count, no lock, and no "is it loaded yet" question in the forward path. `Qwen4VisionModel` holds one mapping holder for the process lifetime; residency is the operating system's decision, exactly as it already is for the 47.7 GiB PLE table.

Explicitly *not* doing: `VirtualUnlock` / `EmptyWorkingSet` after an encode to force the pages out. Windows trims a clean file-backed working set for free under pressure, and forcing it guarantees the next picture is cold. If the "after picture work" text benchmark shows the retained working set hurting, that is the knob to try, and it should be measured, not assumed.

## Failure modes and fallbacks

| failure | detection | response |
| --- | --- | --- |
| no shard contains `model.visual.*`, or the index disagrees | layout build at boot | fatal, named error — the picture weights genuinely are not there |
| a vision tensor's dtype is not the model dtype | layout validation | fatal, named error, before any view is built |
| `mmap.mmap` fails (file locked, address space, granularity) | boot | log once at WARNING, fall back to `ram` for the whole tower, keep serving |
| `PrefetchVirtualMemory` returns FALSE | first prefetch | set the one-shot disable like `weight.py:541-559`, log once, fall back to fault-on-copy — measured 0.26-0.82 s instead of 0.30-0.45 s cold, still inside budget |
| POSIX, no `MADV_WILLNEED` | first prefetch | same fallback path |
| checkpoint file deleted, moved, or the drive drops while serving | page fault at copy time | `EXCEPTION_IN_PAGE_ERROR` — **a new, unrecoverable crash mode that resident RAM does not have.** Identical exposure to the already-accepted 47.7 GiB PLE mapping, which faults on every decoded token; document it, do not try to catch it |
| checkpoint file modified in place while serving | undefined | same standing rule as PLE: the model directory is immutable while the server is up |
| a mapped page is written by accident | copy-on-write makes it private and dirty, silently negating the saving | never write through a view; add a test that asserts the installed `visual.*` tensors' `data_ptr()` lies inside the mapped window after a full encode |
| encode raises mid-stream | existing `try/finally` (`vision.py:302-309`) | unchanged — workspace cleared, `empty_cache`, one terminal request error, raw pixels released (`scheduler.py:643-648`) |
| FTW checkpoint | `is_ftw_checkpoint` at load | log once, fall back to `ram` |

Rollback is the flag: drop `-VisionWeights mmap` and the server is byte-identical to the accepted `layer-stream` build.

## Acceptance criteria

Everything the layer-stream design accepted must still hold, plus the new saving.

1. All 333 picture tensors report `device=cpu` and total 897,862,112 bytes in `weight_placement_report`, now with `backing=mmap`.
2. Every installed `visual.*` tensor except `visual.pos_embed.weight` has a `data_ptr()` inside the mapped window; `pos_embed` is 2-byte aligned and outside it.
3. Engine-process **private** working set at idle, in `mmap` mode, is at least 800 MiB below the same boot in `ram` mode.
4. `/v1/cache/status` reports 4,097 pages of 64 tokens: 262,208 allocated, exactly 262,144 usable.
5. Automatic sizing restores 4,063 cached experts (or a count that still meets criterion 6 without reducing context).
6. The exact deterministic 512-input/512-output benchmark averages **≥ 50 output tokens/second** across its three runs, both before any picture and after two picture requests. The accepted record is 50.58 before / 52.10 after.
7. The retained 1920×1280 screenshot encodes in **≤ 6 s** on a cold page cache (expected ~1.9 s) and on a warm one (expected ~0.8 s).
8. Streamed features from mapped weights are **bit-identical** to streamed features from resident weights for the same input. Not "close" — the bytes are the same bytes, so anything less means a layout bug.
9. Five back-to-back screenshots: no growth in allocator high-water mark, no retained staged component, no expert-cache reduction, and encode 2-5 within 20% of encode 2.
10. A picture failure is followed by a passing direct text-health request.
11. `-VisionWeights ram` reproduces today's behavior exactly; text-only boot builds no tower and opens no mapping.

## Measurement plan

All of it in the private evidence tree, nothing published.

**Off-GPU (can run while another agent holds the GPU):**

1. Layout unit test on the real header: 333 tensors, one shard, one contiguous 897,862,112-byte extent, all BF16.
2. `torch.frombuffer` view test at the real odd offsets: shapes, dtypes, and byte equality against `safetensors.safe_open(...).get_tensor(...)` for a sample of tensors including the largest (`merger.linear_fc1.weight`, 42,467,328 B) and a bias.
3. Alignment test: every installed view's `data_ptr() % 2` recorded; `pos_embed` asserted aligned.
4. Cold/warm extent read benchmark, re-run of the table in **Measurements** above, as a regression baseline.
5. Fallback tests: prefetch returning FALSE, `mmap` failing, FTW checkpoint, `gpu` execution mode rejected, unknown flag value rejected.
6. A small synthetic vision model: streamed encode from mapped weights vs. from resident weights, asserted bit-identical, and asserted to touch all 27 blocks exactly once.

**On-GPU, in order, once the GPU is free:**

7. Boot `ram`, record engine PID, `PrivateMemorySize64`, `WorkingSet64`, expert count, `/v1/cache/status`, and the placement report.
8. Run the deterministic 512/512 benchmark three times. Record.
9. Restart in `mmap`. Record the same four things. Criterion 3 is the delta.
10. **Cold** screenshot immediately after boot. Record `Picture encoder request N: X seconds` and a new INFO line reporting prefetch milliseconds and whether the prefetch succeeded.
11. Warm screenshot ×4 back to back. Record all five encode times and the allocator high-water mark before/after.
12. The 512/512 benchmark three more times, after picture work. Criterion 6.
13. The 33K chunked-picture regression (`benchmarks/run_qwen38_vision_smoke.py`).
14. Malformed picture → error → text health.
15. Normal pi screenshot and a following pi text request, with normal context, skills, templates, and extensions.
16. `PrivateMemorySize64` and standby-list size sampled at boot, after the cold picture, and 60 s after the last picture, to answer whether the prefetched pages stay charged to the working set.

## Files that would change

| file | change |
| --- | --- |
| `python/freetoken/models/config.py` | `vision_weights_backing()` gate, validated against `vision_load_enabled` and `vision_execution_mode` |
| `python/freetoken/models/qwen4_exp/weight.py` | `VisionLayout` + `MmapVisionWeights` (header parse, aligned window map, `frombuffer` views, `prefetch`, `close`); `iter_weights` branch that yields views instead of `get_tensor` results; `pos_embed` carve-out |
| `python/freetoken/models/qwen4_exp/vision.py` | `Qwen4VisionModel._weight_source`, `prefetch_weights()`, defensive prefetch at `forward_layer_streamed` entry |
| `python/freetoken/models/qwen4_exp/model.py` | `prefetch_picture_weights()` delegate; `backing=` field in `weight_placement_report` |
| `python/freetoken/scheduler/scheduler.py` | one optional `prefetch_picture_weights()` hook call at the top of `_prepare_multimodal_request` |
| `scripts/start-qwen38-flash-next-mmap-windows.ps1` | `-VisionWeights` switch, validation, banner line |
| `tests/models/qwen4_exp/test_weight.py` | layout, view, dtype-validation, fallback tests |
| `tests/models/qwen4_exp/test_vision.py` | mapped-vs-resident bit-identity, prefetch idempotence, `data_ptr()` containment |
| `tests/scheduler/test_multimodal_admission.py` | prefetch hook called before MRoPE; absent hook is not an error |
| `tests/engine/test_vision_weight_placement.py` | `backing=` reported; engine materializer still a no-op on a mapped CPU BF16 tensor |

`python/freetoken/engine/engine.py` is deliberately **not** in this list.

## Open questions

1. Does `PrefetchVirtualMemory` charge the pages to the process working set, and if so does the 856 MiB show up as shareable (harmless, trimmable) or does it depress the standby pages the PLE table needs? Step 16 of the measurement plan answers it. If it hurts, the knob is an `EmptyWorkingSet` after the encode, at the price of a cold next picture.
2. Is the pageable H2D from a mapped page measurably slower than from a heap page, once the page is resident? If yes, option (a2) — one extra 66.7 ms CPU memcpy through a 63 MiB pinned buffer — is the fix.
3. Should `_position_data`'s embedding gather move onto the GPU workspace instead of keeping a 5.06 MiB aligned CPU copy of `pos_embed`? Cleaner, but it changes a numerically-validated path for 5 MiB; not worth it in the first implementation.
4. Should the whole file be mapped instead of an aligned window? Simpler code, 1.27 GiB of commit charge instead of 857 MiB. The window is recommended, but only the commit-charge argument separates them.
5. Once the concurrent `python/freetoken/moe/` FILE_FLAG_NO_BUFFERING reader lands, is option (c) worth revisiting for the cold case? It would keep 856 MiB out of the page cache entirely, which is the same argument that motivates this whole change.
6. Should the same treatment be offered for anything else? `model.embed_tokens` is 1.27 GB, but under `FREETOKEN_EMBED_HOST` it is deliberately *pinned* for UVA gathers (`model.py:314-327`) and is read every decode step — the opposite workload. No other tensor set fits.
7. Does the real (non-shim) Windows path need `drop_page_cache` to become a real `FILE_FLAG_NO_BUFFERING`-based call for the expert loader, and would that then evict the vision extent as collateral? Worth checking before the shim is retired.
