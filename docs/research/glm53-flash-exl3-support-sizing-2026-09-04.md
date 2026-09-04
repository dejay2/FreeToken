# GLM-5.3-Flash EXL3 support sizing (2026-09-04)

## Recommendation — three sentences

Do not promise broad EXL3 support as the next full feature; fund a narrow 2.05-bit proof first. If that proof works, adapt ExLlamaV3's permitted graphics-card math to FreeToken instead of expanding each expert before multiplying, because only the 2-bit build is small enough for this computer and the expand-first route throws away most of the speed benefit. Removing half the NVFP4 experts would take less engine work, but published GLM tests show a large quality loss, while the other 2–3-bit formats still need new fast math and a newly tested model.

## Machine fit in plain words

The computer has 95.6 GB of PC memory and can give at most 24 GB of card memory to experts, so the expert budget is about 119.6 GB. The four-bit experts take about 160 GiB (171.4 GB), so they are too large for that budget. The 2.05-bit EXL3 build R2 measured at 85.1 GB would fit that budget by size alone, but only if FreeToken could read it; it cannot today.

## Bottom line

- The best first target is the `2.05bpw` branch of `turboderp/GLM-5.3-Flash-exl3`, restricted to its fixed K=2, `mul1` routed experts and graphics-card expert execution.
- That branch contains 85,148,910,958 bytes (79.3 GiB) of `.safetensors` files. Its routed experts would occupy about 71.29 GiB in FreeToken's host banks, versus 159.65 GiB for the current NVFP4 experts. It is the only main turboderp branch that is even plausible on this computer's 95.6 GB of PC memory; a full boot-memory proof is still required.
- A correctness-only reconstruct-first proof is about **8–12 files, 1,200–2,200 lines and 3–5 person-weeks**. It should not be shipped as the normal serving path.
- A narrow production K=2 path that adapts ExLlamaV3's MIT-licensed fused CUDA work is about **20–27 files, 3,500–6,200 lines and 8–12 person-weeks**, including Windows build work, tests and measurements.
- A general EXL3 implementation covering mixed rates, both codebooks and a new pure-Triton implementation is about **24–35 files, 6,000–10,000 lines and 16–24 person-weeks**.
- The riskiest piece is the direct fused expert operation: it must be fast, preserve GLM's exact clamped activation, and replay repeatedly and concurrently inside FreeToken's recorded CUDA work on Windows without stale locks, moving addresses or hidden setup calls.

This is research only. No model weights were downloaded, no GPU was used, and no serving process was changed.

## 1. What EXL3 is

### Scheme

EXL3 is ExLlamaV3's very-low-bit weight format. It is described by its author as a streamlined variant of QTIP: weights are regularized with Hadamard transforms and half-precision factors, then every 16 by 16 tile is represented as a path through a procedural codebook using an optimal tail-biting trellis. It is not an ordinary packed integer format, which is why its small size comes with more complicated reconstruction and multiplication.

ExLlamaV3's documentation says a full format description is still forthcoming. The implementation is therefore part of the practical specification today.

### Stored tensors

For a linear weight whose ordinary shape is `[out_features, in_features]`, the stored pieces are:

```text
trellis  [in_features // 16, out_features // 16, 16 * K]  int16
suh      [in_features]                                     float16
svh      [out_features]                                    float16
mcg or mul1                                                 one marker value
bias     [out_features]                                    float16, optional
```

`K = trellis.shape[-1] // 16`. Each trellis tile describes 256 ordinary weights and uses `256 * K` bits. `suh` is the input-side factor and `svh` is the output-side factor around the Hadamard transforms. `mcg` and `mul1` select one of the two procedural codebooks.

The converter accepts a floating target average. The stored matrix rate is an integer K, the general CUDA operations dispatch K=1 through K=8, and the small-batch operation specializes K=2, K=3 and K=4. Public releases are advertised mainly as about 2.0 through 8.0 bits per weight; mixed rates across tensors let a complete model land at labels such as 2.05 or 3.05 bits per weight.

The 47,905,719-byte `quantization_config.json` on turboderp's branches is metadata, not model weights. It contains 37,032 matrix records. The routed experts account for 36,288 of them: 12,096 gate, 12,096 up and 12,096 down matrices, exactly `42 layers * 288 experts * 3 projections`. Every routed-expert record on the 2.05 branch is K=2 with `mul1`; the 3.05 and 4.05 branches use K=3 and K=4 respectively. This uniformity is important because FreeToken requires every row of one bank to have a fixed shape.

### Operations, Windows and licence

ExLlamaV3 ships custom C++/CUDA operations for:

- direct EXL3 matrix-vector multiplication for small batches;
- a general matrix multiplication operation;
- reconstruction to half precision followed by a normal half-precision multiplication;
- multi-matrix pointer-table multiplication;
- a fused mixture-of-experts operation that sorts routes, runs gate/up/down projections, applies the activation and route weights, and reduces the result.

No CPU EXL3 multiplication operation was found. Initial FreeToken support would therefore have to reject or fall back from CPU-only and CPU/GPU-mixed expert modes rather than silently selecting them.

The normal setup requires CUDA 12.4 or newer. ExLlamaV3 v1.4.6 was published on 2026-09-02 with prebuilt Linux and `win_amd64` wheels for several Python, Torch and CUDA combinations, and its README also describes a Windows source build using Visual Studio Build Tools; `triton-windows` is recommended. A FreeToken integration still has to match the exact Torch/CUDA build used by the Desktop Python, so the existence of wheels does not make this a drop-in dependency.

ExLlamaV3 is MIT-licensed. Its source may be used and modified, but copied code must retain the Turboderp copyright and MIT licence notice.

### Published GLM-5.3-Flash builds from the Hugging Face API

The inventory below was taken on 2026-09-04 with the Hugging Face model-search API and each repository's `?blobs=true` metadata. “Bytes” is the exact sum of files ending in `.safetensors`; no model file was downloaded. The search returned 22 repositories. Turboderp keeps its weights on three branches and leaves `main` empty.

| repository / revision | advertised rate | `.safetensors` files | exact bytes | note |
|---|---:|---:|---:|---|
| [`turboderp/GLM-5.3-Flash-exl3`, `2.05bpw`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2.05bpw) | 2.05 | 12 | 85,148,910,958 | unaltered; strongest first target |
| [`turboderp/GLM-5.3-Flash-exl3`, `3.05bpw`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw) | 3.05 | 17 | 125,179,521,581 | unaltered; too large here |
| [`turboderp/GLM-5.3-Flash-exl3`, `4.05bpw`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/4.05bpw) | 4.05 | 20 | 165,067,349,597 | unaltered; too large here |
| [`0xSero/GLM-5.3-Flash-EXL3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw) | 3.0 | 130 | 149,402,871,912 | model files present |
| [`0xSero/GLM-5.3-Flash-EXL3-Q4`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Q4) | Q4; exact average not stated | 217 | 187,453,172,472 | model files present |
| [`JANGQ-AI/GLM-5.3-Flash-EXL3`](https://huggingface.co/JANGQ-AI/GLM-5.3-Flash-EXL3) | 4.0 | 23 | 163,803,892,556 | model files present |
| [`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) | 4.0 | 120 | 175,642,157,752 | model files present |
| [`satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw`](https://huggingface.co/satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw) | 3.5 | 120 | 156,163,802,568 | model files present |
| [`Terra3312/GLM-5.3-Flash-EXL3-4bpw-MUL1`](https://huggingface.co/Terra3312/GLM-5.3-Flash-EXL3-4bpw-MUL1) | 4.0 | 22 | 163,274,767,460 | model files present |
| [`vcruz305/GLM-5.3-Flash-EXL3-K2`](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2) | K=2 | 120 | 97,728,721,536 | model files present; 91.0 GiB |
| [`vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix`](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix) | K=2/3 mix | 120 | 103,164,540,888 | variable rows need padding or bucketing |
| [`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1) | K=3 | 16 | 136,686,260,192 | model files present |
| [`neko-legends/GLM-5.3-Flash-Uncensored-EXL3`](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3) | 4.0 | 92 | 175,644,133,320 | altered/uncensored model |
| [`Olt1z/GLM-5.3-Flash-Uncensored-EXL3-q4`](https://huggingface.co/Olt1z/GLM-5.3-Flash-Uncensored-EXL3-q4) | 4.30 overall | 23 | 174,953,071,794 | altered/uncensored model |
| [`s-zaizen/GLM-5.3-Flash-EXL3-TR3-3.51bpw-Uncensored`](https://huggingface.co/s-zaizen/GLM-5.3-Flash-EXL3-TR3-3.51bpw-Uncensored) | 3.51 | 120 | 156,616,786,440 | altered/uncensored model |
| [`unsignedchad/GLM-5.3-Flash-ablit-exl3-4bpw`](https://huggingface.co/unsignedchad/GLM-5.3-Flash-ablit-exl3-4bpw) | 4.0 | 23 | 165,067,832,334 | altered/ablated model |
| [`vcruz305/GLM-5.3-Flash-Uncensored-EXL3-K2`](https://huggingface.co/vcruz305/GLM-5.3-Flash-Uncensored-EXL3-K2) | K=2 | 62 | 97,728,939,920 | altered/uncensored model |
| [`0xSero/GLM-5.3-Flash-EXL3`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3) | not stated | 0 | 0 | empty/documentation only |
| [`0xSero/GLM-5.3-Flash-EXL3-2.0bpw`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-2.0bpw) | 2.0 | 0 | 0 | empty reservation |
| [`0xSero/GLM-5.3-Flash-EXL3-2.5bpw`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-2.5bpw) | 2.5 | 0 | 0 | empty reservation |
| [`0xSero/GLM-5.3-Flash-EXL3-TR3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-TR3-3.0bpw) | 3.0 | 0 | 0 | empty reservation |
| [`gitcommit90/GLM-5.3-Flash-EXL3-2.05-One-Spark`](https://huggingface.co/gitcommit90/GLM-5.3-Flash-EXL3-2.05-One-Spark) | 2.05 | 0 | 0 | empty/documentation only |
| [`turboderp/GLM-5.3-Flash-exl3`, `main`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) | branch index only | 0 | 0 | weights are on the three branches above |
| [`vcruz305/GLM-5.3-Flash-EXL3`](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3) | not stated | 0 | 0 | empty/documentation only |
| [`vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm`](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm) | K=2 | 0 | 0 | wheel-only: 3 wheels, 576,563,197 bytes; no model weights |

## 2. What FreeToken would need

### The existing contract

FreeToken's expert cache is deliberately unaware of the mathematical format. A format declares an ordered set of banks in `_BANK_SCHEMAS`, supplies fixed-size contiguous rows for every streaming layer, and adds an expert operation that reads those banks. The provider boundary is summarized in `python/freetoken/moe/expert_banks.py:1-13`; current providers are registered at `python/freetoken/moe/expert_banks.py:312-318`.

The cache verifies that the loader returned exactly the declared names and that every streaming layer has the same row shape and type (`python/freetoken/moe/offload_cache.py:504-605`). Decode loads only missing expert rows and rewrites expert numbers to cache-slot numbers; prompt processing materializes a complete layer and keeps raw expert numbers (`python/freetoken/layers/moe.py:693-758`, `python/freetoken/layers/moe.py:838-894`). The EXL3 prompt and decode operations must therefore agree mathematically even if they use different launch shapes.

### Recommended bank layout

For general fixed K values per projection, use nine logical banks:

```text
gate_trellis [E, H//16, I//16, 16*K_gate]  int16
gate_suh     [E, H]                         float16
gate_svh     [E, I]                         float16
up_trellis   [E, H//16, I//16, 16*K_up]    int16
up_suh       [E, H]                         float16
up_svh       [E, I]                         float16
down_trellis [E, I//16, H//16, 16*K_down]  int16
down_suh     [E, I]                         float16
down_svh     [E, H]                         float16
```

`K_gate`, `K_up`, `K_down` and the codebook choice should be model-level validated constants, not tiny per-expert banks. A one-value marker bank would break the 16-byte fused-copy rule and is unnecessary for the uniform turboderp builds. Gate and up factors are independently learned values; they may be stored as two slices of one physical bank for a K=2-only optimization, but they must never be treated as one shared factor.

Nine banks are compatible with the cache's bank-count-independent movement. All GLM dimensions above are multiples of the copy alignment. A mixed-K repository such as `K2K3-mix` is not compatible with one fixed row shape without padding, separate buckets or a wider cache contract, so it should be explicitly rejected in the first version.

The schema and total row formula go in `python/freetoken/moe/offload_cache.py:32-96`. Exact per-bank row sizes must also be added to `python/freetoken/kernel/aot_models.py:73-129`, and GLM's AOT model entry must advertise the format at `python/freetoken/kernel/aot_models.py:277-290`. The torch-free settings helper mirrors the total-size formulas, so a later product implementation must also keep `python/freetoken/daemon/settings/model_info.py:6-13` in step.

### Exact GLM bank sizes

GLM-5.3-Flash has 42 routed-expert layers, 288 experts per layer, hidden width H=4096 and expert width I=2048. There are 12,096 routed experts in total.

| format | bytes per expert | MiB per expert | total bytes | total GiB | share of NVFP4 |
|---|---:|---:|---:|---:|---:|
| EXL3 K=2 | 6,328,320 | 6.035 | 76,547,358,720 | 71.290 | 44.65% |
| EXL3 K=3 | 9,474,048 | 9.035 | 114,598,084,608 | 106.728 | 66.85% |
| EXL3 K=4 | 12,619,776 | 12.035 | 152,648,810,496 | 142.165 | 89.05% |
| current NVFP4 | 14,172,160 | 13.516 | 171,426,447,360 | 159.653 | 100% |

For K=2, each gate/up/down trellis is 2,097,152 bytes and the six factor vectors add 36,864 bytes per expert. Only K=2 clearly enters the right memory class for this computer. K=3 and K=4 remain too large for FreeToken's current all-host-bank design even though their file labels sound small.

The total branch size is larger than the routed-expert bank total because it also contains the ordinary language weights, configuration and other model pieces. FreeToken drops the picture tower and MTP layer for this GLM milestone and places dense language weights on the graphics card, but 71.29 GiB of pinned expert banks still leaves little room for loading buffers and the rest of the process. A Windows loader must reuse the existing unbuffered/direct-shard path rather than temporarily filling the operating-system file cache beside the pinned copy (`python/freetoken/models/nvfp4_banks.py:36-55`).

### Loader and format selection

A new reusable EXL3 bank loader should follow the NVFP4 pattern at `python/freetoken/models/nvfp4_banks.py:88-242`:

1. Read the safetensors index and `quantization_config.json` metadata.
2. Match only routed expert layers 3–44 and skip the three dense layers, picture tower and trailing MTP layer.
3. Validate all gate/up/down `trellis`, `suh`, `svh` and marker records before allocating most of memory.
4. Reject mixed K across experts in the first version; validate one K per projection and one common `mcg` or `mul1` codebook, matching ExLlamaV3's fused-operation requirement.
5. Allocate one fixed-shape tensor per bank per layer, fill by shard, pin only after a layer is complete, and support the existing per-layer sink.
6. Use direct whole-shard or range reads on Windows so the load does not keep a second cached copy of tens of gigabytes.
7. Check the exact expected tensor count and every shape, just as the NVFP4 loader checks `num_layers * E * 6` at `python/freetoken/models/nvfp4_banks.py:193-242`.

`detect_expert_quant` already returns an unknown lower-case method string at `python/freetoken/models/config.py:80-110`, so a checkpoint whose method is literally `exl3` will likely reach `expert_quant="exl3"`. The work is still incomplete until a provider is registered, GLM's model loader accepts the EXL3 tensor names instead of asserting NVFP4-only, and configuration tests prove the exact published metadata. GLM's current NVFP4-only assumptions and key patterns are at `python/freetoken/models/glm5_next/weight.py:1-10`, `python/freetoken/models/glm5_next/weight.py:58-106` and `python/freetoken/models/glm5_next/weight.py:186-190`.

### Direct fused operation versus reconstruct-first

**Production route: adapt ExLlamaV3's fused CUDA operation.** It already accepts nine pointer arrays, independent K values for gate/up/down, K=1–8 dispatch, both codebooks, route sorting, gate/up/down multiplication and reduction. FreeToken should either build fixed pointer tables for every cache slot and the two prompt buffers or modify the operation to use base addresses plus row strides. The addresses must be allocated once so recorded CUDA work always sees the same locations.

The source cannot be copied unchanged:

- ExLlamaV3's fused path has global device locks, a self-resetting ticket scheduler and cross-block barriers. It requires all block groups to be resident together (`exl3_moe.cu` and `exl3_moe_kernel.cuh`). FreeToken must warm all one-time setup before recording, prove the locks reset after every replay, and prove two requests or graph instances cannot corrupt each other.
- The operation skips an expert when its route count is above `max_tokens_per_expert` and expects reconstruction to handle that expert outside the fused operation. Long FreeToken prompts can concentrate many routes on one expert, so the integration needs either a bounded reconstruct fallback or enough fixed scratch memory for the admitted prompt size.
- ExLlamaV3's limited SiLU is very close to, but not bit-for-bit the same as, FreeToken's GLM activation. ExLlamaV3 computes SiLU first and then caps that result; FreeToken specifies `min(gate, limit) * sigmoid(gate) * clamp(up, -limit, limit)` (`python/freetoken/kernel/triton/activation.py:176-184`). A port should implement FreeToken's ordering and test values above the limit rather than accepting a small silent difference.
- The current operation uses FP16 inputs and intermediates and a FP32 accumulation buffer. FreeToken's normal hidden type and expected tolerance need an explicit reference comparison.
- No EXL3 CPU operation exists, so the provider and automatic backend choice must reject CPU/hybrid expert execution for this format.

**Correctness route: reconstruct selected experts, then multiply normally.** It is simpler because ExLlamaV3 already exposes reconstruction and FreeToken has ordinary grouped multiplication. It is a poor serving design: one reconstructed GLM expert is 48 MiB in FP16, versus a 13.516 MiB NVFP4 row, so the multiplication alone reads at least 3.55 times as many weight bytes as the current inline-dequant path, before counting the compressed read and the 48 MiB reconstruction write/read. The existing NVFP4 design deliberately avoids this temporary expansion (`python/freetoken/kernel/triton/nvfp4_fused_moe.py:1-21`).

### Expected speed

A K=2 miss would copy 44.65% as many expert bytes as NVFP4, so the host-to-card part has a theoretical 2.24-times byte advantage. That is not a 2.24-times serving claim: trellis decoding and Hadamard transforms are more work than NVFP4's direct lookup and scales, and ExLlamaV3 itself says its lower-bit general multiplication still needs work to stay limited by memory speed.

The defensible expectation is:

- cache misses and full-layer prompt copies should move materially fewer bytes;
- cache-hit decode may be slower than NVFP4 unless the adapted K=2 operation is very well tuned;
- long prompts may be much slower if they trigger reconstruct fallback;
- reconstruct-first is expected to be slower than NVFP4 and is suitable only as a correctness proof;
- no exact tokens-per-second number should be promised before the same-model Windows benchmark exists.

FreeToken fixes decode launch settings because run-time tuning is unsafe inside recorded CUDA work (`python/freetoken/moe/fused_nvfp4.py:56-74`). EXL3 needs the same rule: tune before recording, then use fixed shapes, fixed scratch buffers, fixed pointer tables and no CPU synchronization during replay.

### Copy constraints

The fused multi-bank copy is enabled only when every row and source/destination address is 16-byte aligned; the older per-bank fallback additionally requires row bytes divisible by 128 (`python/freetoken/moe/offload_cache.py:607-686`). The proposed GLM trellis and factor rows satisfy both rules.

Rows below 256 KiB are copied as one whole-layer entry during hit-aware prompt loading (`python/freetoken/moe/offload_cache.py:17-26`, `python/freetoken/moe/offload_cache.py:1071-1089`). All six EXL3 factor banks fall into that class. This is desirable: it avoids the measured CUDA behavior where mixing 5–22 KiB entries with large entries makes `cudaMemcpyBatchAsync` block the CPU. The three trellis banks remain route-aware large-row copies.

### Tests expected by this repository

At minimum:

- CPU-only format-detection tests for the real turboderp metadata and clean rejection of mixed-K or mixed-codebook expert rows (`tests/models/test_glm5_next_config.py:260-320`).
- A synthetic safetensors checkpoint that drives the real EXL3 loader without model weights or a GPU, following `tests/moe/test_gpu_owned_banks.py:40-149`.
- Exact bank names, shapes, bytes, pinned-after-fill behavior, layer sinking and direct-reader tensor counts.
- Byte-for-byte fused versus per-bank movement for 0, 1, 4 and 8 misses across all nine banks, modeled on `tests/moe/test_fused_copy.py:1-76`.
- A pure Torch reconstruction reference and operation comparisons for K=2 first, then K=3/K=4 if claimed.
- Prompt and decode comparisons through raw expert numbers and cache-slot numbers; cache overwrite/reload; the two-buffer prompt path; and activation inputs below, at and above limit 10. The neighboring depth is illustrated by `tests/moe/test_nvfp4_backends.py:1-15` and its prompt/decode reference tests.
- Repeated graph record/replay with unchanged output; replay after a prompt; multiple graph widths; two simultaneous requests; a forced zero-route expert; a route count at and above the scratch limit; and lock/scheduler reset after an error.
- Windows build/import smoke tests for the supported Desktop Python/Torch/CUDA combination and no-JIT/AOT coverage for every bank row size.
- Refusal tests for CPU-only/hybrid expert execution until a real CPU EXL3 operation exists.

### Work estimate and reasoning

| slice | files touched or added | approximate lines | why |
|---|---:|---:|---|
| schemas, sizing, provider and selection | 5 | 200–350 | bank names/bytes, provider registration, AOT model list, CPU-mode refusal, helper sizing |
| checkpoint loader and GLM key handling | 2–3 | 450–750 | index/config parsing, nine bank allocations, direct reads, validation, layer sink |
| Python operation wrapper and dispatch | 2–3 | 350–650 | route preparation, fixed pointer tables, scratch ownership, prompt/decode dispatch |
| adapted CUDA sources, build wiring and licence | 6–9 | 1,500–2,800 | trellis/codebook/Hadamard math, fused expert launch, Windows build, retained notice |
| tests and benchmark drivers | 5–7 | 900–1,400 | loader, math, movement, graph replay, concurrency, Windows smoke |
| operator documentation | 1–2 | 150–250 | supported branches, K=2 limits, memory and speed evidence |
| **narrow production K=2 total** | **20–27** | **3,500–6,200** | overlap between slices is allowed in the file count |

The 8–12 person-week estimate assumes one experienced CUDA/Triton developer can reuse and adapt the MIT source, the first shipped scope is fixed-K K=2 plus `mul1`, and model conversion is out of scope. Supporting arbitrary per-tensor K, both codebooks, CPU/hybrid execution or a new pure-Triton implementation would move the estimate toward 16–24 person-weeks.

## 3. Cheaper ways to reach about 2–3 bits per weight

### A. Simple packed integers through compressed-tensors, GPTQ or AWQ

This is mathematically simpler than EXL3. A reasonable FreeToken layout would be four banks: packed gate/up integers, gate/up group scales, packed down integers and down group scales. A direct operation could unpack and scale inside the multiplication loop, much like the current NVFP4 operation.

`compressed-tensors` now defines symmetric W2A16 and W3A16 presets with group size 128, and its packer accepts 1–8-bit integer weights. Its current simple `int32` pack uses `pack_factor = 32 // num_bits`; at 3 bits it stores ten values in 30 bits and leaves two bits unused. That makes it a usable file container, not a ready FreeToken streamed-GLM expert operation.

For GLM's exact dimensions with FP16 group-128 scales:

| scheme | bytes per expert | total expert GiB | result here |
|---|---:|---:|---|
| packed 2-bit, simple int32 packing | 6,684,672 | 75.305 | plausible memory class |
| packed 3-bit, compressed-tensors ten-per-int32 packing | 10,469,376 | 117.940 | too large |
| ideal no-waste 3-bit packing | 9,830,400 | 110.742 | still too large |

Engine work would be roughly **10–15 files, 2,000–3,500 lines and 5–8 person-weeks**, about one-third cheaper than narrow EXL3, because unpack/scale math is simpler and there are no Hadamard transforms or trellis scheduler. End to end it is not a quick solution: no published GLM-5.3-Flash checkpoint was found in the exact required bank layout, so somebody must quantize it, calibrate it, test quality and publish a model before FreeToken support is useful.

Published quality evidence is model-dependent, not a GLM prediction:

- GPTQ's paper reports that for OPT-175B/BLOOM-176B, about 2.2 bits with group size 128 raised WikiText-2 perplexity by less than 1.5, while about 2.6 bits with group size 32 raised it by about 0.6–0.7. Its LLaMA table also shows that ungrouped 3-bit quantization can lose badly and group size 128 closes much of the gap.
- AWQ's paper reports Mixtral-8x7B WikiText-2 perplexity of 5.94 at FP16 and 6.52 at INT3 group 128, an increase of 0.58. Its fast public inference work is centered on 4-bit; AutoAWQ says other rates such as 3-bit can be requested but may have no inference operation.
- No GLM-5.3-Flash 2/3-bit GPTQ or AWQ quality result was found, so these numbers must not be presented as expected GLM quality.

**Verdict:** materially cheaper in engine code than EXL3, but not materially faster to a usable GLM product unless a trustworthy 2-bit checkpoint and evaluation arrive first.

### B. Repack llama.cpp IQ2/IQ3 into FreeToken banks

IQ2 and IQ3 are not simple packed integers. llama.cpp's IQ2 uses selected E8-lattice codebook points over groups of weights; IQ3_XXS uses a D4-style lattice and its own sign/scale packing. Several very-low-bit modes require an importance matrix, and llama.cpp refuses IQ2_XXS/IQ2_XS without one because quality can collapse.

Copying the stored bytes into FreeToken banks would not copy the operation that gives those bytes meaning. A real port still needs custom vector and grouped matrix operations, codebook tables, route sorting, prompt and decode paths, and Windows graph tests. Expected work is **14–22 files, 3,000–5,000 lines and 8–14 person-weeks** if the existing CUDA math can be cleanly adapted; that is close to EXL3 rather than a cheap shortcut.

GLM-specific community measurements show the quality cliff:

| build | reported measure |
|---|---:|
| 6block BF16 | perplexity 6.6974 |
| 6block IQ4_XS | perplexity 7.1537 |
| 6block IQ3_XXS | perplexity 8.2485 |
| 6block IQ2_XS | perplexity 20.2651 |
| 6block IQ1_M | perplexity 73.9234 |

Unsloth reports top-token agreement with the original model, not perplexity: UD-IQ2_XXS 76.30%, UD-Q2_K_XL 78.34%, UD-IQ3_XXS 81.63%, UD-Q3_K_XL 86.25%, and UD-Q4_K_XL 92.22%. Both are community model-card measurements, not official Z.ai benchmark results, but both say the same practical thing: quality drops sharply in the 2-bit range.

**Verdict:** not materially cheaper than EXL3, and the available GLM evidence is less encouraging at 2 bits.

### C. Keep NVFP4 but remove experts

This is the smallest FreeToken change only if somebody supplies a coherent pruned checkpoint. Merely deleting expert files is wrong: the router's output, expert numbering and checkpoint configuration must be remapped together. With a correctly produced checkpoint, FreeToken can keep its existing NVFP4 loader and fast operation and mainly needs model-shape acceptance, validation and tests.

Halving the 288 routed experts would halve the current 159.653 GiB routed-expert banks to about 79.8 GiB. A rough engine estimate is **4–8 files, 500–1,200 lines and 2–4 person-weeks**, excluding the much larger model-pruning, calibration and quality project.

A published GLM-5.3-Flash REAP model card reports, against an 8-bit perplexity of 3.4607:

| pruned experts | mixed 4/8-bit size | perplexity |
|---:|---:|---:|
| 25% | 139.1 GB | 4.2249 |
| 37% | 118.3 GB | 4.8752 |
| 50% | 96.3 GB | 6.0757 |

These are not NVFP4 results, but they are direct GLM-5.3-Flash evidence that removing enough experts to fit has a large quality cost.

**Verdict:** cheapest engine work, but not the recommended quality/size trade unless the user accepts the measured loss.

## Sources

### EXL3 primary sources

- https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md
- https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md
- https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/modules/quant/exl3.py
- https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/modules/quant/exl3_lib/quantize.py
- https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/exllamav3_ext/quant/exl3_gemv.cu
- https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/exllamav3_ext/quant/exl3_gemm.cu
- https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/exllamav3_ext/quant/exl3_moe.cu
- https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/exllamav3_ext/quant/exl3_moe_common.cuh
- https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh
- https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/exllamav3_ext/quant/hadamard_inner.cuh
- https://github.com/turboderp-org/exllamav3/blob/master/README.md
- https://github.com/turboderp-org/exllamav3/releases/tag/v1.4.6
- https://github.com/turboderp-org/exllamav3/blob/master/LICENSE

### Hugging Face metadata

- https://huggingface.co/api/models?search=GLM-5.3-Flash-EXL3&limit=100
- https://huggingface.co/api/models/turboderp/GLM-5.3-Flash-exl3/revision/2.05bpw?blobs=true
- https://huggingface.co/api/models/turboderp/GLM-5.3-Flash-exl3/revision/3.05bpw?blobs=true
- https://huggingface.co/api/models/turboderp/GLM-5.3-Flash-exl3/revision/4.05bpw?blobs=true
- https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/blob/2.05bpw/quantization_config.json

### Alternatives and quality evidence

- https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py
- https://github.com/vllm-project/compressed-tensors/pull/715
- https://github.com/vllm-project/vllm/blob/c227aaa3/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py
- https://arxiv.org/html/2210.17323v2
- https://github.com/IST-DASLab/gptq
- https://arxiv.org/html/2306.00978v2
- https://github.com/mit-han-lab/llm-awq
- https://github.com/casper-hansen/AutoAWQ/blob/main/docs/examples.md
- https://github.com/ggerganov/llama.cpp/pull/4856
- https://github.com/ggerganov/llama.cpp/pull/4897
- https://github.com/ggerganov/llama.cpp/pull/5196
- https://huggingface.co/6block/GLM-5.3-Flash-GGUF
- https://unsloth.ai/docs/models/glm-5.3-flash
- https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit
