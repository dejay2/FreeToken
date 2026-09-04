# GLM-5.3-Flash GGUF and 2-bit options

_Date: 2026-09-04. Research only; no model weights were downloaded and no product files were changed._

## Plain-language answer

Run [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) instead; it is the supported model that fits this computer. GLM-5.3-Flash needs a memory upgrade before it can run here, and its smaller files cannot be read by FreeToken today.

There are many small GLM-5.3-Flash files on Hugging Face, including 2-bit GGUF, EXL3, and MLX builds. None can be loaded by FreeToken today: FreeToken's native GGUF reader is for Gemma-4, while the GLM-5.3 reader expects a particular safetensors NVFP4 layout. A new GLM-specific reader and conversion pipeline would be needed.

Converting a 2-bit GGUF would not turn it into a small native FreeToken model. It would first unpack the already-quantized numbers and then write them again as q4_0 or NVFP4. That makes a four-bit model again, needs about 171 GB of routed experts, and cannot restore quality discarded by the original 2-bit step. This computer has a combined expert budget of about 119.6 decimal GB, so the four-bit GLM-5.3-Flash model misses by about 52 GB.

The supported Qwen checkpoint's routed experts occupy 18.182308 GB (16.933594 GiB), leaving 101.417692 GB in the same budget calculation.

## Scope and accounting

The machine facts supplied for this job are Windows 11, one RTX 5090 with 32 GB of VRAM, and 95.6 GB of PC memory. At most about 24 GB of the card is assigned to routed experts; the rest must hold resident weights, chat state, work space, and headroom. I therefore use:

```
expert budget = 95.6 GB host + 24.0 GB card = 119.6 decimal GB
```

The table's margin is `119.6 GB - routed-expert bytes`. A positive margin means that the expert bytes alone fit the stated budget. It does not mean that a checkpoint is loadable: resident tensors, format overhead, the reader, and the model architecture still matter.

For GLM-5.3-Flash, the public configuration gives hidden size `H=4096`, expert width `I=2048`, 45 layers, a dense prefix of 3 layers, and 288 routed experts in each of the remaining 42 layers:

```
routed expert instances = (45 - 3) * 288 = 12,096
values per expert       = 2*I*H + I*H = 25,165,824
routed expert parameters = 12,096 * 25,165,824 = 304,405,807,104
```

A raw bit-depth number is only a lower bound. It excludes scales, codebooks, headers, and format-specific padding:

| Raw precision | Raw routed-expert bytes | Decimal GB |
|---|---:|---:|
| 1 bit | 38,050,725,888 | 38.050726 |
| 2 bits | 76,101,451,776 | 76.101452 |
| 3 bits | 114,152,177,664 | 114.152178 |
| 4 bits | 152,202,903,552 | 152.202904 |
| 16 bits | 608,811,614,208 | 608.811614 |

## What FreeToken can read

The relevant source evidence is:

* `docs/models.md:3-5` says that FreeToken reads Hugging Face safetensors directly and has native GGUF support only for Gemma-4.
* `python/freetoken/models/register.py:111-117` registers `Gemma4GGUFForCausalLM` with the GGUF parser. `:133-145` registers GLM-5.3 as `Glm5NextForConditionalGeneration` / `Glm5NextForCausalLM`; there is no GLM-5.3 GGUF registration.
* `python/freetoken/kernel/aot_models.py:208-220` describes the Gemma-4 q4_0 path. The GLM-5.3 entry at `:277-289` offers the NVFP4 expert formats, not a GGUF parser.
* `python/freetoken/moe/offload_cache.py:36-78` defines eight bank layouts in `_BANK_SCHEMAS`: `bf16`, `fp8_block`, `q4_0`, `nvfp4`, `nvfp4_marlin`, `nvfp4_b12x`, `mxfp4_triton`, and `ds_fp4`. Lines `:84-96` define six expert byte formulas in `_BANK_BYTES_PER_EXPERT`: `bf16`, `fp8_block`, `q4_0`, `nvfp4`, `mxfp4`, and `ds_fp4`. There is no q2, q3, EXL3, AWQ, GPTQ, HQQ, or MLX bank layout.
* `python/freetoken/models/gemma4/gguf.py:96-116,187-209,382-400` is a Gemma-specific GGUF iterator. Its q4_0 expert banks have Gemma's `gate_up` and `down` tensors; this is not a generic GGUF importer and does not describe GLM-5.3 tensors.
* `python/freetoken/models/glm5_next/weight.py:1-10,58-92,180-202` expects a GLM multimodal-wrapper checkpoint under `model.language_model.*`. It recognizes ModelOpt NVFP4 source keys or the compressed-tensors NVFP4 keys `weight_packed`, `weight_scale`, and `weight_global_scale`.
* `python/freetoken/models/config.py:80-150` recognizes ModelOpt NVFP4 and one exact compressed-tensors NVFP4 arrangement: four-bit float weights, group size 16, and `tensor_group` strategy. MXFP4's group size 32 is rejected.
* `python/freetoken/checkpoint/convert.py:63-86,163-231` does not provide a generic GGUF-to-FreeToken converter. A single GGUF source is treated as a metadata copy and expert loading remains model-specific.

Therefore:

1. A GLM-5.3-Flash GGUF cannot be opened as-is by FreeToken.
2. Changing GGUF metadata cannot make it a GLM safetensors checkpoint. A real route would need a GLM-specific architecture implementation, tensor-name mapping, GGUF tensor decoder, resident-tensor conversion, and expert-bank writer.
3. A hypothetical 2-bit-GGUF-to-q4_0/NVFP4 conversion would be a lossy dequantize-then-requantize operation. The resulting four-bit expert banks still need about 171 GB, and the second quantization cannot recreate information lost by the first.

## GGUF repositories and sizes

The search request was:

`https://huggingface.co/api/models?search=GLM-5.3-Flash&filter=gguf&limit=100&sort=downloads&direction=-1`

It returned 40 repository records. The full search inventory is listed below; search records are not proof that every record has a usable model file. The size table contains the quantization directories whose files were summed with the Hugging Face API's `?blobs=true` metadata. Sizes are decimal GB, and bytes are included so the arithmetic can be reproduced.

### Search inventory (40 repositories)

`unsloth/GLM-5.3-Flash-GGUF`, `antirez/glm-5.3-flash-gguf`, `avar6/GLM-5.3-Flash-BF16-gguf`, `DevQuasar/zai-org.GLM-5.3-Flash-GGUF`, `AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF`, `orcarouter/GLM-5.3-Flash-Uncensored-GGUF`, `AMAImedia/GLM-5.3-Flash`, `Anbeeld/GLM-5.3-Flash-DFlash2-GGUF`, `meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers`, `vcruz305/GLM-5.3-Flash-GGUF`, `qtum/GLM-5.3-Flash-GGUF`, `BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF`, `patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF`, `6block/GLM-5.3-Flash-GGUF`, `aj9o9/GLM-5.3-Flash-GGUF`, `AesSedai/GLM-5.3-Flash-GGUF`, `Sciguy429/GLM-5.3-Flash-BF16`, `vcruz305/GLM-5.3-Flash-DFlash2-GGUF`, `darask0/GLM-5.3-Flash-UNCENSORED-GGUF`, `CoboSan/GLM-5.3-Flash-GGUF`, `sayyidfareed/GLM-5.3-Flash-Spark-Q2XL-MTP`, `DogContext/GLM-5.3-Flash-Uncensored-Q2-ds4`, `ashuaria/GLM-5.3-Flash-GGUF-Finetuning`, `Hagwell/GLM-5.3-Flash-GGUF`, `zurichquants/GLM-5.3-Flash-GGUF`, `lausannequants/GLM-5.3-Flash-GGUF`, `zurichquants/GLM-5.3-Flash-UD-Q4_K_XL-layers`, `Nil626yhh/GLM-5.3-Flash-GGUF`, `lazywold/GLM-5.3-Flash-GGUF`, `lausannequants/GLM-5.3-Flash-UD-Q4_K_XL-layers`, `batiai/GLM-5.3-Flash-GGUF`, `0ppxnhximxr/GLM-5.3-Flash-GGUF`, `Blackfrost-AI/GLM-5.3-Flash-DERISKED-GGUF`, `axiomofmind/GLM-5.3-Flash-W4A16-NVFP4-GGUF`, `D9010/GLM-5.3-Flash-D9010-86GB`, `MorinoNushi/GLM-5.3-Flash-Heretic-LoRA-V1-GGUF`, `Serpen/GLM-5.3-Flash`, `msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44`, `AtomicChat/GLM-5.3-Flash-GGUF`, `petr567/GLM-5.3-Flash-AJ-IQ2-XXS-Strix-Halo-256K`.

### Size-measured quantization builds

| Repository | Quantization directory / build | Files | Bytes | Decimal GB |
|---|---|---:|---:|---:|
| `unsloth/GLM-5.3-Flash-GGUF` | UD-IQ1_M | 3 | 97,579,344,547 | 97.579345 |
| same | UD-IQ1_S | 3 | 93,087,244,963 | 93.087245 |
| same | UD-IQ2_XXS | 4 | 101,844,951,808 | 101.844952 |
| same | UD-Q2_K_XL | 4 | 108,720,071,427 | 108.720071 |
| same | UD-IQ3_XXS | 4 | 120,367,571,715 | 120.367572 |
| same | UD-Q3_K_XL | 4 | 147,535,921,955 | 147.535922 |
| same | UD-IQ4_XS | 5 | 156,822,111,075 | 156.822111 |
| same | UD-Q4_K_XL | 6 | 199,707,321,347 | 199.707321 |
| same | UD-Q5_K_XL | 6 | 240,306,086,912 | 240.306087 |
| same | UD-Q6_K_XL | 7 | 291,833,111,712 | 291.833112 |
| `vcruz305/GLM-5.3-Flash-GGUF` | Q2_K | 1 | 116,728,212,640 | 116.728213 |
| same | Q4_K_M | 1 | 192,880,442,528 | 192.880443 |
| `qtum/GLM-5.3-Flash-GGUF` | IQ1_M | 15 | 70,679,933,792 | 70.679934 |
| same | IQ2_XS | 15 | 92,109,221,728 | 92.109222 |
| same | IQ3_XXS | 15 | 120,994,791,264 | 120.994791 |
| same | IQ4_XS | 15 | 167,320,261,472 | 167.320261 |
| `AesSedai/GLM-5.3-Flash-GGUF` | IQ2_S | 4 | 113,618,904,128 | 113.618904 |
| same | IQ3_S | 4 | 124,717,032,512 | 124.717033 |
| same | IQ4_XS | 5 | 159,175,648,448 | 159.175648 |
| same | Q4_K_M | 6 | 201,982,715,200 | 201.982715 |
| same | Q5_K_M | 6 | 240,826,164,512 | 240.826165 |
| `0ppxnhximxr/GLM-5.3-Flash-GGUF` | IQ2_XS | 16 | 91,964,207,200 | 91.964207 |
| same | IQ3_XXS | 16 | 120,697,688,160 | 120.697688 |
| same | Q4_K_M | 16 | 189,003,386,976 | 189.003387 |
| `aj9o9/GLM-5.3-Flash-GGUF` | AJ-IQ2_XXS | 2 | 87,346,006,560 | 87.346007 |
| same | AJ-IQ3_XXS | 3 | 112,411,167,424 | 112.411167 |
| `DevQuasar/zai-org.GLM-5.3-Flash-GGUF` | Q2_K | 9 | 114,385,292,256 | 114.385292 |
| same | Q3_K_M | 12 | 149,415,931,200 | 149.415931 |
| same | Q4_K_M release | 15 | 189,003,386,528 | 189.003387 |
| same | Q4_K_M f16-named release | 15 | 188,673,828,288 | 188.673828 |
| same | Q5_K_M | 17 | 222,206,152,704 | 222.206153 |
| same | Q6_K | 21 | 257,484,091,616 | 257.484092 |
| `antirez/glm-5.3-flash-gguf` | Q2 | 1 | 96,505,816,384 | 96.505816 |
| same | Q4_K | 1 | 190,875,526,464 | 190.875526 |
| same | FP8 | 1 | 327,209,059,584 | 327.209060 |
| `sayyidfareed/GLM-5.3-Flash-Spark-Q2XL-MTP` | Spark Q2XL mixed build | 8 | 112,194,217,088 | 112.194217 |
| `DogContext/GLM-5.3-Flash-Uncensored-Q2-ds4` | IQ2-imatrix / Q2-ds4 mixed build | 1 | 96,505,818,432 | 96.505818 |

The root totals are not usable as a single quantization size when a repository contains many directories. For example, the API root sum for `unsloth/GLM-5.3-Flash-GGUF` was 2,540,426,767,765 bytes because it included all variants. `Hagwell` and `CoboSan` expose unsloth-style mirrors; `6block` exposes IQ1_M/IQ2_XS/IQ3_XXS/IQ4_XS families. `AtomicChat` had zero payload files at the check time. Layer-only repositories and BF16/Q8/mirror/derivative records were not treated as additional usable low-bit candidates.

No actual GLM GGUF `Q4_0` build was found by the exact API query `GLM-5.3-Flash-Q4_0`; it returned one non-GGUF EXL3-Q4 search record, not a GGUF payload.

## Expert-memory arithmetic for the GGUF families

For the 256-value GGML formats, one GLM expert has 98,304 blocks across gate/up and down. The rows below model the routed-expert banks using the advertised block size. This answers the memory question only; it does not make the formats readable by FreeToken.

| Candidate family (repositories/builds) | Advertised bits | Per expert | All 12,096 experts | Margin vs 119.6 GB | Expert-only fit |
|---|---:|---:|---:|---:|---|
| IQ1_S / UD-IQ1_S | 1 | 4,915,200 B | 59,454,259,200 B / 59.454259 GB | +60.145741 GB | Yes, hypothetical |
| IQ1_M / UD-IQ1_M | 1 | 5,505,024 B | 66,588,770,304 B / 66.588770 GB | +53.011230 GB | Yes, hypothetical |
| IQ2_XXS / UD-IQ2_XXS | 2 | 6,488,064 B | 78,479,622,144 B / 78.479622 GB | +41.120378 GB | Yes, hypothetical |
| IQ2_XS | 2 | 7,274,496 B | 87,992,303,616 B / 87.992304 GB | +31.607696 GB | Yes, hypothetical |
| IQ2_S | 2 | 8,060,928 B | 97,504,985,088 B / 97.504985 GB | +22.095015 GB | Yes, hypothetical |
| Q2_K / Q2 / UD-Q2_K_XL | 2 | 8,257,536 B | 99,883,155,456 B / 99.883155 GB | +19.716845 GB | Yes, hypothetical |
| IQ3_XXS / UD-IQ3_XXS | 3 | 9,633,792 B | 116,530,348,032 B / 116.530348 GB | +3.069652 GB | Barely, hypothetical |
| Q3_K / Q3_K_M / IQ3_S / UD-Q3_K_XL | 3 | 10,813,440 B | 130,799,370,240 B / 130.799370 GB | -11.199370 GB | No |
| IQ4_XS / UD-IQ4_XS | 4 | 13,369,344 B | 161,715,585,024 B / 161.715585 GB | -42.115585 GB | No |
| Q4_K / Q4_K_M / UD-Q4_K_XL | 4 | 14,155,776 B | 171,228,266,496 B / 171.228266 GB | -51.628266 GB | No |

The `AJ-IQ2_XXS` and `AJ-IQ3_XXS` rows are custom-named variants. Their API metadata does not expose a new bank schema, so the corresponding canonical IQ2_XXS/IQ3_XXS row is an estimate, not a claim that the AJ payload is byte-identical. The same caution applies to UD names: the table uses their underlying advertised GGML family, while the full file size in the previous table is the authoritative API payload size.

The two mixed builds have enough theoretical expert memory but still fail the reader test:

| Mixed build | Expert layout | Per expert | All routed experts | Margin |
|---|---|---:|---:|---:|
| Spark Q2XL | gate/up IQ2_XS + down IQ3_XXS | 8,060,928 B | 97,504,985,088 B / 97.504985 GB | +22.095015 GB |
| DogContext Q2-ds4 | gate/up IQ2_XXS + down Q2_K | 7,077,888 B | 85,614,133,248 B / 85.614133 GB | +33.985867 GB |

Spark's README describes a 2.80 BPW mixture. DogContext's README describes an IQ2_XXS + Q2_K imatrix mix and requires ds4/DwarfStar. Neither is a FreeToken bank format.

For context, the higher-bit rows are already above the four-bit floor even before block overhead: Q5 has a raw lower bound of 190.253629 GB, Q6 228.304355 GB, Q8/FP8 304.405807 GB, and BF16 608.811614 GB. They cannot fit this budget and have no direct GLM GGUF path.

## Non-GGUF low-bit builds

The search request was:

`https://huggingface.co/api/models?search=GLM-5.3-Flash-EXL3&limit=100&sort=downloads&direction=-1`

The exact size is the sum of the model payload files returned by `https://huggingface.co/api/models/<repo>?blobs=true`; it is not a download of those files.

| Repository | Format and metadata | Model payload | FreeToken result |
|---|---|---:|---|
| [`vcruz305/GLM-5.3-Flash-EXL3-K2`](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2) | EXL3, 2 bit, MCG trellis, routed-experts-only scope | 97,728,721,536 B / 97.728722 GB | Not recognized; needs ExLlamaV3/custom vLLM |
| [`vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix`](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix) | EXL3 mixed precision, base 2 bit with K3 layers | 103,164,540,888 B / 103.164541 GB | Not recognized; model card calls for a custom plugin |
| [`vcruz305/GLM-5.3-Flash-Uncensored-EXL3-K2`](https://huggingface.co/vcruz305/GLM-5.3-Flash-Uncensored-EXL3-K2) | EXL3, 2 bit, uncensored derivative | 97,728,939,920 B / 97.728940 GB | Not recognized |
| [`Vontra/GLM-5.3-Flash-MLX-oQ2-MTP`](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-oQ2-MTP) | MLX/oQ, 2 bit, group size 64, affine | 110,128,198,215 B / 110.128198 GB | Not recognized |
| [`Vontra/GLM-5.3-Flash-MLX-2bit-MTP`](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-2bit-MTP) | MLX, 2 bit, group size 64, affine | 111,333,617,475 B / 111.333617 GB | Not recognized |
| [`0xSero/GLM-5.3-Flash-EXL3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw) | selective EXL3, 3.0 BPW, TP=4, custom loader | 149,402,871,912 B / 149.402872 GB | Not recognized |
| [`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1) | EXL3, 3 bit, MCG trellis | 136,686,260,192 B / 136.686260 GB | Not recognized |
| `0xSero/GLM-5.3-Flash-EXL3-2.0bpw` | EXL3 search hit, no model payload in API response | 0 B reported | Not a usable measured build |
| `0xSero/GLM-5.3-Flash-EXL3-2.5bpw` | EXL3 search hit, no model payload in API response | 0 B reported | Not a usable measured build |
| `gitcommit90/GLM-5.3-Flash-EXL3-2.05-One-Spark` | EXL3/DFlash2 search hit, no model payload in API response | 0 B reported | Not a usable measured build |

The 149,402,871,912-byte total is all 130 `.safetensors` files: 81 under `layers/` and 49 under `retained/`. The earlier 115,567,508,400-byte figure covered only the 81 `layers/` files.

Other tagged search hits included `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, `turboderp/GLM-5.3-Flash-exl3`, `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, `0xSero/GLM-5.3-Flash-EXL3-Q4`, and `Terra3312/GLM-5.3-Flash-EXL3-4bpw-MUL1`. They are 4-bit/general EXL3 records rather than additional measured sub-3-bit payloads; `turboderp/GLM-5.3-Flash-exl3` reported zero payload files at the check time.

The related four-bit safetensors results were:

| Repository | Format | Payload | Why it is not a FreeToken bank |
|---|---|---:|---|
| [`cyankiwi/GLM-5.3-Flash-AWQ-INT4`](https://huggingface.co/cyankiwi/GLM-5.3-Flash-AWQ-INT4) | AWQ / compressed-tensors integer 4 bit, group 32 | 212,721,952,636 B / 212.721953 GB | Packed integer AWQ keys, not GLM NVFP4 keys |
| [`wtdcode/GLM-5.3-Flash-AWQ-W4A16`](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16) | AWQ / compressed-tensors integer 4 bit, group 128 | 190,810,457,040 B / 190.810457 GB | Wrong packed format and group geometry |
| [`Intel/GLM-5.3-Flash-W4A16-AutoRound`](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound) | AutoRound / GPTQ-like integer 4 bit, group 128 | 181,472,931,028 B / 181.472931 GB | No AutoRound/GPTQ reader or bank schema |
| [`OneNexus/GLM-5.3-Flash-MXFP4`](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4) | MXFP4 safetensors, group 32 | 227,496,161,368 B / 227.496161 GB | `detect_compressed_tensors_nvfp4` rejects group 32; not GLM's group-16 NVFP4 layout |

No HQQ or usable compressed-tensors int2 build was found in the search. The exact official-style compressed-tensors GLM NVFP4 build is a different, four-bit format and is handled separately below.

### Why the formats do not interchange

* EXL3 uses `quant_method=exl3`, trellis/codebook data, and in one build `exl3_selective_tp4`; none is a `_BANK_SCHEMAS` key.
* MLX/oQ uses MLX metadata and group-size-64 affine tensors; it is not the GLM loader's packed/scales/global-scale NVFP4 source.
* AWQ and AutoRound/GPTQ-like builds store integer packed weights with their own group sizes. `compressed-tensors` is a container label, not proof that the payload is the NVFP4 format FreeToken expects.
* MXFP4 uses group size 32. FreeToken's compressed-tensors detector requires the GLM-compatible four-bit float, group-size-16, `tensor_group` arrangement and rejects MXFP4.
* The engine has no q2/q3 bank schema, so a format being small enough in theory does not make it loadable.

For EXL3 and MLX, the raw routed-expert lower bounds are 76.101452 GB at 2 bits and 114.152178 GB at 3 bits. Those are not decoded bank sizes: trellis/codebook/scales and resident tensors add data. The API payload numbers above are full model payloads, not expert-bank measurements.

## Exact four-bit conversion arithmetic

### q4_0

The q4_0 layout uses 32 values in an 18-byte block. Therefore:

```
blocks per expert = 25,165,824 / 32 = 786,432
bytes per expert  = 786,432 * 18 = 14,155,776
all experts       = 14,155,776 * 12,096 = 171,228,266,496 B
                 = 171.228266 GB = 159.468750 GiB
margin            = 119.6 - 171.228266 = -51.628266 GB
```

### NVFP4

The GLM-5.3 NVFP4 source has these exact per-expert banks for `H=4096`, `I=2048`:

```
gate_up_packed  8,388,608 B
gate_up_scale   1,048,576 B
gate_up_global      8,192 B
down_packed     4,194,304 B
down_scale        524,288 B
down_global         8,192 B
per expert     14,172,160 B
all experts  171,426,447,360 B = 171.426447 GB = 159.653320 GiB
margin        -51.826447 GB
```

The actual [`RedHatAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) model payload measured 197,843,812,476 B / 197.843812 GB. Its config has the GLM wrapper, 45 layers, 3 dense layers, 288 experts, and compressed-tensors mixed-precision NVFP4. It has the right sort of GLM reader but still cannot fit the four-bit experts in this machine's budget.

If all q4_0/NVFP4 experts stay in PC memory, the honest PC-memory requirement is about 160 GiB of experts plus about 10 decimal GB for Windows, resident weights, and always-on serving state: about 168.8-169.0 GiB, rounded to roughly 170 GiB. Even putting the full 24 GB card allowance into experts leaves about 146.4-146.6 GiB of PC memory needed after the same 10 GB allowance. The current 95.6 GB is well below both figures.

## Supported alternatives checked

The public model list at `docs/models.md:10-14` includes GLM-4.7, GLM-5.2, GLM-5.3, and Qwen3.6/Qwen3.5 MoE checkpoints. The exact config/API calculations were:

| Supported checkpoint | Routed expert instances | Per expert | All routed experts | Margin vs 119.6 GB | Full model payload from HF API |
|---|---:|---:|---:|---:|---:|
| `RedHatAI/GLM-5.3-Flash-NVFP4` | 12,096 | 14,172,160 B | 171,426,447,360 B / 171.426447 GB | -51.826447 GB | 197.843812 GB |
| `nvidia/GLM-4.7-NVFP4` | 14,240 | 13,287,424 B | 189,212,917,760 B / 189.212918 GB | -69.612918 GB | 229.916298 GB |
| `nvidia/GLM-5.2-NVFP4` | 19,200 | 21,254,144 B | 408,079,564,800 B / 408.079565 GB | -288.479565 GB | 464.823042 GB |
| [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | 10,240 | 1,775,616 B | 18,182,307,840 B / 18.182308 GB | +101.417692 GB | 23.424338 GB |

The Qwen3.6 config returned `Qwen3_5MoeForConditionalGeneration`, `H=2048`, `I=512`, 256 experts, 40 sparse layers, and group-16 W4A16 NVFP4. A direct public request for `nvidia/Qwen3.5-35B-A3B-NVFP4` returned HTTP 401 during the check, so the public Qwen3.6 NVFP4 config was used; the recommendation is still directly listed in FreeToken's supported model documentation and has a large margin.

## Recommendation

There is no direct, no-upgrade path to GLM-5.3-Flash in FreeToken from any of the measured GGUF, EXL3, MLX, AWQ, AutoRound/GPTQ-like, or MXFP4 builds. The only theoretical way to use a GGUF would be a new GLM-specific importer plus a new conversion pipeline; converting its 2-bit experts to q4_0/NVFP4 would still exceed this machine's memory and would not restore lost quality. Use `nvidia/Qwen3.6-35B-A3B-NVFP4` instead; its routed experts are 18.182308 GB and fit with about 101.417692 GB of the stated budget.

## External context

* llama.cpp pull request [#27752, “model: add GLM-5.3-Flash (glm5next)”](https://github.com/ggml-org/llama.cpp/pull/27752) was open at the check time. Its initial description said text-only, not numerically validated against HF, and not run against real weights; later comments reported community conversion/testing. This confirms that a separate GGUF runtime was still being developed, not that FreeToken can read the files.
* The llama.cpp [model-addition guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md) requires a model class, architecture registration, tensor definitions, and tensor-name mappings for a new GGUF architecture.
* [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and its [EXL3 documentation](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md) list GLM-5.3-Flash/EXL3 support. That is a possible separate runtime for EXL3 files, not a FreeToken reader.
