# EXL3 Qwen3.8-Flash-Next in FreeToken — design

Date: 2026-09-25. Branch: `feat/exl3-qwen-flash` (off `mtp-upstream-merge` at 1e1aa1d).

## In plain words

FreeToken learns to run turboderp's 3-bit EXL3 copy of Qwen3.8 Flash as an extra model,
next to today's NVFP4 Flash, which stays the default and is not changed. The model stays in
its small EXL3 form on disk, in PC memory and on the graphics card; each piece is turned
back into normal numbers only at the moment it is used, like a phone's speed-dial. The one
exception is the big word table (the n-gram PLE table): it is written out once onto the SSD
in the format FreeToken already reads, because the SSD has room and a squeezed row costs the
same SSD read as a full one. Pictures and the guess-ahead helper (MTP) work, and the model is
picked from the control panel and the switcher like any other.

Expected payoff: about 20 GB less PC memory for the specialists (routed experts), about
1.5 GB less card memory for the ordinary parts than today's int8 setup, and EXL3's better
quality per bit. Expected cost: some decode speed; we measure and write it down.

## Decisions taken with Jay (2026-09-25)

| Question | Decision |
|---|---|
| NInfer side | Left alone for now (research in this session: weeks of C++ work, EXL3 slower than NInfer's NVFP4 path, QUASAR is NVFP4-trained). |
| Which copy | `turboderp/Qwen3.8-Flash-Next-exl3`, branch `3.05bpw_h5_ng5` (85.1 GB download). |
| Approach | B: every linear stays EXL3 at runtime (not reconstructed to bf16 at load, not re-quantized). |
| Word table | Converted once, offline, to FreeToken's existing table layout on the SSD. |
| First version includes | Control-panel / switcher entry, pictures, guess-ahead (MTP) from the checkpoint. |
| Purpose | "Have the option". Slower than NVFP4 is acceptable if measured and recorded. |

## The checkpoint (measured from the Hugging Face headers, 2026-09-25)

- `config.json` `quantization_config`: `{quant_method: exl3, version: 1.4.4, bits: 3.05,
  head_bits: 5, codebook: mul1, out_scales: always, vision_bits: 5, mtp_bits: 3}`.
- Seven model shards, 52.4 GB, 304,105 tensors; `ngram_embedding.safetensors` 32.64 GB;
  `quantization_config.json` 0.1 GB.
- Every quantized linear is stored as `<name>.trellis` (int16 `[in/16, out/16, 16·K]`),
  `.suh` (fp16 `[in]`), `.svh` (fp16 `[out]`), `.mul1` (scalar). K per tensor, from the trellis
  last dim / 16:

| Tensors | K (bits) |
|---|---|
| routed experts `layers.N.mlp.experts.E.{gate,up,down}_proj` (48 × 512) | 3 |
| `self_attn.indexer.index_qk_proj` (12), MTP indexer | 3 |
| `self_attn.{q,k,v,o}_proj` (12), GDN `in_proj_qkv`, `in_proj_z`, `out_proj` (36) | 5 |
| shared expert `{gate,up,down}_proj` (48), `lm_head` | 5 |
| vision blocks `attn.{q,k,v}_proj`, `attn.proj`, `mlp.linear_fc{1,2}`, merger fc1/fc2 | 5 |
| `mtp.fc_embedding`, `mtp.fc_hidden` | 4 |
| MTP layer: experts 3, attention / shared expert 5 | 3 / 5 |

- Still bf16/fp16: embed_tokens, GDN `in_proj_a`/`in_proj_b`, `conv1d`, `A_log`, `dt_bias`,
  all norms, all hyper-connection weights, router `mlp.gate`, `shared_expert_gate`, PLE
  `key_proj`/`value_proj`/`conv1d`/norms, vision `patch_embed`, `pos_embed`, norms, biases,
  and a bf16 `visual.blocks.N.attn.qkv.weight` shipped **alongside** the q/k/v trellis.
- Expert shapes: gate/up trellis `[160, 40, 48]`, down `[40, 160, 48]` (H=2560, I=640).
  Per expert: 3 × 160·40·48·2 B = 1.84 MB of trellis + suh/svh, against 2.77 MB for NVFP4
  (`daemon/settings/model_info.py:11`). 24,576 experts ≈ 45 GB against ≈ 68 GB.
- Word table: format `exl3_ngram_trellis` v1, K=5, row_dim 160, 320,001,536 rows in 128
  shards of 2,500,012 rows, each row 51 int16 (word 0 = fp16 row scale, words 1-50 = 800
  tail-biting ring bits), plus `head_bias` fp16 `[16, 160]` and `head_offsets`,
  `head_vocab_sizes`, `layer_multipliers`. Decode per row:
  `row[i] = mul1(state_i) · scale + head_bias[head(row)]` (exllamav3
  `modules/quant/exl3_lib/ngram_codec.py:1-82`, GPU kernel `exllamav3_ext/ngram.cu:308-388`).

## Box facts (read-only check, 2026-09-25)

- WSL `vllm`: `/` has 344 GB free, `/mnt/d` 1.4 TB free. Model goes to
  `~/models/Qwen3.8-Flash-Next-exl3-3.05bpw` (85 GB + ~48 GB converted table).
- Installed wheel: `exllamav3 1.4.6+cu132.torch2.11.0` (the one the GLM work uses). Upstream
  head is 1.5.1 (6b84a21). Model card requires ≥ 1.4.5. MIT licence.

## Architecture

Three seams change. Everything else in the Qwen model (GDN recurrence, attention kernels, HC,
PLE lookup, KV, scheduler, governor, parking) is untouched.

### 1. Detection: `exl3` becomes a real Qwen quant flag

`models/qwen4_exp/config.py:224-265` today maps `quant_method: exl3` to `"none"` everywhere,
which would silently build bf16 layers. New rule: `quant_method == "exl3"` sets
`expert_quant = attn_quant = dense_quant = lm_head_quant = "exl3"`. `FREETOKEN_DENSE_QUANT`
does not apply to exl3 layers (it only upgrades `"none"`, which stays true). Any EXL3
checkpoint whose tensors do not match the expected component set fails at load with the
tensor name — never a silent fallback.

### 2. A dense EXL3 linear (`exl3` storage format)

New op `Exl3Linear` (in `kernel/exl3_linear.py`, built by the `models/quant_linear.py`
factories when the format is `exl3`). It owns the trellis/suh/svh/mul1 tensors on the card
and calls `exllamav3_ext.exl3_gemm(x, trellis, y, suh, xh, svh, -1, mcg=None, mul1, 0)`.

- **Workspace:** the wheel's own wrapper allocates `xh` per call when rows > 1, which breaks
  CUDA graphs (`exllamav3_ext/libtorch/linear.cpp:41-43`). `Exl3Linear` instead takes `xh`
  and `y` from a shared, preallocated, size-bucketed workspace (the same pattern as our
  `kernel/exl3_mgemm.py` wrapper), so decode graphs capture fixed addresses.
- **Large prompts:** above 144 rows, reconstruct-then-GEMM is faster in the toolkit
  (`modules/quant/exl3.py:10,166-216`). We use the same threshold: reconstruct the weight into
  a reusable bf16 scratch (`reconstruct_had_slice` in 32,768-column slices), then a normal
  GEMM. Scratch is sized once for the largest dense linear (lm_head slices) and accounted in
  the cache budget.
- **Shape rule:** in-features multiple of 16, out-features multiple of 128
  (`exl3_kernel_map.cu:86-91`). All Flash dense shapes pass (checked in step 1 with
  `exl3_gemm_shape_compat`).
- **Known speed limit:** the wheel's fast one-token GEMV only runs for K 2-4
  (`exl3_gemv.cu:115-122`); our K=5 dense layers take the general GEMM kernel. Measured in
  step 7; fallback below.

Where it plugs in:

- **Attention** (`qwen4_exp/attention.py:131-144`): `qkv_proj` today is one fused bf16/int8
  weight. Trellis tensors cannot be concatenated, so for exl3 we build three `Exl3Linear`s and
  write into one preallocated `[T, 2·qo + 2·kv]` output (for T=1 each output is a contiguous
  slice; for T>1 we concatenate). The existing split at :144 stays as is. `o_proj`,
  `index_qk_proj` (:88, K=3) become `Exl3Linear`.
- **GDN** (`qwen4_exp/gdn.py:91-117,182-187`): mirror the existing fp8 split path: EXL3
  `in_proj_qkv` + `in_proj_z` (two `Exl3Linear`s into one output) and bf16 `in_proj_b|a`
  (fused as today). `out_proj` becomes `Exl3Linear`.
- **Shared expert** (`qwen3_5_moe/moe.py:22-55`): `gate` and `up` as two `Exl3Linear`s into
  one `[T, 2·I]` buffer feeding the existing `silu_and_mul`; `down` as `Exl3Linear`.
- **lm_head** (`qwen4_exp/model.py:223-243`): new `Exl3LMHead` (K=5, 248,320 × 2,560).
- **HC, router, `shared_expert_gate`, GDN a/b, PLE projections:** unchanged bf16.
- **Loader** (`qwen4_exp/weight.py:92-161`): for exl3 checkpoints, `_FUSIONS` entries whose
  parts are trellis-stored are skipped (bf16 fusions such as HC and GDN b|a remain), and
  `.trellis/.suh/.svh/.mul1` keys route to the matching `Exl3Linear` state instead of being
  unknown keys.

### 3. Routed experts: generalise the GLM EXL3 banks to any K

GLM fixed K=2 in several places. Each becomes "K read from the checkpoint, one K for all
routed experts, validated":

- `models/exl3_banks.py:46,199-202,328,404-405` (`_EXL3_K`), docstrings.
- `moe/fused_exl3.py:26,116-118,292,303,462-531` (`_EXL3_K`), and `_MAX_TOP_K` /
  `_MAX_RECONSTRUCT_EXPERTS = 8` (:25): Qwen routes top-10, so the graph-safety rule
  (`decode_is_graph_safe`, :938; `engine.py:3700-3716`) must accept top_k 10 on the packed
  `mgemm` path. The reconstruct path keeps its own limit.
- `kernel/exl3_mgemm.py:44,230-235,303` (`EXL3_MGEMM_K`).
- `moe/offload_cache.py:113-117` and `kernel/aot_models.py:112-120` byte formulas
  (`16·K·2` bytes per 16×16 tile + fp16 suh/svh).
- Activation: Qwen's MoE passes `silu` (`layers/moe.py:374,1016`); the mgemm wrapper already
  has a `silu` branch (`kernel/exl3_mgemm.py:707`). No new activation.
- Engine rules already in place stay: exl3 needs `--moe-backend offload` and card-resident
  inputs (`engine.py:409-413,3645-3664`); GPU-owned layers and learned routing work as for GLM.

**Concurrency:** today's exl3 graph rule allows `max_running_requests == 1` and graph batch
size [1]. The box runs 2 chats. Step 7 measures whether extending the mgemm graph path to
batch sizes 1-2 is cheap; if not, the EXL3 Flash profile ships at 1 chat with graphs and that
limit is written into the control panel entry. Either result is recorded.

### 4. Word table: one-time converter

`scripts/convert_exl3_ngram_table.py <model-dir>`, run once on the box:

- Reads the header, `head_offsets` and `head_bias` from `ngram_embedding.safetensors`
  (layout as in exllamav3 `conversion/ngram.py:465-552`), memory-maps the contiguous
  `(rows, 51)` int16 trellis block, decodes chunks of rows on the GPU with
  `exllamav3_ext.ngram_dequant` (pure-torch `ngram_codec.dequant_rows` as fallback, and as
  the test reference), adds `head_bias`, and writes FreeToken's layout:
  `*.ple.ple_embedding.ngram_embedding.shard_<i>.weight` F8_E4M3 `[2,500,012, 160]` for
  i in 0..127 plus one bf16 `weight_scale` (`qwen4_exp/weight.py:69-73,717-800`).
- Streams slice by slice: never more than one shard (≈0.4 GB fp8) in PC memory; writes to
  `ple-fp8-00001-of-000NN.safetensors` and adds those names to a FreeToken sidecar index
  that `_ple_table_files` (`weight.py:726`) reads, leaving turboderp's own files untouched.
- **Precision gate:** the converter measures the fp8 table against the fp16 decode on a
  1M-row sample: per-row RMS of (fp8·scale − fp16) divided by the row's RMS. If the median of
  that ratio is above 5 %, it writes a bf16 table instead (~102 GB), which works with the
  `disk` and `mmap` backends only; the control panel then refuses `pinned` for this model.
  The report (median, p99, max) is written into the step-7 research note either way.
- The toolkit's `ngram_embedding.safetensors` is ignored at serve time (`_rename` skips it).
- Hash parameters (`head_offsets`, `head_vocab_sizes`, `layer_multipliers`) must equal what
  FreeToken derives from `config.json`; the converter checks and fails loudly on a mismatch.

### 5. Pictures

- `visual.blocks.N.attn.{q,k,v}_proj.*` trellis tensors are skipped: the bf16
  `attn.qkv.weight` shipped next to them is used as today (confirmed equal-shape in step 5
  against the reconstructed q/k/v).
- `attn.proj`, `mlp.linear_fc1/fc2`, merger fc1/fc2 become `Exl3Linear`
  (`vision.py:106-111,149-154,186-187`); `models/vision_weight.py:6-16` accepts EXL3
  components for those names only.
- Layer-stream execution (`vision.py:298-395`) copies each block into one GPU workspace;
  the workspace is built with the same EXL3 component names, shapes and dtypes, so
  `_copy_component_state_` works unchanged. mmap vision (`weight.py:834-989`) assumes one
  bf16 extent; for exl3 the first version uses the plain CPU-held path (not mmap), and the
  control panel sets that automatically.

### 6. Guess-ahead helper (MTP) from the checkpoint

- `models/qwen4_exp/mtp_spike.py:1716-1765` (`build_mtp_weight_plan`, `MTPWeightStore`) learns
  exl3 component names: MTP attention, shared expert, `fc_embedding`, `fc_hidden` become
  `Exl3Linear`; norms and HC stay bf16.
- MTP routed experts (512, K=3, ≈0.94 GB) stay EXL3 on the card through the same
  mgemm banks as the main layers, as a third draft-expert format `exl3` next to `bf16` and
  `nvfp4` (`engine/spec_draft.py:94-178`), chosen automatically for exl3 checkpoints; no
  manifest and no private root needed (`FREETOKEN_MTP_PRIVATE_ROOT` is only the shadow
  observer's trace dir).
- `lm_head` is shared with the main model as today.

### 7. Control panel and switcher

- `daemon/settings/model_info.py:118-230`: `expert_format` returns `exl3` for
  `quant_method: exl3`; `BYTES_PER_EXPERT` gets a K-aware exl3 entry (K from the first
  expert trellis header, not from `bits`, since `bits` is an average); label "3-bit (EXL3)".
- `engine/memory_plan.py:268` `_BANK_BYTES_PER_EXPERT` uses the same K-aware formula, so
  `memory_fit` estimates host and card need. Card need adds the dense EXL3 tensors and the
  reconstruct scratch.
- `model_detect.py` recognises the folder (architecture already supported) and warns if the
  converted word table is missing ("run the converter first").
- The model gets its own registry entry and switcher id (`qwen3.8-flash-exl3`), with the
  dials the exl3 path needs pinned (`--moe-backend offload`, graph batch sizes, vision
  non-mmap). Fit estimate must be within 5 % of a real boot (same bar as the NVFP4 entry).

## Error handling

- Missing or unexpected EXL3 components, a K outside the supported set, a shape the wheel's
  kernel table cannot run, or a wheel missing `exl3_gemm` / `exl3_mgemm`: refuse to boot with
  the tensor name and the reason.
- CUDA graphs requested outside the graph-safe rule: refuse to boot with a clear message
  (same policy as GLM, `2f944ac`), never silently disable.
- Converted table missing or its hash parameters mismatched: refuse to boot, name the
  converter command.
- The NVFP4 Flash path must be bit-for-bit unchanged: every new branch is keyed on
  `quant_method == "exl3"`.

## Build order and tests

Each step ends with a check. Devbox (no GPU) runs the CPU tests; GPU tests skip there and run
on the 5090. Live boots on the 5090 happen only with Jay's OK, one at a time, with the daily
server stopped through the control panel first.

1. **Download and wheel check.** Download `3.05bpw_h5_ng5` to the box. Check the 1.4.6 wheel
   has `exl3_gemm`, `exl3_mgemm`, `ngram_dequant`, `reconstruct_had_slice`, and that
   `exl3_gemm_shape_compat` / `exl3_mgemm_shape_supported` accept every Flash shape at K 3, 4
   and 5. If not, build 1.5.1 from source for cu132 / torch 2.11 and pin it. Try the toolkit's
   own Qwen4Exp loader (CPU-offloaded experts) on 8 fixed prompts to record reference
   greedy tokens and last-layer logits; if it cannot run in the box's memory, the reference is
   per-layer agreement (step 3-4) plus top-token agreement with NVFP4 Flash.
2. **Word-table converter.** CPU test on a synthetic 4-shard table (decode equals
   `ngram_codec.dequant_rows`); box run on the real table with the precision report.
3. **`Exl3Linear` + detection + loader.** CPU tests with synthetic K=3/4/5 tensors against
   `kernel/exl3.py` reconstruct-then-matmul; config test that exl3 is no longer mapped to
   `"none"`; loader test on a synthetic Qwen exl3 index. GPU test: agreement with
   reconstruct+bf16 GEMM at T = 1, 2, 8, 145, 4096; graph capture/replay at T=1.
4. **Experts at any K.** Existing GLM EXL3 tests keep passing at K=2; new K=3 Qwen-shaped
   fixtures; top_k 10 graph rule test.
5. **First boot on the 5090.** Vision and MTP off, 1 chat, graphs on. Check: 8 reference
   prompts (greedy tokens match the step-1 reference or, failing that, stay coherent and agree
   with NVFP4 Flash on the easy ones), and the usual health / `/ready`.
6. **Pictures, then MTP.** Red-circle probe; MTP acceptance rate and tokens/s with MTP on.
7. **Measurements.** Same `ab_send` workloads as the NVFP4 numbers (8k-chat, numbers, essay,
   code, cold/warm TTFT), PC memory free during serving, card memory, boot time; one-chat and
   (if graphs allow) two-chat. Written to
   `docs/research/exl3-qwen-flash-2026-09-XX.md`, with the NVFP4 numbers from the same day.
8. **Control panel.** Detect, fit estimate within 5 % of the step-7 boot, registry/switcher
   entry, start/stop from the page and through the switcher.

## Backup plans

| If | Then |
|---|---|
| Dense K=5 layers make decode much slower (no GEMV at K=5) | Add `FREETOKEN_EXL3_DENSE=int8`: reconstruct the dense EXL3 layers at load and convert them with the existing int8 path (`kernel/triton/int8_linear.py`). Experts stay EXL3, so the 20 GB PC-memory saving stays; the 1.5 GB card saving is lost. |
| The 1.4.6 wheel lacks a routine or a shape | Build 1.5.1 from source on the box; pin it in the venv. |
| fp8 word table adds noticeable error | bf16 table (~102 GB), `disk`/`mmap` backends only. |
| mgemm graphs cannot do 2 chats cheaply | Ship the EXL3 entry at 1 chat; follow-up item. |
| The toolkit's reference run does not fit | Per-layer agreement + NVFP4 top-token agreement as the correctness bar. |

## Out of scope

- NInfer EXL3 support.
- Keeping the word table squeezed at serve time (possible later with `ngram_dequant` after
  the gather; the row read cost on SSD is the same either way).
- Fusing the split projections back into one kernel launch (`exl3_mgemm` over qkv / qkv+z as
  the toolkit does) — a later speed item if step 7 shows launch overhead matters.
- Other EXL3 copies (2.05, 4.05, community quants) — the design is K-general, but only
  3.05 is tested.
- Windows-native serving of the EXL3 model (the box serves from WSL).

## Done means

The 3-bit EXL3 Flash starts from the control panel and the switcher, answers correctly
including pictures, with guess-ahead on; its speed, PC memory and card memory are written
down next to today's NVFP4 Flash; and today's NVFP4 Flash boots and serves exactly as before.
