# FP8 KV accuracy — QSA layer 3

This is the repaired offline FP8 E4M3 replay of `D:\kv-context-parallel\qsa-layer-3.pt`. The snapshot has one captured final query row, 8,192 active written KV tokens, 128 active pages, and 2,048 captured selected tokens. Temperature-0 48-token server identity remains **DEFERRED** because FP8 storage is not implemented in the engine.

## Repaired result summary

The snapshot contains the production main K/V slabs and the captured selection, but it does not contain the separate BF16 compressed QSA index slab or indexer query. The repaired checker therefore performs two independent, non-aliased BF16-main-K and FP8-dequantized-main-K replays through the same QSA-shaped block score (`ReLU(dot)`, top-k, and causal-tail expansion), and labels that fallback explicitly. A future capture with the compressed slab can replace this diagnostic path; these results must not be described as a complete production compressed-index proof.

| scaling | true selection agreement % (independent main-K replay) | aggregate relative L2 % | worst per-head relative L2 % | K saturation % | V saturation % | scale bytes/token | PASS/FAIL against P0 as written | meets recommended gate (selection agreement 100%) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| per-layer | 99.0234375 | 2.24723297 | 5.06217033 | 0.0000476837158 | 0 | 0.0009765625 | **FAIL / RP** | no |
| per-head | 99.4140625 | 2.20654501 | 3.78707871 | 0.0000715255737 | 0.0000238418579 | 0.001953125 | **FAIL / RP** | no |
| per-head-per-block-64 | 99.4140625 | 2.45133569 | 3.51648256 | 0.0101566315 | 0.0106573105 | 0.25 | **FAIL / RP** | no |

The aggregate-output and worst-head error limits fail in all three modes. The independent selection replay is below the recommended 100% gate in all three modes. Block-mode K/V saturation also exceeds the P0 0.01% limit after using the correct active-token denominator.

## Superseded (buggy M2/M2b measurements)

These rows are retained for audit only. The old checker compared `indices` with itself and divided saturation by the entire allocated pool, so they are not evidence for the repaired gate.

| scaling | scale values | scale bytes | scale bytes/token | agreement | aggregate L2 | worst per-head L2 | K saturation | V saturation | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| per-layer | 2 | 8 | 0.0009765625 | 100% (self-comparison) | 2.24723297% | 5.06217033% | 0.00000595464939% (wrong denominator) | 0% | RP |
| per-head | 4 | 16 | 0.001953125 | 100% (self-comparison) | 2.20654501% | 3.78707871% | 0.00000893197409% (wrong denominator) | 0.0000029773247% (wrong denominator) | RP |
| per-head-per-block-64 | 512 | 2048 | 0.25 | 100% (self-comparison) | 2.45133569% | 3.51648256% | 0.00126834032% (wrong denominator) | 0.00133086414% (wrong denominator) | RP |

## Reference validation

The repaired checker now compares its BF16 CPU replay with the captured production `out` tensor before measuring FP8 error. The BF16 replay differs from the captured output by `0.168923842%` relative L2 and `0.00467467308` maximum absolute error. This drift is recorded for audit and is not silently ignored.

## Per-mode results

### Scaling: `per-layer`

- K/V scales: one FP32 K scalar and one FP32 V scalar for the layer.
- Active written pages/tokens used for calibration and saturation: `128` / `8192`.
- Scale metadata: `8` bytes total, `0.0009765625` bytes per active token.
- Selection path: `main-K score replay (compressed index absent from snapshot)`.
- Selection agreement by query block: query block `0`, `2,028/2,048` tokens, `99.0234375%`.
- P0 decision: **FAIL / RP**.
- Meets recommended gate (selection agreement 100%): **no**.

| commit | layer | prompt_tokens | valid_kv_tokens | k_scale | v_scale | k_max_abs | v_max_abs | k_saturation_pct | v_saturation_pct | selected_tokens | selected_agreement_pct | output_relative_l2_pct | head_relative_l2_mean_pct | head_relative_l2_p99_pct | head_relative_l2_max_pct | output_max_abs | nan_inf_count | bf16_48_hash | fp8_48_hash | first_48_identical | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 355a735 | 3 | 8192 | 8192 | 0.0138811385 | 0.0103934156 | 6.21875 | 4.65625 | 0.0000476837158 | 0 | 2048 | 99.0234375 | 2.24723297 | 2.1097213 | 4.94714902 | 5.06217033 | 0.0837649107 | 0 | DEFERRED | DEFERRED | DEFERRED | RP |

### Scaling: `per-head`

- K/V scales: one FP32 K/V scale pair per KV head.
- Active written pages/tokens used for calibration and saturation: `128` / `8192`.
- Scale metadata: `16` bytes total, `0.001953125` bytes per active token.
- Selection path: `main-K score replay (compressed index absent from snapshot)`.
- Selection agreement by query block: query block `0`, `2,036/2,048` tokens, `99.4140625%`.
- P0 decision: **FAIL / RP**.
- Meets recommended gate (selection agreement 100%): **no**.

| commit | layer | prompt_tokens | valid_kv_tokens | k_scale | v_scale | k_max_abs | v_max_abs | k_saturation_pct | v_saturation_pct | selected_tokens | selected_agreement_pct | output_relative_l2_pct | head_relative_l2_mean_pct | head_relative_l2_p99_pct | head_relative_l2_max_pct | output_max_abs | nan_inf_count | bf16_48_hash | fp8_48_hash | first_48_identical | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 355a735 | 3 | 8192 | 8192 | min=0.0119977677;max=0.0138811385;groups=2 | min=0.00645228801;max=0.0103934156;groups=2 | 6.21875 | 4.65625 | 0.0000715255737 | 0.0000238418579 | 2048 | 99.4140625 | 2.20654501 | 2.06559766 | 3.54957595 | 3.78707871 | 0.100328922 | 0 | DEFERRED | DEFERRED | DEFERRED | RP |

### Scaling: `per-head-per-block-64`

- K/V scales: one FP32 K/V scale pair per KV head in each 64-token page.
- Active written pages/tokens used for calibration and saturation: `128` / `8192`.
- Scale metadata: `2,048` bytes total, `0.25` bytes per active token.
- Selection path: `main-K score replay (compressed index absent from snapshot)`.
- Selection agreement by query block: query block `0`, `2,036/2,048` tokens, `99.4140625%`.
- P0 decision: **FAIL / RP**.
- Meets recommended gate (selection agreement 100%): **no**.

| commit | layer | prompt_tokens | valid_kv_tokens | k_scale | v_scale | k_max_abs | v_max_abs | k_saturation_pct | v_saturation_pct | selected_tokens | selected_agreement_pct | output_relative_l2_pct | head_relative_l2_mean_pct | head_relative_l2_p99_pct | head_relative_l2_max_pct | output_max_abs | nan_inf_count | bf16_48_hash | fp8_48_hash | first_48_identical | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 355a735 | 3 | 8192 | 8192 | min=0.0107421875;max=0.0138811385;groups=256 | min=0.00495256716;max=0.0103934156;groups=256 | 6.21875 | 4.65625 | 0.0101566315 | 0.0106573105 | 2048 | 99.4140625 | 2.45133569 | 2.0990992 | 3.5097311 | 3.51648256 | 0.11141789 | 0 | DEFERRED | DEFERRED | DEFERRED | RP |

## P0 gate

- Selection agreement threshold: at least 97.0% (the repaired independent fallback passes this arithmetic threshold, but not the recommended 100% gate).
- Aggregate relative L2 threshold: at most 1.0% — **failed in all modes**.
- Maximum per-head relative L2 threshold: at most 2.0% — **failed in all modes**.
- K/V saturation threshold: at most 0.01% of active written values — **passed for per-layer and per-head; failed for both block-mode K and V**.
- NaN/Inf threshold: zero — **passed in all modes**.

**FAIL / RP.** Do not start B1 from these offline numbers. The live BF16-versus-FP8 48-token identity test remains **DEFERRED** until FP8 storage exists.
