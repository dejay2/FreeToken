# GPU-owned MoE layers -- implementation status and live checklist

Spec: `docs/design/2026-09-02-qwen38-gpu-owned-moe-layers-design.md`
Plan: `docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-plan.md`
Branch: `gpu-owned-layers` (worktree `D:\FreeToken-gpu-owned-layers`, cut from
`mtp-upstream-merge` @ cc085cf). Not pushed, no PR opened.

## Commits

One row per plan task.

| Task | Commit | Subject |
|---|---|---|
| 1 Config, flag, parser, validation | `5f0f740` | feat(moe): --moe-gpu-owned-layers spec, resolver and validation |
| 2 VRAM budget arithmetic | `35fedb7` | feat(moe): reserve VRAM for GPU-owned MoE layers in the cache budget |
| 3 Cache owned-layer registry | `6163701` | feat(moe): GPU-owned layer registry in OffloadMoeCache |
| 4 Forward paths | `23479f6` | feat(moe): forward path for GPU-owned MoE layers |
| 5 Loader and refusals | `5bebdfc` | feat(moe): load GPU-owned MoE layers straight into VRAM, no host bank |
| 6 Engine wiring and reports | `a01736d` | feat(moe): wire GPU-owned MoE layers through the engine and the reports |
| 7 Launcher and docs | `f0acdf1` | feat(launcher): -GpuOwnedLayers passthrough and docs |
| 8 This checklist | `00ffa83` | docs(moe): operator checklist for the GPU-owned MoE layer live run |

## CPU test results

Runner (no GPU -- PowerShell **deletes** an env var assigned `''`, so `'-1'` is the
portable spelling of `CUDA_VISIBLE_DEVICES=""`; both hide every device):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONPATH = 'D:\FreeToken-gpu-owned-layers\scripts\windows-ple-mmap;D:\FreeToken-gpu-owned-layers\python;<scratch>\pytest-site'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest <paths> -q -p no:cacheprovider --timeout=900
```

New and changed files (all green):

| Suite | Command | Passed | Failed | Skipped |
|---|---|---|---|---|
| new: parser/validation/launcher/checklist | `pytest tests/engine/test_moe_gpu_owned_layers.py` | 28 | 0 | 0 |
| new: loader | `pytest tests/moe/test_gpu_owned_banks.py` | 9 | 0 | 0 |
| new: geometry | `pytest tests/server/test_gpu_owned_geometry.py` | 4 | 0 | 0 |
| changed: cache + forward | `pytest tests/moe/test_offload.py` | 37 | 1 (pre-existing) | 4 |
| changed: budget | `pytest tests/engine/test_cache_budget.py` | 29 | 2 (pre-existing) | 0 |
| changed: routing | `pytest tests/moe/test_routing_stats.py` | 24 | 0 | 0 |
| changed: MTP verify | `pytest tests/engine/test_mtp_fast_verify.py` | 37 | 0 | 0 |

Whole suites, measured before the branch (at `cc085cf`) and after task 8, same box, same
runner. **No failure count moved**; every new test is additive:

| Suite | Baseline (cc085cf) | After |
|---|---|---|
| `tests/engine` | 12 failed, 589 passed, 96 skipped | 12 failed, 622 passed, 96 skipped |
| `tests/moe` | 31 failed, 222 passed, 148 skipped | 31 failed, 248 passed, 149 skipped |
| `tests/server` | 548 passed | 552 passed |
| `tests/models/qwen4_exp` | 1 failed, 254 passed, 167 skipped | 1 failed, 254 passed, 167 skipped |

(The extra `tests/moe` skip is
`test_offload.py::test_copy_plan_holds_a_zero_placeholder_for_gpu_owned_layers`, which is
`skipif(not torch.cuda.is_available())` -- see item 5 below.)

### Pre-existing failures on this box (unchanged by this branch, do NOT chase)

| Count | File | Cause |
|---|---|---|
| 22 | `tests/moe/test_small_prefill_movement.py` | triton: `0 active drivers` -- no CUDA visible |
| 6 | `tests/engine/test_spec_draft.py` | `RuntimeError: No CUDA GPUs are available` |
| 6 | `tests/moe/test_moe_predict_log.py` | triton: `0 active drivers` |
| 4 | `tests/engine/test_spec_sampler.py` | `RuntimeError: No CUDA GPUs are available` |
| 2 | `tests/engine/test_cache_budget.py` | flashinfer not installed (attention-backend `fi`) |
| 1 | `tests/moe/test_offload.py` | flashinfer not installed (attention-backend `fi`) |
| 1 | `tests/moe/test_mtp_spike_experts.py` | triton: `0 active drivers` |
| 1 | `tests/moe/test_mtp_spike_nvfp4_resident.py` | triton: `0 active drivers` |
| 1 | `tests/models/qwen4_exp/test_qsa_step_workspace.py` | needs a device |

`tests/scheduler` (not touched by this branch) fails 107 cases on this box, all
`RuntimeError: No CUDA GPUs are available`.

## Live verification

One server at a time. Boot the baseline and the candidate as FRESH boots, same prompts.

- Baseline: today's flags, `-MoECacheSize 6750`, no `-GpuOwnedLayers`.
- Candidate: the same flags plus `-GpuOwnedLayers auto -MoECacheSize <N>`, with N computed
  from the baseline boot log's free memory so ~1 GiB stays free after the MTP graphs (first
  guess 4400; the engine prints the fit and refuses a size that does not).

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath D:\Models\Qwen3.8-Flash-Next-NVFP4 `
  -Port 2020 -ContextTokens 65536 -KVCacheTokens 65536 `
  -MoECacheSize 4400 -GpuOwnedLayers auto `
  -DenseQuant int8 -EmbedHost -EnableVision `
  -VisionPackagesPath D:\FreeToken\.local\vision-packages `
  -VisionExecution layer-stream -EnableCacheReport
```

| check | pass criterion | baseline | candidate | verdict |
|---|---|---|---|---|
| boot log shows owned set, LRU size, MTP graphs 6/6 + 7/7 captured | yes | | | |
| scheduler private bytes and whole-system commit | -7.9 GiB +/- 0.3 | | | |
| whole-system physical in-use | -7.9 GiB +/- 0.5 | | | |
| boot peak host RAM | <= baseline + 1.5 GiB | | | |
| 8k-chat decode tok/s (same prompt as the sweep) | recorded; operator decides | | | |
| TTFT on the same prompt | recorded | | | |
| answers at temperature 0 | identical to baseline | | | |
| picture request (`-VisionWeights mmap`) | works, latency recorded | | | |
| `/v1/cache/routing` | owned rows `resident: true`; streaming rows sane | | | |
| owned-layer rows on device | byte-identical to a host-bank load (one-off probe script) | | | |

Hazards, from the boot notes for this box: only ever kill `python.exe` from
`nvidia-smi --query-compute-apps`; settle 45-60 s between servers or the last expert bank
dies in `cudaHostRegister failed ... out of memory`; one Claude session booting servers at a
time; send test requests with `chat_template_kwargs.enable_thinking=false`.

## Only a live GPU run can decide this

The CPU suite pins every branch, every refusal and all the arithmetic. These cannot be
covered without the device, and are what the table above exists to settle:

1. The `cudaHostAlloc`'d staging banks and the real `copy_(non_blocking=True)` H2D, including
   the CUDA event that gates staging reuse when two owned layers are in flight. The CPU test
   proves the pool's cap-2 back-pressure and slot recycling; it cannot prove the event.
2. That the resident VRAM rows are byte-identical to the host-bank rows a normal load
   produces for the same layer (the one-off probe script).
3. Boot peak host RAM with cap-2 staging live (the CPU test proves the bound, not the cost).
4. That the decode and MTP CUDA graphs still capture (bs=1, MTP widths 1-6): the owned path
   is fixed-shape reads of fixed-address tensors, strictly simpler than today's, so no
   capture change is expected -- but "expected" is not "observed".
5. The `_build_fused_copy_plan` 0-placeholder assertion for an owned layer (the test is
   `@pytest.mark.skipif(not torch.cuda.is_available())`; the fused plan is only built on a
   CUDA device). The same run also exercises the corrected `feat` row-geometry, which now
   reads the first STREAMING layer instead of layer 0.
6. The actual speed cost of dropping the LRU from 6,750 to ~4,400 slots over 42 streaming
   layers -- spec section 11's open risk, and the only reason to keep, shrink or drop the
   owned set.

## Deviations from the plan, and residual risks

1. `_adjust_config`'s new validator call is gated on `config.moe_gpu_owned_layers` being set.
   As written in the plan it ran for every MoE config and broke seven existing stub-config
   tests that build a partial `model_config` without `num_moe_layers`. Behaviour is identical
   (`_validate_gpu_owned_layers` returns immediately on a falsy spec).
2. `Engine._gpu_owned_layer_ids` is also a CLASS attribute, not only an `__init__` binding:
   `tests/engine/test_cache_budget.py` builds an `Engine.__new__(Engine)` stub to exercise
   `_resolve_auto_moe_cache_size` without a GPU.
3. `pin_banks` refuses a GPU_OWNED plan up front, before settling any bank, in addition to
   the `_settle` guard the plan specifies. With only the `_settle` guard the failure depended
   on which layer was settled first: layer 0's `pin()` ran first and died in
   `cudaHostRegister` instead of the intended message.
4. `_build_fused_copy_plan` reads its per-bank row geometry (`feat`) from
   `self._first_streaming_layer`, not `per_layer[0]`. This is a FOURTH layer-0 special case
   beyond the three spec section 11 names; a GPU-owned layer 0 whose row shape differs would
   otherwise have set the wrong copy width for every layer.
5. The plan's task-5 verification command names `tests/checkpoint`, which does not exist in
   this tree (there are no FTW-loader tests outside the new one). `tests/moe` was run instead.
6. Residual risk, out of the spec's scope: `alloc_layer_banks` now honours the ambient plan
   for EVERY provider, but only the NVFP4 loaders were converted to write through
   `bank.fill`. A non-NVFP4 MoE checkpoint booted with `--moe-gpu-owned-layers` fails loudly
   rather than silently: the `pin_banks` providers hit the refusal in item 3, and the
   `PinPipeline` providers hit `GpuOwnedStagingPool.flush`'s
   `"GPU-owned layer N never acquired staging"` assert. Neither is a wrong-memory read, but
   neither is a friendly message either. The flag is documented as Qwen3.8-Flash-Next-NVFP4.
7. `--moe-gpu-owned-layers` is validated against `--moe-cpu-layers`, the backend, the FTW
   path and the LRU floor, but nothing checks the checkpoint's expert quant format. See 6.
