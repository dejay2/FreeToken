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
| Live run 1 results | `53cdbc4` | docs(moe): live run 1 results (loader hang) |
| Live run 1 fix | `8a63977` | fix(moe): fill GPU-owned banks directly in the placement thread |

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

| Suite | Baseline (cc085cf) | After task 8 | After the live-run fix (`8a63977`) |
|---|---|---|---|
| `tests/engine` | 12 failed, 589 passed, 96 skipped | 12 failed, 622 passed, 96 skipped | 12 failed, 622 passed, 96 skipped |
| `tests/moe` | 31 failed, 222 passed, 148 skipped | 31 failed, 248 passed, 149 skipped | 31 failed, 251 passed, 149 skipped |
| `tests/server` | 548 passed | 552 passed | 552 passed |
| `tests/models/qwen4_exp` | 1 failed, 254 passed, 167 skipped | 1 failed, 254 passed, 167 skipped | not re-run |

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
  -VisionExecution layer-stream -EnableCacheReport -CollectRoutingStats
```

Three corrections the live runs forced on this checklist; all are already applied above
and in the check table:

0. **The RAM criterion named the wrong counter.** It asked for -7.9 GiB of *scheduler
   private bytes*, which this design cannot deliver: the host expert banks are **mapped**
   pages, not private commit (~103 GiB private across the four processes against ~167 GiB
   of attributable commit; the loader reads 63.3 GiB of experts through the mmap path).
   Never allocating six layers' banks removes *mapped* pages, so working set and
   whole-system physical in use move by the full amount while private bytes does not move
   at all: working set **-8.02 GiB**, physical in use **-7.50 GiB**, commit **-4.37 GiB**,
   private bytes **+1.52 GiB**. Read working set and physical in use; treat commit as a
   partial signal and private bytes as no signal. Run 2's row is a PASS on the corrected
   criterion, not the FAIL the original wording produced. Detail in run 2's *Where the RAM
   saving shows up*; the spec's section 9 table carries the same correction.

1. **`-CollectRoutingStats` is required on both boots.** It is what passes
   `--moe-collect-decode-freq`, and without it `/v1/cache/routing` answers **409** -- the
   baseline server did exactly that, so the owned-rows check could not have passed as the
   command was originally written.
2. **The temperature-0 answer-identity check must ask for 48 tokens, not 512.** A 512-token
   temperature-0 answer is not reproducible even against the *same* server (5 runs, 5
   different answers, first divergence 247-1775 B), so it cannot decide anything about the
   candidate. The 48-token form of the same 8k prompt IS stable and was byte-identical
   across two different server processes: md5 `29f0e74dfed744b538f856f38553e1ae` (5 samples,
   3 legacy + 2 fresh). Send it with `"temperature": 0, "max_tokens": 48` and
   `chat_template_kwargs.enable_thinking=false`, and compare that md5.

Live run 1, 2026-09-02, results in `docs/research/measurements-gpu-owned-layers-2026-09-02.md`.
**The candidate never booted**: four attempts (`auto` parallel, `auto` serial, `auto` parallel
again, `auto:2`), all dead in the expert-bank load. Root cause in the section after the table;
fixed in `8a63977`.

Live run 2, 2026-09-02, results in
`docs/research/measurements-gpu-owned-layers-run2-2026-09-02.md`. **The candidate booted and
served** at `auto` / 4,400 slots, in 64.2 s to `state == serving`. The table below is filled
from run 2; its baseline column is run 2's same-session restore boot of `D:\FreeToken` at
6,750 slots, which reproduces run 1's fresh baseline to within 2.4 % on every speed number.

| check | pass criterion | baseline | candidate | verdict |
|---|---|---|---|---|
| boot log shows owned set, LRU size, MTP graphs 6/6 + 7/7 captured | yes | 6/6 + 7/7, 71 s to ready, free VRAM 4.58 GiB | `MoE GPU-owned layers: [0, 1, 2, 6, 7, 22] (6 x 1.32 GiB resident, no host bank); LRU cache 4400 slots for 42 streaming layers`; CUDA graph bs=1 captured; spec 6/6 in 2.485 s, draft 7/7 in 0.649 s; free VRAM 3.27 GiB; 64.2 s to serving | **PASS** |
| scheduler working set (CORRECTED; was "private bytes") | -7.9 GiB +/- 0.5 | - | **-8.02 GiB** | **PASS** |
| whole-system commit | lower; NOT the full 7.9 GiB | 213.74 GiB | 209.37 GiB (**-4.37**) | **PASS** |
| whole-system physical in-use | -7.9 GiB +/- 0.5 | 87.83 GiB (empty ref 18.99 GiB) | 80.33 GiB (empty ref 20.84 GiB) = **-7.50 GiB** | **PASS** |
| boot peak host RAM | <= baseline + 1.5 GiB | 215.45 GiB peak commit, 6.72 GiB min available | 209.16 GiB peak commit (**-6.29**), 14.33 GiB min available | **PASS** |
| 8k-chat decode tok/s (same prompt as the sweep) | recorded; operator decides | 70.4 tok/s (run 1 fresh: 72.1) | **34.6 tok/s** | recorded -- **-50.9 %**, the finding of run 2 |
| TTFT on the same prompt | recorded | cold-7k 5.86 s, warm turn 1.64 s | cold-7k **15.57 s**, warm turn 1.87 s | recorded -- cold TTFT +9.71 s |
| answers at temperature 0, `max_tokens` 48 | identical to baseline (md5 `29f0e74dfed744b538f856f38553e1ae`) | md5 `29f0e74d...`, reproduced again this session | md5 `29f0e74dfed744b538f856f38553e1ae` on 3/3 samples | **PASS** |
| picture request (`-VisionWeights mmap`) | works, latency recorded | 6.62 s first / 4.68 s warm, correct answer | correct answer (`738214`) on all three; **76.91 s first**, then 5.19 s / 6.74 s | **PASS with a caveat** -- the first-request 11x anomaly is unexplained and deserves a probe |
| `/v1/cache/routing` (boot with `-CollectRoutingStats`) | owned rows `resident: true`; streaming rows sane | not collected (step 7 required `boot-2020.ps1` unchanged, and it omits the flag) | all 6 owned rows `resident: true, miss_rate: null, steps: 0`; no non-owned row resident; 42 streaming rows all `steps: 2688`, miss rate 0.120-0.234 (median 0.188); `summary.slots_per_layer 104.76 = 4400/42`, so the summary excludes the owned layers | **PASS** |
| `/v1/cache/status` geometry | shows `gpu_owned_layers` and the LRU size | no `gpu_owned_layers` key; `moe_cache_size: 6750` | `gpu_owned_layers: [0,1,2,6,7,22]`, `moe_cache_size: 4400`; every other geometry field identical to the baseline | **PASS** |
| owned-layer rows on device | byte-identical to a host-bank load (one-off probe script) | - | - | **NOT RUN** -- no such probe script exists in the tree, and comparing device rows to host-bank rows for the same real layer needs a second model process. Indirect live evidence: the 48-token md5 identity above; `tests/moe/test_gpu_owned_banks.py` pins byte-identity on a synthetic checkpoint. |

### Why run 1 did not boot

`GpuOwnedStagingPool.flush` (`host_banks.py:288`) does `dst.copy_(staging.tensor,
non_blocking=True)` on the `PinPipeline` drain thread. The engine loads weights inside
`torch.inference_mode()`, which is **thread-local**, so `alloc_layer_banks` makes each owned
layer's device bank an *inference tensor* while the drain thread is not in inference mode:

```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
```

With `auto` (six layers) that error is invisible: `PinPipeline._run` stores it in `self._exc`
and then drains the queue without running anything, so no staging slot is ever returned, and
the single-threaded NVFP4 placement loop blocks forever in `_acquire_locked` (cap 2) the moment
it touches a third owned layer. `self._exc` is only re-raised by `wait()`/`__exit__`, which the
blocked loop never reaches -> a silent, CPU-idle hang. With `auto:2` there is no third acquire,
so the loop finishes and the boot crashes with the stack above instead.

Both faces reproduce off-server in seconds (`repro_pipeline_real.py`, `repro_loader.py`,
`repro_staging_deadlock.py` in the run's scratchpad); the hang was also confirmed on the live
process with `py-spy dump`. Note the second face is a design issue in its own right: with a
single-threaded placement loop, cap-2 back-pressure has no other thread that can complete a
layer, so `auto`'s six layers deadlock the parallel reader even once the copy is fixed.

### How it was fixed (`8a63977`, before the next live run)

The staging pool is **gone**. `GpuOwnedBank.fill` IS its device tensor, so each
`fill[expert] = row` is a synchronous pageable H2D copy issued by the placement thread
itself, inside the loader's own `inference_mode()`: no staging layer, no cap, no
back-pressure, no CUDA event, no second thread, and therefore neither failure above by
construction. Six owned layers are 7.9 GiB of such copies (a few seconds of boot); that cost
is the trade. `PinPipeline.__call__` now settles nothing for a `GPU_OWNED` layer (the
completion tracker still counts it, so the loaders' `placed` asserts are unchanged), and
`submit_flush` is removed. With no blocking path left on the drain thread its post-failure
"drain without settling" can strand no one -- a failing settle surfaces at `wait()`, which is
now pinned by a test.

Regression tests (`tests/moe/test_gpu_owned_banks.py`) run the REAL NVFP4 loaders -- serial,
and parallel with a strict round-robin interleave of three owned layers -- over a synthetic
checkpoint **inside `torch.inference_mode()`**, and assert byte-identity against a plain
host-bank load. All three hang or fail on the pre-fix code. That one context manager was the
whole difference between the green suite and this run.

Hazards, from the boot notes for this box: only ever kill `python.exe` from
`nvidia-smi --query-compute-apps`; settle 45-60 s between servers or the last expert bank
dies in `cudaHostRegister failed ... out of memory`; one Claude session booting servers at a
time; send test requests with `chat_template_kwargs.enable_thinking=false`.

## Only a live GPU run can decide this

The CPU suite pins every branch, every refusal and all the arithmetic. These cannot be
covered without the device, and are what the table above exists to settle. **Run 2 settled all
six**; each item carries its outcome, and the detail is in
`docs/research/measurements-gpu-owned-layers-run2-2026-09-02.md`.

| # | outcome in run 2 |
|---|---|
| 1 | **WORKS, and it is free.** The expert-bank load phase took 38 s against the baseline's 43 s -- the 7.9 GiB of synchronous pageable H2D copies cost *less* than building the same six host banks. The predicted "a few seconds of boot" overhead did not appear. |
| 2 | **not run** -- needs a second model process (see the check table). Indirect: the 48-token greedy answer is byte-identical to the baseline. |
| 3 | **lower, as predicted**: 209.16 vs 215.45 GiB peak commit; 14.33 vs 6.72 GiB minimum available. |
| 4 | **PASS, no change**: bs=1 captured, spec 6/6 in 2.485 s, draft 7/7 in 0.649 s, ladder replays 6/6. |
| 5 | **PASS** -- `test_copy_plan_holds_a_zero_placeholder_for_gpu_owned_layers` was run with the device visible (in the empty window between servers) and passed; the whole `-k "gpu_owned or fused_copy_plan"` selection is 12 passed, 0 skipped. |
| 6 | **-50.9 % decode, +9.71 s cold TTFT** (34.6 vs 70.4 tok/s). This is the finding of run 2 and the reason to reconsider the owned set. |


1. The real pageable H2D of the owned layers (`E*6` per-slice copies each) against a CUDA
   device: that it works at all -- the CPU test only proves the placement -- and what it
   costs in boot seconds. Rough expectation: 7.9 GiB of pageable H2D, a few seconds.
2. That the resident VRAM rows are byte-identical to the host-bank rows a normal load
   produces for the same layer (the one-off probe script).
3. Boot peak host RAM. It should now be *lower* than the staging design's: nothing extra is
   allocated on the host at all for an owned layer.
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
6. Residual risk, out of the spec's scope: `alloc_layer_banks` honours the ambient plan for
   EVERY provider, but only the NVFP4 loaders were reviewed. `pin_banks` providers still
   refuse loudly (item 3). Since `8a63977` the `PinPipeline` providers no longer hit an
   assert: with no staging indirection, a provider that writes through either `.fill` or
   `.tensor` writes the device tensor directly, so it may now *appear* to work -- untested,
   on a checkpoint whose row geometry the owned path was never checked against. Prefer the
   quant-format guard in item 7 over relying on an accident. The flag is documented as
   Qwen3.8-Flash-Next-NVFP4 only.
7. `--moe-gpu-owned-layers` is validated against `--moe-cpu-layers`, the backend, the FTW
   path and the LRU floor, but nothing checks the checkpoint's expert quant format. See 6.
