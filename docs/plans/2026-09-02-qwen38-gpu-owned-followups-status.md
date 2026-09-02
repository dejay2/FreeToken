# GPU-owned MoE layers: follow-up issues -- status

Closes the seven issues left open by the GPU-owned MoE layer feature
(`docs/design/2026-09-02-qwen38-gpu-owned-moe-layers-design.md`,
`docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-status.md`), found during the build and
the three live runs.

Branch: `gpu-owned-followups`, worktree `D:\FreeToken-gpu-owned-followups`, cut from
`mtp-upstream-merge` @ `0964145` (the merge of the feature). **Not pushed, no PR opened.**
The code here was not run against the GPU while it was written: a live server owned the
card throughout, and every test ran with `CUDA_VISIBLE_DEVICES=-1`. What needs the device
is in the [operator live checklist](#operator-live-checklist) below, which was **run live
on 2026-09-02** -- see `docs/research/measurements-gpu-owned-followups-live-2026-09-02.md`.
Ten of the twelve checks pass; **L4 failed** (the headroom refusal named a size that could
not be typed) and L12 is inconclusive. L4 is **fixed in 42e3134** and needs one live boot to
confirm.

## Commits

| Issue | Commit | Subject |
|---|---|---|
| 1 VRAM budget ignores post-cache allocations | `3b1cc69` | fix(moe): charge post-cache VRAM reservations before sizing the expert cache |
| 1 (review follow-up) | `c0bb740` | fix(moe): reserve only what the boot allocates, and print the ledger where it is true |
| 2 `/v1/cache/status` hides the owned reservation | `6f2f807` | fix(server): expose the GPU-owned VRAM reservation in the cache geometry |
| 3 quant-format guard | `9fdb2a7` | fix(moe): refuse --moe-gpu-owned-layers on non-NVFP4 expert banks |
| 4 placement report cannot list owned layers | `d1fbdb3` | fix(engine): report weight placement where the placement is known |
| 5 docs and criteria corrections | `1427088` | docs(moe): correct the RAM criterion and the -MoECacheSize floor |
| 6 operator stop script | `72f5bc6` | feat(scripts): stop-qwen38-flash-next-windows.ps1 |
| 7 refusal messages and `auto:N` | `709272c` | fix(moe): name the fix in both owned-layer refusals, and refuse a too-large auto:N |
| L4 live-check failure (the refusal named an LRU count under the total's flag) | `42e3134` | fix(moe): the explicit-cache refusal quotes the typed total and names a total |

`3b1cc69`, `6f2f807` and `9fdb2a7` are omp's; the rest are this session's. `c0bb740` is a
review follow-up on omp's issue-1 commit and is described under issue 1 below.

## What was fixed, per issue

### 1. The VRAM budget ignored everything allocated after the MoE cache was sized

The real bug behind live run 2. `net_cache_budget_bytes` / `plan_cache_budget`
(`engine/cache_budget.py`) and `Engine._resolve_auto_moe_cache_size` subtracted KV + GDN
pools, weights and the GPU-owned reservation, but nothing allocated *afterwards*: the
integrated MTP resident draft head (2.17 GiB), the decode/spec/draft CUDA-graph pools, and
the vision layer-stream workspace. `--moe-cache-auto` would spend those bytes on expert
slots and the card would page at decode peak.

**omp's `3b1cc69`:**

* `engine/cache_budget.py`: `DEFAULT_MOE_VRAM_RESERVE_BYTES`,
  `DEFAULT_MOE_CACHE_HEADROOM_BYTES` (1.5 GiB), `post_cache_reserved_bytes()` as the single
  declaration point, `format_vram_ledger()`, and `check_explicit_moe_cache_fits()` extended
  with `reserved_bytes`.
* `engine/engine.py`: the reserve joins `fixed_cache_size` in `_resolve_auto_moe_cache_size`
  exactly like `state_pool_bytes`; `_check_gpu_owned_cache_fits` became
  `_check_explicit_cache_fits` and now runs for **every** explicit `--moe-cache-size`, with
  or without owned layers, failing loudly (never shrinking) and naming the largest slot
  count that fits.
* `engine/config.py`, `server/args.py`: `--moe-vram-reserve-bytes`,
  `--moe-cache-headroom-bytes`. `docs/cli.md`: rows for both.

**Reviewed, and two gaps fixed in `c0bb740`:**

1. *The reserve was charged unconditionally.* Its biggest term is the resident MTP draft
   head, which only exists when speculation is on, so every boot **without** speculation
   lost ~850 expert slots to bytes nothing would allocate -- and an explicit
   `--moe-cache-size` that fits perfectly well could be refused. Reserving bytes nothing
   will allocate is the same sizing error as spending bytes something will.
   `cache_budget.py` now names the two halves the 3 GiB was composed of --
   `MTP_DRAFT_HEAD_RESERVE_BYTES` (2.25 GiB) and `GRAPH_POOL_RESERVE_BYTES` (0.75 GiB, the
   graph pools and vision workspace, allocated on every boot) -- and
   `auto_vram_reserve_bytes()` / `resolve_vram_reserve_bytes()` compose the reserve for the
   boot at hand. `--moe-vram-reserve-bytes` defaults to `-1` (auto);
   `Engine._post_cache_reserve` decides `mtp_resident` from `config.spec_decode.enabled` or
   a resident MTP shadow observer. `>= 0` is still taken as typed, `0` still reserves
   nothing, and `DEFAULT_MOE_VRAM_RESERVE_BYTES` is unchanged at 3 GiB (both halves).
2. *The ledger was printed where its KV row could not be true.* It was emitted inside the
   MoE cache build, which runs **before** the KV pool is sized (the pool takes what the
   cache left), so its KV row read `num_page_override or 0` -- zero pages on every boot that
   does not pass `--num-tokens` -- and `unaccounted` silently absorbed the whole KV pool. It
   also printed the raw `-1` sentinel for the reserve. The cache build now stashes what only
   it knows (`_stash_vram_ledger_inputs`) and `_log_vram_ledger(config)` prints the block
   straight after `create_kv_pool`, where every term is the number the boot took; the
   reserve row prints the resolved bytes. The whole block is wrapped: a ledger is a log line
   and must never fail a boot.

   Also in `c0bb740`: the launcher had no way to pass either knob, so a wrong reserve on the
   operator's box could only be worked around by editing the engine.
   `-MoEVramReserveBytes` / `-MoECacheHeadroomBytes` pass through when `>= 0`. The Windows
   guide gained a section for both flags and a worked ledger block.

### 2. `/v1/cache/status` hid the owned reservation

`cache_budget_bytes` was byte-identical with and without six owned layers
(23,564,753,305 B), so nothing an operator could poll showed that 7.93 GiB of the card was
spoken for. omp's `6f2f807`:

* `kvcache/cache_status.py`: `compute_cache_pools` reports `gpu_owned_reserved_bytes`,
  **measured** from `OffloadMoeCache.resident_banks` rather than derived, beside the
  existing `gpu_owned_layers`.
* `server/api_server.py`: `cache_geometry` carries `gpu_owned_reserved_bytes` (engine ack
  first, the `num_experts x per-expert` product as the fallback for an older ack);
  `_cache_limits`' `moe_experts` ceiling counts **streaming** experts only and spends a MoE
  budget with the owned bytes removed -- as a separate budget, so the KV/window/GDN ceilings
  (which the owned banks do not compete with) are untouched.

Reviewed and left as written. `cache_report.cache_rate` and `model_meta.moe_total_experts`
already denominated over streaming layers; the tests now pin that alongside the new fields.

### 3. Quant-format guard

Only the NVFP4 providers were ever reviewed for filling an owned layer's **device** banks,
and since `8a63977` removed the staging indirection a provider writing through
`.fill`/`.tensor` writes device memory directly -- so an unreviewed one no longer trips an
assert and may silently appear to work. omp's `9fdb2a7` adds two guards:
`_validate_gpu_owned_layers` refuses at config time (before the 40 s expert load), and
`host_banks.plan_gpu_owned()` -- the narrowest gate every provider's owned-bank allocation
passes through -- refuses again before a device tensor exists.
`GPU_OWNED_EXPERT_QUANTS = {"nvfp4"}`; a plan that declares no format (hand-built ones in
tests and the shadow tooling) is not second-guessed.

Reviewed and left as written. Note `ModelConfig.expert_quant` defaults to `"none"`, so a
plain bf16 MoE checkpoint is refused too, which is the intent.

### 4. The placement report could not list owned layers

`weight_placement_report()` was emitted from `Engine._install_model_weights`, which runs
before `load_host_tables` and before `_init_offload_moe_cache` -- so half of what a
placement report is for did not exist yet. `d1fbdb3` moves it, the same way `977ec62` moved
it for the picture-weight backing:

* `_install_model_weights` only loads and adopts. `Engine._log_weight_placement_report()` is
  called from `__init__` after both, and logs **one** block: the model's own lines plus
  `_expert_placement_lines()`, a pure helper over the config and the two resolved layer
  sets:

  ```
  Token embedding: host-resident, bytes=..., device=cpu
  PLE table: backend=mmap, mapped_bytes=..., layers=1
  Picture weights: mode=layer-stream, backing=mmap, tensors=333, bytes=..., devices=cpu
  Dense weights: quant=int8
  Expert placement: backend=offload, moe_layers=48, experts=512,
    gpu_owned_layers=[0, 1, 2, 6, 7, 22], cpu_layers=[], streaming_layers=42, lru_slots=3678
  ```

* `models/qwen4_exp/model.py`: `load_host_tables` records which PLE backing it took (pinned
  / mmap / disk / dummy-zero, with the bytes) and the report emits it. The PLE table is
  47.7 GiB of this checkpoint's host residency and had no line anywhere.
* The dedicated `MoE GPU-owned layers: ...` boot line is untouched -- it is what an operator
  greps for, and it carries the resident GiB this block does not. The dense line is only
  emitted when a dense quant is configured, so a plain bf16 model with no MoE keeps the
  block exactly as silent as before.

### 5. Docs and criteria corrections

`1427088`. The live checklist asked for "-7.9 GiB of scheduler **private bytes**", which
this design cannot deliver and which read run 2 as a FAIL of a feature that worked: the host
expert banks are **mapped** pages, not private commit, so never allocating six layers' banks
moves working set and physical-in-use by the full amount while private bytes does not move
at all (measured: working set -8.02, physical in use -7.50, commit -4.37, private bytes
**+1.52** GiB).

* Design section 9 and the status doc's check table: the criterion is now scheduler working
  set (-7.9 +/- 0.5) plus whole-system commit as a partial signal, with a note on why
  private bytes is no signal at all. Run 2's row becomes the PASS it always was. The
  corrections list in the status doc gains this as correction 0.
* `docs/windows-qwen38-flash-next-mmap.md`: spells out the floor an explicit
  `-MoECacheSize` has to clear -- `512 * owned + 1024`, so `auto` needs **>= 4096** -- with
  the refusal text, and says 4096 is a floor, not a recommendation. It already stated that
  `-MoECacheSize` is the TOTAL budget with the owned layers charged to it, and how to read
  the `resident: true, miss_rate: null` routing rows.
* The launcher's `-GpuOwnedLayers` help still said "lower `-MoECacheSize` by ~512 per owned
  layer", which `19286e1` made wrong. Corrected, with the floor.
* `docs/cli.md`'s row: NVFP4 only, the slot floor, and the routing-row shape.

The research docs are deliberately left as written: they record what was measured against
the criterion as it stood, and run 2's own analysis is where the correction came from.

### 6. Operator stop script

`72f5bc6`. Live runs left `python.exe ... spawn_main` orphans -- children whose launcher had
exited -- holding the ZMQ side ports, so the next boot died with
`ZMQError: Address in use (tcp://127.0.0.1:2033)`. There was no script to clear them, and
`taskkill /im python.exe` is worse than the problem.

`scripts/stop-qwen38-flash-next-windows.ps1` (PowerShell 5.1: no `&&`, no `||`, no ternary):

* **selects** `python.exe` / `pythonw.exe` running `freetoken.cli serve` on `-Port` (or every
  port when `-Port` is 0), every python descendant to any depth (iterative, cycle-safe --
  Windows recycles pids), and any orphaned `spawn_main` python whose parent is not in the
  snapshot. Servers and orphans print under separate headings; `-SkipOrphans` opts out,
  `-DryRun` prints and stops.
* **never** touches a non-python process. `ft.exe`, the Desktop daemon on port 1900, has
  `freetoken.cli serve` in its own command line and would be caught by any
  command-line-only rule; the name allowlist is checked in the matcher *and* again at the
  kill site. This shell and its own process tree are excluded too.
* **waits** up to `-TimeoutSeconds` (120) for both `nvidia-smi` under `-VramFreeThresholdMB`
  (3072) and no listener on `-Port..-Port+9`, then reports which one is still holding.
  Exit 0 settled, 1 timed out. `nvidia-smi` is a query, not a CUDA context.

Documented in the Windows guide's *Port already occupied* section, which now names the
actual failure and the `+9` side-port range (it said `+6`).

### 7. `--moe-cpu-layers` interplay, the `< 4096` refusal, and `auto:N`

`709272c`. Both messages diagnosed without prescribing, and one `auto:N` case answered with
the wrong set instead of an error:

* the `--moe-cpu-layers` clash now says a layer cannot be both, why (an owned layer has no
  host bank for the CPU executor to read), and to drop the clashing ids from one of the two
  flags;
* the LRU-floor refusal now reads "Raise `--moe-cache-size` (launcher: `-MoECacheSize`) to
  at least 4096" -- the operator drives the launcher, and the flag it takes is not the flag
  the engine names;
* `auto:N` on a model with more MoE layers than the measured ranking has entries returned
  `ranked[:N]`, silently owning `len(ranked)` layers instead of the N asked for. It now
  refuses, naming how many the list covers. `auto:N` for N > 8 on *this* model was already
  correct (the list has all 48 entries; only the docs stop at eight) and is now pinned by a
  test rather than assumed.

## Tests

Runner (from the worktree; PowerShell **deletes** an env var assigned `''`, so `'-1'` is the
portable spelling of `CUDA_VISIBLE_DEVICES=""`):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONPATH = 'D:\FreeToken-gpu-owned-followups\scripts\windows-ple-mmap;D:\FreeToken-gpu-owned-followups\python;<scratch>\pytest-site'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine -q -p no:cacheprovider --timeout=600
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/moe -q -p no:cacheprovider --timeout=600
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/server -q -p no:cacheprovider --timeout=600
```

| Suite | Before (`0964145` + omp's three commits) | After | Failure count |
|---|---|---|---|
| `tests/engine` | 12 failed, 636 passed, 96 skipped | 12 failed, 664 passed, 96 skipped | unmoved |
| `tests/moe` | 31 failed, 255 passed, 149 skipped | 31 failed, 255 passed, 149 skipped | unmoved |
| `tests/server` | 0 failed, 557 passed | 0 failed, 557 passed | unmoved |
| `tests/models/qwen4_exp` | 1 failed, 254 passed, 167 skipped | 1 failed, 254 passed, 167 skipped | unmoved |

The brief's baseline at `0964145` itself was 12 / 31-33 / 0 for engine / moe / server, so
neither omp's three commits nor these five moved a failure. The failing **test ids** were
compared set-wise before and after, not just the counts: identical in both suites that have
any. Every pre-existing failure is environmental on this box (flashinfer absent, triton
"0 active drivers", no CUDA device) and none of them is in a file this work touched.

New and changed test files:

| File | Cases added | Covers |
|---|---|---|
| `tests/engine/test_cache_budget.py` | 5 (this session) + 6 (omp) | reserve composition, the auto override, engine-level draft-head gating, the ledger's KV row, the refusal message |
| `tests/engine/test_moe_gpu_owned_layers.py` | 6 | launcher passthrough for both knobs, the docs, both refusal texts, `auto:N` at 9/13/24/47/48, the too-large `auto:N` refusal, corrected checklist criteria |
| `tests/engine/test_vision_weight_placement.py` | 4 + 3 rewritten | the report is emitted after the cache exists, the expert placement line, the PLE line, a dense model's block |
| `tests/engine/test_stop_script.py` | 10 (new file) | the stop script's selection helpers, driven from PowerShell over a synthetic process snapshot |
| `tests/server/test_gpu_owned_geometry.py` | 5 (omp) | `gpu_owned_reserved_bytes`, streaming-only ceilings |
| `tests/moe/test_gpu_owned_banks.py` | 4 (omp) | the non-NVFP4 refusal, including through the real `load_expert_banks` dispatch |

`tests/engine/test_stop_script.py` shells out to `powershell.exe` and dot-sources the script
with `-DotSourceOnly`, which returns before it enumerates a single process. No test starts,
signals or enumerates a real process, and nothing here needs a GPU.

## Operator live checklist

**Run live on 2026-09-02** (five candidate boots on port 2030 plus the restore) — results in
[`docs/research/measurements-gpu-owned-followups-live-2026-09-02.md`](../research/measurements-gpu-owned-followups-live-2026-09-02.md),
evidence in `<scratch>\gpu-owned-live\run4\`. Run each on a **fresh** boot, one server at a
time, and settle 45-60 s between servers.

| # | Check | How | Expected | Result |
|---|---|---|---|---|
| L1 | The VRAM ledger block appears once, after the KV pool | boot with `-GpuOwnedLayers auto -MoECacheSize 6750`, grep the log for `VRAM ledger` | one block; weights / KV / GDN / owned / LRU / reserve / headroom / unaccounted / named total, KV row NOT 0.00 GiB | **PASS** (at 5781; 6750 is refused, see L4) — one block right after `Allocating 65536 tokens for KV cache`, KV row 1.55 GiB |
| L2 | The ledger's numbers are the real ones | compare the ledger against `nvidia-smi` at `state: serving` and against `/v1/cache/status` | `unaccounted` positive and small (allocator slack + activations); if it is negative the plan is over the card | **PASS** — owned/LRU/KV rows reconcile to the byte with the geometry; `unaccounted` +3.03 GiB = exactly `(1-memory_ratio) x 30.25 GiB` |
| L3 | The auto reserve is the right size on this box | boot `-MoECacheSize 0` (i.e. `--moe-cache-auto`) with speculation on, note the resolved slot count; check free VRAM at decode peak | >= ~1.5 GiB free at decode peak; no throughput cliff | **PASS** — 3,029 MiB free at decode peak with the reserve on; 1,112 MiB with it zeroed, and 9 % slower |
| L4 | The headroom refusal fires and names a size that works | boot with an `-MoECacheSize` deliberately ~1000 slots too large | refuses at boot naming the largest slot count that fits; that number then boots | **FAIL as measured, fixed in 42e3134, needs one live boot to confirm** — it refused (correctly, and already at 6750), but quoted the post-charge LRU as `--moe-cache-size` and named 2709, which the 4096 floor then refused; the size that works is 2709+3072=5781. The refusal now quotes 6750 and names 5781. Re-run: boot `-GpuOwnedLayers auto -MoECacheSize 6750`, paste the named size back into `-MoECacheSize`, expect SERVING |
| L5 | `-MoEVramReserveBytes 0 -MoECacheHeadroomBytes 0` restores the old sizing | boot with both zeroed | boots at the pre-2026-09 slot count | **PASS** — auto resolves 4452 LRU + 3072 owned = 7,524 slot-equivalents = the whole 19.43 GiB MoE budget; free after init 3.14 GiB |
| L6 | `/v1/cache/status` shows the reservation | `curl /v1/cache/status` with and without `-GpuOwnedLayers auto` | `gpu_owned_reserved_bytes` ~8.5e9 with, 0 without; `limits.moe_experts.max` lower with | **PASS** — 8,517,058,560 with (field absent on the pre-feature baseline); `moe_experts.max` 5427 vs 8499 |
| L7 | The placement report is one block and names the owned layers | grep the boot log | one block containing `Expert placement: ... gpu_owned_layers=[0, 1, 2, 6, 7, 22] ... streaming_layers=42`, plus the PLE line with `backend=mmap`; the dedicated `MoE GPU-owned layers:` line still present | **PASS** — all five lines in one record, `PLE table: backend=mmap, mapped_bytes=51200245760`; dedicated line still one line above |
| L8 | The non-NVFP4 refusal | not testable here without another checkpoint | a non-NVFP4 MoE checkpoint + `--moe-gpu-owned-layers` refuses before the expert load | **NOT TESTABLE** — `D:\Models` holds only the NVFP4 checkpoint |
| L9 | The stop script against real processes | boot, then `stop-qwen38-flash-next-windows.ps1 -Port 2020 -DryRun`, read the two headings, then run it for real | the server tree listed under "server processes", orphans under their own heading, `ft.exe` (port 1900) absent from both; after the kill it reports settled and the Desktop daemon is still alive | **PASS** — exactly the four serve pythons listed, no `ft.exe`, no unrelated python; settled at 1618 MB, daemon still listening on 1900 |
| L10 | The stop script clears the ZMQ-port orphan | reproduce the orphan (kill the launcher from the console, leave the children), then run the script, then boot | no `ZMQError: Address in use`; the boot proceeds | **PASS** — a real orphan occurred twice on its own (a refused boot leaves one holding 2033); listed under the orphan heading, killed, next boot clean |
| L11 | Speed at the corrected budget | `-GpuOwnedLayers auto -MoECacheSize 6750` vs no owned layers at 6750 | the run-3 fix's ~63 tok/s vs the 70.4 baseline, i.e. the ~-10 % trade, not run 2's -50.9 % | **PASS with a caveat** — 6750 is refused; at the largest allowed size (5781) the candidate runs 60.0 tok/s against the same-session no-owned baseline's 58.5, i.e. parity. Forcing 6750 with the knobs zeroed gives 55.5 |
| L12 | The corrected RAM criterion | scheduler **working set** and whole-system physical in use, not private bytes | working set -7.9 +/- 0.5 GiB | **INCONCLUSIVE** — measured -4.98 GiB working set / -5.49 GiB physical in use. The candidate's absolutes match run 3's post-fix boot; today's baseline read 3.6 GiB lower than run 3's because it had been trimmed after 75 minutes idle |

Hazards, unchanged from the feature's status doc: only ever kill `python.exe` from
`nvidia-smi --query-compute-apps` (or use L9's script); settle 45-60 s between servers or
the last expert bank dies in `cudaHostRegister failed ... out of memory`; one session booting
servers at a time; send test requests with `chat_template_kwargs.enable_thinking=false`.

## Open risks

1. **The reserve is a documented constant, not a measurement.** 2.25 GiB for the draft head
   and 0.75 GiB for the graph pools + vision workspace were derived from one box's boot log
   (2.17 GiB measured draft head). The draft head's bytes *are* computable from the manifest
   before the load -- the brief suggested it -- but doing so means building the MTP config
   and weight plan before the MoE cache exists, which is a larger change than this follow-up
   took on. A box whose draft head is materially bigger will still oversubscribe. L2/L3 are
   the checks that would catch it.
2. **The auto reserve gates on `spec_decode.enabled`, not on what is actually allocated.**
   A boot that enables speculation and then fails to build the head reserves 2.25 GiB it
   does not use (harmless), and the vision workspace is charged even with vision off
   (0.75 GiB, also charged when the graph pools are the only user). Both err toward
   reserving.
3. **The explicit-size refusal can block a boot that used to work.** That is the operator's
   stated preference ("fail loudly, never silently shrink"), but the first boot after this
   change with a hand-tuned `-MoECacheSize` may refuse. The message names the size that
   fits -- as a TOTAL, in the unit `-MoECacheSize` takes, since 42e3134 -- and
   `-MoEVramReserveBytes 0 -MoECacheHeadroomBytes 0` restores the old behaviour (L4, L5).
   The check still runs after the ~40 s expert-bank read, because the per-slot byte count it
   needs is measured off the loaded banks; the LRU-floor refusal fires immediately.
4. **The orphan rule is blind to which tool spawned the child.** A `spawn_main` python
   command line says nothing about its parent's module, so a stale multiprocessing child of
   *any* python tool on the box matches. That is deliberate -- an orphan of the last server
   is exactly what holds the ZMQ port -- but it means `-DryRun` first, and `-SkipOrphans`
   exists. Verified on the live box during this work: with a server running, `-DryRun`
   listed the server's own tree and no orphans.
5. **The stop script's waits are untested against a real teardown.** The selection logic is
   pinned by tests; the `nvidia-smi` threshold, the listener polling and the timeout path
   have only been exercised with `-DryRun` (L9, L10).
6. **The placement report now runs after `load_host_tables`.** `977ec62` deliberately kept
   it before, because `load_host_tables` re-homes `model.embed_tokens` into pinned storage
   and that changes the "Token embedding" line. The line is now *more* truthful (it
   describes where the table ended up), but it is a behaviour change in an existing log
   line: L7 should confirm it still reads sensibly on the real checkpoint.
7. **`gpu_owned_reserved_bytes` is measured from `resident_banks`, so it is 0 on any path
   that does not populate them.** The geometry falls back to the `num_experts x per-expert`
   product only when the engine ack omits the field entirely, not when the ack reports 0.
8. **Nothing here was run on the GPU at all.** Every number in this document is either from
   the CPU suite or quoted from the earlier live runs.
