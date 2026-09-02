# Public Windows PLE mmap Fork Implementation Plan

**Goal:** Publish a reproducible `dejay2/FreeToken` fork and `windows-ple-mmap` branch showing Qwen3.8-Flash-Next NVFP4 with SSD-backed PLE on native Windows, then post one measured validation comment on FreeToken PR #279 without opening a pull request.

**Design source:** `docs/design/public-windows-ple-mmap-fork.md`

**Context sources:** `CONTEXT.md`, `CONTEXT-MAP.md`

**ADR sources:** `docs/adr/0001-desktop-assisted-native-windows-fork.md`

**Architecture:** Keep PR #279's original commit unchanged. Add a public, parameter-based PowerShell launcher that runs the fork's source with the Windows Python/DLL/kernel files supplied by an existing FreeToken Desktop installation. A generalized `sitecustomize.py` applies the proven Windows bridge at Python startup. Public smoke and benchmark programs cross the OpenAI-compatible HTTP API. Documentation and raw results describe this as an unofficial Desktop-assisted command-line setup, not a standalone Windows package.

**Global constraints:**
- Preserve commit `feaeaa31c0cea385a1c9ee107d4b1053f83b35db` and its author, Ruslan Suleimanov (`ryseek`).
- Credit relevant Windows approaches from closed PR #232 by Max K (`MaxKerkula`) and its contributors.
- Do not modify `C:\Users\jay\AppData\Local\FreeToken` or downloaded model files.
- Publish no pi files or pi instructions.
- Publish no personal absolute paths, secrets, local logs, generated cache/build files, or internal planning documents.
- Bind the server to `127.0.0.1` only.
- Default to one active request and 262,144 usable context tokens.
- State that FreeToken Desktop must be installed but does not need to be open.
- Create no upstream pull request and no new upstream issue.
- Post only one upstream comment, on PR #279, after the exact comment text and measured facts pass final review.

**Out of scope:**
- A standalone Windows wheel or command-only installer.
- Integration with or control from the FreeToken Desktop UI.
- Image input, more than one active request, multi-GPU support, or changing PR #279's PLE design.
- Upstream-quality replacement of every Windows shim behavior in FreeToken product source.
- Performance comparisons against Linux, other engines, or FreeToken `main`.

## Codebase map

### Public files

- Create: `scripts/start-qwen38-flash-next-mmap-windows.ps1` — parameter-based launcher; discovers the fork root and standard Desktop paths, validates prerequisites, and starts the local-only server.
- Create: `scripts/windows-ple-mmap/sitecustomize.py` — generalized Windows bridge derived from the proven local shim, with no personal paths.
- Create: `benchmarks/run_qwen38_mmap_smoke.py` — generalized public API smoke test migrated from `windows-shim/smoke_test.py`.
- Create: `benchmarks/bench_qwen38_mmap_windows.py` — deterministic streaming benchmark for fixed input/output sizes with raw timing, usage, cache geometry, memory, and disk-counter samples.
- Create after measurement: `benchmarks/qwen38-flash-next-rtx5090-windows-mmap.json` — raw machine facts, commands, server settings, and benchmark samples for the 50K pool (50,048 total/49,984 usable) and full pool (262,208 total/262,144 usable).
- Create after measurement: `docs/windows-qwen38-flash-next-mmap.md` — FreeToken-only guide, measured table, limits, troubleshooting, and attribution.
- Modify: `docs/models.md` — add one link from the Qwen3.8-Flash-Next note to the Windows guide, clearly labeling it Desktop-assisted and unofficial.

### Local-only files used during preparation

- `serve-pr279.cmd`, `serve-pr279-windows-safe.cmd`, and `windows-shim/` — existing local proof; read/migrate but never stage.
- `CONTEXT.md`, `CONTEXT-MAP.md`, `docs/design/`, `docs/plans/`, and `docs/adr/` — internal preparation record; never stage.
- `.local/pr279-comment.md` — exact comment draft generated from measured JSON; never stage.
- Background logs under `C:\Users\jay\.pi\agent\extensions\bg\logs` — startup/performance evidence; never stage.
- `C:\Users\jay\.pi\agent\models.json` and `settings.json` — local pi setup; inspect only for continued local use and never stage.

### Existing interfaces and evidence

- `python/freetoken/scheduler/config.py` and `python/freetoken/server/args.py` — Unix-only worker addresses patched by the shim.
- `python/freetoken/server/api_server.py` — Uvicorn startup whose Windows loop is patched by the shim.
- `python/freetoken/models/loader.py`, `python/freetoken/models/weight.py`, and `python/freetoken/moe/host_banks.py` — optional POSIX cache hints absent on Windows.
- `python/freetoken/kernel/utils.py` and installed Desktop `freetoken/kernel/csrc` — kernel discovery/build behavior adapted by the shim.
- `/v1/models`, `/v1/cache/status`, and `/v1/chat/completions` — public test/measurement seams.
- Windows counter `\PhysicalDisk(1 D:)\Disk Read Bytes/sec` — established D-drive read-rate source.
- `CONTRIBUTING.md` — requires exact hardware, software, model, commands, and genuinely run measurements.
- Closed PR #232 — prior art for loopback TCP, selector loop, installed-kernel reuse, Windows C++ flags, and a parameter-based launcher.

## Testing strategy

Tests cross the same public boundaries a user relies on:

1. PowerShell parses the launcher without executing it.
2. Python compiles the shim, smoke test, and benchmark helper.
3. A static scan rejects `C:\Users\jay`, `D:\`, credentials, and local pi paths from every public file.
4. The launcher starts the real PR #279 model with the Desktop UI closed and files untouched.
5. The smoke test validates model metadata, sequential text, streamed reasoning/content/usage, parsed tools, and final server health.
6. The benchmark uses deterministic prompts and the streaming API, not server-private timing helpers.
7. Raw JSON is independently checked for expected scenario/repeat counts, actual completion-token counts, finite positive timings, both context allocations, and the full tested machine facts.
8. The guide's numeric table is generated/read from that JSON rather than typed from memory.
9. Git staging uses exact paths; a staged-file allow-list and path/secret scan prevent internal files from entering commits.
10. GitHub checks verify the fork branch exists, original commit attribution remains, and no pull request was opened.

### Focused syntax/static commands

```powershell
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
  'D:\FreeToken-ple-mmap\scripts\start-qwen38-flash-next-mmap-windows.ps1',
  [ref]$null,
  [ref]$errors
) | Out-Null
if ($errors.Count) { $errors | Format-List; exit 1 }
```

```bat
"C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe" -m py_compile ^
  D:\FreeToken-ple-mmap\scripts\windows-ple-mmap\sitecustomize.py ^
  D:\FreeToken-ple-mmap\benchmarks\run_qwen38_mmap_smoke.py ^
  D:\FreeToken-ple-mmap\benchmarks\bench_qwen38_mmap_windows.py
```

### Full live smoke command

```powershell
& 'D:\FreeToken-ple-mmap\scripts\start-qwen38-flash-next-mmap-windows.ps1' `
  -ModelPath 'D:\Models\Qwen3.8-Flash-Next-NVFP4' `
  -Port 2020 `
  -ContextTokens 262144
```

In a second window:

```bat
"C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe" ^
  D:\FreeToken-ple-mmap\benchmarks\run_qwen38_mmap_smoke.py ^
  --base-url http://127.0.0.1:2020/v1 ^
  --model Qwen3.8-Flash-Next-NVFP4
```

Expected completion evidence: every smoke section reports PASS, `num_pages * page_size - page_size == 262144`, and `/v1/models` still answers afterward.

### Benchmark commands

For each fresh 50K and 262,144-usable setting, launch with the public script and run:

```bat
"C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe" ^
  D:\FreeToken-ple-mmap\benchmarks\bench_qwen38_mmap_windows.py ^
  --base-url http://127.0.0.1:2020/v1 ^
  --model Qwen3.8-Flash-Next-NVFP4 ^
  --tokenizer D:\Models\Qwen3.8-Flash-Next-NVFP4 ^
  --context-label 49984-or-262144 ^
  --seed 3805090 ^
  --output D:\FreeToken-ple-mmap\results\benchmark-50k-or-262144.json
```

The helper itself defines one unrecorded warm-up, three 512-in/512-out runs, three 8,192-in/512-out runs, and one 32,768-in/128-out run. It sends `reasoning_effort=off`, `temperature=0`, `top_k=1`, and `ignore_eos=true` so output length and raw engine work are controlled. It records server usage counts rather than assuming character counts.

The full relevant suite is the live smoke plus both measured configurations. The repository-wide pytest suite is not required because no tracked FreeToken product module or test is changed; this fork adds scripts, a shim, benchmark artifacts, and documentation. No package is installed into the untouched Desktop environment.

## Task breakdown

### Task 1: Create a clean public Windows launch path

**Outcome:** Another Windows user with FreeToken Desktop can start this fork without Jay's paths or edits to Desktop files.

**Blocked by:** None.

**Files:**
- Create: `scripts/start-qwen38-flash-next-mmap-windows.ps1` — validation and launch flow.
- Create: `scripts/windows-ple-mmap/sitecustomize.py` — generalized bridge.
- Create: `benchmarks/run_qwen38_mmap_smoke.py` — public live validation.
- Test: the three new files plus the live server API.

**Interfaces:**
- Consumes: required `-ModelPath`; optional `-Port`, `-ContextTokens`, `-MaxRunningRequests`, and `-DesktopPython`; standard `%LOCALAPPDATA%\FreeToken` installation; `CUDA_PATH` or PyTorch CUDA discovery.
- Produces: local OpenAI-compatible service at `http://127.0.0.1:<Port>/v1`.

**Acceptance criteria:**
- [ ] Launcher finds repo paths relative to `$PSScriptRoot` and accepts a custom Desktop Python path.
- [ ] Missing model, Python, kernel directory, or CUDA toolkit produces a specific failure before workers start.
- [ ] Shim contains no fixed user, drive, model, checkout, or CUDA-version path.
- [ ] Desktop files remain unchanged.
- [ ] Full-context smoke passes and server stays alive.

- [ ] **Step 1: Write the failing test**
  - Test: static path scan and public smoke command.
  - Expected behavior: clean public files exist, contain no personal paths, and launch/smoke pass; before implementation the files do not exist and the check fails.

- [ ] **Step 2: Run the focused test**
  - Run: test that the three public paths exist, then run the PowerShell parser and Python compile commands.
  - Expected: missing-file failure establishes the absent public launch path.

- [ ] **Step 3: Implement the smallest change**
  - Migrate the local launch arguments into typed PowerShell parameters.
  - Migrate only the proven shim behavior; derive installed/CUDA locations dynamically.
  - Migrate the existing smoke test and replace local wording with public, FreeToken-only wording.
  - Add source comments crediting PR #232 where its approach is adapted.

- [ ] **Step 4: Run the focused test again**
  - Run: PowerShell parse, Python compile, `--help` for both Python programs, launcher prerequisite checks with one intentionally missing model path, and the static path scan.
  - Expected: parse/compile/help pass; missing model fails before launch with its named path; path scan has no match.

- [ ] **Step 5: Run relevant checks**
  - Stop the current local server cleanly, confirm ports 2020–2026 are free, launch full context with the public script as a supervised task, and run the public smoke test.
  - Expected: 262,144 usable context, parsed tool call, live usage, final health PASS, and no scheduler/worker exit.

### Task 2: Measure the 50K and 256K memory choices reproducibly

**Outcome:** The fork contains raw, repeatable evidence for speed, context, memory, expert cache, and SSD activity on Jay's exact hardware, with total versus usable KV capacity stated separately.

**Blocked by:** Task 1.

**Files:**
- Create: `benchmarks/bench_qwen38_mmap_windows.py` — deterministic measurement helper.
- Create: `benchmarks/qwen38-flash-next-rtx5090-windows-mmap.json` — combined raw results.
- Use locally: `results/benchmark-50k.json`, `results/benchmark-262144.json`, two server logs, and disk samples; ignored/not staged.

**Interfaces:**
- Consumes: streaming chat completions, usage events, `/v1/cache/status`, NVIDIA `nvidia-smi`, Windows memory facts, and `\PhysicalDisk(1 D:)\Disk Read Bytes/sec`.
- Produces: one public JSON artifact with commands, metadata, raw runs, and derived summaries.

**Acceptance criteria:**
- [ ] Both context labels are present with usable tokens 49,984 and 262,144, and total pool tokens 50,048 and 262,208 are also recorded.
- [ ] The speed-focused and full-context expert-cache counts come from live status, not estimates.
- [ ] Every repeated run has actual prompt/completion counts and positive TTFT/TPOT/end-to-end values.
- [ ] Warm-up is labeled and excluded from averages.
- [ ] System table exactly records Windows 11 Pro build 26200, Ryzen 9 9950X3D, 95.6 GiB RAM, RTX 5090 32,607 MiB, driver 610.62, CUDA 13.1.115, and Samsung 990 PRO 4 TB.
- [ ] Full-context server is restored after testing.

- [ ] **Step 1: Write the failing test**
  - Test: benchmark helper `--dry-run` scenario manifest and result-schema validator.
  - Expected behavior: one warm-up, 3×512/512, 3×8192/512, and 1×32768/128 with seed 3805090; before implementation the helper is absent.

- [ ] **Step 2: Run the focused test**
  - Run: Python compile and `--dry-run` before the helper exists.
  - Expected: missing-file failure.

- [ ] **Step 3: Implement the smallest change**
  - Build deterministic tokenizer-exact prompts, streaming timing, usage capture, cache/memory/disk snapshots, JSON output, and a mode that combines/validates the two local result files.
  - Preserve raw samples; derive summaries with documented formulas.

- [ ] **Step 4: Run the focused test again**
  - Run: compile, `--help`, `--dry-run`, and schema self-check against a tiny synthetic fixture created under ignored `results/`.
  - Expected: exact scenario manifest and validator PASS.

- [ ] **Step 5: Run relevant checks**
  - Measure the fresh 50K mode, stop it cleanly, measure fresh 262,144-usable mode, then leave the full server running.
  - Expected: all seven recorded requests per mode succeed; raw files combine; public JSON validation passes; no server crash.

### Task 3: Write the public guide from measured evidence

**Outcome:** A Windows user can understand requirements, reproduce the setup, choose 50K or 256K, and interpret the measured trade-off without mistaking the fork for an official standalone package.

**Blocked by:** Task 2.

**Files:**
- Create: `docs/windows-qwen38-flash-next-mmap.md` — public guide/results.
- Modify: `docs/models.md` — one guide link.
- Test: public JSON, every command/path in the guide, link targets, and public path/secret scan.

**Interfaces:**
- Consumes: public launcher CLI, smoke/benchmark CLI, combined JSON, PR #279, PR #232, issue #130, and official install/release facts.
- Produces: human-facing guide and model-page link.

**Acceptance criteria:**
- [ ] Guide says Desktop is required but need not be open; setup is not Desktop-configurable and not standalone.
- [ ] Hardware/software and benchmark tables match JSON exactly.
- [ ] Commands use placeholders/parameters, never Jay's paths.
- [ ] Limits include text-only, one active request, PR #279 unmerged base, and context/expert-cache trade-off.
- [ ] Attribution links PR #279 and closed PR #232.

- [ ] **Step 1: Write the failing test**
  - Test: guide existence, required-section headings, JSON-number agreement, local-link existence, and forbidden-path scan.
  - Expected behavior: all checks pass after implementation; guide is absent before implementation.

- [ ] **Step 2: Run the focused test**
  - Run: existence/required-heading check.
  - Expected: missing guide failure.

- [ ] **Step 3: Implement the smallest change**
  - Write only FreeToken setup, results, limitations, troubleshooting, and attribution.
  - Add one concise link in `docs/models.md`.

- [ ] **Step 4: Run the focused test again**
  - Run: heading/link/path scan and a Python comparison of guide table values to JSON summaries.
  - Expected: all checks PASS and no forbidden paths/secrets appear.

- [ ] **Step 5: Run relevant checks**
  - Follow the guide's full-context launch and smoke commands once from a fresh shell.
  - Expected: commands work as written and final health remains PASS.

### Task 4: Curate and publish the fork without a pull request

**Outcome:** `https://github.com/dejay2/FreeToken/tree/windows-ple-mmap` contains only the intended public work and original PR #279 history.

**Blocked by:** Task 3.

**Files:**
- Stage only the seven public paths listed in the codebase map.
- Never stage local-only paths listed above.

**Interfaces:**
- Consumes: local branch at PR #279 commit, GitHub account `dejay2`, and exact staged allow-list.
- Produces: public fork and branch.

**Acceptance criteria:**
- [ ] Branch name is `windows-ple-mmap`.
- [ ] Staged files match the allow-list exactly.
- [ ] Two commits follow FreeToken's naming rule.
- [ ] PR #279 commit remains present with `ryseek` as author.
- [ ] Fork branch is public and no pull request exists.

- [ ] **Step 1: Write the failing test**
  - Test: `gh repo view dejay2/FreeToken` and remote branch lookup.
  - Expected behavior after publication: fork and branch exist; currently the repository lookup fails because the fork does not exist.

- [ ] **Step 2: Run the focused test**
  - Run: current `gh repo view` and `git ls-remote` checks.
  - Expected: fork/branch absent.

- [ ] **Step 3: Implement the smallest change**
  - Create/switch local `windows-ple-mmap` branch.
  - Stage exact launcher/shim/smoke/benchmark files and commit `feat(windows): add Desktop-assisted PLE mmap launcher`.
  - Stage exact result/guide/model-page files and commit `docs(windows): document Qwen3.8 mmap setup and results`.
  - Create `dejay2/FreeToken`, add remote named `fork`, and push only `windows-ple-mmap`.

- [ ] **Step 4: Run the focused test again**
  - Run: GitHub fork/branch lookup, remote commit log, tree listing, author check for `feaeaa3`, and staged/public forbidden-path scans.
  - Expected: public branch contains only intended files plus upstream history; attribution is preserved.

- [ ] **Step 5: Run relevant checks**
  - Run: `gh pr list` against both fork and upstream for head `dejay2:windows-ple-mmap`.
  - Expected: zero pull requests.

### Task 5: Post one factual PR #279 validation comment

**Outcome:** PR #279 readers see what SSD-backed PLE makes possible on this Windows machine, with reproducible measurements and honest delivery limitations.

**Blocked by:** Task 4.

**Files:**
- Create locally only: `.local/pr279-comment.md` — exact comment body derived from public JSON and fork URL.
- Publish: one comment on `https://github.com/FlashML-org/FreeToken/pull/279`.

**Interfaces:**
- Consumes: measured JSON, public guide URL, fork branch URL, tested commit, and known limitations.
- Produces: one PR comment; no PR or issue.

**Acceptance criteria:**
- [ ] Comment identifies native Windows, exact hardware/software/model, 50K and 256K trade-off, speed/memory results, and smoke/tool success.
- [ ] Comment states Desktop supplied the Windows engine, Desktop cannot configure this server, and this is not a standalone Windows CLI package.
- [ ] Comment credits PR #279 and links the fork/guide.
- [ ] Every number appears in public JSON; no personal path or unsupported claim appears.
- [ ] Only one comment is posted by `dejay2`.

- [ ] **Step 1: Write the failing test**
  - Test: local draft existence and fact/path scan.
  - Expected: draft absent before implementation.

- [ ] **Step 2: Run the focused test**
  - Run: draft existence check.
  - Expected: missing-file failure.

- [ ] **Step 3: Implement the smallest change**
  - Generate the draft from measured facts and public links.
  - Present the exact draft and final benchmark summary to Jay before the one irreversible comment action.

- [ ] **Step 4: Run the focused test again**
  - Run: compare all numeric strings in the draft to the public JSON and scan for forbidden paths/secrets.
  - Expected: exact factual match and clean scan.

- [ ] **Step 5: Run relevant checks**
  - After Jay's final approval, post with `gh pr comment 279 --repo FlashML-org/FreeToken --body-file .local/pr279-comment.md`.
  - Expected: comment URL is returned; comment body matches the reviewed draft; no PR/new issue exists.

## Checkpoint commits

1. `feat(windows): add Desktop-assisted PLE mmap launcher`
   - `scripts/start-qwen38-flash-next-mmap-windows.ps1`
   - `scripts/windows-ple-mmap/sitecustomize.py`
   - `benchmarks/run_qwen38_mmap_smoke.py`
   - `benchmarks/bench_qwen38_mmap_windows.py`
2. `docs(windows): document Qwen3.8 mmap setup and results`
   - `benchmarks/qwen38-flash-next-rtx5090-windows-mmap.json`
   - `docs/windows-qwen38-flash-next-mmap.md`
   - `docs/models.md`

## Self-review

- **Coverage:** Fork, guide, balanced benchmark, attribution, limitations, no PR, and one reviewed PR comment each map to a task.
- **Evidence:** Hardware/software facts came from live system commands; current context geometry came from `/v1/cache/status`; package availability came from official docs/releases/PyPI; prior art came from PR #232; contribution rules came from `CONTRIBUTING.md`.
- **Interfaces:** Task 1's launcher/API feed Task 2; Task 2's JSON feeds Task 3 and Task 5; Task 3's public files feed Task 4; Task 4's URLs feed Task 5.
- **Dependencies:** Tasks are a strict acyclic sequence. The full server is restored after benchmark restarts.
- **Seams:** Runtime checks use public HTTP APIs and Windows public counters. Publication checks use Git/GitHub public state.
- **Scope:** No product source rewrite, standalone wheel, pi content, Desktop edit, upstream PR, or new issue is included.
- **Attribution:** Original PR commit authorship is preserved; adapted PR #232 ideas are credited.
- **Placeholders:** User-facing examples use named parameters rather than unresolved implementation markers. Every acceptance result has an exact source/check.
- **Fresh-context check:** Each task names exact files, commands, expected outcomes, blockers, and handoff artifacts.
