# Public Windows PLE mmap fork design

## Status

Implemented and published on 2026-08-30. Public branch `dejay2/FreeToken:windows-ple-mmap`; no pull request opened; one verified results comment posted on PR #279. Jay later authorized publishing the separately accepted still-picture series, advancing this same public branch from text-only `14ee7b0` to `c5876d5` without publishing private evidence.

## Goal

Publish `dejay2/FreeToken` with a `windows-ple-mmap` branch that preserves FreeToken PR #279's SSD-backed PLE work and adds a reproducible, FreeToken-only native-Windows setup for users who already have FreeToken Desktop installed. Open no upstream pull request. After measurement and review, post one factual validation comment on PR #279 linking the fork and explaining the setup's limits.

## Audience

Windows users with an NVIDIA CUDA 13 GPU, enough system RAM for the model, an SSD holding `RadixArk/Qwen3.8-Flash-Next-NVFP4`, and FreeToken Desktop v0.1.2 or a compatible later version.

## Public contents

1. A parameter-based PowerShell launcher under `scripts/`.
2. A Windows compatibility shim under `scripts/` with no absolute personal paths.
3. A public API smoke test for model information, sequential text, streamed reasoning/content/usage, tools, and final server health.
4. A FreeToken-only Windows guide under `docs/`.
5. A repeatable benchmark helper plus raw and summarized RTX 5090 results under `benchmarks/`.
6. Attribution to PR #279 author Ruslan Suleimanov (`ryseek`) and relevant prior Windows work in closed PR #232 by Max K (`MaxKerkula`) and its contributors.
7. A locally reviewed PR #279 comment reporting the tested SSD-backed PLE result, exact hardware, measured speed/memory, and limitations.

## Excluded from publication

- `CONTEXT.md`, `CONTEXT-MAP.md`, `docs/design/`, `docs/plans/`, and `docs/adr/` internal preparation files.
- pi model/settings files or pi instructions.
- `D:\...` and `C:\Users\jay\...` paths.
- API keys, account data, local logs, downloaded model files, generated caches, or build outputs.
- Claims that this is an official FreeToken release or a standalone Windows package.
- Any pull request, new upstream issue, or upstream comment other than the one approved PR #279 validation comment.

## Launcher behavior

The PowerShell launcher will accept:

- required model path;
- optional server port (default 2020);
- optional context tokens (default 262,144);
- optional maximum active requests (default 1);
- optional Desktop Python path when the standard installation path is not used.

It will:

- find the fork root relative to itself;
- check the model, Desktop Python, installed kernel directory, and CUDA toolkit;
- prepend the generalized shim and fork source to `PYTHONPATH`;
- start `freetoken.cli serve` with `--ple-backend mmap`, `--moe-backend offload`, automatic expert-cache sizing, serial expert loading, and the chosen context reservation;
- bind to `127.0.0.1` only;
- leave FreeToken Desktop files untouched.

## Shim behavior

The shim will keep only behavior proven necessary on this machine:

- loopback TCP worker addresses because Windows ZeroMQ lacks Unix `ipc://` support;
- Windows Selector event-loop use for PyZMQ;
- discovery of installed compiled kernel modules and matching installed CUDA source/header files;
- no-op handling for optional POSIX page-cache advice absent on Windows;
- Windows-safe C++20/NVCC generated flags;
- explicit CUDA runtime library linking when the toolkit provides it.

The shim will derive locations from `sys.prefix`, `CUDA_PATH`, or PyTorch's discovered CUDA home. It will contain no model path, user name, drive letter, or fixed FreeToken checkout path.

## Balanced benchmark

Run both memory settings with the same model and server options:

- speed-focused: 50,048 total KV tokens, including FreeToken's reserved 64-token dummy page, leaving 49,984 usable tokens;
- full-context: 262,208 total KV tokens, including the reserved page, leaving 262,144 usable tokens.

For each setting:

1. Start from a fresh server and record startup time.
2. Record live context allocation, GPU expert-cache size, system RAM, GPU memory, and the D-drive physical disk identity.
3. Warm up once; do not include warm-up in averages.
4. Run three deterministic requests with 512 input tokens and 512 output tokens.
5. Run three deterministic requests with 8,192 input tokens and 512 output tokens.
6. Run one 32,768-input/128-output context check.
7. Disable reasoning for raw engine speed, use temperature 0/top-k 1, force the requested output length, and record the random seed.
8. Report time to first token, time per output token, output tokens/second, end-to-end time, prompt-processing speed where the server reports it, peak memory, and sampled SSD read rate.
9. Restore the full 262,144-token server after measurement.

Raw timings and commands will be saved. The guide will distinguish controlled benchmark numbers from ordinary manual-use observations.

## Public documentation structure

- What this fork is and is not.
- Tested hardware/software table.
- Required disk, RAM, GPU, Desktop installation, CUDA toolkit, and model files.
- Exact launch examples for 50K and 256K.
- API health and text request examples.
- Memory/speed results with methodology.
- Known limits: text-only server path, one active request by default, unmerged PR #279 base, Desktop-provided Windows engine requirement, and slower behavior when fewer experts fit in GPU memory.
- Troubleshooting for occupied ports, missing CUDA toolkit, missing Desktop runtime, and first-use kernel building.
- Attribution and upstream links.

## Validation and release gate

Before publication:

- all public scripts parse and contain no personal absolute path;
- smoke tests pass on the full-context server;
- benchmark raw data matches the written table;
- git diff contains only intended public files plus PR #279 history;
- secret/path scans pass;
- the original PR #279 commit remains authored by `ryseek`;
- the fork is created under `dejay2`, branch `windows-ple-mmap` is pushed, and no PR exists;
- the single PR #279 comment links the fork, reports only measured facts, says the setup cannot be configured inside Desktop, and says it is not a standalone Windows command-line package.

## Rollback

The local branch remains available if publication fails. The public branch can be deleted without affecting the upstream project, the local model, pi settings, or FreeToken Desktop.
