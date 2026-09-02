# Windows PLE mmap pi Test Implementation Plan

**Goal:** Keep the SSD-backed Qwen server stable across repeated requests and make it a selectable, fully usable local model in pi without changing the existing default model.

**Design source:** `docs/design/windows-native-ple-mmap-pi-test.md`

**Context sources:** `CONTEXT.md`, `CONTEXT-MAP.md`

**ADR sources:** None apply.

**Architecture:** A checkout-local Python startup shim adapts PR #279 to Windows while reusing the untouched FreeToken Desktop binaries. The server exposes its existing OpenAI-shaped API on `127.0.0.1:2020`. Pi uses that public API through a new `freetoken-local` model entry. A small smoke-test program exercises the same API sequence a real pi session needs.

**Global constraints:**
- Keep `C:\Users\jay\AppData\Local\FreeToken` untouched.
- Keep the model at `D:\Models\Qwen3.8-Flash-Next-NVFP4` and the PR checkout at `D:\FreeToken-ple-mmap`.
- Bind the server only to `127.0.0.1:2020`.
- Preserve pi's current default provider, model, and thinking level.
- Make no commit, push, fork, or public documentation change in this plan.
- Advertise text input only because FreeToken's current request renderer rejects non-text content parts.
- Advertise a 50,048-token pi context limit because the successful launch allocated 50,048 KV tokens, even though the checkpoint's own maximum is 262,144.
- Offer only the reasoning levels the server reports: `low`, `medium`, and `xhigh`.

**Out of scope:**
- Editing the installed Desktop version.
- Picture input through pi.
- Increasing the live context allowance beyond the measured 50,048 tokens.
- Performance benchmarking, final packaging, GitHub publication, or Desktop UI integration.

## Codebase map

- `windows-shim/sitecustomize.py` — existing checkout-local Windows bridge. Its `_generate_ninja_windows` wrapper currently fixes CUDA host flags but leaves ordinary C++ helper builds at C++17.
- `windows-shim/smoke_test.py` — new repeatable public-API check. It will validate model information, sequential text generation, streamed reasoning/content/usage, a parsed tool call, and server health afterward using Python's built-in HTTP tools.
- `serve-pr279.cmd` — existing normal launcher. It supplies PR #279 source and the Windows shim through `PYTHONPATH`, enables `--ple-backend mmap`, and accepts `--port 2020` through `%*`.
- `serve-pr279-windows-safe.cmd` — existing lower-video-memory fallback. It is not changed; it remains available if the normal launcher runs out of memory.
- `python/freetoken/kernel/radix.py` — existing cache comparison interface. `fast_compare_key` triggers the delayed helper build that failed in `bg_142`; it is tested but not modified.
- `C:\Users\jay\AppData\Local\FreeToken\venv\Lib\site-packages\freetoken\kernel\csrc\src\radix.cpp` — reused installed Windows source. It requires C++20 and is read/tested but not modified.
- `C:\Users\jay\.pi\agent\models.json` — existing pi model list. Add one `freetoken-local` provider and one model while preserving `llama-local` exactly.
- `C:\Users\jay\.pi\agent\settings.json` — existing pi choices. Add `freetoken-local/Qwen3.8-Flash-Next-NVFP4` to `enabledModels`; leave `defaultProvider`, `defaultModel`, and `defaultThinkingLevel` unchanged.
- `C:\Users\jay\.pi\agent\extensions\local-llama-sampling.ts` — inspected only. It remains limited to `llama-local`; FreeToken already reads the model's recommended `temperature=1.0`, `top_k=20`, and `top_p=0.95` from `generation_config.json`.
- `C:\Users\jay\.pi\agent\extensions\bg\logs\bg_142.out` — existing failure evidence. It shows `/std:c++17`, ignored `-std=c++20`, the radix build failure, and the scheduler exit.
- `tests/README.md` — existing testing policy. It says command-line surfaces are checked by actually running them, so the live smoke test is the highest useful seam for this checkout-local bridge.

## Testing strategy

The main test crosses the public server boundary at `/v1/models` and `/v1/chat/completions`, exactly as pi will. It covers:

- the correct model identity and 262,144-token checkpoint maximum;
- two or more sequential generations, which forces prefix-cache comparison and catches the delayed Windows build failure;
- live reasoning and answer pieces plus the final usage count;
- a `get_weather` tool call with `city` set to `Paris`;
- a final model-list request proving the scheduler stayed alive.

The focused helper check crosses the existing `freetoken.kernel.radix.fast_compare_key` interface and compares `[1, 2]` with `[1, 3]`; the independently known first mismatch position is `1`.

The pi check crosses pi's normal model loading and OpenAI-shaped request path. It first sends a no-tool text prompt, then permits only pi's harmless `read` tool and asks the model to read `CONTEXT.md`. Success proves that pi can send a tool definition, receive a tool call, send the result back, and receive the final answer.

The focused helper command is:

```bat
set "PYTHONPATH=D:\FreeToken-ple-mmap\windows-shim;D:\FreeToken-ple-mmap\python"
"C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe" -c "import torch; from freetoken.kernel.radix import fast_compare_key; assert fast_compare_key(torch.tensor([1,2]), torch.tensor([1,3])) == 1; print('radix helper works')"
```

The full relevant FreeToken check is:

```bat
"C:\Users\jay\AppData\Local\FreeToken\venv\Scripts\python.exe" "D:\FreeToken-ple-mmap\windows-shim\smoke_test.py" --base-url http://127.0.0.1:2020/v1 --model Qwen3.8-Flash-Next-NVFP4
```

The repository-wide pytest command documented by the project is `uv run pytest tests/ -m "not slow"`. It is not part of this local bridge gate: no tracked FreeToken product source is changed, the reused Desktop Python has no pytest installed, and `tests/README.md` identifies real command execution as the proper check for this surface. No package installation will be added just to test the local shim.

Each slice is complete only when its command passes and a subsequent `GET /v1/models` still succeeds.

## Task breakdown

### Task 1: Keep the Windows server alive across cache reuse and tool requests

**Outcome:** The normal SSD-backed launch survives repeated text, stream, and tool requests instead of stopping when the radix cache helper first builds.

**Blocked by:** None.

**Files:**
- Create: `windows-shim/smoke_test.py` — repeatable end-to-end API checks using only Python's standard library.
- Modify: `windows-shim/sitecustomize.py` — normalize ordinary MSVC and CUDA host compiler flags to C++20 inside `_generate_ninja_windows`.
- Test: `python/freetoken/kernel/radix.py` — exercise `fast_compare_key` through its current interface.
- Test: `serve-pr279.cmd` — launch the full server on port 2020.
- Inspect: background server log — confirm SSD PLE mode, 50,048 allocated KV tokens, readiness, and absence of a scheduler exit after tests.

**Interfaces:**
- Consumes: `tvm_ffi.cpp.extension._generate_ninja_build`, `freetoken.kernel.radix.fast_compare_key`, `/v1/models`, and `/v1/chat/completions`.
- Produces: a stable local OpenAI-shaped service at `http://127.0.0.1:2020/v1` and a repeatable `smoke_test.py` command for later fork work.

**Acceptance criteria:**
- [ ] Generated Windows build text contains `/std:c++20` and no `/std:c++17`.
- [ ] CUDA's two host flags remain comma-packed so `/O2` is not mistaken for another input file.
- [ ] `fast_compare_key([1,2], [1,3])` returns `1`.
- [ ] The smoke test sees the expected model, successful sequential responses, streamed reasoning/content/usage, and a valid `get_weather({"city":"Paris"})` call.
- [ ] `GET /v1/models` succeeds after all requests and the server log has no new scheduler-exit message.

- [ ] **Step 1: Write the failing test**
  - Test: `windows-shim/smoke_test.py` against the real local server.
  - Expected behavior: model information is correct; sequential text, stream, and tool requests all finish; the final health request succeeds.
  - Independent expectations: model facts come from `config.json`; stream requirements come from the OpenAI event format; the tool name/argument come from the test's own forced tool definition.

- [ ] **Step 2: Run the focused test before the fix**
  - Run: start `serve-pr279.cmd --port 2020`, then run `smoke_test.py` with the full-relevant-check command above.
  - Expected: reproduce the current failure when cache comparison first calls the delayed radix helper; the log contains `/std:c++17`, `source_location`/`integral` errors, `ninja exited with status 1`, and a scheduler exit.
  - Preserve the existing `bg_142` log as the baseline if repeating the expensive pre-fix startup would add no new evidence.

- [ ] **Step 3: Implement the smallest change**
  - Change: in `_generate_ninja_windows`, first replace every generated `/std:c++17` with `/std:c++20`, then replace `-Xcompiler /std:c++20 /O2` with `-Xcompiler=/std:c++20,/O2`.
  - Preserve: all existing worker-address, event-loop, binary-discovery, CUDA-header, CUDA-library, and POSIX no-op behavior.
  - Preserve: the installed Desktop files and all tracked PR #279 source.

- [ ] **Step 4: Run the focused test again**
  - Run: compile-check `windows-shim/sitecustomize.py`, then run the focused `fast_compare_key` command above.
  - Expected: Python reports no syntax error, the helper builds with `/std:c++20`, and prints `radix helper works`.

- [ ] **Step 5: Run relevant checks**
  - Run: start a supervised background task with `powershell.exe -NoProfile -Command "& 'D:\FreeToken-ple-mmap\serve-pr279.cmd' --port 2020"`, waiting for `API server is ready` or an error.
  - Run: execute the full `smoke_test.py` command above.
  - Expected: every smoke-test section reports PASS; the last `/v1/models` request succeeds; logs still show `ple_backend='mmap'`, `Allocating 50048 tokens for KV cache`, and no post-test worker exit.

### Task 2: Add the stable FreeToken model to pi and prove a complete tool round trip

**Outcome:** Pi can select the local FreeToken model, display its safe limits, generate a normal answer, and complete a read-tool task while the previous default remains unchanged.

**Blocked by:** Task 1: Keep the Windows server alive across cache reuse and tool requests.

**Files:**
- Modify: `C:\Users\jay\.pi\agent\models.json` — add provider `freetoken-local` with the measured service capabilities.
- Modify: `C:\Users\jay\.pi\agent\settings.json` — append the FreeToken model to `enabledModels` only.
- Test: pi's `--list-models`, no-tool print mode, and read-tool print mode.

**Interfaces:**
- Consumes: stable `http://127.0.0.1:2020/v1`, model ID `Qwen3.8-Flash-Next-NVFP4`, pi's `openai-completions` provider shape, and pi's built-in `read` tool.
- Produces: selectable model key `freetoken-local/Qwen3.8-Flash-Next-NVFP4`.

**Provider entry:**
- `api`: `openai-completions`
- `apiKey`: a non-secret local placeholder
- `supportsStore`: false
- `supportsDeveloperRole`: false, proven by the direct 400 response for that role
- `supportsReasoningEffort`: true
- `supportsUsageInStreaming`: true
- `supportsFinishReason`: true
- `supportsStrictMode`: false until strict tool definitions are separately proven
- `maxTokensField`: `max_tokens`
- model input: text only
- `contextWindow`: 50048
- `maxTokens`: 16384
- costs: all zero
- sampling: temperature 1.0, top-p 0.95, top-k 20
- thinking map: low→low, medium→medium, xhigh→xhigh; off, minimal, high, and max are explicitly unavailable

**Acceptance criteria:**
- [ ] Both edited files remain valid JSON.
- [ ] `pi --list-models freetoken-local` shows one text-only model with about 50K context, 16.4K output, and reasoning enabled.
- [ ] A pi no-tool request returns `pi text works` through FreeToken.
- [ ] A pi request limited to the `read` tool reads `D:\FreeToken-ple-mmap\CONTEXT.md` and reports its first heading.
- [ ] `settings.json` still uses `llama-local`, `qwen38-flash-next-atomic-427`, and `xhigh` as its three defaults.
- [ ] The FreeToken server still answers `/v1/models` after both pi requests.

- [ ] **Step 1: Write the failing test**
  - Test: pi model discovery before the configuration change.
  - Run: `pi --list-models freetoken-local`.
  - Expected behavior after implementation: exactly one `freetoken-local` row; before implementation there is no row.

- [ ] **Step 2: Run the focused test**
  - Run: `pi --list-models freetoken-local` before editing.
  - Expected: no FreeToken model appears, proving the missing configuration rather than an endpoint problem.

- [ ] **Step 3: Implement the smallest change**
  - Change: add the provider/model object to `models.json` without changing `llama-local`.
  - Change: append `freetoken-local/Qwen3.8-Flash-Next-NVFP4` to `enabledModels` without changing default settings.
  - Preserve: all existing model entries, packages, extensions, and settings.

- [ ] **Step 4: Run the focused test again**
  - Run: parse both files with Python's `json.load`, then run `pi --list-models freetoken-local`.
  - Expected: both files parse and exactly one FreeToken model row shows text input, about 50K context, 16.4K output, and reasoning.

- [ ] **Step 5: Run relevant checks**
  - Run text check:
    `pi --provider freetoken-local --model Qwen3.8-Flash-Next-NVFP4 --thinking low --no-session --no-context-files --no-skills --no-prompt-templates --no-tools -p "Reply with exactly: pi text works"`
  - Expected: final visible answer is `pi text works`.
  - Run tool check:
    `pi --provider freetoken-local --model Qwen3.8-Flash-Next-NVFP4 --thinking low --no-session --no-context-files --no-skills --no-prompt-templates --tools read -p "Use the read tool on D:\FreeToken-ple-mmap\CONTEXT.md, then reply with only its first heading."`
  - Expected: pi executes `read` and the final answer is `Local Windows PLE mmap test context` (a leading `#` is also acceptable).
  - Run: assert with Python that `defaultProvider == "llama-local"`, `defaultModel == "qwen38-flash-next-atomic-427"`, and `defaultThinkingLevel == "xhigh"`; then request `/v1/models` once more.
  - Expected: defaults are unchanged and the FreeToken model list still returns successfully.

## Self-review

- **Coverage:** Every acceptance check in `CONTEXT.md` maps to Task 1 or Task 2. Picture input, publishing, benchmarking, and Desktop changes remain out of scope.
- **Evidence:** Paths come from the checkout, pi documentation, current config, model `config.json`, and `bg_142`. Commands use the installed Python, current launch script, and pi's documented flags.
- **Interfaces:** Task 1 produces the exact `http://127.0.0.1:2020/v1` address and model ID consumed by Task 2.
- **Dependencies:** Task 2 waits for Task 1; there is no cycle and no claimed parallel work on shared files.
- **Seams:** The smoke test uses the public HTTP API; the pi check uses normal pi model and tool paths. The one focused helper command directly targets the failure shown by the log and compares against an independent expected mismatch position.
- **Scope:** No tracked upstream source, installed Desktop file, package list, public fork, or unrelated pi setting is changed.
- **Placeholders:** No unresolved marker or open design choice remains.
- **Fresh-context check:** Each task names exact files, inputs, commands, expected results, preservation rules, and completion evidence.
