# Windows native PLE mmap and pi test design

## Status

Approved by the user's request to add the working local endpoint to pi and test it fully before documenting or forking.

## Approach

1. Keep PR #279 source in its separate checkout.
2. Reuse FreeToken Desktop's installed Windows binary helpers and CUDA headers through a local Python startup shim; do not edit the Desktop installation.
3. Replace Unix-only worker addresses with private local TCP addresses.
4. Force the Windows event loop form required by the worker messaging library.
5. Normalize generated Windows compiler flags to C++20 for both CUDA and ordinary C++ helper builds.
6. Start the Qwen model with `--ple-backend mmap`, one running request at a time, and 262,144 reserved KV tokens. Let `--moe-cache-auto` reduce the GPU expert cache to fit the full model context.
7. Add an OpenAI-shaped local provider to pi at `http://127.0.0.1:2020/v1`.
8. Mark the provider as not supporting the newer `developer` role because FreeToken's model template rejects it.
9. Preserve the existing pi default model and only add FreeToken to the selectable list.

## Expected pi model facts

- Provider ID: `freetoken-local`
- Model ID: `Qwen3.8-Flash-Next-NVFP4`
- Input: text and images, subject to a direct image check
- Model and configured pi context limit: 262,144 tokens
- Output limit for the initial safe entry: 16,384 tokens
- Reasoning choices: low, medium, xhigh
- Cost: zero because it runs on the user's computer

## Failure handling

- If a helper build fails, fix only the local shim and restart; do not change Desktop files.
- If picture handling is not proven, list text-only until it is proven.
- If tool use fails, do not call the pi setup complete; capture the failure and repair or clearly mark the blocker.
- If the full-context memory split is too tight, keep the existing Windows-safe launch file as a fallback and restore the previously proven 50,048-token limit.

## Verification

- Check server model information and verify that live cache pages total 262,144 tokens.
- Send sequential non-live and live text requests.
- Send a forced or strongly requested tool call and inspect its returned name and arguments.
- Confirm server health after each request.
- Validate pi's model list.
- Run pi once with an ordinary prompt and once with a task that requires one harmless read-only tool.

## Rollback

All FreeToken changes are untracked local files. The pi edit is one added provider plus one enabled-model string. Removing those additions restores the earlier setup.
