# J6 report — memory governor controls

## Verdict

Implemented all four J6 parts. The focused J6 tests pass. The required suite has four known failures that were already present before this work; none is in a changed J6 file.

## Built

- Added the `Memory governor` group and seven helper-only dials:
  - `GovernorUpMarginGB`: `0.5` GB
  - `GovernorStepIntervalS`: `5` s
  - `GovernorUpHoldS`: `60` s
  - `GovernorMaxHoldS`: `600` s
  - `GovernorPostUpGraceS`: `10` s
  - `GovernorRAMRungsBeforeUp`: `2` rungs
  - `GovernorVRAMRungsBeforeUp`: `1` rung
- Moved `MemoryGovernor`, `GovernorVRAMFreeGB`, and `GovernorRAMFreeGB` into the new group.
- Added all seven names to `_LAUNCHER_ORDER`; they validate, save, and load through the real boot-file parser. Their sensible slider ranges and long help text are present. `build_launch` leaves them out of the model command and documents that they are helper-only.
- Added configurable `ram_rungs_before_up` and `vram_rungs_before_up` to `GovernorPolicy`, with defaults `2` and `1`.
- `ProcessManager.start_governor` maps every governor dial. `apply_governor_settings` rebuilds and swaps the running policy without stopping or contacting the model server, and logs one line with the new values.
- `PUT /api/settings` calls `apply_governor_settings` after the boot file is saved.
- `Engine.residency_report` now reports `layer_bytes`, calculated as `num_experts * expert_bytes_per_slot(bank_sources, owned_layers)`, or `0` when the geometry is unavailable. The existing residency message dictionary and HTTP route carry the field unchanged.
- `GovernorLoop` refreshes residency at most once per minute while serving, uses a positive `layer_bytes` value for `policy.rung_bytes`, and uses the measured Qwen value only as the zero/unknown fallback.
- Older profile scripts remain exact: helper-only dials are omitted until a profile explicitly stores them, while explicit helper lines still round-trip.

## Tests and proof

Focused J6 run: `44 passed, 2 warnings`.

Required run:

- `803 passed`
- `2 skipped`
- `4 failed` (known pre-existing failures):
  - `tests/settings/test_browse.py::test_unresolvable_paths_fall_back_to_the_drive_list`
  - `tests/settings/test_memory_plan.py::test_checkpoint_metadata_scanned_once_across_candidates[False]`
  - `tests/settings/test_memory_plan.py::test_checkpoint_metadata_scanned_once_across_candidates[True]`
  - `tests/server/test_parser_auto_selection.py::test_ple_backend_is_exposed_by_the_server_cli`
- `git diff --check`: clean.
- No GPU-box/live serving test was run; this devbox has no CUDA. The required live check remains for L3.

Full command output and status are in `J6-proof.txt`; cumulative patch is in `J6.diff`.

## Assumptions and follow-ups

- The job card names only the primary J6 source files, but persistence order lives in `boot_parser.py`, launch omission lives in `linux_launch.py`, and profile compatibility lives in `profile_scripts.py`; those three small supporting edits are necessary for the requested behavior and are included in the cumulative diff.
- `layer_bytes` uses the same first-streaming-layer geometry as `cache_budget.expert_bytes_per_slot`; an all-owned or otherwise unreadable bank layout reports zero and therefore uses the Qwen fallback.
- The daemon remains torch-free: all policy construction and live swapping stay in the settings process.
- A live Windows Save/status check and the defaults-before/after boot-file comparison still need to be performed on the RTX 5090 box.

## Commits

- `ee70c1d` — `feat(settings): expose memory governor controls`
- `43d78ac` — `feat(settings): apply governor policy changes live`
- `d52e905` — `feat(server): report model-derived governor rung size`
