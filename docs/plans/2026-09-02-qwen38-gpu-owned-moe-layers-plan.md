# GPU-owned MoE layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a chosen subset of the 48 Qwen3.8-Flash-Next-NVFP4 MoE layers keep all 512 experts permanently resident in VRAM and allocate **no pinned host bank at all** for them, handing 1.322 GiB of host RAM back per owned layer while every other layer keeps today's pinned-bank + global-LRU behaviour.

**Architecture:** A new per-layer residency class `GPU_OWNED` runs end to end through the existing `--moe-cpu-layers` machinery: CLI flag -> `EngineConfig` field -> spec parser -> resolver -> per-layer residency label vector -> ambient `_ResidencyPlan` consulted by `alloc_layer_banks` (which allocates a device tensor plus a *reusable* pinned staging layer instead of a `HostBank`) -> `ExpertBanks.layer_residency` -> `OffloadMoeCache.set_bank_sources`, which stores the owned layers' device banks in a `resident_banks` registry instead of registering them with the slot cache. The NVFP4 decode/prefill kernels index row `topk_ids[m,k]` of whatever tensors they are handed, so an owned layer needs **no kernel change and no LRU bookkeeping** -- `OffloadMoELayer` simply passes `cache.resident_views(layer_id)` and the raw (un-remapped) `topk_ids`. Every pointer-table builder and movement entry point excludes or loudly rejects owned layers.

**Tech Stack:** Python 3.12, PyTorch 2.11+cu130, pytest 9.x. Windows 11 / PowerShell host, RTX 5090. No new dependencies, no C++/CUDA changes.

**Spec:** docs/design/2026-09-02-qwen38-gpu-owned-moe-layers-design.md

## Global Constraints

- Host is Windows 11; every command in this plan is written for PowerShell (the `PowerShell` tool), not bash.
- Tests run WITHOUT a GPU. Set `$env:CUDA_VISIBLE_DEVICES = '-1'` before pytest (PowerShell **deletes** an env var assigned `''`, so `'-1'` is the portable spelling of the spec's `CUDA_VISIBLE_DEVICES=""`; both hide every device from `torch.cuda.is_available()`).
- One-time test-runner install (the Desktop venv has no pip/pytest): `uv pip install --target D:\FreeToken\.local\pytest-site pytest pytest-timeout` (`D:\FreeToken\.local\` is gitignored).
- Test runner, verbatim, for every "run the test" step (the sitecustomize shim MUST be first on `PYTHONPATH`):
  ```powershell
  $env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
  $env:CUDA_VISIBLE_DEVICES = '-1'
  & "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest <paths> -q -p no:cacheprovider
  ```
- Never modify anything under `D:\Models`.
- Never launch the server and never touch the GPU during implementation. Live verification is Task 8's operator checklist only.
- Commit after every task; every commit message ends with the two trailer lines:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Hnf1bGBLU4HLq9uHPtNjwU
  ```
- Default owned set is `{0, 1, 2, 6, 7, 22}` (`auto`); per-expert row is 2,772,480 B; one owned layer is 512 x 2,772,480 = 1,419,509,760 B = 1.322 GiB.
- Names, fixed for the whole plan: flag `--moe-gpu-owned-layers`, config field `moe_gpu_owned_layers`, launcher param `-GpuOwnedLayers`, env `FREETOKEN_MOE_GPU_OWNED_LAYERS` (launcher-only), residency label `GPU_OWNED` (`HostResidency.GPU_OWNED.value == "gpu_owned"`).

## File Structure

| File | Created/Modified | Responsibility |
|---|---|---|
| `python/freetoken/engine/config.py` | Modified | `EngineConfig.moe_gpu_owned_layers: str \| None` beside `moe_cpu_layers`. |
| `python/freetoken/server/args.py` | Modified | `--moe-gpu-owned-layers` argparse entry beside `--moe-cpu-layers`. |
| `python/freetoken/engine/engine.py` | Modified | `GPU_OWNED_LAYER_RANK`, `_parse_gpu_owned_layers_spec`, `_resolve_gpu_owned_layers`, `_gpu_owned_boot_line`, `_DENSE_MOE_SETTINGS` entry, `_adjust_config` validation, owned-aware budget in `_resolve_auto_moe_cache_size` / `_target_moe_and_expert_bytes`, `_check_gpu_owned_cache_fits`, residency-vector + `set_bank_sources` wiring, boot log. |
| `python/freetoken/engine/cache_budget.py` | Modified | `expert_bytes_per_slot(..., gpu_owned_layers=)`, `gpu_owned_reservation_bytes`, `check_explicit_moe_cache_fits`. |
| `python/freetoken/moe/offload_cache.py` | Modified | Owned-layer registry (`gpu_owned_layer_ids`, `resident_banks`, `resident_views`, `is_gpu_owned_layer`), copy-plan/prefetch/prefill exclusions, movement guards, `_note_decode_routing` helper, owned-aware reporting. |
| `python/freetoken/layers/moe.py` | Modified | `_decode_routed` / `_prefill_routed` owned branches (raw ids + resident views, no LRU, no overlap buffer). |
| `python/freetoken/moe/host_banks.py` | Modified | `HostResidency.GPU_OWNED`, `HostBank.fill`, `GpuOwnedBank`, `GpuOwnedStagingPool`, `alloc_layer_banks(gpu_owned=, device=)`, `requested_residency(device=)`, `PinPipeline.submit_flush` + owned layer-completion sink, `_settle`/`pin_banks` loud refusal. |
| `python/freetoken/models/nvfp4_banks.py` | Modified | Serial + parallel NVFP4 loaders fill through `bank.fill` and return `bank.tensor`, so owned layers land in VRAM. |
| `python/freetoken/moe/expert_banks.py` | Modified | `bank_bytes_estimate(..., gpu_owned=)`, `_echo_residency` hard failure for an unapplied `GPU_OWNED` request, `requested_residency(..., device=device)`. |
| `python/freetoken/checkpoint/ftw.py` | Modified | `load_ftw_banks` refuses `GPU_OWNED` labels. |
| `python/freetoken/moe/cpu_executor.py` | Modified | `_resolve_banks` refuses CUDA bank sources. |
| `python/freetoken/kvcache/cache_status.py` | Modified | `compute_cache_pools` reports `gpu_owned_layers`. |
| `python/freetoken/server/api_server.py` | Modified | `cache_geometry` surfaces `gpu_owned_layers`. |
| `python/freetoken/server/model_meta.py` | Modified | `moe_total_experts` subtracts owned layers. |
| `python/freetoken/cache_report.py` | Modified | `cache_rate` denominates over streaming layers. |
| `python/freetoken/scheduler/scheduler.py` | Modified | `_reply_routing_stats` returns `gpu_owned_layers`. |
| `python/freetoken/engine/mtp_fast_verify.py` | Modified | `_movement_result` reports `gpu_owned_layers`; reconciliation invariant re-stated for resident layers. |
| `scripts/start-qwen38-flash-next-mmap-windows.ps1` | Modified | `-GpuOwnedLayers` param, `FREETOKEN_MOE_GPU_OWNED_LAYERS` fallback, banner line, flag passthrough. |
| `docs/cli.md` | Modified | `--moe-gpu-owned-layers` row in the MoE flag table. |
| `docs/windows-qwen38-flash-next-mmap.md` | Modified | "GPU-owned MoE layers" section. |
| `tests/engine/test_moe_gpu_owned_layers.py` | **Created** | Parser / resolver / `_adjust_config` validation / dense inertness / launcher passthrough. |
| `tests/engine/test_cache_budget.py` | Modified | Owned reservation arithmetic, first-streaming-layer slot bytes, explicit-size overflow message. |
| `tests/moe/test_offload.py` | Modified | Owned-layer cache registry, movement guards, rebuild, forward paths, overlap alternation. |
| `tests/moe/test_gpu_owned_banks.py` | **Created** | Loader: device placement, staging reuse/back-pressure, flush fidelity, FTW + CPU-executor refusals, bank-bytes estimate. |
| `tests/moe/test_routing_stats.py` | Modified | `resident: true` / `miss_rate: null` rows, `decode_freq` counts owned layers, `slots_per_layer` denominator. |
| `tests/engine/test_mtp_fast_verify.py` | Modified | `gpu_owned_layers` in the movement report; reconciliation holds with resident layers. |
| `tests/server/test_gpu_owned_geometry.py` | **Created** | `/v1/cache/status` geometry + `cache_rate` + `moe_total_experts` subtract owned layers. |
| `docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-status.md` | **Created** | Operator live-verification checklist the implementer fills in. |

## Tasks

### Task 1: Config field, CLI flag, spec parser, resolver, validation

**Files:**
- Modify `python/freetoken/engine/config.py` (insert after line 350, the `moe_cpu_layers` field)
- Modify `python/freetoken/server/args.py` (insert after line 578, the `--moe-cpu-layers` block)
- Modify `python/freetoken/engine/engine.py` (new constants + functions after `_resolve_cpu_layers`, which ends at line 1767; `_DENSE_MOE_SETTINGS` at 1848-1859; new validation after the `--moe-cpu-layers` check at 2161-2166)
- Create `tests/engine/test_moe_gpu_owned_layers.py`

**Interfaces:**
- Consumes: `EngineConfig` (`engine/config.py`), `ServerArgs` (`server/args.py`, a `SchedulerConfig` -> `EngineConfig` subclass, so the dataclass field IS the CLI default), `_parse_cpu_layers_spec` grammar (`engine.py:1727-1753`), `_resolve_cpu_layers` (`engine.py:1756-1767`), `is_offload_moe_backend`, `freetoken.checkpoint.ftw.is_ftw_checkpoint`.
- Produces:
  - `EngineConfig.moe_gpu_owned_layers: str | None = None`
  - `freetoken.engine.engine.GPU_OWNED_LAYER_RANK: tuple[int, ...]` (48 entries, hungriest first)
  - `freetoken.engine.engine._parse_gpu_owned_layers_spec(spec: str, num_moe_layers: int) -> frozenset[int]`
  - `freetoken.engine.engine._resolve_gpu_owned_layers(config, num_moe_layers: int) -> frozenset[int]`
  - `_DENSE_MOE_SETTINGS["moe_gpu_owned_layers"] = None`
  - CLI `--moe-gpu-owned-layers <spec>`

- [x] **Step 1: Write the failing test**

Create `tests/engine/test_moe_gpu_owned_layers.py`:

```python
"""--moe-gpu-owned-layers: spec grammar, resolver, and the _adjust_config gate.

CPU-only: nothing here builds an engine, allocates a bank, or touches CUDA.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from freetoken.engine.engine import GPU_OWNED_LAYER_RANK
from freetoken.engine.engine import _parse_gpu_owned_layers_spec as parse
from freetoken.engine.engine import _resolve_gpu_owned_layers as resolve

L = 48
ANON_MODEL = "/models/anon"


def _cfg(**over):
    base = dict(
        moe_backend="offload",
        moe_cpu_layers=None,
        moe_gpu_owned_layers=None,
        model_path=ANON_MODEL,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------ the grammar


def test_the_ranked_list_is_the_measured_order_and_covers_every_layer():
    # derived from docs/research/routing-skew-2026-09-02/{code,prose,chat8k,toolcall}.json
    assert GPU_OWNED_LAYER_RANK[:8] == (1, 6, 0, 2, 7, 22, 10, 13)
    assert len(GPU_OWNED_LAYER_RANK) == L
    assert sorted(GPU_OWNED_LAYER_RANK) == list(range(L))


def test_auto_is_the_six_hungriest_layers():
    assert parse("auto", L) == frozenset({0, 1, 2, 6, 7, 22})


@pytest.mark.parametrize("n,expected", [(1, {1}), (3, {1, 6, 0}), (8, {1, 6, 0, 2, 7, 22, 10, 13})])
def test_auto_n_takes_the_first_n_of_the_ranked_list(n, expected):
    assert parse(f"auto:{n}", L) == frozenset(expected)


def test_auto_zero_is_empty_and_auto_all_is_every_layer():
    assert parse("auto:0", L) == frozenset()
    assert len(parse(f"auto:{L}", L)) == L


def test_explicit_list_count_and_fraction_reuse_the_cpu_layers_grammar():
    assert parse("3,7,11", L) == frozenset({3, 7, 11})
    assert parse("3, 7 ,11,", L) == frozenset({3, 7, 11})
    assert parse("5,5,5", L) == frozenset({5})
    assert parse("6", L) == frozenset({0, 8, 16, 24, 32, 40})
    assert len(parse("0.125", L)) == 6
    assert parse("", L) == frozenset()
    assert parse("   ", L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "48,1", "-1", "1.5", "auto:-1", "auto:49", "auto:x"])
def test_out_of_range_specs_raise(spec):
    with pytest.raises(ValueError):
        parse(spec, L)


def test_the_error_names_the_flag_not_moe_cpu_layers():
    with pytest.raises(ValueError, match="--moe-gpu-owned-layers"):
        parse("99", L)


# ------------------------------------------------------------------ the resolver


def test_resolve_needs_an_offload_backend_and_a_spec():
    assert resolve(_cfg(moe_gpu_owned_layers="auto"), L) == frozenset({0, 1, 2, 6, 7, 22})
    assert resolve(_cfg(moe_gpu_owned_layers=None), L) == frozenset()
    assert resolve(_cfg(moe_backend="fused", moe_gpu_owned_layers="auto"), L) == frozenset()
    assert resolve(_cfg(moe_backend="cpu", moe_gpu_owned_layers="auto"), L) == frozenset()


# ------------------------------------------------------------ the _adjust_config gate


class _HFConfig:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse_args(*extra: str):
    from freetoken.server.args import parse_args

    config = _HFConfig({"architectures": ["Qwen2ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        args, _run_shell = parse_args(["--model", ANON_MODEL, *extra])
    return args


def test_the_flag_round_trips_through_server_args():
    assert _parse_args().moe_gpu_owned_layers is None
    assert _parse_args("--moe-gpu-owned-layers", "auto").moe_gpu_owned_layers == "auto"
    assert _parse_args("--moe-gpu-owned-layers", "0,1,2").moe_gpu_owned_layers == "0,1,2"


def _adjust(**over):
    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        is_moe=True,
        num_moe_layers=L,
        num_experts=512,
        expert_quant="nvfp4",
        moe_backend="offload",
        nvfp4_backend="triton",
    )
    config = SimpleNamespace(
        model_config=model_config,
        model_path=ANON_MODEL,
        moe_backend="offload",
        moe_cpu_layers=None,
        moe_gpu_owned_layers="auto",
        moe_prefill_overlap=True,
        moe_cache_size=6750,
    )
    for name, value in over.items():
        setattr(config, name, value)
    return config, _adjust_config


def test_validation_requires_the_offload_backend():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config, _ = _adjust(moe_backend="cpu")
    with pytest.raises(ValueError, match="requires --moe-backend offload"):
        _validate_gpu_owned_layers(config, L)


def test_validation_rejects_overlap_with_the_cpu_layer_set():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config, _ = _adjust(moe_cpu_layers="0,1", moe_gpu_owned_layers="auto")
    with pytest.raises(ValueError, match=r"both GPU-owned and CPU layers: \[0, 1\]"):
        _validate_gpu_owned_layers(config, L)


def test_validation_rejects_an_ftw_checkpoint(tmp_path):
    from freetoken.engine.engine import _validate_gpu_owned_layers

    (tmp_path / "freetoken_weight.json").write_text('{"tensors": []}', encoding="utf-8")
    config, _ = _adjust(model_path=str(tmp_path))
    with pytest.raises(ValueError, match="FTW packed checkpoint"):
        _validate_gpu_owned_layers(config, L)


def test_validation_keeps_two_expert_layers_of_lru_when_overlap_is_on():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    # 512 experts, prefill overlap on -> the LRU floor is 2 * 512 = 1024 slots
    config, _ = _adjust(moe_cache_size=1023)
    with pytest.raises(ValueError, match="prefill overlap needs at least 1024"):
        _validate_gpu_owned_layers(config, L)
    config, _ = _adjust(moe_cache_size=1024)
    assert _validate_gpu_owned_layers(config, L) == frozenset({0, 1, 2, 6, 7, 22})


def test_validation_is_inert_without_the_flag():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config, _ = _adjust(moe_gpu_owned_layers=None, moe_backend="fused")
    assert _validate_gpu_owned_layers(config, L) == frozenset()


def test_the_dense_override_table_clears_the_flag():
    """A dense checkpoint has no routed experts; the knob must not survive the reset."""
    from freetoken.engine.engine import _DENSE_MOE_SETTINGS

    assert _DENSE_MOE_SETTINGS["moe_gpu_owned_layers"] is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
```

- [x] **Step 2: Run test to verify it fails**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine/test_moe_gpu_owned_layers.py -q -p no:cacheprovider
```

Expected: collection error, every test errors out with
`ImportError: cannot import name 'GPU_OWNED_LAYER_RANK' from 'freetoken.engine.engine'`.

- [x] **Step 3: Write minimal implementation**

`python/freetoken/engine/config.py` -- insert immediately after the `moe_cpu_layers` field (line 350):

```python
    moe_cpu_layers: str | None = None
    # GPU-owned MoE layers (--moe-backend offload only): layers whose full expert set is
    # permanently VRAM-resident and which allocate NO host bank at all. Spec is the
    # --moe-cpu-layers grammar (explicit ids "0,1,2", a count "6", a fraction "0.125")
    # plus "auto" (the measured six hungriest layers) and "auto:N". None = off.
    # Each owned layer costs num_experts LRU slots of VRAM and returns one host bank of RAM.
    moe_gpu_owned_layers: str | None = None
```

`python/freetoken/server/args.py` -- insert immediately after the `--moe-cpu-layers` block (which closes at line 578):

```python
    parser.add_argument(
        "--moe-gpu-owned-layers",
        type=str,
        default=ServerArgs.moe_gpu_owned_layers,
        help=(
            "With --moe-backend offload: which MoE layers keep every expert permanently "
            "resident in VRAM and allocate no host bank at all (the rest keep the pinned "
            "bank + LRU slot cache). Explicit id list ('0,1,2'), a count ('6' = 6 layers "
            "evenly strided), a fraction ('0.125'), 'auto' (the measured hungriest layers) "
            "or 'auto:N'. Each owned layer costs num_experts cache slots of VRAM and "
            "returns one layer of host RAM. Unset = off."
        ),
    )
```

`python/freetoken/engine/engine.py` -- append after `_resolve_cpu_layers` (which ends at line 1767, just before the `_CPU_MOE_ACTS` comment):

```python
# MoE layers ranked by measured decode miss rate, hungriest first: the mean of
# per_layer[].miss_rate over the four cache_size=6750 decode captures in
# docs/research/routing-skew-2026-09-02/{code,prose,chat8k,toolcall}.json. The order is
# identical under mean missing_per_step and under the pooled union.json. A fixed constant,
# never a runtime heuristic; _auto_cpu_layers' U-shaped head+tail guess is NOT supported by
# this data (the tail 39-47 is mid-pack, the minimum is layer 31) and must not be reused.
GPU_OWNED_LAYER_RANK = (
    1, 6, 0, 2, 7, 22, 10, 13, 5, 18, 21, 38, 8, 12, 34, 11,
    24, 26, 29, 14, 28, 17, 19, 4, 35, 9, 37, 30, 33, 20, 3, 23,
    27, 25, 45, 36, 42, 47, 41, 46, 44, 40, 43, 39, 16, 32, 15, 31,
)


def _parse_gpu_owned_layers_spec(spec: str, num_moe_layers: int) -> frozenset[int]:
    """Parse ``--moe-gpu-owned-layers``: the ``--moe-cpu-layers`` grammar (explicit id list
    ``"0,1,2"``, count ``"6"``, fraction ``"0.125"``) plus ``"auto"`` (the six hungriest
    layers of :data:`GPU_OWNED_LAYER_RANK`) and ``"auto:N"`` (its first N)."""
    s = spec.strip()
    if s == "auto":
        s = "auto:6"
    if s.startswith("auto:"):
        try:
            n = int(s[len("auto:"):])
        except ValueError as exc:
            raise ValueError(f"--moe-gpu-owned-layers {spec!r}: 'auto:N' needs an integer N") from exc
        if not 0 <= n <= num_moe_layers:
            raise ValueError(
                f"--moe-gpu-owned-layers auto:{n} must be in [0, {num_moe_layers}]"
            )
        ranked = [i for i in GPU_OWNED_LAYER_RANK if i < num_moe_layers]
        return frozenset(ranked[:n])
    try:
        return _parse_cpu_layers_spec(s, num_moe_layers)
    except ValueError as exc:
        # reuse the grammar, not its error text: the operator typed a different flag
        raise ValueError(str(exc).replace("--moe-cpu-layers", "--moe-gpu-owned-layers")) from None


def _resolve_gpu_owned_layers(config: EngineConfig, num_moe_layers: int) -> frozenset[int]:
    """MoE layer ids whose experts are permanently VRAM-resident (no host bank).

    Only ``--moe-backend offload`` supports it: ``cpu``/``hybrid`` read every expert on the
    CPU (a VRAM-resident layer has no host bank to read), and ``fused`` keeps every expert
    resident already. Validation of the resolved set lives in
    :func:`_validate_gpu_owned_layers`.
    """
    spec = config.moe_gpu_owned_layers
    if not spec or config.moe_backend != "offload":
        return frozenset()
    return _parse_gpu_owned_layers_spec(spec, num_moe_layers)


def _validate_gpu_owned_layers(config: EngineConfig, num_moe_layers: int) -> frozenset[int]:
    """Resolve and fully validate the owned set, or raise. Returns the empty set when off."""
    from freetoken.checkpoint.ftw import is_ftw_checkpoint

    spec = config.moe_gpu_owned_layers
    if not spec:
        return frozenset()
    if config.moe_backend != "offload":
        raise ValueError(
            "--moe-gpu-owned-layers requires --moe-backend offload (got "
            f"{config.moe_backend!r}): a VRAM-resident layer has no host bank for the CPU "
            "executor to read, and 'fused' keeps every expert resident already"
        )
    owned = _parse_gpu_owned_layers_spec(spec, num_moe_layers)
    if not owned:
        return owned
    cpu_layer_ids = _resolve_cpu_layers(config, num_moe_layers)
    clash = sorted(owned & cpu_layer_ids)
    if clash:
        raise ValueError(
            f"--moe-gpu-owned-layers and --moe-cpu-layers name layers that are "
            f"both GPU-owned and CPU layers: {clash}"
        )
    if config.model_path and is_ftw_checkpoint(config.model_path):
        raise ValueError(
            "--moe-gpu-owned-layers is not supported on an FTW packed checkpoint "
            "(load_ftw_banks always allocates one host bank per layer); serve the original "
            "checkpoint or drop the flag"
        )
    num_experts = config.model_config.num_experts
    floor = 2 * num_experts if config.moe_prefill_overlap else num_experts
    if config.moe_cache_size and config.moe_cache_size < floor:
        raise ValueError(
            f"--moe-gpu-owned-layers leaves an LRU of {config.moe_cache_size} slots for the "
            f"{num_moe_layers - len(owned)} streaming layers, but prefill overlap needs at "
            f"least {floor} slots (2 x num_experts); raise --moe-cache-size, own fewer "
            f"layers, or pass --disable-moe-prefill-overlap"
        )
    return owned
```

`python/freetoken/engine/engine.py` -- add the dense-model entry to `_DENSE_MOE_SETTINGS` (line 1852, beside `"moe_cpu_layers": None`):

```python
    "moe_cpu_layers": None,
    "moe_gpu_owned_layers": None,
```

`python/freetoken/engine/engine.py` -- call the validator in `_adjust_config`, immediately after the `--moe-cpu-layers` backend check (which closes at line 2166):

```python
    if is_moe and config.moe_cpu_layers and config.moe_backend not in ("offload", "hybrid"):
        # the layer split needs the offload host banks + slot cache; 'cpu' already runs every layer on CPU, 'fused' keeps experts resident on the GPU (no host banks)
        raise ValueError(
            "--moe-cpu-layers requires --moe-backend offload or hybrid (got "
            f"{config.moe_backend!r}); use --moe-backend cpu to run all layers on CPU"
        )

    if is_moe:
        # resolved once here so a bad spec fails before any weight is read; the engine
        # re-resolves at cache build (the backend may still be 'auto' at parse time)
        _validate_gpu_owned_layers(config, model_config.num_moe_layers)
```

- [x] **Step 4: Run test to verify it passes**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine/test_moe_gpu_owned_layers.py tests/engine/test_moe_cpu_layers.py -q -p no:cacheprovider
```

- [x] **Step 5: Commit**

```powershell
git add python/freetoken/engine/config.py python/freetoken/server/args.py python/freetoken/engine/engine.py tests/engine/test_moe_gpu_owned_layers.py
git commit -m @'
feat(moe): --moe-gpu-owned-layers spec, resolver and validation

Adds the config field, the CLI flag, the auto/auto:N grammar over the measured
routing-skew rank, the offload-only resolver, and the _adjust_config gate
(backend, cpu-layer disjointness, FTW refusal, LRU floor, dense inertness).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hnf1bGBLU4HLq9uHPtNjwU
'@
```

### Task 2: VRAM budget arithmetic for the owned reservation

**Files:**
- Modify `python/freetoken/engine/cache_budget.py` (`expert_bytes_per_slot` at 17-28; two new functions after it)
- Modify `python/freetoken/engine/engine.py` (`_resolve_auto_moe_cache_size` at 631-660, specifically 639-642 and 649-651; `_target_moe_and_expert_bytes` at 904-916; new `_check_gpu_owned_cache_fits`)
- Modify `tests/engine/test_cache_budget.py` (append at the end, after line 527)

**Interfaces:**
- Consumes: `resolve_moe_cache_auto`, `plan_cache_budget`, `net_cache_budget_bytes`, `div_ceil`, `Engine._gpu_owned_layer_ids` (set by Task 6; default `frozenset()` from this task on).
- Produces:
  - `expert_bytes_per_slot(sources: dict[str, list[Tensor]], gpu_owned_layers: frozenset[int] = frozenset()) -> int`
  - `gpu_owned_reservation_bytes(owned_layers: int, num_experts: int, per_expert_bytes: int) -> int`
  - `check_explicit_moe_cache_fits(*, moe_cache_size: int, per_expert_bytes: int, budget_bytes: int, owned_layers: int, num_experts: int) -> None`
  - `Engine._gpu_owned_layer_ids: frozenset[int]` (attribute, initialised in `Engine.__init__`)
  - `Engine._check_gpu_owned_cache_fits(config, banks) -> None`

Note on scope: the two pure functions and `expert_bytes_per_slot` are unit-tested here. The three engine call sites are pure glue over them (no new arithmetic) and are exercised at boot -- Task 8's checklist records the boot line that proves the reservation landed.

- [x] **Step 1: Write the failing test**

Append to `tests/engine/test_cache_budget.py`:

```python
# --------------------------------------------------- GPU-owned MoE layers (--moe-gpu-owned-layers)

_PER_EXPERT = 2_772_480  # one Qwen3.8-Flash-Next NVFP4 expert row across the 6 banks
_E = 512                 # num_experts
_L = 48                  # num_moe_layers


def _auto_with_owned(owned: int):
    """resolve_moe_cache_auto exactly as Engine._resolve_auto_moe_cache_size calls it with
    ``owned`` GPU-owned layers: their bytes join fixed_cache_size, and they leave
    total_experts."""
    from freetoken.engine.cache_budget import gpu_owned_reservation_bytes

    return resolve_moe_cache_auto(
        baseline_free=40 << 30,
        weights_bytes=8 << 30,
        memory_ratio=0.9,
        cache_per_page=1 << 20,
        fixed_cache_size=gpu_owned_reservation_bytes(owned, _E, _PER_EXPERT),
        per_expert_bytes=_PER_EXPERT,
        num_experts=_E,
        total_experts=(_L - owned) * _E,
        prefill_overlap=True,
        kv_reserve_tokens=0,
        page_size=1,
        quant_format="nvfp4",
    )


def test_gpu_owned_reservation_is_a_whole_layer_of_slots():
    from freetoken.engine.cache_budget import gpu_owned_reservation_bytes

    assert gpu_owned_reservation_bytes(0, _E, _PER_EXPERT) == 0
    assert gpu_owned_reservation_bytes(6, _E, _PER_EXPERT) == 6 * _E * _PER_EXPERT
    assert gpu_owned_reservation_bytes(1, _E, _PER_EXPERT) == 1_419_509_760  # 1.322 GiB


def test_gpu_owned_reservation_shrinks_the_auto_slot_count_by_exactly_one_layer_each():
    base, _, base_overlap = _auto_with_owned(0)
    owned6, _, owned_overlap = _auto_with_owned(6)

    assert base - owned6 == 6 * _E  # 3072 slots, one full expert layer per owned layer
    assert base_overlap is True and owned_overlap is True


def test_expert_bytes_per_slot_reads_the_first_streaming_layer():
    from freetoken.engine.cache_budget import expert_bytes_per_slot

    # layer 0 is GPU-owned and (deliberately) a different row shape; the slot cost must come
    # from layer 1, the first STREAMING layer -- layer 0 is in the default owned set.
    sources = {
        "gate_up": [torch.zeros(4, 99, 8, dtype=torch.float16), torch.zeros(4, 32, 8, dtype=torch.float16)],
        "down": [torch.zeros(4, 99, 16, dtype=torch.float16), torch.zeros(4, 8, 16, dtype=torch.float16)],
    }

    assert expert_bytes_per_slot(sources, frozenset({0})) == 32 * 8 * 2 + 8 * 16 * 2
    assert expert_bytes_per_slot(sources) == 99 * 8 * 2 + 99 * 16 * 2  # no owned set: layer 0


def test_explicit_cache_size_that_overflows_the_budget_names_what_would_fit():
    from freetoken.engine.cache_budget import check_explicit_moe_cache_fits

    # budget funds 5000 slots total; 6 owned layers already take 3072 of them
    budget = 5000 * _PER_EXPERT
    check_explicit_moe_cache_fits(
        moe_cache_size=1928, per_expert_bytes=_PER_EXPERT, budget_bytes=budget,
        owned_layers=6, num_experts=_E,
    )  # exactly fits
    with pytest.raises(ValueError) as excinfo:
        check_explicit_moe_cache_fits(
            moe_cache_size=4400, per_expert_bytes=_PER_EXPERT, budget_bytes=budget,
            owned_layers=6, num_experts=_E,
        )
    message = str(excinfo.value)
    assert "--moe-cache-size 4400" in message
    assert "6 GPU-owned MoE layers" in message
    assert "lower --moe-cache-size to 1928 slots" in message
    assert "own at most 1 layer" in message
```

- [x] **Step 2: Run test to verify it fails**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine/test_cache_budget.py -q -p no:cacheprovider
```

Expected: 4 failures, each an
`ImportError: cannot import name 'gpu_owned_reservation_bytes' from 'freetoken.engine.cache_budget'`
(and `... name 'check_explicit_moe_cache_fits' ...` for the last one).

- [x] **Step 3: Write minimal implementation**

`python/freetoken/engine/cache_budget.py` -- replace `expert_bytes_per_slot` (lines 17-28) and add the two new functions after it:

```python
def expert_bytes_per_slot(
    sources: dict[str, "list[torch.Tensor]"],
    gpu_owned_layers: "frozenset[int]" = frozenset(),
) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size. ``gpu_owned_layers`` are permanently resident and never enter the slot
    cache, so the geometry is read from the first STREAMING layer (layer 0 is in
    the default owned set, and an owned layer's tensor lives on the device).
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[layer][0].numel() is the per-row element count (one expert slot); see the
    # matching slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4_paged_pool.py.
    first = next(i for i in range(len(next(iter(sources.values())))) if i not in gpu_owned_layers)
    return sum(t[first][0].numel() * t[first].element_size() for t in sources.values())


def gpu_owned_reservation_bytes(
    owned_layers: int, num_experts: int, per_expert_bytes: int
) -> int:
    """VRAM the GPU-owned MoE layers hold permanently: one full expert layer each.

    Accounted exactly like ``state_pool_bytes`` -- it joins ``fixed_cache_size`` before the
    MoE-vs-KV split, so the greedy slot fill never spends bytes the owned layers already own.
    """
    return owned_layers * num_experts * per_expert_bytes


def check_explicit_moe_cache_fits(
    *,
    moe_cache_size: int,
    per_expert_bytes: int,
    budget_bytes: int,
    owned_layers: int,
    num_experts: int,
) -> None:
    """Raise when an explicit ``--moe-cache-size`` plus the GPU-owned reservation exceeds the
    net MoE budget. Operator decision: fail loudly naming both sizes that would fit, never
    silently shrink the cache the operator asked for."""
    owned_bytes = gpu_owned_reservation_bytes(owned_layers, num_experts, per_expert_bytes)
    need = moe_cache_size * per_expert_bytes + owned_bytes
    if need <= budget_bytes:
        return
    fits_slots = max(0, (budget_bytes - owned_bytes) // per_expert_bytes)
    fits_owned = max(
        0, (budget_bytes - moe_cache_size * per_expert_bytes) // (num_experts * per_expert_bytes)
    )
    raise ValueError(
        f"--moe-cache-size {moe_cache_size} plus {owned_layers} GPU-owned MoE layers "
        f"({owned_bytes} B resident) needs {need} B of the {budget_bytes} B MoE budget. "
        f"Either lower --moe-cache-size to {fits_slots} slots at this owned set, or "
        f"own at most {fits_owned} layer(s) at this cache size."
    )
```

`python/freetoken/engine/engine.py` -- in `Engine.__init__`, beside `self.moe_offload_cache = None` (line 412):

```python
        self.moe_offload_cache = None
        # MoE layer ids resolved from --moe-gpu-owned-layers; read by the budget helpers
        # below and by the cache build. Empty until _init_offload_moe_cache resolves it.
        self._gpu_owned_layer_ids: frozenset = frozenset()
```

`python/freetoken/engine/engine.py` -- `_resolve_auto_moe_cache_size`, replacing lines 639-651:

```python
        from freetoken.engine.cache_budget import (
            expert_bytes_per_slot,
            gpu_owned_reservation_bytes,
            resolve_moe_cache_auto,
        )

        cache_per_page, fixed_cache_size, page_tokens, min_reserve = self._pool_cls.kv_cost(config)
        fixed_cache_size += state_pool_bytes(config)  # sibling GDN state pool, engine-summed
        num_experts = config.model_config.num_experts
        owned = self._gpu_owned_layer_ids
        per_expert_bytes = expert_bytes_per_slot(banks.sources, owned)
        # GPU-owned layers hold a full expert layer of VRAM forever and leave the slot cache
        # entirely, exactly as state_pool_bytes accounts for the GDN pool.
        fixed_cache_size += gpu_owned_reservation_bytes(len(owned), num_experts, per_expert_bytes)
        total_experts = (config.model_config.num_moe_layers - len(owned)) * num_experts
        moe_cache_size, num_pages, overlap = resolve_moe_cache_auto(
            baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes,
            memory_ratio=config.memory_ratio,
            cache_per_page=cache_per_page,
            fixed_cache_size=fixed_cache_size,
            per_expert_bytes=per_expert_bytes,
            num_experts=num_experts,
            total_experts=total_experts,
            prefill_overlap=config.moe_prefill_overlap,
            kv_reserve_tokens=max(config.kv_reserve_tokens, min_reserve),
            page_size=page_tokens,
            quant_format=banks.quant_format,
        )
```

`python/freetoken/engine/engine.py` -- new method immediately after `_resolve_auto_moe_cache_size` (after line 660):

```python
    def _check_gpu_owned_cache_fits(self, config: EngineConfig, banks) -> None:
        """Explicit --moe-cache-size + the owned reservation must fit the same budget the
        auto path solves against. Fails loudly (never shrinks) -- see spec section 6."""
        from freetoken.engine.cache_budget import (
            check_explicit_moe_cache_fits,
            expert_bytes_per_slot,
            net_cache_budget_bytes,
        )
        from freetoken.utils import div_ceil

        owned = self._gpu_owned_layer_ids
        if not owned or config.moe_cache_auto:
            return
        cache_per_page, fixed_cache_size, page_tokens, min_reserve = self._pool_cls.kv_cost(config)
        fixed_cache_size += state_pool_bytes(config)
        budget = net_cache_budget_bytes(
            config.memory_ratio, self._baseline_free, self._weights_bytes, fixed_cache_size
        )
        kv_reserve_pages = (
            div_ceil(max(config.kv_reserve_tokens, min_reserve), page_tokens) + 1
        )
        check_explicit_moe_cache_fits(
            moe_cache_size=config.moe_cache_size,
            per_expert_bytes=expert_bytes_per_slot(banks.sources, owned),
            budget_bytes=budget - kv_reserve_pages * cache_per_page,
            owned_layers=len(owned),
            num_experts=config.model_config.num_experts,
        )
```

`python/freetoken/engine/engine.py` -- `_target_moe_and_expert_bytes` (lines 912-915):

```python
        per_expert_bytes = (
            expert_bytes_per_slot(
                self.moe_offload_cache.bank_sources, self._gpu_owned_layer_ids
            )
            if self.moe_offload_cache is not None else 0
        )
```

- [x] **Step 4: Run test to verify it passes**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine/test_cache_budget.py -q -p no:cacheprovider
```

- [x] **Step 5: Commit**

```powershell
git add python/freetoken/engine/cache_budget.py python/freetoken/engine/engine.py tests/engine/test_cache_budget.py
git commit -m @'
feat(moe): reserve VRAM for GPU-owned MoE layers in the cache budget

expert_bytes_per_slot reads the first STREAMING layer; the owned layers join
fixed_cache_size and leave total_experts, so --moe-cache-auto shrinks by exactly
num_experts slots per owned layer. An explicit --moe-cache-size that no longer
fits now fails with the slot/layer counts that would.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hnf1bGBLU4HLq9uHPtNjwU
'@
```

### Task 3: OffloadMoeCache owned-layer registry, guards and reporting

**Files:**
- Modify `python/freetoken/moe/host_banks.py` (`HostResidency` enum at 52-60 -- the label only; the loader work is Task 5)
- Modify `python/freetoken/moe/offload_cache.py` (`__post_init__` 145-288; `set_bank_sources` 494-556; `_build_fused_copy_plan` 571-632; `rebuild` 651-723; predicates 756-763; `bank_views` 787-793; `prefetch_ready` 331-345; `prefetch_prefill_layer` 857-890; `ensure_experts` 1032-1042; `ensure_experts_hybrid` 1044-1063; `materialize_layer` 1065-1070; `copy_missing` 1236-1251; `decode_miss_stats_per_layer` 1171-1197; `decode_routing_stats` 1199-1234)
- Modify `tests/moe/test_offload.py` (append after `test_lock_failure_downgrades_echoed_residency`)

**Interfaces:**
- Consumes: `HostResidency`, `_BANK_SCHEMAS`, `Stat`, `cache.cpu_layer_ids`.
- Produces (all on `OffloadMoeCache`):
  - `HostResidency.GPU_OWNED` (`"gpu_owned"`)
  - `cache.gpu_owned_layer_ids: frozenset[int]` (set by the engine before `set_bank_sources`, like `cpu_layer_ids`)
  - `cache.resident_banks: dict[int, tuple[Tensor, ...]]`
  - `cache._first_streaming_layer: int`
  - `set_bank_sources(sources, layer_residency=None, gpu_owned_layers=None)` -- `None` means "use `self.gpu_owned_layer_ids`"
  - `is_gpu_owned_layer(layer_id) -> bool`
  - `resident_views(layer_id) -> tuple[Tensor, ...]`
  - `_note_decode_routing(layer_id, expert_ids) -> None`
  - `decode_miss_stats_per_layer()` rows carry `"resident": bool` and `"miss_rate": None` for owned layers

- [x] **Step 1: Write the failing test**

Append to `tests/moe/test_offload.py`:

```python
# ----------------------------------------------------- GPU-owned MoE layers (spec sections 5, 7)


def _make_owned_cache(num_layers=3, owned=(1,), prefill_overlap=False, head_shape=(32, 8)):
    """A [gate_up, down] bf16 cache with the given layers GPU_OWNED (the rest pinned).

    The "device" tensors are plain CPU tensors: an owned layer's whole point is that it never
    takes a device address and never enters a pointer table, so every assertion below is
    exercisable without CUDA.
    """
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=4, cache_size=8,
        device=torch.device("cpu"), prefill_overlap=prefill_overlap,
    )
    cache.gpu_owned_layer_ids = frozenset(owned)
    sources = {
        "gate_up": [
            torch.randn(4, *(head_shape if i in owned else (32, 8))) for i in range(num_layers)
        ],
        "down": [torch.randn(4, 8, 16) for _ in range(num_layers)],
    }
    residency = [
        HostResidency.GPU_OWNED.value if i in owned else HostResidency.PINNED.value
        for i in range(num_layers)
    ]
    cache.set_bank_sources(sources, layer_residency=residency)
    return cache, sources


def test_gpu_owned_layer_is_registered_as_resident_views_not_slot_cache_rows():
    cache, sources = _make_owned_cache(num_layers=3, owned=(1,))

    assert cache.is_gpu_owned_layer(1)
    assert not cache.is_gpu_owned_layer(0) and not cache.is_gpu_owned_layer(2)
    # views come back in bank REGISTRATION order and are the source tensors themselves
    views = cache.resident_views(1)
    assert views[0] is sources["gate_up"][1]
    assert views[1] is sources["down"][1]
    # an owned layer is NOT unpinned: it has a device address, it just has no host bank
    assert not cache.is_unpinned_layer(1)
    # the slot cache is unchanged: still cache_size rows of the streaming geometry
    assert cache.bank_caches["gate_up"].shape == (8, 32, 8)


def test_the_slot_cache_geometry_comes_from_the_first_streaming_layer():
    # layer 0 is owned (it is in the default owned set) and deliberately a different row
    # shape; the slot cache must be sized from layer 1.
    cache, _ = _make_owned_cache(num_layers=3, owned=(0,), head_shape=(99, 8))

    assert cache._first_streaming_layer == 1
    assert cache.bank_caches["gate_up"].shape == (8, 32, 8)


def test_set_bank_sources_rejects_a_layer_that_is_both_gpu_owned_and_a_cpu_layer():
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
    )
    cache.cpu_layer_ids = frozenset({1})
    cache.gpu_owned_layer_ids = frozenset({1})
    sources = {
        "gate_up": [torch.randn(4, 32, 8) for _ in range(2)],
        "down": [torch.randn(4, 8, 16) for _ in range(2)],
    }
    with pytest.raises(ValueError, match="both GPU-owned and CPU layers"):
        cache.set_bank_sources(
            sources,
            layer_residency=[HostResidency.PINNED.value, HostResidency.GPU_OWNED.value],
        )


def test_set_bank_sources_rejects_labels_that_disagree_with_the_owned_set():
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
    )
    sources = {
        "gate_up": [torch.randn(4, 32, 8) for _ in range(2)],
        "down": [torch.randn(4, 8, 16) for _ in range(2)],
    }
    with pytest.raises(ValueError, match="disagree with the GPU_OWNED residency labels"):
        cache.set_bank_sources(
            sources,
            layer_residency=[HostResidency.PINNED.value, HostResidency.GPU_OWNED.value],
            gpu_owned_layers=frozenset(),
        )


@pytest.mark.parametrize("call", ["ensure_experts", "ensure_experts_hybrid", "materialize_layer"])
def test_gpu_owned_layer_refuses_every_movement_entry_point(call):
    # a wiring bug that routed an owned layer into the LRU would gather the wrong rows;
    # mirror test_locked_layer_copy_missing_rejects_ensure_experts_staging and fail loudly
    cache, _ = _make_owned_cache(num_layers=3, owned=(1,))
    args = (1,) if call == "materialize_layer" else (1, torch.zeros(1, 2, dtype=torch.int32))

    with pytest.raises(RuntimeError, match="GPU-owned layer 1"):
        getattr(cache, call)(*args)


def test_gpu_owned_layer_refuses_copy_missing():
    cache, _ = _make_owned_cache(num_layers=3, owned=(1,))
    cache._pending_src_layer = 1
    cache._pending_whole_layer = True

    with pytest.raises(RuntimeError, match="GPU-owned layer 1"):
        cache.copy_missing()


def test_gpu_owned_layer_is_excluded_from_prefetch_and_the_prefill_double_buffer():
    cache, _ = _make_owned_cache(num_layers=3, owned=(1,), prefill_overlap=True)

    assert cache.prefetch_ready(1) is False  # no PCIe fetch to hide
    # the caller pre-issues layer_id + 1 blindly, so the guard is keyed on the TARGET and
    # returns quietly rather than raising
    cache.prefetch_prefill_layer(1)
    assert cache._prefill_buffer_layer == [None, None]
    cache.prefetch_prefill_layer(0)
    assert cache._prefill_buffer_layer == [0, None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_copy_plan_holds_a_zero_placeholder_for_gpu_owned_layers():
    # device_ptr() on a CUDA tensor returns data_ptr() -- a plausible but WRONG copy source;
    # the owned layer's descriptor row must stay the 0 placeholder
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    dev = torch.device("cuda")
    cache = OffloadMoeCache(num_layers=2, num_experts=4, cache_size=8, device=dev)
    cache.gpu_owned_layer_ids = frozenset({1})
    sources = {
        "gate_up": [torch.randn(4, 32, 8, device=dev) for _ in range(2)],
        "down": [torch.randn(4, 8, 16, device=dev) for _ in range(2)],
    }
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.PINNED.value, HostResidency.GPU_OWNED.value],
    )

    assert cache._copy_fused_ok
    assert (cache._copy_src_ptrs[1] == 0).all(), "GPU-owned layer row must stay 0"
    assert (cache._copy_src_ptrs[0] != 0).all()


def test_rebuild_preserves_resident_banks_and_resizes_only_the_slot_cache():
    cache, sources = _make_owned_cache(num_layers=3, owned=(0,), head_shape=(99, 8))
    before = cache.resident_banks[0]

    cache.rebuild(12)

    assert cache.resident_banks[0] is before
    assert cache.resident_views(0)[0] is sources["gate_up"][0]
    assert cache.bank_caches["gate_up"].shape == (12, 32, 8)  # from streaming layer 1


def test_per_layer_rows_report_gpu_owned_layers_as_resident_with_no_miss_rate():
    cache, _ = _make_owned_cache(num_layers=3, owned=(1,))
    cache.collect_stats = True
    cache.lru_stats[0, Stat.ACTIVE] = 10
    cache.lru_stats[0, Stat.MISS] = 2
    cache.lru_stats[0, Stat.CALLS] = 1

    rows = cache.decode_miss_stats_per_layer()["per_layer"]

    assert rows[0]["resident"] is False
    assert rows[0]["miss_rate"] == pytest.approx(0.2)
    # a resident layer never misses BY CONSTRUCTION; 0.0 would read as a perfect streaming
    # layer to any future heuristic, so it must be null
    assert rows[1]["resident"] is True
    assert rows[1]["miss_rate"] is None
    assert rows[1]["missing_per_step"] == 0.0
```

Add the `Stat` import that `tests/moe/test_offload.py` does not yet have -- put it at the top of the file beside the existing imports:

```python
from flashlib.kernels.slot_cache import Stat
```

- [x] **Step 2: Run test to verify it fails**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/moe/test_offload.py -q -p no:cacheprovider -k "gpu_owned or streaming or resident"
```

Expected failures:
- `test_set_bank_sources_rejects_labels_that_disagree_with_the_owned_set` ->
  `TypeError: OffloadMoeCache.set_bank_sources() got an unexpected keyword argument 'gpu_owned_layers'`
- every other new test -> `AssertionError: bank 'gate_up' layer 1 must be contiguous` is NOT reached; the
  first failure is `AttributeError: 'OffloadMoeCache' object has no attribute 'is_gpu_owned_layer'`
  (and, for `test_the_slot_cache_geometry_comes_from_the_first_streaming_layer` /
  `test_rebuild_preserves_resident_banks_and_resizes_only_the_slot_cache`, the earlier
  `AssertionError: ('gate_up', 1, torch.Size([4, 32, 8]), torch.float32)` from the head-shape
  check that still keys on layer 0).

- [x] **Step 3: Write minimal implementation**

`python/freetoken/moe/host_banks.py` -- extend the enum (after `PAGEABLE = "pageable"`, line 60):

```python
class HostResidency(str, Enum):
    """Residency class of a host bank layer.

    Only PINNED (cudaHostRegister'd) memory can feed the GPU movement paths; LOCKED (mlock'd, no device address) and PAGEABLE layers must decode on the CPU executor.
    The non-pinned classes exist for hosts that cap CUDA pin quota (WSL/WDDM: ~half of RAM).
    GPU_OWNED is the odd one out: there is no host bank at all -- the layer's experts live in
    VRAM for the process lifetime, so it neither spends pin quota nor holds host pages.
    """

    PINNED = "pinned"
    LOCKED = "locked"
    PAGEABLE = "pageable"
    GPU_OWNED = "gpu_owned"
```

`python/freetoken/moe/offload_cache.py` -- in `__post_init__`, after `self.cpu_layer_ids` (line 156):

```python
        self.cpu_layer_ids: frozenset = frozenset()
        # MoE layer ids whose full expert set is permanently VRAM-resident: no host bank, no
        # LRU slot, no copy plan row. Set by the engine BEFORE set_bank_sources (like
        # cpu_layer_ids), which validates it against the residency labels and stores the
        # per-layer device banks in resident_banks (schema order) for resident_views().
        self.gpu_owned_layer_ids: frozenset = frozenset()
        self.resident_banks: dict[int, tuple[torch.Tensor, ...]] = {}
        # Index of the lowest STREAMING layer: layer 0 is in the default owned set, so every
        # "read the head layer's geometry" site must use this instead (spec section 11).
        self._first_streaming_layer = 0
```

`python/freetoken/moe/offload_cache.py` -- `set_bank_sources` (replacing lines 494-556):

```python
    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
        gpu_owned_layers: "frozenset[int] | None" = None,
    ) -> None:
        """Attach the host (CPU pinned) expert source banks and allocate a GPU slot
        cache per bank, following the format's bank schema.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes); each slot cache mirrors the bank's row shape and dtype as one
        unified GPU pool. The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.moe.nvfp4_backends`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``layer_residency`` labels each layer with a ``HostResidency`` value (default: all pinned).
        Non-pinned (LOCKED/PAGEABLE) layers have no device address: they must already be routed to the CPU executor (``cpu_layer_ids``, set BEFORE this call), the copy plan skips their rows, and their only movement is ``copy_missing``'s whole-layer pageable prefill branch -- which is why prefill overlap is incompatible with them.

        ``gpu_owned_layers`` (``None`` = use ``self.gpu_owned_layer_ids``, set before this
        call like ``cpu_layer_ids``) are GPU_OWNED layers: their entries are already-resident
        device tensors, kept in ``resident_banks`` and read straight by the kernels. They
        never enter the slot cache, the copy plan, the prefill buffers or the LRU.
        """
        from freetoken.moe.host_banks import HostResidency

        assert set(sources) == set(self.bank_schema), (
            f"banks {sorted(sources)} do not match the {self.quant_format!r} "
            f"schema {self.bank_schema}"
        )
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        owned = (
            self.gpu_owned_layer_ids if gpu_owned_layers is None else frozenset(gpu_owned_layers)
        )
        labelled = frozenset(
            i for i, r in enumerate(residency) if r == HostResidency.GPU_OWNED.value
        )
        if owned != labelled:
            raise ValueError(
                f"gpu_owned_layers {sorted(owned)} disagree with the GPU_OWNED residency "
                f"labels {sorted(labelled)}: the loader and the cache must name the same set"
            )
        clash = sorted(owned & self.cpu_layer_ids)
        if clash:
            raise ValueError(
                f"layers {clash} are both GPU-owned and CPU layers: a VRAM-resident layer "
                f"has no host bank for the CPU executor to read"
            )
        if owned and len(owned) == self.num_layers:
            raise ValueError(
                "every MoE layer is GPU-owned; there is no streaming layer left to size the "
                "slot cache from (drop --moe-gpu-owned-layers or own fewer layers)"
            )
        self.gpu_owned_layer_ids = owned
        self._first_streaming_layer = min(set(range(self.num_layers)) - owned)
        unpinned = frozenset(
            i for i, r in enumerate(residency)
            if r not in (HostResidency.PINNED.value, HostResidency.GPU_OWNED.value)
        )
        if unpinned:
            if not unpinned <= self.cpu_layer_ids:
                raise ValueError(
                    f"non-pinned layers {sorted(unpinned - self.cpu_layer_ids)} are not in "
                    f"cpu_layer_ids: a layer without a device address can only decode on "
                    f"the CPU executor (set cache.cpu_layer_ids before set_bank_sources)"
                )
            if self.prefill_overlap:
                raise ValueError(
                    "prefill overlap DMAs from registered banks; it must be disabled "
                    "when any layer is LOCKED/PAGEABLE (the engine does this)"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[self._first_streaming_layer]
            for layer_id, source in enumerate(per_layer):
                assert source.size(0) == self.num_experts, (name, layer_id, source.shape)
                if layer_id in owned:
                    # a resident layer's rows never enter the slot cache or a pointer table:
                    # the kernel indexes them directly, and the loader owns their geometry
                    continue
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                assert source.shape == head.shape and source.dtype == head.dtype, (
                    name, layer_id, source.shape, source.dtype,
                )
            self.bank_sources[name] = list(per_layer)
            self.bank_caches[name] = torch.empty(
                (self.cache_size, *head.shape[1:]),
                dtype=head.dtype,
                device=self.device,
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self.resident_banks = {
            layer_id: tuple(self.bank_sources[n][layer_id] for n in self.bank_schema)
            for layer_id in sorted(owned)
        }
        self._build_copy_plan()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()
```

`python/freetoken/moe/offload_cache.py` -- `_build_fused_copy_plan`, the per-layer loop (line 600):

```python
            for layer_id, source in enumerate(per_layer):
                if layer_id in self._unpinned_layers or layer_id in self.gpu_owned_layer_ids:
                    # unregistered layer: no device alias exists, and the row is never consumed (CPU decode; pageable prefill)
                    # GPU-owned layer: device_ptr() WOULD succeed on its CUDA tensor
                    # (kernel/pinned.py:66) and hand the kernel a plausible but wrong source
                    # a 0 placeholder keeps the descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
```

`python/freetoken/moe/offload_cache.py` -- `rebuild`, the reallocation loop (line 680):

```python
        for name in self.bank_schema:
            head = self.bank_sources[name][self._first_streaming_layer]
```

(and immediately before the `for name in self.bank_schema:` loop, a comment:)

```python
        # 3. Reallocate the slot cache from the retained host sources. resident_banks are NOT
        #    slot-cache rows -- they survive a rebuild untouched, and _build_copy_plan below
        #    keeps skipping them.
```

`python/freetoken/moe/offload_cache.py` -- new predicates and views, after `is_unpinned_layer` (line 763):

```python
    def is_gpu_owned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s full expert set is permanently VRAM-resident.

        Such a layer has no host bank, no slot-cache rows and no LRU bookkeeping: the kernels
        read ``resident_views(layer_id)`` with the RAW expert ids."""
        return layer_id in self.gpu_owned_layer_ids

    def _reject_gpu_owned(self, layer_id: int, what: str) -> None:
        if layer_id in self.gpu_owned_layer_ids:
            raise RuntimeError(
                f"{what} was called for GPU-owned layer {layer_id}: its experts are already "
                f"resident, so there is no host bank to fetch from and no slot to fill "
                f"(the forward path must use resident_views)"
            )
```

after `bank_views` (line 793):

```python
    def resident_views(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """A GPU-owned layer's device banks in registration order.

        Position == expert id (the whole layer is resident), so routing ids pass through
        unmapped -- exactly the contract the prefill double buffer offers, without the copy."""
        views = self.resident_banks.get(layer_id)
        assert views is not None, f"layer {layer_id} is not GPU-owned"
        return views
```

`python/freetoken/moe/offload_cache.py` -- `prefetch_ready` (add one clause, line 342):

```python
            and target_layer not in self.cpu_layer_ids
            and target_layer not in self.gpu_owned_layer_ids
            and target_layer not in self._unpinned_layers
```

`python/freetoken/moe/offload_cache.py` -- `prefetch_prefill_layer` (line 858):

```python
    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")
        if layer_id in self.gpu_owned_layer_ids:
            # keyed on the TARGET: _wait_prefill_overlap pre-issues layer_id + 1 blindly, so
            # an owned layer must be a quiet no-op here, not an error. It borrows no buffer,
            # so there is nothing to invalidate, wait on or release either.
            return
```

`python/freetoken/moe/offload_cache.py` -- the movement guards and the routing-histogram helper (replacing lines 1032-1070):

```python
    def _note_decode_routing(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Accumulate the raw per-(layer, expert) routing histogram for ``layer_id``.

        Called from BOTH the streaming path (before ``ensure_experts``' kernel rewrites the
        ids to slots in place) and the GPU-owned decode branch (which never rewrites them),
        so ``/v1/cache/routing`` still counts owned layers. Graph-safe: a device-side
        scatter_add_ over fixed shapes, captured and replayed like any other decode op."""
        if not self.collect_decode_freq:
            return
        ids = expert_ids.reshape(-1).long()
        self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        from freetoken.moe.offload_kernels import ensure_experts

        self._reject_gpu_owned(layer_id, "ensure_experts")
        # ``expert_ids`` still holds raw expert ids here (the kernel rewrites them to
        # slot ids in place), so snapshot the routing histogram before that happens.
        self._note_decode_routing(layer_id, expert_ids)
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts(self, layer_id, expert_ids)

    def ensure_experts_hybrid(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        self._reject_gpu_owned(layer_id, "ensure_experts_hybrid")
        self._note_decode_routing(layer_id, expert_ids)
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts_hybrid(
            self, layer_id, expert_ids, self.hybrid_max_fetch, self.hybrid_fetch_fraction
        )

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._reject_gpu_owned(layer_id, "materialize_layer")
        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)
```

`python/freetoken/moe/offload_cache.py` -- `copy_missing` (after the `assert layer_id is not None`, line 1239):

```python
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        self._reject_gpu_owned(layer_id, "copy_missing")
```

`python/freetoken/moe/offload_cache.py` -- `decode_miss_stats_per_layer`, the row loop (replacing lines 1186-1197):

```python
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            resident = L in self.gpu_owned_layer_ids
            per_layer.append({
                "layer": L,
                "steps": s,
                # A GPU-owned layer never misses BY CONSTRUCTION. Reporting 0.0 would read
                # as a perfectly cacheable STREAMING layer to any later heuristic, so the
                # rate is null and the row says why.
                "resident": resident,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": 0.0 if resident else ((m / s) if s else 0.0),
                "miss_rate": None if resident else ((m / a) if a else 0.0),
                "fetched_per_step": 0.0 if resident else ((f / s) if s else 0.0),
            })
        return {"per_layer": per_layer}
```

`python/freetoken/moe/offload_cache.py` -- `decode_routing_stats` (replacing lines 1211-1219):

```python
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if self.gpu_owned_layer_ids:
            # the summary describes the STREAMING cache: a resident layer's oracle hit is 1.0
            # by construction and would bias the very number used to size the slot cache.
            # Its raw counts stay in decode_freq for offline study.
            owned = torch.zeros_like(valid)
            owned[sorted(self.gpu_owned_layer_ids)] = True
            valid = valid & ~owned
        if int(valid.sum()) == 0:
            return prefetch
        streaming_layers = self.num_layers - len(self.gpu_owned_layer_ids)
        slots_per_layer = self.cache_size / max(1, streaming_layers)
        C = max(1, int(round(slots_per_layer)))
```

- [x] **Step 4: Run test to verify it passes**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/moe/test_offload.py tests/moe/test_routing_stats.py tests/moe/test_hybrid_fetch.py tests/moe/test_prefill_hit_d2d.py -q -p no:cacheprovider
```

- [x] **Step 5: Commit**

```powershell
git add python/freetoken/moe/host_banks.py python/freetoken/moe/offload_cache.py tests/moe/test_offload.py
git commit -m @'
feat(moe): GPU-owned layer registry in OffloadMoeCache

HostResidency.GPU_OWNED, gpu_owned_layer_ids/resident_banks/resident_views, the
first-streaming-layer head for the slot-cache geometry and rebuild, a 0 copy-plan
placeholder (never device_ptr on a CUDA source), loud refusals in every movement
entry point, the lifted decode-histogram helper, and honest per-layer reporting
(resident: true, miss_rate: null).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hnf1bGBLU4HLq9uHPtNjwU
'@
```

### Task 4: OffloadMoELayer forward paths for owned layers

**Files:**
- Modify `python/freetoken/layers/moe.py` (`_decode_routed` 693-738; `_prefill_routed` 818-855)
- Modify `tests/moe/test_offload.py` (append after the Task 3 block)

**Interfaces:**
- Consumes: `cache.is_gpu_owned_layer`, `cache.resident_views`, `cache.alphas_for_layer`, `cache.prefetch_prefill_layer`, `cache.begin_prefill`, `self._expert_gemm(..., views=, n=, alphas=, is_prefill=)`.
- Produces: no new public names. The owned branch of `_decode_routed` is the SINGLE entry point both real decode and the `FREETOKEN_MOE_SMALL_PREFILL_ROWS` narrow-prefill path reach (`forward`/`routed_forward` -> `_use_decode_movement` -> `_decode_routed`), so one branch covers both. `_wait_prefill_overlap` needs no owned branch: it is only reachable from `_prefill_routed`, which returns before it.

- [x] **Step 1: Write the failing test**

Append to `tests/moe/test_offload.py`:

```python
def _make_owned_layer_and_cache(layer_id=1, num_layers=3, owned=(1,), prefill_overlap=False):
    from freetoken.layers.moe import OffloadMoELayer

    cache, sources = _make_owned_cache(
        num_layers=num_layers, owned=owned, prefill_overlap=prefill_overlap
    )
    layer = OffloadMoELayer(
        layer_id=layer_id, num_experts=4, top_k=2, hidden_size=8, intermediate_size=16
    )
    layer.offload_cache = cache
    return layer, cache, sources


def test_decode_forward_on_a_gpu_owned_layer_uses_raw_ids_and_resident_views(monkeypatch):
    layer, cache, sources = _make_owned_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    calls = {}

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("a GPU-owned layer must not touch the LRU")

    monkeypatch.setattr(cache, "ensure_experts", unexpected)
    monkeypatch.setattr(cache, "copy_missing", unexpected)
    monkeypatch.setattr(cache, "materialize_layer", unexpected)

    def fake_fused_decode(
        hidden_states, w1, w2, got_topk_weights, got_topk_ids, activation,
        apply_router_weight_on_input,
    ):
        calls["w1"] = w1
        calls["w2"] = w2
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("freetoken.layers.moe.fused_experts_decode_impl", fake_fused_decode)

    out = layer.decode_forward(hidden_states, router_logits)

    assert out is hidden_states
    # position == expert id on a fully resident layer: the ids pass through UNMAPPED
    assert calls["topk_ids"].tolist() == [[2, 1]]
    assert calls["w1"] is sources["gate_up"][1]
    assert calls["w2"] is sources["down"][1]
    assert calls["w1"] is not cache.bank_caches["gate_up"]
    # and the routing histogram still counts the layer
    cache.collect_decode_freq = True
    layer.decode_forward(hidden_states, router_logits)
    assert cache.decode_freq[1].tolist() == [0, 1, 1, 0]


def test_a_narrow_prefill_on_a_gpu_owned_layer_takes_the_same_owned_branch(monkeypatch):
    # FREETOKEN_MOE_SMALL_PREFILL_ROWS routes narrow prefills through _decode_routed; the
    # owned branch has to be correct there too. (The admission conditions themselves are
    # covered by tests/moe/test_small_prefill_movement.py.)
    layer, cache, sources = _make_owned_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    calls = {}

    monkeypatch.setattr(layer, "_use_decode_movement", lambda _hidden: True)
    monkeypatch.setattr(cache, "materialize_layer", lambda *a, **k: pytest.fail("streamed"))

    def fake_fused_decode(
        hidden_states, w1, w2, got_topk_weights, got_topk_ids, activation,
        apply_router_weight_on_input,
    ):
        calls["w1"] = w1
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("freetoken.layers.moe.fused_experts_decode_impl", fake_fused_decode)

    layer.routed_forward(hidden_states, topk_weights, topk_ids)

    assert calls["w1"] is sources["gate_up"][1]
    assert calls["topk_ids"].tolist() == [[2, 1]]


def test_prefill_overlap_skips_a_gpu_owned_layer_and_still_alternates_buffers(monkeypatch):
    # layers 0 and 2 stream (both land in buffer 0, layer_id % 2), layer 1 is resident.
    # The owned layer borrows no buffer but must still pre-issue layer 2's copy, or the
    # pipeline stalls one layer every time an owned layer sits in the middle.
    from freetoken.layers.moe import OffloadMoELayer

    cache, sources = _make_owned_cache(num_layers=3, owned=(1,), prefill_overlap=True)
    layers = [
        OffloadMoELayer(
            layer_id=layer_id, num_experts=4, top_k=2, hidden_size=8, intermediate_size=16
        )
        for layer_id in range(3)
    ]
    for layer in layers:
        layer.offload_cache = cache

    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    fused_calls = []

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (
            topk_weights, topk_ids.clone()
        ),
    )

    def fake_fused(
        hidden_states, w1, w2, got_topk_weights, got_topk_ids, activation,
        apply_router_weight_on_input,
    ):
        fused_calls.append({"w1_ptr": w1.data_ptr(), "w1": w1.clone(), "ids": got_topk_ids.clone()})
        return hidden_states

    monkeypatch.setattr("freetoken.layers.moe.fused_experts_impl", fake_fused)

    out = hidden_states
    for layer in layers:
        out = layer.prefill_forward(out, router_logits)

    # every layer computed against its OWN weights, ids unmapped throughout
    for layer_id in range(3):
        assert torch.equal(fused_calls[layer_id]["w1"], sources["gate_up"][layer_id])
        assert fused_calls[layer_id]["ids"].tolist() == [[2, 1]]
    # layers 0 and 2 share double buffer 0 (0 % 2 == 2 % 2), the owned layer reads its
    # resident bank and never enters the buffers at all
    assert fused_calls[0]["w1_ptr"] == fused_calls[2]["w1_ptr"]
    assert fused_calls[1]["w1_ptr"] == sources["gate_up"][1].data_ptr()
    assert cache._prefill_buffer_layer == [2, None]  # buffer 1 was never claimed
```

- [x] **Step 2: Run test to verify it fails**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/moe/test_offload.py -q -p no:cacheprovider -k "owned_layer or narrow_prefill or alternates_buffers"
```

Expected: `test_decode_forward_on_a_gpu_owned_layer_uses_raw_ids_and_resident_views` and
`test_a_narrow_prefill_on_a_gpu_owned_layer_takes_the_same_owned_branch` fail with
`AssertionError: a GPU-owned layer must not touch the LRU` (the unpatched `_decode_routed`
still calls `ensure_experts`); `test_prefill_overlap_skips_a_gpu_owned_layer_and_still_alternates_buffers`
fails with `AssertionError: assert 139... == 139...` on
`fused_calls[1]["w1_ptr"] == sources["gate_up"][1].data_ptr()` (layer 1 was copied into
double buffer 1 instead of read in place).

- [x] **Step 3: Write minimal implementation**

`python/freetoken/layers/moe.py` -- `_decode_routed`, inserting the owned branch after the CPU-layer branch (line 714) and before the hybrid branch:

```python
        cache = self.offload_cache
        assert cache is not None
        if cache.is_cpu_layer(self.layer_id):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            return executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
        if cache.is_gpu_owned_layer(self.layer_id):
            # Every expert of this layer is already in VRAM at position == expert id, so
            # there is nothing to predict, fetch, evict or remap: hand the kernel the RAW
            # topk_ids and the resident banks. Fixed shapes over fixed addresses -- strictly
            # simpler than the streaming path, so CUDA-graph capture is unaffected.
            cache._note_decode_routing(self.layer_id, topk_ids)
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.resident_views(self.layer_id),
                n=None,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=False,
            )
        if cache.decode_target == "hybrid":
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids)
```

Also extend the method docstring's final paragraph (line 704-708) with one sentence:

```python
        For a GPU-owned layer (``--moe-gpu-owned-layers``) the experts are already resident
        at position == expert id, so the ids pass through unmapped and no LRU state is
        touched; ``alphas_for_layer`` is the matching (position == expert id) scale lookup.
```

`python/freetoken/layers/moe.py` -- `_prefill_routed`, inserting the owned branch right after the `assert cache is not None` (line 829):

```python
        cache = self.offload_cache
        assert cache is not None
        if cache.is_gpu_owned_layer(self.layer_id):
            # No overlap buffer, no materialize, no release: the layer is already resident.
            # Keep the double-buffer pipeline moving anyway -- the next streaming layer's
            # copy still has to start one layer early, and prefetch_prefill_layer is a quiet
            # no-op for an owned target.
            if cache.prefill_overlap:
                if self.layer_id == 0:
                    cache.begin_prefill()
                cache.prefetch_prefill_layer(self.layer_id + 1)
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.resident_views(self.layer_id),
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
        if cache.prefill_overlap:
```

- [x] **Step 4: Run test to verify it passes**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/moe/test_offload.py tests/moe/test_small_prefill_movement.py tests/moe/test_moe_prefetch_config.py -q -p no:cacheprovider
```

- [x] **Step 5: Commit**

```powershell
git add python/freetoken/layers/moe.py tests/moe/test_offload.py
git commit -m @'
feat(moe): forward path for GPU-owned MoE layers

_decode_routed and _prefill_routed read resident_views with raw topk_ids for an
owned layer: no ensure/copy/materialize, no overlap buffer, no release. The owned
prefill branch still pre-issues layer+1 so the double buffer keeps alternating
across a resident layer in the middle. The narrow-prefill knob shares the decode
branch, so one implementation covers both.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hnf1bGBLU4HLq9uHPtNjwU
'@
```

### Task 5: Loader -- device banks, reusable pinned staging, refusals

**Files:**
- Modify `python/freetoken/moe/host_banks.py` (`_ResidencyPlan` 227-248; `requested_residency` 253-265; `_settle` 268-274; `pin_banks` 276-291; `PinPipeline` 294-356; `alloc_layer_banks` 215-224; `HostBank` 83-177 -- add the `fill` property; two new classes)
- Modify `python/freetoken/models/nvfp4_banks.py` (`_alloc_nvfp4_host_banks` 88-103; serial loader 177-242; parallel loader 301-361)
- Modify `python/freetoken/moe/expert_banks.py` (`bank_bytes_estimate` 403-419; `load_expert_banks` 497; `_echo_residency` 510-534)
- Modify `python/freetoken/checkpoint/ftw.py` (`load_ftw_banks` 453-455)
- Modify `python/freetoken/moe/cpu_executor.py` (`_resolve_banks` 336-345)
- Create `tests/moe/test_gpu_owned_banks.py`

**Interfaces:**
- Consumes: `HostResidency.GPU_OWNED` (Task 3), `HostBank`, `LayerCompletionTracker`, `PinPipeline`, `_requested_residency`.
- Produces:
  - `HostBank.fill -> torch.Tensor` (property; `== self.tensor`)
  - `GpuOwnedBank(tensor, pool, layer_id, name)` with `.tensor`, `.fill`, `.pool`, `.residency`
  - `GpuOwnedStagingPool(specs, *, cap=2, device=None)` with `.fill_view(layer_id, name)`, `.flush(layer_id, device_banks)`
  - `plan_gpu_owned() -> tuple[frozenset[int], torch.device | None]`
  - `alloc_layer_banks(specs, num_layers, *, gpu_owned=None, device=None)`
  - `requested_residency(labels, device=None)`; `_ResidencyPlan(labels, device=None)` with `.gpu_owned`, `.device`
  - `PinPipeline.submit_flush(fn)`
  - `bank_bytes_estimate(model_config, gpu_owned: int = 0)`
  - `freetoken.moe.cpu_executor._reject_cuda_sources(banks) -> None`

- [x] **Step 1: Write the failing test**

Create `tests/moe/test_gpu_owned_banks.py`:

```python
"""GPU-owned MoE layers, loader half (--moe-gpu-owned-layers).

CPU-only. The "device" is ``torch.device("cpu")``, so the staging -> destination copy, the
two-slot staging pool with back-pressure, the layer-completion flush and every refusal are
exercised without CUDA.

What is NOT covered here and only a live GPU run proves (see the operator checklist,
docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-status.md): the cudaHostAlloc'd staging
banks, the ``copy_(non_blocking=True)`` H2D itself, the CUDA event that gates staging reuse,
and that the resident VRAM rows are byte-identical to a normal host-bank load.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

import freetoken.moe.host_banks as hb

_SPECS = {
    "gate_up": ((2, 4), torch.float32),
    "down": ((2, 3), torch.float32),
}
CPU = torch.device("cpu")


def test_alloc_layer_banks_puts_owned_layers_on_the_device_and_allocates_no_host_bank():
    banks = hb.alloc_layer_banks(_SPECS, 3, gpu_owned=frozenset({1}), device=CPU)

    for name, (shape, dtype) in _SPECS.items():
        assert isinstance(banks[name][0], hb.HostBank)
        assert isinstance(banks[name][2], hb.HostBank)
        owned = banks[name][1]
        assert isinstance(owned, hb.GpuOwnedBank)
        assert owned.tensor.shape == shape and owned.tensor.dtype == dtype
        assert owned.tensor.device == CPU
        assert owned.residency is hb.HostResidency.GPU_OWNED
        # the per-layer list keeps length num_layers so every consumer still sees one entry
        assert len(banks[name]) == 3
    # a plain host bank writes where it lives; an owned bank writes into shared staging
    assert banks["gate_up"][0].fill is banks["gate_up"][0].tensor
    assert banks["gate_up"][1].fill.data_ptr() != banks["gate_up"][1].tensor.data_ptr()


def test_alloc_layer_banks_reads_the_owned_set_from_the_ambient_plan():
    labels = [
        hb.HostResidency.PINNED.value,
        hb.HostResidency.GPU_OWNED.value,
    ]
    with hb.requested_residency(labels, device=CPU):
        banks = hb.alloc_layer_banks(_SPECS, 2)

    assert isinstance(banks["gate_up"][0], hb.HostBank)
    assert isinstance(banks["gate_up"][1], hb.GpuOwnedBank)


def test_the_staging_pool_hands_out_two_layers_and_back_pressures_the_third():
    # the parallel reader interleaves layers, so the pool must bound in-flight staging
    # (spec section 4.2 choice (a): cap 2) instead of serializing the read
    pool = hb.GpuOwnedStagingPool(_SPECS, cap=2, device=CPU)
    first = pool.fill_view(0, "gate_up")
    second = pool.fill_view(1, "gate_up")

    assert first.data_ptr() != second.data_ptr()
    assert pool.fill_view(0, "gate_up").data_ptr() == first.data_ptr()  # sticky per layer

    started, got = threading.Event(), []

    def third() -> None:
        started.set()
        got.append(pool.fill_view(2, "gate_up").data_ptr())

    worker = threading.Thread(target=third, daemon=True)
    worker.start()
    started.wait(timeout=5)
    worker.join(timeout=0.5)
    assert worker.is_alive(), "a third owned layer must wait for a staging slot"

    pool.flush(0, {"gate_up": torch.zeros(2, 4), "down": torch.zeros(2, 3)})
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert got == [first.data_ptr()], "the freed slot must be recycled, not a third allocation"


def test_the_layer_completion_sink_flushes_staging_into_the_device_tensor():
    banks = hb.alloc_layer_banks(_SPECS, 2, gpu_owned=frozenset({1}), device=CPU)
    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]
    payload = {
        "gate_up": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "down": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    for name, value in payload.items():
        banks[name][1].fill.copy_(value)
    staging_ptr = banks["gate_up"][1].fill.data_ptr()

    with hb.requested_residency(labels, device=CPU):
        with hb.PinPipeline() as pins:
            pins(1, {name: per[1] for name, per in banks.items()})

    for name, value in payload.items():
        assert torch.equal(banks[name][1].tensor, value)
        assert banks[name][1].tensor.data_ptr() != staging_ptr
    # the staging slot came back to the pool, so the next owned layer reuses it
    pool = banks["gate_up"][1].pool
    assert pool.fill_view(7, "gate_up").data_ptr() == staging_ptr


def test_a_gpu_owned_layer_at_the_plain_settle_path_fails_loudly():
    # a provider without a per-layer completion sink would leave the device banks unfilled
    banks = hb.alloc_layer_banks(_SPECS, 2, gpu_owned=frozenset({1}), device=CPU)
    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]

    with hb.requested_residency(labels, device=CPU):
        with pytest.raises(RuntimeError, match="no per-layer completion sink"):
            hb.pin_banks(banks)


def test_bank_bytes_estimate_subtracts_gpu_owned_layers():
    from freetoken.moe.expert_banks import bank_bytes_estimate

    config = SimpleNamespace(
        expert_quant="nvfp4", moe_weight_format=None,
        num_moe_layers=48, num_experts=512, hidden_size=4096, moe_intermediate_size=512,
    )
    full = bank_bytes_estimate(config)
    owned6 = bank_bytes_estimate(config, gpu_owned=6)

    assert full is not None
    assert owned6 == full * 42 // 48


def test_echo_residency_refuses_an_unapplied_gpu_owned_request():
    from freetoken.moe.expert_banks import ExpertBanks, _echo_residency

    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]
    banks = ExpertBanks("bf16", {"gate_up": [], "down": []})
    stale = hb._ResidencyPlan(labels)  # never consulted -> the layers got host banks

    with pytest.raises(RuntimeError, match="--moe-gpu-owned-layers"):
        _echo_residency(banks, labels, stale)


def test_ftw_bank_loader_refuses_gpu_owned_layers(tmp_path):
    from freetoken.checkpoint.ftw import load_ftw_banks

    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]
    with pytest.raises(ValueError, match="FTW packed checkpoint"):
        load_ftw_banks(str(tmp_path), num_layers=2, layer_residency=labels)


def test_cpu_moe_executor_refuses_cuda_bank_sources():
    # _make_table hands C++ raw data_ptr()s it dereferences on the CPU; a CUDA source
    # (a GPU-owned layer) would be a silent wrong-memory read
    from freetoken.moe.cpu_executor import _reject_cuda_sources

    ok = {"gate_up": [SimpleNamespace(is_cuda=False)], "down": [SimpleNamespace(is_cuda=False)]}
    _reject_cuda_sources(ok)

    bad = {
        "gate_up": [SimpleNamespace(is_cuda=False), SimpleNamespace(is_cuda=True)],
        "down": [SimpleNamespace(is_cuda=False), SimpleNamespace(is_cuda=False)],
    }
    with pytest.raises(ValueError, match=r"bank 'gate_up' layer 1"):
        _reject_cuda_sources(bad)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
```

- [x] **Step 2: Run test to verify it fails**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/moe/test_gpu_owned_banks.py -q -p no:cacheprovider
```

Expected: 9 failures. The first is
`TypeError: alloc_layer_banks() got an unexpected keyword argument 'gpu_owned'`; the pool
tests fail with `AttributeError: module 'freetoken.moe.host_banks' has no attribute 'GpuOwnedStagingPool'`;
the executor test with `ImportError: cannot import name '_reject_cuda_sources' from 'freetoken.moe.cpu_executor'`.

- [x] **Step 3: Write minimal implementation**

`python/freetoken/moe/host_banks.py` -- add the `fill` property to `HostBank` (after the `residency` property, line 131):

```python
    @property
    def fill(self) -> torch.Tensor:
        """Where a loader writes this bank's rows.

        Identical to :attr:`tensor` for a host bank; :class:`GpuOwnedBank` redirects it to a
        shared pinned staging layer, so the assignment loops read the same attribute either
        way and stay one code path."""
        return self.tensor
```

`python/freetoken/moe/host_banks.py` -- new classes, immediately before `alloc_banks` (line 210):

```python
class GpuOwnedStagingPool:
    """Reusable pinned staging layers for GPU-owned MoE layers.

    A GPU-owned layer has no host bank, but the shard readers fill it one expert tensor at a
    time (``num_experts * 6`` separate assignments for NVFP4), and writing those straight to
    VRAM would be that many small pageable H2D copies. So the fill lands in a pinned staging
    LAYER and the layer-completion sink issues one ``copy_(non_blocking=True)`` per bank kind.

    The parallel reader may deliver several owned layers' rows interleaved, so up to ``cap``
    staging layers are handed out at once and a further request blocks (back-pressure) until
    one is returned -- choice (a) of spec section 4.2, cap 2: boot pays at most 2 x one layer
    of extra host RAM (2.6 GiB for Qwen3.8), released after the last owned layer. A slot is
    reusable only once the CUDA event recording its staging -> device copy has completed.

    Staging banks are born pinned (``backing="cuda"``) on a CUDA device: they are written
    many times, so the pin-after-fill trick does not apply and a plain mmap could not feed an
    async copy. Allocation happens under the pool lock, at most ``cap`` times per load.
    """

    def __init__(
        self,
        specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
        *,
        cap: int = 2,
        device: "torch.device | None" = None,
    ) -> None:
        self.specs = dict(specs)
        self.cap = cap
        self.device = device
        self._free: list[tuple[dict[str, HostBank], object]] = []
        self._inflight: dict[int, dict[str, HostBank]] = {}
        self._live = 0
        self._cv = threading.Condition()

    def fill_view(self, layer_id: int, name: str) -> torch.Tensor:
        """The staging tensor ``layer_id``'s ``name`` bank is filled through, acquiring a
        staging layer on first touch. Thread-safe: the parallel reader writes one layer from
        many threads."""
        with self._cv:
            staging = self._inflight.get(layer_id)
            if staging is None:
                staging = self._acquire_locked()
                self._inflight[layer_id] = staging
        return staging[name].tensor

    def _acquire_locked(self) -> dict[str, HostBank]:
        while not self._free and self._live >= self.cap:
            self._cv.wait()
        if self._free:
            staging, event = self._free.pop()
            if event is not None:
                event.synchronize()  # its previous layer's H2D must land before we overwrite
            return staging
        self._live += 1
        backing = "cuda" if (self.device is not None and self.device.type == "cuda") else "mmap"
        return {
            name: HostBank(shape, dtype, backing=backing)
            for name, (shape, dtype) in self.specs.items()
        }

    def flush(self, layer_id: int, device_banks: dict[str, torch.Tensor]) -> None:
        """Copy the completed staging layer into its device banks and return the slot."""
        with self._cv:
            staging = self._inflight.pop(layer_id, None)
        assert staging is not None, f"GPU-owned layer {layer_id} never acquired staging"
        for name, dst in device_banks.items():
            dst.copy_(staging[name].tensor, non_blocking=True)
        event = None
        if self.device is not None and self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record()
        with self._cv:
            self._free.append((staging, event))
            self._cv.notify()


class GpuOwnedBank:
    """One bank kind of a GPU-owned MoE layer: the device tensor consumers read, plus the
    reusable pinned staging tensor the loader fills through.

    Duck-types the two attributes the loaders touch on a :class:`HostBank` -- ``tensor``
    (what the bank IS, here already on the device) and ``fill`` (where to write). There are
    no host pages, so ``pin``/``lock``/``release`` have no meaning: the layer-completion sink
    flushes staging into ``tensor`` instead of settling (see :meth:`PinPipeline.__call__`),
    and :func:`_settle` refuses this label outright.
    """

    __slots__ = ("tensor", "pool", "layer_id", "name")

    def __init__(
        self, tensor: torch.Tensor, pool: GpuOwnedStagingPool, layer_id: int, name: str
    ) -> None:
        self.tensor = tensor
        self.pool = pool
        self.layer_id = layer_id
        self.name = name

    @property
    def fill(self) -> torch.Tensor:
        return self.pool.fill_view(self.layer_id, self.name)

    @property
    def residency(self) -> HostResidency:
        return HostResidency.GPU_OWNED
```

`python/freetoken/moe/host_banks.py` -- replace `alloc_layer_banks` (lines 215-224):

```python
def alloc_layer_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    num_layers: int,
    *,
    gpu_owned: "frozenset[int] | None" = None,
    device: "torch.device | None" = None,
) -> dict[str, list]:
    """Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}``
    -> one independently allocated (page-aligned, independently pin/lock-able)
    ``HostBank`` per layer per name.

    ``gpu_owned`` layer ids get NO host bank at all: their entry is a :class:`GpuOwnedBank`
    holding the ``[num_experts, ...]`` tensor on ``device`` plus the shared
    :class:`GpuOwnedStagingPool` it is filled through. The per-layer list keeps length
    ``num_layers`` and every entry still has ``size(0) == num_experts``, so downstream
    consumers are unchanged. ``None`` (the default) reads the set and the device from the
    ambient :func:`requested_residency` plan, so every provider honors
    ``--moe-gpu-owned-layers`` without a new parameter; pass ``frozenset()`` to force plain
    host banks.
    """
    if gpu_owned is None:
        gpu_owned, device = plan_gpu_owned()
    pool = GpuOwnedStagingPool(specs, device=device) if gpu_owned else None
    banks: dict[str, list] = {name: [] for name in specs}
    for layer_id in range(num_layers):
        for name, (shape, dtype) in specs.items():
            if layer_id in gpu_owned:
                banks[name].append(
                    GpuOwnedBank(
                        torch.empty(shape, dtype=dtype, device=device), pool, layer_id, name
                    )
                )
            else:
                banks[name].append(HostBank(shape, dtype))
    return banks
```

`python/freetoken/moe/host_banks.py` -- `_ResidencyPlan` (lines 232-248) and `requested_residency` (253-265) and the new accessor:

```python
    __slots__ = ("labels", "applied", "has_unpinned", "actual", "gpu_owned", "device")

    def __init__(self, labels: list[str], device=None):
        self.labels = list(labels)
        self.applied = False
        self.has_unpinned = any(r != HostResidency.PINNED.value for r in labels)
        self.actual: dict[int, str] = {}
        # GPU_OWNED layers are resolved once here so alloc_layer_banks can consult the plan
        # ambiently, exactly like pin_banks/PinPipeline consult it for LOCKED.
        self.gpu_owned = frozenset(
            i for i, r in enumerate(labels) if r == HostResidency.GPU_OWNED.value
        )
        self.device = device
```

```python
@contextlib.contextmanager
def requested_residency(labels: list[str] | None, device=None):
    """Install the ambient per-layer residency plan for the enclosed bank load (``None`` = no plan, everything pins).
    ``device`` is where GPU_OWNED layers' banks are allocated."""
    global _requested_residency
    if labels is None:
        yield None
        return
    plan = _ResidencyPlan(labels, device)
    prev, _requested_residency = _requested_residency, plan
    try:
        yield plan
    finally:
        _requested_residency = prev


def plan_gpu_owned() -> "tuple[frozenset[int], torch.device | None]":
    """The ambient plan's GPU_OWNED layer ids and their target device (empty without a plan).

    Consulting the plan for owned layers counts as applying it (``_echo_residency`` keys its
    "this loader ignored the request" failure on that), but only when there is an owned set
    to honor -- a LOCKED-only plan is still only applied by a settle point."""
    plan = _requested_residency
    if plan is None:
        return frozenset(), None
    if plan.gpu_owned:
        plan.applied = True
    return plan.gpu_owned, plan.device
```

`python/freetoken/moe/host_banks.py` -- `_settle` (lines 268-274):

```python
def _settle(bank, residency: str) -> None:
    """Route a filled bank to its residency class (PAGEABLE = leave the plain mmap)."""
    if residency == HostResidency.GPU_OWNED.value:
        raise RuntimeError(
            "a GPU-owned MoE layer reached the host settle path: this checkpoint's bank "
            "loader has no per-layer completion sink, so its device banks would never be "
            "filled; drop --moe-gpu-owned-layers for this model"
        )
    if residency == HostResidency.PINNED.value:
        bank.pin()
    elif residency == HostResidency.LOCKED.value:
        bank.lock()
```

`python/freetoken/moe/host_banks.py` -- `PinPipeline._run` (lines 313-325), `submit_flush` and `__call__` (331-338):

```python
    def _run(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._exc is not None:
                continue  # drain without settling after a failure
            try:
                if callable(item):
                    item()  # GPU-owned layer flush; runs on THIS thread, which set the device
                    continue
                bank, residency, plan, layer_id = item
                _settle(bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, bank.residency.value)
            except BaseException as exc:  # surfaced by wait()/__exit__
                self._exc = exc

    def submit_flush(self, fn) -> None:
        """Queue a zero-argument callable to run on the drain thread (which has the creator's
        CUDA device set and serializes against the settles already queued)."""
        self._q.put(fn)
```

```python
    def __call__(self, layer_id: int, banks: dict) -> None:
        """Layer-completion sink: queue every bank of the completed layer at its ambient :func:`requested_residency` label.
        A GPU_OWNED layer has no host pages to settle -- its staging layer is flushed into the device banks instead, and the staging slot returns to the pool."""
        plan = _requested_residency
        residency = (
            HostResidency.PINNED.value if plan is None else plan.residency_for(layer_id)
        )
        if residency == HostResidency.GPU_OWNED.value:
            pool = next(iter(banks.values())).pool
            device_banks = {name: bank.tensor for name, bank in banks.items()}
            self.submit_flush(lambda: pool.flush(layer_id, device_banks))
            return
        for bank in banks.values():
            self.submit(bank, residency, plan, layer_id)
```

`python/freetoken/models/nvfp4_banks.py` -- `_alloc_nvfp4_host_banks` docstring (lines 88-92) gains one sentence, body unchanged (it already delegates to `alloc_layer_banks`, which now consults the ambient plan):

```python
def _alloc_nvfp4_host_banks(num_layers: int, E: int, H: int, I: int):
    """6 NVFP4 source banks, one ``[E, ...]`` tensor per layer (independent allocations),
    unpinned (pin-after-fill): register only after fill to skip cudaHostAlloc's slow
    commit. Caller fills each layer's ``.fill`` then settles it (per-layer, via
    ``PinPipeline``, as its writes complete).

    GPU-owned layers (the ambient ``requested_residency`` plan) get a device tensor plus a
    shared pinned staging layer instead of a host bank; see ``alloc_layer_banks``.
    """
```

`python/freetoken/models/nvfp4_banks.py` -- serial loader, replace lines 177-183:

```python
    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I)  # unpinned; pinned after fill
    # bank OBJECTS, not tensors: a GPU-owned layer's ``.fill`` resolves to a shared staging
    # view only at write time, and its ``.tensor`` already lives on the device.
    gate_up_packed = _hb["gate_up_packed"]
    gate_up_scale = _hb["gate_up_scale"]
    gate_up_global = _hb["gate_up_global"]
    down_packed = _hb["down_packed"]
    down_scale = _hb["down_scale"]
    down_global = _hb["down_global"]
```

and every assignment target in `_load` (lines 202-219) gains `.fill`:

```python
                    if kind == "weight":
                        if role == "gate":
                            gate_up_packed[bank_layer_id].fill[expert, :I] = tensor
                        elif role == "up":
                            gate_up_packed[bank_layer_id].fill[expert, I:] = tensor
                        elif role == "down":
                            down_packed[bank_layer_id].fill[expert] = tensor
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    else:
                        global_scale = globals_map[(layer, expert, proj)]
                        if role == "gate":
                            gate_up_scale[bank_layer_id].fill[expert, :I] = tensor
                            gate_up_global[bank_layer_id].fill[expert, :I] = global_scale
                        elif role == "up":
                            gate_up_scale[bank_layer_id].fill[expert, I:] = tensor
                            gate_up_global[bank_layer_id].fill[expert, I:] = global_scale
                        elif role == "down":
                            down_scale[bank_layer_id].fill[expert] = tensor
                            down_global[bank_layer_id].fill[expert] = global_scale
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
```

and the return (lines 235-242) reads the banks AFTER the load, so a GPU-owned layer hands back its device tensor:

```python
    expected = num_layers * E * 6
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    # after _load: a GPU-owned layer's ``.tensor`` is its (now filled) device bank
    return {name: [bank.tensor for bank in per_layer] for name, per_layer in _hb.items()}
```

`python/freetoken/models/nvfp4_banks.py` -- the parallel loader takes exactly the same three edits: the alias block (lines 301-307), the ten assignment targets in its `_load` (lines 326-341, each gaining `.fill`), and the return (354-361 -> the same dict comprehension).

`python/freetoken/moe/expert_banks.py` -- `bank_bytes_estimate` (403-419):

```python
def bank_bytes_estimate(model_config, gpu_owned: int = 0) -> int | None:
    """Estimated total expert-bank bytes of a raw checkpoint, from the model config alone.

    Sizes the pin-budget decisions where FTW metadata is not available; ``None`` for unknown formats or missing dims (callers then skip the pre-load sizing).
    nvfp4 uses the native-row formula, a slight over-estimate for the repacked backends.
    ``gpu_owned`` MoE layers allocate no host bank, so they are subtracted -- the pin budget
    and the boot banner must not claim RAM that is never asked for."""
    ...
    if per_expert is None or not all((layers, experts, hidden, inter)):
        return None
    return max(0, layers - gpu_owned) * experts * per_expert(hidden, inter)
```

`python/freetoken/moe/expert_banks.py` -- `load_expert_banks` (line 497):

```python
    with requested_residency(layer_residency, device=device) as residency_plan:
```

`python/freetoken/moe/expert_banks.py` -- `_echo_residency`, the "plan never consulted" branch (lines 526-533):

```python
    from freetoken.moe.host_banks import HostResidency

    if any(r == HostResidency.GPU_OWNED.value for r in requested):
        # unlike a pin/lock downgrade this is not a degradation: the owned layers were
        # allocated as HOST banks, so no device tensor exists and set_bank_sources would
        # register host rows as resident. Fail the boot instead.
        raise RuntimeError(
            "--moe-gpu-owned-layers: this checkpoint's bank loader settles banks without "
            "per-layer residency, so the owned layers were allocated in host RAM and no "
            "resident device banks exist; drop the flag for this model"
        )
    if any(r != HostResidency.PINNED.value for r in requested):
        logger.warning_rank0(
            "--moe-cpu-layers: this checkpoint's bank loader settles banks without "
            "per-layer residency (pre-pins everything); CPU-layer decode still works "
            "but saves no pinned quota"
        )
    return banks
```

`python/freetoken/checkpoint/ftw.py` -- `load_ftw_banks`, after the residency assert (line 454):

```python
    residency = layer_residency or [HostResidency.PINNED.value] * num_layers
    assert len(residency) == num_layers, (len(residency), num_layers)
    if any(r == HostResidency.GPU_OWNED.value for r in residency):
        # the FTW reader always reads into per-layer HostBanks (flat-region windowing needs
        # a page-aligned host scratch); repacking it for VRAM residency is out of scope
        raise ValueError(
            "--moe-gpu-owned-layers is not supported on an FTW packed checkpoint; serve the "
            "original checkpoint or drop the flag"
        )
```

`python/freetoken/moe/cpu_executor.py` -- new module function (place it immediately above `CpuMoeExecutor._resolve_banks`, i.e. before line 336, at module scope) and its call:

```python
def _reject_cuda_sources(banks: dict) -> None:
    """Refuse CUDA per-layer bank sources.

    ``_make_table`` hands the C++ executor raw ``data_ptr()`` values that it dereferences on
    the CPU. A GPU-owned MoE layer's source is a CUDA tensor, whose ``data_ptr()`` is a
    device address -- a silent wrong-memory read. Unreachable today (the flag requires
    ``--moe-backend offload`` with no CPU layers), which is exactly why it must be a guard.
    """
    for name, layers in banks.items():
        for layer_id, tensor in enumerate(layers):
            if tensor.is_cuda:
                raise ValueError(
                    f"CPU MoE executor cannot read bank {name!r} layer {layer_id}: it is a "
                    f"CUDA tensor (a GPU-owned MoE layer). --moe-gpu-owned-layers requires "
                    f"--moe-backend offload with no CPU layers"
                )
```

and inside `_resolve_banks`, as its first statement after the docstring:

```python
        _reject_cuda_sources(banks)
        if fmt == "bf16":
```

- [x] **Step 4: Run test to verify it passes**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/moe/test_gpu_owned_banks.py tests/moe/test_offload.py tests/moe/test_cpu_moe.py tests/checkpoint -q -p no:cacheprovider
```

- [x] **Step 5: Commit**

```powershell
git add python/freetoken/moe/host_banks.py python/freetoken/models/nvfp4_banks.py python/freetoken/moe/expert_banks.py python/freetoken/checkpoint/ftw.py python/freetoken/moe/cpu_executor.py tests/moe/test_gpu_owned_banks.py
git commit -m @'
feat(moe): load GPU-owned MoE layers straight into VRAM, no host bank

alloc_layer_banks gives a GPU_OWNED layer a device tensor plus a slot in a
bounded (cap 2) reusable pinned staging pool; the NVFP4 assignment loops write
through bank.fill and the layer-completion sink copies staging -> device behind a
CUDA event before the slot is reused. bank_bytes_estimate subtracts owned layers;
the FTW loader, the plain settle path, an unapplied residency plan and the CPU
executor all refuse the flag loudly rather than reading the wrong memory.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hnf1bGBLU4HLq9uHPtNjwU
'@
```

### Task 6: Engine wiring, boot log, and the reporting surfaces

**Files:**
- Modify `python/freetoken/engine/engine.py` (`_auto_cpu_layers` 1816-1843; `_init_offload_moe_cache` 662-817 -- specifically 680-689, 706-711, 731-750, 751-768, 783-786; new `_gpu_owned_boot_line` beside `_resolve_gpu_owned_layers`)
- Modify `python/freetoken/kvcache/cache_status.py` (`compute_cache_pools` 148-184)
- Modify `python/freetoken/server/api_server.py` (`cache_geometry` 788-810)
- Modify `python/freetoken/cache_report.py` (`cache_rate` 84-91)
- Modify `python/freetoken/server/model_meta.py` (`moe_total_experts` 122-133)
- Modify `python/freetoken/scheduler/scheduler.py` (`_reply_routing_stats` 856-870)
- Modify `python/freetoken/engine/mtp_fast_verify.py` (`_movement_result` 388-435)
- Modify `tests/engine/test_moe_gpu_owned_layers.py`, `tests/moe/test_routing_stats.py`, `tests/engine/test_mtp_fast_verify.py` (the two exact-dict assertions at ~120-134 and ~223-237)
- Create `tests/server/test_gpu_owned_geometry.py`

**Interfaces:**
- Consumes: `_resolve_gpu_owned_layers`, `_validate_gpu_owned_layers`, `Engine._gpu_owned_layer_ids`, `Engine._check_gpu_owned_cache_fits`, `cache.gpu_owned_layer_ids`, `expert_bytes_per_slot`.
- Produces:
  - `freetoken.engine.engine._gpu_owned_boot_line(owned, num_moe_layers, num_experts, per_expert_bytes, cache_size) -> str`
  - `compute_cache_pools(engine)["gpu_owned_layers"]: list[int]`
  - `cache_geometry(state)["gpu_owned_layers"]: list[int]`
  - `cache_rate(cache_size, geometry)` denominates over `num_moe_layers - len(geometry["gpu_owned_layers"])`
  - `moe_total_experts(config)` subtracts the resolved owned layers
  - routing-stats payload key `"gpu_owned_layers": list[int]`
  - `_movement_result(...)["gpu_owned_layers"]: int`

Note: the model-level `weight_placement_report` hook (spec section 7) is deliberately NOT extended -- `Engine._load_weights` calls it at line 606, before `_init_offload_moe_cache` exists, so it cannot see the owned set. The boot log line below is the single place that reports the owned layers and their resident bytes.

- [x] **Step 1: Write the failing test**

Append to `tests/engine/test_moe_gpu_owned_layers.py`:

```python
# ------------------------------------------------------------------ the boot line


def test_the_boot_line_reports_the_owned_set_the_resident_bytes_and_the_lru():
    from freetoken.engine.engine import _gpu_owned_boot_line

    line = _gpu_owned_boot_line(
        owned=frozenset({0, 1, 2, 6, 7, 22}),
        num_moe_layers=48,
        num_experts=512,
        per_expert_bytes=2_772_480,
        cache_size=4400,
    )

    assert line == (
        "MoE GPU-owned layers: [0, 1, 2, 6, 7, 22] (6 x 1.32 GiB resident, no host bank); "
        "LRU cache 4400 slots for 42 streaming layers"
    )
```

Append to `tests/moe/test_routing_stats.py`:

```python
# ------------------------------------------------ GPU-owned layers (--moe-gpu-owned-layers)


def test_the_summary_denominates_slots_over_the_streaming_layers_only():
    # 8 slots over 4 layers is 2 slots/layer; with 2 layers resident the STREAMING cache is
    # 8 slots over 2 layers, and the resident layers are excluded from every average
    cache = _cpu_cache(num_layers=4, num_experts=4, cache_size=8)
    cache.gpu_owned_layer_ids = frozenset({0, 3})
    cache.decode_freq[1] = torch.tensor([97, 1, 1, 1], dtype=torch.int64)
    cache.decode_freq[0] = torch.tensor([25, 25, 25, 25], dtype=torch.int64)

    stats = cache.decode_routing_stats()

    assert stats["slots_per_layer"] == 4.0
    assert stats["experts_for_90pct"] == 1.0        # layer 1 only; the resident layer 0 is out
    assert stats["oracle_hit_at_slots"] == pytest.approx(1.0)


def test_the_raw_histogram_still_counts_a_resident_layer():
    # the summary is about the streaming cache, but decode_freq is the input to every offline
    # skew study, so a resident layer's routing must still be recorded
    cache = _cpu_cache(num_layers=2, num_experts=4, cache_size=8)
    cache.gpu_owned_layer_ids = frozenset({1})
    cache.collect_decode_freq = True

    cache._note_decode_routing(1, torch.tensor([[2, 3]], dtype=torch.int32))

    assert cache.decode_freq[1].tolist() == [0, 0, 1, 1]


def test_the_routing_reply_names_the_gpu_owned_layers():
    from types import SimpleNamespace

    from freetoken.message.backend import RoutingStatsBackendMsg
    from freetoken.scheduler.scheduler import Scheduler

    cache = _cpu_cache(num_layers=3, num_experts=4, cache_size=8)
    cache.collect_stats = True
    cache.collect_decode_freq = True
    cache.gpu_owned_layer_ids = frozenset({1})
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(moe_offload_cache=cache)
    sent: list = []
    scheduler.send_result = sent.append

    scheduler._reply_routing_stats(RoutingStatsBackendMsg(request_id="r1"))

    stats = sent[0][0].stats
    assert stats["gpu_owned_layers"] == [1]
    assert stats["per_layer"][1]["resident"] is True
    assert stats["per_layer"][1]["miss_rate"] is None
    assert stats["per_layer"][0]["resident"] is False
```

Create `tests/server/test_gpu_owned_geometry.py`:

```python
"""GPU-owned MoE layers in the reported geometry: /v1/cache/status, the residency rate the
cache report prints, and the --moe-cache-rate denominator. CPU-only, no engine, no CUDA."""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.cache_report import cache_rate
from freetoken.server.api_server import cache_geometry
from freetoken.server.model_meta import moe_total_experts


def _state(gpu_owned_layers):
    return SimpleNamespace(
        stats=SimpleNamespace(kv_total_pages=0, mamba_total_slots=0),
        config=SimpleNamespace(
            page_size=64,
            moe_cache_policy="lru",
            moe_cache_size=4400,
            moe_cache_rate=None,
            moe_gpu_owned_layers="auto",
            model_config=SimpleNamespace(num_experts=512, num_moe_layers=48, dsv4_args=None),
        ),
        last_rebuild=None,
        cache_pools={
            "num_pages": 100, "page_size": 64, "moe_cache_size": 4400,
            "num_mamba_slots": 0, "swa_page_size": 0, "num_swa_pages": 0,
            "gpu_owned_layers": gpu_owned_layers,
        },
        unit_bytes={"kv_bytes_per_token": 1, "moe_bytes_per_expert": 2_772_480},
        swa_full_tokens_ratio=0.0,
        cache_budget_bytes=0,
        free_vram_bytes=0,
        cache_floors={},
    )


def test_the_geometry_carries_the_owned_layer_ids():
    geo = cache_geometry(_state([0, 1, 2, 6, 7, 22]))

    assert geo["gpu_owned_layers"] == [0, 1, 2, 6, 7, 22]
    assert geo["num_moe_layers"] == 48  # the MODEL is unchanged; only the rate denominator moves


def test_the_geometry_defaults_to_no_owned_layers():
    state = _state([])
    state.cache_pools.pop("gpu_owned_layers")

    assert cache_geometry(state)["gpu_owned_layers"] == []


def test_the_residency_rate_is_denominated_over_streaming_layers():
    # 4400 slots serve the 42 streaming layers, not all 48: reporting 4400 / (48 * 512)
    # would understate residency by an eighth
    geometry = {"num_experts": 512, "num_moe_layers": 48, "gpu_owned_layers": [0, 1, 2, 6, 7, 22]}

    assert cache_rate(4400, geometry) == 4400 / (42 * 512)
    assert cache_rate(4400, {"num_experts": 512, "num_moe_layers": 48}) == 4400 / (48 * 512)


def test_moe_total_experts_subtracts_the_resolved_owned_layers():
    config = SimpleNamespace(
        moe_backend="offload",
        moe_cpu_layers=None,
        moe_gpu_owned_layers="auto",
        model_config=SimpleNamespace(num_experts=512, num_moe_layers=48),
    )

    assert moe_total_experts(config) == 42 * 512
    config.moe_gpu_owned_layers = None
    assert moe_total_experts(config) == 48 * 512
```

Modify `tests/engine/test_mtp_fast_verify.py`: add `"gpu_owned_layers": 0,` to the two exact
`result.expert_movement == {...}` dicts (immediately after `"bytes_per_expert": 16,` in both),
and append a new test:

```python
def test_movement_reconciles_when_resident_layers_contribute_no_counters():
    """A GPU-owned layer never calls ensure_experts, so it adds nothing to lru_stats or
    stat_fetched. The hit + missing == active invariant therefore still holds, and the report
    says how many layers were resident so a reader is not left wondering where the bytes went
    (bytes_per_expert/actual_h2d_bytes are over bank_caches only -- spec section 7)."""
    cache = _StatsCache()
    cache.gpu_owned_layer_ids = frozenset({0, 1})
    ctx = _TargetContext()
    target = SimpleNamespace(model=_TargetTextModel(cache, ctx), lm_head=_LMHead())
    verifier = MTPFastVerifier(ctx, target, cache, torch.device("cpu"))

    movement = verifier.forward_eager(_batch(rows=3)).expert_movement

    assert movement["gpu_owned_layers"] == 2
    assert movement["active_experts"] == 20 and movement["missing_experts"] == 5
    assert movement["hit_experts"] + movement["missing_experts"] == movement["active_experts"]
```

- [x] **Step 2: Run test to verify it fails**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine/test_moe_gpu_owned_layers.py tests/moe/test_routing_stats.py tests/server/test_gpu_owned_geometry.py tests/engine/test_mtp_fast_verify.py -q -p no:cacheprovider
```

Expected failures:
- `ImportError: cannot import name '_gpu_owned_boot_line' from 'freetoken.engine.engine'`
- `KeyError: 'gpu_owned_layers'` in `test_the_routing_reply_names_the_gpu_owned_layers`
- `KeyError: 'gpu_owned_layers'` in `test_the_geometry_carries_the_owned_layer_ids`
- `assert 4400 / 24576 == 4400 / 21504` in `test_the_residency_rate_is_denominated_over_streaming_layers`
- `assert 24576 == 21504` in `test_moe_total_experts_subtracts_the_resolved_owned_layers`
- `KeyError: 'gpu_owned_layers'` in `test_movement_reconciles_when_resident_layers_contribute_no_counters`
- `test_the_summary_denominates_slots_over_the_streaming_layers_only` -> `assert 2.0 == 4.0`

- [x] **Step 3: Write minimal implementation**

`python/freetoken/engine/engine.py` -- the pure boot-line formatter, immediately after `_validate_gpu_owned_layers`:

```python
def _gpu_owned_boot_line(
    owned: frozenset[int],
    num_moe_layers: int,
    num_experts: int,
    per_expert_bytes: int,
    cache_size: int,
) -> str:
    """The single boot line reporting what --moe-gpu-owned-layers actually did (spec 3)."""
    layer_gib = num_experts * per_expert_bytes / 2**30
    return (
        f"MoE GPU-owned layers: {sorted(owned)} ({len(owned)} x {layer_gib:.2f} GiB "
        f"resident, no host bank); LRU cache {cache_size} slots for "
        f"{num_moe_layers - len(owned)} streaming layers"
    )
```

`python/freetoken/engine/engine.py` -- `_auto_cpu_layers` (line 1822), so the automatic
CPU-layer sizing does not count host banks that are never allocated:

```python
def _auto_cpu_layers(
    config: EngineConfig, num_moe_layers: int, reserved: int = 0, gpu_owned: int = 0
) -> frozenset[int]:
    ...
    bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(
        config.model_config, gpu_owned=gpu_owned
    )
```

`python/freetoken/engine/engine.py` -- `_init_offload_moe_cache`, replacing lines 680-689:

```python
        cpu_layer_ids = _resolve_cpu_layers(config, config.model_config.num_moe_layers)
        # Resolved (and fully validated) here, not just in _adjust_config: the backend may
        # still have been 'auto' at parse time. Stored on the engine because the budget
        # helpers read it.
        gpu_owned_layer_ids = _validate_gpu_owned_layers(
            config, config.model_config.num_moe_layers
        )
        self._gpu_owned_layer_ids = gpu_owned_layer_ids
        if (
            not cpu_layer_ids
            and config.moe_cpu_layers is None
            and config.moe_backend in ("offload", "hybrid")
            and _pin_budget_bytes(self._host_tables_bytes) is not None
        ):
            cpu_layer_ids = _auto_cpu_layers(
                config,
                config.model_config.num_moe_layers,
                reserved=self._host_tables_bytes,
                gpu_owned=len(gpu_owned_layer_ids),
            ) - gpu_owned_layer_ids  # an owned layer has no host bank to lock
```

`python/freetoken/engine/engine.py` -- the `--moe-backend cpu` pin-budget probe (line 711) passes the owned count too:

```python
                bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(
                    config.model_config, gpu_owned=len(gpu_owned_layer_ids)
                )
```

`python/freetoken/engine/engine.py` -- the residency vector, replacing lines 732-740:

```python
            requested_residency = None
            if split_residency or gpu_owned_layer_ids:
                from freetoken.moe.host_banks import HostResidency

                requested_residency = [
                    HostResidency.GPU_OWNED.value if i in gpu_owned_layer_ids
                    else HostResidency.LOCKED.value
                    if (split_residency and i in cpu_layer_ids)
                    else HostResidency.PINNED.value
                    for i in range(config.model_config.num_moe_layers)
                ]
```

`python/freetoken/engine/engine.py` -- the explicit-size fit check and the cache wiring, replacing lines 751-786:

```python
            if config.moe_cache_auto:
                size, pages, overlap = self._resolve_auto_moe_cache_size(config, banks)
                object.__setattr__(config, "moe_cache_size", size)
                object.__setattr__(config, "moe_prefill_overlap", overlap)
                if config.num_page_override is None:
                    # Honor the plan's KV half too: MoE slots and KV pages were solved
                    # against ONE budget (ratio x baseline - weights), so both must come
                    # from it. Re-solving pages later from a fresh free-memory reading
                    # double-counts everything allocated since the weights measurement
                    # (this expert cache, the CPU-executor GPU buffers, allocator
                    # slack) and goes negative whenever the expert fill is exact --
                    # a greedy fill leaves no headroom for the measurement delta.
                    object.__setattr__(config, "num_page_override", pages)
                logger.info_rank0(
                    f"--moe-cache-auto resolved moe_cache_size={size} "
                    f"num_pages={pages} (prefill_overlap={overlap})"
                )
            else:
                self._check_gpu_owned_cache_fits(config, banks)
            _require_offload_cache_size(config.moe_cache_size, config.model_config.num_experts)
            cache = OffloadMoeCache(
                # Models with leading dense layers (GLM-4) only have experts on the MoE
                # layers; num_moe_layers == num_layers when first_k_dense_replace == 0.
                num_layers=config.model_config.num_moe_layers,
                num_experts=config.model_config.num_experts,
                cache_size=config.moe_cache_size,
                device=self.device,
                cache_policy=config.moe_cache_policy,
                prefill_overlap=config.moe_prefill_overlap,
                prefill_hit_d2d=config.moe_prefill_hit_d2d,
                quant_format=banks.quant_format,
                decode_target=decode_target,
                hybrid_max_fetch=config.moe_hybrid_max_fetch,
            )
            # before set_bank_sources: the residency validation and the copy plan's skip of non-pinned layers key on the CPU-layer set; the GPU-owned set is validated against the residency labels the loader honored
            cache.cpu_layer_ids = cpu_layer_ids
            cache.gpu_owned_layer_ids = gpu_owned_layer_ids
            cache.set_bank_sources(
                banks.sources,
                layer_residency=banks.layer_residency,
                gpu_owned_layers=gpu_owned_layer_ids,
            )
            cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
            if gpu_owned_layer_ids:
                from freetoken.engine.cache_budget import expert_bytes_per_slot

                logger.info_rank0(
                    _gpu_owned_boot_line(
                        gpu_owned_layer_ids,
                        config.model_config.num_moe_layers,
                        config.model_config.num_experts,
                        expert_bytes_per_slot(banks.sources, gpu_owned_layer_ids),
                        config.moe_cache_size,
                    )
                )
```

`python/freetoken/kvcache/cache_status.py` -- `compute_cache_pools`: widen the annotation and report the owned set (lines 148-157 and the `moe` block at 175-177):

```python
def compute_cache_pools(engine: "Engine") -> Dict[str, Any]:
    """... 0 for pools the model lacks; never raises.
    ``gpu_owned_layers`` are the MoE layer ids served from permanently resident VRAM banks
    (--moe-gpu-owned-layers); [] when the flag is off."""
    pools: Dict[str, Any] = {
        "num_pages": 0, "page_size": 0, "moe_cache_size": 0, "num_mamba_slots": 0,
        "swa_page_size": 0, "num_swa_pages": 0, "gpu_owned_layers": [],
    }
    ...
        moe = engine.moe_offload_cache
        if moe is not None:
            pools["moe_cache_size"] = int(moe.cache_size or 0)
            pools["gpu_owned_layers"] = sorted(getattr(moe, "gpu_owned_layer_ids", ()) or ())
```

`python/freetoken/server/api_server.py` -- `cache_geometry`, in the `geo = {...}` literal beside `"num_moe_layers"` (line 794):

```python
        "num_moe_layers": num_moe_layers,
        # MoE layers served from permanently resident VRAM banks (--moe-gpu-owned-layers).
        # num_moe_layers still describes the MODEL; only the residency-rate denominator
        # (cache_report.cache_rate) drops these layers.
        "gpu_owned_layers": list(pools.get("gpu_owned_layers") or []),
```

`python/freetoken/cache_report.py` -- `cache_rate` (84-91):

```python
def cache_rate(cache_size: int, geometry: dict) -> float | None:
    """MoE residency: cached slots / the routed experts the SLOT CACHE actually serves
    (experts per layer x streaming MoE layers, the same basis the engine sizes the cache
    against). GPU-owned layers are permanently resident and never occupy a slot, so they
    leave the denominator. None for a non-MoE model, or a server that reports no expert
    counts."""
    owned = len((geometry or {}).get("gpu_owned_layers") or ())
    total = _int(geometry, "num_experts") * max(0, _int(geometry, "num_moe_layers") - owned)
    if total <= 0 or cache_size <= 0:
        return None
    return cache_size / total
```

`python/freetoken/server/model_meta.py` -- `moe_total_experts` (122-133):

```python
def moe_total_experts(config: Any) -> int:
    """Total routed-expert slots the SLOT CACHE serves: experts per layer x streaming MoE
    layers. Matches the engine's own basis (``Engine._resolve_auto_moe_cache_size``), so a
    residency rate or a ``--moe-cache-rate`` derived from it agrees with the size the engine
    resolved -- ``num_moe_layers`` excludes the leading dense layers a model like DSV4
    carries, and ``--moe-gpu-owned-layers`` layers are permanently resident, never cached."""
    try:
        model_config = config.model_config
    except Exception:  # noqa: BLE001 -- dummy/absent config: report "unknown", never raise
        return 0
    layers = int(getattr(model_config, "num_moe_layers", 0) or 0)
    if getattr(config, "moe_gpu_owned_layers", None):
        # local import: the frontend must not pay for the engine module unless the flag is set
        from freetoken.engine.engine import _resolve_gpu_owned_layers

        try:
            layers -= len(_resolve_gpu_owned_layers(config, layers))
        except Exception:  # noqa: BLE001 -- a bad spec is reported by _adjust_config, not here
            pass
    return max(0, layers) * int(getattr(model_config, "num_experts", 0) or 0)
```

`python/freetoken/scheduler/scheduler.py` -- `_reply_routing_stats`, in the `stats = {...}` literal (after `"cache_size"`, line 862):

```python
                    "cache_size": cache.cache_size,
                    # MoE layers served from resident VRAM banks: their per_layer rows carry
                    # resident: true / miss_rate: null, and they are excluded from the
                    # streaming-cache summary.
                    "gpu_owned_layers": sorted(getattr(cache, "gpu_owned_layer_ids", ()) or ()),
```

`python/freetoken/engine/mtp_fast_verify.py` -- `_movement_result`: add the key to BOTH returned dicts (the `counters is None` one at 392-406 and the real one at 421-435), immediately after `"bytes_per_expert"`:

```python
                "bytes_per_expert": bytes_per_expert,
                # GPU-owned MoE layers never call ensure_experts, so they add nothing to
                # active/missing/fetched and the reconciliation invariant below is unchanged.
                # Reported so a reader knows why h2d_bytes covers fewer layers than the model
                # has (bytes_per_expert / actual_h2d_bytes are over bank_caches only).
                "gpu_owned_layers": len(getattr(cache, "gpu_owned_layer_ids", ()) or ()),
```

- [x] **Step 4: Run test to verify it passes**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine tests/moe tests/server tests/scheduler -q -p no:cacheprovider
```

(Expect only the pre-existing failures listed in "Known pre-existing failures" below.)

- [x] **Step 5: Commit**

```powershell
git add python/freetoken/engine/engine.py python/freetoken/kvcache/cache_status.py python/freetoken/server/api_server.py python/freetoken/cache_report.py python/freetoken/server/model_meta.py python/freetoken/scheduler/scheduler.py python/freetoken/engine/mtp_fast_verify.py tests/engine/test_moe_gpu_owned_layers.py tests/moe/test_routing_stats.py tests/engine/test_mtp_fast_verify.py tests/server/test_gpu_owned_geometry.py
git commit -m @'
feat(moe): wire GPU-owned MoE layers through the engine and the reports

The engine resolves and validates the owned set, labels those layers GPU_OWNED in
the residency vector, hands them to the cache before set_bank_sources, checks an
explicit --moe-cache-size against the reservation, and prints one boot line naming
the set, the resident bytes and the LRU left for the streaming layers.
/v1/cache/status, /v1/cache/routing, the residency rate, the --moe-cache-rate
denominator and the MTP movement report all account for them.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hnf1bGBLU4HLq9uHPtNjwU
'@
```

### Task 7: Windows launcher parameter and documentation

**Files:**
- Modify `scripts/start-qwen38-flash-next-mmap-windows.ps1` (param block ends at line 76; banner 157-166; `$serveArgs` 177-200)
- Modify `docs/cli.md` (MoE flag table, insert after line 87)
- Modify `docs/windows-qwen38-flash-next-mmap.md` (new section before "### Expert routing statistics" at line 140)
- Modify `tests/engine/test_moe_gpu_owned_layers.py`

**Interfaces:**
- Consumes: `--moe-gpu-owned-layers` (Task 1).
- Produces: launcher parameter `-GpuOwnedLayers <spec>` (default `''` = off), env fallback `FREETOKEN_MOE_GPU_OWNED_LAYERS`, banner line `  GPU-owned MoE layers: <spec|off>`.

The launcher is tested the way `-ExpertLoad` and `-CollectRoutingStats` already are (`tests/engine/test_expert_load_flag.py:56-63`, `tests/moe/test_routing_stats.py:58-69`): parse the script text and assert the parameter, the passthrough and the absence of a hard-coded value.

- [x] **Step 1: Write the failing test**

Append to `tests/engine/test_moe_gpu_owned_layers.py`:

```python
# ------------------------------------------------------------------ the Windows launcher


def _launcher_text() -> str:
    from pathlib import Path

    return (
        Path(__file__).parents[2] / "scripts" / "start-qwen38-flash-next-mmap-windows.ps1"
    ).read_text(encoding="utf-8")


def test_the_launcher_exposes_gpu_owned_layers_and_defaults_to_off():
    launcher = _launcher_text()

    assert "[string]$GpuOwnedLayers = ''" in launcher
    assert "$env:FREETOKEN_MOE_GPU_OWNED_LAYERS" in launcher
    assert "'--moe-gpu-owned-layers', $GpuOwnedLayers" in launcher
    assert "if ($GpuOwnedLayers) {" in launcher
    # never a hard-coded set: the flag is only ever built from the parameter
    assert "'--moe-gpu-owned-layers', 'auto'" not in launcher


def test_the_launcher_banner_reports_the_owned_spec():
    assert "GPU-owned MoE layers: $(if ($GpuOwnedLayers) { $GpuOwnedLayers } else { 'off' })" in _launcher_text()


def test_the_docs_describe_the_flag():
    from pathlib import Path

    root = Path(__file__).parents[2]
    assert "--moe-gpu-owned-layers" in (root / "docs" / "cli.md").read_text(encoding="utf-8")
    windows = (root / "docs" / "windows-qwen38-flash-next-mmap.md").read_text(encoding="utf-8")
    assert "### GPU-owned MoE layers" in windows
    assert "-GpuOwnedLayers" in windows
```

- [x] **Step 2: Run test to verify it fails**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine/test_moe_gpu_owned_layers.py -q -p no:cacheprovider -k "launcher or docs"
```

Expected: 3 failures, each `AssertionError: assert "[string]$GpuOwnedLayers = ''" in '[CmdletBinding()]\nparam(\n...'` (and the equivalent for the banner and the docs assertions).

- [x] **Step 3: Write minimal implementation**

`scripts/start-qwen38-flash-next-mmap-windows.ps1` -- add the parameter at the end of the `param(...)` block (after `[switch]$CollectRoutingStats`, line 75; remember the comma):

```powershell
    [switch]$CollectRoutingStats,

    # MoE layers that keep every expert permanently resident in VRAM and allocate NO host
    # bank at all (--moe-gpu-owned-layers). Each owned layer hands back 1.32 GiB of host RAM
    # and costs 1.32 GiB of VRAM (about 512 LRU slots), so lower -MoECacheSize by ~512 per
    # owned layer. 'auto' is the six hungriest layers measured on this box
    # (docs/research/routing-skew-2026-09-02); 'auto:N' takes the first N; an explicit id
    # list, a count or a fraction also work. Empty (the default) leaves the feature off.
    # FREETOKEN_MOE_GPU_OWNED_LAYERS is the env fallback, read only here.
    [string]$GpuOwnedLayers = ''
)
```

`scripts/start-qwen38-flash-next-mmap-windows.ps1` -- the env fallback, beside the other
"switches only ever SET these" lines (after line 153):

```powershell
if ($DenseQuant -ne '') { $env:FREETOKEN_DENSE_QUANT = $DenseQuant }
if ($EmbedHost) { $env:FREETOKEN_EMBED_HOST = '1' }
# the env var is a fallback for the parameter, never an override of it
if (-not $GpuOwnedLayers -and $env:FREETOKEN_MOE_GPU_OWNED_LAYERS) {
    $GpuOwnedLayers = $env:FREETOKEN_MOE_GPU_OWNED_LAYERS
}
```

`scripts/start-qwen38-flash-next-mmap-windows.ps1` -- the banner, after the MoE cache line (162):

```powershell
Write-Host "  MoE cache slots: $(if ($MoECacheSize -gt 0) { $MoECacheSize } else { 'auto' })"
Write-Host "  GPU-owned MoE layers: $(if ($GpuOwnedLayers) { $GpuOwnedLayers } else { 'off' })"
```

`scripts/start-qwen38-flash-next-mmap-windows.ps1` -- the passthrough, beside the other optional flags (after the `$CollectRoutingStats` block, line 194):

```powershell
if ($GpuOwnedLayers) {
    $serveArgs += @('--moe-gpu-owned-layers', $GpuOwnedLayers)
}
```

`docs/cli.md` -- new row after the `--moe-cpu-layers` row (line 87):

```markdown
| `--moe-gpu-owned-layers` | off | With `offload`: MoE layers whose experts stay permanently resident in VRAM with no host bank (`0,1,2`, a count, a fraction, `auto`, or `auto:N`); each costs `num_experts` cache slots and returns one layer of host RAM |
```

`docs/windows-qwen38-flash-next-mmap.md` -- new section immediately before `### Expert routing statistics` (line 140) (outer fence is four backticks; the inner fences are part of the doc text):

````markdown
### GPU-owned MoE layers

`-GpuOwnedLayers` passes `--moe-gpu-owned-layers`. The named MoE layers keep all 512
experts permanently resident in VRAM and allocate **no pinned host bank at all**, so each
one hands 1.322 GiB of host RAM back and costs 1.322 GiB of VRAM (about 512 LRU slots).
Host RAM is the binding constraint on the tested system (95.6 GiB, ~89 GiB commit), and a
pinned bank cannot be partially released on Windows -- never allocating it is the only way
to give the RAM back.

| Value | Meaning |
| --- | --- |
| *(empty)* | Off. The default. |
| `auto` | The six hungriest layers by measured decode miss rate: `0, 1, 2, 6, 7, 22`. |
| `auto:N` | The first N of that ranked list (`1, 6, 0, 2, 7, 22, 10, 13, 5, 18, ...`). |
| `0,1,2` | An explicit MoE-layer id list. |
| `6` / `0.125` | A count (evenly strided) or a fraction, as `--moe-cpu-layers` reads them. |

The ranking comes from four decode captures on this box
(`docs/research/routing-skew-2026-09-02/`) and is a fixed built-in list, not a runtime
heuristic. Lower `-MoECacheSize` by roughly 512 slots per owned layer, or the server
refuses to boot and names the size that would fit:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start-qwen38-flash-next-mmap-windows.ps1 `
  -ModelPath $ModelPath `
  -GpuOwnedLayers auto `
  -MoECacheSize 4400
```

The boot log says exactly what happened:

```
INFO MoE GPU-owned layers: [0, 1, 2, 6, 7, 22] (6 x 1.32 GiB resident, no host bank); LRU cache 4400 slots for 42 streaming layers
```

`GET /v1/cache/routing` reports the owned layers as `resident: true` with a null
`miss_rate` (a resident layer cannot miss, and reporting `0.0` would read as a perfectly
cacheable streaming layer), and the `summary` block describes the streaming cache only.
The flag needs `--moe-backend offload`, refuses to overlap with `--moe-cpu-layers`, and is
not supported on an FTW packed checkpoint.
````

- [x] **Step 4: Run test to verify it passes**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine/test_moe_gpu_owned_layers.py tests/engine/test_expert_load_flag.py tests/moe/test_routing_stats.py -q -p no:cacheprovider
```

Then confirm the script still parses (this does NOT run it):

```powershell
$null = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path .\scripts\start-qwen38-flash-next-mmap-windows.ps1), [ref]$null, [ref]$null); "parsed"
```

- [x] **Step 5: Commit**

```powershell
git add scripts/start-qwen38-flash-next-mmap-windows.ps1 docs/cli.md docs/windows-qwen38-flash-next-mmap.md tests/engine/test_moe_gpu_owned_layers.py
git commit -m @'
feat(launcher): -GpuOwnedLayers passthrough and docs

The Windows launcher gains -GpuOwnedLayers (default off, FREETOKEN_MOE_GPU_OWNED_LAYERS
as the env fallback), prints the spec in the boot banner, and passes it as
--moe-gpu-owned-layers. docs/cli.md and the Windows guide describe the trade and
the slot arithmetic.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hnf1bGBLU4HLq9uHPtNjwU
'@
```

### Task 8: Operator live-verification checklist

**Files:**
- Create `docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-status.md`
- Modify `tests/engine/test_moe_gpu_owned_layers.py`

**Interfaces:**
- Consumes: the boot line from Task 6, the launcher flags from Task 7, spec section 9's check table.
- Produces: no code. The document is the hand-off: the implementer fills the commit list, the CPU test results and the "what only a live run can decide" section; the operator fills the measurement rows.

- [x] **Step 1: Write the failing test**

Append to `tests/engine/test_moe_gpu_owned_layers.py`:

```python
# ------------------------------------------------------------------ the operator checklist


def test_the_operator_checklist_covers_every_live_check_the_spec_asks_for():
    from pathlib import Path

    status = (
        Path(__file__).parents[2] / "docs" / "plans"
        / "2026-09-02-qwen38-gpu-owned-moe-layers-status.md"
    ).read_text(encoding="utf-8")

    for heading in (
        "## Commits",
        "## CPU test results",
        "## Live verification",
        "## Only a live GPU run can decide this",
    ):
        assert heading in status
    for check in (
        "boot log shows owned set",
        "scheduler private bytes",
        "whole-system physical in-use",
        "boot peak host RAM",
        "8k-chat decode tok/s",
        "TTFT",
        "answers at temperature 0",
        "picture request",
        "/v1/cache/routing",
        "owned-layer rows on device",
    ):
        assert check in status, check
```

- [x] **Step 2: Run test to verify it fails**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine/test_moe_gpu_owned_layers.py -q -p no:cacheprovider -k operator_checklist
```

Expected: `FileNotFoundError: [Errno 2] No such file or directory: 'D:\\FreeToken\\docs\\plans\\2026-09-02-qwen38-gpu-owned-moe-layers-status.md'`.

- [x] **Step 3: Write minimal implementation**

Create `docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-status.md` (the implementer fills the
result cells of the first two sections before handing over; the operator fills the third):

````markdown
# GPU-owned MoE layers -- implementation status and live checklist

Spec: `docs/design/2026-09-02-qwen38-gpu-owned-moe-layers-design.md`
Plan: `docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-plan.md`
Branch: `mtp-upstream-merge`

## Commits

One row per plan task, filled as each is committed.

| Task | Commit | Subject |
|---|---|---|
| 1 Config, flag, parser, validation | | |
| 2 VRAM budget arithmetic | | |
| 3 Cache owned-layer registry | | |
| 4 Forward paths | | |
| 5 Loader and refusals | | |
| 6 Engine wiring and reports | | |
| 7 Launcher and docs | | |
| 8 This checklist | | |

## CPU test results

Runner (no GPU):

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine tests/moe tests/server tests/scheduler tests/checkpoint -q -p no:cacheprovider
```

| Suite | Command | Passed | Failed | Notes |
|---|---|---|---|---|
| new: parser/validation | `pytest tests/engine/test_moe_gpu_owned_layers.py` | | | |
| new: loader | `pytest tests/moe/test_gpu_owned_banks.py` | | | |
| new: geometry | `pytest tests/server/test_gpu_owned_geometry.py` | | | |
| changed: cache | `pytest tests/moe/test_offload.py` | | | |
| changed: budget | `pytest tests/engine/test_cache_budget.py` | | | |
| changed: routing | `pytest tests/moe/test_routing_stats.py` | | | |
| changed: MTP verify | `pytest tests/engine/test_mtp_fast_verify.py` | | | |
| whole suite | the runner above | | | pre-existing failures listed in the plan |

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
   the CUDA event that gates staging reuse when two owned layers are in flight.
2. That the resident VRAM rows are byte-identical to the host-bank rows a normal load
   produces for the same layer (the one-off probe script).
3. Boot peak host RAM with cap-2 staging live (the CPU test proves the bound, not the cost).
4. That the decode and MTP CUDA graphs still capture (bs=1, MTP widths 1-6): the owned path
   is fixed-shape reads of fixed-address tensors, strictly simpler than today's, so no
   capture change is expected -- but "expected" is not "observed".
5. The `_build_fused_copy_plan` 0-placeholder assertion for an owned layer (the test is
   `@pytest.mark.skipif(not torch.cuda.is_available())`; the fused plan is only built on a
   CUDA device).
6. The actual speed cost of dropping the LRU from 6,750 to ~4,400 slots over 42 streaming
   layers -- spec section 11's open risk, and the only reason to keep, shrink or drop the
   owned set.
````

- [x] **Step 4: Run test to verify it passes**

```powershell
$env:PYTHONPATH = 'D:\FreeToken\scripts\windows-ple-mmap;D:\FreeToken\python;D:\FreeToken\.local\pytest-site'
$env:CUDA_VISIBLE_DEVICES = '-1'
& "$env:LOCALAPPDATA\FreeToken\venv\Scripts\python.exe" -m pytest tests/engine/test_moe_gpu_owned_layers.py -q -p no:cacheprovider
```

- [x] **Step 5: Commit**

```powershell
git add docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-status.md tests/engine/test_moe_gpu_owned_layers.py
git commit -m @'
docs(moe): operator checklist for the GPU-owned MoE layer live run

Spec section 9's check table plus the commit list, the CPU test commands and the
explicit list of what only a live GPU run can settle.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hnf1bGBLU4HLq9uHPtNjwU
'@
```

## Known pre-existing failures on this box (do NOT chase them)

Recorded 2026-09-02 on `mtp-upstream-merge`, Windows 11 without CUDA visible. None of these
is a regression and none is touched by this plan:

| Failure | Cause |
|---|---|
| 3 `_adjust_config` tests (attention-backend resolution) | flashinfer is not installed on this box |
| 6 `tests/*/test_weight*` cases | no `os.O_DIRECT` on Windows; the parallel reader path is skipped/short-circuited |
| q4_0 CPU-MoE cases | the `_cpu_moe` extension is not built for this venv |
| fp8 per-tensor MoE cases | same missing extension / no device |

Additionally, `tests/moe/test_offload.py::test_copy_plan_holds_a_zero_placeholder_for_gpu_owned_layers`
and `::test_copy_plan_skips_locked_layers_and_keeps_fused_path` SKIP without CUDA -- expected,
not a failure.

## Spec coverage

Every numbered spec requirement maps to a task:

| Spec | Task |
|---|---|
| 2 layer choice (`GPU_OWNED_LAYER_RANK`, `auto`, `auto:N`, no U-shaped heuristic) | 1 |
| 3 interface (CLI, config, env, launcher, validation, boot log, status/routing routes) | 1, 6, 7 |
| 4.1-4.3 loading (no host bank, staging pool cap 2, `placed` asserts unchanged) | 5 |
| 4.4-4.5 bank byte estimates, `_echo_residency` | 5 |
| 4.6 FTW refusal | 1 (config gate) + 5 (loader gate) |
| 4.7 `cpu_executor` CUDA-source guard | 5 |
| 5 cache and forward path (registry, predicates, copy plan, prefetch, guards, rebuild, decode-freq helper) | 3, 4 |
| 6 VRAM budget (reservation, first streaming layer, explicit-size overflow, `slots_per_layer`, report denominators) | 2, 3, 6 |
| 7 reporting honesty (`resident`/`miss_rate: null`, boot log, MTP reconciliation, documented `bytes_per_expert`) | 3, 6 |
| 8 tests | every task |
| 9 live verification | 8 |
| 11 risks (`device_ptr`, three layer-0 special cases, parallel interleaving, two allocation sites) | 3, 5 |
