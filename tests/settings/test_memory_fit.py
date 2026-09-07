from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.memory_fit import MemoryFitService
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.profiles_manager import ProfilesManager


@dataclass
class FakeCompleted:
    stdout: str
    stderr: str = ""
    returncode: int = 0


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "model_type": "qwen4_exp",
                "num_hidden_layers": 48,
                "num_experts": 512,
                "max_position_embeddings": 262144,
                "hidden_size": 16,
                "moe_intermediate_size": 32,
                "quantization_config": {"quant_algo": "NVFP4"},
                "ple_layer_ids": [0],
                "mtp_num_hidden_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    return model


def _boot(tmp_path: Path, model: Path) -> Path:
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text(
        f"$env:FREETOKEN_MTP_SPEC_DEPTH = '3'\n"
        f"& $launcher `\n    -ModelPath '{model}' `\n    -Port 2020\n",
        encoding="utf-8",
    )
    return boot


def _planner_result(need: int, *, suggestion: dict | None = None) -> dict:
    phase = {
        "need_bytes": need,
        "resident_bytes": need - 10,
        "boot_peak_bytes": need,
        "shortfall_bytes": 0,
    }
    return {
        "version": 1,
        "status": "ok",
        "fits": True,
        "fits_now": True,
        "fits_empty": True,
        "sampled_at": None,
        "effective_settings": {},
        "machine": {},
        "effective": {
            "page_size": 64,
            "attention_backend": "qsa_sparse",
            "ple_backend": "mmap",
            "pin_budget_bytes": 80_000,
            "bank_cuda_alloc": True,
        },
        "resources": {
            "ram": {
                "free_bytes": 100,
                "total_bytes": 200,
                "now": dict(phase),
                "empty": dict(phase),
            },
            "vram": {
                "free_bytes": 100,
                "total_bytes": 200,
                "now": dict(phase),
                "empty": dict(phase),
            },
        },
        "geometry": {
            "now": {
                "total_slots": 120,
                "lru_slots": 116,
                "owned_layers": [0],
                "num_pages": 8,
                "page_size": 64,
                "usable_kv_tokens": 448,
                "prefill_overlap": True,
            },
            "empty": {
                "total_slots": 120,
                "lru_slots": 116,
                "owned_layers": [0],
                "num_pages": 8,
                "page_size": 64,
                "usable_kv_tokens": 448,
                "prefill_overlap": True,
            },
        },
        "pinning": {"need_bytes": 50, "cap_bytes": 80_000, "shortfall_bytes": 0, "cpu_layers": []},
        "components": [],
        "issues": [],
        "assumptions": [],
        "suggestion": suggestion,
    }


def _memory_plan_module():
    path = Path(__file__).parents[2] / "python" / "freetoken" / "engine" / "memory_plan.py"
    name = "settings_test_memory_plan"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _service(tmp_path: Path, result: dict, snapshots: list[dict]) -> tuple[MemoryFitService, list[dict]]:
    calls: list[dict] = []
    iterator = iter(snapshots)

    def snapshot() -> dict:
        return next(iterator)

    def launch_builder(settings, *, base_env):
        return type(
            "Plan",
            (),
            {
                "argv": ["python", "-m", "freetoken.cli", "serve", "--model", settings["ModelPath"]],
                "env": dict(base_env),
            },
        )()

    def runner(argv, **kwargs):
        calls.append({"argv": argv, **kwargs})
        return FakeCompleted(json.dumps(result))

    return MemoryFitService(snapshot=snapshot, runner=runner, launch_builder=launch_builder), calls


def test_fitting_estimate_merges_saved_extension_settings_without_writing(tmp_path):
    model = _model(tmp_path)
    boot = _boot(tmp_path, model)
    before = boot.read_bytes()
    result = _planner_result(50)
    service, calls = _service(
        tmp_path,
        result,
        [
            {"ram_free_bytes": 100, "ram_total_bytes": 200, "vram_free_bytes": 100, "vram_total_bytes": 200},
            {"ram_free_bytes": 100, "ram_total_bytes": 200, "vram_free_bytes": 100, "vram_total_bytes": 200},
        ],
    )

    response = service.estimate_settings(
        {"MoECacheSize": 120}, boot_file=BootFile(boot), environ={"PATH": "/bin"}
    )

    assert response["status"] == "ok"
    assert response["fits"] is True
    assert response["fits_now"] is True and response["fits_empty"] is True
    assert response["effective_settings"]["FREETOKEN_MTP_SPEC_DEPTH"] == 3
    assert response["effective_settings"]["MoECacheSize"] == 120
    assert response["machine"]["ram_source"]
    assert json.loads(calls[0]["input"])["settings"]["FREETOKEN_MTP_SPEC_DEPTH"] == 3
    assert boot.read_bytes() == before


def test_non_fitting_estimate_reports_shortfall_and_suggestion(tmp_path):
    model = _model(tmp_path)
    boot = _boot(tmp_path, model)
    suggestion = {
        "target": "now",
        "settings": {"MoECacheSize": 64},
        "fits": True,
        "fits_now": True,
        "fits_empty": True,
        "changes": [{"name": "MoECacheSize", "from": 120, "to": 64, "reason": "Free card memory."}],
    }
    service, _ = _service(
        tmp_path,
        _planner_result(150, suggestion=suggestion),
        [
            {"ram_free_bytes": 100, "ram_total_bytes": 200, "vram_free_bytes": 100, "vram_total_bytes": 200},
            {"ram_free_bytes": 100, "ram_total_bytes": 200, "vram_free_bytes": 100, "vram_total_bytes": 200},
        ],
    )

    response = service.estimate_settings({"MoECacheSize": 120}, boot_file=BootFile(boot))

    assert response["fits"] is False
    assert response["fits_now"] is False and response["fits_empty"] is True
    assert response["resources"]["vram"]["now"]["shortfall_bytes"] == 50
    assert response["resources"]["vram"]["empty"]["shortfall_bytes"] == 0
    assert response["suggestion"]["settings"] == {"MoECacheSize": 64}


def test_free_now_and_empty_machine_verdicts_are_separate(tmp_path):
    model = _model(tmp_path)
    boot = _boot(tmp_path, model)
    service, _ = _service(
        tmp_path,
        _planner_result(75),
        [
            {"ram_free_bytes": 80, "ram_total_bytes": 100, "vram_free_bytes": 50, "vram_total_bytes": 100},
            {"ram_free_bytes": 80, "ram_total_bytes": 100, "vram_free_bytes": 50, "vram_total_bytes": 100},
        ],
    )

    response = service.estimate_settings({}, boot_file=BootFile(boot))

    assert response["fits"] is False
    assert response["fits_now"] is False
    assert response["fits_empty"] is True
    assert response["resources"]["vram"]["now"]["shortfall_bytes"] == 25
    assert response["resources"]["vram"]["empty"]["shortfall_bytes"] == 0


def test_estimate_route_returns_unavailable_as_503_and_invalid_settings_as_422(tmp_path):
    model = _model(tmp_path)
    boot = _boot(tmp_path, model)

    class Unavailable(MemoryFitService):
        def _run_child(self, settings, snapshot, environ):
            from freetoken.daemon.settings.memory_fit import EstimateUnavailable

            raise EstimateUnavailable("planner_timeout", "planner timed out")

    process = ProcessManager(
        boot_file=boot,
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *args, **kwargs: None,
        readiness=lambda: {"state": "unreachable"},
        sleep=lambda _: None,
        poll_interval=0,
    )
    app = create_app(
        boot_file=boot,
        process_manager=process,
        profiles=ProfilesManager(tmp_path / "profiles.json"),
        static_path=tmp_path / "missing.html",
        estimate_service=Unavailable(snapshot=lambda: {
            "ram_free_bytes": 100, "ram_total_bytes": 200, "vram_free_bytes": 100, "vram_total_bytes": 200,
        }),
    )
    with TestClient(app) as client:
        unavailable = client.post("/api/settings/estimate", json={"settings": {}})
        assert unavailable.status_code == 503
        assert unavailable.json()["status"] == "unavailable"
        assert unavailable.json()["fits_now"] is None
        invalid = client.post("/api/settings/estimate", json={"settings": {"ContextTokens": 999999999}})
        assert invalid.status_code == 422
        assert invalid.json()["detail"][0]["field"] == "ContextTokens"


@pytest.mark.parametrize("action", ["estimate", "start", "restart"])
def test_metadata_preparation_is_once_and_keeps_stop_status_responsive(tmp_path, monkeypatch, action):
    from freetoken.daemon.settings import memory_fit

    model = _model(tmp_path)
    boot = _boot(tmp_path, model)
    snapshot = {"ram_free_bytes": 100, "ram_total_bytes": 200, "vram_free_bytes": 100, "vram_total_bytes": 200}
    service, _ = _service(tmp_path, _planner_result(50), [snapshot, snapshot])
    process = ProcessManager(
        boot_file=boot, stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log", lock_path=tmp_path / "gpu.lock",
        readiness=lambda: {"state": "serving"}, stats=lambda: {}, gpu_probe=lambda: {},
        platform_windows=False, linux_stop=lambda *_args, **_kw: {"ok": True},
        launch_builder=lambda *_args, **_kw: SimpleNamespace(
            argv=["python"], env={}, notes=(), command_line=lambda: "python"),
        popen=lambda *_args, **_kw: SimpleNamespace(poll=lambda: None),
    )
    app = create_app(
        boot_file=boot, process_manager=process, estimate_service=service,
        profiles=ProfilesManager(tmp_path / "profiles.json"), static_path=tmp_path / "missing.html",
    )
    entered, release = threading.Event(), threading.Event()
    reads = []
    read_model = memory_fit.read_model

    def blocked_read(path):
        reads.append(path)
        entered.set()
        assert release.wait(2)
        return read_model(path)

    monkeypatch.setattr(memory_fit, "read_model", blocked_read)
    url = "/api/settings/estimate" if action == "estimate" else f"/api/server/{action}"
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as callers:
        request = callers.submit(client.post, url, json={"settings": {"MoECacheSize": 120}})
        try:
            assert entered.wait(1)
            assert callers.submit(client.get, "/api/status").result(timeout=0.5).status_code == 200
            assert callers.submit(client.post, "/api/server/stop", json={}).result(timeout=0.5).status_code == 202
            process._executor.submit(lambda: None).result(timeout=1)
        finally:
            release.set()
        response = request.result(timeout=2)
    assert response.status_code == (200 if action == "estimate" else 202), response.text
    assert reads == [str(model)]
    process.close()


def test_incomplete_child_result_is_unavailable_instead_of_zero_fit():
    from freetoken.daemon.settings.memory_fit import EstimateUnavailable

    malformed = {
        "version": 1,
        "status": "ok",
        "fits": True,
        "fits_now": True,
        "fits_empty": True,
        "resources": {"ram": {}, "vram": {}},
        "geometry": {"now": {}, "empty": {}},
        "pinning": {},
        "components": [],
        "issues": [],
        "assumptions": [],
    }

    with pytest.raises(EstimateUnavailable, match="complete"):
        MemoryFitService._assemble(
            malformed,
            {},
            {
                "ram_free_bytes": 100,
                "ram_total_bytes": 200,
                "vram_free_bytes": 100,
                "vram_total_bytes": 200,
            },
            "2026-09-06T00:00:00Z",
        )


def test_ram_probe_uses_effective_nested_cgroup_limit(monkeypatch):
    from freetoken.daemon.settings import memory_fit

    gib = 1 << 30
    child = Path("/sys/fs/cgroup/user.slice/freetoken.service")
    parent = Path("/sys/fs/cgroup/user.slice")
    limits = {
        child / "memory.max": 4 * gib,
        child / "memory.current": 1 * gib,
        parent / "memory.max": 8 * gib,
        parent / "memory.current": 2 * gib,
    }

    def fake_open(path, *args, **kwargs):
        assert path == "/proc/meminfo"
        return io.StringIO(
            f"MemTotal: {16 * gib // 1024} kB\n"
            f"MemAvailable: {12 * gib // 1024} kB\n"
        )

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(memory_fit, "_cgroup_v2_paths", lambda: (child, parent), raising=False)
    monkeypatch.setattr(memory_fit, "_finite_cgroup_value", lambda path: limits.get(path))

    free, total, source, limited = memory_fit._read_ram_snapshot()

    assert total == 4 * gib
    assert free == 3 * gib
    assert "cgroup-v2" in source
    assert limited is True


@pytest.mark.parametrize("mount_root", ["/", "/user.slice"])
def test_ram_probe_discovers_cgroup_relative_to_mount_root(monkeypatch, mount_root):
    from freetoken.daemon.settings import memory_fit

    gib = 1 << 30
    mount = Path("/sandbox/cgroup")
    leaf = mount / ("user.slice/freetoken.service" if mount_root == "/" else "freetoken.service")
    files = {
        Path("/proc/self/cgroup"): "0::/user.slice/freetoken.service\n",
        Path("/proc/self/mountinfo"): (
            f"21 20 0:28 {mount_root} {mount} rw - cgroup2 cgroup rw\n"
            "22 20 0:28 /other.slice /unrelated rw - cgroup2 cgroup rw\n"
        ),
        leaf / "memory.max": str(4 * gib),
        leaf / "memory.current": str(gib),
        mount / "memory.max": "max",
    }

    def read_text(path, **_kwargs):
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    def fake_open(path, *args, **kwargs):
        assert path == "/proc/meminfo"
        return io.StringIO(f"MemTotal: {16 * gib // 1024} kB\nMemAvailable: {12 * gib // 1024} kB\n")

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr("builtins.open", fake_open)

    free, total, source, limited = memory_fit._read_ram_snapshot()

    assert (free, total) == (3 * gib, 4 * gib)
    assert limited and "cgroup-v2:effective-limit" in source
    paths = memory_fit._cgroup_v2_paths()
    assert paths[0] == leaf and paths[-1] == mount
    assert all(path.is_relative_to(mount) for path in paths)


def test_vision_stream_workspace_uses_one_real_component_not_all_cpu_weights(monkeypatch, tmp_path):
    from freetoken.daemon.settings import model_info

    memory_plan = _memory_plan_module()

    class Tensor:
        def __init__(self, size):
            self._size = size

        def numel(self):
            return self._size

        def element_size(self):
            return 1

    class Component:
        def __init__(self, size):
            self._state = {"weight": Tensor(size)}

        def state_dict(self):
            return self._state

    visual = SimpleNamespace(
        patch_embed=Component(100),
        blocks=SimpleNamespace(op_list=[Component(200), Component(300)]),
        merger=Component(50),
    )
    model = SimpleNamespace(visual=visual)
    model_dir = tmp_path / "vision-model"
    model_dir.mkdir()
    monkeypatch.setattr(
        model_info,
        "read_model",
        lambda _path: SimpleNamespace(ple_bytes=0),
    )
    config = SimpleNamespace(
        model_path=str(model_dir),
        model_config=SimpleNamespace(is_multimodal=True),
        ple_backend="disk",
    )
    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    monkeypatch.setenv("FREETOKEN_VISION_EXECUTION", "layer-stream")

    _pinned, workspace, components = memory_plan._ple_and_vision_bytes(
        config,
        {"cpu": 2_169_260_512, "cuda": 4_422_165_272},
        model=model,
    )

    assert workspace == 200
    assert workspace != 2_169_260_512
    assert {item["bytes"] for item in components} == {100, 200, 50}


def test_candidate_normalization_is_allowed_to_use_launch_environment():
    memory_plan = _memory_plan_module()

    base = SimpleNamespace(
        moe_cache_size=120,
        moe_cache_auto=False,
        moe_gpu_owned_layers=None,
        kv_reserve_tokens=1024,
        page_size=64,
        num_page_override=None,
        num_token_override=None,
    )

    def builder(settings, *, base_env):
        assert settings["GpuOwnedLayers"] == ""
        assert base_env["FREETOKEN_MOE_GPU_OWNED_LAYERS"] == "auto:2"
        return SimpleNamespace(
            argv=["python", "-m", "freetoken.cli", "serve", "--moe-cache-size", "64"],
            env=dict(base_env),
        )

    def parser(argv, environment):
        assert argv == ["--moe-cache-size", "64"]
        assert environment["FREETOKEN_MOE_GPU_OWNED_LAYERS"] == "auto:2"
        return SimpleNamespace(
            moe_cache_size=64,
            moe_cache_auto=False,
            moe_gpu_owned_layers="auto:2",
            kv_reserve_tokens=1024,
            page_size=64,
            num_page_override=None,
            num_token_override=None,
        )

    candidate = memory_plan._candidate_config(
        base,
        {"MoECacheSize": 64, "GpuOwnedLayers": ""},
        environment={"FREETOKEN_MOE_GPU_OWNED_LAYERS": "auto:2"},
        launch_builder=builder,
        config_parser=parser,
    )

    assert candidate.moe_gpu_owned_layers == "auto:2"


def test_suggestion_revalidates_the_winning_full_snapshot(monkeypatch):
    memory_plan = _memory_plan_module()
    calls = []

    base_config = SimpleNamespace(
        model_config=SimpleNamespace(num_moe_layers=0, num_experts=0),
    )
    base_settings = {
        "MoECacheSize": 0,
        "GpuOwnedLayers": "",
        "ContextTokens": 1024,
        "KVCacheTokens": 0,
    }
    base_result = {
        "fits_now": False,
        "fits_empty": False,
        "geometry": {"now": {"page_size": 64, "total_slots": 0}},
    }

    monkeypatch.setattr(memory_plan, "_owned_layers", lambda _config: frozenset())
    monkeypatch.setattr(memory_plan, "_ceil_div", lambda value, divisor: (value + divisor - 1) // divisor)
    monkeypatch.setattr(
        memory_plan,
        "_candidate_config",
        lambda _base, settings, **_kwargs: calls.append(("candidate", dict(settings), _kwargs))
        or SimpleNamespace(context=settings["ContextTokens"]),
    )
    monkeypatch.setattr(memory_plan, "_evaluate_scenarios", lambda *_args, **_kwargs: {})

    def render(_request, _machine, _evaluated, configs):
        calls.append(("render",))
        fits = configs[0].context <= 512
        return {"fits_now": fits, "fits_empty": fits}

    monkeypatch.setattr(memory_plan, "_render_plan", render)

    suggestion = memory_plan._suggestion(
        base_settings=base_settings,
        base_result=base_result,
        base_config=base_config,
        machine={},
        environment={},
        model_bytes={},
        host_tables=0,
        vision_workspace=0,
        table_components=[],
        per_expert=0,
        source_format="none",
        runtime_format="none",
    )

    assert suggestion is not None and suggestion["settings"]["ContextTokens"] == 512
    assert sum(1 for item in calls if item[0] == "render") == 3
    candidates = [item[1] for item in calls if item[0] == "candidate"]
    assert candidates[-2] == candidates[-1]


def test_suggested_ownership_survives_apply_estimate_and_save(monkeypatch, tmp_path):
    from freetoken.daemon.settings.memory_fit import prepare_settings

    memory_plan = _memory_plan_module()
    model = _model(tmp_path)
    boot = _boot(tmp_path, model)
    original = prepare_settings(
        {"MoECacheSize": 4096, "GpuOwnedLayers": "", "ContextTokens": 1024, "KVCacheTokens": 1088},
        boot_file=BootFile(boot),
    )
    base_config = SimpleNamespace(model_config=SimpleNamespace(num_moe_layers=48, num_experts=512))
    normalized_ownership = []

    def parser(argv, _environment):
        flag = "--moe-gpu-owned-layers"
        owned = argv[argv.index(flag) + 1] if flag in argv else ""
        normalized_ownership.append(owned)
        return SimpleNamespace(moe_gpu_owned_layers=owned)

    monkeypatch.setattr(memory_plan, "_parse_config", parser)
    monkeypatch.setattr(memory_plan, "_owned_layers", lambda _config: frozenset())
    monkeypatch.setattr(memory_plan, "_evaluate_scenarios", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        memory_plan, "_render_plan",
        lambda _request, _machine, _result, configs: {
            "fits_now": bool(configs[0].moe_gpu_owned_layers),
            "fits_empty": bool(configs[1].moe_gpu_owned_layers),
        },
    )
    suggestion = memory_plan._suggestion(
        base_settings=original,
        base_result={
            "fits_now": False, "fits_empty": False,
            "geometry": {"now": {"page_size": 64, "total_slots": 4096, "prefill_overlap": True}},
        },
        base_config=base_config, machine={}, environment={}, model_bytes={}, host_tables=0,
        vision_workspace=0, table_components=[], per_expert=1,
        source_format="nvfp4", runtime_format="nvfp4",
    )
    assert suggestion is not None
    applied = {**original, **suggestion["settings"]}
    snapshot = {"ram_free_bytes": 100, "ram_total_bytes": 200, "vram_free_bytes": 100, "vram_total_bytes": 200}
    service, _ = _service(tmp_path, _planner_result(50), [snapshot, snapshot])
    process = ProcessManager(
        boot_file=boot, stop_script=tmp_path / "stop.ps1", log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock", runner=lambda *_args, **_kwargs: None,
    )
    app = create_app(
        boot_file=boot, process_manager=process, profiles=ProfilesManager(tmp_path / "profiles.json"),
        static_path=tmp_path / "missing.html", estimate_service=service,
    )
    with TestClient(app) as client:
        estimated = client.post("/api/settings/estimate", json={"settings": applied})
        saved = client.put("/api/settings", json={"settings": applied})

    assert estimated.status_code == saved.status_code == 200
    assert applied == estimated.json()["effective_settings"] == saved.json()["settings"]
    assert suggestion["settings"]["GpuOwnedLayers"] == "auto:1"
    assert normalized_ownership[-2:] == ["auto:1", "auto:1"]


def test_context_fallback_uses_resolved_page_size_without_geometry(monkeypatch):
    memory_plan = _memory_plan_module()
    candidates = []
    base_config = SimpleNamespace(
        model_config=SimpleNamespace(num_moe_layers=0, num_experts=0),
        page_size=64,
    )
    base_settings = {
        "MoECacheSize": 0,
        "GpuOwnedLayers": "",
        "ContextTokens": 128,
        "KVCacheTokens": 0,
    }
    base_result = {
        "fits_now": False,
        "fits_empty": False,
        "geometry": {"now": None, "empty": None},
        "effective": {"page_size": 64},
    }

    monkeypatch.setattr(memory_plan, "_owned_layers", lambda _config: frozenset())
    monkeypatch.setattr(memory_plan, "_ceil_div", lambda value, divisor: (value + divisor - 1) // divisor)
    monkeypatch.setattr(
        memory_plan,
        "_candidate_config",
        lambda _base, settings, **_kwargs: candidates.append(dict(settings)) or SimpleNamespace(),
    )
    monkeypatch.setattr(memory_plan, "_evaluate_scenarios", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        memory_plan,
        "_render_plan",
        lambda request, *_args: {
            "fits_now": request["settings"].get("ContextTokens", 0) <= 64,
            "fits_empty": request["settings"].get("ContextTokens", 0) <= 64,
        },
    )

    suggestion = memory_plan._suggestion(
        base_settings=base_settings,
        base_result=base_result,
        base_config=base_config,
        machine={},
        environment={},
        model_bytes={},
        host_tables=0,
        vision_workspace=0,
        table_components=[],
        per_expert=0,
        source_format="none",
        runtime_format="none",
    )

    assert suggestion is not None
    assert suggestion["settings"]["ContextTokens"] == 64
    assert suggestion["settings"]["KVCacheTokens"] == 128
    assert all(candidate["KVCacheTokens"] % 64 == 0 for candidate in candidates)
