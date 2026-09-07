"""linux_launch: the Windows launcher's rules mapped onto ft serve, the WSL env, and stop."""

from __future__ import annotations

import json

import pytest

from freetoken.daemon.settings import linux_launch as ll


def _facts(**over):
    base = dict(is_moe=True, expert_count=24576, max_context=262144, has_ple=True, has_vision=True, has_mtp=True,
                parking_supported=True, model_type="qwen4_exp")
    base.update(over)
    return ll.ModelFacts(**base)


def _plan(settings, **facts):
    return ll.build_launch({"ModelPath": "/models/demo", **settings}, python="py", base_env={}, facts=_facts(**facts), wsl=False)


def _arg(plan, flag):
    argv = plan.argv
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_the_fast_single_chat_profile_maps_like_the_windows_launcher():
    plan = _plan({
        "Port": 2020, "ContextTokens": 65536, "KVCacheTokens": 65536, "MaxRunningRequests": 1,
        "MoECacheSize": 6750, "MoEVramReserveBytes": 0, "MoECacheHeadroomBytes": 0, "MemoryGovernor": False, "DenseQuant": "int8",
        "EmbedHost": True, "EnableVision": False, "ExpertLoad": "parallel", "EnableCacheReport": True,
        "CollectRoutingStats": False, "GpuOwnedLayers": "", "CudaGraphMaxBS": -1, "KVPark": "off",
    })
    assert plan.argv[:9] == ["py", "-m", "freetoken.cli", "serve", "--model", "/models/demo", "--host", "127.0.0.1", "--port"]
    assert _arg(plan, "--port") == "2020"
    assert _arg(plan, "--ple-backend") == "disk", "MTP off: the Linux-native disk table"
    assert _arg(plan, "--moe-backend") == "offload"
    assert _arg(plan, "--moe-cache-size") == "6750"
    assert _arg(plan, "--max-running-requests") == "1"
    assert _arg(plan, "--kv-reserve-tokens") == "65536"
    assert _arg(plan, "--num-tokens") == "65536"
    assert _arg(plan, "--moe-vram-reserve-bytes") == "0" and _arg(plan, "--moe-cache-headroom-bytes") == "0"
    assert "--enable-cache-report" in plan.argv and "--moe-collect-decode-freq" not in plan.argv
    assert "--moe-gpu-owned-layers" not in plan.argv and "--cuda-graph-max-bs" not in plan.argv
    assert plan.env["FREETOKEN_DENSE_QUANT"] == "int8" and plan.env["FREETOKEN_EMBED_HOST"] == "1"
    assert plan.env["FREETOKEN_LOAD_VISION"] == "0" and plan.env["FREETOKEN_VISION_EXECUTION"] == "gpu"
    assert plan.notes == []


def test_mtp_on_forces_the_mmap_table_and_vision_and_parking_pass_through():
    plan = _plan({
        "FREETOKEN_MTP_SPECULATE": "1", "EnableVision": True, "VisionExecution": "layer-stream", "VisionWeights": "mmap",
        "KVPark": "ram", "KVParkIdleMs": 5, "GpuOwnedLayers": "auto", "CollectRoutingStats": True, "CudaGraphMaxBS": 4,
    })
    assert _arg(plan, "--ple-backend") == "mmap"
    assert plan.env["FREETOKEN_MTP_SPECULATE"] == "1"
    assert plan.env["FREETOKEN_LOAD_VISION"] == "1" and plan.env["FREETOKEN_VISION_WEIGHTS"] == "mmap"
    assert _arg(plan, "--kv-park") == "ram" and _arg(plan, "--kv-park-idle-ms") == "5"
    assert _arg(plan, "--moe-gpu-owned-layers") == "auto"
    assert "--moe-collect-decode-freq" in plan.argv and _arg(plan, "--cuda-graph-max-bs") == "4"


def test_the_guess_depth_reaches_the_engine_as_a_number_and_does_not_count_as_mtp_on():
    plan = _plan({"FREETOKEN_MTP_SPEC_DEPTH": 3})
    assert plan.env["FREETOKEN_MTP_SPEC_DEPTH"] == "3"
    assert _arg(plan, "--ple-backend") == "disk", "a depth alone does not switch MTP on"
    assert _plan({}).env["FREETOKEN_MTP_SPEC_DEPTH"] == "5", "the catalogue default, not a toggle's 0"
    assert _plan({"FREETOKEN_MTP_SPEC_DEPTH": "2"}).env["FREETOKEN_MTP_SPEC_DEPTH"] == "2"


def test_the_guess_safety_catches_reach_the_engine_and_the_bar_is_capped_at_the_chain():
    plan = _plan({})
    assert plan.env["FREETOKEN_MTP_SPEC_CONF_CUT"] == "0.8"
    assert plan.env["FREETOKEN_MTP_SPEC_MIN_EMITTED"] == "2.4"
    assert plan.env["FREETOKEN_MTP_SPEC_COST_AWARE"] == "1"
    assert _arg(plan, "--ple-backend") == "disk" and plan.notes == [], "catches alone do not switch MTP on"
    custom = _plan({"FREETOKEN_MTP_SPEC_CONF_CUT": 0.65, "FREETOKEN_MTP_SPEC_MIN_EMITTED": "3.6", "FREETOKEN_MTP_SPEC_COST_AWARE": False})
    assert custom.env["FREETOKEN_MTP_SPEC_CONF_CUT"] == "0.65" and custom.env["FREETOKEN_MTP_SPEC_MIN_EMITTED"] == "3.6"
    assert custom.env["FREETOKEN_MTP_SPEC_COST_AWARE"] == "0"
    # The engine refuses a bar above 1 + depth, so a shallow chain caps the bar with a note.
    capped = _plan({"FREETOKEN_MTP_SPEC_DEPTH": 1, "FREETOKEN_MTP_SPEC_MIN_EMITTED": 2.4})
    assert capped.env["FREETOKEN_MTP_SPEC_MIN_EMITTED"] == "2.0"
    assert any("capped" in note and "FREETOKEN_MTP_SPEC_MIN_EMITTED" in note for note in capped.notes)


def test_a_model_without_the_features_gets_the_windows_launcher_notes():
    plan = _plan(
        {"FREETOKEN_MTP_SPECULATE": "1", "EnableVision": True, "KVPark": "ssd", "MoECacheSize": 10, "GpuOwnedLayers": "auto",
         "PleBackend": "mmap", "ContextTokens": 0},
        is_moe=False, expert_count=0, has_ple=False, has_vision=False, has_mtp=False, parking_supported=False, max_context=4096,
    )
    notes = " | ".join(plan.notes)
    assert "forced off: this model ships no MTP head" in notes and plan.env["FREETOKEN_MTP_SPECULATE"] == "0"
    assert "Picture input switched off" in notes and plan.env["FREETOKEN_LOAD_VISION"] == "0"
    assert "KV parking 'ssd' forced off" in notes and _arg(plan, "--kv-park") == "off"
    assert "Expert-slot settings ignored" in notes and "--moe-backend" not in plan.argv
    assert "PleBackend mmap ignored" in notes and "--ple-backend" not in plan.argv
    assert "using the model's own limit 4096" in notes and _arg(plan, "--kv-reserve-tokens") == "4096"


def test_cache_slots_are_clamped_and_context_is_bounded():
    plan = _plan({"MoECacheSize": 99999}, expert_count=1000)
    assert _arg(plan, "--moe-cache-size") == "1000" and "clamped" in plan.notes[0]
    with pytest.raises(ValueError, match="longer than this model can read"):
        _plan({"ContextTokens": 999999}, max_context=8192)


def test_wsl_environment_adds_the_pin_settings_without_overriding(monkeypatch):
    monkeypatch.setattr(ll, "_wsl_memory_gib", lambda: 88.0)
    monkeypatch.setattr(ll, "_compute_capability", lambda: "12.0")
    env = ll.wsl_environment({"PATH": "/usr/bin"}, wsl=True)
    assert env["FREETOKEN_BANK_CUDA_ALLOC"] == "1"
    assert env["FREETOKEN_PIN_BUDGET_GB"] == str(int(88 * ll.WSL_PIN_BUDGET_FRACTION))
    assert env["TVM_FFI_CUDA_ARCH_LIST"] == "12.0"
    kept = ll.wsl_environment({"PATH": "/usr/bin", "FREETOKEN_PIN_BUDGET_GB": "40", "FREETOKEN_BANK_CUDA_ALLOC": "0"}, wsl=True)
    assert kept["FREETOKEN_PIN_BUDGET_GB"] == "40" and kept["FREETOKEN_BANK_CUDA_ALLOC"] == "0"
    assert ll.wsl_environment({"PATH": "/usr/bin"}, wsl=False) == {"PATH": "/usr/bin"}


def test_launcher_arguments_use_the_powershell_spelling():
    settings = ll.parse_launcher_args(["-ModelPath", "/m", "-Port", "2034", "-EmbedHost", "-EnableVision", "false", "-ExpertLoad", "serial"])
    assert settings == {"ModelPath": "/m", "Port": "2034", "EmbedHost": True, "EnableVision": False, "ExpertLoad": "serial"}
    with pytest.raises(ValueError, match="needs a value"):
        ll.parse_launcher_args(["-Port"])
    with pytest.raises(ValueError, match="expected -Name value"):
        ll.parse_launcher_args(["--port", "1"])


def test_model_facts_come_from_config_json(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["DemoForCausalLM"],
        "text_config": {"model_type": "qwen4_exp_text", "ple_layer_ids": [3], "num_hidden_layers": 48, "first_k_dense_replace": 0,
                        "num_experts": 512, "mtp_num_hidden_layers": 1, "max_position_embeddings": 262144},
        "vision_config": {"depth": 1},
    }), encoding="utf-8")
    facts = ll.read_model_facts(tmp_path)
    assert facts.is_moe and facts.expert_count == 48 * 512 and facts.has_ple and facts.has_vision and facts.has_mtp
    assert facts.max_context == 262144 and facts.parking_supported and facts.architecture == "DemoForCausalLM"
    with pytest.raises(FileNotFoundError):
        ll.read_model_facts(tmp_path / "missing")


def test_stop_servers_kills_the_selection_and_waits_for_the_ports(monkeypatch):
    killed = []
    alive = {41, 42}
    monkeypatch.setattr(ll, "find_server_pids", lambda port: set(alive))
    monkeypatch.setattr(ll.os.path, "exists", lambda path: int(path.split("/")[2]) in alive if path.startswith("/proc/") else False)

    def fake_kill(pid, sig):
        killed.append((pid, sig))
        alive.discard(pid)

    monkeypatch.setattr(ll.os, "kill", fake_kill)
    listens = iter([{2021}, set()])
    monkeypatch.setattr(ll, "_listeners", lambda ports: next(listens, set()))
    monkeypatch.setattr(ll, "_vram_used_mb", lambda: 500)
    clock = iter(range(0, 1000))
    report = ll.stop_servers(2020, timeout=30, sleep=lambda _: None, monotonic=lambda: float(next(clock)))
    assert report["ok"] is True and report["killed"] == [41, 42] and report["remaining"] == []
    assert {pid for pid, _ in killed} == {41, 42}
