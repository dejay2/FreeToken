from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PRIVATE = Path(r"D:\FreeToken-ple-mmap-vision\.local\mtp-spike\prototypes")


def _load(name: str):
    path = PRIVATE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_fixture_has_only_the_approved_neutral_categories():
    generator = _load("generate_mtp_fast_workloads")

    first = generator.build_fixture("run-one", "nonce-one")
    second = generator.build_fixture("run-two", "nonce-two")

    assert first != second
    assert first["run_id"] == "run-one"
    assert first["generation_nonce"] == "nonce-one"
    assert "no prior conversations" in first["provenance"].lower()
    assert {item["category"] for item in first["workloads"]} == {
        "coding",
        "reasoning",
        "prose",
        "long-chat",
    }
    assert all("screenshot" not in item["category"] for item in first["workloads"])
    assert all("run-one" in item["id"] for item in first["workloads"])
    long_chat = next(item for item in first["workloads"] if item["category"] == "long-chat")
    assert [message["role"] for message in long_chat["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    generator.validate_fixture(first)


def test_candidate_request_bodies_are_private_control_independent_and_fixed():
    generator = _load("generate_mtp_fast_workloads")
    harness = _load("run_mtp_live_shadow_requests")
    fixture = generator.build_fixture("run-contract", "nonce-contract")

    cases = harness.build_request_cases(fixture, output_tokens=4)

    assert len(cases) == 16
    assert {case["sampling"] for case in cases} == {"greedy", "normal"}
    assert {case["temperature_state"] for case in cases} == {"cold", "warm"}
    for case in cases:
        body = case["body"]
        assert body["max_tokens"] == 5
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["messages"] in [item["messages"] for item in fixture["workloads"]]
        assert not any(term in str(body).lower() for term in ("mtp", "speculative"))
    by_pair = {}
    for case in cases:
        key = (case["workload"], case["sampling"])
        by_pair.setdefault(key, []).append(case["request_body_sha256"])
    assert all(len(values) == 2 and len(set(values)) == 1 for values in by_pair.values())


def test_repair_sender_uses_one_normal_request_and_refuses_overwrite(tmp_path):
    sender = _load("run_one_mtp_repair_correctness")

    assert sender.validate_candidate_base_url("http://127.0.0.1:2030") == (
        "http://127.0.0.1:2030"
    )
    body = sender.build_request_body("repair-contract")
    assert body["temperature"] == 0.8
    assert body["top_k"] == 40
    assert body["top_p"] == 0.9
    assert body["max_tokens"] >= 4
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert not any(
        term in str(body).lower() for term in ("mtp", "draft", "speculative")
    )

    output = tmp_path / "repair-correctness.json"
    sender.write_new_json(output, {"passed": True})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sender.write_new_json(output, {"passed": False})


def test_candidate_url_is_exactly_loopback_port_2030():
    harness = _load("run_mtp_live_shadow_requests")

    assert harness.validate_candidate_base_url("http://127.0.0.1:2030") == (
        "http://127.0.0.1:2030"
    )
    for bad in (
        "http://127.0.0.1:2020",
        "http://localhost:2030",
        "http://0.0.0.0:2030",
        "http://127.0.0.1:2031",
    ):
        with pytest.raises(ValueError, match="127.0.0.1:2030"):
            harness.validate_candidate_base_url(bad)


def test_candidate_controls_disable_normal_decode_graphs():
    launcher = Path(
        r"D:\FreeToken-ple-mmap-mtp-spike\scripts\start-qwen38-flash-next-mmap-windows.ps1"
    ).read_text(encoding="utf-8")
    matrix = (PRIVATE / "run_guarded_live_mtp_matrix.ps1").read_text(
        encoding="utf-8"
    )

    assert "[int]$CudaGraphMaxBS" in launcher
    assert "--cuda-graph-max-bs" in launcher
    assert "'-CudaGraphMaxBS', '0'" in matrix
    assert matrix.count("'-CudaGraphMaxBS', '0'") == 1
    assert "Start-Candidate" in matrix


def test_state_diagnostic_sender_uses_one_ordinary_exact_port_request_and_refuses_overwrite(
    tmp_path,
):
    diagnostic = _load("run_one_mtp_state_diagnostic")

    assert diagnostic.validate_candidate_base_url("http://127.0.0.1:2030") == (
        "http://127.0.0.1:2030"
    )
    for bad in (
        "http://127.0.0.1:2020",
        "http://localhost:2030",
        "http://0.0.0.0:2030",
        "http://127.0.0.1:2031",
    ):
        with pytest.raises(ValueError, match="127.0.0.1:2030"):
            diagnostic.validate_candidate_base_url(bad)

    body = diagnostic.build_request_body("20260831T120000000Z")
    assert body["model"] == "Qwen3.8-Flash-Next-NVFP4"
    assert body["max_tokens"] == 5
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert not any(
        term in str(body).lower() for term in ("mtp", "draft", "speculative")
    )

    output = tmp_path / "state-diagnostic-client.json"
    diagnostic.write_new_json(output, {"status": "CONNECTION_FAILED"})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.write_new_json(output, {"status": "ORDINARY_RESPONSE"})
