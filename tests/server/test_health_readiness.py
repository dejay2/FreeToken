"""The readiness contract: /health is 200 from the moment uvicorn binds, so only
``/v1/cache/status`` -> ``state == "serving"`` means the model is actually loaded.

The desktop app, the daemon's ``/engine/health`` proxy and ``ft shell`` all rely on
/health answering 200 while the weights load (they render the ``loading`` body's phase
and progress), so the status code must not become conditional. These tests pin both
halves: /health stays 200 and names the phase, and the cache-status gate is what flips.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.server.control_api import build_health, register_control_routes


def _state(maintenance="serving", *, phase=None, fatal=None):
    load_progress = (
        SimpleNamespace(phase=phase, done_bytes=7, total_bytes=11)
        if phase is not None
        else None
    )
    return SimpleNamespace(
        maintenance_state=maintenance,
        fatal_error=fatal,
        instance_id="generation-1",
        config=SimpleNamespace(served_model_name="unit-model"),
        load_progress=load_progress,
        ready_at=None,
    )


def _client(state):
    app = FastAPI(version="0.0-test")
    register_control_routes(app, lambda: state, lambda: {})
    return TestClient(app)


def test_health_is_200_while_the_model_is_still_loading():
    response = _client(_state("loading", phase="expert_banks")).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "loading"
    assert body["phase"] == "expert_banks"
    assert body["progress"] == {"done_bytes": 7, "total_bytes": 11}


def test_health_is_200_once_serving_and_echoes_the_gate():
    response = _client(_state("serving")).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["maintenance"] == "serving"


def test_health_is_200_even_when_the_engine_died():
    response = _client(_state("failed", fatal="cuda oom")).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "error"


def test_health_status_ok_does_not_mean_serving():
    """"ok" covers rebuilding/stopping too -- the gate is the `maintenance` field."""
    for maintenance in ("serving", "rebuilding", "stopping"):
        doc = build_health(_state(maintenance), "0.0-test")
        assert doc["status"] == "ok"
        assert doc["maintenance"] == maintenance


def test_cache_status_state_is_the_readiness_signal():
    import freetoken.server.api_server as api

    prev_state, prev_geometry = api._GLOBAL_STATE, api.cache_geometry
    api.cache_geometry = lambda _state: {}
    try:
        client = TestClient(api.app)
        for maintenance in ("loading", "serving"):
            api._GLOBAL_STATE = SimpleNamespace(
                maintenance_state=maintenance, last_rebuild=None
            )
            response = client.get("/v1/cache/status")
            assert response.status_code == 200
            assert response.json()["state"] == maintenance
    finally:
        api._GLOBAL_STATE, api.cache_geometry = prev_state, prev_geometry


def test_ready_is_503_while_the_model_is_still_loading():
    response = _client(_state("loading", phase="expert_banks")).get("/ready")

    assert response.status_code == 503
    assert response.json()["health"] == "loading"


def test_ready_is_200_once_serving():
    response = _client(_state("serving")).get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": "unit-model"}


def test_ready_waits_out_a_runtime_rebuild_like_the_chat_gate():
    assert _client(_state("rebuilding")).get("/ready").status_code == 200


def test_ready_is_503_when_stopping_or_failed():
    assert _client(_state("stopping")).get("/ready").status_code == 503
    assert _client(_state("serving", fatal="worker died")).get("/ready").status_code == 503


def _model_state(path):
    state = _state("serving")
    state.config = SimpleNamespace(served_model_name="served-name", model_path=path)
    return state


@pytest.mark.parametrize("wanted", ["served-name", "/m/Qwen-A", "/m/Qwen-A/", "Qwen-A"])
def test_ready_for_the_loaded_model_by_any_of_its_names(wanted):
    assert _client(_model_state("/m/Qwen-A")).get("/ready", params={"model": wanted}).status_code == 200


def test_ready_is_503_for_a_different_model_even_while_serving():
    # A llama-swap swap between two FreeToken models: the old one still serves for a few
    # seconds while the wrapper stops it, and must not pass the new model's check.
    response = _client(_model_state("/m/Qwen-A")).get("/ready", params={"model": "Qwen-B"})
    assert response.status_code == 503
    assert response.json()["reason"] == "another model is loaded"


def test_ready_matches_a_differently_spelled_path_to_the_same_folder(tmp_path):
    real = tmp_path / "Qwen-A"
    real.mkdir()
    link = tmp_path / "link-to-A"
    link.symlink_to(real)
    assert _client(_model_state(str(real))).get("/ready", params={"model": str(link)}).status_code == 200


def test_ready_before_the_config_exists_reports_loading_not_another_model():
    state = _state("loading", phase="expert_banks")
    state.config = None
    response = _client(state).get("/ready", params={"model": "Qwen-A"})
    assert response.status_code == 503 and response.json()["health"] == "loading"
