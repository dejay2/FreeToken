"""The readiness contract: /health is 200 from the moment uvicorn binds, so only
``/v1/cache/status`` -> ``state == "serving"`` means the model is actually loaded.

The desktop app, the daemon's ``/engine/health`` proxy and ``ft shell`` all rely on
/health answering 200 while the weights load (they render the ``loading`` body's phase
and progress), so the status code must not become conditional. These tests pin both
halves: /health stays 200 and names the phase, and the cache-status gate is what flips.
"""

from __future__ import annotations

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
