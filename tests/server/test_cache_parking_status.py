from __future__ import annotations

import asyncio
from types import SimpleNamespace


def test_cache_status_exposes_latest_parking_metrics(monkeypatch):
    import freetoken.server.api_server as api

    parking = {
        "mode": "ssd",
        "parked_count": 3,
        "parked_bytes": 1234,
        "hits": 2,
        "misses": 1,
        "last_restore_ms": 17.5,
        "disabled": False,
        "last_error": None,
    }
    state = SimpleNamespace(
        maintenance_state="serving",
        last_rebuild=None,
        parking_status=parking,
        _create_listener_once=lambda: None,
    )
    monkeypatch.setattr(api, "get_global_state", lambda: state)
    monkeypatch.setattr(api, "cache_geometry", lambda _state: {})

    result = asyncio.run(api.cache_status())

    assert result["parking"] == parking


def test_cache_status_starts_listener_before_reading_scheduler_snapshot(monkeypatch):
    import freetoken.server.api_server as api

    calls = []
    state = SimpleNamespace(
        maintenance_state="serving",
        last_rebuild=None,
        parking_status={"mode": "ram"},
        _create_listener_once=lambda: calls.append("listen"),
    )
    monkeypatch.setattr(api, "get_global_state", lambda: state)
    monkeypatch.setattr(api, "cache_geometry", lambda _state: {})

    asyncio.run(api.cache_status())

    assert calls == ["listen"]
