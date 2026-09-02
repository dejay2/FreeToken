"""--moe-collect-decode-freq: flag plumbing, the histogram's reset window, and the
GET /v1/cache/routing round trip. All CPU -- no CUDA, no weights."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from freetoken.engine.engine import _env_flag
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.server.args import parse_args

ANON_MODEL = "/models/anon"


class _HFConfig:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(*extra: str):
    config = _HFConfig({"architectures": ["Qwen2ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        args, _run_shell = parse_args(["--model", ANON_MODEL, *extra])
    return args


# ---------------------------------------------------------------- flag plumbing


def test_collect_decode_freq_is_off_by_default():
    assert _parse().moe_collect_decode_freq is False


def test_collect_decode_freq_flag_reaches_the_config():
    assert _parse("--moe-collect-decode-freq").moe_collect_decode_freq is True


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("on", True),
                                            ("0", False), ("", False), ("no", False)])
def test_env_flag_spelling(monkeypatch, value, expected):
    monkeypatch.setenv("FREETOKEN_MOE_COLLECT_DECODE_FREQ", value)
    assert _env_flag("FREETOKEN_MOE_COLLECT_DECODE_FREQ") is expected


def test_env_flag_is_false_when_unset(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MOE_COLLECT_DECODE_FREQ", raising=False)
    assert _env_flag("FREETOKEN_MOE_COLLECT_DECODE_FREQ") is False


def test_launcher_exposes_collect_routing_stats():
    from pathlib import Path

    launcher = (
        Path(__file__).parents[2]
        / "scripts"
        / "start-qwen38-flash-next-mmap-windows.ps1"
    ).read_text(encoding="utf-8")

    assert "[switch]$CollectRoutingStats" in launcher
    assert "if ($CollectRoutingStats) {" in launcher
    assert "'--moe-collect-decode-freq'" in launcher


def test_the_dense_override_table_clears_the_flag():
    """A dense checkpoint has no experts to count; the knob must not survive the reset."""
    from freetoken.engine.engine import _DENSE_MOE_SETTINGS

    assert _DENSE_MOE_SETTINGS["moe_collect_decode_freq"] is False


# ---------------------------------------------------------------- the histogram


def _cpu_cache(num_layers=2, num_experts=4, cache_size=8):
    return OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=torch.device("cpu"),
    )


def test_the_histogram_starts_empty_and_unarmed():
    cache = _cpu_cache()

    assert cache.collect_decode_freq is False
    assert cache.decode_freq.shape == (2, 4)
    assert int(cache.decode_freq.sum()) == 0
    # unarmed and unrouted: nothing to summarize, and no prefetch counters either
    assert cache.decode_routing_stats() == {}


def test_reset_stats_windows_the_routing_histogram_too():
    """decode_routing_stats merges reset-delimited prefetch counters into the histogram's
    own numbers, so both halves have to share one window."""
    cache = _cpu_cache()
    cache.decode_freq[0, 1] = 5
    cache.decode_freq[1, 3] = 7

    cache.reset_stats()

    assert int(cache.decode_freq.sum()) == 0


def test_routing_summary_describes_a_skewed_layer():
    # 4 slots over 2 layers = 2 slots per layer; only layer 0 sees traffic, and layers with
    # no traffic are excluded from every average.
    cache = _cpu_cache(num_layers=2, num_experts=4, cache_size=4)
    # one expert takes 97% of the traffic
    cache.decode_freq[0] = torch.tensor([97, 1, 1, 1], dtype=torch.int64)

    stats = cache.decode_routing_stats()

    assert stats["working_set_mean"] == 4.0
    assert stats["working_set_max"] == 4
    assert stats["experts_for_90pct"] == 1.0
    assert stats["slots_per_layer"] == 2.0
    assert stats["oracle_hit_at_slots"] == pytest.approx(0.98)
    assert 0.0 < stats["norm_entropy"] < 0.2


def test_a_uniform_layer_has_maximal_entropy():
    cache = _cpu_cache(num_layers=2, num_experts=4, cache_size=8)
    cache.decode_freq[0] = torch.tensor([10, 10, 10, 10], dtype=torch.int64)

    stats = cache.decode_routing_stats()

    assert stats["norm_entropy"] == pytest.approx(1.0)
    assert stats["experts_for_90pct"] == 4.0


# ---------------------------------------------------------------- the route


class _RoutingState:
    """A frontend stand-in that answers a RoutingStatsMsg the way the backend would."""

    def __init__(self, *, stats=None, error=None, maintenance="serving", drop=False):
        self.maintenance_state = maintenance
        self.routing_futures: dict = {}
        self._stats = stats if stats is not None else {}
        self._error = error
        self._drop = drop
        self.sent: list = []

    async def send_one(self, msg):
        self.sent.append(msg)
        if self._drop:
            return
        fut = self.routing_futures.pop(msg.request_id, None)
        if fut is not None and not fut.done():
            fut.set_result({"stats": self._stats, "error": self._error})


def _routing_client(state):
    import freetoken.server.api_server as api

    prev = api._GLOBAL_STATE
    api._GLOBAL_STATE = state
    return TestClient(api.app), (lambda: setattr(api, "_GLOBAL_STATE", prev))


def test_routing_route_returns_the_raw_histogram():
    payload = {
        "collect_decode_freq": True,
        "num_layers": 2,
        "num_experts": 3,
        "cache_size": 6,
        "summary": {"norm_entropy": 0.5},
        "per_layer": [{"layer": 0, "miss_rate": 0.25}],
        "decode_freq": [[1, 2, 3], [4, 5, 6]],
    }
    state = _RoutingState(stats=payload)
    client, restore = _routing_client(state)
    try:
        response = client.get("/v1/cache/routing")
    finally:
        restore()

    assert response.status_code == 200
    assert response.json()["decode_freq"] == [[1, 2, 3], [4, 5, 6]]
    assert state.sent[0].reset is False


def test_routing_route_forwards_the_reset_flag():
    state = _RoutingState(stats={"decode_freq": []})
    client, restore = _routing_client(state)
    try:
        assert client.get("/v1/cache/routing?reset=true").status_code == 200
    finally:
        restore()

    assert state.sent[0].reset is True


def test_routing_route_409s_when_the_counters_were_never_armed():
    state = _RoutingState(error="decode counters are off; boot with --moe-collect-decode-freq")
    client, restore = _routing_client(state)
    try:
        response = client.get("/v1/cache/routing")
    finally:
        restore()

    assert response.status_code == 409
    assert "--moe-collect-decode-freq" in response.json()["error"]


def test_routing_route_503s_while_the_model_is_loading():
    state = _RoutingState(maintenance="loading")
    client, restore = _routing_client(state)
    try:
        response = client.get("/v1/cache/routing")
    finally:
        restore()

    assert response.status_code == 503
    assert state.sent == []  # never dispatched


def test_routing_route_504s_and_drops_the_future_on_timeout():
    state = _RoutingState(drop=True)
    client, restore = _routing_client(state)
    try:
        response = client.get("/v1/cache/routing?timeout=0.05")
    finally:
        restore()

    assert response.status_code == 504
    assert state.routing_futures == {}


def test_a_dead_backend_wakes_a_pending_routing_waiter():
    """fail_pending_rebuilds is the crash path; a routing waiter must not hang on it."""
    from freetoken.server.api_server import FrontendManager

    async def scenario():
        state = FrontendManager.__new__(FrontendManager)
        state.rebuild_futures = {}
        state.routing_futures = {}
        state._loop = asyncio.get_running_loop()
        fut = state._loop.create_future()
        state.routing_futures["r1"] = fut
        state.fail_pending_rebuilds("worker died")
        return await asyncio.wait_for(fut, timeout=2)

    assert asyncio.run(scenario()) == {"stats": {}, "error": "worker died"}


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
