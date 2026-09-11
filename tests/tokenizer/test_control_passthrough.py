"""The tokenizer worker's control-message passthrough (api -> scheduler -> api).

The worker asserts that every pending message is a tokenize/detokenize/abort message or one of
``_CONTROL_MSG_TYPES``; a control type without a branch here kills the worker on its first use
(that is how the J1 step/residency messages surfaced). Each shape is forwarded field by field.
"""

from __future__ import annotations

import dataclasses

from freetoken.message import (
    CacheProgressMsg,
    CacheProgressReply,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
    CacheResidencyBackendMsg,
    CacheResidencyMsg,
    CacheResidencyReply,
    CacheResidencyResultMsg,
    CacheStepBackendMsg,
    CacheStepMsg,
    CacheStepReply,
    CacheStepResultMsg,
    RoutingStatsMsg,
    RoutingStatsResultMsg,
)
from freetoken.tokenizer.server import _CONTROL_MSG_TYPES, _forward_control_msg


def test_completed_prefill_progress_survives_both_message_wires():
    from freetoken.message import BaseTokenizerMsg, BaseFrontendMsg, PrefillProgressMsg, PrefillProgressReply
    import queue

    original = PrefillProgressMsg(processed_tokens=8192, batch_size=2)
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(original))
    output = queue.Queue()
    assert isinstance(decoded, _CONTROL_MSG_TYPES)
    assert _forward_control_msg(decoded, queue.Queue(), output)
    delivered = output.get_nowait()
    delivered = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(delivered))
    assert delivered == PrefillProgressReply(processed_tokens=8192, batch_size=2)


class _Q:
    def __init__(self) -> None:
        self.items: list = []

    def put(self, m) -> None:
        self.items.append(m)


def _forward(m):
    backend, frontend = _Q(), _Q()
    assert _forward_control_msg(m, backend, frontend)
    assert len(backend.items) + len(frontend.items) == 1
    return (backend.items or frontend.items)[0]


def _same_fields(src, dst, rename: dict[str, str] | None = None) -> None:
    rename = rename or {}
    for f in dataclasses.fields(src):
        assert getattr(dst, rename.get(f.name, f.name)) == getattr(src, f.name), f.name


def test_every_control_shape_is_counted():
    for t in (
        CacheRebuildMsg, CacheRebuildResultMsg, CacheStepMsg, CacheStepResultMsg,
        CacheResidencyMsg, CacheResidencyResultMsg, RoutingStatsMsg, RoutingStatsResultMsg,
        CacheProgressMsg,
    ):
        assert t in _CONTROL_MSG_TYPES, t.__name__


def test_rebuild_msg_forwards_layer_moves():
    out = _forward(CacheRebuildMsg(request_id="r", moe_cache_size=3000, layer_moves=[(5, "pinned")]))
    assert isinstance(out, CacheRebuildBackendMsg)
    _same_fields(CacheRebuildMsg(request_id="r", moe_cache_size=3000, layer_moves=[(5, "pinned")]), out)
    assert out.layer_moves == [(5, "pinned")]


def test_step_msg_forwards_to_backend():
    src = CacheStepMsg(request_id="s1", axis="vram", direction="down", ram_tight=True)
    out = _forward(src)
    assert isinstance(out, CacheStepBackendMsg)
    _same_fields(src, out)


def test_step_result_forwards_to_frontend():
    src = CacheStepResultMsg(
        request_id="s1", status="ok", applied="gpu_owned->pinned", layer=7, at_floor=False,
        moe_cache_size=3678, layers={"owned": 5, "pinned": 43, "disk": 0},
        vram_free_bytes=2 << 30, error=None,
    )
    out = _forward(src)
    assert isinstance(out, CacheStepReply)
    _same_fields(src, out)


def test_residency_round_trip_shapes():
    out = _forward(CacheResidencyMsg(request_id="q"))
    assert isinstance(out, CacheResidencyBackendMsg) and out.request_id == "q"
    src = CacheResidencyResultMsg(request_id="q", status="ok", residency={"owned": 1}, error=None)
    out = _forward(src)
    assert isinstance(out, CacheResidencyReply)
    _same_fields(src, out)


def test_unknown_message_is_not_forwarded():
    backend, frontend = _Q(), _Q()
    assert not _forward_control_msg(object(), backend, frontend)
    assert backend.items == [] and frontend.items == []


def test_progress_msg_forwards_to_frontend():
    """One unit of maintenance work (scheduler -> api): the API restarts its stuck clock on it."""
    m = CacheProgressMsg(request_id="op-1", phase="waiting", detail="drained 1 prefill")
    out = _forward(m)
    assert isinstance(out, CacheProgressReply)
    _same_fields(m, out)
