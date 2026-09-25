"""Sleep/wake control messages survive the wire and the tokenizer worker's passthrough."""

from __future__ import annotations

from freetoken.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    CacheSleepBackendMsg,
    CacheSleepMsg,
    CacheSleepReply,
    CacheSleepResultMsg,
)
from freetoken.tokenizer.server import _CONTROL_MSG_TYPES, _forward_control_msg


class Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_every_sleep_message_round_trips():
    msg = CacheSleepMsg(request_id="a", action="wake")
    assert BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg)) == msg
    backend = CacheSleepBackendMsg(request_id="b", action="sleep")
    assert BaseBackendMsg.decoder(backend.encoder()) == backend
    result = CacheSleepResultMsg(request_id="c", action="sleep", status="ok", asleep=True,
                                 released_bytes=24 << 30, vram_free_bytes=23 << 30, elapsed_s=6.5)
    assert BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(result)) == result
    reply = CacheSleepReply(request_id="d", action="wake", status="rejected", asleep=True,
                            error="the graphics card has 2.0 GB free")
    assert BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply)) == reply


def test_the_tokenizer_worker_forwards_both_directions_field_by_field():
    assert CacheSleepMsg in _CONTROL_MSG_TYPES and CacheSleepResultMsg in _CONTROL_MSG_TYPES
    backend, frontend = Queue(), Queue()
    assert _forward_control_msg(CacheSleepMsg(request_id="r", action="sleep"), backend, frontend)
    assert backend.items == [CacheSleepBackendMsg(request_id="r", action="sleep")]
    result = CacheSleepResultMsg(request_id="r", action="sleep", status="ok", asleep=True,
                                 released_bytes=7, vram_free_bytes=9, elapsed_s=1.25, error=None)
    assert _forward_control_msg(result, backend, frontend)
    assert frontend.items == [CacheSleepReply(request_id="r", action="sleep", status="ok", asleep=True,
                                              released_bytes=7, vram_free_bytes=9, elapsed_s=1.25)]
