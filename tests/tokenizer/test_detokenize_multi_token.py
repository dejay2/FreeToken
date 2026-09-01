"""One DetokenizeMsg may carry a run of tokens; it must stream exactly what the same
tokens would have streamed one message at a time.

The detokenizer keeps per-uid incremental offsets, so a run has to be appended and
decoded inside a SINGLE message -- N messages sharing a uid in one call would advance
surr_offset/read_offset against the already fully-appended id list and corrupt the delta.
These pin the equivalence over the paths that own those offsets: the printable-text
hold-back, the partial-UTF8 replacement char, the trailing-EOS suppression and the
stop-string hold-back/trim.
"""

from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


_EOS = 9

_PIECES: dict[int, bytes] = {
    1: b"He",
    2: b"llo",
    3: b" wor",
    4: b"ld",
    5: b"!",
    6: b"\xe4\xb8",  # first two bytes of a 3-byte CJK char
    7: b"\x96",  # ... completed by this one
    10: b"a",
    11: b"</",
    12: b"s>",
    13: b"b",
    _EOS: b"<|eos|>",
}


class _ByteTokenizer:
    """Byte-level stand-in for the real tokenizer: a piece is raw bytes and decoding is
    utf-8 with replacement, so a character split across two pieces behaves as it does in
    production (a lone piece decodes to the replacement char)."""

    eos_token_id = _EOS

    def batch_decode(self, seqs):
        return [b"".join(_PIECES[i] for i in seq).decode("utf-8", errors="replace") for seq in seqs]

    def decode(self, seq):
        return self.batch_decode([seq])[0]


def _manager() -> DetokenizeManager:
    return DetokenizeManager(_ByteTokenizer(), frozenset({_EOS}))


def _msg(uid: int, tokens, *, finished: bool = False, **kwargs) -> DetokenizeMsg:
    return DetokenizeMsg(uid=uid, next_tokens=tuple(tokens), finished=finished, **kwargs)


def _stream_one_by_one(tokens, *, finished: bool = False, **kwargs) -> str:
    """The reference: one message per token, the terminal fields on the last one."""
    mgr = _manager()
    out = ""
    for i, token in enumerate(tokens):
        last = i == len(tokens) - 1
        msg = _msg(1, [token], finished=finished and last, **(kwargs if last else _streaming(kwargs)))
        out += mgr.detokenize([msg])[0]
    return out


def _streaming(kwargs: dict) -> dict:
    """Non-terminal messages carry only the request's stop strings."""
    return {"stop_strs": kwargs["stop_strs"]} if "stop_strs" in kwargs else {}


def _stream_in_one_message(tokens, *, finished: bool = False, **kwargs) -> str:
    mgr = _manager()
    return mgr.detokenize([_msg(1, tokens, finished=finished, **kwargs)])[0]


def test_token_run_streams_the_same_text_as_one_message_per_token():
    tokens = [1, 2, 3, 4, 5]
    assert _stream_in_one_message(tokens, finished=True) == "Hello world!"
    assert _stream_one_by_one(tokens, finished=True) == "Hello world!"


def test_a_character_split_across_the_run_is_not_streamed_as_a_replacement_char():
    tokens = [1, 6, 7, 5]
    expected = _stream_one_by_one(tokens, finished=True)
    assert expected == "He世!"
    assert _stream_in_one_message(tokens, finished=True) == expected


def test_only_a_trailing_eos_on_a_finished_run_is_suppressed():
    tokens = [1, 2, _EOS]
    assert _stream_one_by_one(tokens, finished=True) == "Hello"
    assert _stream_in_one_message(tokens, finished=True) == "Hello"


def test_an_unfinished_runs_eos_is_kept_like_any_other_token():
    # ignore_eos requests stream the id as text; suppression is a terminal-message rule.
    assert _stream_in_one_message([1, _EOS, 2], finished=False) == _stream_one_by_one(
        [1, _EOS, 2], finished=False
    )


def test_a_partial_stop_string_at_the_end_of_a_run_is_held_back():
    tokens = [10, 11]
    expected = _stream_one_by_one(tokens, stop_strs=["</s>"])
    assert expected == "a"  # "</" is a proper prefix of the stop string
    assert _stream_in_one_message(tokens, stop_strs=["</s>"]) == expected


def test_a_run_that_completes_a_stop_string_trims_at_the_match():
    # The scheduler truncates the run at the token that completes the stop string, so the
    # completing token is always the last one of a finished message.
    tokens = [10, 11, 12]
    kwargs = {"stop_strs": ["</s>"], "matched_stop": "</s>", "finish_reason": "stop"}
    expected = _stream_one_by_one(tokens, finished=True, **kwargs)
    assert expected == "a"
    assert _stream_in_one_message(tokens, finished=True, **kwargs) == expected


def test_a_finished_run_releases_the_uid_decode_state():
    mgr = _manager()
    mgr.detokenize([_msg(1, [1, 2])])
    assert 1 in mgr.decode_map
    mgr.detokenize([_msg(1, [3, 4], finished=True)])
    assert mgr.decode_map == {}


def test_one_message_per_uid_per_call_across_several_uids():
    mgr = _manager()
    out = mgr.detokenize([_msg(1, [1, 2]), _msg(2, [10, 13])])
    assert out == ["Hello", "ab"]
