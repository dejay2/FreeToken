"""The TTFT accounting seam: every prefill stage opens a ``diag.*`` range, in order.

A 7k-token cold prompt takes 12-17 s to first token while ``diag.prefill_forward`` measures
~5.5 s of it, so the rest of the pipeline is instrumented too. These tests pin what the
ranges are, that they nest/close in pipeline order, that a decode step opens none of them,
and -- the property the profiler's arming rule depends on -- that an idle scheduler opens no
range at all (``freetoken/diag.py``: entering ANY range is what marks an iteration non-idle).
"""

from __future__ import annotations

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from freetoken import diag
from freetoken.core import Batch
from freetoken.engine.engine import Engine
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler

# The stages a first token passes through, in pipeline order.
PREFILL_STAGES = (
    "diag.prefill_tokenize",
    "diag.prefill_admit",
    "diag.prefill_batch",
    "diag.prefill_restore_state",
    "diag.prefill_forward",
    "diag.prefill_sample",
    "diag.prefill_prime_draft",
    "diag.prefill_cache_commit",
    "diag.prefill_emit",
)


@pytest.fixture
def ranges(monkeypatch):
    """Record every range open/close, keeping the real (disabled, no-sync) region object."""
    events: list[str] = []
    real = diag.region

    def region(name):
        inner = real(name)

        class _Recorded:
            def __enter__(self):
                if name is not None:
                    events.append(f"enter {name}")
                inner.__enter__()
                return self

            def __exit__(self, *exc):
                result = inner.__exit__(*exc)
                if name is not None:
                    events.append(f"exit {name}")
                return result

        return _Recorded()

    monkeypatch.setattr(diag, "region", region)
    return events


def _last_data(batch):
    return (
        SimpleNamespace(batch=batch),
        (
            None,
            torch.tensor([42], dtype=torch.int32),
            SimpleNamespace(synchronize=lambda: None),
        ),
    )


class _DrainReq:
    """A drained request: hashable (SimpleNamespace is not) and not a ``ChunkedReq``."""

    def __init__(self, uid: int) -> None:
        self.uid = uid
        self.aborted = False
        self.table_idx = 0


def _drain_stub(commits):
    return SimpleNamespace(
        finished_reqs=set(),
        _spec_record_plain=lambda _batch: None,
        cache_manager=SimpleNamespace(
            lazy_free_region=contextlib.nullcontext,
            cache_req=lambda req, finished: commits.append((req.uid, finished)),
        ),
        _emit_step_tokens=lambda req, tokens: SimpleNamespace(
            finished=False, next_tokens=(7,)
        ),
        _ship_replies=lambda *args, **kwargs: None,
    )


def test_every_named_prefill_stage_exists_in_the_pipeline():
    """The nine stage names are live call sites, not documentation."""
    sources = "\n".join(
        inspect.getsource(module)
        for module in (
            inspect.getmodule(Scheduler),
            inspect.getmodule(PrefillManager),
            inspect.getmodule(Engine),
        )
    )
    for stage in PREFILL_STAGES:
        assert f'"{stage}"' in sources, stage


def test_engine_stages_are_gated_on_the_prefill_phase():
    """Sampling and draft priming are shared with decode; only prefill gets a named range."""
    body = inspect.getsource(Engine.forward_batch)
    for stage in ("diag.prefill_sample", "diag.prefill_prime_draft"):
        assert f'"{stage}" if batch.is_prefill else None' in body, stage


def test_admit_range_stays_shut_while_the_queue_is_empty(ranges):
    """An armed but trafficless server must not burn its step budget on an empty poll."""
    manager = PrefillManager.__new__(PrefillManager)
    manager.pending_list = []
    assert PrefillManager.schedule_next_batch(manager, 128) is None
    assert ranges == []


def test_admit_range_covers_the_admission_work(ranges):
    manager = PrefillManager.__new__(PrefillManager)
    manager.pending_list = [object()]
    manager._admit_next_batch = lambda budget: ("admitted", budget)
    assert PrefillManager.schedule_next_batch(manager, 128) == ("admitted", 128)
    assert ranges == ["enter diag.prefill_admit", "exit diag.prefill_admit"]


@pytest.mark.parametrize("phase,expected", [("prefill", True), ("decode", False)])
def test_batch_construction_range_names_only_the_prefill_phase(ranges, phase, expected):
    batch = SimpleNamespace(is_prefill=phase == "prefill", prompt_admissions=[])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = 99
    scheduler.prefill_manager = SimpleNamespace(schedule_next_batch=lambda budget: batch)
    scheduler.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None)
    scheduler._prepare_batch = lambda value: "forward-input"
    scheduler.send_result = lambda messages: None

    assert Scheduler._schedule_next_batch(scheduler) == "forward-input"
    opened = ["enter diag.prefill_batch", "exit diag.prefill_batch"] if expected else []
    assert ranges == opened


def test_drain_ranges_open_and_close_in_order(ranges):
    """Emit, then the prefix-cache commit, then the reply ship -- each closed before the next."""
    req = _DrainReq(1)
    commits: list = []
    Scheduler._process_last_data(
        _drain_stub(commits), _last_data(Batch(reqs=[req], phase="prefill"))
    )
    assert commits == [(1, False)]
    assert ranges == [
        "enter diag.prefill_emit",
        "exit diag.prefill_emit",
        "enter diag.prefill_cache_commit",
        "exit diag.prefill_cache_commit",
        "enter diag.prefill_emit",
        "exit diag.prefill_emit",
    ]


def test_a_decode_drain_opens_no_prefill_range(ranges):
    req = _DrainReq(1)
    commits: list = []
    Scheduler._process_last_data(
        _drain_stub(commits), _last_data(Batch(reqs=[req], phase="decode"))
    )
    assert commits == []
    assert ranges == []


def test_ranges_are_free_while_the_profiler_is_unset():
    """The disabled seam is the shared singleton -- no allocation on the prefill path."""
    assert not diag.ENABLED
    assert diag.region("diag.prefill_admit") is diag.region(None)
