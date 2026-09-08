"""A cache rebuild must re-arm the speculative verify widths whether or not it could capture
graphs (2026-09-08: the SSD spill deferred the graphs, dropped the runner, and the recall then
had nothing to re-arm, leaving MTP-on decode eager for the rest of the boot)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine import engine as engine_mod
from freetoken.engine.engine import Engine


class _FakeRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.widths = tuple(kwargs["widths"])
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def _engine(widths, runner=None):
    fe = SimpleNamespace(
        ctx=object(), model=object(), attn_backend=object(), device=torch.device("cpu"),
        spec_graph_runner=runner, _spec_graph_widths=tuple(widths),
    )
    fe._rearm_spec_graphs = lambda: Engine._rearm_spec_graphs(fe)
    return fe


def test_rearm_builds_a_fresh_runner_from_the_boot_widths_when_none_is_armed(monkeypatch):
    monkeypatch.setattr(engine_mod, "_SPEC_GRAPH_GUARD_BYTES", 7, raising=True)
    import freetoken.engine.spec_graph as sg

    monkeypatch.setattr(sg, "SpecVerifyGraphRunner", _FakeRunner)
    fe = _engine((1, 2, 3))
    fe._rearm_spec_graphs()
    assert isinstance(fe.spec_graph_runner, _FakeRunner)
    assert fe.spec_graph_runner.widths == (1, 2, 3)
    assert fe.spec_graph_runner.kwargs["guard_bytes"] == 7
    assert fe.spec_graph_runner.kwargs["device"].type == "cpu"


def test_rearm_is_a_no_op_without_boot_widths_or_with_a_live_runner(monkeypatch):
    import freetoken.engine.spec_graph as sg

    monkeypatch.setattr(sg, "SpecVerifyGraphRunner", _FakeRunner)
    fe = _engine(())
    fe._rearm_spec_graphs()
    assert fe.spec_graph_runner is None, "FREETOKEN_MTP_SPEC_GRAPH unset: the step stays eager"
    live = _FakeRunner(widths=(1, 2))
    fe = _engine((1, 2), runner=live)
    fe._rearm_spec_graphs()
    assert fe.spec_graph_runner is live, "an armed runner is left alone"


def test_the_rearm_is_a_statement_of_the_rebuild_body_not_of_a_branch():
    """Structural guard: the call must sit in the method body, outside the has_disk if/else."""
    import ast
    import inspect
    import textwrap

    fn = ast.parse(textwrap.dedent(inspect.getsource(Engine.rebuild_runtime_cache))).body[0]
    assert any(
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
        and getattr(s.value.func, "attr", None) == "_rearm_spec_graphs"
        for s in fn.body
    ), "the re-arm must be a statement of the method body, not of a branch"


def test_a_deferred_engine_runs_the_verify_step_without_its_graph_runner():
    """With a layer on the SSD the verify batch cannot be captured; the runner is left aside."""
    import inspect

    src = inspect.getsource(Engine.speculative_decode_batch)
    assert 'graph_runner = None if getattr(self, "_graphs_deferred", None) else self.spec_graph_runner' in src
