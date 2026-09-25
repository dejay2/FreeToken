"""The two helpers extracted from rebuild_runtime_cache step 0 and step 4 (shared with wake)."""

from __future__ import annotations

from types import SimpleNamespace

import freetoken.engine.engine as engine_mod
from freetoken.engine.engine import Engine


def test_graph_bs_for_recapture_reads_the_live_runner_then_the_parked_set():
    eng = SimpleNamespace(_graphs_deferred=None, graph_runner=SimpleNamespace(graph_bs_list=[1, 2]))
    assert Engine._graph_bs_for_recapture(eng) == [1, 2] and eng._deferred_graph_bs == [1, 2]
    eng._graphs_deferred, eng.graph_runner = "deferred until no disk layers", SimpleNamespace(graph_bs_list=[])
    assert Engine._graph_bs_for_recapture(eng) == [1, 2]
    no_graphs = SimpleNamespace(_graphs_deferred=None, graph_runner=SimpleNamespace(graph_bs_list=[]))
    assert Engine._graph_bs_for_recapture(no_graphs) == []  # a graphs-off boot stays off


def test_recapture_defers_while_a_layer_is_on_the_ssd(monkeypatch):
    built = []
    monkeypatch.setattr(engine_mod, "GraphRunner", lambda **kw: built.append(kw) or SimpleNamespace(**kw))
    config = SimpleNamespace(page_size=64, cuda_graph_max_bs=4,
                             model_config=SimpleNamespace(vocab_size=10, model_is_mrope=False))
    eng = SimpleNamespace(max_seq_len=4096, stream=None, device=None, model=None, attn_backend=None,
                          dummy_req=None, moe_offload_cache=SimpleNamespace(has_disk_layers=True))
    Engine._recapture_graphs(eng, config, [1, 2], free_min=5)
    assert built[-1]["cuda_graph_bs"] == [] and eng._graphs_deferred == "deferred until no disk layers"
    eng.moe_offload_cache.has_disk_layers = False
    Engine._recapture_graphs(eng, config, [1, 2], free_min=5)
    assert built[-1]["cuda_graph_bs"] == [1, 2] and eng._graphs_deferred is None
    assert built[-1]["max_seq_len"] == 4096 and built[-1]["free_memory"] == 5
