"""The EXL3 launch choke point (kernel/exl3_launch.py): cross-stream serialisation, strict
mode and the trace file. CPU tests drive it through the ``_OPS`` seam with fake streams, so
the ordering rules are checked without a card."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel import exl3_launch as L


class FakeStream:
    def __init__(self, name, log, capturing=False):
        self.name, self.log, self.capturing = name, log, capturing

    def wait_event(self, event):
        self.log.append(("wait", self.name, event.recorded_on))

    def synchronize(self):
        self.log.append(("sync", self.name))

    def __repr__(self):
        return f"FakeStream({self.name})"


class FakeEvent:
    def __init__(self, log):
        self.log, self.recorded_on = log, None

    def record(self, stream):
        self.recorded_on = stream.name
        self.log.append(("record", stream.name))


class FakeOps:
    def __init__(self):
        self.log = []
        self.current = None

    def applies(self, device):
        return True

    def current_stream(self, device):
        return self.current

    def is_capturing(self, stream):
        return stream.capturing

    def current_capturing(self, device):
        return self.current.capturing

    def new_event(self):
        return FakeEvent(self.log)


@pytest.fixture
def ops(monkeypatch):
    fake = FakeOps()
    monkeypatch.setattr(L, "_OPS", fake)
    monkeypatch.delenv(L.STRICT_ENV, raising=False)
    monkeypatch.delenv(L.TRACE_ENV, raising=False)
    L._reset_for_tests()
    yield fake
    L._reset_for_tests()


DEV = torch.device("cuda", 0)


def _go(ops, stream, op="exl3_gemm", label=None, m=4, k=128, n=256, bits=5):
    ops.current = stream
    return L.launch(op, lambda: ops.log.append(("kernel", stream.name, op)), device=DEV,
                    m=m, k=k, n=n, bits=bits, label=label)


def test_same_stream_launches_add_no_waits(ops):
    s = FakeStream("engine", ops.log)
    _go(ops, s)
    _go(ops, s, op="exl3_mgemm")
    assert ops.log == [("kernel", "engine", "exl3_gemm"), ("kernel", "engine", "exl3_mgemm")]


def test_switching_stream_waits_on_the_previous_stream_before_launching(ops):
    engine, sched = FakeStream("engine", ops.log), FakeStream("sched", ops.log)
    _go(ops, engine)
    _go(ops, sched)
    _go(ops, engine)
    assert ops.log == [
        ("kernel", "engine", "exl3_gemm"),
        ("record", "engine"), ("wait", "sched", "engine"), ("kernel", "sched", "exl3_gemm"),
        ("record", "sched"), ("wait", "engine", "sched"), ("kernel", "engine", "exl3_gemm"),
    ]


def test_devices_are_tracked_separately(ops):
    a, b = FakeStream("a", ops.log), FakeStream("b", ops.log)
    ops.current = a
    L.launch("exl3_gemm", lambda: None, device=torch.device("cuda", 0), m=1, k=1, n=1, bits=5)
    ops.current = b
    L.launch("exl3_gemm", lambda: None, device=torch.device("cuda", 1), m=1, k=1, n=1, bits=5)
    assert ops.log == []


def test_fork_inside_one_capture_becomes_a_graph_edge(ops):
    c1 = FakeStream("cap1", ops.log, capturing=True)
    c2 = FakeStream("cap2", ops.log, capturing=True)
    _go(ops, c1)
    _go(ops, c2)
    assert ("record", "cap1") in ops.log and ("wait", "cap2", "cap1") in ops.log


def test_no_wait_between_an_eager_and_a_captured_launch(ops):
    # CUDA forbids a capturing stream waiting on an event recorded outside the capture (needs
    # cudaEventWaitExternal) and the reverse merges the capture: both directions skip.
    eager = FakeStream("engine", ops.log)
    cap = FakeStream("capture", ops.log, capturing=True)
    _go(ops, eager)
    _go(ops, cap)
    cap.capturing = False  # capture ended
    other = FakeStream("other", ops.log, capturing=True)
    _go(ops, other)
    assert [e for e in ops.log if e[0] in ("record", "wait")] == []


def test_a_capture_in_between_does_not_hide_the_last_eager_stream(ops):
    # review minor 1: eager on E -> captured on S -> eager on X must make X wait on E
    e = FakeStream("E", ops.log)
    s = FakeStream("S", ops.log, capturing=True)
    x = FakeStream("X", ops.log)
    _go(ops, e)
    _go(ops, s)
    s.capturing = False
    _go(ops, x)
    assert ("record", "E") in ops.log and ("wait", "X", "E") in ops.log
    assert ("record", "S") not in ops.log


def test_eager_launch_does_not_wait_on_a_stream_that_is_capturing_now(ops):
    engine = FakeStream("engine", ops.log)
    _go(ops, engine)
    engine.capturing = True
    _go(ops, engine)  # captured on engine
    other = FakeStream("other", ops.log)
    _go(ops, other)  # eager while engine captures: recording on engine would be captured
    assert [e for e in ops.log if e[0] in ("record", "wait")] == []


def test_strict_accepts_a_fenced_side_stream_warmup(ops, monkeypatch, tmp_path):
    # review Important: the MTP graph warm-ups run eager on the capture side stream, fenced
    # both ways; strict must not raise there (the capture's except Exception swallowed it).
    monkeypatch.setenv(L.STRICT_ENV, "1")
    path = tmp_path / "t"
    monkeypatch.setenv(L.TRACE_ENV, str(path))
    engine, side = FakeStream("engine", ops.log), FakeStream("side", ops.log)
    _go(ops, engine)
    with L.fenced_side_stream("mtp-verify-graph-warmup"):
        _go(ops, side)
    assert ("kernel", "side", "exl3_gemm") in ops.log
    text = path.read_text()
    assert " FENCED " in text and "fence=mtp-verify-graph-warmup" in text
    _go(ops, engine)  # home stays engine
    with pytest.raises(RuntimeError):
        _go(ops, side)  # outside the fence it is a violation again


def test_a_fenced_block_does_not_claim_the_home_stream(ops, monkeypatch):
    monkeypatch.setenv(L.STRICT_ENV, "1")
    side, engine = FakeStream("side", ops.log), FakeStream("engine", ops.log)
    with L.fenced_side_stream("warm"):
        _go(ops, side)
    _go(ops, engine)
    with pytest.raises(RuntimeError):
        _go(ops, side)


def test_strict_violation_is_logged_at_error_before_raising(ops, monkeypatch):
    monkeypatch.setenv(L.STRICT_ENV, "1")
    errors = []
    monkeypatch.setattr(L.logger, "error", lambda msg, *a: errors.append(msg % a if a else msg))
    _go(ops, FakeStream("engine", ops.log))
    try:
        _go(ops, FakeStream("sched", ops.log), label="x")
    except RuntimeError:
        pass  # a broad except in the caller must still leave the log line
    assert len(errors) == 1 and "second CUDA stream" in errors[0] and "label=x" in errors[0]


def test_the_mtp_graph_warmups_are_fenced():
    import inspect

    from freetoken.engine import spec_draft_graph, spec_graph

    assert 'fenced_side_stream("mtp-verify-graph-warmup")' in inspect.getsource(spec_graph)
    assert 'fenced_side_stream("mtp-draft-graph-warmup")' in inspect.getsource(spec_draft_graph)


def test_cpu_tensors_bypass_the_stream_rules():
    calls = []
    L.launch("exl3_gemm", lambda: calls.append(1), device=torch.device("cpu"),
             m=1, k=1, n=1, bits=5)
    assert calls == [1]


def test_strict_mode_rejects_a_second_eager_stream_before_launching(ops, monkeypatch):
    monkeypatch.setenv(L.STRICT_ENV, "1")
    engine, sched = FakeStream("engine", ops.log), FakeStream("sched", ops.log)
    _go(ops, engine)
    with L.scope("picture"), pytest.raises(RuntimeError) as err:
        _go(ops, sched, label="blocks.0.mlp.linear_fc1", m=64, k=1152, n=4352)
    msg = str(err.value)
    assert "exl3_gemm" in msg and "picture/blocks.0.mlp.linear_fc1" in msg
    assert "m=64 k=1152 n=4352" in msg
    assert ("kernel", "sched", "exl3_gemm") not in ops.log
    _go(ops, engine)  # the home stream keeps working


def test_strict_mode_allows_captured_launches_on_the_capture_stream(ops, monkeypatch):
    monkeypatch.setenv(L.STRICT_ENV, "1")
    engine = FakeStream("engine", ops.log)
    cap = FakeStream("capture", ops.log, capturing=True)
    _go(ops, engine)
    _go(ops, cap)
    assert ("kernel", "capture", "exl3_gemm") in ops.log


def test_strict_off_by_default_serialises_instead(ops):
    engine, sched = FakeStream("engine", ops.log), FakeStream("sched", ops.log)
    _go(ops, engine)
    _go(ops, sched)
    assert ("kernel", "sched", "exl3_gemm") in ops.log


def test_trace_writes_begin_end_and_syncs_eager_launches(ops, monkeypatch, tmp_path):
    path = tmp_path / "exl3.trace"
    monkeypatch.setenv(L.TRACE_ENV, str(path))
    engine = FakeStream("engine", ops.log)
    with L.scope("L3.routed"):
        _go(ops, engine, op="exl3_mgemm", label="mgemm.gate", m=30, k=2560, n=640, bits=3)
    # the END line is written only after the stream synchronised (the kernel finished)
    assert ops.log == [("kernel", "engine", "exl3_mgemm"), ("sync", "engine")]
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert " BEGIN op=exl3_mgemm label=L3.routed/mgemm.gate m=30 k=2560 n=640 bits=3 " in lines[0]
    assert "capturing=no" in lines[0] and "stream=" in lines[0] and "pid=" in lines[0]
    assert " END op=exl3_mgemm " in lines[1] and "ms=" in lines[1]
    float(lines[0].split()[0])  # monotonic timestamp first


def test_trace_does_not_sync_while_capturing(ops, monkeypatch, tmp_path):
    path = tmp_path / "exl3.trace"
    monkeypatch.setenv(L.TRACE_ENV, str(path))
    _go(ops, FakeStream("capture", ops.log, capturing=True))
    assert ("sync", "capture") not in ops.log
    assert "capturing=yes" in path.read_text()


def test_trace_leaves_a_begin_without_end_when_the_launch_fails(ops, monkeypatch, tmp_path):
    path = tmp_path / "exl3.trace"
    monkeypatch.setenv(L.TRACE_ENV, str(path))
    ops.current = FakeStream("engine", ops.log)

    def boom():
        raise ValueError("bad shape")

    with pytest.raises(ValueError):
        L.launch("exl3_gemm", boom, device=DEV, m=1, k=1, n=1, bits=5, label="x")
    text = path.read_text()
    assert " BEGIN " in text and " FAIL " in text and " END " not in text


def test_trace_logs_the_cross_stream_wait(ops, monkeypatch, tmp_path):
    path = tmp_path / "exl3.trace"
    monkeypatch.setenv(L.TRACE_ENV, str(path))
    _go(ops, FakeStream("engine", ops.log))
    _go(ops, FakeStream("sched", ops.log))
    assert " WAIT " in path.read_text()


def test_scopes_nest_and_unwind():
    with L.scope("a"):
        with L.scope("b"):
            assert L._full_label("c") == "a/b/c"
        assert L._full_label(None) == "a"
    assert L._full_label(None) == "?"


# --------------------------------------------------------------------------------------
# Wiring: every ExLlamaV3 GPU launch FreeToken makes reaches the choke point, labelled.


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_launch(op, fn, *, device, m, k, n, bits, label=None):
        calls.append({"op": op, "device": torch.device(device), "m": m, "k": k, "n": n,
                      "bits": bits, "label": L._full_label(label)})
        return fn()

    monkeypatch.setattr(L, "launch", fake_launch)
    return calls


def test_dense_gemm_goes_through_the_choke_point(monkeypatch, recorded):
    from freetoken.kernel import exl3_linear as el

    wheel = []

    class Ext:
        @staticmethod
        def exl3_gemm(*args):
            wheel.append(len(recorded))  # called INSIDE launch, after it was recorded

    monkeypatch.setattr(el, "_load_ext", lambda: Ext)
    x16 = torch.zeros(3, 128, dtype=torch.float16)
    trellis = torch.zeros(8, 16, 48, dtype=torch.int16)
    y16 = torch.zeros(3, 256, dtype=torch.float16)
    with L.scope("model.layers.0.attn.q_proj"):
        el._exl3_gemm(x16, trellis, y16, torch.zeros(128).half(), x16.clone(), torch.zeros(256).half())
    assert wheel == [1]
    assert recorded == [{"op": "exl3_gemm", "device": torch.device("cpu"), "m": 3, "k": 128,
                         "n": 256, "bits": 3, "label": "model.layers.0.attn.q_proj"}]


def test_reconstruct_goes_through_the_choke_point(monkeypatch, recorded):
    from freetoken.kernel import exl3 as exl3_kernel

    wheel = []

    class Ext:
        @staticmethod
        def reconstruct_had_slice(work, trellis, suh, svh, k, _flag, mul1, _zero):
            wheel.append((k, mul1))

    monkeypatch.setattr(exl3_kernel, "_exllamav3_ext", Ext)
    work = torch.zeros(128, 256, dtype=torch.float16)
    with L.scope("L2.routed.decode"), L.scope("reconstruct.down"):
        exl3_kernel._reconstruct_had_slice(work, None, None, None, k=2, mul1=True)
    assert wheel == [(2, True)]
    assert recorded[0]["op"] == "reconstruct" and recorded[0]["label"] == "L2.routed.decode/reconstruct.down"
    assert (recorded[0]["k"], recorded[0]["n"], recorded[0]["bits"]) == (128, 256, 2)


def test_mgemm_goes_through_the_choke_point(monkeypatch, recorded):
    from freetoken.kernel import exl3_mgemm as mg

    wheel = []

    class Ext:
        @staticmethod
        def exl3_gemm_num_kernel_shapes():
            return 1

        @staticmethod
        def exl3_gemm_shape_compat(*args):
            return True

        @staticmethod
        def exl3_mgemm(a, *args):
            wheel.append(tuple(a.shape))

    monkeypatch.setattr(mg, "_load_extension", lambda: Ext)
    slots, H, I, K = 2, 128, 256, 3
    banks = (
        torch.zeros(slots, H // 16, I // 16, 16 * K, dtype=torch.int16),
        torch.zeros(slots, H, dtype=torch.float16), torch.zeros(slots, I, dtype=torch.float16),
        torch.zeros(slots, H // 16, I // 16, 16 * K, dtype=torch.int16),
        torch.zeros(slots, H, dtype=torch.float16), torch.zeros(slots, I, dtype=torch.float16),
        torch.zeros(slots, I // 16, H // 16, 16 * K, dtype=torch.int16),
        torch.zeros(slots, I, dtype=torch.float16), torch.zeros(slots, H, dtype=torch.float16),
    )
    ptrs = tuple(torch.zeros(slots, dtype=torch.int64) for _ in banks)
    tables = mg.Exl3MgemmBanks(banks, ptrs, k=K)
    with L.scope("L5.routed.prefill"):
        mg.exl3_mgemm_projection(torch.zeros(4, H, dtype=torch.bfloat16), tables,
                                 torch.zeros(1, dtype=torch.int64), projection="up")
    assert wheel == [(1, 4, H)]
    assert recorded == [{"op": "exl3_mgemm", "device": torch.device("cpu"), "m": 4, "k": H,
                         "n": I, "bits": K, "label": "L5.routed.prefill/mgemm.up"}]


def test_routed_experts_label_their_layer_and_phase(monkeypatch):
    from freetoken.moe import fused_exl3

    seen = []
    monkeypatch.setattr(fused_exl3, "_fused_experts_exl3",
                        lambda *a, **kw: seen.append((L._full_label(None), kw["layer_id"])) or "out")
    got = fused_exl3.fused_experts_exl3(None, (), None, None, is_prefill=True, activation="silu",
                                        apply_router_weight_on_input=False, swiglu_limit=None,
                                        hidden_act_alpha=1.0, scratch=None, layer_id=7)
    assert got == "out" and seen == [("L7.routed.prefill", 7)]


def _labelled_tree():
    from freetoken.kernel.exl3_linear import Exl3ColMerged, Exl3Linear
    from freetoken.layers.base import BaseOP, OPList

    class Attn(BaseOP):
        def __init__(self):
            self.qkv = Exl3ColMerged(128, [("q", 128), ("k", 128)])
            self.o_proj = Exl3Linear(256, 128)

    class Layer(BaseOP):
        def __init__(self):
            self.attn = Attn()

    class Model(BaseOP):
        def __init__(self):
            self.layers = OPList([Layer(), Layer()])
            self.lm_head = Exl3Linear(128, 256)

    return Model()


def test_workspace_preparation_labels_every_dense_op_with_its_module_path():
    from freetoken.kernel import exl3_linear as el

    model = _labelled_tree()
    el.prepare_exl3_dense_workspace(model, torch.device("cpu"), _reset=True)
    try:
        assert model.layers.op_list[1].attn.qkv.k.label == "layers.1.attn.qkv.k"
        assert model.layers.op_list[0].attn.o_proj.label == "layers.0.attn.o_proj"
        assert model.lm_head.label == "lm_head"
        # labels are not state: they must never show up as weights
        assert all(not key.endswith("label") for key in model.state_dict())
    finally:
        el._WORKSPACES.pop(torch.device("cpu"), None)


def test_dense_gemm_launch_carries_the_op_label(monkeypatch):
    from freetoken.kernel import exl3_linear as el

    model = _labelled_tree()
    el.prepare_exl3_dense_workspace(model, torch.device("cpu"), _reset=True)
    seen = []
    monkeypatch.setattr(el, "_exl3_gemm", lambda *a: seen.append(L._full_label(None)))
    try:
        model.layers.op_list[1].attn.o_proj._gemm(torch.zeros(2, 256, dtype=torch.bfloat16))
    finally:
        el._WORKSPACES.pop(torch.device("cpu"), None)
    assert seen == ["layers.1.attn.o_proj"]


# --- real card: the _CudaOps event/capture path (the fakes above cannot prove CUDA accepts it) ---

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture
def real_ops():
    L._reset_for_tests()
    yield
    L._reset_for_tests()


def _spin(ms: int) -> None:
    # ~ms of GPU busy time on the current stream (torch.cuda._sleep counts clock cycles).
    torch.cuda._sleep(int(ms * 1.5e6))


@needs_cuda
def test_real_stream_switch_orders_the_new_launch_after_the_old_stream(real_ops):
    device = torch.device("cuda")
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    done_a, done_b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(a):
        L.launch("t", lambda: (_spin(200), done_a.record(a)), device=device, m=1, k=1, n=1, bits=3)
    with torch.cuda.stream(b):
        L.launch("t", lambda: done_b.record(b), device=device, m=1, k=1, n=1, bits=3)
    torch.cuda.synchronize()
    # Without the cross-stream wait, b's record would finish ~200 ms before a's spin ends.
    assert done_a.elapsed_time(done_b) >= 0.0


@needs_cuda
def test_real_capture_after_eager_launch_skips_the_wait_and_replays(real_ops):
    device = torch.device("cuda")
    x = torch.ones(1024, device=device)
    L.launch("t", lambda: x.mul_(1.0), device=device, m=1, k=1, n=1, bits=3)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
        L.launch("t", lambda: x.add_(1.0), device=device, m=1, k=1, n=1, bits=3)
    torch.cuda.current_stream().wait_stream(side)
    graph.replay()
    torch.cuda.synchronize()
    assert float(x[0]) == 2.0  # the captured add runs only at replay


@needs_cuda
def test_real_fork_inside_one_capture_becomes_a_graph_edge(real_ops):
    device = torch.device("cuda")
    x = torch.zeros(1024, device=device)
    cap = torch.cuda.Stream()
    fork = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(cap), torch.cuda.graph(graph, stream=cap):
        L.launch("t", lambda: x.add_(1.0), device=device, m=1, k=1, n=1, bits=3)
        fork.wait_stream(cap)
        with torch.cuda.stream(fork):
            L.launch("t", lambda: x.mul_(2.0), device=device, m=1, k=1, n=1, bits=3)
        cap.wait_stream(fork)  # join, or capture end fails
    graph.replay()
    torch.cuda.synchronize()
    assert float(x[0]) == 2.0


@needs_cuda
def test_real_strict_mode_raises_on_a_second_eager_stream(real_ops, monkeypatch, tmp_path):
    monkeypatch.setenv(L.STRICT_ENV, "1")
    monkeypatch.setenv(L.TRACE_ENV, str(tmp_path / "trace.log"))
    device = torch.device("cuda")
    L.launch("t", lambda: None, device=device, m=1, k=1, n=1, bits=3)
    with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match="second CUDA stream"):
        L.launch("t", lambda: None, device=device, m=1, k=1, n=1, bits=3)
    text = (tmp_path / "trace.log").read_text()
    assert "BEGIN" in text and "END" in text and "STRICT-FAIL" in text
