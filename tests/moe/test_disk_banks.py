"""Tests for ExpertDiskCopy, DiskLayerReader, BackgroundWriter, and disk-fed MoE (J2).

Covers:
  1. Manifest round trip, serialization, and identity mismatch detection.
  2. write_layer -> read_rows returns byte-identical rows for random expert IDs on bf16 bank.
  3. Spill -> disk_gather -> GEMM output matches pinned-path output on CPU dummies.
  4. Refusal to spill an incomplete layer.
  5. Recall (disk -> pinned) restores host bank tensor equal to original.
  6. can_use_cuda_graph returns False when disk layers are present; GraphRunner defers capture.
  7. GPU decode step with disk layer (@pytest.mark.slow, skips on CPU).
  8. Staging read throughput measurement at 64-row chunks (MB/s).
  9. RAM axis stepping in step_memory (lowest routing / highest pinned; reverse spill recall)
     and ram_tight=True VRAM ladder branch (gpu_owned -> disk).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from freetoken.engine.engine import Engine
from freetoken.engine.graph import GraphRunner
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.disk_banks import DiskLayerReader, ExpertDiskCopy
from freetoken.moe.host_banks import HostBank, HostResidency
from freetoken.moe.offload_cache import OffloadMoeCache


def _init_tp() -> None:
    if not torch.distributed.is_initialized():
        import tempfile

        f = tempfile.NamedTemporaryFile(delete=False)
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{f.name}",
            rank=0,
            world_size=1,
        )


def _make_dummy_checkpoint(model_dir: Path, num_shards: int = 2) -> list[Path]:
    shards = []
    for i in range(num_shards):
        p = model_dir / f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        p.write_bytes(b"dummy_shard_bytes_" + str(i).encode("utf-8") * 100)
        shards.append(p)
    return shards


class FakeDiskEngine:
    """Harness for testing _move_layer and step_memory with disk rungs."""

    def __init__(
        self,
        num_layers: int = 4,
        num_experts: int = 64,
        cache_size: int = 128,
        owned_layers: tuple[int, ...] = (0,),
        model_dir: Path | None = None,
        disk_dir: Path | None = None,
    ):
        _init_tp()
        self.device = torch.device("cpu")
        self.model_dir = model_dir or Path("/tmp/dummy_model")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        _make_dummy_checkpoint(self.model_dir)

        self.config = SimpleNamespace(
            model_config=SimpleNamespace(
                num_moe_layers=num_layers,
                num_experts=num_experts,
                vocab_size=100,
            ),
            model_path=str(self.model_dir),
            moe_backend="offload",
            moe_prefill_overlap=False,
            moe_cache_size=cache_size,
            cuda_graph_max_bs=1,
            max_running_req=4,
            page_size=16,
            max_seq_len=1024,
            memory_ratio=0.9,
            tp_info=SimpleNamespace(rank=0, size=1),
        )
        self.num_pages = 100
        self._initial_num_pages = 100
        self._initial_moe_cache_size = cache_size
        self._gpu_owned_layer_ids = frozenset(owned_layers)
        self._host_banks: dict[int, dict[str, HostBank]] = {}
        self._ram_spilled_layers: list[int] = []
        self._demoted_layers: list[int] = []
        self._vram_ledger_inputs = None

        # Build OffloadMoeCache
        self.moe_offload_cache = OffloadMoeCache(
            num_layers=num_layers,
            num_experts=num_experts,
            cache_size=cache_size,
            device=self.device,
        )
        self.bank_schema = ("gate_up", "down")
        self.shapes = {
            "gate_up": (num_experts, 32, 16),
            "down": (num_experts, 16, 32),
        }
        self.dtypes = {
            "gate_up": torch.bfloat16,
            "down": torch.bfloat16,
        }
        sources = {
            "gate_up": [
                torch.randn(self.shapes["gate_up"], dtype=torch.bfloat16)
                for _ in range(num_layers)
            ],
            "down": [
                torch.randn(self.shapes["down"], dtype=torch.bfloat16)
                for _ in range(num_layers)
            ],
        }
        residency = [
            HostResidency.GPU_OWNED.value if i in self._gpu_owned_layer_ids else HostResidency.PINNED.value
            for i in range(num_layers)
        ]
        self.moe_offload_cache.set_bank_sources(
            sources, layer_residency=residency, gpu_owned_layers=self._gpu_owned_layer_ids
        )

        disk_path = disk_dir or (self.model_dir / "freetoken-expert-cache")
        self.expert_disk_copy = ExpertDiskCopy(
            root=disk_path,
            model_path=self.model_dir,
            schema=self.bank_schema,
            shapes=self.shapes,
            dtypes=self.dtypes,
        )
        self.moe_offload_cache.expert_disk_copy = self.expert_disk_copy
        self.rebuild_runtime_cache = MagicMock(side_effect=self._mock_rebuild)

    def _stash_vram_ledger_inputs(self, sources, owned):
        self._vram_ledger_inputs = (sources, owned)

    def _sync_get_memory(self):
        return (10 << 30, 20 << 30)

    def _mock_rebuild(self, *, moe_cache_size=None, num_pages=None, layer_moves=None, **kwargs):
        if layer_moves:
            for lid, tgt in layer_moves:
                Engine._move_layer(self, lid, tgt)
        if moe_cache_size is not None:
            self.moe_offload_cache.rebuild(moe_cache_size)

    def step_memory(self, *args, **kwargs):
        return Engine.step_memory(self, *args, **kwargs)

    def _move_layer(self, layer_id: int, target: str):
        return Engine._move_layer(self, layer_id, target)


# --------------------------------------------------------------------------------------
# 1. Manifest round trip and identity mismatch
# --------------------------------------------------------------------------------------

def test_manifest_round_trip_and_identity_mismatch(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shards = _make_dummy_checkpoint(model_dir)

    disk_dir = tmp_path / "disk_cache"
    schema = ("gate_up", "down")
    shapes = {"gate_up": (16, 8, 4), "down": (16, 4, 8)}
    dtypes = {"gate_up": torch.bfloat16, "down": torch.bfloat16}

    # Initial creation writes manifest
    copy1 = ExpertDiskCopy(disk_dir, model_dir, schema, shapes, dtypes)
    assert copy1.manifest_path.is_file()
    assert copy1.manifest["schema"] == list(schema)
    assert not copy1.layer_complete(0)

    # Fake writing layer 0
    tensors = {
        "gate_up": torch.randn(shapes["gate_up"], dtype=torch.bfloat16),
        "down": torch.randn(shapes["down"], dtype=torch.bfloat16),
    }
    copy1.write_layer(0, tensors)
    assert copy1.layer_complete(0)

    # Reload from existing directory: layer 0 must stay complete
    copy2 = ExpertDiskCopy(disk_dir, model_dir, schema, shapes, dtypes)
    assert copy2.layer_complete(0)

    # Shard modification triggers identity mismatch and invalidates completed layers
    time.sleep(0.01)
    shards[0].write_bytes(b"modified_shard_bytes_different_size_and_mtime")
    copy3 = ExpertDiskCopy(disk_dir, model_dir, schema, shapes, dtypes)
    assert not copy3.layer_complete(0)


# --------------------------------------------------------------------------------------
# 2. write_layer -> read_rows byte-identical for random expert IDs
# --------------------------------------------------------------------------------------

def test_write_and_read_rows_byte_identical(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _make_dummy_checkpoint(model_dir)

    disk_dir = tmp_path / "disk_cache"
    num_experts = 128
    schema = ("gate_up", "down")
    shapes = {"gate_up": (num_experts, 32, 16), "down": (num_experts, 16, 32)}
    dtypes = {"gate_up": torch.bfloat16, "down": torch.bfloat16}

    copy = ExpertDiskCopy(disk_dir, model_dir, schema, shapes, dtypes)
    banks = {
        "gate_up": torch.randn(shapes["gate_up"], dtype=torch.bfloat16),
        "down": torch.randn(shapes["down"], dtype=torch.bfloat16),
    }
    copy.write_layer(0, banks)
    assert copy.layer_complete(0)

    # Allocate staging buffer (64 rows)
    staging = {
        name: torch.empty((64, *shapes[name][1:]), dtype=dtypes[name])
        for name in schema
    }
    reader = DiskLayerReader(copy, layer_id=0, staging=staging)

    # Query random expert IDs
    g = torch.Generator().manual_seed(42)
    sample_ids = torch.randperm(num_experts, generator=g)[:10].tolist()

    views = reader.read_rows(sample_ids)
    for idx, exp_id in enumerate(sample_ids):
        assert torch.equal(views[0][idx], banks["gate_up"][exp_id])
        assert torch.equal(views[1][idx], banks["down"][exp_id])

    reader.close()


# --------------------------------------------------------------------------------------
# 3. Spill -> disk_gather -> GEMM output matches pinned path on CPU dummy
# --------------------------------------------------------------------------------------

def test_spill_disk_gather_gemm_output_matches_pinned(tmp_path: Path):
    _init_tp()
    num_experts = 32
    num_layers = 2
    H, I = 16, 32

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _make_dummy_checkpoint(model_dir)
    disk_dir = tmp_path / "disk_cache"

    schema = ("gate_up", "down")
    shapes = {"gate_up": (num_experts, 2 * I, H), "down": (num_experts, H, I)}
    dtypes = {"gate_up": torch.bfloat16, "down": torch.bfloat16}

    copy = ExpertDiskCopy(disk_dir, model_dir, schema, shapes, dtypes)

    dev = torch.device("cpu")
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=num_experts,
        device=dev,
    )
    cache.expert_disk_copy = copy

    orig_banks = {
        "gate_up": [torch.randn(shapes["gate_up"], dtype=torch.bfloat16) for _ in range(num_layers)],
        "down": [torch.randn(shapes["down"], dtype=torch.bfloat16) for _ in range(num_layers)],
    }
    cache.set_bank_sources(orig_banks, layer_residency=[HostResidency.PINNED.value] * num_layers)

    # Write layer 0 to disk copy
    copy.write_layer(0, {n: orig_banks[n][0] for n in schema})
    assert copy.layer_complete(0)

    # Create OffloadMoELayer
    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    layer.layer_id = 0
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.top_k = 4
    layer.offload_cache = cache

    # Dummy inputs for 2 tokens
    hidden = torch.randn(2, H, dtype=torch.bfloat16)
    weights = torch.tensor([[0.6, 0.4, 0.0, 0.0], [0.5, 0.3, 0.2, 0.0]], dtype=torch.float32)
    topk_ids = torch.tensor([[3, 15, 0, 7], [15, 2, 8, 31]], dtype=torch.int32)

    # Define a clean reference GEMM that uses views indexed by topk_ids
    def reference_gemm(cache_obj, hidden_in, topk_w, ids_in, *, views, **kwargs):
        B, K = ids_in.shape
        out = torch.zeros_like(hidden_in)
        for b in range(B):
            for k in range(K):
                slot = ids_in[b, k].item()
                w = topk_w[b, k].item()
                gu = views[0][slot].float()
                dn = views[1][slot].float()
                # Simple projection
                mid = hidden_in[b].float() @ gu[:I, :].t()
                res = mid @ dn.t()
                out[b] += w * res.to(torch.bfloat16)
        return out

    layer._expert_gemm = reference_gemm

    # Path A: Pinned path. The expert weights come from the pinned bank sources.
    pinned_views = tuple(orig_banks[n][0] for n in schema)
    out_pinned = layer._expert_gemm(
        cache, hidden, weights, topk_ids, views=pinned_views, n=None, alphas=None, is_prefill=False
    )

    # Path B: Spill layer 0 to DISK
    cache.rebind_layer(0, HostResidency.DISK.value, None)
    assert cache.is_disk_layer(0)
    assert cache.has_disk_layers

    # Disk decode via _decode_routed
    out_disk = layer._decode_routed(hidden, weights, topk_ids)

    # Bitwise identical
    assert torch.equal(out_pinned, out_disk)


# --------------------------------------------------------------------------------------
# 4. Refuse spill on incomplete layer
# --------------------------------------------------------------------------------------

def test_refuse_spill_on_incomplete_layer(tmp_path: Path):
    eng = FakeDiskEngine(num_layers=3, model_dir=tmp_path / "model", disk_dir=tmp_path / "disk")
    # Layer 1 is not written to disk yet
    assert not eng.expert_disk_copy.layer_complete(1)

    with pytest.raises(RuntimeError, match="incomplete"):
        eng._move_layer(1, "disk")


# --------------------------------------------------------------------------------------
# 5. Recall restores pinned bank equal to original
# --------------------------------------------------------------------------------------

def test_recall_restores_pinned_bank_equal_to_original(tmp_path: Path):
    eng = FakeDiskEngine(num_layers=3, model_dir=tmp_path / "model", disk_dir=tmp_path / "disk")
    # Write layer 1 to disk
    banks = {n: eng.moe_offload_cache.bank_sources[n][1] for n in eng.bank_schema}
    orig_gate_up = banks["gate_up"].clone()
    orig_down = banks["down"].clone()
    eng.expert_disk_copy.write_layer(1, banks)

    # Spill to disk
    eng._move_layer(1, "disk")
    assert eng.moe_offload_cache.layer_residency[1] == "disk"

    # Recall to pinned
    eng._move_layer(1, "pinned")
    assert eng.moe_offload_cache.layer_residency[1] == "pinned"
    recalled_gu = eng._host_banks[1]["gate_up"].tensor
    recalled_dn = eng._host_banks[1]["down"].tensor

    assert torch.equal(recalled_gu, orig_gate_up)
    assert torch.equal(recalled_dn, orig_down)


# --------------------------------------------------------------------------------------
# 6. can_use_cuda_graph False with disk layer
# --------------------------------------------------------------------------------------

def test_can_use_cuda_graph_false_with_disk_layer(tmp_path: Path):
    eng = FakeDiskEngine(num_layers=3, model_dir=tmp_path / "model", disk_dir=tmp_path / "disk")
    banks = {n: eng.moe_offload_cache.bank_sources[n][1] for n in eng.bank_schema}
    eng.expert_disk_copy.write_layer(1, banks)

    dummy_runner = GraphRunner.__new__(GraphRunner)
    dummy_runner.cuda_graph_bs = [1, 2, 4]
    dummy_runner.max_graph_bs = 4
    dummy_runner.moe_offload_cache = eng.moe_offload_cache
    dummy_runner.graph_map = {1: MagicMock(), 2: MagicMock()}

    batch = SimpleNamespace(is_decode=True, size=1)

    # Before spill: no disk layers
    assert not eng.moe_offload_cache.has_disk_layers
    assert dummy_runner.can_use_cuda_graph(batch)

    # After spill: has disk layers
    eng._move_layer(1, "disk")
    assert eng.moe_offload_cache.has_disk_layers
    assert not dummy_runner.can_use_cuda_graph(batch)


# --------------------------------------------------------------------------------------
# 7. Slow-marked GPU test (skips on CPU devbox)
# --------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_decode_step_with_disk_layer(tmp_path: Path):
    _init_tp()
    dev = torch.device("cuda")
    num_experts = 16
    H, I = 32, 64

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _make_dummy_checkpoint(model_dir)
    disk_dir = tmp_path / "disk_cache"

    schema = ("gate_up", "down")
    shapes = {"gate_up": (num_experts, 2 * I, H), "down": (num_experts, H, I)}
    dtypes = {"gate_up": torch.bfloat16, "down": torch.bfloat16}

    copy = ExpertDiskCopy(disk_dir, model_dir, schema, shapes, dtypes)
    cache = OffloadMoeCache(num_layers=1, num_experts=num_experts, cache_size=num_experts, device=dev)
    cache.expert_disk_copy = copy

    banks = {
        "gate_up": [torch.randn(shapes["gate_up"], dtype=torch.bfloat16, device=dev)],
        "down": [torch.randn(shapes["down"], dtype=torch.bfloat16, device=dev)],
    }
    cache.set_bank_sources(banks, layer_residency=["pinned"])
    copy.write_layer(0, {n: banks[n][0] for n in schema})

    cache.rebind_layer(0, "disk", None)
    assert cache.has_disk_layers

    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    layer.layer_id = 0
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.top_k = 2

    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=dev)
    weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32, device=dev)
    topk_ids = torch.tensor([[1, 5]], dtype=torch.int32, device=dev)

    out = layer._decode_routed(hidden, weights, topk_ids)
    assert out.shape == hidden.shape
    assert out.device == dev


# --------------------------------------------------------------------------------------
# 8. Measure staging read throughput in MB/s at 64-row chunks
# --------------------------------------------------------------------------------------

def test_staging_read_throughput_benchmark(tmp_path: Path):
    """Measures staging read throughput in MB/s using 64-row chunks on CPU."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _make_dummy_checkpoint(model_dir)

    disk_dir = tmp_path / "disk_cache"
    # Sized dummy: 128 rows, gate_up [128, 512, 256], down [128, 256, 512]
    # Each row is (512*256*2) + (256*512*2) = 524,288 bytes = 0.5 MB
    # 64 rows = 32 MB
    num_experts = 128
    schema = ("gate_up", "down")
    shapes = {"gate_up": (num_experts, 512, 256), "down": (num_experts, 256, 512)}
    dtypes = {"gate_up": torch.bfloat16, "down": torch.bfloat16}

    copy = ExpertDiskCopy(disk_dir, model_dir, schema, shapes, dtypes)
    banks = {
        "gate_up": torch.randn(shapes["gate_up"], dtype=torch.bfloat16),
        "down": torch.randn(shapes["down"], dtype=torch.bfloat16),
    }
    copy.write_layer(0, banks)
    assert copy.layer_complete(0)

    staging = {
        name: torch.empty((64, *shapes[name][1:]), dtype=dtypes[name])
        for name in schema
    }
    reader = DiskLayerReader(copy, layer_id=0, staging=staging, max_workers=4)

    # Warmup
    for _ in reader.iter_layer_chunks(rows=64):
        pass

    # Timed run: read 10 full passes (20 chunks of 64 rows = 640 MB)
    passes = 10
    total_bytes = passes * sum(copy.bank_bytes(n) for n in schema)

    t0 = time.perf_counter()
    for _ in range(passes):
        for start_row, chunk_size, chunk_tup in reader.iter_layer_chunks(rows=64):
            pass
    dt = time.perf_counter() - t0

    mb_per_s = (total_bytes / 1e6) / dt
    print(f"\n[MEASUREMENT] Staging read throughput at 64-row chunks: {mb_per_s:.1f} MB/s ({total_bytes / 1e6:.1f} MB in {dt:.3f} s)")
    assert mb_per_s > 50.0  # Basic sanity check on modern SSD / tmpfs
    reader.close()


# --------------------------------------------------------------------------------------
# 9. RAM axis stepping and ram_tight in step_memory
# --------------------------------------------------------------------------------------

def test_step_memory_ram_axis_and_ram_tight(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    disk_dir = tmp_path / "disk_cache"

    eng = FakeDiskEngine(
        num_layers=4,
        num_experts=64,
        cache_size=128,
        owned_layers=(0,),  # layer 0 is gpu_owned; 1, 2, 3 are pinned
        model_dir=model_dir,
        disk_dir=disk_dir,
    )

    # Make layers 0, 1, 2, 3 complete on disk
    for lid in range(4):
        banks = {n: eng.moe_offload_cache.bank_sources[n][lid] for n in eng.bank_schema}
        eng.expert_disk_copy.write_layer(lid, banks)

    # 1. RAM down without learned stats: picks highest pinned layer (layer 3)
    r1 = eng.step_memory(axis="ram", direction="down")
    assert r1["applied"] == "pinned->disk"
    assert r1["layer"] == 3
    assert not r1["at_floor"]
    assert eng.moe_offload_cache.layer_residency[3] == "disk"

    # 2. RAM down with learned routing stats file:
    # Create routing stats where layer 1 has fewest routes
    stats_path = model_dir / "freetoken-routing-stats.json"
    stats_data = {
        "schema": 1,
        "decode_freq": [
            [100] * 64,  # layer 0
            [1] * 64,    # layer 1 (total 64) -> FEWEST
            [50] * 64,   # layer 2 (total 3200)
            [10] * 64,   # layer 3 (already disk)
        ],
    }
    stats_path.write_text(json.dumps(stats_data), encoding="utf-8")

    r2 = eng.step_memory(axis="ram", direction="down")
    assert r2["applied"] == "pinned->disk"
    assert r2["layer"] == 1
    assert eng.moe_offload_cache.layer_residency[1] == "disk"

    # 3. RAM up recalls in reverse order (layer 1 was spilled last, so layer 1 recalls first)
    r3 = eng.step_memory(axis="ram", direction="up")
    assert r3["applied"] == "disk->pinned"
    assert r3["layer"] == 1
    assert eng.moe_offload_cache.layer_residency[1] == "pinned"

    # Next RAM up recalls layer 3
    r4 = eng.step_memory(axis="ram", direction="up")
    assert r4["applied"] == "disk->pinned"
    assert r4["layer"] == 3
    assert eng.moe_offload_cache.layer_residency[3] == "pinned"

    # 4. VRAM down with ram_tight=True: spills gpu_owned directly to disk
    r5 = eng.step_memory(axis="vram", direction="down", ram_tight=True)
    assert r5["applied"] == "gpu_owned->disk"
    assert r5["layer"] == 0
    assert eng.moe_offload_cache.layer_residency[0] == "disk"

    # 5. VRAM down with ram_tight=False: spills gpu_owned to pinned
    # First recall layer 0 back to gpu_owned
    eng._move_layer(0, "gpu_owned")
    assert eng.moe_offload_cache.layer_residency[0] == "gpu_owned"

    r6 = eng.step_memory(axis="vram", direction="down", ram_tight=False)
    assert r6["applied"] == "gpu_owned->pinned"
    assert r6["layer"] == 0
    assert eng.moe_offload_cache.layer_residency[0] == "pinned"


def test_disk_materialize_layer(tmp_path: Path):
    """Verifies that disk_materialize_layer populates slot cache [0, num_experts) through staging."""
    _init_tp()
    num_experts = 64
    num_layers = 1
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _make_dummy_checkpoint(model_dir)
    disk_dir = tmp_path / "disk_cache"

    schema = ("gate_up", "down")
    shapes = {"gate_up": (num_experts, 32, 16), "down": (num_experts, 16, 32)}
    dtypes = {"gate_up": torch.bfloat16, "down": torch.bfloat16}

    dev = torch.device("cpu")
    copy = ExpertDiskCopy(disk_dir, model_dir, schema, shapes, dtypes)
    cache = OffloadMoeCache(num_layers=1, num_experts=num_experts, cache_size=num_experts, device=dev)
    cache.expert_disk_copy = copy

    orig_banks = {
        "gate_up": [torch.randn(shapes["gate_up"], dtype=torch.bfloat16)],
        "down": [torch.randn(shapes["down"], dtype=torch.bfloat16)],
    }
    cache.set_bank_sources(orig_banks, layer_residency=["pinned"])
    copy.write_layer(0, {n: orig_banks[n][0] for n in schema})

    cache.rebind_layer(0, "disk", None)
    assert cache.is_disk_layer(0)

    # Call disk_materialize_layer
    cache.disk_materialize_layer(0)

    # Check slot cache contents match original banks
    for b_idx, (per_layer, slot_cache) in enumerate(cache.banks):
        name = schema[b_idx]
        assert torch.equal(slot_cache[:num_experts], orig_banks[name][0])

