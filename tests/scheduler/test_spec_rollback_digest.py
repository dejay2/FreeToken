"""The observer's state-family digest as the Phase-2 rollback oracle.

``MTPShadowObserver._target_state_family_digests`` already enumerates every byte a target
transaction may touch (page-table prefix, GDN conv/recurrent/extra slot states, both QSA
pending rings, the K/V page and the compressed slab page). It runs offline against CPU pools,
so it can gate the scheduler-side rollback without a GPU.

What it proves here: building a w-row speculative batch and rolling it fully back leaves every
enumerated family byte-identical. The load-bearing families for Phase 2 are the page-table
prefix and the request metadata -- an unrestored page-table row would re-point the K/V and
compressed-slab reads at a page the free list has already handed back. The ring/KV families
are unchanged by construction here because no forward runs offline; Phase 3's live gate is
what exercises them.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.engine.config import SpecDecodeConfig
from freetoken.engine.mtp_shadow import MTPShadowObserver, MTPTargetCapture
from freetoken.kvcache import create_kvcache_pool
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.scheduler import Scheduler
from tests.models.qwen4_exp.common import parsed_config

CPU = torch.device("cpu")
PAGE_SIZE = 64
NUM_PAGES = 8
PROMPT_LEN = 64  # cached_len lands on a page boundary, so the step must allocate


def _linear_pool():
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(
        group=group, num_slots=4, dtype=torch.bfloat16, device=CPU, tp_size=1
    )


def _fill(tensor: torch.Tensor) -> None:
    """Distinct bytes everywhere, so a digest over the wrong page cannot collide."""
    flat = tensor.view(-1)
    values = torch.arange(1, flat.numel() + 1, dtype=torch.float32) % 251 + 1
    flat.copy_(values.to(flat.dtype))


def _harness():
    model_config = parsed_config()
    spec = model_config.kv_cache_group_specs()[0]
    kv_pool = create_kvcache_pool(
        model_config=model_config,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device=CPU,
        num_req_slots=2,
        num_speculative_tokens=3,
    )
    assert kv_pool.ring_capacity == 8
    layer_id = spec.layer_ids[0]
    _fill(kv_pool.k_cache(layer_id))
    _fill(kv_pool.v_cache(layer_id))
    _fill(kv_pool.cmp_k_cache(0))
    _fill(kv_pool.pending_ring(0))
    kv_pool.pending_position_ring(0).copy_(
        torch.arange(kv_pool.pending_position_ring(0).numel(), dtype=torch.int64).view(
            kv_pool.pending_position_ring(0).shape
        )
    )

    linear_pool = _linear_pool()
    _fill(linear_pool.conv_states)
    _fill(linear_pool.recurrent_states)

    page_table = torch.zeros(2, 512, dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, PAGE_SIZE, page_table, "naive")
    stub = Scheduler.__new__(Scheduler)
    stub.device = CPU
    stub.cache_manager = cm
    stub.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
    stub.engine = SimpleNamespace(
        page_table=page_table,
        kv_cache=kv_pool,
        linear_state_pool=linear_pool,
        attn_backend=SimpleNamespace(_idx_slot={layer_id: 0}, ratio=kv_pool.index_ratio),
        sampler=SimpleNamespace(prepare=lambda batch: None),
    )
    stub.engine.attn_backend.prepare_metadata = lambda batch: None

    observer = MTPShadowObserver.__new__(MTPShadowObserver)
    observer.engine = stub.engine
    observer.cache_manager = SimpleNamespace(page_size=PAGE_SIZE)

    req = Req(
        input_ids=torch.arange(PROMPT_LEN, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=64,
        uid=7,
        sampling_params=SamplingParams(max_tokens=64),
        cache_handle=cm.prefix_cache.match_prefix(
            torch.zeros(0, dtype=torch.int32)
        ).cuda_handle,
    )
    cm.allocate_paged([req])
    req.complete_one()
    req.append_host(torch.tensor([5], dtype=torch.int32))
    req.linear_slot_idx = 1
    stub.token_pool[0, req.cached_len] = 5
    stub.config = SimpleNamespace(
        page_size=PAGE_SIZE, spec_decode=SpecDecodeConfig(enabled=True, depth=3)
    )
    return stub, observer, req


def _capture(req) -> MTPTargetCapture:
    return MTPTargetCapture(
        uid=req.uid,
        is_chunked=False,
        cached_len=req.cached_len,
        device_len=req.device_len,
        table_idx=req.table_idx,
        linear_slot_idx=req.linear_slot_idx,
        protected_linear_slots=(),
        input_ids_cpu=req.input_ids.clone(),
        multi_stream_cpu=torch.zeros(0),
        inputs_embeds_cpu=torch.zeros(0),
        rope_positions_cpu=None,
        mrope_position_delta=0,
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
    )


def test_a_full_rollback_leaves_every_state_family_digest_unchanged():
    stub, observer, req = _harness()
    capture = _capture(req)
    before = observer._target_state_family_digests(capture)
    assert set(before) == {
        "request-metadata",
        "page-table-prefix",
        "linear-conv",
        "linear-recurrent",
        "qsa-ring:0",
        "qsa-position-ring:0",
        "qsa-compressed:0",
        "target-k:3",
        "target-v:3",
    }

    stub._prepare_spec_batch(req, [11, 12, 13])
    stub._rollback_spec_tokens(req, 0)

    assert observer._target_state_family_digests(_capture(req)) == before
    assert observer._combine_state_family_digests(
        observer._target_state_family_digests(_capture(req))
    ) == observer._combine_state_family_digests(before)


def test_the_oracle_would_have_seen_an_unrolled_step():
    # Guards against a vacuous gate: the in-flight step DOES move the enumerated families.
    stub, observer, req = _harness()
    before = observer._target_state_family_digests(_capture(req))

    stub._prepare_spec_batch(req, [11, 12, 13])
    during = observer._target_state_family_digests(_capture(req))

    # The step opens a page at position 64, so the row's last page-table cell names a
    # different physical page -- which moves the K/V and compressed-slab reads with it. An
    # unrestored row would leave every one of these pointing into a freed page.
    assert {name for name in before if before[name] != during[name]} == {
        "request-metadata",
        "page-table-prefix",
        "qsa-compressed:0",
        "target-k:3",
        "target-v:3",
    }


def test_a_partial_rollback_keeps_the_page_the_accepted_rows_sit_on():
    stub, observer, req = _harness()
    before = observer._target_state_family_digests(_capture(req))

    stub._prepare_spec_batch(req, [11, 12, 13])
    during = observer._target_state_family_digests(_capture(req))
    stub._rollback_spec_tokens(req, 2)
    after = observer._target_state_family_digests(_capture(req))

    assert (req.cached_len, req.device_len) == (66, 67)
    # rows 64..66 are kept, so the page opened for them -- and every read that follows the
    # page-table cell naming it -- must survive the rollback unchanged. (page-table-prefix is
    # excluded: it digests [:device_len], whose length legitimately shrank.)
    for name in ("qsa-compressed:0", "target-k:3", "target-v:3"):
        assert after[name] == during[name] != before[name]
    assert int(stub.engine.page_table[0, 64]) not in stub.cache_manager.free_slots.tolist()
