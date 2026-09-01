from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.qsa_sparse import QSASparseAttnBackend, QSASparseMetadata
from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache import create_kvcache_pool, resolve_pool_class
from freetoken.models.qwen4_exp.mtp_spike import derive_mtp_model_config
from tests.models.qwen4_exp.common import parsed_config


def test_private_qsa_ring_is_widened_for_three_recursive_tokens():
    config = derive_mtp_model_config(parsed_config())
    pool = create_kvcache_pool(
        model_config=config,
        num_pages=4,
        page_size=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        num_req_slots=2,
        num_speculative_tokens=3,
    )
    assert pool.ring_capacity == 8
    assert pool.num_req_slots == 2
    assert pool._kv_buffer.shape[2] == 4
    assert pool._kv_buffer.shape[3] == 64


def test_full_private_qsa_geometry_has_exact_predicted_bytes():
    config = derive_mtp_model_config(parsed_config())
    # Keep shipping QSA dimensions while toy hidden/head counts come from the shared fixture.
    pool_cls = resolve_pool_class(config)
    fake = SimpleNamespace(
        model_config=config,
        page_size=64,
        max_running_req=1,
        tp_info=SimpleNamespace(size=1),
        dtype=torch.bfloat16,
    )
    per_page, fixed, _, _ = pool_cls.kv_cost(fake)
    assert per_page > 0
    assert fixed > 0
    predicted = 4097 * per_page + fixed
    assert predicted == 553_786_048
    assert 4097 * 64 - 64 == 262_144


def _qsa_metadata(width: int, value: int) -> QSASparseMetadata:
    return QSASparseMetadata(
        is_decode=False,
        last_indices=torch.tensor([width - 1], dtype=torch.int32),
        qo_indptr_cpu=torch.tensor([0, width], dtype=torch.int32),
        kv_len_cpu=torch.tensor([value], dtype=torch.int32),
        token_to_req=torch.zeros(width, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, width], dtype=torch.int32),
        seq_lens=torch.tensor([value], dtype=torch.int32),
        ring_slots=torch.tensor([value + 1], dtype=torch.int32),
        block_table=torch.full((1, 5), value + 2, dtype=torch.int32),
    )


def _verify_batch(width: int, value: int) -> Batch:
    cached_len = 5
    req = Req(
        input_ids=torch.arange(cached_len + width, dtype=torch.int32),
        table_idx=9,
        cached_len=cached_len,
        output_len=0,
        uid=value,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    req.linear_slot_idx = 3
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.arange(width, dtype=torch.int32)
    batch.positions = torch.arange(cached_len, cached_len + width, dtype=torch.int32)
    batch.out_loc = torch.arange(100, 100 + width, dtype=torch.int32)
    batch.linear_table_idx = torch.tensor([req.linear_slot_idx], dtype=torch.int32)
    batch.attn_metadata = _qsa_metadata(width, value)
    batch.mtp_verify = True
    return batch


def test_private_qsa_graph_accepts_one_request_with_two_to_six_verify_tokens():
    backend = object.__new__(QSASparseAttnBackend)
    backend.device = torch.device("cpu")
    backend._mtp_verify_graph = {}
    backend._mtp_verify_scratch = {}
    backend._mtp_verify_active_width = None
    backend._ensure_mtp_verify_scratch = lambda width: None

    # 2..6 = w for depth 1..5, the whole range FREETOKEN_MTP_SPEC_DEPTH admits. Each width
    # gets its own static metadata and its own scratch dict; nothing is special-cased.
    for width in (2, 3, 4, 5, 6):
        captured = _verify_batch(width, 10 + width)
        assert captured.size == captured.padded_size == 1
        assert captured.reqs[0].extend_len == width

        backend.prepare_mtp_verify_graph(captured)

        assert backend._mtp_verify_active_width == width
        assert width in backend._mtp_verify_graph
        assert captured.attn_metadata.token_to_req.tolist() == [0] * width
        assert captured.attn_metadata.cu_seqlens.tolist() == [0, width]
        runtime = _verify_batch(width, 90 + width)
        backend.stage_mtp_verify_graph(runtime, captured)
        assert captured.attn_metadata.seq_lens.tolist() == [90 + width]
        backend.finish_mtp_verify_graph_capture(width)


def test_private_qsa_graph_rejects_a_width_past_the_depth_ceiling():
    """One past the widest verify block any configuration can produce. The bound exists to
    catch a shape bug (a padded request count used as a token width, slice-27's live failure),
    so it must stay tight to the ceiling rather than being dropped."""
    backend = object.__new__(QSASparseAttnBackend)
    backend.device = torch.device("cpu")
    backend._mtp_verify_graph = {}
    backend._mtp_verify_scratch = {}
    backend._mtp_verify_active_width = None
    backend._ensure_mtp_verify_scratch = lambda width: None

    with pytest.raises(ValueError, match="width must be 2..6"):
        backend.prepare_mtp_verify_graph(_verify_batch(7, 70))
    with pytest.raises(ValueError, match="width must be 2..6"):
        backend.prepare_mtp_verify_graph(_verify_batch(1, 71))


def test_the_qsa_verify_width_cap_mirrors_the_engine_configs_depth_ceiling():
    """The attention backend must not import the engine config, so it mirrors the ceiling.
    Pinned here: a mirror that rots would refuse a legal depth-5 boot's graph capture."""
    from freetoken.attention.qsa_sparse import _MAX_MTP_VERIFY_WIDTH
    from freetoken.engine.config import _MAX_SPEC_DEPTH

    assert _MAX_MTP_VERIFY_WIDTH == 1 + _MAX_SPEC_DEPTH


def test_private_qsa_ring_widens_with_a_deeper_draft_chain():
    """The draft head sizes its OWN pool from the configured depth (spec_draft passes
    ``num_speculative_tokens=self.depth``). At depth 5 the ring must cover a whole
    index_ratio-token group plus the 5 recursive rows -- 4 + 5 = 9, so ratio-4 rounding gives
    12. A narrower ring aliases a draft row onto the open group's still-needed members and is
    silently wrong, never a crash."""
    config = derive_mtp_model_config(parsed_config())
    ratio = config.kv_cache_group_specs()[0].index_ratio
    for depth, expected in ((3, 8), (4, 8), (5, 12)):
        pool = create_kvcache_pool(
            model_config=config,
            num_pages=4,
            page_size=64,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            num_req_slots=2,
            num_speculative_tokens=depth,
        )
        assert pool.ring_capacity == expected
        assert pool.ring_capacity >= ratio + depth
        assert pool.ring_capacity % ratio == 0
        assert pool._pending_ring.shape[2] == expected
        assert pool._pending_position_ring.shape[2] == expected


def test_private_qsa_graph_rejects_non_prefill_or_multi_request_shapes():
    backend = object.__new__(QSASparseAttnBackend)
    backend.device = torch.device("cpu")
    backend._mtp_verify_graph = {}
    backend._mtp_verify_scratch = {}
    backend._mtp_verify_active_width = None
    backend._ensure_mtp_verify_scratch = lambda width: None

    decode = _verify_batch(2, 10)
    decode.phase = "decode"
    with pytest.raises(ValueError, match="one prefill request"):
        backend.prepare_mtp_verify_graph(decode)

    multiple = _verify_batch(2, 20)
    multiple.reqs.append(_verify_batch(2, 21).reqs[0])
    multiple.padded_reqs = multiple.reqs
    with pytest.raises(ValueError, match="one prefill request"):
        backend.prepare_mtp_verify_graph(multiple)

    unmarked = _verify_batch(2, 30)
    unmarked.mtp_verify = False
    with pytest.raises(ValueError, match="private verification marker"):
        backend.prepare_mtp_verify_graph(unmarked)

    captured = _verify_batch(2, 40)
    backend.prepare_mtp_verify_graph(captured)
    with pytest.raises(ValueError, match="replay width does not match"):
        backend.stage_mtp_verify_graph(_verify_batch(3, 41), captured)


def test_private_qsa_graph_staging_keeps_addresses_and_updates_values():
    backend = object.__new__(QSASparseAttnBackend)
    backend.device = torch.device("cpu")
    backend._mtp_verify_graph = {}
    backend._mtp_verify_scratch = {}
    backend._mtp_verify_active_width = None
    backend._ensure_mtp_verify_scratch = lambda width: None
    captured = _verify_batch(3, 10)

    backend.prepare_mtp_verify_graph(captured)
    static = captured.attn_metadata
    pointers = {
        "last": static.last_indices.data_ptr(),
        "seq": static.seq_lens.data_ptr(),
        "ring": static.ring_slots.data_ptr(),
        "table": static.block_table.data_ptr(),
    }

    runtime = _verify_batch(3, 90)
    backend.stage_mtp_verify_graph(runtime, captured)

    assert static.last_indices.tolist() == [2]
    assert static.seq_lens.tolist() == [90]
    assert static.ring_slots.tolist() == [91]
    assert static.block_table.tolist() == [[92] * 5]
    assert static.last_indices.data_ptr() == pointers["last"]
    assert static.seq_lens.data_ptr() == pointers["seq"]
    assert static.ring_slots.data_ptr() == pointers["ring"]
    assert static.block_table.data_ptr() == pointers["table"]

    backend.reset_mtp_verify_graph()
    assert backend._mtp_verify_graph == {}
    assert backend._mtp_verify_scratch == {}
    assert backend._mtp_verify_active_width is None


def test_private_qsa_failed_capture_discards_only_failed_width_state():
    backend = object.__new__(QSASparseAttnBackend)
    backend.device = torch.device("cpu")
    backend._mtp_verify_graph = {}
    backend._mtp_verify_scratch = {2: {"indices": torch.empty(2, 3)}}
    backend._mtp_verify_active_width = 2
    backend._ensure_mtp_verify_scratch = lambda width: None
    captured = _verify_batch(2, 10)
    backend.prepare_mtp_verify_graph(captured)

    backend.discard_mtp_verify_graph(2)

    assert 2 not in backend._mtp_verify_graph
    assert 2 not in backend._mtp_verify_scratch
    assert backend._mtp_verify_active_width is None


def test_private_qsa_graph_scratch_uses_fixed_address_only_while_capturing():
    backend = object.__new__(QSASparseAttnBackend)
    normal = torch.empty(4, 7)
    private = torch.empty(3, 7)
    backend._graph = {"indices": normal}
    backend._mtp_verify_scratch = {3: {"indices": private}}
    backend._mtp_verify_active_width = 3

    assert backend._scratch("indices", 3, 7, dtype=torch.float32).data_ptr() == private.data_ptr()
    backend.finish_mtp_verify_graph_capture(3)
    assert backend._scratch("indices", 3, 7, dtype=torch.float32).data_ptr() == normal.data_ptr()
