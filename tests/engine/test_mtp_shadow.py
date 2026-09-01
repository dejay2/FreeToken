from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.mtp_shadow import (
    _GUARD_BYTES,
    MTPShadowConfig,
    MTPShadowObserver,
    build_shifted_pairs,
    build_shifted_rope_positions,
)
from freetoken.models.qwen4_exp.mtp_spike import (
    MTPExactExpertRunner,
    MTPGPUExpertRunner,
    MTPNVFP4ExpertRunner,
    MTPStagedModelRunner,
    Qwen4ExpMTPModel,
    derive_mtp_model_config,
)
from tests.models.qwen4_exp.common import parsed_config


def _engine_config(**overrides):
    values = dict(
        max_running_req=1,
        page_size=64,
        max_seq_len=262_144,
        model_config=SimpleNamespace(model_type="qwen4_exp"),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_shadow_config_is_disabled_by_default(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("FREETOKEN_MTP_"):
            monkeypatch.delenv(name, raising=False)
    assert MTPShadowConfig.from_env(_engine_config()).enabled is False


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("FREETOKEN_MTP_SHADOW", "yes", "must be 0 or 1"),
        ("FREETOKEN_MTP_EXPERT_FORMAT", "fp8", "bf16 or nvfp4"),
        ("FREETOKEN_MTP_DEPTH", "4", "1, 2, or 3"),
        ("FREETOKEN_MTP_CPU_THREADS", "0", "positive"),
        (
            "FREETOKEN_MTP_VERIFY_MODE",
            "public-fast",
            "oracle, compare, fast-eager, or fast-graph",
        ),
    ],
)
def test_shadow_config_rejects_malformed_private_values(monkeypatch, name, value, message):
    monkeypatch.setenv("FREETOKEN_MTP_SHADOW", "1")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        MTPShadowConfig.from_env(_engine_config())


def test_disabled_shadow_ignores_fast_verifier_settings(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_SHADOW", "0")
    monkeypatch.setenv("FREETOKEN_MTP_VERIFY_MODE", "not-a-mode")
    config = MTPShadowConfig.from_env(_engine_config())
    assert config.enabled is False
    assert config.verify_mode == "oracle"


def test_shadow_config_rejects_scope_or_geometry_drift(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_SHADOW", "1")
    monkeypatch.setenv("FREETOKEN_MTP_PRIVATE_ROOT", r"D:\other")
    with pytest.raises(ValueError, match="private root"):
        MTPShadowConfig.from_env(_engine_config())
    monkeypatch.setenv(
        "FREETOKEN_MTP_PRIVATE_ROOT",
        r"D:\FreeToken-ple-mmap-vision\.local\mtp-spike",
    )
    for field, value, message in (
        ("max_running_req", 2, "one active"),
        ("max_seq_len", 131_072, "262144"),
        ("model_config", SimpleNamespace(model_type="other"), "Qwen3.8"),
    ):
        with pytest.raises(ValueError, match=message):
            MTPShadowConfig.from_env(_engine_config(**{field: value}))


def test_shadow_config_resident_is_off_by_default_and_env_enables_it(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_SHADOW", "1")
    monkeypatch.delenv("FREETOKEN_MTP_RESIDENT", raising=False)

    assert MTPShadowConfig(enabled=False).resident is False
    assert MTPShadowConfig.from_env(_engine_config()).resident is False
    monkeypatch.setenv("FREETOKEN_MTP_RESIDENT", "1")
    assert MTPShadowConfig.from_env(_engine_config()).resident is True


def test_shadow_config_target_geometry_defaults_to_the_full_reservation(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_SHADOW", "1")
    for name in ("FREETOKEN_MTP_TARGET_PAGES", "FREETOKEN_MTP_TARGET_EXPERTS"):
        monkeypatch.delenv(name, raising=False)

    default = MTPShadowConfig.from_env(_engine_config())
    assert (default.target_pages, default.target_experts) == (4097, 4063)
    assert (MTPShadowConfig(enabled=False).target_pages, MTPShadowConfig(
        enabled=False
    ).target_experts) == (4097, 4063)

    monkeypatch.setenv("FREETOKEN_MTP_TARGET_PAGES", "1025")
    monkeypatch.setenv("FREETOKEN_MTP_TARGET_EXPERTS", "3000")
    reduced = MTPShadowConfig.from_env(_engine_config())
    assert (reduced.target_pages, reduced.target_experts) == (1025, 3000)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("FREETOKEN_MTP_RESIDENT", "yes", "must be 0 or 1"),
        ("FREETOKEN_MTP_TARGET_PAGES", "0", "positive"),
        ("FREETOKEN_MTP_TARGET_EXPERTS", "-1", "positive"),
    ],
)
def test_shadow_config_rejects_malformed_resident_values(
    monkeypatch, name, value, message
):
    monkeypatch.setenv("FREETOKEN_MTP_SHADOW", "1")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        MTPShadowConfig.from_env(_engine_config())


def _draft_rng_observer(*, seed=411, uid=37):
    observer = object.__new__(MTPShadowObserver)
    observer.device = torch.device("cpu")
    observer.config = SimpleNamespace(seed=seed, depth=3)
    observer._init_private_rng()
    observer._reset_private_rng(uid)
    return observer


def test_persistent_draft_rng_advances_and_replays_only_after_request_reset():
    observer = _draft_rng_observer()
    logits = torch.zeros(8)
    global_before = torch.random.get_rng_state().clone()
    target_generator = torch.Generator().manual_seed(1881)
    target_before = target_generator.get_state().clone()

    def run_two_cycles():
        records = []
        for _ in range(2):
            token, rng = observer._sample_draft(
                logits,
                temperature=1.0,
                top_k=-1,
                top_p=1.0,
            )
            records.append((token, rng))
        return records

    first = run_two_cycles()
    assert (first[0][1]["draw_count_before"], first[0][1]["draw_count_after"]) == (
        0,
        1,
    )
    assert (first[1][1]["draw_count_before"], first[1][1]["draw_count_after"]) == (
        1,
        2,
    )
    assert first[0][1]["state_sha256_after"] == first[1][1]["state_sha256_before"]
    assert first[0][1]["cycle_index"] == first[1][1]["cycle_index"] == 0
    assert first[0][1]["default_rng_unchanged"] is True
    assert first[0][1]["state_sha256_before"] != first[1][1]["state_sha256_before"]
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert torch.equal(target_generator.get_state(), target_before)

    observer._reset_private_rng(37)
    replay = run_two_cycles()
    assert replay == first


def test_complete_private_draft_rng_replay_is_deterministic_across_fresh_observers():
    logits = torch.tensor([0.2, 0.1, 0.4, 0.3])

    def run(observer):
        return [
            observer._sample_draft(
                logits,
                temperature=0.8,
                top_k=-1,
                top_p=0.95,
            )
            for _ in range(3)
        ]

    assert run(_draft_rng_observer(seed=601, uid=43)) == run(
        _draft_rng_observer(seed=601, uid=43)
    )


def test_private_draft_and_acceptance_rng_streams_are_distinct_and_greedy_is_exact():
    """A greedy draft still returns the argmax exactly -- and now costs one draw doing it.

    ``MTPDraftSampler`` is branch-free on the host so the integrated draft CHAIN can be a CUDA
    graph, which means the greedy id is selected out of the same fixed sequence of kernels the
    sampled draw runs, and that sequence contains the uniform. What the observer still owns is
    that the draft's stream and the acceptance streams are disjoint, and that a draft never
    touches the acceptance one.
    """
    observer = _draft_rng_observer()
    draft_before = observer._rng_stream_snapshot("draft")
    acceptance_before = observer._rng_stream_snapshot("acceptance", depth=1)
    assert draft_before["state_sha256"] != acceptance_before["state_sha256"]

    token, transition = observer._sample_draft(
        torch.tensor([0.0, 4.0, 1.0]),
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
    )

    assert token == 1
    assert transition["greedy"] is True
    assert transition["draw_count_before"] == 0
    assert transition["draw_count_after"] == 1
    assert transition["state_sha256_before"] != transition["state_sha256_after"]
    assert observer._rng_stream_snapshot("acceptance", depth=1) == acceptance_before


def test_private_rng_objects_are_request_lifecycle_owned_and_cycle_indexes_reset():
    observer = _draft_rng_observer(seed=509, uid=11)
    first_draft = observer._draft_sampler
    first_acceptance = dict(observer._acceptance_generators)
    assert set(first_acceptance) == {1, 2, 3}
    assert observer._rng_cycle_index == 0
    assert observer._acceptance_cycle_indices == {1: 0, 2: 0, 3: 0}

    observer._reset_private_rng(12)

    assert observer._draft_sampler is not first_draft
    assert all(
        observer._acceptance_generators[depth] is not first_acceptance[depth]
        for depth in (1, 2, 3)
    )
    states = {
        observer._rng_stream_snapshot("draft")["state_sha256"],
        *(
            observer._rng_stream_snapshot("acceptance", depth=depth)["state_sha256"]
            for depth in (1, 2, 3)
        ),
    }
    assert len(states) == 4
    assert observer._rng_cycle_index == 0
    assert observer._acceptance_cycle_indices == {1: 0, 2: 0, 3: 0}


def test_draft_rng_rejects_default_rng_leak(monkeypatch):
    observer = _draft_rng_observer()
    original = observer._draft_sampler.sample
    global_before = torch.random.get_rng_state().clone()

    def leaking_sample(*args, **kwargs):
        torch.rand(())
        return original(*args, **kwargs)

    monkeypatch.setattr(observer._draft_sampler, "sample", leaking_sample)
    try:
        with pytest.raises(
            RuntimeError,
            match="MTP checker changed target/global RNG state",
        ):
            observer._sample_draft(
                torch.zeros(4),
                temperature=1.0,
                top_k=-1,
                top_p=1.0,
            )
    finally:
        torch.random.set_rng_state(global_before)


def test_target_expert_temperature_alternates_cold_then_warm_and_resets_only_cold():
    class Cache:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    observer = object.__new__(MTPShadowObserver)
    observer.engine = SimpleNamespace(moe_offload_cache=Cache())
    observer._init_target_expert_temperature()

    cold = observer._assign_target_expert_temperature(101)
    cold_first = observer._prepare_target_expert_temperature(101)
    cold_second = observer._prepare_target_expert_temperature(101)
    warm = observer._assign_target_expert_temperature(102)
    warm_record = observer._prepare_target_expert_temperature(102)
    next_cold = observer._assign_target_expert_temperature(103)

    assert cold == {"state": "cold", "pair_index": 0, "request_uid": 101}
    assert cold_first["residency_reset"] is True
    assert cold_second["residency_reset"] is True
    assert cold_first["reset_setup_ms"] >= 0
    assert warm == {"state": "warm", "pair_index": 0, "request_uid": 102}
    assert warm_record["residency_reset"] is False
    assert next_cold == {"state": "cold", "pair_index": 1, "request_uid": 103}
    assert observer.engine.moe_offload_cache.reset_calls == 2


def test_proposal_timing_includes_lm_head_sampling_and_recursive_steps():
    first = MTPShadowObserver._proposal_component_times(
        request_cycle_index=0,
        prompt_setup_ms=10.0,
        update_ms=2.0,
        draft_step_ms=[1.0, 1.5, 2.0],
        recursive_step_ms=[3.0, 4.0],
        cleanup_ms=0.5,
        graph_capture_setup_ms=6.0,
        depth=3,
    )
    later = MTPShadowObserver._proposal_component_times(
        request_cycle_index=1,
        prompt_setup_ms=0.0,
        update_ms=2.0,
        draft_step_ms=[1.0, 1.5, 2.0],
        recursive_step_ms=[3.0, 4.0],
        cleanup_ms=0.5,
        graph_capture_setup_ms=0.0,
        depth=2,
    )

    assert first == {"P": 18.0, "D": 12.0}
    assert later == {"P": 0.0, "D": 8.0}
    with pytest.raises(ValueError, match="later request cycle"):
        MTPShadowObserver._proposal_component_times(
            request_cycle_index=2,
            prompt_setup_ms=0.0,
            update_ms=1.0,
            draft_step_ms=[1.0],
            recursive_step_ms=[],
            cleanup_ms=0.5,
            graph_capture_setup_ms=2.0,
            depth=1,
        )


def test_target_expert_temperature_fails_closed_on_unexpected_request_order():
    observer = object.__new__(MTPShadowObserver)
    observer.engine = SimpleNamespace(
        moe_offload_cache=SimpleNamespace(reset=lambda: None)
    )
    observer._init_target_expert_temperature()
    observer._assign_target_expert_temperature(201)

    with pytest.raises(RuntimeError, match="target expert temperature request order"):
        observer._prepare_target_expert_temperature(202)
    with pytest.raises(RuntimeError, match="cold request must be followed by warm"):
        observer._assign_target_expert_temperature(203)


def test_shifted_pairs_cross_chunks_and_one_row_tail_exactly():
    all_hidden = torch.arange(17, dtype=torch.float32).view(17, 1)
    all_embeds = (100 + torch.arange(17, dtype=torch.float32)).view(17, 1)
    pending = None
    got_hidden = []
    got_embeds = []
    start = 0
    for width in (8, 8, 1):
        end = start + width
        final = end == 17
        next_embedding = torch.tensor([[999.0]]) if final else None
        h, e, pending = build_shifted_pairs(
            pending,
            all_hidden[start:end],
            all_embeds[start:end],
            next_embedding=next_embedding,
        )
        got_hidden.append(h)
        got_embeds.append(e)
        start = end
    assert pending is None
    assert torch.equal(torch.cat(got_hidden), all_hidden)
    expected_embeds = torch.cat((all_embeds[1:], torch.tensor([[999.0]])))
    assert torch.equal(torch.cat(got_embeds), expected_embeds)


def test_shifted_picture_positions_cross_chunks_with_hidden_rows():
    first = torch.tensor([[0, 1, 2], [10, 11, 12], [20, 21, 22]])
    paired1, pending = build_shifted_rope_positions(
        None, first, had_pending_hidden=False, final_chunk=False
    )
    torch.testing.assert_close(paired1, first[:, :2])
    torch.testing.assert_close(pending, first[:, 2:])

    second = torch.tensor([[3, 4], [13, 14], [23, 24]])
    paired2, pending = build_shifted_rope_positions(
        pending, second, had_pending_hidden=True, final_chunk=True
    )
    torch.testing.assert_close(
        paired2, torch.tensor([[2, 3, 4], [12, 13, 14], [22, 23, 24]])
    )
    assert pending is None


def test_shifted_pairs_preserve_actual_picture_embedding_at_boundary():
    pending = torch.tensor([[7.0, 8.0]])
    hidden = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    picture = torch.tensor([[101.0, 102.0], [13.0, 14.0]])
    paired_hidden, paired_embeds, tail = build_shifted_pairs(
        pending, hidden, picture, next_embedding=None
    )
    assert torch.equal(paired_hidden[0], pending[0])
    assert torch.equal(paired_embeds[0], picture[0])
    assert torch.equal(tail, hidden[-1:])


def _meta_mtp_model_and_weights(*, fill: bool = False):
    from freetoken.utils.torch_utils import torch_dtype

    config = derive_mtp_model_config(parsed_config())
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = Qwen4ExpMTPModel(config)
    generator = torch.Generator().manual_seed(7)
    weights = {}
    for name, tensor in model.state_dict().items():
        if ".experts." in name:
            continue
        value = torch.zeros(tensor.shape, dtype=tensor.dtype)
        if fill and value.is_floating_point():
            value.normal_(0.0, 0.05, generator=generator)
        weights[name] = value
    return model, weights


def _group_owners(model, names):
    owners = []
    for name in names:
        parts = name.split(".")
        owner = model
        for part in parts[:-1]:
            owner = owner.op_list[int(part)] if part.isdigit() else getattr(owner, part)
        owners.append((owner, parts[-1]))
    return owners


def test_dense_weights_are_present_only_inside_their_stage():
    from freetoken.layers.rotary import get_rope, set_rope_device

    get_rope.cache_clear()
    set_rope_device(torch.device("cpu"))
    try:
        model, weights = _meta_mtp_model_and_weights()
        runner = MTPStagedModelRunner(model, weights, device=torch.device("cpu"))
        owners = _group_owners(model, runner.groups["attention"])
        assert all(getattr(owner, attr).is_meta for owner, attr in owners)
        with runner.stage("attention"):
            assert all(not getattr(owner, attr).is_meta for owner, attr in owners)
        assert all(getattr(owner, attr).is_meta for owner, attr in owners)
        assert runner.stats.stages == 1
        assert runner.stats.copied_bytes == runner.max_component_bytes
        assert runner.staging_bytes == runner.max_component_bytes
        assert runner.resident_bytes == 0
    finally:
        get_rope.cache_clear()
        set_rope_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def test_resident_dense_weights_never_leave_the_device_or_move_per_call():
    from freetoken.layers.rotary import get_rope, set_rope_device

    get_rope.cache_clear()
    set_rope_device(torch.device("cpu"))
    try:
        model, weights = _meta_mtp_model_and_weights()
        runner = MTPStagedModelRunner(
            model, weights, device=torch.device("cpu"), resident=True
        )
        owners = [
            owner
            for group in runner.groups
            for owner in _group_owners(model, runner.groups[group])
        ]
        assert all(not getattr(owner, attr).is_meta for owner, attr in owners)
        for group in runner.groups:
            with runner.stage(group):
                assert all(not getattr(owner, attr).is_meta for owner, attr in owners)
        assert all(not getattr(owner, attr).is_meta for owner, attr in owners)

        assert runner.stats.stages == len(runner.groups)
        assert runner.stats.copied_bytes == 0
        assert runner.stats.copy_ms == 0.0
        assert runner.stats.peak_component_bytes == 0
        assert runner.staging_bytes == 0
        assert runner.resident_bytes == sum(
            tensor.numel() * tensor.element_size() for tensor in weights.values()
        )
    finally:
        get_rope.cache_clear()
        set_rope_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def test_resident_and_staged_dense_paths_compute_the_same_fusion():
    from freetoken.layers.rotary import get_rope, set_rope_device

    get_rope.cache_clear()
    set_rope_device(torch.device("cpu"))
    try:
        staged_model, weights = _meta_mtp_model_and_weights(fill=True)
        resident_model, _ = _meta_mtp_model_and_weights(fill=True)
        staged = MTPStagedModelRunner(
            staged_model, weights, device=torch.device("cpu")
        )
        resident = MTPStagedModelRunner(
            resident_model, weights, device=torch.device("cpu"), resident=True
        )
        config = staged_model.config
        embeddings = torch.randn(2, config.hidden_size).to(torch.bfloat16)
        hidden = torch.randn(
            2, config.qwen4_args.hc_count * config.hidden_size
        ).to(torch.bfloat16)

        with staged.stage("pre_fc"):
            expected = staged_model.fuse_inputs(embeddings, hidden)
        with resident.stage("pre_fc"):
            got = resident_model.fuse_inputs(embeddings, hidden)

        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    finally:
        get_rope.cache_clear()
        set_rope_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# --------------------------------------------------------------------------------------
# overlap-skewed capture: the scheduler launches decode k before append_host lands for k-1,
# so at prepare_capture time req.device_len is one ahead of req.input_ids
# --------------------------------------------------------------------------------------

def _skewed_decode_batch(prompt_len: int = 6, next_token: int = 4242):
    from freetoken.core import Batch, Req, SamplingParams

    req = Req(
        input_ids=torch.arange(100, 100 + prompt_len, dtype=torch.int32),
        table_idx=3,
        cached_len=0,
        output_len=8,
        uid=17,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    req.linear_slot_idx = 5
    # exactly what engine.py does before append_host catches up
    req.complete_one()
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.tensor([next_token], dtype=torch.int32)
    batch.rope_positions = None
    return batch, req


def _capture_observer():
    observer = object.__new__(MTPShadowObserver)
    observer._uid = 17
    return observer


def test_prepare_capture_completes_the_overlap_skewed_token_snapshot():
    observer = _capture_observer()
    batch, req = _skewed_decode_batch()
    assert req.input_ids.numel() == req.device_len - 1, "test must reproduce the skew"

    captured = observer.prepare_capture(
        batch, (None, torch.zeros(1, 2), torch.zeros(1, 3))
    )

    assert captured.device_len == req.device_len
    assert captured.input_ids_cpu.numel() == captured.device_len
    assert captured.input_ids_cpu.dtype == torch.int32
    assert captured.input_ids_cpu[:-1].tolist() == req.input_ids.tolist()
    assert int(captured.input_ids_cpu[-1]) == int(batch.input_ids[0])


def test_prepare_capture_keeps_an_in_sync_snapshot_untouched():
    from freetoken.core import Batch, Req, SamplingParams

    req = Req(
        input_ids=torch.arange(100, 106, dtype=torch.int32),
        table_idx=3,
        cached_len=0,
        output_len=8,
        uid=17,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    req.linear_slot_idx = 5
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = req.input_ids.clone()
    batch.rope_positions = None

    captured = _capture_observer().prepare_capture(
        batch, (None, torch.zeros(1, 2), torch.zeros(1, 3))
    )

    assert captured.input_ids_cpu.tolist() == req.input_ids.tolist()
    assert captured.input_ids_cpu.data_ptr() != req.input_ids.data_ptr()


def test_prepare_capture_fails_closed_when_the_snapshot_cannot_be_completed():
    observer = _capture_observer()
    batch, req = _skewed_decode_batch()
    batch.input_ids = torch.tensor([], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="token snapshot"):
        observer.prepare_capture(batch, (None, torch.zeros(1, 2), torch.zeros(1, 3)))


def _verify_batch_observer():
    observer = object.__new__(MTPShadowObserver)
    observer.device = torch.device("cpu")
    observer.engine = SimpleNamespace(
        page_table=torch.zeros((4, 64), dtype=torch.int32),
        attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None),
    )
    return observer


def _capture_stub(device_len: int, id_count: int):
    from freetoken.engine.mtp_shadow import MTPTargetCapture

    return MTPTargetCapture(
        uid=17,
        is_chunked=False,
        cached_len=device_len - 1,
        device_len=device_len,
        table_idx=3,
        linear_slot_idx=5,
        protected_linear_slots=(5,),
        input_ids_cpu=torch.arange(100, 100 + id_count, dtype=torch.int32),
        multi_stream_cpu=torch.zeros(1, 2),
        inputs_embeds_cpu=torch.zeros(1, 3),
        rope_positions_cpu=None,
        mrope_position_delta=0,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
    )


def test_target_verify_batch_matches_extend_len_to_the_candidate_rows():
    batch = _verify_batch_observer()._target_verify_batch(
        _capture_stub(device_len=7, id_count=7),
        [11, 12, 13],
        dummy_table_idx=1,
        shadow_slot=5,
    )

    assert batch.reqs[0].extend_len == 3
    assert batch.input_ids.shape[0] == 3
    assert batch.fla_metadata.cu_seqlens.tolist() == [0, 3]


def test_target_verify_batch_refuses_a_short_token_snapshot():
    with pytest.raises(RuntimeError, match="extend_len"):
        _verify_batch_observer()._target_verify_batch(
            _capture_stub(device_len=7, id_count=6),
            [11, 12, 13],
            dummy_table_idx=1,
            shadow_slot=5,
        )


def test_observer_init_arms_expert_movement_stats(tmp_path):
    cache = SimpleNamespace(collect_stats=False, cache_size=0)
    engine = SimpleNamespace(
        device=torch.device("cpu"),
        ctx=SimpleNamespace(),
        model=SimpleNamespace(),
        config=SimpleNamespace(model_config=parsed_config(), model_path=tmp_path),
        moe_offload_cache=cache,
        num_pages=0,
    )
    config = MTPShadowConfig(enabled=True, private_root=tmp_path)

    # the geometry gate stops __init__ well after the cache is taken hold of
    with pytest.raises(RuntimeError, match="KV geometry"):
        MTPShadowObserver(engine, config)

    assert cache.collect_stats is True


def test_observer_init_tolerates_a_target_without_an_offload_cache(tmp_path):
    engine = SimpleNamespace(
        device=torch.device("cpu"),
        ctx=SimpleNamespace(),
        model=SimpleNamespace(),
        config=SimpleNamespace(model_config=parsed_config(), model_path=tmp_path),
        moe_offload_cache=None,
        num_pages=0,
    )

    with pytest.raises(RuntimeError, match="KV geometry"):
        MTPShadowObserver(engine, MTPShadowConfig(enabled=True, private_root=tmp_path))


def _geometry_engine(tmp_path, *, num_pages, cache_size):
    return SimpleNamespace(
        device=torch.device("cpu"),
        ctx=SimpleNamespace(),
        model=SimpleNamespace(),
        config=SimpleNamespace(
            model_config=parsed_config(), model_path=tmp_path, page_size=64
        ),
        moe_offload_cache=SimpleNamespace(collect_stats=False, cache_size=cache_size),
        num_pages=num_pages,
        linear_state_pool=None,
    )


def test_observer_geometry_gate_follows_the_configured_target_expectations(tmp_path):
    engine = _geometry_engine(tmp_path, num_pages=1025, cache_size=3000)

    with pytest.raises(RuntimeError, match="KV geometry"):
        MTPShadowObserver(engine, MTPShadowConfig(enabled=True, private_root=tmp_path))
    with pytest.raises(RuntimeError, match="requires 4063 target experts, got 3000"):
        MTPShadowObserver(
            engine,
            MTPShadowConfig(
                enabled=True, private_root=tmp_path, target_pages=1025
            ),
        )
    # both gates cleared: the run stops at the next unrelated target requirement
    with pytest.raises(RuntimeError, match="eight target recurrent slots"):
        MTPShadowObserver(
            engine,
            MTPShadowConfig(
                enabled=True,
                private_root=tmp_path,
                target_pages=1025,
                target_experts=3000,
            ),
        )


def test_observer_still_pins_the_page_size_when_the_page_count_is_reduced(tmp_path):
    engine = _geometry_engine(tmp_path, num_pages=1025, cache_size=3000)
    engine.config.page_size = 128

    with pytest.raises(RuntimeError, match="KV geometry"):
        MTPShadowObserver(
            engine,
            MTPShadowConfig(
                enabled=True,
                private_root=tmp_path,
                target_pages=1025,
                target_experts=3000,
            ),
        )


def test_memory_plan_counts_resident_bytes_and_leaves_the_staged_plan_unchanged():
    staged = MTPShadowObserver._build_memory_plan(
        free=1 << 30, qsa_bytes=100, staging_bytes=50, resident_bytes=0
    )
    resident = MTPShadowObserver._build_memory_plan(
        free=1 << 30, qsa_bytes=100, staging_bytes=0, resident_bytes=7
    )

    assert staged == {
        "free_before": 1 << 30,
        "mtp_qsa_bytes": 100,
        "max_dense_stage_bytes": 50,
        "guard_bytes": _GUARD_BYTES,
        "required_bytes": 150 + _GUARD_BYTES,
    }
    assert resident["resident_bytes"] == 7
    assert resident["max_dense_stage_bytes"] == 0
    assert resident["required_bytes"] == 107 + _GUARD_BYTES


def _placement_observer(tmp_path, **overrides):
    observer = object.__new__(MTPShadowObserver)
    observer.config = MTPShadowConfig(
        enabled=True, private_root=tmp_path, **overrides
    )
    observer.engine = SimpleNamespace(
        config=SimpleNamespace(model_path=tmp_path)
    )
    observer._weight_store = None
    return observer


def test_resident_mode_selects_the_device_expert_runner(tmp_path):
    assert _placement_observer(tmp_path)._expert_runner_type() is MTPExactExpertRunner
    assert (
        _placement_observer(tmp_path, placement="nvfp4")._expert_runner_type()
        is MTPNVFP4ExpertRunner
    )
    assert (
        _placement_observer(tmp_path, resident=True)._expert_runner_type()
        is MTPGPUExpertRunner
    )


def test_resident_expert_placement_refuses_nvfp4(tmp_path):
    observer = _placement_observer(tmp_path, placement="nvfp4", resident=True)

    with pytest.raises(ValueError, match="bf16"):
        observer._load_expert_banks()


def test_launcher_passes_an_explicit_moe_cache_size_only_when_asked():
    launcher = (
        Path(__file__).parents[2]
        / "scripts"
        / "start-qwen38-flash-next-mmap-windows.ps1"
    ).read_text(encoding="utf-8")

    assert "[int]$MoECacheSize" in launcher
    assert "if ($MoECacheSize -gt 0)" in launcher
    assert "'--moe-cache-size', \"$MoECacheSize\"" in launcher
    assert "'--moe-cache-auto'" in launcher
