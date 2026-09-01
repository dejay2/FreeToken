from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.mtp_fast_verify import MTPVerifyForwardResult
from freetoken.engine.mtp_shadow import MTPShadowObserver
from freetoken.scheduler.cache import CacheManager


def _manager(pages=10, page_size=64):
    table = torch.zeros((2, pages * page_size), dtype=torch.int32)
    return CacheManager(pages, page_size, table, "naive")


def test_temporary_page_lease_conserves_exact_free_set():
    manager = _manager()
    before = manager.free_slots.clone()
    with manager.temporary_page_lease(3) as pages:
        assert pages.tolist() == before[:3].tolist()
        assert pages.untyped_storage().data_ptr() != (
            manager.free_slots.untyped_storage().data_ptr()
        )
        assert len(manager.free_slots) == len(before) - 3
    assert torch.equal(manager.free_slots, before)


def test_temporary_page_lease_conserves_on_injected_failure():
    manager = _manager()
    before = manager.free_slots.clone()
    with pytest.raises(RuntimeError, match="injected"):
        with manager.temporary_page_lease(2):
            raise RuntimeError("injected verification failure")
    assert torch.equal(manager.free_slots, before)


def test_temporary_page_lease_never_evicts_prefix_state():
    manager = _manager(pages=2)
    manager.free_slots = manager.free_slots[:1]
    before = manager.free_slots.clone()
    with pytest.raises(RuntimeError, match="immediately free"):
        with manager.temporary_page_lease(2):
            pass
    assert torch.equal(manager.free_slots, before)


def test_borrowed_recurrent_snapshot_restores_every_state_family_bit_exactly():
    from types import SimpleNamespace

    pool = SimpleNamespace(
        conv_states=torch.arange(24, dtype=torch.bfloat16).view(2, 3, 4),
        recurrent_states=torch.arange(36, dtype=torch.float32).view(2, 3, 6),
        slot_states={
            "ple": torch.arange(18, dtype=torch.int64).view(2, 3, 3),
            "api": torch.arange(6, dtype=torch.int32).view(1, 3, 2),
        },
    )
    before = MTPShadowObserver._linear_slot_digest(pool, 1)
    snapshot = MTPShadowObserver._linear_slot_snapshot(pool, 1)
    assert MTPShadowObserver._linear_snapshot_digest(snapshot) == before
    pool.conv_states[:, 1].fill_(-1)
    pool.recurrent_states[:, 1].fill_(-2)
    for tensor in pool.slot_states.values():
        tensor[:, 1].fill_(-3)
    MTPShadowObserver._restore_linear_slot(pool, 1, snapshot)
    assert MTPShadowObserver._linear_slot_digest(pool, 1) == before


def test_temporary_page_lease_rejects_zero_pages():
    with pytest.raises(ValueError, match="at least one"):
        with _manager().temporary_page_lease(0):
            pass


@pytest.mark.parametrize(
    ("free_slots", "needed_pages", "forbidden_pages", "message"),
    [
        ([64, 64, 128], 2, [], "unique"),
        ([1, 64, 128], 1, [], "page-aligned"),
        ([640, 64, 128], 1, [], "out of range"),
        ([64, 128, 192], 2, [128], "forbidden"),
    ],
)
def test_temporary_page_lease_rejects_invalid_or_live_pages(
    free_slots, needed_pages, forbidden_pages, message
):
    manager = _manager()
    manager.free_slots = torch.tensor(free_slots, dtype=torch.int32)
    before = manager.free_slots.clone()

    with pytest.raises((ValueError, RuntimeError), match=message):
        with manager.temporary_page_lease(
            needed_pages,
            forbidden_pages=torch.tensor(forbidden_pages, dtype=torch.int32),
        ):
            pass

    assert torch.equal(manager.free_slots, before)


def test_temporary_page_lease_detects_allocator_mutation_without_overwriting_it():
    manager = _manager(pages=6)
    expected_remainder = manager.free_slots[2:].clone()
    unexpected = torch.cat((expected_remainder, torch.tensor([320], dtype=torch.int32)))

    with pytest.raises(
        RuntimeError,
        match="MTP temporary page lease observed allocator mutation",
    ):
        with manager.temporary_page_lease(2):
            manager.free_slots = unexpected.clone()

    assert torch.equal(manager.free_slots, unexpected)


class _BorrowOnlyPool:
    def __init__(self):
        self.num_slots = 4
        self.num_free_slots = 0
        self.conv_states = torch.arange(32, dtype=torch.bfloat16).view(2, 4, 4)
        self.recurrent_states = torch.arange(48, dtype=torch.float32).view(2, 4, 6)
        self.slot_states = {
            "ple": torch.arange(24, dtype=torch.int64).view(2, 4, 3),
        }

    def copy_from(self, source, destination):
        self.conv_states[:, destination].copy_(self.conv_states[:, source])
        self.recurrent_states[:, destination].copy_(self.recurrent_states[:, source])
        for tensor in self.slot_states.values():
            tensor[:, destination].copy_(tensor[:, source])


class _FreeSlotPool(_BorrowOnlyPool):
    def __init__(self):
        super().__init__()
        self._free_slots = [3]
        self.num_free_slots = 1

    def alloc(self, count):
        assert count == 1
        self.num_free_slots -= 1
        return [self._free_slots.pop()]

    def free(self, slot):
        self._free_slots.append(int(slot))
        self.num_free_slots += 1


class _TargetKV:
    def __init__(self, pages=4, page_size=64, index_ratio=4):
        self._pending_ring = torch.arange(32, dtype=torch.bfloat16).view(2, 1, 16)
        self._pending_position_ring = torch.arange(96, dtype=torch.int64).view(2, 1, 16, 3)
        self.cmp_scratch_base = pages * (page_size // index_ratio)
        cmp_rows = self.cmp_scratch_base + 8
        self._cmp_k_buffer = torch.arange(
            cmp_rows * 4, dtype=torch.bfloat16
        ).view(1, cmp_rows, 4)
        self._k_buffer = torch.arange(
            pages * page_size * 4, dtype=torch.bfloat16
        ).view(1, pages, page_size, 4)
        self._v_buffer = self._k_buffer.clone() + 1000

    def pending_ring(self, slot):
        return self._pending_ring[:, slot]

    def pending_position_ring(self, slot):
        return self._pending_position_ring[:, slot]

    def k_cache(self, layer_id):
        return self._k_buffer[layer_id]

    def v_cache(self, layer_id):
        return self._v_buffer[layer_id]

    def cmp_k_cache(self, slot):
        return self._cmp_k_buffer[slot]


def test_target_page_view_maps_physical_token_bases_to_page_axis():
    tensor = torch.arange(3 * 64 * 2, dtype=torch.int32).view(3, 64, 2)

    for physical_token_base, page_index in ((0, 0), (64, 1), (128, 2)):
        page = MTPShadowObserver._target_kv_page(
            tensor,
            physical_token_base,
            page_size=64,
        )
        assert torch.equal(page, tensor[page_index])


@pytest.mark.parametrize("physical_token_base", [-64, 1, 192])
def test_target_page_view_rejects_invalid_physical_token_base(physical_token_base):
    tensor = torch.zeros(3, 64, 2)

    with pytest.raises((ValueError, IndexError)):
        MTPShadowObserver._target_kv_page(
            tensor,
            physical_token_base,
            page_size=64,
        )


def test_partial_target_page_copy_uses_one_production_page():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()
    k_before = target_kv._k_buffer.clone()
    v_before = target_kv._v_buffer.clone()
    cmp_before = target_kv._cmp_k_buffer.clone()

    observer._copy_partial_target_page(0, 64)

    assert torch.equal(target_kv._k_buffer[0, 1], k_before[0, 0])
    assert torch.equal(target_kv._v_buffer[0, 1], v_before[0, 0])
    assert torch.equal(target_kv._cmp_k_buffer[0, 16:32], cmp_before[0, 0:16])
    assert torch.equal(target_kv._k_buffer[0, 0], k_before[0, 0])
    assert torch.equal(target_kv._v_buffer[0, 0], v_before[0, 0])
    assert torch.equal(target_kv._k_buffer[0, 2:], k_before[0, 2:])
    assert torch.equal(target_kv._v_buffer[0, 2:], v_before[0, 2:])
    assert torch.equal(target_kv._cmp_k_buffer[0, 32:], cmp_before[0, 32:])


def _all_scratch_bytes(engine):
    tensors = [
        engine.page_table,
        engine.linear_state_pool.conv_states,
        engine.linear_state_pool.recurrent_states,
        engine.kv_cache._pending_ring,
        engine.kv_cache._pending_position_ring,
        engine.kv_cache._cmp_k_buffer,
        engine.kv_cache._k_buffer,
        engine.kv_cache._v_buffer,
        *engine.linear_state_pool.slot_states.values(),
    ]
    return b"".join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in tensors)


def _named_transaction_fixture():
    observer, captured, manager, pool, target_kv, engine = _transaction_fixture()
    del observer.__dict__["_target_state_family_digests"]
    return observer, captured, manager, pool, target_kv, engine


def _successful_forward(mode="fast-eager"):
    def forward(batch):
        return MTPVerifyForwardResult(
            mode=mode,
            logits=torch.zeros(2, 4),
            required_wall_ms=1.0,
            core_cuda_ms=1.0,
            required_synchronizations=0,
            instrumentation_wall_ms=0.25,
            instrumentation_synchronizations=0,
            expert_movement={"available": False},
        )

    return forward


def test_target_scratch_transaction_does_not_advance_private_global_or_target_rng():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()
    observer.config = SimpleNamespace(seed=913, depth=3)
    observer._init_private_rng()
    observer._reset_private_rng(captured.uid)
    draft_before = observer._rng_stream_snapshot("draft")
    acceptance_before = observer._rng_stream_snapshot("acceptance", depth=1)
    global_before = torch.random.get_rng_state().clone()
    target_generator = torch.Generator().manual_seed(914)
    target_before = target_generator.get_state().clone()

    observer._run_target_transaction(
        captured,
        [7, 8],
        forward=_successful_forward(),
        mode="fast-eager",
    )

    assert observer._rng_stream_snapshot("draft") == draft_before
    assert observer._rng_stream_snapshot("acceptance", depth=1) == acceptance_before
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert torch.equal(target_generator.get_state(), target_before)


def test_target_transaction_rejects_default_rng_leak():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()
    observer.config = SimpleNamespace(seed=919, depth=3)
    observer._init_private_rng()
    observer._reset_private_rng(captured.uid)
    global_before = torch.random.get_rng_state().clone()

    def leaking_forward(batch):
        torch.rand(())
        return _successful_forward()(batch)

    try:
        with pytest.raises(
            RuntimeError,
            match="MTP checker changed target/global RNG state",
        ):
            observer._run_target_transaction(
                captured,
                [7, 8],
                forward=leaking_forward,
                mode="fast-eager",
            )
    finally:
        torch.random.set_rng_state(global_before)


def _configure_partial_prefix(captured, manager, engine, *, device_len=70):
    captured.device_len = device_len
    live_row_end = ((device_len + manager.page_size - 1) // manager.page_size) * (
        manager.page_size
    )
    engine.page_table[captured.table_idx, :live_row_end].copy_(
        torch.arange(live_row_end, dtype=torch.int32)
    )
    manager.free_slots = torch.arange(
        live_row_end,
        manager.num_pages * manager.page_size,
        manager.page_size,
        dtype=torch.int32,
    )


def _install_synthetic_target_batch(observer, captured, engine):
    def build(captured_arg, candidate_ids, *, dummy_table_idx, shadow_slot):
        assert captured_arg is captured
        batch = type("SyntheticTargetBatch", (), {})()
        batch.out_loc = engine.page_table[
            dummy_table_idx,
            captured.device_len : captured.device_len + len(candidate_ids),
        ].clone()
        batch.linear_table_idx = torch.tensor([shadow_slot], dtype=torch.int32)
        return batch

    observer._target_verify_batch = build


def _scratch_writing_forward(engine, seen_out_loc, mode="fast-eager"):
    def forward(batch):
        physical_tokens = [int(value) for value in batch.out_loc.cpu().tolist()]
        seen_out_loc.extend(physical_tokens)
        for physical_token in physical_tokens:
            page_index, row = divmod(physical_token, 64)
            engine.kv_cache._k_buffer[:, page_index, row].fill_(-101)
            engine.kv_cache._v_buffer[:, page_index, row].fill_(-202)
            engine.kv_cache._cmp_k_buffer[:, physical_token // 4].fill_(-303)
        dummy = engine.dummy_req.table_idx
        engine.kv_cache._pending_ring[dummy].add_(1)
        engine.kv_cache._pending_position_ring[dummy].add_(1)
        shadow_slot = int(batch.linear_table_idx[0].item())
        engine.linear_state_pool.conv_states[:, shadow_slot].add_(1)
        engine.linear_state_pool.recurrent_states[:, shadow_slot].add_(1)
        for tensor in engine.linear_state_pool.slot_states.values():
            tensor[:, shadow_slot].add_(1)
        return MTPVerifyForwardResult(
            mode=mode,
            logits=torch.zeros(len(physical_tokens), 4),
            required_wall_ms=1.0,
            core_cuda_ms=1.0,
            required_synchronizations=0,
            instrumentation_wall_ms=0.25,
            instrumentation_synchronizations=0,
            expert_movement={"available": False},
        )

    return forward


def test_scratch_transaction_shares_complete_pages_and_restores_partial_page_bytes():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()
    _configure_partial_prefix(captured, manager, engine)
    _install_synthetic_target_batch(observer, captured, engine)
    scratch_before = _all_scratch_bytes(engine)
    free_before = manager.free_slots.clone()
    seen_out_loc = []
    dummy = engine.dummy_req.table_idx

    def forward(batch):
        assert torch.equal(
            engine.page_table[dummy, :64], torch.arange(64, dtype=torch.int32)
        )
        assert torch.equal(
            engine.page_table[dummy, 64:128],
            torch.arange(128, 192, dtype=torch.int32),
        )
        return _scratch_writing_forward(engine, seen_out_loc)(batch)

    result = observer._run_target_transaction(
        captured,
        [7, 8],
        forward=forward,
        mode="fast-eager",
    )

    assert seen_out_loc == [134, 135]
    assert result["state_digest_unchanged"] is True
    assert result["scratch_live_disjoint"] is True
    assert result["candidate_out_loc_lease_owned"] is True
    assert result["copied_partial_pages"] == 1
    assert result["scratch_state_restored"] is True
    assert result["free_list_order_restored"] is True
    assert result["recurrent_state_restored"] is True
    assert _all_scratch_bytes(engine) == scratch_before
    assert torch.equal(manager.free_slots, free_before)


def test_state_timing_covers_complete_transaction_not_hashes():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()
    _configure_partial_prefix(captured, manager, engine)
    _install_synthetic_target_batch(observer, captured, engine)

    result = observer._run_target_transaction(
        captured,
        [7, 8],
        forward=_successful_forward(),
        mode="fast-eager",
    )

    assert result["state_required_scope"] == "scratch-backup-through-cleanup"
    assert result["state_prepare_ms"] >= 0
    assert result["state_cleanup_ms"] >= 0
    assert result["state_ms"] == pytest.approx(
        result["state_prepare_ms"] + result["state_cleanup_ms"]
    )
    assert result["state_instrumentation_ms"] >= 0
    assert result["state_instrumentation_synchronizations"] >= 0
    assert result["target_required_wall_ms"] == 1.0
    assert result["target_instrumentation_wall_ms"] >= 0


def test_scratch_transaction_restores_owned_recurrent_slot_and_free_order():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()
    pool = _FreeSlotPool()
    engine.linear_state_pool = pool
    _configure_partial_prefix(captured, manager, engine)
    _install_synthetic_target_batch(observer, captured, engine)
    scratch_before = _all_scratch_bytes(engine)
    free_before = tuple(pool._free_slots)
    seen_out_loc = []

    result = observer._run_target_transaction(
        captured,
        [7, 8],
        forward=_scratch_writing_forward(engine, seen_out_loc),
        mode="fast-eager",
    )

    assert result["borrowed_recurrent_snapshot"] is False
    assert result["recurrent_state_restored"] is True
    assert tuple(pool._free_slots) == free_before
    assert _all_scratch_bytes(engine) == scratch_before


def test_scratch_transaction_rejects_live_request_page_table_as_dummy():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()
    _configure_partial_prefix(captured, manager, engine)
    _install_synthetic_target_batch(observer, captured, engine)
    engine.dummy_req.table_idx = captured.table_idx
    scratch_before = _all_scratch_bytes(engine)
    free_before = manager.free_slots.clone()

    with pytest.raises(RuntimeError, match="live request page-table row"):
        observer._run_target_transaction(
            captured,
            [7, 8],
            forward=_successful_forward(),
            mode="fast-eager",
        )

    assert _all_scratch_bytes(engine) == scratch_before
    assert torch.equal(manager.free_slots, free_before)


def test_scratch_transaction_rejects_collision_with_live_partial_page():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()
    _configure_partial_prefix(captured, manager, engine)
    _install_synthetic_target_batch(observer, captured, engine)
    manager.free_slots = torch.tensor([64, 128, 192], dtype=torch.int32)
    scratch_before = _all_scratch_bytes(engine)
    free_before = manager.free_slots.clone()

    with pytest.raises(RuntimeError, match="forbidden"):
        observer._run_target_transaction(
            captured,
            [7, 8],
            forward=_successful_forward(),
            mode="fast-eager",
        )

    assert _all_scratch_bytes(engine) == scratch_before
    assert torch.equal(manager.free_slots, free_before)


@pytest.mark.parametrize(
    "fault_point",
    [
        "after-page-table-rewrite",
        "after-kv-copy",
        "after-recurrent-copy",
        "after-target-forward",
    ],
)
def test_scratch_transaction_restores_every_byte_at_named_fault_point(fault_point):
    observer, captured, manager, pool, target_kv, engine = _transaction_fixture()
    _configure_partial_prefix(captured, manager, engine)
    _install_synthetic_target_batch(observer, captured, engine)
    scratch_before = _all_scratch_bytes(engine)
    free_before = manager.free_slots.clone()
    seen_out_loc = []

    def inject(stage):
        if stage == fault_point:
            raise RuntimeError(f"injected {stage}")

    observer._transaction_fault_hook = inject
    with pytest.raises(RuntimeError, match=f"injected {fault_point}"):
        observer._run_target_transaction(
            captured,
            [7, 8],
            forward=_scratch_writing_forward(engine, seen_out_loc),
            mode="fast-eager",
        )

    assert _all_scratch_bytes(engine) == scratch_before
    assert torch.equal(manager.free_slots, free_before)


@pytest.mark.parametrize(
    ("family", "mutate"),
    [
        ("request-metadata", lambda captured, pool, kv, engine: setattr(captured, "uid", 10)),
        ("page-table-prefix", lambda captured, pool, kv, engine: engine.page_table[0, 1].add_(1)),
        ("linear-conv", lambda captured, pool, kv, engine: pool.conv_states[:, 1].add_(1)),
        (
            "linear-recurrent",
            lambda captured, pool, kv, engine: pool.recurrent_states[:, 1].add_(1),
        ),
        (
            "linear-extra:ple",
            lambda captured, pool, kv, engine: pool.slot_states["ple"][:, 1].add_(1),
        ),
        ("qsa-ring:0", lambda captured, pool, kv, engine: kv._pending_ring[0, 0].add_(1)),
        (
            "qsa-position-ring:0",
            lambda captured, pool, kv, engine: kv._pending_position_ring[0, 0].add_(1),
        ),
        ("target-k:0", lambda captured, pool, kv, engine: kv._k_buffer[0, 0].add_(1)),
        ("target-v:0", lambda captured, pool, kv, engine: kv._v_buffer[0, 0].add_(16)),
        (
            "qsa-compressed:0",
            lambda captured, pool, kv, engine: kv._cmp_k_buffer[0, 0].add_(1),
        ),
    ],
)
def test_target_transaction_names_exact_changed_live_state_family(family, mutate):
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()

    def forward(batch):
        mutate(captured, pool, target_kv, engine)
        return _successful_forward()(batch)

    with pytest.raises(
        RuntimeError,
        match=rf"^MTP verifier changed live target state families: {family}$",
    ):
        observer._run_target_transaction(
            captured,
            [7, 8],
            forward=forward,
            mode="fast-eager",
        )


def test_target_transaction_state_family_orders_multiple_changes():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()

    def forward(batch):
        pool.conv_states[:, captured.linear_slot_idx].add_(1)
        target_kv._pending_ring[captured.table_idx, 0].add_(1)
        return _successful_forward()(batch)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^MTP verifier changed live target state families: "
            r"linear-conv,qsa-ring:0$"
        ),
    ):
        observer._run_target_transaction(
            captured,
            [7, 8],
            forward=forward,
            mode="fast-eager",
        )


def test_target_transaction_state_family_map_is_silent_when_live_state_is_unchanged():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()

    result = observer._run_target_transaction(
        captured,
        [7, 8],
        forward=_successful_forward(),
        mode="fast-eager",
    )

    assert result["state_digest_unchanged"] is True


def test_target_state_family_hashes_one_production_page():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()

    families = observer._target_state_family_digests(captured)

    assert families["target-k:0"] == observer._state_tensor_digest(
        target_kv.k_cache(0)[0]
    )
    assert families["target-v:0"] == observer._state_tensor_digest(
        target_kv.v_cache(0)[0]
    )
    assert families["qsa-compressed:0"] == observer._state_tensor_digest(
        target_kv.cmp_k_cache(0)[0:16]
    )


def test_target_state_family_map_is_deterministic_sha256():
    observer, captured, manager, pool, target_kv, engine = _named_transaction_fixture()

    first = observer._target_state_family_digests(captured)
    second = observer._target_state_family_digests(captured)

    assert first == second
    assert set(first) == {
        "request-metadata",
        "page-table-prefix",
        "linear-conv",
        "linear-recurrent",
        "linear-extra:ple",
        "qsa-ring:0",
        "qsa-position-ring:0",
        "target-k:0",
        "target-v:0",
        "qsa-compressed:0",
    }
    assert all(len(value) == 64 for value in first.values())
    assert all(int(value, 16) >= 0 for value in first.values())
    assert observer._combine_state_family_digests(first) == (
        observer._combine_state_family_digests(second)
    )


def _transaction_fixture():
    manager = _manager(pages=4)
    manager.page_table[0, :64].copy_(torch.arange(64, dtype=torch.int32))
    manager.free_slots = torch.tensor([64, 128, 192], dtype=torch.int32)
    pool = _BorrowOnlyPool()
    target_kv = _TargetKV()
    engine = type("Engine", (), {})()
    engine.page_table = manager.page_table
    engine.linear_state_pool = pool
    engine.kv_cache = target_kv
    engine.attn_backend = type("Backend", (), {"_idx_slot": {0: 0}, "ratio": 4})()
    engine.dummy_req = type("Dummy", (), {"table_idx": 1})()

    observer = object.__new__(MTPShadowObserver)
    observer.engine = engine
    observer.cache_manager = manager
    observer.device = torch.device("cpu")
    observer._target_state_family_digests = lambda captured: {
        "synthetic": hashlib.sha256(_all_scratch_bytes(engine)).hexdigest()
    }
    captured = type(
        "Capture",
        (),
        {
            "uid": 9,
            "cached_len": 64,
            "device_len": 64,
            "table_idx": 0,
            "linear_slot_idx": 1,
            "protected_linear_slots": (1, 2),
        },
    )()
    _install_synthetic_target_batch(observer, captured, engine)
    return observer, captured, manager, pool, target_kv, engine


def _poison_transaction_scratch(pool, target_kv, engine):
    engine.page_table[1, :128].fill_(-1)
    pool.conv_states[:, 3].fill_(-2)
    pool.recurrent_states[:, 3].fill_(-3)
    pool.slot_states["ple"][:, 3].fill_(-4)
    target_kv._pending_ring[1].fill_(-5)
    target_kv._pending_position_ring[1].fill_(-6)
    target_kv._cmp_k_buffer[:, target_kv.cmp_scratch_base + 1].fill_(-7)


@pytest.mark.parametrize("fault_stage", ["recurrent", "qsa", "forward", "logits"])
def test_eager_transaction_restores_every_resource_at_each_fault_stage(fault_stage):
    observer, captured, manager, pool, target_kv, engine = _transaction_fixture()
    before = _all_scratch_bytes(engine)
    free_before = manager.free_slots.clone()

    if fault_stage == "recurrent":
        original_copy = pool.copy_from

        def fail_copy(source, destination):
            original_copy(source, destination)
            raise RuntimeError("injected recurrent-copy failure")

        pool.copy_from = fail_copy
    elif fault_stage == "qsa":
        observer._target_verify_batch = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected QSA-preparation failure")
        )

    def forward(batch):
        _poison_transaction_scratch(pool, target_kv, engine)
        if fault_stage == "forward":
            raise RuntimeError("injected eager target forward failure")
        return MTPVerifyForwardResult(
            mode="fast-eager",
            logits=torch.zeros(1 if fault_stage == "logits" else 2, 4),
            required_wall_ms=1.0,
            core_cuda_ms=1.0,
            required_synchronizations=0,
            instrumentation_wall_ms=0.25,
            instrumentation_synchronizations=0,
            expert_movement={"available": False},
        )

    message = "injected" if fault_stage != "logits" else "wrong logit-row count"
    with pytest.raises(RuntimeError, match=message):
        observer._run_target_transaction(
            captured,
            [7, 8],
            forward=forward,
            mode="fast-eager",
        )

    assert _all_scratch_bytes(engine) == before
    assert torch.equal(manager.free_slots, free_before)
    assert pool.num_free_slots == 0


@pytest.mark.parametrize("mode", ["graph-capture", "fast-graph"])
@pytest.mark.parametrize("fault_stage", ["forward", "logits"])
def test_graph_transaction_restores_every_resource_on_capture_or_replay_fault(
    mode, fault_stage
):
    observer, captured, manager, pool, target_kv, engine = _transaction_fixture()
    before = _all_scratch_bytes(engine)
    free_before = manager.free_slots.clone()

    def forward(batch):
        _poison_transaction_scratch(pool, target_kv, engine)
        if fault_stage == "forward":
            raise RuntimeError("injected graph target forward failure")
        return MTPVerifyForwardResult(
            mode=mode,
            logits=torch.zeros(1, 4),
            required_wall_ms=1.0,
            core_cuda_ms=1.0,
            required_synchronizations=1,
            instrumentation_wall_ms=0.25,
            instrumentation_synchronizations=0,
            expert_movement={"available": False},
        )

    message = "injected" if fault_stage == "forward" else "wrong logit-row count"
    with pytest.raises(RuntimeError, match=message):
        observer._run_target_transaction(
            captured,
            [7, 8],
            forward=forward,
            mode=mode,
        )

    assert _all_scratch_bytes(engine) == before
    assert torch.equal(manager.free_slots, free_before)
    assert pool.num_free_slots == 0
