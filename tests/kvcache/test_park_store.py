from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.park_store import ParkStore, rolling_page_keys
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _qsa_pool(num_pages: int = 6) -> QSAKVCache:
    return QSAKVCache(
        num_kv_heads=1,
        num_layers=2,
        head_dim=4,
        num_pages=num_pages,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_head_dim=3,
        num_index_layers=2,
        index_ratio=2,
        num_req_slots=2,
        layer_ids=(0, 1),
    )


def _state_pool(num_slots: int = 5) -> LinearStatePool:
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 1),
        num_key_heads=1,
        num_value_heads=1,
        key_head_dim=2,
        value_head_dim=2,
        conv_kernel_dim=3,
        output_gate="silu",
    )
    sibling = SlotStateSpec(
        name="ple_sibling",
        shape=(2, 3),
        layer_ids=(0,),
        dtype=torch.int32,
        fill_value=-1,
    )
    return LinearStatePool(
        group=group,
        num_slots=num_slots,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        tp_size=1,
        slot_states=(sibling,),
    )


def _fill_entry(
    kv_pool: QSAKVCache,
    state_pool: LinearStatePool,
    page_bases: torch.Tensor,
    state_slot: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    originals: list[torch.Tensor] = []
    value = 1
    for base in page_bases.tolist():
        for view in kv_pool.page_byte_views(base // 4):
            payload = torch.arange(view.numel(), dtype=torch.int64).reshape(view.shape) + value
            view.copy_(payload.to(view.dtype))
            originals.append(view.clone())
            value += view.numel() + 7
    state_originals: list[torch.Tensor] = []
    for view in state_pool.slot_byte_views(state_slot):
        payload = torch.arange(view.numel(), dtype=torch.int64).reshape(view.shape) + value
        view.copy_(payload.to(view.dtype))
        state_originals.append(view.clone())
        value += view.numel() + 11
    return originals, state_originals


def _assert_entry_equal(
    kv_pool: QSAKVCache,
    state_pool: LinearStatePool,
    page_bases: torch.Tensor,
    state_slot: int,
    kv_expected: list[torch.Tensor],
    state_expected: list[torch.Tensor],
) -> None:
    restored_kv = [
        view.clone()
        for base in page_bases.tolist()
        for view in kv_pool.page_byte_views(base // 4)
    ]
    restored_state = [view.clone() for view in state_pool.slot_byte_views(state_slot)]
    assert len(restored_kv) == len(kv_expected)
    assert len(restored_state) == len(state_expected)
    assert all(torch.equal(got, want) for got, want in zip(restored_kv, kv_expected, strict=True))
    assert all(
        torch.equal(got, want)
        for got, want in zip(restored_state, state_expected, strict=True)
    )


def _store(
    mode: str,
    tmp_path: Path,
    kv_pool: QSAKVCache,
    state_pool: LinearStatePool,
    *,
    fingerprint: str = "model-A",
    ram_budget_bytes: int = 1 << 20,
    disk_budget_bytes: int = 1 << 20,
) -> ParkStore:
    return ParkStore(
        mode=mode,
        page_size=4,
        kv_pool=kv_pool,
        state_pool=state_pool,
        fingerprint=fingerprint,
        min_tokens=8,
        ram_budget_bytes=ram_budget_bytes,
        ssd_dir=tmp_path,
        disk_budget_bytes=disk_budget_bytes,
        pinned_window_bytes=4096,
    )


def test_rolling_keys_are_prefix_chained_and_model_scoped():
    tokens = torch.arange(12, dtype=torch.int32)
    keys = rolling_page_keys(tokens, page_size=4, fingerprint="model-A")
    assert len(keys) == 3
    assert keys[:2] == rolling_page_keys(tokens[:8], page_size=4, fingerprint="model-A")
    assert keys != rolling_page_keys(tokens, page_size=4, fingerprint="model-B")
    changed = tokens.clone()
    changed[1] += 100
    assert keys[0] != rolling_page_keys(changed, page_size=4, fingerprint="model-A")[0]


def test_ram_round_trip_preserves_qsa_and_all_linear_state_bytes(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    source_pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    kv_expected, state_expected = _fill_entry(kv_pool, state_pool, source_pages, source_slot)
    store = _store("ram", tmp_path, kv_pool, state_pool)
    tokens = torch.arange(8, dtype=torch.int32)

    assert store.save(tokens, source_pages, source_slot)
    for view in kv_pool.page_byte_views(2):
        view.fill_(99)
    for view in kv_pool.page_byte_views(3):
        view.fill_(99)
    for view in state_pool.slot_byte_views(target_slot):
        view.fill_(99)

    hit = store.lookup(torch.cat([tokens, torch.tensor([99], dtype=torch.int32)]))
    assert hit is not None and hit.token_count == 8
    store.restore(hit, torch.tensor([8, 12], dtype=torch.int32), target_slot)
    _assert_entry_equal(
        kv_pool,
        state_pool,
        torch.tensor([8, 12], dtype=torch.int32),
        target_slot,
        kv_expected,
        state_expected,
    )
    assert store.status()["hits"] == 1


def test_ssd_round_trip_rebuilds_index_without_manifest(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    source_pages = torch.tensor([0, 4], dtype=torch.int32)
    source_slot, target_slot = state_pool.alloc(2)
    kv_expected, state_expected = _fill_entry(kv_pool, state_pool, source_pages, source_slot)
    tokens = torch.arange(8, dtype=torch.int32)

    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, source_pages, source_slot)
    assert (tmp_path / "park.json").is_file()
    store.close()
    (tmp_path / "park.json").unlink()

    rebuilt = _store("ssd", tmp_path, kv_pool, state_pool)
    hit = rebuilt.lookup(torch.cat([tokens, torch.tensor([88], dtype=torch.int32)]))
    assert hit is not None
    rebuilt.restore(hit, torch.tensor([8, 12], dtype=torch.int32), target_slot)
    _assert_entry_equal(
        kv_pool,
        state_pool,
        torch.tensor([8, 12], dtype=torch.int32),
        target_slot,
        kv_expected,
        state_expected,
    )
    header = next(tmp_path.glob("*.park")).read_bytes()[:16]
    assert header.startswith(b"FTKVPARK")
    assert json.loads((tmp_path / "park.json").read_text(encoding="utf-8"))["version"] == 1


def test_ssd_pinned_window_failure_disables_store_without_failing_boot(
    tmp_path: Path, monkeypatch
):
    kv_pool, state_pool = _qsa_pool(), _state_pool()

    def fail_window(_self, _nbytes):
        raise RuntimeError("pin quota exhausted")

    monkeypatch.setattr(ParkStore, "_allocate_window", fail_window)
    store = _store("ssd", tmp_path, kv_pool, state_pool)
    assert store.status()["disabled"] is True
    assert store.lookup(torch.arange(8, dtype=torch.int32)) is None
    store.close()


def test_fingerprint_mismatch_is_rejected_on_startup(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    _fill_entry(kv_pool, state_pool, pages, slot)
    tokens = torch.arange(8, dtype=torch.int32)
    first = _store("ssd", tmp_path, kv_pool, state_pool, fingerprint="model-A")
    assert first.save(tokens, pages, slot)
    first.close()

    wrong_model = _store("ssd", tmp_path, kv_pool, state_pool, fingerprint="model-B")
    assert wrong_model.lookup(tokens) is None
    assert wrong_model.status()["parked_count"] == 0


def test_token_verification_turns_a_hash_collision_into_a_miss(tmp_path: Path):
    kv_pool, state_pool = _qsa_pool(), _state_pool()
    pages = torch.tensor([0, 4], dtype=torch.int32)
    slot = state_pool.alloc(1)[0]
    _fill_entry(kv_pool, state_pool, pages, slot)
    tokens = torch.arange(8, dtype=torch.int32)
    store = _store("ram", tmp_path, kv_pool, state_pool)
    assert store.save(tokens, pages, slot)
    key = rolling_page_keys(tokens, 4, "model-A")[-1]
    store._entries[key].token_ids[0] = -123

    assert store.lookup(tokens) is None
    status = store.status()
    assert status["misses"] == 1
    assert status["parked_count"] == 0


def test_ram_and_ssd_budgets_evict_the_oldest_entry(tmp_path: Path):
    for mode in ("ram", "ssd"):
        root = tmp_path / mode
        kv_pool, state_pool = _qsa_pool(), _state_pool()
        pages = torch.tensor([0, 4], dtype=torch.int32)
        slot = state_pool.alloc(1)[0]
        _fill_entry(kv_pool, state_pool, pages, slot)
        probe = _store(mode, root, kv_pool, state_pool)
        one_entry = probe.storage_bytes(8)
        probe.close()
        store = _store(
            mode,
            root,
            kv_pool,
            state_pool,
            ram_budget_bytes=one_entry + 64,
            disk_budget_bytes=one_entry + 64,
        )
        first = torch.arange(8, dtype=torch.int32)
        second = torch.arange(100, 108, dtype=torch.int32)
        assert store.save(first, pages, slot)
        assert store.save(second, pages, slot)
        assert store.lookup(first) is None
        assert store.lookup(second) is not None
        assert store.status()["parked_count"] == 1
