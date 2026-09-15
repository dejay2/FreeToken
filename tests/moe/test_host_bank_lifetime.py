"""Host ownership must follow resident expert tensors through governor moves."""
import gc
import weakref

import pytest
import torch

from freetoken.moe import host_banks as hb
from freetoken.kernel import pinned


@pytest.fixture
def registration(monkeypatch):
    calls = []
    monkeypatch.setattr(pinned, "host_register", lambda addr, size: calls.append(("register", addr)))
    monkeypatch.setattr(pinned, "host_unregister", lambda addr: calls.append(("unregister", addr)), raising=False)
    yield calls
    # Keep independent tests from inheriting deliberately failed unregisters.
    for bank in list(getattr(hb, "_REGISTERED_BANKS", {}).values()):
        bank.free()


def test_gpu_owned_plan_does_not_disable_requested_cuda_host_alloc(monkeypatch):
    monkeypatch.setenv("FREETOKEN_BANK_CUDA_ALLOC", "1")
    monkeypatch.setattr(pinned, "alloc_pinned_tensor", lambda size, dtype: torch.empty(size, dtype=dtype))
    with hb.requested_residency(["gpu_owned", "pinned"], device=torch.device("cpu")):
        bank = hb.HostBank((16,), torch.uint8)
    assert bank._backing == "cuda"
    bank.free()


@pytest.mark.parametrize("unpinned", ["locked", "pageable"])
def test_cpu_host_layers_still_veto_cuda_allocation(monkeypatch, unpinned):
    monkeypatch.setenv("FREETOKEN_BANK_CUDA_ALLOC", "1")
    with hb.requested_residency([unpinned, "pinned"], device=torch.device("cpu")):
        bank = hb.HostBank((16,), torch.uint8)
    assert bank._backing == "mmap"
    bank.free()


def test_unpinned_buffer_lives_with_tensor_without_process_lifetime_registry():
    bank = hb.HostBank((4096,), torch.uint8, backing="mmap")
    buffer = weakref.ref(bank._buf)
    alias = bank.tensor[7:]
    alias.fill_(19)
    bank.free()
    del bank
    gc.collect()
    assert buffer() is not None
    assert int(alias[0]) == 19
    del alias
    gc.collect()
    assert buffer() is None


def test_registered_buffer_unregisters_once_and_alias_stays_valid(registration):
    bank = hb.HostBank((4096,), torch.uint8, backing="mmap")
    addr = bank.addr
    bank.tensor.fill_(23)
    alias = bank.tensor[12:]
    buffer = weakref.ref(bank._buf)
    bank.pin()
    bank.pin()
    bank.free()
    bank.free()
    assert registration == [("register", addr), ("unregister", addr)]
    assert int(alias[0]) == 23
    assert bank.tensor is None
    del alias
    gc.collect()
    assert buffer() is None


def test_claim_resolves_interior_and_reinterpreted_startup_sources(registration):
    bank = hb.HostBank((4096,), torch.uint8, backing="mmap")
    bank.pin()
    alias = bank.tensor[16:80].view(torch.float32)
    owners = hb.claim_host_banks({"weights": [alias]})
    assert owners == {0: {"weights": bank}}
    assert hb.claim_host_banks({"weights": [alias]}) == {}
    bank.free()


def test_claim_refuses_owner_shared_between_movable_layers(registration):
    bank = hb.HostBank((4096,), torch.uint8, backing="mmap")
    bank.pin()
    with pytest.raises(ValueError, match="multiple layers"):
        hb.claim_host_banks({"weights": [bank.tensor[:64], bank.tensor[64:128]]})
    assert hb.claim_host_banks({"weights": [bank.tensor]}) == {0: {"weights": bank}}
    bank.free()


def test_unregister_error_keeps_allocation_owned_for_retry(registration, monkeypatch):
    bank = hb.HostBank((4096,), torch.uint8, backing="mmap")
    bank.pin()
    def fail(_addr):
        raise RuntimeError("unregister failed")
    monkeypatch.setattr(pinned, "host_unregister", fail)
    with pytest.raises(RuntimeError, match="unregister failed"):
        bank.free()
    assert bank._pinned and bank.tensor is not None and bank.addr
    monkeypatch.setattr(pinned, "host_unregister", lambda addr: None)
    bank.free()
