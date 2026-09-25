"""Package-wide runtime hygiene: TP info set once, the global ctx never leaks across tests."""

import dataclasses

import pytest


@pytest.fixture(autouse=True)
def _runtime():
    import freetoken.core as core
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None


@pytest.fixture
def exl3_vision_config():
    """The dense vision config test_vision.py builds, packed EXL3 (spec 2026-09-25 section 5).

    Exl3Linear needs out_features % 128 == 0, so hidden/intermediate/out_hidden are bumped to
    128/256/128 (the merged patch-merger width, hidden*spatial_merge_size**2, lands on 512).
    """
    from .test_vision import _config

    return dataclasses.replace(
        _config(),
        exl3=True,
        exl3_k=5,
        hidden_size=128,
        intermediate_size=256,
        out_hidden_size=128,
    )
