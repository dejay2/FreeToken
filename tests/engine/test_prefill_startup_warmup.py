from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import Engine


@pytest.mark.parametrize('mrope', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_short_warmup_uses_dummy_state_and_restores_it(monkeypatch, fail, mrope):
    row = torch.full((128,), 999, dtype=torch.int32)
    seen = []
    reset = []
    @contextmanager
    def forward_batch(batch):
        assert batch.phase == 'prefill'
        if mrope:
            assert torch.equal(batch.mrope_positions, batch.positions.expand(3, -1))
        assert batch.reqs[0].linear_slot_idx == 0
        seen.append(batch.input_ids.numel())
        yield
    def forward():
        if fail:
            raise RuntimeError('warmup failed')
    event = SimpleNamespace(record=lambda _: None, elapsed_time=lambda _: 1)
    monkeypatch.setattr(torch.cuda, 'Event', lambda **kw: event)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda _: None)
    engine = SimpleNamespace(config=SimpleNamespace(model_config=SimpleNamespace(model_is_mrope=mrope)), max_seq_len=128, page_table=row.unsqueeze(0),
        dummy_req=SimpleNamespace(table_idx=0, linear_slot_idx=0), stream=None,
        device=torch.device('cpu'), ctx=SimpleNamespace(forward_batch=forward_batch),
        model=SimpleNamespace(forward=forward),
        attn_backend=SimpleNamespace(prepare_metadata=lambda _:None),
        moe_offload_cache=SimpleNamespace(reset=lambda:reset.append(True)))
    if fail:
        with pytest.raises(RuntimeError, match='warmup failed'):
            Engine._warmup_prefill(engine, lengths=[64])
    else:
        Engine._warmup_prefill(engine, lengths=[64])
    assert seen == [64]
    assert row.tolist() == [999] * 128
    assert reset == [True]
