"""Engine._run_mm_encoder against a fake model: chunked gathers, a shared image, precomputed embeddings, orphan jobs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import Engine
from freetoken.message.backend import MMItem
from freetoken.mm.encoder_cache import EncoderCache

H = 8


class _FakeVision:
    def __init__(self):
        self.calls = 0

    def encode(self, item: MMItem) -> torch.Tensor:
        self.calls += 1
        return torch.full((item.num_tokens, H), float(item.hash))


def _engine(cache: EncoderCache, model=None) -> SimpleNamespace:
    return SimpleNamespace(
        encoder_cache=cache, device=torch.device("cpu"), dtype=torch.float32,
        model=model or _FakeVision(),
    )


def _item(h: int, n_tokens: int, precomputed: torch.Tensor | None = None) -> MMItem:
    feature = None if precomputed is not None else torch.zeros(1)
    return MMItem(
        modality="image", hash=h, pad_value=0, offsets=[[0, n_tokens]],
        feature=feature, precomputed_embeddings=precomputed,
    )


def _batch(jobs, plan) -> SimpleNamespace:
    return SimpleNamespace(mm_encoder_jobs=jobs, mm_gather_plan=plan, mm_embeds=None, mm_rows=None)


def test_entry_lives_until_its_consumer_gathers_the_last_row():
    cache = EncoderCache(storage="cpu")
    eng = _engine(cache)
    item = _item(h=3, n_tokens=4)
    cache.register(3, 7, 4)  # admission claims the whole image
    # chunk 1 consumes rows [0, 2) of a 4-token image
    Engine._run_mm_encoder(eng, _batch([item], [(7, 3, 0, 2, 4, 0)]))
    assert cache._entries[3].remaining == {7: 2}
    assert item.feature is None
    # chunk 2 finishes the image; the entry dies with its last claim
    Engine._run_mm_encoder(eng, _batch([], [(7, 3, 2, 4, 4, 0)]))
    assert not cache.has(3)
    assert eng.model.calls == 1


def test_shared_image_is_encoded_once_and_sliced_per_request():
    cache = EncoderCache(storage="cpu")
    eng = _engine(cache)
    a, b = _item(h=5, n_tokens=4), _item(h=5, n_tokens=4)
    cache.register(5, 1, 4)
    cache.register(5, 2, 4)
    batch = _batch([a, b], [(1, 5, 0, 4, 4, 0), (2, 5, 0, 2, 4, 4)])
    Engine._run_mm_encoder(eng, batch)
    assert eng.model.calls == 1
    assert batch.mm_embeds.shape == (6, H)
    assert torch.equal(batch.mm_embeds, torch.full((6, H), 5.0))
    # request 1 consumed its image; request 2 still has rows to gather in its next chunk
    assert cache._entries[5].remaining == {2: 2}


def test_precomputed_embeddings_bypass_the_encoder():
    cache = EncoderCache(storage="cpu")

    class _NoVision:
        def encode(self, item):
            raise AssertionError("encoder must not run for precomputed embeddings")

    eng = _engine(cache, model=_NoVision())
    emb = torch.arange(2 * H, dtype=torch.float32).view(2, H)
    item = _item(h=4, n_tokens=2, precomputed=emb)
    cache.register(4, 1, 2)
    batch = _batch([item], [(1, 4, 0, 2, 2, 0)])
    Engine._run_mm_encoder(eng, batch)
    assert torch.equal(batch.mm_embeds, emb)
    assert item.precomputed_embeddings is None
    assert not cache.has(4)


def test_job_without_gather_row_fails_loudly():
    eng = _engine(EncoderCache(storage="cpu"))
    with pytest.raises(AssertionError, match="encoder jobs without gather rows"):
        Engine._run_mm_encoder(eng, _batch([_item(h=1, n_tokens=2)], []))


def test_mixed_legacy_and_content_keyed_images_keep_both_embeddings():
    from freetoken.scheduler.scheduler import Scheduler

    legacy = torch.full((2, H), 9.0)
    old = SimpleNamespace(uid=1, mm_items=None, mm_embeds=legacy,
        input_ids=torch.tensor([1, 99, 99]), cached_len=0, device_len=3, extend_len=3)
    item = _item(5, 2)
    new = SimpleNamespace(uid=2, mm_items=[item], mm_embeds=None,
        input_ids=torch.tensor([1000, 1000]), cached_len=0, device_len=2, extend_len=2)
    cache = EncoderCache()
    cache.register(5, 2, 2)
    batch = _batch(None, None)
    batch.reqs = batch.padded_reqs = [old, new]
    scheduler = SimpleNamespace(engine=SimpleNamespace(encoder_cache=cache),
        config=SimpleNamespace(model_config=SimpleNamespace(image_token_id=99)),
        device=torch.device('cpu'), _bidirectional_mm=False)
    Scheduler._gather_multimodal(scheduler, batch)
    Engine._run_mm_encoder(_engine(cache), batch)
    rows = dict(zip(batch.mm_rows.tolist(), batch.mm_embeds.tolist()))
    assert rows == {1: [9.0]*H, 2: [9.0]*H, 3: [5.0]*H, 4: [5.0]*H}
    assert old.mm_embeds is None


@pytest.mark.parametrize('architecture,env,enabled', [
    ('Qwen4ExpForConditionalGeneration', None, False),
    ('Qwen4ExpForConditionalGeneration', '1', True),
    ('Gemma4ForConditionalGeneration', None, True),
    ('Gemma4ForConditionalGeneration', '0', False),
])
def test_engine_encoder_activation_preserves_legacy_gate(monkeypatch, architecture, env, enabled):
    from freetoken.engine.config import EngineConfig
    from freetoken.mm.config import MultimodalConfig

    if env is None:
        monkeypatch.delenv('FREETOKEN_LOAD_VISION', raising=False)
    else:
        monkeypatch.setenv('FREETOKEN_LOAD_VISION', env)
    config = EngineConfig(model_path='unused', tp_info=SimpleNamespace(rank=0, size=1), dtype=torch.float32)
    config.__dict__['hf_config'] = SimpleNamespace(architectures=[architecture], vision_config={})
    assert bool(config.active_encoders) is enabled
    assert ('image' in config.served_modalities) is enabled
    disabled = EngineConfig(model_path='unused', tp_info=SimpleNamespace(rank=0, size=1), dtype=torch.float32, mm=MultimodalConfig(disabled_encoders=frozenset({'vision'})))
    disabled.__dict__['hf_config'] = config.hf_config
    assert not disabled.active_encoders
