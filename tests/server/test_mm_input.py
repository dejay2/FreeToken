"""Image input plumbing: content-part rendering, ref collection, fetch, and the gate.

No GPU and no model checkpoint: everything here is pure frontend logic."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from freetoken.mm.media import collect_image_refs, fetch_image_bytes, image_reject_reason
from freetoken.server.generation import GenerationError, render_messages
from freetoken.server.stats import derive_model_card

PNG = base64.b64encode(b"fakepng").decode()


def _config(**overrides):
    fields = dict(
        model_path="/nonexistent", allowed_media_domains="", allowed_local_media_path="",
        served_model_name="unit-model", max_seq_len=8192, model_config=SimpleNamespace(),
    )
    text_model_only = overrides.pop("text_model_only", False)
    serves_images = overrides.pop("vision_enabled", False)
    mm = SimpleNamespace(
        text_model_only=text_model_only,
        disabled_encoders=frozenset({"vision", "audio"}) if text_model_only else frozenset(),
    )
    return SimpleNamespace(mm=mm, served_modalities=frozenset({"image"}) if serves_images else frozenset(), **{**fields, **overrides})


def test_image_url_part_becomes_template_image_part():
    msgs = render_messages(
        [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]}]
    )
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image"
    refs = collect_image_refs(msgs)
    assert refs == [{"kind": "url", "data": "https://x/y.png"}]
    # refs are popped; the template-facing part stays
    assert msgs[0]["content"][1] == {"type": "image"}
    assert collect_image_refs(msgs) == []


def test_collect_refs_preserves_prompt_order():
    msgs = render_messages(
        [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u1"}}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "u2"}},
                {"type": "image_url", "image_url": {"url": "u3"}},
            ]},
        ]
    )
    assert [r["data"] for r in collect_image_refs(msgs)] == ["u1", "u2", "u3"]


def test_fetch_decodes_data_uri_and_raw_b64():
    refs = [
        {"kind": "url", "data": f"data:image/png;base64,{PNG}"},
        {"kind": "b64", "data": PNG},
    ]
    assert asyncio.run(fetch_image_bytes(refs, _config())) == [b"fakepng", b"fakepng"]


def test_fetch_failures_surface_as_generation_error(monkeypatch):
    from freetoken.server import generation as gen

    # bypass the capability gate; the fetch failure itself must become a GenerationError
    monkeypatch.setattr(gen, "image_reject_reason", lambda config: None)
    state = SimpleNamespace(config=_config())
    with pytest.raises(GenerationError):
        asyncio.run(gen._resolve_images([{"kind": "url", "data": "ftp://nope"}], state))


def test_image_gate_reasons():
    assert "text-model-only" in image_reject_reason(_config(text_model_only=True))
    assert "vision" in image_reject_reason(_config(model_path="/nonexistent"))


def test_stats_model_card_lists_the_accepted_input_modalities():
    assert derive_model_card(_config())["input_modalities"] == ["text"]
    assert derive_model_card(_config(vision_enabled=True))["input_modalities"] == ["text", "image"]


def test_media_domain_allowlist():
    from freetoken.mm.media import _check_media_domain

    config = _config(allowed_media_domains="cdn.example.com, Other.COM.")
    _check_media_domain("https://cdn.example.com/a.png", config)  # allowed: no raise
    _check_media_domain("https://OTHER.com./b.png", config)  # case/root-dot normalized
    with pytest.raises(ValueError, match="allowed domains"):
        _check_media_domain("https://evil.com/a.png", config)
    # empty allowlist admits any domain
    _check_media_domain("https://evil.com/a.png", _config())

    with pytest.raises(ValueError, match="allowed domains"):
        asyncio.run(
            fetch_image_bytes([{"kind": "url", "data": "https://evil.com/a.png"}], config)
        )


def test_local_media_requires_allowlisted_root(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"fakepng")
    url = f"file://{img}"

    # gate off (default): rejected
    with pytest.raises(ValueError, match="allowed-local-media-path"):
        asyncio.run(fetch_image_bytes([{"kind": "url", "data": url}], _config()))

    # gate on, file under the root: served
    config = _config(allowed_local_media_path=str(tmp_path))
    assert asyncio.run(fetch_image_bytes([{"kind": "url", "data": url}], config)) == [b"fakepng"]

    # a path outside the root is rejected even with the gate on
    with pytest.raises(ValueError, match="subpath"):
        asyncio.run(fetch_image_bytes([{"kind": "url", "data": "file:///etc/hostname"}], config))


def test_image_token_budget_flags_land_in_the_multimodal_config():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3VLForConditionalGeneration"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        args, _ = parse_args([
            "--model", "/models/anon", "--image-min-tokens", "64", "--image-max-tokens", "1024",
            "--mm-processor-kwargs", '{"size": {"longest_edge": 4096}}',
        ])
        assert (args.mm.image_min_tokens, args.mm.image_max_tokens) == (64, 1024)
        assert args.mm.processor_kwargs == {"size": {"longest_edge": 4096}}
        assert parse_args(["--model", "/models/anon"])[0].mm.processor_kwargs == {}
        with pytest.raises(SystemExit):  # argparse reports the bad pair and exits
            parse_args(["--model", "/models/anon", "--image-min-tokens", "2048", "--image-max-tokens", "1024"])


def test_local_image_spelling_and_percent_data_survive_normalization(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"picture")
    messages = render_messages([{"role": "user", "content": [
        {"type": "image", "image": str(image)},
        {"type": "input_image", "image_url": "data:image/png,picture%2Dbytes"},
    ]}])
    refs = collect_image_refs(messages)
    assert asyncio.run(fetch_image_bytes(refs, _config(allowed_local_media_path=str(tmp_path)))) == [
        b"picture", b"picture-bytes",
    ]


def test_submit_keeps_refs_for_reuse_and_waits_for_admission():
    from copy import deepcopy
    from freetoken.core import SamplingParams
    from freetoken.server.generation import GenSpec, submit_generation

    messages = render_messages([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
    ]}])
    before = deepcopy(messages)
    sent = []

    async def new_user():
        return 7

    async def send_one(msg):
        sent.append(msg)

    state = SimpleNamespace(config=_config(vision_enabled=True), new_user=new_user, send_one=send_one)
    spec = GenSpec(messages=messages, sampling_params=SamplingParams())
    assert asyncio.run(submit_generation(spec, state)) == 7
    assert asyncio.run(submit_generation(spec, state)) == 7
    assert messages == before
    assert [msg.images for msg in sent] == [[b"fakepng"], [b"fakepng"]]
    assert sent[0].text[0]["content"] == [{"type": "image"}]


def test_disabled_images_rejected_before_admission():
    from freetoken.core import SamplingParams
    from freetoken.server.generation import GenSpec, submit_generation

    spec = GenSpec(messages=render_messages([{"role": "user", "content": [
        {"type": "image_url", "image_url": f"data:image/png;base64,{PNG}"},
    ]}]), sampling_params=SamplingParams())
    # No admission method: rejecting images must happen before allocating a request.
    with pytest.raises(GenerationError, match="text-model-only"):
        asyncio.run(submit_generation(spec, SimpleNamespace(config=_config(text_model_only=True))))


def test_http_fetch_bounds_size_and_checks_redirect_domains(monkeypatch):
    import httpx
    import freetoken.mm.media as media

    def handle(request):
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "https://other.test/image"})
        return httpx.Response(200, content=b"12345")

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    refs = [{"kind": "url", "data": "https://allowed.test/image"}]
    assert asyncio.run(fetch_image_bytes(refs, _config())) == [b"12345"]
    monkeypatch.setattr(media, "_MAX_IMAGE_BYTES", 4)
    with pytest.raises(ValueError, match="exceeds"):
        asyncio.run(fetch_image_bytes(refs, _config()))
    with pytest.raises(ValueError, match="allowed domains"):
        asyncio.run(fetch_image_bytes(
            [{"kind": "url", "data": "https://allowed.test/redirect"}],
            _config(allowed_media_domains="allowed.test"),
        ))
