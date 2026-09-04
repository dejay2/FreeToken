from __future__ import annotations

import json
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.daemon.settings.download import GIB, create_router, parse_repo


PC_MEMORY = int(95.6 * GIB)
CARD_MEMORY = 32 * GIB


def _glm_config() -> dict:
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "num_hidden_layers": 45,
        "first_k_dense_replace": 3,
        "n_routed_experts": 288,
        "num_experts_per_tok": 8,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "quantization_config": {"quant_algo": "NVFP4"},
    }


def _gpt_oss_config() -> dict:
    return {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "num_hidden_layers": 24,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "quantization_config": {"quant_method": "mxfp4"},
    }


class _FakeApi:
    def __init__(self, manifests: dict[str, list[SimpleNamespace]]) -> None:
        self.manifests = manifests
        self.calls: list[tuple[str, bool]] = []

    def model_info(self, repo: str, files_metadata: bool = False):
        self.calls.append((repo, files_metadata))
        return SimpleNamespace(siblings=self.manifests[repo])


def _app(tmp_path: Path, *, api, config_fetcher, snapshot_downloader=None, **overrides):
    app = FastAPI()
    app.include_router(
        create_router(
            models_dir=tmp_path / "models",
            api_factory=api,
            config_fetcher=config_fetcher,
            snapshot_downloader=snapshot_downloader,
            pc_memory=PC_MEMORY,
            card_memory=CARD_MEMORY,
            disk_free=lambda _: 2_000 * GIB,
            **overrides,
        )
    )
    return TestClient(app)


def _wait_for(client: TestClient, job_id: str, stage: str) -> dict:
    for _ in range(200):
        body = client.get(f"/api/downloads/{job_id}").json()
        if body["stage"] == stage:
            return body
        time.sleep(0.005)
    raise AssertionError(f"download did not reach {stage}: {body}")


def test_parse_repo_accepts_five_hugging_face_forms() -> None:
    forms = [
        "https://huggingface.co/openai/gpt-oss-20b",
        "https://huggingface.co/openai/gpt-oss-20b/",
        "https://huggingface.co/openai/gpt-oss-20b/tree/main",
        "https://huggingface.co/openai/gpt-oss-20b/tree/main/",
        "openai/gpt-oss-20b",
    ]
    assert [parse_repo(form) for form in forms] == ["openai/gpt-oss-20b"] * len(forms)


def test_parse_repo_rejects_other_hosts_and_paths() -> None:
    for value in (
        "https://example.com/openai/gpt-oss-20b",
        "https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json",
    ):
        try:
            parse_repo(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid repository: {value}")


def test_preview_reports_glm_shortfall_and_gpt_oss_fit(tmp_path: Path) -> None:
    manifests = {
        "RedHatAI/GLM-5.3-Flash-NVFP4": [
            SimpleNamespace(rfilename="config.json", size=20_000),
            SimpleNamespace(rfilename="model-00001-of-00021.safetensors", size=197_800_000_000),
        ],
        "openai/gpt-oss-20b": [
            SimpleNamespace(rfilename="config.json", size=20_000),
            SimpleNamespace(rfilename="generation_config.json", size=10_000),
            SimpleNamespace(rfilename="tokenizer.json", size=100_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=5_000),
            SimpleNamespace(rfilename="special_tokens_map.json", size=1_000),
            SimpleNamespace(rfilename="chat_template.jinja", size=8_000),
            SimpleNamespace(rfilename="model.safetensors.index.json", size=50_000),
            SimpleNamespace(rfilename="model.safetensors", size=13_000_000_000),
            # These are alternate source copies, not files the engine loads.
            SimpleNamespace(rfilename="original/model.safetensors", size=39_000_000_000),
            SimpleNamespace(rfilename="original/config.json", size=99_000),
            SimpleNamespace(rfilename="metal/model.safetensors", size=39_000_000_000),
        ],
    }
    api = _FakeApi(manifests)
    configs = {
        "RedHatAI/GLM-5.3-Flash-NVFP4": _glm_config(),
        "openai/gpt-oss-20b": _gpt_oss_config(),
    }
    client = _app(tmp_path, api=api, config_fetcher=lambda repo: configs[repo])

    glm = client.get("/api/downloads/preview", params={"repo": "RedHatAI/GLM-5.3-Flash-NVFP4"})
    assert glm.status_code == 200, glm.text
    glm_body = glm.json()
    assert glm_body["fit"]["fits"] is False
    assert glm_body["fit"]["shortfallBytes"] > 60 * GIB
    assert glm_body["fit"]["expertBytes"] > 170 * 1_000_000_000
    assert glm_body["weightFiles"] == 1

    gpt = client.get("/api/downloads/preview", params={"repo": "openai/gpt-oss-20b"})
    assert gpt.status_code == 200, gpt.text
    gpt_body = gpt.json()
    assert gpt_body["fit"]["fits"] is True
    assert gpt_body["expertFormat"] == "mxfp4"
    assert gpt_body["numLayers"] == 24 and gpt_body["numExperts"] == 32
    assert gpt_body["downloadBytes"] == 13_000_194_000
    assert gpt_body["weightFiles"] == 1
    assert gpt_body["weightBytes"] == 13_000_000_000
    assert gpt_body["fit"]["needsBytes"] == (
        gpt_body["fit"]["expertBytes"] + gpt_body["fit"]["denseBytes"]
    )

    linked = client.get(
        "/api/downloads/preview",
        params={"repo": "https://huggingface.co/openai/gpt-oss-20b/tree/main"},
    )
    assert linked.status_code == 200
    assert linked.json()["downloadBytes"] == gpt_body["downloadBytes"]
    assert all(repo == "openai/gpt-oss-20b" and metadata for repo, metadata in api.calls[-2:])


def test_preview_excludes_remote_ple_header_bytes(tmp_path: Path) -> None:
    config = {
        "architectures": ["Qwen3MoeForConditionalGeneration"],
        "model_type": "tiny_moe",
        "num_hidden_layers": 1,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": 2,
        "moe_intermediate_size": 2,
        "ple_layer_ids": [0],
    }
    files = [
        SimpleNamespace(rfilename="config.json", size=20),
        SimpleNamespace(rfilename="model.safetensors", size=1_000),
    ]
    api = _FakeApi({"owner/tiny": files})
    header = json.dumps(
        {
            "model.layers.0.ple_table": {"data_offsets": [0, 600]},
            "model.layers.0.mlp.gate_proj": {"data_offsets": [600, 1_000]},
        }
    ).encode("utf-8")
    client = _app(
        tmp_path,
        api=api,
        config_fetcher=lambda _: config,
        header_fetcher=lambda repo, filename: header,
    )

    response = client.get("/api/downloads/preview", params={"repo": "owner/tiny"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pleBytes"] == 600
    assert body["fit"]["denseBytes"] == max(0, body["weightBytes"] - body["fit"]["expertBytes"] - 600)
    assert body["fit"]["needsBytes"] == body["fit"]["expertBytes"] + body["fit"]["denseBytes"]


def test_local_models_fit_excludes_ple_header_tensor_bytes(tmp_path: Path) -> None:
    models = tmp_path / "models" / "tiny"
    models.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3MoeForConditionalGeneration"],
        "model_type": "tiny_moe",
        "num_hidden_layers": 1,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": 2,
        "moe_intermediate_size": 2,
        "ple_layer_ids": [0],
    }
    (models / "config.json").write_text(json.dumps(config), encoding="utf-8")
    header = json.dumps(
        {
            "model.layers.0.ple_table": {"data_offsets": [0, 600]},
            "model.layers.0.mlp.gate_proj": {"data_offsets": [600, 1_000]},
        }
    ).encode("utf-8")
    (models / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + (b"x" * 1_000))
    client = _app(tmp_path, api=_FakeApi({}), config_fetcher=lambda _: config)

    response = client.get("/api/models")

    assert response.status_code == 200, response.text
    body = response.json()["models"][0]
    assert body["pleBytes"] == 600
    assert body["fit"]["denseBytes"] == max(0, body["weightBytes"] - body["fit"]["expertBytes"] - 600)


def test_start_refuses_existing_folder(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "gpt-oss-20b").mkdir(parents=True)
    api = _FakeApi({"openai/gpt-oss-20b": []})
    client = _app(tmp_path, api=api, config_fetcher=lambda _: _gpt_oss_config())

    response = client.post("/api/downloads", json={"repo": "openai/gpt-oss-20b"})
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_progress_counts_files_on_disk(tmp_path: Path) -> None:
    files = [
        SimpleNamespace(rfilename="config.json", size=3),
        SimpleNamespace(rfilename="weights.safetensors", size=5),
        SimpleNamespace(rfilename="original/weights.safetensors", size=50),
    ]
    api = _FakeApi({"owner/model": files})
    calls: list[str] = []

    def snapshot(repo: str, *, local_dir: str, allow_patterns=None, ignore_patterns=None, **_kwargs):
        name = allow_patterns[0] if allow_patterns else "weights.safetensors"
        calls.append(name)
        assert ignore_patterns == ["*/*"]
        target = Path(local_dir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * next(item.size for item in files if item.rfilename == name))

    client = _app(tmp_path, api=api, config_fetcher=lambda _: {"architectures": ["LlamaForCausalLM"]}, snapshot_downloader=snapshot)
    started = client.post("/api/downloads", json={"repo": "owner/model"})
    assert started.status_code == 202
    done = _wait_for(client, started.json()["id"], "done")
    assert done["receivedBytes"] == 8
    assert done["totalBytes"] == 8
    assert done["percent"] == 100.0
    assert done["files"] == ["config.json", "weights.safetensors"]
    assert calls == ["config.json", "weights.safetensors"]


def test_cancel_is_honoured_between_files_and_partial_target_can_resume(tmp_path: Path) -> None:
    files = [
        SimpleNamespace(rfilename="one.safetensors", size=3),
        SimpleNamespace(rfilename="two.safetensors", size=4),
    ]
    api = _FakeApi({"owner/model": files})
    first_file = threading.Event()
    release_first = threading.Event()
    calls: list[str] = []

    def snapshot(repo: str, *, local_dir: str, allow_patterns=None, **_kwargs):
        name = allow_patterns[0]
        calls.append(name)
        target = Path(local_dir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * next(item.size for item in files if item.rfilename == name))
        if name == "one.safetensors":
            first_file.set()
            release_first.wait(2)

    client = _app(tmp_path, api=api, config_fetcher=lambda _: {"architectures": ["LlamaForCausalLM"]}, snapshot_downloader=snapshot)
    started = client.post("/api/downloads", json={"repo": "owner/model"})
    assert started.status_code == 202
    job_id = started.json()["id"]
    assert first_file.wait(2)
    cancelled = client.post(f"/api/downloads/{job_id}/cancel")
    assert cancelled.status_code == 202
    release_first.set()
    stopped = _wait_for(client, job_id, "cancelled")
    assert stopped["receivedBytes"] == 3
    assert calls == ["one.safetensors"]
    assert (tmp_path / "models" / "model" / "one.safetensors").is_file()
    assert not (tmp_path / "models" / "model" / "two.safetensors").exists()

    # The manager remembers that this is its own partial target, so a retry can resume without
    # treating a user-created complete model folder as overwriteable.
    resumed = client.post("/api/downloads", json={"repo": "owner/model"})
    assert resumed.status_code == 202
    release_first.set()
    done = _wait_for(client, resumed.json()["id"], "done")
    assert done["receivedBytes"] == 7
    assert calls[-2:] == ["one.safetensors", "two.safetensors"]


def test_models_route_lists_model_folder_and_fit(tmp_path: Path) -> None:
    models = tmp_path / "models" / "gpt-oss-20b"
    models.mkdir(parents=True)
    (models / "config.json").write_text(json.dumps(_gpt_oss_config()), encoding="utf-8")
    (models / "model.safetensors").write_bytes(b"x" * 10)
    api = _FakeApi({})
    client = _app(tmp_path, api=api, config_fetcher=lambda _: _gpt_oss_config())

    response = client.get("/api/models")
    assert response.status_code == 200
    listed = response.json()["models"]
    assert len(listed) == 1
    assert listed[0]["name"] == "gpt-oss-20b"
    assert listed[0]["found"] is True
    assert listed[0]["architecture"] == "GptOssForCausalLM"
    assert "fit" in listed[0] and "verdict" in listed[0]["fit"]
