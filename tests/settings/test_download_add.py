"""The Add model wizard's downloads (DownloadManager.plan_add/start_add). Review focus 4 lives here."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from freetoken.daemon.settings.download import (
    AddUnsupported, DownloadConflict, DownloadManager, InvalidRepository, parse_sha256sums,
)

GIB = 1024 ** 3


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Hub:
    """A fake Hub. model_info lists the files (LFS sha256 on all but config.json and
    SHA256SUMS when published); snapshot writes one file like snapshot_download(local_dir=…),
    including its .cache folder."""

    def __init__(self, files: dict[str, bytes], published: bool = True, config=None, fail_on=None, gate=None):
        self.files, self.published, self.config, self.fail_on, self.gate = files, published, config, fail_on, gate
        self.fetched: list[str] = []

    def model_info(self, repo, files_metadata=False):
        rows = []
        for name, data in self.files.items():
            lfs = (SimpleNamespace(size=len(data), sha256=sha(data))
                   if self.published and name not in ("config.json", "SHA256SUMS") else None)
            rows.append(SimpleNamespace(rfilename=name, size=len(data), lfs=lfs))
        return SimpleNamespace(siblings=rows)

    def snapshot(self, repo, *, local_dir, allow_patterns=None, **_kwargs):
        name = allow_patterns[0]
        self.fetched.append(name)
        if name == self.fail_on:
            raise OSError("connection reset")
        target = Path(local_dir)
        (target / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(self.files[name])
        if self.gate is not None and name == self.gate[0]:
            self.gate[1].set()
            self.gate[2].wait(2)


def manager(tmp_path, hub, free=10 ** 12):
    return DownloadManager(tmp_path / "models", api_factory=hub, config_fetcher=lambda _: hub.config or {},
                           snapshot_downloader=hub.snapshot, pc_memory=64 * GIB, card_memory=32 * GIB,
                           disk_free=lambda _: free)


def roots(tmp_path):
    folder, ninfer = tmp_path / "models", tmp_path / "ninfer"
    folder.mkdir(exist_ok=True)
    ninfer.mkdir(exist_ok=True)
    return {"folder_root": folder, "ninfer_root": ninfer}


def finish(m, job):
    for _ in range(400):
        current = m.get(job.job_id)
        if current.stage in ("done", "failed", "cancelled"):
            return current.as_dict()
        time.sleep(0.005)
    raise AssertionError(m.get(job.job_id).as_dict())


def leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.startswith((".incoming-", ".cache")))


def test_a_ninfer_repo_fetches_the_entry_and_its_parts_and_checks_them(tmp_path):
    hub = Hub({"small.ninfer": b"entry" * 10, "small.ninfer.part-0001": b"part" * 5, "README.md": b"hi"})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    plan = m.plan_add("owner/small-repo", **r)
    assert (plan["engine"], plan["kind"], plan["entry"]) == ("ninfer", "ninfer", "small.ninfer")
    assert [f["name"] for f in plan["files"]] == ["small.ninfer", "small.ninfer.part-0001"]
    assert {f["check"] for f in plan["files"]} == {"published"}
    assert plan["target"] == str(r["ninfer_root"] / "small.ninfer") and plan["exists"] is False
    done = finish(m, m.start_add("https://huggingface.co/owner/small-repo", **r))
    assert done["stage"] == "done", done
    assert done["resultPath"] == str(r["ninfer_root"] / "small.ninfer")
    assert sorted(done["verified"]) == ["small.ninfer", "small.ninfer.part-0001"]
    assert (r["ninfer_root"] / "small.ninfer.part-0001").read_bytes() == b"part" * 5
    assert hub.fetched == ["small.ninfer", "small.ninfer.part-0001"]
    assert leftovers(r["ninfer_root"]) == []


def test_parse_sha256sums_reads_both_common_forms():
    text = f"{'a' * 64}  one.ninfer\n{'B' * 64} *./two.ninfer\nnot a line\n"
    assert parse_sha256sums(text) == {"one.ninfer": "a" * 64, "two.ninfer": "b" * 64}


def test_sha256sums_wins_and_a_mismatch_deletes_everything(tmp_path):
    hub = Hub({"small.ninfer": b"weights" * 100, "SHA256SUMS": f"{'0' * 64}  small.ninfer\n".encode()})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    (r["ninfer_root"] / "other.ninfer").write_bytes(b"keep me")
    plan = m.plan_add("owner/repo", **r)
    assert plan["sumsFile"] == "SHA256SUMS" and plan["files"] == [{"name": "small.ninfer", "bytes": 700, "check": "SHA256SUMS"}]
    done = finish(m, m.start_add("owner/repo", **r))
    assert done["stage"] == "failed" and "checksum" in done["error"], done
    assert sorted(p.name for p in r["ninfer_root"].iterdir()) == ["other.ninfer"]
    assert (r["ninfer_root"] / "other.ninfer").read_bytes() == b"keep me"


def test_two_ninfer_files_need_a_pick_and_only_the_pick_is_fetched(tmp_path):
    hub = Hub({"a.ninfer": b"a", "b.ninfer": b"b", "b.ninfer.part-0001": b"bb"})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    plan = m.plan_add("owner/repo", **r)
    assert plan["entries"] == ["a.ninfer", "b.ninfer"] and plan["entry"] is None and plan["files"] == []
    with pytest.raises(InvalidRepository, match="Pick"):
        m.start_add("owner/repo", **r)
    with pytest.raises(InvalidRepository):
        m.plan_add("owner/repo", "c.ninfer", **r)
    done = finish(m, m.start_add("owner/repo", "b.ninfer", **r))
    assert done["stage"] == "done" and hub.fetched == ["b.ninfer", "b.ninfer.part-0001"]
    assert not (r["ninfer_root"] / "a.ninfer").exists()


def test_nothing_is_ever_overwritten(tmp_path):
    hub = Hub({"small.ninfer": b"new"})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    (r["ninfer_root"] / "small.ninfer").write_bytes(b"old")
    assert m.plan_add("owner/repo", **r)["exists"] is True
    with pytest.raises(DownloadConflict, match="already on this PC"):
        m.start_add("owner/repo", **r)
    assert (r["ninfer_root"] / "small.ninfer").read_bytes() == b"old" and hub.fetched == []


def test_a_file_that_appears_during_the_download_is_not_overwritten(tmp_path):
    started, release = threading.Event(), threading.Event()
    hub = Hub({"small.ninfer": b"new"}, gate=("small.ninfer", started, release))
    m, r = manager(tmp_path, hub), roots(tmp_path)
    job = m.start_add("owner/repo", **r)
    assert started.wait(2)
    (r["ninfer_root"] / "small.ninfer").write_bytes(b"old")
    release.set()
    done = finish(m, job)
    assert done["stage"] == "failed" and "appeared" in done["error"], done
    assert (r["ninfer_root"] / "small.ninfer").read_bytes() == b"old" and leftovers(r["ninfer_root"]) == []


def test_a_drive_that_cannot_hard_link_still_never_overwrites(tmp_path, monkeypatch):
    """WSL drvfs mounts refuse os.link (EPERM); the fallback claims the name exclusively first."""

    def refuse(*_args, **_kwargs):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr("freetoken.daemon.settings.download.os.link", refuse)
    hub = Hub({"small.ninfer": b"new" * 4, "small.ninfer.part-0001": b"p"})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    done = finish(m, m.start_add("owner/repo", **r))
    assert done["stage"] == "done", done
    assert (r["ninfer_root"] / "small.ninfer").read_bytes() == b"new" * 4 and leftovers(r["ninfer_root"]) == []

    started, release = threading.Event(), threading.Event()
    hub2 = Hub({"other.ninfer": b"new"}, gate=("other.ninfer", started, release))
    m2 = manager(tmp_path, hub2)
    job = m2.start_add("owner/repo2", **r)
    assert started.wait(2)
    (r["ninfer_root"] / "other.ninfer").write_bytes(b"old")
    release.set()
    done = finish(m2, job)
    assert done["stage"] == "failed" and "appeared" in done["error"], done
    assert (r["ninfer_root"] / "other.ninfer").read_bytes() == b"old" and leftovers(r["ninfer_root"]) == []


def test_a_model_folder_repo_lands_in_models_name(tmp_path):
    config = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2, "hidden_size": 64}
    hub = Hub({"config.json": b"{}", "model.safetensors": b"w" * 32, "tokenizer.json": b"{}", "notes.md": b"x"},
              config=config)
    m, r = manager(tmp_path, hub), roots(tmp_path)
    plan = m.plan_add("owner/Tiny-Llama", **r)
    assert (plan["engine"], plan["kind"], plan["architecture"]) == ("freetoken", "folder", "LlamaForCausalLM")
    assert plan["target"] == str(r["folder_root"] / "Tiny-Llama")
    assert [f["name"] for f in plan["files"]] == ["config.json", "model.safetensors", "tokenizer.json"]
    done = finish(m, m.start_add("owner/Tiny-Llama", **r))
    assert done["stage"] == "done", done
    folder = r["folder_root"] / "Tiny-Llama"
    assert done["resultPath"] == str(folder) and (folder / "model.safetensors").is_file()
    assert not (folder / ".cache").exists() and leftovers(r["folder_root"]) == []
    assert sorted(done["verified"]) == ["model.safetensors", "tokenizer.json"]


def test_designs_and_repos_the_engines_cannot_run_are_refused_before_downloading(tmp_path):
    bert = Hub({"config.json": b"{}", "model.safetensors": b"w"}, config={"architectures": ["BertModel"]})
    with pytest.raises(AddUnsupported, match=r"BertModel\) is not supported by your engines"):
        manager(tmp_path, bert).plan_add("owner/bert", **roots(tmp_path))
    gguf = Hub({"README.md": b"hi", "model.gguf": b"x"})
    with pytest.raises(AddUnsupported, match="not supported by your engines"):
        manager(tmp_path, gguf).start_add("owner/gguf", **roots(tmp_path))
    assert bert.fetched == [] and gguf.fetched == []


def test_failure_and_cancel_delete_the_partial_files(tmp_path):
    r = roots(tmp_path)
    broken = Hub({"small.ninfer": b"a" * 10, "small.ninfer.part-0001": b"b" * 10}, fail_on="small.ninfer.part-0001")
    m = manager(tmp_path, broken)
    failed = finish(m, m.start_add("owner/repo", **r))
    assert failed["stage"] == "failed" and "connection reset" in failed["error"]
    assert list(r["ninfer_root"].iterdir()) == []

    started, release = threading.Event(), threading.Event()
    slow = Hub({"small.ninfer": b"a" * 10, "small.ninfer.part-0001": b"b" * 10}, gate=("small.ninfer", started, release))
    m2 = manager(tmp_path, slow)
    job = m2.start_add("owner/repo", **r)
    assert started.wait(2)
    assert m2.cancel(job.job_id).cancel_requested is True
    release.set()
    assert finish(m2, job)["stage"] == "cancelled"
    assert list(r["ninfer_root"].iterdir()) == [] and slow.fetched == ["small.ninfer"]


def test_not_enough_drive_space_refuses_to_start(tmp_path):
    m = manager(tmp_path, Hub({"small.ninfer": b"a" * 100}), free=10)
    with pytest.raises(DownloadConflict, match="drive space"):
        m.start_add("owner/repo", **roots(tmp_path))


def test_latest_add_reports_the_newest_wizard_download(tmp_path):
    hub = Hub({"small.ninfer": b"a"})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    assert m.latest_add() is None
    job = m.start_add("owner/repo", **r)
    finish(m, job)
    assert m.latest_add()["id"] == job.job_id and m.latest_add()["kind"] == "add"
