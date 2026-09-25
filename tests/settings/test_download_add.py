"""The Add model wizard's downloads (DownloadManager.plan_add/start_add). Review focus 4 lives here."""

from __future__ import annotations

import hashlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest

from freetoken.daemon.settings import download as dl
from freetoken.daemon.settings.download import (
    AddUnsupported, DownloadCancelled, DownloadConflict, DownloadManager, InvalidRepository, parse_sha256sums,
    sha256_file,
)

GIB = 1024 ** 3


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Hub:
    """A fake Hub. model_info lists the files (LFS sha256 on all but config.json and
    SHA256SUMS when published); fetch writes one file the way the streamed copy does, in
    ``chunk``-byte pieces, polling ``cancelled`` before each piece. A gate (name, started,
    release) pauses after the first piece of that file, mid-file when it has several."""

    def __init__(self, files: dict[str, bytes], published: bool = True, config=None, fail_on=None, gate=None,
                 chunk: int | None = None):
        self.files, self.published, self.config, self.fail_on, self.gate = files, published, config, fail_on, gate
        self.chunk = chunk
        self.fetched: list[str] = []
        self.written: dict[str, int] = {}

    def model_info(self, repo, files_metadata=False):
        rows = []
        for name, data in self.files.items():
            lfs = (SimpleNamespace(size=len(data), sha256=sha(data))
                   if self.published and name not in ("config.json", "SHA256SUMS") else None)
            rows.append(SimpleNamespace(rfilename=name, size=len(data), lfs=lfs))
        return SimpleNamespace(siblings=rows)

    def fetch(self, repo, name, destination, cancelled):
        self.fetched.append(name)
        if name == self.fail_on:
            raise OSError("connection reset")
        data = self.files[name]
        step = self.chunk or max(1, len(data))
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as fh:
            for offset in range(0, max(1, len(data)), step):
                if cancelled():
                    raise DownloadCancelled(name)
                fh.write(data[offset:offset + step])
                fh.flush()
                self.written[name] = self.written.get(name, 0) + len(data[offset:offset + step])
                if self.gate is not None and name == self.gate[0] and offset == 0:
                    self.gate[1].set()
                    self.gate[2].wait(2)


def manager(tmp_path, hub, free=10 ** 12):
    return DownloadManager(tmp_path / "models", api_factory=hub, config_fetcher=lambda _: hub.config or {},
                           file_fetcher=hub.fetch, pc_memory=64 * GIB, card_memory=32 * GIB,
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


def test_cancel_interrupts_a_file_in_progress(tmp_path):
    """One 19 GB .ninfer must stop within a chunk of Cancel, not once it has finished."""
    started, release = threading.Event(), threading.Event()
    hub = Hub({"big.ninfer": b"abcdefghij" * 4}, gate=("big.ninfer", started, release), chunk=10)
    m, r = manager(tmp_path, hub), roots(tmp_path)
    job = m.start_add("owner/repo", **r)
    assert started.wait(2)
    staged = r["ninfer_root"] / f".incoming-{job.job_id}" / "big.ninfer"
    assert staged.stat().st_size == 10  # one chunk in, three to go
    m.cancel(job.job_id)
    release.set()
    done = finish(m, job)
    assert done["stage"] == "cancelled", done
    assert list(r["ninfer_root"].iterdir()) == [] and hub.fetched == ["big.ninfer"]
    assert hub.written == {"big.ninfer": 10}  # the copy itself stopped; the other 30 bytes never came


def test_cancel_after_the_last_file_does_not_end_done(tmp_path):
    """No checksum step (nothing published, no SHA256SUMS): the last look before moving must
    still see the cancel."""
    started, release = threading.Event(), threading.Event()
    hub = Hub({"small.ninfer": b"weights"}, published=False, gate=("small.ninfer", started, release))
    m, r = manager(tmp_path, hub), roots(tmp_path)
    job = m.start_add("owner/repo", **r)
    assert started.wait(2)
    m.cancel(job.job_id)
    release.set()
    done = finish(m, job)
    assert done["stage"] == "cancelled" and done["resultPath"] is None, done
    assert list(r["ninfer_root"].iterdir()) == []


def test_the_streamed_copy_polls_cancel_per_chunk_and_sends_the_token_only_to_the_hub(tmp_path, monkeypatch):
    flag = threading.Event()
    seen: list[Request] = []
    arm = [True]

    class Response:
        def __init__(self):
            self.chunks = [b"one", b"two", b"three"]

        def read(self, size):
            if arm[0]:
                flag.set()  # the cancel lands while the first chunk is in flight
            return self.chunks.pop(0) if self.chunks else b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(request, timeout):
        seen.append(request)
        return Response()

    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    destination = tmp_path / "file.bin"
    with pytest.raises(DownloadCancelled):
        dl._stream_hub_file("owner/repo", "file.bin", destination, flag.is_set, opener=opener)
    assert destination.read_bytes() == b"one"  # stopped after the chunk that was already in flight
    assert seen[0].get_header("Authorization") == "Bearer hf_secret"
    assert "owner/repo" in seen[0].full_url and seen[0].full_url.endswith("file.bin")
    flag.clear()
    arm[0] = False
    dl._stream_hub_file("owner/repo", "file.bin", destination, flag.is_set, opener=opener)
    assert destination.read_bytes() == b"onetwothree"

    handler = dl._DropTokenAcrossHosts()
    hub = Request("https://huggingface.co/o/r/resolve/main/f", headers={"Authorization": "Bearer hf_secret"})
    to_cdn = handler.redirect_request(hub, None, 302, "Found", {}, "https://cdn-lfs.hf.co/blob")
    assert to_cdn.get_header("Authorization") is None
    same_host = handler.redirect_request(hub, None, 302, "Found", {}, "https://huggingface.co/o/r/other")
    assert same_host.get_header("Authorization") == "Bearer hf_secret"


def test_hashing_can_be_cancelled_too(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (dl._HASH_CHUNK + 1))
    reads = iter([False, True])
    with pytest.raises(DownloadCancelled):
        sha256_file(big, lambda: next(reads))
    assert sha256_file(big) == sha(b"x" * (dl._HASH_CHUNK + 1))


def test_orphan_staging_folders_are_swept_but_a_live_jobs_is_kept(tmp_path):
    """A helper restart mid-download leaves .incoming-download-* behind; nothing else cleans it."""
    r = roots(tmp_path)
    for root in r.values():
        orphan = root / ".incoming-download-deadbeef0000"
        orphan.mkdir()
        (orphan / "half.ninfer").write_bytes(b"partial")
        (root / ".incoming-somethingelse").mkdir()  # not ours: left alone
    m = manager(tmp_path, Hub({"small.ninfer": b"new"}))
    assert not (r["folder_root"] / ".incoming-download-deadbeef0000").exists()  # models_dir swept at start
    assert (r["ninfer_root"] / ".incoming-download-deadbeef0000").is_dir()  # not known until a wizard root is

    started, release = threading.Event(), threading.Event()
    hub = Hub({"small.ninfer": b"new"}, gate=("small.ninfer", started, release))
    m = manager(tmp_path, hub)
    job = m.start_add("owner/repo", **r)
    assert started.wait(2)
    assert not (r["ninfer_root"] / ".incoming-download-deadbeef0000").exists()
    assert m.sweep_staging(*r.values()) == []  # the live job's own staging is not an orphan
    assert (r["ninfer_root"] / f".incoming-{job.job_id}" / "small.ninfer").is_file()
    release.set()
    assert finish(m, job)["stage"] == "done"
    assert sorted(p.name for p in r["ninfer_root"].iterdir()) == [".incoming-somethingelse", "small.ninfer"]


def test_a_folder_that_appears_during_the_download_is_not_replaced(tmp_path):
    config = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2, "hidden_size": 64}
    started, release = threading.Event(), threading.Event()
    hub = Hub({"config.json": b"{}", "model.safetensors": b"w" * 8}, config=config,
              gate=("model.safetensors", started, release))
    m, r = manager(tmp_path, hub), roots(tmp_path)
    job = m.start_add("owner/Tiny-Llama", **r)
    assert started.wait(2)
    (r["folder_root"] / "Tiny-Llama").mkdir()  # an empty folder: os.rename would replace it silently
    release.set()
    done = finish(m, job)
    assert done["stage"] == "failed" and "appeared" in done["error"], done
    assert list((r["folder_root"] / "Tiny-Llama").iterdir()) == [] and leftovers(r["folder_root"]) == []


def test_rename_noreplace_refuses_an_empty_folder_target(tmp_path, monkeypatch):
    source, taken = tmp_path / "staging", tmp_path / "taken"
    source.mkdir()
    (source / "a").write_bytes(b"a")
    taken.mkdir()
    with pytest.raises(FileExistsError):
        dl._rename_noreplace(source, taken)
    assert (source / "a").is_file() and list(taken.iterdir()) == []
    dl._rename_noreplace(source, tmp_path / "free")
    assert (tmp_path / "free" / "a").is_file() and not source.exists()
    # The check-then-rename fallback (non-Linux, or a filesystem without RENAME_NOREPLACE).
    monkeypatch.setattr(sys, "platform", "darwin")
    (tmp_path / "free2").mkdir()
    with pytest.raises(FileExistsError):
        dl._rename_noreplace(tmp_path / "free", tmp_path / "free2")
    assert (tmp_path / "free" / "a").is_file()


def test_each_file_is_labelled_with_what_actually_checked_it(tmp_path):
    """SHA256SUMS lists only the entry: the part keeps the Hub's digest, config-like files none."""
    files = {"small.ninfer": b"entry" * 3, "small.ninfer.part-0001": b"part" * 2,
             "SHA256SUMS": f"{sha(b'entry' * 3)}  small.ninfer\n".encode()}
    hub = Hub(files)
    m, r = manager(tmp_path, hub), roots(tmp_path)
    plan = m.plan_add("owner/repo", **r)
    assert [(f["name"], f["check"]) for f in plan["files"]] == [("small.ninfer", "SHA256SUMS"),
                                                                ("small.ninfer.part-0001", "published")]
    done = finish(m, m.start_add("owner/repo", **r))
    assert done["stage"] == "done", done
    assert done["checks"] == {"small.ninfer": "SHA256SUMS", "small.ninfer.part-0001": "published"}
    assert sorted(done["verified"]) == ["small.ninfer", "small.ninfer.part-0001"]

    unpublished = Hub({"small.ninfer": b"x", "SHA256SUMS": f"{'0' * 64}  other.ninfer\n".encode()}, published=False)
    m2 = manager(tmp_path, unpublished)
    assert m2.plan_add("owner/repo2", **r)["files"] == [{"name": "small.ninfer", "bytes": 1, "check": None}]
    (r["ninfer_root"] / "small.ninfer").unlink()
    done2 = finish(m2, m2.start_add("owner/repo2", **r))
    assert done2["stage"] == "done" and done2["checks"] == {"small.ninfer": None} and done2["verified"] == []


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


def test_folder_placement_without_renameat2_never_uses_a_check_then_rename(tmp_path, monkeypatch):
    """review, PR #17: an empty folder made between the check and the rename was silently replaced."""
    import pathlib

    monkeypatch.setattr(sys, "platform", "darwin")  # the path taken when RENAME_NOREPLACE is missing
    source, target = tmp_path / "staging", tmp_path / "Tiny-Llama"
    source.mkdir()
    (source / "config.json").write_bytes(b"{}")
    target.mkdir()  # the user's empty folder, made right after any existence check
    real_exists = pathlib.Path.exists

    def racing_exists(self, *args, **kwargs):
        # A check-then-rename looks before the folder is made; the lie stands in for that window.
        return False if self == target else real_exists(self, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "exists", racing_exists)
    with pytest.raises(FileExistsError):
        dl._rename_noreplace(source, target)
    assert list(target.iterdir()) == [] and (source / "config.json").is_file()


def test_folder_placement_without_renameat2_moves_files_and_backs_out_on_a_clash(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    source, target = tmp_path / "staging", tmp_path / "Model"
    (source / "sub").mkdir(parents=True)
    (source / "a.safetensors").write_bytes(b"a")
    (source / "sub" / "b.json").write_bytes(b"b")
    dl._rename_noreplace(source, target)
    assert (target / "a.safetensors").read_bytes() == b"a" and (target / "sub" / "b.json").read_bytes() == b"b"
    assert not source.exists()

    source.mkdir()
    for name in ("a.json", "b.json"):
        (source / name).write_bytes(name.encode())
    real_claim = dl._claim_file

    def clash(src, dst):
        if dst.name == "b.json":
            dst.write_bytes(b"someone else's")  # appears inside the new folder meanwhile
        real_claim(src, dst)
    monkeypatch.setattr(dl, "_claim_file", clash)
    with pytest.raises(FileExistsError):
        dl._rename_noreplace(source, tmp_path / "Model2")
    assert (source / "a.json").read_bytes() == b"a.json" and (source / "b.json").read_bytes() == b"b.json"
    assert [p.name for p in (tmp_path / "Model2").iterdir()] == ["b.json"]
    assert (tmp_path / "Model2" / "b.json").read_bytes() == b"someone else's"
