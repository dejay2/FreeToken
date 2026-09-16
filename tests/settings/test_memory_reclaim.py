"""Cache probes must never depend on or cause expert rebuilds."""

import errno
from pathlib import Path

import pytest
from freetoken.daemon.settings import memory_reclaim as reclaim

MIB = 1024**2
GIB = 1024**3


def test_success_ramps_only_when_windows_receives_memory():
    policy = reclaim.ReclaimPolicy()
    assert (
        policy.request(
            0, windows=4 * GIB, linux=30 * GIB, cache=20 * GIB, target=6 * GIB
        )
        == 256 * MIB
    )
    policy.finish(1, windows=4 * GIB, cache=20 * GIB, requested=256 * MIB)
    assert (
        policy.request(
            3, windows=4 * GIB, linux=30 * GIB, cache=19 * GIB, target=6 * GIB
        )
        == 0
    )
    # Cache eviction alone is not proof Windows got memory back.
    assert (
        policy.request(
            6, windows=4 * GIB, linux=30 * GIB, cache=19 * GIB, target=6 * GIB
        )
        == 0
    )
    assert policy.status(6)["state"] == "backoff"
    assert (
        policy.request(
            16, windows=4 * GIB, linux=30 * GIB, cache=19 * GIB, target=6 * GIB
        )
        == 256 * MIB
    )
    policy.finish(17, windows=4 * GIB, cache=19 * GIB, requested=256 * MIB)
    assert (
        policy.request(
            22,
            windows=4 * GIB + 128 * MIB,
            linux=30 * GIB,
            cache=18 * GIB,
            target=6 * GIB,
        )
        == 512 * MIB
    )


def test_reclaim_stops_at_target_or_when_cache_or_guest_ram_is_low():
    for windows, linux, cache in [
        (8 * GIB, 30 * GIB, 20 * GIB),
        (4 * GIB, 2 * GIB, 20 * GIB),
        (4 * GIB, 30 * GIB, 200 * MIB),
    ]:
        assert (
            reclaim.ReclaimPolicy().request(
                0, windows=windows, linux=linux, cache=cache, target=6 * GIB
            )
            == 0
        )


def test_ineffective_probes_back_off_and_reset_batch():
    policy = reclaim.ReclaimPolicy()
    t = 0
    for delay in (10, 30, 60, 60):
        assert (
            policy.request(
                t, windows=4 * GIB, linux=30 * GIB, cache=20 * GIB, target=6 * GIB
            )
            == 256 * MIB
        )
        policy.finish(t, windows=4 * GIB, cache=20 * GIB, requested=256 * MIB)
        policy.request(
            t + 5, windows=4 * GIB, linux=30 * GIB, cache=20 * GIB, target=6 * GIB
        )
        assert policy.status(t + 5)["retry_in_s"] == delay
        t += 5 + delay


def test_cgroup_never_retries_without_swappiness_on_old_kernel(tmp_path, monkeypatch):
    (tmp_path / "memory.reclaim").touch()
    calls = []

    def write(path, payload):
        calls.append(payload)
        raise OSError(errno.EINVAL, "old kernel")

    monkeypatch.setattr(reclaim, "_write_reclaim", write)
    with pytest.raises(reclaim.ReclaimUnavailable):
        reclaim.reclaim_cgroup(tmp_path, 256 * MIB)
    assert calls == ["268435456 swappiness=0"]


def test_cgroup_eagain_is_partial_not_unsupported(tmp_path, monkeypatch):
    def write(*args):
        raise OSError(errno.EAGAIN, "partial reclaim")

    monkeypatch.setattr(reclaim, "_write_reclaim", write)
    assert reclaim.reclaim_cgroup(tmp_path, 256 * MIB) == "partial"


def test_discovery_limits_scope_to_current_delegated_user(tmp_path):
    root = tmp_path / "cgroup"
    group = root / "user.slice/user-1000.slice/user@1000.service"
    group.mkdir(parents=True)
    (group / "memory.reclaim").touch()
    proc = tmp_path / "self.cgroup"
    proc.write_text(
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/helper.service\n"
    )
    assert reclaim.discover_cgroup(proc, root, uid=1000) == group
    assert reclaim.discover_cgroup(proc, root, uid=2000) is None


def test_targeted_cleanup_excludes_active_models_symlinks_and_partial_files(tmp_path):
    active = tmp_path / "active"
    inactive = tmp_path / "inactive"
    active.mkdir()
    inactive.mkdir()
    for d in (active, inactive):
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"hello")
    (inactive / "incomplete.safetensors.incomplete").write_bytes(b"partial")
    (inactive / "alias.safetensors").symlink_to(active / "model.safetensors")
    (inactive / "hardlink.safetensors").hardlink_to(active / "model.safetensors")
    assert reclaim.inactive_weight_files([active]) == [inactive / "model.safetensors"]
    # File cleanup preserves bytes, including when cache advice fails.
    reclaim.release_completed_file(inactive / "model.safetensors")
    assert (inactive / "model.safetensors").read_bytes() == b"hello"


def test_download_cleanup_error_does_not_fail_download(tmp_path, monkeypatch):
    p = tmp_path / "model.safetensors"
    p.write_bytes(b"weights")

    def denied(*args):
        raise OSError(errno.EPERM, "unsupported filesystem")

    monkeypatch.setattr(reclaim.os, "posix_fadvise", denied)
    assert reclaim.release_completed_file(p) is False
    assert p.read_bytes() == b"weights"


def test_governor_reclaim_tick_never_posts_cache_step(monkeypatch):
    from types import SimpleNamespace

    from freetoken.daemon.settings import governor

    loop = governor.GovernorLoop(
        SimpleNamespace(), governor.GovernorPolicy(GIB, 4 * GIB)
    )
    monkeypatch.setattr(governor, "_is_wsl", lambda: True)
    loop.last_free_windows_ram = 4 * GIB
    loop.last_free_linux_ram = 30 * GIB
    calls = []
    loop.reclaimer = SimpleNamespace(tick=lambda **kw: calls.append(kw))
    loop._tick_reclaim()
    assert len(calls) == 1
    assert calls[0]["windows"] == 4 * GIB


def test_worker_timeout_keeps_one_probe_in_flight_and_leaves_tick_nonblocking(
    monkeypatch,
):
    clock = [0]
    monkeypatch.setattr(reclaim.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(reclaim, "inactive_cache_bytes", lambda: 20 * GIB)

    class Process:
        killed = False
        done = False

        def poll(self):
            return -9 if self.done else None

        def kill(self):
            self.killed = True

        def communicate(self):
            assert self.done
            return "", ""

        def wait(self, timeout=None):
            return 0

    processes = []

    def start(*args, **kw):
        p = Process()
        processes.append(p)
        return p

    monkeypatch.setattr(reclaim.subprocess, "Popen", start)
    controller = reclaim.ReclaimController(2020)
    tick = lambda: controller.tick(windows=4 * GIB, linux=30 * GIB, target=6 * GIB)
    tick()
    tick()
    assert len(processes) == 1
    clock[0] = 11
    tick()
    assert processes[0].killed
    clock[0] = 30
    tick()
    assert (
        len(processes) == 1
    )  # A killed but uninterruptible worker still owns the slot.
    processes[0].done = True
    tick()
    clock[0] = 35
    tick()
    assert len(processes) == 1
    assert controller.status()["state"] == "backoff"


def test_spawn_failure_backs_off_instead_of_retrying_each_tick(monkeypatch):
    monkeypatch.setattr(reclaim.time, "monotonic", lambda: 0)
    monkeypatch.setattr(reclaim, "inactive_cache_bytes", lambda: 20 * GIB)
    calls = []

    def fail(*args, **kw):
        calls.append(1)
        raise OSError("process limit")

    monkeypatch.setattr(reclaim.subprocess, "Popen", fail)
    controller = reclaim.ReclaimController(2020)
    for _ in range(3):
        controller.tick(windows=4 * GIB, linux=30 * GIB, target=6 * GIB)
    assert len(calls) == 1
    assert "process limit" in controller.status()["last_result"]["error"]


def test_watchdog_kills_worker_without_further_governor_ticks(monkeypatch):
    import subprocess
    import sys

    real_popen = subprocess.Popen
    monkeypatch.setattr(reclaim, "inactive_cache_bytes", lambda: 20 * GIB)
    monkeypatch.setattr(reclaim, "PROBE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(
        reclaim.subprocess,
        "Popen",
        lambda *a, **kw: real_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kw
        ),
    )
    controller = reclaim.ReclaimController(2020)
    controller.tick(windows=4 * GIB, linux=30 * GIB, target=6 * GIB)
    assert controller.process.wait(timeout=3) != 0
    controller.close()


def test_close_prevents_launch_after_stop(monkeypatch):
    monkeypatch.setattr(reclaim, "inactive_cache_bytes", lambda: 20 * GIB)

    def unexpected(*a, **kw):
        pytest.fail("launched after shutdown")

    monkeypatch.setattr(reclaim.subprocess, "Popen", unexpected)
    controller = reclaim.ReclaimController(2020)
    controller.close()
    controller.tick(windows=4 * GIB, linux=30 * GIB, target=6 * GIB)
    assert not controller.status()["running"]


def test_cli_watcher_launches_importable_worker_module(monkeypatch):
    import importlib.util

    monkeypatch.setattr(reclaim, "__name__", "__main__")
    monkeypatch.setattr(reclaim, "inactive_cache_bytes", lambda: 20 * GIB)
    modules = []

    def start(args, **kwargs):
        modules.append(args[2])
        raise OSError("stop before starting real worker")

    monkeypatch.setattr(reclaim.subprocess, "Popen", start)
    reclaim.ReclaimController(2020).tick(
        windows=4 * GIB, linux=30 * GIB, target=6 * GIB
    )
    assert modules != ["__main__"]
    assert importlib.util.find_spec(modules[0]) is not None


def test_completed_download_shards_are_advised_before_next_shard(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from freetoken.daemon.settings import download

    events = []

    def fetch(repo, **kw):
        name = kw["allow_patterns"][0]
        (Path(kw["local_dir"]) / name).write_bytes(b"weight data")
        events.append(("download", name))

    monkeypatch.setattr(
        download, "release_completed_file", lambda p: events.append(("release", p.name))
    )
    manager = download.DownloadManager(tmp_path, snapshot_downloader=fetch)
    manager._hub_files = lambda repo: [
        SimpleNamespace(name="a.safetensors", size=11),
        SimpleNamespace(name="b.safetensors", size=11),
    ]
    # Exercise the job loop with normalized manifest records and a real target.
    monkeypatch.setattr(download, "_downloadable_files", lambda files: files)
    job = download.DownloadJob(
        job_id="probe", repo="owner/model", target_folder=tmp_path / "model"
    )
    manager._jobs["probe"] = job
    manager._run("probe")
    assert job.stage == "done"
    assert events == [
        ("download", "a.safetensors"),
        ("release", "a.safetensors"),
        ("download", "b.safetensors"),
        ("release", "b.safetensors"),
    ]


def test_temporary_watcher_hands_off_to_integrated_governor(monkeypatch):
    import io
    import json
    from contextlib import nullcontext

    records = [
        {
            "governor": {
                "enabled": True,
                "free_windows_ram_gb": 4,
                "free_linux_ram_gb": 30,
            },
            "server": {"reachable": True, "state": "serving"},
        },
        {
            "settings": {
                "GovernorRAMFreeGB": 4,
                "GovernorRAMRungsBeforeUp": 1,
                "GovernorUpMarginGB": 0.5,
            }
        },
        {"governor": {"reclaim": {"state": "idle"}}},
    ]
    monkeypatch.setattr(
        reclaim.urllib.request,
        "urlopen",
        lambda *a, **kw: nullcontext(io.BytesIO(json.dumps(records.pop(0)).encode())),
    )
    calls = []

    class Controller:
        def __init__(self, port):
            pass

        def tick(self, **kw):
            calls.append(kw)

        def status(self):
            return {"state": "reclaiming", "running": True}

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(reclaim, "ReclaimController", Controller)
    monkeypatch.setattr(reclaim.time, "sleep", lambda s: None)
    monkeypatch.setattr(reclaim, "is_wsl", lambda: True)
    reclaim.watch(2031, 2020)
    assert len(calls) == 2
    assert calls[0]["enabled"] is True
    assert calls[-1] == "closed"
