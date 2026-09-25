"""Fakes for the Test tab: a one-card switcher, a scripted chat, a probe and a clock."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.panel import PanelService
from freetoken.daemon.settings.playground import WARMUP_MESSAGES, PlaygroundRunner
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.profiles_manager import ProfilesManager
from freetoken.daemon.settings.registry import RegistryStore, find_model
from freetoken.daemon.settings.swap_config import SwapConfigWriter, render_config
from freetoken.daemon.settings.switcher import SwitcherError
from tests.settings.registry_fixtures import five
from tests.settings.test_panel_routes import FakeEstimates, FakeSwitcher, checker

GIB = 1024 ** 3


def presets_doc() -> dict:
    doc = five()
    find_model(doc, "quasar-27b")["presets"] = {
        "Fast": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 3},
        "Same": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 7},  # equals the saved settings
    }
    find_model(doc, "fable-27b")["presets"] = {"Three": {"kv-dtype": "fp8", "spec": "mtp", "draft-tokens": 3}}
    find_model(doc, "qwen3.8-flash")["presets"] = {"Short": {"KVCacheTokens": 131072}}
    return doc


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def wall(self) -> float:
        return 1_800_000_000.0 + self.now


class CardSwitcher(FakeSwitcher):
    """One graphics card: a load puts every other model away. on_load runs when a load starts
    (it may change states, call runner.stop() or raise); after_load runs once it finished."""

    def __init__(self, cfg: Path, clock: Clock) -> None:
        super().__init__(cfg)
        self.clock, self.load_s, self.stale = clock, {}, False
        self.on_load = self.after_load = None
        self.loading, self.cancelled = None, set()
        self.files_at_load: dict[str, list[str]] = {}

    def config_hash(self):
        return "stale" if self.stale else super().config_hash()

    def unload(self, model_id):
        self.calls.append(("unload", model_id))
        if not self.unload_ok:
            return False
        if self.loading == model_id:
            self.cancelled.add(model_id)  # P1: an unload cancels a half-finished swap
        self.states.pop(model_id, None)
        return True

    def load(self, model_id, *, timeout=900.0):
        self.calls.append(("load", model_id))
        self.loading = model_id
        try:
            self.files_at_load.setdefault(model_id, []).append(self.cfg.read_text())
            if self.on_load is not None:
                self.on_load(model_id)
            if model_id in self.cancelled:
                self.cancelled.discard(model_id)
                raise SwitcherError(502, "load_failed", "the load was cancelled")
            if self.load_error:
                raise self.load_error
            self.clock.advance(self.load_s.get(model_id, 20))
            self.states = {model_id: "ready"}
        finally:
            self.loading = None
        if self.after_load is not None:
            self.after_load(model_id)


def sse(obj) -> str:
    return "data: " + json.dumps(obj) + "\n"


def answer_script(words=("Hello", " there", " friend"), prompt=40, timings=None):
    """(seconds before the line, line): the first word after 0.2 s, then one word each 0.05 s."""
    script = [(0.2, sse({"model": None, "choices": [{"index": 0, "delta": {"content": words[0]}}]}))]
    script += [(0.05, sse({"choices": [{"index": 0, "delta": {"content": word}}]})) for word in words[1:]]
    last = {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": len(words)}}
    if timings:
        last["timings"] = timings
    script += [(0.0, sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})),
               (0.0, sse(last)), (0.0, "data: [DONE]\n")]
    return script


class FakeChat:
    def __init__(self, clock: Clock) -> None:
        self.clock, self.scripts, self.errors = clock, {}, {}
        self.bodies, self.sessions = [], []
        self.on_stream, self.aborted = None, False

    def stream(self, body, session):
        self.bodies.append(body)
        self.sessions.append(session)
        warmup = body["messages"] == WARMUP_MESSAGES
        if self.on_stream is not None:
            self.on_stream(body)
        if not warmup and body["model"] in self.errors:
            raise self.errors[body["model"]]
        for delay, line in answer_script(("OK",)) if warmup else self.scripts.get(body["model"], answer_script()):
            if self.aborted:
                return
            self.clock.advance(delay)
            yield line

    def abort(self):
        self.aborted = True

    def reset(self):
        """Like SwitcherChat.reset: an abort holds until the next test starts."""
        self.aborted = False

    def answers(self):
        return [body for body in self.bodies if body["messages"] != WARMUP_MESSAGES]


class FakeProbe:
    """busy: models llama-swap's P7 if-idle unload refuses (a request it holds that the
    in-flight read did not show). on_inflight runs on every in-flight read."""

    def __init__(self, switcher=None) -> None:
        self.rows, self.used, self.unknown = [], {}, False
        self.switcher, self.busy, self.on_inflight = switcher, set(), None

    def inflight(self):
        if self.on_inflight is not None:
            self.on_inflight()
        return None if self.unknown else list(self.rows)

    def unload_if_idle(self, model_id):
        if model_id in self.busy:
            self.switcher.calls.append(("busy", model_id))
            return "busy"
        return "unloaded" if self.switcher.unload(model_id) else "failed"

    def last_used(self, model_id):
        return self.used.get(model_id)


def make(tmp_path, monkeypatch, loaded=None, doc=None, with_routes=False, static_path=None):
    """Registry with presets, a generated switcher file, one card, and inline threads."""
    monkeypatch.setenv("HOME", "/home/jay")
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -KVDtype 'fp8' `\n    -Port 2020\n", encoding="utf-8")
    cfg = tmp_path / "llama-swap" / "config.yaml"
    cfg.parent.mkdir()
    binary = tmp_path / "llama-swap-bin"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    clock = Clock()
    writer = SwapConfigWriter(cfg, binary=binary, runner=checker())
    switcher = CardSwitcher(cfg, clock)
    profiles = ProfilesManager(tmp_path / "boot-profiles.json", boot_file=boot)
    store = RegistryStore(tmp_path / "freetoken" / "registry.json")
    store.save(doc or presets_doc(), expected_revision=None)
    writer.write(render_config(store.load()[0], {}))
    switcher.states = dict(loaded or {})
    chat, probe = FakeChat(clock), FakeProbe(switcher)

    def build():
        """A fresh PanelService and runner over the same files: what a helper restart gives."""
        service = PanelService(
            store=store, writer=writer, switcher=switcher, profiles=profiles,
            boot_file=lambda: BootFile(boot), default_boot=lambda: boot, estimate_service=FakeEstimates(),
            card_probe=lambda: {"totalBytes": 32 * GIB, "usedBytes": 2 * GIB}, windows_free_probe=lambda: 40 * GIB,
            artifact_size=lambda path: 19_782_132_224, spawn=lambda fn, *args: fn(*args),
            clock=clock, sleep=clock.advance)
        runner = PlaygroundRunner(service, chat=chat, probe=probe, spawn=lambda fn, *args: fn(*args),
                                  clock=clock, wall=clock.wall)
        return service, runner

    service, runner = build()
    proc = ProcessManager(boot_file=boot, stop_script=tmp_path / "stop.ps1", log_path=tmp_path / "server.log",
                          lock_path=tmp_path / "gpu.lock", runner=lambda *a, **k: None,
                          readiness=lambda: {"state": "unreachable"}, sleep=lambda _: None, poll_interval=0)
    extra = {"playground": runner} if with_routes else {}
    app = create_app(boot_file=boot, process_manager=proc, profiles=profiles, log_path=tmp_path / "server.log",
                     static_path=static_path or tmp_path / "missing.html", panel=service, **extra)
    return SimpleNamespace(service=service, runner=runner, store=store, writer=writer, switcher=switcher,
                           chat=chat, probe=probe, clock=clock, cfg=cfg, client=TestClient(app), build=build)
