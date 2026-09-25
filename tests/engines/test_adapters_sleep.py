"""A sleeping FreeToken under llama-swap (review focus 5): adopted by its own model, fully
stopped for any other engine (Jay's rule: switching models while FreeToken sleeps unloads it)."""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from tests.engines.test_adapters import ADAPTERS, FakeHelper, env_for, fake_engine


@pytest.fixture
def helper():
    """The shared fake, plus the real server's /ready answer while asleep: the ``?model=``
    mismatch check runs before the readiness check (control_api.ready), so a sleeping server
    still says "another model is loaded" for a different folder. The shared fake only does
    that while "serving"."""
    h = FakeHelper()
    base = h.httpd.RequestHandlerClass

    class H(base):
        def do_GET(self):
            if self.path.startswith("/ready") and h.state == "sleeping":
                h.calls.append("GET " + self.path)
                want = self.path.split("model=", 1)[1] if "model=" in self.path else ""
                if os.path.basename(h.model_path) != want:
                    return self._send(503, {"status": "not_ready", "reason": "another model is loaded"})
                return self._send(200, {"status": "ready"})
            return base.do_GET(self)

    h.httpd.RequestHandlerClass = H
    yield h
    h.close()


def test_freetoken_adopts_its_own_sleeping_server_without_rebooting(helper, tmp_path):
    helper.state, helper.model_path = "sleeping", "/m/B"
    proc = subprocess.Popen([ADAPTERS / "freetoken.sh", "/m/B"], env=env_for(helper, tmp_path),
                            stderr=subprocess.PIPE, text=True)
    time.sleep(1.5)
    assert proc.poll() is None  # still guarding the model, not exited or crashed
    proc.terminate()
    proc.wait(20)
    err = proc.stderr.read()
    assert "already sleeping /m/B; adopting it" in err  # M8: the log names the real state
    assert "already serving" not in err
    assert "POST /api/server/start" not in helper.calls
    # Adopting it must not touch the sleeping server: a Stop is the one thing that would throw
    # away the fast wake. The stop is the SIGTERM's, so exactly one.
    assert helper.calls.count("POST /api/server/stop") == 1


def test_freetoken_releases_a_sleeping_server_the_page_switched_away_from(helper, tmp_path):
    helper.state, helper.model_path = "sleeping", "/m/B"
    proc = subprocess.Popen([ADAPTERS / "freetoken.sh", "/m/B"], env=env_for(helper, tmp_path),
                            stderr=subprocess.PIPE, text=True)
    time.sleep(1.5)
    assert proc.poll() is None
    helper.model_path = "/m/C"  # the settings page booted another model while B slept
    try:
        proc.wait(20)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("the adapter kept guarding a model the page had switched away from")
    err = proc.stderr.read()
    assert proc.returncode == 0 and "switched to another model" in err
    assert "POST /api/server/stop" not in helper.calls


def test_ninfer_fully_stops_a_sleeping_freetoken_first(helper, tmp_path):
    helper.state = "sleeping"
    eng = fake_engine(tmp_path)
    proc = subprocess.Popen([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "quasar-27b"], env=env_for(helper, tmp_path))
    for _ in range(50):
        if (tmp_path / "argv").exists():
            break
        time.sleep(0.1)
    proc.terminate()
    proc.wait(5)
    assert "POST /api/server/stop" in helper.calls and helper.state == "unreachable"
    assert (tmp_path / "argv").exists()
