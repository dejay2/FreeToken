import json
import os
import sys
import time

import pytest


def _blocked_capture(send, kwargs):
    time.sleep(60)


def test_filesystem_hang_cannot_hold_the_watchdog(monkeypatch):
    from freetoken.daemon.settings import incidents

    monkeypatch.setattr(incidents, "_incident_worker", _blocked_capture, raising=False)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        incidents.capture_incident_bounded(timeout=0.2)
    assert time.monotonic() - started < 3


def test_bounded_capture_returns_a_completed_bundle_from_its_child(tmp_path):
    from freetoken.daemon.settings.incidents import capture_incident_bounded

    log = tmp_path / "server.log"
    log.write_text("bounded child capture\n")
    path = capture_incident_bounded(
        document={"state": "failed"}, reason="test", pids=set(),
        log_path=log, directory=tmp_path / "incidents",
    )
    assert json.loads((path / "manifest.json").read_text())["reason"] == "test"
    assert (path / "server-tail.log").read_text() == "bounded child capture\n"


def test_command_capture_is_bounded_even_when_a_child_floods_or_hangs():
    from freetoken.daemon.settings.incidents import command_output

    started = time.monotonic()
    flood = command_output([sys.executable, '-c', 'import os;\nwhile True: os.write(1,b"x"*8192)'],
                           timeout=.5, max_bytes=4096)
    assert len(flood['output'].encode()) <= 4096
    assert flood['truncated']
    hung = command_output([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=.2)
    assert hung['timed_out']
    assert time.monotonic() - started < 3


def test_incident_is_private_and_keeps_only_bounded_recent_bundles(tmp_path, monkeypatch):
    from freetoken.daemon.settings import incidents

    monkeypatch.setattr(incidents, 'command_output', lambda *a, **kw: {'output': 'diagnostic'})
    log = tmp_path / 'server.log'
    log.write_text('tail marker\n')
    root = tmp_path / 'incidents'
    for i in range(4):
        path = incidents.capture_incident(
            document={'state': 'failed', 'instance_id': str(i)}, reason='runtime failure',
            pids=set(), log_path=log, directory=root, retain=2,
        )
    dirs = list(root.iterdir())
    assert len(dirs) == 2
    assert json.loads((path / 'manifest.json').read_text())['document']['instance_id'] == '3'
    if os.name == 'posix':
        assert path.stat().st_mode & 0o777 == 0o700
        assert (path / 'manifest.json').stat().st_mode & 0o777 == 0o600
    assert (path / 'server-tail.log').read_text().endswith('tail marker\n')
