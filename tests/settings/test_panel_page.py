"""The control panel page: plain-words helpers (node) and the page contract."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[2]
STATIC = REPO / "python" / "freetoken" / "daemon" / "settings" / "static"
PANEL_JS = STATIC / "panel.js"
PAGE = STATIC / "index.html"


def _node(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to run the page's JavaScript")
    prelude = (f"const p = require({json.dumps(str(PANEL_JS))}); "
               "const assert = require('node:assert/strict'); const G = 1024 ** 3;\n")
    run = subprocess.run([node, "-e", prelude + script], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr


def test_helpers_speak_plain_words():
    _node(r"""
assert.equal(p.fmtGB(29.7 * G), '29.7 GB');
assert.equal(p.fmtGB(null), '—');
assert.equal(p.stateWord('ready'), 'Loaded');
assert.equal(p.stateWord('starting'), 'Loading…');
assert.equal(p.sourceText({from: 'model'}), 'changed for this model');
assert.equal(p.sourceText({from: 'preset', preset: 'Fast agents'}), 'from preset “Fast agents”');
assert.equal(p.sourceText({from: 'default', engineLabel: 'NInfer'}), 'from NInfer defaults');
""")


def test_sources_follow_the_draft():
    _node(r"""
const view = {kind: 'model', engineLabel: 'NInfer', activePreset: 'Fast',
  base: {'max-concurrency': 6, 'kv-dtype': 'bf16'}, baseFrom: {'max-concurrency': 'preset', 'kv-dtype': 'default'}};
const draft = {'max-concurrency': 6, 'kv-dtype': 'int8', 'model.name': 'Q'};
assert.deepEqual(p.dialSourceFor(view, draft, 'kv-dtype'), {from: 'model'});
assert.deepEqual(p.dialSourceFor(view, draft, 'max-concurrency'), {from: 'preset', preset: 'Fast', engineLabel: 'NInfer'});
assert.equal(p.dialSourceFor(view, draft, 'model.name'), null);
assert.equal(p.dialSourceFor({kind: 'engine'}, draft, 'kv-dtype'), null);
""")


def test_fit_and_memory_lines():
    _node(r"""
assert.equal(p.fitSummary({verdict: 'fits', needBytes: 29.7 * G, cardTotalBytes: 31.8 * G}),
  "Fits: needs about 29.7 GB of the graphics card's 31.8 GB.");
assert.equal(p.fitSummary({verdict: 'unknown', message: 'The graphics card could not be read.'}),
  "Couldn't check: The graphics card could not be read.");
assert.equal(p.ramSummary({needGB: 18, cushionGB: 6, windowsFreeGB: 40, loadedNow: false}),
  'PC memory: needs 18 GB; Windows has 40.0 GB free and keeps a 6 GB cushion.');
assert.match(p.ramSummary({needGB: 18, cushionGB: 6, windowsFreeGB: 20, loadedNow: false}), /will wait for memory/);
assert.match(p.ramSummary({needGB: 18, cushionGB: 6, windowsFreeGB: 3, loadedNow: true}), /already in use/);
""")


def test_restart_question_lists_every_affected_model():
    _node(r"""
assert.equal(p.restartQuestion([{id: 'quasar-27b', name: 'QUASAR'}], true),
  'QUASAR is loaded right now. Restart now to use the new settings, or keep it running on the old ones until its next load?');
assert.equal(p.restartQuestion([{id: 'a'}, {id: 'b', name: 'B'}], false),
  'a, B are loaded right now and must restart for this. Restart now?');
""")


def test_right_now_and_models_list_escape_and_show_state():
    _node(r"""
assert.match(p.nowStripHtml({switcher: {up: false}}), /switcher not running/);
const now = p.nowStripHtml({switcher: {up: true, running: [{id: 'q', name: '<Q>', state: 'starting'}]},
  card: {usedBytes: G, totalBytes: 2 * G}, windowsFreeBytes: 40 * G, cushionGB: 6, held: []});
for (const part of ['&lt;Q&gt;', 'Loading…', '40.0 GB', '6 GB', 'width:50%']) assert.ok(now.includes(part), part);
const row = {id: 'q', name: '<b>', state: 'ready', engineLabel: 'NInfer', runtimeLabel: 'QUASAR runtime',
  activePreset: null, ramNeedGB: 18, idleMinutes: 0, idleFromSystem: false, held: false};
const table = p.modelsTableHtml([row], true);
for (const part of ['data-unload="q"', '&lt;b&gt;', 'QUASAR runtime', 'never', 'data-settings="q"']) assert.ok(table.includes(part), part);
assert.ok(p.modelsTableHtml([{...row, state: 'stopped'}], true).includes('data-load="q"'));
assert.ok(!p.modelsTableHtml([row], false).includes('data-unload'));
""")


def test_page_contract():
    page, js = PAGE.read_text(encoding="utf-8"), PANEL_JS.read_text(encoding="utf-8")
    assert '<script src="/panel.js"></script>' in page
    for main in ("models", "system", "ninfer", "freetoken"):
        assert f'data-main="{main}"' in page
    for gone in ('id="profile-strip"', 'id="start"', 'id="stop"', 'id="restart"', "serverAction(",
                 'id="restart-dialog"', 'id="fit-panel"', 'id="server-pill"'):
        assert gone not in page, gone
    for present in ('id="restart-ask"', "Restart now", "Next time", 'id="now-strip"', 'id="models-view"',
                    'id="editor"', 'id="preset-picker"', 'id="save-anyway"', 'id="status-strip"'):
        assert present in page, present
    assert "PANEL_NOW_MS = 5000" in js
    for route in ("/api/panel/now", "/api/panel/models", "/api/panel/views/", "/api/panel/import",
                  "/api/panel/registry/restore", "/load", "/unload", "/fit", "/presets/"):
        assert route in js, route
    for code in ("choose_restart", "stale_revision", "switcher_refused"):
        assert code in js
    for banned in ("window.confirm", "window.alert", "window.prompt"):
        assert banned not in js


def test_panel_script_is_served(tmp_path, monkeypatch):
    from freetoken.daemon.settings.app import create_app
    from freetoken.daemon.settings.process_manager import ProcessManager
    from freetoken.daemon.settings.profiles_manager import ProfilesManager

    monkeypatch.setenv("FREETOKEN_REGISTRY", str(tmp_path / "registry.json"))
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -Port 2020\n", encoding="utf-8")
    proc = ProcessManager(boot_file=boot, stop_script=tmp_path / "stop.ps1", log_path=tmp_path / "s.log",
                          lock_path=tmp_path / "gpu.lock", runner=lambda *a, **k: None,
                          readiness=lambda: {"state": "unreachable"}, sleep=lambda _: None, poll_interval=0)
    app = create_app(boot_file=boot, process_manager=proc, profiles=ProfilesManager(tmp_path / "p.json"))
    answer = TestClient(app).get("/panel.js")
    assert answer.status_code == 200 and "javascript" in answer.headers["content-type"]


def test_server_errors_become_plain_words():
    _node(r"""
assert.equal(p.panelErrorText({code: 'switcher_unknown', message: "Can't tell whether Q is loaded right now."}, 'x'),
  "Can't tell whether Q is loaded right now.");
assert.equal(p.panelErrorText({detail: [{field: 'preset', message: 'A preset called Fast already exists.'}]}, 'x'),
  'A preset called Fast already exists.');
assert.equal(p.panelErrorText({detail: 'not found: q'}, 'Could not open these settings.'), 'Could not open these settings.');
assert.equal(p.panelErrorText({}, 'Could not save.'), 'Could not save.');
""")


def test_answering_the_restart_question_sends_the_save_again():
    """Found in the browser check: a `return` inside panelSave's try skipped the retry, so
    "Next time" / "Restart now" closed the question and saved nothing."""
    _node(r"""
const nodes = {};
global.$ = (id) => (nodes[id] ||= {hidden: true, textContent: '', innerHTML: '', value: '', className: '', disabled: false, addEventListener() {}, querySelectorAll: () => []});
global.document = {querySelectorAll: () => [], querySelector: () => null};
global.state = {view: {kind: 'system', url: '/api/panel/views/system'}, settings: {floorGB: 8}, saved: {floorGB: 6}};
global.setBusy = () => {}; global.showFieldErrors = () => {}; global.changedNames = () => [];
const notes = []; global.setNotice = (message) => notes.push(message);
const puts = [];
global.json = async (url, options = {}) => {
  if (options.method === 'PUT') {
    const sent = JSON.parse(options.body); puts.push(sent);
    if (!sent.whenLoaded) return {response: {ok: false, status: 409}, body: {code: 'choose_restart', affected: [{id: 'q', name: 'Q'}], nextTimeAllowed: true}};
    return {response: {ok: true, status: 200}, body: {status: 'saved', revision: 'r2', restarting: [], held: ['q']}};
  }
  return {response: {ok: false, status: 500}, body: {}};
};
(async () => {
  const saving = p.panelSave();
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal($('restart-ask').hidden, false);
  assert.match($('restart-ask-text').textContent, /Q is loaded right now/);
  p.answerRestart('next-time');
  await saving;
  assert.equal(puts.length, 2);
  assert.equal(puts[1].whenLoaded, 'next-time');
  assert.deepEqual(puts[1].system, {floorGB: 8});
  assert.ok(notes.includes('Saved. The loaded model keeps its old settings until its next load.'), notes.join(' / '));
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")
