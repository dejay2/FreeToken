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
                    'id="confirm-ask"', 'id="confirm-ask-ok"', 'id="confirm-ask-cancel"',
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


def test_a_slow_refresh_never_overlaps_the_next_one():
    """Review finding: a 5 s setInterval stacked /now and /models requests on the helper when
    the switcher hung. The loop is chained: the next tick is set only after both settle."""
    _node(r"""
const nodes = {};
global.$ = (id) => (nodes[id] ||= {hidden: true, textContent: '', innerHTML: '', querySelectorAll: () => []});
global.document = {hidden: false, querySelectorAll: () => [], querySelector: () => null};
global.state = {view: null};
const timers = [];
global.setTimeout = (fn, ms) => { timers.push({fn, ms}); return timers.length; };
const pending = [];
let calls = 0;
global.json = (url) => { calls += 1; return new Promise((resolve, reject) => pending.push({url, resolve, reject})); };
const tick = () => new Promise((resolve) => setImmediate(resolve));
(async () => {
  p.startNow(); p.startNow();                         // a second start must not add a chain
  await tick();
  assert.deepEqual(pending.map((row) => row.url), ['/api/panel/now', '/api/panel/models']);
  assert.equal(timers.length, 0, 'no next tick while this one is still waiting');
  pending.shift().resolve({response: {ok: true, status: 200}, body: {switcher: {up: true, running: []}}});
  await tick();
  assert.equal(timers.length, 0, 'still waiting for the model list');
  pending.shift().reject(new Error('helper went away'));  // a failure still settles the tick
  await tick(); await tick();
  assert.equal(timers.length, 1);
  assert.equal(timers[0].ms, 5000);
  assert.equal(calls, 2);
  timers[0].fn();                                      // the next tick starts only now
  await tick();
  assert.equal(calls, 4);
  assert.equal(timers.length, 1);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


# ---- final whole-branch review fix wave ----
def test_load_and_unload_questions_name_the_models():
    """Item 9: Load asks only when another model is loaded; Unload always asks."""
    _node(r"""
const rows = [{id: 'a', name: 'Alpha', state: 'ready'}, {id: 'b', name: 'Beta', state: 'stopped'}];
assert.equal(p.loadQuestion('b', rows), 'This will put away Alpha and load Beta. Carry on?');
assert.equal(p.loadQuestion('a', rows), null);
assert.equal(p.loadQuestion('b', [{id: 'b', name: 'Beta', state: 'stopped'}]), null);
assert.equal(p.loadQuestion('b', [{id: 'a', name: 'Alpha', state: 'starting'}, {id: 'b', name: 'Beta'}]),
  'This will put away Alpha and load Beta. Carry on?');
assert.equal(p.unloadQuestion('a', rows), 'Put away Alpha? Anything using it will stop.');
""")


def test_load_waits_for_the_answer_and_cancel_loads_nothing():
    _node(r"""
const nodes = {};
global.$ = (id) => (nodes[id] ||= {hidden: true, textContent: '', innerHTML: '', querySelectorAll: () => []});
global.document = {hidden: false, querySelectorAll: () => [], querySelector: () => null};
global.state = {view: null};
global.setNotice = () => {};
const posts = [];
global.json = async (url, options = {}) => { if (options.method === 'POST') posts.push(url); return {response: {ok: true, status: 200}, body: {models: [], switcher: {up: true}}}; };
const tick = () => new Promise((resolve) => setImmediate(resolve));
(async () => {
  p.panel.models = [{id: 'a', name: 'Alpha', state: 'ready'}, {id: 'b', name: 'Beta', state: 'stopped'}];
  const first = p.loadModel('b');
  await tick();
  assert.equal($('confirm-ask').hidden, false);
  assert.equal($('confirm-ask-text').textContent, 'This will put away Alpha and load Beta. Carry on?');
  p.answerConfirm(false);
  await first;
  assert.deepEqual(posts, []);
  const second = p.loadModel('b');
  await tick();
  p.answerConfirm(true);
  await second;
  assert.deepEqual(posts, ['/api/panel/models/b/load']);
  p.panel.models = [{id: 'a', name: 'Alpha', state: 'ready'}];  // the refresh after a load replaced the list
  const third = p.unloadModel('a');
  await tick();
  assert.equal($('confirm-ask-text').textContent, 'Put away Alpha? Anything using it will stop.');
  assert.equal($('confirm-ask-ok').textContent, 'Put it away');
  p.answerConfirm(true);
  await third;
  assert.deepEqual(posts, ['/api/panel/models/b/load', '/api/panel/models/a/unload']);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_missing_list_offers_the_backups_too():
    """Item 3: spec, missing or corrupt -> offer restore."""
    _node(r"""
const missing = p.registryProblemHtml({status: 'missing', configPath: '/c.yaml', backups: ['registry.json.bak-20260924-101112-000001']});
assert.ok(missing.includes('id="import-now"'));
assert.ok(missing.includes('data-restore="registry.json.bak-20260924-101112-000001"'));
const bare = p.registryProblemHtml({status: 'missing', configPath: '/c.yaml', backups: []});
assert.ok(bare.includes('id="import-now"') && !bare.includes('data-restore'));
const corrupt = p.registryProblemHtml({status: 'corrupt', message: 'bad', backups: ['registry.json.bak-20260924-101112-000001']});
assert.ok(corrupt.includes('data-restore=') && !corrupt.includes('import-now'));
""")


def test_right_now_says_when_the_switcher_is_on_older_settings():
    """Item 5."""
    _node(r"""
const base = {switcher: {up: true, running: []}, card: null, windowsFreeBytes: null, cushionGB: 6, held: []};
assert.ok(!p.nowStripHtml(base).includes("hasn't picked up"));
assert.ok(p.nowStripHtml({...base, switcher: {up: true, running: [], stale: true}}).includes("The switcher hasn't picked up the latest settings yet."));
""")


def test_a_second_save_press_during_the_fit_check_does_nothing():
    """Item 7: busy is set before the fit check, so a second press cannot send a second PUT."""
    _node(r"""
const nodes = {};
global.$ = (id) => (nodes[id] ||= {hidden: true, textContent: '', innerHTML: '', value: '', className: '', disabled: false, addEventListener() {}, querySelectorAll: () => []});
global.document = {querySelectorAll: () => [], querySelector: () => null};
global.state = {view: {kind: 'model', id: 'q', url: '/api/panel/views/model/q', activePreset: null}, settings: {'kv-dtype': 'int8'}, saved: {}};
const busy = []; global.setBusy = (value) => busy.push(value);
global.showFieldErrors = () => {}; global.changedNames = () => []; global.setNotice = () => {};
global.renderSettings = () => {}; global.renderModelCard = () => {}; global.renderPresetPicker = () => {}; global.showEditor = () => {};
let fits = 0; let puts = 0; let releaseFit;
global.json = (url, options = {}) => {
  if (url.endsWith('/fit')) { fits += 1; return new Promise((resolve) => { releaseFit = () => resolve({response: {ok: true, status: 200}, body: {verdict: 'fits'}}); }); }
  if (options.method === 'PUT') { puts += 1; return Promise.resolve({response: {ok: true, status: 200}, body: {revision: 'r2', restarting: [], held: []}}); }
  return Promise.resolve({response: {ok: false, status: 500}, body: {}});
};
const tick = () => new Promise((resolve) => setImmediate(resolve));
(async () => {
  const first = p.panelSave();
  await tick();
  assert.deepEqual(busy, [true]);
  await p.panelSave();                      // the second press while the check runs
  assert.equal(fits, 1);
  releaseFit();
  await first;
  assert.equal(fits, 1);
  assert.equal(puts, 1);
  assert.deepEqual(busy, [true, false]);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")
