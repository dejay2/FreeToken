"""Control panel stage A: the deferred reviewer minors (page side, node). Each fails without its fix."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.settings.test_panel_page import _FAKE_PAGE, _node
from tests.settings.test_playground_page import STATIC


# ---- 6: the load and remove questions read the list again first ----
def test_load_question_uses_a_fresh_list():
    _node(_FAKE_PAGE + r"""
global.state = {view: null};
global.json = async (url, options = {}) => {
  if (options.method === 'POST') posts.push(url);
  return {response: {ok: true, status: 200}, body: {models: [{id: 'a', name: 'Alpha', state: 'ready'}, {id: 'b', name: 'Beta', state: 'stopped'}], switcherUp: true}};
};
(async () => {
  p.panel.models = [{id: 'a', name: 'Alpha', state: 'stopped'}, {id: 'b', name: 'Beta', state: 'stopped'}];  // 5 s old
  const run = p.loadModel('b'); await tick();
  assert.equal($('confirm-ask').hidden, false, 'Alpha is loaded now, so the load must ask first');
  assert.equal($('confirm-ask-text').textContent, 'This will put away Alpha and load Beta. Carry on?');
  p.answerConfirm(false); await run;
  assert.deepEqual(posts, []);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_remove_question_uses_the_state_now_not_when_the_settings_opened():
    _node(_FAKE_PAGE + r"""
global.state = {view: {kind: 'model', id: 'q', name: 'QUASAR', engine: 'ninfer', state: 'stopped', artifact: '/q', url: '/api/panel/views/model/q'}, settings: {}, saved: {}};
global.json = async () => ({response: {ok: true, status: 200}, body: {models: [{id: 'q', name: 'QUASAR', state: 'ready'}], switcherUp: true}});
(async () => {
  const run = p.openRemove(); await tick();
  assert.equal($('remove-ask-text').textContent, 'Remove QUASAR from the list? It is loaded now and will be put away first. Apps will no longer see it.');
  p.answerRemove(false); await run;
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


# ---- 16: Rename/Delete stay off without a preset, busy or not ----
def test_preset_buttons_follow_the_preset_not_just_busy():
    _node(_FAKE_PAGE + r"""
global.state = {view: {kind: 'model', id: 'q', activePreset: null}, busy: false};
$('preset-rename').disabled = false; $('preset-delete').disabled = false;   // what setBusy(false) leaves
p.presetButtons();
assert.equal($('preset-rename').disabled, true); assert.equal($('preset-delete').disabled, true);
state.view.activePreset = 'Fast';
p.presetButtons();
assert.equal($('preset-rename').disabled, false); assert.equal($('preset-delete').disabled, false);
state.busy = true; p.presetButtons();
assert.equal($('preset-rename').disabled, true);
""")


def test_set_busy_calls_the_preset_rule():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    line = next(row for row in page.splitlines() if "function setBusy(value)" in row)
    assert "presetButtons()" in line


# ---- 17: a row about another model names it ----
def test_unplaced_rows_name_the_model():
    _node(_FAKE_PAGE + r"""
p.panel.models = [{id: 'fable-27b', name: 'Fable 27B'}];
assert.deepEqual(p.unplacedErrors([{field: 'draft-tokens', where: 'fable-27b', message: 'Too many.'},
                                   {field: 'x', where: 'Twin 27B', message: 'Too long.'}]),
  ['Fable 27B: Too many.', 'Twin 27B: Too long.']);
""")


# ---- 18: the next-time note names every held model ----
def test_next_time_note_names_each_model():
    _node(r"""
assert.equal(p.heldNote([{id: 'q', name: 'QUASAR'}]), 'Saved. QUASAR keeps its old settings until its next load.');
assert.equal(p.heldNote([{id: 'f', name: 'Fable'}, {id: 't', name: 'Twin'}, {id: 'q', name: 'QUASAR'}]),
  'Saved. Fable, Twin and QUASAR keep their old settings until their next load.');
assert.equal(p.heldNote([]), 'Saved. The loaded model keeps its old settings until its next load.');
""")


# ---- 19: an unsaved preset pick counts as a change when leaving ----
def test_leaving_with_an_unsaved_preset_pick_is_stopped():
    _node(_FAKE_PAGE + r"""
global.state = {view: {kind: 'model', id: 'q', activePreset: 'Fast', savedPreset: null}, settings: {}, saved: {}};
assert.equal(p.leaveGuard(), false);
assert.match(notes[notes.length - 1], /Save or undo your changes first/);
state.view.savedPreset = 'Fast';
assert.equal(p.leaveGuard(), true);
""")


def test_closing_the_browser_tab_with_an_unsaved_preset_pick_warns():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    line = next(row for row in page.splitlines() if "'beforeunload'" in row)
    assert "panelExtraDirty()" in line


# ---- 21: no polling while the browser tab is hidden or the Test tab is not shown ----
def test_add_download_poll_pauses_while_hidden():
    _node(_FAKE_PAGE + r"""
global.state = {view: null};
const timers = []; global.setTimeout = (fn, ms) => { timers.push({fn, ms}); return timers.length; }; global.clearTimeout = () => {};
const gets = [];
global.json = async (url) => { gets.push(url); return {response: {ok: true, status: 200}, body: {id: 'j1', stage: 'downloading', percent: 5}}; };
(async () => {
  $('add-wizard').hidden = false;
  p.addState.job = {id: 'j1', stage: 'downloading'};
  document.hidden = true;
  await p.pollAddJob(p.addState.pollGen);
  assert.deepEqual(gets, [], 'nothing is asked while the tab is hidden');
  assert.equal(timers.length, 1, 'it looks again later');
  document.hidden = false;
  await timers[0].fn();
  assert.deepEqual(gets, ['/api/panel/add/downloads/j1']);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def _node_playground(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to run the page's JavaScript")
    prelude = ("const assert = require('node:assert/strict');\n"
               f"global.panel = {{main: 'models'}};\n"
               f"const g = require({json.dumps(str(STATIC / 'playground.js'))});\n")
    run = subprocess.run([node, "-e", prelude + script], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr


def test_test_tab_poll_pauses_away_from_the_tab_and_while_hidden():
    _node_playground(r"""
const gets = []; const timers = [];
global.json = async (url) => { gets.push(url); return {response: {ok: true, status: 200}, body: {status: 'idle'}}; };
global.setTimeout = (fn, ms) => { timers.push(fn); return timers.length; }; global.clearTimeout = () => {};
global.document = {hidden: false};
(async () => {
  panel.main = 'models';
  await g.pgPoll();
  assert.deepEqual(gets, [], 'no polling while the Models tab is shown');
  panel.main = 'test'; document.hidden = true;
  await g.pgPoll();
  assert.deepEqual(gets, [], 'no polling while the browser tab is hidden');
  assert.equal(timers.length, 1, 'but it looks again later');
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


# ---- 26: the restore buttons say what they bring back and from when ----
def test_restore_buttons_say_what_and_when():
    _node(r"""
const html = p.backupListHtml(['registry.json.bak-20260924-101112-000001', 'registry.json.bak-20260923-090000-000001']);
assert.match(html, /Bring back the model list as it was on [^<]+ \(newest copy\)<\/button>/);
assert.equal((html.match(/newest copy/g) || []).length, 1);
assert.doesNotMatch(html, /Restore the copy/);
const corrupt = p.registryProblemHtml({status: 'corrupt', message: 'bad', backups: ['registry.json.bak-20260924-101112-000001']});
assert.match(corrupt, /every model and its settings/);
assert.match(p.restoredNote('registry.json.bak-20260924-101112-000001'), /^The model list is back as it was on .+\.$/);
""")
