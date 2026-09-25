"""PR #20 (control panel minors) Codex review findings. Each test fails without its fix."""

from __future__ import annotations

import pytest

from freetoken.daemon.settings import registry as reg
from freetoken.daemon.settings.registry import find_model
from freetoken.daemon.settings.swap_config import render_config
from tests.settings.registry_fixtures import five
from tests.settings.test_panel_page import _FAKE_PAGE, _node
from tests.settings.test_panel_routes import engine_settings, env, run_spawned, seed  # noqa: F401


# ---- 1: Remove carries the editor's revision and the files its question named ----
def test_remove_is_refused_when_the_models_files_changed(env):
    revision = seed(env)
    answer = env.client.post("/api/panel/models/quasar-27b/remove", json={
        "revision": revision, "artifact": "~/ninfer-work/models/an_older_copy.ninfer", "deleteFiles": False})
    assert answer.status_code == 409, answer.text
    assert answer.json()["code"] == "artifact_changed" and "nothing was removed" in answer.json()["message"]
    assert "quasar-27b" in [m["id"] for m in env.store.load()[0]["models"]]
    artifact = find_model(env.store.load()[0], "quasar-27b")["artifact"]
    ok = env.client.post("/api/panel/models/quasar-27b/remove", json={
        "revision": revision, "artifact": artifact, "deleteFiles": False})
    assert ok.status_code == 200, ok.text


def test_remove_does_not_move_the_editors_revision():
    _node(_FAKE_PAGE + r"""
global.state = {view: {kind: 'model', id: 'q', name: 'QUASAR', engine: 'ninfer', state: 'stopped', artifact: '~/m/q.ninfer',
  url: '/api/panel/views/model/q', revision: 'r1'}, settings: {}, saved: {}};
global.json = async (url, options = {}) => {
  if (options.method === 'POST') { posts.push({url, body: JSON.parse(options.body)}); return {response: {ok: false, status: 409}, body: {code: 'artifact_changed', message: 'changed'}}; }
  return {response: {ok: true, status: 200}, body: {revision: 'r9', models: [{id: 'q', name: 'QUASAR', state: 'ready'}], switcherUp: true}};
};
(async () => {
  p.panel.revision = 'r1';
  const first = p.openRemove(); await tick();
  assert.match($('remove-ask-text').textContent, /It is loaded now/);
  p.answerRemove(false); await first;
  assert.equal(p.panel.revision, 'r1', 'a cancelled Remove must not move the revision a later Save sends');
  const second = p.openRemove(); await tick();
  p.answerRemove(true); await second;
  assert.deepEqual(posts[0].body, {revision: 'r1', artifact: '~/m/q.ninfer', deleteFiles: false});
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


# ---- 2: the loaded-entry comparison uses what the switcher actually gets ----
def test_a_hand_spelled_value_that_renders_differently_asks_first(env):
    doc = five()
    find_model(doc, "quasar-27b")["overrides"]["max-concurrency"] = "04"  # renders --max-concurrency 04
    env.store.path.parent.mkdir(parents=True, exist_ok=True)
    env.store.path.write_bytes(reg.dumps(doc))
    env.writer.write(render_config(doc, {}))
    assert "--max-concurrency 04" in env.cfg.read_text()
    revision = env.store.load()[1]
    env.switcher.states = {"quasar-27b": "ready"}
    answer = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}})
    assert answer.status_code == 409 and answer.json()["code"] == "choose_restart", answer.text
    assert [row["id"] for row in answer.json()["affected"]] == ["quasar-27b"]
    held = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7},
                                                     "whenLoaded": "next-time"})
    assert held.status_code == 200 and held.json()["held"] == ["quasar-27b"], held.text
    assert "--max-concurrency 04" in env.cfg.read_text()


# ---- 3: a restart is applied only on its own write or a later panel write ----
def test_an_old_file_copied_back_marks_the_restart_superseded(env):
    revision = seed(env)
    old = env.cfg.read_text()
    env.switcher.states = {"quasar-27b": "ready"}
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    answer = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "whenLoaded": "restart"})
    assert answer.status_code == 200 and answer.json()["restarting"] == ["quasar-27b"], answer.text
    env.cfg.write_text(old)  # the old config copied back by hand: both hashes are the old file's
    env.service.restart_wait_s = 0
    run_spawned(env)
    restart = env.service.last_restart
    assert restart["ok"] is False and "changed outside the control panel" in restart["message"], restart
    assert ("load", "quasar-27b") not in env.switcher.calls


# ---- 6: a hand-written file's copy is made before anything changes ----
def test_a_failed_hand_copy_changes_nothing(env, monkeypatch):
    first = seed(env)
    revision = env.client.put("/api/panel/system", json={"revision": first, "system": {"floorGB": 8}}).json()["revision"]
    backup = env.store.backups()[0]
    env.cfg.write_text("# my own file\nmodels: {}\n")
    before = env.store.path.read_bytes()

    def fail():
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(env.writer, "backup_hand_written", fail)
    with pytest.raises(OSError):
        env.service.save_system({"floorGB": 7}, revision, None)
    assert env.store.path.read_bytes() == before
    with pytest.raises(OSError):
        env.service.restore(backup)
    assert env.store.path.read_bytes() == before
    assert env.cfg.read_text() == "# my own file\nmodels: {}\n"
    assert not (env.cfg.parent / "config.yaml.new").exists()


# ---- 4: a saved preset change is saved, and picking the saved preset back is allowed ----
_VIEW_PAGE = _FAKE_PAGE + r"""
global.showFieldErrors = () => {}; global.renderModelCard = () => {}; global.renderSettings = () => {};
const gets = [];
global.json = async (url, options = {}) => {
  if (options.method === 'PUT') { posts.push(url); return {response: {ok: true, status: 200}, body: {revision: 'r2', restarting: []}}; }
  gets.push(url);
  if (url.startsWith('/api/panel/views/model/q')) {
    const picked = url.includes('?preset=') ? decodeURIComponent(url.split('?preset=')[1]) || null : 'Fast';
    return {response: {ok: true, status: 200}, body: {kind: 'model', id: 'q', name: 'Q', revision: 'r2', settings: {}, dials: [], groups: [],
      presets: ['Fast'], activePreset: picked, savedPreset: url.includes('?preset=') ? null : 'Fast'}};
  }
  return {response: {ok: true, status: 200}, body: {}};
};
"""


def test_saving_a_preset_pick_reloads_a_clean_view():
    _node(_VIEW_PAGE + r"""
global.state = {view: {kind: 'model', id: 'q', url: '/api/panel/views/model/q?preset=Fast', revision: 'r1', activePreset: 'Fast', savedPreset: null},
  settings: {}, saved: {}};
(async () => {
  p.panel.revision = 'r1';
  await p.panelSave({anyway: true});
  assert.deepEqual(posts, ['/api/panel/models/q']);
  assert.ok(gets.includes('/api/panel/views/model/q'), `the view reloads after the save: ${gets}`);
  assert.equal(state.view.savedPreset, 'Fast');
  assert.equal(p.viewDirty(), false);
  assert.ok(notes.includes('Saved.'), notes.join(' / '));
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_picking_the_saved_preset_back_is_allowed():
    _node(_VIEW_PAGE + r"""
global.state = {view: {kind: 'model', id: 'q', url: '/api/panel/views/model/q?preset=Fast', revision: 'r1', activePreset: 'Fast', savedPreset: null},
  settings: {}, saved: {}};
(async () => {
  $('preset-picker').value = '';
  await p.pickPreset();
  assert.deepEqual(gets, ['/api/panel/views/model/q?preset=']);
  assert.equal(state.view.activePreset, null);
  assert.equal(p.viewDirty(), false);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


# ---- 5: Load stops when the switcher's state is not known ----
@pytest.mark.parametrize("answer", [
    "{models: [{id: 'a', name: 'Alpha', state: 'unknown'}, {id: 'b', name: 'Beta', state: 'unknown'}], switcherUp: false}",
    "{models: [{id: 'a', name: 'Alpha', state: 'unknown'}, {id: 'b', name: 'Beta', state: 'stopped'}], switcherUp: true}",
])
def test_load_stops_when_the_switcher_is_not_answering(answer):
    _node(_FAKE_PAGE + r"""
global.state = {view: null};
global.json = async (url, options = {}) => { if (options.method === 'POST') posts.push(url); return {response: {ok: true, status: 200}, body: ANSWER}; };
(async () => {
  p.panel.models = [{id: 'a', name: 'Alpha', state: 'ready'}, {id: 'b', name: 'Beta', state: 'stopped'}];
  await p.loadModel('b');
  assert.deepEqual(posts, []);
  assert.equal($('confirm-ask').hidden, true);
  assert.equal(notes[notes.length - 1], "The model switcher isn't answering; try again.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
""".replace("ANSWER", answer))
