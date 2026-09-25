"""Sleep and Wake on the control panel (review focus 5): rows, routes and plain words."""

from __future__ import annotations

from tests.settings.test_panel_page import _node
from tests.settings.test_panel_routes import env, seed  # noqa: F401 - the fixture


def _wire(env, state="serving", reply=None):
    calls = []
    env.service._freetoken_state = lambda: state
    env.service._freetoken_control = lambda action: calls.append(action) or (reply or {"status": "ok"})
    return calls


def test_only_the_loaded_freetoken_row_offers_sleep(env):
    seed(env)
    env.switcher.states = {"qwen3.8-flash": "ready"}
    _wire(env, state="serving")
    rows = {row["id"]: row for row in env.client.get("/api/panel/models").json()["models"]}
    assert rows["qwen3.8-flash"]["sleep"] == "awake"
    assert rows["qwen3.8-flash-abliterated"]["sleep"] is None and rows["quasar-27b"]["sleep"] is None
    _wire(env, state="sleeping")
    rows = {row["id"]: row for row in env.client.get("/api/panel/models").json()["models"]}
    assert rows["qwen3.8-flash"]["sleep"] == "asleep"
    now = env.client.get("/api/panel/now").json()
    assert now["switcher"]["running"][0]["sleep"] == "asleep"


def test_an_unwired_or_failing_state_probe_reads_as_awake(env):
    seed(env)
    env.switcher.states = {"qwen3.8-flash": "ready"}
    rows = {row["id"]: row for row in env.client.get("/api/panel/models").json()["models"]}
    assert rows["qwen3.8-flash"]["sleep"] == "awake"

    def boom():
        raise OSError("helper away")

    env.service._freetoken_state = boom
    rows = {row["id"]: row for row in env.client.get("/api/panel/models").json()["models"]}
    assert rows["qwen3.8-flash"]["sleep"] == "awake"


def test_sleep_and_wake_go_to_the_helper_not_the_switcher(env):
    seed(env)
    env.switcher.states = {"qwen3.8-flash": "ready"}
    calls = _wire(env)
    assert env.client.post("/api/panel/models/qwen3.8-flash/sleep").json()["sleep"] == "asleep"
    assert env.client.post("/api/panel/models/qwen3.8-flash/wake").json()["sleep"] == "awake"
    assert calls == ["sleep", "wake"] and env.switcher.calls == []


def test_sleep_refusals_speak_plainly(env):
    seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    _wire(env)
    r = env.client.post("/api/panel/models/quasar-27b/sleep")
    assert r.status_code == 409 and r.json()["code"] == "not_freetoken"
    r = env.client.post("/api/panel/models/qwen3.8-flash/sleep")
    assert r.status_code == 409 and r.json()["code"] == "not_loaded"
    env.switcher.states = {"qwen3.8-flash": "ready"}
    _wire(env, reply={"status": "busy"})
    r = env.client.post("/api/panel/models/qwen3.8-flash/sleep")
    assert r.status_code == 409 and "chat is still running" in r.json()["message"]
    _wire(env, reply={"status": "rejected", "error": "the graphics card has 2.0 GB free and waking needs 24.5 GB; close the game"})
    r = env.client.post("/api/panel/models/qwen3.8-flash/wake")
    assert r.status_code == 503 and "close the game" in r.json()["message"]
    _wire(env, reply={"status": "timeout"})
    r = env.client.post("/api/panel/models/qwen3.8-flash/wake")
    assert r.status_code == 504 and r.json()["code"] == "wake_timeout"
    _wire(env, reply={"status": "unreachable", "error": "refused"})
    r = env.client.post("/api/panel/models/qwen3.8-flash/sleep")
    assert r.status_code == 503 and r.json()["code"] == "sleep_unreachable" and "not answering" in r.json()["message"]
    env.service._freetoken_control = None
    r = env.client.post("/api/panel/models/qwen3.8-flash/sleep")
    assert r.status_code == 503 and r.json()["code"] == "helper_missing"
    r = env.client.post("/api/panel/models/no-such-model/sleep")
    assert r.status_code == 404


def test_the_page_words_and_buttons():
    _node(r"""
assert.equal(p.stateWord(p.rowState({state: 'ready', sleep: 'asleep'})), 'Asleep (graphics card free)');
assert.equal(p.stateWord(p.rowState({state: 'ready', sleep: 'awake'})), 'Loaded');
assert.equal(p.rowState({state: 'stopped', sleep: null}), 'stopped');
const rows = [
  {id: 'qwen3.8-flash', name: 'Q', state: 'ready', sleep: 'asleep', engineLabel: 'FreeToken', ramNeedGB: 62},
  {id: 'quasar-27b', name: 'QS', state: 'stopped', sleep: null, engineLabel: 'NInfer', ramNeedGB: 18},
];
const html = p.modelsTableHtml(rows, true);
assert.ok(html.includes('data-wake="qwen3.8-flash"') && html.includes('data-unload="qwen3.8-flash"'));
assert.ok(html.includes('dot asleep') && !html.includes('data-sleep='));
assert.ok(html.includes('Asleep (graphics card free)'));
const awake = p.modelsTableHtml([{...rows[0], sleep: 'awake'}], true);
assert.ok(awake.includes('data-sleep="qwen3.8-flash"') && awake.includes('Frees the graphics card for games'));
assert.ok(!awake.includes('data-wake='));
assert.equal(p.sleepButtonHtml(rows[1]), '');
// With the switcher down there are no buttons at all, sleep included.
assert.ok(!p.modelsTableHtml(rows, false).includes('data-wake='));
// The "Right now" strip says asleep too.
const strip = p.nowStripHtml({switcher: {up: true, running: [{id: 'qwen3.8-flash', name: 'Q', state: 'ready', sleep: 'asleep'}]}});
assert.ok(strip.includes('Asleep (graphics card free)'));
""")
