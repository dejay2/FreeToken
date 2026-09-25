"""The Test tab's browser behaviour after review: the page's functions run in node against a
small fake DOM (a `$` that hands out plain objects), so each finding has a test that fails
without its fix."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
STATIC = REPO / "python" / "freetoken" / "daemon" / "settings" / "static"
PG_JS = STATIC / "playground.js"
PANEL_JS = STATIC / "panel.js"

# A fake page: every $('id') is one plain object; events fire by hand; timers are recorded.
FAKE_DOM = r"""
const els = {};
function fakeEl(id) {
  if (!els[id]) els[id] = { id, value: '', checked: false, hidden: false, disabled: false, innerHTML: '', textContent: '', dataset: {}, listeners: {},
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    fire(type, target) { for (const fn of this.listeners[type] || []) fn({ type, target: target || this }); },
    closest() { return null; }, querySelectorAll() { return []; }, querySelector() { return null; } };
  return els[id];
}
global.$ = fakeEl;
global.window = { localStorage: null };
global.document = { hidden: false };
global.setNotice = () => {};
global.askConfirm = async () => true;
global.panelErrorText = (body, fallback) => (body && body.message) || fallback;
global.json = async () => { throw new TypeError('Failed to fetch'); };
const scheduled = [];
global.setTimeout = (fn, ms) => { scheduled.push(ms); return 1; };
global.clearTimeout = () => {};
const OFFLINE = "Couldn't reach the settings page. Try again.";
const okJson = (routes) => async (url, options) => {
  const body = JSON.parse((options && options.body) || 'null');
  const answer = routes(url, body, options);
  return { response: { ok: true, status: 200 }, body: answer };
};
"""


def _node(script: str, module: Path = PG_JS) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to run the page's JavaScript")
    prelude = (f"const assert = require('node:assert/strict');\n{FAKE_DOM}\n"
               f"const p = require({json.dumps(str(module))});\n(async () => {{\n")
    run = subprocess.run([node, "-e", prelude + script + "\n})().catch((e) => { console.error(e); process.exit(1); });"],
                         capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr


def test_editing_the_form_hides_the_plan_and_start_sends_the_plan_that_was_shown():
    """Review item 1: Start confirmed whatever the form held at that moment, not the plan on
    screen. Now any input or change clears the plan, and Start posts the planned request."""
    _node(r"""
const posts = [];
global.json = okJson((url, body) => { posts.push([url, body]);
  return url.endsWith('/plan') ? { before: 'quasar-27b', steps: [], estimateS: 5 } : { id: 'j1', status: 'running', steps: [], sides: [] }; });
p.pgWire();
$('pg-prompt').value = 'first prompt';
await p.pgRun();
assert.equal($('pg-plan').hidden, false);
$('pg-prompt').value = 'edited after planning';   // no event: the plan stays, Start must send 'first prompt'
await p.pgStart();
const start = posts.find(([url]) => url === '/api/playground/runs');
assert.equal(start[1].prompt, 'first prompt');
assert.equal(start[1].confirm, true);
assert.equal(start[1].expectBefore, 'quasar-27b');
assert.equal(p.pg.plan, null);
// an input or change in the form clears the plan and needs a fresh Run
await p.pgRun();
assert.equal($('pg-plan').hidden, false);
$('test-view').fire('input', $('pg-prompt'));
assert.equal(p.pg.plan, null); assert.equal($('pg-plan').hidden, true);
await p.pgRun();
$('test-view').fire('change', $('pg-b-on'));
assert.equal(p.pg.plan, null); assert.equal($('pg-plan').hidden, true);
const before = posts.length;
await p.pgStart();   // nothing shown: nothing sent
assert.equal(posts.length, before);
""")


def test_network_errors_are_plain_words_and_never_stop_the_polling():
    """Review item 2: a fetch that throws (helper restarting, Wi-Fi gone) left an unhandled
    rejection and a dead page. Every button says so in plain words; Start always schedules a
    poll; opening the tab polls even when the option list fails."""
    _node(r"""
p.pgWire();
await p.pgRun();
assert.equal($('pg-error').textContent, OFFLINE); assert.equal($('pg-error').hidden, false);
p.pg.plan = { before: 'quasar-27b', steps: [] }; p.pg.planRequest = { prompt: 'Hi', sides: [] };
scheduled.length = 0;
await p.pgStart();
assert.equal($('pg-error').textContent, OFFLINE);
assert.equal(scheduled.length, 1, 'a Start attempt always schedules the next poll');
$('pg-error').textContent = ''; $('pg-error').hidden = true;
await p.pgStop();
assert.equal($('pg-error').textContent, OFFLINE);
$('pg-error').textContent = '';
scheduled.length = 0;
await p.pgOpen();
assert.equal($('pg-error').textContent, OFFLINE);
assert.equal(scheduled.length, 1, 'the tab polls (and re-polls) even when the option list could not be read');
""")


def test_results_keep_thinking_open_and_scroll_between_polls():
    """Review item 3: rewriting #pg-results every 500 ms closed every Thinking box and reset
    the answers' scroll. Unchanged results are not rewritten; changed ones keep each side's
    details.open and scroll (a box scrolled to the bottom stays at the bottom)."""
    _node(r"""
const mk = (side, open, boxes) => ({ dataset: { side }, details: { open },
  answers: boxes.map(([top, height, client]) => ({ scrollTop: top, scrollHeight: height, clientHeight: client })),
  querySelector(sel) { return sel.startsWith('details') ? this.details : null; },
  querySelectorAll() { return this.answers; } });
const box = $('pg-results');
const a = mk('A', true, [[120, 2000, 400], [1600, 2000, 400]]);
box.querySelectorAll = () => [a];
const kept = p.pgResultsState(box);
const a2 = mk('A', false, [[0, 3000, 400], [0, 3000, 400]]);
const b2 = mk('B', false, [[0, 100, 400]]);
box.querySelectorAll = () => [a2, b2];
p.pgResultsRestore(box, kept);
assert.equal(a2.details.open, true);
assert.equal(a2.answers[0].scrollTop, 120, 'a box scrolled part-way keeps its place');
assert.equal(a2.answers[1].scrollTop, 3000, 'a box at the bottom follows the stream');
assert.equal(b2.details.open, false);
// unchanged results are not rewritten at all
let writes = 0;
Object.defineProperty(box, 'innerHTML', { get() { return this._html || ''; }, set(v) { this._html = v; writes += 1; } });
p.pg.shown = null;
p.pg.job = { id: 'j1', status: 'running', steps: [], sides: [{ key: 'A', name: 'M', answer: 'hello', reasoning: 'why', stats: null }] };
p.pgRenderJob(); p.pgRenderJob();
assert.equal(writes, 1);
p.pg.job.sides[0].answer = 'hello there';
p.pgRenderJob();
assert.equal(writes, 2);
""")


def test_cleared_history_does_not_come_back_from_the_last_finished_job():
    """Review item 5: Clear history, then reopen the tab: the poll saw the last finished job,
    found it missing from the history and put it back. Cleared ids are remembered (and kept
    in this browser) so the finished job is not remembered again."""
    _node(r"""
const store = { data: {}, setItem(k, v) { this.data[k] = v; }, getItem(k) { return this.data[k] ?? null; }, removeItem(k) { delete this.data[k]; } };
global.window = { localStorage: store };
const job = { id: 'j1', startedAt: '2026-09-25T10:00:00Z', status: 'done', prompt: 'Hi', steps: [], sides: [] };
global.json = okJson((url) => url.endsWith('/options') ? { models: [] } : job);
p.pgWire();
await p.pgOpen();
assert.equal(p.pg.history.length, 1);
await p.pgClear();
assert.equal(p.pg.history.length, 0);
p.pgRemember(job);
assert.equal(p.pg.history.length, 0, 'the cleared job is not remembered again');
await p.pgOpen();
assert.equal(p.pg.history.length, 0, 'reopening the tab does not bring it back');
p.pg.cleared = [];   // a fresh page reads the cleared ids from the browser store
await p.pgOpen();
assert.equal(p.pg.history.length, 0);
assert.deepEqual(p.loadCleared(store), ['j1']);
assert.deepEqual(p.loadCleared({ getItem() { return '{broken'; } }), []);
const next = { ...job, id: 'j2' };
p.pgRemember(next);
assert.equal(p.pg.history.length, 1, 'a new test is still remembered');
""")


def test_a_late_view_render_never_paints_over_the_test_tab():
    """Review item 6: a slow /api/panel/views answer arriving after a switch to the Test tab
    drew the settings editor over it."""
    _node(r"""
global.state = { view: null, settings: {}, saved: {}, dials: [], groups: [], model: null };
global.$ = () => { throw new Error('the editor must not be touched under the Test tab'); };
p.panel.main = 'test';
p.applyView({ kind: 'model', settings: { a: 1 }, dials: [], groups: [], revision: 'r9' }, '/api/panel/views/model/x');
assert.equal(state.view, null);
assert.notEqual(p.panel.revision, 'r9');
""", module=PANEL_JS)


def test_speed_labels_carry_the_technical_names_as_tooltips():
    """Review item 7."""
    _node(r"""
const rows = p.speedRows({ key: 'A', stats: { firstWordMs: 400, writeTps: 40, writeSource: 'engine' } }, null);
const title = (key) => rows.find((r) => r.key === key).title;
assert.match(title('firstWordMs'), /TTFT/);
assert.match(title('writeTps'), /tok\/s/);
assert.match(title('promptTokens'), /prompt_tokens/);
assert.match(title('finishReason'), /finish_reason/);
for (const row of rows) assert.ok(row.title && row.title.length, row.key);
const html = p.pgResultsHtml({ sides: [{ key: 'A', name: 'M', answer: 'x', stats: { firstWordMs: 400 } }] });
assert.match(html, /<th scope="row" title="TTFT[^"]*">First word after<\/th>/);
""")


def test_showing_an_earlier_test_offers_the_way_back_to_the_current_one():
    """Review item 8: after Show on an earlier test while a test runs (or after one finished),
    nothing led back to the current test."""
    _node(r"""
p.pgWire();
p.pg.job = { id: 'j2', status: 'running', steps: [], sides: [] };
p.pg.shown = { id: 'j1', at: '2026-09-25T10:00:00Z', sides: [] };
p.pgRenderJob();
assert.match($('pg-status').innerHTML, /Earlier test from [^<]*Sep/);
assert.match($('pg-status').innerHTML, /<button[^>]*data-pg-back[^>]*>Back to the current test<\/button>/);
const back = { dataset: {}, closest: (sel) => sel === '[data-pg-back]' ? back : null };
$('pg-status').fire('click', back);
assert.equal(p.pg.shown, null);
assert.equal($('pg-status').innerHTML, 'Running…');
// no current test: no button
p.pg.job = null; p.pg.shown = { id: 'j1', at: 'then', sides: [] };
p.pgRenderJob();
assert.doesNotMatch($('pg-status').innerHTML, /data-pg-back/);
assert.equal(p.statusHtml(null, { at: '<b>' }), 'Earlier test from &lt;b&gt;');
""")
