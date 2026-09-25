"""The Test tab page: plain-words helpers (node) and the page contract."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
STATIC = REPO / "python" / "freetoken" / "daemon" / "settings" / "static"
PG_JS = STATIC / "playground.js"
PANEL_JS = STATIC / "panel.js"
PAGE = STATIC / "index.html"


def _node(script: str, module: Path = PG_JS) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to run the page's JavaScript")
    prelude = f"const p = require({json.dumps(str(module))}); const assert = require('node:assert/strict');\n"
    run = subprocess.run([node, "-e", prelude + script], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr


def test_times_and_speeds_in_plain_words():
    _node(r"""
assert.equal(p.fmtMs(420), '420 ms');
assert.equal(p.fmtMs(12840), '12.8 s');
assert.equal(p.fmtMs(119999), '2 min 0 s');
assert.equal(p.fmtMs(151000), '2 min 31 s');
assert.equal(p.fmtMs(null), '—');
assert.equal(p.fmtGuess(20), '20 s');
assert.equal(p.fmtGuess(150), '2.5 min');
assert.equal(p.fmtRate({writeTps: 41.26, writeSource: 'engine'}), '41.3 tokens a second');
assert.equal(p.fmtRate({writeTps: 9.94, writeSource: 'measured'}), '~9.9 tokens a second');
assert.equal(p.fmtRate({writeTps: null}), '—');
assert.equal(p.guessWords({guesses: {proposed: 514, kept: 365, keptPct: 71}}), '71% (365 of 514 guesses)');
assert.equal(p.guessWords({guesses: null}), 'not reported by this engine');
assert.equal(p.stopWords('length'), 'hit the longest-answer limit');
assert.equal(p.stopWords('cancelled'), 'stopped');
assert.equal(p.tokensWords(812, false, 640), '812 tokens (640 reused)');
assert.equal(p.tokensWords(12, true), '~12 tokens');
""")


def test_better_tag_needs_a_real_difference():
    _node(r"""
const a = {key: 'A', loadMs: 20000, stats: {firstWordMs: 400, writeTps: 40, writeSource: 'engine', totalMs: 12000,
  guesses: {keptPct: 71, kept: 1, proposed: 1}}};
const b = {key: 'B', loadMs: null, stats: {firstWordMs: 410, writeTps: 30, writeSource: 'engine', totalMs: 16000, guesses: null}};
assert.equal(p.betterSide(a, b, 'writeTps'), 'A');
assert.equal(p.betterSide(a, b, 'totalMs'), 'A');
assert.equal(p.betterSide(a, b, 'firstWordMs'), null);   // 2.4% apart: no tag
assert.equal(p.betterSide(a, b, 'guessPct'), null);      // B did not report
assert.equal(p.betterSide(a, b, 'loadMs'), null);        // loading is never judged
const measured = {...b, stats: {...b.stats, writeSource: 'measured'}};
assert.equal(p.betterSide(a, measured, 'writeTps'), null);  // engine vs measured: not comparable
assert.equal(p.betterSide(a, measured, 'totalMs'), 'A');
const stopped = {...b, stats: {...b.stats, finishReason: 'cancelled'}};
assert.equal(p.betterSide(a, stopped, 'totalMs'), null);    // a stopped answer is never judged
assert.equal(p.betterSide(stopped, a, 'writeTps'), null);
const rowsB = p.speedRows(b, a);
assert.equal(rowsB.find((r) => r.key === 'loadMs').text, 'already loaded');
assert.equal(rowsB.find((r) => r.key === 'writeTps').better, false);
assert.equal(p.speedRows(a, b).find((r) => r.key === 'writeTps').better, true);
assert.deepEqual(p.speedRows(a, b).map((r) => r.label), ['Loading (not counted)', 'First word after', 'Writing speed',
  'Whole answer', 'Prompt size', 'Answer size', 'Guesses kept', 'Stopped because']);
""")


def test_steps_plan_and_status_lines():
    _node(r"""
assert.equal(p.stepLine({state: 'running', label: 'Load Fable', elapsedMs: 4200}), '◐ Load Fable · 4.2 s');
assert.equal(p.stepLine({state: 'done', label: 'Put away QUASAR', ms: 5100}), '✓ Put away QUASAR · 5.1 s');
assert.equal(p.stepLine({state: 'waiting', label: 'Load Fable', guessS: 20}), '○ Load Fable · about 20 s');
assert.equal(p.stepLine({state: 'skipped', label: 'Answer with setup B', guessS: 30}), '– Answer with setup B');
assert.equal(p.stepLine({state: 'failed', label: 'Load Fable', ms: 900, detail: 'Loading Fable failed: no memory'}),
  '✗ Load Fable · 900 ms — Loading Fable failed: no memory');
const lines = p.planLines({steps: [{label: 'Put away QUASAR', guessS: 5}, {label: 'Load Qwen', guessS: 150}],
  estimateS: 155, warnings: ['QUASAR was used 40 seconds ago. The test will put it away.']});
assert.deepEqual(lines.steps, ['Put away QUASAR · about 5 s', 'Load Qwen · about 2.5 min']);
assert.equal(lines.total, 'About 2.5 min in all.');
assert.equal(lines.warnings.length, 1);
assert.equal(p.statusWords({status: 'yielded', message: 'Another app asked for a different model, so the test stopped to let it through.'}),
  'Stopped to let another app through. Another app asked for a different model, so the test stopped to let it through.');
assert.equal(p.statusWords({status: 'stopped', message: 'Stopped.'}), 'Stopped.');
assert.equal(p.statusWords({status: 'done', message: ''}), 'Done.');
assert.equal(p.statusWords(null), '');
""")


def test_history_keeps_twenty_newest_and_survives_a_full_store():
    _node(r"""
let list = [];
for (let i = 0; i < 25; i++) list = p.historyAdd(list, {id: 'j' + i});
assert.equal(list.length, 20); assert.equal(list[0].id, 'j24'); assert.equal(list[19].id, 'j5');
assert.equal(p.historyAdd(list, {id: 'j24', x: 1}).filter((r) => r.id === 'j24').length, 1);
const store = { data: {}, setItem(k, v) { if (JSON.parse(v).length > 2) throw new Error('QuotaExceededError'); this.data[k] = v; },
  getItem(k) { return this.data[k] ?? null; }, removeItem(k) { delete this.data[k]; } };
assert.equal(p.saveHistory(store, list.slice(0, 5)).length, 2);
assert.deepEqual(p.loadHistory(store).map((r) => r.id), ['j24', 'j23']);
assert.deepEqual(p.saveHistory(store, []), []);
assert.equal(store.data[p.PG_HISTORY_KEY], undefined);
assert.deepEqual(p.loadHistory({ getItem() { return '{broken'; } }), []);
assert.deepEqual(p.loadHistory({ getItem() { throw new Error('SecurityError'); } }), []);
const rec = p.historyRecord({id: 'x', startedAt: '2026-09-25T10:00:00Z', prompt: 'Hi', status: 'done', steps: [{}],
  sides: [{key: 'A', model: 'm', name: 'M', answer: 'a'.repeat(30000), stats: null}]});
assert.equal(rec.sides[0].answer.length, 20000); assert.equal(rec.steps, undefined); assert.equal(p.PG_HISTORY_MAX, 20);
""")


def test_markdown_copy_has_prompt_table_and_answers():
    _node(r"""
const md = p.historyMarkdown({at: '2026-09-25T10:00:00Z', prompt: 'Line one\nLine two',
  restore: 'QUASAR is loaded again on its saved settings (21.0 s).', sides: [
  {key: 'A', name: 'Fable', preset: 'Fast', settingsLabel: 'preset “Fast”', loadMs: 20000, answer: 'Answer A',
   stats: {firstWordMs: 400, writeTps: 41.3, writeSource: 'engine', totalMs: 12000, promptTokens: 30, completionTokens: 500,
           guesses: {proposed: 514, kept: 365, keptPct: 71}, finishReason: 'stop'}},
  {key: 'B', name: 'Fable', preset: null, settingsLabel: 'Saved settings', loadMs: null, stats: null, answer: ''}]});
assert.match(md, /^## Test [^\n]*Sep[^\n]*\n\n> Line one\n> Line two\n/);
assert.match(md, /\| \| A · Fable · preset “Fast” \| B · Fable · Saved settings \|/);
assert.match(md, /\| Writing speed \| 41\.3 tokens a second \| — \|/);
assert.match(md, /\| Guesses kept \| 71% \(365 of 514 guesses\) \| not reported by this engine \|/);
assert.match(md, /### A · Fable · preset “Fast”\n\nAnswer A\n/);
assert.match(md, /### B · Fable · Saved settings\n\n\(no answer\)\n/);
assert.match(md, /_QUASAR is loaded again on its saved settings \(21\.0 s\)\._\n$/);
assert.doesNotMatch(md, /\n\n\n/);
""")


def test_page_has_the_test_tab_and_every_id_the_script_uses():
    page = PAGE.read_text(encoding="utf-8")
    for needle in ('data-main="test"', 'id="test-view"', '<script src="/playground.js"></script>',
                   'id="pg-prompt"', 'id="pg-system"', 'id="pg-a-model"', 'id="pg-a-preset"', 'id="pg-b-on"',
                   'id="pg-b-fields"', 'id="pg-warmup"', 'id="pg-putback"', 'id="pg-run"', 'id="pg-start"',
                   'id="pg-plan-cancel"', 'id="pg-stop"', 'id="pg-steps"', 'id="pg-results"', 'id="pg-restore"',
                   'id="pg-history"', 'id="pg-export"', 'id="pg-clear"'):
        assert needle in page, needle
    assert page.index('<script src="/panel.js">') < page.index('<script src="/playground.js">')
    script = PG_JS.read_text(encoding="utf-8")
    for element in set(re.findall(r"\$\('(pg-[\w-]+)'\)", script)):
        assert f'id="{element}"' in page, element
    for side in ("a", "b"):
        for field in ("model", "preset", "temperature", "top_p", "top_k", "maxTokens"):
            assert f'id="pg-{side}-{field}"' in page, (side, field)


def test_show_main_knows_the_test_tab():
    text = PANEL_JS.read_text(encoding="utf-8")
    assert "'freetoken', 'test']" in text and "pgOpen()" in text


def test_right_now_strip_shows_a_running_test_and_a_leftover():
    _node(r"""
const base = {switcher: {up: true, running: [{id: 'quasar-27b', name: 'QUASAR', state: 'ready'}]}, card: null, held: []};
const running = p.nowStripHtml({...base, test: {running: true, model: 'fable-27b', name: 'Fable', preset: 'Three'}});
assert.match(running, /Test running: Fable on preset “Three”/);
assert.match(p.nowStripHtml({...base, test: {running: true, model: null}}), /A test is running on the Test tab\./);
const left = p.nowStripHtml({...base, testLeftover: {model: 'quasar-27b', name: 'QUASAR', preset: 'Fast'}});
assert.match(left, /QUASAR is still on test settings \(preset “Fast”\) because an app was using it\. It goes back to its saved settings at its next load\./);
assert.doesNotMatch(p.nowStripHtml(base), /test/i);
""", module=PANEL_JS)


def test_history_times_read_as_local_time_not_iso():
    _node(r"""
const when = p.pgWhen('2026-09-25T09:06:06Z');
assert.ok(!when.includes('T09:06') && !when.endsWith('Z'), when);
assert.ok(/Sep/.test(when), when);
assert.equal(p.pgWhen('not a date'), 'not a date');
""")
