'use strict';
/* The Test tab (docs/superpowers/specs/2026-09-25-playground-design.md). The helper runs the
   test (switch, answer, time, put back); this page plans it, starts it, polls it every 0.5 s
   and keeps the last 20 tests in this browser. The helpers above the browser section are pure
   and exported for node. $, json, setNotice and askConfirm come from index.html and panel.js. */

const PG_POLL_MS = 500;
const PG_HISTORY_KEY = 'ft-test-history-v1';
const PG_HISTORY_MAX = 20;
const PG_ANSWER_KEEP = 20000;
const PG_ACTIVE = ['running', 'stopping', 'restoring'];
const pg = { options: null, job: null, timer: null, plan: null, history: [], shown: null, wired: false };

function pgEsc(value) { return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function fmtMs(ms) {
  const n = Number(ms);
  if (ms == null || ms === '' || !Number.isFinite(n)) return '—';
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60000) return `${(n / 1000).toFixed(1)} s`;
  const total = Math.round(n / 1000);
  return `${Math.floor(total / 60)} min ${total % 60} s`;
}
function fmtGuess(seconds) {
  const s = Number(seconds) || 0;
  return s < 60 ? `${Math.round(s)} s` : `${Math.round((s / 60) * 2) / 2} min`;
}
function fmtRate(stats) {
  if (!stats || stats.writeTps == null) return '—';
  const value = Number(stats.writeTps).toFixed(1);
  return stats.writeSource === 'engine' ? `${value} tokens a second` : `~${value} tokens a second`;
}
function guessWords(stats) {
  const g = stats && stats.guesses;
  return g ? `${g.keptPct}% (${g.kept} of ${g.proposed} guesses)` : 'not reported by this engine';
}
const PG_STOP_WORDS = { stop: 'finished', length: 'hit the longest-answer limit', tool_calls: 'wanted to use a tool', cancelled: 'stopped' };
function stopWords(reason) { return reason ? (PG_STOP_WORDS[reason] || String(reason)) : '—'; }
function tokensWords(n, approx, reused) {
  if (n == null) return '—';
  return `${approx ? '~' : ''}${Number(n).toLocaleString('en-GB')} tokens${reused ? ` (${Number(reused).toLocaleString('en-GB')} reused)` : ''}`;
}

// [key, label, lower is better (true), higher is better (false), never judged (null)]
const PG_METRICS = [
  ['loadMs', 'Loading (not counted)', null],
  ['firstWordMs', 'First word after', true],
  ['writeTps', 'Writing speed', false],
  ['totalMs', 'Whole answer', true],
  ['promptTokens', 'Prompt size', null],
  ['completionTokens', 'Answer size', null],
  ['guessPct', 'Guesses kept', false],
  ['finishReason', 'Stopped because', null],
];
function metricValue(side, key) {
  const s = (side && side.stats) || {};
  if (key === 'loadMs') return side ? side.loadMs : null;
  if (key === 'guessPct') return s.guesses ? s.guesses.keptPct : null;
  return s[key];
}
function metricText(side, key) {
  const s = (side && side.stats) || {};
  switch (key) {
    case 'loadMs': return side && side.loadMs != null ? fmtMs(side.loadMs) : 'already loaded';
    case 'firstWordMs': case 'totalMs': return fmtMs(s[key]);
    case 'writeTps': return fmtRate(s);
    case 'promptTokens': return tokensWords(s.promptTokens, false, s.cachedTokens);
    case 'completionTokens': return tokensWords(s.completionTokens, s.approxTokens);
    case 'guessPct': return guessWords(s);
    default: return stopWords(s.finishReason);
  }
}
function betterSide(a, b, key) {
  const metric = PG_METRICS.find((row) => row[0] === key);
  if (!metric || metric[2] == null || !a || !b) return null;
  const rawA = metricValue(a, key); const rawB = metricValue(b, key);
  if (rawA == null || rawB == null) return null;
  const x = Number(rawA); const y = Number(rawB);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  const top = Math.max(Math.abs(x), Math.abs(y));
  if (top === 0 || Math.abs(x - y) / top < 0.03) return null;
  return (metric[2] ? x < y : x > y) ? a.key : b.key;
}
function speedRows(side, other) {
  return PG_METRICS.map(([key, label]) => ({ key, label, text: metricText(side, key), better: !!other && betterSide(side, other, key) === side.key }));
}

const PG_STEP_ICON = { waiting: '○', running: '◐', done: '✓', failed: '✗', skipped: '–' };
function stepLine(step) {
  let time = '';
  if (step.state === 'running') time = fmtMs(step.elapsedMs || 0);
  else if (step.ms != null) time = fmtMs(step.ms);
  else if (step.state === 'waiting' && step.guessS) time = `about ${fmtGuess(step.guessS)}`;
  return `${PG_STEP_ICON[step.state] || '○'} ${step.label}${time ? ` · ${time}` : ''}${step.detail ? ` — ${step.detail}` : ''}`;
}
function planLines(plan) {
  return {
    steps: (plan.steps || []).map((step) => `${step.label}${step.guessS ? ` · about ${fmtGuess(step.guessS)}` : ''}`),
    total: `About ${fmtGuess(plan.estimateS || 0)} in all.`,
    warnings: plan.warnings || [],
  };
}
const PG_STATUS_WORDS = { running: 'Running…', stopping: 'Stopping…', restoring: 'Putting things back…', done: 'Done.',
  failed: "The test didn't finish.", stopped: 'Stopped.', yielded: 'Stopped to let another app through.' };
function statusWords(job) {
  if (!job) return '';
  const base = PG_STATUS_WORDS[job.status] || String(job.status);
  return job.message && job.message !== base ? `${base} ${job.message}` : base;
}

function historyRecord(job) {
  const cut = (text) => String(text || '').slice(0, PG_ANSWER_KEEP);
  return { id: job.id, at: job.startedAt, prompt: job.prompt, system: job.system || '', status: job.status,
    message: job.message || '', restore: job.restore || '',
    sides: (job.sides || []).map((s) => ({ key: s.key, model: s.model, name: s.name, preset: s.preset || null,
      settingsLabel: s.settingsLabel, sampling: s.sampling, loadMs: s.loadMs, stats: s.stats, error: s.error || null,
      answer: cut(s.answer), reasoning: cut(s.reasoning) })) };
}
function historyAdd(list, record, max = PG_HISTORY_MAX) { return [record, ...(list || []).filter((row) => row.id !== record.id)].slice(0, max); }
function sideTitle(side) { return `${side.key} · ${side.name} · ${side.settingsLabel || (side.preset ? `preset “${side.preset}”` : 'Saved settings')}`; }
function historyMarkdown(record) {
  const sides = record.sides || [];
  const lines = [`## Test ${record.at || ''}`.trim(), '', ...String(record.prompt || '').split('\n').map((line) => `> ${line}`), ''];
  lines.push(`| | ${sides.map(sideTitle).join(' | ')} |`, `|---|${sides.map(() => '---').join('|')}|`);
  for (const [key, label] of PG_METRICS) lines.push(`| ${label} | ${sides.map((s) => metricText(s, key)).join(' | ')} |`);
  for (const s of sides) lines.push('', `### ${sideTitle(s)}`, '', s.error ? `(${s.error})` : '', String(s.answer || '').trim() || '(no answer)');
  if (record.restore) lines.push('', `_${record.restore}_`);
  return `${lines.filter((line, i, all) => !(line === '' && all[i - 1] === '')).join('\n')}\n`;
}
function loadHistory(storage) {
  try { const value = JSON.parse(storage.getItem(PG_HISTORY_KEY) || '[]'); return Array.isArray(value) ? value : []; } catch (_) { return []; }
}
function saveHistory(storage, list) {
  let items = (list || []).slice();
  while (items.length) {
    try { storage.setItem(PG_HISTORY_KEY, JSON.stringify(items)); return items; } catch (_) { items = items.slice(0, -1); }
  }
  try { storage.removeItem(PG_HISTORY_KEY); } catch (_) {}
  return [];
}

/* ---------- browser ---------- */
function pgStore() { try { return window.localStorage; } catch (_) { return null; } }
function pgShowError(text) { const box = $('pg-error'); box.textContent = text || ''; box.hidden = !text; }
function pgPost(url, body) { return json(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body || {}) }); }
function pgFillPresets(side) {
  const model = ((pg.options && pg.options.models) || []).find((m) => m.id === $(`pg-${side}-model`).value);
  const select = $(`pg-${side}-preset`);
  const keep = select.value;
  const presets = (model && model.presets) || [];
  select.innerHTML = `<option value="">Saved settings</option>${presets.map((name) => `<option value="${pgEsc(name)}">Preset “${pgEsc(name)}”</option>`).join('')}`;
  select.value = presets.includes(keep) ? keep : '';
}
function pgFillModels() {
  const models = (pg.options && pg.options.models) || [];
  const loaded = models.find((m) => m.state === 'ready');
  for (const side of ['a', 'b']) {
    const select = $(`pg-${side}-model`);
    const keep = select.value;
    select.innerHTML = models.map((m) => `<option value="${pgEsc(m.id)}">${pgEsc(m.name)}${m.state === 'ready' ? ' (loaded)' : ''}</option>`).join('');
    select.value = models.some((m) => m.id === keep) ? keep : ((loaded || models[0] || {}).id || '');
    pgFillPresets(side);
  }
}
function pgSide(side) {
  const value = (field) => $(`pg-${side}-${field}`).value.trim();
  return { model: value('model'), preset: value('preset') || null, temperature: value('temperature'), top_p: value('top_p'),
    top_k: value('top_k'), maxTokens: value('maxTokens') || 512 };
}
function pgRequest() {
  const sides = [pgSide('a')];
  if ($('pg-b-on').checked) sides.push(pgSide('b'));
  return { prompt: $('pg-prompt').value, system: $('pg-system').value, sides, warmup: $('pg-warmup').checked, putBack: $('pg-putback').checked };
}
async function pgLoadOptions() {
  const { response, body } = await json('/api/playground/options');
  if (!response.ok) { pgShowError(panelErrorText(body, "Couldn't read the model list.")); return; }
  pg.options = body;
  pgFillModels();
}
function pgRenderPlan() {
  const box = $('pg-plan');
  if (!pg.plan) { box.hidden = true; return; }
  const lines = planLines(pg.plan);
  $('pg-plan-steps').innerHTML = lines.steps.map((line) => `<li>${pgEsc(line)}</li>`).join('');
  $('pg-plan-total').textContent = lines.total;
  $('pg-plan-warnings').innerHTML = lines.warnings.map((w) => `<p class="hint warn">${pgEsc(w)}</p>`).join('');
  box.hidden = false;
}
async function pgRun() {
  pgShowError('');
  const { response, body } = await pgPost('/api/playground/plan', pgRequest());
  if (!response.ok) { pg.plan = null; pgRenderPlan(); pgShowError(panelErrorText(body, "Couldn't plan the test.")); return; }
  pg.plan = body;
  pgRenderPlan();
}
async function pgStart() {
  if (!pg.plan) return;
  const { response, body } = await pgPost('/api/playground/runs', { ...pgRequest(), expectBefore: pg.plan.before, confirm: true });
  if (!response.ok) {
    if (body && body.plan) { pg.plan = body.plan; pgRenderPlan(); }
    pgShowError(panelErrorText(body, "Couldn't start the test."));
    return;
  }
  pg.plan = null; pgRenderPlan(); pg.shown = null; pg.job = body; pgRenderJob(); pgSchedule();
}
async function pgStop() { const { body } = await pgPost('/api/playground/runs/current/stop'); if (body && body.status) { pg.job = body; pgRenderJob(); } }
function pgSchedule() { clearTimeout(pg.timer); pg.timer = setTimeout(pgPoll, PG_POLL_MS); }
async function pgPoll() {
  clearTimeout(pg.timer);
  let result;
  try { result = await json('/api/playground/runs/current'); } catch (_) { pgSchedule(); return; }
  if (!result.response.ok) { pgSchedule(); return; }
  pg.job = result.body.status === 'idle' ? null : result.body;
  pgRenderJob();
  if (pg.job && PG_ACTIVE.includes(pg.job.status)) pgSchedule();
  else if (pg.job) { pgRemember(pg.job); if (pg.options) pgLoadOptions(); }
}
function pgRemember(job) {
  if (!job || !job.id || PG_ACTIVE.includes(job.status) || pg.history.some((row) => row.id === job.id)) return;
  const next = historyAdd(pg.history, historyRecord(job));
  const store = pgStore();
  pg.history = store ? saveHistory(store, next) : next;
  pgRenderHistory();
}
function pgResultsHtml(record) {
  const sides = record.sides || [];
  return sides.map((side) => {
    const other = sides.find((s) => s.key !== side.key);
    const rows = speedRows(side, other).map((row) => `<tr><th scope="row">${pgEsc(row.label)}</th><td>${pgEsc(row.text)}${row.better ? ' <span class="pg-better">better</span>' : ''}</td></tr>`).join('');
    const thinking = side.reasoning ? `<details class="pg-thinking"><summary>Thinking</summary><div class="pg-answer">${pgEsc(side.reasoning)}</div></details>` : '';
    const error = side.error ? `<p class="error">${pgEsc(side.error)}</p>` : '';
    const answer = side.answer ? pgEsc(side.answer) : '<span class="muted">(no answer yet)</span>';
    return `<article class="pg-result" data-side="${pgEsc(side.key)}"><h3>${pgEsc(sideTitle(side))}</h3><table class="pg-speed">${rows}</table>${error}${thinking}<div class="pg-answer">${answer}</div></article>`;
  }).join('');
}
function pgRenderJob() {
  const record = pg.shown || pg.job;
  const active = !!(pg.job && PG_ACTIVE.includes(pg.job.status));
  $('pg-run').disabled = active;
  $('pg-stop').hidden = !(active && pg.job.status === 'running');
  $('pg-steps').innerHTML = record && record.steps ? record.steps.map((step) => `<li class="pg-step ${pgEsc(step.state)}">${pgEsc(stepLine(step))}</li>`).join('') : '';
  $('pg-status').textContent = pg.shown ? `Earlier test from ${pg.shown.at || ''}` : statusWords(pg.job);
  $('pg-results').innerHTML = record ? pgResultsHtml(record) : '';
  $('pg-restore').textContent = record ? (record.restore || '') : '';
}
function pgRenderHistory() {
  const list = pg.history;
  $('pg-history').innerHTML = list.length ? list.map((row, i) => `<div class="pg-history-row"><div><strong>${pgEsc(row.at || '')}</strong> <span class="small">${pgEsc(String(row.prompt || '').slice(0, 80))}</span><div class="small">${(row.sides || []).map((s) => `${pgEsc(s.key)}: ${pgEsc(s.name)}, ${pgEsc(fmtRate(s.stats))}`).join(' · ')}</div></div><div class="actions"><button class="button small" type="button" data-pg-show="${i}">Show</button><button class="button small" type="button" data-pg-copy="${i}">Copy</button></div></div>`).join('') : '<p class="empty">No tests yet.</p>';
}
async function pgCopy(i) {
  const record = pg.history[i];
  if (!record) return;
  try { await navigator.clipboard.writeText(historyMarkdown(record)); setNotice('Copied.', 'good'); } catch (_) { setNotice("Couldn't copy: the browser blocked it.", 'warn'); }
}
function pgExport() {
  const blob = new Blob([JSON.stringify(pg.history, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `freetoken-tests-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
async function pgClear() {
  if (!(await askConfirm('Clear every earlier test from this browser?', 'Clear'))) return;
  const store = pgStore();
  pg.history = store ? saveHistory(store, []) : [];
  pg.shown = null;
  pgRenderHistory(); pgRenderJob();
}
function pgWire() {
  if (pg.wired) return;
  pg.wired = true;
  $('pg-a-model').addEventListener('change', () => pgFillPresets('a'));
  $('pg-b-model').addEventListener('change', () => pgFillPresets('b'));
  $('pg-b-on').addEventListener('change', () => { $('pg-b-fields').disabled = !$('pg-b-on').checked; });
  $('pg-run').addEventListener('click', pgRun);
  $('pg-start').addEventListener('click', pgStart);
  $('pg-plan-cancel').addEventListener('click', () => { pg.plan = null; pgRenderPlan(); });
  $('pg-stop').addEventListener('click', pgStop);
  $('pg-export').addEventListener('click', pgExport);
  $('pg-clear').addEventListener('click', pgClear);
  $('pg-history').addEventListener('click', (event) => {
    const show = event.target.closest('[data-pg-show]');
    const copy = event.target.closest('[data-pg-copy]');
    if (show) { pg.shown = pg.history[Number(show.dataset.pgShow)] || null; pgRenderJob(); }
    if (copy) pgCopy(Number(copy.dataset.pgCopy));
  });
}
async function pgOpen() {
  pgWire();
  const store = pgStore();
  pg.history = store ? loadHistory(store) : [];
  pgRenderHistory();
  await pgLoadOptions();
  await pgPoll();
}

if (typeof module !== 'undefined') module.exports = { fmtMs, fmtGuess, fmtRate, guessWords, stopWords, tokensWords, metricText, betterSide,
  speedRows, stepLine, planLines, statusWords, historyRecord, historyAdd, historyMarkdown, sideTitle, loadHistory, saveHistory,
  PG_HISTORY_KEY, PG_HISTORY_MAX, pg };
