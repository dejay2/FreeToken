'use strict';
/* Control panel, Stage A (docs/superpowers/specs/2026-09-24-control-panel-part2-design.md).
   The Models list, the "Right now" strip, the System / NInfer defaults / FreeToken defaults
   tabs and each model's settings. Dials are drawn by index.html's renderer: a view fills
   state.dials, state.groups, state.settings and state.saved, sets state.view and calls
   renderSettings(). The helpers above the browser section are pure and exported for node.
   Stage B (spec section 7): the Add a model wizard and the Remove question. */

const PANEL_NOW_MS = 5000;
const PANEL_GB = 1024 ** 3;
const panel = { main: 'models', revision: '', models: [], now: null, busy: false, fit: null, fitDraft: '', nowTimer: null, nowLoop: false, restartResolve: null, confirmResolve: null, presetMode: 'add', removeResolve: null };

function panelEsc(value) { return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function fmtGB(bytes) { const n = Number(bytes); if (bytes == null || bytes === '' || !Number.isFinite(n)) return '—'; return `${(n / PANEL_GB).toFixed(1)} GB`; }
const STATE_WORDS = { ready: 'Loaded', starting: 'Loading…', stopping: 'Unloading…', stopped: 'Not loaded', shutdown: 'Not loaded', unknown: 'Switcher not running' };
function stateWord(value) { return STATE_WORDS[value] || String(value || 'unknown'); }
function sourceText(source) {
  if (!source) return '';
  if (source.from === 'model') return 'changed for this model';
  if (source.from === 'preset') return `from preset “${source.preset}”`;
  return `from ${source.engineLabel} defaults`;
}
function sameValue(a, b) { return JSON.stringify(a ?? '') === JSON.stringify(b ?? ''); }
function dialSourceFor(view, settings, name) {
  if (!view || view.kind !== 'model' || String(name).startsWith('model.') || !view.base || !(name in view.base)) return null;
  if (!sameValue(settings[name], view.base[name])) return { from: 'model' };
  return view.baseFrom && view.baseFrom[name] === 'preset' ? { from: 'preset', preset: view.activePreset, engineLabel: view.engineLabel } : { from: 'default', engineLabel: view.engineLabel };
}
function verdictWords(verdict) { return ({ fits: 'Fits', tight: 'Tight', wont_fit: "Won't fit" })[verdict] || "Couldn't check"; }
function fitSummary(fit) {
  if (!fit) return '';
  if (fit.needBytes == null || !fit.cardTotalBytes) return `${verdictWords(fit.verdict)}: ${fit.message || 'no estimate'}`;
  // NInfer's startup check (fit round 2) can make a model tight or won't-fit while the memory it
  // uses once running looks fine; its message carries the numbers behind that verdict, so show it
  // instead of the used-memory line, which would contradict the verdict.
  if (fit.message && (fit.verdict === 'tight' || fit.verdict === 'wont_fit')) return `${verdictWords(fit.verdict)}: ${fit.message}`;
  return `${verdictWords(fit.verdict)}: needs about ${fmtGB(fit.needBytes)} of the graphics card's ${fmtGB(fit.cardTotalBytes)}.`;
}
function ramSummary(ram) {
  if (!ram) return '';
  const need = `${Number(ram.needGB)} GB`;
  if (ram.loadedNow) return `PC memory: this model is loaded now, so its ${need} is already in use.`;
  if (ram.windowsFreeGB == null) return `PC memory: needs ${need}; Windows free memory could not be read.`;
  const short = Number(ram.windowsFreeGB) - Number(ram.needGB) < Number(ram.cushionGB);
  return `PC memory: needs ${need}; Windows has ${Number(ram.windowsFreeGB).toFixed(1)} GB free and keeps a ${ram.cushionGB} GB cushion.${short ? ' The switcher will wait for memory before loading.' : ''}`;
}
function restartQuestion(affected, nextTimeAllowed) {
  const rows = affected || [];
  const names = rows.map((row) => row.name || row.id).join(', ');
  const one = rows.length === 1;
  if (!nextTimeAllowed) return `${names} ${one ? 'is' : 'are'} loaded right now and must restart for this. Restart now?`;
  return `${names} ${one ? 'is' : 'are'} loaded right now. Restart now to use the new settings, or keep ${one ? 'it' : 'them'} running on the old ones until ${one ? 'its' : 'their'} next load?`;
}
// Load/Unload questions (final review item 9). Load asks only when another model is loaded,
// because the switcher puts that one away first; Unload always asks.
function isLoadedState(value) { return value === 'ready' || value === 'starting'; }
function loadQuestion(id, rows) {
  const list = rows || [];
  const target = list.find((row) => row.id === id) || { id };
  const others = list.filter((row) => row.id !== id && isLoadedState(row.state));
  if (!others.length) return null;
  return `This will put away ${others.map((row) => row.name || row.id).join(', ')} and load ${target.name || target.id}. Carry on?`;
}
function unloadQuestion(id, rows) {
  const target = (rows || []).find((row) => row.id === id) || { id };
  return `Put away ${target.name || target.id}? Anything using it will stop.`;
}
const STALE_WORDS = "The switcher hasn't picked up the latest settings yet.";
function nowStripHtml(now) {
  if (!now) return '<p class="empty">Checking…</p>';
  const sw = now.switcher || {};
  const running = !sw.up ? 'Model switcher not running' : ((sw.running || []).length ? sw.running.map((row) => `${panelEsc(row.name || row.id)} <span class="small">${panelEsc(stateWord(row.state))}</span>`).join('<br>') : 'Nothing loaded');
  const card = now.card ? `${fmtGB(now.card.usedBytes)} <span class="muted">/ ${fmtGB(now.card.totalBytes)}</span>` : '—';
  const pct = now.card && now.card.totalBytes ? Math.round(Math.min(100, (now.card.usedBytes / now.card.totalBytes) * 100)) : 0;
  const win = now.windowsFreeBytes == null ? '—' : fmtGB(now.windowsFreeBytes);
  const cushion = now.cushionGB == null ? '—' : `${now.cushionGB} GB`;
  const held = (now.held || []).length ? `<div class="sub">Old settings until the next load: ${panelEsc(now.held.map((id) => ((sw.running || []).find((row) => row.id === id) || {}).name || id).join(', '))}</div>` : '';
  const restart = now.lastRestart && !now.lastRestart.ok ? `<div class="error">${panelEsc(now.lastRestart.message)}</div>` : '';
  const stale = sw.up && sw.stale ? `<div class="error">${STALE_WORDS}</div>` : '';
  const test = now.test && now.test.running
    ? `<div class="sub">${now.test.model ? `Test running: ${panelEsc(now.test.name || now.test.model)} on preset “${panelEsc(now.test.preset)}”` : 'A test is running on the Test tab.'}</div>` : '';
  const leftover = now.testLeftover
    ? `<div class="error">${panelEsc(now.testLeftover.name || now.testLeftover.model)} is still on test settings${now.testLeftover.preset ? ` (preset “${panelEsc(now.testLeftover.preset)}”)` : ''} because an app was using it. It goes back to its saved settings at its next load.</div>` : '';
  return [
    `<div class="stat"><span class="small">Loaded</span><strong class="now-list">${running}</strong>${held}${test}${leftover}${stale}${restart}</div>`,
    `<div class="stat"><span class="small">Graphics card</span><strong>${card}</strong><div class="bar" aria-label="Graphics card memory used"><span style="width:${pct}%"></span></div></div>`,
    `<div class="stat"><span class="small">Windows free memory</span><strong>${win}</strong></div>`,
    `<div class="stat"><span class="small">Cushion kept free</span><strong>${cushion}</strong></div>`,
  ].join('');
}
function modelsTableHtml(rows, switcherUp) {
  if (!rows || !rows.length) return '<p class="empty">No models yet.</p>';
  const body = rows.map((row) => {
    const loaded = row.state === 'ready' || row.state === 'starting';
    const engine = row.runtimeLabel ? `${panelEsc(row.engineLabel)} <span class="small">(${panelEsc(row.runtimeLabel)})</span>` : panelEsc(row.engineLabel);
    const idle = row.idleMinutes ? `${panelEsc(row.idleMinutes)} min` : 'never';
    const action = !switcherUp ? '' : loaded ? `<button class="button small danger" type="button" data-unload="${panelEsc(row.id)}">Unload</button>` : `<button class="button small primary" type="button" data-load="${panelEsc(row.id)}">Load</button>`;
    return `<tr><td><span class="dot ${panelEsc(row.state)}"></span> ${panelEsc(stateWord(row.state))}${row.held ? '<div class="small">old settings until the next load</div>' : ''}</td><td><strong>${panelEsc(row.name)}</strong><div class="small">${panelEsc(row.id)}</div></td><td data-label="Engine">${engine}</td><td data-label="Preset">${panelEsc(row.activePreset || '—')}</td><td data-label="PC memory">${panelEsc(row.ramNeedGB)} GB</td><td data-label="Unload when idle">${idle}${row.idleFromSystem ? ' <span class="small">(System)</span>' : ''}</td><td class="row-actions"><div class="actions">${action}<button class="button small" type="button" data-settings="${panelEsc(row.id)}">Settings</button></div></td></tr>`;
  }).join('');
  return `<table class="models-table"><thead><tr><th>Status</th><th>Model</th><th>Engine</th><th>Preset</th><th>PC memory</th><th>Unload when idle</th><th></th></tr></thead><tbody>${body}</tbody></table>`;
}
// The server answers with {message} (plain words), {detail: "..."} or {detail: [{field, message}]}
// (422). Pick the plain words; never show "[object Object]" or a bare status code.
function panelErrorText(body, fallback) {
  if (body && typeof body.message === 'string' && body.message) return body.message;
  const detail = body && body.detail;
  if (Array.isArray(detail)) { const words = detail.map((row) => row && (row.message || row.msg)).filter(Boolean); if (words.length) return words.join(' '); }
  if (typeof detail === 'string' && detail && !detail.startsWith('not found')) return detail;
  return fallback;
}
// Backups are named registry.json.bak-YYYYmmdd-HHMMSS-ffffff; show the time, not the file name.
function backupWhen(name) {
  const m = String(name).match(/bak-(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d)(\d\d)/);
  if (!m) return name;
  return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4]), Number(m[5]), Number(m[6])).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
function backupListHtml(backups) {
  return (backups || []).map((name) => `<li><button class="button small" type="button" data-restore="${panelEsc(name)}">Restore the copy from ${panelEsc(backupWhen(name))}</button></li>`).join('');
}
// The fix-it box on Models. Missing or damaged, the backups are offered (spec: Error handling);
// a missing list also offers a fresh copy of today's settings (final review item 3).
function registryProblemHtml(body) {
  const backups = backupListHtml(body.backups);
  if (body.status === 'missing') {
    const restore = backups ? `<p class="small">Or bring back a saved copy of the panel's own list:</p><ul class="backups">${backups}</ul>` : '';
    return `<div class="panel-head"><div><h2>Set up the control panel</h2><p class="small">This copies today's models and settings from ${panelEsc(body.configPath)} and the helper's start-up file into the panel's own list. The old switcher file is kept as a backup.</p></div><button class="button primary" id="import-now" type="button">Copy today's settings</button></div>${restore}`;
  }
  return `<div class="panel-head"><div><h2>The model list is damaged</h2><p class="small">${panelEsc(body.message || '')}</p><p class="small">Nothing was changed. Restore the last good backup:</p></div></div>${backups ? `<ul class="backups">${backups}</ul>` : '<p class="empty">No backups were found.</p>'}`;
}
// ---- Stage B: add and remove (spec section 7) ----
const ADD_ID_RE = /^[a-z0-9][a-z0-9._-]{0,62}$/; // registry.MODEL_ID_RE (a test keeps them equal)
const ADD_ID_RULE = 'Use 1 to 63 small letters, numbers, dots, dashes or underscores, starting with a letter or number.';
// The stage names are download.py's DownloadJob.stage values.
const ADD_STAGE_WORDS = { queued: 'Waiting to start', downloading: 'Downloading', verifying: 'Checking the download', moving: 'Putting it in place', done: 'Downloaded', failed: 'Download failed', cancelled: 'Download cancelled' };
const ADD_TERMINAL = new Set(['done', 'failed', 'cancelled']);
const ADD_POLL_MS = 1500;
// pollGen numbers the poll loop: only the newest loop's answers are shown (review item 4).
// checkSeq / planSeq number the detect and plan requests the same way (item 6). seenJob is the
// finished download already shown once on opening (item 3). cancelling holds "Cancelling…" until
// a poll says cancelled (item 5). saving keeps Save off while the POST runs (item 6).
const addState = { found: null, plan: null, job: null, timer: null, roots: {}, pollGen: 0, checkSeq: 0, planSeq: 0, seenJob: null, cancelling: false, saving: false };
const ADD_LOST_TOUCH = 'Lost touch with the settings page, trying again…';
const ADD_KEEPS_GOING = 'The download keeps going. Open Add a model to see it.';
function idProblem(id, rows) {
  const text = String(id ?? '').trim();
  if (!ADD_ID_RE.test(text)) return ADD_ID_RULE;
  const wanted = text.toLowerCase();
  const owner = (rows || []).find((row) => String(row.id).toLowerCase() === wanted || (row.aliases || []).some((alias) => String(alias).toLowerCase() === wanted));
  return owner ? `That id is already used by ${owner.name || owner.id}.` : '';
}
// /api/panel/add/detect: {kind, format, engineLabel, runtimeLabel, bytes, reason, already}.
function detectionText(found) {
  if (!found) return '';
  if (found.kind === 'unsupported') return found.reason || 'Not supported by your engines.';
  const engine = found.runtimeLabel ? `${found.engineLabel} (${found.runtimeLabel})` : found.engineLabel;
  const already = found.already ? ` It is already in the list as ${found.already}.` : '';
  return `This is a ${found.format || 'model'} for ${engine}, ${fmtGB(found.bytes)}.${already}`;
}
// /api/panel/add/plan: {kind: 'ninfer'|'folder', entry, entries, name, files: [{name, bytes, check}],
// totalBytes, target, exists, diskFits, diskFreeBytes}. check is "SHA256SUMS", "published" or null.
function planSummary(plan) {
  if (!plan) return '';
  if (plan.kind === 'ninfer' && !plan.entry) return `This repo has ${(plan.entries || []).length} NInfer files. Pick one.`;
  const what = plan.kind === 'ninfer' ? `NInfer file ${plan.entry}` : `model folder ${plan.name}`;
  const files = plan.files || [];
  const checked = files.filter((file) => file.check).length;
  const sums = !checked ? ' The repo publishes no checksums, so the files cannot be checked.' : checked === files.length ? ` All ${files.length} files will be checked against the checksums the repo publishes.` : ` ${checked} of ${files.length} files will be checked against the checksums the repo publishes.`;
  const space = plan.diskFits === false ? ` Not enough drive space: ${fmtGB(plan.diskFreeBytes)} free.` : '';
  const exists = plan.exists ? ' It is already on this PC, so use “On this PC” to add it.' : '';
  return `Downloads the ${what} (${fmtGB(plan.totalBytes)}) into ${plan.target}.${sums}${space}${exists}`;
}
// A download job: {id, stage, percent, receivedBytes, totalBytes, error, verified, resultPath}.
// Server errors end with a full stop, so the failed line reads as two sentences.
function downloadLine(job) {
  if (!job) return '';
  const words = ADD_STAGE_WORDS[job.stage] || String(job.stage || '');
  if (job.stage === 'failed') return `${words}: ${job.error || 'no reason was given.'} Its partial files were deleted.`;
  if (job.stage === 'cancelled') return `${words}. Its partial files were deleted.`;
  const sizes = job.totalBytes ? ` · ${fmtGB(job.receivedBytes)} of ${fmtGB(job.totalBytes)}` : '';
  const checked = job.stage === 'done' && (job.verified || []).length ? ` · ${job.verified.length} file(s) matched the published checksums` : '';
  return `${words} · ${Math.round(Number(job.percent) || 0)}%${sizes}${checked}`;
}
function removeQuestion(row, loaded) {
  return `Remove ${row.name || row.id} from the list?${loaded ? ' It is loaded now and will be put away first.' : ''} Apps will no longer see it.`;
}
// What "Also delete the model files" would do: a NInfer model is one file plus its part files;
// a FreeToken model is a folder, and its settings profile goes with it (review item 8).
function removeFilesNote(view) {
  const where = view.artifact || 'its files';
  if (view.engine === 'freetoken') return `If you tick this, the model folder at ${where} is deleted for good, and so is its settings profile.`;
  return `If you tick this, the model file (and its part files) at ${where} are deleted for good.`;
}
function removeOkLabel(deleteFiles) { return deleteFiles ? 'Remove and delete files' : 'Remove'; }
// The RAM box: whole gigabytes from 0 to 512 (the input's own min/max, checked here too because
// a typed value ignores them; review item 7).
function ramProblem(value) {
  const text = String(value ?? '').trim();
  const n = Number(text);
  if (!text || !Number.isFinite(n) || n < 0 || n > 512) return 'Use a number of gigabytes from 0 to 512.';
  return '';
}
// pi_sync's answer: {status: 'updated'|'not_updated', message, notes}.
function piNote(pi) {
  if (!pi) return '';
  const notes = (pi.notes || []).join(' ');
  if (pi.status === 'not_updated') return `Pi not updated: ${pi.message}`;
  if (pi.status === 'updated') return `Pi updated.${notes ? ` ${notes}` : ''}`;
  return notes;
}
function addedNote(body) {
  return [`Added ${body.name || body.id}.`, ...(body.adjusted || []), piNote(body.pi)].filter(Boolean).join(' ');
}
function removedNote(body) {
  return [`Removed ${body.name || body.id}.`, body.files && body.files.message, body.profile && body.profile.message, piNote(body.pi)].filter(Boolean).join(' ');
}
if (typeof module !== 'undefined') module.exports = { fmtGB, stateWord, sourceText, dialSourceFor, verdictWords, fitSummary, ramSummary, restartQuestion, nowStripHtml, modelsTableHtml, panelErrorText,
  loadQuestion, unloadQuestion, registryProblemHtml, panelSave, answerRestart, answerConfirm, startNow, loadModel, unloadModel, panel, applyView,
  idProblem, detectionText, planSummary, downloadLine, removeQuestion, removeFilesNote, removeOkLabel, ramProblem, piNote, addedNote, removedNote,
  addState, openAdd, closeAdd, addCheckPath, addValidate, addPlan, addPathEdited, addRepoEdited, wirePanel, addDownload, addCancelDownload, pollAddJob, addSave, openRemove, answerRemove };

/* ---------- browser side: uses index.html's state, json, $, setNotice and dial renderer ---------- */
const postJson = (url, payload) => json(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload ?? {}) });
// A 409 that means the model list itself is missing or damaged: show the fix-it box on Models.
function registryProblem(response, body) { return response.status === 409 && (body.code === 'registry_missing' || body.code === 'registry_corrupt'); }
async function backToRegistryProblem(body) { setNotice(panelErrorText(body, 'The model list needs attention.'), 'bad'); clearView(); await loadRegistry(); showMain('models'); }
// 422 rows whose field has no dial on screen (the preset name, the System block, another
// model's limit) would otherwise vanish: say them in the notice instead.
function unplacedErrors(errors) {
  const shown = new Set(Array.from(document.querySelectorAll('[data-error]')).map((node) => node.dataset.error));
  return (Array.isArray(errors) ? errors : []).filter((row) => row && !shown.has(row.field)).map((row) => `${row.where ? `${row.where}: ` : ''}${row.message || 'A value is not allowed.'}`);
}
function dialSource(name) { return dialSourceFor(state.view, state.settings, name); }
function panelExtraDirty() { const view = state.view; return !!(view && view.kind === 'model' && (view.activePreset || null) !== (view.savedPreset || null)); }
function panelDraftChanged() { panel.fit = null; renderPanelFit(); }
function refreshSourceLine(name) {
  const node = document.querySelector(`[data-source="${CSS.escape(name)}"]`);
  const dial = dialByName(name);
  if (!node || !dial) return;
  node.outerHTML = sourceLine(dial);
  wireSourceLines();
}
function wireSourceLines() {
  document.querySelectorAll('[data-reset]').forEach((button) => {
    if (button.dataset.wired) return;
    button.dataset.wired = '1';
    button.addEventListener('click', () => resetDial(button.dataset.reset));
  });
}
function resetDial(name) {
  if (!state.view || !state.view.base || !(name in state.view.base)) return;
  state.settings[name] = JSON.parse(JSON.stringify(state.view.base[name]));
  renderSettings();
  markChanged(name);
}
function showEditor(on) { $('models-view').hidden = on; $('editor').hidden = !on; $('save').hidden = !on; }
function leaveGuard() { if (state.view && changedNames().length) { setNotice('Save or undo your changes first.', 'warn'); return false; } return true; }
function clearView() { state.view = null; state.settings = {}; state.saved = {}; state.dials = []; state.groups = []; state.model = null; }

function showMain(name) {
  panel.main = ['models', 'system', 'ninfer', 'freetoken', 'test'].includes(name) ? name : 'models';
  document.querySelectorAll('[data-main]').forEach((button) => button.setAttribute('aria-selected', String(button.dataset.main === panel.main)));
  try { localStorage.setItem('ft-main', panel.main); } catch (_) {}
  const testView = $('test-view');
  if (testView) testView.hidden = panel.main !== 'test';
  if (panel.main === 'test') { clearView(); showEditor(false); $('models-view').hidden = true; if (typeof pgOpen === 'function') pgOpen(); return; }
  if (panel.main === 'models') { clearView(); showEditor(false); loadModels(); return; }
  openView(panel.main === 'system' ? '/api/panel/views/system' : `/api/panel/views/engine/${panel.main}`);
}

async function loadRegistry() {
  const { response, body } = await json('/api/panel/registry');
  if (!response.ok) { setNotice('Could not read the model list.', 'bad'); return false; }
  panel.revision = body.revision || '';
  renderRegistryProblem(body);
  return body.status === 'ok';
}
function renderRegistryProblem(body) {
  const box = $('registry-problem');
  $('models-head').hidden = body.status !== 'ok';
  if (body.status === 'ok') { box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  box.innerHTML = registryProblemHtml(body);
  if (body.status === 'missing') $('import-now').addEventListener('click', () => importLive());
  box.querySelectorAll('[data-restore]').forEach((button) => button.addEventListener('click', () => restoreBackup(button.dataset.restore)));
}
async function importLive(whenLoaded = null) {
  const { response, body } = await postJson('/api/panel/import', { whenLoaded });
  if (response.status === 409 && body.code === 'choose_restart') {
    const choice = await askRestart(body.affected || [], false);
    if (choice) return importLive(choice);
    setNotice('Nothing was copied.');
    return;
  }
  if (!response.ok) { setNotice(panelErrorText(body, 'Could not copy the settings.'), 'bad'); return; }
  setNotice(body.warnings && body.warnings.length ? `Copied. Notes: ${body.warnings.join(' ')}` : "Copied today's settings.", 'good');
  await loadRegistry();
  await loadModels();
}
async function restoreBackup(name) {
  const { response, body } = await postJson('/api/panel/registry/restore', { backup: name });
  if (!response.ok) { setNotice(panelErrorText(body, 'Could not restore the backup.'), 'bad'); await loadRegistry(); return; }
  setNotice('Backup restored.', 'good');
  await loadRegistry();
  await loadModels();
}

async function loadNow() {
  const { response, body } = await json('/api/panel/now');
  if (!response.ok) return;
  panel.now = body;
  $('now').innerHTML = nowStripHtml(body);
  $('now-updated').textContent = `Updated ${new Date().toLocaleTimeString()}`;
}
// One refresh loop, chained: the next tick is scheduled only after this one's requests have
// settled (errors too). /api/panel/now can take longer than 5 s when the switcher hangs
// (running() waits 5 s, then the card and Windows probes up to 3 s each), and a setInterval
// then piled overlapping requests onto the helper's threadpool (Task 11 review).
async function nowTick() {
  if (document.hidden) return;
  const jobs = [loadNow()];
  if (panel.main === 'models' && !state.view) jobs.push(loadModels());
  await Promise.allSettled(jobs);
}
function startNow() {
  if (panel.nowLoop) return; // tab switches or a second boot never start a second chain
  panel.nowLoop = true;
  const run = async () => {
    try { await nowTick(); } catch (_) { /* the next tick tries again */ }
    panel.nowTimer = setTimeout(run, PANEL_NOW_MS);
  };
  run();
}

async function loadModels() {
  const { response, body } = await json('/api/panel/models');
  if (response.status === 409) { await loadRegistry(); $('models-list').innerHTML = ''; return; }
  if (!response.ok) { $('models-list').innerHTML = `<p class="empty">${panelEsc(panelErrorText(body, 'Could not read the model list.'))}</p>`; return; }
  panel.revision = body.revision || panel.revision;
  panel.models = body.models || [];
  const list = $('models-list');
  list.innerHTML = modelsTableHtml(panel.models, body.switcherUp);
  list.querySelectorAll('[data-load]').forEach((button) => button.addEventListener('click', () => loadModel(button.dataset.load)));
  list.querySelectorAll('[data-unload]').forEach((button) => button.addEventListener('click', () => unloadModel(button.dataset.unload)));
  list.querySelectorAll('[data-settings]').forEach((button) => button.addEventListener('click', () => openModel(button.dataset.settings)));
}
async function loadModel(id) {
  const question = loadQuestion(id, panel.models);
  if (question && !(await askConfirm(question, 'Carry on'))) return;
  setNotice(`Loading ${id}… FreeToken models take a few minutes.`);
  loadNow();
  const { response, body } = await json(`/api/panel/models/${encodeURIComponent(id)}/load`, { method: 'POST' });
  if (response.ok) setNotice(`${id} is loaded.`, 'good'); else setNotice(panelErrorText(body, `Could not load ${id}.`), 'bad');
  loadNow(); loadModels();
}
async function unloadModel(id) {
  if (!(await askConfirm(unloadQuestion(id, panel.models), 'Put it away'))) return;
  setNotice(`Unloading ${id}…`);
  const { response, body } = await json(`/api/panel/models/${encodeURIComponent(id)}/unload`, { method: 'POST' });
  if (response.ok) setNotice(`${id} is unloaded.`, 'good'); else setNotice(panelErrorText(body, `Could not unload ${id}.`), 'bad');
  loadNow(); loadModels();
}

async function openModel(id, preset) {
  document.querySelectorAll('[data-main]').forEach((button) => button.setAttribute('aria-selected', String(button.dataset.main === 'models')));
  panel.main = 'models';
  await openView(`/api/panel/views/model/${encodeURIComponent(id)}${preset === undefined ? '' : `?preset=${encodeURIComponent(preset)}`}`);
}
async function openView(url) {
  if (!leaveGuard()) return;
  const { response, body } = await json(url);
  if (response.status === 409) { await backToRegistryProblem(body); return; }
  if (response.status === 404) { setNotice('That model or preset was not found. It may have been removed; the list has been refreshed.', 'bad'); clearView(); showMain('models'); return; }
  if (!response.ok) { setNotice(panelErrorText(body, 'Could not open these settings.'), 'bad'); return; }
  applyView(body, url);
}
function viewNote(body) {
  if (body.kind === 'engine') return `Every ${body.engineLabel} model starts from these.${body.models && body.models.length ? ` Used by: ${body.models.join(', ')}.` : ''}`;
  if (body.kind === 'system') return 'Settings for the model switcher itself.';
  return `${body.engineLabel}${body.runtimeLabel ? ` (${body.runtimeLabel})` : ''} · ${body.id} · ${stateWord(body.state)}`;
}
function applyView(body, url) {
  // A slow view answer arriving after a switch to the Test tab must not paint the editor over it.
  if (panel.main === 'test') return;
  state.view = { ...body, url };
  panel.revision = body.revision || panel.revision;
  state.settings = JSON.parse(JSON.stringify(body.settings || {}));
  state.saved = JSON.parse(JSON.stringify(state.settings));
  state.dials = Array.isArray(body.dials) ? body.dials : [];
  state.groups = Array.isArray(body.groups) ? body.groups : [];
  state.model = body.model || null;
  state.modelPath = state.model ? String(state.model.path ?? '') : null;
  state.tab = '';
  panel.fit = null;
  $('view-title').textContent = body.title || '';
  $('view-note').textContent = viewNote(body);
  $('view-back').hidden = body.kind !== 'model';
  $('preset-tools').hidden = body.kind !== 'model';
  $('panel-fit').hidden = body.kind !== 'model';
  $('model-remove').hidden = body.kind !== 'model';
  $('preset-form').hidden = true;
  if (body.kind === 'model') renderPresetPicker();
  showEditor(true);
  renderModelCard();
  renderSettings();
  renderPanelFit();
}
async function reopenView() {
  if (!state.view || !state.view.url) return;
  const view = state.view;
  state.saved = JSON.parse(JSON.stringify(state.settings));
  if (view.kind === 'model') await openModel(view.id); else await openView(view.url);
}

function renderPresetPicker() {
  const view = state.view;
  $('preset-picker').innerHTML = `<option value="">No preset</option>${(view.presets || []).map((name) => `<option value="${panelEsc(name)}">${panelEsc(name)}</option>`).join('')}`;
  $('preset-picker').value = view.activePreset || '';
  $('preset-rename').disabled = !view.activePreset;
  $('preset-delete').disabled = !view.activePreset;
}
async function pickPreset() {
  const value = $('preset-picker').value;
  if (changedNames().length) { $('preset-picker').value = state.view.activePreset || ''; setNotice('Save or undo your changes before switching presets.', 'warn'); return; }
  await openModel(state.view.id, value);
  setNotice(value ? `Showing preset “${value}”. Press Save to use it.` : 'Showing no preset. Press Save to use it.');
}
function presetForm(mode) {
  panel.presetMode = mode;
  const view = state.view;
  $('preset-form').hidden = false;
  $('preset-name').hidden = mode === 'delete';
  $('preset-form-note').textContent = mode === 'delete' ? `Delete preset “${view.activePreset}”?` : mode === 'rename' ? 'New name:' : 'Name for the new preset:';
  $('preset-name').value = mode === 'rename' ? (view.activePreset || '') : '';
  if (mode !== 'delete') $('preset-name').focus();
}
async function presetSubmit(whenLoaded = null) {
  const view = state.view;
  const mode = panel.presetMode;
  const name = $('preset-name').value.trim();
  if (mode !== 'delete' && !name) { $('preset-name').focus(); return; }
  const payload = mode === 'rename' ? { name: view.activePreset, newName: name } : mode === 'delete' ? { name: view.activePreset } : { name };
  const { response, body } = await postJson(`/api/panel/models/${encodeURIComponent(view.id)}/presets/${mode}`, { ...payload, revision: panel.revision, whenLoaded });
  if (response.status === 409 && body.code === 'choose_restart') {
    const choice = await askRestart(body.affected || [], body.nextTimeAllowed !== false);
    if (choice) return presetSubmit(choice);
    return;
  }
  if (response.status === 409 && body.code === 'stale_revision') { setNotice(body.message, 'bad'); $('preset-form').hidden = true; await reopenView(); return; }
  if (registryProblem(response, body)) { await backToRegistryProblem(body); return; }
  if (!response.ok) { setNotice(panelErrorText(body, 'Could not change the preset.'), 'bad'); return; }
  panel.revision = body.revision;
  $('preset-form').hidden = true;
  setNotice(mode === 'add' ? `Saved as preset “${name}”.` : mode === 'rename' ? 'Preset renamed.' : 'Preset deleted.', 'good');
  await reopenView();
}

function askRestart(affected, nextTimeAllowed) {
  $('restart-ask-text').textContent = restartQuestion(affected, nextTimeAllowed);
  $('restart-ask-later').hidden = !nextTimeAllowed;
  $('restart-ask').hidden = false;
  return new Promise((resolve) => { panel.restartResolve = resolve; });
}
function answerRestart(choice) {
  $('restart-ask').hidden = true;
  const resolve = panel.restartResolve;
  panel.restartResolve = null;
  if (resolve) resolve(choice);
}
// A yes/no question in the page's own dialog, never the browser's built-in one.
function askConfirm(text, okLabel) {
  $('confirm-ask-text').textContent = text;
  $('confirm-ask-ok').textContent = okLabel || 'Carry on';
  $('confirm-ask').hidden = false;
  return new Promise((resolve) => { panel.confirmResolve = resolve; });
}
function answerConfirm(yes) {
  $('confirm-ask').hidden = true;
  const resolve = panel.confirmResolve;
  panel.confirmResolve = null;
  if (resolve) resolve(!!yes);
}

/* ---------- Stage B: add a model, remove a model ---------- */
function addSource(which) {
  $('add-pc').hidden = which !== 'pc';
  $('add-link').hidden = which !== 'link';
  document.querySelectorAll('[data-add-source]').forEach((button) => button.setAttribute('aria-selected', String(button.dataset.addSource === which)));
}
function addReset() {
  addState.found = null;
  $('add-step-found').hidden = true;
  $('add-step-identity').hidden = true;
  $('add-errors').textContent = '';
  $('add-save').disabled = true;
}
async function openAdd() {
  if (!leaveGuard()) return;
  addReset();
  addState.plan = null;
  addState.cancelling = false;
  $('add-path').value = ''; $('add-repo').value = '';
  ['add-plan-card', 'add-entry', 'add-download-actions', 'add-progress', 'add-last-download'].forEach((id) => { $(id).hidden = true; });
  addSource('pc');
  $('add-wizard').hidden = false;
  $('add-path').focus();
  const { response, body } = await json('/api/panel/add/info');
  if (!response.ok || $('add-wizard').hidden) return;
  addState.roots = body.roots || {};
  // A download started earlier keeps running in the helper; reopening the wizard picks it up.
  const job = body.download;
  if (!job) return;
  if (!ADD_TERMINAL.has(job.stage)) { addSource('link'); addState.job = job; renderAddJob(job); startAddPoll(); return; }
  // One that finished while the wizard was closed: a good download is offered as the path to
  // add; a failed or cancelled one is said once, then left alone.
  if (job.stage === 'done' && job.resultPath) {
    addState.job = job;
    $('add-last-download').hidden = false;
    $('add-last-download').textContent = `Your download finished: ${downloadLine(job)}`;
    $('add-path').value = job.resultPath;
    await addCheckPath(job.resultPath);
    return;
  }
  if (job.id !== addState.seenJob) { addState.seenJob = job.id; addState.job = job; addSource('link'); renderAddJob(job); }
}
function closeAdd() {
  const wasOpen = !$('add-wizard').hidden;
  $('add-wizard').hidden = true;
  clearTimeout(addState.timer); addState.timer = null;
  addState.pollGen += 1; // an answer still on its way is dropped
  if (wasOpen && addState.job && !ADD_TERMINAL.has(addState.job.stage)) setNotice(ADD_KEEPS_GOING);
}
function addBrowse() {
  openBrowser('add', 'add', { title: 'Choose a model file or folder', start: addState.roots.ninfer || '', onPick: (path) => { $('add-path').value = path; addCheckPath(path); } });
}
async function addCheckPath(path) {
  addReset();
  const seq = ++addState.checkSeq;
  const text = String(path ?? '').trim();
  if (!text) return;
  $('add-step-found').hidden = false;
  $('add-found-text').className = '';
  $('add-found-text').textContent = 'Checking…';
  const { response, body } = await postJson('/api/panel/add/detect', { path: text });
  if (seq !== addState.checkSeq) return; // a newer check has been sent; its answer wins
  if (registryProblem(response, body)) { closeAdd(); await backToRegistryProblem(body); return; }
  if (!response.ok) { $('add-found-text').textContent = panelErrorText(body, 'Could not check that path.'); $('add-found-text').className = 'error'; return; }
  showFound(body);
}
function showFound(found) {
  addState.found = found;
  const ok = found.kind !== 'unsupported' && !found.already;
  $('add-step-found').hidden = false;
  $('add-found-text').textContent = detectionText(found);
  $('add-found-text').className = ok ? '' : 'error';
  $('add-step-identity').hidden = !ok;
  if (ok) { $('add-id').value = found.suggested.id; $('add-name').value = found.suggested.name; $('add-ram').value = String(found.suggested.ramNeedGB); }
  addValidate();
}
function addValidate() {
  const found = addState.found;
  const usable = !!found && found.kind !== 'unsupported' && !found.already;
  const problem = usable ? idProblem($('add-id').value, panel.models) : '';
  const ram = usable ? ramProblem($('add-ram').value) : '';
  $('add-id-error').textContent = problem;
  $('add-ram-error').textContent = ram;
  $('add-ram').className = ram ? 'bad' : '';
  $('add-save').disabled = addState.saving || !usable || !!problem || !!ram || !String($('add-name').value ?? '').trim();
}
// Typing in the path or link field makes the last answer stale: an Add or Download must never
// act on a path or repo other than the one on screen, and a check still on its way for the
// old text is dropped (review round 2, PR #17).
function addPathEdited() {
  addState.checkSeq += 1;
  addReset();
}
function addRepoEdited() {
  addState.planSeq += 1;
  addState.plan = null;
  $('add-download').disabled = true;
  ['add-plan-card', 'add-entry', 'add-download-actions'].forEach((id) => { $(id).hidden = true; });
}
async function addPlan() {
  addReset();
  const seq = ++addState.planSeq;
  $('add-download-actions').hidden = true;
  const entry = $('add-entry').hidden ? null : ($('add-entry').value || null);
  $('add-plan-card').hidden = false;
  $('add-plan-card').textContent = 'Reading the repo…';
  const { response, body } = await postJson('/api/panel/add/plan', { link: String($('add-repo').value ?? '').trim(), entry });
  if (seq !== addState.planSeq) return; // a newer plan request has been sent; its answer wins
  if (!response.ok) { addState.plan = null; $('add-plan-card').textContent = panelErrorText(body, 'Could not read that link.'); return; }
  addState.plan = body;
  $('add-plan-card').textContent = planSummary(body);
  const pick = body.kind === 'ninfer' && (body.entries || []).length > 1;
  $('add-entry').hidden = !pick;
  if (pick && !entry) $('add-entry').innerHTML = `<option value="">Pick a NInfer file…</option>${body.entries.map((name) => `<option value="${panelEsc(name)}">${panelEsc(name)}</option>`).join('')}`;
  const ready = body.kind === 'folder' || !!body.entry;
  $('add-download-actions').hidden = !ready;
  $('add-download').disabled = !ready || !!body.exists || !body.diskFits;
}
async function addDownload() {
  const plan = addState.plan;
  if (!plan || $('add-download').disabled) return;
  $('add-download').disabled = true;
  // plan.repo is the canonical owner/name, which the route's parse_repo accepts as a link.
  const { response, body } = await postJson('/api/panel/add/downloads', { link: plan.repo, entry: plan.entry || null });
  if (!response.ok) { $('add-download').disabled = false; $('add-plan-card').textContent = panelErrorText(body, 'Could not start the download.'); return; }
  addState.job = body;
  addState.cancelling = false;
  $('add-last-download').hidden = true;
  renderAddJob(body);
  startAddPoll();
}
function renderAddJob(job) {
  const over = ADD_TERMINAL.has(job.stage);
  if (over) addState.cancelling = false;
  $('add-progress').hidden = false;
  $('add-progress-stage').textContent = addState.cancelling ? 'Cancelling…' : (ADD_STAGE_WORDS[job.stage] || job.stage);
  $('add-progress-bar').style.width = `${Math.max(0, Math.min(100, Number(job.percent) || 0))}%`;
  $('add-progress-detail').textContent = downloadLine(job);
  $('add-download-cancel').disabled = over || addState.cancelling;
}
// One poll loop at a time: a new start retires the old loop, whose in-flight answer is dropped.
function startAddPoll() {
  clearTimeout(addState.timer); addState.timer = null;
  addState.pollGen += 1;
  return pollAddJob(addState.pollGen);
}
async function pollAddJob(gen = addState.pollGen) {
  if (gen !== addState.pollGen) return;
  clearTimeout(addState.timer); addState.timer = null;
  const job = addState.job;
  if (!job) return;
  let answer;
  try {
    answer = await json(`/api/panel/add/downloads/${encodeURIComponent(job.id)}`);
  } catch (_) {
    // The helper is restarting or the network blinked: say so and keep asking (item 2).
    if (gen !== addState.pollGen) return;
    $('add-progress-detail').textContent = ADD_LOST_TOUCH;
    if (!$('add-wizard').hidden) addState.timer = setTimeout(() => pollAddJob(gen), ADD_POLL_MS);
    return;
  }
  if (gen !== addState.pollGen) return;
  const { response, body } = answer;
  if (response.status === 404) { $('add-progress-detail').textContent = 'The settings page restarted and lost this download. Start it again.'; $('add-download').disabled = false; $('add-download-cancel').disabled = true; addState.cancelling = false; return; }
  const current = response.ok ? body : job;
  addState.job = current;
  renderAddJob(current);
  if (current.stage === 'done') { $('add-path').value = current.resultPath || ''; await addCheckPath(current.resultPath); return; }
  if (current.stage === 'failed' || current.stage === 'cancelled') { $('add-download').disabled = false; return; }
  if (!$('add-wizard').hidden) addState.timer = setTimeout(() => pollAddJob(gen), ADD_POLL_MS);
}
async function addCancelDownload() {
  const job = addState.job;
  if (!job || addState.cancelling || ADD_TERMINAL.has(job.stage)) return;
  // The helper only flags the job; the worker stops at its next chunk. The button stays off and
  // the stage reads "Cancelling…" until a poll brings back "cancelled" (item 5).
  addState.cancelling = true;
  renderAddJob(job);
  let answer;
  try { answer = await postJson(`/api/panel/add/downloads/${encodeURIComponent(job.id)}/cancel`, {}); } catch (_) { answer = { response: { ok: false }, body: {} }; }
  if (addState.job !== job && !(addState.job && addState.job.id === job.id)) return;
  if (!answer.response.ok) { addState.cancelling = false; renderAddJob(addState.job); $('add-progress-detail').textContent = panelErrorText(answer.body, 'Could not cancel the download. Try again.'); return; }
  if (ADD_TERMINAL.has(answer.body.stage)) { addState.job = answer.body; renderAddJob(answer.body); }
}
async function addSave() {
  const found = addState.found;
  if (!found || addState.saving || $('add-save').disabled) return;
  addState.saving = true;
  $('add-save').disabled = true;
  $('add-errors').textContent = '';
  try {
    const { response, body } = await postJson('/api/panel/models', { revision: panel.revision, path: found.path, id: String($('add-id').value ?? '').trim(), name: String($('add-name').value ?? '').trim(), ramNeedGB: $('add-ram').value });
    addState.saving = false;
    if (response.status === 409 && body.code === 'stale_revision') { await loadModels(); $('add-errors').textContent = 'The model list changed meanwhile. Check the details and press Add model again.'; addValidate(); return; }
    if (registryProblem(response, body)) { closeAdd(); await backToRegistryProblem(body); return; }
    // 422 here is {detail: [{field: 'add.…', message}]} or a plain {message}; 409 already_added and
    // 503 switcher_unknown carry plain words too. panelErrorText picks the words out of each.
    if (!response.ok) { $('add-errors').textContent = panelErrorText(body, 'Could not add the model.'); addValidate(); return; }
    panel.revision = body.revision || panel.revision;
    closeAdd();
    setNotice(addedNote(body), body.pi && body.pi.status === 'not_updated' ? 'warn' : 'good');
    await loadModels();
    loadNow();
  } finally {
    addState.saving = false;
  }
}

async function openRemove() {
  const view = state.view;
  if (!view || view.kind !== 'model' || panel.busy || !leaveGuard()) return;
  const row = (panel.models || []).find((item) => item.id === view.id) || { id: view.id, name: view.name };
  $('remove-ask-text').textContent = removeQuestion(row, isLoadedState(view.state));
  $('remove-files').checked = false; // spec: "also delete the model files" is off by default, every time
  $('remove-files-note').textContent = removeFilesNote(view);
  $('remove-ask-ok').textContent = removeOkLabel(false);
  $('remove-ask').hidden = false;
  $('remove-ask-cancel').focus();
  const yes = await new Promise((resolve) => { panel.removeResolve = resolve; });
  if (!yes || panel.busy) return;
  // Busy while the remove runs (item 1): the unload it may do takes seconds, and a second press
  // of Remove, Save or Load meanwhile would race it.
  panel.busy = true; setBusy(true);
  try {
    const { response, body } = await postJson(`/api/panel/models/${encodeURIComponent(view.id)}/remove`, { revision: panel.revision, deleteFiles: !!$('remove-files').checked });
    if (response.status === 409 && body.code === 'stale_revision') { setNotice(body.message, 'bad'); await reopenView(); return; }
    if (registryProblem(response, body)) { await backToRegistryProblem(body); return; }
    // 409 files_missing / files_unsafe / files_shared and 503 switcher_unknown / unload_failed all
    // carry plain words that say nothing was removed.
    if (!response.ok) { setNotice(panelErrorText(body, 'Could not remove the model.'), 'bad'); return; }
    panel.revision = body.revision || panel.revision;
    clearView();
    showMain('models');
    setNotice(removedNote(body), (body.files && !body.files.deleted) || (body.profile && body.profile.message) || (body.pi && body.pi.status === 'not_updated') ? 'warn' : 'good');
    loadNow();
  } finally {
    panel.busy = false; setBusy(false);
  }
}
function answerRemove(yes) {
  $('remove-ask').hidden = true;
  const resolve = panel.removeResolve;
  panel.removeResolve = null;
  if (resolve) resolve(!!yes);
}

function draftBody() {
  const view = state.view;
  if (view.kind === 'system') return { system: { ...state.settings } };
  if (view.kind === 'engine') return { settings: { ...state.settings } };
  const identity = {}; const settings = {};
  Object.entries(state.settings).forEach(([name, value]) => { if (name.startsWith('model.')) identity[name.slice(6)] = value; else settings[name] = value; });
  return { identity, settings, activePreset: view.activePreset || null };
}
async function runFit() {
  const view = state.view;
  if (!view || view.kind !== 'model') return null;
  const draft = draftBody();
  $('panel-fit-text').textContent = 'Checking memory…';
  const { response, body } = await postJson(`/api/panel/models/${encodeURIComponent(view.id)}/fit`, { settings: draft.settings, identity: draft.identity });
  if (state.view !== view) return null;
  if (response.status === 422) { showFieldErrors(body.detail); panel.fit = { verdict: 'unknown', message: 'Some values need attention first.' }; }
  else panel.fit = response.ok ? body : { verdict: 'unknown', message: panelErrorText(body, 'The memory check failed.') };
  panel.fitDraft = JSON.stringify(draft);
  renderPanelFit();
  return panel.fit;
}
function renderPanelFit() {
  if (!state.view || state.view.kind !== 'model') return;
  const fit = panel.fit;
  $('panel-fit-text').textContent = fit ? fitSummary(fit) : 'Memory is checked when you press Save.';
  $('panel-fit-text').className = `fit-verdict ${fit ? fit.verdict : ''}`;
  $('panel-fit-ram').textContent = fit ? ramSummary(fit.ram) : '';
  $('panel-fit-suggest').hidden = !(fit && fit.suggestion);
  $('save-anyway').hidden = !(fit && fit.verdict === 'wont_fit');
}
function useSuggestion() {
  const suggestion = panel.fit && panel.fit.suggestion;
  if (!suggestion) return;
  const names = Object.keys(suggestion.settings).filter((name) => name in state.settings);
  names.forEach((name) => { state.settings[name] = suggestion.settings[name]; });
  renderSettings();
  names.forEach((name) => markChanged(name));
  setNotice('Suggested values filled in. Press Save to check and keep them.', 'good');
}

async function panelSave(options = {}) {
  const view = state.view;
  if (!view || panel.busy) return;
  // Busy before the fit check (final review): a second Save press while the check ran sent a
  // second PUT, which came back stale_revision or raised a second restart question.
  panel.busy = true; setBusy(true);
  let retry = null;
  try {
    if (view.kind === 'model' && !options.anyway && !options.whenLoaded) {
      const fit = panel.fit && panel.fitDraft === JSON.stringify(draftBody()) ? panel.fit : await runFit();
      if (fit && fit.verdict === 'wont_fit') { setNotice("This won't fit on the graphics card. Change the values, or press Save anyway.", 'bad'); return; }
    }
    const url = view.kind === 'system' ? '/api/panel/system' : view.kind === 'engine' ? `/api/panel/engines/${encodeURIComponent(view.engine)}/defaults` : `/api/panel/models/${encodeURIComponent(view.id)}`;
    showFieldErrors([]);
    retry = await panelPut(url, options);
  } finally {
    panel.busy = false; setBusy(false);
  }
  // Asked "restart now / next time": send the same save again with the answer, after the busy
  // flag is down (a return inside the try used to skip this, so the answer was never sent).
  if (retry) await panelSave(retry);
}
// One PUT. Returns the options for a second try when the restart question was answered.
async function panelPut(url, options) {
  const { response, body } = await json(url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ revision: panel.revision, whenLoaded: options.whenLoaded || null, ...draftBody() }) });
  if (response.status === 409 && body.code === 'choose_restart') {
    const choice = await askRestart(body.affected || [], body.nextTimeAllowed !== false);
    if (choice) return { ...options, whenLoaded: choice, anyway: true };
    setNotice('Nothing was saved.');
    return null;
  }
  if (response.status === 409 && body.code === 'stale_revision') { setNotice(body.message, 'bad'); state.saved = JSON.parse(JSON.stringify(state.settings)); await reopenView(); return null; }
  if (registryProblem(response, body)) { state.saved = JSON.parse(JSON.stringify(state.settings)); await backToRegistryProblem(body); return null; }
  if (response.status === 422 && body.code === 'switcher_refused') { setNotice(body.message, 'bad'); return null; }
  if (response.status === 422) {
    showFieldErrors(body.detail);
    const other = unplacedErrors(body.detail);
    setNotice(other.length ? `Not saved: ${other.join(' ')}` : 'Some values need attention. They are marked in red.', 'bad');
    return null;
  }
  // 503 switcher_unknown and switcher_down carry a plain message: nothing was saved.
  if (!response.ok) { setNotice(panelErrorText(body, 'Could not save.'), 'bad'); return null; }
  panel.revision = body.revision;
  let note = 'Saved.';
  if (body.restarting && body.restarting.length) note = `Saved. Restarting ${body.restarting.join(', ')} with the new settings…`;
  else if (options.whenLoaded === 'next-time') note = 'Saved. The loaded model keeps its old settings until its next load.';
  else if (body.inherits && body.inherits.length) note = `Saved. Used by: ${body.inherits.join(', ')}.`;
  state.saved = JSON.parse(JSON.stringify(state.settings));
  await reopenView();
  setNotice(note, 'good');
  loadNow();
  return null;
}

function wirePanel() {
  document.querySelectorAll('[data-main]').forEach((button) => button.addEventListener('click', () => { if (leaveGuard()) { clearView(); showMain(button.dataset.main); } }));
  $('view-back').addEventListener('click', () => { if (leaveGuard()) { clearView(); showMain('models'); } });
  $('preset-picker').addEventListener('change', pickPreset);
  $('preset-save-new').addEventListener('click', () => { if (changedNames().length) { setNotice('Save your changes first; a new preset copies the saved settings.', 'warn'); return; } presetForm('add'); });
  $('preset-rename').addEventListener('click', () => presetForm('rename'));
  $('preset-delete').addEventListener('click', () => presetForm('delete'));
  $('preset-form-ok').addEventListener('click', () => presetSubmit());
  $('preset-form-cancel').addEventListener('click', () => { $('preset-form').hidden = true; });
  $('panel-fit-check').addEventListener('click', runFit);
  $('panel-fit-suggest').addEventListener('click', useSuggestion);
  $('save-anyway').addEventListener('click', () => panelSave({ anyway: true }));
  $('restart-ask-now').addEventListener('click', () => answerRestart('restart'));
  $('restart-ask-later').addEventListener('click', () => answerRestart('next-time'));
  $('restart-ask-cancel').addEventListener('click', () => answerRestart(null));
  $('confirm-ask-ok').addEventListener('click', () => answerConfirm(true));
  $('confirm-ask-cancel').addEventListener('click', () => answerConfirm(false));
  $('add-model').addEventListener('click', openAdd);
  $('add-close').addEventListener('click', closeAdd);
  $('add-cancel').addEventListener('click', closeAdd);
  document.querySelectorAll('[data-add-source]').forEach((button) => button.addEventListener('click', () => { addReset(); addSource(button.dataset.addSource); }));
  $('add-browse').addEventListener('click', addBrowse);
  $('add-check').addEventListener('click', () => addCheckPath($('add-path').value));
  $('add-path').addEventListener('input', addPathEdited);
  $('add-repo').addEventListener('input', addRepoEdited);
  $('add-path').addEventListener('keydown', (event) => { if (event.key === 'Enter') addCheckPath($('add-path').value); });
  $('add-plan').addEventListener('click', () => { $('add-entry').hidden = true; addPlan(); });
  $('add-repo').addEventListener('keydown', (event) => { if (event.key === 'Enter') { $('add-entry').hidden = true; addPlan(); } });
  $('add-entry').addEventListener('change', addPlan);
  $('add-download').addEventListener('click', addDownload);
  $('add-download-cancel').addEventListener('click', addCancelDownload);
  ['add-id', 'add-name', 'add-ram'].forEach((id) => $(id).addEventListener('input', addValidate));
  $('add-save').addEventListener('click', addSave);
  $('model-remove').addEventListener('click', openRemove);
  $('remove-files').addEventListener('change', () => { $('remove-ask-ok').textContent = removeOkLabel(!!$('remove-files').checked); });
  $('remove-ask-ok').addEventListener('click', () => answerRemove(true));
  $('remove-ask-cancel').addEventListener('click', () => answerRemove(false));
}
async function panelBoot() {
  wirePanel();
  const ok = await loadRegistry();
  let main = 'models';
  try { main = localStorage.getItem('ft-main') || 'models'; } catch (_) {}
  showMain(ok ? main : 'models');
  startNow();
}
