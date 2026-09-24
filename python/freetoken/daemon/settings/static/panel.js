'use strict';
/* Control panel, Stage A (docs/superpowers/specs/2026-09-24-control-panel-part2-design.md).
   The Models list, the "Right now" strip, the System / NInfer defaults / FreeToken defaults
   tabs and each model's settings. Dials are drawn by index.html's renderer: a view fills
   state.dials, state.groups, state.settings and state.saved, sets state.view and calls
   renderSettings(). The helpers above the browser section are pure and exported for node. */

const PANEL_NOW_MS = 5000;
const PANEL_GB = 1024 ** 3;
const panel = { main: 'models', revision: '', models: [], now: null, busy: false, fit: null, fitDraft: '', nowTimer: null, restartResolve: null, presetMode: 'add' };

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
  return [
    `<div class="stat"><span class="small">Loaded</span><strong class="now-list">${running}</strong>${held}${restart}</div>`,
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
if (typeof module !== 'undefined') module.exports = { fmtGB, stateWord, sourceText, dialSourceFor, verdictWords, fitSummary, ramSummary, restartQuestion, nowStripHtml, modelsTableHtml, panelErrorText,
  panelSave, answerRestart };

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
  panel.main = ['models', 'system', 'ninfer', 'freetoken'].includes(name) ? name : 'models';
  document.querySelectorAll('[data-main]').forEach((button) => button.setAttribute('aria-selected', String(button.dataset.main === panel.main)));
  try { localStorage.setItem('ft-main', panel.main); } catch (_) {}
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
// Backups are named registry.json.bak-YYYYmmdd-HHMMSS-ffffff; show the time, not the file name.
function backupWhen(name) {
  const m = String(name).match(/bak-(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d)(\d\d)/);
  if (!m) return name;
  return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4]), Number(m[5]), Number(m[6])).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
function renderRegistryProblem(body) {
  const box = $('registry-problem');
  $('models-head').hidden = body.status !== 'ok';
  if (body.status === 'ok') { box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  if (body.status === 'missing') {
    box.innerHTML = `<div class="panel-head"><div><h2>Set up the control panel</h2><p class="small">This copies today's models and settings from ${panelEsc(body.configPath)} and the helper's start-up file into the panel's own list. The old switcher file is kept as a backup.</p></div><button class="button primary" id="import-now" type="button">Copy today's settings</button></div>`;
    $('import-now').addEventListener('click', () => importLive());
    return;
  }
  const backups = (body.backups || []).map((name) => `<li><button class="button small" type="button" data-restore="${panelEsc(name)}">Restore the copy from ${panelEsc(backupWhen(name))}</button></li>`).join('');
  box.innerHTML = `<div class="panel-head"><div><h2>The model list is damaged</h2><p class="small">${panelEsc(body.message || '')}</p><p class="small">Nothing was changed. Restore the last good backup:</p></div></div>${backups ? `<ul class="backups">${backups}</ul>` : '<p class="empty">No backups were found.</p>'}`;
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
function startNow() {
  clearInterval(panel.nowTimer);
  loadNow();
  panel.nowTimer = setInterval(() => { if (document.hidden) return; loadNow(); if (panel.main === 'models' && !state.view) loadModels(); }, PANEL_NOW_MS);
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
  setNotice(`Loading ${id}… FreeToken models take a few minutes.`);
  loadNow();
  const { response, body } = await json(`/api/panel/models/${encodeURIComponent(id)}/load`, { method: 'POST' });
  if (response.ok) setNotice(`${id} is loaded.`, 'good'); else setNotice(panelErrorText(body, `Could not load ${id}.`), 'bad');
  loadNow(); loadModels();
}
async function unloadModel(id) {
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
  if (view.kind === 'model' && !options.anyway && !options.whenLoaded) {
    const fit = panel.fit && panel.fitDraft === JSON.stringify(draftBody()) ? panel.fit : await runFit();
    if (fit && fit.verdict === 'wont_fit') { setNotice("This won't fit on the graphics card. Change the values, or press Save anyway.", 'bad'); return; }
  }
  const url = view.kind === 'system' ? '/api/panel/system' : view.kind === 'engine' ? `/api/panel/engines/${encodeURIComponent(view.engine)}/defaults` : `/api/panel/models/${encodeURIComponent(view.id)}`;
  panel.busy = true; setBusy(true); showFieldErrors([]);
  let retry = null;
  try {
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
}
async function panelBoot() {
  wirePanel();
  const ok = await loadRegistry();
  let main = 'models';
  try { main = localStorage.getItem('ft-main') || 'models'; } catch (_) {}
  showMain(ok ? main : 'models');
  startNow();
}
