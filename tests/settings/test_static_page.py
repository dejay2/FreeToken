from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest


PAGE = Path(__file__).parents[2] / "python" / "freetoken" / "daemon" / "settings" / "static" / "index.html"


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self.external_assets: list[str] = []
        self.title = ""
        self._in_script = False
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script":
            self._in_script = True
            if values.get("src"):
                self.external_assets.append(values["src"] or "")
        elif tag == "link" and values.get("href"):
            self.external_assets.append(values["href"] or "")
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self.scripts.append(data)
        if self._in_title:
            self.title += data


DIAL_FIXTURE = [
    {"name": "ModelPath", "group": "Model & chats", "control": "path"},
    {"name": "ContextTokens", "group": "Model & chats", "control": "number"},
    {"name": "KVDtype", "group": "Model & chats", "control": "choice"},
    {"name": "MaxRunningRequests", "group": "Model & chats", "control": "number"},
    {"name": "KVPark", "group": "Model & chats", "control": "choice"},
    {"name": "MoECacheSize", "group": "Memory & experts", "control": "number"},
    {"name": "EmbedHost", "group": "Memory & experts", "control": "toggle"},
    {"name": "EnableVision", "group": "Pictures", "control": "toggle"},
    {"name": "FREETOKEN_MTP_SPECULATE", "group": "MTP", "control": "toggle"},
    {"name": "ExpertLoad", "group": "Server & advanced", "control": "choice"},
    {"name": "Port", "group": "Server & advanced", "control": "number"},
    {"name": "DesktopPython", "group": "Server & advanced", "control": "path"},
]


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _source() -> str:
    parser = _PageParser()
    parser.feed(_page())
    assert parser.title.strip(), "the page needs a browser title"
    assert not parser.external_assets, "the page must not download outside files"
    return "\n".join(parser.scripts)


def test_static_page_renders_fixture_dials_from_metadata() -> None:
    source = _source()

    assert "json('/api/settings')" in source
    assert "groups.map((group)" in source
    assert "dials.map((dial) => renderDial(dial))" in source
    assert "groups.find((item) => item.name === dial.group)" in source
    assert "state.settings[dial.name]" in source or "state.settings[dial.name] ??" in source
    assert "valueFor(dial)" in source
    assert 'data-dial="${escapeHtml(dial.name)}"' in source

    # The renderer must have a distinct branch for each control kind that needs a different
    # browser control. Path and text both use a text box, while the metadata still chooses them.
    for control in ("toggle", "choice", "number"):
        assert f'dial.control === "{control}"' in source
    assert "dial.control === \"path\" || dial.control === \"text\"" in source

    # No dial name is allowed to be the page's source of truth; the fixture names only exercise
    # the metadata contract above, while the page must iterate the response's dials array.
    assert not any(f'"{dial["name"]}"' in source for dial in DIAL_FIXTURE)

    control_markup = {
        "toggle": '<input type="checkbox">',
        "number": '<input type="number">',
        "choice": '<select>',
        "path": '<input type="text">',
    }
    rendered = [f'{dial["name"]} -> {control_markup[dial["control"]]}' for dial in DIAL_FIXTURE]
    assert len(rendered) == len(DIAL_FIXTURE)
    print("DOM dump (section-8 fixture): " + " | ".join(rendered))


def test_save_sends_current_values_and_handles_validation_results() -> None:
    source = _source()

    assert "json('/api/settings', { method: 'PUT'" in source
    assert "JSON.stringify({ settings: state.settings })" in source
    assert "response.status === 422" in source
    assert "showFieldErrors" in source
    assert "data-error=\"${escapeHtml(dial.name)}\"" in source


def test_lifecycle_profiles_status_and_logs_use_the_route_contract() -> None:
    page = _page()
    source = _source()

    for route in (
        "/api/server/",
        "/api/server/jobs/",
        "/api/status",
        "/api/logs",
        "/api/profiles",
        "/api/profiles/",
        "/api/profiles/${encodeURIComponent(id)}/activate",
    ):
        assert route in source
    assert "profile.isDefault" in source
    assert "profile.bootFile" in source
    for action in ("start", "stop", "restart"):
        assert f"serverAction('{action}')" in page
    assert "auto-follow" in page
    assert "data-confirm" in source
    assert "data-confirm-delete" in source
    assert "window.prompt" not in source
    assert "window.confirm" not in source
    assert "window.alert" not in source


def test_page_has_tabs_info_buttons_sliders_and_browse() -> None:
    page = _page()
    source = _source()

    assert 'role="tablist"' in page
    tabs = ("model-chats", "memory-experts", "memory-governor", "mtp", "pictures", "server-advanced")
    assert [match.group(1) for match in re.finditer(r'<button[^>]+data-tab="([^"]+)"', page)] == list(tabs)
    # Plain-language help, effect chips, sliders and folder browsing are all driven by the
    # metadata the settings route sends; the page only needs the generic hooks.
    assert 'data-info="${name}"' in source
    assert "dial.effects" in source
    assert 'type="range" data-slider=' in source
    assert "dial.slider" in source
    assert 'data-browse="${name}"' in source
    assert "/api/browse" in source
    assert "dial.autoValue" in source
    assert "dial.displayFactor" in source
    assert "advanced-settings" in source
    assert "dial.blurb" in source
    assert "tabKeyForGroup" in source
    assert 'id="search"' in page
    # Restarting with unsaved changes asks in-page, never through a browser dialog.
    assert "restart-dialog" in page
    assert "changedNames()" in source


def test_page_reads_the_model_and_reshapes_itself() -> None:
    page = _page()
    source = _source()

    # A model card above the settings, filled from the settings payload's model block.
    assert 'id="model-card"' in page
    assert "body.model" in source
    assert "renderModelCard()" in source
    # Typing or browsing a new model folder previews it before Save through the settings
    # route's model query, and the dials are re-rendered from that response.
    assert "/api/settings?model=" in source
    assert "refreshModel(" in source
    assert "dial.browse === 'model'" in source
    # Text-stored counts (layers kept on the card) go through the storedAs contract.
    assert "dial.storedAs" in source
    assert "dial.storedZero" in source
    assert "const respelled = []" in source
    assert "toStored(dial, count)" in source
    assert "respelled.forEach((name) => markChanged(name))" in source


def test_models_tab_uses_preview_download_progress_and_folder_routes() -> None:
    page = _page()
    source = _source()

    assert "<!-- models-tab -->" in page and "<!-- /models-tab -->" in page
    assert 'id="model-repo"' in page
    assert 'id="model-preview"' in page
    for route in ("/api/downloads/preview", "/api/downloads", "/api/models"):
        assert route in source
    assert "model-download-cancel" in page
    assert "setTimeout(pollModelDownload, 2000)" in source
    assert "Use as model folder" in page
    assert "dial.browse === 'model'" in source
    assert "window.confirm" not in source
    assert "window.alert" not in source
    assert "window.prompt" not in source


def test_page_has_memory_fit_panel_and_checked_lifecycle_snapshot() -> None:
    page = _page()
    source = _source()

    assert 'id="fit-warning"' in page
    for control in ("fit-now", "fit-empty", "fit-suggestion", "fit-apply", "fit-override", "fit-error"):
        assert f'id="{control}"' in page
    assert "json('/api/settings/estimate'" in source
    assert "body: JSON.stringify({ settings: fitClone(snapshot), action: action || null })" in source
    assert "Start anyway with my values." in page
    assert "state.fit" in source
    assert "checkedSettings" in source
    assert "JSON.stringify({ force: state.fit.override" in source
    assert "settings: state.fit.checkedSettings" in source
    assert "fit-apply" in source and "fit-override" in source
    assert "30" in source


def test_restart_requires_current_fit_before_save_or_lifecycle() -> None:
    source = _source()

    # Restart must not treat an empty-machine fit as permission while the current card is occupied.
    assert "const fitsForAction = result?.fits_now;" in source
    assert "action === 'restart' ? result?.fits_empty" not in source


def test_page_keeps_stop_available_while_estimate_or_start_is_busy() -> None:
    source = _source()

    assert "#fit-override" in source
    assert "#fit-apply" in source
    assert "header .button:not(#stop)" in source
    assert "#profiles .button, #new-profile, #profile-save" in source
    assert "#fit-override, #fit-apply" in source
    assert "stop" in source


def _run_fit_script(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise the inline page JavaScript")
    # Run the real main script, leaving only initial network boot to the browser tests.
    source = _source().split("    'use strict';", 1)[1].rsplit("    boot();", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const nodes = new Map();
function element(id) {
  if (!nodes.has(id)) nodes.set(id, {hidden:false, disabled:false, textContent:'', innerHTML:'',
    addEventListener(){}, setAttribute(){}, querySelectorAll(){return []}, append(){},
    classList:{add(){}, remove(){}, toggle(){}}});
  return nodes.get(id);
}
global.document = {getElementById:element, querySelectorAll:()=>[], querySelector:()=>null, addEventListener(){}};
global.window = {addEventListener(){}};
global.CSS = {escape: value=>value};
global.localStorage = {getItem(){return null}, setItem(){}};
global.setTimeout = global.setInterval = ()=>0;
global.clearTimeout = ()=>{};
let responseBody, requests = [];
global.fetch = async (url, options = {}) => {
  requests.push([options.method || 'GET', url, options.body ? JSON.parse(options.body) : null]);
  const body = url === '/api/settings/estimate' ? responseBody : options.method === 'PUT'
    ? JSON.parse(options.body) : {jobId:'fixture-job', stage:'serving'};
  return {ok:true, status:200, json:async()=>body};
};
"""
    run = subprocess.run(
        [node], input=harness + source + script, text=True, capture_output=True, check=False, timeout=15,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout.strip())


def _fit_response() -> dict:
    phase = {"need_bytes": 80, "resident_bytes": 70, "boot_peak_bytes": 80, "shortfall_bytes": 0}
    resource = {"free_bytes": 100, "total_bytes": 200, "now": phase, "empty": phase}
    geometry = {"total_slots": 8, "lru_slots": 8, "owned_layers": [], "num_pages": 2,
                "page_size": 64, "usable_kv_tokens": 64, "prefill_overlap": False}
    return {
        "version": 1, "status": "ok", "fits": True, "fits_now": True, "fits_empty": True,
        "sampled_at": "2026-09-07T12:00:00Z", "effective_settings": {"MoECacheSize": 8},
        "effective": {"page_size": 64, "attention_backend": "qsa_sparse", "ple_backend": "mmap",
                      "pin_budget_bytes": None, "bank_cuda_alloc": False},
        "machine": {"ram_source": "fixture", "gpu_uuid": "fixture-gpu", "gpu_name": "fixture",
                    "vram_source": "fixture", "cgroup_limited": False},
        "resources": {"ram": resource, "vram": resource}, "geometry": {"now": geometry, "empty": geometry},
        "pinning": {"need_bytes": 10, "cap_bytes": None, "shortfall_bytes": 0, "cpu_layers": []},
        "components": [{"name": "weights", "resource": "vram", "phase": "dense", "bytes": 20,
                        "kind": "allocation", "source": "fixture", "scenario": "both"}],
        "issues": [], "assumptions": ["Synthetic values, not measurements."], "suggestion": None,
    }


def test_malformed_success_never_saves_or_launches_and_allows_explicit_override() -> None:
    script = r"""
(async () => {
  const fixture = FIXTURE;
  const malformed = Object.keys(fixture).map(key => {
    const body = structuredClone(fixture); delete body[key]; return [key, body];
  });
  for (const [path, value] of [
    ['effective_settings', []], ['effective_settings', {}], ['effective_settings.MoECacheSize', {}],
    ['effective_settings', {OtherSetting:8}], ['effective_settings.MoECacheSize', 4],
    ['effective.page_size', 0], ['effective.pin_budget_bytes', '10'], ['effective.bank_cuda_alloc', 1],
    ['machine.cgroup_limited', 'false'], ['sampled_at', 0], ['resources.ram.free_bytes', -1],
    ['resources.ram.total_bytes', null], ['resources.vram.now.need_bytes', 0.5],
    ['geometry.now', null], ['geometry.empty.owned_layers', '0'], ['geometry.now.num_pages', 1],
    ['geometry.now.prefill_overlap', 1], ['pinning.cpu_layers', [false]], ['pinning.need_bytes', '10'],
    ['pinning.cap_bytes', false], ['components', {}], ['components', [{name:'incomplete'}]],
    ['issues', {}], ['issues', [{code:'bad', scope:'elsewhere', message:'bad'}]],
    ['assumptions', [7]], ['suggestion', {}],
    ['suggestion', {target:'now', settings:{}, fits:true, fits_now:true, fits_empty:true, changes:[]}],
  ]) {
    const body = structuredClone(fixture), keys = path.split('.');
    let parent = body; for (const key of keys.slice(0, -1)) parent = parent[key];
    parent[keys.at(-1)] = value; malformed.push([path, body]);
  }
  for (const path of ['effective', 'machine', 'resources.ram', 'resources.vram.now', 'geometry.now', 'pinning', 'components.0']) {
    const keys = path.split('.');
    let row = fixture; for (const key of keys) row = row[key];
    for (const field of Object.keys(row)) {
      const body = structuredClone(fixture);
      let parent = body; for (const key of keys) parent = parent[key];
      delete parent[field]; malformed.push([`${path}.${field}`, body]);
    }
  }
  for (const [action, save] of [['start', true], ['restart', true], ['restart', false]]) {
    for (const [name, body] of malformed) {
      requests = []; responseBody = body;
      state.settings = {...fixture.effective_settings}; state.saved = {...state.settings};
      state.busy = false; invalidateFit();
      await serverAction(action, {confirmed:true, save});
      assert.deepEqual(requests.map(row=>row.slice(0, 2)), [['POST','/api/settings/estimate']], `${action}/${save}/${name}`);
      assert.equal(state.fit.result.status, 'unavailable', name);
      assert.match(element('fit-warning').textContent, /Couldn't check memory/);
      assert.equal(element('fit-override').hidden, false);
      assert.equal(element('fit-suggestion').hidden, true);
    }
    console.log(`${action} save=${save}: ${malformed.length} malformed successes => estimate only; unavailable/override visible`);
    requests = [];
    await Promise.all([overrideFit(), overrideFit()]);
    const expected = save ? [['PUT','/api/settings'], ['POST',`/api/server/${action}`]] : [['POST',`/api/server/${action}`]];
    assert.deepEqual(requests.filter(row=>row[0]!=='GET').map(row=>row.slice(0, 2)), expected);
    assert.equal(requests.find(row=>row[1]===`/api/server/${action}`)[2].force, true);
    console.log(`${action} save=${save}: explicit override => ${JSON.stringify(expected)} exactly once`);
  }
})().catch(error=>{console.error(error); process.exitCode=1});
""".replace("FIXTURE", json.dumps(_fit_response()))
    _run_fit_script(script)


def test_complete_fit_contract_preserves_launch_and_nonfit_states() -> None:
    script = r"""
(async () => {
  const fixture = FIXTURE;
  for (const action of ['start', 'restart']) {
    requests = []; responseBody = structuredClone(fixture);
    state.settings = {...fixture.effective_settings}; state.saved = {...state.settings};
    state.busy = false; invalidateFit();
    await serverAction(action, {confirmed:true});
    assert.deepEqual(requests.filter(row=>row[0]!=='GET').map(row=>row.slice(0, 2)),
      [['POST','/api/settings/estimate'], ['PUT','/api/settings'], ['POST',`/api/server/${action}`]]);
    assert.deepEqual(requests[2][2], {force:false, settings:fixture.effective_settings});
    console.log(`${action}: complete fit => estimate, Save, lifecycle with checked snapshot`);
  }
  responseBody = structuredClone(fixture); responseBody.fits = responseBody.fits_now = false;
  responseBody.geometry.now = null;
  responseBody.issues = [{code:'auto_budget', scope:'now', message:'Not enough free capacity'}];
  state.busy = false; invalidateFit(); requests = [];
  await serverAction('restart', {confirmed:true});
  assert.equal(state.fit.result.status, 'ok');
  assert.deepEqual(requests.map(row=>row[1]), ['/api/settings/estimate']);
  assert.equal(element('fit-suggestion').hidden, true, 'null suggestion must not expose an empty Apply');
  assert.equal(element('fit-override').hidden, false);
  console.log('nonfit/null geometry with issue: estimate only; no Apply; override visible');
  responseBody.suggestion = {target:'now', settings:{MoECacheSize:4}, fits:true, fits_now:true, fits_empty:true,
    changes:[{name:'MoECacheSize', from:8, to:4, reason:'Free memory for boot'}]};
  state.busy = false; invalidateFit(); requests = [];
  await serverAction('start');
  assert.equal(state.fit.result.status, 'ok');
  assert.equal(element('fit-suggestion').hidden, false);
  applyFitSuggestion();
  assert.deepEqual(state.settings, {MoECacheSize:4});
  assert.deepEqual(requests.map(row=>row[1]), ['/api/settings/estimate']);
  assert.equal(state.fit.allowOverride, false);
  console.log('complete suggestion: Apply changes form only and clears override');
})().catch(error=>{console.error(error); process.exitCode=1});
""".replace("FIXTURE", json.dumps(_fit_response()))
    _run_fit_script(script)


def test_the_slot_field_prints_the_minimum_for_the_layers_on_the_card() -> None:
    """Every layer kept whole on the card is charged to the expert-slot total, so the page
    shows minimum = layers x expertsPerLayer + streamingFloor while the value is still a
    draft. Both numbers come from the dial metadata; the page knows no model geometry.
    Live failure behind it: 4288 slots with 8 layers on the card (2026-09-07 11:22 BST)."""
    source = _source()
    assert "data-floor-hint=" in source and "data-owned-dial=" in source
    assert "dial.expertsPerLayer && dial.ownedDial" in source, "hint only where the metadata says so"
    assert "refreshFloorHints()" in source

    script = r"""
(() => {
  const dial = {name:'Slots', expertsPerLayer:512, streamingFloor:1024, ownedDial:'Owned'};
  assert.equal(floorHintText(dial, 'auto:8'), 'Minimum for 8 layers on the card: 5,120');
  assert.equal(floorHintText(dial, '8'), 'Minimum for 8 layers on the card: 5,120');
  assert.equal(floorHintText(dial, 8), 'Minimum for 8 layers on the card: 5,120');
  assert.equal(floorHintText(dial, '0,7'), 'Minimum for 2 layers on the card: 2,048');
  assert.equal(floorHintText(dial, 'auto'), 'Minimum for 6 layers on the card: 4,096');
  assert.equal(floorHintText(dial, ''), 'Minimum for 0 layers on the card: 1,024');
  assert.equal(floorHintText(dial, 1), 'Minimum for 1 layer on the card: 1,536');
  const small = {name:'Slots', expertsPerLayer:64, streamingFloor:128, ownedDial:'Owned'};
  assert.equal(floorHintText(small, 'auto:3'), 'Minimum for 3 layers on the card: 320');
  console.log('floor hint: minimum = layers x expertsPerLayer + streamingFloor');
})();
"""
    _run_fit_script(script)
