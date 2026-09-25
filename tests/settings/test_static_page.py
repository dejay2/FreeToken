from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
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
    assert parser.external_assets == ["/panel.js", "/playground.js"], "the page loads only its own panel.js and playground.js"
    return "\n".join(parser.scripts)


def test_static_page_renders_fixture_dials_from_metadata() -> None:
    source = _source()

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


def test_save_goes_through_the_panel_and_marks_bad_values() -> None:
    source = _source()
    panel_js = (PAGE.parent / "panel.js").read_text(encoding="utf-8")
    assert "$('save').addEventListener('click', () => panelSave());" in source
    assert "method: 'PUT'" in panel_js and "response.status === 422" in panel_js
    assert "showFieldErrors(body.detail)" in panel_js
    assert "data-error=\"${escapeHtml(dial.name)}\"" in source


def test_status_and_logs_stay_and_the_old_controls_are_gone() -> None:
    page = _page()
    source = _source()
    for route in ("/api/status", "/api/logs"):
        assert route in source
    assert "auto-follow" in page
    for gone in ("serverAction(", "/api/server/", "/api/profiles", 'id="profile-strip"'):
        assert gone not in page, gone
    assert "window.prompt" not in source and "window.confirm" not in source and "window.alert" not in source


def test_page_has_tabs_info_buttons_sliders_and_browse() -> None:
    page = _page()
    source = _source()

    assert 'role="tablist"' in page
    assert [m.group(1) for m in re.finditer(r'<button[^>]+data-main="([^"]+)"', page)] == ["models", "system", "ninfer", "freetoken", "test"]
    assert "renderTabs(groups)" in source
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
    assert "restart-ask" in page
    assert "changedNames()" in source


def test_page_reads_the_model_and_reshapes_itself() -> None:
    page = _page()
    source = _source()

    # A model card above the settings, filled from the settings payload's model block.
    assert 'id="model-card"' in page
    assert "renderModelCard()" in source
    # Text-stored counts (layers kept on the card) go through the storedAs contract.
    assert "dial.storedAs" in source
    assert "dial.storedZero" in source


def test_dead_model_folder_code_is_gone():
    """Stage A deferred minor: the panel never shows the model-folder dial (ModelPath is set per
    model), so the folder preview (refreshModel) never ran, and the hidden "Model downloads and
    folders" box still called /api/models on every page load."""
    page = _page()
    source = _source()
    for dead in ("refreshModel", "scheduleModelRefresh", "'/api/models'", "models-tools", "model-folders",
                 "dial.browse === 'model'", "/api/settings?model="):
        assert dead not in source and dead not in page, dead


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


def test_prompt_cache_form_preserves_exact_advanced_request_and_shared_boundary():
    _run_fit_script(r"""
element('cache-mode').value = 'json';
element('cache-format').value = 'anthropic';
element('cache-name').value = 'agent-test';
element('cache-prefix').value = '512';
element('cache-ttl').value = '300';
const raw = {model:'served', system:'instructions', messages:[{role:'user',content:'test'}],
  tools:[{name:'lookup',input_schema:{type:'object'}}], thinking:{type:'disabled'}, max_tokens:10};
element('cache-request').value = JSON.stringify(raw);
assert.deepEqual(cacheRegistration(), {name:'agent-test', format:'anthropic', request:raw,
  prefix_tokens:512, ttl_seconds:300});
element('cache-prefix').value = '';
assert.equal(Object.hasOwn(cacheRegistration(), 'prefix_tokens'), false);
element('cache-request').value = '[]';
assert.throws(cacheRegistration, /JSON object/);
element('cache-mode').value = 'simple';
element('cache-prompt').value = 'Shared text';
element('cache-task').value = 'A private task';
cacheUI.model = 'served';
assert.deepEqual(cacheRegistration().request.messages, [
  {role:'system',content:'Shared text'}, {role:'user',content:'A private task'}]);
element('cache-prefix').value = '1.5';
assert.throws(cacheRegistration, /whole number/);
""")


def test_prompt_cache_rejects_bad_success_and_preserves_form_on_failure():
    _run_fit_script(r"""
(async () => {
  element('cache-prompt').value = 'Keep my draft';
  global.fetch = async () => ({ok:true, status:200, json:async()=>({status:'ok',result:{}})});
  await loadPromptCache();
  assert.equal(cacheUI.connected, false);
  assert.match(element('cache-connection').textContent, /invalid|unexpected/i);
  global.fetch = async () => ({ok:false,status:400,json:async()=>({error:'prefix exceeds pool'})});
  await cacheAction(async()=>{await cacheCall('/prefixes', 'POST', {});});
  assert.match(element('cache-message').textContent, /prefix exceeds pool/);
  assert.equal(element('cache-prompt').value, 'Keep my draft');
  assert.equal(cacheUI.busy, false);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_prompt_cache_treats_queued_warming_as_success():
    _run_fit_script(r"""
(async () => {
  global.fetch = async () => ({ok:true,status:202,json:async()=>({status:'warming',result:{state:'warming'}})});
  assert.equal((await cacheCall('/prefixes/agent/warm', 'POST')).status, 'warming');
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_recent_prompt_selection_preserves_payload_and_resets_old_boundary():
    _run_fit_script(r"""
const request = {model:'agent-model',system:'shared',messages:[{role:'user',content:'task'}],
  tools:[{name:'read',input_schema:{type:'object'}}],thinking:{type:'disabled'},max_tokens:42};
element('cache-prefix').value='24960';
element('cache-ttl').value='300';
applyRecentPrompt({id:'0123456789abcdef0123456789abcdef',format:'anthropic',request});
assert.equal(element('cache-mode').value,'json');
assert.equal(element('cache-format').value,'anthropic');
assert.equal(element('cache-prefix').value,'');
assert.deepEqual(cacheRegistration().request,request);
assert.equal(requests.length,0,'Selecting a request never registers or generates implicitly');
assert.match(element('cache-message').textContent,/system prompt/i);
assert.equal(cacheRegistration().prefix_scope,'system');
""")


def test_selected_prompt_shows_system_and_tools_separately_from_user_text():
    _run_fit_script(r"""
const request={model:'pi',messages:[{role:'system',content:[{type:'text',text:'<rules>\nAll instructions'}]},
  {role:'user',content:'hi'}],tools:[{type:'function',function:{name:'read',parameters:{type:'object'}}}]};
element('cache-ttl').value='300';
applyRecentPrompt({id:'0123456789abcdef0123456789abcdef',format:'openai',request});
assert.match(element('cache-system-preview').textContent, /<rules>\nAll instructions/);
assert.match(element('cache-tools-preview').textContent, /read/);
assert.equal(element('cache-scope').value,'system');
assert.deepEqual(cacheRegistration().request,request);
assert.equal(cacheRegistration().prefix_tokens,undefined);
element('cache-scope').value='request';
assert.equal(cacheRegistration().prefix_scope,undefined);
applyRecentPrompt({id:'0123456789abcdef0123456789abcdef',format:'openai',request:{messages:[{role:'user',content:'hi'}]}});
assert.equal(element('cache-scope').value,'request');
assert.match(element('cache-system-preview').textContent,/No leading system/i);
""")


def test_old_model_server_cannot_silently_register_system_scope_as_whole_request():
    _run_fit_script(r"""
(async()=>{
  element('cache-ttl').value='300';
  applyRecentPrompt({id:'0123456789abcdef0123456789abcdef',format:'openai',
    request:{messages:[{role:'system',content:'rules'},{role:'user',content:'hi'}]}});
  cacheUI.data={prefixes:[]};
  await registerPrompt();
  assert.equal(requests.length,0);
  assert.match(element('cache-message').textContent,/Update and restart/);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_recent_prompt_list_escapes_content_and_expired_selection_keeps_draft():
    _run_fit_script(r"""
(async()=>{
  recentUI.data={enabled:true,capacity:50,stored_bytes:123,max_bytes:16777216,ttl_seconds:3600,skipped_count:0,
    prompts:[{id:'0123456789abcdef0123456789abcdef',format:'openai',model:'local',preview:'<img src=x onerror=alert(1)>',message_count:2,received_at:'2026-09-14T12:00:00Z'}]};
  renderRecentPrompts();
  assert.equal(element('cache-recent-list').innerHTML.includes('<img'),false);
  assert.match(element('cache-recent-list').innerHTML,/&lt;img/);
  element('cache-request').value='keep draft';
  global.fetch=async()=>({ok:false,status:404,json:async()=>({error:'This request has expired or was cleared.'})});
  await useRecentPrompt('0123456789abcdef0123456789abcdef');
  assert.equal(element('cache-request').value,'keep draft');
  assert.match(element('cache-message').textContent,/expired/);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_malformed_recent_selection_keeps_every_editor_field():
    _run_fit_script(r"""
const fields=['cache-mode','cache-format','cache-request','cache-name','cache-prefix'];
fields.forEach(id=>{element(id).value='existing '+id});
for(const id of [undefined,42,'bad/id']) {
  assert.throws(()=>applyRecentPrompt({id,format:'openai',request:{messages:[]}}),/invalid/i);
  fields.forEach(id=>assert.equal(element(id).value,'existing '+id));
}
""")


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
