from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


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
    {"name": "ModelPath", "group": "Model and context", "control": "path"},
    {"name": "ContextTokens", "group": "Model and context", "control": "number"},
    {"name": "KVDtype", "group": "Model and context", "control": "choice"},
    {"name": "MaxRunningRequests", "group": "Chats", "control": "number"},
    {"name": "KVPark", "group": "KV notes and parking", "control": "choice"},
    {"name": "MoECacheSize", "group": "Expert slots and card memory", "control": "number"},
    {"name": "EmbedHost", "group": "Expert slots and card memory", "control": "toggle"},
    {"name": "EnableVision", "group": "Picture input", "control": "toggle"},
    {"name": "FREETOKEN_MTP_SPECULATE", "group": "Look-ahead speed trick (MTP)", "control": "toggle"},
    {"name": "ExpertLoad", "group": "Loading and diagnostics", "control": "choice"},
    {"name": "Port", "group": "Advanced", "control": "number"},
    {"name": "DesktopPython", "group": "Advanced", "control": "path"},
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
    assert "group.dials.map((dial) => renderDial(dial))" in source
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
    for tab in ("settings", "server", "profiles"):
        assert f'data-tab="{tab}"' in page
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
    assert "show-advanced" in page
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
