"""NInfer settings catalogue for the control panel (own model system part 2).

Every ``ninfer-serve`` option of the two frozen runtimes -- engines/ninfer (the QUASAR-capable
mobile fork, d4bc75db) and engines/ninfer-upstream (f76e19c0) -- in the Dial shape the settings
page already renders for FreeToken. Limits and built-in defaults are copied from the runtimes'
own parser: src/serve/serve_options.{h,cpp}, src/product/speculative_options.h and
include/ninfer/types.h at the frozen commits. tests/settings/test_ninfer_dials.py checks this
list against those sources, and against each built binary's --help where one exists.

A value equal to the runtime's built-in default is left off the command line. An empty value
("") means "not set": the runtime, or for answer style the model itself, decides. Host, port and
model id are fixed by the config generator (swap_config.py) and never shown.

Ruling F13 (control panel stage A): a setting that only one runtime supports (here,
``chat-template``, upstream-only) still keeps its runtime marker in this catalogue -- the
catalogue itself never rejects it. ``validate`` only raises a runtime-support error when a
caller passes ``runtime=`` explicitly (as model-override/preset validation will in Tasks 2/10);
engine *defaults* are validated with ``runtime=None`` (the default), which skips that check
entirely, and ``settings_for(runtime)``/``dial_dicts(..., runtime)``/``render_flags`` simply
leave the setting out of a runtime that does not carry it rather than rejecting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .dials import Dial

RUNTIMES = ("ninfer", "ninfer-upstream")
RUNTIME_LABELS = {"ninfer": "QUASAR runtime", "ninfer-upstream": "upstream runtime"}
FIXED_FLAGS = ("--host", "--port", "--model-id")
# The context-cache options; --no-prefix-reuse refuses them (serve_options.cpp), so the
# generator leaves them out when reuse is off.
CONTEXT_CACHE = (
    "host-kv-mib", "host-state-slots", "device-state-slots",
    "max-private-continuations", "max-shared-prefixes", "max-long-anchors-per-continuation",
)
# speculative_options.h: mtp takes 1..5 draft tokens, dflash and dflash2 1..15.
SPEC_DRAFT_LIMIT = {"mtp": 5, "dflash": 15, "dflash2": 15}

MC, MEM, GA, AS, PIC, ADV = "Model & chats", "Memory", "Guess-ahead", "Answer style", "Pictures", "Advanced"
GROUP_INFO = {
    MC: "How long chats can be and how many run at once.",
    MEM: "Chat memory on the graphics card and in PC memory.",
    GA: "Guessing several words at once so answers type faster.",
    AS: "How varied answers are, and the thinking step.",
    PIC: "Reading pictures in chats.",
    ADV: "Logging, limits and switches that rarely need changing.",
}


@dataclass(frozen=True)
class NinferSetting:
    dial: Dial
    runtimes: tuple[str, ...] = RUNTIMES
    clamp: str = ""  # the request field this range also clamps (generated clampParams)
    step: int = 0  # the value must be a multiple of this

    @property
    def name(self) -> str:
        return self.dial.name

    @property
    def flag(self) -> str:
        return "--" + self.dial.name

    @property
    def builtin(self) -> Any:
        return self.dial.default

    @property
    def is_switch(self) -> bool:
        return self.dial.control == "toggle"


def _s(name, control, default, unit, group, plain, blurb, info="", *, runtimes=RUNTIMES, clamp="", step=0, **extra):
    dial = Dial(
        name, control, default, unit, blurb, group,
        source="ninfer", engine_mapping=f"--{name}", plain=plain, blurb=blurb, info=info or blurb, **extra,
    )
    return NinferSetting(dial, runtimes=runtimes, clamp=clamp, step=step)


CATALOGUE: tuple[NinferSetting, ...] = (
    # ---- Model & chats ----
    _s("max-context", "number", 8192, "tokens", MC, "Longest chat",
       "The most tokens one chat can hold. A token is about three quarters of a word.",
       "Each chat may grow to this many tokens, question and answers together. QUASAR, Fable and "
       "Twin run 150,000 on this PC.",
       minimum=1, maximum=1048576, slider=(8192, 262144, 1024), effects=("vram:up",)),
    _s("kv-capacity", "number", "", "tokens", MC, "Shared chat memory",
       "Card memory for all chats together, in tokens. Empty: the same as the longest chat.",
       "Room on the graphics card that all running chats share. It must be at least the longest "
       "chat. 'Fill the card' lets the engine use whatever is left, keeping 1 GB spare. Measured "
       "on this PC: 200,000 tokens at the compact int8 precision take about 6.3 GB.",
       minimum=0, maximum=4194304, slider=(8192, 524288, 8192), auto_value=0, auto_label="Fill the card",
       effects=("vram:up", "speed:up")),
    _s("max-concurrency", "number", 1, "chats", MC, "Chats at the same time",
       "How many chats the model answers at once. Each one gets a little slower.",
       minimum=1, maximum=8, slider=(1, 8, 1), effects=("vram:up",)),
    _s("max-pending-requests", "number", 16, "chats", MC, "Chats that may wait",
       "How many more chats may queue while the others are answered.",
       minimum=1, maximum=1024, advanced=True),
    _s("pending-timeout-ms", "number", 30000, "ms", MC, "Longest wait in the queue",
       "A waiting chat gives up after this long.",
       minimum=1, maximum=3600000, display_unit="seconds", display_factor=1000),
    _s("prefill-chunk", "number", 1024, "tokens", MC, "Reading step",
       "How many prompt tokens are read in one go. Must be a multiple of 128.",
       minimum=128, maximum=65536, step=128, advanced=True, effects=("speed:mixed", "vram:up")),
    _s("default-max-tokens", "number", 8192, "tokens", MC, "Longest answer",
       "The answer limit when the app does not set one.",
       minimum=1, maximum=1048576),
    # ---- Memory ----
    _s("kv-dtype", "choice", "bf16", "", MEM, "Chat memory precision",
       "Smaller settings fit longer chats on the card, at a small cost in accuracy.",
       "Card memory per token of chat on a 27B model: full 64 KB, int8 33 KB, fp8 32 KB, "
       "nvfp4 18 KB, mixed 25 KB. QUASAR runs int8 and Fable and Twin fp8 on this PC.",
       options=("bf16", "int8", "fp8", "nvfp4", "k8v4"),
       option_labels=("Full (bf16)", "Compact (int8)", "Compact (fp8)", "Smallest (nvfp4)", "Mixed (fp8 + nvfp4)"),
       effects=("vram:mixed", "accuracy:mixed")),
    _s("host-kv-mib", "number", 8192, "MiB", MEM, "PC memory for parked chats",
       "Finished chats are kept here so they pick up quickly next time.",
       minimum=0, maximum=262144, display_unit="GB", display_factor=1024, effects=("ram:up", "speed:up")),
    _s("host-state-slots", "number", 8, "chats", MEM, "Parked chats in PC memory",
       "How many finished chats are kept for a quick restart.",
       minimum=0, maximum=256, advanced=True),
    _s("device-state-slots", "number", "", "chats", MEM, "Extra chat checkpoints on the card",
       "Empty: one per chat at the same time.", minimum=0, maximum=64, advanced=True),
    _s("max-request-mib", "number", 384, "MiB", MEM, "Largest request",
       "Requests bigger than this many MB are refused before they are read.",
       minimum=1, maximum=4096, advanced=True),
    _s("max-private-continuations", "number", "", "", MEM, "Kept chat endings",
       "Empty: twice the chats at the same time.", minimum=0, maximum=1024, advanced=True),
    _s("max-shared-prefixes", "number", "", "", MEM, "Shared chat beginnings",
       "Empty: the chats at the same time, and at least 4.", minimum=0, maximum=1024, advanced=True),
    _s("max-long-anchors-per-continuation", "number", "", "", MEM, "Long-chat bookmarks",
       "Empty: 2 per chat.", minimum=0, maximum=64, advanced=True),
    # ---- Guess-ahead ----
    _s("spec", "choice", "off", "", GA, "Guess-ahead method",
       "Guess several words ahead and check them in one go, so answers type faster.",
       "The model file must carry the chosen helper. QUASAR uses DFlash2; Fable and Twin use MTP.",
       options=("off", "mtp", "dflash", "dflash2"), option_labels=("Off", "MTP", "DFlash", "DFlash2"),
       effects=("speed:up", "vram:up")),
    _s("draft-tokens", "number", "", "words", GA, "Words guessed ahead",
       "MTP takes 1 to 5; DFlash and DFlash2 take 1 to 15.", minimum=1, maximum=15, slider=(1, 15, 1)),
    _s("lm-head-draft", "toggle", False, "switch", GA, "Quick guessing head",
       "A lighter guessing step. Usually faster."),
    # ---- Answer style ----
    _s("temperature", "number", "", "", AS, "Adventurousness",
       "Empty: the model's own. 0 always picks the likeliest word; higher is more varied.",
       minimum=0, maximum=2, numeric_kind="float", slider=(0, 2, 0.05), clamp="temperature"),
    _s("top-p", "number", "", "", AS, "Word pool (share)",
       "Empty: the model's own. Only the likeliest words adding up to this share are considered.",
       minimum=0, maximum=1, numeric_kind="float", slider=(0, 1, 0.01), clamp="top_p"),
    _s("top-k", "number", "", "words", AS, "Word pool (count)",
       "Empty: the model's own. Only this many likeliest words are considered; 0 means no limit.",
       minimum=0, maximum=20, slider=(0, 20, 1), clamp="top_k"),
    _s("min-p", "number", "", "", AS, "Least likely word allowed",
       "Empty: the model's own. Words much less likely than the best one are skipped.",
       minimum=0, maximum=1, numeric_kind="float", clamp="min_p", advanced=True),
    _s("presence-penalty", "number", "", "", AS, "Avoid repeating topics",
       "Empty: the model's own. Higher moves on to new topics sooner.",
       minimum=-2, maximum=2, numeric_kind="float", clamp="presence_penalty", advanced=True),
    _s("frequency-penalty", "number", "", "", AS, "Avoid repeating words",
       "Empty: the model's own. Higher repeats the same words less.",
       minimum=-2, maximum=2, numeric_kind="float", clamp="frequency_penalty", advanced=True),
    _s("seed", "number", "", "", AS, "Fixed dice",
       "Empty: random. A number makes the same question give the same answer.",
       minimum=0, maximum=18446744073709551615, advanced=True),
    _s("greedy", "toggle", False, "switch", AS, "Always the likeliest word",
       "The same as adventurousness 0.", advanced=True),
    _s("no-thinking", "toggle", False, "switch", AS, "Answer without thinking",
       "Skip the thinking step unless the app asks for it."),
    _s("default-thinking-budget", "number", "", "tokens", AS, "Thinking limit",
       "Empty: no limit. Caps the thinking when thinking is on.", minimum=1, maximum=4294967295),
    _s("preserve-thinking", "toggle", False, "switch", AS, "Remember earlier thinking",
       "Keep the model's thinking from earlier turns in the chat."),
    # ---- Pictures ----
    _s("vision", "toggle", False, "switch", PIC, "Pictures",
       "Let chats include pictures. Uses some extra card memory.", effects=("vram:up",)),
    _s("media-cache-mib", "number", 1024, "MiB", PIC, "Picture reuse memory",
       "Pictures kept for reuse; 0 turns reuse off.", minimum=0, maximum=65536,
       display_unit="GB", display_factor=1024),
    _s("media-live-mib", "number", 2048, "MiB", PIC, "Picture working memory",
       "The limit for pictures being read at once.", minimum=1, maximum=65536,
       display_unit="GB", display_factor=1024),
    _s("media-preprocess-threads", "number", 0, "workers", PIC, "Picture preparation workers",
       "Automatic uses up to 16.", minimum=0, maximum=64, auto_value=0, auto_label="Automatic", advanced=True),
    # ---- Advanced ----
    _s("no-cuda-graph", "toggle", False, "switch", ADV, "Turn off speed recordings",
       "Slower. Only for troubleshooting.", effects=("speed:down", "vram:down")),
    _s("no-prefix-reuse", "toggle", False, "switch", ADV, "Turn off chat reuse",
       "Re-read every chat from the start. Also turns off parked chats."),
    _s("log-stats-interval-ms", "number", 5000, "ms", ADV, "Speed log interval",
       "0 turns the regular speed lines off.", minimum=0, maximum=3600000,
       display_unit="seconds", display_factor=1000),
    _s("response-store-max-records", "number", 1024, "answers", ADV, "Stored answers (count)",
       "How many answers are kept for apps that ask for them later.", minimum=1, maximum=1000000),
    _s("response-store-max-mib", "number", 256, "MiB", ADV, "Stored answers (size)",
       "How much memory the stored answers may use, in MB.", minimum=1, maximum=65536),
    _s("request-log-jsonl", "path", "", "", ADV, "Request log file",
       "Empty: no request log.", browse="file"),
    _s("log-level", "choice", "info", "", ADV, "Log detail", "How much the engine writes to its log.",
       options=("trace", "debug", "info", "warning", "error", "critical", "off")),
    _s("cors", "toggle", False, "switch", ADV, "Allow browser pages",
       "Only needed when a web page talks to the engine directly."),
    _s("context-cost-presets", "path", "", "", ADV, "Context cost file",
       "Empty: none. A file of chat-length cost presets.", browse="file"),
    _s("chat-template", "path", "", "", ADV, "Chat template file",
       "Upstream runtime only. Empty: the model's own.", browse="file", runtimes=("ninfer-upstream",)),
    _s("api-key", "text", "", "", ADV, "Engine password",
       "Leave empty. The switcher does not send it, so a password stops chats through the one address.",
       "The switcher shows each model's start command in its status list, so a password here is "
       "also visible there."),
    # serve_options.cpp:260 takes any non-negative --device and only fails later, at CUDA
    # start-up, after the switcher has waited for memory. The serving PC has one card (RTX
    # 5090), and the switcher runs one model at a time on it, so 0 is the only value that can
    # start; the old 0..7 range let a typo save and then fail at every load.
    _s("device", "number", 0, "", ADV, "Graphics card number",
       "Which card to use. This PC has one graphics card, so this stays 0.", minimum=0, maximum=0),
)

BY_NAME: dict[str, NinferSetting] = {setting.name: setting for setting in CATALOGUE}
BUILTINS: dict[str, Any] = {setting.name: setting.builtin for setting in CATALOGUE}


def settings_for(runtime: str | None) -> tuple[NinferSetting, ...]:
    return tuple(setting for setting in CATALOGUE if runtime is None or runtime in setting.runtimes)


def group_list() -> list[dict[str, str]]:
    return [{"name": name, "plain": name, "info": info} for name, info in GROUP_INFO.items()]


def _plain_number(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def clamp_params() -> dict[str, list]:
    return {
        setting.clamp: [_plain_number(setting.dial.minimum), _plain_number(setting.dial.maximum)]
        for setting in CATALOGUE
        if setting.clamp
    }


def canonical(setting: NinferSetting, value: Any) -> Any:
    """Canonical stored value; raises ValueError with a plain reason."""
    control = setting.dial.control
    if value is None:
        value = ""
    if control == "toggle":
        if isinstance(value, bool):
            return value
        raise ValueError("must be on or off")
    if control == "choice":
        if value not in (setting.dial.options or ()):
            raise ValueError("must be one of " + ", ".join(setting.dial.options or ()))
        return value
    if control in ("path", "text"):
        if not isinstance(value, str):
            raise ValueError("must be text")
        if any(char in value for char in "\x00\r\n"):
            raise ValueError("must be one line")
        return value.strip() if control == "path" else value
    if value == "":
        if setting.builtin == "":
            return ""
        raise ValueError("needs a number")
    if isinstance(value, bool):
        raise ValueError("must be a number")
    # float()/int() raise with Python's own words ("invalid literal for int() with base 10"),
    # which reached the page through validate(); plain words instead (stage A deferred minor).
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("must be a number") from None
    if not math.isfinite(number):
        raise ValueError("must be a number")
    if setting.dial.numeric_kind == "float":
        return number
    if not number.is_integer():
        raise ValueError("must be a whole number")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(number)


def canonical_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {name: canonical(BY_NAME[name], value) for name, value in settings.items()}


def complete(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {**BUILTINS, **{name: value for name, value in settings.items() if name in BY_NAME}}


def normalized(settings: Mapping[str, Any]) -> dict[str, Any]:
    """What the engine will actually run with: switched-off features drop their options."""
    full = complete(settings)
    if full["spec"] == "off":
        full["draft-tokens"] = BUILTINS["draft-tokens"]
        full["lm-head-draft"] = BUILTINS["lm-head-draft"]
    if full["no-prefix-reuse"]:
        for name in CONTEXT_CACHE:
            full[name] = BUILTINS[name]
    return full


def _fmt(value: Any) -> str:
    value = _plain_number(value)
    return f"{value:,}" if isinstance(value, int) else str(value)


def validate(settings: Mapping[str, Any], runtime: str | None = None, *, cross: bool = True) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    values: dict[str, Any] = {}
    for name, raw in settings.items():
        setting = BY_NAME.get(name)
        if setting is None:
            errors.append({"field": name, "message": f"Unknown NInfer setting {name}"})
            continue
        plain = setting.dial.plain
        try:
            value = canonical(setting, raw)
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append({"field": name, "message": f"{plain}: {exc}"})
            continue
        is_auto = setting.dial.auto_value is not None and value == setting.dial.auto_value
        if setting.dial.control == "number" and value != "" and not is_auto:
            low, high = setting.dial.minimum, setting.dial.maximum
            if low is not None and value < low:
                errors.append({"field": name, "message": f"{plain}: at least {_fmt(low)}"})
            if high is not None and value > high:
                errors.append({"field": name, "message": f"{plain}: at most {_fmt(high)}"})
            if setting.step and value % setting.step:
                errors.append({"field": name, "message": f"{plain}: must be a multiple of {setting.step}"})
        if runtime is not None and runtime not in setting.runtimes and value != setting.builtin:
            only = RUNTIME_LABELS[setting.runtimes[0]]
            errors.append({"field": name, "message": f"{plain}: only the {only} has this setting"})
        values[name] = value
    if errors or not cross:
        return errors
    full = complete(values)
    capacity, longest = full["kv-capacity"], full["max-context"]
    if isinstance(capacity, int) and capacity != 0 and capacity < longest:
        errors.append({"field": "kv-capacity",
                       "message": f"Shared chat memory must be at least the longest chat ({longest:,} tokens)."})
    spec, drafts = full["spec"], full["draft-tokens"]
    if spec != "off":
        top = SPEC_DRAFT_LIMIT[spec]
        if drafts == "":
            errors.append({"field": "draft-tokens", "message": f"Guess-ahead {spec.upper()} needs words guessed ahead (1 to {top})."})
        elif drafts > top:
            errors.append({"field": "draft-tokens", "message": f"{spec.upper()} takes 1 to {top} words guessed ahead."})
    return errors


def render_flags(settings: Mapping[str, Any], runtime: str, *, path: Callable[[str], str] = lambda p: p) -> list[str]:
    """The ninfer-serve flags for these settings: only what differs from the runtime's own."""
    full = normalized(settings)
    out: list[str] = []
    for setting in settings_for(runtime):
        value = full[setting.name]
        if value == "" or value == setting.builtin:
            continue
        if setting.is_switch:
            if value is True:
                out.append(setting.flag)
            continue
        if setting.name == "kv-capacity" and value == 0:
            out += [setting.flag, "auto"]
            continue
        if setting.dial.control == "path":
            out += [setting.flag, path(value)]
            continue
        out += [setting.flag, str(_plain_number(value)) if isinstance(value, (int, float)) else str(value)]
    return out


def parse_flags(args: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
    """Read ninfer-serve flags back into settings; host/port/model-id go to the second dict."""
    settings: dict[str, Any] = {}
    fixed: dict[str, str] = {}
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in FIXED_FLAGS:
            if index + 1 >= len(args):
                raise ValueError(f"{arg} needs a value")
            fixed[arg] = args[index + 1]
            index += 2
            continue
        setting = BY_NAME.get(arg[2:]) if arg.startswith("--") else None
        if setting is None:
            raise ValueError(f"unknown NInfer option {arg}")
        if setting.is_switch:
            settings[setting.name] = True
            index += 1
            continue
        if index + 1 >= len(args):
            raise ValueError(f"{arg} needs a value")
        raw = args[index + 1]
        if setting.name == "kv-capacity" and raw == "auto":
            settings[setting.name] = 0
        elif setting.dial.control == "number":
            settings[setting.name] = float(raw) if setting.dial.numeric_kind == "float" else int(raw)
        else:
            settings[setting.name] = raw
        index += 2
    return settings, fixed


def dial_dicts(values: Mapping[str, Any], runtime: str | None = None) -> list[dict[str, Any]]:
    return [setting.dial.as_dict(values.get(setting.name, setting.builtin)) for setting in settings_for(runtime)]


__all__ = [
    "BUILTINS", "BY_NAME", "CATALOGUE", "CONTEXT_CACHE", "FIXED_FLAGS", "GROUP_INFO", "NinferSetting",
    "RUNTIMES", "RUNTIME_LABELS", "canonical", "canonical_settings", "clamp_params", "complete",
    "dial_dicts", "group_list", "normalized", "parse_flags", "render_flags", "settings_for", "validate",
]
