"""The five models of engines/config/config.example.yaml as the import produces them."""

from __future__ import annotations

import copy

FT_DEFAULTS = {"KVCacheTokens": 262208, "KVDtype": "fp8", "MoECacheSize": 5332}

_FIVE = {
    "version": 1,
    "system": {"floorGB": 6, "waitSeconds": 300, "latestWins": True, "defaultIdleMinutes": 0,
               "helperURL": "http://127.0.0.1:2031"},
    "engines": {
        "ninfer": {"defaults": {"max-context": 150000, "kv-capacity": 200000, "max-concurrency": 4,
                                "pending-timeout-ms": 300000, "lm-head-draft": True,
                                "preserve-thinking": True, "vision": True}},
        "freetoken": {"defaults": dict(FT_DEFAULTS)},
    },
    "models": [
        {"id": "qwen3.8-flash", "name": "Qwen3.8 Flash Next NVFP4 (FreeToken)", "engine": "freetoken",
         "runtime": "freetoken", "artifact": "~/models/Qwen3.8-Flash-Next-NVFP4", "ramNeedGB": 58,
         "idleMinutes": 0, "aliases": ["Qwen3.8-Flash-Next-NVFP4"], "overrides": {}, "presets": {},
         "activePreset": None},
        {"id": "qwen3.8-flash-abliterated", "name": "Qwen3.8 Flash Next ABLITERATED NVFP4 (FreeToken, uncensored)",
         "engine": "freetoken", "runtime": "freetoken", "artifact": "~/models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4",
         "ramNeedGB": 58, "idleMinutes": 0, "aliases": ["Qwen3.8-Flash-Next-ABLITERATED-NVFP4"],
         "overrides": {}, "presets": {}, "activePreset": None},
        {"id": "quasar-27b", "name": "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)", "engine": "ninfer",
         "runtime": "ninfer", "artifact": "~/ninfer-work/models/quasar_27b_nvfp4.ninfer", "ramNeedGB": 18,
         "idleMinutes": 0, "aliases": [], "overrides": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 7},
         "presets": {}, "activePreset": None},
        {"id": "fable-27b", "name": "Fable 27B NVFP4 (NInfer)", "engine": "ninfer", "runtime": "ninfer-upstream",
         "artifact": "~/ninfer-work/models/fable_27b_nvfp4.ninfer", "ramNeedGB": 18, "idleMinutes": 0,
         "aliases": [], "overrides": {"kv-dtype": "fp8", "spec": "mtp", "draft-tokens": 4}, "presets": {},
         "activePreset": None},
        {"id": "twin-27b", "name": "Twin 27B NVFP4 (NInfer)", "engine": "ninfer", "runtime": "ninfer-upstream",
         "artifact": "~/ninfer-work/models/twin_nvfp4.ninfer", "ramNeedGB": 18, "idleMinutes": 0,
         "aliases": [], "overrides": {"kv-dtype": "fp8", "spec": "mtp", "draft-tokens": 4}, "presets": {},
         "activePreset": None},
    ],
}


def five() -> dict:
    return copy.deepcopy(_FIVE)
