from __future__ import annotations

from freetoken.daemon.settings.ninfer_fit import GIB, estimate, kv_bytes_per_token, verdict
from freetoken.daemon.settings.registry import effective_settings, find_model
from tests.settings.registry_fixtures import five

QUASAR_BYTES = 19_782_132_224
MEASURED = 29.7 * GIB  # card in use, quasar-27b loaded: kv 200000, int8, dflash2, vision (2026-09-24)


def quasar():
    doc = five()
    return effective_settings(doc, find_model(doc, "quasar-27b"))


def test_quasar_estimate_is_within_10_percent_of_the_measured_boot():
    need = estimate(quasar(), QUASAR_BYTES)["needBytes"]
    print(f"estimate {need / GIB:.2f} GB against measured 29.7 GB")
    assert abs(need - MEASURED) / MEASURED < 0.10


def test_kv_rate_follows_the_precision():
    assert kv_bytes_per_token("bf16") == 65536
    assert kv_bytes_per_token("int8") == 33792
    assert kv_bytes_per_token("fp8") == 33024
    assert kv_bytes_per_token("nvfp4") == 18432
    assert kv_bytes_per_token("k8v4") == 25728


def test_verdicts():
    assert verdict(29 * GIB, 32 * GIB) == "fits"
    assert verdict(31 * GIB, 32 * GIB) == "tight"
    assert verdict(33 * GIB, 32 * GIB) == "wont_fit"
    assert verdict(1, None) == "unknown"


def test_empty_capacity_means_the_longest_chat():
    settings = quasar()
    settings["kv-capacity"] = ""
    assert estimate(settings, QUASAR_BYTES)["kvTokens"] == 150000


def test_fill_the_card_is_capped_by_the_card():
    settings = quasar()
    settings["kv-capacity"] = 0
    result = estimate(settings, QUASAR_BYTES, card_total_bytes=32 * GIB)
    assert result["needBytes"] == 31 * GIB
    assert result["notes"]


def test_turning_off_pictures_and_guess_ahead_lowers_the_estimate():
    full = estimate(quasar(), QUASAR_BYTES)["needBytes"]
    lighter = quasar()
    lighter.update({"vision": False, "spec": "off"})
    assert estimate(lighter, QUASAR_BYTES)["needBytes"] < full


def test_more_concurrent_chats_use_more_card_memory():
    # GDN keeps a fixed-size recurrent-state + conv-history StateImage per active lane
    # (src/core/linear_attention_state.{h,cpp}; slot count = max-concurrency + device-state-slots,
    # src/runtime/engine/engine.cpp:63,75-76), so raising concurrency alone must raise the estimate.
    needs = []
    for concurrency in (1, 2, 4, 8):
        settings = quasar()
        settings["max-concurrency"] = concurrency
        needs.append(estimate(settings, QUASAR_BYTES)["needBytes"])
    assert needs == sorted(needs)
    assert len(set(needs)) == len(needs)


def test_more_device_state_slots_uses_more_card_memory():
    needs = []
    for slots in (0, 4, 8):
        settings = quasar()
        settings["device-state-slots"] = slots
        needs.append(estimate(settings, QUASAR_BYTES)["needBytes"])
    assert needs == sorted(needs)
    assert len(set(needs)) == len(needs)


def test_unknown_kv_dtype_raises_a_plain_error():
    try:
        kv_bytes_per_token("q9")
    except ValueError as exc:
        assert "q9" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
