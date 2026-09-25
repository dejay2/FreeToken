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


# ---- startup refusal check (fit round 2, measured live 2026-09-25, RTX 5090, WSL) ----
from freetoken.daemon.settings.ninfer_fit import startup_room  # noqa: E402

CARD_5090 = 34_190_917_632  # cudaMemGetInfo total
REFUSED_NEEDS = 13_177_821_184  # "requested Engine runtime reservation requires ..." at mc6
REFUSED_AVAILABLE = 13_111_561_216  # "... only ... bytes are available for runtime capacity"


def quasar_at(concurrency, dtype="int8", context=150000):
    settings = quasar()
    settings.update({"max-concurrency": concurrency, "kv-dtype": dtype, "max-context": context})
    return estimate(settings, QUASAR_BYTES, card_total_bytes=CARD_5090)


def test_room_matches_what_ninfer_reported_at_the_refusal():
    assert startup_room(QUASAR_BYTES, CARD_5090) == REFUSED_AVAILABLE


def test_mc6_int8_reproduces_the_refusal():
    result = quasar_at(6)
    assert result["runtimeReservationBytes"] == REFUSED_NEEDS
    assert result["startupVerdict"] == "wont_fit"
    assert result["startupMessage"].startswith("NInfer would refuse to start. It needs to set aside 12.3 GB")
    assert "fewer chats" in result["startupMessage"]


def test_mc4_and_mc5_start_and_match_the_logged_runtime():
    # capacity | ... runtime 10.6 GiB (mc4) / 11.4 GiB (mc5), int8, at both 150000 and 100000.
    for concurrency, logged, expected in ((4, 10.6, "fits"), (5, 11.4, "tight")):
        for context in (150000, 100000):
            result = quasar_at(concurrency, context=context)
            assert abs(result["runtimeReservationBytes"] / GIB - logged) <= 0.1
            assert result["startupVerdict"] == expected


def test_fp8_plans_slightly_less_and_starts():
    int8, fp8 = quasar_at(4), quasar_at(4, "fp8")
    assert abs(fp8["runtimeReservationBytes"] / GIB - 10.4) <= 0.1  # logged runtime 10.4 GiB
    assert 0 < int8["runtimeReservationBytes"] - fp8["runtimeReservationBytes"] < 0.3 * GIB
    assert fp8["startupVerdict"] == "fits"
    assert quasar_at(5, "fp8")["startupVerdict"] != "wont_fit"


def test_reservation_does_not_depend_on_max_context():
    assert quasar_at(4)["runtimeReservationBytes"] == quasar_at(4, context=100000)["runtimeReservationBytes"]


def test_used_memory_tracks_the_measured_card_after_start():
    # nvidia-smi card in use after start (MiB), 2026-09-25; includes that day's ~2.6 GB desktop.
    for concurrency, dtype, mib in ((3, "int8", 30077), (4, "int8", 30484), (5, "int8", 30892), (4, "fp8", 30354)):
        need = quasar_at(concurrency, dtype)["needBytes"]
        assert abs(need - mib * 1024 ** 2) / (mib * 1024 ** 2) < 0.05


def test_no_card_reading_leaves_the_startup_check_unknown():
    result = estimate(quasar(), QUASAR_BYTES)
    assert result["startupVerdict"] == "unknown" and result["runtimeRoomBytes"] is None
    assert result["runtimeReservationBytes"] > 0


def test_fill_the_card_only_refuses_when_the_minimum_does_not_fit():
    settings = quasar()
    settings["kv-capacity"] = 0
    assert estimate(settings, QUASAR_BYTES, card_total_bytes=CARD_5090)["startupVerdict"] == "fits"
    assert estimate(settings, QUASAR_BYTES, card_total_bytes=24 * GIB)["startupVerdict"] == "wont_fit"


def test_a_larger_live_desktop_shrinks_the_room_and_a_smaller_one_does_not():
    base = startup_room(QUASAR_BYTES, CARD_5090)
    assert startup_room(QUASAR_BYTES, CARD_5090, int(1.9 * GIB)) == base
    assert startup_room(QUASAR_BYTES, CARD_5090, 3 * GIB) == base - (3 * GIB - int(2.6 * GIB))
    settings = quasar()
    settings["max-concurrency"] = 4
    assert estimate(settings, QUASAR_BYTES, card_total_bytes=CARD_5090,
                    desktop_bytes=5 * GIB)["startupVerdict"] == "wont_fit"


def test_a_bigger_prefill_chunk_raises_the_reservation():
    small, big = quasar(), quasar()
    small["prefill-chunk"], big["prefill-chunk"] = 512, 2048
    at_1024 = estimate(quasar(), QUASAR_BYTES)["runtimeReservationBytes"]
    assert estimate(small, QUASAR_BYTES)["runtimeReservationBytes"] == at_1024
    assert estimate(big, QUASAR_BYTES)["runtimeReservationBytes"] > at_1024


# ---- upstream runtime (Fable, Twin), measured live 2026-09-25, RTX 5090, WSL ----
# engines/ninfer-upstream ninfer-serve run by hand with the config.yaml command line (kv-capacity
# 200000, fp8, mtp draft-tokens 4, lm-head-draft, vision, max-context 150000) plus
# --request-log-jsonl; exact bytes from its startup "memory" record and the refusal line.
from freetoken.daemon.settings.ninfer_fit import runtime_reservation  # noqa: E402
from freetoken.daemon.settings import ninfer_dials  # noqa: E402

FABLE_BYTES = 21_500_080_900
TWIN_BYTES = 21_492_920_836
UPSTREAM_AVAILABLE = 10_948_182_016  # available_after_weights_bytes, both models, every run
UPSTREAM_RESERVATION = {4: 9_523_597_057, 6: 10_337_772_033, 8: 11_151_947_009}  # mc8 = refusal


def upstream(model_id, concurrency):
    doc = five()
    model = find_model(doc, model_id)
    settings = effective_settings(doc, model)
    settings["max-concurrency"] = concurrency
    return settings, model["runtime"]


def test_upstream_reservation_matches_the_measured_bytes():
    for model_id in ("fable-27b", "twin-27b"):
        for concurrency, measured in UPSTREAM_RESERVATION.items():
            settings, runtime = upstream(model_id, concurrency)
            assert runtime == "ninfer-upstream"
            parts = runtime_reservation(ninfer_dials.normalized(settings), 3125, runtime)
            assert parts["total"] == measured
    # kv_payload_bytes at mc4 and the MTP graph allowance (82 MiB x 4), to the byte.
    parts = runtime_reservation(ninfer_dials.normalized(upstream("fable-27b", 4)[0]), 3125, "ninfer-upstream")
    assert parts["kv"] == 7_018_128_384 and parts["graphs"] == 343_932_928


def test_upstream_room_is_within_8_mb_and_never_above_fable():
    fable = startup_room(FABLE_BYTES, CARD_5090, runtime="ninfer-upstream")
    twin = startup_room(TWIN_BYTES, CARD_5090, runtime="ninfer-upstream")
    assert twin == UPSTREAM_AVAILABLE
    assert 0 <= UPSTREAM_AVAILABLE - fable < 8 * 1024 ** 2


def test_upstream_mc4_mc6_start_and_mc8_refuses():
    for model_id, size in (("fable-27b", FABLE_BYTES), ("twin-27b", TWIN_BYTES)):
        verdicts = {}
        for concurrency in (4, 6, 8):
            settings, runtime = upstream(model_id, concurrency)
            verdicts[concurrency] = estimate(settings, size, card_total_bytes=CARD_5090,
                                             runtime=runtime)["startupVerdict"]
        # Logged planned slack: mc4 1,424,584,959 B and mc6 610,409,983 B (both started), mc8
        # refused. The upstream margin is 1.0 GiB (STARTUP_TIGHT_MARGIN): the shipped mc4 fits, mc6
        # (582 MiB slack) is tight.
        assert verdicts == {4: "fits", 6: "tight", 8: "wont_fit"}


def test_quasar_runtime_default_is_unchanged():
    assert quasar_at(6)["runtimeReservationBytes"] == REFUSED_NEEDS
    assert estimate(quasar(), QUASAR_BYTES, card_total_bytes=CARD_5090, runtime="ninfer") == \
        estimate(quasar(), QUASAR_BYTES, card_total_bytes=CARD_5090)
