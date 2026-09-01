from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

ROOT = Path(r"D:\FreeToken-ple-mmap-vision\.local\mtp-spike\prototypes")


def _load_summary():
    path = ROOT / "summarize_mtp_fast_verify.py"
    spec = importlib.util.spec_from_file_location("summarize_mtp_fast_verify", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle():
    categories = ("coding", "reasoning", "prose", "long-chat")
    samplings = ("greedy", "normal")
    temperatures = ("cold", "warm")
    placements = ("bf16", "nvfp4")
    baseline = {
        "runs": [
            {"input_tokens": 512, "output_tokens": 512, "output_tokens_per_second": rate}
            for rate in (49.0, 50.0, 51.0)
        ]
    }
    def rng_transition(stream, cycle, draws):
        before = f"{stream}-state-{0 if draws == 0 else cycle}"
        after = before if draws == 0 else f"{stream}-state-{cycle + 1}"
        return {
            "stream": stream,
            "request_uid": 77,
            "cycle_index": cycle,
            "draw_count_before": cycle * draws,
            "draw_count_after": (cycle + 1) * draws,
            "draws": draws,
            "state_sha256_before": before,
            "state_sha256_after": after,
            "greedy": draws == 0,
            "default_rng_unchanged": True,
        }

    requests = []
    for category in categories:
        for sampling in samplings:
            for temperature in temperatures:
                key = f"{category}:{sampling}:{temperature}"
                common = {
                    "request_key": key,
                    "category": category,
                    "sampling": sampling,
                    "temperature_state": temperature,
                    "request_body_sha256": f"body-{category}-{sampling}",
                    "response_sha256": f"response-{key}",
                    "response_schema_sha256": "ordinary-schema",
                    "usage": {"prompt_tokens": 80, "completion_tokens": 4},
                    "status": 200,
                    "response_has_private_keys": False,
                    "ordinary_execution": "eager",
                }
                for placement in placements:
                    for phase in ("compare-a", "compare-b", "fast-eager", "fast-graph"):
                        requests.append(
                            {
                                **common,
                                "phase": "mtp-off",
                                "placement": "off",
                                "control_for": phase,
                                "control_placement": placement,
                            }
                        )
                        requests.append(
                            {**common, "phase": phase, "placement": placement}
                        )

    matrix = []
    for placement in placements:
        for category in categories:
            for sampling in samplings:
                for temperature in temperatures:
                    request_key = f"{category}:{sampling}:{temperature}"
                    for replay in ("a", "b"):
                        for depth in (1, 2, 3):
                            matrix.append(
                                {
                                    "record_type": "comparison",
                                    "phase": f"compare-{replay}",
                                    "replay": replay,
                                    "request_key": request_key,
                                    "placement": placement,
                                    "category": category,
                                    "sampling": sampling,
                                    "temperature_state": temperature,
                                    "depth": depth,
                                    "oracle_matches": True,
                                    "graph_matches": True,
                                    "normal_support_order_matches": True,
                                    "state_digest_unchanged": True,
                                    "pages_conserved": True,
                                    "recurrent_slots_conserved": True,
                                    "draft_tokens": [11, 12, 13][:depth],
                                    "draft_logits_sha256": ["d1", "d2", "d3"][:depth],
                                    "eager_target_logits_sha256": f"target-{depth}",
                                    "graph_target_logits_sha256": f"target-{depth}",
                                    "accepted_tokens": depth,
                                    "graph_supported": True,
                                    "target_expert_temperature": {
                                        "state": temperature,
                                        "pair_index": 0,
                                        "residency_reset": temperature == "cold",
                                    },
                                    "draft_rng_steps": [
                                        rng_transition(
                                            "draft",
                                            index,
                                            0 if sampling == "greedy" else 1,
                                        )
                                        for index in range(depth)
                                    ],
                                    "acceptance_rng": rng_transition(
                                        f"acceptance-depth-{depth}",
                                        0,
                                        0 if sampling == "greedy" else 2 * depth + 1,
                                    ),
                                    "timing": {
                                        "proposal_required_wall_ms": 11.0,
                                        "proposal_instrumentation_wall_ms": 0.2,
                                        "proposal_required_synchronizations": 4,
                                        "proposal_instrumentation_synchronizations": 0,
                                        "target_required_wall_ms": 2.0,
                                        "target_core_cuda_ms": 1.5,
                                        "target_required_synchronizations": 1,
                                        "target_instrumentation_wall_ms": 0.2,
                                        "target_instrumentation_synchronizations": 2,
                                        "state_required_wall_ms": 1.0,
                                        "state_required_scope": "scratch-backup-through-cleanup",
                                        "state_required_synchronizations": 2,
                                        "state_instrumentation_wall_ms": 0.2,
                                        "state_instrumentation_synchronizations": 0,
                                        "acceptance_required_wall_ms": 1.0,
                                        "acceptance_required_synchronizations": 0,
                                        "acceptance_instrumentation_wall_ms": 0.2,
                                        "acceptance_instrumentation_synchronizations": 0,
                                    },
                                    "expert_movement": {
                                        "available": True,
                                        "active_experts": 4,
                                        "hit_experts": 2,
                                        "missing_experts": 2,
                                        "fetched_experts": 2,
                                        "cpu_experts": 0,
                                        "d2d_rows": 0,
                                        "bytes_per_expert": 64,
                                        "h2d_bytes": 128,
                                        "d2d_bytes": 0,
                                        "transfer_bytes": 128,
                                        "movement_reconciled": True,
                                    },
                                }
                            )
                    for verifier in ("fast-eager", "fast-graph"):
                        for depth in (1, 2, 3):
                            for cycle in (0, 1):
                                components = {
                                    "P": 10.0 if cycle == 0 else 0.0,
                                    "D": 1.0,
                                    "V": 2.0,
                                    "S": 1.0,
                                    "A": 1.0,
                                    "E": 1 + depth,
                                }
                                matrix.append(
                                    {
                                        "record_type": "performance",
                                        "phase": verifier,
                                        "request_key": request_key,
                                        "placement": placement,
                                        "verifier": verifier,
                                        "category": category,
                                        "sampling": sampling,
                                        "temperature_state": temperature,
                                        "depth": depth,
                                        "cycle": cycle,
                                        "components": components,
                                        "component_total_ms": sum(
                                            components[name] for name in "PDVSA"
                                        ),
                                        "projection_eligible": True,
                                        "component_reconciled": True,
                                        "target_expert_temperature": {
                                            "state": temperature,
                                            "pair_index": 0,
                                            "residency_reset": temperature == "cold",
                                        },
                                        "timing": {
                                            "proposal_required_wall_ms": components["P"] + components["D"],
                                            "proposal_instrumentation_wall_ms": 0.2,
                                            "proposal_required_synchronizations": 4,
                                            "proposal_instrumentation_synchronizations": 0,
                                            "target_required_wall_ms": components["V"],
                                            "target_core_cuda_ms": 1.5,
                                            "target_required_synchronizations": 1,
                                            "target_instrumentation_wall_ms": 0.2,
                                            "target_instrumentation_synchronizations": 2,
                                            "state_required_wall_ms": components["S"],
                                            "state_required_scope": "scratch-backup-through-cleanup",
                                            "state_required_synchronizations": 2,
                                            "state_instrumentation_wall_ms": 0.2,
                                            "state_instrumentation_synchronizations": 0,
                                            "acceptance_required_wall_ms": components["A"],
                                            "acceptance_required_synchronizations": 0,
                                            "acceptance_instrumentation_wall_ms": 0.2,
                                            "acceptance_instrumentation_synchronizations": 0,
                                        },
                                        "rng": {
                                            "draft": [
                                                rng_transition(
                                                    "draft",
                                                    cycle * 3 + index,
                                                    0 if sampling == "greedy" else 1,
                                                )
                                                for index in range(depth)
                                            ],
                                            "acceptance": rng_transition(
                                                f"acceptance-depth-{depth}",
                                                cycle,
                                                0 if sampling == "greedy" else 2 * depth + 1,
                                            ),
                                        },
                                        "state_digest_unchanged": True,
                                        "pages_conserved": True,
                                        "recurrent_slots_conserved": True,
                                        "failure": None,
                                        "rejected_cycle": False,
                                        "expert_movement": {
                                            "available": True,
                                            "active_experts": 4,
                                            "hit_experts": 2,
                                            "missing_experts": 2,
                                            "fetched_experts": 2,
                                            "cpu_experts": 0,
                                            "d2d_rows": 0,
                                            "bytes_per_expert": 64,
                                            "h2d_bytes": 128,
                                            "d2d_bytes": 0,
                                            "transfer_bytes": 128,
                                            "movement_reconciled": True,
                                        },
                                        "graph_support": {
                                            "status": "captured",
                                            "memory_bytes": 1024,
                                        } if verifier == "fast-graph" else None,
                                    }
                                )
    return {
        "schema_version": 3,
        "run_id": "contract-run",
        "baseline": baseline,
        "correctness_gate": {
            "passed": True,
            "request_count": 1,
            "candidate_base_url": "http://127.0.0.1:2030",
            "normal_sampling": True,
            "response_has_private_keys": False,
            "cycles": 2,
        },
        "requests": requests,
        "matrix": matrix,
        "required_categories": list(categories),
        "required_placements": list(placements),
    }


def test_strict_summary_emits_only_the_fixed_eligible_verdict_on_full_pass():
    summary = _load_summary().summarize_bundle(_bundle())

    assert summary["schema_version"] == 3
    assert summary["verdict"] == "ELIGIBLE_FOR_SEPARATE_FULL_SERVING_DESIGN"
    assert summary["baseline"]["median_output_tokens_per_second"] == 50.0
    assert summary["speed"]["best_normal_warm_median_projected_512"] > 50.0
    assert set(summary["speed"]["categories"]) == {
        "coding",
        "reasoning",
        "prose",
        "long-chat",
    }
    assert summary["scope"] == "component projection, not end-to-end speculative serving"


def test_strict_summary_stops_when_correct_but_not_faster():
    bundle = _bundle()
    for run in bundle["baseline"]["runs"]:
        run["output_tokens_per_second"] = 5000.0

    summary = _load_summary().summarize_bundle(bundle)

    assert summary["verdict"] == "STOP_NOT_FAST_ENOUGH"
    assert summary["correctness_and_safety"]["passed"] is True


def test_summary_requires_advancing_replayable_rng_and_exact_timing():
    module = _load_summary()
    for mutation in ("rng", "timing", "execution", "gate"):
        bundle = copy.deepcopy(_bundle())
        if mutation == "rng":
            row = next(
                row
                for row in bundle["matrix"]
                if row["record_type"] == "performance" and row["sampling"] == "normal"
            )
            row["rng"]["acceptance"]["state_sha256_after"] = row["rng"][
                "acceptance"
            ]["state_sha256_before"]
        elif mutation == "timing":
            row = next(
                row for row in bundle["matrix"] if row["record_type"] == "performance"
            )
            del row["timing"]["target_instrumentation_wall_ms"]
        elif mutation == "execution":
            next(
                request for request in bundle["requests"] if request["phase"] == "mtp-off"
            )["ordinary_execution"] = "graph"
        else:
            bundle["correctness_gate"]["passed"] = False

        summary = module.summarize_bundle(bundle)

        assert summary["verdict"] == "STOP_INCORRECT_OR_UNSAFE"
        assert summary["correctness_and_safety"]["passed"] is False


def test_strict_summary_fails_closed_for_mismatch_leak_or_component_omission():
    module = _load_summary()
    for mutation in ("oracle", "response", "component"):
        bundle = copy.deepcopy(_bundle())
        if mutation == "oracle":
            next(row for row in bundle["matrix"] if row["record_type"] == "comparison")[
                "oracle_matches"
            ] = False
        elif mutation == "response":
            next(row for row in bundle["requests"] if row["phase"] == "fast-eager")[
                "response_sha256"
            ] = "changed"
        else:
            row = next(row for row in bundle["matrix"] if row["record_type"] == "performance")
            del row["components"]["V"]

        summary = module.summarize_bundle(bundle)

        assert summary["verdict"] == "STOP_INCORRECT_OR_UNSAFE"
        assert summary["correctness_and_safety"]["passed"] is False
