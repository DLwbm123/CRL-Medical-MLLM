"""Seal engineering acceptance from completed tests; never opens answer labels."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from state import atomic_json


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    tests = {key: json.loads((folder / (key + ".json")).read_text()) for key in [
        "cpu_acceptance", "core_acceptance", "evaluator_acceptance", "recovery_acceptance"]}
    if not all(test["passed"] for test in tests.values()):
        raise AssertionError("An engineering test has not passed")
    run_ids = json.loads(os.environ.get("P2_ENGINEERING_RUNS", '{"a":"eng2-a-full","b":"eng2-b-full","c":"eng2-c-full","d":"eng2-d-full"}'))
    expected = {"a": "Frozen greedy", "b": "Frozen SC-8", "c": "TTRL", "d": "SPINE"}
    if set(run_ids) != set(expected):
        raise AssertionError("All four engineering methods are required")
    manifest_digest = (folder / "manifest_engineering.sha256").read_text().strip()
    reports = {}
    initial_probes, shared_config = [], None
    for key, run_id in run_ids.items():
        run = folder / "runs" / run_id
        result = json.loads((run / "result.json").read_text())
        config = json.loads((run / "configuration.json").read_text())
        common = {name: value for name, value in config.items() if name != "method"}
        if shared_config is not None and common != shared_config:
            raise AssertionError("Configurations differ beyond the declared method")
        shared_config = common
        initial_probes.append(json.loads((run / "probe/c000000.json").read_text())["predictions"])
        if result["status"] != "completed" or result["cursor"] != 2 or result["method"] != expected[key] or not result["reference_unchanged"] or result["manifest_sha256"] != manifest_digest:
            raise AssertionError("Incomplete or mismatched actual-model acceptance run")
        cases = [json.loads((run / "cases" / f"i{i:06d}.json").read_text()) for i in range(2)]
        checks = []
        for case in cases:
            opt = case.get("optimization")
            if not opt:
                continue
            if not opt["masters_persisted"] or not opt["moments_persisted"]:
                raise AssertionError("Persistent-state check failed")
            checks.append({"cursor": case["index"] + 1,
                           "vision_gradient_nonzero": opt["gradient_norms"]["vision"] > 0,
                           "language_gradient_nonzero": opt["gradient_norms"]["language"] > 0,
                           "tracked_actor_changed": all(x["changed_elements"] > 0 for x in opt["tracked_actor_updates"].values()),
                           "behavior_logp_recompute_error": opt["max_behavior_recompute_error"]})
        if key in {"c", "d"} and not any(check["tracked_actor_changed"] and check["vision_gradient_nonzero"] and check["language_gradient_nonzero"] for check in checks):
            raise AssertionError("Actual-model test did not verify an effective parameter update")
        reports[key] = {"method": expected[key], "cursor": result["cursor"], "updates": result["optimizer_updates"],
                        "reference_unchanged": result["reference_unchanged"], "code_commit": result["code_commit"],
                        "update_checks": checks, "wall_seconds": result["elapsed_seconds"],
                        "case_seconds": [case["elapsed_seconds"] for case in cases],
                        "phase_seconds": result["phase_seconds"],
                        "peak_allocated_bytes": result["peak_allocated_bytes"],
                        "peak_cpu_rss_kib": result["cpu_peak_rss_kib"],
                        "checkpoint_bytes": result["checkpoint"]["state_bytes"]}
    if not all(probe == initial_probes[0] for probe in initial_probes):
        raise AssertionError("Fresh base-model probe predictions differ between methods")
    probability = json.loads((folder / "runs" / run_ids["d"] / "probability_check.json").read_text())
    if probability["max_logp_difference"] > probability["atol"]:
        raise AssertionError("Actual probability check failed")
    acceptance = {"passed": True, "sealed_utc": datetime.now(timezone.utc).isoformat(),
                  "tests": tests, "actual_methods": reports, "probability": probability,
                  "initial_probe_predictions_identical": True, "configurations_match_except_method": True,
                  "validated_configuration_without_method": shared_config,
                  "clinical_accuracy_scored": False, "labels_read": False,
                  "isolation": "data-flow validation, not an OS sandbox"}
    atomic_json(folder / "acceptance.json", acceptance)
    print(json.dumps({"passed": True, "actual_methods": list(reports), "labels_read": False}))


if __name__ == "__main__":
    main()
