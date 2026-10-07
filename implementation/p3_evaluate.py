"""Seal both seeds before joining labels; preserve original primary scores."""
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from continual import validate_development_configuration
from evaluate import (METHODS, fraction, is_correct, prediction_transition_audit,
                      resource_summary, score_cases, self_test)
from p3_diagnostics import (eval_only_choice, options_by_id, output_units,
                            read_labels, reward_direction, summarize_formats)
from state import atomic_json


def predictions(rows, mode):
    return [{"id": row["id"], "answer": row[mode]["answer"]} for row in rows]


def revised_vote(row, options):
    counts = Counter(eval_only_choice(text, options) for text in row["completions"])
    counts.pop(None, None)
    return min(counts, key=lambda answer: (-counts[answer], answer)) if counts else None


def available_run(run, manifest, digest, expected_cfg, acceptance):
    result_path = run / "result.json"
    cfg_path = run / "configuration.json"
    if not cfg_path.exists():
        return [], {"status": "not_started", "cursor": 0}, {}
    cfg = json.loads(cfg_path.read_text())
    if cfg != expected_cfg:
        raise ValueError("Run configuration differs from locked seed/method")
    validate_development_configuration(cfg, acceptance, manifest)
    rows = []
    for path in sorted((run / "cases").glob("i*.json")):
        row = json.loads(path.read_text())
        index = len(rows)
        if path.name != f"i{index:06d}.json" or row["index"] != index or row["id"] != manifest["stream"][index]["id"]:
            raise ValueError("Available cases are not the fixed contiguous prefix")
        rows.append(row)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["manifest_sha256"] != digest or not result["reference_unchanged"] or result["cursor"] != len(rows) or result["method"] != cfg["method"]:
            raise ValueError("Result provenance or cursor differs")
    else:
        result = {"status": "incomplete_no_final_checkpoint", "cursor": len(rows)}
    probes = {}
    ids = [entry["id"] for entry in manifest["probe"]]
    for path in sorted((run / "probe").glob("c*.json")):
        record = json.loads(path.read_text())
        if record["cursor"] not in manifest["probe_cursors"] or [x["id"] for x in record["predictions"]] != ids:
            raise ValueError("Probe plan/order differs")
        if not record["rng_and_modes_restored"]:
            raise ValueError("Probe changed RNG or model modes")
        probes[record["cursor"]] = record
    return rows, result, probes


def main():
    self_test()
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip():
        raise ValueError("Manifest changed before scoring")
    manifest = json.loads(raw)
    if manifest["rollout_seeds"] != [43, 44] or manifest["selected_n"] != 16:
        raise ValueError("Unexpected locked matrix")
    acceptance = json.loads((folder / "acceptance.json").read_text())
    controller = json.loads((folder / "controllers/main-plan.json").read_text())
    if "finished_utc" not in controller or any("return_code" not in job for job in controller["jobs"]):
        raise ValueError("Main prediction controller is still active")
    if not json.loads((folder / "owned_cleanup_before_score.json").read_text())["owned_main_processes_ended"]:
        raise ValueError("Owned main processes have not ended")
    options = options_by_id(folder, manifest)
    loaded, units = {}, {}
    for name in ["a"] + [key + str(seed) for seed in [43, 44] for key in "bcd"]:
        cfg_name = "p2_a" if name == "a" else "p3_" + name
        cfg = json.loads((root / "configs" / (cfg_name + ".json")).read_text())
        run = folder / "runs" / ("main-" + name)
        loaded[name] = available_run(run, manifest, digest, cfg, acceptance)
        units[name] = output_units(run, loaded[name][0], options)
    readouts = {}
    for path in sorted((folder / "readout").glob("*.json")):
        record = json.loads(path.read_text())
        if "actor" not in record or "protocol" not in record:
            continue
        if [x["id"] for x in record["predictions"]] != [x["id"] for x in manifest["probe"]]:
            raise ValueError("Readout probe coverage/order differs")
        if record["protocol"]["label_fields_read"] or not record["parameters_unchanged_by_readout"]:
            raise ValueError("Readout entered labels or updated the actor")
        for row in record["predictions"]:
            if set(row["scores"]) != set(options[row["id"]]) or row["answer"] != min(row["scores"], key=lambda k: (-row["scores"][k], k)):
                raise ValueError("Readout does not score every legal identifier")
        readouts[record["actor"]] = record
    # No correctness labels have been opened by this evaluator above this point.
    seal = {"sealed_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": digest,
            "rollout_seeds": [43, 44], "labels_read_before_seal": False,
            "coverage": {name: {"cases": len(rows), "status": result["status"], "probe_cursors": sorted(probes)}
                         for name, (rows, result, probes) in loaded.items()},
            "controller_status": controller["status"], "closed_readout_actors": sorted(readouts)}
    atomic_json(folder / "scores/prediction_seal.json", seal)
    labels = read_labels(root, manifest)
    public = {"manifest_sha256": digest, "data_status": manifest["data_status"],
              "unique_stream_groups": 16, "unique_probe_groups": 16, "seeds": {},
              "prediction_seal": seal, "parser_version": "P2 original primary; marked_choice_v1_eval_only diagnostic",
              "patient_independence_verified": False, "resources": {}, "readout": {}}
    private = {"case_scores": {}, "reward_groups": {}, "gain_sets": {}}
    for seed in [43, 44]:
        names = {"a": "a", **{key: key + str(seed) for key in "bcd"}}
        n = min(len(loaded[name][0]) for name in names.values())
        context = {"matched_prefix_n": n, "actual_coverage": {key: len(loaded[name][0]) for key, name in names.items()},
                   "complete_matrix": n == manifest["selected_n"], "methods": {}}
        public["seeds"][str(seed)] = context
        if n == 0:
            context["unavailable_reason"] = "No completed four-method same-seed prefix"
            continue
        base_rows = loaded["a"][0][:n]
        base_predictions = predictions(base_rows, "before")
        base_right = [is_correct(x["answer"], labels[x["id"]]) for x in base_predictions]
        for key, name in names.items():
            rows, result, probes = loaded[name]
            rows = rows[:n]
            score, private["case_scores"][name] = score_cases(rows, labels, base_right)
            score.update(method=METHODS[key], rollout_seed=seed if key != "a" else None,
                         cached=key == "a", actual_completed=len(loaded[name][0]), probes=[])
            if "before" in rows[0]:
                before = predictions(rows, "before")
                score["history_parse_transition"] = prediction_transition_audit(base_predictions, before, labels)
                private["gain_sets"][name] = [x["id"] for x, base in zip(before, base_right)
                                              if not base and is_correct(x["answer"], labels[x["id"]])]
            if "after" in rows[0]:
                score["current_parse_transition"] = prediction_transition_audit(before, predictions(rows, "after"), labels)
            if key != "a":
                score["reward_direction"], private["reward_groups"][name] = reward_direction(rows, options, labels)
                score["eval_only_consensus"] = fraction(sum(is_correct(revised_vote(row, options[row["id"]]), labels[row["id"]]) for row in rows), n)
            for cursor, probe in sorted(probes.items()):
                preds = probe["predictions"]
                metric = {"cursor": cursor, "correct": sum(is_correct(x["answer"], labels[x["id"]]) for x in preds),
                          "parsed": sum(x["answer"] is not None for x in preds), "n": len(preds)}
                if 0 in probes:
                    metric["transition_from_initial"] = prediction_transition_audit(probes[0]["predictions"], preds, labels)
                score["probes"].append(metric)
            selected_units = [unit for unit in units[name] if unit["kind"] == "probe" or unit["cursor"] <= n]
            score["format"] = summarize_formats(selected_units, labels)
            context["methods"][key] = score
        ttrl = predictions(loaded[names["c"]][0][:n], "before")
        spine = predictions(loaded[names["d"]][0][:n], "before")
        context["spine_vs_ttrl_before"] = prediction_transition_audit(ttrl, spine, labels)
        context["delta_spine_vs_ttrl_before_pp"] = context["spine_vs_ttrl_before"]["gain_pp"]
    public["cross_seed_gain_overlap"] = {}
    for key in "cd":
        left, right = [set(private["gain_sets"].get(key + str(seed), [])) for seed in [43, 44]]
        public["cross_seed_gain_overlap"][key] = {"seed43_gained": len(left), "seed44_gained": len(right),
                                                   "intersection": len(left & right), "union": len(left | right),
                                                   "compared_when_both_seeds_available": all(key + str(seed) in private["gain_sets"] for seed in [43, 44])}
    for name, (rows, result, _) in loaded.items():
        if name == "a":
            public["resources"][name] = {"cached": True, "new_generation_seconds": 0,
                                          "checkpoint_bytes_written": 0, "source_code_commit": result["cache_provenance"]["source_code_commit"]}
        elif result["status"] == "completed":
            public["resources"][name] = resource_summary(folder / "runs" / ("main-" + name), rows, result)
        else:
            public["resources"][name] = {"status": result["status"], "actual_completed_cursor": len(rows)}
    for actor, record in readouts.items():
        preds = record["predictions"]
        public["readout"][actor] = {"correct": sum(is_correct(x["answer"], labels[x["id"]]) for x in preds),
                                     "parsed": len(preds), "n": len(preds), "protocol": record["protocol"],
                                     "elapsed_seconds": record["elapsed_seconds"], "provenance": record["provenance"],
                                     "peak_allocated_bytes": record["peak_allocated_bytes"],
                                     "peak_reserved_bytes": record["peak_reserved_bytes"]}
        if actor != "frozen_base" and "frozen_base" in readouts:
            public["readout"][actor]["transition_vs_frozen_readout"] = prediction_transition_audit(readouts["frozen_base"]["predictions"], preds, labels)
    atomic_json(folder / "scores/private_paired_scores.json", private)
    atomic_json(folder / "scores/public_summary.json", public)
    print(json.dumps({"scored_after_global_seal": True,
                      "matched_prefix_by_seed": {s: x["matched_prefix_n"] for s, x in public["seeds"].items()},
                      "closed_readout_actors": sorted(readouts), "primary_parser_unchanged": True}))


if __name__ == "__main__":
    main()
