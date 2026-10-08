"""Offline old-pool gate and sealed main comparison for fixed-reference rewards."""
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import torch
from continual import group_rewards
from evaluate import is_correct, fraction, prediction_transition_audit, score_cases, resource_summary
from p3_diagnostics import read_labels, options_by_id, output_units, summarize_formats
from p3_evaluate import available_run
from state import atomic_json
from core import extract_answer


def reward_audit(rows, choices, labels, targets=None):
    counts = {"groups": len(rows), "correct_parsed": 0, "correct_negative": 0,
              "positive_reward": 0, "wrong_positive": 0, "target_correct": 0,
              "all_invalid": 0, "zero_advantage": 0}
    for row in rows:
        target = None if targets is None else targets[row["id"]]
        rewards, advantages, vote = group_rewards(row["completions"], choices[row["id"]], target)
        counts["all_invalid"] += int(vote["all_unparseable"])
        winner = vote["winner"] if target is None else target
        counts["target_correct"] += int(is_correct(winner, labels[row["id"]]))
        if rewards is None:
            continue
        counts["zero_advantage"] += int(vote["zero_advantage_group"])
        for answer, reward, advantage in zip(vote["answers"], rewards.tolist(), advantages.tolist()):
            correct = is_correct(answer, labels[row["id"]])
            counts["correct_parsed"] += int(correct)
            counts["correct_negative"] += int(correct and advantage < 0)
            counts["positive_reward"] += int(reward > 0)
            counts["wrong_positive"] += int(reward > 0 and answer is not None and not correct)
    return {"counts": counts, "correct_negative": fraction(counts["correct_negative"], counts["correct_parsed"]),
            "wrong_positive": fraction(counts["wrong_positive"], counts["positive_reward"]),
            "target_accuracy": fraction(counts["target_correct"], counts["groups"])}


def gate_passes(reference_correct, majority, reference):
    m, v = majority, reference
    return (reference_correct >= 3 and v["correct_negative"]["percent"] is not None
            and v["wrong_positive"]["percent"] is not None
            and v["correct_negative"]["percent"] < m["correct_negative"]["percent"]
            and v["wrong_positive"]["percent"] <= m["wrong_positive"]["percent"] - 10)


def stable_development_success(seeds, expected_seeds=(45, 46, 47)):
    return len(seeds) == 3 and {s["seed"] for s in seeds} == set(expected_seeds) and all(s["candidate_before_correct"] > s["frozen_before_correct"]
                                  and s["candidate_before_correct"] > s["control_before_correct"]
                                  and s["candidate_initial_correct_retained"] == 3
                                  and s["n"] == 16 for s in seeds)


def checked_signal(folder, manifest, digest):
    record = json.loads((folder / "reference/stream_targets.json").read_text())
    if record["labels_read"] or record["manifest_sha256"] != digest or not record["parameters_unchanged"] or not record["rng_and_modes_preserved"]:
        raise ValueError("Reference signal provenance differs")
    if [x["id"] for x in record["predictions"]] != [x["id"] for x in manifest["stream"]]:
        raise ValueError("Reference signal scope/order differs")
    for prediction in record["predictions"]:
        scores = prediction["scores"]
        entry = next(e for e in manifest["stream"] if e["id"] == prediction["id"])
        options = json.loads((folder / entry["input"]).read_text())["options"]
        if set(scores) != set(options) or not all(math.isfinite(v) for v in scores.values()):
            raise ValueError("Reference scores are not finite legal-option scores")
        if prediction["winner"] != min(scores, key=lambda k: (-scores[k], k)):
            raise ValueError("Reference winner differs from locked maximum")
    return record


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip():
        raise ValueError("Manifest changed")
    manifest = json.loads(raw)
    p5 = manifest.get("p5_version") == 1
    p6 = manifest.get("p6_version") == 1
    p7 = manifest.get("p7_version") == 1
    no_reference = p5 or p6 or p7
    seeds = [54, 55, 56] if p7 else ([51, 52, 53] if p6 else ([48, 49, 50] if p5 else [45, 46, 47]))
    if manifest["rollout_seeds"] != seeds:
        raise ValueError("Unexpected locked seed matrix")
    signal = None if no_reference else checked_signal(folder, manifest, digest)
    targets = None if no_reference else {p["id"]: p["winner"] for p in signal["predictions"]}
    choices = options_by_id(folder, manifest)
    if os.environ["P2_MODE"] == "p4_audit":
        completion = json.loads((folder / "reference/completion.json").read_text())
        if completion["coverage"] != {"stream": 16, "probe": 16}:
            raise ValueError("Reference stage is not closed")
        old = Path(os.environ["P4_P3_FOLDER"])
        rows = [json.loads(p.read_text()) for name in ["c43", "d43", "c44", "d44"]
                for p in sorted((old / "runs" / ("main-" + name) / "cases").glob("i*.json"))]
        if len(rows) != 64:
            raise ValueError("Old rollout pool incomplete")
        expected_ids = [e["id"] for e in manifest["stream"]]
        if any([r["id"] for r in rows[start:start+16]] != expected_ids for start in range(0,64,16)):
            raise ValueError("Old rollout pool scope/order differs")
        labels = read_labels(root, manifest)
        majority = reward_audit(rows, choices, labels)
        reference = reward_audit(rows, choices, labels, targets)
        correct = sum(is_correct(p["winner"], labels[p["id"]]) for p in signal["predictions"])
        decision = {"scored_utc": datetime.now(timezone.utc).isoformat(),
                    "reference_stream_correct": correct, "reference_stream_n": 16,
                    "majority_old_pool": majority, "reference_old_pool": reference,
                    "go": gate_passes(correct, majority, reference),
                    "only_previously_observed_development_labels": True,
                    "clinical_labels_enter_trainer": False, "thresholds_fixed_before_scoring": True}
        atomic_json(folder / "reference/audit_public.json", decision)
        print(json.dumps(decision))
        return
    controller = json.loads((folder / "controllers/main-plan.json").read_text())
    if "finished_utc" not in controller or controller["status"] != "all_planned_jobs_completed" or len(controller["jobs"]) != 6 or any(j.get("return_code") != 0 for j in controller["jobs"]):
        raise ValueError("Main matrix not closed successfully")
    if not json.loads((folder / "owned_cleanup_before_score.json").read_text())["owned_main_processes_ended"]:
        raise ValueError("Owned workers not verified ended")
    if not no_reference and not json.loads((folder / "reference/audit_public.json").read_text())["go"]:
        raise ValueError("Reference gate did not pass")
    acceptance = json.loads((folder / "acceptance.json").read_text())
    loaded = {}
    for name in ["a"] + [k + str(seed) for seed in seeds for k in "mv"]:
        cfgname = "p2_a" if name == "a" else ("p7_" if p7 else ("p6_" if p6 else ("p5_" if p5 else "p4_"))) + name
        cfg = json.loads((root / "configs" / (cfgname + ".json")).read_text())
        run = folder / "runs" / ("main-" + name)
        rows, result, probes = available_run(run, manifest, digest, cfg, acceptance)
        if len(rows) != 16 or result["status"] != "completed" or set(probes) != {0, 16}:
            raise ValueError("Incomplete fixed main matrix")
        if not no_reference and name.startswith("v") and json.loads((run / "reference_signal_snapshot.json").read_text()) != signal:
            raise ValueError("Candidate reward signal differed between runs")
        if p5 and name.startswith("v"):
            probability = json.loads((run / "probability_check.json").read_text())
            if not probability["raw_sampler_configuration_matches_check"] or probability["max_logp_difference"] > cfg["probability_check_bf16_atol"]:
                raise ValueError("Actual-model raw sampling probability check failed")
        for row in rows:
            for phase in ["before", "after"]:
                if phase in row and row[phase]["answer"] != extract_answer(row[phase]["text"], choices[row["id"]]):
                    raise ValueError("Stored primary parsing differs")
            target = targets[row["id"]] if not no_reference and name.startswith("v") else None
            if name != "a":
                rewards, advantages, vote = group_rewards(row["completions"], choices[row["id"]], target)
                if (None if rewards is None else rewards.tolist()) != row["rewards"]:
                    raise ValueError("Stored rewards differ from declared source")
                if not no_reference and name.startswith("v") and row["vote"]["reward_target"] != target:
                    raise ValueError("Stored target differs")
        loaded[name] = rows, result, probes
    expected_initial = loaded["a"][2][0]["predictions"]
    for name, (_, _, probes) in loaded.items():
        if probes[0]["predictions"] != expected_initial:
            raise ValueError("Initial fixed probe predictions differ from frozen cache")
    seal = {"sealed_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": digest,
            "seeds": seeds, "main_labels_read_before_seal": False, "coverage": {k: len(v[0]) for k,v in loaded.items()}}
    atomic_json(folder / "scores/prediction_seal.json", seal)
    labels = read_labels(root, manifest)
    base = [{"id": r["id"], "answer": r["before"]["answer"]} for r in loaded["a"][0]]
    base_right = [is_correct(p["answer"], labels[p["id"]]) for p in base]
    public = {"scored_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": digest,
              "evaluator_commit": os.environ["P2_EVAL_CODE_COMMIT"], "prediction_seal": seal,
              "unique_stream_groups": 16, "unique_probe_groups": 16, "data_status": manifest["data_status"],
              "reference_audit": None if no_reference else json.loads((folder / "reference/audit_public.json").read_text()),
              "sampling_variants": manifest.get("sampling_variants"),
              "learning_rate_variants": manifest.get("learning_rate_variants"),
              "kl_variants": manifest.get("kl_variants"),
              "seeds": {}, "resources": {}}
    criteria = []
    for seed in seeds:
        context = {}
        for kind in "mv":
            name = kind + str(seed);rows,result,probes = loaded[name]
            metric, _ = score_cases(rows, labels, base_right)
            before = [{"id": r["id"], "answer": r["before"]["answer"]} for r in rows]
            after = [{"id": r["id"], "answer": r["after"]["answer"]} for r in rows]
            metric["history_parse_transition"] = prediction_transition_audit(base, before, labels)
            metric["current_parse_transition"] = prediction_transition_audit(before, after, labels)
            initial, final = [probes[c]["predictions"] for c in [0,16]]
            metric["probe"] = {"initial_correct": sum(is_correct(p["answer"], labels[p["id"]]) for p in initial),
                              "final_correct": sum(is_correct(p["answer"], labels[p["id"]]) for p in final),
                              "initial_parsed": sum(p["answer"] is not None for p in initial),
                              "final_parsed": sum(p["answer"] is not None for p in final),
                              "transition": prediction_transition_audit(initial, final, labels)}
            metric["reward_direction"] = reward_audit(rows, choices, labels, targets if not no_reference and kind=="v" else None)
            if p5 and kind=="v":
                metric["sampling_probability_check"] = json.loads((folder / "runs" / ("main-"+name) / "probability_check.json").read_text())
            metric["format"] = summarize_formats(output_units(folder / "runs" / ("main-" + name), rows, choices), labels)
            context[kind] = metric
            public["resources"][name] = resource_summary(folder / "runs" / ("main-" + name), rows, result)
        left = [{"id": r["id"], "answer": r["before"]["answer"]} for r in loaded["m"+str(seed)][0]]
        right = [{"id": r["id"], "answer": r["before"]["answer"]} for r in loaded["v"+str(seed)][0]]
        context["paired_candidate_vs_control"] = prediction_transition_audit(left,right,labels)
        public["seeds"][str(seed)] = context
        criteria.append({"seed": seed, "n":16, "frozen_before_correct":sum(base_right),
                         "control_before_correct":context["m"]["greedy_before"]["correct"],
                         "candidate_before_correct":context["v"]["greedy_before"]["correct"],
                         "candidate_initial_correct_retained":context["v"]["probe"]["transition"]["correct_to_correct"]})
    public["success_checks"] = criteria
    public["stable_positive_development_result"] = stable_development_success(criteria, seeds)
    public["independent_generalization_established"] = False
    atomic_json(folder / "scores/public_summary.json", public)
    print(json.dumps({"scored_after_matrix_seal": True, "stable_positive_development_result":public["stable_positive_development_result"]}))


if __name__ == "__main__":
    main()
