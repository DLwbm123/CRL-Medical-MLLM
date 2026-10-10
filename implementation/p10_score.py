"""Offline gate and complete-matrix scoring of previously observed groups."""
import hashlib
import json
import os
from pathlib import Path
from continual import group_rewards
from core import extract_answer
from evaluate import fraction, is_correct, prediction_transition_audit, resource_summary, score_cases
from p3_diagnostics import options_by_id, output_units, read_labels, summarize_formats
from p3_evaluate import available_run
from p4_evaluate import reward_audit
from state import atomic_json
from verified_reward import checked_targets, qualification_gate, verified_group_rewards


def verified_audit(rows, choices, labels, targets):
    counts = dict(groups=len(rows), correct_parsed=0, correct_negative=0, positive_reward=0, wrong_positive=0,
                  abstained=0, zero_advantage=0, minority_rescued=0, minority_available=0)
    for row in rows:
        rewards, advantages, vote = verified_group_rewards(row["completions"], choices[row["id"]], targets[row["id"]])
        counts["abstained"] += int(targets[row["id"]] is None)
        counts["zero_advantage"] += int(vote["zero_advantage_group"])
        available = any(is_correct(a, labels[row["id"]]) for a in vote["answers"]) and not is_correct(vote["winner"], labels[row["id"]])
        counts["minority_available"] += int(available)
        counts["minority_rescued"] += int(available and is_correct(targets[row["id"]], labels[row["id"]]))
        if rewards is not None:
            for answer, reward, advantage in zip(vote["answers"], rewards.tolist(), advantages.tolist()):
                correct = is_correct(answer, labels[row["id"]])
                counts["correct_parsed"] += int(correct)
                counts["correct_negative"] += int(correct and advantage < 0)
                counts["positive_reward"] += int(reward > 0)
                counts["wrong_positive"] += int(reward > 0 and not correct)
    return {"counts": counts, "correct_negative": fraction(counts["correct_negative"], counts["correct_parsed"]),
            "wrong_positive": fraction(counts["wrong_positive"], counts["positive_reward"])}


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip():
        raise ValueError("Locked manifest differs")
    manifest = json.loads(raw)
    signal = json.loads((folder / "reference/targets.json").read_text())
    targets = checked_targets(signal, manifest, digest)
    choices = options_by_id(folder, manifest)
    if os.environ["P2_MODE"] == "gate":
        if not json.loads((folder / "qualifier_owned_ended.json").read_text())["owned_main_processes_ended"]:
            raise ValueError("Frozen teacher must end before label scoring")
        source = Path(os.environ["P10_SOURCE_ROOT"])
        rows = []
        for campaign in manifest["source_campaigns"]:
            old = source / "outputs" / campaign
            old_manifest = json.loads((old / "manifest.json").read_text())
            old_digest = (old / "manifest.sha256").read_text().strip()
            controller = json.loads((old / "controllers/main-plan.json").read_text())
            if controller["status"] != "all_planned_jobs_completed" or len(controller["jobs"]) != 6 or any(j["return_code"] != 0 for j in controller["jobs"]):
                raise ValueError("Old candidate trajectories have not all closed")
            if [e["id"] for e in old_manifest["stream"]] != [e["id"] for e in manifest["stream"]]:
                raise ValueError("Legacy candidate pool scope differs")
            for seed in old_manifest["rollout_seeds"]:
                for arm in ["m", "v"]:
                    run = old / "runs" / f"main-{arm}{seed}"
                    result = json.loads((run / "result.json").read_text())
                    if result["status"] != "completed" or result["cursor"] != 16 or result["manifest_sha256"] != old_digest:
                        raise ValueError("Old rollout result seal differs")
                    data = [json.loads(p.read_text()) for p in sorted((run / "cases").glob("i*.json"))]
                    if len(data) != 16 or [r["id"] for r in data] != [e["id"] for e in manifest["stream"]]:
                        raise ValueError("All 18 sealed old candidate trajectories are required")
                    for row in data:
                        if row["vote"]["answers"] != [extract_answer(t, choices[row["id"]]) for t in row["completions"]]:
                            raise ValueError("Old strict candidate parsing differs")
                    rows += data
        if len(rows) != 288:
            raise ValueError("Candidate pool must contain 18 x 16 groups")
        labels = read_labels(root, manifest)
        majority, verified = reward_audit(rows, choices, labels), verified_audit(rows, choices, labels, targets)
        accepted = sum(t is not None for t in targets.values())
        correct = sum(is_correct(t, labels[k]) for k, t in targets.items())
        decision = qualification_gate(majority, verified, correct, accepted, verified["counts"]["minority_rescued"])
        public = {**decision, "accepted": accepted, "correct_targets": correct, "stream_n": 16,
                  "candidate_trajectory_count": 18, "repeated_group_observations": len(rows), "unique_group_count": 16,
                  "majority": majority, "verified": verified, "independent_generalization_established": False,
                  "labels_used_only_after_teacher_closed": True, "clinical_labels_enter_reward_targets": False,
                  "teacher_probe_accepted": sum(p["target"] is not None for p in signal["probe"]),
                  "teacher_probe_correct": sum(is_correct(p["target"], labels[p["id"]]) for p in signal["probe"])}
        atomic_json(folder / "scores/qualification.json", public)
        teacher_rows = []
        for section in ["stream", "probe"]:
            for index, row in enumerate(signal[section]):
                teacher_rows.append({"section": section, "anonymous_index": index,
                    "accepted": row["target"] is not None,
                    "accepted_target_correct": bool(row["target"] is not None and is_correct(row["target"], labels[row["id"]])),
                    "first_order_parsed": row["judgments"][0] is not None, "reverse_order_parsed": row["judgments"][1] is not None,
                    "same_canonical_choice": bool(all(j is not None for j in row["judgments"]) and row["judgments"][0]["choice"] == row["judgments"][1]["choice"]),
                    "first_tokens": row["readouts"][0]["tokens"], "reverse_tokens": row["readouts"][1]["tokens"],
                    "any_length_cap": any(r["hit_length_cap"] for r in row["readouts"]),
                    "actor_update_harm": "not_applicable_no_actor_update"})
        atomic_json(folder / "scores/anonymous_teacher_cases.json", teacher_rows)
        print(json.dumps(public), flush=True)
        return
    ledger = json.loads((folder / "controllers/main-plan.json").read_text())
    if ledger["status"] != "all_planned_jobs_completed" or len(ledger["jobs"]) != 10 or any(j["return_code"] != 0 for j in ledger["jobs"]):
        raise ValueError("The entire three-seed, three-arm matrix must close")
    if not json.loads((folder / "owned_cleanup_before_score.json").read_text())["owned_main_processes_ended"]:
        raise ValueError("All main workers must end before scoring")
    acceptance = json.loads((folder / "acceptance.json").read_text())
    loaded = {}
    for arm in ["a", "s", "t", "v"]:
        for seed in ([42] if arm == "a" else manifest["rollout_seeds"]):
            key = "a" if arm == "a" else f"{arm}{seed}"
            cfg = manifest["configurations"][arm][str(seed)]
            run = folder / "runs" / ("main-" + key)
            rows, result, probes = available_run(run, manifest, digest, cfg, acceptance)
            if len(rows) != 16 or result["status"] != "completed" or set(probes) != {0, 16} or result["code_commit"] != os.environ["P2_CODE_COMMIT"]:
                raise ValueError("Incomplete matrix member")
            output_units(run, rows, choices)
            if arm == "v" and json.loads((run / "reference_signal_snapshot.json").read_text()) != signal:
                raise ValueError("Frozen reward changed between seeds")
            for row in rows:
                if arm != "a":
                    rewards, _, _ = (verified_group_rewards(row["completions"], choices[row["id"]], targets[row["id"]]) if arm == "v"
                                     else group_rewards(row["completions"], choices[row["id"]]))
                    if row["rewards"] != (None if rewards is None else rewards.tolist()):
                        raise ValueError("Saved rewards differ from the frozen source")
            loaded[key] = rows, result, probes
    initial = loaded["a"][2][0]["predictions"]
    if any(probes[0]["predictions"] != initial for _, _, probes in loaded.values()):
        raise ValueError("Initial probe predictions differ between arms")
    atomic_json(folder / "scores/prediction_seal.json", {"manifest_sha256": digest, "coverage": {k: len(v[0]) for k, v in loaded.items()}, "labels_read_before_seal": False})
    labels = read_labels(root, manifest)
    base_predictions = [{"id": r["id"], "answer": r["before"]["answer"]} for r in loaded["a"][0]]
    base = [is_correct(p["answer"], labels[p["id"]]) for p in base_predictions]
    public = {"seeds": {}, "resources": {}, "frozen_correct": sum(base), "n": 16, "independent_generalization_established": False}
    success = []
    probe_rows = []
    for seed in manifest["rollout_seeds"]:
        per_seed = {}
        for arm in ["s", "t", "v"]:
            key = f"{arm}{seed}"; rows, result, probes = loaded[key]
            run = folder / "runs" / ("main-" + key)
            metric, _ = score_cases(rows, labels, base)
            before = [{"id": r["id"], "answer": r["before"]["answer"]} for r in rows]
            after = [{"id": r["id"], "answer": r["after"]["answer"]} for r in rows]
            metric["history_transition"] = prediction_transition_audit(base_predictions, before, labels)
            metric["current_transition"] = prediction_transition_audit(before, after, labels)
            final = probes[16]["predictions"]
            metric["probe_transition"] = prediction_transition_audit(initial, final, labels)
            metric["format"] = summarize_formats(output_units(run, rows, choices), labels)
            metric["reward_audit"] = verified_audit(rows, choices, labels, targets) if arm == "v" else reward_audit(rows, choices, labels)
            per_seed[arm] = metric
            public["resources"][key] = resource_summary(run, rows, result)
            for index, (old, new) in enumerate(zip(initial, final)):
                probe_rows.append({"seed": seed, "arm": arm, "probe_index": index,
                    "initial_correct": is_correct(old["answer"], labels[old["id"]]), "final_correct": is_correct(new["answer"], labels[new["id"]]),
                    "initial_parsed": old["answer"] is not None, "final_parsed": new["answer"] is not None})
        per_seed["success"] = (per_seed["v"]["greedy_before"]["correct"] > max(sum(base), per_seed["s"]["greedy_before"]["correct"], per_seed["t"]["greedy_before"]["correct"])
                               and per_seed["v"]["probe_transition"]["correct_to_correct"] == 3)
        public["seeds"][str(seed)] = per_seed
        success.append(per_seed["success"])
    public.update(stable_positive_development_result=len(success) == 3 and all(success), probe_rows=probe_rows,
                  qualification=json.loads((folder / "scores/qualification.json").read_text()))
    atomic_json(folder / "scores/public_summary.json", public)
    print(json.dumps({"matrix_closed": True, "stable_positive_development_result": public["stable_positive_development_result"]}), flush=True)


if __name__ == "__main__":
    main()
