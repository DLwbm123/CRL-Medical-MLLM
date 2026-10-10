"""Offline scoring only after all new frozen workers have sealed and ended."""
import hashlib
import json
import os
from pathlib import Path
from continual import group_rewards, load_input
from core import extract_answer
from evaluate import is_correct
from p4_evaluate import reward_audit
from p10_score import verified_audit
from state import atomic_json
from verified_reward import checked_targets

CHECKS = ["coverage_at_least_half", "at_least_twelve_correct_targets", "accepted_precision_at_least_three_quarters",
          "correct_negative_rate_lower", "wrong_positive_rate_ten_pp_lower", "correct_minority_rescued"]


def block_gate(majority, verified, correct, accepted):
    mcn, vcn = majority["correct_negative"]["percent"], verified["correct_negative"]["percent"]
    mwp, vwp = majority["wrong_positive"]["percent"], verified["wrong_positive"]["percent"]
    values = [accepted >= 16, correct >= 12, accepted > 0 and correct / accepted >= .75,
              mcn is not None and vcn is not None and vcn < mcn,
              mwp is not None and vwp is not None and vwp <= mwp - 10,
              verified["counts"]["minority_rescued"] >= 1]
    checks = dict(zip(CHECKS, values))
    return {"go": all(values), "checks": checks}


def main(reference="reference", score_directory="scores", teacher=None):
    root = Path(os.environ["P0_ROOT"]); folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    for name in ["qualifier_owned_ended.json", "owned_cleanup_before_score.json"]:
        if not json.loads((folder / name).read_text())["owned_main_processes_ended"]:
            raise ValueError("Every teacher and pool session must end before scoring")
    ledger = json.loads((folder / "controllers/main-plan.json").read_text())
    if ledger["status"] != "all_planned_jobs_completed" or len(ledger["jobs"]) != 6 or any(j["return_code"] != 0 for j in ledger["jobs"]):
        raise ValueError("All six frozen pools must close before any labels are read")
    raw = (folder / "manifest.json").read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip(): raise ValueError("Manifest changed")
    manifest = json.loads(raw); entries = manifest["stream"] + manifest["probe"]
    if len(manifest["stream"]) != 32 or len(manifest["probe"]) != 32 or len({e["id"] for e in entries}) != 64:
        raise ValueError("Fixed 32+32 image-group scope differs")
    scores = folder / score_directory
    scores.mkdir(exist_ok=True)
    signal = json.loads((folder / reference / "targets.json").read_text())
    targets = checked_targets({**signal, "stream": signal["stream"] + signal["probe"]}, {**manifest, "stream": entries, "teacher": teacher or manifest["teacher"]}, digest)
    if signal["code_commit"] != os.environ["P2_CODE_COMMIT"]: raise ValueError("Teacher source differs")
    choices = {e["id"]: load_input(folder, e)["options"] for e in entries}
    loaded = {}
    for name, cfg in manifest["configurations"].items():
        run = folder / "runs" / name; result = json.loads((run / "result.json").read_text())
        if result["status"] != "completed" or not result["sealed"] or result["cursor"] != 64 or result["manifest_sha256"] != digest or result["configuration"] != cfg or result["code_commit"] != os.environ["P2_CODE_COMMIT"] or result["optimizer_updates"] != 0 or result["labels_read"] or not result["weights_unchanged"]:
            raise ValueError("Frozen pool seal/configuration/source differs")
        probability = json.loads((run / "probability_check.json").read_text())
        if not 0 <= probability["max_logp_difference"] <= cfg["probability_check_bf16_atol"] or probability["temperature"] != 1 or probability["top_p"] != 1:
            raise ValueError("Actual frozen-backbone probability check failed")
        rows = [json.loads(p.read_text()) for p in sorted((run / "cases").glob("i*.json"))]
        if len(rows) != 64 or [r["id"] for r in rows] != [e["id"] for e in entries]: raise ValueError("Incomplete frozen pool")
        for row in rows:
            if len(row["completions"]) != 8 or row["optimizer_updates_before"] or row["optimizer_updates_after"]:
                raise ValueError("Pool differs from SC-8 without updates")
            rewards, _, vote = group_rewards(row["completions"], choices[row["id"]])
            if row["vote"]["answers"] != [extract_answer(t, choices[row["id"]]) for t in row["completions"]] or row["vote"]["winner"] != vote["winner"] or row["rewards"] != (None if rewards is None else rewards.tolist()):
                raise ValueError("Original strict parse or majority rewards differ")
        loaded[name] = rows
    atomic_json(scores / "prediction_seal.json", {"manifest_sha256": digest, "coverage": {k: len(v) for k, v in loaded.items()}, "labels_read_before_seal": False})
    labels = {}
    source = Path(os.environ["P10_SOURCE_ROOT"]) / "views/evaluation_labels/SLAKE/train.jsonl"
    for line in source.open():
        row = json.loads(line)
        if row["id"] in choices:
            answer = row["answer"].strip().lower()
            if answer not in {"yes", "no"}: raise ValueError("Selected binary question has a nonbinary label; no reselection permitted")
            labels[row["id"]] = "A" if answer == "yes" else "B"
    if set(labels) != set(choices): raise ValueError("Selected training labels are incomplete")
    blocks = {}; pools = []; cases = []; teacher_rows = []
    for section, offset in [("calibration", 0), ("verification", 32)]:
        part = entries[offset:offset + 32]; ids = {e["id"] for e in part}
        accepted = sum(targets[k] is not None for k in ids); correct = sum(is_correct(targets[k], labels[k]) for k in ids)
        for arm in ["s", "t"]:
            rows = [r for seed in manifest["rollout_seeds"] for r in loaded[f"pool-{arm}{seed}"][offset:offset + 32]]
            majority, verified = reward_audit(rows, choices, labels), verified_audit(rows, choices, labels, targets)
            decision = block_gate(majority, verified, correct, accepted)
            blocks[f"{section}_{arm}"] = {**decision, "unique_groups": 32, "repeated_group_observations": 96,
                "accepted": accepted, "correct_targets": correct, "majority": majority, "verified": verified}
    for name, rows in loaded.items():
        seed = manifest["configurations"][name]["seed"]; arm = name.split("-")[1][0]
        majority, verified = reward_audit(rows, choices, labels), verified_audit(rows, choices, labels, targets)
        pools.append({"seed": seed, "sampler": arm, "groups": 64, "majority": majority, "verified": verified,
                      "optimizer_updates": 0, "retention_harm": "NA_no_update"})
        for i, row in enumerate(rows):
            m, v = reward_audit([row], choices, labels), verified_audit([row], choices, labels, targets)
            cases.append({"seed": seed, "sampler": arm, "section": "calibration" if i < 32 else "verification", "anonymous_index": i % 32,
                "majority_correct": is_correct(row["vote"]["winner"], labels[row["id"]]), "valid_candidates": row["vote"]["valid"],
                "correct_candidates": sum(is_correct(a, labels[row["id"]]) for a in row["vote"]["answers"]),
                "any_length_cap": any(row["hit_length_cap"]), "accepted": targets[row["id"]] is not None,
                "majority_correct_negative_count": m["counts"]["correct_negative"], "verified_correct_negative_count": v["counts"]["correct_negative"],
                "majority_wrong_positive_count": m["counts"]["wrong_positive"], "verified_wrong_positive_count": v["counts"]["wrong_positive"],
                "minority_rescued": bool(v["counts"]["minority_rescued"]), "retention_harm": "NA_no_update"})
    for section, key in [("calibration", "stream"), ("verification", "probe")]:
        for i, row in enumerate(signal[key]):
            teacher_rows.append({"section": section, "anonymous_index": i, "accepted": row["target"] is not None,
                "accepted_target_correct": is_correct(row["target"], labels[row["id"]]),
                "first_order_parsed": row["judgments"][0] is not None, "reverse_order_parsed": row["judgments"][1] is not None,
                "same_canonical_choice": bool(all(j is not None for j in row["judgments"]) and row["judgments"][0]["choice"] == row["judgments"][1]["choice"]),
                "first_tokens": row["readouts"][0]["tokens"], "reverse_tokens": row["readouts"][1]["tokens"],
                "any_length_cap": any(r["hit_length_cap"] for r in row["readouts"]), "retention_harm": "NA_no_update"})
    public = {"go": all(b["go"] for b in blocks.values()), "blocks": blocks, "pool_metrics": pools,
        "unique_groups": 64, "repeated_group_observations": 384, "candidate_count": 3072, "teacher_readouts": 128,
        "optimizer_updates": 0, "training_performed": False, "stable_positive_development_result": False,
        "independent_clinical_generalization_established": False, "labels_used_only_after_all_workers_sealed_and_ended": True}
    atomic_json(scores / "qualification.json", public)
    atomic_json(scores / "anonymous_teacher_cases.json", teacher_rows)
    atomic_json(scores / "anonymous_pool_cases.json", cases)
    print(json.dumps({"go": public["go"], "groups": 64, "optimizer_updates": 0}), flush=True)


if __name__ == "__main__": main()
