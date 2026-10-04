"""Independent offline scoring; this is the only runtime entry that opens labels."""
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from state import atomic_json

METHODS = {"a": "Frozen greedy", "b": "Frozen SC-8", "c": "TTRL", "d": "SPINE"}


def fraction(correct, total):
    return {"correct": int(correct), "total": int(total),
            "percent": 100 * correct / total if total else None}


def is_correct(answer, label):
    return answer is not None and answer == label


def paired(before, after):
    assert len(before) == len(after)
    return {"wrong_to_correct": sum(not x and y for x, y in zip(before, after)),
            "correct_to_wrong": sum(x and not y for x, y in zip(before, after)),
            "correct_to_correct": sum(x and y for x, y in zip(before, after)),
            "wrong_to_wrong": sum(not x and not y for x, y in zip(before, after)),
            "gain_pp": 100 * (sum(after) - sum(before)) / len(before) if before else None}


def mean(values):
    return sum(values) / len(values) if values else None


def range_summary(values):
    return {"count": len(values), "mean": mean(values),
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def score_cases(rows, labels, base):
    n = len(rows)
    before, after, voted = [], [], []
    for row in rows:
        label = labels[row["id"]]
        if "before" in row:
            before.append(is_correct(row["before"]["answer"], label))
        if "after" in row:
            after.append(is_correct(row["after"]["answer"], label))
        if "vote" in row:
            voted.append(is_correct(row["vote"]["winner"], label))
    result = {"n": n}
    for name, values in [("greedy_before", before), ("greedy_after", after), ("consensus", voted)]:
        if values:
            result[name] = fraction(sum(values), n)
    if before:
        result["historical_gain_vs_frozen_pp"] = 100 * (sum(before) - sum(base)) / n
        result["historical_transitions_vs_frozen"] = paired(base, before)
        result["before_parse_rate"] = fraction(sum(row["before"]["answer"] is not None for row in rows), n)
    if after:
        result["current_case_transitions"] = paired(before, after)
        result["after_parse_rate"] = fraction(sum(row["after"]["answer"] is not None for row in rows), n)
    votes = [row["vote"] for row in rows if "vote" in row]
    if votes:
        candidate_count = sum(len(vote["answers"]) for vote in votes)
        candidate_available_wrong_vote = sum(
            any(is_correct(answer, labels[row["id"]]) for answer in row["vote"]["answers"])
            and not is_correct(row["vote"]["winner"], labels[row["id"]]) for row in rows)
        available = sum(any(is_correct(answer, labels[row["id"]]) for answer in row["vote"]["answers"]) for row in rows)
        half = n // 2
        result["reward_audit"] = {
            "parsed_candidates": fraction(sum(vote["valid"] for vote in votes), candidate_count),
            "ties": fraction(sum(vote["tie"] for vote in votes), n),
            "all_unparseable": fraction(sum(vote["all_unparseable"] for vote in votes), n),
            "zero_advantage_groups": fraction(sum(vote["zero_advantage_group"] for vote in votes), n),
            "top_vote_count": range_summary([vote["top_count"] for vote in votes]),
            "top_vote_margin": range_summary([vote["margin"] for vote in votes]),
            "correct_candidate_exists": fraction(available, n),
            "correct_candidate_but_wrong_vote": fraction(candidate_available_wrong_vote, n),
            "wrong_vote_given_correct_candidate": fraction(candidate_available_wrong_vote, available),
            "first_half_top_vote_mean": mean([vote["top_count"] for vote in votes[:half]]),
            "second_half_top_vote_mean": mean([vote["top_count"] for vote in votes[half:]]),
            "before_matches_vote": fraction(sum(row["before"]["answer"] == row["vote"]["winner"] and row["vote"]["winner"] is not None for row in rows), n) if before else None,
            "after_matches_vote": fraction(sum(row["after"]["answer"] == row["vote"]["winner"] and row["vote"]["winner"] is not None for row in rows), n) if after else None,
        }
    optimized = [row["optimization"] for row in rows if row.get("update_applied")]
    result["optimization"] = {
        "updates_in_prefix": len(optimized),
        "skips_in_prefix": dict(Counter(row["skip_reason"] for row in rows if row.get("skip_reason"))),
        "zero_advantage_updates": sum(bool(row.get("update_applied")) and row["vote"]["zero_advantage_group"] for row in rows if "vote" in row),
        "selected_token_fraction": sum(x["selected_tokens"] for x in optimized) / sum(x["response_tokens"] for x in optimized) if optimized else None,
        "loss_components": {key: range_summary([x["loss_components"][key] for x in optimized]) for key in ["policy", "band", "kl"]},
        "gradient_norms": {key: range_summary([x["gradient_norms"][key] for x in optimized]) for key in ["vision", "language"]},
        "masters_persisted_all": all(x["masters_persisted"] for x in optimized) if optimized else None,
        "moments_persisted_all": all(x["moments_persisted"] for x in optimized) if optimized else None,
        "ratio_mean_before_step": range_summary([x["ratio_mean_before_step"] for x in optimized]),
        "clip_fraction_before_step": range_summary([x["clip_fraction_before_step"] for x in optimized]),
        "exact_full_vocabulary_kl_before_step": range_summary([x["exact_full_vocabulary_kl_mean_before_step"] for x in optimized]),
        "post_update_drift": [{"cursor": row["index"] + 1, **row["optimization"]["post_update_drift"]} for row in rows if "post_update_drift" in row.get("optimization", {})],
    }
    sequences = sum(len(row.get("response_lengths", [])) for row in rows)
    result["prefix_generation"] = {
        "rollouts": sequences,
        "rollout_tokens": sum(sum(row.get("response_lengths", [])) for row in rows),
        "rollouts_at_length_cap": fraction(sum(sum(row.get("hit_length_cap", [])) for row in rows), sequences),
        "greedy_before_tokens": sum(row.get("before", {}).get("tokens", 0) for row in rows),
        "greedy_after_tokens": sum(row.get("after", {}).get("tokens", 0) for row in rows),
        "case_wall_seconds": sum(row["elapsed_seconds"] for row in rows),
    }
    # Eight-case blocks avoid a misleading smooth trend and expose no sample IDs.
    result["online_blocks"] = []
    for stop in range(8, n + 1, 8):
        block = {"cursor": stop}
        for name, values in [("greedy_before", before), ("greedy_after", after), ("consensus", voted)]:
            if values:
                block[name + "_cumulative"] = fraction(sum(values[:stop]), stop)
                block[name + "_block"] = fraction(sum(values[stop - 8:stop]), 8)
        result["online_blocks"].append(block)
    private = [{"id": row["id"], "before_correct": before[i] if before else None,
                "after_correct": after[i] if after else None, "vote_correct": voted[i] if voted else None}
               for i, row in enumerate(rows)]
    return result, private


def prediction_transition_audit(before, after, labels):
    """Offline paired correctness changes, decomposed by parser validity."""
    if [x["id"] for x in before] != [x["id"] for x in after]:
        raise ValueError("Paired prediction IDs differ")
    bc = [is_correct(x["answer"], labels[x["id"]]) for x in before]
    ac = [is_correct(x["answer"], labels[x["id"]]) for x in after]
    bp = [x["answer"] is not None for x in before]
    ap = [x["answer"] is not None for x in after]
    result = {"n": len(before), **paired(bc, ac),
              "before_parsed": sum(bp), "after_parsed": sum(ap),
              "newly_parsed": sum(not x and y for x, y in zip(bp, ap)),
              "newly_unparseable": sum(x and not y for x, y in zip(bp, ap)),
              "invalid_to_correct": sum(not p and c for p, c in zip(bp, ac)),
              "parsed_wrong_to_correct": sum(p and not c and a for p, c, a in zip(bp, bc, ac)),
              "correct_to_invalid": sum(c and not p for c, p in zip(bc, ap)),
              "correct_to_parsed_wrong": sum(c and p and not a for c, p, a in zip(bc, ap, ac))}
    assert result["wrong_to_correct"] == result["invalid_to_correct"] + result["parsed_wrong_to_correct"]
    assert result["correct_to_wrong"] == result["correct_to_invalid"] + result["correct_to_parsed_wrong"]
    assert result["after_parsed"] - result["before_parsed"] == result["newly_parsed"] - result["newly_unparseable"]
    return result


def read_cases(run, manifest, digest):
    result = json.loads((run / "result.json").read_text())
    if result["manifest_sha256"] != digest or not result["reference_unchanged"]:
        raise ValueError("Manifest/reference validation failed")
    rows = []
    for i in range(result["cursor"]):
        row = json.loads((run / "cases" / f"i{i:06d}.json").read_text())
        if row["index"] != i or row["id"] != manifest["stream"][i]["id"]:
            raise ValueError("Prediction order differs from locked manifest")
        rows.append(row)
    return rows, result


def resource_summary(run, rows, result):
    segments = [json.loads(path.read_text()) for path in sorted((run / "segments").glob("c*.json"))]
    phases = Counter()
    for segment in segments:
        phases.update(segment["phase_seconds"])
    probes = [json.loads(path.read_text()) for path in sorted((run / "probe").glob("c*.json"))]
    jobs = []
    campaign = run.parent.parent
    for path in (campaign / "controllers").glob("*.json"):
        controller = json.loads(path.read_text())
        plan = json.loads((campaign / controller["plan"]).read_text())
        owners = {job["label"]: job["environment"].get("P2_RUN") for job in plan["jobs"]}
        jobs.extend(job for job in controller["jobs"] if owners[job["label"]] == run.name)
    return {
        "actual_completed_cursor": result["cursor"], "actual_optimizer_updates": result["optimizer_updates"],
        "segment_wall_seconds": sum(segment["elapsed_seconds"] for segment in segments),
        "process_wall_seconds": sum(job["wall_seconds"] for job in jobs) if jobs and all("wall_seconds" in job for job in jobs) else None,
        "process_jobs": len(jobs),
        "process_failures": sum(job.get("return_code", 0) != 0 for job in jobs),
        "process_jobs_missing_final_metadata": sum("return_code" not in job or "wall_seconds" not in job for job in jobs),
        "phase_seconds": dict(phases),
        "peak_allocated_bytes": max(segment["peak_allocated_bytes"] for segment in segments),
        "peak_reserved_bytes": max(segment["peak_reserved_bytes"] for segment in segments),
        "whole_device_sampled_peak_used_bytes": max(segment["sampled_device_memory"]["peak_used_bytes"] for segment in segments),
        "whole_device_memory_includes_other_jobs": True,
        "peak_cpu_rss_kib": max(segment["cpu_peak_rss_kib"] for segment in segments),
        "probe_generated_tokens": sum(item["tokens"] for probe in probes for item in probe["predictions"]),
        "actual_rollouts": sum(len(row.get("response_lengths", [])) for row in rows),
        "actual_rollout_tokens": sum(sum(row.get("response_lengths", [])) for row in rows),
        "checkpoint_writes": len(segments),
        "checkpoint_bytes_written": sum(segment["checkpoint"]["state_bytes"] for segment in segments),
        "retained_checkpoint_bytes": sum(path.stat().st_size for path in (run / "checkpoints").glob("c*/state.pt")),
        "reference_unchanged_all_segments": all(segment["reference_unchanged"] for segment in segments),
    }


def self_test():
    # A skipped/invalid prediction remains in the denominator; harms are counted.
    assert fraction(1, 4) == {"correct": 1, "total": 4, "percent": 25.0}
    assert not is_correct(None, "A")
    assert paired([False, True, False, True], [True, False, False, True]) == {
        "wrong_to_correct": 1, "correct_to_wrong": 1, "correct_to_correct": 1,
        "wrong_to_wrong": 1, "gain_pp": 0.0}
    rows = []
    for i in range(8):
        rows.append({"id": str(i), "index": i, "before": {"answer": "A" if i == 0 else None, "tokens": 1},
                     "after": {"answer": "A" if i == 1 else None, "tokens": 1},
                     "vote": {"winner": None, "answers": [None] * 8, "valid": 0, "tie": False,
                              "all_unparseable": True, "zero_advantage_group": False, "top_count": 0, "margin": 0},
                     "update_applied": False, "skip_reason": "all_unparseable", "elapsed_seconds": 1,
                     "response_lengths": [1] * 8, "hit_length_cap": [False] * 8})
    summary, private = score_cases(rows, {str(i): "A" for i in range(8)}, [False] * 8)
    assert summary["greedy_before"] == fraction(1, 8)
    assert summary["current_case_transitions"]["wrong_to_correct"] == 1
    assert summary["current_case_transitions"]["correct_to_wrong"] == 1
    assert summary["reward_audit"]["all_unparseable"] == fraction(8, 8)
    assert summary["optimization"]["skips_in_prefix"] == {"all_unparseable": 8}
    assert summary["prefix_generation"]["rollouts"] == 64 and len(private) == 8
    before = [{"id": str(i), "answer": answer} for i, answer in enumerate([None, "B", "A", "A", "B"])]
    after = [{"id": str(i), "answer": answer} for i, answer in enumerate(["A", "A", None, "B", "B"])]
    audit = prediction_transition_audit(before, after, {str(i): "A" for i in range(5)})
    assert audit["wrong_to_correct"] == audit["correct_to_wrong"] == 2
    assert all(audit[key] == 1 for key in ["invalid_to_correct", "parsed_wrong_to_correct", "correct_to_invalid", "correct_to_parsed_wrong"])
    assert audit["newly_parsed"] == audit["newly_unparseable"] == 1
    print(json.dumps({"passed": True, "scoring": "denominators, paired gains/harms, skips, invalid answers, token counts and parser-attributed transitions", "real_labels_read": False}))


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip():
        raise ValueError("Manifest changed before scoring")
    manifest = json.loads(raw)
    if manifest["kind"] != "development" or not json.loads((folder / "acceptance.json").read_text())["passed"]:
        raise ValueError("Development manifest and successful engineering acceptance required")
    run_ids = json.loads(os.environ.get("P2_SCORE_RUNS", '{"a":"main-a","b":"main-b","c":"main-c","d":"main-d"}'))
    runs = {key: folder / "runs" / run_ids[key] for key in METHODS}
    loaded = {key: read_cases(run, manifest, digest) for key, run in runs.items()}
    if any(result["method"] != METHODS[key] for key, (_, result) in loaded.items()):
        raise ValueError("Method mapping differs from prediction artifacts")
    n = min(len(rows) for rows, _ in loaded.values())
    if n == 0:
        raise ValueError("There is no completed four-method prefix to score")
    # Validate every available probe artifact BEFORE opening any label file.
    probes = {}
    expected_probe_ids = [entry["id"] for entry in manifest["probe"]]
    for key, run in runs.items():
        probes[key] = {}
        for path in sorted((run / "probe").glob("c*.json")):
            record = json.loads(path.read_text())
            if [row["id"] for row in record["predictions"]] != expected_probe_ids:
                raise ValueError("Probe predictions differ from locked manifest")
            probes[key][record["cursor"]] = record
    common_probes = sorted(set.intersection(*(set(x) for x in probes.values())))
    common_probes = [cursor for cursor in common_probes if cursor <= n]
    # The only label read in this executable pipeline occurs below, after outputs.
    labels = {}
    for split in sorted({entry["split"] for section in ["stream", "probe"] for entry in manifest[section]}):
        path = root / "views/evaluation_labels/MedXpertQA-MM" / (split + ".jsonl")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["id"] in labels:
                raise ValueError("Duplicate label ID")
            labels[row["id"]] = row["label"]
    base = [is_correct(row["before"]["answer"], labels[row["id"]]) for row in loaded["a"][0][:n]]
    public = {"status": "completed" if n == manifest["selected_n"] else "common_prefix_only",
              "scored_at_utc": datetime.now(timezone.utc).isoformat(), "common_n": n,
              "evaluator_code_commit": os.environ.get("P2_EVAL_CODE_COMMIT"),
              "transition_audit": "post-prediction paired parse/correctness decomposition; not used to select any experiment",
              "planned_n": manifest["selected_n"], "probe_n": len(expected_probe_ids),
              "manifest_sha256": digest, "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
              "seed": manifest["seed"], "source_split": "test retired for development use",
              "selection_used_correctness": False, "methods": {},
              "common_probe_cursors": common_probes, "inference": "single-seed fixed random development stream; no significance claim",
              "gpu_memory_caveat": "whole-device sample includes unrelated jobs; allocator figures cover this PyTorch process",
              "cost_caveat": "SC-8 and RL are not compute-matched; wall time is not GPU active time"}
    private = {}
    for key, (rows, result) in loaded.items():
        method, private[key] = score_cases(rows[:n], labels, base)
        method.update(name=METHODS[key], actual_coverage=len(rows), code_commit=result["code_commit"],
                      resources=resource_summary(runs[key], rows, result))
        method["probe"] = []
        initial = None
        for cursor in common_probes:
            predictions = probes[key][cursor]["predictions"]
            correct = sum(is_correct(row["answer"], labels[row["id"]]) for row in predictions)
            if cursor == 0:
                initial = correct
            method["probe"].append({"cursor": cursor, **fraction(correct, len(predictions)),
                                    "delta_from_initial_pp": 100 * (correct - initial) / len(predictions) if initial is not None else None,
                                    "parsed": sum(row["answer"] is not None for row in predictions), "decoding": "greedy"})

        # Post-prediction audit only; these counts never control training.
        compact = lambda data, field: [{"id": row["id"], "answer": row[field]["answer"]} for row in data]
        if key in {"c", "d"}:
            method["historical_transition_audit"] = prediction_transition_audit(
                compact(loaded["a"][0][:n], "before"), compact(rows[:n], "before"), labels)
            method["current_transition_audit"] = prediction_transition_audit(
                compact(rows[:n], "before"), compact(rows[:n], "after"), labels)
        method["probe_transition_audit"] = [
            {"cursor": cursor, **prediction_transition_audit(probes[key][0]["predictions"],
                                                           probes[key][cursor]["predictions"], labels)}
            for cursor in common_probes if 0 in probes[key]]
        public["methods"][key] = method
    atomic_json(folder / "scores/private_case_scores.json", private)
    atomic_json(folder / "scores/public_summary.json", public)
    print(json.dumps({"status": public["status"], "common_n": n, "summary_written": True}))


if __name__ == "__main__":
    self_test() if os.environ.get("P2_EVAL_SELFTEST") == "1" else main()
