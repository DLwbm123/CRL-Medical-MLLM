"""Separate CPU evaluator for existing, sealed development artifacts only."""
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: "N/A" if v is None else v for k, v in row.items()} for row in rows)


def audit():
    # Import the evaluator from this round's actual executed source checkout.
    workspace = Path(os.environ["P8_SOURCE_WORKSPACE"])
    sys.path.insert(0, str(workspace / "implementation"))
    from continual import group_rewards
    from evaluate import is_correct, prediction_transition_audit, score_cases
    from p3_diagnostics import read_labels, options_by_id, output_units, summarize_formats
    from p3_evaluate import available_run
    from p4_evaluate import reward_audit
    from state import atomic_json

    root, folder, dest = (Path(os.environ[k]) for k in ["P8_ROOT", "P8_SOURCE_FOLDER", "P8_DEST"])
    dest.mkdir(parents=True, exist_ok=True)
    raw = (folder / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == (folder / "manifest.sha256").read_text().strip()
    manifest = json.loads(raw)
    published = json.loads((folder / "scores/public_summary.json").read_text())
    acceptance = json.loads((folder / "acceptance.json").read_text())
    source = json.loads((workspace / "metadata/source_receipt.json").read_text())["commit"]
    assert published["evaluator_commit"] == source
    seeds = manifest["rollout_seeds"]
    round_name = os.environ["P8_ROUND"]
    options = options_by_id(folder, manifest)
    aliases = {entry["id"]: f"{section[0].upper()}{i+1:02d}"
               for section in ["stream", "probe"] for i, entry in enumerate(manifest[section])}
    loaded = {}
    for name in ["a"] + [arm + str(seed) for seed in seeds for arm in "mv"]:
        run = folder / "runs" / ("main-" + name)
        cfg = json.loads((run / "configuration.json").read_text())
        if name != "a":
            latest = json.loads((run / "checkpoints/latest.json").read_text())
            assert latest["code_commit"] == source and latest["manifest_sha256"] == digest
            assert latest["model_revision"] == cfg["model_revision"] and latest["configuration"] == cfg
            assert latest["cursor"] == latest["optimizer_updates"] == 16
            assert cfg["seed"] == int(name[1:]) and cfg["method"] == "SPINE"
        rows, result, probes = available_run(run, manifest, digest, cfg, acceptance)
        assert len(rows) == 16 and set(probes) == {0, 16} and result["status"] == "completed"
        assert result["code_commit"] == source or name == "a"
        units = output_units(run, rows, options)
        loaded[name] = rows, probes, units
    # All parsing/provenance checks precede the independent label read.
    labels = read_labels(root, manifest)
    base = [{"id": r["id"], "answer": r["before"]["answer"]} for r in loaded["a"][0]]
    base_right = [is_correct(p["answer"], labels[p["id"]]) for p in base]
    initial = loaded["a"][1][0]["predictions"]
    initial_correct = {p["id"] for p in initial if is_correct(p["answer"], labels[p["id"]])}
    assert len(initial_correct) == 3
    primary, cases, formats, private_rows, discrepancies = [], [], [], [], []
    for seed in seeds:
        for arm in "mv":
            rows, probes, units = loaded[arm + str(seed)]
            expected = published["seeds"][str(seed)][arm]
            metric, _ = score_cases(rows, labels, base_right)
            before = [{"id": r["id"], "answer": r["before"]["answer"]} for r in rows]
            after = [{"id": r["id"], "answer": r["after"]["answer"]} for r in rows]
            metric["history_parse_transition"] = prediction_transition_audit(base, before, labels)
            metric["current_parse_transition"] = prediction_transition_audit(before, after, labels)
            first, final = [probes[c]["predictions"] for c in [0, 16]]
            assert first == initial
            metric["probe"] = {"initial_correct": sum(is_correct(p["answer"], labels[p["id"]]) for p in first),
                               "final_correct": sum(is_correct(p["answer"], labels[p["id"]]) for p in final),
                               "initial_parsed": sum(p["answer"] is not None for p in first),
                               "final_parsed": sum(p["answer"] is not None for p in final),
                               "transition": prediction_transition_audit(first, final, labels)}
            metric["reward_direction"] = reward_audit(rows, options, labels)
            metric["format"] = summarize_formats(units, labels)
            for key, value in metric.items():
                if expected[key] != value:
                    discrepancies.append({"round": round_name, "seed": seed, "arm": arm, "metric": key,
                                          "old": expected[key], "new": value})
            primary.append({"round": round_name, "seed": seed, "arm": arm,
                            "before_correct": metric["greedy_before"]["correct"],
                            "after_correct": metric["greedy_after"]["correct"],
                            "before_parsed": metric["before_parse_rate"]["correct"],
                            "after_parsed": metric["after_parse_rate"]["correct"],
                            "probe_final_correct": metric["probe"]["final_correct"],
                            "original_correct_retained": metric["probe"]["transition"]["correct_to_correct"],
                            "correct_negative_n": metric["reward_direction"]["correct_negative"]["correct"],
                            "correct_negative_d": metric["reward_direction"]["correct_negative"]["total"],
                            "wrong_positive_n": metric["reward_direction"]["wrong_positive"]["correct"],
                            "wrong_positive_d": metric["reward_direction"]["wrong_positive"]["total"]})
            for i, row in enumerate(rows):
                rewards, advantages, vote = group_rewards(row["completions"], options[row["id"]])
                assert (None if rewards is None else rewards.tolist()) == row["rewards"]
                correct = [is_correct(a, labels[row["id"]]) for a in vote["answers"]]
                adv = [None] * 8 if advantages is None else advantages.tolist()
                rew = [0.] * 8 if rewards is None else rewards.tolist()
                correct_adv = [a for a, c in zip(adv, correct) if c and a is not None]
                b, a = row["before"], row["after"]
                value = {"round": round_name, "seed": seed, "arm": arm, "group": aliases[row["id"]],
                         "frozen_parsed": base[i]["answer"] is not None, "frozen_correct": base_right[i],
                         "before_parsed": b["answer"] is not None, "before_correct": is_correct(b["answer"], labels[row["id"]]),
                         "after_parsed": a["answer"] is not None, "after_correct": is_correct(a["answer"], labels[row["id"]]),
                         "valid_candidates": vote["valid"], "answer_count_profile": json.dumps(sorted(vote["counts"].values(), reverse=True)),
                         "tie": vote["tie"], "top_count": vote["top_count"], "margin": vote["margin"],
                         "correct_candidate_exists": any(correct), "correct_candidate_count": sum(correct),
                         "consensus_correct": is_correct(vote["winner"], labels[row["id"]]),
                         "correct_negative_n": sum(c and x is not None and x < 0 for c, x in zip(correct, adv)),
                         "correct_negative_d": sum(correct),
                         "wrong_positive_n": sum(x > 0 and ans is not None and not c for x, ans, c in zip(rew, vote["answers"], correct)),
                         "wrong_positive_d": sum(x > 0 for x in rew),
                         "mean_correct_advantage": sum(correct_adv)/len(correct_adv) if correct_adv else None,
                         "before_tokens": b["tokens"], "after_tokens": a["tokens"],
                         "rollout_tokens": json.dumps(row["response_lengths"]),
                         "rollout_cap_count": sum(n == 2048 for n in row["response_lengths"]),
                         "before_capped": b["tokens"] == 2048, "after_capped": a["tokens"] == 2048,
                         "missing": False}
                cases.append(value)
                private_rows.append({**value, "id": row["id"], "consensus": vote["winner"], "answer_counts": vote["counts"]})
            for unit in units:
                if unit["kind"] not in ["probe", "before", "after"]:
                    continue
                formats.append({"round": round_name, "seed": seed, "arm": arm,
                                "group": aliases[unit["id"]], "kind": unit["kind"], "cursor": unit["cursor"],
                                "tokens": unit["tokens"], "capped": unit["capped"],
                                "strict_parsed": unit["answer"] is not None,
                                "strict_correct": is_correct(unit["answer"], labels[unit["id"]]),
                                "extra_parsed": unit["eval_only_answer"] is not None,
                                "extra_correct": is_correct(unit["eval_only_answer"], labels[unit["id"]]),
                                "failure_type": unit["failure_type"],
                                "initial_strict_correct_probe": unit["kind"] == "probe" and unit["id"] in initial_correct,
                                "semantic_manual_status": "not_assessed_no_semantic_guess"})
    atomic_json(dest / (round_name + "_discrepancies.json"), discrepancies)
    if discrepancies:
        raise ValueError("Original sealed scores differ; mechanism analysis blocked")
    groups = []
    for arm in "mv":
        for group in [f"S{i+1:02d}" for i in range(16)]:
            observations = [r for r in private_rows if r["arm"] == arm and r["group"] == group]
            assert len(observations) == 3
            groups.append({"round": round_name, "arm": arm, "group": group, "observations": 3, "missing": 0,
                           "consensus_correct_observations": sum(r["consensus_correct"] for r in observations),
                           "unique_consensus_answers": len({r["consensus"] for r in observations}),
                           "all_consensus_wrong": all(not r["consensus_correct"] for r in observations),
                           "correct_candidate_groups": sum(r["correct_candidate_exists"] for r in observations),
                           "correct_candidates_of_24": sum(r["correct_candidate_count"] for r in observations),
                           "correct_candidate_wrong_vote": sum(r["correct_candidate_exists"] and not r["consensus_correct"] for r in observations),
                           "top_vote_counts": [r["top_count"] for r in observations],
                           "vote_margins": [r["margin"] for r in observations],
                           "history_gain_observations": sum(not r["frozen_correct"] and r["before_correct"] for r in observations),
                           "history_loss_observations": sum(r["frozen_correct"] and not r["before_correct"] for r in observations)})
    atomic_json(dest / (round_name + "_private.json"), {"mapping": aliases, "rows": private_rows})
    atomic_json(dest / (round_name + "_public.json"), {"primary": primary, "cases": cases, "formats": formats,
                "groups": groups, "recomputed_matches_sealed": True, "evaluator_source": source,
                "manifest_sha256": digest, "unique_stream_groups": 16, "unique_probe_groups": 16})
    print(json.dumps({"round": round_name, "trajectories": 6, "case_rows": len(cases),
                      "format_rows": len(formats), "all_original_metrics_match": True}), flush=True)


if __name__ == "__main__":
    audit()
