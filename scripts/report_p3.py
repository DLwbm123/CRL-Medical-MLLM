"""Export explicit public aggregate tables; no medical text, IDs or label files."""
import csv
import json
import os
from pathlib import Path

METHODS = {"a": "Frozen greedy", "b": "Frozen SC-8", "c": "TTRL", "d": "SPINE"}
TRANSITIONS = ["parsed_wrong_to_correct", "invalid_to_correct", "correct_to_parsed_wrong",
               "correct_to_invalid", "correct_to_correct", "newly_parsed", "newly_unparseable"]
FIXED_BINS = {"top_votes": ["0-3", "4-5", "6-8"], "vote_margin": ["0", "1-2", "3-8"],
              "valid_fraction": ["0-3/8", "4-7/8", "8/8"]}


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def transition_columns(prefix, value):
    return {prefix + key: value.get(key) for key in TRANSITIONS}


def export_tables(public, p2, destination):
    main, reward, probes = [], [], []
    sources = [("P2", 42, key, value) for key, value in p2["methods"].items()]
    for seed, context in public["seeds"].items():
        for key, method in context["methods"].items():
            row = {"seed_context": seed, "rollout_seed": None if key == "a" else seed,
                   "method": METHODS[key], "n": method["n"], "actual_coverage": method["actual_completed"],
                   "shared_frozen_cache": method["cached"]}
            for metric in ["greedy_before", "greedy_after", "consensus"]:
                row[metric + "_correct"] = method.get(metric, {}).get("correct")
            row.update(before_parsed=method.get("before_parse_rate", {}).get("correct"),
                       after_parsed=method.get("after_parse_rate", {}).get("correct"),
                       consensus_parsed_groups=method["n"] - method["reward_direction"]["counts"].get("all_invalid", 0) if key != "a" else None,
                       historical_gain_pp=method.get("historical_gain_vs_frozen_pp"),
                       current_gain_pp=method.get("current_case_transitions", {}).get("gain_pp"),
                       delta_spine_vs_ttrl_before_pp=context["delta_spine_vs_ttrl_before_pp"] if key == "d" else None)
            row.update(transition_columns("history_", method.get("history_parse_transition", {})))
            row.update(transition_columns("current_", method.get("current_parse_transition", {})))
            for metric in method["format"]["paired_scores"]:
                if metric["kind"] in {"before", "after"}:
                    row["eval_only_" + metric["kind"] + "_correct"] = metric["eval_only_correct"]
            row["eval_only_consensus_correct"] = method.get("eval_only_consensus", {}).get("correct")
            for mode in ["before", "after"]:
                correct = row["greedy_" + mode + "_correct"]
                if correct is not None:
                    assert 0 <= correct <= row[mode + "_parsed"] <= row["n"]
            main.append(row)
            if key != "a" or seed == "43":
                sources.append(("P3", None if key == "a" else int(seed), key, method))
    for stage, seed, key, method in sources:
        audit = method.get("reward_direction")
        if audit:
            for metric, value in audit["metrics"].items():
                assert 0 <= value["correct"] <= value["total"]
                level = "rollout" if metric in {"parsed_rollouts", "correct_negative_advantage", "wrong_parsed_positive_reward"} else "group"
                reward.append({"stage": stage, "seed": seed, "method": METHODS[key], "level": level,
                               "metric": metric, "numerator": value["correct"], "denominator": value["total"],
                               "percent": value["percent"], "mean_advantage": None})
            reward.append({"stage": stage, "seed": seed, "method": METHODS[key], "level": "rollout",
                           "metric": "mean_advantage_correct_parsed", "numerator": None,
                           "denominator": audit["correct_advantage_denominator"], "percent": None,
                           "mean_advantage": audit["mean_advantage_correct_parsed"]})
            for axis, names in FIXED_BINS.items():
                for name in names:
                    value = next((x for x in audit["fixed_bins"] if x["axis"] == axis and x["bin"] == name), {})
                    n, correct = value.get("groups", 0), value.get("vote_correct", 0)
                    reward.append({"stage": stage, "seed": seed, "method": METHODS[key], "level": "group",
                                   "metric": "vote_accuracy_" + axis + "_" + name, "numerator": correct,
                                   "denominator": n, "percent": 100 * correct / n if n else None,
                                   "mean_advantage": None})
        for probe in method.get("original_probe", method.get("probes", [])):
            row = {"stage": stage, "seed": seed, "method": METHODS[key], "readout_protocol": "original_free",
                   "cursor": probe["cursor"], "n": probe["n"], "correct": probe["correct"], "parsed": probe["parsed"]}
            row.update(transition_columns("from_initial_", probe.get("transition_from_initial", {})))
            probes.append(row)
            revised = next(x for x in method["format"]["paired_scores"] if x["kind"] == "probe" and x["cursor"] == probe["cursor"])
            probes.append({**row, "readout_protocol": "eval_only_syntax", "correct": revised["eval_only_correct"],
                           "parsed": revised["transition"]["after_parsed"],
                           **transition_columns("from_initial_", {})})
    for actor, value in public["readout"].items():
        probes.append({"stage": "P2_actor_diagnosis", "seed": None, "method": actor,
                       "readout_protocol": "legal_option_likelihood", "cursor": value["provenance"].get("cursor", 0),
                       "n": value["n"], "correct": value["correct"], "parsed": value["parsed"],
                       **transition_columns("from_initial_", value.get("transition_vs_frozen_readout", {}))})
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "p3_main_results.csv", main)
    write_csv(destination / "p3_reward_audit.csv", reward)
    write_csv(destination / "p3_probe_results.csv", probes)
    return {"main_rows": len(main), "reward_rows": len(reward), "probe_rows": len(probes)}


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    public = json.loads((folder / "scores/public_summary.json").read_text())
    p2 = json.loads((folder / "diagnostics/p2_diagnostics.json").read_text())
    counts = export_tables(public, p2, root / "reports")
    print(json.dumps({"aggregate_tables_written": True, **counts, "labels_opened": False}))


if __name__ == "__main__":
    main()
