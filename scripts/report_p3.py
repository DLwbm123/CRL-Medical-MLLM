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
FAILURE_TYPES = ["normal_end_missing_final_marker_or_explicit_choice",
                 "explicit_choice_unsupported_syntax", "token_cap_without_parseable_final_choice",
                 "conflicting_or_nonunique_final_choices", "prose_final_payload_without_unambiguous_identifier",
                 "explicit_choice_outside_legal_options", "other_or_undetermined"]


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def transition_columns(prefix, value):
    return {prefix + key: value.get(key) for key in TRANSITIONS}


def markdown_table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
                     + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def format_report(public, p2, sources):
    failures, paired, votes = [], [], []
    for stage, seed, key, method in sources:
        label = METHODS[key] + (" / " + str(seed) if seed is not None else " / shared cache")
        audit = method["format"]
        counts = audit["failure_types"]
        assert sum(counts.get(name, 0) for name in FAILURE_TYPES) == audit["original_invalid"]
        failures.append([stage, label, audit["outputs"], audit["original_invalid"],
                         *[counts.get(name, 0) for name in FAILURE_TYPES]])
        for row in audit["paired_scores"]:
            t = row["transition"]
            paired.append([stage, label, row["kind"], row["cursor"] if row["cursor"] is not None else "all",
                           row["n"], str(t["before_parsed"]) + " → " + str(t["after_parsed"]),
                           str(row["original_correct"]) + " → " + str(row["eval_only_correct"]),
                           t["newly_parsed"], t["newly_unparseable"], t["invalid_to_correct"],
                           t["correct_to_invalid"], t["correct_to_parsed_wrong"]])
        if key != "a":
            original = method.get("original_scores", method)["consensus"]
            votes.append([stage, label, original["total"], original["correct"], method["eval_only_consensus"]["correct"]])
    readout, lost, tokenization = [], [], []
    for actor, value in public["readout"].items():
        t = value["same_actor_free_vs_legal_readout"]
        free_correct = t["correct_to_correct"] + t["correct_to_wrong"]
        readout.append([actor, value["n"], free_correct, t["before_parsed"], value["correct"], value["parsed"],
                        t["parsed_wrong_to_correct"], t["invalid_to_correct"],
                        t["correct_to_parsed_wrong"], t["correct_to_invalid"]])
        fraction = value["readout_correct_among_original_free_correct_to_invalid"]
        lost.append([actor, fraction["correct"], fraction["total"],
                     "NA" if fraction["percent"] is None else f'{fraction["percent"]:.2f}'])
        tokenization.append([actor, value["option_token_length_counts"]])
    return "\n\n".join([
        "# P3 output-format and legal-option readout audit",
        "The P2 original four-method, candidate and probe aggregates/transitions were reproduced exactly. "
        f'All {p2["invalid_outputs_reviewed"]} P2 invalid records and {p2["stratified_parsed_controls"]} '
        "stratified parsed controls were reviewed privately. Classification used answer-marker/ending evidence, "
        "without guessing intent from truth labels or clinical plausibility. P3 uses the same locked offline rules.",
        "## Failure counts",
        "Counts cover stream greedy outputs, rollouts and every retained probe output. They are output records, "
        "not independent patients. P2 has three probe cursors; P3 has two, so their raw failure totals are not directly comparable. "
        "Missing denotes missing final marker/explicit choice; syntax denotes explicit unsupported choice syntax; "
        "cap denotes reaching 2048 tokens without a parseable final choice. A syntax failure can include an explicit "
        "identifier followed by text, while the extra parser recovers only a bare identifier. Categories are exclusive "
        "under the locked priority order; a record counted as syntax can also have reached the cap.",
        markdown_table(["Stage", "Method / seed", "Outputs", "Invalid", "Missing", "Syntax", "Cap", "Conflict", "Prose", "Illegal ID", "Other"], failures),
        "## Original versus eval-only syntax parser",
        "`marked_choice_v1_eval_only` adds explicit, unambiguous bare identifiers after marked answer fields, "
        "including newline, Markdown and boxed syntax. Conflicting marked identifiers abstain. It does not infer "
        "clinical answers or treat option lists/reasoning candidates as final answers. The P2 original parser and "
        "consensus rewards remain the primary scoring/training protocol; P2 reports are unchanged. The same extra "
        "rules apply to every method, seed, output type and probe cursor. Parsed/correct entries below show original → extra syntax.",
        markdown_table(["Stage", "Method / seed", "Output", "Cursor", "N", "Parsed", "Correct", "New parsed", "New invalid", "Invalid→correct", "Correct→invalid", "Correct→parsed wrong"], paired),
        "Consensus is recomputed uniformly for this diagnostic using the extra parser and original majority/tie rule. "
        "Stored original rewards and actor updates are never recomputed or replaced.",
        markdown_table(["Stage", "Method / seed", "Groups", "Original vote correct", "Extra-syntax vote correct"], votes),
        "## Fixed legal-option likelihood readout on old P2 actors",
        "All four actors used all 16 old probe groups, the same image/question/options, and an answer-only instruction "
        "with assistant prefix `Final answer:`. For identifier k with tokens t₁…tₘ, score(k) = Σⱼ log p(tⱼ | common prefix, t<ⱼ). "
        "The complete identifier sequence is scored, without EOS or length normalization. Maximum score wins; ties "
        "choose the lexicographically first identifier. Tokenization and option scores remain private. No generated "
        "reasoning, truth labels, optimizer or parameter update enters this path; checkpoint loading is read-only "
        "and inference preserves RNG/model modes. These are P2 actors, not newly adapted P3 actors.",
        markdown_table(["Actor", "N", "Free correct", "Free parsed", "Legal correct", "Legal parsed", "Parsed wrong→correct", "Invalid→correct", "Correct→parsed wrong", "Correct→invalid"], readout),
        "Observed identifier token lengths (length: count) are recorded below. The CPU check also covers a "
        "two-token identifier and checks the complete summed score rather than a first-token shortcut.",
        markdown_table(["Actor", "Identifier token-length counts"], tokenization),
        "The following subset is diagnostic only: cases initially correct under frozen free generation that became "
        "invalid under the indicated actor's old free generation. Every actor was evaluated on the full probe first.",
        markdown_table(["Actor", "Legal correct in lost subset", "Subset denominator", "Percent"], lost),
        "The free-generation and legal-readout protocols are different conditional distributions. A difference in "
        "correct counts is not a pure-format-loss estimate. Legal validity alone is not medical competence; preserved "
        "or reduced legal-readout accuracy cannot prove retention or forgetting of a particular kind of medical knowledge. "
        "Format/reward associations do not identify the cause of any individual update's degradation.",
    ]) + "\n"


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
            row.update(transition_columns("spine_vs_ttrl_", context["spine_vs_ttrl_before"] if key == "d" else {}))
            for probe in method["probes"]:
                for metric in ["correct", "parsed", "n"]:
                    row[f'probe_{probe["cursor"]}_{metric}'] = probe[metric]
            for metric in method["format"]["paired_scores"]:
                if metric["kind"] in {"before", "after"}:
                    row["eval_only_" + metric["kind"] + "_correct"] = metric["eval_only_correct"]
                    row["eval_only_" + metric["kind"] + "_parsed"] = metric["transition"]["after_parsed"]
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
    (destination / "p3_format_audit.md").write_text(format_report(public, p2, sources))
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
