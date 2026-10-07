"""Offline reward direction and format diagnostics; never called by a trainer."""
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from core import consensus_rewards, extract_answer
from evaluate import fraction, is_correct, prediction_transition_audit, read_cases, score_cases
from state import atomic_json

PARSER_VERSION = "marked_choice_v1_eval_only"
MARKER = re.compile(
    r"(?<!\w)(?:final[ \t*_`]*answer|(?:the[ \t]+)?correct[ \t*_`]*answer|answer|最终答案|答案)"
    r"[ \t*_`]*(?:[：:]|[ \t]+is\b|(?=\n))\s*", re.I)


def marked_choices(text, options):
    text = unicodedata.normalize("NFKC", text)
    matches = list(MARKER.finditer(text))
    choices, payloads = [], []
    for i, match in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        lines = [line.strip() for line in text[match.end():stop].splitlines() if line.strip()]
        lines = [line for line in lines if not line.startswith("```")]
        payload = lines[0] if lines else ""
        payloads.append(payload)
        clean = re.sub(r"\\(?:boxed|text|mathrm|mathbf)\s*\{", "{", payload)
        clean = re.sub(r"[*`$]", "", clean).strip()
        # Added recoveries require an explicit marker and a bare identifier.
        # Clinical prose, option lists and reasoning discussions are not mapped.
        found = re.fullmatch(r"[\s({\[]*([A-Za-z])[\s)}\].,;:!?]*", clean)
        if found and found.group(1).upper() in options:
            choices.append(found.group(1).upper())
    return choices, payloads


def eval_only_choice(text, options):
    choices, _ = marked_choices(text, options)
    if len(set(choices)) > 1:
        return None
    original = extract_answer(text, options)
    if original is not None:
        return original if not choices or choices[-1] == original else None
    return choices[0] if choices else None


def failure_type(text, options, capped):
    if extract_answer(text, options) is not None:
        return "parsed_control"
    choices, payloads = marked_choices(text, options)
    if len(set(choices)) > 1 or any(re.fullmatch(r"[\s*`({\[]*[A-Za-z][\s*`)}\].]*(?:or|and|/)[\s*`({\[]*[A-Za-z][\s*`)}\].]*", x, re.I) for x in payloads):
        return "conflicting_or_nonunique_final_choices"
    if eval_only_choice(text, options) is not None:
        return "explicit_choice_unsupported_syntax"
    if capped:
        return "token_cap_without_parseable_final_choice"
    if payloads and any(payloads):
        return "prose_final_payload_without_unambiguous_identifier"
    if text.strip() and not payloads:
        return "normal_end_missing_final_marker_or_explicit_choice"
    return "other_or_undetermined"


def read_labels(root, manifest):
    # This function is used exclusively by the separate offline evaluators.
    wanted = {entry["id"] for section in ["stream", "probe"] for entry in manifest[section]}
    labels = {}
    for split in sorted({entry["split"] for section in ["stream", "probe"] for entry in manifest[section]}):
        path = root / "views/evaluation_labels/MedXpertQA-MM" / (split + ".jsonl")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["id"] in wanted:
                if row["id"] in labels:
                    raise ValueError("Duplicate requested label ID")
                labels[row["id"]] = row["label"]
    if set(labels) != wanted:
        raise ValueError("Missing requested labels")
    return labels


def options_by_id(folder, manifest):
    result = {}
    for section in ["stream", "probe"]:
        for entry in manifest[section]:
            row = json.loads((folder / entry["input"]).read_text())
            result[row["id"]] = row["options"]
    return result


def output_units(run, rows, options):
    units = []
    for row in rows:
        for kind in ["before", "after"]:
            if kind in row:
                value = row[kind]
                units.append({"id": row["id"], "kind": kind, "cursor": row["index"] + 1,
                              "slot": 0, **value})
        for slot, text in enumerate(row.get("completions", [])):
            units.append({"id": row["id"], "kind": "rollout", "cursor": row["index"] + 1,
                          "slot": slot, "text": text, "answer": row["vote"]["answers"][slot],
                          "tokens": row["response_lengths"][slot]})
    for path in sorted((run / "probe").glob("c*.json")):
        record = json.loads(path.read_text())
        for slot, value in enumerate(record["predictions"]):
            units.append({"kind": "probe", "cursor": record["cursor"], "slot": slot, **value})
    for unit in units:
        choices = options[unit["id"]]
        if extract_answer(unit["text"], choices) != unit["answer"]:
            raise ValueError("Original parser differs from stored predictions")
        unit["capped"] = unit["tokens"] == 2048
        unit["eval_only_answer"] = eval_only_choice(unit["text"], choices)
        unit["failure_type"] = failure_type(unit["text"], choices, unit["capped"])
    return units


def reward_direction(rows, options, labels):
    counts = Counter()
    bins = defaultdict(lambda: Counter())
    groups = []
    correct_advantages = []
    for row in rows:
        if "vote" not in row:
            continue
        vote, identifier = row["vote"], row["id"]
        answers = vote["answers"]
        if vote["all_unparseable"]:
            rewards, advantages = [0.0] * len(answers), [None] * len(answers)
        else:
            reward, advantage, rebuilt = consensus_rewards(row["completions"], options[identifier])
            if rebuilt["answers"] != answers or rebuilt["winner"] != vote["winner"] or reward.tolist() != row["rewards"]:
                raise ValueError("Stored original rewards cannot be reconstructed")
            rewards, advantages = reward.tolist(), advantage.tolist()
        correct = [is_correct(answer, labels[identifier]) for answer in answers]
        coverage, voted_correct = any(correct), is_correct(vote["winner"], labels[identifier])
        counts.update(groups=1, rollouts=len(answers), coverage=int(coverage), vote_correct=int(voted_correct),
                      candidate_but_wrong_vote=int(coverage and not voted_correct), ties=int(vote["tie"]),
                      fewer_than_eight_valid=int(vote["valid"] < 8), all_invalid=int(vote["all_unparseable"]),
                      zero_advantage=int(vote["zero_advantage_group"]))
        for answer, right, reward, advantage in zip(answers, correct, rewards, advantages):
            counts["parsed_rollouts"] += answer is not None
            counts["correct_parsed_rollouts"] += right
            counts["correct_negative_advantage"] += right and advantage is not None and advantage < 0
            counts["positive_reward_rollouts"] += reward > 0
            counts["wrong_parsed_positive_reward"] += reward > 0 and answer is not None and not right
            if right:
                correct_advantages.append(advantage)
        tags = {
            "top_votes": "0-3" if vote["top_count"] <= 3 else "4-5" if vote["top_count"] <= 5 else "6-8",
            "vote_margin": "0" if vote["margin"] == 0 else "1-2" if vote["margin"] <= 2 else "3-8",
            "valid_fraction": "0-3/8" if vote["valid"] < 4 else "4-7/8" if vote["valid"] < 8 else "8/8",
        }
        for axis, name in tags.items():
            bins[(axis, name)].update(groups=1, vote_correct=int(voted_correct), coverage=int(coverage),
                                     candidate_but_wrong_vote=int(coverage and not voted_correct))
        groups.append({"id": identifier, "index": row["index"], "valid": vote["valid"],
                       "top_votes": vote["top_count"], "margin": vote["margin"],
                       "vote_correct": voted_correct, "correct_candidate_exists": coverage})
    metrics = {
        "empirical_candidate_coverage_at_8": fraction(counts["coverage"], counts["groups"]),
        "vote_accuracy": fraction(counts["vote_correct"], counts["groups"]),
        "correct_candidate_but_wrong_vote": fraction(counts["candidate_but_wrong_vote"], counts["groups"]),
        "wrong_vote_given_correct_candidate": fraction(counts["candidate_but_wrong_vote"], counts["coverage"]),
        "correct_negative_advantage": fraction(counts["correct_negative_advantage"], counts["correct_parsed_rollouts"]),
        "wrong_parsed_positive_reward": fraction(counts["wrong_parsed_positive_reward"], counts["positive_reward_rollouts"]),
        "ties": fraction(counts["ties"], counts["groups"]),
        "fewer_than_eight_valid": fraction(counts["fewer_than_eight_valid"], counts["groups"]),
        "all_invalid": fraction(counts["all_invalid"], counts["groups"]),
        "zero_advantage": fraction(counts["zero_advantage"], counts["groups"]),
        "parsed_rollouts": fraction(counts["parsed_rollouts"], counts["rollouts"]),
    }
    return {"counts": dict(counts), "metrics": metrics,
            "mean_advantage_correct_parsed": sum(correct_advantages) / len(correct_advantages) if correct_advantages else None,
            "correct_advantage_denominator": len(correct_advantages),
            "fixed_bins": [{"axis": axis, "bin": name, **dict(value),
                            "vote_accuracy": fraction(value["vote_correct"], value["groups"])}
                           for (axis, name), value in sorted(bins.items())]}, groups


def summarize_formats(units, labels):
    types = Counter(unit["failure_type"] for unit in units)
    summaries = []
    for kind, cursor in sorted({(unit["kind"], unit["cursor"] if unit["kind"] == "probe" else None) for unit in units}):
        selected = [unit for unit in units if unit["kind"] == kind and (cursor is None or unit["cursor"] == cursor)]
        original = [{"id": unit["id"], "answer": unit["answer"]} for unit in selected]
        revised = [{"id": unit["id"], "answer": unit["eval_only_answer"]} for unit in selected]
        summaries.append({"kind": kind, "cursor": cursor, "n": len(selected),
                          "original_correct": sum(is_correct(unit["answer"], labels[unit["id"]]) for unit in selected),
                          "eval_only_correct": sum(is_correct(unit["eval_only_answer"], labels[unit["id"]]) for unit in selected),
                          "transition": prediction_transition_audit(original, revised, labels)})
    return {"outputs": len(units), "original_invalid": sum(unit["answer"] is None for unit in units),
            "failure_types": dict(types), "parser_version": PARSER_VERSION, "paired_scores": summaries}


def audit_p2(root, folder, destination):
    raw = (folder / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip():
        raise ValueError("P2 manifest checksum differs")
    manifest = json.loads(raw)
    loaded = {key: read_cases(folder / "runs" / ("main-" + key), manifest, digest) for key in "abcd"}
    options = options_by_id(folder, manifest)
    units = {key: output_units(folder / "runs" / ("main-" + key), rows, options) for key, (rows, _) in loaded.items()}
    # Prediction structure and original parse outputs were checked before labels.
    labels = read_labels(root, manifest)
    published = json.loads((folder / "scores/public_summary.json").read_text())
    base = [is_correct(row["before"]["answer"], labels[row["id"]]) for row in loaded["a"][0]]
    methods, private_groups = {}, {}
    failures, controls = [], []
    for key, (rows, _) in loaded.items():
        recomputed, _ = score_cases(rows, labels, base)
        for metric, value in recomputed.items():
            if published["methods"][key][metric] != value:
                raise ValueError("P2 original aggregate differs: " + key + "/" + metric)
        run = folder / "runs" / ("main-" + key)
        probes = []
        for path in sorted((run / "probe").glob("c*.json")):
            record = json.loads(path.read_text())
            cursor = record["cursor"]
            stored = next(x for x in published["methods"][key]["probe"] if x["cursor"] == cursor)
            predictions = record["predictions"]
            right = sum(is_correct(x["answer"], labels[x["id"]]) for x in predictions)
            parsed = sum(x["answer"] is not None for x in predictions)
            if (right, parsed, len(predictions)) != (stored["correct"], stored["parsed"], stored["total"]):
                raise ValueError("P2 original probe aggregate differs")
            initial = json.loads((run / "probe/c000000.json").read_text())["predictions"]
            transition = prediction_transition_audit(initial, predictions, labels)
            if {"cursor": cursor, **transition} != next(x for x in published["methods"][key]["probe_transition_audit"] if x["cursor"] == cursor):
                raise ValueError("P2 original probe transitions differ")
            probes.append({"cursor": cursor, "correct": right, "parsed": parsed, "n": len(predictions)})
        method = {"original_scores_match": True, "original_scores": recomputed, "original_probe": probes,
                  "format": summarize_formats(units[key], labels)}
        if key != "a":
            method["reward_direction"], private_groups[key] = reward_direction(rows, options, labels)
        methods[key] = method
        control_counts = Counter()
        for unit in units[key]:
            record = {"method": key, **unit}
            if unit["answer"] is None:
                failures.append(record)
            else:
                bucket = (unit["kind"], unit["cursor"] if unit["kind"] == "probe" else None)
                if control_counts[bucket] < 2:
                    controls.append(record)
                    control_counts[bucket] += 1
    atomic_json(destination / "p2_diagnostics.json", {"source_manifest_sha256": digest,
                "original_scores_match": True, "primary_parser_unchanged": True, "methods": methods,
                "invalid_outputs_reviewed": len(failures), "stratified_parsed_controls": len(controls),
                "interpretation": "offline associations, not causal attribution of individual degradation"})
    atomic_json(destination / "private_format_failures.json", failures)
    atomic_json(destination / "private_format_controls.json", controls)
    atomic_json(destination / "private_reward_groups.json", private_groups)
    print(json.dumps({"p2_original_scores_match": True, "invalid_outputs": len(failures),
                      "parsed_controls": len(controls), "diagnostics_written": True}))


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = Path(os.environ["P3_P2_FOLDER"])
    destination = root / "outputs" / os.environ["P2_CAMPAIGN"] / "diagnostics"
    audit_p2(root, folder, destination)


if __name__ == "__main__":
    main()
