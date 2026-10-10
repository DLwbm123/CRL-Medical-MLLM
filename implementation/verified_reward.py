"""Frozen visual pseudo-rewards; clinical labels are never accepted here."""
import json
import math


def judge_messages(row, reverse=False, evidence_first=False):
    from core import INPUT_KEYS
    if set(row) != INPUT_KEYS or not row["options"] or not row["images"] or len(row["images"]) != len(row["image_paths"]):
        raise ValueError("Verifier requires an exact label-free ordered-image MCQ")
    keys = sorted(row["options"], reverse=reverse)
    mapping = {chr(65 + i): key for i, key in enumerate(keys)}
    text = row["question"] + "\nOptions:\n" + "\n".join(f"{alias}. {row['options'][key]}" for alias, key in mapping.items())
    text += ('\nEvaluate the images and the question independently. No candidate votes or reference answer are supplied. '
             'Reason briefly about visual findings and clinical compatibility. If either is insufficient, say uncertain. '
             'End with exactly one JSON object on its own final line, with only these keys: '
             '{"choice":"<one option letter>","visual_support":"supported or uncertain",'
             '"clinical_consistency":"consistent or uncertain"}. Do not put instructions or extra text after the JSON.')
    if evidence_first:
        text += (' Before the final JSON, write exactly two short sentences. '
                 'First describe specific findings actually visible in the supplied images. '
                 'Second explain how those findings support or fail to support the chosen option in the clinical question. '
                 'Do not invent visual findings or infer them solely from the question. '
                 'If the visual evidence or clinical compatibility is insufficient, keep the corresponding judgment uncertain.')
    return [{"role": "user", "content": [{"type": "image", "image": p} for p in row["image_paths"]] + [{"type": "text", "text": text}]}], mapping


def parse_judgment(text, mapping, json_suffix=False):
    if json_suffix:
        def unique_object(pairs):
            value = dict(pairs)
            if len(value) != len(pairs):
                raise ValueError("Duplicate JSON keys")
            return value
        decoder = json.JSONDecoder(object_pairs_hook=unique_object)
        objects = []
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, end = decoder.raw_decode(text[index:])
            except ValueError:
                continue
            if isinstance(value, dict):
                objects.append((value, text[index + end:].strip()))
        if len(objects) != 1 or objects[0][1]:
            return None
        value = objects[0][0]
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        try:
            value = json.loads(lines[-1])
        except (ValueError, IndexError):
            return None
    if not isinstance(value, dict) or set(value) != {"choice", "visual_support", "clinical_consistency"}:
        return None
    if not all(isinstance(v, str) for v in value.values()) or value["choice"] not in mapping or value["visual_support"] not in {"supported", "uncertain"} or value["clinical_consistency"] not in {"consistent", "uncertain"}:
        return None
    return {**value, "choice": mapping[value["choice"]]}


def agreed_target(judgments):
    if len(judgments) != 2:
        raise ValueError("Exactly two prespecified option orders are required")
    if any(j is None or j["visual_support"] != "supported" or j["clinical_consistency"] != "consistent" for j in judgments):
        return None
    return judgments[0]["choice"] if judgments[0]["choice"] == judgments[1]["choice"] else None


def checked_targets(signal, manifest, digest):
    if signal.get("labels_read") is not False or signal.get("weights_unchanged") is not True or signal.get("sealed") is not True or signal.get("manifest_sha256") != digest:
        raise ValueError("Frozen verifier provenance is incomplete or differs")
    if signal.get("teacher_revision") != manifest["teacher"]["revision"]:
        raise ValueError("Frozen teacher revision differs")
    predictions = signal["stream"]
    if [p["id"] for p in predictions] != [e["id"] for e in manifest["stream"]]:
        raise ValueError("Reward signal scope/order differs from the locked stream")
    for p in predictions:
        if p["target"] != agreed_target(p["judgments"]) or (p["target"] is not None and p["target"] not in p["option_keys"]):
            raise ValueError("Reward target differs from the two-order decision")
    return {p["id"]: p["target"] for p in predictions}


def verified_group_rewards(completions, options, target):
    import torch
    from continual import group_rewards
    rewards, advantages, detail = group_rewards(completions, options, target)
    detail.update(reward_source="frozen_visual", reward_target=target)
    if target is None:
        if rewards is not None:
            rewards, advantages = torch.zeros_like(rewards), torch.zeros_like(advantages)
        detail.update(skip_update_reason="verifier_abstained", zero_advantage_group=True)
    elif detail["zero_advantage_group"]:
        detail["skip_update_reason"] = "zero_advantage"
    return rewards, advantages, detail


def qualification_gate(majority, verified, target_correct, accepted, minority_rescued):
    # The reused sixteen groups provide a development gate, not independent validation.
    checks = {
        "coverage_at_least_half": accepted >= 8,
        "at_least_four_correct_targets": target_correct >= 4,
        "accepted_precision_at_least_half": accepted > 0 and target_correct / accepted >= .5,
        "correct_negative_rate_lower": verified["correct_negative"]["percent"] is not None and majority["correct_negative"]["percent"] is not None and verified["correct_negative"]["percent"] < majority["correct_negative"]["percent"],
        "wrong_positive_rate_ten_pp_lower": verified["wrong_positive"]["percent"] is not None and majority["wrong_positive"]["percent"] is not None and verified["wrong_positive"]["percent"] <= majority["wrong_positive"]["percent"] - 10,
        "correct_minority_rescued": minority_rescued >= 1,
    }
    return {"go": all(checks.values()), "checks": checks}


def admitted_seconds(remaining, round_remaining, estimate, reserve=.2):
    if not all(math.isfinite(x) and x >= 0 for x in [remaining, round_remaining, estimate, reserve]):
        raise ValueError("Nonfinite or negative compute admission")
    return min(remaining, round_remaining) >= estimate * (1 + reserve)
