"""Fixed visual answer contrast and retired-development calibration; no GPU imports."""
import json
import math
import os
from pathlib import Path
from state import atomic_json

SEEDS = [96, 97, 98]
TOTAL = 518400.
REPO = "lingshu-medical-mllm/Lingshu-7B"
REVISION = "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"
SCHEME = "calibrated_visual_v1"


def messages(row, reverse=False):
    from core import INPUT_KEYS
    if set(row) != INPUT_KEYS or row["options"] != {"A": "yes", "B": "no"} or len(row["images"]) != 1 or len(row["image_paths"]) != 1:
        raise ValueError("Exact answer-free binary image input required")
    keys = ["B", "A"] if reverse else ["A", "B"]
    mapping = dict(zip(["A", "B"], keys))
    text = row["question"] + "\nOptions:\n" + "\n".join(f"{alias}. {row['options'][key]}" for alias, key in mapping.items())
    text += "\nUse the supplied image to answer the question. Output only the option letter."
    return [{"role": "user", "content": [{"type": "image", "image": row["image_paths"][0]}, {"type": "text", "text": text}]}], mapping


def probabilities(scores, temperature):
    if set(scores) != {"A", "B"} or not .25 <= temperature <= 8 or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Invalid fixed binary score or temperature")
    delta = max(-700., min(700., (scores["B"] - scores["A"]) / temperature))
    a = 1 / (1 + math.exp(delta))
    return {"A": a, "B": 1 - a}


def decision(orders, calibration):
    if len(orders) != 2 or not calibration.get("admitted") or not .75 <= calibration["threshold"] <= 1:
        raise ValueError("Missing frozen two-order calibration")
    winners, confidence = [], []
    for index, order in enumerate(orders):
        if order.get("reverse") is not bool(index) or set(order["real"]) != {"A", "B"} or set(order["masked"]) != {"A", "B"}:
            raise ValueError("Two canonical orders differ")
        contrast = {key: 2 * order["real"][key] - order["masked"][key] for key in ["A", "B"]}
        p = probabilities(contrast, calibration["temperature"])
        winner = min(p, key=lambda key: (-p[key], key))
        winners.append(winner); confidence.append(p[winner])
    accepted = winners[0] == winners[1] and min(confidence) >= calibration["threshold"]
    return winners[0] if accepted else None, min(confidence), winners[0] == winners[1]


def calibrate(rows, labels):
    if len(rows) != 64 or [r["id"] for r in rows] != list(labels) or any(v not in {"A", "B"} for v in labels.values()):
        raise ValueError("All64 retired labels and ordered readouts required")
    # One scalar NLL fit; no prompt, mask, score-factor or new-case threshold search.
    def loss(log_temperature):
        t = min(8., max(.25, math.exp(log_temperature)))
        return sum(-math.log(max(probabilities({k: 2 * o["real"][k] - o["masked"][k] for k in ["A", "B"]}, t)[labels[r["id"]]], 1e-300)) for r in rows for o in r["orders"]) / 128
    lo, hi = math.log(.25), math.log(8)
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if loss(a) <= loss(b): hi = b
        else: lo = a
    candidates = [math.log(.25), (lo + hi) / 2, math.log(8)]
    log_t = min(candidates, key=lambda v: (loss(v), v)); t = min(8., max(.25, math.exp(log_t)))
    basic = {"admitted": True, "temperature": t, "threshold": .75}
    observations = [(r, *decision(r["orders"], basic)) for r in rows]
    thresholds = sorted({.75} | {confidence for _, _, confidence, agree in observations if agree and confidence >= .75})
    result = {"admitted": False, "temperature": t, "threshold": None, "retired_groups": 64,
              "calibration_labels_read_after_seal": True, "new_labels_used": False, "nll": loss(log_t)}
    for threshold in thresholds:
        accepted = [(r, target) for r, target, confidence, agree in observations if agree and target is not None and confidence >= threshold]
        correct = sum(target == labels[r["id"]] for r, target in accepted)
        if len(accepted) >= 32 and correct >= 24 and correct / len(accepted) >= .75:
            return {**result, "admitted": True, "threshold": threshold, "accepted": len(accepted), "correct": correct}
    return {**result, "accepted": 0, "correct": 0, "failure": "no_threshold_meets_fixed_retired_calibration_rule"}


def validate(manifest):
    if manifest.get("p23_version") != 1 or manifest["rollout_seeds"] != SEEDS or manifest["primary_teacher"] != "v" or manifest["training_authorized_in_this_round"]:
        raise ValueError("Fixed P23 complete scope differs")
    if (len(manifest["calibration_entries"]), len(manifest["stream"]), len(manifest["probe"])) != (64, 32, 32) or set(manifest["teachers"]) != {"v", "c"}:
        raise ValueError("Calibration/fresh paired scope incomplete")
    for spec in manifest["teachers"].values():
        if (spec["repo"], spec["revision"]) != (REPO, REVISION): raise ValueError("Fixed medical source differs")
        if spec.get("max_new_tokens") != 1024 or spec.get("evidence_first") is not True or spec.get("json_suffix") is not True:
            raise ValueError("Matched JSON control settings changed")
    if manifest["teachers"]["v"] != manifest["teachers"]["c"] or manifest["teacher"] != manifest["teachers"]["v"]:
        raise ValueError("Matched teacher checkpoint differs")
    if manifest["gate"] != {"minimum_accepted": 16, "minimum_correct": 12, "minimum_precision": .75, "wrong_positive_drop_pp": 10}:
        raise ValueError("Original reward gates changed")
    if manifest["visual_reward"] != {"scheme": SCHEME, "real_coefficient": 2, "masked_coefficient": -1, "masked_rgb": [128, 128, 128], "temperature_bounds": [.25, 8], "minimum_confidence": .75, "retired_minimum_accepted": 32, "retired_minimum_correct": 24}:
        raise ValueError("Preregistered contrast/calibration rule differs")
    if set(manifest["configurations"]) != {f"pool-{arm}{seed}" for seed in SEEDS for arm in ["s", "t"]}:
        raise ValueError("Six pools required")
    for seed in SEEDS:
        for arm, sampler in [("s", (.7, .95)), ("t", (1., 1.))]:
            cfg = manifest["configurations"][f"pool-{arm}{seed}"]
            if cfg["seed"] != seed or cfg["method"] != "Frozen SC-8" or cfg["evaluation_labels_allowed"] or (cfg["temperature"], cfg["top_p"]) != sampler or cfg["probability_check_bf16_atol"] != .1:
                raise ValueError("Frozen pool method differs")


def checked_signal(signal, manifest, digest):
    if signal.get("reward_scheme") != SCHEME or signal.get("labels_read") is not False or signal.get("weights_unchanged") is not True or signal.get("sealed") is not True or signal.get("manifest_sha256") != digest or signal.get("teacher_revision") != REVISION or manifest["teacher"]["revision"] != REVISION:
        raise ValueError("Calibrated frozen signal provenance differs")
    calibration = signal["calibration"]
    if not calibration.get("sealed") or calibration.get("new_labels_used") is not False or calibration.get("admitted") is not True or calibration.get("retired_groups") != 64:
        raise ValueError("Calibration did not freeze before fresh readouts")
    if manifest.get("p23_version") == 1 and calibration.get("manifest_sha256") != digest:
        raise ValueError("Retired calibration belongs to another fresh scope")
    if manifest.get("p24_version") == 1:
        if signal.get("source_binding") != manifest["reward_binding"] or signal["source_binding"]["primary_teacher"] != "v" or signal["source_binding"]["calibration"] != calibration:
            raise ValueError("Immutable P24 calibrated-source binding differs")
    if [r["id"] for r in signal["stream"]] != [e["id"] for e in manifest["stream"]]:
        raise ValueError("Calibrated signal scope/order differs")
    for row in signal["stream"]:
        target, confidence, agree = decision(row["orders"], calibration)
        if row["option_keys"] != ["A", "B"] or row["target"] != target or row["confidence"] != confidence or row["same_canonical_choice"] != agree:
            raise ValueError("Target differs from the frozen calibrated decision")
    return {r["id"]: r["target"] for r in signal["stream"]}


def main():
    folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
    manifest = json.loads((folder / "manifest.json").read_text()); validate(manifest)
    mode = os.environ["P2_MODE"]
    if mode == "p23_resources":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("CPU resource stage has CUDA")
        spec = manifest["teacher"]; path = Path(spec["model_path"]); receipt = json.loads((path / "download_receipt.json").read_text())
        if receipt["repo"] != REPO or receipt["revision"] != REVISION or not receipt["publisher_size_check"]: raise ValueError("Warm resource receipt differs")
        for item in receipt["files"]:
            file = path / item["name"]
            if not file.resolve().is_relative_to(path.resolve()) or file.stat().st_size != item["bytes"]: raise ValueError("Warm resource file differs")
        atomic_json(folder / "resource_check.json", {"sealed": True, "publisher_size_check": True, "gpu_process_seconds": 0})
    elif mode == "p23_control":
        from p10_verifier import main as verify
        verify("c")
    elif mode == "p23_pool":
        import torch
        from p16_pool import main as collect
        from p20_pool import CachedCheckEngine
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        collect(CachedCheckEngine)
    else: raise ValueError("Unknown P23 stage")


if __name__ == "__main__": main()
