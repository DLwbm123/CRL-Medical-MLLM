"""Freeze fresh binary training inputs and a complete calibrated visual reward comparison."""
import copy
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from prepare_p16 import binary_domain_metadata, heldout_image_references, select_rows, thumbnail, validate_binary_domain
from prepare_campaign import write_manifest
from p23_reward import SEEDS, TOTAL, REPO, REVISION, SCHEME, validate
from state import atomic_json


def main():
    root = Path(os.environ["P0_ROOT"]); source = Path(os.environ["P10_SOURCE_ROOT"])
    name = "p21-20261010T225707"; previous = source / "workspaces" / name / "outputs" / name
    end = json.loads((previous / "FINAL.json").read_text()); audit = json.loads((root / "metadata/p21_completion_audit.json").read_text())
    delivery = json.loads((root / "metadata/p21_delivery_receipt.json").read_text())
    if end["status"] != "reward_qualification_negative" or not audit["owned_processes_ended"] or not audit["all_eight_gpu_workers_exit_zero"] or not delivery["public_delivery_complete"] or not delivery["remote_sha_verified"] or not delivery["anonymous_access_verified"]:
        raise ValueError("P21 must fully close and be publicly delivered")
    receipt = json.loads((source / "metadata/slake_download.json").read_text())
    if receipt["status"] != "complete" or receipt["revision"] != "a9083ce6c34ac3ffb17671a605962924d8a8f9e9": raise ValueError("Pinned SLAKE source differs")
    raw = json.loads((source / "data/SLAKE/raw/train.json").read_text()); metadata = binary_domain_metadata(raw); del raw
    excluded = set.union(*(heldout_image_references(source / "views/adaptation_inputs/SLAKE" / (split + ".jsonl")) for split in ["validation", "test"]))
    heldout = len(excluded); prints = set(); prior_images = set()
    for name in ["p15-20261010T095303", "p16-20261010T105716", "p17-20261010T160004", "p21-20261010T225707"]:
        old_folder = source / "workspaces" / name / "outputs" / name
        old = json.loads((old_folder / "manifest.json").read_text())
        for entry in old["stream"] + old["probe"]:
            if not name.startswith("p15-"): prior_images.update(entry["images"])
            row = json.loads((old_folder / entry["input"]).read_text()); prints.update(thumbnail(Path(p)) for p in row["image_paths"])
    excluded.update(prior_images); selected, eligible = select_rows(metadata, excluded, seed=20261011)
    image_root = (source / "data/SLAKE/extracted/imgs").resolve(); clean = []; skipped = 0
    for row in selected:
        path = (image_root / row["img_name"]).resolve()
        if not path.is_relative_to(image_root): raise ValueError("Image path escapes source")
        value = thumbnail(path)
        if value in prints: skipped += 1; continue
        prints.add(value); clean.append((row, path))
        if len(clean) == 64: break
    if len(clean) != 64: raise ValueError("Fewer than64 disjoint new eligible groups; no reduced scope")
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]; folder.mkdir(parents=True, exist_ok=False)
    for part in ["inputs", "reference", "scores", "runs", "controllers"]: (folder / part).mkdir()
    for key in ["calibration", "v", "c"]: (folder / "reference" / key).mkdir(); (folder / "scores" / key).mkdir()
    entries = []
    for i, (row, path) in enumerate(clean):
        sample = {"id": f"SLAKE/train/{row['qid']}", "source_id": row["qid"], "dataset": "SLAKE", "split": "train", "question": row["question"],
            "options": {"A": "yes", "B": "no"}, "images": [row["img_name"]], "image_paths": [str(path)], "language": "en"}
        rel = f"inputs/i{i:03d}.json"; atomic_json(folder / rel, sample)
        entries.append({"id": sample["id"], "split": "train", "group": f"g{i:03d}", "input": rel, "images": sample["images"]})
    old = json.loads((previous / "manifest.json").read_text()); manifest = copy.deepcopy(old)
    configs = {f"pool-{arm}{seed}": {**old["configurations"][f"pool-{arm}90"], "seed": seed} for seed in SEEDS for arm in ["s", "t"]}
    teachers = {key: {**old["teachers"]["m"]} for key in ["v", "c"]}
    calibration_entries = []
    for i, entry in enumerate(old["stream"] + old["probe"]):
        rel = f"inputs/cal{i:03d}.json"
        atomic_json(folder / rel, json.loads((previous / entry["input"]).read_text()))
        calibration_entries.append({**entry, "input": rel})
    manifest.pop("p21_version", None)
    manifest.update(p23_version=1, stream=entries[:32], probe=entries[32:], rollout_seeds=SEEDS, configurations=configs,
        teachers=teachers, teacher=teachers["v"], primary_teacher="v", calibration_entries=calibration_entries, calibration_source="p21-20261010T225707",
        visual_reward={"scheme": SCHEME, "real_coefficient": 2, "masked_coefficient": -1, "masked_rgb": [128, 128, 128], "temperature_bounds": [.25, 8], "minimum_confidence": .75, "retired_minimum_accepted": 32, "retired_minimum_correct": 24}, input_scope_reused_from=None, selected_n=64,
        selection_seed=20261011, selection_receives_labels=True, annotation_domain_only=True, correctness_used_for_selection=False,
        annotations_reread_for_preparation=True, teacher_readouts_reused=False, training_authorized_in_this_round=False)
    validate(manifest); digest = write_manifest(folder / "manifest.json", manifest)
    domain = validate_binary_domain(source / "views/evaluation_labels/SLAKE/train.jsonl", {e["id"] for e in entries})
    atomic_json(folder / "binary_domain_admission.json", {**domain, "manifest_sha256": digest, "previous_image_references_excluded": len(prior_images)})
    for name, cfg in configs.items(): atomic_json(root / "configs" / ("p23_" + name.split("-")[1] + ".json"), cfg)
    now = datetime.now(timezone.utc); proxy = 23105.95653328098 + 3 * 7200; caps = 6 * 18000 + 3 * 7200
    estimate = max(proxy, caps)
    budget = {"total_gpu_process_seconds": TOTAL, "prior_gpu_process_seconds": end["cumulative_gpu_process_seconds"],
        "remaining_total_gpu_process_seconds_at_start": (TOTAL - end["cumulative_gpu_process_seconds"]), "budget_reset": False,
        "gpu_stop_utc": (now + timedelta(hours=46)).isoformat(), "hard_deadline_utc": (now + timedelta(hours=48)).isoformat(),
        "recovery_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "historical_complete_cost_proxy_seconds": proxy,
        "whole_scope_stop_caps_seconds": caps, "complete_scope_estimated_gpu_process_seconds": estimate,
        "complete_scope_required_with_reserve": estimate * 1.2, "remaining_pool_scope_estimated_seconds": max(23105.95653328098, 6 * 18000),
        "reserve_fraction": .2, "qualified_training_pipeline_enabled": False, "authorized_physical_gpus": [0, 1], "new_training_cost_measured": False}
    if estimate * 1.2 > min((TOTAL - end["cumulative_gpu_process_seconds"]), 46 * 3600): raise ValueError("Whole paired source scope plus20% cannot fit")
    atomic_json(root / "metadata/campaign_budget.json", budget)
    atomic_json(folder / "authorization.json", json.loads((root / "metadata/new_authorization.json").read_text()))
    atomic_json(folder / "resources-plan.json", {"jobs": [{"label": "resource", "gpu": False, "environment": {"P2_MODE": "p23_resources"}, "max_seconds": 1800}]})
    atomic_json(folder / "calibration-plan.json", {"minimum_free_mib": 24000, "jobs": [
        {"label": "retired-calibration", "environment": {"P2_MODE": "p23_calibration_read"}, "max_seconds": 7200}]})
    atomic_json(folder / "qualification-plan.json", {"minimum_free_mib": 24000, "jobs": [
        {"label": "teacher-v", "environment": {"P2_MODE": "p23_visual"}, "max_seconds": 7200},
        {"label": "teacher-c", "environment": {"P2_MODE": "p23_control"}, "max_seconds": 7200}]})
    atomic_json(folder / "main-plan.json", {"minimum_free_mib": 24000, "jobs": [
        {"label": name, "environment": {"P2_MODE": "p23_pool", "P2_RUN": name}, "max_seconds": 18000, "expected_cursor": 64} for name in configs]})
    admission = {"admitted": True, "training_admitted": False, "followup_full_training_required_with_reserve": 233280., "combined_scope_required_with_reserve": 388800., "combined_scope_budget_admitted": TOTAL - end["cumulative_gpu_process_seconds"] >= 388800., "new_training_groups": 64, "calibration_groups": 32, "verification_groups": 32,
        "eligible_image_groups": eligible, "heldout_image_refs_excluded": heldout, "previous_image_refs_excluded": len(prior_images),
        "thumbnail_duplicates_skipped": skipped, "annotation_domain_only": True, "correctness_used_for_selection": False,
        "patient_disjointness_verified": False, "pretraining_overlap_excluded": False,
        "modalities": dict(Counter(r["modality"] for r, _ in clean)), "conditional_answer_measurements": 1024, "control_teacher_readouts": 128, "retired_calibration_groups": 64,
        "complete_frozen_pools": 6, "complete_candidates": 3072, "primary_teacher": "v", "budget": budget}
    atomic_json(folder / "budget_admission.json", admission); atomic_json(folder / "preparation.json", admission)
    print(json.dumps(admission), flush=True)


if __name__ == "__main__": main()
