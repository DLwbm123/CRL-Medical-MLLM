"""Reuse the admitted binary task after complete same-token numerical evidence."""
import copy
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from p20_pool import validate_manifest
from prepare_campaign import write_manifest
from state import atomic_json


def main():
    root = Path(os.environ["P0_ROOT"]); source = Path(os.environ["P10_SOURCE_ROOT"])
    previous = source / "workspaces/p19-20261010T175812/outputs/p19-20261010T175812"
    original = source / "workspaces/p17-20261010T160004/outputs/p17-20261010T160004"
    end = json.loads((previous / "FINAL.json").read_text()); audit = json.loads((previous / "completion_audit.json").read_text())
    delivery = json.loads((previous / "public_delivery_receipt.json").read_text()); numeric = json.loads((previous / "scores/numerical_summary.json").read_text())
    if end["status"] != "numerical_checks_passed" or not audit["owned_processes_ended"] or not audit["all384_measurements_sealed"] or not delivery["remote_sha_verified"] or not delivery["anonymous_access_verified"] or not numeric["diagnostic_passed"] or numeric["cached_checks_passed"] != 192:
        raise ValueError("Complete cached-path evidence and public delivery required")
    old = json.loads((original / "manifest.json").read_text()); reused = json.loads((previous / "manifest.json").read_text())
    domain = json.loads((previous / "binary_domain_admission.json").read_text())
    if not domain["admitted"] or domain["groups"] != 64 or domain["class_labels_exported"] or domain["manifest_sha256"] != (previous / "manifest.sha256").read_text().strip() or (old["stream"], old["probe"]) != (reused["stream"], reused["probe"]):
        raise ValueError("Original binary task scope differs")
    manifest = copy.deepcopy(old); configs = {}
    for seed in [87, 88, 89]:
        for arm in ["s", "t"]:
            configs[f"pool-{arm}{seed}"] = {**old["configurations"][f"pool-{arm}78"], "seed": seed}
    manifest.update(p20_version=1, rollout_seeds=[87, 88, 89], configurations=configs,
        frozen_pool_probability_recompute_path="incremental_cache", input_scope_reused_from="p17",
        annotations_reread_for_preparation=False, teacher_readouts_reused=False, training_authorized_in_this_round=False)
    validate_manifest(manifest)
    if manifest["teacher"] != old["teacher"]: raise ValueError("Teacher changed")
    now = datetime.now(timezone.utc); proxy = 6 * 4 * 4590.984604918864 + 7200; caps = 6 * 21600 + 7200
    estimate = max(proxy, caps)
    budget = {"total_gpu_process_seconds": 259200., "prior_gpu_process_seconds": end["cumulative_gpu_process_seconds"],
        "remaining_total_gpu_process_seconds_at_start": end["remaining_gpu_process_seconds"], "budget_reset": False,
        "gpu_stop_utc": (now + timedelta(hours=46)).isoformat(), "hard_deadline_utc": (now + timedelta(hours=48)).isoformat(),
        "recovery_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "measured_old_full_worker_cost_proxy_seconds": proxy, "whole_scope_stop_caps_seconds": caps,
        "complete_scope_estimated_gpu_process_seconds": estimate, "complete_scope_required_with_reserve": estimate * 1.2,
        "reserve_fraction": .2, "qualified_training_pipeline_enabled": False, "authorized_physical_gpus": [0, 1],
        "new_training_cost_measured": False}
    if estimate * 1.2 > min(budget["remaining_total_gpu_process_seconds_at_start"], 46 * 3600): raise ValueError("Complete reward scope plus20% cannot fit")
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]; folder.mkdir(parents=True, exist_ok=False)
    for part in ["inputs", "reference", "scores", "runs", "controllers"]: (folder / part).mkdir()
    for entry in manifest["stream"] + manifest["probe"]:
        src = (previous / entry["input"]).resolve(); dst = folder / entry["input"]
        if not src.is_relative_to((previous / "inputs").resolve()) or not dst.resolve().is_relative_to((folder / "inputs").resolve()): raise ValueError("Input boundary changed")
        dst.write_bytes(src.read_bytes())
    digest = write_manifest(folder / "manifest.json", manifest)
    for name, cfg in configs.items(): atomic_json(root / "configs" / ("p20_" + name.split("-")[1] + ".json"), cfg)
    atomic_json(folder / "binary_domain_admission.json", {**domain, "prior_manifest_sha256": domain["manifest_sha256"], "manifest_sha256": digest, "reused_without_reading_annotations": True})
    atomic_json(root / "metadata/campaign_budget.json", budget)
    atomic_json(folder / "resources-plan.json", {"jobs": [{"label": "resource", "gpu": False, "environment": {"P2_MODE": "download_transport"}, "max_seconds": 7200}]})
    atomic_json(folder / "qualification-plan.json", {"minimum_free_mib": 24000, "jobs": [{"label": "frozen-verifier", "environment": {"P2_MODE": "verify"}, "max_seconds": 7200}]})
    atomic_json(folder / "main-plan.json", {"minimum_free_mib": 24000, "jobs": [
        {"label": name, "environment": {"P2_MODE": "p20_pool", "P2_RUN": name}, "max_seconds": 21600, "expected_cursor": 64} for name in configs]})
    admission = {"admitted": True, "reward_validation_only": True, "training_admitted": False, "unique_image_groups": 64,
        "input_scope_reused_from": "p17", "new_data_access": False, "annotations_reread_for_preparation": False,
        "new_teacher_readouts": 128, "complete_frozen_pools": 6, "complete_candidates": 3072,
        "paired_probability_check": "incremental_cache; original0.1; unchanged production/training path", "budget": budget}
    atomic_json(folder / "budget_admission.json", admission); atomic_json(folder / "preparation.json", admission)
    print(json.dumps(admission))


if __name__ == "__main__": main()
