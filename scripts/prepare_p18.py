"""Freeze a complete read-only precision diagnostic on the unchanged P17 inputs."""
import copy
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from prepare_campaign import write_manifest
from p18_numeric import validate_manifest
from state import atomic_json


def main():
    root = Path(os.environ["P0_ROOT"]); source = Path(os.environ["P10_SOURCE_ROOT"])
    cache_round = os.environ["P2_MODE"] == "p19_prepare"
    previous_name = "p18-20261010T165940" if cache_round else "p17-20261010T160004"
    previous = source / "workspaces" / previous_name / "outputs" / previous_name
    end = json.loads((previous / "FINAL.json").read_text()); delivery = json.loads((previous / "public_delivery_receipt.json").read_text())
    audit = json.loads((previous / "completion_audit.json").read_text())
    documented = end["status"] == "numerical_checks_negative" and audit.get("all384_measurements_sealed") if cache_round else end["status"] == "failed" and json.loads((previous / "engineering_failure_audit.json").read_text())["failure_type"] == "probability_check_failure"
    if not documented or not audit["owned_processes_ended"] or not delivery["remote_sha_verified"] or not delivery["anonymous_access_verified"]:
        raise ValueError("Prior failure or negative result must close and be publicly delivered")
    old = json.loads((previous / "manifest.json").read_text())
    domain = json.loads((previous / "binary_domain_admission.json").read_text())
    if not domain["admitted"] or domain["groups"] != 64 or domain["class_labels_exported"] or domain["manifest_sha256"] != (previous / "manifest.sha256").read_text().strip():
        raise ValueError("Prior fixed binary task admission differs")
    prefix = "p19" if cache_round else "p18"; seeds = [84, 85, 86] if cache_round else [81, 82, 83]
    cfg = old["configurations"]["numeric-b81" if cache_round else "pool-t78"]
    configs = {f"numeric-k{seed}": {**cfg, "seed": seed} for seed in seeds} if cache_round else {f"numeric-{arm}{seed}": {**cfg, "seed": seed, "model_dtype": dtype}
        for seed in seeds for arm, dtype in [("b", "bfloat16"), ("f", "float32")]}
    manifest = {prefix + "_version": 1, "stream": copy.deepcopy(old["stream"]), "probe": copy.deepcopy(old["probe"]),
        "rollout_seeds": seeds, "configurations": configs, "source_revision": old["source_revision"],
        "input_scope_reused_from": "p17", "prior_manifest_sha256": domain["manifest_sha256"], "labels_read": False,
        "training_authorized_in_this_round": False, "diagnostic_only": True}
    if cache_round:
        from p19_cache import validate_manifest as validate_cache_manifest
        validate_cache_manifest(manifest)
    else: validate_manifest(manifest)
    # Reserve all stop caps above the old measured proxy; no new speed assumption.
    measured_proxy = len(configs) * 4 * 4590.984604918864; caps = len(configs) * 21600
    gpu_hours, hard_hours = (22, 24) if cache_round else (46, 48)
    required = max(measured_proxy, caps) * 1.2
    now = datetime.now(timezone.utc)
    budget = {"total_gpu_process_seconds": 259200., "prior_gpu_process_seconds": end["cumulative_gpu_process_seconds"],
        "remaining_total_gpu_process_seconds_at_start": end["remaining_gpu_process_seconds"], "budget_reset": False,
        "gpu_stop_utc": (now + timedelta(hours=gpu_hours)).isoformat(), "hard_deadline_utc": (now + timedelta(hours=hard_hours)).isoformat(),
        "recovery_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "measured_old_full_worker_cost_proxy_seconds": measured_proxy, "whole_scope_stop_caps_seconds": caps,
        "complete_scope_estimated_gpu_process_seconds": max(measured_proxy, caps), "complete_scope_required_with_reserve": required,
        "reserve_fraction": .2, "diagnostic_complete_scope_cost_measured": False, "training_admitted": False,
        "authorized_physical_gpus": [0, 1]}
    if required > min(budget["remaining_total_gpu_process_seconds_at_start"], gpu_hours * 3600):
        raise ValueError("Complete bounded diagnostic and20% reserve do not fit")
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]; folder.mkdir(parents=True, exist_ok=False)
    for part in ["inputs", "runs", "scores", "controllers"]: (folder / part).mkdir()
    for entry in manifest["stream"] + manifest["probe"]:
        src = (previous / entry["input"]).resolve(); dst = folder / entry["input"]
        if not src.is_relative_to((previous / "inputs").resolve()) or not dst.resolve().is_relative_to((folder / "inputs").resolve()):
            raise ValueError("Frozen input path boundary differs")
        dst.write_bytes(src.read_bytes())
    digest = write_manifest(folder / "manifest.json", manifest)
    for name, value in configs.items(): atomic_json(root / "configs" / (prefix + "_" + name.split("-", 1)[1] + ".json"), value)
    atomic_json(folder / "binary_domain_admission.json", {**domain, "prior_manifest_sha256": domain["manifest_sha256"], "manifest_sha256": digest, "reused_without_reading_annotations": True})
    atomic_json(root / "metadata/campaign_budget.json", budget)
    atomic_json(folder / "main-plan.json", {"minimum_free_mib": 48000, "jobs": [
        {"label": name, "environment": {"P2_MODE": "p19_cache" if cache_round else "p18_numeric", "P2_RUN": name}, "max_seconds": 21600, "expected_cursor": 64}
        for name in configs]})
    admission = {"admitted": True, "diagnostic_only": True, "training_admitted": False, "unique_image_groups": 64,
        "input_scope_reused": True, "new_data_access": False, "accuracy_scoring": False, "budget": budget}
    atomic_json(folder / "budget_admission.json", admission); print(json.dumps(admission))


if __name__ == "__main__": main()
