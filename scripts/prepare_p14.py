"""Shared gated preparation of evidence-prompt and teacher-interface interventions."""
import json
import os
from pathlib import Path
from prepare_p10 import prepare
from state import atomic_json


def main(prefix="p14", seeds=(69, 70, 71)):
    source = Path(os.environ["P10_SOURCE_ROOT"])
    previous = "p14-20261010T085425" if prefix == "p15" else "p13-20261010T074553"
    old = source / "workspaces" / previous / "outputs" / previous
    final = json.loads((old / "FINAL.json").read_text())
    audit = json.loads((old / "completion_audit.json").read_text())
    delivery = json.loads((old / "public_delivery_receipt.json").read_text())
    if final["status"] != "reward_qualification_negative" or not audit["owned_processes_ended"] or audit["actor_updates"] != 0 or not delivery["remote_sha_verified"] or not delivery["anonymous_access_verified"]:
        raise ValueError("The preceding reward rejection must close and be publicly delivered")
    prepare(prefix, seeds, final["cumulative_gpu_process_seconds"],
            "lingshu-medical-mllm/Lingshu-7B", "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9", evidence_first=True, json_suffix=prefix == "p15")
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    budget_path = root / "metadata/campaign_budget.json"
    budget = json.loads(budget_path.read_text())
    estimate = budget["qualification_gpu_limit_seconds"] + budget["main_estimated_gpu_process_seconds"] + budget["acceptance_gpu_limit_seconds"]
    budget.update(resource_preparation_cpu_limit_seconds=7200,
                  recovery_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  complete_scope_estimated_gpu_process_seconds=estimate,
                  complete_scope_required_with_reserve=estimate * (1 + budget["reserve_fraction"]))
    if budget["remaining_total_gpu_process_seconds_at_start"] < budget["complete_scope_required_with_reserve"]:
        raise ValueError("The full prompt-intervention scope plus reserve cannot fit")
    atomic_json(budget_path, budget)
    atomic_json(folder / "resources-plan.json", {"jobs": [{"label": "resource", "gpu": False,
        "environment": {"P2_MODE": "download_transport"}, "max_seconds": 7200}]})
    atomic_json(folder / "evidence_admission.json", {"admitted": True, "training_admitted": False,
        "single_scientific_change": "strict-schema teacher JSON suffix adapter" if prefix == "p15" else "explicit visible-evidence reasoning before the unchanged strict JSON judgment",
        "preceding_closed_round": previous, "old_deadlines_unchanged": True,
        "new_data_authorized": False, "qualification_thresholds_unchanged": True, "budget": budget})


if __name__ == "__main__":
    main()
