"""Isolate the interrupted medical route with supervised CPU resource preparation."""
import json
import os
from pathlib import Path
from prepare_p10 import prepare
from state import atomic_json


def main():
    source = Path(os.environ["P10_SOURCE_ROOT"])
    audit = json.loads((source / "workspaces/p11-20261010T014142/outputs/p11-20261010T014142/interruption_audit.json").read_text())
    if audit["status"] != "interrupted_before_reward_qualification" or not audit["owned_processes_ended"] or audit["gpu_process_seconds_used"] != 0 or audit["cumulative_gpu_process_seconds"] != 67363.9203985543:
        raise ValueError("Interrupted round or continuous cost differs from the recovery admission")
    prepare("p12", (63, 64, 65), audit["cumulative_gpu_process_seconds"],
            "lingshu-medical-mllm/Lingshu-7B", "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")
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
        raise ValueError("The complete recovery scope plus reserve cannot fit the continuous budget")
    atomic_json(budget_path, budget)
    environment = {"P2_MODE": "download", "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "HF_HUB_DISABLE_XET": "1",
                   "HTTP_PROXY": os.environ["P10_PROXY"], "HTTPS_PROXY": os.environ["P10_PROXY"],
                   "http_proxy": os.environ["P10_PROXY"], "https_proxy": os.environ["P10_PROXY"], "NO_PROXY": "", "no_proxy": ""}
    atomic_json(folder / "resources-plan.json", {"jobs": [
        {"label": "resource", "gpu": False, "environment": environment, "max_seconds": 7200}]})
    atomic_json(folder / "recovery_admission.json", {"admitted": True, "training_admitted": False,
        "single_engineering_change": "supervised CPU resource stage and same-boot completion fence before any GPU stage",
        "supersedes_interrupted_round": "p11-20261010T014142", "old_deadlines_unchanged": True,
        "new_data_authorized": False, "budget": budget})


if __name__ == "__main__":
    main()
