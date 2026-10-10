"""Isolate the interrupted medical route with supervised CPU resource preparation."""
import json
import os
from pathlib import Path
from prepare_p10 import prepare
from state import atomic_json


def main(prefix="p12", seeds=(63, 64, 65)):
    source = Path(os.environ["P10_SOURCE_ROOT"])
    previous = "p12-20261010T064959" if prefix == "p13" else "p11-20261010T014142"
    old = source / "workspaces" / previous / "outputs" / previous
    audit = json.loads((old / ("completion_audit.json" if prefix == "p13" else "interruption_audit.json")).read_text())
    final = json.loads((old / "FINAL.json").read_text()) if prefix == "p13" else audit
    expected_status = "failed" if prefix == "p13" else "interrupted_before_reward_qualification"
    if prefix == "p13" and (audit.get("failure_type") != "resource_download_tls_eof" or audit.get("gpu_work_started") is not False or audit.get("readouts_generated") != 0):
        raise ValueError("Transport recovery requires the verified pre-GPU TLS failure")
    if prefix not in {"p12", "p13"} or final["status"] != expected_status or not audit["owned_processes_ended"] or final["gpu_process_seconds_used"] != 0 or final["cumulative_gpu_process_seconds"] != 67363.9203985543:
        raise ValueError("Interrupted round or continuous cost differs from the recovery admission")
    prepare(prefix, seeds, final["cumulative_gpu_process_seconds"],
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
    environment = {"P2_MODE": "download_transport" if prefix == "p13" else "download", "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "HF_HUB_DISABLE_XET": "1",
                   "HTTP_PROXY": os.environ["P10_PROXY"], "HTTPS_PROXY": os.environ["P10_PROXY"],
                   "http_proxy": os.environ["P10_PROXY"], "https_proxy": os.environ["P10_PROXY"], "NO_PROXY": "", "no_proxy": ""}
    atomic_json(folder / "resources-plan.json", {"jobs": [
        {"label": "resource", "gpu": False, "environment": environment, "max_seconds": 7200}]})
    atomic_json(folder / "recovery_admission.json", {"admitted": True, "training_admitted": False,
        "single_engineering_change": "bounded native curl HTTP1.1 per-file resume instead of threaded Python HTTPS" if prefix == "p13" else "supervised CPU resource stage and same-boot completion fence before any GPU stage",
        "supersedes_interrupted_round": previous, "old_deadlines_unchanged": True,
        "new_data_authorized": False, "budget": budget})


if __name__ == "__main__":
    main()
