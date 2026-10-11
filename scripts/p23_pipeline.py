"""Retired calibration closes first; complete fresh paired audit then seals."""
import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
import p10_pipeline as stages
from p4_pipeline import gpu_seconds
from state import atomic_json
from verified_reward import admitted_seconds
from p23_reward import TOTAL, validate


def main():
    folder, root = stages.FOLDER, stages.ROOT
    if (folder / "PIPELINE_STARTED.json").exists() or (folder / "FINAL.json").exists(): raise RuntimeError("Duplicate P23 forbidden")
    for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, stages.stop)
    final = {"started_utc": datetime.now(timezone.utc).isoformat(), "code_commit": os.environ["P2_CODE_COMMIT"],
             "status": "resource_preparation", "public_delivery_complete": False, "training_performed": False}
    atomic_json(folder / "PIPELINE_STARTED.json", final)
    try:
        budget = json.loads((root / "metadata/campaign_budget.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text()); validate(manifest)
        domain = json.loads((folder / "binary_domain_admission.json").read_text())
        if budget["total_gpu_process_seconds"] != TOTAL or not domain["admitted"] or domain["groups"] != 64 or domain["class_labels_exported"] or domain["manifest_sha256"] != (folder / "manifest.sha256").read_text().strip(): raise ValueError("Full authorization/data admission differs")
        def room(estimate):
            seconds = (datetime.fromisoformat(budget["gpu_stop_utc"]) - datetime.now(timezone.utc)).total_seconds()
            if not admitted_seconds(TOTAL - budget["prior_gpu_process_seconds"] - gpu_seconds(), seconds, estimate): raise ValueError("Complete remaining scope plus20% cannot fit")
        room(budget["complete_scope_estimated_gpu_process_seconds"])
        stages.supervise("resources", 0, os.environ["P10_GPU0_UUID"])
        ledger = json.loads((folder / "controllers/resources-plan.json").read_text()); spec = manifest["teacher"]
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        stages.check_resource_closure(ledger, json.loads((Path(spec["model_path"]) / "download_receipt.json").read_text()), spec, budget["recovery_boot_id"], boot)
        atomic_json(folder / "resource_ready.json", {"sealed": True, "cpu_worker_ended": True, "boot_id": boot, "gpu_process_seconds": 0})
        atomic_json(folder / "STATUS.json", {"stage": "retired_calibration_readouts"})
        stages.supervise("calibration", 1, os.environ["P10_GPU1_UUID"])
        atomic_json(folder / "calibration_owned_ended.json", json.loads((folder / "owned_cleanup_before_score.json").read_text()))
        stages.invoke("p23_calibrate", "retired-calibration-fit.log")
        calibration = json.loads((folder / "calibration.json").read_text())
        if not calibration["admitted"]:
            final["status"] = "retired_calibration_negative"
        else:
            room(2 * 7200 + 6 * 18000)
            atomic_json(folder / "STATUS.json", {"stage": "fresh_paired_teacher_readouts"})
            stages.supervise("qualification", 1, os.environ["P10_GPU1_UUID"])
            room(budget["remaining_pool_scope_estimated_seconds"])
            atomic_json(folder / "main_budget_lock.json", {"used_seconds": gpu_seconds(), "required_with_reserve": budget["remaining_pool_scope_estimated_seconds"] * 1.2,
                "complete_seeds": [96, 97, 98], "complete_samplers": ["s", "t"], "training_authorized": False})
            atomic_json(folder / "STATUS.json", {"stage": "complete_shared_candidate_pools"})
            stages.supervise("main", 0, os.environ["P10_GPU0_UUID"])
            stages.invoke("p23_score", "offline-score.log")
            final["status"] = "reward_qualified" if json.loads((folder / "scores/qualification.json").read_text())["go"] else "reward_qualification_negative"
    except Exception as exc:
        final.update(status="failed", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        budget = json.loads((root / "metadata/campaign_budget.json").read_text()); used = gpu_seconds()
        final.update(finished_utc=datetime.now(timezone.utc).isoformat(), gpu_process_seconds_used=used,
            cumulative_gpu_process_seconds=budget["prior_gpu_process_seconds"] + used, remaining_gpu_process_seconds=TOTAL - budget["prior_gpu_process_seconds"] - used,
            total_gpu_process_seconds=TOTAL, whole_gpu_worker_lifetimes_counted=True, budget_reset=False, stable_positive_development_result=False)
        atomic_json(folder / "FINAL.json", final); atomic_json(folder / "STATUS.json", final)
        print(json.dumps(final), flush=True)


if __name__ == "__main__": main()
