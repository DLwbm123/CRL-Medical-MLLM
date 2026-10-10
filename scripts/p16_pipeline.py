"""One complete reward-only campaign; no conditional training or label peeking."""
import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
import p10_pipeline as stages
from p4_pipeline import gpu_seconds
from state import atomic_json
from verified_reward import admitted_seconds


def main():
    folder, root = stages.FOLDER, stages.ROOT
    if (folder / "PIPELINE_STARTED.json").exists() or (folder / "FINAL.json").exists(): raise RuntimeError("Duplicate campaign forbidden")
    for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, stages.stop)
    final = {"started_utc": datetime.now(timezone.utc).isoformat(), "code_commit": os.environ["P2_CODE_COMMIT"],
             "status": "resource_preparation", "public_delivery_complete": False, "training_performed": False}
    atomic_json(folder / "PIPELINE_STARTED.json", final)
    try:
        budget = json.loads((root / "metadata/campaign_budget.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest.get("p17_version") == 1:
            domain = json.loads((folder / "binary_domain_admission.json").read_text())
            if not domain["admitted"] or domain["groups"] != 64 or domain["class_labels_exported"] or domain["manifest_sha256"] != (folder / "manifest.sha256").read_text().strip():
                raise ValueError("Pre-generation binary-domain admission differs")
        remaining = (datetime.fromisoformat(budget["gpu_stop_utc"]) - datetime.now(timezone.utc)).total_seconds()
        if not admitted_seconds(budget["remaining_total_gpu_process_seconds_at_start"], remaining, budget["complete_scope_estimated_gpu_process_seconds"]):
            raise ValueError("Complete frozen-pool scope plus20% no longer fits")
        stages.supervise("resources", 0, os.environ["P10_GPU0_UUID"])
        spec = json.loads((folder / "manifest.json").read_text())["teacher"]
        stages.check_resource_closure(json.loads((folder / "controllers/resources-plan.json").read_text()),
            json.loads((Path(spec["model_path"]) / "download_receipt.json").read_text()), spec,
            budget["recovery_boot_id"], Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        atomic_json(folder / "resource_ready.json", {"sealed": True, "cpu_worker_ended": True, "gpu_process_seconds": 0,
            "boot_id": budget["recovery_boot_id"], "teacher_revision": spec["revision"], "code_commit": os.environ["P2_CODE_COMMIT"]})
        atomic_json(folder / "STATUS.json", {"stage": "reward_readouts"})
        stages.supervise("qualification", 1, os.environ["P10_GPU1_UUID"])
        used = gpu_seconds(); remaining = (datetime.fromisoformat(budget["gpu_stop_utc"]) - datetime.now(timezone.utc)).total_seconds()
        estimate = budget["complete_scope_estimated_gpu_process_seconds"] - 7200
        if not admitted_seconds(budget["remaining_total_gpu_process_seconds_at_start"] - used, remaining, estimate):
            raise ValueError("Whole six-pool scope plus20% cannot fit after teacher")
        atomic_json(folder / "main_budget_lock.json", {"used_seconds": used, "required_with_reserve": estimate * 1.2,
            "remaining_total_seconds": budget["remaining_total_gpu_process_seconds_at_start"] - used, "remaining_round_seconds": remaining,
            "complete_seeds": json.loads((folder / "manifest.json").read_text())["rollout_seeds"], "complete_samplers": ["s", "t"], "training_authorized": False})
        atomic_json(folder / "STATUS.json", {"stage": "complete_frozen_candidate_pools"})
        stages.supervise("main", 0, os.environ["P10_GPU0_UUID"])
        stages.invoke("p16_score", "offline-score.log")
        final["status"] = "reward_qualified" if json.loads((folder / "scores/qualification.json").read_text())["go"] else "reward_qualification_negative"
    except Exception as exc:
        final.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        budget = json.loads((root / "metadata/campaign_budget.json").read_text()); used = gpu_seconds()
        final.update(finished_utc=datetime.now(timezone.utc).isoformat(), gpu_process_seconds_used=used,
            cumulative_gpu_process_seconds=budget["prior_gpu_process_seconds"] + used,
            remaining_gpu_process_seconds=budget["total_gpu_process_seconds"] - budget["prior_gpu_process_seconds"] - used,
            budget_reset=False, whole_gpu_worker_lifetimes_counted=True, stable_positive_development_result=False)
        atomic_json(folder / "FINAL.json", final); atomic_json(folder / "STATUS.json", final)
        print(json.dumps(final), flush=True)


if __name__ == "__main__": main()
