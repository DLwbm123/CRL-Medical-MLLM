"""One bounded complete numerical matrix, followed by label-free aggregation."""
import json
import math
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
import p10_pipeline as stages
from p4_pipeline import gpu_seconds
from p18_numeric import validate_manifest
from state import atomic_json


def aggregate(folder):
    manifest = json.loads((folder / "manifest.json").read_text()); validate_manifest(manifest)
    ledger = json.loads((folder / "controllers/main-plan.json").read_text())
    if ledger["status"] != "all_planned_jobs_completed" or len(ledger["jobs"]) != 6 or any(j["return_code"] != 0 for j in ledger["jobs"]):
        raise ValueError("Incomplete numerical matrix")
    if not json.loads((folder / "owned_cleanup_before_score.json").read_text())["owned_main_processes_ended"]:
        raise ValueError("Numerical workers have not closed")
    rows = []
    source = json.loads((folder / "PIPELINE_STARTED.json").read_text())["code_commit"]
    for name, cfg in manifest["configurations"].items():
        result = json.loads((folder / "runs" / name / "result.json").read_text())
        if result["status"] != "completed" or not result["sealed"] or result["cursor"] != 64 or result["configuration"] != cfg or result["code_commit"] != source or result["labels_read"] or not result["weights_unchanged"] or result["optimizer_updates"] != 0 or result["manifest_sha256"] != (folder / "manifest.sha256").read_text().strip():
            raise ValueError("Closed numerical result differs from frozen scope")
        if len(result["measurements"]) != 64: raise ValueError("Incomplete numerical measurements")
        for index, row in enumerate(result["measurements"]):
            error = row["max_logp_difference"]
            if row["section"] != ("calibration" if index < 32 else "verification") or row["anonymous_index"] != index % 32 or row["seed"] != cfg["seed"] or row["precision"] != cfg["model_dtype"] or not 1 <= row["tokens"] <= 4 or not math.isfinite(error) or error < 0 or row["within_tolerance"] != (error <= .1):
                raise ValueError("Numerical case coverage or unchanged threshold differs")
        rows.extend(result["measurements"])
    fp32 = [r for r in rows if r["precision"] == "float32"]
    summary = {"complete_workers": 6, "measurements": len(rows), "fp32_checks_passed": sum(r["within_tolerance"] for r in fp32),
        "fp32_all_checks_passed": all(r["within_tolerance"] for r in fp32), "absolute_tolerance": .1,
        "numerical_diagnostic_only": True, "reward_qualification": None, "training_performed": False,
        "stable_positive_development_result": False}
    atomic_json(folder / "scores/numerical_summary.json", summary)
    return summary


def main():
    folder, root = stages.FOLDER, stages.ROOT
    if (folder / "PIPELINE_STARTED.json").exists() or (folder / "FINAL.json").exists(): raise RuntimeError("Duplicate diagnostic forbidden")
    for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, stages.stop)
    final = {"started_utc": datetime.now(timezone.utc).isoformat(), "code_commit": os.environ["P2_CODE_COMMIT"],
             "status": "complete_numerical_matrix", "public_delivery_complete": False, "training_performed": False}
    atomic_json(folder / "PIPELINE_STARTED.json", final)
    try:
        budget = json.loads((root / "metadata/campaign_budget.json").read_text())
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != budget["recovery_boot_id"]:
            raise ValueError("Host boot changed after preparation")
        remaining = (datetime.fromisoformat(budget["gpu_stop_utc"]) - datetime.now(timezone.utc)).total_seconds()
        if min(remaining, budget["remaining_total_gpu_process_seconds_at_start"]) < budget["complete_scope_required_with_reserve"]:
            raise ValueError("Whole bounded matrix and reserve do not fit")
        atomic_json(folder / "STATUS.json", {"stage": "complete_numerical_matrix"})
        stages.supervise("main", 0, os.environ["P10_GPU0_UUID"])
        summary = aggregate(folder)
        final["status"] = "numerical_checks_passed" if summary["fp32_all_checks_passed"] else "numerical_checks_negative"
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


if __name__ == "__main__": main()
