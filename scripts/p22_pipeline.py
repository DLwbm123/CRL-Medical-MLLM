"""Reward/delivery and full-cost barriers precede even GPU acceptance."""
import json
import os
import signal
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from p10_pipeline import ROOT, FOLDER, stop, supervise, invoke
from p4_pipeline import gpu_seconds
from p22_contract import admission
from state import atomic_json


def main():
    for signum in [signal.SIGTERM, signal.SIGINT]: signal.signal(signum, stop)
    if (FOLDER / "PIPELINE_STARTED.json").exists() or (FOLDER / "FINAL.json").exists(): raise RuntimeError("Duplicate P22 execution forbidden")
    budget = json.loads((ROOT / "metadata/campaign_budget.json").read_text())
    final = {"started_utc": datetime.now(timezone.utc).isoformat(), "code_commit": os.environ["P2_CODE_COMMIT"], "public_delivery_complete": False}
    atomic_json(FOLDER / "PIPELINE_STARTED.json", final)
    try:
        prep = json.loads((FOLDER / "preparation.json").read_text())
        gate = json.loads((FOLDER / "scores/qualification.json").read_text())
        if not prep["sealed"] or not gate["go"] or gate["primary_teacher"] != "q": raise ValueError("Qualified delivered binding is missing")
        def lock(accepted=False):
            remaining = (datetime.fromisoformat(budget["gpu_stop_utc"]) - datetime.now(timezone.utc)).total_seconds()
            decision = admission(budget["prior_gpu_process_seconds"], max(0., remaining), gpu_seconds(), accepted)
            atomic_json(FOLDER / "budget_admission.json", decision)
            if decision["admitted"]: atomic_json(FOLDER / "main_budget_lock.json", decision)
            return decision["admitted"]
        if not lock(): final["status"] = "full_matrix_budget_admission_blocked"
        else:
            if not ROOT.resolve().is_relative_to(Path("/data/bmw")): raise ValueError("GPU output must remain on authorized data storage")
            mount = subprocess.check_output(["findmnt", "-no", "TARGET", "-T", str(ROOT)], text=True).strip()
            if mount == "/" or not mount: raise ValueError("Authorized data mount is unavailable")
            space = os.statvfs(ROOT)
            if space.f_bavail * space.f_frsize < 400 * 1024**3: raise ValueError("Full checkpoint scope requires400GiB available storage")
            memory = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
            if memory < 100 * 1024**2: raise ValueError("Actual full-parameter optimizer requires100GiB available CPU memory")
            probe = FOLDER / ".storage_probe"; probe.write_text("ok")
            if probe.read_text() != "ok": raise ValueError("Data mount read/write probe failed")
            probe.unlink()
            invoke("p22_test", "cpu-acceptance.log")
            atomic_json(FOLDER / "STATUS.json", {"stage": "actual_training_acceptance"})
            supervise("acceptance", 0, os.environ["P10_GPU0_UUID"])
            if not json.loads((FOLDER / "acceptance.json").read_text())["passed"]: raise ValueError("Actual training acceptance failed")
            if not lock(True): final["status"] = "full_matrix_budget_admission_blocked_after_acceptance"
            else:
                atomic_json(FOLDER / "STATUS.json", {"stage": "full_matrix"})
                supervise("main", 0, os.environ["P10_GPU0_UUID"])
                invoke("p22_score", "main-score.log")
                final.update(status="scored_awaiting_public_delivery", stable_positive_development_result=json.loads((FOLDER / "scores/public_summary.json").read_text())["stable_positive_development_result"])
    except Exception as exc:
        final.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        used = gpu_seconds()
        final.update(finished_utc=datetime.now(timezone.utc).isoformat(), gpu_process_seconds_used=used,
            cumulative_gpu_process_seconds=budget["prior_gpu_process_seconds"] + used,
            remaining_gpu_process_seconds=budget["total_gpu_process_seconds"] - budget["prior_gpu_process_seconds"] - used,
            whole_gpu_worker_lifetimes_counted=True, budget_reset=False)
        atomic_json(FOLDER / "FINAL.json", final); atomic_json(FOLDER / "STATUS.json", final)


if __name__ == "__main__": main()
