"""Bounded qualification, conditional admission, complete matrix, sealed scoring."""
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from p4_pipeline import gpu_seconds, verify_ended
from state import atomic_json
from verified_reward import admitted_seconds

ROOT = Path(os.environ["P0_ROOT"])
FOLDER = ROOT / "outputs" / os.environ["P2_CAMPAIGN"]
CHILD = None
STOP = False


def stop(*_):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def invoke(mode, log_name, **values):
    global CHILD
    if STOP:
        raise RuntimeError("Owned pipeline received a stop request")
    env = dict(os.environ, P2_MODE=mode, CUDA_VISIBLE_DEVICES="", **values)
    if mode == "download":
        env.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0", HF_HUB_DISABLE_XET="1",
                   HTTP_PROXY=os.environ["P10_PROXY"], HTTPS_PROXY=os.environ["P10_PROXY"],
                   http_proxy=os.environ["P10_PROXY"], https_proxy=os.environ["P10_PROXY"], NO_PROXY="", no_proxy="")
    with (FOLDER / log_name).open("x") as log:
        CHILD = subprocess.Popen([sys.executable, "-u", os.environ["P2_ENTRY"]], env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = CHILD.wait(timeout=7200 if mode == "download" else 172800)
        except subprocess.TimeoutExpired:
            CHILD.send_signal(signal.SIGTERM)
            try: CHILD.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(CHILD.pid, signal.SIGKILL); CHILD.wait()
            raise RuntimeError("Owned stage exceeded its fixed deadline")
    CHILD = None
    if code:
        raise RuntimeError(f"Owned stage {mode} exited {code}")


def supervise(plan_name, gpu_index, gpu_uuid):
    invoke("supervise", f"supervisor-{plan_name}.log", P2_PLAN=plan_name + "-plan.json",
           P2_GPU_INDEX=str(gpu_index), P2_GPU_UUID=gpu_uuid)
    ledger = json.loads((FOLDER / "controllers" / (plan_name + "-plan.json")).read_text())
    if ledger["status"] != "all_planned_jobs_completed" or any(j["return_code"] != 0 for j in ledger["jobs"]):
        raise ValueError("Owned stage did not fully close")
    verify_ended(ledger)
    if plan_name == "qualification":
        atomic_json(FOLDER / "qualifier_owned_ended.json", json.loads((FOLDER / "owned_cleanup_before_score.json").read_text()))


def main():
    for signum in [signal.SIGTERM, signal.SIGINT]: signal.signal(signum, stop)
    if (FOLDER / "PIPELINE_STARTED.json").exists() or (FOLDER / "FINAL.json").exists():
        raise RuntimeError("A P10 pipeline already exists; duplicate execution forbidden")
    final = {"started_utc": datetime.now(timezone.utc).isoformat(), "code_commit": os.environ["P2_CODE_COMMIT"],
             "status": "resource_preparation", "public_delivery_complete": False}
    atomic_json(FOLDER / "PIPELINE_STARTED.json", final)
    try:
        invoke("download", "teacher-download.log")
        atomic_json(FOLDER / "STATUS.json", {"stage": "reward_qualification"})
        supervise("qualification", 1, os.environ["P10_GPU1_UUID"])
        invoke("gate", "qualification-score.log")
        gate = json.loads((FOLDER / "scores/qualification.json").read_text())
        if not gate["go"]:
            final["status"] = "reward_qualification_negative"
        else:
            budget = json.loads((ROOT / "metadata/campaign_budget.json").read_text())
            used = gpu_seconds()
            remaining = budget["remaining_total_gpu_process_seconds_at_start"] - used
            round_remaining = (datetime.fromisoformat(budget["gpu_stop_utc"]) - datetime.now(timezone.utc)).total_seconds()
            estimated = budget["main_estimated_gpu_process_seconds"] + budget["acceptance_gpu_limit_seconds"]
            if not admitted_seconds(remaining, round_remaining, estimated, budget["reserve_fraction"]):
                final["status"] = "full_matrix_budget_admission_blocked"
            else:
                atomic_json(FOLDER / "main_budget_lock.json", {"gpu_seconds_already_used": used, "remaining_gpu_seconds": remaining,
                    "round_remaining_seconds": round_remaining, "estimated_seconds": estimated, "required_with_reserve": estimated * 1.2,
                    "complete_arms": ["s", "t", "v"], "complete_seeds": budget["main_seeds"]})
                atomic_json(FOLDER / "STATUS.json", {"stage": "actual_model_acceptance"})
                supervise("acceptance", 0, os.environ["P10_GPU0_UUID"])
                atomic_json(FOLDER / "STATUS.json", {"stage": "full_matrix"})
                supervise("main", 0, os.environ["P10_GPU0_UUID"])
                invoke("score", "main-score.log")
                final.update(status="scored_awaiting_public_delivery", stable_positive_development_result=
                             json.loads((FOLDER / "scores/public_summary.json").read_text())["stable_positive_development_result"])
    except Exception as exc:
        final.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        budget = json.loads((ROOT / "metadata/campaign_budget.json").read_text())
        used = gpu_seconds()
        final.update(finished_utc=datetime.now(timezone.utc).isoformat(), gpu_process_seconds_used=used,
                     cumulative_gpu_process_seconds=budget["prior_gpu_process_seconds"] + used,
                     remaining_gpu_process_seconds=budget["total_gpu_process_seconds"] - budget["prior_gpu_process_seconds"] - used,
                     budget_reset=False, whole_gpu_worker_lifetimes_counted=True)
        atomic_json(FOLDER / "FINAL.json", final)
        atomic_json(FOLDER / "STATUS.json", final)
        print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
