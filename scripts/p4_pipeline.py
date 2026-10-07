"""Run the locked reference gate and, if passed, the full paired matrix once."""
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from state import atomic_json

ROOT = Path(os.environ["P0_ROOT"])
FOLDER = ROOT / "outputs" / os.environ["P2_CAMPAIGN"]
CHILD = None
STOP = False


def request_stop(*_):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def invoke(mode, logfile, **values):
    global CHILD
    if STOP:
        raise RuntimeError("Owned pipeline received a stop request")
    env = dict(os.environ, P2_MODE=mode, CUDA_VISIBLE_DEVICES="", **values)
    with (FOLDER / logfile).open("x") as log:
        CHILD = subprocess.Popen([sys.executable, "-u", os.environ["P2_ENTRY"]],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        code = CHILD.wait()
    CHILD = None
    if code:
        raise RuntimeError(f"Owned stage {mode} exited {code}")


def gpu_seconds():
    total = 0.0
    for path in (FOLDER / "controllers").glob("*.json"):
        ledger = json.loads(path.read_text())
        plan = json.loads((FOLDER / ledger["plan"]).read_text())
        gpu_labels = {j["label"] for j in plan["jobs"] if j.get("gpu", True)}
        for job in ledger["jobs"]:
            if job["label"] in gpu_labels:
                end = datetime.fromisoformat(ledger["finished_utc"])
                fallback = (end - datetime.fromisoformat(job["started_utc"])).total_seconds()
                total += job.get("wall_seconds", fallback)
    return total


def verify_ended(ledger):
    sessions = {j["pid"] for j in ledger["jobs"]}
    live = []
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:
                raw = (path / "stat").read_text()
                fields = raw[raw.rfind(")") + 2:].split()
                if int(fields[3]) in sessions and fields[0] != "Z":
                    live.append({"pid": int(path.name), "start_ticks": fields[19]})
            except (FileNotFoundError, ProcessLookupError):
                continue
    if live:
        raise RuntimeError("Recorded main sessions still have live members")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    if any(int(line.split(",")[0]) in sessions for line in gpu_rows.splitlines() if line.strip()):
        raise RuntimeError("Recorded main worker is still present on a GPU")
    atomic_json(FOLDER / "owned_cleanup_before_score.json",
                {"checked_utc": datetime.now(timezone.utc).isoformat(),
                 "owned_main_processes_ended": True, "live_owned_members": live,
                 "recorded_worker_count": len(sessions)})


def main():
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, request_stop)
    final = {"started_utc": datetime.now(timezone.utc).isoformat(),
             "code_commit": os.environ["P2_CODE_COMMIT"], "status": "running"}
    try:
        manifest = json.loads((FOLDER / "manifest.json").read_text())
        p5 = manifest.get("p5_version") == 1
        p6 = manifest.get("p6_version") == 1
        if p5 or p6:
            audit = {"go": True}
        else:
            invoke("supervise", "supervisor-reference.log", P2_PLAN="reference-plan.json")
            audit = json.loads((FOLDER / "reference/audit_public.json").read_text())
        if not audit["go"]:
            final["status"] = "reference_gate_negative"
        else:
            budget = json.loads((ROOT / "metadata/campaign_budget.json").read_text())
            used = gpu_seconds()
            allowance = float(budget["remaining_total_gpu_process_seconds_at_start"]) - used
            remaining = (datetime.fromisoformat(budget["gpu_stop_utc"]) - datetime.now(timezone.utc)).total_seconds()
            expected = 6 * (3966 if p6 else 3554)
            required = expected * 1.2
            if min(allowance, remaining) < required:
                raise RuntimeError("Remaining budget cannot cover measured matrix cost plus 20 percent reserve")
            atomic_json(FOLDER / "main_budget_lock.json",
                        {"locked_utc": datetime.now(timezone.utc).isoformat(), "gpu_seconds_already_used": used,
                         "expected_seconds": expected, "required_with_reserve": required,
                         "round_seconds_remaining": remaining, "total_seconds_remaining": allowance})
            invoke("supervise", "supervisor-main.log", P2_PLAN="main-plan.json")
            ledger = json.loads((FOLDER / "controllers/main-plan.json").read_text())
            if ledger["status"] != "all_planned_jobs_completed" or len(ledger["jobs"]) != 6:
                raise RuntimeError("Fixed main matrix did not fully close")
            verify_ended(ledger)
            invoke("p6_score" if p6 else ("p5_score" if p5 else "p4_score"), "offline-score.log")
            final["status"] = "scored_awaiting_public_delivery"
            final["stable_positive_development_result"] = json.loads((FOLDER / "scores/public_summary.json").read_text())["stable_positive_development_result"]
    except Exception as exc:
        final.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        final.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                     gpu_process_seconds_used=gpu_seconds(), public_delivery_complete=False)
        atomic_json(FOLDER / "FINAL.json", final)
        print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
