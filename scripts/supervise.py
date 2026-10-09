"""Bounded controller for recorded child processes on one authorized GPU."""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ["P0_ROOT"])
FOLDER = ROOT / "outputs" / os.environ["P2_CAMPAIGN"]
PLAN = json.loads((FOLDER / os.environ["P2_PLAN"]).read_text())
BUDGET = json.loads((ROOT / "metadata/campaign_budget.json").read_text())
STOP_UTC = datetime.fromisoformat(BUDGET["gpu_stop_utc"])
START = time.monotonic()
STOP_MONO = START + (STOP_UTC - datetime.now(timezone.utc)).total_seconds()
STOP_REQUESTED = False
FORBIDDEN = [word.lower() for word in json.loads(os.environ.get("P2_FORBIDDEN_ARGV", "[]"))]


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


for signum in [signal.SIGTERM, signal.SIGINT]:
    signal.signal(signum, request_stop)


def event(stage, **values):
    record = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(),
              "elapsed_seconds": time.monotonic() - START, "stage": stage, **values}
    with (FOLDER / "jobs.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
    print(json.dumps(record), flush=True)


def process_info(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return {"pid": int(pid), "ppid": int(fields[1]), "group": int(fields[2]),
                "session": int(fields[3]), "start_ticks": fields[19], "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_processes(pid):
    # Every worker is a new POSIX session. Unrelated jobs cannot be members.
    records = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            item = process_info(entry.name)
            if item and item["session"] == pid:
                records.append(item)
    return records


def inspect_owned(pid, gpu):
    owned = owned_processes(pid)
    event("owned_process_record", root_pid=pid, processes=owned)
    if owned:
        args = subprocess.check_output(["ps", "-ww", "-o", "pid=,ppid=,args=", "-p", ",".join(str(x["pid"]) for x in owned)], text=True)
        if any(word in args.lower() for word in FORBIDDEN):
            raise RuntimeError("Owned process arguments violate neutral-entry policy")
        event("argv_checked", root_pid=pid, commands=args.strip())
    if gpu:
        rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
        ids = {str(x["pid"]) for x in owned}
        rows = [line for line in rows.splitlines() if line.split(",")[0].strip() in ids]
        if any(word in " ".join(rows).lower() for word in FORBIDDEN):
            raise RuntimeError("GPU process name violates neutral-entry policy")
        event("gpu_argv_checked", root_pid=pid, rows=rows)


def stop_owned(pid):
    records = owned_processes(pid)
    event("termination_records", root_pid=pid, processes=records)
    for record in records:
        current = process_info(record["pid"])
        if current and current["start_ticks"] == record["start_ticks"]:
            os.kill(record["pid"], signal.SIGTERM)
    end = min(time.monotonic() + 30, STOP_MONO)
    while time.monotonic() < end and any(x["state"] != "Z" for x in owned_processes(pid)):
        time.sleep(0.5)
    records = owned_processes(pid)
    event("final_termination_records", root_pid=pid, processes=records)
    for record in records:
        current = process_info(record["pid"])
        if current and current["start_ticks"] == record["start_ticks"] and current["state"] != "Z":
            os.kill(record["pid"], signal.SIGKILL)


def gpu_ready():
    index = int(os.environ["P2_GPU_INDEX"])
    text = subprocess.check_output(["nvidia-smi", "-i", str(index), "--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
    uuid, free, total = [part.strip() for part in text.split(",")]
    if uuid != os.environ["P2_GPU_UUID"] or int(free) < PLAN.get("minimum_free_mib", 48000):
        raise RuntimeError("Authorized GPU mapping or available memory changed")
    event("gpu_preflight", gpu_index=index, free_mib=int(free), total_mib=int(total))


def main():
    if PLAN.get("requires_acceptance") and not json.loads((FOLDER / "acceptance.json").read_text())["passed"]:
        raise RuntimeError("Engineering acceptance is required before the main plan")
    ledger = {"controller_pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "stop_utc": STOP_UTC.isoformat(), "plan": os.environ["P2_PLAN"], "jobs": []}
    pid = None
    try:
        for index, job in enumerate(PLAN["jobs"]):
            remaining = STOP_MONO - time.monotonic()
            if STOP_REQUESTED or remaining < 300:
                ledger["status"] = "stopped_before_next_job"
                break
            gpu = job.get("gpu", True)
            if gpu:
                gpu_ready()
            environment = dict(os.environ, **job["environment"])
            environment["CUDA_VISIBLE_DEVICES"] = os.environ["P2_GPU_UUID"] if gpu else ""
            limit = min(float(job.get("max_seconds", 3600)), remaining - 60)
            environment["P2_MAX_JOB_SECONDS"] = str(limit - 30)
            label = job["label"]
            log_path = FOLDER / "logs" / (label + ".log")
            log_path.parent.mkdir(exist_ok=True)
            started = time.monotonic()
            started_utc = datetime.now(timezone.utc).isoformat()
            with log_path.open("x") as log:
                child = subprocess.Popen([sys.executable, "-u", os.environ["P2_ENTRY"]], env=environment,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pid = child.pid
            job_record = {"label": label, "pid": pid, "started_utc": started_utc,
                          "max_seconds": limit, "identity": process_info(pid), "log": str(log_path.relative_to(FOLDER))}
            ledger["jobs"].append(job_record)
            (FOLDER / "controller.json").write_text(json.dumps(ledger, indent=2))
            event("job_started", **job_record)
            inspected = False
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if not inspected and elapsed >= 10:
                    inspect_owned(pid, gpu)
                    inspected = True
                if STOP_REQUESTED or elapsed >= limit or time.monotonic() >= STOP_MONO - 30:
                    stop_owned(pid)
                    child.wait(timeout=10)
                    job_record["timeout_or_stop"] = True
                    break
                try:
                    child.wait(timeout=min(5, max(0.1, limit - elapsed)))
                except subprocess.TimeoutExpired:
                    pass
            if not inspected:
                event("short_job_ended_before_argv_sample", root_pid=pid)
            job_record.update(return_code=child.returncode, wall_seconds=time.monotonic() - started)
            event("job_finished", **job_record)
            pid = None
            if child.returncode != 0 or job_record.get("timeout_or_stop"):
                ledger["status"] = "stopped_after_job_failure"
                break
            if "expected_cursor" in job:
                result = json.loads((FOLDER / "runs" / job["environment"]["P2_RUN"] / "result.json").read_text())
                if result["cursor"] != job["expected_cursor"] or result["status"] not in {"completed", "segment_complete"}:
                    raise RuntimeError("Worker did not complete its declared prefix")
        else:
            ledger["status"] = "all_planned_jobs_completed"
    except Exception as exc:
        ledger.update(status="controller_error", error=f"{type(exc).__name__}: {exc}")
        if pid is not None:
            stop_owned(pid)
        raise
    finally:
        ledger["elapsed_seconds"] = time.monotonic() - START
        ledger["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (FOLDER / "controller.json").write_text(json.dumps(ledger, indent=2))
        archive = FOLDER / "controllers"
        archive.mkdir(exist_ok=True)
        (archive / (Path(os.environ["P2_PLAN"]).stem + ".json")).write_text(json.dumps(ledger, indent=2))
        event("controller_finished", status=ledger["status"])
    if ledger["status"] != "all_planned_jobs_completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
