"""Export sealed aggregates and compute receipts only, including negative outcomes."""
import csv
import json
import os
from pathlib import Path


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main():
    folder = Path(os.environ["P10_LOCAL_FOLDER"])
    root = Path(__file__).resolve().parents[1]
    final = json.loads((folder / "FINAL.json").read_text())
    if "finished_utc" not in final or final["public_delivery_complete"]:
        raise ValueError("A newly completed private receipt is required")
    jobs = []
    for name in ["qualification-plan", "acceptance-plan", "main-plan"]:
        path = folder / "controllers" / (name + ".json")
        if not path.exists(): continue
        ledger = json.loads(path.read_text())
        for job in ledger["jobs"]:
            if job.get("wall_seconds") is None:
                from datetime import datetime
                seconds = (datetime.fromisoformat(ledger["finished_utc"]) - datetime.fromisoformat(job["started_utc"])).total_seconds()
            else: seconds = job["wall_seconds"]
            jobs.append({"stage": name, "job": job["label"], "seconds": seconds, "return_code": job.get("return_code"), "timeout": job.get("timeout_or_stop", False)})
    used = sum(j["seconds"] for j in jobs)
    if abs(used - final["gpu_process_seconds_used"]) > 1e-5:
        raise ValueError("Unique whole-worker costs differ from FINAL")
    receipt = {"original_authorized_seconds": 86400., "new_authorized_seconds": 172800., "total_authorized_seconds": 259200.,
               "prior_used_seconds": 66990.15299156541, "p10_used_seconds": used,
               "cumulative_seconds": 66990.15299156541 + used, "remaining_seconds": 259200. - 66990.15299156541 - used,
               "budget_reset": False, "whole_worker_lifetime_counted": True, "jobs": jobs}
    gate_path = folder / "scores/qualification.json"
    main_path = folder / "scores/public_summary.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    matrix = json.loads(main_path.read_text()) if main_path.exists() else None
    public = {"status": final["status"], "source_commit": final["code_commit"], "qualification": gate, "matrix": matrix,
              "independent_generalization_established": False, "compute": receipt,
              "failure_type": final.get("error", "").split(":", 1)[0] or None}
    output = root / "reports"
    for name, value in [("p10_results.json", public), ("p10_compute_receipt.json", receipt)]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    main_rows = []
    if matrix:
        for seed in [57, 58, 59]:
            for arm in ["s", "t", "v"]:
                metric = matrix["seeds"][str(seed)][arm]
                main_rows.append({"seed": seed, "arm": arm, "n": metric["n"],
                    "before_correct": metric["greedy_before"]["correct"], "after_correct": metric["greedy_after"]["correct"],
                    "retained_initial_correct": metric["probe_transition"]["correct_to_correct"],
                    "correct_probe_to_invalid": metric["probe_transition"]["correct_to_invalid"],
                    "correct_probe_to_parsed_wrong": metric["probe_transition"]["correct_to_parsed_wrong"]})
        write_csv(output / "p10_main_results.csv", main_rows)
        write_csv(output / "p10_all_probes.csv", matrix["probe_rows"])
    write_csv(output / "p10_gpu_jobs.csv", jobs)
    table = "\n".join(["| Seed | Arm | Before /16 | After /16 | Same correct probes retained /3 |", "| --- | --- | --- | --- | --- |"] +
        [f"| {r['seed']} | {r['arm']} | {r['before_correct']} | {r['after_correct']} | {r['retained_initial_correct']} |" for r in main_rows]) if main_rows else "No complete training matrix was scored."
    text = ["# P10 completed frozen visual reward campaign", f"Final status: {final['status']}. Executed source: {final['code_commit']}.",
            "Qualification used all18 old candidate trajectories, but only16 unique already observed stream groups. Frozen7B self-reported evidence/consistency and two-order agreement are fallible proxies. No independent medical or domain-shift generalization is established.",
            "Gate outcome: " + (json.dumps(gate, ensure_ascii=False) if gate else "NA; no sealed gate outcome."), table,
            "All nine seed-arm outcomes and every anonymous probe correctness/parse harm are retained when the matrix closes. No checkpoint/seed/format selection substitutes for the original strict prequential primary metric.",
            "Stable-positive development: " + (str(matrix["stable_positive_development_result"]) if matrix else "NA; training not completed."),
            f"Whole GPU-worker lifetime: {used / 3600:.6f}h this campaign; {receipt['cumulative_seconds'] / 3600:.6f}h cumulative of72h; {receipt['remaining_seconds'] / 3600:.6f}h remaining. Old costs were not reset or counted twice.",
            "New actual-backbone probability acceptance is separate from reused original full-model optimizer recovery. Raw medical material, target/readout text, checkpoints, full logs and private paths are excluded. Public delivery still requires proxy push, remote SHA and anonymous final-commit verification."]
    (output / "p10_report.md").write_text("\n\n".join(text) + "\n")
    print(json.dumps({"exported": True, "status": final["status"], "main_rows": len(main_rows), "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
