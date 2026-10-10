"""Export complete anonymous reward diagnostics, including explicit missing outcomes."""
import json
import os
from datetime import datetime
from pathlib import Path
from report_p10 import write_csv
from p16_score import CHECKS


def export(folder, output):
    final = json.loads((folder / "FINAL.json").read_text())
    if "finished_utc" not in final: raise ValueError("Campaign has not ended")
    jobs, cpu = [], []
    for stage in ["resources", "qualification", "main"]:
        path = folder / "controllers" / (stage + "-plan.json")
        if not path.exists(): continue
        ledger = json.loads(path.read_text())
        for job in ledger["jobs"]:
            seconds = job.get("wall_seconds")
            if seconds is None: seconds = (datetime.fromisoformat(ledger["finished_utc"]) - datetime.fromisoformat(job["started_utc"])).total_seconds()
            (cpu if stage == "resources" else jobs).append({"stage": stage, "job": job["label"], "seconds": seconds,
                "return_code": job.get("return_code"), "timeout": job.get("timeout_or_stop", False)})
    used = sum(j["seconds"] for j in jobs)
    if abs(used - final["gpu_process_seconds_used"]) > 1e-5: raise ValueError("Unique complete GPU job accounting differs")
    receipt = {"total_authorized_seconds": 259200., "prior_used_seconds": final["cumulative_gpu_process_seconds"] - used,
        "p16_used_seconds": used, "cumulative_seconds": final["cumulative_gpu_process_seconds"],
        "remaining_seconds": final["remaining_gpu_process_seconds"], "budget_reset": False,
        "whole_worker_lifetime_counted": True, "gpu_jobs": jobs, "cpu_resource_jobs": cpu}
    path = folder / "scores/qualification.json"; gate = json.loads(path.read_text()) if path.exists() else None
    teacher_path = folder / "scores/anonymous_teacher_cases.json"; pool_path = folder / "scores/anonymous_pool_cases.json"
    teachers = json.loads(teacher_path.read_text()) if teacher_path.exists() else [
        {"section": section, "anonymous_index": i, "accepted": "NA", "accepted_target_correct": "NA", "retention_harm": "NA_no_update"}
        for section in ["calibration", "verification"] for i in range(32)]
    cases = json.loads(pool_path.read_text()) if pool_path.exists() else [
        {"seed": seed, "sampler": arm, "section": section, "anonymous_index": i, "majority_correct": "NA", "retention_harm": "NA_no_update"}
        for seed in [75, 76, 77] for arm in ["s", "t"] for section in ["calibration", "verification"] for i in range(32)]
    pools = []
    for seed in [75, 76, 77]:
        for arm in ["s", "t"]:
            metric = next((p for p in gate["pool_metrics"] if p["seed"] == seed and p["sampler"] == arm), None) if gate else None
            row = {"seed": seed, "sampler": arm, "groups": 64, "optimizer_updates": 0, "retention_harm": "NA_no_update"}
            for reward in ["majority", "verified"]:
                for rate in ["correct_negative", "wrong_positive"]:
                    row[reward + "_" + rate + "_percent"] = metric[reward][rate]["percent"] if metric else "NA"
            pools.append(row)
    checks = [{"block_sampler": block, "check": name, "passed": gate["blocks"][block]["checks"][name] if gate else "NA"}
        for block in ["calibration_s", "calibration_t", "verification_s", "verification_t"] for name in CHECKS]
    if (len(teachers), len(pools), len(cases), len(checks)) != (64, 6, 384, 24): raise ValueError("Anonymous scope is incomplete")
    output.mkdir(exist_ok=True, parents=True)
    for name, rows in [("teacher_cases", teachers), ("pool_results", pools), ("all_pool_cases", cases), ("qualification_checks", checks), ("gpu_jobs", jobs)]:
        write_csv(output / ("p16_" + name + ".csv"), rows)
    for name, value in [("results", {"status": final["status"], "source_commit": final["code_commit"], "qualification": gate,
        "failure_type": final.get("error", "").split(":", 1)[0] or None, "stable_positive_development_result": False,
        "training_performed": False, "compute": receipt}), ("compute_receipt", receipt)]:
        (output / ("p16_" + name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    table = "\n".join(["| Block / sampler | Accepted /32 | Correct accepted /32 | Six checks passed |", "| --- | --- | --- | --- |"] +
        [f"| {b} | {v['accepted']} | {v['correct_targets']} | {sum(v['checks'].values())}/6 |" for b, v in gate["blocks"].items()]) if gate else "No complete sealed qualification; every unscored outcome remains NA."
    text = ["# P16 medical reward validation", f"Final status: {final['status']}. Executed source: {final['code_commit']}.",
        "64 SLAKE English binary training image groups, fixed32 calibration +32 verification; six frozen SC-8 pools at seeds75/76/77 and two sampling distributions. Validation/test image-reference overlaps and conservative thumbnail duplicates were excluded. Patient independence and absence of pretraining overlap remain unverified.",
        "This new binary development task differs from retired MedXpertQA and cannot reverse its negative results or establish independent clinical generalization. Teacher self-reported visual support is a fallible proxy.", table,
        "Reward qualified: " + (str(gate["go"]) if gate else "NA") + ". No optimizer/backward/training was performed; actor retention transitions are NA. This qualification does not establish stable RL improvement, causality or success on the original task.",
        "All64 teacher rows, six pool summaries,384 repeated case observations and24 checks are retained, including negative/invalid/NA outcomes. Repeated group observations are not384 independent images. No truth entered the reward signal; scoring occurred only after every frozen worker sealed and ended.",
        f"Complete GPU-worker lifetime: {used / 3600:.6f}h this round; {receipt['cumulative_seconds'] / 3600:.6f}/72h cumulative; {receipt['remaining_seconds'] / 3600:.6f}h remaining. CPU resource cost is separate; no budget reset or archive double counting.",
        "Private identifiers, medical text/options/labels/images, teacher targets/readout text, checkpoints/full logs and private paths are omitted. Future training requires a separately frozen complete protocol and budget admission. Public delivery requires effective-proxy push and final remote SHA/anonymous access verification."]
    (output / "p16_report.md").write_text("\n\n".join(text) + "\n")
    return {"teacher_rows": len(teachers), "pool_rows": len(pools), "case_rows": len(cases), "check_rows": len(checks)}


if __name__ == "__main__":
    print(json.dumps(export(Path(os.environ["P10_LOCAL_FOLDER"]), Path(__file__).resolve().parents[1] / "reports")))
