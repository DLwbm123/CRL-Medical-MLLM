"""Anonymous complete numerical outcomes or NA, with continuous full-life cost."""
import json
import os
from datetime import datetime
from pathlib import Path
from report_p10 import write_csv


def export(folder, output):
    final = json.loads((folder / "FINAL.json").read_text())
    if "finished_utc" not in final: raise ValueError("Diagnostic has not ended")
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / "controllers/main-plan.json"; ledger = json.loads(path.read_text()) if path.exists() else {"jobs": []}
    jobs = [{"job": j["label"], "return_code": j.get("return_code"), "seconds": j["wall_seconds"] if j.get("wall_seconds") is not None else
             (datetime.fromisoformat(ledger["finished_utc"]) - datetime.fromisoformat(j["started_utc"])).total_seconds()} for j in ledger["jobs"]]
    used = sum(j["seconds"] for j in jobs)
    if abs(used - final["gpu_process_seconds_used"]) > 1e-5: raise ValueError("Unique GPU lifetime receipt differs")
    rows, summaries = [], []
    for name, cfg in manifest["configurations"].items():
        path = folder / "runs" / name / "result.json"; result = json.loads(path.read_text()) if path.exists() else None
        job = next((j for j in jobs if j["job"] == name), None)
        closed = result is not None and result.get("sealed") and job is not None and job["return_code"] == 0
        observed = result["measurements"] if closed else [{"section": section, "anonymous_index": index, "seed": cfg["seed"], "precision": cfg["model_dtype"],
            "tokens": "NA", "max_logp_difference": "NA", "within_tolerance": "NA"} for section in ["calibration", "verification"] for index in range(32)]
        rows.extend(observed)
        summaries.append({"seed": cfg["seed"], "precision": cfg["model_dtype"], "execution_status": "completed" if closed else "failed" if job else "not_started",
            "measurements": len(observed) if closed else 0, "checks_passed": sum(r["within_tolerance"] for r in observed) if closed else "NA",
            "max_logp_difference": max(r["max_logp_difference"] for r in observed) if closed else "NA", "optimizer_updates": 0, "retention_harm": "NA_no_update"})
    if len(rows) != 384 or len(summaries) != 6: raise ValueError("Anonymous numerical scope incomplete")
    path = folder / "scores/numerical_summary.json"; summary = json.loads(path.read_text()) if path.exists() else None
    receipt = {"total_authorized_seconds": 259200., "prior_used_seconds": final["cumulative_gpu_process_seconds"] - used, "p18_used_seconds": used,
        "cumulative_seconds": final["cumulative_gpu_process_seconds"], "remaining_seconds": final["remaining_gpu_process_seconds"], "budget_reset": False,
        "whole_worker_lifetime_counted": True, "gpu_jobs": jobs}
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [("measurements", rows), ("workers", summaries), ("gpu_jobs", jobs)]: write_csv(output / ("p18_" + name + ".csv"), values)
    for name, value in [("results", {"source_commit": final["code_commit"], "status": final["status"], "numerical_summary": summary,
        "training_performed": False, "reward_qualification": None, "stable_positive_development_result": False, "compute": receipt}), ("compute_receipt", receipt)]:
        (output / ("p18_" + name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    table = "\n".join(["| Seed | Precision | Status | Passed /64 | Max absolute log-p difference |", "| --- | --- | --- | --- | --- |"] +
        [f"| {r['seed']} | {r['precision']} | {r['execution_status']} | {r['checks_passed']} | {r['max_logp_difference']} |" for r in summaries])
    text = ["# P18 paired numerical precision diagnostic", f"Status: {final['status']}. Executed source: {final['code_commit']}.",
        "The same64 P17 SLAKE training image groups were reused:32 calibration and32 verification. No annotation, correctness score or teacher reward was read. Three new seeds81/82/83 compare BF16 and FP32 at fixed unwarped four-token generation. Both precisions retain the original absolute log-probability error limit0.1; the BF16 control records failures without changing its production guard. The sampled token paths can differ between precisions, so this is not an identical-token causal attribution.", table,
        "All six planned worker outcomes and384 planned measurements, including failed and unstarted NA, are retained. Repeated measurements refer to64 image groups. No optimizer/backward/weight update occurred; all retention transitions are NA. Numerical checks do not establish reward quality, RL improvement, independent medical generalization or a training-cost estimate. No training is automatically launched.",
        f"Whole GPU-worker lifetimes: {used / 3600:.6f}h this round; {receipt['cumulative_seconds'] / 3600:.6f}/72h cumulative; {receipt['remaining_seconds'] / 3600:.6f}h remaining. No budget reset or archive double counting.",
        "Medical identifiers, questions/options/labels/images, token/readout text, checkpoints/full logs, private paths and keys are omitted. Public completion requires a proxy push, remote SHA and anonymous access verification."]
    (output / "p18_report.md").write_text("\n\n".join(text) + "\n")
    return {"worker_rows": len(summaries), "measurement_rows": len(rows)}


if __name__ == "__main__": print(json.dumps(export(Path(os.environ["P10_LOCAL_FOLDER"]), Path(__file__).resolve().parents[1] / "reports")))
