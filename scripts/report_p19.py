"""Complete anonymous same-token numerical outcomes and continuous cost."""
import json
import os
from datetime import datetime
from pathlib import Path
from report_p10 import write_csv


def export(folder, output):
    final = json.loads((folder / "FINAL.json").read_text())
    if "finished_utc" not in final: raise ValueError("Diagnostic not ended")
    manifest = json.loads((folder / "manifest.json").read_text()); p = folder / "controllers/main-plan.json"
    ledger = json.loads(p.read_text()) if p.exists() else {"jobs": []}
    jobs = [{"job": j["label"], "return_code": j.get("return_code"), "seconds": j["wall_seconds"] if j.get("wall_seconds") is not None else
        (datetime.fromisoformat(ledger["finished_utc"]) - datetime.fromisoformat(j["started_utc"])).total_seconds()} for j in ledger["jobs"]]
    used = sum(j["seconds"] for j in jobs)
    if abs(used - final["gpu_process_seconds_used"]) > 1e-5: raise ValueError("Unique complete lifetime accounting differs")
    rows, summaries = [], []
    for name, cfg in manifest["configurations"].items():
        p = folder / "runs" / name / "result.json"; result = json.loads(p.read_text()) if p.exists() else None
        job = next((j for j in jobs if j["job"] == name), None)
        closed = result is not None and result.get("sealed") and job is not None and job["return_code"] == 0
        values = result["measurements"] if closed else [{"section": section, "anonymous_index": i, "seed": cfg["seed"], "path": path,
            "tokens": "NA", "max_logp_difference": "NA", "within_tolerance": "NA", "same_sampled_tokens": "NA"}
            for section in ["calibration", "verification"] for i in range(32) for path in ["uncached", "cached"]]
        rows.extend(values)
        for path in ["uncached", "cached"]:
            observed = [r for r in values if r["path"] == path]
            summaries.append({"seed": cfg["seed"], "path": path, "execution_status": "completed" if closed else "failed" if job else "not_started",
                "checks_passed": sum(r["within_tolerance"] for r in observed) if closed else "NA",
                "max_logp_difference": max(r["max_logp_difference"] for r in observed) if closed else "NA",
                "optimizer_updates": 0, "retention_harm": "NA_no_update"})
    if len(rows) != 384 or len(summaries) != 6: raise ValueError("Incomplete anonymous scope")
    p = folder / "scores/numerical_summary.json"; summary = json.loads(p.read_text()) if p.exists() else None
    receipt = {"total_authorized_seconds": 259200., "prior_used_seconds": final["cumulative_gpu_process_seconds"] - used,
        "p19_used_seconds": used, "cumulative_seconds": final["cumulative_gpu_process_seconds"], "remaining_seconds": final["remaining_gpu_process_seconds"],
        "budget_reset": False, "whole_worker_lifetime_counted": True, "gpu_jobs": jobs}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [("measurements", rows), ("path_results", summaries), ("gpu_jobs", jobs)]: write_csv(output / ("p19_" + name + ".csv"), value)
    for name, value in [("results", {"source_commit": final["code_commit"], "status": final["status"], "numerical_summary": summary,
        "reward_qualification": None, "training_performed": False, "stable_positive_development_result": False, "compute": receipt}), ("compute_receipt", receipt)]:
        (output / ("p19_" + name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    table = "\n".join(["| Seed | Recompute path | Status | Passed /64 | Max absolute log-p difference |", "| --- | --- | --- | --- | --- |"] +
        [f"| {r['seed']} | {r['path']} | {r['execution_status']} | {r['checks_passed']} | {r['max_logp_difference']} |" for r in summaries])
    text = ["# P19 same-token cache-path numerical diagnostic", f"Status: {final['status']}. Executed source: {final['code_commit']}.",
        "Exactly64 previously admitted P17 SLAKE training inputs were reused,32 calibration and32 verification, without reading annotations or reward accuracy. BF16 actor, original prompt/processor and unwarped four-token generation are frozen. Each new seed84/85/86 draws one response per input; that exact response supplies both complete-prefix uncached and manually incremental cached recomputation. The cached path uses the installed model's generation preparation, cache positions, attention-mask growth and vision-prefill handling. Original0.1 tolerance remains unchanged and all384 paired measurements/NA are retained.", table,
        "Only all192 cached measurements within0.1 plus complete worker closure/provenance/coverage/frozen state constitute this numerical diagnostic passing. No production probability guard or RL loss path was changed. Cache-path agreement does not prove medical reward quality, an uncached training loss is valid, RL improvement, or an underlying library bug. No optimizer/backward/weight update/teacher generation/correctness scoring or automatic training occurred. Retention transitions remain NA; repeated observations are64 image groups, not384 independent patients.",
        f"Complete GPU-worker lifetime: {used / 3600:.6f}h this round; {receipt['cumulative_seconds'] / 3600:.6f}/72h cumulative; {receipt['remaining_seconds'] / 3600:.6f}h remaining. No reset or archived/active double counting.",
        "Private identifiers, medical text/options/labels/images, token/readout text, checkpoints/full logs, private paths and keys are omitted. Public completion requires proxy push, final remote SHA and anonymous access verification. Reward qualification and future RL require separately frozen full scope and real training-cost admission."]
    (output / "p19_report.md").write_text("\n\n".join(text) + "\n")
    return {"path_rows": 6, "measurement_rows": 384}


if __name__ == "__main__": print(json.dumps(export(Path(os.environ["P10_LOCAL_FOLDER"]), Path(__file__).resolve().parents[1] / "reports")))
