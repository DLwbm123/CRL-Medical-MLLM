"""Full planned matrix and probe outcomes, including failures and explicit NA."""
import json
import os
from datetime import datetime
from pathlib import Path
from p24_contract import ARMS, SEEDS, TOTAL
from report_p10 import write_csv


def export(folder, output, prefix="p24"):
    if (folder / "FINAL.json").exists():
        final = json.loads((folder / "FINAL.json").read_text())
    else:
        startup = json.loads((folder / "startup_admission.json").read_text())
        if startup["pipeline_started"] or startup["GPU_work_started"] or startup["FINAL_exists"]:
            raise ValueError("An unclosed execution cannot become an admission-only report")
        source = json.loads((folder / "source_receipt.json").read_text())
        final = {"status": startup["startup_admission"], "code_commit": source["code_commit"],
            "finished_utc": startup["checked_utc"], "gpu_process_seconds_used": 0.,
            "cumulative_gpu_process_seconds": startup["continuous_gpu_seconds_used"],
            "remaining_gpu_process_seconds": startup["remaining_gpu_seconds"]}
    if "finished_utc" not in final: raise ValueError("An ended campaign is required")
    jobs = []
    for stage in ["acceptance", "main"]:
        path = folder / "controllers" / (stage + "-plan.json")
        if not path.exists(): continue
        ledger = json.loads(path.read_text())
        for job in ledger["jobs"]:
            cost = job.get("wall_seconds")
            if cost is None: cost = (datetime.fromisoformat(ledger["finished_utc"]) - datetime.fromisoformat(job["started_utc"])).total_seconds()
            jobs.append({"stage": stage, "job": job["label"], "seconds": cost, "return_code": job.get("return_code"), "timeout": job.get("timeout_or_stop")})
    used = sum(j["seconds"] for j in jobs)
    if abs(used - final["gpu_process_seconds_used"]) > 1e-5 or abs(TOTAL - final["cumulative_gpu_process_seconds"] - final["remaining_gpu_process_seconds"]) > 1e-5:
        raise ValueError("Continuous unique whole-worker ledger differs")
    receipt = {"total_authorized_seconds": TOTAL, "prior_used_seconds": final["cumulative_gpu_process_seconds"] - used,
        "this_round_seconds": used, "cumulative_seconds": final["cumulative_gpu_process_seconds"], "remaining_seconds": final["remaining_gpu_process_seconds"],
        "budget_reset": False, "whole_worker_lifetime_counted": True, "jobs": jobs}
    path = folder / "scores/public_summary.json"; matrix = json.loads(path.read_text()) if path.exists() else None
    path = folder / "acceptance.json"; acceptance = json.loads(path.read_text()) if path.exists() else None
    path = folder / "budget_admission.json"; budget = json.loads(path.read_text()) if path.exists() else None
    rows, cases, probes = [], [], []
    for seed in SEEDS:
        for arm in ARMS:
            metric = matrix["seeds"][str(seed)][arm] if matrix else None
            job = next((j for j in jobs if j["job"] == f"main-{arm}{seed}"), None)
            rows.append({"seed": seed, "arm": arm, "n": 64, "execution_status": "completed" if job and job["return_code"] == 0 else "failed" if job else "not_started",
                "before_correct": metric["greedy_before"]["correct"] if metric else "NA", "after_correct": metric["greedy_after"]["correct"] if metric else "NA",
                "original_correct_probes_retained": metric["probe_transition"]["correct_to_correct"] if metric else "NA",
                "correct_probe_to_invalid": metric["probe_transition"]["correct_to_invalid"] if metric else "NA",
                "correct_probe_to_parsed_wrong": metric["probe_transition"]["correct_to_parsed_wrong"] if metric else "NA"})
            run = folder / "runs" / f"main-{arm}{seed}" / "cases"
            for i in range(64):
                path = run / f"i{i:06d}.json"; row = json.loads(path.read_text()) if path.exists() else None
                cases.append({"seed": seed, "arm": arm, "anonymous_index": i, "observed": bool(row),
                    "update_applied": row["update_applied"] if row else "NA", "skip_reason": row.get("skip_reason") if row else "NA",
                    "invalid_rollouts": 8 - row["vote"]["valid"] if row else "NA", "any_length_cap": any(row["hit_length_cap"]) if row else "NA"})
            probes.extend([p for p in matrix["probe_rows"] if p["seed"] == seed and p["arm"] == arm] if matrix else
                [{"seed": seed, "arm": arm, "probe_index": i, "initial_correct": "NA", "final_correct": "NA", "initial_parsed": "NA", "final_parsed": "NA"} for i in range(16)])
    if (len(rows), len(cases), len(probes)) != (3 * len(SEEDS), 192 * len(SEEDS), 48 * len(SEEDS)): raise ValueError("Public planned coverage differs")
    public = {"status": final["status"], "source_commit": final["code_commit"], "failure_type": final.get("error_type"),
        "training_acceptance": acceptance, "budget": budget, "matrix": matrix, "compute": receipt,
        "stable_positive_development_result": matrix["stable_positive_development_result"] if matrix else None,
        "independent_clinical_generalization_established": False,
        "pipeline_started": (folder / "PIPELINE_STARTED.json").exists(),
        "original_FINAL_exists": (folder / "FINAL.json").exists()}
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [("main_results", rows), ("all_training_cases", cases), ("all_probes", probes), ("gpu_jobs", jobs)]: write_csv(output / f"{prefix}_{name}.csv", values)
    for name, value in [("results", public), ("compute_receipt", receipt)]: (output / f"{prefix}_{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    table = "\n".join(["| Seed | Arm | Strict before /64 | Strict after /64 | Original correct probes retained /3 |", "| --- | --- | --- | --- | --- |"] +
        [f"| {r['seed']} | {r['arm']} | {r['before_correct']} | {r['after_correct']} | {r['original_correct_probes_retained']} |" for r in rows])
    (output / (prefix + "_report.md")).write_text("\n\n".join(["# " + prefix.upper() + " complete conditional RL development matrix", f"Status: {final['status']}. Source: {final['code_commit']}. Pipeline started: {public['pipeline_started']}; original FINAL exists: {public['original_FINAL_exists']}. An admission-only report does not create an experiment FINAL.",
        f"All64 already observed P23 SLAKE binary training groups in their frozen order, original16 MedXpert retention probes, seeds{SEEDS} and original SPINE(s), raw majority TTRL(t), fixed calibrated visual-reward TTRL(v). No final-test access, data reselection, changed parser or relaxed tolerances. The original objective uses differentiable generation replay shared by actor statistics, gradients and fixed reference; original full-prefix observations remain paired same-token controls. A cached frozen generation check is not training acceptance.",
        table, f"All{3 * len(SEEDS)} planned trajectories,{192 * len(SEEDS)} planned case observations and{48 * len(SEEDS)} individual probe outcomes are published, with missing scientific outcomes explicitly NA. Complete negative transitions, reward-direction and format diagnostics remain in the aggregate when the entire matrix seals. Engineering diagnostic advantages do not constitute scientific training rewards; those temporary weights are discarded.",
        f"This-round GPU full-worker lifetime {used/3600:.6f}h; continuous{receipt['cumulative_seconds']/3600:.6f}/144h; remaining{receipt['remaining_seconds']/3600:.6f}h. Acceptance/loading/CPU optimizer/saving/failure lifetimes count; P23 reused signal jobs are not counted again.",
        "Reused small development results do not reverse the old MedXpert negative experiments or establish patient independence, pretraining separation or clinical generalization. Raw medical IDs/text/options/class labels/images, teacher targets/readouts/token text, checkpoints/full logs/private paths are excluded. Effective-proxy push, remote SHA and anonymous access verification remain required for actual delivery."]) + "\n")
    return {"matrix_rows": 3 * len(SEEDS), "case_rows": 192 * len(SEEDS), "probe_rows": 48 * len(SEEDS)}


if __name__ == "__main__": print(json.dumps(export(Path(os.environ["P10_LOCAL_FOLDER"]), Path(__file__).resolve().parents[1] / "reports")))
