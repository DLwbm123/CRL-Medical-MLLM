"""Aggregate only the closed full same-token numerical scope."""
import json
import math
from p18_pipeline import main
from p19_cache import validate_manifest
from state import atomic_json


def aggregate(folder):
    manifest = json.loads((folder / "manifest.json").read_text()); validate_manifest(manifest)
    ledger = json.loads((folder / "controllers/main-plan.json").read_text())
    if ledger["status"] != "all_planned_jobs_completed" or len(ledger["jobs"]) != 3 or any(j["return_code"] != 0 for j in ledger["jobs"]): raise ValueError("Incomplete numerical matrix")
    if not json.loads((folder / "owned_cleanup_before_score.json").read_text())["owned_main_processes_ended"]: raise ValueError("Workers not closed")
    source = json.loads((folder / "PIPELINE_STARTED.json").read_text())["code_commit"]; rows = []
    for name, cfg in manifest["configurations"].items():
        result = json.loads((folder / "runs" / name / "result.json").read_text())
        if result["status"] != "completed" or not result["sealed"] or result["cursor"] != 64 or len(result["measurements"]) != 128 or result["configuration"] != cfg or result["code_commit"] != source or result["labels_read"] or not result["weights_unchanged"] or result["optimizer_updates"] != 0 or result["manifest_sha256"] != (folder / "manifest.sha256").read_text().strip(): raise ValueError("Closed result differs from scope")
        for index, pair in enumerate(zip(result["measurements"][::2], result["measurements"][1::2])):
            if pair[0]["tokens"] != pair[1]["tokens"]: raise ValueError("Token scope not paired")
            for row, path in zip(pair, ["uncached", "cached"]):
                e = row["max_logp_difference"]
                if row["path"] != path or row["seed"] != cfg["seed"] or row["section"] != ("calibration" if index < 32 else "verification") or row["anonymous_index"] != index % 32 or not row["same_sampled_tokens"] or row["absolute_tolerance"] != .1 or not 1 <= row["tokens"] <= 4 or not math.isfinite(e) or e < 0 or row["within_tolerance"] != (e <= .1): raise ValueError("Paired coverage or original threshold differs")
                errors = row["per_token_abs_difference"]
                if len(errors) != row["tokens"] or any(not math.isfinite(x) or x < 0 for x in errors) or max(errors) != e or row["first_token_abs_difference"] != errors[0]: raise ValueError("Per-token numerical evidence differs")
                rows.append(row)
    cached = [r for r in rows if r["path"] == "cached"]
    summary = {"complete_workers": 3, "measurements": len(rows), "cached_checks_passed": sum(r["within_tolerance"] for r in cached),
        "diagnostic_passed": all(r["within_tolerance"] for r in cached), "absolute_tolerance": .1,
        "same_sampled_tokens": True, "numerical_diagnostic_only": True, "training_performed": False,
        "reward_qualification": None, "stable_positive_development_result": False}
    atomic_json(folder / "scores/numerical_summary.json", summary); return summary


if __name__ == "__main__": main(aggregate)
