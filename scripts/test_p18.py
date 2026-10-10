"""Boundary, unchanged production guard and complete diagnostic aggregation checks."""
import copy
import json
import tempfile
from pathlib import Path
from continual import Engine
from p18_numeric import validate_manifest
from p18_pipeline import aggregate
from report_p18 import export
from prepare_campaign import write_manifest


def main():
    engine = object.__new__(Engine)
    engine.probability_measurement = lambda _: {"within_tolerance": False, "max_logp_difference": .13839125633239746}
    try: engine.probability_check(None)
    except RuntimeError: pass
    else: raise AssertionError("Production probability guard was relaxed")
    engine.probability_measurement = lambda _: {"within_tolerance": True, "max_logp_difference": .01}
    assert engine.probability_check(None)["max_logp_difference"] == .01
    cfg = {"method": "Frozen SC-8", "evaluation_labels_allowed": False, "probability_check_bf16_atol": .1,
           "temperature": 1., "top_p": 1., "top_k": 0, "repetition_penalty": 1.}
    configs = {f"numeric-{arm}{seed}": {**cfg, "seed": seed, "model_dtype": precision} for seed in [81, 82, 83] for arm, precision in [("b", "bfloat16"), ("f", "float32")]}
    manifest = {"p18_version": 1, "configurations": configs, "rollout_seeds": [81, 82, 83], "stream": [{}] * 32, "probe": [{}] * 32}
    validate_manifest(manifest)
    broken = copy.deepcopy(manifest); broken["configurations"]["numeric-f81"]["probability_check_bf16_atol"] = .2
    try: validate_manifest(broken)
    except ValueError: pass
    else: raise AssertionError("Paired check accepted a relaxed threshold")
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name); (folder / "controllers").mkdir(); (folder / "scores").mkdir()
        digest = write_manifest(folder / "manifest.json", manifest)
        (folder / "PIPELINE_STARTED.json").write_text(json.dumps({"code_commit": "synthetic"}))
        jobs = []
        for name, conf in configs.items():
            run = folder / "runs" / name; run.mkdir(parents=True)
            rows = [{"section": section, "anonymous_index": i, "seed": conf["seed"], "precision": conf["model_dtype"], "tokens": 4,
                     "max_logp_difference": .01 if conf["model_dtype"] == "float32" else .14, "within_tolerance": conf["model_dtype"] == "float32"} for section in ["calibration", "verification"] for i in range(32)]
            (run / "result.json").write_text(json.dumps({"status": "completed", "code_commit": "synthetic", "sealed": True, "cursor": 64, "configuration": conf, "labels_read": False,
                "weights_unchanged": True, "optimizer_updates": 0, "manifest_sha256": digest, "measurements": rows}))
            jobs.append({"label": name, "return_code": 0, "wall_seconds": 1})
        (folder / "controllers/main-plan.json").write_text(json.dumps({"status": "all_planned_jobs_completed", "jobs": jobs}))
        closure = folder / "owned_cleanup_before_score.json"; closure.write_text(json.dumps({"owned_main_processes_ended": False}))
        try: aggregate(folder)
        except ValueError: pass
        else: raise AssertionError("Aggregation ignored live numerical workers")
        closure.write_text(json.dumps({"owned_main_processes_ended": True}))
        summary = aggregate(folder)
        assert summary["fp32_checks_passed"] == 192 and summary["fp32_all_checks_passed"] and not summary["training_performed"]
        (folder / "FINAL.json").write_text(json.dumps({"finished_utc": "2026-10-10T00:00:00+00:00", "status": "numerical_checks_passed", "code_commit": "synthetic",
            "gpu_process_seconds_used": 6, "cumulative_gpu_process_seconds": 83148.44163591831, "remaining_gpu_process_seconds": 176051.5583640817}))
        assert export(folder, folder / "public") == {"worker_rows": 6, "measurement_rows": 384}
        (folder / "controllers/main-plan.json").write_text(json.dumps({"status": "stopped_after_job_failure", "jobs": []}))
        for run in (folder / "runs").iterdir(): (run / "result.json").unlink()
        (folder / "scores/numerical_summary.json").unlink()
        final = json.loads((folder / "FINAL.json").read_text()); final.update(status="failed", gpu_process_seconds_used=0); (folder / "FINAL.json").write_text(json.dumps(final))
        assert export(folder, folder / "public") == {"worker_rows": 6, "measurement_rows": 384}
        assert "not_started" in (folder / "public/p18_workers.csv").read_text()
    print(json.dumps({"unchanged_probability_guard_and_full_diagnostic_boundary_checks_passed": True}))


if __name__ == "__main__": main()
