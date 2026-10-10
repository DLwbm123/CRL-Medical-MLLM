"""Add CPU recovery closure and complete anonymous NA scope to the sealed exporter."""
import json
import os
from pathlib import Path
from report_p10 import export, write_csv


def main():
    export("p12")
    folder = Path(os.environ["P10_LOCAL_FOLDER"])
    output = Path(__file__).resolve().parents[1] / "reports"
    resource_path = folder / "controllers/resources-plan.json"
    resource = {"status": "not_completed", "cpu_only": True, "gpu_process_seconds": 0,
                "same_boot_completion_fence_passed": (folder / "resource_ready.json").exists()}
    if resource_path.exists():
        ledger = json.loads(resource_path.read_text())
        plan = json.loads((folder / ledger["plan"]).read_text())
        if len(plan["jobs"]) != 1 or plan["jobs"][0].get("gpu") is not False:
            raise ValueError("Resource stage was not the single declared CPU job")
        resource.update(status=ledger["status"], jobs=[{key: job.get(key) for key in
            ["return_code", "wall_seconds", "timeout_or_stop"]} for job in ledger["jobs"]])
    public_path = output / "p12_results.json"
    public = json.loads(public_path.read_text()); public["resource_preparation"] = resource
    public_path.write_text(json.dumps(public, indent=2) + "\n")
    if not (folder / "scores/anonymous_teacher_cases.json").exists():
        write_csv(output / "p12_teacher_cases.csv", [{"scope": scope, "anonymous_case": f"{i:02d}",
            "qualification": "NA", "reason": "no sealed teacher scoring"}
            for scope in ["retired_stream", "legacy_probe"] for i in range(1, 17)])
    if public["matrix"] is None:
        started = {}
        path = folder / "controllers/main-plan.json"
        if path.exists():
            started = {j["label"]: j for j in json.loads(path.read_text())["jobs"]}
        rows = [{"seed": seed, "arm": arm, "before_correct": "NA", "after_correct": "NA",
                 "worker_started": f"main-{arm}{seed}" in started,
                 "return_code": started.get(f"main-{arm}{seed}", {}).get("return_code", "NA"),
                 "reason": "no sealed complete matrix scoring"} for seed in [63, 64, 65] for arm in ["s", "t", "v"]]
        write_csv(output / "p12_main_results.csv", rows)
        write_csv(output / "p12_all_probes.csv", [{"seed": row["seed"], "arm": row["arm"],
            "anonymous_probe": f"{i:02d}", "transition": "NA", "reason": row["reason"]}
            for row in rows for i in range(1, 17)])
    with (output / "p12_report.md").open("a") as stream:
        stream.write(f"\nCPU resource-stage status: {resource['status']}; same-boot completion fence: {resource['same_boot_completion_fence_passed']}. CPU preparation is recorded separately and adds zero GPU-process seconds. An unscored planned outcome remains NA, including every declared seed/arm and all144 planned probe transitions.\n")


if __name__ == "__main__":
    main()
