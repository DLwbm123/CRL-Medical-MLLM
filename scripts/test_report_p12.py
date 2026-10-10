"""Unsealed outcomes and undefined reward rates must remain NA in public reports."""
import csv
import json
import os
import runpy
import shutil
import tempfile
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); folder = root / "private"; reports = root / "reports"
        folder.mkdir(); reports.mkdir(); (root / "scripts").mkdir()
        for name in ["report_p10.py", "report_p12.py", "report_p13.py", "report_p14.py"]:
            shutil.copyfile(source / name, root / "scripts" / name)
        (folder / "FINAL.json").write_text(json.dumps({"finished_utc": "fixed", "public_delivery_complete": False,
            "status": "failed", "code_commit": "0" * 40, "gpu_process_seconds_used": 0,
            "cumulative_gpu_process_seconds": 12, "error": "RuntimeError: resource failed"}))
        os.environ["P10_LOCAL_FOLDER"] = str(folder)
        import sys
        sys.path.insert(0, str(root / "scripts")); sys.modules.pop("report_p10", None)
        runpy.run_path(str(root / "scripts/report_p12.py"), run_name="__main__")
        for name, count in [("teacher_cases", 32), ("main_results", 9), ("all_probes", 144)]:
            assert len(list(csv.DictReader((reports / f"p12_{name}.csv").open()))) == count
        public = json.loads((reports / "p12_results.json").read_text())
        assert public["qualification"] is None and public["matrix"] is None
        assert public["compute"]["cumulative_seconds"] == 12
        assert public["resource_preparation"]["gpu_process_seconds"] == 0
        (folder / "scores").mkdir()
        rate = {"correct_negative": {"percent": None}, "wrong_positive": {"percent": None}}
        (folder / "scores/qualification.json").write_text(json.dumps({"go": False, "checks": {"coverage_at_least_half": False},
            "accepted": 0, "correct_targets": 0, "majority": rate, "verified": rate,
            "teacher_probe_accepted": 0, "teacher_probe_correct": 0}))
        runpy.run_path(str(root / "scripts/report_p12.py"), run_name="__main__")
        assert "versus teacher NA" in (reports / "p12_report.md").read_text()
        sys.modules.pop("report_p12", None)
        runpy.run_path(str(root / "scripts/report_p13.py"), run_name="__main__")
        with (reports / "p13_main_results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 9 and {int(row["seed"]) for row in rows} == {66, 67, 68}
        runpy.run_path(str(root / "scripts/report_p14.py"), run_name="__main__")
        with (reports / "p14_main_results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 9 and {int(row["seed"]) for row in rows} == {69, 70, 71}
    print("NA scope and undefined reward-rate export passed")


if __name__ == "__main__":
    main()
