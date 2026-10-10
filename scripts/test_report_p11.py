"""Check that pre-GPU interruption reporting cannot invent scientific results."""
import csv
import json
import tempfile
from pathlib import Path
from report_p11 import export_interruption


def main():
    audit = {"status": "interrupted_before_reward_qualification",
             "receipt_kind": "external_observer_audit_not_pipeline_final", "owned_processes_ended": True,
             "download_receipt_present": False, "prior_gpu_process_seconds": 12.,
             "cumulative_gpu_process_seconds": 12., "remaining_gpu_process_seconds": 259188.,
             "source_commit": "0" * 40, "observed_utc": "2026-10-10T03:37:38+00:00",
             "approximate_host_boot_utc": "2026-10-10T03:10:26+00:00",
             "failure_type": "host_reboot_and_gpu_driver_unavailable", "nvidia_smi_exit_code": 9,
             "partial_weight_bytes": 10, "gpu_zero_evidence": "download never sealed; no GPU stage",
             "original_gpu_stop_utc": "2026-10-11T23:42:28+00:00", "original_hard_deadline_utc": "2026-10-12T01:42:28+00:00"}
    zero_keys = ["gpu_workers_started", "gpu_controllers_present", "gpu_process_seconds_used",
                 "qualification_readouts_generated", "old_candidate_trajectories_audited", "actor_updates", "main_trajectories_started"]
    audit.update(dict.fromkeys(zero_keys, 0))
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / "private"; output = Path(directory) / "public"
        folder.mkdir(); output.mkdir(); path = folder / "interruption_audit.json"
        path.write_text(json.dumps(audit)); export_interruption(folder, output)
        for name, count in [("teacher_cases", 32), ("main_results", 9), ("all_probes", 144)]:
            assert len(list(csv.DictReader((output / f"p11_{name}.csv").open()))) == count
        result = json.loads((output / "p11_results.json").read_text())
        assert result["qualification"] is None and result["matrix"] is None
        assert result["compute"]["jobs"] == [] and result["compute"]["cumulative_seconds"] == 12.
        for key in zero_keys + ["download_receipt_present", "cumulative_gpu_process_seconds"]:
            bad = dict(audit); bad[key] = 1; path.write_text(json.dumps(bad))
            try: export_interruption(folder, output)
            except ValueError: pass
            else: raise AssertionError(f"Accepted inconsistent audit: {key}")
        path.write_text(json.dumps(audit)); (folder / "FINAL.json").write_text("{}")
        try: export_interruption(folder, output)
        except ValueError: pass
        else: raise AssertionError("Overrode a pipeline FINAL")
    print("Interruption export and rejection guards passed")


if __name__ == "__main__":
    main()
