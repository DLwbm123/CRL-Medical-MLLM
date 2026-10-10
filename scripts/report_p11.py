"""Export sealed results or an explicitly audited pre-GPU host interruption."""
import json
import os
from pathlib import Path
from report_p10 import export, write_csv


def export_interruption(folder, output):
    audit = json.loads((folder / "interruption_audit.json").read_text())
    if (folder / "FINAL.json").exists() or audit["receipt_kind"] != "external_observer_audit_not_pipeline_final":
        raise ValueError("Do not replace a pipeline completion receipt")
    if (audit["status"] != "interrupted_before_reward_qualification" or not audit["owned_processes_ended"]
            or audit["gpu_workers_started"] != 0 or audit["gpu_controllers_present"] != 0
            or audit["gpu_process_seconds_used"] != 0 or audit["download_receipt_present"]
            or any(audit[key] != 0 for key in ["qualification_readouts_generated", "old_candidate_trajectories_audited",
                                             "actor_updates", "main_trajectories_started"])):
        raise ValueError("This exporter only accepts verified pre-GPU interruptions")
    prior = audit["prior_gpu_process_seconds"]
    if audit["cumulative_gpu_process_seconds"] != prior or audit["remaining_gpu_process_seconds"] != 259200. - prior:
        raise ValueError("Continuous accounting differs from the interruption audit")
    receipt = {"original_authorized_seconds": 86400., "new_authorized_seconds": 172800.,
               "total_authorized_seconds": 259200., "prior_used_seconds": prior, "p11_used_seconds": 0.,
               "cumulative_seconds": prior, "remaining_seconds": 259200. - prior,
               "budget_reset": False, "whole_worker_lifetime_counted": True, "jobs": [],
               "cpu_download_excluded": True, "evidence": audit["gpu_zero_evidence"]}
    public = {"status": audit["status"], "source_commit": audit["source_commit"],
              "receipt_kind": audit["receipt_kind"], "observed_utc": audit["observed_utc"],
              "approximate_host_boot_utc": audit["approximate_host_boot_utc"],
              "failure_type": audit["failure_type"], "nvidia_smi_exit_code": audit["nvidia_smi_exit_code"],
              "partial_weight_bytes": audit["partial_weight_bytes"], "qualification": None, "matrix": None,
              "stable_positive_development_result": None, "independent_generalization_established": False,
              "original_gpu_stop_utc": audit["original_gpu_stop_utc"],
              "original_hard_deadline_utc": audit["original_hard_deadline_utc"], "compute": receipt}
    for name, value in [("p11_results.json", public), ("p11_compute_receipt.json", receipt)]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    write_csv(output / "p11_teacher_cases.csv", [
        {"section": section, "index": index, "readout_status": "not_generated",
         "accepted": "NA", "accepted_target_correct": "NA", "actor_update_harm": "NA_no_actor_update"}
        for section in ["stream", "probe"] for index in range(16)])
    write_csv(output / "p11_main_results.csv", [
        {"seed": seed, "arm": arm, "status": "not_started", "before_correct": "NA",
         "after_correct": "NA", "retained_initial_correct": "NA"}
        for seed in [60, 61, 62] for arm in ["s", "t", "v"]])
    write_csv(output / "p11_all_probes.csv", [
        {"seed": seed, "arm": arm, "probe_index": index, "status": "not_started",
         "before_correct": "NA", "after_correct": "NA", "harm": "NA_no_actor_update"}
        for seed in [60, 61, 62] for arm in ["s", "t", "v"] for index in range(16)])
    text = f"""# P11 host interruption before reward qualification

Status: {audit['status']}. Executed source: {audit['source_commit']}.

At {audit['observed_utc']}, SSH access had recovered but the recorded pipeline PID/start identity and all owned processes were absent. Host uptime implies a reboot around {audit['approximate_host_boot_utc']} (approximate, not an observed process exit time). NVIDIA-SMI exited9 and could not communicate with the driver; no NVIDIA driver version or device nodes were present. The data filesystem remained mounted. The exact reboot cause and pipeline termination time are unknown.

The pinned medical teacher download did not seal; {audit['partial_weight_bytes'] / 1e9:.3f}GB of partial weight files remain for possible recovery. The last download log contains a network timeout/resume and an incomplete traceback. No pipeline FINAL was written. This report uses a separately identified external interruption audit and does not manufacture a pipeline completion or a scientific negative result.

| Frozen scope | Executed | Outcome |
| --- | --- | --- |
| 32 teacher cases /64 readouts | 0 | Not generated; all32 anonymous rows marked NA |
| 18 old candidate trajectories | 0 | Not audited |
| Reward qualification | 0 | Not computed |
| Actual actor acceptance and fresh baseline | 0 | Not started |
| 9 seed-arm trajectories (60/61/62) | 0 | Not started; scores and probe harms NA |

No GPU worker, actor update, reward selection, new data/final-test access or scientific score occurred. Stable-positive is NA, not established. No P11-specific performance conclusion is possible, and the previously reused development groups do not establish independent generalization.

P11 adds0 GPU-process seconds: download is CUDA-free and the pipeline starts GPU stages only after a sealed successful download; no GPU controllers, reference output, acceptance or main runs exist. Continuous use remains{prior:.12f}s ({prior / 3600:.6f}h) of259200s, leaving{(259200. - prior) / 3600:.6f}h. No reset, archived-controller double counting, or deadline change. Original GPU stop: {audit['original_gpu_stop_utc']}; hard deadline: {audit['original_hard_deadline_utc']}.

Further GPU execution is blocked until the server driver and authorized device mapping are restored. Preserve the partial download and all old artifacts. A subsequent recovery requires separate frozen engineering changes and complete cost/time admission; this interrupted pipeline is not relaunched unchanged. Code, all32 anonymous NA teacher rows, all9 unstarted matrix rows and the continuous compute receipt are public. Raw medical material, private paths, full logs, checkpoints and weights are excluded.
"""
    (output / "p11_report.md").write_text(text)
    print(json.dumps({"exported": True, "status": audit["status"], "gpu_jobs": 0, "teacher_rows": 32, "unstarted_matrix_rows": 9}))


if __name__ == "__main__":
    folder = Path(os.environ["P10_LOCAL_FOLDER"])
    if (folder / "FINAL.json").exists():
        export("p11")
    else:
        export_interruption(folder, Path(__file__).resolve().parents[1] / "reports")
