"""Complete paired negative/NA diagnostics with one continuous 144h cost ledger."""
import json
import os
from pathlib import Path
from report_p16 import export as reward_export
from report_p10 import write_csv


def export(folder, output):
    results = {}; counts = {}
    for key in ["v", "c"]:
        prefix = "p23_" + key
        counts[key] = reward_export(folder, output, prefix, (96, 97, 98), "scores/" + key, 518400., ("calibration",))
        results[key] = json.loads((output / (prefix + "_results.json")).read_text())
    calibration_path = folder / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    public_calibration = {k: calibration[k] for k in ["admitted", "temperature", "threshold", "retired_groups", "accepted", "correct", "nll", "new_labels_used"] if k in calibration} if calibration else None
    checks = []
    for seed in [96, 97, 98]:
        for arm in ["s", "t"]:
            path = folder / "runs" / f"pool-{arm}{seed}" / "probability_check.json"
            v = json.loads(path.read_text()) if path.exists() else None
            checks.append({"seed": seed, "sampler": arm, "cached_max_logp_difference": v["max_logp_difference"] if v else "NA",
                "uncached_max_logp_difference": v["paired_uncached_control"]["max_logp_difference"] if v else "NA", "absolute_tolerance": .1, "training_loss_validated": False})
    write_csv(output / "p23_probability_checks.csv", checks)
    result = {"primary_teacher": "v", "candidate_qualified": results["v"]["qualification"]["go"] if results["v"]["qualification"] else None,
              "calibration": public_calibration, "teachers": results, "compute": results["v"]["compute"],
              "same_compute_views_must_not_be_added": True, "training_performed": False, "training_loss_validated": False}
    (output / "p23_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "p23_report.md").write_text("# P23 calibrated visual answer reward\n\n"
        "Fixed medical Lingshu checkpoint; candidate v uses two-order mean answer-token log probability contrast 2 real minus one fixed RGB128 mask. c uses the same teacher's original evidence-first JSON support/consistency rule. Retired64 P21 training groups fit one bounded temperature and the lowest admissible confidence threshold before any fresh readouts. Calibration is label-assisted development, not independent evidence. A failed retired calibration stops before fresh GPU work with all fresh outcomes NA.\n\n"
        "New64 binary SLAKE training image references exclude every heldout/P16/P17/P21 reference and conservative thumbnail duplicate. Binary annotation vocabulary membership defines task eligibility and is not label-blind. No correctness/class balancing/model-score selection. Patient and pretraining independence remain unverified. A fixed mask probes image dependence, not correctness or causality.\n\n"
        "All original24 candidate gates and24 control gates,64 rows per definition, six pools and384 case rows per definition are reported, including failures/NA. Same-accepted-subset diagnostics prevent global abstention effects being read as reward repair. Six cached0.1 probability guards preserve uncached observations; training native loss/backward remains unvalidated. Calibration worker and all other GPU lifetimes appear once in the continuous144h receipt; paired receipt views must not be added.\n\n"
        "Candidate qualification: " + str(result["candidate_qualified"]) + ". No RL update occurred. Reward qualification is not stable RL improvement or clinical generalization. Medical identifiers/text/options/class labels/images/target readouts/checkpoints/full logs/private paths are excluded. Complete closure and effective-proxy final remote SHA/anonymous access verification are required.\n")
    return counts


if __name__ == "__main__": print(json.dumps(export(Path(os.environ["P10_LOCAL_FOLDER"]), Path(__file__).resolve().parents[1] / "reports")))
