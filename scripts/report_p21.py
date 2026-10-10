"""Both complete source audits share one continuous compute ledger."""
import json
import os
from pathlib import Path
from report_p16 import export as reward_export
from report_p10 import write_csv


def export(folder, output):
    counts = {}; results = {}
    for key in ["q", "m"]:
        prefix = "p21_" + key
        counts[key] = reward_export(folder, output, prefix, (90, 91, 92), "scores/" + key)
        results[key] = json.loads((output / (prefix + "_results.json")).read_text())
    checks = []
    for seed in [90, 91, 92]:
        for arm in ["s", "t"]:
            path = folder / "runs" / f"pool-{arm}{seed}" / "probability_check.json"
            v = json.loads(path.read_text()) if path.exists() else None
            checks.append({"seed": seed, "sampler": arm, "cached_max_logp_difference": v["max_logp_difference"] if v else "NA",
                "uncached_max_logp_difference": v["paired_uncached_control"]["max_logp_difference"] if v else "NA",
                "cached_within_original_tolerance": v["within_tolerance"] if v else "NA", "same_sampled_tokens": v["same_sampled_tokens"] if v else "NA",
                "absolute_tolerance": .1, "training_loss_validated": False})
    write_csv(output / "p21_probability_checks.csv", checks)
    result = {"primary_teacher": "q", "candidate_qualified": results["q"]["qualification"]["go"] if results["q"]["qualification"] else None,
        "control_is_not_a_selectable_replacement": True, "teachers": results, "compute": results["q"]["compute"],
        "unique_input_groups": 64, "teacher_readouts_planned": 256, "underlying_candidate_groups_planned": 384,
        "paired_teacher_case_rows_planned": 768, "probability_checks": checks, "training_performed": False,
        "stable_positive_development_result": False, "training_loss_validated": False}
    (output / "p21_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "p21_report.md").write_text("# P21 paired frozen teacher-source reward validation\n\n"
        "Predeclared candidate q is the pinned Qwen2.5-VL-7B source; m is the pinned Lingshu-7B medical control. Both sources and all48 gates are reported. A passing control cannot replace a failed primary. The same64 fresh SLAKE training image groups and3072 frozen candidate answers are used for both audits. No optimizer/backward/weight update or automatic RL.\n\n"
        "Training-answer vocabulary membership defines this binary task and is not label-blind; no correctness/class balancing/model-score selection. Validation/test and every P16/P17 image reference plus conservative thumbnail duplicates are excluded. No heldout labels/generation/scoring. Patient and pretraining independence remain unverified. This new binary development task does not reverse MedXpertQA negatives or establish clinical generalization.\n\n"
        "Both reports contain64 teacher rows, six pool summaries,384 case rows and24 checks including every failure/NA. The paired768 reward observations reuse384 underlying groups on64 images; q/m compute receipts are views of the same eight GPU workers, never separate costs to add together. Frozen cached checks retain original0.1 and uncached observations; they do not validate training loss.\n\n"
        "Primary reward qualified: " + str(result["candidate_qualified"]) + ". Stable RL improvement: false. See p21_q_report.md and p21_m_report.md for complete blocks, negatives and the continuous72h receipt. Raw medical data/text/IDs/labels/images/targets/readouts/checkpoints/full logs/private paths are omitted. Actual closure and effective-proxy final SHA/anonymous verification are required for delivery.\n")
    return {"teacher_rows": 128, "paired_pool_rows": 12, "paired_case_rows": 768, "check_rows": 48, "probability_check_rows": 6, "source_counts": counts}


if __name__ == "__main__": print(json.dumps(export(Path(os.environ["P10_LOCAL_FOLDER"]), Path(__file__).resolve().parents[1] / "reports")))
