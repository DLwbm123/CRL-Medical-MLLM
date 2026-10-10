"""Complete reward evidence plus original and cached numerical guard observations."""
import json
import os
from pathlib import Path
from report_p16 import export as reward_export
from report_p10 import write_csv


def export(folder, output):
    counts = reward_export(folder, output, "p20", (87, 88, 89)); checks = []
    for seed in [87, 88, 89]:
        for arm in ["s", "t"]:
            path = folder / "runs" / f"pool-{arm}{seed}" / "probability_check.json"
            value = json.loads(path.read_text()) if path.exists() else None
            checks.append({"seed": seed, "sampler": arm, "cached_max_logp_difference": value["max_logp_difference"] if value else "NA",
                "uncached_max_logp_difference": value["paired_uncached_control"]["max_logp_difference"] if value else "NA",
                "cached_within_original_tolerance": value["within_tolerance"] if value else "NA", "absolute_tolerance": .1,
                "same_sampled_tokens": value["same_sampled_tokens"] if value else "NA", "training_loss_validated": False})
    write_csv(output / "p20_probability_checks.csv", checks)
    path = output / "p20_results.json"; results = json.loads(path.read_text()); results.update(
        frozen_pool_probability_recompute_path="incremental_cache", probability_checks=checks, training_loss_validated=False)
    path.write_text(json.dumps(results, indent=2) + "\n")
    path = output / "p20_report.md"; text = path.read_text()
    text += "\nP20 reuses the exact64 P17 admitted binary training inputs; no annotation selection or new data access occurred. Its sole engineering change is the frozen-pool probability check using P19's same-token incremental cached path. The original0.1 threshold and uncached control observations are retained. The normal Engine probability check and RL loss were not modified; this frozen-pool path is not admitted for training. P19's cached192/192 versus uncached172/192 was numerical evidence only. Teacher, prompt, parsers, six reward conditions and both sampling distributions are unchanged, with fresh128 teacher readouts and six complete SC8 pools. All24 block checks must pass; this still does not establish RL improvement or independent clinical generalization.\n"
    path.write_text(text); return {**counts, "probability_check_rows": 6}


if __name__ == "__main__": print(json.dumps(export(Path(os.environ["P10_LOCAL_FOLDER"]), Path(__file__).resolve().parents[1] / "reports")))
