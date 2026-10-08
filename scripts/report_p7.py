"""Export the completed P7 aggregates without opening medical inputs or labels."""
import json
import os
from pathlib import Path
from report_p3 import write_csv, markdown_table, transition_columns


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    public = json.loads((folder / "public_summary.json").read_text())
    validation = json.loads((folder / "delivery_validation.json").read_text())
    monitor = json.loads((sorted((folder / "monitor").glob("*.json"))[-1]).read_text())
    assert public["prediction_seal"]["main_labels_read_before_seal"] is False
    assert monitor["owned_cleanup_before_score.json"]["owned_main_processes_ended"]
    assert validation["source_receipt"]["commit"] == public["evaluator_commit"]
    assert all(j["return_code"] == 0 and j["completed_case_count"] == 16 for j in monitor["jobs"])
    destination = root / "reports"
    main_rows, reward_rows, probe_rows, resource_rows, paired_rows = [], [], [], [], []
    for seed in [54, 55, 56]:
        context = public["seeds"][str(seed)]
        paired_rows.append({"seed": seed, **context["paired_candidate_vs_control"]})
        for arm in "mv":
            value = context[arm]
            probe = value["probe"]
            row = {"seed": seed, "arm": arm, "n": value["n"],
                   "before_correct": value["greedy_before"]["correct"],
                   "after_correct": value["greedy_after"]["correct"],
                   "before_parsed": value["before_parse_rate"]["correct"],
                   "after_parsed": value["after_parse_rate"]["correct"],
                   "consensus_correct": value["consensus"]["correct"]}
            row.update(transition_columns("history_", value["history_parse_transition"]))
            row.update(transition_columns("current_", value["current_parse_transition"]))
            assert 0 <= row["before_correct"] <= row["before_parsed"] <= 16
            assert 0 <= row["after_correct"] <= row["after_parsed"] <= 16
            main_rows.append(row)
            probe_rows.append({"seed": seed, "arm": arm, **{k: v for k, v in probe.items() if k != "transition"},
                               **probe["transition"]})
            for metric in ["correct_negative", "wrong_positive", "target_accuracy"]:
                ratio = value["reward_direction"][metric]
                assert 0 <= ratio["correct"] <= ratio["total"]
                reward_rows.append({"seed": seed, "arm": arm, "metric": metric,
                                    "numerator": ratio["correct"], "denominator": ratio["total"], "percent": ratio["percent"]})
            resource_rows.append({"seed": seed, "arm": arm,
                                  **{k: v for k, v in public["resources"][arm + str(seed)].items() if k != "phase_seconds"}})
    statuses = validation["initial_correct_probe_statuses"]
    assert len(statuses) == len({(p["seed"], p["arm"], p["initial_correct_probe"]) for p in statuses}) == 18
    assert all({p["initial_correct_probe"] for p in statuses if p["seed"] == seed and p["arm"] == arm} == {"A", "B", "C"} for seed in [54,55,56] for arm in "mv")
    for check in public["success_checks"]:
        assert sum(p["final_correct"] for p in statuses if p["seed"] == check["seed"] and p["arm"] == "v") == check["candidate_initial_correct_retained"]
    for name, rows in [("main_results", main_rows), ("paired_transitions", paired_rows),
                       ("reward_audit", reward_rows), ("probe_results", probe_rows),
                       ("initial_correct_probes", statuses), ("resources", resource_rows)]:
        write_csv(destination / ("p7_" + name + ".csv"), rows)
    receipt = {"total_authorized_gpu_process_seconds": 86400,
               "p4_gpu_process_seconds": sum(r["seconds"] for r in monitor["budget_records"] if r["campaign"].startswith("p4-")),
               "p5_gpu_process_seconds": sum(r["seconds"] for r in monitor["budget_records"] if r["campaign"].startswith("p5-")),
               "p6_gpu_process_seconds": sum(r["seconds"] for r in monitor["budget_records"] if r["campaign"].startswith("p6-")),
               "p7_gpu_process_seconds": monitor["FINAL.json"]["gpu_process_seconds_used"],
               "cumulative_gpu_process_seconds": monitor["total_gpu_process_seconds_used"],
               "remaining_gpu_process_seconds": monitor["remaining_gpu_process_seconds"],
               "whole_worker_lifetime_counted": True, "budget_reset": False,
               "jobs": [{"arm": j["label"][1:], "seconds": j["wall_seconds"], "exit_code": j["return_code"]}
                        for j in monitor["jobs"]]}
    assert abs(sum(j["seconds"] for j in receipt["jobs"]) - receipt["p7_gpu_process_seconds"]) < 1e-6
    assert abs(sum(receipt[f"p{n}_gpu_process_seconds"] for n in [4,5,6,7]) - receipt["cumulative_gpu_process_seconds"]) < 1e-6
    assert abs(receipt["cumulative_gpu_process_seconds"] + receipt["remaining_gpu_process_seconds"] - 86400) < 1e-6
    (destination / "p7_compute_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (destination / "p7_results.json").write_text(json.dumps(public, indent=2) + "\n")
    table = markdown_table(["Seed", "Arm", "Before /16", "After /16", "Parsed before/after", "Probe final /16", "Original correct retained /3"],
        [[r["seed"], r["arm"], r["before_correct"], r["after_correct"],
          str(r["before_parsed"]) + "/" + str(r["after_parsed"]),
          public["seeds"][str(r["seed"])][r["arm"]]["probe"]["final_correct"],
          public["seeds"][str(r["seed"])][r["arm"]]["probe"]["transition"]["correct_to_correct"]] for r in main_rows])
    harms = markdown_table(["Seed", "Arm", "Probe A", "Probe B", "Probe C"],
        [[seed, arm, *[("correct" if p["final_correct"] else ("parsed wrong" if p["final_parsed"] else "unparseable"))
                       for p in statuses if p["seed"] == seed and p["arm"] == arm]] for seed in [54,55,56] for arm in "mv"])
    reward = markdown_table(["Seed", "Arm", "Correct response negative advantage", "Wrong response positive reward", "Vote correct /16"],
        [[seed, arm, *[str(public["seeds"][str(seed)][arm]["reward_direction"][k]["correct"]) + "/" +
                       str(public["seeds"][str(seed)][arm]["reward_direction"][k]["total"])
                       for k in ["correct_negative", "wrong_positive", "target_accuracy"]]]
         for seed in [54,55,56] for arm in "mv"])
    text = "\n\n".join([
        "# P7 completed reference-anchor experiment: stable criterion not met",
        "All six declared trajectories completed, without failed workers or retries. The stronger fixed-base KL coefficient1.0 produced primary counts2,3,5; paired original coefficient0.01 produced3,3,5. The frozen baseline is2/16. No candidate seed meets all three conditions. Seed54 equals the frozen baseline and is below control; seeds55/56 equal their controls, and seed55 also loses the original correct probe B. The preregistered all-three-seed success is false.",
        "## Scope and executed source",
        "Six independent initializations, seeds54/55/56, the same16 known stream groups and16 known probe groups, orderseed42. Control m uses fixed-base KL coefficient0.01; candidate v uses1.0. This coefficient is the sole change. Both retain original SPINE majority reward, learning rate1e-6, temperature0.7/top-p0.95, full parameters, entropy mask/band, original strict parser, greedy primary/probe generation and all other locked settings. Executed source and evaluator: " + public["evaluator_commit"] + ". See p7_protocol.md and p4_reproduce.md.",
        "## All outcomes", table,
        "Before denotes each group's greedy prediction before its update and measures historical adaptation. After denotes the current group's prediction after its update. No after-only result, changed parser or selected checkpoint substitutes for the primary. CSVs include all history/current/paired transitions and gains/harms; p7_results.json retains the complete aggregate diagnostics.",
        "## Individual retention of the same initial correct probes", harms,
        "A/B/C are anonymous ordinal aliases of the same three initially correct probe groups. They expose no original IDs, choices, text or labels. Candidate seed55 B becomes unparseable. Its final total remains3/16 because one other probe becomes correct: total accuracy alone hides the lost original correct probe. The other five trajectories preserve all three original correct probes. Every original correct probe outcome is included above and in the CSV. Unparseability establishes strict-output failure, not a medical knowledge diagnosis.",
        "## Reward-direction audit", reward,
        "Response-level ratios count repeated sampled responses, not independent patients. Wrong majority rewards remain common in both arms. A stronger anchor does not validate clinical reward correctness or establish a causal explanation of an individual loss.",
        "## Anchor and optimization diagnostics",
        "Mean exact full-vocabulary KL to the fixed base, measured before updates on each arm's own sampled prefixes, is0.00174151 versus0.00086736 for seed54,0.00089177 versus0.00090169 for seed55, and0.00306600 versus0.00095890 for seed56 (control versus candidate). Candidate KL is lower for two seeds, but these prefixes differ between arms and there is no uniform primary or retention improvement. This is a descriptive diagnostic, not a causal effect estimate. Scalar loss magnitudes do not measure component gradient influence. Full loss components, gradient norms, token fractions and post-update prefix diagnostics remain in p7_results.json.",
        "## Closure, numerical check and resources",
        "All six workers exited0, completed16 optimizer updates, retained probes0/16 and one final full state. Owned sessions and GPU workers ended before scoring. Configurations, primary parsing, declared rewards, frozen initial probes, source and reference integrity passed the offline checks. Predictions were sealed before the main development label read. Before launch, an actual synthetic KL objective and gradient check verified the coefficient scaling; it did not predict clinical efficacy.",
        "Both arms retain the original transformed sampler with raw-softmax policy loss. Its documented sampling/loss mismatch remains; raw-sampler equality checks are inapplicable here and no strict on-policy claim is made. Exact behavior recomputation and fixed-reference checks passed.",
        f'P7 consumed {receipt["p7_gpu_process_seconds"]:.6f} whole GPU-worker seconds ({receipt["p7_gpu_process_seconds"]/3600:.4f} hours). Including P4, P5 and P6, cumulative use is {receipt["cumulative_gpu_process_seconds"]:.6f} seconds ({receipt["cumulative_gpu_process_seconds"]/3600:.4f} hours); remaining authorization is {receipt["remaining_gpu_process_seconds"]:.6f} seconds ({receipt["remaining_gpu_process_seconds"]/3600:.4f} hours). Model loading, CPU optimization and checkpoint saving are counted. p7_compute_receipt.json and p7_resources.csv record every arm; full phase timings remain in p7_results.json. Whole-device sampled memory may include other jobs and is distinct from this worker allocated/reserved peaks.',
        "## Next-round budget admission",
        f'The remaining {receipt["remaining_gpu_process_seconds"]/3600:.4f} hours cannot cover another full six-trajectory matrix at the measured P7 cost of {receipt["p7_gpu_process_seconds"]/3600:.4f} hours, even without reserve. With the existing20% reserve that measured cost requires {receipt["p7_gpu_process_seconds"]*1.2/3600:.4f} hours. The previous P6 matrix also consumed {receipt["p6_gpu_process_seconds"]/3600:.4f} hours before reserve. No next GPU round is launched. Authorization is not exhausted or reset; unused time remains. A cheaper implementation has not been measured, and an assumed speedup cannot establish admission. See p7_next_admission.json for the exact decision. Stable positive performance remains unestablished.',
        "## Limits and public boundary",
        "This is development on reused, previously observed small groups. Different rollout seeds do not create independent clinical samples or establish generalization. There was no held-out access, true-label training, new model/GPU or budget reset. All negative outcomes remain reported. No raw medical data, images/text/IDs/labels, checkpoints, full logs, private paths or credentials are included."
    ]) + "\n"
    (destination / "p7_report.md").write_text(text)
    print(json.dumps({"all_six_reported": True, "stable_positive": public["stable_positive_development_result"],
                      "cumulative_seconds": receipt["cumulative_gpu_process_seconds"]}))


if __name__ == "__main__":
    main()
