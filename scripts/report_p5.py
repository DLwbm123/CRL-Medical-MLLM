"""Export the completed P5 aggregates without opening medical inputs or labels."""
import json
import os
from pathlib import Path
from report_p3 import write_csv, markdown_table, transition_columns


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    public = json.loads((folder / "public_summary.json").read_text())
    validation = json.loads((folder / "delivery_validation.json").read_text())
    monitor = json.loads((folder / "monitor/20261007T222437.json").read_text())
    assert public["prediction_seal"]["main_labels_read_before_seal"] is False
    assert monitor["owned_cleanup_before_score.json"]["owned_main_processes_ended"]
    assert validation["source_receipt"]["commit"] == public["evaluator_commit"]
    assert all(j["return_code"] == 0 and j["completed_case_count"] == 16 for j in monitor["jobs"])
    destination = root / "reports"
    main_rows, reward_rows, probe_rows, resource_rows, paired_rows = [], [], [], [], []
    for seed in [48, 49, 50]:
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
    assert len(statuses) == 18
    for check in public["success_checks"]:
        assert sum(p["final_correct"] for p in statuses if p["seed"] == check["seed"] and p["arm"] == "v") == check["candidate_initial_correct_retained"]
    for name, rows in [("main_results", main_rows), ("paired_transitions", paired_rows),
                       ("reward_audit", reward_rows), ("probe_results", probe_rows),
                       ("initial_correct_probes", statuses), ("resources", resource_rows)]:
        write_csv(destination / ("p5_" + name + ".csv"), rows)
    receipt = {"total_authorized_gpu_process_seconds": 86400,
               "p4_gpu_process_seconds": monitor["budget_records"][0]["seconds"],
               "p5_gpu_process_seconds": monitor["FINAL.json"]["gpu_process_seconds_used"],
               "cumulative_gpu_process_seconds": monitor["total_gpu_process_seconds_used"],
               "remaining_gpu_process_seconds": monitor["remaining_gpu_process_seconds"],
               "whole_worker_lifetime_counted": True, "budget_reset": False,
               "jobs": [{"arm": j["label"][1:], "seconds": j["wall_seconds"], "exit_code": j["return_code"]}
                        for j in monitor["jobs"]]}
    (destination / "p5_compute_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (destination / "p5_results.json").write_text(json.dumps(public, indent=2) + "\n")
    table = markdown_table(["Seed", "Arm", "Before /16", "After /16", "Parsed before/after", "Probe final /16", "Original correct retained /3"],
        [[r["seed"], r["arm"], r["before_correct"], r["after_correct"],
          str(r["before_parsed"]) + "/" + str(r["after_parsed"]),
          public["seeds"][str(r["seed"])][r["arm"]]["probe"]["final_correct"],
          public["seeds"][str(r["seed"])][r["arm"]]["probe"]["transition"]["correct_to_correct"]] for r in main_rows])
    harms = markdown_table(["Seed", "Arm", "Probe A", "Probe B", "Probe C"],
        [[seed, arm, *[("correct" if p["final_correct"] else ("parsed wrong" if p["final_parsed"] else "unparseable"))
                       for p in statuses if p["seed"] == seed and p["arm"] == arm]] for seed in [48,49,50] for arm in "mv"])
    reward = markdown_table(["Seed", "Arm", "Correct response negative advantage", "Wrong response positive reward", "Vote correct /16"],
        [[seed, arm, *[str(public["seeds"][str(seed)][arm]["reward_direction"][k]["correct"]) + "/" +
                       str(public["seeds"][str(seed)][arm]["reward_direction"][k]["total"])
                       for k in ["correct_negative", "wrong_positive", "target_accuracy"]]]
         for seed in [48,49,50] for arm in "mv"])
    text = "\n\n".join([
        "# P5 completed sampling-alignment experiment: stable criterion not met",
        "All six declared trajectories completed, without failed workers or retries. Raw-softmax sampling produced primary counts 4,3,3; paired original sampling produced 3,3,5. The frozen baseline is 2/16. Only seed48 meets all three conditions. Seed49 ties its control and loses two initially correct probes; seed50 is below its control. The preregistered all-three-seed success is false.",
        "## Scope and executed source",
        "Six independent initializations, seeds48/49/50, the same16 known stream groups and16 known probe groups, orderseed42. Control m uses temperature0.7/top-p0.95; candidate v uses temperature1/top-p1. Both retain original SPINE majority reward, full parameters, learning rate1e-6, original strict parser, greedy primary/probe generation and all other locked settings. Executed source and evaluator: " + public["evaluator_commit"] + ". See p5_protocol.md and p4_reproduce.md.",
        "## All outcomes", table,
        "Before denotes each group's greedy prediction before its update and measures historical adaptation. After denotes the current group's prediction after its update. No after-only result, changed parser or selected checkpoint substitutes for the primary. CSVs include all history/current/paired transitions and gains/harms; p5_results.json retains the complete aggregate diagnostics.",
        "## Individual retention of the same initial correct probes", harms,
        "A/B/C are anonymous ordinal aliases of the same three initially correct probe groups. They expose no original IDs, choices, text or labels. Seed49 candidate A and B become unparseable. Original-sampler seed50 loses B to a parsed wrong answer and gains a different correct group, so its net total3 does not imply full retention. Unparseability establishes strict-output failure, not a medical knowledge diagnosis.",
        "## Reward-direction audit", reward,
        "Response-level ratios count repeated sampled responses, not independent patients. Wrong majority rewards remain common in both arms. Sampling alignment does not validate their clinical correctness or establish a causal explanation of an individual loss.",
        "## Closure, numerical check and resources",
        "All six workers exited0, completed16 optimizer updates, retained probes0/16 and one final full state. Owned sessions and GPU workers ended before scoring. Configurations, primary parsing, declared rewards, frozen initial probes, source and reference integrity passed the offline checks. Predictions were sealed before the main development label read.",
        "The candidate four-token actual-model probability checks passed at BF16 tolerance0.1: maximum log-probability differences0.03994596,0.03994596,0.00915903. Production sampling matched the check configuration. This short-prefix check does not establish consistency for every sequence.",
        f'P5 consumed {receipt["p5_gpu_process_seconds"]:.6f} whole GPU-worker seconds ({receipt["p5_gpu_process_seconds"]/3600:.4f} hours). Including P4, cumulative use is {receipt["cumulative_gpu_process_seconds"]:.6f} seconds ({receipt["cumulative_gpu_process_seconds"]/3600:.4f} hours); remaining authorization is {receipt["remaining_gpu_process_seconds"]:.6f} seconds ({receipt["remaining_gpu_process_seconds"]/3600:.4f} hours). Model loading, CPU optimization and checkpoint saving are counted. p5_compute_receipt.json and p5_resources.csv record every arm; full phase timings remain in p5_results.json.',
        "## Limits and public boundary",
        "This is development on reused, previously observed small groups. Different rollout seeds do not create independent clinical samples or establish generalization. There was no held-out access, true-label training, new model/GPU or budget reset. All negative outcomes remain reported. No raw medical data, images/text/IDs/labels, checkpoints, full logs, private paths or credentials are included."
    ]) + "\n"
    (destination / "p5_report.md").write_text(text)
    print(json.dumps({"all_six_reported": True, "stable_positive": public["stable_positive_development_result"],
                      "cumulative_seconds": receipt["cumulative_gpu_process_seconds"]}))


if __name__ == "__main__":
    main()
