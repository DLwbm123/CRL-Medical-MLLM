"""Fixed complete RL scope and fail-closed admission; no model or labels loaded."""
import math

SEEDS = [102]
ARMS = ["s", "t", "v"]
TOTAL = 518400.
TRAIN_CAP = 40000.
BASE_CAP = 7200.
ACCEPT_CAP = 14400.
TRAJECTORIES = len(SEEDS) * len(ARMS)
TRAIN_PROXY = TRAJECTORIES * 4 * 4590.984604918864
FULL_ESTIMATE = max(TRAIN_PROXY + BASE_CAP + ACCEPT_CAP, TRAJECTORIES * TRAIN_CAP + BASE_CAP + ACCEPT_CAP)
RESERVE = .2
CHECKS = {"coverage_at_least_half", "at_least_twelve_correct_targets", "accepted_precision_at_least_three_quarters",
          "correct_negative_rate_lower", "wrong_positive_rate_ten_pp_lower", "correct_minority_rescued"}


def physical_gpu(value):
    if type(value) is not int or value != 2:
        raise ValueError("Only physical GPU2 is authorized")
    return value


def configurations(base):
    result = {arm: {} for arm in ["a"] + ARMS}
    for arm in result:
        for seed in ([42] if arm == "a" else SEEDS):
            cfg = {**base, "seed": seed, "gpu_index": 0, "p10_arm": arm,
                   "response_recompute_path": "differentiable_generation_replay",
                   "method": "Frozen greedy" if arm == "a" else "SPINE" if arm == "s" else "TTRL",
                   "reward_source": "frozen_visual" if arm == "v" else "majority"}
            if arm in ["t", "v"]: cfg.update(temperature=1., top_p=1., zero_advantage_policy="skip", strict_on_policy_consistency_claimed=False)
            result[arm][str(seed)] = cfg
    return result


def admission(prior, round_seconds, used=0., acceptance_complete=False):
    if not all(math.isfinite(x) and x >= 0 for x in [prior, round_seconds, used]) or prior + used > TOTAL:
        raise ValueError("Invalid continuous full-worker cost")
    estimate = FULL_ESTIMATE - (ACCEPT_CAP if acceptance_complete else 0)
    remaining = TOTAL - prior - used
    return {"admitted": min(remaining, round_seconds) >= estimate * (1 + RESERVE),
            "remaining_gpu_seconds": remaining, "round_remaining_seconds": round_seconds,
            "estimated_seconds": estimate, "required_with_reserve": estimate * (1 + RESERVE),
            "additional_seconds_required": max(0., estimate * (1 + RESERVE) - remaining),
            "complete_seeds": SEEDS, "complete_arms": ARMS, "budget_reset": False,
            "new_task_training_cost_measured": False, "basis": "max historical training proxy and full fixed stop caps; not frozen-generation speed"}


def require_predecessor(final, gate, delivery, ledgers, closures):
    if final.get("status") != "reward_qualified" or gate.get("go") is not True or gate.get("primary_teacher") != "v" or gate.get("primary_selected_after_scoring") is not False:
        raise ValueError("Only the predeclared v reward can qualify")
    candidate = gate.get("teachers", {}).get("v", {})
    blocks = candidate.get("blocks", {})
    if candidate.get("go") is not True or set(blocks) != {"calibration_s", "calibration_t", "verification_s", "verification_t"} or any(set(v.get("checks", {})) != CHECKS or any(x is not True for x in v["checks"].values()) for v in blocks.values()):
        raise ValueError("All24 original candidate reward conditions must pass")
    if any(delivery.get(k) is not True for k in ["public_delivery_complete", "remote_sha_verified", "anonymous_access_verified"]):
        raise ValueError("P23 public delivery is incomplete")
    for stage, count in [("calibration", 1), ("qualification", 2), ("main", 6)]:
        ledger = ledgers[stage]
        if ledger.get("status") != "all_planned_jobs_completed" or len(ledger.get("jobs", [])) != count or any(j.get("return_code") != 0 or j.get("timeout_or_stop") for j in ledger["jobs"]) or closures[stage].get("owned_main_processes_ended") is not True:
            raise ValueError("Entire P23 scope must seal and every session must end")
    if final.get("whole_gpu_worker_lifetimes_counted") is not True or final.get("budget_reset") is not False or final["cumulative_gpu_process_seconds"] < 120691.1088064673 or abs(final["remaining_gpu_process_seconds"] - (TOTAL - final["cumulative_gpu_process_seconds"])) > 1e-5:
        raise ValueError("Continuous P23 budget receipt differs")
