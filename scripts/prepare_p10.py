"""Lock the new verifier qualification and all three-arm trajectories once."""
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from state import atomic_json


def prepare(prefix="p10", seeds=(57, 58, 59), prior=66990.15299156541,
            teacher_repo="Qwen/Qwen2.5-VL-7B-Instruct", teacher_revision="cc594898137f460bfe9f0759e9844b3ce807cfb5", evidence_first=False):
    root = Path(os.environ["P0_ROOT"])
    source = Path(os.environ["P10_SOURCE_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    folder.mkdir(parents=True, exist_ok=False)
    for name in ["inputs", "reference", "scores", "runs", "controllers"]:
        (folder / name).mkdir()
    old_folder = source / "outputs/p7-20261008T044250"
    manifest = json.loads((old_folder / "manifest.json").read_text())
    for key in ["p3_version", "p4_version", "p5_version", "p6_version", "p7_version"]:
        manifest.pop(key, None)
    if prefix not in {"p10", "p11", "p12", "p13", "p14"} or len(seeds) != 3 or len(set(seeds)) != 3 or not 0 <= prior < 259200 or evidence_first != (prefix == "p14"):
        raise ValueError("Invalid complete campaign or continuous cost")
    seeds = list(seeds)
    manifest.update(p10_version=1, campaign_prefix=prefix, rollout_seeds=seeds, probe_cursors=[0, 16], drift_cursors=[16],
                    new_final_test_access=False, data_status="same previously observed retired development groups",
                    teacher={"repo": teacher_repo, "revision": teacher_revision,
                             "model_path": str(source / "models" / teacher_repo.split("/")[-1]),
                             "max_new_tokens": 1024, "option_orders": ["sorted", "reversed"], "gpu_index": 1},
                    configurations={"a": {}, "s": {}, "t": {}, "v": {}}, source_campaigns=[
                        "p5-20261007T154208", "p6-20261007T223003", "p7-20261008T044250"])
    if len(manifest["stream"]) != 16 or len(manifest["probe"]) != 16:
        raise ValueError("The authorized legacy scope must be exactly 16+16")
    if evidence_first:
        manifest["teacher"]["evidence_first"] = True
    for entry in manifest["stream"] + manifest["probe"]:
        shutil.copyfile(old_folder / entry["input"], folder / entry["input"])
    base = json.loads((source / "configs/p2_c.json").read_text())
    (root / "configs").mkdir(exist_ok=True)
    for arm in ["a", "s", "t", "v"]:
        for seed in ([42] if arm == "a" else seeds):
            cfg = copy.deepcopy(base)
            cfg.update(seed=seed, gpu_index=0, p10_arm=arm,
                       method="Frozen greedy" if arm == "a" else ("SPINE" if arm == "s" else "TTRL"),
                       reward_source="frozen_visual" if arm == "v" else "majority")
            if arm in {"t", "v"}:
                cfg.update(temperature=1., top_p=1., zero_advantage_policy="skip", strict_on_policy_consistency_claimed=False)
            manifest["configurations"][arm][str(seed)] = cfg
            atomic_json(root / "configs" / f"{prefix}_{arm}{seed}.json", cfg)
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (folder / "manifest.json").write_bytes(raw)
    (folder / "manifest.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
    now = datetime.now(timezone.utc)
    budget = {"total_gpu_process_seconds": 259200., "prior_gpu_process_seconds": prior,
              "remaining_total_gpu_process_seconds_at_start": 259200. - prior,
              "added_authorization_gpu_process_seconds": 172800., "budget_reset": False,
              "gpu_stop_utc": (now + timedelta(hours=46)).isoformat(), "hard_deadline_utc": (now + timedelta(hours=48)).isoformat(),
              "qualification_gpu_limit_seconds": 7200., "acceptance_gpu_limit_seconds": 1800.,
              "main_estimated_gpu_process_seconds": 9 * 4590.984604918864 + 1800., "reserve_fraction": .2,
              "main_seeds": seeds, "main_arms": ["s", "t", "v"], "authorized_physical_gpus": [0, 1],
              "completed_controller_archives_counted_again": False}
    atomic_json(root / "metadata/campaign_budget.json", budget)
    atomic_json(folder / "acceptance.json", {"passed": False})
    atomic_json(folder / "qualification-plan.json", {"minimum_free_mib": 24000, "jobs": [
        {"label": "frozen-verifier", "environment": {"P2_MODE": "verify"}, "max_seconds": 7200}]})
    atomic_json(folder / "acceptance-plan.json", {"minimum_free_mib": 48000, "jobs": [
        {"label": "actual-acceptance", "environment": {"P2_MODE": "actual_accept"}, "max_seconds": 1800}]})
    jobs = [{"label": "main-a", "environment": {"P2_MODE": "run", "P2_RUN": "main-a", "P2_CONFIG": f"configs/{prefix}_a42.json"},
             "max_seconds": 1800, "expected_cursor": 16}]
    jobs += [{"label": f"main-{arm}{seed}", "environment": {"P2_MODE": "run", "P2_RUN": f"main-{arm}{seed}",
              "P2_CONFIG": f"configs/{prefix}_{arm}{seed}.json"}, "max_seconds": 7200, "expected_cursor": 16}
             for seed in seeds for arm in ["s", "t", "v"]]
    atomic_json(folder / "main-plan.json", {"requires_acceptance": True, "minimum_free_mib": 48000, "jobs": jobs})
    atomic_json(folder / "preparation.json", {"prepared_utc": now.isoformat(), "stream": 16, "probe": 16,
                "new_labels_read": False, "new_final_test_access": False, "main_trajectory_count": 9,
                "matrix_requires_reward_gate": True, "budget": budget})


if __name__ == "__main__":
    prepare()
