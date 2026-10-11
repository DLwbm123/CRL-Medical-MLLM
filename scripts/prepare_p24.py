"""Bind a delivered, qualified P23 signal to the complete new RL matrix once."""
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from p23_reward import validate
from p24_contract import SEEDS, ARMS, TOTAL, FULL_ESTIMATE, RESERVE, TRAIN_CAP, BASE_CAP, ACCEPT_CAP, admission, require_predecessor, configurations, physical_gpu
from prepare_campaign import write_manifest
from state import atomic_json
from verified_reward import checked_targets


def read(path):
    return json.loads(path.read_text())


def prepare():
    root = Path(os.environ["P0_ROOT"]); source = Path(os.environ["P10_SOURCE_ROOT"])
    device = physical_gpu(int(os.environ.get("P24_PHYSICAL_GPU", "0")))
    name = os.environ["P24_PREDECESSOR"]
    if not name.startswith("p23-"): raise ValueError("Only the fixed P23 candidate can bind")
    previous = source / "workspaces" / name / "outputs" / name
    final, gate = read(previous / "FINAL.json"), read(previous / "scores/qualification.json")
    delivery = read(root / "metadata/p23_delivery_receipt.json")
    require_predecessor(final, gate, delivery,
        {s: read(previous / "controllers" / (s + "-plan.json")) for s in ["calibration", "qualification", "main"]},
        {"calibration": read(previous / "calibration_owned_ended.json"), "qualification": read(previous / "qualifier_owned_ended.json"), "main": read(previous / "owned_cleanup_before_score.json")})
    raw = (previous / "manifest.json").read_bytes(); old_digest = hashlib.sha256(raw).hexdigest()
    if old_digest != (previous / "manifest.sha256").read_text().strip(): raise ValueError("P23 manifest differs")
    old = json.loads(raw); validate(old)
    signal = read(previous / "reference/v/targets.json")
    entries = old["stream"] + old["probe"]
    checked_targets({**signal, "stream": signal["stream"] + signal["probe"]}, {**old, "stream": entries}, old_digest)
    if signal["code_commit"] != final["code_commit"] or len(entries) != 64: raise ValueError("P23 signal source differs")
    legacy = source / "workspaces/p15-20261010T095303/outputs/p15-20261010T095303"
    probes = read(legacy / "manifest.json")["probe"]
    if len(probes) != 16 or set(e["id"] for e in entries) & set(e["id"] for e in probes): raise ValueError("Full stream and original probes differ")
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]; folder.mkdir(parents=True, exist_ok=False)
    for part in ["inputs", "reference", "scores", "runs", "controllers"]: (folder / part).mkdir()
    anchor = read(source / "outputs/p7-20261008T044250/runs/main-m54/probe/c000000.json")["predictions"]
    if [p["id"] for p in anchor] != [e["id"] for e in probes]: raise ValueError("Original retention probe ordering differs")
    atomic_json(folder / "reference/original_probe_predictions.json", [{"id": p["id"], "answer": p["answer"]} for p in anchor])
    manifest = copy.deepcopy(read(legacy / "manifest.json"))
    for key in list(manifest):
        if key.endswith("_version"): manifest.pop(key)
    stream, retained = [], []
    for section, records, origin, target in [("s", entries, previous, stream), ("p", probes, legacy, retained)]:
        for i, entry in enumerate(records):
            input_path = (origin / entry["input"]).resolve()
            if not input_path.is_relative_to((origin / "inputs").resolve()): raise ValueError("Input path escapes sealed source")
            rel = f"inputs/{section}{i:03d}.json"; shutil.copyfile(input_path, folder / rel)
            target.append({**entry, "input": rel})
    manifest.update(kind="development", p10_version=1, p24_version=1, campaign_prefix="p24", stream=stream, probe=retained,
        selected_n=64, rollout_seeds=SEEDS, probe_cursors=[0, 64], drift_cursors=[64], teacher=old["teachers"]["v"], execution_physical_gpu=device,
        configurations={arm: {} for arm in ["a"] + ARMS}, new_final_test_access=False,
        data_status="all64 already observed P23 SLAKE binary development groups, original16 MedXpert retention probes",
        reward_binding={"campaign": name, "source_commit": final["code_commit"], "manifest_sha256": old_digest,
                        "primary_teacher": "v", "calibration": signal["calibration"], "readouts_reused": True, "reselection": False})
    base = read(source / "configs/p2_c.json")
    manifest["configurations"] = configurations(base)
    for arm, values in manifest["configurations"].items():
        for seed, cfg in values.items():
            atomic_json(root / "configs" / f"p24_{arm}{seed}.json", cfg)
    digest = write_manifest(folder / "manifest.json", manifest)
    bound = {**signal, "stream": signal["stream"] + signal["probe"], "probe": [], "manifest_sha256": digest,
             "source_binding": manifest["reward_binding"]}
    checked_targets(bound, manifest, digest)
    atomic_json(folder / "reference/targets.json", bound); atomic_json(folder / "scores/qualification.json", gate)
    now = datetime.now(timezone.utc)
    budget = {"total_gpu_process_seconds": TOTAL, "prior_gpu_process_seconds": final["cumulative_gpu_process_seconds"],
        "remaining_total_gpu_process_seconds_at_start": final["remaining_gpu_process_seconds"], "budget_reset": False,
        "gpu_stop_utc": (now + timedelta(hours=66)).isoformat(), "hard_deadline_utc": (now + timedelta(hours=68)).isoformat(),
        "complete_scope_estimated_gpu_process_seconds": FULL_ESTIMATE, "reserve_fraction": RESERVE,
        "main_seeds": SEEDS, "main_arms": ARMS, "authorized_physical_gpus": [0, 1], "new_task_training_cost_measured": False}
    atomic_json(root / "metadata/campaign_budget.json", budget)
    atomic_json(folder / "acceptance.json", {"passed": False, "status": "not_started"})
    atomic_json(folder / "acceptance-plan.json", {"minimum_free_mib": 48000, "jobs": [
        {"label": "actual-acceptance", "environment": {"P2_MODE": "p24_accept"}, "max_seconds": ACCEPT_CAP}]})
    jobs = [{"label": "main-a", "environment": {"P2_MODE": "p24_run", "P2_RUN": "main-a", "P2_CONFIG": "configs/p24_a42.json", "P2_PROBABILITY_CHECK": "1"}, "max_seconds": BASE_CAP, "expected_cursor": 64}]
    jobs += [{"label": f"main-{arm}{seed}", "environment": {"P2_MODE": "p24_run", "P2_RUN": f"main-{arm}{seed}", "P2_CONFIG": f"configs/p24_{arm}{seed}.json", "P2_PROBABILITY_CHECK": "1"}, "max_seconds": TRAIN_CAP, "expected_cursor": 64} for seed in SEEDS for arm in ARMS]
    atomic_json(folder / "main-plan.json", {"requires_acceptance": True, "minimum_free_mib": 48000, "jobs": jobs})
    decision = admission(budget["prior_gpu_process_seconds"], 66 * 3600)
    atomic_json(folder / "budget_admission.json", decision)
    atomic_json(folder / "preparation.json", {"sealed": True, "stream": 64, "probe": 16, "seed_count": 3, "trajectory_count": 9,
        "clinical_labels_read": False, "new_final_test_access": False, "reward_source_rebound_explicitly": True, "predecessor_public_commit": delivery["code_commit"], "budget": decision})
    print(json.dumps(decision), flush=True)


if __name__ == "__main__": prepare()
