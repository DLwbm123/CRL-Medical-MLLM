"""Actual-backbone raw probability check; reuse unchanged optimizer recovery evidence."""
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from continual import Deadline, Engine, load_input
from state import atomic_json


def main():
    root = Path(os.environ["P0_ROOT"]); folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    manifest = json.loads((folder / "manifest.json").read_text())
    if not json.loads((folder / "scores/qualification.json").read_text())["go"] or not (folder / "main_budget_lock.json").exists():
        raise RuntimeError("Reward and compute admission must precede actual-model acceptance")
    seed = manifest["rollout_seeds"][0]
    cfg = manifest["configurations"]["t"][str(seed)]
    torch.set_num_threads(1); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    deadline = Deadline(time.monotonic() + float(os.environ["P2_MAX_JOB_SECONDS"]))
    out = folder / "actual_acceptance"; out.mkdir(exist_ok=False)
    engine = None
    try:
        engine = Engine(root, out, cfg, deadline)
        check = engine.probability_check(load_input(folder, manifest["stream"][0]))
        if not check["raw_sampler_configuration_matches_check"] or engine.state.updates != 0:
            raise ValueError("Actual raw-sampling check or no-update acceptance differs")
        engine.reference_unchanged()
        cpu = json.loads((folder / "reward_acceptance.json").read_text())
        recovery = json.loads((folder / "cpu_recovery/result.json").read_text())
        if not cpu["passed"] or not recovery["passed"]:
            raise ValueError("CPU reward/recovery acceptance failed")
        atomic_json(folder / "acceptance.json", {"passed": True, "labels_read": False, "probability": check,
            "reward_tests": cpu, "tiny_state_recovery": recovery, "actual_new_optimizer_step_tested": False,
            "prior_actual_model_recovery_reused": True, "optimizer_and_checkpoint_code_unchanged": True})
    finally:
        if engine: engine.close()


if __name__ == "__main__":
    main()
