"""Compare full real-model state and discrete outputs after independent restarts."""
import json
import os
from pathlib import Path

import torch

from state import atomic_json
from test_continual import compare_values

ROOT = Path(os.environ["P0_ROOT"])
FOLDER = ROOT / "outputs" / os.environ["P2_CAMPAIGN"]
NAMES = json.loads(os.environ.get("P2_COMPARE_RUNS", '["eng-d-full", "eng-d-split"]'))


def read_state(name):
    folder = FOLDER / "runs" / name
    latest = json.loads((folder / "checkpoints/latest.json").read_text())
    return folder, torch.load(folder / "checkpoints" / latest["directory"] / "state.pt",
                              map_location="cpu", mmap=True, weights_only=False)


def main():
    torch.set_num_threads(16)
    folder_a, a = read_state(NAMES[0])
    folder_b, b = read_state(NAMES[1])
    checked = {}
    # Predeclared before running either branch: full floating state within these
    # tolerances; RNG/counters, rollout token IDs and stored greedy/probe outputs exactly equal.
    for key in ["master_weights", "optimizer", "actor_buffers"]:
        checked[key] = compare_values(a[key], b[key], atol=1e-7, rtol=1e-6, path=key)
    for key in ["rng", "cursor", "processed_ids", "optimizer_updates", "configuration", "code_commit", "manifest_sha256", "model_revision"]:
        compare_values(a[key], b[key], path=key)
    if a["cursor"] != 2 or a["optimizer_updates"] < 1 or not a["reference_unchanged"] or not b["reference_unchanged"]:
        raise AssertionError("Real-model acceptance did not exercise updates and frozen reference")
    for index in range(a["cursor"]):
        x = json.loads((folder_a / "cases" / f"i{index:06d}.json").read_text())
        y = json.loads((folder_b / "cases" / f"i{index:06d}.json").read_text())
        for key in ["id", "before", "after", "response_token_ids", "vote", "rewards", "optimizer_updates_after", "skip_reason"]:
            compare_values(x[key], y[key], path=f"case{index}/{key}")
    for name in ["c000000.json", "c000001.json", "c000002.json"]:
        x = json.loads((folder_a / "probe" / name).read_text())
        y = json.loads((folder_b / "probe" / name).read_text())
        compare_values(x, y, path="probe/" + name)
    result = {"passed": True, "continuous_steps": 2, "restart_after": 1, "fresh_process": True,
              "floating_atol": 1e-7, "floating_rtol": 1e-6, "rng_and_outputs_exact": True,
              "full_state_comparison": checked, "reference_unchanged": True,
              "clinical_accuracy_scored": False, "runs": NAMES}
    atomic_json(FOLDER / "recovery_acceptance.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
