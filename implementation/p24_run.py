"""Run the original loss using the accepted differentiable generation replay."""
import json
import os
from pathlib import Path
import torch
from continual import main
from p24_contract import ARMS

if __name__ == "__main__":
    folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
    manifest = json.loads((folder / "manifest.json").read_text())
    accepted = json.loads((folder / "acceptance.json").read_text())
    if manifest.get("p24_version") != 1 or not accepted["passed"] or set(accepted["arms"]) != set(ARMS) or any(v["passed"] is not True for v in accepted["arms"].values()):
        raise ValueError("New actual training acceptance is incomplete")
    if any(cfg.get("response_recompute_path") != "differentiable_generation_replay" for group in manifest["configurations"].values() for cfg in group.values()):
        raise ValueError("Scientific matrix changed the accepted replay path")
    if manifest.get("memory_acceptance_required") and any(v.get("memory_challenge", {}).get("passed") is not True or v["memory_challenge"].get("largest_prefix", {}).get("passed") is not True for v in accepted["arms"].values()):
        raise ValueError("Larger-prefix actual memory acceptance is incomplete")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    main()
