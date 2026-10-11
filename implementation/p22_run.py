"""Run the unchanged native loss after the new actual training acceptance."""
import json
import os
from pathlib import Path
import torch
from continual import main

if __name__ == "__main__":
    folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
    manifest = json.loads((folder / "manifest.json").read_text())
    accepted = json.loads((folder / "acceptance.json").read_text())
    if manifest.get("p22_version") != 1 or not accepted["passed"] or set(accepted["arms"]) != {"s", "t", "v"} or any(v["passed"] is not True for v in accepted["arms"].values()):
        raise ValueError("New actual training acceptance is incomplete")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    main()
