"""Neutral external launcher; select an owned job entirely via environment."""
import os
import runpy
import sys
from pathlib import Path

root = Path(os.environ["P0_ROOT"])
sys.path.insert(0, str(root / "implementation"))
paths = {
    "prepare": "scripts/prepare_campaign.py",
    "test": "implementation/test_continual.py",
    "run": "implementation/continual.py",
    "score": "implementation/evaluate.py",
    "supervise": "scripts/supervise.py",
    "compare": "scripts/compare_recovery.py",
}
runpy.run_path(str(root / paths[os.environ["P2_MODE"]]), run_name="__main__")
