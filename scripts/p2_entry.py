"""Neutral external launcher; select an owned job entirely via environment."""
import os
import runpy
import sys
from pathlib import Path

root = Path(os.environ["P0_ROOT"])
sys.path.insert(0, str(root / "implementation"))
sys.path.insert(0, str(root / "scripts"))
paths = {
    "prepare": "scripts/prepare_campaign.py",
    "test": "implementation/test_continual.py",
    "run": "implementation/continual.py",
    "score": "implementation/evaluate.py",
    "supervise": "scripts/supervise.py",
    "compare": "scripts/compare_recovery.py",
    "accept": "scripts/accept_campaign.py",
    "p3_prepare": "scripts/prepare_p3.py",
    "p3_audit": "implementation/p3_diagnostics.py",
    "p3_readout": "implementation/p3_readout.py",
    "p3_score": "implementation/p3_evaluate.py",
    "p3_test": "implementation/test_p3.py",
    "p4_prepare": "scripts/prepare_p4.py",
    "p4_reference": "implementation/p4_reference.py",
    "p4_audit": "implementation/p4_evaluate.py",
    "p4_score": "implementation/p4_evaluate.py",
    "p4_test": "implementation/test_p4.py",
    "p4_pipeline": "scripts/p4_pipeline.py",
    "p5_prepare": "scripts/prepare_p4.py",
    "p5_score": "implementation/p4_evaluate.py",
    "p5_pipeline": "scripts/p4_pipeline.py",
}
runpy.run_path(str(root / paths[os.environ["P2_MODE"]]), run_name="__main__")
