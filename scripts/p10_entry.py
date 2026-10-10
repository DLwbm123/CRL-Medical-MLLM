"""Copy outside the project; all task and path selection is environment-only."""
import os
import runpy
import sys
from pathlib import Path
root = Path(os.environ["P0_ROOT"])
sys.path[:0] = [str(root / "implementation"), str(root / "scripts")]
paths = {"prepare": "scripts/prepare_p10.py", "download": "scripts/p10_download.py", "reward_test": "implementation/test_verified_reward.py",
         "prepare_medical": "scripts/prepare_p11.py",
         "prepare_recovery": "scripts/prepare_p12.py",
         "prepare_transport": "scripts/prepare_p13.py", "download_transport": "scripts/p13_download.py",
         "prepare_evidence": "scripts/prepare_p14.py",
         "prepare_suffix": "scripts/prepare_p15.py",
         "p16_prepare": "scripts/prepare_p16.py", "p16_pool": "implementation/p16_pool.py",
         "p16_score": "implementation/p16_score.py", "p16_pipeline": "scripts/p16_pipeline.py",
         "p16_test": "scripts/test_p16.py",
         "p17_prepare": "scripts/prepare_p16.py", "p17_pipeline": "scripts/p16_pipeline.py",
         "transport_test": "scripts/test_transport.py",
         "resource_test": "scripts/test_resource_closure.py",
         "report_test": "scripts/test_report_p12.py",
         "test": "implementation/test_continual.py", "verify": "implementation/p10_verifier.py", "gate": "implementation/p10_score.py",
         "legacy_test": "implementation/test_p4.py",
         "actual_accept": "implementation/p10_accept.py", "run": "implementation/continual.py", "score": "implementation/p10_score.py",
         "supervise": "scripts/supervise.py", "pipeline": "scripts/p10_pipeline.py"}
runpy.run_path(str(root / paths[os.environ["P2_MODE"]]), run_name="__main__")
