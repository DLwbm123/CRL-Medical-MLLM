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
         "p18_prepare": "scripts/prepare_p18.py", "p18_pipeline": "scripts/p18_pipeline.py",
         "p18_numeric": "implementation/p18_numeric.py", "p18_test": "scripts/test_p18.py",
         "p19_prepare": "scripts/prepare_p18.py", "p19_pipeline": "scripts/p19_pipeline.py",
         "p19_cache": "implementation/p19_cache.py", "p19_test": "scripts/test_p19.py",
         "p20_prepare": "scripts/prepare_p20.py", "p20_pipeline": "scripts/p16_pipeline.py",
         "p20_pool": "implementation/p20_pool.py", "p20_test": "scripts/test_p20.py",
         "p21_prepare": "scripts/prepare_p21.py", "p21_pipeline": "scripts/p21_pipeline.py",
         "p21_resources": "implementation/p21_reward.py", "p21_verify": "implementation/p21_reward.py",
         "p21_pool": "implementation/p21_reward.py", "p21_score": "implementation/p21_reward.py", "p21_test": "scripts/test_p21.py",
         "p22_prepare": "scripts/prepare_p22.py", "p22_pipeline": "scripts/p22_pipeline.py",
         "p22_accept": "implementation/p22_accept.py", "p22_run": "implementation/p22_run.py",
         "p22_score": "implementation/p22_score.py", "p22_test": "scripts/test_p22.py",
         "p23_prepare": "scripts/prepare_p23.py", "p23_pipeline": "scripts/p23_pipeline.py",
         "p23_resources": "implementation/p23_reward.py", "p23_control": "implementation/p23_reward.py",
         "p23_calibration_read": "implementation/p23_verifier.py", "p23_visual": "implementation/p23_verifier.py",
         "p23_calibrate": "implementation/p23_calibration.py", "p23_pool": "implementation/p23_reward.py",
         "p23_score": "implementation/p23_score.py", "p23_test": "scripts/test_p23.py",
         "p24_prepare": "scripts/prepare_p24.py", "p24_pipeline": "scripts/p24_pipeline.py",
         "p24_accept": "implementation/p24_accept.py", "p24_run": "implementation/p24_run.py",
         "p24_score": "implementation/p24_score.py", "p24_test": "scripts/test_p24.py",
         "p29_test": "scripts/test_p28.py", "p29_prepare": "scripts/prepare_p28.py",
         "p28_test": "scripts/test_p28.py", "p28_prepare": "scripts/prepare_p28.py", "p26_test": "scripts/test_p26.py", "p27_test": "scripts/test_p27.py", "p27_prepare": "scripts/prepare_p27.py",
         "transport_test": "scripts/test_transport.py",
         "resource_test": "scripts/test_resource_closure.py",
         "report_test": "scripts/test_report_p12.py",
         "test": "implementation/test_continual.py", "verify": "implementation/p10_verifier.py", "gate": "implementation/p10_score.py",
         "legacy_test": "implementation/test_p4.py",
         "actual_accept": "implementation/p10_accept.py", "run": "implementation/continual.py", "score": "implementation/p10_score.py",
         "supervise": "scripts/supervise.py", "pipeline": "scripts/p10_pipeline.py"}
runpy.run_path(str(root / paths[os.environ["P2_MODE"]]), run_name="__main__")
