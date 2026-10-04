"""Copy to a neutral external path; choose the job through environment variables."""
import os
import runpy
import sys
from pathlib import Path

root = Path(os.environ["P0_ROOT"])
sys.path.insert(0, str(root / "implementation"))
task = os.environ["P1_TASK"]
jobs = {
    "run_update.py": "implementation/run_update.py",
    "test_core.py": "implementation/test_core.py",
    "download_resources.py": "scripts/download_resources.py",
    "prepare_views.py": "scripts/prepare_views.py",
}
runpy.run_path(str(root / jobs[task]), run_name="__main__")
