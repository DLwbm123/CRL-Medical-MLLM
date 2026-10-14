"""Neutral copied entry; no scientific training mode exists."""
import os
import runpy
import sys
from pathlib import Path
root=Path(os.environ['P0_ROOT'])
sys.path[:0]=[str(root/'implementation'),str(root/'scripts')]
paths={'cpu_test':'scripts/test_kv_paired_accept.py','block_test':'scripts/test_block_generation_replay.py',
       'regression_test':'scripts/test_p30.py','supervise':'scripts/supervise.py',
       'paired_accept':'implementation/kv_paired_accept.py','validation_pipeline':'scripts/kv_validation_pipeline.py'}
runpy.run_path(str(root/paths[os.environ['P2_MODE']]),run_name='__main__')
