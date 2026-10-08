"""One bounded diagnostic worker, whole-life receipt, no persistent polling."""
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    folder=Path(os.environ['P8_FOLDER'])
    plan=json.loads((folder/'diagnostic_manifest.json').read_text())
    seconds=min(plan['gpu_worker_limit_seconds'],
        (datetime.fromisoformat(plan['gpu_stop_utc'])-datetime.now(timezone.utc)).total_seconds())
    if seconds <= 60:raise RuntimeError('GPU deadline already reached')
    if (folder/'worker_receipt.json').exists() or (folder/'worker_started.json').exists():
        raise RuntimeError('P8 worker already admitted; duplicate launch forbidden')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',P8_WORKER_SECONDS=str(seconds))
    bootstrap="import os,sys;sys.path.insert(0,os.environ['P8_IMPLEMENTATION']);from p8_diagnostics import main;main()\n"
    start=time.monotonic(); utc=datetime.now(timezone.utc).isoformat()
    with (folder/'worker.log').open('w') as log:
        worker=subprocess.Popen([plan['runtime'],'-u','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                                env=env,start_new_session=True,text=True)
        ticks=Path(f'/proc/{worker.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        record={'pid':worker.pid,'start_ticks':ticks,'started_utc':utc,'worker_limit_seconds':seconds}
        (folder/'worker_started.json').write_text(json.dumps(record,indent=2)+'\n')
        worker.stdin.write(bootstrap);worker.stdin.close()
        timed_out=False
        try:worker.wait(timeout=max(1,seconds-(time.monotonic()-start)))
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(worker.pid,signal.SIGTERM)
            try:worker.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
    elapsed=time.monotonic()-start
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    ended=not Path(f'/proc/{worker.pid}').exists() and str(worker.pid) not in gpu.stdout.split()
    assert ended
    receipt={**record,'finished_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':elapsed,
             'return_code':worker.returncode,'timed_out':timed_out,'owned_worker_ended':ended,
             'gpu_process_seconds_before_p8':plan['prior_gpu_process_seconds'],
             'cumulative_gpu_process_seconds':plan['prior_gpu_process_seconds']+elapsed,
             'remaining_gpu_process_seconds':86400-plan['prior_gpu_process_seconds']-elapsed,
             'whole_worker_lifetime_counted':True,'budget_reset':False}
    (folder/'worker_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
