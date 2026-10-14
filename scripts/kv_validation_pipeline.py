"""One serial paired engineering acceptance, full ledger, no training entry."""
import json
import os
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from state import atomic_json


def main():
    root=Path(os.environ['P0_ROOT']);folder=root/'outputs'/os.environ['P2_CAMPAIGN']
    plan=json.loads((folder/'paired-plan.json').read_text())
    admission=json.loads((folder/'budget_admission.json').read_text())
    budget=json.loads((root/'metadata/campaign_budget.json').read_text())
    final={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
           'code_commit':os.environ['P2_CODE_COMMIT'],'engineering_only':True,'labels_read':False,'scientific_training_started':False}
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or plan['jobs'][0]['label']!='original' or [j['label'] for j in plan['jobs']]!=['original','candidate']:
            raise ValueError('Only hidden-CUDA parent and complete serial original/candidate plan admitted')
        now=datetime.now(timezone.utc)
        if not admission['admitted'] or admission['required_with_reserve_seconds']!=46080 or (datetime.fromisoformat(budget['gpu_stop_utc'])-now).total_seconds()<46080:
            raise ValueError('Full paired budget or independent frozen clock admission failed')
        env=dict(os.environ,P2_MODE='supervise',P2_PLAN='paired-plan.json')
        with (folder/'logs/controller.log').open('x') as log:
            controller=subprocess.Popen([sys.executable,'-u',os.environ['P2_ENTRY']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(folder/'pipeline-controller.json',{'pid':controller.pid,'started_utc':now.isoformat()})
            remaining=(datetime.fromisoformat(budget['hard_deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
            try:code=controller.wait(timeout=max(1,remaining-60))
            except subprocess.TimeoutExpired:
                controller.terminate();controller.wait(timeout=45);raise RuntimeError('Independent hard deadline reached')
            if code!=0:raise RuntimeError('Paired acceptance stopped after worker failure')
        for side in ['original','candidate']:
            result=json.loads((root/'outputs'/(os.environ['P2_CAMPAIGN']+'-'+side)/'acceptance.json').read_text())
            if not result['passed'] or set(result['arms'])!={'s','t','v','h'}:raise ValueError('Incomplete actual paired acceptance')
            for arm in ['s','t','v','h']:
                r=json.loads((root/'outputs'/(os.environ['P2_CAMPAIGN']+'-'+side)/'runs'/('accept-'+arm)/'paired_engineering.json').read_text())
                if r['challenge_count']!=3 or r['optimizer_calls']!=(5 if arm in ['t','h'] else 3):raise ValueError('Incomplete paired update evidence')
        final['status']='paired_acceptance_passed_awaiting_public_delivery'
    except Exception as exc:
        final.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        ledger=folder/'controllers/paired-plan.json';used=0.
        if ledger.exists():
            records=json.loads(ledger.read_text())
            for job in records['jobs']:
                used+=job['wall_seconds'] if job.get('wall_seconds') is not None else (datetime.fromisoformat(records['finished_utc'])-datetime.fromisoformat(job['started_utc'])).total_seconds()
        cumulative=admission['prior_final_gpu_seconds']+used
        final.update(finished_utc=datetime.now(timezone.utc).isoformat(),gpu_process_seconds_used=used,
                     cumulative_gpu_process_seconds=cumulative,remaining_gpu_process_seconds=518400-cumulative,
                     whole_gpu_worker_lifetimes_counted=True,budget_reset=False,public_delivery_complete=False)
        atomic_json(folder/'FINAL.json',final)

if __name__=='__main__':main()
