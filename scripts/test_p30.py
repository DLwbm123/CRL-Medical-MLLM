"""Policy-only mask equivalence and fail-closed four-arm execution barriers."""
import copy,json,os,runpy,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from core import objective,select_and_band
from continual import validate_main_budget_lock
from p24_contract import SEEDS,ARMS,configurations,admission,physical_gpu
from state import atomic_json


def main():
    root=Path(__file__).resolve().parents[1];base=json.loads((root/'configs/p2_c.json').read_text());cfg=configurations(base)
    assert SEEDS==[103] and ARMS==['s','t','v','h']
    h=copy.deepcopy(cfg['h']['103']);h.pop('policy_token_selection');h['p10_arm']='v';assert h==cfg['v']['103']
    decision=admission(232909.6123750563,66*3600);assert decision['admitted'] and decision['required_with_reserve']==223680
    assert not admission(400000,66*3600)['admitted']
    for bad in [0,1,6,7,True,'2',2.0]:
        try:physical_gpu(bad)
        except ValueError:pass
        else:raise AssertionError('Unauthorized device admitted')
    x=torch.tensor([-.2,-.3,-.4],requires_grad=True);entropy=torch.tensor([.1,1.,2.]);kl=torch.tensor([.2,.3,.4],requires_grad=True);mask=torch.tensor([False,True,True]);full=torch.ones(3,dtype=torch.bool)
    c=cfg['h']['103'];loss,part=objective(x,x.detach(),entropy,kl,torch.tensor(1.),full,entropy,entropy,3,3,c,mask)
    loss.backward(); assert x.grad[0]==0 and x.grad[1]<0 and x.grad[2]<0
    assert torch.equal(kl.grad,torch.full_like(kl,c['kl_coefficient']/3))
    old,_=objective(x,x.detach(),entropy,kl,torch.tensor(1.),full,entropy,entropy,3,3,c)
    same,_=objective(x,x.detach(),entropy,kl,torch.tensor(1.),full,entropy,entropy,3,3,c,full);assert torch.equal(old,same)
    m={'p24_version':1,'p30_version':1,'rollout_seeds':SEEDS,'configurations':cfg,'memory_acceptance_required':True};lock={'admitted':True,'complete_seeds':SEEDS,'complete_arms':ARMS}
    validate_main_budget_lock(m,{'go':True},lock)
    for changed in [dict(lock,complete_arms=['s','t','v']),dict(lock,admitted=False),dict(lock,complete_seeds=[102])]:
        try:validate_main_budget_lock(m,{'go':True},changed)
        except RuntimeError:pass
        else:raise AssertionError('Partial matrix admitted')
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'outputs/fixture';atomic_json(f/'manifest.json',m)
        acceptance={'passed':True,'arms':{a:{'passed':True,'memory_challenge':{'passed':True,'largest_prefix':{'passed':True}}} for a in ARMS}}
        with patch.dict(os.environ,{'P0_ROOT':d,'P2_CAMPAIGN':'fixture'}),patch('continual.main') as entry:
            atomic_json(f/'acceptance.json',acceptance);runpy.run_path(str(root/'implementation/p24_run.py'),run_name='__main__');entry.assert_called_once()
            acceptance['arms'].pop('h');atomic_json(f/'acceptance.json',acceptance)
            try:runpy.run_path(str(root/'implementation/p24_run.py'),run_name='__main__')
            except ValueError:pass
            else:raise AssertionError('Missing candidate accepted')
    # Closure barriers must reject the old four-worker scope before reading truth.
    from p10_score import main as score
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'outputs/fixture';raw=json.dumps(m).encode()
        import hashlib
        atomic_json(f/'manifest.json',m);(f/'manifest.sha256').write_text(hashlib.sha256(raw).hexdigest())
        atomic_json(f/'reference/targets.json',{})
        atomic_json(f/'controllers/main-plan.json',{'status':'all_planned_jobs_completed','jobs':[{'return_code':0}]*4})
        with patch.dict(os.environ,{'P0_ROOT':d,'P2_CAMPAIGN':'fixture','P2_MODE':'p24_score'}),patch('p10_score.checked_targets',return_value={}),patch('p10_score.options_by_id',return_value={}):
            def forbidden(*args):raise AssertionError('Labels opened before whole five-worker closure')
            try:score(forbidden)
            except ValueError:pass
            else:raise AssertionError('Incomplete four-arm matrix scored')
        from report_p24 import export
        atomic_json(f/'FINAL.json',{'status':'failed','code_commit':'fixture','finished_utc':'2026-10-13T00:00:00+00:00','gpu_process_seconds_used':0.,'cumulative_gpu_process_seconds':0.,'remaining_gpu_process_seconds':518400.})
        # Remove incomplete fixture ledger so missing workers remain explicit NA.
        (f/'controllers/main-plan.json').unlink()
        assert export(f,Path(d)/'reports','fixture')=={'matrix_rows':4,'case_rows':256,'probe_rows':64}
    from test_generation_replay import main as replay
    replay()
    print(json.dumps({'policy_gradient_mask_exact':True,'KL_full_response_unchanged':True,'four_arm_barrier':True,'original_point_one_and_behavior_guards_preserved':True}))

if __name__=='__main__':main()
