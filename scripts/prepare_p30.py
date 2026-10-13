"""Bind the delivered P29 task and qualified P23 reward to one paired mask factor."""
import copy, hashlib, json, os, shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from p24_contract import SEEDS, ARMS, TOTAL, FULL_ESTIMATE, ACCEPT_CAP, BASE_CAP, TRAIN_CAP, admission, physical_gpu
from state import atomic_json
from verified_reward import checked_targets
from prepare_campaign import write_manifest


def main():
    root=Path(os.environ['P0_ROOT']); source=Path(os.environ['P10_SOURCE_ROOT'])
    previous=source/'workspaces/p29-20261012T072138/outputs/p29-20261012T072138'
    read=lambda p:json.loads(p.read_text())
    final=read(previous/'FINAL.json'); delivery=read(root/'metadata/p29_delivery_receipt.json'); closure=read(root/'metadata/p29_completion_audit.json')
    if (final['code_commit']!='a9e11b87835108de8524d8641ac4495db973701b' or final['status']!='scored_awaiting_public_delivery'
            or delivery['code_commit']!='07c1a721577c702c62ecae387b8c92c05c9074bd'
            or not all(delivery.get(k) is True for k in ['public_delivery_complete','remote_sha_verified','anonymous_access_verified'])
            or closure.get('all_owned_sessions_ended') is not True): raise ValueError('P29 closeout/delivery differs')
    for stage,count in [('acceptance',1),('main',4)]:
        ledger=read(previous/'controllers'/f'{stage}-plan.json')
        if not ledger.get('finished_utc') or len(ledger['jobs'])!=count or any(j['return_code']!=0 for j in ledger['jobs']):raise ValueError('P29 full worker closure missing')
    continuous=read(root/'metadata/continuous_compute_receipt.json');prior=continuous['cumulative_seconds']
    if prior!=final['cumulative_gpu_process_seconds'] or continuous['total_authorized_seconds']!=TOTAL or continuous['budget_reset'] or not continuous['whole_worker_lifetime_counted']:raise ValueError('Continuous final ledger differs')
    device=physical_gpu(int(os.environ['P24_PHYSICAL_GPU'])); decision=admission(prior,66*3600)
    if not decision['admitted']:raise ValueError('Full four-arm budget must precede GPU acceptance')
    raw=(previous/'manifest.json').read_bytes(); old_digest=hashlib.sha256(raw).hexdigest()
    if old_digest!=(previous/'manifest.sha256').read_text().strip():raise ValueError('P29 frozen manifest changed')
    old=json.loads(raw); signal=read(previous/'reference/targets.json');checked_targets(signal,old,old_digest)
    gate=read(previous/'scores/qualification.json')
    from p24_contract import require_predecessor
    p23=source/'workspaces/p23-20261011T081301/outputs/p23-20261011T081301'
    require_predecessor(read(p23/'FINAL.json'),gate,read(root/'metadata/p23_delivery_receipt.json'),
        {s:read(p23/'controllers'/f'{s}-plan.json') for s in ['calibration','qualification','main']},
        {'calibration':read(p23/'calibration_owned_ended.json'),'qualification':read(p23/'qualifier_owned_ended.json'),'main':read(p23/'owned_cleanup_before_score.json')})
    folder=root/'outputs'/os.environ['P2_CAMPAIGN'];folder.mkdir(parents=True,exist_ok=False)
    for name in ['inputs','reference','scores','runs','controllers']:(folder/name).mkdir()
    manifest=copy.deepcopy(old);manifest.update(p30_version=1,campaign_prefix='p30',rollout_seeds=SEEDS,engineering_change='high_entropy_policy_mask_only',memory_acceptance_required=True)
    configs={}
    for arm in ['a']+ARMS:
        original='v' if arm=='h' else arm
        cfg=copy.deepcopy(old['configurations'][original]['42' if arm=='a' else '102'])
        cfg.update(seed=42 if arm=='a' else SEEDS[0],p10_arm=arm)
        if arm=='h':cfg['policy_token_selection']='high_entropy'
        configs[arm]={str(cfg['seed']):cfg};atomic_json(root/'configs'/f'p24_{arm}{cfg["seed"]}.json',cfg)
    manifest['configurations']=configs
    for section in ['stream','probe']:
        for e in old[section]:shutil.copyfile(previous/e['input'],folder/e['input'])
    for name in ['original_probe_predictions.json','memory_challenge.json']:shutil.copyfile(previous/'reference'/name,folder/'reference'/name)
    shutil.copyfile(previous/'memory_challenge.json',folder/'memory_challenge.json')
    digest=write_manifest(folder/'manifest.json',manifest);signal['manifest_sha256']=digest;checked_targets(signal,manifest,digest)
    atomic_json(folder/'reference/targets.json',signal);atomic_json(folder/'scores/qualification.json',gate)
    now=datetime.now(timezone.utc)
    budget={'total_gpu_process_seconds':TOTAL,'prior_gpu_process_seconds':prior,'remaining_total_gpu_process_seconds_at_start':continuous['remaining_seconds'],
        'budget_reset':False,'gpu_stop_utc':(now+timedelta(hours=66)).isoformat(),'hard_deadline_utc':(now+timedelta(hours=68)).isoformat(),
        'complete_scope_estimated_gpu_process_seconds':FULL_ESTIMATE,'reserve_fraction':.2,'main_seeds':SEEDS,'main_arms':ARMS,'authorized_physical_gpus':[device],'new_task_training_cost_measured':False}
    atomic_json(root/'metadata/campaign_budget.json',budget)
    atomic_json(folder/'acceptance.json',{'passed':False,'status':'not_started'})
    atomic_json(folder/'acceptance-plan.json',{'minimum_free_mib':48000,'jobs':[{'label':'actual-acceptance','environment':{'P2_MODE':'p24_accept'},'max_seconds':ACCEPT_CAP}]})
    jobs=[{'label':'main-a','environment':{'P2_MODE':'p24_run','P2_RUN':'main-a','P2_CONFIG':'configs/p24_a42.json','P2_PROBABILITY_CHECK':'1'},'max_seconds':BASE_CAP,'expected_cursor':64}]
    jobs += [{'label':f'main-{a}{s}','environment':{'P2_MODE':'p24_run','P2_RUN':f'main-{a}{s}','P2_CONFIG':f'configs/p24_{a}{s}.json','P2_PROBABILITY_CHECK':'1'},'max_seconds':TRAIN_CAP,'expected_cursor':64} for s in SEEDS for a in ARMS]
    atomic_json(folder/'main-plan.json',{'requires_acceptance':True,'minimum_free_mib':48000,'jobs':jobs})
    atomic_json(folder/'budget_admission.json',decision)
    atomic_json(folder/'preparation.json',{'sealed':True,'stream':64,'probe':16,'seed_count':1,'trajectory_count':4,'clinical_labels_read':False,'new_final_test_access':False,'reward_source_rebound_explicitly':True,'predecessor_public_commit':delivery['code_commit'],'budget':decision})
    print(json.dumps(decision))

if __name__=='__main__':main()
