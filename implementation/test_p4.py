"""One small check of reward changes, gate and all-seed success boundaries."""
import copy
import json
import os
import tempfile
from pathlib import Path
import torch
from continual import group_rewards, validate_development_configuration
from core import consensus_rewards
from p4_evaluate import reward_audit, gate_passes, stable_development_success
import p4_pipeline


def main():
    options = {"A":"wrong example","B":"correct example"}
    completions = ["Final answer: A"]*6 + ["Final answer: B"]*2
    old, old_adv, detail = group_rewards(completions, options)
    expected, expected_adv, _ = consensus_rewards(completions, options)
    assert torch.equal(old, expected) and torch.equal(old_adv, expected_adv)
    new, adv, vote = group_rewards(completions, options, "B")
    assert new.tolist() == [0.0]*6+[1.0]*2 and torch.all(adv[:6]<0) and torch.all(adv[6:]>0)
    assert vote["winner"]=="A" and vote["reward_target"]=="B"
    assert group_rewards(["invalid"]*8, options, "B")[:2] == (None,None)
    try:
        group_rewards(completions,options,"C")
    except ValueError:
        pass
    else:
        raise AssertionError("Illegal reward target accepted")
    rows=[{"id":"synthetic","completions":completions}]
    labels={"synthetic":"B"}; choices={"synthetic":options}
    m=reward_audit(rows,choices,labels);v=reward_audit(rows,choices,labels,labels)
    assert m["correct_negative"]["correct"]==2 and v["correct_negative"]["correct"]==0
    assert m["wrong_positive"]["correct"]==6 and v["wrong_positive"]["correct"]==0
    assert gate_passes(3,m,v) and not gate_passes(2,m,v)
    assert not gate_passes(3,m,m)
    cfg=json.loads((Path(os.environ["P0_ROOT"])/"configs/p3_d43.json").read_text())
    acceptance={"validated_configuration_without_method":{k:v for k,v in cfg.items() if k!="method"}}
    newcfg=dict(cfg,seed=45,reward_source="frozen_legal")
    manifest={"p4_version":1,"rollout_seeds":[45,46,47],"reward_sources":["majority","frozen_legal"]}
    validate_development_configuration(newcfg,acceptance,manifest)
    validate_development_configuration(dict(cfg,method="Frozen greedy"),acceptance,manifest)
    for changed in [dict(newcfg,seed=44),dict(newcfg,reward_source="labels"),dict(newcfg,learning_rate=2e-6)]:
        try:
            validate_development_configuration(changed,acceptance,manifest)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Undeclared P4 change accepted")
    good=[{"seed":s,"n":16,"frozen_before_correct":2,"control_before_correct":3,"candidate_before_correct":4,"candidate_initial_correct_retained":3} for s in [45,46,47]]
    assert stable_development_success(good) and not stable_development_success(good[:2])
    assert not stable_development_success([good[0]]*3)
    aligned=dict(cfg,seed=48,reward_source="majority",sampling_source="raw_softmax",temperature=1.0,top_p=1.0)
    p5_manifest={"p4_version":1,"p5_version":1,"rollout_seeds":[48,49,50],"reward_sources":["majority"],
                 "sampling_variants":{"original":{"temperature":0.7,"top_p":0.95},"raw_softmax":{"temperature":1.0,"top_p":1.0}}}
    validate_development_configuration(aligned,acceptance,p5_manifest)
    validate_development_configuration(dict(aligned,sampling_source="original",temperature=0.7,top_p=0.95),acceptance,p5_manifest)
    for changed in [dict(aligned,temperature=0.9),dict(aligned,sampling_source="original"),dict(aligned,reward_source="frozen_legal")]:
        try:
            validate_development_configuration(changed,acceptance,p5_manifest)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Undeclared P5 change accepted")
    shifted=[dict(s,seed=s["seed"]+3) for s in good]
    assert stable_development_success(shifted,[48,49,50]) and not stable_development_success(shifted)
    for key,value in [("candidate_before_correct",3),("candidate_initial_correct_retained",2)]:
        bad=copy.deepcopy(good);bad[0][key]=value
        assert not stable_development_success(bad)
    half=dict(cfg,seed=51,reward_source="majority",learning_rate_source="half",learning_rate=5e-7)
    p6_manifest={"p4_version":1,"p6_version":1,"rollout_seeds":[51,52,53],"reward_sources":["majority"],
                 "learning_rate_variants":{"original":1e-6,"half":5e-7}}
    validate_development_configuration(half,acceptance,p6_manifest)
    validate_development_configuration(dict(half,learning_rate_source="original",learning_rate=1e-6),acceptance,p6_manifest)
    for changed in [dict(half,learning_rate=1e-7),dict(half,seed=50),dict(half,temperature=1),
                    dict(half,reward_source="frozen_legal"),dict(half,kl_coefficient=1),
                    dict(half,learning_rate_source="original")]:
        try:
            validate_development_configuration(changed,acceptance,p6_manifest)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Undeclared P6 change accepted")
    p6_good=[dict(s,seed=s["seed"]+6) for s in good]
    assert stable_development_success(p6_good,[51,52,53]) and not stable_development_success(p6_good)
    old_folder=p4_pipeline.FOLDER
    with tempfile.TemporaryDirectory() as temporary:
        p4_pipeline.FOLDER=Path(temporary)
        (p4_pipeline.FOLDER/'controllers').mkdir()
        (p4_pipeline.FOLDER/'plan.json').write_text(json.dumps({'jobs':[{'label':'g','gpu':True},{'label':'c','gpu':False}]}))
        ledger={'plan':'plan.json','finished_utc':'2020-01-01T00:00:07+00:00',
                'jobs':[{'label':'g','wall_seconds':2,'started_utc':'2020-01-01T00:00:00+00:00'},
                        {'label':'c','wall_seconds':5,'started_utc':'2020-01-01T00:00:00+00:00'},
                        {'label':'g','started_utc':'2020-01-01T00:00:00+00:00'}]}
        (p4_pipeline.FOLDER/'controllers/plan.json').write_text(json.dumps(ledger))
        assert p4_pipeline.gpu_seconds()==9
    p4_pipeline.FOLDER=old_folder
    print(json.dumps({"passed":True,"checks":["original rewards identical","reference rewards and original vote separated","all-invalid retained","no illegal target","offline gate denominators","declared scope guard","no best-seed success"],"real_labels_read":False}))


if __name__ == "__main__":
    main()
