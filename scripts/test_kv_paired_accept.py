"""CPU trust-boundary checks; never hardware acceptance."""
import torch
from kv_paired_accept import compare_tensors


def reject(a, b):
    try:
        compare_tensors(a, b)
    except ValueError:
        return
    raise AssertionError('Invalid paired evidence accepted')


def main():
    a = {'vision': torch.tensor([1.]), 'language': torch.tensor([2.])}
    assert compare_tensors(a, a) == 0
    reject(a, {'vision': a['vision']})
    reject(a, {**a, 'vision': torch.tensor([1., 1.])})
    reject(a, {**a, 'vision': torch.tensor([1.], dtype=torch.float64)})
    reject(a, {**a, 'vision': torch.tensor([float('nan')])})
    reject(a, {**a, 'vision': torch.tensor([1.01])})
    assert 0 < compare_tensors(a, {**a, 'vision': torch.tensor([1.000001])}) < 1e-5
    print('paired gradient coverage/schema/finite/tolerance checks passed; CPU only')


if __name__ == '__main__':
    main()


def integration():
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from types import SimpleNamespace
    from continual import Engine
    from state import PersistentAdam
    from kv_paired_accept import PairedEngine
    cfg = {'learning_rate':1e-4,'adam_betas':[.9,.999],'adam_epsilon':1e-8,
           'weight_decay':0.,'max_grad_norm':1.,'p10_arm':'t'}
    def init(engine, root, out, configuration, deadline):
        engine.out=out; engine.cfg=configuration; engine.deadline=deadline
        engine.actor=torch.nn.Linear(2,1); engine.state=PersistentAdam(engine.actor,cfg)
        engine.timings={};engine.loads=[]
        original=engine.state.load
        def load(value):
            engine.loads.append(value['optimizer_updates']);original(value)
        engine.state.load=load
    def encode(engine,row):return {'x':torch.ones(1,2)}
    def update(engine,encoded,responses,advantages,drift):
        engine.state.zero_grad();engine.actor(encoded['x']).sum().backward();return engine.state.step()
    def close(engine):pass
    with tempfile.TemporaryDirectory() as directory:
        pairs=Path(directory);deadline=SimpleNamespace(check=lambda:None)
        with patch.object(Engine,'__init__',init),patch.object(Engine,'encode',encode),patch.object(Engine,'update',update),patch.object(Engine,'close',close),patch.dict(os.environ,KV_PAIR_FOLDER=directory):
            originals=[]
            for side in ['original','candidate']:
                os.environ['KV_IMPLEMENTATION']=side
                out=pairs/side;out.mkdir();engine=PairedEngine(pairs,out,cfg,deadline)
                for i in range(3):
                    x=engine.encode({'id':str(i)});engine.encode({'id':str(i)})
                    for j in range(2):
                        before=len(engine.loads)
                        engine.update(x,[torch.tensor([1,2])],torch.ones(1),False)
                        assert len(engine.loads)==before,'Update hook replaced recovery state'
                assert engine.challenge==2 and engine.update_index==6
                if side=='original':originals=[p.clone() for p in engine.actor.parameters()]
                else:
                    assert engine.loads==[0,2,4]
                    assert all(torch.equal(a,b) for a,b in zip(originals,engine.actor.parameters()))
                    assert all(r['full_gradient_max_abs_difference']==0 for r in engine.pair_metrics)
                    try:engine.update(x,[torch.tensor([9])],torch.ones(1),False)
                    except (FileNotFoundError,ValueError):pass
                    else:raise AssertionError('Unpaired extra update accepted')
            try:engine.encode({'id':'unexpected'})
            except ValueError:pass
            else:raise AssertionError('Extra challenge accepted')
    print('paired encode/update/full-gradient/Adam-state integration passed; CPU only')

if __name__ == '__main__':
    integration()


def candidate_barrier():
    import json,os,tempfile
    from pathlib import Path
    from unittest.mock import patch
    import kv_paired_accept as module
    with tempfile.TemporaryDirectory() as directory:
        f=Path(directory)/'outputs/pair-original';f.mkdir(parents=True)
        p=f/'acceptance.json'
        with patch.dict(os.environ,P0_ROOT=directory,P2_CAMPAIGN='pair-candidate',KV_IMPLEMENTATION='candidate'),patch.object(module.p24_accept,'main') as actual,patch.object(module.p24_accept,'Engine'):
            p.write_text(json.dumps({'passed':False,'arms':{}}))
            try:module.main()
            except ValueError:pass
            else:raise AssertionError('Incomplete original admitted candidate')
            actual.assert_not_called()
            p.write_text(json.dumps({'passed':True,'arms':{a:{'passed':True,'memory_challenge':{'passed':True,'largest_prefix':{'passed':True}}} for a in ['s','t','v','h']}}))
            module.main();actual.assert_called_once()
    print('candidate rejects incomplete original acceptance before model load; CPU only')

if __name__=='__main__':candidate_barrier()
