"""Score sealed read-only diagnostics in a separate CPU-only process."""
import hashlib
import json
import os
from pathlib import Path
from evaluate import is_correct
from p3_diagnostics import read_labels
from state import atomic_json


def main():
    root, folder, source = (Path(os.environ[k]) for k in ['P8_ROOT','P8_FOLDER','P8_P7_FOLDER'])
    receipt=json.loads((folder/'worker_receipt.json').read_text())
    assert receipt['owned_worker_ended'] and receipt['return_code']==0
    directory=folder/'diagnostics'
    completion=json.loads((directory/'completion.json').read_text())
    seal=json.loads((directory/'readout_seal.json').read_text())
    assert completion['all_c_completed'] and not completion['labels_read'] and completion['optimizer_steps']==0
    assert seal['actors']==['base','m54','v54','m55','v55','m56','v56'] and seal['probes_per_actor']==16
    plan=json.loads((folder/'diagnostic_manifest.json').read_text())
    raw=(source/'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==plan['p7_manifest_sha256']
    manifest=json.loads(raw)
    readouts={name:json.loads((directory/(name+'_readout.json')).read_text())['predictions'] for name in seal['actors']}
    identifiers=[e['id'] for e in manifest['probe']]
    assert all([p['id'] for p in rows]==identifiers for rows in readouts.values())
    labels=read_labels(root,manifest)
    initial=json.loads((source/'runs/main-a/probe/c000000.json').read_text())['predictions']
    strict_initial={p['id'] for p in initial if is_correct(p['answer'],labels[p['id']])}
    legal_initial={p['id'] for p in readouts['base'] if is_correct(p['answer'],labels[p['id']])}
    rows, kl_rows, summary, gradient_rows=[],[],[],[]
    for name in seal['actors']:
        strict=initial if name=='base' else json.loads((source/'runs'/('main-'+name)/'probe/c000016.json').read_text())['predictions']
        assert [p['id'] for p in strict]==identifiers
        for prediction, free in zip(readouts[name],strict):
            identifier=prediction['id'];scores=prediction['scores'];target=labels[identifier]
            assert target in scores
            correct=is_correct(prediction['answer'],target)
            rows.append({'actor':name,'group':prediction['group'],'strict_parsed':free['answer'] is not None,
                         'strict_correct':is_correct(free['answer'],target),'legal_correct':correct,
                         'initial_strict_correct':identifier in strict_initial,'initial_legal_correct':identifier in legal_initial,
                         'legal_top1_top2_gap':prediction['top1_top2_gap'],
                         'correct_minus_best_wrong_gap':scores[target]-max(v for k,v in scores.items() if k!=target),
                         'option_identifier_lengths':json.dumps(sorted(len(ids) for ids in prediction['option_token_ids'].values()))})
            kl_rows.append({'actor':name,'group':prediction['group'],'kl_actor_base':prediction['kl_actor_base'],
                            'base_top_token_retained':prediction['base_top_token_retained'],
                            'legal_top1_top2_gap':prediction['top1_top2_gap'],
                            'correct_minus_best_wrong_gap':rows[-1]['correct_minus_best_wrong_gap']})
        selected=[r for r in rows if r['actor']==name]
        summary.append({'actor':name,'n':16,'strict_correct':sum(r['strict_correct'] for r in selected),
                        'strict_parsed':sum(r['strict_parsed'] for r in selected),
                        'strict_initial_n':len(strict_initial),
                        'strict_initial_retained':sum(r['strict_correct'] and r['initial_strict_correct'] for r in selected),
                        'legal_correct':sum(r['legal_correct'] for r in selected),'legal_initial_n':len(legal_initial),
                        'legal_initial_retained':sum(r['legal_correct'] and r['initial_legal_correct'] for r in selected),
                        'strict_fail_legal_correct':sum(not r['strict_correct'] and r['legal_correct'] for r in selected),
                        'strict_initial_loss_legal_correct':sum(r['initial_strict_correct'] and not r['strict_correct'] and r['legal_correct'] for r in selected),
                        'mean_kl':sum(r['kl_actor_base'] for r in kl_rows if r['actor']==name)/16})
        path=directory/(name+'_gradients.json')
        if path.exists():
            gradient_rows.extend({'actor':name,**r} for r in json.loads(path.read_text())['rows'])
    offline={p:json.loads((folder/'offline'/(p+'_public.json')).read_text()) for p in ['P5','P6','P7']}
    atomic_json(folder/'public_diagnostics.json',{'readout':rows,'prefix_kl':kl_rows,'readout_summary':summary,
                'gradient_rows':gradient_rows,'gradient_groups':completion['gradient_groups'],
                'gradient_complete':completion['all_d_completed'],'completion':completion,'receipt':receipt,
                'offline':offline,'p9_unused_seeds':plan['p9_unused_seeds'],'used_seeds':plan['used_seeds'],
                'output_sealed_before_label_read':True,'scoring_process_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})
    print(json.dumps({'scored_actors':7,'all_probes':112,'gradient_rows':len(gradient_rows),'labels_in_gpu_process':False}),flush=True)


if __name__=='__main__':main()
