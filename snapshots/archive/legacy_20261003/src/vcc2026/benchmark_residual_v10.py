"""Fixed-grid low-LR continuation, weight averaging and bounded response bias."""
from pathlib import Path
from datetime import datetime, timezone
import json, argparse
import numpy as np
import pandas as pd
from residual_v6_common import dump, sha, generate, CONTEXTS, SEED, METRICS
from benchmark_residual_v7 import clone_panel, reuse_score, link, tag_for, choose, SECOND_SEED
from benchmark_residual_v9 import tag_for_cfg

ROOT=Path('runs/residual_v10'); OLD=Path('runs/residual_v9'); MODEL=Path('models/residual_v10')
PROTOCOL=Path('configs/residual_v10_protocol.json')
NEW=['continued','ema','swa','bounded_bias']
KINDS=['incumbent','tiny_transformer']+NEW


def configurations(kind):
    if kind=='incumbent':return [{'kind':kind,'step':0,'scale':1.}]
    return [{'kind':kind,'step':s,'scale':a} for s in [600,1200] for a in [.5,1.]]


def setup():
    manifest=Path('reports/delivery_manifest_residual_v9.json')
    assert manifest.exists(),'Finish V9 before V10 fitting'
    if (ROOT/'protocol_lock.json').exists():
        lock=json.loads((ROOT/'protocol_lock.json').read_text())
        assert lock['sha256']==sha(PROTOCOL) and lock['V9_manifest_sha256']==sha(manifest)
        return
    decision=json.loads((OLD/'decision.json').read_text())
    sel=json.loads((OLD/'selection.json').read_text())
    prior=json.loads((OLD/'protocol_lock.json').read_text())
    mapping={line:sel['selections'][line]['selected_family'] if decision['advance_preselected_strategy'] else prior['incumbent_per_context'][line] for line in CONTEXTS}
    tiny=json.loads(Path('runs/residual_v7/selection.json').read_text())
    dump(ROOT/'protocol_lock.json',{'locked_at':datetime.now(timezone.utc).isoformat(),
        'sha256':sha(PROTOCOL),'V9_manifest_sha256':sha(manifest),'incumbent_per_context':mapping,
        'V9_advanced':decision['advance_preselected_strategy'],
        'warm_start_per_context':{line:tiny['selections'][line]['families']['tiny_transformer'] for line in CONTEXTS},
        'full_warm_start_step':600})


def inner():
    from train_residual_v10 import fit,predict,predict_features,init_runtime
    setup();init_runtime()
    lock=json.loads((ROOT/'protocol_lock.json').read_text())
    prior=json.loads((OLD/'selection.json').read_text())
    jobs=[]
    for line in CONTEXTS:
        old=OLD/'inner_panels'/line;path=ROOT/'inner_panels'/line;clone_panel(old,path)
        link(old/'pred__fixed_v1.h5ad',path/'pred__fixed_v1.h5ad')
        reuse_score(old/'scores/fixed_v1',path/'scores/fixed_v1','fixed_v1')
    for trainline in CONTEXTS:
        left=[line for line in CONTEXTS if line!=trainline]
        origin=Path('models/residual_v7')/f'inner_train_{trainline}'/f'seed_{SEED}'
        for outer,innerline in [left,left[::-1]]:
            start=lock['warm_start_per_context'][outer]['step'];name=f'inner_train_{trainline}_start_{start}'
            fit(name,[trainline,'orion_hct116'],origin,start,SEED)
            neworigin=MODEL/name/f'seed_{SEED}';old=OLD/'inner_panels'/innerline;path=ROOT/'inner_panels'/innerline
            inputs,base=predict_features(neworigin,path);cache={}
            for kind in KINDS:
                for cfg in configurations(kind):
                    tag=tag_for(outer,kind,cfg['step'],cfg['scale'])
                    jobs.append({'path':str(path),'tag':tag,'outer':outer,'inner':innerline,'warm_start_step':start,**cfg})
                    if kind in ['tiny_transformer','incumbent']:
                        src=tag if kind=='tiny_transformer' else tag_for_cfg(outer,prior['selections'][outer]['families'][lock['incumbent_per_context'][outer]])
                        link(old/f'pred__{src}.h5ad',path/f'pred__{tag}.h5ad')
                        reuse_score(old/'scores'/src,path/'scores'/tag,tag)
                        link(old/f'effects__{src}.npz',path/f'effects__{tag}.npz')
                        continue
                    key=(kind,cfg['step'])
                    if key not in cache:cache[key]=predict(neworigin,kind,cfg['step'],inputs)
                    raw,gate=cache[key];r=raw*cfg['scale']
                    if not (path/f'pred__{tag}.h5ad').exists():generate(path,tag,base+r,SEED)
                    np.savez_compressed(path/f'effects__{tag}.npz',effect=base+r,residual=r,gate=gate,
                        extra_steps=cfg['step'],scale=cfg['scale'],warm_start_step=start)
            print('V10_INNER_READY',outer,innerline,flush=True)
    dump(ROOT/'inner_jobs.json',jobs)


def select():
    if (ROOT/'selection.json').exists():return
    jobs=json.loads((ROOT/'inner_jobs.json').read_text());selected={};allrows=[]
    for outer in CONTEXTS:
        selected[outer]={'families':{}}
        for kind in KINDS:
            rows=[]
            for cfg in configurations(kind):
                jj=[j for j in jobs if j['outer']==outer and all(j[k]==v for k,v in cfg.items())]
                assert len(jj)==2
                recs=[json.loads((Path(j['path'])/'scores'/j['tag']/'summary.json').read_text()) for j in jj]
                r={'outer':outer,**cfg,'overall':float(np.mean([x['replicate_overall'] for x in recs]))}
                r.update({m:float(np.mean([x[m] for x in recs])) for m in METRICS})
                rows.append(r);allrows.append(r)
            selected[outer]['families'][kind]=choose(rows)
        choices=list(selected[outer]['families'].values());best=max(r['overall'] for r in choices)
        selected[outer]['selected_family']=min([r for r in choices if r['overall']>=best-.005],key=lambda r:KINDS.index(r['kind']))['kind']
    a=pd.DataFrame(allrows);a.to_csv(ROOT/'inner_selection_candidates.csv',index=False)
    global_rows=a.groupby(['kind','step','scale'],as_index=False).overall.mean()
    choices=[choose(global_rows[global_rows.kind==kind].to_dict('records')) for kind in KINDS]
    best=max(r['overall'] for r in choices)
    final=min([r for r in choices if r['overall']>=best-.005],key=lambda r:KINDS.index(r['kind']))
    dump(ROOT/'selection.json',{'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(PROTOCOL),
        'selection_uses_outer_scores':False,'selections':selected,'conditional_final_candidate':final})
    print(json.dumps(selected,indent=2),flush=True)


def outer():
    from train_residual_v10 import fit,predict,predict_features,init_runtime
    init_runtime();sel=json.loads((ROOT/'selection.json').read_text());lock=json.loads((ROOT/'protocol_lock.json').read_text());jobs=[]
    for line in CONTEXTS:
        name=f'outer_{line}';lines=[l for l in CONTEXTS if l!=line]+['orion_hct116'];start=lock['warm_start_per_context'][line]['step']
        tiny=sel['selections'][line]['families']['tiny_transformer'];assert (tiny['step'],tiny['scale'])==(start,1.)
        for ts in [SEED,SECOND_SEED]:
            origin=Path('models/residual_v7')/name/f'seed_{ts}';fit(name,lines,origin,start,ts)
            neworigin=MODEL/name/f'seed_{ts}';oldfirst=OLD/'outer_panels'/line/f'train_seed_{ts}'/f'seed_{SEED}'
            first=ROOT/'outer_panels'/line/f'train_seed_{ts}'/f'seed_{SEED}';clone_panel(oldfirst,first)
            inputs,base=predict_features(neworigin,first);effects={}
            for kind in NEW:
                cfg=sel['selections'][line]['families'][kind];r,gate=predict(neworigin,kind,cfg['step'],inputs)
                effects[kind]=(r*cfg['scale'],gate)
            for ss in ([SEED,SEED+1,SEED+2] if ts==SEED else [SEED]):
                old=OLD/'outer_panels'/line/f'train_seed_{ts}'/f'seed_{ss}';path=ROOT/'outer_panels'/line/f'train_seed_{ts}'/f'seed_{ss}';clone_panel(old,path)
                for tag in ['fixed_v1','incumbent_B','tiny_transformer']:
                    link(old/f'pred__{tag}.h5ad',path/f'pred__{tag}.h5ad');reuse_score(old/'scores'/tag,path/'scores'/tag,tag)
                inc=lock['incumbent_per_context'][line]
                link(old/f'pred__{inc}.h5ad',path/'pred__incumbent.h5ad');reuse_score(old/'scores'/inc,path/'scores/incumbent','incumbent')
                for kind in NEW:
                    cfg=sel['selections'][line]['families'][kind];r,gate=effects[kind]
                    if not (path/f'pred__{kind}.h5ad').exists():generate(path,kind,base+r,ss)
                    np.savez_compressed(path/f'effects__{kind}.npz',effect=base+r,residual=r,gate=gate,
                        extra_steps=cfg['step'],scale=cfg['scale'],warm_start_step=start)
                    jobs.append({'path':str(path),'tag':kind,'context':line,'training_seed':ts,'sampling_seed':ss})
                dump(path/'selection.json',{'source_selection_sha256':sha(ROOT/'selection.json'),
                    'selected_family':sel['selections'][line]['selected_family'],'incumbent':inc,
                    'training_seed':ts,'sampling_seed':ss,'warm_start_step':start})
            print('V10_OUTER_READY',line,ts,flush=True)
    dump(ROOT/'outer_jobs.json',jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inner','select','outer']);a=p.parse_args();globals()[a.mode]()
