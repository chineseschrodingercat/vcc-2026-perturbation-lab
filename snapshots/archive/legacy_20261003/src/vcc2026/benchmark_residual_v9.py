"""Predeclared attention variants and bounded expert-response voting."""
from pathlib import Path
from datetime import datetime,timezone
import json,argparse
import numpy as np
import pandas as pd
from residual_v6_common import dump,sha,generate,CONTEXTS,SEED,METRICS
from benchmark_residual_v7 import clone_panel,reuse_score,link,tag_for,SECOND_SEED

ROOT=Path('runs/residual_v9');OLD=Path('runs/residual_v8');MODEL=Path('models/residual_v9');V7=Path('runs/residual_v7')
NEW=['cross_attention','attention_moe'];VOTES=['mean_vote','median_vote'];KINDS=['tiny_transformer']+NEW+VOTES
PRIORITY=['tiny_transformer','mean_vote','median_vote','cross_attention','attention_moe']
PROTOCOL=Path('configs/residual_v9_protocol.json')

def configurations(kind):
    if kind in VOTES:return [{'kind':kind,'step':0,'beta':b,'scale':a} for b in [.5,1.] for a in [.5,1.]]
    return [{'kind':kind,'step':s,'beta':0.,'scale':a} for s in [600,1200] for a in [.5,1.]]

def tag_for_cfg(outer,c):
    if c['kind'] in VOTES:return f"outer_{outer}__{c['kind']}_b{int(c['beta']*100):03d}_a{int(c['scale']*100):03d}"
    return tag_for(outer,c['kind'],c['step'],c['scale'])

def choose(rows):
    best=max(r['overall'] for r in rows)
    return min([r for r in rows if r['overall']>=best-.005],key=lambda r:(r['scale'],r['step'] if r['step'] else r['beta']))

def vote(experts,c):
    rr=np.stack(experts,axis=0);v=np.mean(rr,axis=0) if c['kind']=='mean_vote' else np.median(rr,axis=0)
    return np.asarray(c['scale']*((1-c['beta'])*rr[0]+c['beta']*v),dtype=np.float32)

def setup():
    manifest=Path('reports/delivery_manifest_residual_v8.json');assert manifest.exists(),'Preserve and verify V8 before V9 fitting'
    if (ROOT/'protocol_lock.json').exists():assert json.loads((ROOT/'protocol_lock.json').read_text())['sha256']==sha(PROTOCOL);return
    d=json.loads((OLD/'decision.json').read_text());s=json.loads((OLD/'selection.json').read_text());prior=json.loads((OLD/'protocol_lock.json').read_text())
    mapping={line:s['selections'][line]['selected_family'] if d['advance_preselected_strategy'] else prior['incumbent_per_context'][line] for line in CONTEXTS}
    dump(ROOT/'protocol_lock.json',{'locked_at':datetime.now(timezone.utc).isoformat(),'sha256':sha(PROTOCOL),'V8_manifest_sha256':sha(manifest),'incumbent_per_context':mapping,'V8_advanced':d['advance_preselected_strategy']})

def inner_experts(outer,innerline):
    s=json.loads((OLD/'selection.json').read_text());rr=[];sources=[]
    for kind in ['tiny_transformer','medium_transformer','soft_moe']:
        cfg=s['selections'][outer]['families'][kind];p=OLD/'inner_panels'/innerline/f"effects__{tag_for(outer,kind,cfg['step'],cfg['scale'])}.npz"
        with np.load(p) as z:rr.append(z['residual'].copy())
        sources.append({'kind':kind,'path':str(p),'sha256':sha(p),'setting':cfg})
    return rr,sources

def outer_experts(line,ts,ss):
    rr=[];sources=[]
    for kind in ['tiny_transformer','medium_transformer','soft_moe']:
        root=V7 if kind=='tiny_transformer' else OLD;p=root/'outer_panels'/line/f'train_seed_{ts}'/f'seed_{ss}'/f'effects__{kind}.npz'
        with np.load(p) as z:rr.append(z['residual'].copy())
        sources.append({'kind':kind,'path':str(p),'sha256':sha(p)})
    return rr,sources

def inner():
    from train_residual_v9 import fit,predict,predict_features,init_runtime
    setup();init_runtime();jobs=[]
    for line in CONTEXTS:
        old=OLD/'inner_panels'/line;path=ROOT/'inner_panels'/line;clone_panel(old,path)
        link(old/'pred__fixed_v1.h5ad',path/'pred__fixed_v1.h5ad');reuse_score(old/'scores/fixed_v1',path/'scores/fixed_v1','fixed_v1')
    for trainline in CONTEXTS:
        left=[l for l in CONTEXTS if l!=trainline];name=f'inner_train_{trainline}';origin=Path('models/residual_v7')/name/f'seed_{SEED}'
        fit(name,[trainline,'orion_hct116'],origin,SEED,NEW);neworigin=MODEL/name/f'seed_{SEED}'
        for outer,innerline in [left,left[::-1]]:
            old=OLD/'inner_panels'/innerline;path=ROOT/'inner_panels'/innerline;inputs,base=predict_features(neworigin,path);experts,sources=inner_experts(outer,innerline)
            dump(path/f'vote_sources__outer_{outer}.json',sources);cache={}
            for kind in KINDS:
                for cfg in configurations(kind):
                    tag=tag_for_cfg(outer,cfg);jobs.append({'path':str(path),'tag':tag,'outer':outer,'inner':innerline,**cfg})
                    if kind=='tiny_transformer':
                        link(old/f'pred__{tag}.h5ad',path/f'pred__{tag}.h5ad');reuse_score(old/'scores'/tag,path/'scores'/tag,tag);link(old/f'effects__{tag}.npz',path/f'effects__{tag}.npz');continue
                    if kind in VOTES:r=vote(experts,cfg);gate=np.full((len(r),1),np.nan,dtype=np.float32)
                    else:
                        key=(kind,cfg['step'])
                        if key not in cache:cache[key]=predict(neworigin,kind,cfg['step'],inputs)
                        raw,gate=cache[key];r=raw*cfg['scale']
                    if not (path/f'pred__{tag}.h5ad').exists():generate(path,tag,base+r,SEED)
                    np.savez_compressed(path/f'effects__{tag}.npz',effect=base+r,residual=r,gate=gate,step=cfg['step'],scale=cfg['scale'],beta=cfg['beta'])
            print('V9_INNER_READY',outer,innerline,flush=True)
    dump(ROOT/'inner_jobs.json',jobs)

def select():
    if (ROOT/'selection.json').exists():return
    jobs=json.loads((ROOT/'inner_jobs.json').read_text());selected={};allrows=[]
    for outer in CONTEXTS:
        selected[outer]={'families':{}}
        for kind in KINDS:
            rows=[]
            for cfg in configurations(kind):
                jj=[j for j in jobs if j['outer']==outer and all(j[k]==v for k,v in cfg.items())];assert len(jj)==2
                recs=[json.loads((Path(j['path'])/'scores'/j['tag']/'summary.json').read_text()) for j in jj]
                r={'outer':outer,**cfg,'overall':float(np.mean([x['replicate_overall'] for x in recs]))};r.update({m:float(np.mean([x[m] for x in recs])) for m in METRICS});rows.append(r);allrows.append(r)
            selected[outer]['families'][kind]=choose(rows)
        choices=list(selected[outer]['families'].values());best=max(r['overall'] for r in choices);selected[outer]['selected_family']=min([r for r in choices if r['overall']>=best-.005],key=lambda r:PRIORITY.index(r['kind']))['kind']
    a=pd.DataFrame(allrows);a.to_csv(ROOT/'inner_selection_candidates.csv',index=False);g=a.groupby(['kind','step','scale','beta'],as_index=False).overall.mean();choices=[choose(g[g.kind==k].to_dict('records')) for k in KINDS];best=max(r['overall'] for r in choices);final=min([r for r in choices if r['overall']>=best-.005],key=lambda r:PRIORITY.index(r['kind']))
    expert_settings={}
    olda=pd.read_csv(OLD/'inner_selection_candidates.csv');oldg=olda.groupby(['kind','step','scale'],as_index=False).overall.mean()
    for kind in ['tiny_transformer','medium_transformer','soft_moe']:
        from benchmark_residual_v7 import choose as previous_choose
        expert_settings[kind]=previous_choose(oldg[oldg.kind==kind].to_dict('records'))
    dump(ROOT/'selection.json',{'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(PROTOCOL),'selection_uses_outer_scores':False,'selections':selected,'conditional_final_candidate':final,'conditional_final_experts':expert_settings})
    print(json.dumps(selected,indent=2),flush=True)

def outer():
    from train_residual_v9 import fit,predict,predict_features,init_runtime
    init_runtime();sel=json.loads((ROOT/'selection.json').read_text());prev=json.loads((OLD/'selection.json').read_text());lock=json.loads((ROOT/'protocol_lock.json').read_text());jobs=[]
    for line in CONTEXTS:
        name=f'outer_{line}';lines=[l for l in CONTEXTS if l!=line]+['orion_hct116'];tiny=sel['selections'][line]['families']['tiny_transformer'];oldtiny=prev['selections'][line]['families']['tiny_transformer'];assert (tiny['step'],tiny['scale'])==(oldtiny['step'],oldtiny['scale'])
        for ts in [SEED,SECOND_SEED]:
            origin=Path('models/residual_v7')/name/f'seed_{ts}';fit(name,lines,origin,ts,NEW);neworigin=MODEL/name/f'seed_{ts}'
            oldfirst=OLD/'outer_panels'/line/f'train_seed_{ts}'/f'seed_{SEED}';first=ROOT/'outer_panels'/line/f'train_seed_{ts}'/f'seed_{SEED}';clone_panel(oldfirst,first);inputs,base=predict_features(neworigin,first);effects={}
            for kind in NEW:
                cfg=sel['selections'][line]['families'][kind];r,gate=predict(neworigin,kind,cfg['step'],inputs);effects[kind]=(r*cfg['scale'],gate)
            for ss in ([SEED,SEED+1,SEED+2] if ts==SEED else [SEED]):
                old=OLD/'outer_panels'/line/f'train_seed_{ts}'/f'seed_{ss}';path=ROOT/'outer_panels'/line/f'train_seed_{ts}'/f'seed_{ss}';clone_panel(old,path)
                for tag in ['fixed_v1','incumbent_B','B_tuned','joint_mlp','tiny_transformer','medium_transformer','soft_moe']:
                    link(old/f'pred__{tag}.h5ad',path/f'pred__{tag}.h5ad');reuse_score(old/'scores'/tag,path/'scores'/tag,tag)
                inc=lock['incumbent_per_context'][line];link(old/f'pred__{inc}.h5ad',path/'pred__incumbent.h5ad');reuse_score(old/'scores'/inc,path/'scores/incumbent','incumbent')
                experts,sources=outer_experts(line,ts,ss);dump(path/'vote_sources.json',sources)
                for kind in NEW+VOTES:
                    cfg=sel['selections'][line]['families'][kind]
                    if kind in VOTES:r=vote(experts,cfg);gate=np.full((len(r),1),np.nan,dtype=np.float32)
                    else:r,gate=effects[kind]
                    if not (path/f'pred__{kind}.h5ad').exists():generate(path,kind,base+r,ss)
                    np.savez_compressed(path/f'effects__{kind}.npz',effect=base+r,residual=r,gate=gate,step=cfg['step'],scale=cfg['scale'],beta=cfg['beta'])
                    jobs.append({'path':str(path),'tag':kind,'context':line,'training_seed':ts,'sampling_seed':ss})
                dump(path/'selection.json',{'source_selection_sha256':sha(ROOT/'selection.json'),'selected_family':sel['selections'][line]['selected_family'],'incumbent':inc,'training_seed':ts,'sampling_seed':ss})
            print('V9_OUTER_READY',line,ts,flush=True)
    dump(ROOT/'outer_jobs.json',jobs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inner','select','outer']);a=p.parse_args();globals()[a.mode]()
