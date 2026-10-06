"""Finite, predeclared overall search with whole-background inner selection."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,shutil
import numpy as np
import pandas as pd
from residual_v6_common import axes,frozen_v1,generate,dump,sha,CONTEXTS,SEED,METRICS

ROOT=Path('runs/residual_v7');OLD=Path('runs/residual_v6');MODEL=Path('models/residual_v7')
KINDS=['B','joint_mlp','tiny_transformer'];SECOND_SEED=20261005

def link(src,dst):
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():os.link(src,dst)

def clone_panel(src,dest):
    dest.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage('.').free <75*2**30:raise RuntimeError('Need 75 GiB free before new predictions')
    for f in ['real.h5ad','generation_pool.npz','genes.txt','panel.json']:link(src/f,dest/f)
    if not (dest/'references').exists():shutil.copytree(src/'references',dest/'references')
    assert json.loads((dest/'references/reference.json').read_text())['real_sha256']==sha(dest/'real.h5ad')

def reuse_score(src,dest,tag):
    if (dest/'summary.json').exists():return
    shutil.copytree(src,dest,dirs_exist_ok=True)
    rec=json.loads((dest/'summary.json').read_text());rec['model']=tag;dump(dest/'summary.json',rec)
    dump(dest/'reuse.json',{'source':str(src),'raw_metrics_unchanged':True,'references':'identical real panel and V6 anchors'})

def tag_for(outer,kind,step,scale):return f'outer_{outer}__{kind}_s{step}_a{int(scale*100):03d}'

def inner():
    from train_residual_v7 import fit,predict,predict_features,init_runtime
    init_runtime();ROOT.mkdir(parents=True,exist_ok=True)
    if not (ROOT/'protocol_lock.json').exists():dump(ROOT/'protocol_lock.json',{'sha256':sha('configs/residual_v7_protocol.json'),'locked_at':datetime.now(timezone.utc).isoformat()})
    assert json.loads((ROOT/'protocol_lock.json').read_text())['sha256']==sha('configs/residual_v7_protocol.json')
    jobs=[]
    for line in CONTEXTS:
        path=ROOT/'inner_panels'/line;old=OLD/'inner_panels'/line;clone_panel(old,path)
        link(old/'pred__fixed_v1.h5ad',path/'pred__fixed_v1.h5ad');reuse_score(old/'scores/fixed_v1',path/'scores/fixed_v1','fixed_v1')
    # Identical training sets are shared between the two swapped outer/inner roles.
    for trainline in CONTEXTS:
        left=[l for l in CONTEXTS if l!=trainline];origin=Path('models/residual_v6')/f'outer_{left[0]}__inner_{left[1]}'
        name=f'inner_train_{trainline}';fit(name,[trainline,'orion_hct116'],origin)
        neworigin=MODEL/name/f'seed_{SEED}'
        for outer,innerline in [left,left[::-1]]:
            path=ROOT/'inner_panels'/innerline;old=OLD/'inner_panels'/innerline
            inputs,base=predict_features(neworigin,path)
            for kind in KINDS:
                for step in [600,1200]:
                    oldtag=f'outer_{outer}__inner_{innerline}__B_{step}'
                    if kind=='B':
                        with np.load(old/f'effects__{oldtag}.npz') as z:r=z['residual'];gate=z['gate'];assert np.array_equal(z['effect'],base+r)
                    else:r,gate=predict(neworigin,kind,step,inputs)
                    for scale in [.5,1.]:
                        tag=tag_for(outer,kind,step,scale);jobs.append({'path':str(path),'tag':tag,'outer':outer,'inner':innerline,'kind':kind,'step':step,'scale':scale})
                        if kind=='B' and scale==1:
                            link(old/f'pred__{oldtag}.h5ad',path/f'pred__{tag}.h5ad');reuse_score(old/'scores'/oldtag,path/'scores'/tag,tag)
                        elif not (path/f'pred__{tag}.h5ad').exists():generate(path,tag,base+r*scale,SEED)
                        np.savez_compressed(path/f'effects__{tag}.npz',effect=base+r*scale,residual=r*scale,gate=gate,step=step,scale=scale)
            print('INNER_READY',outer,innerline,flush=True)
    dump(ROOT/'inner_jobs.json',jobs)

def choose(rows):
    top=max(x['overall'] for x in rows);near=[r for r in rows if r['overall']>=top-.005]
    return min(near,key=lambda x:(x['scale'],x['step']))

def select():
    path=ROOT/'selection.json'
    if path.exists():print('REUSE_FROZEN_SELECTION',flush=True);return
    allrows=[];selected={}
    jobs=json.loads((ROOT/'inner_jobs.json').read_text())
    for outer in CONTEXTS:
        innerlines=[l for l in CONTEXTS if l!=outer]
        v1=float(np.mean([json.loads((ROOT/'inner_panels'/l/'scores/fixed_v1/summary.json').read_text())['replicate_overall'] for l in innerlines]))
        selected[outer]={'V1_inner_overall':v1,'families':{}}
        for kind in KINDS:
            rows=[]
            for step in [600,1200]:
                for scale in [.5,1.]:
                    matches=[j for j in jobs if j['outer']==outer and j['kind']==kind and j['step']==step and j['scale']==scale];assert len(matches)==2
                    recs=[json.loads((Path(j['path'])/'scores'/j['tag']/'summary.json').read_text()) for j in matches]
                    rec={'outer':outer,'kind':kind,'step':step,'scale':scale,'overall':float(np.mean([r['replicate_overall'] for r in recs]))}
                    rec.update({m:float(np.mean([r[m] for r in recs])) for m in METRICS});rows.append(rec);allrows.append(rec)
            picked=choose(rows);selected[outer]['families'][kind]=picked
        competitors=[{'kind':'fixed_v1','overall':v1}]+list(selected[outer]['families'].values())
        best=max(r['overall'] for r in competitors);order=['fixed_v1']+KINDS
        selected[outer]['selected_family']=min([r for r in competitors if r['overall']>=best-.005],key=lambda r:order.index(r['kind']))['kind']
    pd.DataFrame(allrows).to_csv(ROOT/'inner_selection_candidates.csv',index=False)
    dump(path,{'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha('configs/residual_v7_protocol.json'),'primary':'replicate_overall','selection_uses_outer_scores':False,'selections':selected})
    print(json.dumps(selected,indent=2),flush=True)

def outer():
    from train_residual_v7 import fit,predict,predict_features,init_runtime
    init_runtime();selection=json.loads((ROOT/'selection.json').read_text());assert selection['protocol_sha256']==sha('configs/residual_v7_protocol.json')
    jobs=[]
    for line in CONTEXTS:
        origin=Path('models/residual_v6')/f'outer_{line}__refit';lines=[l for l in CONTEXTS if l!=line]+['orion_hct116'];name=f'outer_{line}'
        for trainseed in [SEED,SECOND_SEED]:
            fit(name,lines,origin,trainseed,KINDS if trainseed!=SEED else KINDS[1:])
            neworigin=MODEL/name/f'seed_{trainseed}';samples=[SEED,SEED+1,SEED+2] if trainseed==SEED else [SEED]
            first=ROOT/'outer_panels'/line/f'train_seed_{trainseed}'/f'seed_{SEED}';clone_panel(OLD/'outer_panels'/line/f'seed_{SEED}',first)
            inputs,base=predict_features(neworigin,first);effects={}
            for kind in KINDS:
                cfg=selection['selections'][line]['families'][kind];ckorigin=origin if kind=='B' and trainseed==SEED else neworigin
                r,s=predict(ckorigin,kind,cfg['step'],inputs);effects[kind]=(base+r*cfg['scale'],r*cfg['scale'],s)
            if trainseed!=SEED:
                r,s=predict(neworigin,'B',600,inputs);effects['incumbent_B']=(base+r,r,s)
            for sample in samples:
                old=OLD/'outer_panels'/line/f'seed_{sample}';path=ROOT/'outer_panels'/line/f'train_seed_{trainseed}'/f'seed_{sample}';clone_panel(old,path)
                link(old/'pred__fixed_v1.h5ad',path/'pred__fixed_v1.h5ad');reuse_score(old/'scores/fixed_v1',path/'scores/fixed_v1','fixed_v1')
                if trainseed==SEED:
                    link(old/'pred__B_learned.h5ad',path/'pred__incumbent_B.h5ad');reuse_score(old/'scores/B_learned',path/'scores/incumbent_B','incumbent_B')
                for kind,(effect,r,s) in effects.items():
                    tag='B_tuned' if kind=='B' else kind
                    if not (path/f'pred__{tag}.h5ad').exists():generate(path,tag,effect,sample)
                    np.savez_compressed(path/f'effects__{tag}.npz',effect=effect,residual=r,gate=s)
                    jobs.append({'path':str(path),'tag':tag,'context':line,'training_seed':trainseed,'sampling_seed':sample})
                dump(path/'selection.json',{'source_selection_sha256':sha(ROOT/'selection.json'),'selected_family':selection['selections'][line]['selected_family'],'training_seed':trainseed,'sampling_seed':sample})
            print('OUTER_READY',line,trainseed,flush=True)
    dump(ROOT/'outer_jobs.json',jobs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inner','select','outer']);a=p.parse_args();globals()[a.mode]()
