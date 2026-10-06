"""SLIM-style two-sided ridge on a frozen K562 unseen-target split.

Uses the published gene-basis/PCA and solve_W equations, adapted to the
normalized pseudobulk count representation. This is not a paper reproduction.
Hyperparameters use only non-test development targets. All 300 competition
targets and the prior matched comparison set are excluded from all fitting.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import h5py
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from select_transfer_model import read_source,profile,project,Pair,evaluate,sha256


def basis(y,k):
    centered=y-y.mean(axis=1,keepdims=True)
    u,s,_=randomized_svd(centered.T.astype(np.float32),n_components=k,n_iter=5,random_state=20260927)
    return u*s


def fit_candidates(y,p):
    items={}
    for bias_name in ['control','train_mean']:
        b=np.zeros(y.shape[1]) if bias_name=='control' else y.mean(axis=0)
        yc=y-b
        g=basis(yc,32).astype(np.float64)
        for k in [10,32]:
            gg=g[:,:k]
            for lam in [.1,10.,100.]:
                z=np.linalg.solve(gg.T@gg+lam*np.eye(k),gg.T@yc.T)
                w=np.linalg.solve(p.T@p+lam*np.eye(p.shape[1]),(z@p).T).T
                key=f'bilinear_{bias_name}_k{k}_l{lam:g}'
                items[key]={'g':gg.astype(np.float32),'w':w.astype(np.float32),'b':b.astype(np.float32)}
    return items


def predict(model,p,control):
    y=(p@model['w'].T)@model['g'].T+model['b']
    return project(y,control)[0]


def main():
    out=Path('runs/model_selection_v2/bilinear_unseen_targets')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source=read_source('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz')
    splitpath=Path('runs/baseline_v1/unseen_target_validation/split_manifest.csv')
    split=pd.read_csv(splitpath).set_index('target_gene')
    official=set(np.loadtxt('data/raw/vcc2026/validation/files/gene_names.csv',dtype=str,skiprows=1))
    idx=np.array([i for i,g in enumerate(source['gene_names']) if g in official])
    genes=source['gene_names'][idx];control=source['control_mean_counts'][idx]
    targets=source['target_names'];n=source['target_n_cells']
    y=profile(source['target_mean_counts'][:,idx])-profile(control)
    p=np.zeros((len(targets),64),dtype=np.float32);found=np.zeros(len(targets),dtype=bool)
    with h5py.File('data/raw/external/slim_embeddings/gene_string_embeddings.v0.3.h5') as f:
        for i,t in enumerate(targets):
            if t in f:p[i]=f[t][()];found[i]=True
    original_split=np.array([str(split.loc[t,'split']) if t in split.index else 'excluded' for t in targets])
    development=(original_split=='train') & found & (n>=30)
    fold=np.array([int(hashlib.sha256(('slim-v2:'+t).encode()).hexdigest(),16)%5 for t in targets])
    train=development & (fold!=0);val=development & (fold==0)
    official_test=(original_split=='official_panel_test') & found & (n>=30)
    # Match by the audited split names; abort instead of silently scoring no targets.
    if not official_test.any():
        raise ValueError(f'Unexpected split names: {set(original_split)}')
    matched=(original_split=='matched_nonpanel_test') & found & (n>=30)
    assert not set(targets[development]) & set(targets[official_test|matched])
    (out/'protocol.json').write_text(json.dumps({'frozen_prior_split_sha256':sha256(splitpath),
        'normalization':'bulk log1p with 50000 target sum on 7679 measured official genes',
        'selection':'minimum MSE on hash-held-out nonpanel development targets only',
        'equation':'W=(G.T G+lambda I)^-1 G.T Y P (P.T P+lambda I)^-1',
        'embedding_dimensions':64,'ranks':[10,32],'lambda':[.1,10,100],
        'bias':['control','training mean'],'created_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    def pair(mask):
        return Pair('k562',targets[mask],genes,np.zeros_like(y[mask]),y[mask],np.zeros_like(y[mask]),
                    np.zeros_like(y[mask]),control,control,n[mask],n[mask])
    print('fit inner',train.sum(),'validation',val.sum(),flush=True)
    models=fit_candidates(y[train],p[train].astype(np.float64));rows=[]
    for key,m in models.items():
        sc,_=evaluate(predict(m,p[val],control),pair(val),pds=False);rows.append({'model':key,**sc})
    zero,_=evaluate(np.zeros_like(y[val]),pair(val),pds=False);rows.append({'model':'zero',**zero})
    valid=pd.DataFrame(rows).sort_values('mse');valid.to_csv(out/'inner_validation.csv',index=False)
    chosen=str(valid.iloc[0]['model']); print('preselected',chosen,flush=True)
    models=fit_candidates(y[development],p[development].astype(np.float64));result=[]
    for group,mask in [('official_panel',official_test),('matched_nonpanel',matched)]:
        for key in [chosen,'zero'] if chosen!='zero' else ['zero',str(valid[valid.model!='zero'].iloc[0]['model'])]:
            pred=np.zeros_like(y[mask]) if key=='zero' else predict(models[key],p[mask],control)
            sc,df=evaluate(pred,pair(mask));result.append({'group':group,'model':key,'preselected':key==chosen,**sc})
            df.to_csv(out/f'per_target__{group}__{key}.csv',index=False)
    pd.DataFrame(result).to_csv(out/'test_results.csv',index=False)
    (out/'selection.json').write_text(json.dumps({'selected_model':chosen,'train_targets':int(development.sum()),
        'official_test':int(official_test.sum()),'matched_test':int(matched.sum()),
        'fit_used_test_targets':False,'models_not_full_paper_reproduction':True},indent=2)+'\n')
    print(pd.DataFrame(result).to_string(index=False),flush=True)


if __name__=='__main__':main()
