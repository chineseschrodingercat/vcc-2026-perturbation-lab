"""Disjoint-control, whole-context six-metric validation of v4 checkpoints."""
from pathlib import Path
from dataclasses import replace, asdict
import argparse, hashlib, json, sys, time, shutil, os
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from foundation_io import metadata,dense_rows
from prepare_state_cells import SOURCES
from generate_counts import transform_counts

# Python 3.13 scoring environment uses pandas nullable strings; opt in explicitly.
ad.settings.allow_write_nullable_strings=True

OUT=Path('runs/context_v4')
SEED=20260928
METRICS=['pds_cosine','expr_mse_unbiased_capped_norm','de_wilcoxon_direction_fidelity_yield_raw',
         'de_wilcoxon_direction_reach_raw','de_wilcoxon_sig_jaccard','de_wilcoxon_lfc_nmae']
MODELS=['v1_raw__round','shrink_only__round','residual_no_context__round','context_residual__round',
        'context_residual__posterior','learned_alpha_only__round','mean_response__round','zero__round']


def fold(t):return int(hashlib.sha256(('vcc-v4:'+str(t)).encode()).hexdigest(),16)%5


def thin(x,p,rng):
    x=sparse.csr_matrix(x,dtype=np.float32)
    if p<1:x.data=rng.binomial(x.data.astype(np.int64),p).astype(np.float32);x.eliminate_zeros()
    return x


def save(path,blocks,labels,genes):
    x=sparse.vstack(blocks,format='csr',dtype=np.float32)
    assert np.isfinite(x.data).all() and (x.data>=0).all() and np.array_equal(x.data,np.rint(x.data))
    assert float(x.sum(1).max())<1e6
    a=ad.AnnData(x,obs=pd.DataFrame({'target_gene':pd.Categorical(labels)},index=[f'c_{i}' for i in range(len(labels))]),
        var=pd.DataFrame(index=np.asarray(genes,dtype=str)))
    a.write_h5ad(path,compression='gzip',compression_opts=1)


def make_panel(line,seed):
    path=OUT/f'benchmark_{line}'/f'seed_{seed}';path.mkdir(parents=True,exist_ok=True)
    if (path/'panel.json').exists():return path,json.loads((path/'panel.json').read_text())
    if seed!=SEED:
        base=OUT/f'benchmark_{line}'/f'seed_{SEED}'
        info=json.loads((base/'panel.json').read_text())
        for filename in ['real.h5ad','generation_pool.npz','genes.txt','controls.csv']:
            shutil.copy2(base/filename,path/filename)
        info.update(seed=seed,resampling_scope='Generated NTC-cell subsampling and integer-count randomness only; fixed reference cells and generator pool from seed20260928')
        (path/'panel.json').write_text(json.dumps(info,indent=2)+'\n')
        return path,info
    rng=np.random.default_rng(seed)
    if line=='h1':
        a=ad.read_h5ad('data/raw/external/vcc2025_h1/adata_Training.h5ad',backed='r')
        labels=np.asarray(a.obs.target_gene,dtype=str);genes=np.array(a.var_names,dtype=str)
        def read(rows):
            rows=np.sort(rows);return sparse.vstack([sparse.csr_matrix(a.X[rows[i:i+128]]) for i in range(0,len(rows),128)],format='csr')
        chosen=json.loads(Path('runs/model_selection_v2/h1_single_cell_benchmark/protocol.json').read_text())['targets']
    else:
        labels,genes=metadata(SOURCES[line])
        # Preserve unambiguous native measured genes; never zero-pad unmeasured genes for evaluation.
        unambiguous=~pd.Index(genes).duplicated(keep=False);columns=np.flatnonzero(unambiguous);genes=genes[columns]
        def read(rows):return sparse.csr_matrix(dense_rows(SOURCES[line],rows,columns))
        with np.load('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz') as z:
            eligible=set(z['target_names'][z['target_n_cells']>=30])
        with np.load('data/processed/v2/external_pseudobulk/vcc2025_h1.npz') as z:h1targets=set(z['target_names'])
        vc=pd.Series(labels).value_counts()
        chosen=[t for t in vc.index if vc[t]>=128 and t in eligible and t not in h1targets and fold(t)==0]
        chosen=sorted(chosen,key=lambda t:hashlib.sha256(('vcc-v4-panel:'+t).encode()).hexdigest())[:16]
        if len(chosen)<8:raise ValueError(f'{line}: insufficient eligible targets')
    index=pd.Series(np.arange(len(labels))).groupby(labels,sort=False).indices
    ntc=index['non-targeting'].copy();rng.shuffle(ntc)
    refrows=ntc[:1000];poolrows=ntc[1000:4000]
    assert not set(refrows)&set(poolrows)
    rawpool=read(poolrows);rawref=read(refrows)
    p=min(1,20000/float(np.median(np.asarray(rawpool.sum(1)).ravel())))
    pool=thin(rawpool,p,rng);ref=thin(rawref,p,rng)
    blocks=[ref];labs=['non-targeting']*ref.shape[0];rowsdict={};sizes={};batch_records=[]
    for t in chosen:
        n=min(400,len(index[t]));rows=rng.choice(index[t],n,replace=False)
        rowsdict[t]=rows.tolist();sizes[t]=n
        blocks.append(thin(read(rows),p,rng));labs.extend([t]*n)
    save(path/'real.h5ad',blocks,labs,genes)
    sparse.save_npz(path/'generation_pool.npz',pool)
    np.savetxt(path/'genes.txt',genes,fmt='%s')
    pd.DataFrame({'row':np.r_[refrows,poolrows],'role':['reference']*len(refrows)+['generation']*len(poolrows)}).to_csv(path/'controls.csv',index=False)
    info={'context':line,'seed':seed,'targets':chosen,'target_cells':sizes,'reference_controls':len(refrows),'generation_controls':len(poolrows),
        'thinning_probability':p,'genes':len(genes),'target_selection':'fixed hash, source>=30, destination>=128, globally heldout target fold0; H1 uses old development targets',
        'gene_axis':'native measured unambiguous genes; effects on official union genes only','target_original_rows':rowsdict,
        'independent_units':'target identities and cell backgrounds; cells are subsamples within historical experiments',
        'controls_disjoint':True,'no_cross_context_outcomes_in_training':True,'models':MODELS}
    (path/'panel.json').write_text(json.dumps(info,indent=2)+'\n')
    if line=='h1':a.file.close()
    print('PANEL_READY',line,seed,len(chosen),flush=True)
    return path,info


def posterior(block,ratio,mu,theta,rng):
    x=block.toarray().astype(np.int32);down=ratio<1
    if down.any():x[:,down]=rng.binomial(x[:,down],ratio[down]).astype(np.int32)
    up=np.flatnonzero(ratio>1+1e-8)
    for start in range(0,len(up),256):
        j=up[start:start+256];scale=mu[j]/(theta[j]+mu[j])
        lam=rng.gamma(theta[j][None,:]+x[:,j],scale[None,:])
        x[:,j]+=rng.poisson(lam*(ratio[j]-1)).astype(np.int32)
    return sparse.csr_matrix(x,dtype=np.float32)


def desired_ratio(effect,model,mu,genes,modeled,core):
    gp={g:i for i,g in enumerate(genes)};pick=[(j,gp[g]) for j,g in enumerate(modeled) if g in gp]
    si=np.array([a for a,b in pick]);di=np.array([b for a,b in pick]);desired=mu.copy()
    if model=='v1_raw':desired[di]=np.maximum((mu[di]+.5)*np.exp2(effect[si])-.5,0)
    elif model!='zero':
        tot=sum(mu[gp[g]] for g in core if g in gp)
        profile=np.log1p(mu[di]*50000/tot)
        desired[di]=np.maximum(np.expm1(np.clip(profile+effect[si],0,15))*tot/50000,0)
        # Work at the destination's total depth. Closure is recorded as part of the model.
        desired*=mu.sum()/max(desired.sum(),1e-12)
    ratio=np.clip(np.divide(desired,mu,out=np.ones_like(mu),where=mu>0),0,16)
    if model=='zero':ratio[:]=1
    return ratio


def build(line,seed,only=None):
    import torch
    from train_context_v4 import load_data,predict,MODELS as MODEL_DIR
    torch.set_num_threads(4)
    path,info=make_panel(line,seed)
    needed=list(only) if only else list(info['models'])
    unknown=set(needed)-set(MODELS+['context_se_frozen__round'])
    if unknown:raise ValueError(f'Unknown candidates {unknown}')
    if any(m not in info['models'] for m in needed):
        info['models']=list(dict.fromkeys(info['models']+needed))
        (path/'panel.json').write_text(json.dumps(info,indent=2)+'\n')
    if all((path/f'pred__{m}.h5ad').exists() for m in needed):return
    data,modeled,core=load_data();genes=np.loadtxt(path/'genes.txt',dtype=str)
    pool=sparse.load_npz(path/'generation_pool.npz');mu=np.asarray(pool.mean(0)).ravel().astype(np.float64)
    variance=np.asarray(pool.power(2).mean(0)).ravel()-mu**2
    theta=np.clip(mu**2/np.maximum(variance-mu,1e-8),.05,1e5)
    foldname='final' if line=='h1' else f'holdout_{line}'
    effects,params=predict(data,modeled,core,foldname,line,info['targets'],genes,mu)
    ck=torch.load(MODEL_DIR/foldname/'context_residual.pt',map_location='cpu',weights_only=False)
    learned=float(ck['state_dict']['alpha'].clamp(0,2))
    effects['learned_alpha_only']=effects['shrink_only']*learned/float(params['alpha'])
    effectfile='predicted_effects_se.npz' if 'context_se_frozen' in effects else 'predicted_effects.npz'
    if not (path/effectfile).exists():np.savez_compressed(path/effectfile,genes=modeled,targets=info['targets'],**effects)
    rng=np.random.default_rng(seed+100);samples={t:rng.choice(pool.shape[0],info['target_cells'][t],replace=False) for t in info['targets']}
    for name in needed:
        if (path/f'pred__{name}.h5ad').exists():continue
        model,sampler=name.split('__');blocks=[];labels=[];diagnostics=[]
        for i,t in enumerate(info['targets']):
            x=pool[samples[t]];ratio=desired_ratio(effects[model][i],model,mu,genes,modeled,core)
            rr=np.random.default_rng(np.random.SeedSequence([seed,i,1 if sampler=='posterior' else 0]))
            p=posterior(x,ratio,mu,theta,rr) if sampler=='posterior' else transform_counts(x,ratio,rr)
            new=int(((p>0).astype(np.int8)-(x>0).astype(np.int8)).maximum(0).sum())
            diagnostics.append({'target':t,'new_nonzero':new,'ratio_clip_high':int((ratio>=16).sum()),
                'median_depth':float(np.median(np.asarray(p.sum(1)).ravel()))})
            blocks.append(p);labels.extend([t]*p.shape[0])
        save(path/f'pred__{name}.h5ad',blocks,labels,genes)
        pd.DataFrame(diagnostics).to_csv(path/f'generation__{name}.csv',index=False)
        print('PRED_READY',line,seed,name,flush=True)


def score(line,seed,only=None):
    from cell_eval2 import EvalConfig,compute_metrics,aggregate_metrics,__version__
    import polars as pl
    from diagnose_v4 import decomposition
    path=OUT/f'benchmark_{line}'/f'seed_{seed}';info=json.loads((path/'panel.json').read_text())
    real=ad.read_h5ad(path/'real.h5ad')
    unresolved=[t for t in info['targets'] if t not in set(real.var_names)]
    targets=[t for t in info['targets'] if t not in unresolved]
    # Readthrough/fused historical labels can lack an unambiguous own-gene mapping.
    # Exclude by identity, before comparing models; never invent a transcript alias.
    real=real[real.obs.target_gene.astype(str).isin(targets+['non-targeting'])].copy()
    evaluated_path=path/'real_resolved_targets_v2.h5ad'
    if not evaluated_path.exists():real.write_h5ad(evaluated_path,compression='gzip',compression_opts=1)
    (path/'target_identity_audit.json').write_text(json.dumps({'original_targets':info['targets'],
        'evaluated_targets':targets,'excluded_unresolved_own_transcript':unresolved,
        'reason':'Exact target label must occur on measured gene axis for unambiguous own-transcript masking; no guessed alias',
        'preliminary_unresolved_scores':'Any scores outside scores_resolved are superseded; not used for selection'},indent=2))
    control=real[real.obs.target_gene.astype(str)=='non-targeting'].copy()
    rows=[]
    for name in info['models']:
        if only and name not in only:continue
        dest=path/'scores_resolved'/name;dest.mkdir(exist_ok=True,parents=True)
        if not (dest/'aggregate.csv').exists():
            pred=ad.read_h5ad(path/f'pred__{name}.h5ad')
            pred=pred[pred.obs.target_gene.astype(str).isin(targets)].copy()
            joined=ad.concat([pred,control],index_unique='-',merge='same')
            cfg=EvalConfig.from_preset('vcc2026')
            threads=int(os.environ.get('VCC_SCORE_THREADS','4'))
            cfg=replace(cfg,pert_col='target_gene',device='cpu',num_threads=threads,gather_threads=threads,
                cache_real=str(path/'cache_real_resolved'),outdir=str(dest),de=replace(cfg.de,backend='scanpy'))
            (dest/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
            m=compute_metrics(joined,evaluated_path,config=cfg,write_de=True)
            m.write_csv(dest/'metrics.csv');aggregate_metrics(m).write_csv(dest/'aggregate.csv')
            del joined,pred
        a=pd.read_csv(dest/'aggregate.csv');row={'context':line,'seed':seed,'model':name,'n_targets':len(targets)}
        row.update({r.metric:r.mean for r in a.itertuples() if r.metric in METRICS});rows.append(row)
        modeled=np.loadtxt(OUT/'gene_axis.txt',dtype=str)
        d=decomposition(pl.read_parquet(dest/'de_real.parquet'),pl.read_parquet(dest/'de_pred.parquet'),modeled,targets)
        d.to_csv(dest/'fid_decomposition.csv',index=False)
        assert abs(d.fid_reconstructed.mean()-row[METRICS[2]])<1e-8
        print(json.dumps(row),flush=True)
    summary_path=path/'six_metrics_resolved.csv'
    current=pd.DataFrame(rows)
    if summary_path.exists():
        old=pd.read_csv(summary_path);current=pd.concat([old[~old.model.isin(current.model)],current],ignore_index=True)
    current.to_csv(summary_path,index=False)
    allrows=[]
    for p in OUT.glob('benchmark_*/seed_*/six_metrics_resolved.csv'):allrows.append(pd.read_csv(p))
    if allrows:pd.concat(allrows,ignore_index=True).to_csv(OUT/'all_six_metrics.csv',index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['build','score'],required=True)
    ap.add_argument('--contexts',nargs='+',default=['hepg2','jurkat','rpe1','h1']);ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--models',nargs='+');args=ap.parse_args()
    for line in args.contexts:
        (build if args.mode=='build' else score)(line,args.seed,args.models)


if __name__=='__main__':main()
