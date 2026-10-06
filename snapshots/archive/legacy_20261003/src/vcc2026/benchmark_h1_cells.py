"""Run the official six-metric implementation on an external H1 pilot panel.

No leaderboard anchors are available: the outputs are raw local metrics, never
a VCC2026 scaled competition score. The panel is selected by a fixed hash,
without looking at prediction accuracy or perturbation effect strength.
"""
from __future__ import annotations
import argparse
from dataclasses import replace, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from cell_eval2 import EvalConfig, compute_metrics, aggregate_metrics, __version__ as eval_version
from generate_counts import transform_counts
from select_transfer_model import project, read_source, sha256

SEED=20260927
BASE=Path('runs/model_selection_v2/h1_confirmation')


def posterior_transport(block, multiplier, mu, theta, rng):
    """Gamma-Poisson augmentation for upregulation, exact thinning for decreases.

    The baseline latent rate posterior Gamma(theta + x, rate=theta/mu + 1)
    lets previously zero cells acquire counts. It uses NTC moments only.
    """
    x=block.toarray().astype(np.int32)
    down=multiplier < 1
    if down.any():
        x[:,down]=rng.binomial(x[:,down],multiplier[down]).astype(np.int32)
    up=np.flatnonzero(multiplier > 1+1e-8)
    for start in range(0,len(up),256):
        j=up[start:start+256]
        rate_scale=mu[j]/(theta[j]+mu[j])
        lam=rng.gamma(theta[j][None,:]+x[:,j],rate_scale[None,:])
        x[:,j]+=rng.poisson(lam*(multiplier[j]-1)).astype(np.int32)
    return sparse.csr_matrix(x,dtype=np.float32)


def read_counts(a, rows):
    rows=np.sort(np.asarray(rows,dtype=np.int64))
    blocks=[sparse.csr_matrix(a.X[rows[i:i+256]]) for i in range(0,len(rows),256)]
    return sparse.vstack(blocks,format='csr',dtype=np.float32)


def thin(x,p,rng):
    x=x.copy()
    if p < 1:
        x.data=rng.binomial(x.data.astype(np.int64),p).astype(np.float32)
        x.eliminate_zeros()
    return x


def save_adata(path, blocks, labels, genes):
    a=ad.AnnData(sparse.vstack(blocks,format='csr'),
                obs=pd.DataFrame({'target_gene':pd.Categorical(labels)},index=[f'cell_{i}' for i in range(len(labels))]),
                var=pd.DataFrame(index=genes))
    if not np.isfinite(a.X.data).all() or np.any(a.X.data<0) or np.any(a.X.data != np.rint(a.X.data)):
        raise ValueError('Invalid raw output')
    a.write_h5ad(path,compression='gzip',compression_opts=1)
    return {'shape':list(a.shape),'nnz':int(a.X.nnz),'max_library_size':float(a.X.sum(axis=1).max()),'bytes':path.stat().st_size}


def build(out,panel_size):
    if (out/'manifest.json').exists():
        return json.loads((out/'manifest.json').read_text())
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Partial benchmark directory; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    pred_paths={m:BASE/f'prediction__{m}.npz' for m in ['zero','v1_raw_scalar','norm_scalar','hybrid_64']}
    pred={}
    for m,p in pred_paths.items():
        with np.load(p) as z:pred[m]={k:np.asarray(z[k]) for k in z.files}
    h1=read_source('data/processed/v2/external_pseudobulk/vcc2025_h1.npz')
    candidates=set(h1['target_names'][h1['target_n_cells']>=400])&set(pred['zero']['target_symbols'])
    candidates=sorted(candidates,key=lambda t:hashlib.sha256(('vcc-h1-cells:'+t).encode()).hexdigest())
    targets=candidates[:panel_size]
    protocol={'targets':targets,'target_selection':'First N SHA256(vcc-h1-cells:target) among H1 and K562 >=400/30 cells; no outcome/error-based selection',
              'real_cells_per_target':400,'generated_cells_per_target':400,'reference_ntc_cells':2000,
              'generation_ntc_pool':8000,'seed':SEED,'gene_axis':'all 18080 H1 genes; model changes only 6122 common genes',
              'unmodeled_gene_rule':'unchanged NTC counts','models':['zero__round','v1_raw_scalar__round','norm_scalar__round','hybrid_64__round','hybrid_64__posterior'],
              'ntc_separation':'Reference 2000 and generator 8000 are disjoint sampled original cells',
              'depth_rule':'One binomial thinning probability applied to real and control counts, making sampled NTC median depth about 20000; no inflation',
              'created_utc':datetime.now(timezone.utc).isoformat()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    rng=np.random.default_rng(SEED)
    a=ad.read_h5ad('data/raw/external/vcc2025_h1/adata_Training.h5ad',backed='r')
    genes=np.asarray(a.var_names,dtype=str); labels=np.asarray(a.obs.target_gene,dtype=str)
    ntc_rows=np.flatnonzero(labels=='non-targeting');rng.shuffle(ntc_rows)
    ref_rows=ntc_rows[:2000];pool_rows=ntc_rows[2000:10000]
    assert not set(ref_rows)&set(pool_rows)
    ref_raw=read_counts(a,ref_rows);pool_raw=read_counts(a,pool_rows)
    p=min(1,20000/float(np.median(np.asarray(pool_raw.sum(axis=1)).ravel())))
    ref=thin(ref_raw,p,rng);pool=thin(pool_raw,p,rng)
    mu=np.asarray(pool.mean(axis=0)).ravel().astype(np.float64)
    variance=np.asarray(pool.power(2).mean(axis=0)).ravel()-mu**2
    theta=np.clip(np.divide(mu**2,np.maximum(variance-mu,1e-8)),.05,1e5)
    realblocks=[ref];reallabels=['non-targeting']*len(ref_rows)
    target_rows={}
    for t in targets:
        rows=rng.choice(np.flatnonzero(labels==t),400,replace=False)
        target_rows[t]=rows.tolist()
        realblocks.append(thin(read_counts(a,rows),p,rng));reallabels += [t]*400
    real_info=save_adata(out/'real.h5ad',realblocks,reallabels,genes)
    del realblocks,ref_raw,pool_raw
    gp={g:i for i,g in enumerate(genes)}
    modeled=np.array([gp[g] for g in pred['zero']['gene_symbols']])
    index={t:i for i,t in enumerate(pred['zero']['target_symbols'])}
    # Shared random NTC samples make model comparisons less sensitive to sampling.
    sampled={t:rng.choice(len(pool_rows),400,replace=False) for t in targets}
    files={}
    for variant in protocol['models']:
        model,sampler=variant.split('__')
        blocks=[];plabels=[];diagnostics=[]
        for ti,t in enumerate(targets):
            effect=pred[model]['effect_bulk'][index[t]][None,:]
            desired=project(effect,mu[modeled])[1][0]
            ratio=np.ones(len(genes),dtype=np.float64)
            ratio[modeled]=np.clip(np.divide(desired,mu[modeled],out=np.ones(len(modeled)),where=mu[modeled]>0),0,16)
            if model=='zero':ratio[:]=1
            genrng=np.random.default_rng(np.random.SeedSequence([SEED,ti,0 if sampler=='round' else 1]))
            block=pool[sampled[t]].copy()
            if sampler=='round':
                x=transform_counts(block,ratio,genrng)
            else:
                x=posterior_transport(block,ratio,mu,theta,genrng)
            diagnostics.append({'target':t,'sampler':sampler,'new_nonzero_entries':int(((x>0).astype(np.int8)-(block>0).astype(np.int8)).maximum(0).sum()),
                                'median_library':float(np.median(np.asarray(x.sum(axis=1)).ravel()))})
            blocks.append(x);plabels += [t]*400
        files[variant]=save_adata(out/f'pred__{variant}.h5ad',blocks,plabels,genes)
        pd.DataFrame(diagnostics).to_csv(out/f'generation__{variant}.csv',index=False)
        print('built',variant,files[variant],flush=True)
    a.file.close()
    pd.DataFrame({'role':['reference']*len(ref_rows)+['generation']*len(pool_rows),
                  'raw_row_index':np.r_[ref_rows,pool_rows]}).to_csv(out/'ntc_split.csv',index=False)
    (out/'real_target_rows.json').write_text(json.dumps(target_rows,indent=2)+'\n')
    manifest={**protocol,'thinning_probability':p,'raw_ntc_generator_median_library':20000/p,
              'real':real_info,'prediction_files':files,'input_prediction_hashes':{m:sha256(f) for m,f in pred_paths.items()},
              'cell_eval2_version':eval_version,'official_score':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def score(out,manifest,only=None):
    # Windows releases this advisory lock when a worker exits, including on
    # interruption. Concurrent resume commands cannot overwrite active metrics.
    import msvcrt,time
    lock=Path(out)/'.scoring.lock'
    with lock.open('a+b') as handle:
        if lock.stat().st_size==0:handle.write(b'0');handle.flush()
        while True:
            try:
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(5)
        try:_score_impl(out,manifest,only)
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def _score_impl(out,manifest,only=None):
    real_path=Path(manifest.get('real_path',out/'real.h5ad'))
    cfg=replace(EvalConfig.from_preset('vcc2026'),pert_col='target_gene',device='cpu',num_threads=4,gather_threads=4,
                cache_real=str(manifest.get('cache_real_path',out/'cache_real')),de=replace(EvalConfig.from_preset('vcc2026').de,backend='scanpy'))
    (out/'evaluation_config.json').write_text(json.dumps(asdict(cfg),indent=2)+'\n')
    summaries=[]
    for model in manifest['models']:
        if only and model!=only:continue
        metrics_path=out/f'metrics__{model}.csv';summary_path=out/f'aggregate__{model}.csv'
        if not summary_path.exists():
            # The local library validates identical label sets, unlike the VCC CLI
            # payload (which forbids NTC rows). The official evaluation uses real
            # controls on BOTH sides. Add these already-provided controls in a
            # separate evaluation-only file; target predictions are unchanged.
            eval_path=out/f'eval_input__{model}.h5ad'
            if not eval_path.exists():
                real=ad.read_h5ad(real_path)
                control=real[real.obs.target_gene.astype(str)=='non-targeting'].copy()
                del real
                pred=ad.read_h5ad(out/f'pred__{model}.h5ad')
                joined=ad.concat([pred,control],index_unique='-',merge='same')
                joined.write_h5ad(eval_path,compression='gzip',compression_opts=1)
                del joined,pred,control
            print('scoring',model,flush=True)
            m=compute_metrics(eval_path,real_path,config=cfg,write_de=False)
            m.write_csv(metrics_path)
            aggregate_metrics(m).write_csv(summary_path)
        summary=pd.read_csv(summary_path);summary.insert(0,'model',model);summaries.append(summary)
        print(summary.to_string(index=False),flush=True)
    if summaries:pd.concat(summaries).to_csv(out/'six_metric_results.csv',index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel-size',type=int,default=24)
    ap.add_argument('--out',type=Path,default=Path('runs/model_selection_v2/h1_single_cell_benchmark'))
    ap.add_argument('--build-only',action='store_true');ap.add_argument('--only')
    args=ap.parse_args();manifest=build(args.out,args.panel_size)
    if not args.build_only:score(args.out,manifest,args.only)


if __name__=='__main__':main()
