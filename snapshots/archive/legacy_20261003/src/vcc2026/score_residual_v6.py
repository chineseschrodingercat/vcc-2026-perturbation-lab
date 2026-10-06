"""Official metric engine and explicit six-member local overall references."""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,json,os,sys,hashlib,shutil
import numpy as np
import pandas as pd
import polars as pl
import anndata as ad
from cell_eval2 import EvalConfig,compute_metrics,aggregate_metrics,__version__
from cell_eval2.baseline import generic_response_profile,build_baseline_prediction,baseline_config
from cell_eval2.anchor import compute_replicate_anchor
from cell_eval2.score import _replicate_entries
from cell_eval2.scoring import score_one
from cell_eval2.scales import SCALES,scale_payload
from residual_v6_common import *
from diagnose_v4 import decomposition

ad.settings.allow_write_nullable_strings=True
SCALE='low-random_high-1_v10'
def cfg_for(path,dest):
    c=EvalConfig.from_preset('vcc2026');threads=int(os.environ.get('VCC_SCORE_THREADS','2'))
    cache=path/'cache_real'
    if path.name.startswith('seed_'):cache=path.parent/f'seed_{SEED}'/'cache_real'
    return replace(c,pert_col='target_gene',device='cpu',num_threads=threads,gather_threads=threads,cache_real=str(cache),outdir=str(dest),de=replace(c.de,backend='scanpy'))
def raw_values(path):
    a=pd.read_csv(path);return {r.metric:float(r.mean) for r in a.itertuples() if r.metric in METRICS}
def references(path):
    path=Path(path);dest=path/'references';dest.mkdir(exist_ok=True)
    if (dest/'reference.json').exists():return
    cfg=cfg_for(path,dest);real=ad.read_h5ad(path/'real.h5ad')
    if not (dest/'baseline_aggregate.csv').exists():
        profile=generic_response_profile(real,pert_col='target_gene',control='non-targeting')
        base=build_baseline_prediction(profile,real,pert_col='target_gene',control='non-targeting',emit='dispersed',seed=0)
        m=compute_metrics(base,real,config=baseline_config(cfg),write_de=False)
        m.write_csv(dest/'baseline_metrics.csv');aggregate_metrics(m).write_csv(dest/'baseline_aggregate.csv')
        del base
    if not (dest/'anchor_aggregate.parquet').exists():
        splits,anchor=compute_replicate_anchor(real,config=cfg,base_seed=0,n_splits=5)
        splits.write_parquet(dest/'anchor_splits.parquet');anchor.write_parquet(dest/'anchor_aggregate.parquet')
    bv=raw_values(dest/'baseline_aggregate.csv');anchor=pl.read_parquet(dest/'anchor_aggregate.parquet');valid=True;reason=None
    try:
        entries=_replicate_entries(bv,anchor);assert set(entries)==set(METRICS)
    except (ValueError,AssertionError) as e:valid=False;reason=str(e)
    dump(dest/'config.json',asdict(cfg));dump(dest/'fixed_scale.json',scale_payload(SCALES[SCALE]))
    dump(dest/'reference.json',{'cell_eval2':__version__,'real_sha256':sha(path/'real.h5ad'),'replicate_usable':valid,'reason':reason,
        'baseline':bv,'replicate':{r['metric']:r['replicate'] for r in anchor.to_dicts() if r['metric'] in METRICS},
        'five_split_seed':0,'baseline_release':'official dispersed generic response without unavailable competition knockdown filter',
        'protocol_sha256':sha('configs/residual_v6_protocol.json'),'local_only':True})
    print('REFERENCE_COMPLETE',path,valid,reason,flush=True)
def score_path(path,names=None,do_reference=False):
    path=Path(path)
    if do_reference:references(path)
    real=ad.read_h5ad(path/'real.h5ad');ctrl=real[real.obs.target_gene.astype(str)=='non-targeting'].copy();targets=json.loads((path/'panel.json').read_text())['targets']
    assert all(t in real.var_names for t in targets)
    models=names or [p.stem.removeprefix('pred__') for p in path.glob('pred__*.h5ad')]
    for name in models:
        dest=path/'scores'/name;dest.mkdir(parents=True,exist_ok=True)
        if name in ['A_selected','B_selected'] and not (dest/'aggregate.csv').exists():
            meta=json.loads((path/f'{name[0]}_selection.json').read_text());source=meta['source'];origin=path/'scores'/source
            assert os.path.samefile(path/f'pred__{name}.h5ad',path/f'pred__{source}.h5ad')
            if (origin/'aggregate.csv').exists():
                for f in ['aggregate.csv','metrics.csv','fid_decomposition.csv','config.json']:
                    if (origin/f).exists():shutil.copy2(origin/f,dest/f)
                dump(dest/'score_reuse.json',{'identical_prediction_hardlink':source,'real_sha256':sha(path/'real.h5ad')})
        if not (dest/'aggregate.csv').exists():
            pred=ad.read_h5ad(path/f'pred__{name}.h5ad');joined=ad.concat([pred,ctrl],index_unique='-',merge='same')
            cfg=cfg_for(path,dest);dump(dest/'config.json',asdict(cfg))
            m=compute_metrics(joined,path/'real.h5ad',config=cfg,write_de=True)
            m.write_csv(dest/'metrics.csv');aggregate_metrics(m).write_csv(dest/'aggregate.csv');del pred,joined
        u=raw_values(dest/'aggregate.csv');assert set(u)==set(METRICS)
        modeled,_=axes()
        if not (dest/'fid_decomposition.csv').exists():
            d=decomposition(pl.read_parquet(dest/'de_real.parquet'),pl.read_parquet(dest/'de_pred.parquet'),modeled,targets)
            d.to_csv(dest/'fid_decomposition.csv',index=False)
        else:d=pd.read_csv(dest/'fid_decomposition.csv')
        assert abs(d.fid_reconstructed.mean()-u[METRICS[2]])<1e-8
        entries=SCALES[SCALE].entries;fixed={m:score_one(u[m],entries[m].base,entries[m].scoring) for m in METRICS}
        rec={'model':name,**u,'fixed_overall':float(np.mean(list(fixed.values()))),'replicate_overall':None}
        metricrows=[{'metric':m,'raw':u[m],'fixed_scaled':fixed[m]} for m in METRICS]
        rp=path/'references/reference.json'
        if rp.exists():
            ref=json.loads(rp.read_text());assert ref['real_sha256']==sha(path/'real.h5ad')
            if ref['replicate_usable']:
                en=_replicate_entries(ref['baseline'],pl.read_parquet(path/'references/anchor_aggregate.parquet'))
                scores={m:score_one(u[m],en[m].base,en[m].scoring) for m in METRICS}
                rec['replicate_overall']=float(np.mean(list(scores.values())))
                for row in metricrows:
                    m=row['metric'];row.update(replicate_scaled=scores[m],baseline=ref['baseline'][m],replicate=ref['replicate'][m])
        pd.DataFrame(metricrows).to_csv(dest/'overall_components.csv',index=False);dump(dest/'summary.json',rec)
        print('SCORE',path.name,json.dumps(rec),flush=True)
    records=[json.loads(p.read_text()) for p in (path/'scores').glob('*/summary.json')]
    pd.DataFrame(records).to_csv(path/'six_metrics_overall.csv',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--path');ap.add_argument('--inner-all',action='store_true');ap.add_argument('--references',action='store_true');ap.add_argument('--models',nargs='+');a=ap.parse_args()
    paths=[ROOT/'inner_panels'/l for l in CONTEXTS] if a.inner_all else [Path(a.path)]
    for p in paths:score_path(p,a.models,a.references)
