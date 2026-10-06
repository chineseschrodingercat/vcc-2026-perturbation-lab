"""Immutable V1 counts semantics and shared local V6 panels."""
from pathlib import Path
import hashlib,json
from functools import lru_cache
import numpy as np
from scipy import sparse
from generate_counts import multiplier_from_effects,transform_counts

ROOT=Path('runs/residual_v6'); MODEL=Path('models/residual_v6')
CONTEXTS=['hepg2','jurkat','rpe1']; SEED=20260928
METRICS=['pds_cosine','expr_mse_unbiased_capped_norm','de_wilcoxon_direction_fidelity_yield_raw','de_wilcoxon_direction_reach_raw','de_wilcoxon_sig_jaccard','de_wilcoxon_lfc_nmae']
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fold(t):return int(hashlib.sha256(('vcc-v4:'+str(t)).encode()).hexdigest(),16)%5
def axes():return np.loadtxt('runs/context_v4/gene_axis.txt',dtype=str),np.loadtxt('runs/context_v4/normalization_core.txt',dtype=str)
def frozen_v1(targets,genes):
    cfg=json.loads(Path('configs/baseline_v1.json').read_text())
    with np.load('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz') as z:
        tg=z['target_names'].astype(str);gn=z['gene_names'].astype(str);n=z['target_n_cells'];cm=z['control_mean_counts'];tm=z['target_mean_counts']
        ti={t:i for i,t in enumerate(tg)};gi={g:i for i,g in enumerate(gn)}
        dest=np.array([j for j,g in enumerate(genes) if g in gi]);src=np.array([gi[genes[j]] for j in dest])
        out=np.zeros((len(targets),len(genes)),np.float32)
        for i,t in enumerate(targets):
            if t not in ti or n[ti[t]]<30:continue
            j=ti[t];out[i,dest]=np.clip(cfg['cross_context_scale']*(n[j]/(n[j]+50))*np.log2((tm[j,src]+.5)/(cm[src]+.5)),-3,3)
    return out
def positions(genes,modeled):
    gm={g:i for i,g in enumerate(genes)};pairs=[(i,gm[g]) for i,g in enumerate(modeled) if g in gm]
    return np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs])
@lru_cache(maxsize=1)
def source_gene_set():
    with np.load('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz') as z:return set(z['gene_names'].astype(str))
def count_ratio(raw_effect,mu,genes,modeled):
    si,di=positions(genes,modeled)
    # Preserve the submitted V1's literal multiplier=1 outside its source axis.
    # Zero residual must not introduce a floating-point inversion on new genes.
    keep=np.array([modeled[i] in source_gene_set() or raw_effect[i]!=0 for i in si])
    return multiplier_from_effects(mu,di[keep],np.asarray(raw_effect)[si[keep]])[0]
def generate(path,name,effects,seed=SEED,raw_ratio=False):
    from benchmark_context_v4 import save
    path=Path(path);info=json.loads((path/'panel.json').read_text());genes=np.loadtxt(path/'genes.txt',dtype=str);modeled,_=axes()
    pool=sparse.load_npz(path/'generation_pool.npz');mu=np.asarray(pool.mean(0)).ravel().astype(np.float64)
    rng=np.random.default_rng(seed+100);blocks=[];labels=[]
    order=info.get('generation_targets',info['targets']);sizes=info.get('generation_target_cells',info['target_cells'])
    samples={t:rng.choice(pool.shape[0],sizes[t],replace=False) for t in order}
    for i,t in enumerate(info['targets']):
        ids=samples[t]
        ratio=effects[i] if raw_ratio else count_ratio(effects[i],mu,genes,modeled)
        rr=np.random.default_rng(np.random.SeedSequence([seed,order.index(t),0]))
        blocks.append(transform_counts(pool[ids].tocsr(copy=True),ratio,rr));labels.extend([t]*len(ids))
    dest=path/f'pred__{name}.h5ad';save(dest,blocks,labels,genes)
    return dest
