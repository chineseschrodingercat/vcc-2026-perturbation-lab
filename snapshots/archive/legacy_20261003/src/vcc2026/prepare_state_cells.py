"""Prepare fixed, bounded cell samples for official State architecture pilots."""
from pathlib import Path
import json,hashlib,time
import h5py
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from foundation_io import metadata
from select_transfer_model import read_source,profile

SEED=20260927
OUT=Path('data/processed/v3/state_cells')
SOURCES={
 'hepg2':'data/raw/external/nadig_2025/GSE264667_hepg2_raw_singlecell_01.h5ad',
 'jurkat':'data/raw/external/nadig_2025/GSE264667_jurkat_raw_singlecell_01.h5ad',
 'rpe1':'data/raw/external/replogle_2022/rpe1/rpe1_raw_singlecell_01.h5ad',
 'k562':'data/raw/external/replogle_2022/k562_gwps/K562_gwps_raw_singlecell_01.h5ad'}
IDS={'hepg2':'nadig_hepg2','jurkat':'nadig_jurkat','rpe1':'replogle_rpe1','k562':'replogle_k562_gwps'}

def fold(t):return int(hashlib.sha256(('state-v3:'+t).encode()).hexdigest(),16)%10

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():return
    bulks={k:read_source(f'data/processed/v1/external_pseudobulk/{v}.npz') for k,v in IDS.items()}
    h1=read_source('data/processed/v2/external_pseudobulk/vcc2025_h1.npz');h1targets=set(h1['target_names'])
    with np.load('runs/model_selection_v2/h1_confirmation/prediction__zero.npz') as z:core=z['gene_symbols'].astype(str)
    variance=[]
    for line,b in bulks.items():
        gi=np.array([list(b['gene_names']).index(g) for g in core])
        keep=(b['target_n_cells']>=30)&np.array([t not in h1targets and fold(t)!=0 for t in b['target_names']])
        eff=profile(b['target_mean_counts'][keep][:,gi])-profile(b['control_mean_counts'][gi])
        variance.append(np.mean(eff**2,axis=0))
    hvg=np.sort(np.argsort(np.mean(variance,axis=0))[-1024:]);genes=core[hvg]
    targets=np.array(sorted(set(np.concatenate([b['target_names'] for b in bulks.values()]))))
    embed=np.zeros((len(targets),64),dtype=np.float32);found=[]
    with h5py.File('data/raw/external/slim_embeddings/gene_string_embeddings.v0.3.h5') as f:
        for i,t in enumerate(targets):
            if t in f:embed[i]=f[t][()];found.append(i)
    development=np.array([t not in h1targets and fold(t)!=0 for t in targets]);fmask=np.zeros(len(targets),bool);fmask[found]=True
    emean=embed[development&fmask].mean(0);estd=np.maximum(embed[development&fmask].std(0),.01)
    embed[fmask]=(embed[fmask]-emean)/estd
    source=bulks['k562'];gi=np.array([list(source['gene_names']).index(g) for g in core]);si={t:i for i,t in enumerate(source['target_names'])}
    source_effect=profile(source['target_mean_counts'][:,gi])-profile(source['control_mean_counts'][gi])
    source_effect*= (source['target_n_cells']/(source['target_n_cells']+50))[:,None]
    gp={g:i for i,g in enumerate(core)}
    for i,t in enumerate(source['target_names']):
        if t in gp:source_effect[i,gp[t]]=0
    train=np.array([t not in h1targets and fold(t)!=0 for t in source['target_names']])&(source['target_n_cells']>=30)
    _,_,vt=randomized_svd(source_effect[train],n_components=64,n_iter=5,random_state=SEED)
    latent=source_effect@vt.T;ls=np.maximum(latent[train].std(0),.01)
    source_embedding=np.zeros((len(targets),64),dtype=np.float32)
    for i,t in enumerate(targets):
        if t in si and source['target_n_cells'][si[t]]>=30:source_embedding[i]=latent[si[t]]/ls
    np.savez_compressed(OUT/'features.npz',targets=targets,genes=genes,core_genes=core,hvg_positions=hvg,
        string_embedding=embed,source_embedding=source_embedding,string_mean=emean,string_std=estd,
        source_svd_components=vt,source_latent_std=ls,source_control=source['control_mean_counts'][gi])
    protocol={'seed':SEED,'features':1024,'normalization':'per cell library over 6122 measured common genes -> 10000 -> log1p; output 1024 selected genes',
        'gene_selection':'Top 1024 training-only mean squared normalized pseudobulk responses, equal source weights',
        'cells_per_target':32,'control_cells':2048,'validation':'SHA256(state-v3:target) mod 10 == 0, same target excluded from fitting in all lines',
        'h1_exclusion':'All 150 H1 target labels excluded from neural network fitting across all four lines; H1 raw perturbed cells never used',
        'source_embedding':'Optional 64d SVD of observed K562 response; basis excludes validation and H1 targets. Source observations for evaluation targets are allowed inputs.',
        'sources':SOURCES,'source_NTC_label':'non-targeting','missing_string':'all-zero normalized vector','input_raw_files_modified':False}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    groups=[];rng=np.random.default_rng(SEED)
    for line,pathstr in SOURCES.items():
        dest=OUT/f'{line}.npy';meta=OUT/f'{line}_groups.csv'
        path=Path(pathstr)
        if not path.exists():
            print('Waiting for verified source',path.name,flush=True)
            while not path.exists():time.sleep(15)
        labels,srcgenes=metadata(path);lookup={g:i for i,g in enumerate(srcgenes)}
        if pd.Index(srcgenes).duplicated(keep=False)[[lookup[g] for g in core]].any():raise ValueError('Ambiguous common gene')
        cols=np.array([lookup[g] for g in core]);counts=pd.Series(labels).value_counts()
        # Build the label-to-row index once; scanning two million strings for
        # every one of ~10,000 targets wastes hundreds of GB of memory traffic.
        grouped_rows=pd.Series(np.arange(len(labels))).groupby(labels,sort=False).indices
        selected=[];linegroups=[];position=0
        for t in ['non-targeting']+sorted(t for t in counts.index if t!='non-targeting' and counts[t]>=32 and t not in h1targets):
            ids=grouped_rows[t];n=min(2048,len(ids)) if t=='non-targeting' else 32
            rows=rng.choice(ids,n,replace=False);selected.extend(rows)
            linegroups.append({'line':line,'target':t,'start':position,'n':n,'split':'control' if t=='non-targeting' else ('validation' if fold(t)==0 else 'train')})
            position+=n
        selected=np.asarray(selected);order=np.argsort(selected);sortedrows=selected[order]
        if dest.exists() and meta.exists():
            original=np.load(OUT/f'{line}_original_row_indices.npy')
            if not np.array_equal(original,selected):raise ValueError(f'{line}: cached row selection differs from deterministic replay')
            groups.extend(pd.read_csv(meta).to_dict('records'));continue
        x=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(selected),len(genes)))
        with h5py.File(path,'r') as f:
            for start in range(0,len(selected),512):
                raw=np.asarray(f['X'][sortedrows[start:start+512],:],dtype=np.float32)[:,cols]
                if np.any(raw<0) or not np.isfinite(raw).all() or np.any(raw!=np.rint(raw)):raise ValueError('Not raw UMI')
                total=raw.sum(1,keepdims=True)
                normalized=np.log1p(raw[:,hvg]*10000/np.maximum(total,1))
                x[order[start:start+512]]=normalized
                if start%65536==0:print(line,start,'/',len(selected),flush=True)
        x.flush();del x
        pd.DataFrame(linegroups).to_csv(meta,index=False)
        np.save(OUT/f'{line}_original_row_indices.npy',selected)
        groups.extend(linegroups);print('prepared',line,len(selected),len(linegroups),flush=True)
    df=pd.DataFrame(groups);df.to_csv(OUT/'all_groups.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps({'cells':int(df.n.sum()),'groups':len(df),'genes':1024,'lines':list(SOURCES)},indent=2)+'\n')

if __name__=='__main__':main()
