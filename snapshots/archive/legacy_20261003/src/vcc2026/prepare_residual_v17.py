"""Training-only, stratified count bootstrap and control-cell inputs for V17.

Bootstrap draws are technical views, not biological replicates. Frozen V6 labels
are kept; only centered, matched-control response fluctuations augment them.
No fold0/fold1 or H1 perturbation counts are extracted into training caches.
"""
from pathlib import Path
import json, hashlib, time, argparse
import numpy as np
import pandas as pd
import anndata as ad
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from residual_v6_common import axes, fold, dump, SEED, CONTEXTS
from prepare_orion_counts import matrix

ROOT=Path('data/processed/residual_v17')
HCT=Path('data/processed/supplemental/orion/v17_training_views')
SOURCES={
 'hepg2':'data/raw/external/nadig_2025/GSE264667_hepg2_raw_singlecell_01.h5ad',
 'jurkat':'data/raw/external/nadig_2025/GSE264667_jurkat_raw_singlecell_01.h5ad',
 'rpe1':'data/raw/external/replogle_2022/rpe1/rpe1_raw_singlecell_01.h5ad',
 'k562':'data/raw/external/replogle_2022/k562_gwps/K562_gwps_raw_singlecell_01.h5ad'}
VIEWS=4

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def cache(line):return HCT if line=='orion_hct116' else ROOT/line
def ids_for(line):
    with np.load(f'runs/residual_v6/training_data/{line}.npz') as z:t=z['targets'].astype(str)
    with np.load('data/processed/v2/external_pseudobulk/vcc2025_h1.npz') as z:excluded=set(z['target_names'])
    assert all(fold(x)>=2 and x not in excluded for x in t)
    return t

def plan(m,targets,seed):
    """Fixed guide/batch allocations, multinomial cells within each stratum."""
    rng=np.random.default_rng(seed);n=len(m);tm={t:i for i,t in enumerate(targets)}
    rid=m.target.map(tm).fillna(-1).to_numpy(np.int32)
    ntc=m.target.eq('non-targeting').to_numpy()&m.eligible.to_numpy()
    batches=sorted(m.loc[ntc,'batch'].unique());bm={b:i for i,b in enumerate(batches)}
    bid=m.batch.map(bm).fillna(-1).to_numpy(np.int32)
    keep=m.eligible.to_numpy()&((rid>=0)|ntc)&(bid>=0)
    weights=np.zeros((VIEWS+1,n),np.float32);weights[0,keep]=1
    strata=[]
    for (target,batch,guide),ix in m.loc[keep].groupby(['target','batch','guide'],sort=True).groups.items():
        ix=np.asarray(ix);size=len(ix)
        for v in range(1,VIEWS+1):weights[v,ix]=rng.multinomial(size,np.full(size,1/size))
        strata.append({'target':target,'batch':batch,'guide':guide,'cells':size,'fold':-1 if target=='non-targeting' else fold(target)})
    count=np.bincount(rid[keep&(rid>=0)],minlength=len(targets));assert (count>=30).all(),targets[count<30]
    nc=np.bincount(bid[ntc],minlength=len(batches));assert (nc>0).all()
    bw=np.zeros((len(targets),len(batches)),np.float64)
    j=keep&(rid>=0);np.add.at(bw,(rid[j],bid[j]),1)
    assert np.array_equal(bw.sum(1),count)
    # Every view preserves guide/batch allocation, not expression-state proportions.
    for v in range(1,VIEWS+1):
        assert weights[v].sum()==weights[0].sum()
    return rid,bid,keep,ntc,weights,count,nc,bw,batches,pd.DataFrame(strata)

def aggregate(x,lo,hi,plan_,out,ctrl,features):
    rid,bid,keep,ntc,w,n,nc,bw,batches,strata=plan_
    xp=x[:,features['src']]
    # The two totals include the actual measured native panel and common core.
    totals=sparse.csr_matrix(np.c_[np.asarray(x[:,features['native']].sum(1)).ravel(),np.asarray(x[:,features['core']].sum(1)).ravel()])
    xp=sparse.hstack([xp,totals],format='csr')
    rr=rid[lo:hi];bb=bid[lo:hi];kk=keep[lo:hi]&(rr>=0);cc=ntc[lo:hi]
    for v in range(VIEWS+1):
        ww=w[v,lo:hi]
        for valid,idx,size,dest in [(kk,rr,len(n),out[v]),(cc,bb,len(nc),ctrl[v])]:
            valid=valid&(ww>0);ii=np.flatnonzero(valid)
            if not len(ii):continue
            group=sparse.csr_matrix((ww[ii],(idx[ii],np.arange(len(ii)))),shape=(size,len(ii)))
            # CSR multiplication aggregates each row/column coordinate once.
            a=(group@xp[ii]).tocoo();dest[a.row,a.col]+=a.data

def finish(line,targets,features,plan_,out,ctrl,source_records):
    dest=cache(line);genes,core=axes();rid,bid,keep,ntc,w,n,nc,bw,batches,strata=plan_
    out/=n[None,:,None]
    ctrl/=nc[None,:,None]
    matched=np.stack([(bw/n[:,None])@ctrl[v] for v in range(VIEWS+1)]).astype(np.float32)
    ng=len(genes);di=features['dest'];mask=np.zeros(ng,np.uint8);mask[di]=1
    response=np.log1p(out[:,:,:-2]*50000/out[:,:,-2: -1])-np.log1p(matched[:,:,:-2]*50000/matched[:,:,-2:-1])
    effect_core=np.log1p(out[:,:,:-2]*50000/out[:,:,-1:])-np.log1p(matched[:,:,:-2]*50000/matched[:,:,-1:])
    noise=np.zeros((VIEWS,len(targets),ng),np.float32);noise[:,:,di]=response[1:]-response[:1]
    effects=np.zeros((VIEWS+1,len(targets),ng),np.float32);effects[:,:,di]=effect_core
    gp={g:i for i,g in enumerate(genes)}
    for j,t in enumerate(targets):
        if t in gp:noise[:,j,gp[t]]=0;effects[:,j,gp[t]]=0
    assert np.isfinite(noise).all() and np.isfinite(effects).all()
    np.save(dest/'response_noise.npy',noise);np.save(dest/'source_effects.npy',effects)
    np.savez_compressed(dest/'metadata.npz',targets=targets,genes=genes,measured_mask=mask,n_cells=n,batch_weights=(bw/n[:,None]).astype(np.float32),batch_names=np.asarray(batches))
    strata.to_csv(dest/'guide_batch_strata.csv',index=False)
    audit={'context':line,'n_targets':len(targets),'qualified_target_cells':int(n.sum()),'matched_controls':int(nc.sum()),'batches':len(batches),'guide_batch_strata':len(strata),'views':VIEWS,'all_targets_fold_ge_2_and_H1_excluded':True,'unmeasured_genes_masked':True,'native_measured_genes':len(features['native']),'modeled_measured_genes':int(mask.sum()),'response_noise_RMS':float(np.sqrt(np.mean(noise**2))),'source_files':source_records,'view_rule':'Four multinomial cell bootstraps within target x GEM batch x guide; original stratum sizes retained. Matched control bootstraps from same GEM batches weighted by target batch counts. No cell-state matching/constraint. Views are technical samples, not biological repeats.','training_use':'Center each view response on full matched response, add to immutable V6 normalized target profile and clamp nonnegative. This preserves original cohort/labels and isolates technical augmentation; it is not a new measured cell or new biological replicate. Source response for alternative training contexts is projected using measured genes only.','original_label_replacement':False}
    dump(dest/'complete.json',audit);print('V17_DATA_COMPLETE',line,json.dumps({k:v for k,v in audit.items() if k not in ['source_files','training_use','view_rule']}),flush=True)

def native_features(names):
    genes,core=axes();names=np.asarray(names,str);valid=~pd.Index(names).duplicated(keep=False)&(names!='')&(names!='HSPA14')
    native=np.flatnonzero(valid);gm={names[i]:i for i in native};di=np.array([i for i,g in enumerate(genes) if g in gm]);si=np.array([gm[genes[i]] for i in di])
    missing=set(core)-set(gm);assert missing.issubset({'C19orf48','PRRC2B','SOD2'})
    return {'src':si,'dest':di,'native':native,'core':np.array([gm[g] for g in core if g in gm]),'core_genes':np.array([g for g in core if g in gm])}

def native(line):
    dest=cache(line);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    targets=ids_for(line);a=ad.read_h5ad(SOURCES[line],backed='r');obs=a.obs
    m=pd.DataFrame({'target':obs.gene.astype(str).to_numpy(),'batch':obs.gem_group.astype(str).to_numpy(),'guide':obs.sgID_AB.astype(str).to_numpy(),'eligible':True})
    f=native_features(a.var.gene_name.to_numpy(str));pp=plan(m,targets,SEED+sum(map(ord,line)))
    width=len(f['src'])+2;out=np.zeros((5,len(targets),width),np.float64);ctrl=np.zeros((5,len(pp[6]),width),np.float64)
    for lo in range(0,len(m),1024):
        hi=min(lo+1024,len(m));valid=pp[2][lo:hi]
        if valid.any():
            x=sparse.csr_matrix(a.X[lo:hi]);assert np.isfinite(x.data).all() and (x.data>=0).all() and np.array_equal(x.data,np.rint(x.data));aggregate(x,lo,hi,pp,out,ctrl,f)
        if lo%51200==0:print('V17_NATIVE',line,lo,len(m),flush=True)
    a.file.close();finish(line,targets,f,pp,out,ctrl,[{'path':SOURCES[line],'bytes':Path(SOURCES[line]).stat().st_size}])

def hct():
    dest=HCT;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    base=Path('data/processed/supplemental/orion');raw=Path('data/raw/external/supplemental/orion')
    m=pd.read_parquet(base/'HCT116_cell_metadata.parquet');e=pd.read_parquet(base/'HCT116_eligibility_library_v3.parquet')
    assert m.cell_barcode.tolist()==e.cell_barcode.tolist()
    m['target']=e.canonical_target.fillna('').to_numpy();m['eligible']=e.eligible_library_v3.to_numpy();m['batch']=m['sample'];m['guide']=m.guide_target
    targets=ids_for('orion_hct116');pp=plan(m,targets,SEED+171)
    gm=pd.read_csv(base/'gene_mapping.csv');gm=gm[(gm.gene_token_id>=0)&(gm.official_symbol!='HSPA14')]
    names=np.full(38606,'',dtype='<U40');names[gm.gene_token_id.to_numpy()]=gm.official_symbol.to_numpy(str);f=native_features(names)
    width=len(f['src'])+2;out=np.zeros((5,len(targets),width),np.float64);ctrl=np.zeros((5,len(pp[6]),width),np.float64)
    rng=np.random.default_rng(SEED+172);poolrows=[]
    for b in pp[8]:
        ix=np.flatnonzero(pp[3]&(m.batch.to_numpy()==b));poolrows.extend(rng.choice(ix,min(64,len(ix)),replace=False).tolist())
    poolset=set(poolrows);pool=[];pool_ids=[];files=sorted(m.shard.unique());source=[]
    for fi,name in enumerate(files):
        mm=m[m.shard==name];assert np.array_equal(mm.row_in_shard,np.arange(len(mm)))
        globalids=mm.index.to_numpy();assert np.array_equal(globalids,np.arange(globalids[0],globalids[0]+len(mm)))
        offset=0;path=raw/'data'/name
        for b in pq.ParquetFile(path).iter_batches(batch_size=1024,columns=['gene_token_id','gene_expression']):
            hi=offset+len(b);lo_global=int(globalids[0]+offset);hi_global=lo_global+len(b)
            wanted=pp[2][lo_global:hi_global].any();chosen=[j for j,k in enumerate(globalids[offset:hi]) if int(k) in poolset]
            if wanted or chosen:
                x=matrix(b);assert np.array_equal(np.asarray(x.sum(1)).ravel(),mm.total_counts.iloc[offset:hi].to_numpy())
                aggregate(x,lo_global,hi_global,pp,out,ctrl,f)
                if chosen:pool.append(x[chosen][:,f['core']]);pool_ids.extend(globalids[offset:hi][chosen].tolist())
            offset=hi
        assert offset==len(mm)
        source.append({'path':str(path),'bytes':path.stat().st_size,'published_download_manifest':str(path)+'.manifest.json'})
        print('V17_ORION',fi+1,len(files),name,flush=True)
    pool=sparse.vstack(pool,format='csr');p=read('runs/residual_v6/training_data/complete.json')
    with np.load('runs/residual_v6/training_data/orion_hct116.npz') as z:prob=float(z['thinning_probability'])
    pool.data=rng.binomial(pool.data.astype(np.int64),prob).astype(np.float32);pool.eliminate_zeros()
    sparse.save_npz(dest/'control_core_counts.npz',pool);np.savetxt(dest/'control_genes.txt',f['core_genes'],fmt='%s')
    m.loc[pool_ids,['cell_barcode','sample','shard','row_in_shard','guide_target','total_counts']].to_csv(dest/'control_cells.csv',index=False)
    finish('orion_hct116',targets,f,pp,out,ctrl,source)

def control_pools():
    genes,core=axes()
    for line in CONTEXTS:
        dest=cache(line);dest.mkdir(parents=True,exist_ok=True);old=Path('runs/residual_v16/inner_panels')/line
        native=np.loadtxt(old/'genes.txt',dtype=str);gp={g:i for i,g in enumerate(native)}
        x=sparse.load_npz(old/'generation_pool.npz')[:,[gp[g] for g in core]]
        sparse.save_npz(dest/'control_core_counts.npz',x);np.savetxt(dest/'control_genes.txt',core,fmt='%s')
        dump(dest/'control_source.json',{'source':str(old/'generation_pool.npz'),'sha256':sha_file(old/'generation_pool.npz'),'genes_sha256':sha_file(old/'genes.txt'),'role':'generation controls only; scoring reference controls excluded','cells':x.shape[0]})
    dest=ROOT/'k562';dest.mkdir(parents=True,exist_ok=True)
    if not (dest/'control_core_counts.npz').exists():
        a=ad.read_h5ad(SOURCES['k562'],backed='r');ix=np.flatnonzero(a.obs.gene.astype(str).to_numpy()=='non-targeting')
        rng=np.random.default_rng(SEED+173);ix=np.sort(rng.choice(ix,min(3000,len(ix)),replace=False));gp={g:i for i,g in enumerate(a.var.gene_name.astype(str))};columns=[gp[g] for g in core]
        blocks=[sparse.csr_matrix(a.X[ix[j:j+128]])[:,columns] for j in range(0,len(ix),128)];sparse.save_npz(dest/'control_core_counts.npz',sparse.vstack(blocks,format='csr'))
        pd.DataFrame({'original_row':ix,'cell_id':a.obs_names[ix]}).to_csv(dest/'control_cells.csv',index=False);np.savetxt(dest/'control_genes.txt',core,fmt='%s');a.file.close()
        dump(dest/'control_source.json',{'source':SOURCES['k562'],'purpose':'source context representation only; never PCA fitting unless explicitly a training destination (not done here)','cells':len(ix)})

def main():
    pa.set_cpu_count(2);pa.set_io_thread_count(2);ROOT.mkdir(parents=True,exist_ok=True)
    control_pools()
    for line in CONTEXTS:native(line)
    hct()
    files=[p for base in [ROOT,HCT] for p in base.rglob('*') if p.is_file() and p.name!='input_manifest.json']
    dump(ROOT/'input_manifest.json',{'files':{str(p):sha_file(p) for p in files},'no_heldout_supervised_targets':True,'no_HEK293T_or_H1_count_access':True})
    print('V17_INPUTS_COMPLETE',len(files),flush=True)
if __name__=='__main__':main()
