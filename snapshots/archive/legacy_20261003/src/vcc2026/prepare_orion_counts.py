"""Stream published sparse counts; retain matched-batch controls and missing masks."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

RAW=Path('data/raw/external/supplemental/orion')
OUT=Path('data/processed/supplemental/orion')


def matrix(batch):
    g=batch.column(batch.schema.get_field_index('gene_token_id'))
    v=batch.column(batch.schema.get_field_index('gene_expression'))
    go=g.offsets.to_numpy();vo=v.offsets.to_numpy()
    assert np.array_equal(go-go[0],vo-vo[0])
    tokens=g.values.to_numpy()[go[0]:go[-1]].astype(np.int32)
    values=v.values.to_numpy()[vo[0]:vo[-1]]
    assert np.isfinite(values).all() and (values>=0).all() and np.array_equal(values,np.rint(values))
    assert tokens.min()>=0 and tokens.max()<38606
    x=sparse.csr_matrix((values.astype(np.float32),tokens,(vo-vo[0]).astype(np.int64)),shape=(len(batch),38606))
    assert x.has_canonical_format,'Repeated or unsorted gene tokens'
    return x


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-shards',type=int);args=ap.parse_args()
    pa.set_cpu_count(2);pa.set_io_thread_count(2)
    line='HCT116';dest=OUT/('HCT116_pilot_counts' if args.max_shards else 'HCT116_counts');dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():print('REUSE_COMPLETE',dest);return
    metadata=pd.read_parquet(OUT/f'{line}_cell_metadata.parquet');meta=metadata[metadata.eligible]
    gm=pd.read_csv(OUT/'gene_mapping.csv');gm=gm[gm.gene_token_id>=0]
    genes=gm.official_symbol.to_numpy(str);tokens=gm.gene_token_id.to_numpy(np.int32)
    with np.load('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz') as z:source=set(z['target_names'][z['target_n_cells']>=30])
    competition=set(pd.read_csv('data/raw/vcc2026/validation/files/pert_counts.csv').target_gene)
    targets=sorted((source|competition)&set(meta.target)-{'non-targeting'});tmap={t:i for i,t in enumerate(targets)}
    n=len(targets);ng=len(genes)
    total=np.zeros((n,ng),np.float64);ctrl_matched=np.zeros_like(total);ns=np.zeros(n,np.int64)
    ctrl_all=np.zeros(ng,np.float64);ctrl_n=0;audits=[];files=sorted(metadata.shard.unique())
    if args.max_shards:files=files[:args.max_shards]
    start=time.monotonic()
    for fi,name in enumerate(files):
        m=metadata[metadata.shard==name].sort_values('row_in_shard').reset_index(drop=True)
        assert np.array_equal(m.row_in_shard,np.arange(len(m)))
        ids=m.target.map(tmap).fillna(-1).to_numpy(np.int64);keep=m.eligible.to_numpy();ntc=(m.target=='non-targeting').to_numpy()&keep
        assert ntc.sum()>=30
        batch_ns=np.zeros(n,np.int64);control=np.zeros(ng,np.float64);offset=0;checked=0
        for b in pq.ParquetFile(RAW/'data'/name).iter_batches(batch_size=1024,columns=['gene_token_id','gene_expression']):
            x=matrix(b);end=offset+x.shape[0]
            assert np.allclose(np.asarray(x.sum(1)).ravel(),m.total_counts.iloc[offset:end].to_numpy(),rtol=0,atol=0)
            assert np.array_equal(np.diff(x.indptr),m.n_genes_by_counts.iloc[offset:end].to_numpy())
            rows=keep[offset:end]&(ids[offset:end]>=0);rid=ids[offset:end][rows]
            xp=x[:,tokens]
            if rows.any():
                grouping=sparse.csr_matrix((np.ones(len(rid)),(rid,np.arange(len(rid)))),shape=(n,len(rid)))
                agg=(grouping@xp[rows]).tocoo();total[agg.row,agg.col]+=agg.data
                batch_ns+=np.bincount(rid,minlength=n)
            if ntc[offset:end].any():control+=np.asarray(xp[ntc[offset:end]].sum(0,dtype=np.float64)).ravel()
            checked+=x.shape[0];offset=end
        assert offset==len(m)
        mu=control/ntc.sum();nonzero=np.flatnonzero(batch_ns)
        for startrow in range(0,len(nonzero),256):
            idx=nonzero[startrow:startrow+256];ctrl_matched[idx]+=batch_ns[idx,None]*mu[None,:]
        ns+=batch_ns;ctrl_all+=control;ctrl_n+=int(ntc.sum())
        audits.append({'shard':name,'all_count_rows_checked':checked,'selected_target_cells':int(batch_ns.sum()),'controls':int(ntc.sum())})
        pd.DataFrame(audits).to_csv(dest/'batch_count_audit.csv',index=False)
        print('COUNTS',fi+1,'/',len(files),name,'seconds',round(time.monotonic()-start,1),flush=True)
    valid=ns>0
    # Every output column is measured; genes absent from this list remain missing, not zeros.
    means=(total[valid]/ns[valid,None]).astype(np.float32);controls=(ctrl_matched[valid]/ns[valid,None]).astype(np.float32)
    assert np.isfinite(means).all() and np.isfinite(controls).all()
    np.savez_compressed(dest/'pseudobulk.npz',gene_names=genes,gene_ids=gm.ensembl_id.to_numpy(str),
        target_names=np.asarray(targets)[valid],target_mean_counts=means,target_control_mean_counts=controls,
        control_mean_counts=(ctrl_all/ctrl_n).astype(np.float32),target_n_cells=ns[valid],control_n_cells=ctrl_n)
    summary={'context':line,'source':'Orion pinned Parquet','selected_targets':int(valid.sum()),'selected_cells':int(ns.sum()),
      'all_count_rows_checked':sum(a['all_count_rows_checked'] for a in audits),'controls':ctrl_n,'genes':ng,
      'target_selection':'all K562>=30 supported targets plus all 300 competition targets; cell-count coverage for all remaining targets is separately retained',
      'control_matching':'weighted mean of GEM-batch-specific NTC profiles, with each target own eligible batch cell counts as weights',
      'count_validation':'all downloaded HCT116 count values finite, nonnegative integers; sparse lengths and UMI totals match metadata',
      'elapsed_seconds':time.monotonic()-start,'created_utc':datetime.now(timezone.utc).isoformat()}
    (dest/'complete.json').write_text(json.dumps(summary,indent=2)+'\n');print('COUNTS_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
