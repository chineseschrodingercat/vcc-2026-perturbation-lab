"""A small real-data count-generation check, explicitly not a VCC submission."""
from pathlib import Path
import json
import numpy as np
import h5py
import anndata as ad
from scipy import sparse
import lean_transfer as lt
import lean_union as lu


def main():
    root=Path(__file__).resolve().parents[1]
    ids=['k562_gwps','k562_essential','rpe1_bulk','hepg2_sc','jurkat_sc','rpe1_sc']
    sources=[lt.load_source(root/'data/processed/restart'/f'{s}.npz') for s in ids]
    model=json.loads((root/'outputs/restart/union_cv/selected_model.json').read_text())
    lt.validate_source_locks(sources,model)
    query_source=next(s for s in sources if s.source_id=='rpe1_sc')
    training=[s for s in sources if s.background!='RPE1']
    genes=query_source.genes
    core=np.asarray(model['core_genes'])
    cp=np.asarray([query_source.gmap[g] for g in core])
    targets=query_source.targets[:2]
    input_path=root/'data/raw/external/scperturb/ReplogleWeissman2022_rpe1.h5ad'
    a=ad.read_h5ad(input_path,backed='r')
    try:
        indices=np.flatnonzero(a.obs['gene'].astype(str).to_numpy()=='non-targeting')
        pool=a.X[indices]
        mean=np.asarray(pool.mean(axis=0)).ravel().astype(float)
        second=(np.asarray(pool.astype(np.float64).power(2).sum(axis=0)).ravel()
                if sparse.issparse(pool) else np.sum(np.asarray(pool,dtype=np.float64)**2,axis=0))
        variance=np.maximum((second-len(indices)*mean**2)/(len(indices)-1),0)
    finally:
        a.file.close()
    delta,mask,audit=lu.predict_delta(training,core,genes,targets,mean,
        model=model['model'],alpha=model['alpha'],temperature=model['temperature'],
        prior_cells=model['prior_cells'],min_cells=model['min_cells'])
    _,desired=lu.project(delta,mean,cp,mask)
    folder=root/'outputs/restart/smoke'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'local_RPE1_counts.h5ad'
    if path.exists():
        raise FileExistsError(path)
    lt.init_output(path,['RPE1_LOCAL_SMOKE'],targets,genes)
    row=nnz=0
    libraries=[]
    with h5py.File(path,'r+') as f:
        for i,target in enumerate(targets):
            block=lt.sample_counts(desired[i],mean,variance,np.random.default_rng(20261103+i),generator='nb')
            libraries.extend(np.asarray(block.sum(axis=1)).ravel().tolist())
            nnz=lt.append_block(f['X'],block,row,nnz)
            row+=400
    b=ad.read_h5ad(path,backed='r')
    group_counts=b.obs.groupby(['context','target_gene'],observed=True).size()
    shape=list(b.shape)
    b.file.close()
    assert len(group_counts)==2 and (group_counts==400).all()
    report={'scope':'Real external RPE1 controls -> two local predictions -> integer CSR AnnData. Engineering smoke check, not an official score or competition submission.',
            'shape':shape,'source_control_cells':len(indices),'targets':targets.tolist(),
            'context':'RPE1_LOCAL_SMOKE','cells_per_target':400,'stored_entries':nnz,
            'min_library_counts':min(libraries),'max_library_counts':max(libraries),
            'mean_library_counts':float(np.mean(libraries)),
            'ntc_mean_max_absolute_difference_from_aggregate':float(np.max(np.abs(mean-query_source.control_mean))),
            'output':str(path.relative_to(root)),'bytes':path.stat().st_size,'all_checks_passed':True,
            'excluded_training_background':'RPE1','source_audit':audit}
    lt.write_json(root/'reports/real_count_smoke.json',report)
    print(json.dumps(report))


if __name__=='__main__':
    main()
