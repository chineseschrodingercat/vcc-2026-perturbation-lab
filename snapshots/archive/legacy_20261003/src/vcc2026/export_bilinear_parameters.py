"""Rebuild and save the frozen bilinear winner without reselecting on test data."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from select_transfer_model import read_source,profile,sha256
from validate_bilinear_targets import basis

def main():
    run=Path('runs/model_selection_v2/bilinear_unseen_targets');selection=json.loads((run/'selection.json').read_text())
    key=selection['selected_model']
    if key!='bilinear_train_mean_k32_l100':raise ValueError('Frozen winner changed; review before export')
    out=Path('models/bilinear_v2');out.mkdir(parents=True,exist_ok=True)
    if (out/'parameters.npz').exists():raise FileExistsError(out/'parameters.npz')
    src=read_source('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz')
    split=pd.read_csv('runs/baseline_v1/unseen_target_validation/split_manifest.csv').set_index('target_gene')
    official=set(np.loadtxt('data/raw/vcc2026/validation/files/gene_names.csv',dtype=str,skiprows=1))
    idx=np.array([i for i,g in enumerate(src['gene_names']) if g in official]);targets=src['target_names']
    p=np.zeros((len(targets),64),np.float32);found=np.zeros(len(targets),bool)
    with h5py.File('data/raw/external/slim_embeddings/gene_string_embeddings.v0.3.h5') as f:
        for i,t in enumerate(targets):
            if t in f:p[i]=f[t][()];found[i]=True
    keep=np.array([t in split.index and split.loc[t,'split']=='train' for t in targets])&found&(src['target_n_cells']>=30)
    y=profile(src['target_mean_counts'][keep][:,idx])-profile(src['control_mean_counts'][idx]);p=p[keep].astype(np.float64)
    b=y.mean(0);yc=y-b;g=basis(yc,32).astype(np.float64)
    z=np.linalg.solve(g.T@g+100*np.eye(32),g.T@yc.T)
    w=np.linalg.solve(p.T@p+100*np.eye(64),(z@p).T).T
    np.savez_compressed(out/'parameters.npz',gene_symbols=src['gene_names'][idx],training_targets=targets[keep],
        g=g.astype(np.float32),w=w.astype(np.float32),b=b.astype(np.float32),control=src['control_mean_counts'][idx])
    (out/'config.json').write_text(json.dumps({**selection,'model_id':key,'rank':32,'ridge_lambda':100,
        'seed':20260927,'source_selection_hash':sha256(run/'selection.json'),'validation_status':'Not selected for competition fallback; official target MSE worse than zero'},indent=2)+'\n')
    print('Exported',key,int(keep.sum()),flush=True)

if __name__=='__main__':main()
