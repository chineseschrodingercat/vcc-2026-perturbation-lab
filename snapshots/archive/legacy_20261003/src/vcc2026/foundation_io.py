"""Bounded-memory raw-cell readers for the audited external H5AD sources."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad

def strings(x):
    return np.asarray([v.decode() if isinstance(v,bytes) else str(v) for v in x])

def field(f,key):
    z=f[key]
    if isinstance(z,h5py.Group):
        return strings(z['categories'][:])[z['codes'][:]]
    if 'categories' in z.attrs:
        return strings(f[z.attrs['categories']][:])[z[:]]
    return strings(z[:])

def metadata(path):
    with h5py.File(path,'r') as f:
        return field(f,'obs/gene'),field(f,'var/gene_name')

def dense_rows(path,rows,columns=None):
    # h5py requires sorted unique indices; restore requested order and repeats.
    unique,inverse=np.unique(np.asarray(rows,dtype=np.int64),return_inverse=True)
    blocks=[]
    with h5py.File(path,'r') as f:
        for start in range(0,len(unique),128):
            b=np.asarray(f['X'][unique[start:start+128],:],dtype=np.float32)
            if columns is not None:b=b[:,columns]
            if not np.isfinite(b).all() or np.any(b<0) or np.any(b!=np.rint(b)):
                raise ValueError('Invalid raw UMI cells')
            blocks.append(b)
    return np.concatenate(blocks)[inverse]

def align(x,genes,destination):
    genes=np.asarray(genes,dtype=str);destination=np.asarray(destination,dtype=str)
    valid=~pd.Index(genes).duplicated(keep=False)
    lookup={g:i for i,g in enumerate(genes) if valid[i]}
    out=np.zeros((len(x),len(destination)),dtype=np.float32)
    dst=np.array([j for j,g in enumerate(destination) if g in lookup])
    out[:,dst]=x[:,[lookup[destination[j]] for j in dst]]
    return sparse.csr_matrix(out),dst

def adata(x,genes):
    return ad.AnnData(sparse.csr_matrix(x),var=pd.DataFrame(index=np.asarray(genes,dtype=str)))
