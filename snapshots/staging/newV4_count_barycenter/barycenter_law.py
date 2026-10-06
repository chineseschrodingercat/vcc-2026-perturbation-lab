from __future__ import annotations
import copy
import numpy as np

WEIGHTS=(.25,.5,.75)

def count_barycenter(n1,carrier,weight,count_seed,target_index):
    a=np.asarray(n1);b=np.asarray(carrier);w=float(weight)
    if a.shape!=b.shape or a.ndim!=2 or w not in (0.,.25,.5,.75,1.):raise ValueError('Aligned count matrices and frozen weights only')
    for x in (a,b):
        if not np.isfinite(x).all() or np.any(x<0) or np.any(x>2**24) or not np.array_equal(x,np.rint(x)):raise ValueError('Original nonnegative exact integer endpoints required')
    if w==0:return b.copy(),{}
    if w==1:return a.copy(),{}
    mean=w*a.astype(np.float64)+(1-w)*b.astype(np.float64)
    floor=np.floor(mean).astype(np.int64);frac=mean-floor
    rng=np.random.default_rng(np.random.SeedSequence([int(count_seed),int(target_index),45,0]))
    before=copy.deepcopy(rng.bit_generator.state)
    y=floor+(rng.random(mean.shape)<frac)
    if np.any(y<0) or np.any(y>2**24):raise ArithmeticError('Barycenter integer guard failed; no clip/redraw')
    record=dict(namespace=[int(count_seed),int(target_index),45,0],before=before,after=copy.deepcopy(rng.bit_generator.state),conditional_rounding_variance_sum=float(np.sum(frac*(1-frac))))
    return y,record
