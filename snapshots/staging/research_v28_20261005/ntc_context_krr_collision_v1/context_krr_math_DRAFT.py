"""Unexecuted mathematical draft, no file IO/import side effects or data fits.

Only an exact per-output observed context submatrix is used. This is not a
paper reproduction and not an authorization to run a real experiment.
"""
import numpy as np

LAMBDA=1.0

def component_weights(components):
    c=np.asarray(components)
    return np.asarray([1.0/np.sum(c==v) for v in c],np.float64)

def fit_ntc_geometry(source_logCP,components):
    """Future fit: source global NTCs only; no query/response argument."""
    X=np.asarray(source_logCP,np.float64)
    assert X.ndim==2 and len(X)==len(components) and np.isfinite(X).all()
    w=component_weights(components);w/=w.sum()
    center=np.sum(w[:,None]*X,axis=0)
    sd=np.sqrt(np.sum(w[:,None]*(X-center)**2,axis=0))
    varying=sd>0 # measured zero and exact constants are not unmeasured values.
    assert varying.any()
    Z=(X[:,varying]-center[varying])/sd[varying]
    diff=Z[:,None,:]-Z[None,:,:]
    D=np.mean(diff*diff,axis=2);np.fill_diagonal(D,0.0)
    off=D[np.triu_indices(len(Z),1)];positive=off[off>0]
    assert len(positive),'No identifiable source geometry; abstain, no epsilon'
    bandwidth=float(np.median(positive))
    K=np.exp(-D/bandwidth)
    return dict(center=center,sd=sd,varying=varying,source_Z=Z,bandwidth=bandwidth,K=K)

def query_kernel(geometry,query_logCP):
    """Apply to a complete measurement of the frozen source feature axis."""
    q=np.asarray(query_logCP,np.float64)
    assert q.shape==geometry['center'].shape and np.isfinite(q).all()
    v=geometry['varying'];z=(q[v]-geometry['center'][v])/geometry['sd'][v]
    D=np.mean((geometry['source_Z']-z[None,:])**2,axis=1)
    return np.exp(-D/geometry['bandwidth']),dict(nearest_squared_distance=float(D.min()),
        source_bandwidth=geometry['bandwidth'],maximum_kernel_similarity=float(np.exp(-D/geometry['bandwidth']).max()))

def context_response_prediction(K,k_query,beta,known,components):
    """Future solve for one target/gene; unknown does not enter intercept/Gram.

    Objective sum(w_i*(beta_i - b - f(x_i))**2) + ||f||_H**2.
    Each available closure component has total weight one; b is unpenalized.
    Last block row sum(alpha)=0 enforces the unpenalized intercept exactly.
    """
    K=np.asarray(K,np.float64);k=np.asarray(k_query,np.float64)
    y=np.asarray(beta,np.float64);mask=np.asarray(known,bool);components=np.asarray(components)
    assert K.shape==(len(y),len(y)) and k.shape==y.shape==mask.shape==components.shape
    I=np.flatnonzero(mask)
    if not len(I):return dict(tau=0.0,known=False,reason='no measured supported source; policy H0, not observed zero')
    yy=y[I];assert np.isfinite(yy).all()
    cc=components[I];w=component_weights(cc)
    mean=float(np.average(yy,weights=w))
    if len(set(cc.tolist()))<2:
        return dict(tau=mean,known=True,reason='one component: equal-component mean fallback; no learned context correction',uniform=mean)
    A=np.empty((len(I)+1,len(I)+1),np.float64)
    A[:-1,:-1]=K[np.ix_(I,I)]+LAMBDA*np.diag(1/w)
    A[:-1,-1]=1;A[-1,:-1]=1;A[-1,-1]=0
    rhs=np.r_[yy,0.0];solution=np.linalg.solve(A,rhs)
    pred=float(k[I]@solution[:-1]+solution[-1])
    assert np.isfinite(pred)
    return dict(tau=float(np.clip(pred,-3,3)),known=True,unclipped_tau=pred,uniform=mean,
        reason='fixed masked context RBF KRR',source_contexts=len(I),source_components=len(set(cc.tolist())),
        coefficient_sum=float(solution[:-1].sum()),unpenalized_intercept=float(solution[-1]))

def decode_original_raw_PC05(tau,Cq,editable):
    """Pilot mean readout; original carrier-limited cap16 policy, no counts.

    A later count experiment must check against the literal original
    count_ratio() ABI and actual donor means; equality is not claimed here.
    """
    t=np.asarray(tau,np.float64);C=np.asarray(Cq,np.float64);e=np.asarray(editable,bool)
    assert t.shape==C.shape==e.shape and np.isfinite(C).all() and (C>=0).all()
    rho=np.ones_like(C);I=np.flatnonzero(e & (C>0) & (t!=0))
    assert np.isfinite(t[I]).all()
    wanted=np.maximum((C[I]+.5)*np.exp2(t[I])-.5,0.)
    rho[I]=np.clip(wanted/C[I],0,16)
    return C*rho,rho
