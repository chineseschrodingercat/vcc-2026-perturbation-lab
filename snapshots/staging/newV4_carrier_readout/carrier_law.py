"""Empirical donor carrier with binomial removal and NB increments only."""
from __future__ import annotations
import copy
import numpy as np

ARMS=('donor_only','posterior_augmentation')
NAMESPACE=44

def array_inputs(x,rho,q,theta):
    x=np.asarray(x);rho=np.asarray(rho,np.float64);q=np.asarray(q,np.float64);theta=np.asarray(theta,np.float64)
    if x.ndim!=2 or rho.shape!=(x.shape[1],) or q.shape!=x.shape or theta.shape!=(x.shape[1],):raise ValueError('Aligned donor-by-gene arrays required')
    if not np.isfinite(x).all() or np.any(x<0) or not np.array_equal(x,np.rint(x)) or np.any(x>2**24):raise ValueError('Exact nonnegative raw integer X required')
    if not np.isfinite(rho).all() or np.any((rho<0)|(rho>16)):raise ValueError('Original actual reached/capped rho in [0,16] required')
    if not np.isfinite(q).all() or np.any(q<0) or not np.isfinite(theta).all() or np.any(theta<=0):raise ValueError('Finite nonnegative allocation and positive theta required')
    return x.astype(np.int64),rho,q,theta

def posterior_mean(x,h,theta):
    x=np.asarray(x,np.float64);h=np.asarray(h,np.float64);theta=np.asarray(theta,np.float64)
    if x.shape!=h.shape or theta.shape!=(x.shape[1],):raise ValueError('Posterior axes differ')
    if np.any(h<0) or not np.isfinite(h).all() or np.any(theta<=0) or not np.isfinite(theta).all():raise ValueError('Invalid Gamma-Poisson prior')
    if np.any((h==0)&(x>0)):raise ValueError('Degenerate zero prior contradicts observed positive X; no invented fallback')
    # Algebraically equivalent to (theta+X)/(theta/h+1), without division by h.
    return h*((theta[None,:]+x)/(theta[None,:]+h))

def posterior_mean_lock(x,h,theta):
    raw=posterior_mean(x,h,theta);mu=np.asarray(x,np.float64).mean(axis=0);origin=raw.mean(axis=0)
    if np.any((mu>0)&(origin==0)):raise ValueError('No state support for positive population mean')
    gain=np.divide(mu,origin,out=np.zeros_like(mu),where=origin>0)
    q=raw*gain[None,:]
    np.testing.assert_allclose(q.mean(0),mu,rtol=3e-12,atol=3e-12)
    if np.any(q[:,mu==0]!=0):raise AssertionError('True zero-mean gene activated')
    return q,gain

def conditional_moments(x,rho,q,theta):
    x,rho,q,theta=array_inputs(x,rho,q,theta)
    mean=x.astype(np.float64);var=np.zeros(x.shape,np.float64)
    down=rho<1;up=rho>1
    mean[:,down]=x[:,down]*rho[None,down]
    var[:,down]=x[:,down]*(rho[None,down]*(1-rho[None,down]))
    if np.any(up):
        inc=(rho[None,up]-1)*q[:,up];shape=theta[None,up]+x[:,up]
        mean[:,up]=x[:,up]+inc;var[:,up]=inc+inc*inc/shape
    return mean,var

def draw_carrier(x,rho,q,theta,count_seed,target_index):
    """Integer Y; identity genes consume no draw in any random stream."""
    x,rho,q,theta=array_inputs(x,rho,q,theta)
    y=x.copy();down=np.flatnonzero(rho<1);up=np.flatnonzero(rho>1)
    records={}
    if len(down):
        rng=np.random.default_rng(np.random.SeedSequence([int(count_seed),int(target_index),NAMESPACE,0]))
        before=copy.deepcopy(rng.bit_generator.state)
        y[:,down]=rng.binomial(x[:,down],rho[None,down])
        records['binomial']={'namespace':[int(count_seed),int(target_index),NAMESPACE,0],'before':before,'after':copy.deepcopy(rng.bit_generator.state)}
    if len(up):
        shape=theta[None,up]+x[:,up]
        incmean=(rho[None,up]-1)*q[:,up]
        grng=np.random.default_rng(np.random.SeedSequence([int(count_seed),int(target_index),NAMESPACE,1]))
        prng=np.random.default_rng(np.random.SeedSequence([int(count_seed),int(target_index),NAMESPACE,2]))
        gb=copy.deepcopy(grng.bit_generator.state);pb=copy.deepcopy(prng.bit_generator.state)
        gamma=grng.gamma(shape,1./shape)
        increment=prng.poisson(incmean*gamma)
        y[:,up]=x[:,up]+increment
        records['gamma']={'namespace':[int(count_seed),int(target_index),NAMESPACE,1],'before':gb,'after':copy.deepcopy(grng.bit_generator.state)}
        records['poisson']={'namespace':[int(count_seed),int(target_index),NAMESPACE,2],'before':pb,'after':copy.deepcopy(prng.bit_generator.state)}
    identity=rho==1
    if not np.array_equal(y[:,identity],x[:,identity]):raise AssertionError('Null carrier changed')
    if np.any(y<0) or np.any(y>2**24):raise ArithmeticError('Integer boundary failed; no clipping or redraw')
    if len(down) and np.any(y[:,down]>x[:,down]):raise AssertionError('Binomial thinning generated molecules')
    if len(up) and np.any(y[:,up]<x[:,up]):raise AssertionError('Increment-only augmentation removed molecules')
    return y,records
