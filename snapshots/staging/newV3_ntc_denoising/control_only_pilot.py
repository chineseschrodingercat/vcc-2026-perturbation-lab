"""Control-only Native count-split EB state denoiser; no perturbation truth.

This pilot reads the immutable 3000-cell generation pools only. The saved
latent intensities may activate donor zeros but retain each pool gene mean.
Count/judge scheduling belongs to the root and is deliberately absent here.
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT=Path('C:/00 - Competition')
ARCHIVE=ROOT/'archive/legacy_20261003'
OUT=Path(__file__).resolve().parent/'native_controls'
SEED=20261004
S_GRID=(0.,.25,.5,.75,1.)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda:fh.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def fit_state(a,b,fit,seed):
    lib=a.sum(1,dtype=np.float64)
    x=np.log1p(10000*a/np.maximum(lib[:,None],1)).astype(np.float32)
    pca=PCA(n_components=16,svd_solver='randomized',random_state=seed)
    zfit=pca.fit_transform(x[fit]);z=pca.transform(x)
    cluster=KMeans(n_clusters=8,n_init=3,max_iter=100,random_state=seed).fit(zfit)
    labels=cluster.predict(z)
    # The other molecular view supplies training-only conditional labels.
    pi=np.stack([b[fit & (labels==k)].sum(0,dtype=np.float64)+.5 for k in range(8)])
    pi/=pi.sum(1,keepdims=True)
    return {'pca':pca,'cluster':cluster,'pi':pi,'labels':labels,'library_a':lib}

def h_raw(x,state,s,full_depth=True,p=.5):
    scale=1. if full_depth else 1./p
    return ((1-s)*x+s*x.sum(1,dtype=np.float64)[:,None]*state['pi'][state['labels']])*scale

def assign_state(x,state):
    library=x.sum(1,dtype=np.float64)
    transformed=np.log1p(10000*x/np.maximum(library[:,None],1)).astype(np.float32)
    labels=state['cluster'].predict(state['pca'].transform(transformed))
    return {**state,'labels':labels,'library_a':library}

def mean_lock(h,reference,fit):
    target=reference[fit].mean(0,dtype=np.float64)
    origin=h[fit].mean(0,dtype=np.float64)
    gain=np.divide(target,origin,out=np.zeros_like(target),where=origin>0)
    if np.any((target>0)&(origin==0)):raise ArithmeticError('Cannot preserve positive population mean')
    return h*gain[None,:],gain

def poisson_score(y,rate):
    r=np.maximum(rate,1e-8)
    return float(np.mean(r-y*np.log(r)+gammaln(y+1))),float(np.mean((r-y)**2)),float(np.mean((r>=.5)&(y==0)))

def predicted_moments(h,theta):
    mu=h.mean(0,dtype=np.float64);hv=h.var(0,dtype=np.float64)
    var=mu+np.mean(h*h,axis=0,dtype=np.float64)/theta+hv
    zero=np.exp(-theta[None,:]*np.log1p(h/theta[None,:])).mean(0)
    return mu,var,zero

def run_context(context):
    start=time.monotonic();dest=OUT/context
    if (dest/'complete.json').exists():raise FileExistsError('Preserve existing pilot')
    panel=ARCHIVE/f'runs/residual_v17/inner_panels/{context}'
    pool_path=panel/'generation_pool.npz';pool=sparse.load_npz(pool_path).tocsr()
    native=np.loadtxt(panel/'genes.txt',dtype=str,ndmin=1)
    with np.load(ROOT/'data/processed/set_response_v22/encoder.npz',allow_pickle=False) as z:genes=z['genes'].astype(str)
    with np.load(ARCHIVE/'retained/native_theta/outer_hepg2/seed_20261002/dispersion_seen.npz',allow_pickle=False) as z:model_genes=z['gene_symbols'].astype(str)
    nm={g:i for i,g in enumerate(native)};mm={g:i for i,g in enumerate(model_genes)}
    pos=np.array([nm[g] for g in genes],dtype=np.int64);mp=np.array([mm[g] for g in genes],dtype=np.int64)
    x=pool[:,pos].toarray().astype(np.float32)
    if len(x)!=3000 or (x<0).any() or not np.isfinite(x).all() or not np.array_equal(x,np.rint(x)):
        raise ValueError('Requires exact original 3000 integer Native controls')
    metadata=pd.read_csv(ROOT/f'data/processed/set_response_v22/controls/{context}/cell_metadata.csv',dtype={'batch':str,'guide':str,'cell_id':str})
    if len(metadata)!=3000 or not np.array_equal(metadata.cache_row.to_numpy(),np.arange(3000)):
        raise ValueError('NTC metadata donor order differs')
    # Whole technical GEM groups held for NTC-only measurement calibration.
    groups=metadata.batch.astype(str).to_numpy()
    split={g:int(hashlib.sha256(f'v3-ntc-gem:{context}:{g}'.encode()).hexdigest(),16)%5 for g in set(groups)}
    val=np.array([split[g]==0 for g in groups]);fit=~val
    if val.sum()<100 or fit.sum()<100:raise ValueError('Insufficient predeclared NTC group split')
    seed=SEED+sum(map(ord,context));rng=np.random.default_rng(seed)
    a=rng.binomial(x.astype(np.int64),.5).astype(np.float32);b=x-a
    state=fit_state(a,b,fit,seed)
    choices=[]
    for s in S_GRID:
        h,gain=mean_lock(h_raw(a,state,s,False),x,fit)
        nll,mse,fp=poisson_score(b[val],.5*h[val])
        choices.append({'s':s,'control_held_countsplit_poisson_NLL':nll,'control_held_countsplit_MSE':mse,'pred_rate_ge_half_b_zero':fp})
    selected=min(choices,key=lambda q:(q['control_held_countsplit_poisson_NLL'],q['s']))['s']
    # Honest calibration diagnostics: fit-state parameters never see val b.
    hf,_=mean_lock(h_raw(a,state,selected,False),x,fit)
    raw_train=x[fit];ht=hf[fit]
    mu=raw_train.mean(0,dtype=np.float64);observed_var=raw_train.var(0,dtype=np.float64)
    hv=ht.var(0,dtype=np.float64);h2=(ht*ht).mean(0,dtype=np.float64)
    remaining=observed_var-mu-hv
    theta=np.clip(np.divide(h2,np.maximum(remaining,1e-8)),1e-4,1e8)
    theta[remaining<=0]=1e8
    val_pred_mu,val_pred_var,val_pred_zero=predicted_moments(hf[val],theta)
    held={'true_gene_mean':x[val].mean(0,dtype=np.float64),'true_gene_var':x[val].var(0,dtype=np.float64),'true_gene_zero':(x[val]==0).mean(0),'pred_gene_mean':val_pred_mu,'pred_gene_var':val_pred_var,'pred_gene_zero':val_pred_zero}
    held_readings={'control_held_raw_gene_mean_MAE':float(np.mean(np.abs(held['true_gene_mean']-val_pred_mu))),
        'control_held_raw_gene_variance_MAE':float(np.mean(np.abs(held['true_gene_var']-val_pred_var))),
        'control_held_zero_fraction_MAE':float(np.mean(np.abs(held['true_gene_zero']-val_pred_zero))),
        'theta_poisson_limit_genes':int((theta==1e8).sum()),'negative_remaining_genes':int((remaining<0).sum()),
        'latent_from_half_view_variance_warning':'These diagnostics use Xa/p intensities. Final whole-count intensities are recalibrated separately and do not inherit this half-view theta.'}
    dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'control_held_diagnostics.npz',**held,theta_train_only=theta,remaining_train_only=remaining,genes=genes)
    # Final fit still has Xa input/Xb conditional labels; X never serves as its
    # own paired reconstruction label. All 3000 controls are legitimate final
    # unsupervised adaptation data after s has been frozen on the technical split.
    paired_state=fit_state(a,b,np.ones(len(x),bool),seed+17)
    # Apply frozen transform to complete observed counts only at inference.
    full_state=assign_state(x,paired_state)
    full_h,full_gain=mean_lock(h_raw(x,full_state,selected,True),x,np.ones(len(x),bool))
    out_mu=full_h.mean(0);original_mu=x.mean(0,dtype=np.float64)
    exact_error=float(np.max(np.abs(out_mu-original_mu)))
    np.testing.assert_allclose(out_mu,original_mu,rtol=5e-12,atol=5e-12)
    # Complete-pool control observation variance calibration, with no target truth.
    v=x.var(0,dtype=np.float64);latent_v=full_h.var(0);second=(full_h*full_h).mean(0)
    residual=v-original_mu-latent_v
    theta_full=np.clip(np.divide(second,np.maximum(residual,1e-8)),1e-4,1e8)
    theta_full[residual<=0]=1e8
    lam=full_h.astype(np.float32);stored_error=float(np.max(np.abs(lam.mean(0,dtype=np.float64)-original_mu)))
    np.save(dest/'lambda_pool_F32.npy',lam)
    np.savez_compressed(dest/'calibration.npz',genes=genes,input_native_positions=pos,input_output_positions=mp,
        theta_control_moments_F64=theta_full,mu_NTC_F64=original_mu,variance_NTC_F64=v,
        variance_latent_F64=latent_v,second_moment_latent_F64=second,remaining_emission_variance_F64=residual,
        mean_lock_gain_F64=full_gain,cluster_labels=full_state['labels'])
    np.savez_compressed(dest/'state_model.npz',genes=genes,pca_mean=full_state['pca'].mean_,pca_components=full_state['pca'].components_,
        cluster_centers=full_state['cluster'].cluster_centers_,cluster_compositions=full_state['pi'],s=np.array(selected))
    metadata=metadata.copy();metadata['calibration_held']=val;metadata['calibration_split_group']='GEM';metadata.to_csv(dest/'donor_metadata.csv',index=False)
    values={'schema':'new_V3_control_only_state_EB_1','context':context,'status':'CONTROL_PILOT_COMPLETED_NO_PERTURBATION_COUNT_OR_JUDGE',
        'pool_path':str(pool_path),'pool_sha256':sha(pool_path),'pool_rows':len(x),'common_genes':len(genes),
        'native_gene_axis_sha256':sha(panel/'genes.txt'),'metadata_sha256':sha(ROOT/f'data/processed/set_response_v22/controls/{context}/cell_metadata.csv'),
        'selected_s':selected,'selection':'Lowest fixed-grid independent half-view NTC held-GEM Poisson NLL; no perturbation outcomes',
        'fixed_rank':16,'fixed_clusters':8,'CPU_threads':2,'GPU':0,'seed':seed,'choices':choices,'control_held_readings':held_readings,
        'training_NTC_cells':int(fit.sum()),'held_NTC_cells':int(val.sum()),'training_GEMs':int(len(set(groups[fit]))),'held_GEMs':int(len(set(groups[val]))),
        'guide_overlap_training_held':int(len(set(metadata.guide[fit])&set(metadata.guide[val]))),
        'independent_biological_replicates_claimed':False,'reference_NTC_read':False,'perturbation_truth_read':False,
        'population_mean_max_abs_error_F64':exact_error,'population_mean_max_abs_error_stored_F32':stored_error,
        'full_pool_zero_mean_genes':int((original_mu==0).sum()),'latent_positive_at_original_donor_zero_coordinates':int(((lam>0)&(x==0)).sum()),
        'final_negative_remaining_variance_genes':int((residual<0).sum()),'final_poisson_limit_theta_genes':int((theta_full==1e8).sum()),
        'emission_variance_calibration':'theta=E[h²]/max(Var(X)-E[h]-Var(h),1e-8), bounded [1e-4,1e8]; impossible negative residual -> Poisson limit with explicit residual',
        'strict_source_preprocessing_policy_exception':'Declared recipient control-only unsupervised adaptation, not old source-only transform; no receiver perturbation information used',
        'protected_and_own_gene_policy':'Artifact provides latent common-axis values; count adapter applies original seen/own masks and preserves protected realized H0 counts',
        'official_NTC_pretraining_used':False,'official_NTC_pretraining_ablation':'Future separate arm; current true execution Native controls only',
        'state_training_inputs':'Xa~Binomial(X,0.5) only','state_training_conditional_labels':'Xb=X-Xa only; no same full observed X paired reconstruction label',
        'final_state_fit':'All generation NTC after control-held selection; PCA/cluster Xa, prototypes Xb, then frozen transform full X for inference',
        'seconds':time.monotonic()-start,'files_sha256':{leaf:sha(dest/leaf) for leaf in ['lambda_pool_F32.npy','calibration.npz','state_model.npz','donor_metadata.csv','control_held_diagnostics.npz']}}
    write(dest/'complete.json',values)
    print(json.dumps({k:values[k] for k in ['context','selected_s','training_NTC_cells','held_NTC_cells','population_mean_max_abs_error_stored_F32','final_negative_remaining_variance_genes','seconds']},ensure_ascii=False),flush=True)
    return values

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--context',choices=['hepg2','jurkat','rpe1','all'],default='all');args=ap.parse_args()
    with threadpool_limits(limits=2):
        values=[run_context(c) for c in (['hepg2','jurkat','rpe1'] if args.context=='all' else [args.context])]
    write(OUT/'complete.json',{'contexts':[v['context'] for v in values],'count_judges':0,'new_perturbation_counts':0,'GPU':0,'CPU_threads':2,'script_sha256':sha(__file__),'results':values})

if __name__=='__main__':main()
