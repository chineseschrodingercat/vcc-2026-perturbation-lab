"""V4 empirical carrier renderer; no perturbation truth or judge on import."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'newV2_native_lock'))
import native_reproducer as native
from carrier_law import ARMS,posterior_mean,conditional_moments,draw_carrier

class CarrierPanelGenerator(native.CompactPanelGenerator):
    def __init__(self,job,allocation_arm='posterior_augmentation',artifact_root=None):
        if allocation_arm not in ARMS:raise ValueError('Only frozen V4 donor_only/posterior_augmentation')
        super().__init__(job)
        self.allocation_arm=allocation_arm
        self.posterior_dir=Path(artifact_root or HERE/'mechanical_pilot')/self.destination
        self.control_complete=native.read(self.posterior_dir/'complete.json')
        if native.sha(self.panel/'generation_pool.npz')!=self.control_complete['pool_sha256']:raise AssertionError('Exact V4 control pool differs')
        if native.sha(self.posterior_dir/'posterior_calibration.npz')!=self.control_complete['posterior_calibration_sha256']:raise AssertionError('V4 posterior calibration changed')
        self.v3dir=HERE.parent/'newV3_ntc_denoising/native_controls'/self.destination
        if native.sha(self.v3dir/'lambda_pool_F32.npy')!=self.control_complete['V3_lambda_sha256']:raise AssertionError('Frozen V3 state changed')
        if native.sha(self.v3dir/'calibration.npz')!=self.control_complete['V3_calibration_sha256']:raise AssertionError('Frozen V3 theta changed')
        with np.load(self.posterior_dir/'posterior_calibration.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['genes'].astype(str),self.input_genes)
            np.testing.assert_array_equal(z['input_native_positions'],self.input_native)
            np.testing.assert_array_equal(z['input_output_positions'],self.input_output)
            self.control_theta=z['theta_control'].astype(np.float64)
            self.posterior_gain=z['posterior_gain'].astype(np.float64)
            self.control_mu=z['population_mu'].astype(np.float64)
        self.lambda_pool=np.load(self.v3dir/'lambda_pool_F32.npy',mmap_mode='r',allow_pickle=False)
        if self.lambda_pool.shape!=(self.pool.shape[0],len(self.input_genes)):raise AssertionError('V3 state/donor coordinates differ')
        self.source_identity.update(dict(V4_allocation_arm=allocation_arm,V4_posterior_complete=str(self.posterior_dir/'complete.json'),V4_posterior_complete_sha256=native.sha(self.posterior_dir/'complete.json'),V4_law_sha256=native.sha(HERE/'carrier_law.py'),V4_adapter_sha256=native.sha(__file__),V4_control_theta_source='recipient_NTC_only_V3_full_pool_moments',V4_new_response_fit=False,V4_null_is_literal_donor=True))

    def build_target(self,target,arm='N1',ratio_override=None):
        if arm!='N1':raise ValueError('V4 preserves original N1 editable domain')
        f=self._scientific;idx=self.targets.index(str(target));donors=self.donors[target];orderidx=self.order.index(target)
        donor=self.pool[donors].tocsr(copy=True)
        ratio=f['count_ratio'](self.effects[idx],self.mean,self.genes,self.output_genes)
        if ratio_override is not None:
            # Only mechanical null pilot passes an explicit override. Actual generation never does.
            ratio=np.asarray(ratio_override,np.float64)
            if ratio.shape!=(len(self.genes),):raise ValueError('Mechanical ratio override axes differ')
        oldrng=np.random.default_rng(np.random.SeedSequence([self.count_seed,orderidx,0]))
        h0=f['transform_counts'](donor.copy(),ratio,oldrng)
        seen=self.inputs['seen'].astype(bool);own=self.inputs['ownmask'][idx].astype(bool)
        common=np.flatnonzero(seen[self.input_output]&own[self.input_output])
        outpos=self.input_output[common];natpos=self.input_native[common]
        protected=np.ones(len(self.genes),bool);protected[natpos]=False
        raw=donor[:,natpos].toarray()
        if not np.isfinite(raw).all() or np.any(raw<0) or not np.array_equal(raw,np.rint(raw)):raise AssertionError('Original carrier must be integer before dtype conversion')
        x=raw.astype(np.int64);theta=self.control_theta[common]
        if self.allocation_arm=='donor_only':q=x.astype(np.float64)
        else:
            h=np.asarray(self.lambda_pool[donors][:,common],np.float64)
            q=posterior_mean(x,h,theta)*self.posterior_gain[common][None,:]
            if np.any(q[:,self.control_mu[common]==0]>0):raise AssertionError('Zero-mean control gene activated')
        rho=ratio[natpos]
        integer,streams=draw_carrier(x,rho,q,theta,self.count_seed,orderidx)
        mean,var=conditional_moments(x,rho,q,theta)
        dense=h0.toarray();dense[:,natpos]=integer.astype(np.float32)
        block=sparse.csr_matrix(dense,dtype=np.float32);block.eliminate_zeros();block.sort_indices()
        if (block[:,protected]!=h0[:,protected]).nnz:raise AssertionError('Protected original H0 changed')
        if np.any(~np.isfinite(block.data)) or np.any(block.data<=0) or np.any(block.data>2**24) or not np.array_equal(block.data,np.rint(block.data)):raise ArithmeticError('Original integer guard failed; no clip/redraw')
        library=np.asarray(block.sum(1),np.float64).ravel()
        if np.any((library<=0)|(library>=1e6)):raise ArithmeticError('Original library guard failed; no clip/redraw')
        identity=rho==1
        if not np.array_equal(integer[:,identity],x[:,identity]):raise AssertionError('Null edited columns changed')
        if np.any(rho<1) and np.any(integer[:,rho<1]>x[:,rho<1]):raise AssertionError('Downreg generated new molecules')
        if np.any(rho>1) and np.any(integer[:,rho>1]<x[:,rho>1]):raise AssertionError('Upreg removed empirical carrier')
        diagonal_zero=(x==0).astype(np.float64)
        down=rho<1;up=rho>1
        diagonal_zero[:,down]=np.power(1-rho[None,down],x[:,down])
        if np.any(up):
            inc=(rho[None,up]-1)*q[:,up];shape=theta[None,up]+x[:,up]
            diagonal_zero[:,up]=(x[:,up]==0)*np.exp(-shape*np.log1p(inc/shape))
        newzero=(x==0)&(integer>0)
        diagnostics=dict(target=target,arm='V4_'+self.allocation_arm,cells=len(donors),editable_genes=len(common),down_genes=int(down.sum()),identity_genes=int(identity.sum()),up_genes=int(up.sum()),
            count_ratio_native_sha256=f['array_sha'](ratio),donor_pool_rows_sha256=f['array_sha'](donors),count_block_sha256=f['csr_sha'](block),H0_count_block_sha256=f['csr_sha'](h0),
            protected_actual_count_block_sha256=f['csr_sha'](block[:,protected]),protected_H0_count_block_sha256=f['csr_sha'](h0[:,protected]),protected_integer_equality_H0=True,null_columns_byte_equal=True,
            original_donor_pool_sha256=self.control_complete['pool_sha256'],posterior_complete_sha256=native.sha(self.posterior_dir/'complete.json'),
            new_nonzero_at_original_donor_zero=int(newzero.sum()),latent_positive_at_original_donor_zero=int(((q>0)&(x==0)).sum()),
            selected_expected_mean_MAE_vs_literal_Xrho=float(np.mean(np.abs(mean-x*rho[None,:]))),actual_library_mean=float(library.mean()),
            conditional_mean_semantics='empirical carrier X plus upreg posterior/donor increment; downreg literal Binomial X',population_mean_lock_applies_to='uniform draw from complete3000 NTC pool, not selected donors/gene margins',
            new_response_fit=False,destination_perturbation_truth_read=False,new_rng_streams=streams,mechanical_ratio_override_used=ratio_override is not None,control_theta_is_perturbation_specific=False)
        moments=dict(native_genes=self.genes,editable_native_positions=natpos,editable_output_positions=outpos,donor_pool_rows=donors,count_ratio_native=ratio,
            conditional_mean_editable=mean,conditional_variance_editable=var,conditional_zero_probability_editable=diagonal_zero,posterior_allocation_editable=q,control_theta_editable=theta,increment_NB_shape_editable=theta[None,:]+x,actual_native_library=library)
        return native.TargetResult(target,block,diagnostics,moments)

def generate_case(job,out,name=None,allocation_arm='posterior_augmentation',artifact_root=None):
    """Root schedules this count-only entry point and its unchanged Native judge."""
    import anndata as ad
    import pandas as pd
    ad.settings.allow_write_nullable_strings=True
    name=name or 'V4_'+allocation_arm
    out=native.prepare_case(job,out);pred=out/f'pred__{name}.h5ad'
    if pred.exists():raise FileExistsError('Preserve existing V4 counts')
    generator=CarrierPanelGenerator(job,allocation_arm,artifact_root)
    blocks=[];labels=[];records=[]
    for target in generator.targets:
        result=generator.build_target(target);blocks.append(result.block);labels.extend([target]*result.block.shape[0]);records.append(result.diagnostics)
        print('V4_GENERATED',job['role'],target,result.block.shape[0],flush=True)
    namespace={'np':np,'sparse':sparse,'ad':ad,'pd':pd}
    native.extract(native.SOURCE/'benchmark_context_v4.py',('save',),namespace);namespace['save'](pred,blocks,labels,generator.genes)
    native.write(out/f'{name}_generation.json',dict(completed=True,prediction_path=str(pred),prediction_sha256=native.sha(pred),job_key=job['key'],source_identity=generator.source_identity,targets=records,same_original_protected_H0=True,new_response_fit=False,new_judges=0))
    return pred
