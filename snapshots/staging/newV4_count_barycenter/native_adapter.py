from __future__ import annotations
from pathlib import Path
import importlib.util,sys
import numpy as np
from scipy import sparse

HERE=Path(__file__).resolve().parent;V4=HERE.parent/'newV4_carrier_readout'
sys.path.insert(0,str(HERE))
from barycenter_law import WEIGHTS,count_barycenter
spec=importlib.util.spec_from_file_location('frozen_V4_posterior_carrier_adapter',V4/'native_adapter.py')
carrier_adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(carrier_adapter)
native=carrier_adapter.native

class BarycenterPanelGenerator:
    def __init__(self,job,weight=.5):
        if float(weight) not in (0.,*WEIGHTS,1.):raise ValueError('Frozen barycenter weights or mechanical endpoints only')
        self.weight=float(weight);self.job=job
        frozen=native.read(V4/'V4_FROZEN_MANIFEST_COMPLETE.json')
        pinned={x['relative_path']:x['sha256'] for x in frozen['files']}
        for leaf in ('carrier_law.py','native_adapter.py'):
            if native.sha(V4/leaf)!=pinned[leaf]:raise AssertionError('Frozen original V4 source changed')
        self.n1=native.CompactPanelGenerator(job,theta_scale=1.)
        self.carrier=carrier_adapter.CarrierPanelGenerator(job,allocation_arm='posterior_augmentation')
        self.targets=self.n1.targets;self.genes=self.n1.genes;self.order=self.n1.order;self.count_seed=self.n1.count_seed
        for name in ('targets','order'):
            if getattr(self.n1,name)!=getattr(self.carrier,name):raise AssertionError('Endpoint target/generation order differs')
        for name in ('genes','output_genes','effects','input_native','input_output'):
            np.testing.assert_array_equal(getattr(self.n1,name),getattr(self.carrier,name))
        for target in self.order:np.testing.assert_array_equal(self.n1.donors[target],self.carrier.donors[target])
        self.source_identity=dict(version='V4_revision2_count_barycenter',weight_N1=self.weight,weight_posterior=1-self.weight,N1_source=self.n1.source_identity,carrier_source=self.carrier.source_identity,original_V4_freeze_sha256=native.sha(V4/'V4_FROZEN_MANIFEST_COMPLETE.json'),barycenter_law_sha256=native.sha(HERE/'barycenter_law.py'),adapter_sha256=native.sha(__file__),per_background_weight_selection=False,additional_response_fit=False)

    def endpoints(self,target):return self.n1.build_target(target),self.carrier.build_target(target)

    def build_target(self,target,endpoints=None):
        n1,carrier=endpoints or self.endpoints(target)
        nativepos=n1.moments['editable_native_positions'];outpos=n1.moments['editable_output_positions']
        np.testing.assert_array_equal(nativepos,carrier.moments['editable_native_positions'])
        np.testing.assert_array_equal(outpos,carrier.moments['editable_output_positions'])
        protected=np.ones(len(self.genes),bool);protected[nativepos]=False
        if (n1.block[:,protected]!=carrier.block[:,protected]).nnz:raise AssertionError('The two original protected H0 blocks differ')
        if self.weight==0:return carrier
        if self.weight==1:return n1
        a=n1.block[:,nativepos].toarray();b=carrier.block[:,nativepos].toarray()
        y,stream=count_barycenter(a,b,self.weight,self.count_seed,self.order.index(target))
        dense=n1.block.toarray();dense[:,nativepos]=y.astype(np.float32)
        block=sparse.csr_matrix(dense,dtype=np.float32);block.eliminate_zeros();block.sort_indices()
        if (block[:,protected]!=n1.block[:,protected]).nnz:raise AssertionError('Original protected H0 changed')
        if np.any(~np.isfinite(block.data)) or np.any(block.data<=0) or np.any(block.data>2**24) or not np.array_equal(block.data,np.rint(block.data)):raise ArithmeticError('Original integer guard failed; no clip/redraw')
        library=np.asarray(block.sum(1),np.float64).ravel()
        if np.any((library<=0)|(library>=1e6)):raise ArithmeticError('Original library guard failed; no clip/redraw')
        mean=self.weight*n1.moments['conditional_mean_editable']+(1-self.weight)*carrier.moments['conditional_mean_editable']
        realized_mean=self.weight*a+(1-self.weight)*b
        frac=realized_mean-np.floor(realized_mean)
        f=self.n1._scientific
        diag=dict(target=target,arm=f'V4_count_barycenter_w{self.weight:g}',cells=block.shape[0],editable_genes=len(nativepos),weight_N1=self.weight,
            integer_count_space_mix=True,old_V122_whole_cell_splice=False,protected_integer_equality_H0=True,
            N1_endpoint_count_sha256=f['csr_sha'](n1.block),posterior_endpoint_count_sha256=f['csr_sha'](carrier.block),
            count_block_sha256=f['csr_sha'](block),protected_actual_count_block_sha256=f['csr_sha'](block[:,protected]),protected_H0_count_block_sha256=f['csr_sha'](n1.block[:,protected]),
            donor_pool_rows_sha256=f['array_sha'](n1.moments['donor_pool_rows']),count_ratio_native_sha256=f['array_sha'](n1.moments['count_ratio_native']),
            mean_semantics='w literal Xrho + (1-w) carrier conditional mean; complete-pool population target mu*rho; realized endpoints rounded unbiased',
            selected_mean_MAE_vs_literal_Xrho=float(np.mean(np.abs(mean-n1.moments['conditional_mean_editable']))),
            covariance_and_libraries_exactly_locked=False,actual_library_mean=float(library.mean()),rounding_stream=stream,new_response_fit=False,destination_perturbation_truth_read=False)
        moments=dict(native_genes=self.genes,editable_native_positions=nativepos,editable_output_positions=outpos,donor_pool_rows=n1.moments['donor_pool_rows'],count_ratio_native=n1.moments['count_ratio_native'],
            conditional_mean_editable=mean,realized_endpoint_mixture_mean_editable=realized_mean,conditional_rounding_variance_editable=frac*(1-frac),
            N1_conditional_variance_editable=n1.moments['conditional_variance_editable'],carrier_conditional_variance_editable=carrier.moments['conditional_variance_editable'],actual_native_library=library)
        return native.TargetResult(target,block,diag,moments)

def generate_case(job,out,name=None,weight=.5):
    if float(weight) not in WEIGHTS:raise ValueError('Only three predeclared actual arms')
    import anndata as ad
    import pandas as pd
    ad.settings.allow_write_nullable_strings=True
    name=name or f'V4_count_barycenter_w{weight:g}'
    out=native.prepare_case(job,out);pred=out/f'pred__{name}.h5ad'
    if pred.exists():raise FileExistsError('Preserve existing barycenter count panels')
    generator=BarycenterPanelGenerator(job,weight);blocks=[];labels=[];diags=[]
    for target in generator.targets:
        result=generator.build_target(target);blocks.append(result.block);labels.extend([target]*result.block.shape[0]);diags.append(result.diagnostics)
        print('V4_BARYCENTER_GENERATED',job['role'],target,weight,result.block.shape[0],flush=True)
    namespace={'np':np,'sparse':sparse,'ad':ad,'pd':pd}
    native.extract(native.SOURCE/'benchmark_context_v4.py',('save',),namespace);namespace['save'](pred,blocks,labels,generator.genes)
    native.write(out/f'{name}_generation.json',dict(completed=True,job_key=job['key'],prediction_path=str(pred),prediction_sha256=native.sha(pred),source_identity=generator.source_identity,targets=diags,new_judges=0,new_response_fit=False))
    return pred
