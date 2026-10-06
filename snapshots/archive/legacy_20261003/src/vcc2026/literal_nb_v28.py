"""V28: exact historical V13 donor-by-gene means, frozen-theta NB cells.

The conditional mean intervention is literal B=X_NTC*historical_V13_ratio.
No model object, model forward, source set, latent anchor, IPF or fit is used.
The frozen role decoder checkpoint supplies dispersion parameters only. H0
is the original sparse-round generator and N1 protects its realized counts
outside the predeclared common/seen/own editable axis.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse
import torch

import latent_counts_v26 as base
import latent_counts_v27 as v27
from latent_flow_v25 import config as resolved_decoder_config
from residual_v6_common import axes, count_ratio, dump, generate as old_generate
from generate_counts import transform_counts
from prepare_response_inputs_v23 import own_for
from train_latent_flow_v25 import directory

PROTOCOL=Path('configs/literal_nb_v28_protocol.json')
ARMS=('H0','N1')
sha=base.sha
array_sha=base.array_sha
csr_sha=base.csr_sha
load_teacher_effect=v27.load_teacher_effect
GenerationValidationError=base.GenerationValidationError


def validate_protocol(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value['primary_arm']!='N1' or set(value['arms'])!=set(ARMS):
        raise ValueError('V28 must have the sole new N1 primary and literal H0')
    if value['effect_source']!='matched_V13' or value['fixed_model']['new_full_fits']!=0:
        raise ValueError('V28 cannot change the saved V13 mean or fit models')
    if value['fixed_model']['candidate']!='flow_nb' or value['fixed_model']['checkpoint']!=3200:
        raise ValueError('V28 requires the fixed V25 role pairing for theta provenance')
    if value['fixed_model']['decoder_checkpoint']!=3000:raise ValueError('V28 decoder step differs')
    if not value['intervention']['literal_B_coordinate_preserved']:
        raise ValueError('V28 must preserve literal conditional B coordinates')
    for field in ['IPF','decoder_forward','flow_forward','encoder_forward','V1_anchor','source_condition_draws','new_theta_fit','new_mean_fit']:
        if value['intervention'][field]:raise ValueError(f'Forbidden V28 intervention: {field}')
    for field,expected in [('donor_seed_offset',100),('protected_namespace',[0]),('nb_namespace',[26,3])]:
        if value['rng'][field]!=expected:raise ValueError(f'V28 RNG mismatch: {field}')
    if value['count_guard']['maximum_exact_float32_integer']!=2**24:
        raise ValueError('V28 integer guard differs from the frozen mechanism')
    if value['count_guard']['maximum_row_library_exclusive']!=1000000:
        raise ValueError('V28 row library guard differs from the frozen mechanism')
    return value


class PanelGenerator:
    """Read frozen identities, masks and theta without constructing a model."""
    _pool_metadata=base.PanelGenerator._pool_metadata

    def __init__(self,panel,effects,role,model_seed,count_seed,protocol_path=PROTOCOL,
                 device='cuda',teacher_provenance_path=None):
        self.panel=Path(panel);self.role=str(role);self.model_seed=int(model_seed)
        self.count_seed=int(count_seed);self.device=torch.device(device)
        self.protocol_path=Path(protocol_path);self.protocol=validate_protocol(protocol_path)
        self.info=json.loads((self.panel/'panel.json').read_text(encoding='utf-8'))
        self.targets=list(map(str,self.info['targets']));self.destination=self.info['context']
        self.genes=np.loadtxt(self.panel/'genes.txt',dtype=str);self.output_genes=axes()[0]
        self.pool=sparse.load_npz(self.panel/'generation_pool.npz').tocsr()
        self.mean=np.asarray(self.pool.mean(0)).ravel().astype(np.float64)
        self.effects=np.asarray(effects,np.float32)
        if self.effects.shape!=(len(self.targets),len(self.output_genes)) or not np.isfinite(self.effects).all():
            raise ValueError('Actual saved V13 effects disagree with locked historical axes')
        if len(set(self.targets))!=len(self.targets) or len(set(self.genes))!=len(self.genes):
            raise ValueError('Duplicate target or measured native gene identity')
        self.order=list(map(str,self.info.get('generation_targets',self.targets)))
        self.sizes=self.info.get('generation_target_cells',self.info['target_cells'])
        rng=np.random.default_rng(self.count_seed+100)
        self.donor_rng_before=rng.bit_generator.state
        self.donors={t:rng.choice(self.pool.shape[0],self.sizes[t],replace=False) for t in self.order}
        self.donor_rng_after=rng.bit_generator.state
        if not set(self.targets).issubset(self.donors):raise ValueError('Evaluated target absent from original generation order')
        # Only encoder gene identities are read; no encoder arrays or forward.
        with np.load('data/processed/set_response_v22/encoder.npz',allow_pickle=False) as z:
            self.input_genes=z['genes'].astype(str)
        native_map={g:i for i,g in enumerate(self.genes)}
        output_map={g:i for i,g in enumerate(self.output_genes)}
        if not all(g in native_map and g in output_map for g in self.input_genes):
            raise ValueError('Common input gene missing on locked native/model axes')
        self.input_native=np.asarray([native_map[g] for g in self.input_genes],np.int64)
        self.input_output=np.asarray([output_map[g] for g in self.input_genes],np.int64)
        control_meta=pd.read_csv(Path('data/processed/set_response_v22/controls')/self.destination/'cell_metadata.csv')
        self.cache={'control_metadata':{self.destination:control_meta}}
        self.pool_metadata=self._pool_metadata()
        self.dest=directory(self.role,'flow_nb',self.model_seed)
        self.decoder_dest=directory(self.role,'decoder',self.model_seed)
        ck=torch.load(self.dest/'checkpoint_3200.pt',map_location='cpu',weights_only=False)
        dck=torch.load(self.decoder_dest/'checkpoint_final.pt',map_location='cpu',weights_only=False)
        self.checkpoint_manifest=ck['manifest'];self.decoder_manifest=dck['manifest']
        if ck['step']!=3200 or dck['step']!=3000:raise ValueError('Frozen role checkpoint step differs')
        for checkpoint in (ck,dck):
            if checkpoint['manifest']['trainname']!=self.role or checkpoint['manifest']['seed']!=self.model_seed:
                raise ValueError('Frozen theta role/model seed pairing differs')
        if self.destination in ck['manifest']['training_contexts'] or self.destination in dck['manifest']['decoder_contexts']:
            raise ValueError('Held-out destination was used to fit the frozen model')
        if ck['manifest']['decoder_checkpoint_sha256']!=sha(self.decoder_dest/'checkpoint_final.pt'):
            raise ValueError('Frozen response role links a different decoder checkpoint')
        for manifest in (ck['manifest'],dck['manifest']):
            for field,source in [('protocol_sha256','configs/latent_flow_v25_protocol.json'),
                                 ('model_sha256','src/vcc2026/latent_flow_v25.py'),
                                 ('trainer_sha256','src/vcc2026/train_latent_flow_v25.py')]:
                if manifest[field]!=sha(source):raise ValueError(f'Frozen V25 {field} drift')
        if ck['manifest']['features_sha256']!=sha(self.dest/'features.npz'):
            raise ValueError('Frozen role features drift')
        with np.load(self.dest/'features.npz',allow_pickle=False) as z:
            seen=z['seen'].astype(np.float32)
        self.inputs={'seen':seen,'ownmask':own_for(self.targets,self.output_genes)}
        if seen.shape!=(len(self.output_genes),):raise ValueError('Role-seen mask axis differs')
        self.resolved_model_config=resolved_decoder_config(dck['manifest']['resolved_model_config'])
        raw_theta=dck['state_dict']['dispersion_unconstrained']
        if tuple(raw_theta.shape)!=(len(self.output_genes),):raise ValueError('Frozen theta gene axis differs')
        # Predeclared CPU float32 parameter transform only. The retained old
        # CUDA transform may differ by one ULP; this is no neural forward.
        with torch.no_grad():
            self.theta=(torch.nn.functional.softplus(raw_theta.cpu())+self.resolved_model_config['theta_floor']).numpy().astype(np.float64)
        if not np.isfinite(self.theta).all() or np.any(self.theta<=0):raise ValueError('Frozen NB theta is invalid')
        self.theta_parameter_sha256=array_sha(raw_theta.cpu().numpy())
        self.theta_sha256=array_sha(self.theta)
        del ck,dck
        self.conditions=[];self.source_draws=[]
        self.v28_teacher_provenance=None
        if teacher_provenance_path is not None:
            self.v28_teacher_provenance=v27.teacher_provenance(teacher_provenance_path,self.effects,self.panel)
            if int(self.v28_teacher_provenance['count_seed'])!=self.count_seed:
                raise ValueError('Actual teacher/count seed pairing differs')
        self.v27_teacher_provenance=self.v28_teacher_provenance
        self.dependencies=self._dependencies()
        if self.v28_teacher_provenance is not None:
            value=self.v28_teacher_provenance
            self.dependencies[value['provenance_path']]=value['provenance_sha256']
            self.dependencies[value['teacher_effect_path']]=value['teacher_effect_sha256']
            self.dependencies.update(value['validated_teacher_and_reference_paths_sha256'])
            if value.get('teacher_effect_metadata_path'):
                self.dependencies[value['teacher_effect_metadata_path']]=value['teacher_effect_metadata_sha256']

    def _dependencies(self):
        paths=[self.panel/'panel.json',self.panel/'generation_pool.npz',self.panel/'genes.txt',
               self.dest/'checkpoint_3200.pt',self.dest/'features.npz',self.dest/'manifest.json',
               self.decoder_dest/'checkpoint_final.pt',self.decoder_dest/'manifest.json',
               self.protocol_path,Path('configs/latent_flow_v25_protocol.json'),
               Path('configs/baseline_v1.json'),Path('runs/context_v4/gene_axis.txt'),
               Path('data/processed/set_response_v22/encoder.npz'),
               Path('data/processed/set_response_v22/complete.json'),
               Path('data/processed/set_response_v22/input_genes.txt'),
               Path('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz'),
               Path('src/vcc2026/literal_nb_v28.py'),Path('src/vcc2026/latent_counts_v27.py'),
               Path('src/vcc2026/latent_counts_v26.py'),Path('src/vcc2026/latent_flow_v25.py'),
               Path('src/vcc2026/train_latent_flow_v25.py'),Path('src/vcc2026/prepare_response_inputs_v23.py'),
               Path('src/vcc2026/prepare_set_response_v22.py'),Path('src/vcc2026/generate_counts.py'),
               Path('src/vcc2026/residual_v6_common.py'),Path('src/vcc2026/benchmark_context_v4.py'),
               Path('runs/residual_v6/inner_panels')/self.destination/'controls.csv']
        controls=Path('data/processed/set_response_v22/controls')/self.destination
        paths.extend([controls/'cell_metadata.csv',controls/'measurement_mask.npy',
                      base.old_control_cache(self.destination)/'control_core_counts.npz',
                      base.old_control_cache(self.destination)/'control_genes.txt'])
        if (self.panel/'controls.csv').exists():paths.append(self.panel/'controls.csv')
        return {str(path):sha(path) for path in paths}

    def build_target(self,target,arm):
        if arm not in ARMS:raise ValueError(arm)
        index=self.targets.index(str(target));donors=self.donors[target];order_index=self.order.index(target)
        donor=self.pool[donors].tocsr(copy=True)
        ratio=count_ratio(self.effects[index],self.mean,self.genes,self.output_genes)
        old_rng=np.random.default_rng(np.random.SeedSequence([self.count_seed,order_index,0]))
        h0=transform_counts(donor.copy(),ratio,old_rng)
        seen=self.inputs['seen'].astype(bool);own=self.inputs['ownmask'][index].astype(bool)
        editable_output=self.input_output[seen[self.input_output]&own[self.input_output]]
        native_map={g:i for i,g in enumerate(self.genes)}
        editable_native=np.asarray([native_map[self.output_genes[g]] for g in editable_output],np.int64)
        protected=np.ones(len(self.genes),bool);protected[editable_native]=False
        literal_B=donor.toarray().astype(np.float64)*ratio[None,:]
        mu=literal_B[:,editable_native].copy()
        literal_coordinate_equal=np.array_equal(mu,literal_B[:,editable_native])
        if not literal_coordinate_equal:raise AssertionError('N1 conditional mean is not literal V13 B')
        theta=self.theta[editable_output]
        sampler_info={'sampler':'historical_sparse_stochastic_round',
                      'seed_sequence_entropy':[self.count_seed,order_index,0],
                      'NB_rejection_sampler_gene_pairing_claimed':False}
        block=h0
        if arm=='N1':
            integer,sampler_info=base.integer_sample(mu,theta,self.count_seed,order_index,'nb')
            if np.any(integer[mu==0]!=0):
                raise GenerationValidationError('NB with literal zero mean emitted a nonzero count',failed_draw=integer,rng=sampler_info)
            dense=h0.toarray();dense[:,editable_native]=integer
            block=sparse.csr_matrix(dense,dtype=np.float32);block.eliminate_zeros();block.sort_indices()
        if (block[:,protected]!=h0[:,protected]).nnz:raise AssertionError('Protected literal H0 integer block changed')
        if np.any(~np.isfinite(block.data)) or np.any(block.data<=0) or not np.array_equal(block.data,np.rint(block.data)) or np.any(block.data>2**24):
            raise GenerationValidationError('Integer count guard violated; do not clip or redraw',failed_draw=block.toarray(),rng=sampler_info)
        library=np.asarray(block.sum(1),np.float64).ravel();h0lib=np.asarray(h0.sum(1),np.float64).ravel()
        if np.any(library<=0) or np.any(library>=1e6):
            raise GenerationValidationError('Row library guard violated; do not clip or redraw',failed_draw=block.toarray(),rng=sampler_info)
        sampled=block.toarray().astype(np.float64);h0dense=h0.toarray().astype(np.float64)
        fraction=literal_B-np.floor(literal_B)
        shared_noise=fraction*(1-fraction);shared_zero=np.maximum(1-literal_B,0)
        if arm=='N1':
            shared_noise[:,editable_native]=mu+mu**2/theta[None,:]
            shared_zero[:,editable_native]=np.exp(-theta[None,:]*np.log1p(mu/theta[None,:]))
        conditional_expected=literal_B.copy();conditional_noise=shared_noise.copy();conditional_zero=shared_zero.copy()
        if arm=='N1':
            conditional_expected[:,protected]=h0dense[:,protected]
            conditional_noise[:,protected]=0.
            conditional_zero[:,protected]=(h0dense[:,protected]==0).astype(np.float64)
        moments={'native_genes':self.genes,'editable_native_positions':editable_native,
                 'editable_output_positions':editable_output,'donor_pool_rows':donors,
                 'literal_B_editable':mu,'conditional_mean_editable':mu.copy(),
                 'conditional_variance_editable':(mu+mu**2/theta[None,:] if arm=='N1' else (mu-np.floor(mu))*(1-(mu-np.floor(mu)))),
                 'conditional_zero_probability_editable':(np.exp(-theta[None,:]*np.log1p(mu/theta[None,:])) if arm=='N1' else np.maximum(1-mu,0)),
                 'count_ratio_native':ratio,'frozen_theta_editable':theta,
                 'expected_gene_mean':literal_B.mean(0),'G0_expected_gene_mean':literal_B.mean(0),
                 'H0_expected_gene_mean':literal_B.mean(0),
                 'expected_gene_variance':shared_noise.mean(0)+literal_B.var(0),
                 'conditional_mu_gene_variance':literal_B.var(0),
                 'expected_gene_zero_fraction':shared_zero.mean(0),
                 'sample_gene_mean':sampled.mean(0),'sample_gene_variance':sampled.var(0),
                 'sample_gene_zero_fraction':(sampled==0).mean(0),
                 'G0_sample_gene_mean':h0dense.mean(0),'G0_sample_gene_variance':h0dense.var(0),
                 'G0_sample_gene_zero_fraction':(h0dense==0).mean(0),
                 'actual_native_library':library,'G0_actual_native_library':h0lib,
                 'expected_native_library':literal_B.sum(1),'G0_expected_native_library':literal_B.sum(1),
                 'expected_editable_library':mu.sum(1),'G0_expected_editable_library':mu.sum(1),
                 'sample_editable_library':sampled[:,editable_native].sum(1),
                 'conditional_given_G0_expected_gene_mean':conditional_expected.mean(0),
                 'conditional_given_G0_expected_gene_variance':conditional_noise.mean(0)+conditional_expected.var(0),
                 'conditional_given_G0_expected_gene_zero_fraction':conditional_zero.mean(0),
                 'conditional_given_G0_expected_native_library':conditional_expected.sum(1)}
        moments.update({k.replace('G0','H0'):v for k,v in list(moments.items()) if 'G0' in k})
        original_zero=donor[:,editable_native].toarray()==0
        newly_positive=(sampled[:,editable_native]>0)&original_zero
        raw_zero_count=int(newly_positive.sum())
        if raw_zero_count:raise AssertionError('Literal mean NB introduced count support at original donor zeros')
        diag={'target':target,'arm':arm,'cells':len(donors),'native_genes':len(self.genes),
              'editable_genes':len(editable_native),'protected_genes':int(protected.sum()),
              'mean_source':'matched_V13_literal_donor_by_gene_B','literal_conditional_mean_equal_H0_B':literal_coordinate_equal,
              'literal_B_editable_sha256':array_sha(mu),'conditional_mean_editable_sha256':array_sha(mu),
              'literal_B_native_sha256':array_sha(literal_B),'count_ratio_native_sha256':array_sha(ratio),
              'conditional_given_H0_native_mean_sha256':array_sha(conditional_expected),
              'sampler':sampler_info,'protected_integer_equality':True,'protected_integer_equality_H0':True,
              'protected_H0_count_block_sha256':csr_sha(h0[:,protected]),
              'protected_actual_count_block_sha256':csr_sha(block[:,protected]),
              'integer_nonnegative_finite_exact_float32':True,'row_library_guard_passed':True,
              'zero_literal_mu_emits_zero':bool(np.all(sampled[:,editable_native][mu==0]==0)),
              'literal_zero_mean_coordinates':int((mu==0).sum()),
              'encoder_forward_calls':0,'decoder_forward_calls':0,'flow_forward_calls':0,
              'source_condition_draws':0,'V1_anchor_used':False,'IPF':{'applied':False,'reason':'literal conditional B already retained coordinatewise'},
              'IPF_calls':0,'neural_forward_calls':0,'IPF_applied':False,
              'expected_gene_mean_max_absolute_difference_G0':0.,
              'expected_native_library_max_relative_difference_G0':0.,
              'sample_gene_mean_MAE_vs_expected':float(np.mean(np.abs(sampled.mean(0)-literal_B.mean(0)))),
              'editable_sample_gene_mean_MAE_vs_expected':float(np.mean(np.abs(sampled[:,editable_native].mean(0)-mu.mean(0)))) if len(editable_native) else 0.,
              'sample_gene_mean_MAE_vs_conditional_given_G0_expectation':float(np.mean(np.abs(sampled.mean(0)-conditional_expected.mean(0)))),
              'sample_native_zero_fraction':float((sampled==0).mean()),'G0_sample_native_zero_fraction':float((h0dense==0).mean()),
              'sample_native_mean_gene_variance':float(sampled.var(0).mean()),
              'G0_sample_native_mean_gene_variance':float(h0dense.var(0).mean()),
              'actual_native_library':base.quantiles(library),'G0_actual_native_library':base.quantiles(h0lib),
              'actual_native_library_MAE_vs_G0':float(np.mean(np.abs(library-h0lib))),
              'actual_native_library_mean_difference_G0':float(np.mean(library-h0lib)),
              'actual_library_correlation_G0':base._corr(library,h0lib),
              'editable_new_nonzero_coordinates':raw_zero_count,
              'editable_new_nonzero_fraction_of_original_zeros':float(raw_zero_count/max(1,original_zero.sum())),
              'theta_editable':base.quantiles(theta) if len(theta) else None,
              'count_block_sha256':csr_sha(block),'G0_count_block_sha256':csr_sha(h0),'H0_count_block_sha256':csr_sha(h0),
              'donor_pool_rows_sha256':array_sha(donors),'editable_output_positions_sha256':array_sha(editable_output),
              'original_NTC_common_library':base.quantiles(np.asarray(donor[:,self.input_native].sum(1)).ravel()),
              'new_mean_conditional_on_frozen_model_not_destination_truth':True,
              'full_axis_expectation_fields':'expected_* retain shared unrounded H0 B; conditional_given_H0 and G0 aliases condition on protected realized H0 integers',
              'latent_diagnostics':{}}
        return base.TargetResult(target,block,diag,moments)

    def iter_target(self,arm):
        for target in self.targets:yield self.build_target(target,arm)

    def write_provenance(self,records_dir):
        records=Path(records_dir);records.mkdir(parents=True,exist_ok=True)
        offset=0;rows=[]
        for target in self.targets:
            ids=self.donors[target];part=self.pool_metadata.iloc[ids].copy()
            part['generated_target']=target;part['generated_row_id']=[f'c_{offset+i}' for i in range(len(ids))]
            part['generated_row_offset']=np.arange(offset,offset+len(ids));offset+=len(ids);rows.append(part)
        csv=records/'donor_provenance.csv';pd.concat(rows,ignore_index=True).to_csv(csv,index=False)
        return csv


def generate(panel_work,name,effects,count_seed,*,model_role,model_seed,arm,
             records_dir,protocol_path=PROTOCOL):
    panel=Path(panel_work);records=Path(records_dir);dest=panel/f'pred__{name}.h5ad'
    if arm not in ARMS:raise ValueError(arm)
    if dest.exists() or (records/'generation_manifest.json').exists():raise FileExistsError('Preserve existing V28 generation artifacts')
    records.mkdir(parents=True,exist_ok=True)
    provenance=records/'teacher_effect_provenance.json'
    if not provenance.exists():raise FileNotFoundError('V28 requires actual matched V13 teacher provenance')
    start=time.monotonic()
    generator=PanelGenerator(panel,effects,model_role,model_seed,count_seed,protocol_path,teacher_provenance_path=provenance)
    csv=generator.write_provenance(records)
    if arm=='H0':old_generate(panel,name,effects,count_seed)
    blocks=[];labels=[];diags=[];paths=[]
    for target in generator.targets:
        try:result=generator.build_target(target,arm)
        except Exception as error:
            failure={'failed':True,'target':target,'arm':arm,'exception_type':type(error).__name__,
                     'exception':str(error),'clipped_or_redrawn':False,'dependencies_sha256':generator.dependencies,
                     'model_role':model_role,'model_seed':int(model_seed),'count_seed':int(count_seed),
                     'mean_source':'matched_V13_literal_donor_by_gene_B','new_full_fits':0}
            if isinstance(error,GenerationValidationError):
                if error.failed_draw is not None:
                    np.savez_compressed(records/'failed_draw.npz',draw=error.failed_draw)
                    failure['failed_draw_sha256']=sha(records/'failed_draw.npz')
                if error.rng is not None:failure['rng']=error.rng
            dump(records/'generation_failure.json',failure);raise
        target_dir=records/'targets';target_dir.mkdir(exist_ok=True)
        moments=target_dir/f'{target}.npz';np.savez_compressed(moments,**result.moments)
        info=target_dir/f'{target}.json';dump(info,result.diagnostics)
        paths.extend([str(moments),str(info)]);diags.append(result.diagnostics)
        if arm!='H0':blocks.append(result.block);labels.extend([target]*result.block.shape[0])
        print('V28_GENERATED_TARGET',arm,target,result.block.shape[0],'literal_B_preserved',flush=True)
    if arm!='H0':
        from benchmark_context_v4 import save
        save(dest,blocks,labels,generator.genes)
    targets_csv=records/'target_diagnostics.csv'
    pd.DataFrame([{k:v for k,v in d.items() if not isinstance(v,(dict,list))} for d in diags]).to_csv(targets_csv,index=False)
    value={'schema':'v28_literal_V13_conditional_mean_frozen_theta_NB_1','completed':True,'finalfinished':True,
           'protocol_sha256':sha(protocol_path),'generator_sha256':sha('src/vcc2026/literal_nb_v28.py'),
           'base_v27_generator_sha256':sha('src/vcc2026/latent_counts_v27.py'),
           'base_v26_generator_sha256':sha('src/vcc2026/latent_counts_v26.py'),
           'arm':arm,'generator_changed':arm!='H0','primary_arm':'N1',
           'mean_source':'matched_V13_literal_donor_by_gene_B','V25_predicted_mean_used':False,
           'teacher_effect_sha256':generator.v28_teacher_provenance['teacher_effect_sha256'],
           'teacher_provenance':generator.v28_teacher_provenance,'model_role':model_role,
           'model_seed':int(model_seed),'count_seed':int(count_seed),'fixed_flow_checkpoint_step':3200,
           'fixed_decoder_checkpoint_step':3000,'new_full_fits':0,'optimizer_updates':0,
           'frozen_checkpoint_parameters_read':True,'flow_checkpoint_used_for_pairing_provenance_only':True,
           'model_objects_instantiated':0,'encoder_forward_calls':0,'decoder_forward_calls':0,
           'flow_forward_calls':0,'neural_forward_calls':0,'IPF_calls':0,'IPF_applied':False,
           'source_condition_draws':0,'V1_anchor_used':False,
           'destination':generator.destination,'targets':generator.targets,
           'native_genes_sha256':array_sha(generator.genes),'effects_array_sha256':array_sha(generator.effects),
           'generation_target_order':generator.order,'donor_rng_before':generator.donor_rng_before,
           'donor_rng_after':generator.donor_rng_after,
           'same_H0_donors_selected_before_any_new_sampler_draw':True,
           'same_G0_donors_selected_before_any_new_sampler_draw':True,
           'donor_provenance_count_verified_on_all_6119_common_genes':True,
           'source_conditions':{'views':0,'used':False},'anchor':None,
           'editable_axis':'unchanged input6119 intersect V25 role-seen own-gene-mask and native measured genes',
           'mean_match':'literal H0 V13 unrounded B preserved coordinatewise on editable E; protected realized H0 fixed',
           'baseline_moment_field_prefix':'H0, with G0 aliases for the same literal V13 baseline',
           'IPF':{'applied':False,'reason':'literal donor-by-gene B is used directly'},
           'sampler':'unchanged literal historical sparse stochastic round' if arm=='H0' else 'numpy_negative_binomial_NB2',
           'NB_inverse_dispersion_source':'V25 role-specific frozen decoder parameter softplus(raw theta)+unchanged default floor',
           'theta_floor':generator.resolved_model_config['theta_floor'],
           'theta_parameter_sha256':generator.theta_parameter_sha256,'theta_sha256':generator.theta_sha256,
           'NB_rejection_sampler_gene_pairing_claimed':False,'actual_row_totals_forced':False,
           'RNG_namespaces':{'protected_H0':[0],'NB':[26,3]},
           'destination_perturbation_outcomes_read':False,'whole_destination_heldout_from_training':True,
           'expected_count_views_averaged_before_integer_sampling':False,
           'all_protected_integer_equal_G0':all(v['protected_integer_equality'] for v in diags),
           'all_protected_integer_equal_H0':all(v['protected_integer_equality_H0'] for v in diags),
           'all_literal_conditional_mean_equal_H0':all(v['literal_conditional_mean_equal_H0_B'] for v in diags),
           'all_literal_B_coordinates_preserved':all(v['literal_conditional_mean_equal_H0_B'] for v in diags),
           'all_zero_literal_mu_emits_zero':all(v['zero_literal_mu_emits_zero'] for v in diags),
           'all_original_zero_means_remain_zero':all(v['zero_literal_mu_emits_zero'] for v in diags),
           'all_IPF_not_applied':all(not v['IPF']['applied'] for v in diags),
           'dependencies_sha256':generator.dependencies,'prediction_sha256':sha(dest),
           'donor_provenance_sha256':sha(csv),'teacher_effect_provenance_sha256':sha(provenance),
           'target_diagnostics_sha256':sha(targets_csv),'seconds':time.monotonic()-start,
           'prediction_path':str(dest),'diagnostics_dir':str(records),'target_diagnostic_files':paths}
    dump(records/'generation_manifest.json',value)
    return {'prediction_path':dest,'generator_changed':arm!='H0',
            'generation_manifest_path':records/'generation_manifest.json',
            'diagnostics_paths':[targets_csv,provenance,*map(Path,paths)],'provenance_path':csv}
