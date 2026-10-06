"""V27: matched V13 means with the frozen V26/V25 count mechanism.

The sole scientific intervention versus V26 is the total effect supplied to
the common expected-marginal calculation. Identity decoder state remains the
frozen V1 anchor. H3 is primary; H4 transports that state descriptively.
The immutable V26 mechanism is explicitly validated under its original
protocol, and V27's different mean source is locked separately.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

import latent_counts_v26 as base
from residual_v6_common import dump,generate as old_generate

PROTOCOL=Path('configs/v13_nb_v27_protocol.json')
BASE_PROTOCOL=Path('configs/latent_counts_v26_protocol.json')
ARMS=('H0','H1','H3','H4')
BASE_ARMS={'H0':'G0','H1':'G1','H3':'G3','H4':'G4'}
sha=base.sha
array_sha=base.array_sha
csr_sha=base.csr_sha


def validate_protocol(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    original=base.validate_protocol(BASE_PROTOCOL)
    assert value['primary_arm']=='H3'
    assert set(value['arms'])==set(ARMS)
    assert value['fixed_model']['candidate']=='flow_nb' and value['fixed_model']['checkpoint']==3200
    assert value['fixed_model']['new_full_fits']==0
    assert value['effect_source']=='matched_V13'
    for key in ['inference','ipf','count_guard']:
        assert value[key]==original[key], f'V27 changes the frozen count mechanism: {key}'
    for key in ['donor_seed_offset','protected_namespace','rounding_namespace','nb_namespace']:
        assert value['rng'][key]==original['rng'][key]
    return value


def load_teacher_effect(path,panel,role=None):
    """Historical V13 NPZs inherit locked panel/model axes explicitly.

    They predate the self-describing V22 NPZ schema. Validate that their saved
    full effect exactly reconstructs from the current immutable V1 prior plus
    their saved residual on those inherited axes; never invent embedded IDs.
    """
    path=Path(path);panel=Path(panel)
    with np.load(path,allow_pickle=False) as stored:
        if 'targets' in stored and 'genes' in stored:
            from score_set_response_v22 import load_effects
            return load_effects(path,panel,role)
        if not {'effect','residual'}.issubset(stored.files):
            raise ValueError('Legacy V13 prediction lacks full-effect/residual provenance')
        effect=np.asarray(stored['effect']);residual=np.asarray(stored['residual'])
    info=json.loads((panel/'panel.json').read_text(encoding='utf-8'))
    genes=base.axes()[0];targets=np.asarray(info['targets'],str)
    if effect.shape!=(len(targets),len(genes)) or residual.shape!=effect.shape:
        raise ValueError('Legacy V13 prediction disagrees with inherited historical target/gene axes')
    if not np.isfinite(effect).all() or not np.isfinite(residual).all():
        raise ValueError('Legacy V13 prediction contains nonfinite values')
    from residual_v6_common import frozen_v1,fold
    effect=np.asarray(effect,np.float32);residual=np.asarray(residual,np.float32)
    if not np.array_equal(effect,frozen_v1(targets,genes)+residual):
        raise ValueError('Legacy V13 effect does not exactly reconstruct on the locked historical axes')
    if role is not None:
        from score_set_response_v22 import parse_role
        outer,inner=parse_role(role)
        if info['context']!=(inner or outer) or any(fold(t)!=(1 if inner else 0) for t in targets):
            raise ValueError('Legacy V13 evaluation role and historical target fold disagree')
    return effect,{'context':info['context'],'targets':targets.tolist(),'genes':len(genes),
        'shape':list(effect.shape),'semantics':'historical saved full V13 effect; no new V1 addition',
        'legacy_NPZ_has_embedded_target_gene_identifiers':False,
        'target_axis_source':'historical panel.json exact target order',
        'gene_axis_source':'runs/context_v4/gene_axis.txt immutable model gene order',
        'panel_sha256':sha(panel/'panel.json'),'model_axis_sha256':sha('runs/context_v4/gene_axis.txt'),
        'full_effect_equal_frozen_V1_plus_saved_residual':True,'generator_dtype':'float32'}


def teacher_provenance(path,effects,panel):
    """Validate a real saved teacher file, not a claimed V13 label."""
    path=Path(path)
    value=json.loads(path.read_text(encoding='utf-8'))
    source=Path(value['teacher_effect_path'])
    if sha(source)!=value['teacher_effect_sha256']:
        raise ValueError('Historical matched V13 effect file hash changed')
    expected=array_sha(np.asarray(effects,np.float32))
    if value['effects_array_sha256']!=expected:
        raise ValueError('V13 provenance does not identify the actual float32 effect input')
    # This read accesses only a historical prediction NPZ, never real cells.
    actual,axis=load_teacher_effect(source,Path(panel),value.get('role'))
    if not np.array_equal(actual,np.asarray(effects,np.float32)):
        raise ValueError('Supplied effects are not the saved, exactly reordered V13 predictions')
    metadata=value.get('teacher_effect_metadata_path')
    if metadata is not None and sha(metadata)!=value['teacher_effect_metadata_sha256']:
        raise ValueError('Historical V13 effect metadata hash changed')
    tag=str(value['teacher_tag'])
    if 'v13' not in tag.lower():
        raise ValueError('Historical baseline tag is not the matched V13 procedure')
    if source.name != f'effects__{tag}.npz':
        raise ValueError('Teacher tag and historical V13 prediction file name disagree')
    historical=Path(value['historical_panel']);validated={}
    for field in ['references_sha256','reference_files_sha256','teacher_files_sha256']:
        declared=value.get(field,{})
        if isinstance(declared,str):
            filename=historical/'references/reference.json'
            if sha(filename)!=declared:raise ValueError('Historical reference JSON hash changed')
            validated[str(filename)]=declared;continue
        for filename,checksum in declared.items():
            source_path=Path(filename)
            if not source_path.is_absolute():
                source_path=(historical/'scores'/tag/source_path if field=='teacher_files_sha256'
                             else historical/'references'/source_path)
            if sha(source_path)!=checksum:
                raise ValueError(f'Matched V13 provenance file hash changed: {source_path}')
            validated[str(source_path)]=checksum
    return {**value,'provenance_path':str(path),'provenance_sha256':sha(path),
            'actual_prediction_array_verified':True,'prediction_axes':axis,
            'validated_teacher_and_reference_paths_sha256':validated}


class PanelGenerator(base.PanelGenerator):
    def __init__(self,panel,effects,role,model_seed,count_seed,protocol_path=PROTOCOL,
                 device='cuda',teacher_provenance_path=None):
        self.v27_protocol_path=Path(protocol_path)
        self.v27_protocol=validate_protocol(protocol_path)
        # Never tell V26's validator that a new protocol is its old protocol.
        super().__init__(panel,effects,role,model_seed,count_seed,BASE_PROTOCOL,device)
        self.v27_teacher_provenance=None
        if teacher_provenance_path is not None:
            self.v27_teacher_provenance=teacher_provenance(teacher_provenance_path,self.effects,self.panel)
            if int(self.v27_teacher_provenance['count_seed'])!=self.count_seed:
                raise ValueError('Teacher/count sampling seed pairing differs')
        self.dependencies.update({str(self.v27_protocol_path):sha(self.v27_protocol_path),
                                  'src/vcc2026/latent_counts_v27.py':sha('src/vcc2026/latent_counts_v27.py')})
        if self.v27_teacher_provenance is not None:
            value=self.v27_teacher_provenance
            self.dependencies[value['provenance_path']]=value['provenance_sha256']
            self.dependencies[value['teacher_effect_path']]=value['teacher_effect_sha256']
            self.dependencies.update(value['validated_teacher_and_reference_paths_sha256'])
            if value.get('teacher_effect_metadata_path'):
                self.dependencies[value['teacher_effect_metadata_path']]=value['teacher_effect_metadata_sha256']

    def build_target(self,target,arm):
        if arm not in ARMS:raise ValueError(arm)
        result=super().build_target(target,BASE_ARMS[arm])
        result.diagnostics.update(arm=arm,base_v26_mechanism_arm=BASE_ARMS[arm],
            mean_source='matched_V13_total_effect',protected_integer_equality_H0=True,
            baseline_diagnostic_field_prefix='G0 means the identical H0 V13 baseline in inherited moments')
        return result


def generate(panel_work,name,effects,count_seed,*,model_role,model_seed,arm,
             records_dir,protocol_path=PROTOCOL):
    panel=Path(panel_work);records=Path(records_dir);dest=panel/f'pred__{name}.h5ad'
    if arm not in ARMS:raise ValueError(arm)
    if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
    manifest=records/'generation_manifest.json'
    if manifest.exists():raise FileExistsError(f'Refusing to overwrite {manifest}')
    records.mkdir(parents=True,exist_ok=True)
    provenance=records/'teacher_effect_provenance.json'
    if not provenance.exists():
        raise FileNotFoundError('V27 requires scorer-written real matched V13 teacher_effect_provenance.json')
    start=time.monotonic()
    generator=PanelGenerator(panel,effects,model_role,model_seed,count_seed,protocol_path,
                             teacher_provenance_path=provenance)
    csv,source_npz,source_json=generator.write_provenance(records)
    if arm=='H0':old_generate(panel,name,effects,count_seed)
    blocks=[];labels=[];diags=[];paths=[]
    for target in generator.targets:
        try:result=generator.build_target(target,arm)
        except Exception as error:
            failure={'failed':True,'target':target,'arm':arm,'exception_type':type(error).__name__,
                     'exception':str(error),'clipped_or_redrawn':False,
                     'dependencies_sha256':generator.dependencies,'model_role':model_role,
                     'model_seed':int(model_seed),'count_seed':int(count_seed),
                     'mean_source':'matched_V13_total_effect'}
            if isinstance(error,base.GenerationValidationError):
                if error.failed_draw is not None:
                    np.savez_compressed(records/'failed_draw.npz',draw=error.failed_draw)
                    failure['failed_draw_sha256']=sha(records/'failed_draw.npz')
                if error.ipf_weights is not None:
                    np.savez_compressed(records/'failed_IPF_weights.npz',weights=error.ipf_weights)
                    failure['failed_IPF_weights_sha256']=sha(records/'failed_IPF_weights.npz')
                if error.rng is not None:failure['rng']=error.rng
            dump(records/'generation_failure.json',failure);raise
        target_dir=records/'targets';target_dir.mkdir(exist_ok=True)
        moments=target_dir/f'{result.target}.npz';np.savez_compressed(moments,**result.moments)
        info=target_dir/f'{result.target}.json';dump(info,result.diagnostics)
        paths.extend([str(moments),str(info)]);diags.append(result.diagnostics)
        if arm!='H0':blocks.append(result.block);labels.extend([result.target]*result.block.shape[0])
        print('V27_GENERATED_TARGET',arm,result.target,result.block.shape[0],result.diagnostics['IPF']['sweeps'],flush=True)
    if arm!='H0':
        from benchmark_context_v4 import save
        save(dest,blocks,labels,generator.genes)
    targets_csv=records/'target_diagnostics.csv';flat=[]
    for row in diags:
        fields={k:v for k,v in row.items() if not isinstance(v,(dict,list))}
        fields.update({f'IPF_{k}':v for k,v in row['IPF'].items()});flat.append(fields)
    pd.DataFrame(flat).to_csv(targets_csv,index=False)
    value={'schema':'v27_V13_mean_frozen_NB_count_1','completed':True,'finalfinished':True,
        'protocol_sha256':sha(protocol_path),'generator_sha256':sha('src/vcc2026/latent_counts_v27.py'),
        'base_v26_protocol_sha256':sha(BASE_PROTOCOL),'base_v26_generator_sha256':sha('src/vcc2026/latent_counts_v26.py'),
        'base_v26_mechanism_arm':BASE_ARMS[arm],'arm':arm,'generator_changed':arm!='H0',
        'primary_arm':'H3','H4_descriptive_only':True,'mean_source':'matched_V13_total_effect',
        'V25_predicted_mean_used':False,
        'teacher_effect_sha256':generator.v27_teacher_provenance['teacher_effect_sha256'],
        'teacher_provenance':generator.v27_teacher_provenance,
        'model_role':model_role,'model_seed':int(model_seed),'count_seed':int(count_seed),
        'fixed_flow_checkpoint_step':3200,'fixed_decoder_checkpoint_step':3000,'new_full_fits':0,
        'destination':generator.destination,'targets':generator.targets,'native_genes_sha256':array_sha(generator.genes),
        'effects_array_sha256':array_sha(generator.effects),'generation_target_order':generator.order,
        'donor_rng_before':generator.donor_rng_before,'donor_rng_after':generator.donor_rng_after,
        'same_G0_donors_selected_before_any_new_sampler_draw':True,
        'same_H0_donors_selected_before_any_new_sampler_draw':True,
        'donor_provenance_count_verified_on_all_6119_common_genes':True,
        'source_conditions':{'views':base.V25['inference']['views'],'cell_seed':base.V25['inference']['cell_seed'],
                            'view_stride':base.V25['inference']['view_stride'],'set_size':base.V25['model']['set_size']},
        'anchor':'unchanged actual scoring NTC donors × frozen V1 capped ratio; V13 changes expected IPF totals only',
        'editable_axis':'unchanged input6119 intersect V25 role-seen own-gene-mask and native measured genes',
        'mean_match':'H0 V13 unrounded editable donor row totals and gene column totals; inherited G0 fields denote H0',
        'baseline_moment_field_prefix':'G0','IPF':{'max_sweeps':base.MAX_SWEEPS,'tolerance':base.IPF_TOLERANCE,
            'dtype':'float64','failure_is_fatal':True},
        'sampler':'unchanged literal historical generator' if arm=='H0' else ('stochastic_round' if arm=='H1' else 'numpy_negative_binomial_NB2'),
        'NB_inverse_dispersion_source':'unchanged V25 role-specific frozen decoder gene-specific softplus(theta)+floor',
        'NB_rejection_sampler_gene_pairing_claimed':False,'actual_row_totals_forced':False,
        'RNG_namespaces_inherited_from_v26':{'round':[26,1],'NB':[26,3]},
        'destination_perturbation_outcomes_read':False,'whole_destination_heldout_from_training':True,
        'expected_count_views_averaged_before_integer_sampling':True,
        'all_protected_integer_equal_G0':all(v['protected_integer_equality'] for v in diags),
        'all_protected_integer_equal_H0':all(v['protected_integer_equality_H0'] for v in diags),
        'all_IPF_converged':all(v['IPF']['converged'] for v in diags),
        'dependencies_sha256':generator.dependencies,'prediction_sha256':sha(dest),
        'donor_provenance_sha256':sha(csv),'source_draws_sha256':sha(source_npz),'source_draws_info_sha256':sha(source_json),
        'teacher_effect_provenance_sha256':sha(provenance),'target_diagnostics_sha256':sha(targets_csv),
        'seconds':time.monotonic()-start,'prediction_path':str(dest),'diagnostics_dir':str(records),
        'target_diagnostic_files':paths}
    dump(manifest,value)
    return {'prediction_path':dest,'generator_changed':arm!='H0','generation_manifest_path':manifest,
            'diagnostics_paths':[targets_csv,source_npz,source_json,provenance,*map(Path,paths)],'provenance_path':csv}
