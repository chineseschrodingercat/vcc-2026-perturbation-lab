"""Independent saved-scalar three-seed postcheck; no judge/model/RNA decoding.

JSON/CSV are provenance and previously saved metric scalars. Any H5AD, NPZ,
parquet, model or RNA file is streamed only for SHA. No producer imports.
--draft writes metadata only. --check requires root authorization after COMPLETE.
"""
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import argparse, hashlib, importlib.metadata, itertools, json, os, re, sys, time, traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ[k]='2'
import numpy as np
import pandas as pd

ROOT=Path('C:/00 - Competition')
OUT=ROOT/'runs/research_v28_20261005/context_KRR_seedrep_conditional_v1'
VCC=ROOT/'envs/vcc313/Scripts/python.exe'
SEEDS=(20260928,20260929,20260930)
ARMS=('clean_B0_NB','context_KRR_NB')
CONTEXTS=('hepg2','jurkat','rpe1')
METRICS=('pds_cosine','expr_mse_unbiased_capped_norm','de_wilcoxon_direction_fidelity_yield_raw','de_wilcoxon_direction_reach_raw','de_wilcoxon_sig_jaccard','de_wilcoxon_lfc_nmae')
EXPECTED_PROTOCOL='22901dd0a9998cf161836d2303f0263f17f82877f8c0cd1604de4cb8e1edfa7a'
EXPECTED_SEAL='8bed547560cdc296971ead26aa14ddd5af60cb5aab34454715f0e3015b704f0c'
checked={}; scalar_inputs={}
worst=dict(raw=0.,fixed_scaled=0.,replicate_scaled=0.,overall=0.,macro=0.,bootstrap=0.)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(2**20):h.update(block)
    return h.hexdigest()


def check(path,digest):
    path=str(Path(path).resolve())
    assert re.fullmatch('[0-9a-f]{64}',digest),path
    if path in checked:assert checked[path]==digest,path
    else:assert sha(path)==digest,path;checked[path]=digest


def read(path):
    path=Path(path);assert path.suffix=='.json'
    scalar_inputs[str(path.resolve())]=sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv(path):
    path=Path(path);assert path.suffix=='.csv'
    scalar_inputs[str(path.resolve())]=sha(path)
    return pd.read_csv(path)


def fresh(path,obj):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(obj,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def bound_tree(obj):
    if isinstance(obj,list):
        for x in obj:bound_tree(x)
    elif isinstance(obj,dict):
        if 'path' in obj and 'sha256' in obj:check(obj['path'],obj['sha256'])
        for key,value in obj.items():
            if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) and Path(key).is_absolute():check(key,value)
            if key.endswith('_path') and isinstance(value,str):
                prefix=key[:-5]
                for suffix in ('_sha256','_SHA256'):
                    if prefix+suffix in obj:check(value,obj[prefix+suffix])
            bound_tree(value)


def close(a,b,kind):
    diff=abs(float(a)-float(b));worst[kind]=max(worst[kind],diff)
    assert np.isfinite(a) and np.isfinite(b) and diff<5e-12,(kind,a,b,diff)


def local(seed,role):return OUT/f'seed_{seed}'/role


def scalar_case(folder,targets,arm,new_judge):
    from cell_eval2.catalog import CATALOG
    from cell_eval2.scoring import score_one
    from cell_eval2.scales import SCALES
    ref=read(folder.parent.parent/'references/reference.json')
    assert ref['replicate_usable'] and ref['cell_eval2']=='0.16.0'
    summary=read(folder/'summary.json');components=csv(folder/'overall_components.csv')
    agg=csv(folder/'aggregate.csv');metrics=csv(folder/'metrics.csv');cfg=read(folder/'config.json')
    assert summary['model']==arm and set(components.metric)==set(METRICS) and len(components)==6
    assert not metrics.duplicated(['perturbation','metric']).any() and set(metrics.perturbation)==targets
    assert cfg['input_type']=='counts' and cfg['target_sum']==1e6 and cfg['control_source']=='real'
    assert cfg['de']['backend']=='scanpy' and cfg['de']['mean_calc']=='arithmetic' and cfg['de']['epsilon']==1e-9
    if new_judge:assert cfg['num_threads']==cfg['gather_threads']==4
    value={};coverage={}
    for metric in METRICS:
        if metric=='expr_mse_unbiased_capped_norm':
            numerator=metrics[metrics.metric=='expr_mse_unbiased_capped'].set_index('perturbation').value
            denominator=metrics[metrics.metric=='expr_distance_unbiased'].set_index('perturbation').value
            assert set(numerator.index)==set(denominator.index)==targets
            denominator=denominator.reindex(numerator.index);keep=np.isfinite(numerator)&np.isfinite(denominator)
            assert denominator[keep].sum()>0
            raw=float(numerator[keep].sum()/denominator[keep].sum())
            coverage[metric]=dict(panel_targets=8,finite_pairs=int(keep.sum()),aggregation='ratio_of_sums')
        else:
            row=metrics[metrics.metric==metric];assert set(row.perturbation).issubset(targets)
            finite=np.isfinite(row.value);raw=float(row.loc[finite,'value'].mean())
            coverage[metric]=dict(panel_targets=8,emitted_metric_rows=len(row),finite_metric_rows=int(finite.sum()),
                undefined_or_absent_targets=sorted(targets-set(row.loc[finite,'perturbation'])),aggregation='canonical_NaN_skipping_mean')
        a=agg[agg.metric==metric];assert len(a)==1
        close(raw,float(a.iloc[0]['mean']),'raw');close(raw,summary[metric],'raw')
        comp=components[components.metric==metric].iloc[0]
        close(raw,comp.raw,'raw');close(comp.baseline,ref['baseline'][metric],'raw');close(comp.replicate,ref['replicate'][metric],'raw')
        policy=replace(CATALOG[metric].scoring,anchor=float(ref['replicate'][metric]),allow_negative_baseline=False)
        rep=score_one(raw,float(ref['baseline'][metric]),policy)
        fixedpolicy=SCALES['low-random_high-1_v10'].entries[metric]
        fixed=score_one(raw,fixedpolicy.base,fixedpolicy.scoring)
        close(rep,comp.replicate_scaled,'replicate_scaled');close(fixed,comp.fixed_scaled,'fixed_scaled')
        value[metric]=dict(raw=raw,replicate_scaled=float(rep),fixed_scaled=float(fixed))
    overall=float(np.mean([value[m]['replicate_scaled'] for m in METRICS]))
    fixedoverall=float(np.mean([value[m]['fixed_scaled'] for m in METRICS]))
    close(overall,summary['replicate_overall'],'overall');close(fixedoverall,summary['fixed_overall'],'overall')
    return dict(overall=overall,fixed_overall=fixedoverall,
        raw={m:value[m]['raw'] for m in METRICS},
        replicate_scaled={m:value[m]['replicate_scaled'] for m in METRICS},
        fixed_scaled={m:value[m]['fixed_scaled'] for m in METRICS},metric_coverage=coverage)


def macro(rows):
    # Exactly two roles per background. Never resample roles/cells as independent.
    per_seed=[];bgseed=[]
    for seed in SEEDS:
        base=[r for r in rows if r['seed']==seed and r['arm']==ARMS[0]];assert len(base)==6
        baseline={bg:float(np.mean([r['overall'] for r in base if r['context']==bg])) for bg in CONTEXTS}
        for arm in ARMS:
            group=[r for r in rows if r['seed']==seed and r['arm']==arm];assert len(group)==6
            assert all(sum(r['context']==bg for r in group)==2 for bg in CONTEXTS)
            bybg={bg:float(np.mean([r['overall'] for r in group if r['context']==bg])) for bg in CONTEXTS}
            per_seed.append(dict(seed=seed,arm=arm,overall=float(np.mean(list(bybg.values()))),
                delta_vs_NB=float(np.mean([bybg[bg]-baseline[bg] for bg in CONTEXTS])),per_background=bybg,
                six_raw={m:float(np.mean([r['raw'][m] for r in group])) for m in METRICS},
                six_scaled={m:float(np.mean([r['replicate_scaled'][m] for r in group])) for m in METRICS},
                six_fixed_scaled={m:float(np.mean([r['fixed_scaled'][m] for r in group])) for m in METRICS}))
            bgseed.extend(dict(seed=seed,arm=arm,context=bg,overall=bybg[bg],
                delta_vs_same_seed_same_background_NB=bybg[bg]-baseline[bg]) for bg in CONTEXTS)
    macros=[]
    for arm in ARMS:
        entries=[r for r in per_seed if r['arm']==arm];assert len(entries)==3
        group=[r for r in rows if r['arm']==arm];assert len(group)==18
        macros.append(dict(arm=arm,three_seed_macro_overall=float(np.mean([r['overall'] for r in entries])),
            three_seed_macro_delta_vs_NB=float(np.mean([r['delta_vs_NB'] for r in entries])),
            per_background_three_seed_mean={bg:float(np.mean([r['per_background'][bg] for r in entries])) for bg in CONTEXTS},
            six_raw={m:float(np.mean([r['raw'][m] for r in group])) for m in METRICS},
            six_scaled={m:float(np.mean([r['replicate_scaled'][m] for r in group])) for m in METRICS},
            six_fixed_scaled={m:float(np.mean([r['fixed_scaled'][m] for r in group])) for m in METRICS},
            all_three_seed_overall_deltas_positive=all(r['delta_vs_NB']>0 for r in entries) if arm!=ARMS[0] else None))
    matrix=np.asarray([[next(r['delta_vs_same_seed_same_background_NB'] for r in bgseed
        if r['seed']==seed and r['context']==bg and r['arm']==ARMS[1]) for seed in SEEDS] for bg in CONTEXTS],float)
    # Independent weighted-count form of the original 27 x 27 crossed draw law.
    draws=list(itertools.product(range(3),repeat=3))
    weights=[np.bincount(draw,minlength=3).astype(np.float64)/3 for draw in draws]
    distribution=np.asarray([wb@matrix@ws for wb in weights for ws in weights],np.float64)
    assert distribution.shape==(729,) and np.isfinite(distribution).all()
    boot=dict(arm=ARMS[1],backgrounds=list(CONTEXTS),seeds=list(SEEDS),paired_delta_matrix=matrix.tolist(),
        paired_macro_delta=float(matrix.mean()),crossed_background_seed_percentile95=np.quantile(distribution,[.025,.975],method='linear').tolist(),
        fraction_positive_bootstrap_draws=float(np.mean(distribution>0)),equally_weighted_bootstrap_draws=729,
        minimum_draw=float(distribution.min()),maximum_draw=float(distribution.max()),
        unit='Three backgrounds and three stochastic count seeds, paired across arms; roles/cells are not resampled as independent')
    return per_seed,bgseed,macros,boot


def compare_saved(saved,per_seed,bgseed,macros,boot):
    assert len(saved['per_seed'])==6 and len(saved['per_background_seed'])==18 and len(saved['three_seed_macro'])==2
    for r in per_seed:
        found=[x for x in saved['per_seed'] if x['seed']==r['seed'] and x['arm']==r['arm']];assert len(found)==1;y=found[0]
        close(r['overall'],y['overall'],'macro');close(r['delta_vs_NB'],y['delta_vs_NB'],'macro')
        for bg in CONTEXTS:close(r['per_background'][bg],y['per_background'][bg],'macro')
        for m in METRICS:close(r['six_scaled'][m],y['six_scaled'][m],'macro')
    for r in bgseed:
        found=[x for x in saved['per_background_seed'] if all(x[k]==r[k] for k in ('seed','arm','context'))];assert len(found)==1;y=found[0]
        for k in ('overall','delta_vs_same_seed_same_background_NB'):close(r[k],y[k],'macro')
    for r in macros:
        found=[x for x in saved['three_seed_macro'] if x['arm']==r['arm']];assert len(found)==1;y=found[0]
        for k in ('three_seed_macro_overall','three_seed_macro_delta_vs_NB'):close(r[k],y[k],'macro')
        for bg in CONTEXTS:close(r['per_background_three_seed_mean'][bg],y['per_background_three_seed_mean'][bg],'macro')
        for m in METRICS:close(r['six_scaled'][m],y['six_scaled'][m],'macro')
        assert r['all_three_seed_overall_deltas_positive']==y['all_three_seed_overall_deltas_positive']
    assert len(saved['paired_crossed_bootstrap'])==1;y=saved['paired_crossed_bootstrap'][0]
    for k in ('arm','backgrounds','seeds','equally_weighted_bootstrap_draws','unit'):assert y[k]==boot[k],k
    assert np.asarray(y['paired_delta_matrix']).shape==(3,3)
    for a,b in zip(np.asarray(boot['paired_delta_matrix']).ravel(),np.asarray(y['paired_delta_matrix']).ravel()):close(a,b,'bootstrap')
    for k in ('paired_macro_delta','fraction_positive_bootstrap_draws','minimum_draw','maximum_draw'):close(boot[k],y[k],'bootstrap')
    for a,b in zip(boot['crossed_background_seed_percentile95'],y['crossed_background_seed_percentile95']):close(a,b,'bootstrap')


def draft():
    # Only metadata schema/count inspection; no bound-tree traversal or score read.
    p=read(OUT/'PROTOCOL.json');assert sha(OUT/'PROTOCOL.json')==EXPECTED_PROTOCOL
    assert tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS
    total=sum(len(r['files']) for r in p['reuse_records'])
    scores=sum(sum(k.startswith('score/') for k in r['files']) for r in p['reuse_records'])
    assert len(p['reuse_records'])==24 and total==294 and scores==222
    path=Path(__file__).with_suffix('.DRAFT.json')
    fresh(path,dict(schema='context_KRR_seedrep_saved_scalar_postcheck_DRAFT_v1',status='DRAFT_NOT_EXECUTED',
        code_path=str(Path(__file__).resolve()),code_sha256=sha(__file__),expected_OUT=str(OUT),
        expected_protocol_sha256=EXPECTED_PROTOCOL,expected_prediction_seal_sha256=EXPECTED_SEAL,
        expected_count_schema='conditional_selected_context_KRR_originalNB400_seedrep_v1',
        arms=list(ARMS),seeds=list(SEEDS),backgrounds=list(CONTEXTS),required_full_logical_cases=36,
        independent_new_count_gate_target_blocks=96,strict_reuse=dict(logical_cases=24,B0_cases=18,selected_firstseed_cases=6,
            reused_source_assets=294,reused_score_assets=222,non_score_assets=72),
        raw6_replay_cases=36,raw_metrics_replayed=216,fixed_scaled_replayed=216,replicate_scaled_replayed=216,
        MSE='canonical ratio of finite numerator/denominator sums per case, not average ratios',
        aggregation='two roles per background then three backgrounds then three seeds, all equally weighted',
        CI='paired 3background x3seed differences, exact729 crossed enumeration, NumPy linear percentile95',
        conditional_selection_scope='selected after firstseed completed development screen; countseeds technical; no final biological validation',
        whole_population=True,negative_seeds_and_backgrounds_retained=True,
        numeric_IO='saved CSV metric/aggregate/components scalars; JSON provenance only; all RNA/mean/count/DEparquet opaque SHA',
        engine_judge_calls=0,producer_imports=0,source_or_count_numeric_reads=0,Native_THPtruth_numeric_reads=0,fits=0,
        eventual_check_interpreter=str(VCC),Numpy='2.3.3',cell_eval2='0.16.0',
        execution_requires_root_authorization_and_COMPLETE_and_full96_count_gate=True))
    print(json.dumps(dict(DRAFT=str(path),DRAFT_SHA256=sha(path),code_SHA256=sha(__file__),actual_check=False)))


def run():
    started=time.perf_counter()
    assert Path(sys.executable).resolve()==VCC.resolve() and np.__version__=='2.3.3'
    assert importlib.metadata.version('cell_eval2')=='0.16.0'
    assert not (OUT/'POSTSCORE_INDEPENDENT_AUDIT.json').exists(),'preserve completed audit'
    check(OUT/'PROTOCOL.json',EXPECTED_PROTOCOL);check(OUT/'PREDICTIONS_SEALED.json',EXPECTED_SEAL)
    p=read(OUT/'PROTOCOL.json');seal=read(OUT/'PREDICTIONS_SEALED.json');score=read(OUT/'SCORE_PROTOCOL.json')
    done=read(OUT/'COMPLETE.json');progress=read(OUT/'PROGRESS.json')
    gate=read(OUT/'GENERATION_INDEPENDENT_AUDIT.json');gc=read(OUT/'GENERATION_COMPLETE.json')
    assert p['schema']=='conditional_selected_context_KRR_originalNB400_seedrep_v1' and p['frozen'] and not p['pending_bindings']
    assert tuple(p['arms'])==ARMS and p['selected_arm']==ARMS[1] and tuple(p['seeds'])==SEEDS and p['cells_per_target']==400
    assert p['evaluation_sha256']=='c8a719bda010e352af4dd7c433313c795eab5460e709897a95b474cd9091e35c'
    assert len(p['cases'])==6 and p['total_outputs']==36 and p['new_outputs']==12 and p['reused_outputs']==24
    assert p['reused_B0_outputs']==18 and p['reused_selected_pilot_outputs']==6
    assert seal['completed'] and seal['all36_before_new_judge'] and seal['newjudge_at_seal']==0 and len(seal['records'])==36
    expected={(seed,c['role'],arm) for seed in SEEDS for c in p['cases'] for arm in ARMS}
    assert {(r['seed'],r['role'],r['arm']) for r in seal['records']}==expected
    assert sum(r['strict_reuse'] for r in seal['records'])==24
    assert done['completed'] and done['completed_score_cases']==36 and done['actual_new_judge_cases']==done['actual_new_scores']==12 and done['strict_reused_score_cases']==24
    assert done['actual_new_integer_count_outputs']==12 and done['actual_reused_count_outputs']==24
    assert score['judge_processes']==done['judge_processes']==1 and score['judge_threads']==done['judge_threads']==4
    assert score['new_score_cases_expected']==12 and score['strict_reused_score_cases']==24 and score['official_calls']==0
    check(score['score_entrypoint_path'],score['score_entrypoint_sha256']);assert done['code_sha256']==score['score_entrypoint_sha256']
    assert Path(score['interpreter']).resolve()==VCC.resolve();check(VCC,score['interpreter_sha256'])
    proto_sha=sha(OUT/'PROTOCOL.json');seal_sha=sha(OUT/'PREDICTIONS_SEALED.json')
    assert seal['protocol_sha256']==score['protocol_sha256']==done['protocol_sha256']==progress['protocol_sha256']==proto_sha
    assert gate['completed'] and gate['passed'] and gate['target_blocks_full_replayed']==96
    assert gate['protocol_sha256']==proto_sha and gate['prediction_seal_sha256']==score['prediction_seal_sha256']==gc['prediction_seal_sha256']==seal_sha
    assert gc['completed'] and gc['newjudge']==0
    assert score['independent_audit_sha256']==sha(OUT/'GENERATION_INDEPENDENT_AUDIT.json')
    for obj in (p,seal,gate):bound_tree(obj)
    ticket=read(p['choice_ticket_path']);check(p['choice_ticket_path'],p['choice_ticket_sha256'])
    assert ticket['chosen_arm']==ARMS[1] and ticket['all24_completed_and_independently_replayed']
    pilot=ROOT/'runs/research_v28_20261005/ntc_context_krr_native400_v1'
    oldaudit=read(pilot/'POSTSCORE_INDEPENDENT_AUDIT.json');assert oldaudit['passed'] and oldaudit['completed']
    assert sha(pilot/'POSTSCORE_INDEPENDENT_AUDIT.json')==p['firstscreen_postscore_audit_sha256']
    assert sha(pilot/'COMPLETE.json')==p['firstscreen_COMPLETE_sha256']
    means=ROOT/'runs/research_v28_20261005/ntc_context_krr_native_mean_v1'
    mg=read(means/'MEAN_INDEPENDENT_AUDIT.json');ms=read(means/'MEANS_SEALED.json')
    assert mg['passed'] and mg['mean_seal_sha256']==sha(means/'MEANS_SEALED.json') and ms['completed'] and len(ms['records'])==18
    assert p['mean_gate_sha256']==sha(means/'MEAN_INDEPENDENT_AUDIT.json')
    assert p['mean_seal_sha256']==sha(means/'MEANS_SEALED.json') and p['mean_protocol_sha256']==sha(means/'PROTOCOL.json')
    original=ROOT/'runs/research_v28_20261005/clean400_v1';base=read(original/'PROTOCOL.json');oldcase={c['role']:c for c in base['cases']}
    assert p['metric_weights']==base['metric_weights'] and p['original_evaluation_aggregation']==base['aggregation'] and all(w==1/6 for w in p['metric_weights'].values())
    cases={c['role']:c for c in p['cases']};assert len(cases)==6
    populations={}
    for c in p['cases']:
        role=c['role'];b=c['base_case'];assert b==oldcase[role]
        targets=b['targets'];assert len(targets)==len(set(targets))==8 and b['prediction_cells_per_target']==400
        populations[role]=targets
        for leaf in b['leaves'].values():check(leaf['path'],leaf['sha256'])
        for seed in SEEDS:
            folder=local(seed,role);assert read(folder/'panel.json')['targets']==targets
            for name in ('panel.json','references/reference.json','references/anchor_aggregate.parquet'):
                check(folder/name,b['leaves'][name]['sha256'])
    reused={(r['seed'],r['role'],r['arm']):r for r in p['reuse_records']};assert len(reused)==24
    assert sum(r['arm']==ARMS[0] for r in reused.values())==18
    assert sum(r['arm']==ARMS[1] and r['seed']==SEEDS[0] for r in reused.values())==6
    assets=score_assets=0
    for old in reused.values():
        role=old['role'];arm=old['arm'];seed=old['seed'];dest=local(seed,role)
        sourcep=ROOT/'runs/research_v28_20261005/clean400_seedrep_v1/PROTOCOL.json' if arm==ARMS[0] else pilot/'PROTOCOL.json'
        assert old['source_protocol_sha256']==sha(sourcep)
        for name,spec in old['files'].items():
            assets+=1;check(spec['path'],spec['sha256'])
            if name=='prediction':target=dest/f'pred__{arm}.h5ad'
            elif name=='moments':target=dest/f'{arm}_moments.npz'
            elif name=='generation':target=dest/f'{arm}_SOURCE_generation.json'
            elif name.startswith('score/'):
                score_assets+=1;target=dest/'scores'/arm/name.removeprefix('score/')
            else:raise AssertionError('Undeclared reused leaf '+name)
            check(target,spec['sha256'])
    assert assets==294 and score_assets==222
    for r in seal['records']:
        seed=r['seed'];role=r['role'];arm=r['arm'];assert r['context']==cases[role]['context']
        assert r['completed'] and r['cells']==3200 and r['cells_per_target']==400 and r['count_seed']==seed and r['donor_seed']==seed+100
        for name in ('prediction','moments','generation_manifest'):check(r[name+'_path'],r[name+'_sha256'])
        manifest=read(r['generation_manifest_path']);copyr=dict(r)
        copyr.pop('generation_manifest_path');copyr.pop('generation_manifest_sha256');assert manifest==copyr
        assert r['protocol_sha256']==proto_sha
        if r['strict_reuse']:
            old=reused[(seed,role,arm)];sr=old['source_record']
            assert r['source_files']==old['files'] and r['source_protocol_sha256']==old['source_protocol_sha256']
            assert r['source_identity']==sr['source_identity'] and r['prediction_sha256']==sr['prediction_sha256'] and r['moments_sha256']==sr['moments_sha256']
            assert r['scores_reused_no_new_judge'] and r['newcount']==r['newjudge']==0
        else:
            assert seed in SEEDS[1:] and arm==ARMS[1] and not r['truth_read_by_generation'] and r['newfits']==r['newjudge']==r['official_calls']==0
            assert [x['target'] for x in r['targets']]==populations[role] and all(x['protected_counts_equal_same_seed_B0'] for x in r['targets'])
            assert [x['target'] for x in r['original_donor_receipts']]==populations[role]
            assert all(len(x['donor_rows'])==len(set(x['donor_rows']))==400 for x in r['original_donor_receipts'])
            bundle=cases[role]['mean_bundle'];ident=r['source_identity']
            assert ident['mean_path']==bundle['bundle_path'] and ident['mean_sha256']==bundle['bundle_sha256'] and ident['mean_workflow']==bundle['arm']
            assert ident['context_mean_protocol_SHA256']==p['mean_protocol_sha256']
            assert ident['source_model_seal_path']==bundle['source_model_seal_path'] and ident['source_model_seal_SHA256']==bundle['source_model_seal_SHA256']
            baseline=next(x for x in seal['records'] if x['seed']==seed and x['role']==role and x['arm']==ARMS[0])
            assert ident['baseline_same_seed_prediction_path']==baseline['prediction_path'] and ident['baseline_same_seed_prediction_sha256']==baseline['prediction_sha256']
            check(r['donor_receipt_path'],r['donor_receipt_sha256'])
    rows=[]
    for seed in SEEDS:
        for c in p['cases']:
            for arm in ARMS:
                folder=local(seed,c['role'])/'scores'/arm
                scalar=scalar_case(folder,set(populations[c['role']]),arm,arm==ARMS[1] and seed in SEEDS[1:])
                rows.append(dict(seed=seed,role=c['role'],context=c['context'],arm=arm,**scalar))
    assert len(rows)==36
    per_seed,bgseed,macros,boot=macro(rows)
    for saved in (done,progress):compare_saved(saved,per_seed,bgseed,macros,boot)
    combined=csv(OUT/'completed_scores.csv')
    assert len(combined)==36 and not combined.duplicated(['seed','role','arm']).any()
    for r in rows:
        found=combined[(combined.seed==r['seed'])&(combined.role==r['role'])&(combined.arm==r['arm'])];assert len(found)==1;x=found.iloc[0]
        assert x.context==r['context'];close(x.replicate_overall,r['overall'],'overall');close(x.fixed_overall,r['fixed_overall'],'overall')
        for m in METRICS:close(x[m],r['raw'][m],'raw');close(x['scaled__'+m],r['replicate_scaled'][m],'replicate_scaled')
    for path,digest in checked.items():assert sha(path)==digest,path
    for path,digest in scalar_inputs.items():assert sha(path)==digest,path
    receipt=dict(schema='context_KRR_seedrep_saved_scalar_postscore_independent_v1',passed=True,completed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=proto_sha,prediction_seal_sha256=seal_sha,complete_sha256=sha(OUT/'COMPLETE.json'),score_protocol_sha256=sha(OUT/'SCORE_PROTOCOL.json'),
        audit_script_path=str(Path(__file__).resolve()),audit_script_sha256=sha(__file__),all36=rows,
        raw_scalar_metrics_replayed=216,fixed_scaled_replayed=216,replicate_scaled_replayed=216,
        independent_scalar_csv_raw_aggregation=True,ratio_of_sums_MSE_replayed=True,frozen_catalog_scalar_policy_only=True,
        strict_reused_cases_and_scores=24,reused_B0_cases=18,reused_firstseed_KRR_cases=6,reused_source_assets=294,reused_score_assets=222,
        new_judge_cases_previously_completed=12,all_original_backgrounds_and_negative_seeds_retained=True,
        population8_eachcase_preserved=True,metric_specific_undefined_coverage_retained=True,source_hashes_unchanged=True,
        checked_opaque_file_hashes=len(checked),max_absolute_numerical_discrepancies=worst,
        per_seed=per_seed,per_background_seed=bgseed,three_seed_macro=macros,paired_crossed_bootstrap=[boot],
        independent_exact729_weighted_count_enumeration=True,CI_percentile_method='linear',
        mean_seal_sha256=p['mean_seal_sha256'],mean_independent_audit_sha256=p['mean_gate_sha256'],
        firstseed_selected_after_complete_development_screen=True,
        independent_units='Three repeatedly tuned Native development backgrounds; countseeds are technical stochastic repetitions; roles/cells are not independent biology; conditional selection scope retained',
        new_judges=0,new_counts=0,new_fits=0,RNA_or_count_numeric_decodes=0,Native_truth_expression_decodes=0,THP_numeric_reads=0,official_calls=0,
        elapsed_wall_seconds=time.perf_counter()-started,peak_RSS_bytes=None,peak_RSS_status='not instrumented',
        checked_assets_sha256=checked,scalar_inputs_sha256=scalar_inputs)
    fresh(OUT/'POSTSCORE_INDEPENDENT_AUDIT.json',receipt)
    print(json.dumps(dict(passed=True,path=str(OUT/'POSTSCORE_INDEPENDENT_AUDIT.json'),sha256=sha(OUT/'POSTSCORE_INDEPENDENT_AUDIT.json'),
        three_seed_macro=macros,paired_crossed_bootstrap=[boot],resources=dict(elapsed_wall_seconds=receipt['elapsed_wall_seconds'],peak_RSS_bytes=None)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--draft',action='store_true');group.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.draft:draft()
    else:
        try:run()
        except BaseException:
            fresh(OUT/f'FAILED_POSTSCORE_AUDIT_{time.time_ns()}.json',dict(error=traceback.format_exc(),original_assets_unchanged=True,new_judges_counts_fits=0));raise
