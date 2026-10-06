"""Root-only pure scorer for one selected fixed-mean three-seed follow-up.

No generation imports, no fits/draws. 24 saved score cases strictly reused,
only 12 new cases after completed independent count96 gate.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,itertools,json,os,shutil,sys,time,traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[k]='4'
ROOT=Path('C:/00 - Competition');OUT=ROOT/'runs/research_v28_20261005/context_KRR_seedrep_conditional_v1';CODE=Path(__file__).resolve()
VCC=ROOT/'envs/vcc313/Scripts/python.exe';SEEDS=(20260928,20260929,20260930);BASE='clean_B0_NB'
OPTIONS=('context_KRR_NB','equalcomponent_uniform_NB','positive_RBF_NB')
sys.path.insert(0,str(ROOT/'scripts'));import native_reproducer as nr
import numpy as np
import pandas as pd

def fresh(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def check(p,h):assert nr.sha(p)==h,str(p)

def validate():
    p=nr.read(OUT/'PROTOCOL.json');assert p['frozen'] and not p['pending_bindings'] and p['schema']=='conditional_selected_context_KRR_originalNB400_seedrep_v1'
    selected=p['selected_arm'];assert selected in OPTIONS and p['arms']==[BASE,selected] and p['seeds']==list(SEEDS) and p['cells_per_target']==400
    assert p['evaluation_sha256']=='c8a719bda010e352af4dd7c433313c795eab5460e709897a95b474cd9091e35c'
    original=nr.read(ROOT/'runs/research_v28_20261005/clean400_v1/PROTOCOL.json')
    assert p['metric_weights']==original['metric_weights'] and p['original_evaluation_aggregation']==original['aggregation'] and all(w==1/6 for w in p['metric_weights'].values())
    assert len(p['cases'])==6 and sorted(c['context'] for c in p['cases'])==sorted(list(nr.CONTEXTS)*2)
    for path,h in p['dependencies'].items():check(path,h)
    assert p['dependencies'][str(CODE.resolve())]==nr.sha(CODE)
    ticket=nr.read(p['choice_ticket_path']);check(p['choice_ticket_path'],p['choice_ticket_sha256']);assert ticket['chosen_arm']==selected and ticket['all24_completed_and_independently_replayed']
    for old in p['reuse_records']:
        dest=OUT/f"seed_{old['seed']}"/old['role']
        for name,s in old['files'].items():
            check(s['path'],s['sha256'])
            target=dest/f"pred__{old['arm']}.h5ad" if name=='prediction' else dest/f"{old['arm']}_moments.npz" if name=='moments' else dest/f"{old['arm']}_SOURCE_generation.json" if name=='generation' else dest/'scores'/old['arm']/name.removeprefix('score/')
            check(target,s['sha256'])
    assert len(p['reuse_records'])==24
    seal=nr.read(OUT/'PREDICTIONS_SEALED.json');assert seal['completed'] and seal['all36_before_new_judge'] and seal['newjudge_at_seal']==0
    assert seal['protocol_sha256']==nr.sha(OUT/'PROTOCOL.json') and len(seal['records'])==36
    assert {(r['seed'],r['role'],r['arm']) for r in seal['records']}=={(seed,c['role'],arm) for seed in SEEDS for c in p['cases'] for arm in p['arms']}
    assert sum(r['strict_reuse'] for r in seal['records'])==24
    for r in seal['records']:
        for name in ('prediction','moments','generation_manifest'):check(r[name+'_path'],r[name+'_sha256'])
        assert r['count_seed']==r['seed'] and r['donor_seed']==r['seed']+100 and r['cells_per_target']==400 and r['cells']==3200
        m=nr.read(r['generation_manifest_path']);copyr=dict(r);copyr.pop('generation_manifest_path');copyr.pop('generation_manifest_sha256');assert m==copyr
        if not r['strict_reuse']:
            assert r['seed'] in SEEDS[1:] and r['arm']==selected
            check(r['donor_receipt_path'],r['donor_receipt_sha256']);assert all(x['protected_counts_equal_same_seed_B0'] for x in r['targets'])
    gate=nr.read(OUT/'GENERATION_INDEPENDENT_AUDIT.json')
    assert gate['completed'] and gate['passed'] and gate['target_blocks_full_replayed']==96
    assert gate['prediction_seal_sha256']==nr.sha(OUT/'PREDICTIONS_SEALED.json') and gate['protocol_sha256']==nr.sha(OUT/'PROTOCOL.json')
    gc=nr.read(OUT/'GENERATION_COMPLETE.json');assert gc['completed'] and gc['prediction_seal_sha256']==nr.sha(OUT/'PREDICTIONS_SEALED.json') and gc['newjudge']==0
    return p

def paired_bootstrap(matrix):
    draws = list(itertools.product(range(3), repeat=3))
    distribution = np.asarray([matrix[np.ix_(b, s)].mean() for b in draws for s in draws], dtype=np.float64)
    return {'paired_macro_delta': float(matrix.mean()), 'crossed_background_seed_percentile95': np.quantile(distribution, [.025, .975]).tolist(),
            'fraction_positive_bootstrap_draws': float(np.mean(distribution > 0)), 'equally_weighted_bootstrap_draws': len(distribution),
            'minimum_draw': float(distribution.min()), 'maximum_draw': float(distribution.max()),
            'unit': 'Three backgrounds and three stochastic count seeds, paired across arms; roles/cells are not resampled as independent'}

def collect(p):
    rows=[]
    for seed in SEEDS:
        for c in p['cases']:
            for arm in p['arms']:
                dest=OUT/f'seed_{seed}'/c['role']/'scores'/arm
                if not (dest/'summary.json').exists():continue
                summary=nr.read(dest/'summary.json');components=pd.read_csv(dest/'overall_components.csv')
                assert len(components)==6 and set(components.metric)==set(nr.METRICS)
                scaled={f'scaled__{r.metric}':float(r.replicate_scaled) for r in components.itertuples()}
                assert np.isclose(summary['replicate_overall'],np.mean(list(scaled.values())),rtol=0,atol=1e-12)
                rows.append(dict(seed=seed,role=c['role'],context=c['context'],arm=arm,**summary,**scaled))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'completed_scores.csv',index=False)
    per_seed=[];bgseed=[]
    for seed in SEEDS:
        baseline=frame[(frame.seed==seed)&(frame.arm==BASE)];assert len(baseline)==6
        bgs=baseline.groupby('context').replicate_overall.mean().to_dict()
        for arm in p['arms']:
            group=frame[(frame.seed==seed)&(frame.arm==arm)]
            if len(group)!=6:continue
            assert group.groupby('context').size().to_dict()=={bg:2 for bg in nr.CONTEXTS}
            bybg=group.groupby('context').replicate_overall.mean().to_dict()
            per_seed.append(dict(seed=seed,arm=arm,overall=float(np.mean(list(bybg.values()))),delta_vs_NB=float(np.mean([bybg[bg]-bgs[bg] for bg in nr.CONTEXTS])),per_background=bybg,six_scaled={m:float(group[f'scaled__{m}'].mean()) for m in nr.METRICS}))
            bgseed.extend(dict(seed=seed,arm=arm,context=bg,overall=v,delta_vs_same_seed_same_background_NB=v-bgs[bg]) for bg,v in bybg.items())
    macro=[];bootstrap=[]
    for arm in p['arms']:
        entries=[x for x in per_seed if x['arm']==arm]
        if len(entries)==3:
            group=frame[frame.arm==arm]
            macro.append(dict(arm=arm,three_seed_macro_overall=float(np.mean([x['overall'] for x in entries])),three_seed_macro_delta_vs_NB=float(np.mean([x['delta_vs_NB'] for x in entries])),per_background_three_seed_mean={bg:float(np.mean([x['per_background'][bg] for x in entries])) for bg in nr.CONTEXTS},six_scaled={m:float(group[f'scaled__{m}'].mean()) for m in nr.METRICS},all_three_seed_overall_deltas_positive=all(x['delta_vs_NB']>0 for x in entries) if arm!=BASE else None))
    if len(macro)==2:
        matrix=np.asarray([[next(x['delta_vs_same_seed_same_background_NB'] for x in bgseed if x['seed']==seed and x['context']==bg and x['arm']==p['selected_arm']) for seed in SEEDS] for bg in nr.CONTEXTS],float)
        bootstrap.append(dict(arm=p['selected_arm'],backgrounds=list(nr.CONTEXTS),seeds=list(SEEDS),paired_delta_matrix=matrix.tolist(),**paired_bootstrap(matrix)))
    progress=dict(completed=len(frame)==36,completed_score_cases=len(frame),strict_reused_score_cases=24,actual_new_scores=max(0,len(frame)-24),selected_arm=p['selected_arm'],per_seed=per_seed,per_background_seed=bgseed,three_seed_macro=macro,paired_crossed_bootstrap=bootstrap,only_promotion_metric='Same-protocol overall',all_original_cases_retained=True,firstseed_selected_after_complete_development_screen=True,independent_units='Three developed backgrounds; countseeds technical stochastic repetitions, roles and cells are not independent biology',new_model_fits=0,new_mean_forwards=0,official_calls=0,THP_truth_opened=False,protocol_sha256=nr.sha(OUT/'PROTOCOL.json'))
    nr.write(OUT/'PROGRESS.json',progress);return progress

def score():
    assert Path(sys.executable).resolve()==VCC.resolve() and np.__version__=='2.3.3'
    p=validate();import cell_eval2;assert cell_eval2.__version__=='0.16.0'
    bindings=dict(protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),prediction_seal_sha256=nr.sha(OUT/'PREDICTIONS_SEALED.json'),independent_audit_sha256=nr.sha(OUT/'GENERATION_INDEPENDENT_AUDIT.json'),score_entrypoint_path=str(CODE),score_entrypoint_sha256=nr.sha(CODE),interpreter=str(VCC),interpreter_sha256=nr.sha(VCC),judge_processes=1,judge_threads=4,new_score_cases_expected=12,strict_reused_score_cases=24,fixed_original_metric_weights_anchors=True,official_calls=0)
    path=OUT/'SCORE_PROTOCOL.json'
    if path.exists():
        old=nr.read(path)
        for k,v in bindings.items():assert old[k]==v
    else:fresh(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),**bindings))
    nr.engine()
    with (OUT/'SCORING.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        for seed in SEEDS[1:]:
            for c in p['cases']:
                assert shutil.disk_usage(ROOT).free>=50*2**30
                dest=OUT/f'seed_{seed}'/c['role']
                if not (dest/'scores'/p['selected_arm']/'summary.json').exists():nr.score_case(dest,[p['selected_arm']],4)
                progress=collect(p);print('SELECTED_CONTEXT_KRR_SEEDREP_PROGRESS',seed,c['role'],json.dumps(progress['per_seed']),flush=True)
        progress=collect(p);assert progress['completed'] and progress['actual_new_scores']==12;validate()
        fresh(OUT/'COMPLETE.json',dict(actual_new_integer_count_outputs=12,actual_reused_count_outputs=24,actual_new_judge_cases=12,judge_processes=1,judge_threads=4,code_sha256=nr.sha(CODE),**progress))
    finally:(OUT/'SCORING.lock').unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('score','collect'));a=parser.parse_args();os.chdir(ROOT)
    try:
        if a.phase=='score':score()
        else:print(json.dumps(collect(validate())))
    except BaseException as e:
        fresh(OUT/f'FAILED_SCORE_{a.phase}_{time.time_ns()}.json',dict(completed=False,error=repr(e),traceback=traceback.format_exc(),draw_calls=0,fit_calls=0,official_calls=0));raise
