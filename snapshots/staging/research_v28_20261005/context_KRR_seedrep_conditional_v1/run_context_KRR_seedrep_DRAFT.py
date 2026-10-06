"""Conditional fixed three-count-seed follow-up, root choice required.

No arm is selected by draft. Freeze requires complete first-screen scores,
independent pure-scalar PASS, and root's exact highest-positive-arm ticket.
Generation only; 18 B0 + 6 chosen pilot strict reuse, 12 new original NB.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,itertools,json,os,shutil,sys,time,traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[k]='4'
ROOT=Path('C:/00 - Competition');HERE=Path(__file__).resolve().parent;STAGING=HERE.parent;CODE=Path(__file__).resolve()
sys.path.insert(0,str(STAGING))
import run_clean400 as clean
import run_null400 as rt
nr,np,pd=clean.nr,clean.np,clean.pd
from scipy import sparse
OUT=ROOT/'runs/research_v28_20261005/context_KRR_seedrep_conditional_v1'
PILOT=ROOT/'runs/research_v28_20261005/ntc_context_krr_native400_v1'
NB=ROOT/'runs/research_v28_20261005/clean400_seedrep_v1'
MEANS=ROOT/'runs/research_v28_20261005/ntc_context_krr_native_mean_v1'
GENPY=ROOT/'envs/foundation312/Scripts/python.exe'
SCORER=HERE/'score_context_KRR_seedrep_DRAFT.py'
POSTCHECK=STAGING/'postcheck_context_KRR400_DRAFT.py'
BASE='clean_B0_NB';CANDIDATES=('context_KRR_NB','equalcomponent_uniform_NB','positive_RBF_NB')
SEEDS=(20260928,20260929,20260930);CELLS=400;MAX_SECONDS=600.;MAX_BYTES=512*2**20

def check(p,h):assert nr.sha(p)==h,str(p)

def resources(start=None):
    import psutil
    m=psutil.Process().memory_info();peak=getattr(m,'peak_wset',m.rss);free=shutil.disk_usage(ROOT).free
    files=[q for q in OUT.glob('seed_*/*/pred__*.h5ad') if BASE not in q.name and 'seed_20260928' not in q.parts]
    size=sum(q.stat().st_size for q in files);elapsed=time.monotonic()-start if start is not None else None
    assert peak<2*2**30 and free>=50*2**30 and size<=MAX_BYTES
    assert elapsed is None or elapsed<MAX_SECONDS
    return dict(peak_RSS_bytes=peak,C_free_bytes=free,new_prediction_bytes=size,numeric_seconds=elapsed)

def spec(p,h=None):
    p=Path(p).resolve();actual=nr.sha(p)
    if h is not None:assert actual==h
    return dict(path=str(p),sha256=actual)

def sourcefiles(r,folder,arm):
    folder=Path(folder);d={k:spec(r[k+'_path'],r[k+'_sha256']) for k in ('prediction','moments')}
    d['generation']=spec(folder/f'{arm}_generation.json')
    score=folder/'scores'/arm;assert (score/'summary.json').is_file()
    for p in sorted(score.rglob('*')):
        if p.is_file():d['score/'+p.relative_to(score).as_posix()]=spec(p)
    return d

def draft(path,ticket=None):
    """Pure code/schema draft: deliberately does not inspect pilot score files."""
    assert not OUT.exists(),'Keep any previous frozen run'
    if ticket is not None:
        value=build_protocol(ticket);value['frozen']=False;value['draft_only']=True
        value['status']='SELECTED_ARM_COMPLETE_ROOT_CODE_REVIEW_REQUIRED_NO_GENERATION'
        clean.fresh(path,value);return
    clean.fresh(path,dict(schema='conditional_context_KRR_seedrep_DRAFT_v1',draft_only=True,
        code_path=str(CODE),code_sha256=nr.sha(CODE),scorer_path=str(SCORER),scorer_sha256=nr.sha(SCORER),
        output_path=str(OUT),selected_arm=None,candidate_options=list(CANDIDATES),seeds=list(SEEDS),
        pending=['all24 first-screen COMPLETE','independent savedscalar24 PASS','root highest-positive-arm ticket'],
        eligibility='After all24 and purepostcheckPASS, highest candidate overall strictly above same-protocol B0. If no positive candidate: unused; no parameter or source sweep.',
        plan=dict(logical_outputs=36,strict_reused_B0_outputs=18,strict_reused_selected_pilot_outputs=6,new_selected_outputs=12,new_target_blocks=96,original_B0_H0_replay_gate=96),
        generator='Original CleanPanelGenerator literal N1 unchanged; selected frozen effect only; count_seed=seed, seed+100 donors in original complete gen.order, original NB namespace26,3.',
        source_identity='New candidate inherits same-seed B0 source_identity then replaces only mean binding and adds same-seed baseline/donor receipt SHA provenance. Pilot reuse source_identity preserved verbatim.',
        mean_gate_path=str(MEANS/'MEAN_INDEPENDENT_AUDIT.json'),firstcount_gate_path=str(PILOT/'GENERATION_INDEPENDENT_AUDIT.json'),firstscore_gate_path=str(PILOT/'POSTSCORE_INDEPENDENT_AUDIT.json'),
        resource=dict(CPU_threads=4,RSS_GiB=2,numeric_seconds=MAX_SECONDS,new_prediction_MiB=512,C_free_GiB=50),
        schema_keys=dict(countgate=['completed','passed','protocol_sha256','prediction_seal_sha256','target_blocks_full_replayed96'],new_record=['seed','role','arm','source_identity','donor_receipt_path','donor_receipt_sha256']),
        firstscreen_score_values_read=0,expression_IO=0,fit_calls=0,draw_calls=0,judge_calls=0,official_calls=0,THP_reads=0))

def build_protocol(ticket_path):
    assert Path(sys.executable).resolve()==GENPY.resolve() and np.__version__=='2.5.3';resources()
    ticket=nr.read(ticket_path)
    assert ticket['schema']=='root_context_KRR400_selected_fixed_three_countseeds_v1'
    assert ticket['status']=='selected_for_paired_validation_not_promoted' and ticket['all24_completed_and_independently_replayed']
    assert ticket['count_seeds']==list(SEEDS)
    for path,h in ticket['immutable_screen_bindings_sha256'].items():check(path,h)
    pilotp=nr.read(PILOT/'PROTOCOL.json');pilots=nr.read(PILOT/'PREDICTIONS_SEALED.json')
    done=nr.read(PILOT/'COMPLETE.json');sa=nr.read(PILOT/'POSTSCORE_INDEPENDENT_AUDIT.json')
    assert done['completed'] and done['completed_score_cases']==24 and done['actual_new_scores']==18
    assert sa['completed'] and sa['passed'] and len(sa['all24'])==24
    assert sa['complete_sha256']==nr.sha(PILOT/'COMPLETE.json') and sa['protocol_sha256']==nr.sha(PILOT/'PROTOCOL.json')
    assert sa['prediction_seal_sha256']==nr.sha(PILOT/'PREDICTIONS_SEALED.json')
    assert ticket['immutable_screen_bindings_sha256'][str((PILOT/'COMPLETE.json').resolve())]==nr.sha(PILOT/'COMPLETE.json')
    assert ticket['immutable_screen_bindings_sha256'][str((PILOT/'POSTSCORE_INDEPENDENT_AUDIT.json').resolve())]==nr.sha(PILOT/'POSTSCORE_INDEPENDENT_AUDIT.json')
    agg={a['arm']:a for a in sa['aggregates']};assert set(agg)=={BASE,*CANDIDATES}
    selected=ticket['chosen_arm'];assert selected in CANDIDATES
    highest=max(agg[a]['overall'] for a in CANDIDATES)
    assert agg[selected]['overall']==highest and highest>agg[BASE]['overall'],'No eligible highest-positive arm'
    assert pilotp['arms']==[BASE,*CANDIDATES] and pilotp['count_seed']==SEEDS[0] and pilotp['cells_per_target']==400
    assert ticket['evaluation_sha256']==pilotp['evaluation_sha256']=='c8a719bda010e352af4dd7c433313c795eab5460e709897a95b474cd9091e35c'
    assert pilots['completed'] and pilots['protocol_sha256']==nr.sha(PILOT/'PROTOCOL.json')
    nbp=nr.read(NB/'PROTOCOL.json');nbs=nr.read(NB/'PREDICTIONS_SEALED.json');assert nr.read(NB/'COMPLETE.json')['completed']
    assert nbp['seeds']==list(SEEDS) and nbs['protocol_sha256']==nr.sha(NB/'PROTOCOL.json')
    means=nr.read(MEANS/'MEANS_SEALED.json');ma=nr.read(MEANS/'MEAN_INDEPENDENT_AUDIT.json')
    ga=nr.read(PILOT/'GENERATION_INDEPENDENT_AUDIT.json');assert ma['passed'] and ma['mean_seal_sha256']==nr.sha(MEANS/'MEANS_SEALED.json')
    assert ga['passed'] and ga['target_blocks_full_replayed']==144 and ga['prediction_seal_sha256']==nr.sha(PILOT/'PREDICTIONS_SEALED.json')
    b0p=nbp['source_clean400_protocol'];original={c['role']:c for c in b0p['cases']}
    assert pilotp['metric_weights']==b0p['metric_weights'] and len(pilotp['metric_weights'])==6 and all(w==1/6 for w in pilotp['metric_weights'].values())
    bm={(r['seed'],r['role'],r['arm']):r for r in nbs['records']};pm={(r['role'],r['arm']):r for r in pilots['records']}
    cases=[];reuse=[]
    for c in pilotp['cases']:
        role=c['role'];base=c['base_case'];assert base==original[role];bundle=c['mean_bundles'][selected]
        panel=nr.read(base['leaves']['panel.json']['path']);order=list(map(str,panel.get('generation_targets',base['targets'])))
        assert len(order)==8 and set(order)==set(base['targets'])
        cases.append(dict(role=role,context=c['context'],base_case=copy.deepcopy(base),mean_bundle=copy.deepcopy(bundle),original_generation_target_order=order))
        for seed in SEEDS:
            r=bm[seed,role,BASE];assert r['cells_per_target']==400 and r['count_seed']==seed and r['donor_seed']==seed+100
            reuse.append(dict(seed=seed,role=role,context=c['context'],arm=BASE,source_record=copy.deepcopy(r),files=sourcefiles(r,NB/f'seed_{seed}'/role,BASE),source_protocol_sha256=nr.sha(NB/'PROTOCOL.json')))
        r=pm[role,selected];assert r['source_identity']['mean_sha256']==bundle['bundle_sha256']
        reuse.append(dict(seed=SEEDS[0],role=role,context=c['context'],arm=selected,source_record=copy.deepcopy(r),files=sourcefiles(r,PILOT/role,selected),source_protocol_sha256=nr.sha(PILOT/'PROTOCOL.json')))
    assert len(reuse)==24 and len(cases)==6
    deps=dict(pilotp['dependencies'])
    for path,h in nbp['dependencies'].items():
        if path in deps:assert deps[path]==h
        deps[path]=h
    runtime_assets=rt.generation_runtime_assets();deps.update(runtime_assets)
    mandatory=(CODE,SCORER,POSTCHECK,STAGING/'run_clean400.py',STAGING/'run_null400.py',STAGING/'run_context_KRR400.py',STAGING/'run_clean400_seedrep.py',
        Path(ticket_path),PILOT/'PROTOCOL.json',PILOT/'PREDICTIONS_SEALED.json',PILOT/'GENERATION_INDEPENDENT_AUDIT.json',PILOT/'SCORE_PROTOCOL.json',PILOT/'COMPLETE.json',PILOT/'POSTSCORE_INDEPENDENT_AUDIT.json',
        NB/'PROTOCOL.json',NB/'PREDICTIONS_SEALED.json',NB/'COMPLETE.json',MEANS/'PROTOCOL.json',MEANS/'MEANS_SEALED.json',MEANS/'MEAN_INDEPENDENT_AUDIT.json')
    for path in mandatory:deps[str(path.resolve())]=nr.sha(path)
    for r in reuse:
        for v in r['files'].values():deps[v['path']]=v['sha256']
    for path,h in deps.items():check(path,h)
    return dict(schema='conditional_selected_context_KRR_originalNB400_seedrep_v1',frozen=True,code_sha256=nr.sha(CODE),selected_arm=selected,arms=[BASE,selected],seeds=list(SEEDS),cells_per_target=400,
        evaluation_sha256=pilotp['evaluation_sha256'],metric_weights=copy.deepcopy(pilotp['metric_weights']),original_evaluation_aggregation=copy.deepcopy(pilotp['aggregation']),
        cases=cases,reuse_records=reuse,dependencies=deps,generation_runtime_assets=runtime_assets,runtime=rt.runtime(),pending_bindings=[],
        choice_ticket_path=str(Path(ticket_path).resolve()),choice_ticket_sha256=nr.sha(ticket_path),firstscreen_COMPLETE_sha256=nr.sha(PILOT/'COMPLETE.json'),firstscreen_postscore_audit_sha256=nr.sha(PILOT/'POSTSCORE_INDEPENDENT_AUDIT.json'),
        mean_protocol_sha256=nr.sha(MEANS/'PROTOCOL.json'),mean_seal_sha256=nr.sha(MEANS/'MEANS_SEALED.json'),mean_gate_sha256=nr.sha(MEANS/'MEAN_INDEPENDENT_AUDIT.json'),
        total_outputs=36,new_outputs=12,reused_outputs=24,reused_B0_outputs=18,reused_selected_pilot_outputs=6,only_selected_highest_positive_arm=True,
        mean_sources_theta_seen_axis_own_masks_frozen=True,newfits=0,newmeanforwards=0,truth_read_by_generation=False,THP_reads=0,official_calls=0,
        root_judge_cases=12,root_judge_processes=1,root_judge_threads=4,all36_before_judge=True,
        budget=dict(CPU_threads=4,numeric_seconds=MAX_SECONDS,RSS_GiB=2,new_prediction_bytes=MAX_BYTES,C_free_GiB=50),
        NB='Unchanged original sampler namespace seed,originalordinal,26,3; no role/newarm namespace',donors='seed+100 choice in complete original gen.order, paired with same-seed B0',
        inference_scope='Firstseed chosen after viewed development screen, other2 fixed before screen; three count seeds technical, three developed backgrounds not final biological validation',
        aggregation='Two roles/background equal, three backgrounds equal, three seeds equal; original sixmetrics/weights/anchors',bootstrap='Exact729 crossed background and countseed paired enumeration')

def verify_protocol(p):
    assert p['frozen'] and p['schema']=='conditional_selected_context_KRR_originalNB400_seedrep_v1' and p['seeds']==list(SEEDS)
    assert p['selected_arm'] in CANDIDATES and p['arms']==[BASE,p['selected_arm']] and p['cells_per_target']==400
    assert p['code_sha256']==nr.sha(CODE) and p['runtime']==rt.runtime()
    for path,h in p['dependencies'].items():check(path,h)
    return p

def freeze(ticket):
    assert not OUT.exists(),'Preserve any old frozen/partial run'
    p=build_protocol(ticket);clean.fresh(OUT/'PROTOCOL.json',p);print('CONTEXT_KRR_SEEDREP_PROTOCOL',nr.sha(OUT/'PROTOCOL.json'),flush=True)

def local(seed,role):return OUT/f'seed_{seed}'/role

def reuse(p):
    records=[]
    for old in p['reuse_records']:
        seed,role,arm=old['seed'],old['role'],old['arm'];dest=local(seed,role)
        c=next(c for c in p['cases'] if c['role']==role);nr.prepare_case(c['base_case']['job'],dest)
        for name,s in old['files'].items():
            path=dest/f'pred__{arm}.h5ad' if name=='prediction' else dest/f'{arm}_moments.npz' if name=='moments' else dest/f'{arm}_SOURCE_generation.json' if name=='generation' else dest/'scores'/arm/name.removeprefix('score/')
            rt.link(s['path'],path,s['sha256'])
        r=dict(completed=True,seed=seed,role=role,context=old['context'],arm=arm,strict_reuse=True,cells=3200,cells_per_target=400,count_seed=seed,donor_seed=seed+100,
            prediction_path=str(dest/f'pred__{arm}.h5ad'),prediction_sha256=old['files']['prediction']['sha256'],moments_path=str(dest/f'{arm}_moments.npz'),moments_sha256=old['files']['moments']['sha256'],
            source_identity=copy.deepcopy(old['source_record']['source_identity']),source_files=copy.deepcopy(old['files']),source_protocol_sha256=old['source_protocol_sha256'],
            protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),scores_reused_no_new_judge=True,newcount=0,newjudge=0)
        manifest=dest/f'{arm}_generation.json';clean.fresh(manifest,r);r.update(generation_manifest_path=str(manifest),generation_manifest_sha256=nr.sha(manifest));records.append(copy.deepcopy(r))
    return records

def bind_candidate(gen,case,same_seed_B0,donor_path):
    base=case['base_case'];b=case['mean_bundle']
    with np.load(base['leaves']['B0_effects']['path'],allow_pickle=False) as z,np.load(b['bundle_path'],allow_pickle=False) as v:
        assert set(z.files)==set(v.files)
        for k in z.files:
            if k!='effect':np.testing.assert_array_equal(z[k],v[k])
        gen.effects=v['effect'].copy();assert gen.effects.dtype==np.float32 and np.isfinite(gen.effects).all()
    gen.source_identity=copy.deepcopy(same_seed_B0['source_identity'])
    gen.source_identity.update(mean_path=b['bundle_path'],mean_sha256=b['bundle_sha256'],mean_workflow=b['arm'],
        context_mean_protocol_path=str(MEANS/'PROTOCOL.json'),context_mean_protocol_SHA256=nr.sha(MEANS/'PROTOCOL.json'),source_model_seal_path=b['source_model_seal_path'],source_model_seal_SHA256=b['source_model_seal_SHA256'],
        baseline_same_seed_prediction_path=same_seed_B0['prediction_path'],baseline_same_seed_prediction_sha256=same_seed_B0['prediction_sha256'],baseline_same_seed_generation_path=same_seed_B0['generation_manifest_path'],baseline_same_seed_generation_sha256=same_seed_B0['generation_manifest_sha256'],
        same_seed_original_donor_receipt_path=str(donor_path),same_seed_original_donor_receipt_sha256=nr.sha(donor_path))

def validate_seal(p):
    s=nr.read(OUT/'PREDICTIONS_SEALED.json');assert s['completed'] and s['protocol_sha256']==nr.sha(OUT/'PROTOCOL.json')
    expected={(seed,c['role'],arm) for seed in SEEDS for c in p['cases'] for arm in p['arms']}
    assert len(s['records'])==36 and {(r['seed'],r['role'],r['arm']) for r in s['records']}==expected and sum(r['strict_reuse'] for r in s['records'])==24
    for r in s['records']:
        for k in ('prediction','moments','generation_manifest'):check(r[k+'_path'],r[k+'_sha256'])
        m=nr.read(r['generation_manifest_path']);copyr=dict(r);copyr.pop('generation_manifest_path');copyr.pop('generation_manifest_sha256');assert m==copyr
        assert r['count_seed']==r['seed'] and r['donor_seed']==r['seed']+100 and r['cells_per_target']==400
        if not r['strict_reuse']:check(r['donor_receipt_path'],r['donor_receipt_sha256'])
    return s

def generate():
    import anndata as ad
    ad.settings.allow_write_nullable_strings=True
    assert Path(sys.executable).resolve()==GENPY.resolve();p=verify_protocol(nr.read(OUT/'PROTOCOL.json'))
    assert not (OUT/'PREDICTIONS_SEALED.json').exists() and not (OUT/'GENERATION_STARTED.json').exists(),'No implicit retries'
    clean.fresh(OUT/'GENERATION_STARTED.json',dict(protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),created_utc=datetime.now(timezone.utc).isoformat(),no_implicit_retry=True))
    start=time.monotonic();records=reuse(p);arm=p['selected_arm']
    with (OUT/'GENERATING.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        for seed in SEEDS[1:]:
            for case in p['cases']:
                base=case['base_case'];role=case['role'];dest=local(seed,role);nr.prepare_case(base['job'],dest)
                gen=clean.CleanPanelGenerator(base);gen.count_seed=seed;assert gen.order==case['original_generation_target_order']
                rng=np.random.default_rng(seed+100);gen.donors={t:rng.choice(3000,400,replace=False) for t in gen.order}
                donor_path=dest/f'DONOR_DRAW__{arm}.json';clean.fresh(donor_path,dict(seed=seed,donor_seed=seed+100,pool_rows=3000,pool_SHA256=base['leaves']['generation_pool.npz']['sha256'],original_order=gen.order,draws={t:gen.donors[t].tolist() for t in gen.order}))
                br=next(r for r in records if r['seed']==seed and r['role']==role and r['arm']==BASE)
                bind_candidate(gen,case,br,donor_path);baseline=ad.read_h5ad(br['prediction_path']);labels0=np.asarray(baseline.obs.target_gene.astype(str))
                assert baseline.shape==(3200,len(gen.genes));np.testing.assert_array_equal(baseline.var_names.astype(str).to_numpy(),gen.genes)
                bm=np.load(br['moments_path'],allow_pickle=False);np.testing.assert_array_equal(bm['targets'].astype(str),gen.targets);base_expected=bm['expected_and_sample_mean'][:,0].copy();bm.close()
                blocks=[];labels=[];diags=[];moments=[]
                for i,target in enumerate(gen.targets):
                    result=gen.build_target(target,'N1');block=result.block;editable=result.moments['editable_native_positions'];protected=np.ones(len(gen.genes),bool);protected[editable]=False
                    old=baseline.X[labels0==target];assert old.shape==block.shape and (block[:,protected]!=old[:,protected]).nnz==0
                    expected=np.asarray(result.moments['conditional_given_H0_expected_gene_mean'],np.float64);sample=np.asarray(block.mean(0)).ravel().astype(float)
                    assert np.isfinite(block.data).all() and (block.data>=0).all() and np.equal(block.data,np.rint(block.data)).all()
                    d=copy.deepcopy(result.diagnostics);d.update(arm=arm,protected_counts_equal_same_seed_B0=True,conditional_expected_library_delta_vs_same_seed_B0=float(expected.sum()-base_expected[i].sum()),same_original_donor_receipt_sha256=nr.sha(donor_path))
                    blocks.append(block);labels.extend([target]*400);diags.append(d);moments.append([expected,sample]);resources(start)
                matrix=sparse.vstack(blocks,format='csr');output=dest/f'pred__{arm}.h5ad';assert not output.exists()
                obj=ad.AnnData(matrix,obs=pd.DataFrame({'target_gene':labels},index=[f'{arm}_{i}' for i in range(3200)]),var=pd.DataFrame(index=gen.genes));obj.write_h5ad(output,compression='gzip')
                moment=dest/f'{arm}_moments.npz';np.savez_compressed(moment,targets=np.asarray(gen.targets),genes=gen.genes,expected_and_sample_mean=np.asarray(moments))
                r=dict(completed=True,seed=seed,role=role,context=case['context'],arm=arm,strict_reuse=False,cells=3200,cells_per_target=400,count_seed=seed,donor_seed=seed+100,
                    prediction_path=str(output),prediction_sha256=nr.sha(output),moments_path=str(moment),moments_sha256=nr.sha(moment),protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),source_identity=copy.deepcopy(gen.source_identity),
                    donor_receipt_path=str(donor_path),donor_receipt_sha256=nr.sha(donor_path),targets=diags,original_donor_receipts=[dict(target=t,donor_rows=gen.donors[t].tolist()) for t in gen.targets],truth_read_by_generation=False,newfits=0,newjudge=0,official_calls=0)
                manifest=dest/f'{arm}_generation.json';clean.fresh(manifest,r);r.update(generation_manifest_path=str(manifest),generation_manifest_sha256=nr.sha(manifest));records.append(copy.deepcopy(r))
                print('CONDITIONAL_KRR_SEEDREP_GENERATED',seed,role,'new',len(records)-24,flush=True);del obj,matrix,blocks,baseline,gen
        # Re-read every saved manifest: no mutable-dictionary alias can enter seal.
        reconstructed=[]
        for r in records:
            rr=nr.read(r['generation_manifest_path']);rr.update(generation_manifest_path=r['generation_manifest_path'],generation_manifest_sha256=r['generation_manifest_sha256']);assert rr==r;reconstructed.append(copy.deepcopy(rr))
        clean.fresh(OUT/'PREDICTIONS_SEALED.json',dict(completed=True,protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),records=reconstructed,total_outputs=36,new_outputs=12,reused_outputs=24,all36_before_new_judge=True,newjudge_at_seal=0,selected_arm=arm))
        validate_seal(p);verify_protocol(p);clean.fresh(OUT/'GENERATION_COMPLETE.json',dict(completed=True,prediction_seal_sha256=nr.sha(OUT/'PREDICTIONS_SEALED.json'),new_outputs=12,reused_outputs=24,newjudge=0,resources=resources(start)))
    finally:(OUT/'GENERATING.lock').unlink()

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('phase',choices=('draft','freeze','generate','validate'));cli.add_argument('--draft-path',type=Path,default=HERE/'CONDITIONAL_PROTOCOL_DRAFT.json');cli.add_argument('--root-ticket',type=Path);a=cli.parse_args();os.chdir(ROOT)
    try:
        if a.phase=='draft':draft(a.draft_path,a.root_ticket)
        elif a.phase=='freeze':assert a.root_ticket;freeze(a.root_ticket)
        elif a.phase=='generate':generate()
        else:validate_seal(verify_protocol(nr.read(OUT/'PROTOCOL.json')))
    except BaseException as e:
        folder=OUT if OUT.exists() else HERE;clean.fresh(folder/f'FAILED_CONDITIONAL_{a.phase}_{time.time_ns()}.json',dict(completed=False,error=repr(e),traceback=traceback.format_exc(),producer_sha256=nr.sha(CODE),newfits=0,newjudge=0,official_calls=0));raise
