"""Generation-only three same-source mean methods; literal original NB400.

Consumes sealed, independently audited means. No source-label reads, fit or judge.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse,copy,json,os,sys,time,traceback,shutil
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[key]='4'
import run_clean400 as clean
import run_null400 as rt
nr,np,pd=clean.nr,clean.np,clean.pd
from scipy import sparse
ROOT=clean.ROOT;HERE=Path(__file__).resolve().parent;CODE=Path(__file__).resolve()
BASE=ROOT/'runs/research_v28_20261005/clean400_v1'
MEANS=ROOT/'runs/research_v28_20261005/ntc_context_krr_native_mean_v1'
OUT=ROOT/'runs/research_v28_20261005/ntc_context_krr_native400_v1'
BASELINE='clean_B0_NB'
MEAN_ARMS=('context_KRR','equalcomponent_uniform','positive_RBF')
NEW_ARMS=tuple(a+'_NB' for a in MEAN_ARMS);ARMS=(BASELINE,)+NEW_ARMS
SCORER=HERE/'score_context_KRR400.py'
DECISION=ROOT/'reports/research_v28_20261005/CONTEXT_KRR400_ROOT_SCREEN_DECISION_CN.md'
GENPY=ROOT/'envs/foundation312/Scripts/python.exe'
SEED=20260928;CELLS=400;MAX_SECONDS=900.;MAX_BYTES=1024*2**20

def check(path,h):
    assert nr.sha(path)==h,'Frozen dependency changed: '+str(path)

def resources(start=None):
    import psutil
    mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss)
    elapsed=time.monotonic()-start if start is not None else None
    total=sum(q.stat().st_size for a in NEW_ARMS for q in OUT.glob('*/pred__'+a+'.h5ad'))
    result=dict(peak_RSS_MiB=peak/2**20,elapsed_seconds=elapsed,new_prediction_bytes=total,C_free_GiB=shutil.disk_usage(ROOT).free/2**30)
    assert peak<2*2**30 and total<=MAX_BYTES and result['C_free_GiB']>=50,result
    assert elapsed is None or elapsed<=MAX_SECONDS,result
    return result

def build_protocol():
    assert Path(sys.executable).resolve()==GENPY.resolve() and np.__version__=='2.5.3'
    resources();bp=nr.read(BASE/'PROTOCOL.json');bs=nr.read(BASE/'PREDICTIONS_SEALED.json')
    assert nr.read(BASE/'COMPLETE.json')['completed'] and bs['completed']
    assert bs['protocol_sha256']==nr.sha(BASE/'PROTOCOL.json')
    assert bp['count_seed']==SEED and bp['donor_seed']==SEED+100 and bp['cell_count']==CELLS
    means=nr.read(MEANS/'MEANS_SEALED.json');gate=nr.read(MEANS/'MEAN_INDEPENDENT_AUDIT.json')
    assert means['completed'] and len(means['records'])==18 and tuple(means['arms'])==MEAN_ARMS
    assert means['held_response_reads']==means['THP_numeric_IO']==means['count_calls']==means['judge_calls']==0
    assert gate['passed'] and gate['mean_seal_sha256']==nr.sha(MEANS/'MEANS_SEALED.json')
    bundles={(r['role'],r['arm']):r for r in means['records']}
    assert len(bundles)==18
    deps=dict(bp['dependencies']);deps.update(rt.generation_runtime_assets())
    for q in (CODE,SCORER,DECISION,HERE/'run_null400.py',HERE/'run_clean400.py',HERE/'run_trace400.py',
              MEANS/'PROTOCOL.json',MEANS/'MEANS_SEALED.json',MEANS/'MEAN_INDEPENDENT_AUDIT.json',
              BASE/'PROTOCOL.json',BASE/'PREDICTIONS_SEALED.json',BASE/'COMPLETE.json'):
        assert q.is_file(),q;deps[str(q.resolve())]=nr.sha(q)
    for q,h in nr.read(MEANS/'PROTOCOL.json')['bindings_SHA256'].items():check(q,h);deps[q]=h
    old={(r['role'],r['arm']):r for r in bs['records']};cases=[];reuses=[]
    for original in bp['cases']:
        for leaf in original['leaves'].values():check(leaf['path'],leaf['sha256'])
        source=old[(original['role'],BASELINE)];score=BASE/original['role']/'scores'/BASELINE
        files={name:rt.spec(source[name+'_path'],source[name+'_sha256']) for name in ('prediction','moments')}
        files['generation']=rt.spec(BASE/original['role']/f'{BASELINE}_generation.json')
        files.update({'score/'+q.relative_to(score).as_posix():rt.spec(q) for q in sorted(score.rglob('*')) if q.is_file()})
        reuse=dict(seed=SEED,role=original['role'],context=original['context'],arm=BASELINE,files=files,
                   source_record=copy.deepcopy(source),source_score_dir=str(score),source_protocol_SHA256=nr.sha(BASE/'PROTOCOL.json'))
        chosen={}
        for arm,meanarm in zip(NEW_ARMS,MEAN_ARMS):
            b=copy.deepcopy(bundles[(original['role'],meanarm)])
            for k,hk in (('bundle_path','bundle_sha256'),('complete_path','complete_SHA256'),('source_model_seal_path','source_model_seal_SHA256')):
                check(b[k],b[hk]);deps[b[k]]=b[hk]
            with np.load(original['leaves']['B0_effects']['path'],allow_pickle=False) as a,np.load(b['bundle_path'],allow_pickle=False) as z:
                assert set(a.files)==set(z.files)
                for k in a.files:
                    if k!='effect':np.testing.assert_array_equal(a[k],z[k])
                effect=z['effect'];assert effect.dtype==np.float32 and effect.shape==a['effect'].shape and np.isfinite(effect).all()
                assert np.count_nonzero(effect[~a['editable_mask'].astype(bool)])==0
            chosen[arm]=b
        cases.append(dict(role=original['role'],context=original['context'],base_case=copy.deepcopy(original),baseline=copy.deepcopy(reuse),mean_bundles=chosen));reuses.append(reuse)
    for reuse in reuses:
        for spec in reuse['files'].values():deps[spec['path']]=spec['sha256']
    for q,h in deps.items():check(q,h)
    return dict(schema='Native400_context_KRR_same_public19_sources_originalNB_v1',frozen=True,
        created_utc=datetime.now(timezone.utc).isoformat(),code_sha256=nr.sha(CODE),runtime=rt.runtime(),
        generation_runtime_assets=rt.generation_runtime_assets(),dependencies=deps,pending_bindings=[],
        cases=cases,reuse_records=reuses,arms=list(ARMS),seeds=[SEED],count_seed=SEED,donor_seed=SEED+100,cells_per_target=CELLS,
        total_outputs=24,new_outputs=18,reused_outputs=6,evaluation_sha256=bp['evaluation_sha256'] if 'evaluation_sha256' in bp else 'c8a719bda010e352af4dd7c433313c795eab5460e709897a95b474cd9091e35c',
        source_scope='Versioned public19 exact same-target labels; same sources and masks for all3 mean arms; no heldNative query rows',
        mean_intervention='Replace only total effect by masked contextKRR or equalcomponentuniform or positiveRBF; original B0 retained control',
        generator='Exact original CleanPanelGenerator/build_target N1; role theta, structural6119 axis, seen, own, original400donors and NB namespace unchanged',
        effect_contract='F32 rawlog2PC0.5, same original B0 editable mask; predictor F64 nominal mean is not a count mean lock',
        controlled_contrasts={'context_kernel_inverse':'context_KRR_NB minus positive_RBF_NB','context_vs_uniform':'context_KRR_NB minus equalcomponent_uniform_NB'},
        all24_predictions_before_judge=True,root_new_judge_cases=18,root_judge_processes=1,root_judge_threads=4,
        primary_promotion_metric='localoverall versus same protocol B0NB; positive initial screen requires fixed3seed paired validation',
        aggregation=bp['aggregation'],metric_weights=bp['metric_weights'],benchmark_status=bp['benchmark_status'],
        budgets=dict(CPU_threads=4,numeric_seconds=MAX_SECONDS,RSS_MiB=2048,new_prediction_bytes=MAX_BYTES,min_C_free_GiB=50),
        normalized_mean_invariance_claimed=False,truth_read_by_generation=False,newfits=0,THP_reads=0,official_calls=0)

def validate_protocol(p):
    assert p['frozen'] and p['schema']=='Native400_context_KRR_same_public19_sources_originalNB_v1'
    assert tuple(p['arms'])==ARMS and p['code_sha256']==nr.sha(CODE) and p['runtime']==rt.runtime()
    for q,h in p['dependencies'].items():check(q,h)
    return p

def freeze():
    assert not (OUT/'PROTOCOL.json').exists(),'Keep all previous frozen/partial attempts'
    p=build_protocol();clean.fresh(OUT/'PROTOCOL.json',p)
    print('CONTEXT_KRR400_PROTOCOL',nr.sha(OUT/'PROTOCOL.json'),flush=True)

def bind_candidate(gen,b,c):
    with np.load(c['leaves']['B0_effects']['path'],allow_pickle=False) as a,np.load(b['bundle_path'],allow_pickle=False) as z:
        assert set(a.files)==set(z.files)
        for k in a.files:
            if k!='effect':np.testing.assert_array_equal(a[k],z[k])
        gen.effects=z['effect'].copy();assert gen.effects.dtype==np.float32
    gen.source_identity=copy.deepcopy(gen.source_identity)
    gen.source_identity.update(mean_path=b['bundle_path'],mean_sha256=b['bundle_sha256'],mean_workflow=b['arm'],
        context_mean_protocol_path=str(MEANS/'PROTOCOL.json'),context_mean_protocol_SHA256=nr.sha(MEANS/'PROTOCOL.json'),
        source_model_seal_path=b['source_model_seal_path'],source_model_seal_SHA256=b['source_model_seal_SHA256'])

def validate_seal(p):
    s=nr.read(OUT/'PREDICTIONS_SEALED.json');assert s['completed'] and s['protocol_sha256']==nr.sha(OUT/'PROTOCOL.json')
    assert len(s['records'])==24 and {(r['seed'],r['role'],r['arm']) for r in s['records']}=={(SEED,c['role'],a) for c in p['cases'] for a in ARMS}
    assert sum(r['strict_reuse'] for r in s['records'])==6
    for r in s['records']:
        for key in ('prediction','moments','generation_manifest'):check(r[key+'_path'],r[key+'_sha256'])
        m=nr.read(r['generation_manifest_path']);assert m['source_identity']==r['source_identity'] and m['protocol_sha256']==nr.sha(OUT/'PROTOCOL.json')
    return s

def generate():
    import anndata as ad
    ad.settings.allow_write_nullable_strings=True
    p=validate_protocol(nr.read(OUT/'PROTOCOL.json'));assert not (OUT/'PREDICTIONS_SEALED.json').exists()
    start=time.monotonic();records=[];lock=OUT/'GENERATING.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        for item in p['cases']:
            c=item['base_case'];dest=OUT/c['role'];nr.prepare_case(c['job'],dest)
            reuse=item['baseline'];r=copy.deepcopy(reuse['source_record'])
            for key,spec in reuse['files'].items():
                name={'prediction':f'pred__{BASELINE}.h5ad','moments':f'{BASELINE}_moments.npz','generation':f'{BASELINE}_original_generation.json'}.get(key)
                if name:rt.link(spec['path'],dest/name,spec['sha256'])
                elif key.startswith('score/'):rt.link(spec['path'],dest/'scores'/BASELINE/key.removeprefix('score/'),spec['sha256'])
            # Separate local wrapper binds strict source bytes to this protocol.
            r.update(seed=SEED,strict_reuse=True,protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),
                     prediction_path=str(dest/f'pred__{BASELINE}.h5ad'),moments_path=str(dest/f'{BASELINE}_moments.npz'))
            manifest=dest/f'{BASELINE}_generation.json';clean.fresh(manifest,r)
            r.update(generation_manifest_path=str(manifest),generation_manifest_sha256=nr.sha(manifest));records.append(r)
            baseline=ad.read_h5ad(dest/f'pred__{BASELINE}.h5ad')
            for arm in NEW_ARMS:
                b=item['mean_bundles'][arm];gen=clean.CleanPanelGenerator(c);bind_candidate(gen,b,c)
                output=dest/f'pred__{arm}.h5ad';assert not output.exists()
                blocks=[];labels=[];diagnostics=[];moments=[];donor_receipts=[]
                for target in gen.targets:
                    result=gen.build_target(target,'N1');block=result.block
                    assert block.shape==(400,len(gen.genes)) and np.isfinite(block.data).all() and (block.data>=0).all() and np.equal(block.data,np.rint(block.data)).all()
                    editable=result.moments['editable_native_positions'];protected=np.ones(len(gen.genes),bool);protected[editable]=False
                    original=baseline.X[np.asarray(baseline.obs.target_gene.astype(str))==target]
                    assert (block[:,protected]!=original[:,protected]).nnz==0,'Protected original H0 changed'
                    expected=np.asarray(result.moments['conditional_given_H0_expected_gene_mean'],np.float64)
                    sample=np.asarray(block.mean(0)).ravel().astype(np.float64)
                    d=copy.deepcopy(result.diagnostics);d.update(arm=arm,protected_counts_equal_B0=True)
                    blocks.append(block);labels.extend([target]*CELLS);diagnostics.append(d);moments.append([expected,sample])
                    donor_receipts.append(dict(target=target,donor_rows=gen.donors[target].astype(int).tolist()))
                    resources(start)
                matrix=sparse.vstack(blocks,format='csr');assert matrix.shape==(3200,len(gen.genes))
                obj=ad.AnnData(matrix,obs=pd.DataFrame({'target_gene':labels},index=[f'{arm}_{i}' for i in range(len(labels))]),var=pd.DataFrame(index=gen.genes))
                obj.write_h5ad(output,compression='gzip');momentpath=dest/f'{arm}_moments.npz'
                np.savez_compressed(momentpath,targets=np.asarray(gen.targets),genes=gen.genes,expected_and_sample_mean=np.asarray(moments))
                r=dict(completed=True,seed=SEED,role=c['role'],context=c['context'],arm=arm,strict_reuse=False,
                    prediction_path=str(output),prediction_sha256=nr.sha(output),moments_path=str(momentpath),moments_sha256=nr.sha(momentpath),
                    protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),cells=3200,cells_per_target=400,count_seed=SEED,donor_seed=SEED+100,
                    source_identity=copy.deepcopy(gen.source_identity),targets=diagnostics,original_donor_receipts=donor_receipts,
                    truth_read_by_generation=False,newfits=0,newjudge=0,official_calls=0)
                manifest=dest/f'{arm}_generation.json';clean.fresh(manifest,r)
                r.update(generation_manifest_path=str(manifest),generation_manifest_sha256=nr.sha(manifest));records.append(r)
                print('CONTEXT_KRR400_GENERATED',c['role'],arm,resources(start),flush=True)
                del obj,matrix,blocks,gen
            del baseline
        clean.fresh(OUT/'PREDICTIONS_SEALED.json',dict(completed=True,protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),records=records,
            all24_predictions_before_judge=True,actual_new_count_outputs=18,strict_reused_count_outputs=6,official_calls=0))
        validate_seal(p);validate_protocol(p)
        clean.fresh(OUT/'GENERATION_COMPLETE.json',dict(completed=True,prediction_seal_sha256=nr.sha(OUT/'PREDICTIONS_SEALED.json'),resources=resources(start),
            new_outputs=18,reused_outputs=6,newjudge=0,official_calls=0))
    finally:lock.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('freeze','generate','validate'));a=parser.parse_args();os.chdir(ROOT)
    try:
        if a.phase=='freeze':freeze()
        elif a.phase=='generate':generate()
        else:validate_seal(validate_protocol(nr.read(OUT/'PROTOCOL.json')))
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True);clean.fresh(OUT/f'FAILED_{a.phase}_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),official_calls=0));raise
