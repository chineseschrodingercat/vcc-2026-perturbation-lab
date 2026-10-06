"""Generation-only Native400 null controls draft. No fit, truth IO, or judge."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,ctypes,json,os,sys,time,traceback,shutil
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[k]='4'
import run_clean400 as clean
nr,np,pd=clean.nr,clean.np,clean.pd
from scipy import sparse
import scipy
from threadpoolctl import threadpool_info
ROOT=clean.ROOT;HERE=Path(__file__).resolve().parent;CODE=Path(__file__).resolve()
BASE=ROOT/'runs/research_v28_20261005/clean400_v1'
OUT=ROOT/'runs/research_v28_20261005/null400_v1'
DRAFT=ROOT/'reports/research_v28_20261005/null400_PROTOCOL_DRAFT_revision3.json'
SCORER=HERE/'score_null400.py'
DECISION=ROOT/'reports/research_v28_20261005/NULL400_ROOT_SCREEN_DECISION_CN.md'
INVENTORY=ROOT/'reports/research_v28_20261005/NATIVE400_NULL_CONTROL_INVENTORY_AGENT.json'
BASELINE='clean_B0_NB';NBNULL='zero_response_NB';IDENTITY='zero_response_NTC_identity'
ARMS=(BASELINE,NBNULL,IDENTITY);NEW_ARMS=ARMS[1:]
COUNT_SEED=20260928;DONOR_SEED=COUNT_SEED+100;CELLS=400;MAX_SECONDS=600
GENPY=ROOT/'envs/foundation312/Scripts/python.exe'

def verify(path,h):
    if nr.sha(path)!=h:raise ValueError('Dependency drift: '+str(path))
def check(s):verify(s['path'],s['sha256'])
def spec(path,h=None):
    path=Path(path).resolve();actual=nr.sha(path)
    if h is not None and h!=actual:raise ValueError('Bound source drift: '+str(path))
    return dict(path=str(path),sha256=actual)
def loaded_python_DLL():
    """Actual loaded runtime using the existing KRR WinAPI contract."""
    from ctypes import wintypes
    assert os.name=='nt'
    fn=ctypes.windll.kernel32.GetModuleFileNameW
    fn.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD];fn.restype=wintypes.DWORD
    buf=ctypes.create_unicode_buffer(32768)
    n=fn(ctypes.pythonapi._handle,buf,len(buf));assert 0<n<len(buf)
    p=Path(buf.value).resolve();assert p.name.lower()=='python312.dll' and p.is_file()
    return p
def loaded_binary_receipt():
    python=loaded_python_DLL();pools=threadpool_info();blas=[p for p in pools if p['user_api']=='blas']
    assert blas and any('numpy.libs' in str(p['filepath']).replace('\\','/') for p in blas)
    bins={'actual_loaded_python312_DLL':python}
    bins.update({'actual_loaded_BLAS::'+str(Path(p['filepath']).resolve()):Path(p['filepath']).resolve() for p in blas})
    return dict(actual_binaries={k:spec(p) for k,p in bins.items()},threadpools=pools,
        guessed_base_pythonDLL=str(Path(sys.base_prefix)/'python312.dll'),
        guessed_base_matches_actual=(Path(sys.base_prefix)/'python312.dll').resolve()==python)
def runtime():return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,interpreter=str(Path(sys.executable).resolve()),loaded_binary_receipt=loaded_binary_receipt())
def resources(start=None):
    class PM(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    p=PM();p.cb=ctypes.sizeof(p);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(PM),ctypes.c_ulong]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb):raise OSError('Memory receipt unavailable')
    predbytes=sum(q.stat().st_size for a in NEW_ARMS for q in OUT.glob('*/pred__'+a+'.h5ad'))
    elapsed=time.monotonic()-start if start is not None else None
    r=dict(RSS_MiB=p.WorkingSetSize/2**20,peak_RSS_MiB=p.PeakWorkingSetSize/2**20,C_free_GiB=shutil.disk_usage(ROOT).free/2**30,new_prediction_bytes=predbytes,elapsed_seconds=elapsed)
    if r['peak_RSS_MiB']>=2048 or r['C_free_GiB']<50 or predbytes>512*2**20 or (elapsed is not None and elapsed>MAX_SECONDS):raise RuntimeError('Preserve fixed resource failure: '+str(r))
    return r
def generation_runtime_assets():
    site=ROOT/'envs/foundation312/Lib/site-packages';receipt=loaded_binary_receipt()
    files=[GENPY]+[Path(s['path']) for s in receipt['actual_binaries'].values()]
    files.extend(sorted((site/'numpy.libs').glob('*.dll')))
    files.append(site/'threadpoolctl.py')
    for area in ('numpy/random','numpy/_core','scipy/sparse','h5py'):
        found=sorted((site/area).rglob('*.pyd'));assert found,area;files.extend(found)
    for pattern in ('numpy-*.dist-info','scipy-*.dist-info','h5py-*.dist-info','anndata-*.dist-info','pandas-*.dist-info','threadpoolctl-*.dist-info'):
        dirs=sorted(site.glob(pattern));assert len(dirs)==1,pattern
        for name in ('METADATA','RECORD'):files.append(dirs[0]/name)
    return {str(p.resolve()):nr.sha(p) for p in files}
def link(source,dest,h):
    source,dest=Path(source),Path(dest);verify(source,h);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():verify(dest,h);return
    try:os.link(source,dest)
    except OSError:shutil.copy2(source,dest)
    verify(dest,h)
def normalized(matrix):
    library=matrix.sum(1);assert np.isfinite(library).all() and (library>0).all()
    cp=matrix/library[:,None]*1e6
    return cp.mean(0),np.log1p(cp).mean(0),library
def rms(v):return float(np.sqrt(np.mean(np.asarray(v)**2)))

def build_protocol():
    resources();assert Path(sys.executable).resolve()==GENPY.resolve() and np.__version__=='2.5.3'
    bp=nr.read(BASE/'PROTOCOL.json');seal=nr.read(BASE/'PREDICTIONS_SEALED.json');assert seal['completed'] and nr.read(BASE/'COMPLETE.json')['completed']
    assert seal['protocol_sha256']==nr.sha(BASE/'PROTOCOL.json') and bp['count_seed']==COUNT_SEED and bp['donor_seed']==DONOR_SEED and bp['cell_count']==CELLS
    inventory=nr.read(INVENTORY);assert inventory['completed'] and not inventory['same_protocol_zero_response_NB_found'] and not inventory['same_protocol_complete_NTC_identity_found']
    lookup={(r['role'],r['arm']):r for r in seal['records']};cases=[];reuses=[];deps=dict(bp['dependencies'])
    assets=generation_runtime_assets();deps.update(assets)
    paths=[CODE,HERE/'run_clean400.py',HERE/'run_trace400.py',ROOT/'scripts/native_reproducer.py',INVENTORY,
        nr.SOURCE/'literal_nb_v28.py',nr.SOURCE/'latent_counts_v26.py',nr.SOURCE/'generate_counts.py',nr.SOURCE/'residual_v6_common.py',
        nr.SOURCE/'prepare_response_inputs_v23.py',ROOT/'data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz',
        BASE/'PROTOCOL.json',BASE/'PREDICTIONS_SEALED.json',BASE/'COMPLETE.json']
    pending=[]
    for q in (SCORER,DECISION):
        if q.exists():paths.append(q)
        else:pending.append(str(q))
    for q in paths:deps[str(q.resolve())]=nr.sha(q)
    for c in bp['cases']:
        for s in c['leaves'].values():check(s)
        r=lookup[(c['role'],BASELINE)];assert r['count_seed']==COUNT_SEED and r['donor_seed']==DONOR_SEED and r['cells']==3200 and r['cells_per_target']==CELLS
        score=BASE/c['role']/'scores'/BASELINE;assert (score/'summary.json').exists()
        files={n:spec(r[n+'_path'],r[n+'_sha256']) for n in ('prediction','moments')}
        files['generation']=spec(BASE/c['role']/f'{BASELINE}_generation.json')
        files.update({'score/'+q.relative_to(score).as_posix():spec(q) for q in sorted(score.rglob('*')) if q.is_file()})
        assert r['source_identity']['mean_sha256']==c['leaves']['B0_effects']['sha256'] and r['source_identity']['compact_theta_sha256']==c['leaves']['theta_seen']['sha256']
        reuse=dict(seed=COUNT_SEED,role=c['role'],context=c['context'],arm=BASELINE,files=files,source_record=copy.deepcopy(r),source_score_dir=str(score),source_protocol_SHA256=nr.sha(BASE/'PROTOCOL.json'))
        cases.append(dict(role=c['role'],context=c['context'],base_case=copy.deepcopy(c),baseline=copy.deepcopy(reuse)));reuses.append(reuse)
    for q,h in deps.items():verify(q,h)
    assert len(cases)==len(reuses)==6
    return dict(schema='Native400_zeroresponse_NB_and_identity_v1',created_utc=datetime.now(timezone.utc).isoformat(),frozen=False,
        code_sha256=nr.sha(CODE),runtime=runtime(),generation_runtime_assets=assets,dependencies=deps,pending_bindings=pending,
        cases=cases,reuse_records=reuses,arms=list(ARMS),seeds=[COUNT_SEED],count_seed=COUNT_SEED,donor_seed=DONOR_SEED,cells_per_target=CELLS,
        reused_outputs=6,new_outputs=12,total_outputs=18,
        mean_intervention='zero_response_NB: set the complete original10916 rawPC.5 effect matrix to exactF32 zero. Recompute literal original count_ratio using original sparseF32mean promotedF64; require every fullnative ratio exactly1 or fail, never force/clip to1.',
        NB_law='Exact original literal_nb_v28 build_target body and integer_sample(mu=D_editable,originaltheta,countseed,originaltargetorder,nb), original seen/nonown/input mask. No new sampler namespace.',
        neutral_H0='Zero-response original stochastic-round transform must equal all-axis original integer donor D exactly; protected zeroNB columns equal donor D and retained B0NB protected H0.',
        identity_law='Complete native-axis original400 donor integer CSR copied directly, all genes including own; no NB or count rounding added. All genes/targets remain.',
        donor_rule='Default_rng(seed+100) choice(3000,400,replaceFalse) in original gen.order, independently replayed; original target index controls NB RNG.',
        preserved='Original source assets/checkpoints/theta/seen/masks/genes/refs/anchors/real target population and original400 donors. No fitting or inference model changes beyond declared total effect zero.',
        comparisons={'mean_only':NBNULL+' minus '+BASELINE,'countlaw_at_zero_response':IDENTITY+' minus '+NBNULL,
            'joint_change_caution':IDENTITY+' minus '+BASELINE+' changes both totalmean and countlaw; do not label a mean-only or independent generator gain.'},
        conditional_mean='Both null arms given donor panel have exact raw conditional mean D per gene, original noneditable/own fixed. Sampled zeroNB raw and normalized means vary; identity retains observed D exactly.',
        diagnostics='Full-axis conditional/sample gene mean/variance/zero fraction, original donor hashes, actual libraries, CP1M arithmetic mean and mean log1pCP. No finalDE/p-value invariant claim.',
        variance_definition='total_D_mixture_variance is the marginal variance after uniformly drawing one of this fixed400 donor intensities then applying the conditional countlaw: Var_D(mu)+mean_D(conditionalvariance). It is not the expectation of a sample variance over400 fixed heterogeneous conditional draws; the integer panel uses those400 nonidentical D values once each.',
        all18_outputs_sealed_before_new_judge=True,root_new_judge_cases=12,root_judge_processes=1,root_judge_threads=4,
        root_primary_metric='Frozen localoverall with original anchors; no sixmetric simultaneous-rise gate; null controls are diagnostic contrasts.',
        aggregation='Three tuned development backgrounds equal; two roles perbg and cells/genes are not independent model repeats.',
        benchmark_status=bp['benchmark_status'],cache_policy='New isolated candidate score cases; old B0 score bytes strictly copied/hashbound.',
        budgets=dict(CPU_threads=4,seconds=MAX_SECONDS,RSS_MiB=2048,new_prediction_bytes=512*2**20,C_free_GiB=50),
        newfits=0,newmeaninference=0,generator_destinationtruthreads=0,THP_reads=0,newjudge=0,official_calls=0)
def validate_protocol(p):
    assert p['schema']=='Native400_zeroresponse_NB_and_identity_v1' and tuple(p['arms'])==ARMS and p['seeds']==[COUNT_SEED]
    assert not p['pending_bindings'] and p['code_sha256']==nr.sha(CODE) and p['runtime']==runtime()
    for q,h in p['dependencies'].items():verify(q,h)
    for c in p['cases']:
        for s in c['base_case']['leaves'].values():check(s)
        for s in c['baseline']['files'].values():check(s)
    return p
def freeze():
    assert not (OUT/'PROTOCOL.json').exists(),'Keep frozen and partial attempts'
    p=nr.read(DRAFT);validate_protocol(p);p=copy.deepcopy(p);p['frozen']=True;p['status']='FROZEN_GENERATION_ONLY_WAIT_ROOT_SCHEDULE'
    p['freeze_utc']=datetime.now(timezone.utc).isoformat();p['draft_protocol_sha256']=nr.sha(DRAFT)
    clean.fresh(OUT/'PROTOCOL.json',p);print('PROTOCOL',nr.sha(OUT/'PROTOCOL.json'),flush=True)
def reuse(c):
    r0=c['baseline'];files=r0['files'];dest=OUT/c['role'];nr.prepare_case(c['base_case']['job'],dest)
    for name,path in [('prediction',dest/f'pred__{BASELINE}.h5ad'),('moments',dest/f'{BASELINE}_moments.npz'),('generation',dest/f'{BASELINE}_SOURCE_generation.json')]:link(files[name]['path'],path,files[name]['sha256'])
    for name,s in files.items():
        if name.startswith('score/'):link(s['path'],dest/'scores'/BASELINE/name.removeprefix('score/'),s['sha256'])
    r=dict(completed=True,seed=COUNT_SEED,role=c['role'],context=c['context'],arm=BASELINE,strict_reuse=True,
        prediction_path=str(dest/f'pred__{BASELINE}.h5ad'),prediction_sha256=files['prediction']['sha256'],moments_path=str(dest/f'{BASELINE}_moments.npz'),moments_sha256=files['moments']['sha256'],
        source_identity=copy.deepcopy(r0['source_record']['source_identity']),source_files=copy.deepcopy(files),source_record=copy.deepcopy(r0['source_record']),
        protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),count_seed=COUNT_SEED,donor_seed=DONOR_SEED,cells=3200,cells_per_target=CELLS,newcount=0,newjudge=0,scores_strictly_reused=True)
    path=dest/f'{BASELINE}_generation.json';clean.fresh(path,r);r.update(generation_manifest_path=str(path),generation_manifest_sha256=nr.sha(path));return r
def validate_seal(p):
    s=nr.read(OUT/'PREDICTIONS_SEALED.json');assert s['completed'] and s['protocol_sha256']==nr.sha(OUT/'PROTOCOL.json') and len(s['records'])==18
    assert {(r['seed'],r['role'],r['arm']) for r in s['records']}=={(COUNT_SEED,c['role'],a) for c in p['cases'] for a in ARMS}
    assert sum(r['strict_reuse'] for r in s['records'])==6
    for r in s['records']:
        assert r['count_seed']==COUNT_SEED and r['donor_seed']==DONOR_SEED and r['cells']==3200 and r['cells_per_target']==CELLS
        for name,q in r.items():
            if name.endswith('_path') and name[:-5]+'_sha256' in r:verify(q,r[name[:-5]+'_sha256'])
        m=nr.read(r['generation_manifest_path']);assert m['protocol_sha256']==s['protocol_sha256'] and m['source_identity']==r['source_identity']
    return s
def generate():
    p=validate_protocol(nr.read(OUT/'PROTOCOL.json'));assert p['frozen'] and not (OUT/'PREDICTIONS_SEALED.json').exists()
    start=time.monotonic();resources(start)
    with (OUT/'GENERATING.lock').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
    try:
        import anndata as ad
        ad.settings.allow_write_nullable_strings=True
        records=[reuse(c) for c in p['cases']]
        for c in p['cases']:
            dest=OUT/c['role'];gen=clean.CleanPanelGenerator(c['base_case']);gen.count_seed=COUNT_SEED
            donor_rng=np.random.default_rng(DONOR_SEED);replayed={t:donor_rng.choice(3000,CELLS,replace=False) for t in gen.order}
            for t in gen.order:np.testing.assert_array_equal(gen.donors[t],replayed[t])
            f=gen._scientific;zero=np.zeros_like(gen.effects,dtype=np.float32);gen.effects=zero
            assert np.count_nonzero(gen.effects)==0
            for ti,t in enumerate(gen.targets):
                ratio=f['count_ratio'](zero[ti],gen.mean,gen.genes,gen.output_genes)
                np.testing.assert_array_equal(ratio,np.ones(len(gen.genes),dtype=np.float64))
            old=c['baseline']['source_record'];oldby={d['target']:d for d in old['targets']}
            baseobj=ad.read_h5ad(c['baseline']['files']['prediction']['path'])
            np.testing.assert_array_equal(baseobj.var_names.astype(str).to_numpy(),gen.genes)
            np.testing.assert_array_equal(baseobj.obs['target_gene'].astype(str).to_numpy(),np.repeat(gen.targets,CELLS))
            baseline=baseobj.X.tocsr();del baseobj
            assert baseline.shape==(3200,len(gen.genes))
            for arm in NEW_ARMS:
                blocks=[];labels=[];diags=[];mom=[];vars_=[];zeros_=[];profiles=[];libs=[]
                for ti,t in enumerate(gen.targets):
                    resources(start);D=gen.pool[gen.donors[t]].tocsr(copy=True);denseD=D.toarray().astype(np.float64)
                    assert f['array_sha'](gen.donors[t])==oldby[t]['donor_pool_rows_sha256']
                    seen=gen.inputs['seen'].astype(bool);own=gen.inputs['ownmask'][ti].astype(bool)
                    eo=gen.input_output[seen[gen.input_output]&own[gen.input_output]];en=gen.input_native[seen[gen.input_output]&own[gen.input_output]]
                    protected=np.ones(len(gen.genes),bool);protected[en]=False
                    assert (D[:,protected]!=baseline[ti*CELLS:(ti+1)*CELLS,protected]).nnz==0
                    if arm==NBNULL:
                        r=gen.build_target(t);block=r.block;diag=copy.deepcopy(r.diagnostics);diag['arm']=arm
                        np.testing.assert_array_equal(r.moments['count_ratio_native'],np.ones(len(gen.genes)))
                        np.testing.assert_array_equal(r.moments['conditional_mean_editable'],denseD[:,en])
                        np.testing.assert_array_equal(r.moments['G0_sample_gene_mean'],denseD.mean(0))
                        condnoise=np.zeros_like(denseD);condnoise[:,en]=denseD[:,en]+denseD[:,en]**2/gen.theta[eo][None,:]
                        condzero=(denseD==0).astype(np.float64);condzero[:,en]=np.exp(-gen.theta[eo][None,:]*np.log1p(denseD[:,en]/gen.theta[eo][None,:]))
                    else:
                        block=D.copy();condnoise=np.zeros_like(denseD);condzero=(denseD==0).astype(np.float64)
                        diag=dict(target=t,arm=arm,cells=CELLS,editable_genes=len(en),protected_genes=int(protected.sum()),sampler='direct_full_native_integer_D_copy',donor_pool_rows_sha256=f['array_sha'](gen.donors[t]),count_ratio_native_sha256=f['array_sha'](np.ones(len(gen.genes))),count_block_sha256=f['csr_sha'](block))
                    assert (block[:,protected]!=D[:,protected]).nnz==0
                    library=np.asarray(block.sum(1),dtype=np.float64).ravel()
                    assert np.isfinite(block.data).all() and (block.data>0).all() and np.equal(block.data,np.rint(block.data)).all() and (block.data<=2**24).all()
                    assert (library>0).all() and (library<1e6).all()
                    sample=block.toarray().astype(np.float64);base=baseline[ti*CELLS:(ti+1)*CELLS].toarray().astype(np.float64)
                    expected=denseD.mean(0);samplemean=sample.mean(0)
                    prof=[normalized(x) for x in (denseD,base,sample)]
                    profiles.append([[x[0],x[1]] for x in prof]);libs.append([x[2] for x in prof])
                    mom.append([expected,samplemean]);vars_.append([denseD.var(0),condnoise.mean(0)+denseD.var(0),sample.var(0)]);zeros_.append([(denseD==0).mean(0),condzero.mean(0),(sample==0).mean(0)])
                    diag.update(mode=arm,mean_source='complete_total_raw_effect_exact0',all_native_count_ratio_exact1=True,protected_equals_original_B0_H0=True,
                        protected_equals_original_D=True,fullaxis_identity_exact=bool((block!=D).nnz==0),conditional_given_donor_mean_sha256=f['array_sha'](denseD),
                        conditional_expected_mean_sha256=f['array_sha'](expected),actual_sample_gene_mean_sha256=f['array_sha'](samplemean),
                        actual_gene_mean_RMS_vs_D=rms(samplemean-expected),actual_CP1M_arithmetic_mean_RMS_vs_D=rms(prof[2][0]-prof[0][0]),
                        actual_mean_log1pCP1M_RMS_vs_D=rms(prof[2][1]-prof[0][1]),sample_library_mean=float(library.mean()),original_D_library_mean=float(prof[0][2].mean()),
                        source_truth_read=False,neural_forward_calls=0,newfits=0,official_calls=0)
                    blocks.append(block);labels.extend([t]*CELLS);diags.append(diag)
                    print('NULL400_TARGET',c['role'],arm,t,flush=True)
                matrix=sparse.vstack(blocks,format='csr');out=dest/f'pred__{arm}.h5ad'
                obj=ad.AnnData(matrix,obs=pd.DataFrame({'target_gene':labels},index=[f'{arm}_{i}' for i in range(len(labels))]),var=pd.DataFrame(index=gen.genes));obj.write_h5ad(out,compression='gzip')
                moment=dest/f'{arm}_moments.npz';np.savez_compressed(moment,targets=np.asarray(gen.targets),genes=gen.genes,expected_and_sample_mean=np.asarray(mom),
                    variance_kinds=np.asarray(['D_conditional_mu_variance','total_D_mixture_variance','actual_sample_variance']),gene_variances=np.asarray(vars_),
                    zero_kinds=np.asarray(['observed_D','expected_given_D_mixture','actual_sample']),gene_zero_fractions=np.asarray(zeros_))
                profile=dest/f'{arm}_full_axis_profiles.npz';np.savez_compressed(profile,targets=np.asarray(gen.targets),genes=gen.genes,
                    sampled_profile_kinds=np.asarray(['original400_D','baselineNB_sampled','null_sampled']),profile_units=np.asarray(['percellCP1M_arithmetic_mean','percelllog1pCP1M_mean']),sampled_profiles=np.asarray(profiles),full_axis_libraries=np.asarray(libs))
                identity=copy.deepcopy(gen.source_identity);identity.update(mean_workflow='total_effect_zero_control',original_mean_bundle_provenance_only=True,total_mean_effect_zero=True,
                    count_law=arm,count_math_path=str(CODE),count_math_sha256=nr.sha(CODE),theta_changed=False,own_effect_zero=True)
                r=dict(completed=True,seed=COUNT_SEED,role=c['role'],context=c['context'],arm=arm,strict_reuse=False,
                    prediction_path=str(out),prediction_sha256=nr.sha(out),moments_path=str(moment),moments_sha256=nr.sha(moment),full_axis_profile_path=str(profile),full_axis_profile_sha256=nr.sha(profile),
                    source_identity=identity,protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),count_seed=COUNT_SEED,donor_seed=DONOR_SEED,cells=3200,cells_per_target=CELLS,targets=diags,
                    generation_runtime=runtime(),truth_read_by_generation=False,newfit=0,newjudge=0,official_calls=0)
                path=dest/f'{arm}_generation.json';clean.fresh(path,r);r.update(generation_manifest_path=str(path),generation_manifest_sha256=nr.sha(path));records.append(r)
                del blocks,matrix,obj;resources(start)
            del gen,baseline
        assert len(records)==18;validate_protocol(p);resources(start)
        clean.fresh(OUT/'PREDICTIONS_SEALED.json',dict(completed=True,protocol_sha256=nr.sha(OUT/'PROTOCOL.json'),records=records,seconds=time.monotonic()-start,
            total_outputs=18,reused_outputs=6,new_outputs=12,all18_outputs_sealed_before_new_judge=True,newjudge_at_seal=0,newfit=0,official_calls=0,resources=resources(start)))
        validate_seal(p)
        clean.fresh(OUT/'GENERATION_COMPLETE.json',dict(completed=True,prediction_seal_sha256=nr.sha(OUT/'PREDICTIONS_SEALED.json'),status='COUNTS_SEALED_WAIT_INDEPENDENT96BLOCK_GATE_AND_ROOT12JUDGE',resources=resources(start),newjudge=0))
    finally:(OUT/'GENERATING.lock').unlink()
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['draft','freeze','generate','validate']);a=ap.parse_args();os.chdir(ROOT)
    try:
        if a.phase=='draft':
            p=build_protocol();p['status']='ROOT_DRAFT_REVIEW_BEFORE_FREEZE';clean.fresh(DRAFT,p);print('DRAFT',nr.sha(DRAFT),'pending',p['pending_bindings'],flush=True)
        elif a.phase=='freeze':freeze()
        elif a.phase=='generate':generate()
        else:validate_seal(validate_protocol(nr.read(OUT/'PROTOCOL.json')));print('SEALED_VALID',flush=True)
    except Exception as e:
        target=OUT if OUT.exists() else DRAFT.parent;failure=dict(error=repr(e),traceback=traceback.format_exc(),newjudge=0,official_calls=0)
        try:failure['resources']=resources()
        except Exception as resource_error:failure['resource_check_error']=repr(resource_error)
        clean.fresh(target/f'NULL400_FAILED_{a.phase}_{time.time_ns()}.json',failure);raise
