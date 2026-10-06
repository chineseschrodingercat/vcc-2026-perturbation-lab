"""DRAFT: versioned public same-target inference labels; no phase on import.

--draft reads metadata/code only. --prepare-source requires this draft's future
root-frozen protocol and materializes public-source rows only. Never opens any
Native response bank. It leaves original F23 locks, NN inputs and models intact.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import argparse,hashlib,json,shutil,time,traceback,zipfile,sys,importlib.metadata,ctypes
import h5py,numpy as np
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path('C:/00 - Competition');HERE=Path(__file__).resolve().parent
META=ROOT/'reports/research_v28_20261005/KRR_PUBLIC_QUERY19_METADATA_AGENT_revision2.json'
NATIVE_META=ROOT/'reports/research_v28_20261005/KRR_NATIVE6_MEASUREMENT_AND_SUPPORT_METADATA_AGENT_v1.json'
AXIS=ROOT/'staging/newV20_native_source_selection/prepared/outer_rpe1__inner_jurkat/inputs.npz'
OUT=ROOT/'runs/research_v28_20261005/ntc_context_krr_native_public19_v2_winapi_repair'
DRAFT=HERE/'PUBLIC_QUERY19_READER_DRAFT_revision4.json'
META_SHA='6f97eb27e382da523399069ccbe2edb705cc194c100ae50051a094429d0c37f4'
CONTEXTS=('K562','HCT116','HEK293T','CD4_Rest','CD4_Stim8hr','CD4_Stim48hr')
MAX_SECONDS=120.;MAX_BYTES=256*1024**2
EXE=ROOT/'envs/foundation312/Scripts/python.exe'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with p.open('x',encoding='utf8') as f:f.write(text)
def arrays(p,keys):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def arrayhash(a):
    a=np.asarray(a);a=a.astype(str) if a.dtype.kind=='O' else a
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(p),SHA256=sha(p))
def stat(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
def axis():
    a=arrays(AXIS,('genes','model_exact_ENSG','common_to_model','seen'))
    assert len(a['genes'])==10916 and len(a['common_to_model'])==6157 and int(a['seen'].sum())==6114
    return a
def contract():
    assert sha(META)==META_SHA
    m=read(META);assert len(m['original_target_IDs'])==19 and m['original_instances']==48
    assert m['public_query_response_values_read']==m['Native_held_response_IO']==m['THP_IO']==0
    assert m['K562']['control_cells']==88839 and m['K562']['measured_mask_exact_corrected_bank']
    assert m['CD4']['TRAIN_donors']==['CE0008678','CE0006864'] and m['CD4']['selected_rows']==148
    return m
def dependencies(m):
    p=[META,NATIVE_META,AXIS,HERE/'audit_public_query19_metadata_revision2.py',
       ROOT/'staging/newV15_ntc_directed_source_selection/prepare_query_arrays.py',
       ROOT/'staging/newV15_ntc_directed_source_selection/prepare_model_arrays.py',
       ROOT/'staging/newV20_native_source_selection/SOURCE_ROLE_GATE.json',
       ROOT/'staging/newV20_native_source_selection/prepare_mean_arrays.py',
       ROOT/'staging/newV20_native_source_selection/train_v20.py',
       ROOT/'archive/legacy_20261003/scripts/build_orion_response_moments_v84_revision4.py',
       ROOT/'staging/newV9_CD4_source_extension/extract_train_bank.py']
    values={str(q):sha(q) for q in p};values.update(m['dependencies_SHA256'])
    values[m['K562']['cache_path']]='92371f55d397a1810421e55e4c4e60573f065683cf58bc5196a4d27238253777'
    values[m['K562']['corrected_bank_control_file']]='7e961fe2ddfaa7257d2d94a4396b1db0de5066f0ddfbac4c4af36520d1f02a62'
    for c,o in m['Orion'].items():
        for name in o['headers']:values[str(Path(o['path'])/name)]=sha(Path(o['path'])/name) # opaque bytes only
    values[m['CD4']['NTC_metadata_path']]=sha(m['CD4']['NTC_metadata_path'])
    return values
def runtime():
    import numpy._core._multiarray_umath as multi
    import numpy.linalg._umath_linalg as lin
    from ctypes import wintypes
    fn=ctypes.windll.kernel32.GetModuleFileNameW;fn.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD];fn.restype=wintypes.DWORD
    buf=ctypes.create_unicode_buffer(32768);n=fn(ctypes.pythonapi._handle,buf,len(buf));assert 0<n<len(buf)
    dll=Path(buf.value).resolve();assert dll.name.lower()=='python312.dll' and dll.is_file()
    files=[EXE,dll,Path(multi.__file__),Path(lin.__file__),Path(h5py.h5.__file__)]
    pools=threadpool_info();blas=[q for q in pools if q['user_api']=='blas'];assert blas
    files.extend(Path(q['filepath']).resolve() for q in blas)
    return dict(executable=str(EXE),Python=sys.version,versions={k:importlib.metadata.version(k) for k in ('numpy','h5py','threadpoolctl')},
        runtime_binary_bindings_SHA256={str(p.resolve()):sha(p) for p in files},actual_CPU_threads=2,
        actual_loaded_python_DLL=str(dll),actual_loaded_BLAS=[str(Path(q['filepath']).resolve()) for q in blas])
def draft():
    m=contract();a=axis()
    v=dict(schema='KRR_public19_inference_source_reader_DRAFT_revision4_winapi',status='DRAFT_NOT_EXECUTED',code=bind(__file__),
      expected_output=str(OUT),metadata_only=True,original_target_IDs=m['original_target_IDs'],original_query_instances=48,
      raw_stat_bindings=[stat(m['K562']['raw_path']),stat(m['CD4']['raw_path'])],
      source_contexts=list(CONTEXTS),bindings_SHA256=dependencies(m),
      new_scope='public same-target inference conditioning, independently versioned; not F23 NN supervised training',
      Native_response_access='NONE: no Native F4 or other heldpanel responses; allowedNative query beta stays unknown',
      Mixscale_query_response_access='NONE: canonical F23 banks lack these IDs, retained unknown',
      K_recipe='original oldB0 cache selected Float32 target means as Float64; corrected pooledNTC88839 Float64 original mask7671',
      Orion_recipe='existing authenticated V84 rawinteger target/matchedGEM Float64; HCT precision differs from older V81 Float32 means, newbranch declared',
      CD4_recipe='148 exactTrain2donor selected PB CSR rows, sums/n; same donor/culture/run original frozenNTCsum; invalid rawrows flagged/excluded per oldQC, denominator records',
      labels='literal rawlog2PC0.5 clip±3 then n/(n+50) exactly once, then Float32; no CP-library response substitution',
      source_eligibility='exact measured finite nonnegative T,C; source n>=30 and positive observed common6157 T/C library, own/seen excluded',
      unknown='T,C and response NaN with measurement and response eligibility separate; zero observed remains measured',
      common6157=a['common_to_model'].astype(int).tolist(),NB_seen=int(a['seen'].sum()),runtime=runtime(),
      prior_source_gene_domains='existing exact measured masks; no gene or NB domain expansion',
      budget=dict(CPU_threads=2,peak_RSS_GiB=2,numeric_seconds=MAX_SECONDS,opaque_hash_seconds_not_numeric_budget=True,new_output_MiB=256,min_C_free_GiB=50),
      future_actions=['root review + freeze explicit independent PROTOCOL','prepare public source labels once','seal all6 sources before Native NTC-only mean inference'],
      output_vectors=dict(context_count=6,target_count=19,model_axis=10916,raw_T_F64='19x10916 rawUMI/cell; NaN if absent/unmeasured',
        raw_matchedC_F64='same; sourceC matched as declared, never queryC',source_measured='raw observation mask separate from response_known',
        source_n='19 vector; metadata cohort/validcounts, NaN truly absent, 0 no valid CD4 rows',source_eligible_n30='19 bool with commonlibrary numeric validity',
        source_beta_F32='19x10916; NaN unknown/own/seenprotected/low-support; observed exact zero distinguished by flag',
        response_known='19x10916',own_mask='19x10916',seen='10916 original6114',low_support='19 bool',absent_target='19 bool',rawbeta_zero_measured='19x10916 bool'),
      assertions=['fixed target/gene order','all known rawmeans finite nonnegative, unknown rawmeans NaN','original Kcontrol known7671',
        'CD4 exact rawinteger+OBS library equality on selected rows','source clip/shrink applied once','own/seen not response','all6 seals and IOtrace before further actions'],
      actual_response_values_read=0,actual_fits=0,actual_mean_exports=0,counts=0,judge=0)
    write(DRAFT,v);print(json.dumps(dict(path=str(DRAFT),SHA256=sha(DRAFT),code_SHA256=sha(__file__))))

def zip_rows(path,key,rows,trace):
    rows=np.asarray(rows,np.int64);order=np.argsort(rows);selected=rows[order]
    event=dict(context='K562',path=str(path),key=key,selected_rows=rows.tolist(),status='OPEN_ATTEMPT');trace.append(event)
    with zipfile.ZipFile(path) as z,z.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f);reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=reader(f);assert len(shape)==2 and not fortran and dtype==np.dtype('float32')
        offset=f.tell();stride=shape[1]*dtype.itemsize;out=np.empty((len(rows),shape[1]),np.float64)
        for i,r in zip(order,selected):
            assert 0<=r<shape[0];f.seek(offset+int(r)*stride);b=f.read(stride);assert len(b)==stride
            out[i]=np.frombuffer(b,dtype=dtype,count=shape[1])
    event.update(status='MATERIALIZED_SELECTED_ONLY',materialized_rows=len(rows),opaque_unselected_ZIP_decompression='discarded bytes; not numeric arrays')
    return out
def gate(start):
    assert time.monotonic()-start<=MAX_SECONDS,'Fixed public19 numeric budget exceeded'
    assert shutil.disk_usage(ROOT).free>=50*2**30
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=MAX_BYTES
    # Process peak Windows working set, no psutil dependency.
    import ctypes
    class PM(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    pm=PM();pm.cb=ctypes.sizeof(pm)
    current=ctypes.windll.kernel32.GetCurrentProcess;current.argtypes=[];current.restype=ctypes.c_void_p
    memory=ctypes.windll.psapi.GetProcessMemoryInfo
    memory.argtypes=[ctypes.c_void_p,ctypes.POINTER(PM),ctypes.c_ulong];memory.restype=ctypes.c_int
    if not memory(current(),ctypes.byref(pm),pm.cb):raise OSError('Could not measure process memory with explicit Windows ABI')
    assert pm.PeakWorkingSetSize<=2*2**30
    return dict(numeric_seconds=time.monotonic()-start,peak_RSS_bytes=int(pm.PeakWorkingSetSize))
def sourceK(m,a,trace):
    k=m['K562'];rows=[r['cache_row'] for r in k['target_support']]
    z=arrays(k['cache_path'],('gene_ids',));pos={str(e):i for i,e in enumerate(z['gene_ids'])};cols=np.array([pos.get(str(e),-1) for e in a['model_exact_ENSG']]);known=cols>=0
    assert int(known.sum())==7671
    T=np.full((19,10916),np.nan);C=T.copy();T[:,known]=zip_rows(k['cache_path'],'target_mean_counts',rows,trace)[:,cols[known]]
    trace.append(dict(context='K562',path=k['corrected_bank_control_file'],selected_rows=[0],status='OPEN_ATTEMPT'))
    ctrl=np.load(k['corrected_bank_control_file'],mmap_mode='r',allow_pickle=False)[0].copy();assert np.isfinite(ctrl[known]).all() and np.isnan(ctrl[~known]).all()
    trace[-1]['status']='MATERIALIZED_SELECTED_ONLY';C[:,known]=ctrl[known]
    n=np.array([r['n_cells'] for r in k['target_support']],np.float64)
    return T,C,np.broadcast_to(known,T.shape).copy(),n,dict(recipe='same oldB0 publicK means precision/control cohort',target_precision='F32 interpretedF64')
def sourceOrion(c,m,a,trace):
    o=m['Orion'][c];rows=o['target_support'];ok=[i for i,r in enumerate(rows) if r['source_available']];I=[rows[i]['source_support_row'] for i in ok]
    mask=arrays(Path(o['path'])/'source_support.npz',('measured',))['measured'].astype(bool)
    T=np.full((19,10916),np.nan);C=T.copy();known=np.zeros(T.shape,bool)
    for filename,result in [('raw_integer_target_mean_F64.npy',T),('raw_integer_matched_control_mean_F64.npy',C)]:
        trace.append(dict(context=c,path=str(Path(o['path'])/filename),selected_rows=I,status='OPEN_ATTEMPT'))
        selected=np.load(Path(o['path'])/filename,mmap_mode='r',allow_pickle=False)[I].copy()
        assert selected.shape==(len(ok),10916) and np.isfinite(selected[:,mask]).all() and np.isnan(selected[:,~mask]).all()
        result[np.ix_(ok,np.flatnonzero(mask))]=selected[:,mask];trace[-1]['status']='MATERIALIZED_SELECTED_ONLY'
    known[ok]=mask;n=np.array([r['n_target'] if r['n_target'] is not None else np.nan for r in rows],np.float64)
    return T,C,known,n,dict(recipe=o['target_C_rule'],unit=o['unit'],precision='F64 exact integer mean; not original HCT V81 F32 bitclaim')
def sourcesCD4(m,a,trace,start):
    cd=m['CD4'];raw=Path(cd['raw_path']);assert raw.stat().st_size==cd['raw_bytes'] and raw.stat().st_mtime_ns==cd['raw_mtime_ns']
    mapping=arrays(ROOT/'staging/newV9_official_matching/CD4_F23_DONOR_LOCK_v1/model10916_exact_mapping.npz',('raw_columns','measured'))
    measured=mapping['measured'].astype(bool);cols=mapping['raw_columns'][measured];assert (cols>=0).all()
    nt=arrays(cd['NTC_metadata_path'],('stratum_keys','raw_UMI_sum','n_cells'));controls={tuple(k):nt['raw_UMI_sum'][i]/nt['n_cells'][i] for i,k in enumerate(nt['stratum_keys'].astype(str))}
    trace.append(dict(context='CD4_all3_SHARED_TRAIN2donor',path=cd['NTC_metadata_path'],keys=['raw_UMI_sum','n_cells','stratum_keys'],status='MATERIALIZED_ORIGINAL6STRATA'))
    rows=cd['target_support'];allrows=sorted({int(j) for r in rows for j in r['PB_rows']});assert len(allrows)==148
    values={};hard=[];stream=hashlib.sha256()
    with h5py.File(raw,'r',rdcc_nbytes=32*1024**2) as f:
        # Only allowed source OBS metadata and exactly selected CSR points. No other donors or target values.
        def hf(k):
            z=f['obs/'+k]
            if isinstance(z,h5py.Group):
                cat=z['categories'][:];cat=cat.astype(str) if cat.dtype.kind in 'OSU' else cat;codes=z['codes'][:]
                return np.asarray([cat[i] if i>=0 else '' for i in codes])
            return z.asstr()[:] if z.dtype.kind in 'OSU' else z[:]
        obs={k:hf(k) for k in ('donor_id','culture_condition','10xrun_id','guide_id','guide_type','_index','perturbed_gene_id','n_cells','total_counts')}
        for k,v in obs.items():assert arrayhash(v)==cd['OBS_field_SHA256'][k],k+' metadata changed'
        uids=obs['_index'].astype(str);assert hashlib.sha256(np.ascontiguousarray(uids[allrows]).tobytes()).hexdigest()==cd['selected_UIDs_SHA256']
        x=f['X'];ptr=x['indptr'][:];assert tuple(x.attrs['shape'])==(278684,18129)
        trace.append(dict(context='CD4_all3_SHARED_TRAIN2donor',path=str(raw),selected_rows=allrows,status='OPEN_SELECTED_POINT_ROWS'))
        for j in allrows:
            assert str(obs['donor_id'][j]) in cd['TRAIN_donors'] and str(obs['guide_type'][j])=='targeting' and str(obs['perturbed_gene_id'][j]) in m['original_target_IDs']
            lo,hi=int(ptr[j]),int(ptr[j+1]);v=x['data'][lo:hi];ind=x['indices'][lo:hi]
            stream.update(np.array([j,lo,hi],np.int64).tobytes());stream.update(ind.tobytes());stream.update(v.tobytes())
            valid=np.isfinite(v).all() and (v>=0).all() and np.equal(v,np.rint(v)).all() and (ind>=0).all() and (ind<18129).all() and float(v.sum())==float(obs['total_counts'][j])
            if not valid:hard.append(dict(original_row=j,reason='nonfinite/negative/noninteger/column/library exact-contract failure'));continue
            dense=np.zeros(18129);np.add.at(dense,ind,v);values[j]=dense;gate(start)
        trace[-1].update(status='MATERIALIZED_SELECTED_ONLY',rows_valid=len(values),selected_stream_SHA256=stream.hexdigest())
    results={}
    for culture in ('Rest','Stim8hr','Stim48hr'):
        c='CD4_'+culture;T=np.full((19,10916),np.nan);C=T.copy();known=np.zeros(T.shape,bool);n=np.zeros(19)
        ledger=[]
        for i,e in enumerate(m['original_target_IDs']):
            sr=next(r for r in rows if r['context']==c and r['exact_ENSG']==e);ts=np.zeros(18129);cs=ts.copy()
            for j in sr['PB_rows']:
                if j not in values:continue
                key=(str(obs['donor_id'][j]),culture,str(obs['10xrun_id'][j]));assert key in controls
                w=float(obs['n_cells'][j]);assert w>0 and w==round(w);n[i]+=w;ts+=values[j];cs+=w*controls[key]
            if n[i]>0:T[i,measured]=ts[cols]/n[i];C[i,measured]=cs[cols]/n[i];known[i]=measured
            ledger.append(dict(exact_ENSG=e,metadata_n=sr['n_cells'],actual_valid_n=float(n[i]),raw_invalid_rows=[j for j in sr['PB_rows'] if j not in values]))
        results[c]=(T,C,known,n,dict(recipe=cd['source_C_rule'],cohort=cd['TRAIN_donors'],support_ledger=ledger,raw_hard_invalid=hard,states_not_independent_donors=True))
    return results
def save_source(c,tup,m,a,protocol_sha):
    T,C,measured,n,receipt=tup;assert T.shape==C.shape==measured.shape==(19,10916)
    assert np.isnan(T[~measured]).all() and np.isnan(C[~measured]).all()
    assert np.isfinite(T[measured]).all() and np.isfinite(C[measured]).all() and (T[measured]>=0).all() and (C[measured]>=0).all()
    eligible=np.isfinite(n)&(n>=30)&(np.nansum(T[:,a['common_to_model']],1)>0)&(np.nansum(C[:,a['common_to_model']],1)>0)
    own=a['model_exact_ENSG'].astype(str)[None,:]==np.asarray(m['original_target_IDs'])[:,None]
    known=measured & eligible[:,None] & ~own & (a['seen'][None,:]>0)
    beta=np.full(T.shape,np.nan,np.float32);I=np.where(known)
    b=np.log2((T[I]+.5)/(C[I]+.5));b=np.clip(b,-3,3)*n[I[0]]/(n[I[0]]+50)
    beta[I]=b.astype(np.float32)
    p=OUT/'public_sources'/f'{c}.npz';p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
    np.savez_compressed(p,context=np.asarray(c),targets=np.asarray(m['original_target_IDs']),model_genes=a['genes'],model_exact_ENSG=a['model_exact_ENSG'],
        raw_T_F64=T,raw_matchedC_F64=C,source_measured=measured,source_n=n,source_eligible_n30=eligible,source_beta_F32=beta,response_known=known,
        own_mask=own,seen=a['seen'],low_support=np.isfinite(n)&(n<30),absent_target=np.isnan(T).all(1),rawbeta_zero_measured=known&(beta==0))
    receipt.update(context=c,path=str(p),SHA256=sha(p),PROTOCOL_SHA256=protocol_sha,eligible_targets=int(eligible.sum()),source_response_known=int(known.sum()),
        own_not_response=True,seen6114=True,source_units='rawlog2PC0.5 clip3 nshrink50 once F32; raw source means also saved')
    q=p.with_suffix('.json');write(q,receipt);return dict(context=c,path=str(p),SHA256=sha(p),receipt_path=str(q),receipt_SHA256=sha(q))
def prepare():
    assert Path(sys.executable).resolve()==EXE.resolve(),'Use pinned foundation312'
    threadpool_limits(2);m=contract();p=read(OUT/'PUBLIC_SOURCE_PROTOCOL.json');assert p['reader_code_SHA256']==sha(__file__)
    assert p['frozen'] is True and p['parent_authorized_scope']=='public-query19-source-labels-only'
    assert p['metadata_SHA256']==META_SHA and not (OUT/'PUBLIC_SOURCE_LABELS_SEALED.json').exists()
    # Verify all opaque bindings before numeric-time budget begins.
    t=time.monotonic()
    for path,h in p['bindings_SHA256'].items():assert sha(path)==h,'Source binding changed '+path
    for st in p['raw_stat_bindings']:assert stat(st['path'])==st,'Raw source stat changed'
    assert p['runtime']==runtime(),'Runtime binding changed'
    opaque=time.monotonic()-t;a=axis();trace=[];start=time.monotonic();done=[];phase='BEGIN';context=None
    try:
        phase='PUBLIC_SOURCE_K';context='K562';done.append(save_source(context,sourceK(m,a,trace),m,a,sha(OUT/'PUBLIC_SOURCE_PROTOCOL.json')));gate(start)
        for context in ('HCT116','HEK293T'):
            phase='PUBLIC_SOURCE_ORION';done.append(save_source(context,sourceOrion(context,m,a,trace),m,a,sha(OUT/'PUBLIC_SOURCE_PROTOCOL.json')));gate(start)
        phase='PUBLIC_SOURCE_CD4_POINT_ROWS';context='CD4_all3';cd=sourcesCD4(m,a,trace,start)
        for context,tup in cd.items():done.append(save_source(context,tup,m,a,sha(OUT/'PUBLIC_SOURCE_PROTOCOL.json')));gate(start)
        numeric_resource=gate(start);phase='POST_NUMERIC_OPAQUE_BINDINGS_AND_STAT_VERIFY';context=None;post=time.monotonic()
        for path,h in p['bindings_SHA256'].items():assert sha(path)==h,'Source binding changed during read '+path
        for st in p['raw_stat_bindings']:assert stat(st['path'])==st,'Raw source stat changed during read'
        assert p['runtime']==runtime(),'Runtime changed during read';post_seconds=time.monotonic()-post
        write(OUT/'PUBLIC_SOURCE_LABELS_SEALED.json',dict(schema='KRR_public19_source_labels_actual_seal_v1',records=done,all6_complete=True,
            actual_source_IO=trace,resource=numeric_resource,opaque_hash_seconds=opaque,post_opaque_hash_seconds=post_seconds,
            post_numeric_all_bindings_rawstats_runtime_verified=True,PROTOCOL_SHA256=sha(OUT/'PUBLIC_SOURCE_PROTOCOL.json'),
            no_Native_response_IO=True,THP_IO=0,NN_fits=0,Native_mean_exports=0,count_calls=0,judge_calls=0))
        print(json.dumps(dict(path=str(OUT/'PUBLIC_SOURCE_LABELS_SEALED.json'),SHA256=sha(OUT/'PUBLIC_SOURCE_LABELS_SEALED.json'))))
    except BaseException as e:
        write(OUT/f'FAILED_SOURCE_PREP_{time.time_ns()}.json',dict(phase=phase,current_context=context,error=repr(e),traceback=traceback.format_exc(),
            actual_source_IO=trace,completed_records=done,partial_preserved=True,science_parameters_not_retried=True))
        raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--draft',action='store_true');g.add_argument('--prepare-source',action='store_true');args=parser.parse_args()
    draft() if args.draft else prepare()
