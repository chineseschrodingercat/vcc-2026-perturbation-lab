"""Fixed12 exact-mask NTC-context KRR driver draft; no count/scorer code.

No phase executes on import. --draft is metadata/code only. --freeze, --predict,
and --diagnose are prepared for separate parent authorization, not run here.
Public purpose-held TRAIN truth is opened only by --diagnose after all120 seal.
A context's TRAIN T can be a legal source in another fold; no program-wide T0
claim is made. Only the original lawful HepG2 TRAIN bank is a Native source.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='2'
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,ctypes,hashlib,importlib,importlib.metadata,importlib.util,json,shutil,sys,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path('C:/00 - Competition');HERE=Path(__file__).resolve().parent
PREV=ROOT/'staging/research_v28_20261005/ntc_context_krr_collision_v1'
META=PREV/'FIXED12_RAW_SOURCE_METADATA.json'
HELPER=PREV/'context_krr_math_DRAFT.py'
LITERAL=ROOT/'staging/newV15_ntc_directed_source_selection/prepare_model_arrays.py'
GATE=ROOT/'staging/newV20_native_source_selection/SOURCE_ROLE_GATE.json'
INPUT=ROOT/'staging/newV20_native_source_selection/prepared/outer_rpe1__inner_jurkat/inputs.npz'
EXE=ROOT/'envs/foundation312/Scripts/python.exe'
OUT=ROOT/'runs/research_v28_20261005/ntc_context_krr_fixed12_revision5_reader_repair_v1'
DRAFT=HERE/'DRIVER_DRAFT_revision5.json'
GEOMETRY_AUDIT=ROOT/'reports/research_v28_20261005/KRR_FULLQUALIFIED_H5954_GEOMETRY_AGENT_v1.json'
OLD_FAILED=ROOT/'runs/research_v28_20261005/ntc_context_krr_fixed12_v1'
ROLE='outer_rpe1__inner_jurkat'
ARMS=('context_KRR','equalcomponent_uniform','corrected_K_if_lawful','zero_response','positive_RBF')
CAP_SECONDS=120.;RAM_BYTES=2*1024**3;OUTPUT_BYTES=256*1024**2
PREV_META_SHA='bf0fac3748f95db2e3e2f4e9ea9a9016445a7a89b851b04f01635c5078a1d4dd'
HELPER_SHA='610128a2e4981fb04cbedb9710e013c2e8e97e61a14ea692e38c3d4dc5d40e26'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p,x,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not replace and p.exists():raise FileExistsError('Preserve '+str(p))
    text=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        tmp=p.with_suffix(p.suffix+'.writing');tmp.write_text(text,encoding='utf8');os.replace(tmp,p)
    else:
        with p.open('x',encoding='utf8') as f:f.write(text)

def arrays(p,keys):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}

def math_module():
    assert sha(HELPER)==HELPER_SHA
    spec=importlib.util.spec_from_file_location('frozen_context_krr_math',HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def literal_prior():
    tree=ast.parse(LITERAL.read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='physical_prior')
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(LITERAL),'exec'),env)
    return env['physical_prior'],hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()

def expected_slots(m):
    return [dict(slot=f'f{fi}__{c}__t{ti:02d}',fold=fi,receiver=c,target_exact_ID=t,target_position=ti,
        status='PENDING',path=None,SHA256=None,public_truth_allowed=c!='HepG2')
        for fi,held in enumerate(m['closure_components']) for c in held for ti,t in enumerate(m['targets'])]

def loaded_python_DLL():
    """Actual loaded Python runtime, not a guessed venv/base path."""
    from ctypes import wintypes
    assert os.name=='nt','This frozen driver requires Windows runtime binding'
    fn=ctypes.windll.kernel32.GetModuleFileNameW
    fn.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD];fn.restype=wintypes.DWORD
    buf=ctypes.create_unicode_buffer(32768)
    n=fn(ctypes.pythonapi._handle,buf,len(buf));assert 0<n<len(buf)
    p=Path(buf.value).resolve();assert p.name.lower()=='python312.dll' and p.is_file()
    return p

def runtime():
    distributions={}
    for name in ('numpy','threadpoolctl'):
        d=importlib.metadata.distribution(name)
        paths=[Path(d.locate_file(f)) for f in d.files if str(f).endswith('.dist-info/METADATA')]
        assert len(paths)==1
        distributions[name]=dict(version=d.version,METADATA_path=str(paths[0]),METADATA_SHA256=sha(paths[0]))
    multi=Path(importlib.import_module('numpy._core._multiarray_umath').__file__).resolve()
    linalg=Path(importlib.import_module('numpy.linalg._umath_linalg').__file__).resolve()
    pools=threadpool_info();blas=[x for x in pools if x['user_api']=='blas']
    assert blas,'No actual loaded BLAS binary; fail closed, no assumed implementation'
    bins={'numpy_multiarray_umath':multi,'numpy_umath_linalg':linalg,'python312_DLL':loaded_python_DLL()}
    bins.update({f'BLAS::{str(Path(x["filepath"]).resolve())}':Path(x['filepath']).resolve() for x in blas})
    strong={k:dict(path=str(p),bytes=p.stat().st_size,SHA256=sha(p)) for k,p in bins.items()}
    return dict(executable=str(Path(sys.executable).resolve()),executable_SHA256=sha(sys.executable),
        python=sys.version,packages=distributions,threads=2,device='CPU',GPU=False,
        threadpools=pools,strong_binary_bindings=strong)

def metadata_contract():
    assert sha(META)==PREV_META_SHA
    m=read(META);assert m['role']==ROLE and len(m['targets'])==12 and m['target_selection_population']==292
    assert len(m['source_contexts'])==10 and len(m['closure_components'])==4 and m['geometry_axis_count']==5954
    gate=read(GATE);assert set(gate['roles'][ROLE]['allowed_response_contexts'])==set(m['source_contexts'])
    assert set(m['source_contexts'])&{'HepG2','Jurkat','RPE1'}=={'HepG2'}
    records={r['context']:r for r in m['source_records']}
    allowed={s['context_id']:s for s in gate['source_contexts']}
    bindings={str(p):sha(p) for p in (META,HELPER,LITERAL,GATE,INPUT,Path(__file__))}
    bindings[str(PREV/'PILOT_DRAFT_NOT_FIT_AUTHORIZATION.json')]=sha(PREV/'PILOT_DRAFT_NOT_FIT_AUTHORIZATION.json')
    for c,r in records.items():
        assert str(Path(r['bank_path']).resolve())==str(Path(allowed[c]['response_bank_path']).resolve())
        complete=Path(r['bank_path'])/'LABELS_COMPLETE.json';metadata=Path(r['bank_path'])/'metadata.npz'
        assert sha(complete)==r['complete_SHA256']==allowed[c]['source_complete_SHA256']
        assert sha(metadata)==r['metadata_SHA256']
        bindings[str(complete)]=r['complete_SHA256'];bindings[str(metadata)]=r['metadata_SHA256']
        for name in ('raw_target_model_F64.npy','raw_matchedNTC_model_F64.npy','fullTlib_F64.npy','fullClib_F64.npy'):
            bindings[r['raw_files_header_only'][name]['path']]=r['raw_files_header_only'][name]['expected_SHA256']
        ctl=r['global_NTC_geometry'];bindings[ctl['profile_path']]=ctl['profile_SHA256'];bindings[ctl['complete_path']]=ctl['complete_SHA256']
    for q in (GEOMETRY_AUDIT,
        ROOT/'reports/research_v28_20261005/KRR_H5954_CANONICAL_INDEX_DIAGNOSTIC_AGENT_v1.json',
        ROOT/'reports/research_v28_20261005/KRR_H5954_PROJECTION_GAP_AGENT_v1.json',
        ROOT/'staging/newV15_ntc_directed_source_selection/prepare_mixscale_and_query.py',
        ROOT/'staging/newV15_ntc_directed_source_selection/identity_bridge_v1/GENCODE_v50_symbol_exact_ID_metadata.csv',
        HERE/'run_fixed12_context_krr_revision4_DRAFT.py',HERE/'DRIVER_DRAFT_revision4.json',
        HERE/'diagnose_H5954_identity_index_agent.py',HERE/'diagnose_H5954_projection_gap_agent.py',
        HERE/'audit_fullqualified_H5954_geometry_agent.py',OLD_FAILED/'PROTOCOL.json',
        OLD_FAILED/'FAILED_PREDICT_1791206629075349200.json'):
        bindings[str(q)]=sha(q)
    audit=read(GEOMETRY_AUDIT)
    assert audit['all_source_H_complete_unique_finite'] and audit['H5954_IDs_and_order_unchanged']
    assert audit['metadata_SHA256']==sha(META)
    assert all(r['known_overlap_numeric_exact'] and r['H_count']==5954 for r in audit['records'])
    return m,records,bindings


def draft():
    m,records,bindings=metadata_contract();_,literal_AST=literal_prior()
    obj=dict(schema='NTC_context_exactmask_rawPC05_KRR_driver_draft_revision5_reader_repair',draft_only=True,actual_fit_authorized=False,
        code_path=str(Path(__file__).resolve()),code_SHA256=sha(__file__),helper_SHA256=HELPER_SHA,literal_prior_AST_SHA256=literal_AST,
        role=ROLE,targets=m['targets'],target_selection_population=292,target_order_SHA256=m['targets_SHA256'],
        context_order=m['source_contexts'],components=m['closure_components'],H5954=m['fixed_global_NTC_geometry_axis'],
        H5954_SHA256=m['geometry_axis_SHA256'],arms=list(ARMS),expected_slots=expected_slots(m),
        expected_output=str(OUT),runtime_executable=str(EXE),bindings_SHA256=bindings,
        runtime_strong_preview=runtime(),
        reader_repair=dict(previous_failed_run=str(OLD_FAILED),previous_protocol_SHA256=sha(OLD_FAILED/'PROTOCOL.json'),
            repair='H5954 was defined from qualified full-source original axis; geometry now reads exactly that axis instead of model projection',
            geometry_audit_path=str(GEOMETRY_AUDIT),geometry_audit_SHA256=sha(GEOMETRY_AUDIT),
            unchanged='H5954/order/lambda1/arms/source-label model masks/queryC model masks/decode/domain/120slots/budget',
            model_unknown_projection='METTL13 vs EEF1AKNMT at same qualified ENSG00000010165; no modelNaN fill/domain expansion',
            all_globalNTC_preflight_before_any_source_response_IO=True,
            failure_receipt='current_phase/context/fold; successful and attempted source file IO; full120 ledger retained'),
        science=dict(geometry='source-only component-balanced center/std log1p CP50000 on full-qualified H5954 rawNTC means; exact constants removed',
            bandwidth='median strictly positive source offdiagonal mean-squared standardized distance; explicit diagonal0',
            KRR='fixedlambda1 exact observed-source subGram, component total weight1, unpenalized intercept sumalpha0',
            implementation='bitmask-grouped multiRHS solve; preset first3 measured output scalar-helper checks atol2e-12',
            geometry_abstention='Only explicit source varying.none or no strictpositive source offdistance; all invalidinput/other assertions hardFAIL. Same sourceweighted math precheck, no broad assertioncatch.',
            source_label='literalV15 F64 rawPC.5 clip3 then n/(n+50) once F32; proper source-matchedC; no responsePCA',
            positive_RBF='samekernel/componentweights; exact local logmax normalization to prevent underflow, no epsilon',
            unknown='NA labels never enter Gram; no source -> policyqueryC; one component -> uniformfallback; fullaxisNaN retained',
            query='globalcontrolprofile only; no target/GEM/support argument in query_api or geometry_api; queryC and fullqualified geometry fixedacross12',
            decode='cap16 floor0 C0 guard tau0exact; finalrawPC.5 computedfromactualM',
            primary='allarms identical availablepublicTRUTH domain commonseen6114/nonown/querymeasured/truthmeasured; candidate-source-unknown fallbackincluded',
            unsupported_truth='allslots retained and metricsNA, no targetreplacement or favorable-source-mask restriction',
            aggregation='target->receiver->heldcomponent equalmacro; targetpairedbootstrap descriptiveB1000seed20261005',
            scope='HepG2 originalTRAINbank lawfulsource inotherfold; Hepquery12 unscored; noNativeheldpanel/THP IO',
            program_wide_recipient_T0_claim=False,old_V36_R1_reuse=False,
            fixed_generator='no counts here; originalNB/theta/seen/donor/countseeds unchanged for future separate authorization'),
        resource=dict(CPU_threads=2,RAM_GiB=2,output_MiB=256,predict_seconds=120,diagnose_seconds=120,
            clock='Numericprofile/source-row decoding, geometryprecheck/fits, inference and resultwrites INCLUDED. Opaque input+actualruntimebinary hashes beforephase EXCLUDED and timedseparately.',
            budget_failure='retainfull120slotledger andpartialoutputs; FAILEDreceipt, noCOMPLETEseal/partialdiagnosis; no budgetextension'),
        current_actual=dict(T_expression_decoded=0,NTC_expression_decoded=0,fit_calls=0,count_calls=0,judge_calls=0))
    write(DRAFT,obj)
    print(json.dumps(dict(draft=str(DRAFT),SHA256=sha(DRAFT),fit_calls=0)))

def check_runtime(protocol=None):
    assert Path(sys.executable).resolve()==EXE.resolve(),'Use frozen foundation312 executable'
    threadpool_limits(2)
    if protocol is not None:
        current=runtime();locked=protocol['runtime']
        assert current['executable_SHA256']==locked['executable_SHA256'],'Executable drift'
        assert current['python']==locked['python'],'Python runtime drift'
        for name in ('numpy','threadpoolctl'):
            assert current['packages'][name]==locked['packages'][name],'Runtime package/METADATA drift: '+name
        assert current['strong_binary_bindings']==locked['strong_binary_bindings'],'Actual NumPy/linalg/BLAS/PythonDLL implementation drift'

def verify_bindings(bindings):
    start=time.monotonic()
    for p,wanted in bindings.items():
        assert sha(p)==wanted,'Frozen input/code drift: '+p
    return time.monotonic()-start

def freeze():
    check_runtime();d=read(DRAFT);assert sha(__file__)==d['code_SHA256']
    assert not OUT.exists(),'Preserve existing run'
    opaque=verify_bindings(d['bindings_SHA256']);assert shutil.disk_usage(ROOT).free>=50*1024**3
    rtstart=time.monotonic();rt=runtime();rtseconds=time.monotonic()-rtstart
    assert rt['strong_binary_bindings']==d['runtime_strong_preview']['strong_binary_bindings']
    assert rt['executable_SHA256']==d['runtime_strong_preview']['executable_SHA256']
    assert rt['packages']==d['runtime_strong_preview']['packages']
    OUT.mkdir(parents=True)
    p=dict(d);p.pop('actual_fit_authorized')
    p.update(schema='NTC_context_exactmask_rawPC05_KRR_RUN_revision5_reader_repair_v1',draft_only=False,
        requested_parent_run_phase='freeze',freeze_executes_fit=False,actual_fits_so_far=0,
        execution_boundary='Freeze/predict/diagnose invoked only after parent phase instruction. Metadata draft alone is not a fit request.',
        UTC=datetime.now(timezone.utc).isoformat(),runtime=rt,opaque_verification_seconds=opaque,
        prephase_runtime_binary_verification_seconds=rtseconds,
        driver_draft_path=str(DRAFT),driver_draft_SHA256=sha(DRAFT))
    write(OUT/'PROTOCOL.json',p)
    write(OUT/'EXPECTED_FULL120_LEDGER.json',dict(PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'),slots=d['expected_slots']))
    print(json.dumps(dict(frozen=str(OUT/'PROTOCOL.json'),SHA256=sha(OUT/'PROTOCOL.json'),fits=0)))

def win_peak_RSS():
    if os.name!='nt':return None
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    pm=PMC();pm.cb=ctypes.sizeof(pm)
    ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(pm),pm.cb)
    return int(pm.PeakWorkingSetSize)

def resource_gate(start):
    elapsed=time.monotonic()-start;peak=win_peak_RSS()
    assert elapsed<=CAP_SECONDS,'Fixed120second numeric phase exceeded'
    assert peak is None or peak<=RAM_BYTES,'Fixed2GiB peakRSS exceeded'
    assert shutil.disk_usage(ROOT).free>=50*1024**3,'C free space <50GiB'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=OUTPUT_BYTES,'Fixed256MiB output exceeded'
    return dict(numeric_seconds=elapsed,peak_RSS_bytes=peak,C_free_GiB=shutil.disk_usage(ROOT).free/2**30)

def load_axis():
    a=arrays(INPUT,('genes','model_exact_ENSG','common_to_model','seen','target_ids'))
    assert len(a['genes'])==10916 and len(a['common_to_model'])==6157 and int(a['seen'].sum())==6114
    assert set(read(META)['targets'])<=set(a['target_ids'].astype(str))
    return a

def query_api(context,record,axis):
    """No target identity, target support/GEM composition or perturbation API."""
    ctl=record['global_NTC_geometry'];co=read(ctl['complete_path'])
    assert co['source_id']==context and co['partition']=='train' and co['source_perturbation_values_read']==0 and not co['targetmatched_C_averaged']
    z=arrays(ctl['profile_path'],('model_genes','model_exact_ENSG','model_measured','raw_model10916_mean_F64'))
    np.testing.assert_array_equal(z['model_exact_ENSG'],axis['model_exact_ENSG']);np.testing.assert_array_equal(z['model_genes'],axis['genes'])
    C=z['raw_model10916_mean_F64'].astype(np.float64);known=z['model_measured'].astype(bool)
    assert np.isfinite(C[known]).all() and (C[known]>=0).all() and np.isnan(C[~known]).all()
    return C,known

def geometry_api(context,record,H_IDs):
    """Source/query same NTC-only reader on frozen qualified original axis.

    Geometry is independent of model10916 measurement projection. This does
    not fill or expand the model queryC/response domain. No target argument.
    """
    ctl=record['global_NTC_geometry'];co=read(ctl['complete_path'])
    assert co['source_id']==context and co['partition']=='train' and co['source_perturbation_values_read']==0 and not co['targetmatched_C_averaged']
    assert len(H_IDs)==5954 and len(set(H_IDs))==5954
    z=arrays(ctl['profile_path'],('qualified_gene_keys','measured','raw_mean_F64','identity_basis','genes',
        'provided_exact_ENSG','model_exact_ENSG','model_measured','raw_model10916_mean_F64'))
    keys=z['qualified_gene_keys'].astype(str);positions={}
    for i,g in enumerate(keys):positions.setdefault(g,[]).append(i)
    bad=[dict(exact_ID=g,original_positions=positions.get(g,[])) for g in H_IDs if len(positions.get(g,[]))!=1]
    if bad:raise ValueError('Incomplete/ambiguous qualified H geometry: '+json.dumps(dict(context=context,genes=bad)))
    J=np.array([positions[g][0] for g in H_IDs]);measured=z['measured'][J].astype(bool)
    G=z['raw_mean_F64'][J].astype(np.float64)
    invalid=~measured | ~np.isfinite(G) | (G<0)
    if invalid.any():
        raise ValueError('Invalid qualified H geometry: '+json.dumps(dict(context=context,
            genes=[dict(exact_ID=H_IDs[i],original_position=int(J[i]),measured=bool(measured[i]),finite=bool(np.isfinite(G[i]))) for i in np.flatnonzero(invalid)])))
    if G.sum()<=0:raise ValueError('Nonpositive qualified H library: '+context)
    exact=z['model_exact_ENSG'].astype(str);lookup={g:i for i,g in enumerate(exact) if g}
    M=np.array([lookup[g] for g in H_IDs]);known=z['model_measured'][M].astype(bool)
    np.testing.assert_array_equal(G[known],z['raw_model10916_mean_F64'][M[known]])
    datahash=hashlib.sha256(np.ascontiguousarray(G).tobytes()).hexdigest()
    locked=next(r for r in read(GEOMETRY_AUDIT)['records'] if r['context']==context)
    assert datahash==locked['geometry_raw_F64_bytes_SHA256'] and float(G.sum())==locked['H_raw_mean_library']
    audit=dict(context=context,profile_path=ctl['profile_path'],profile_SHA256=ctl['profile_SHA256'],
        H5954_original_positions=J.tolist(),H5954_raw_mean_library=float(G.sum()),raw_unit='native RNA UMI per-cell rawmean',
        raw_F64_bytes_SHA256=datahash,all_H_unique_measured_finite=True,
        model_projection_overlap=int(known.sum()),known_model_overlap_numeric_exact=True,
        geometry_only_model_unmeasured_IDs=[H_IDs[i] for i in np.flatnonzero(~known)],
        model_queryC_missing_mask_preserved=True,API_has_target_argument=False,
        identity_basis_counts={str(b):int(n) for b,n in zip(*np.unique(z['identity_basis'][J].astype(str),return_counts=True))})
    return G,audit


def response_metadata(context,record,axis):
    m=arrays(Path(record['bank_path'])/'metadata.npz',('target_exact_ENSG','model_exact_ENSG','available','eligible_n30','target_n','known'))
    np.testing.assert_array_equal(m['model_exact_ENSG'],axis['model_exact_ENSG'])
    return m,{str(t):i for i,t in enumerate(m['target_exact_ENSG'])}

def selected_source(context,fold_held,record,axis,targets,physical,io_trace,fold_index):
    """Gate complete heldcomponent BEFORE each source expression file open."""
    assert context not in fold_held and context in read(META)['source_contexts']
    assert context not in {'Jurkat','RPE1','THP1'}
    m,lookup=response_metadata(context,record,axis);rows=np.array([lookup.get(t,-1) for t in targets])
    valid=rows>=0;n=np.full(len(targets),np.nan)
    for i,r in enumerate(rows):
        if r>=0:n[i]=float(m['target_n'][r]);valid[i]=bool(m['available'][r] and m['eligible_n30'][r] and n[i]>=30)
    pos=np.flatnonzero(valid);beta=np.full((len(targets),10916),np.nan,np.float32);known=np.zeros(beta.shape,bool)
    if len(pos):
        assert context not in fold_held
        event=dict(fold=fold_index,context=context,role='TRAIN_source',kind='raw_target',
            path=str(Path(record['bank_path'])/'raw_target_model_F64.npy'),requested_rows=rows[pos].tolist(),status='OPEN_ATTEMPT')
        io_trace.append(event)
        T=np.load(Path(record['bank_path'])/'raw_target_model_F64.npy',mmap_mode='r',allow_pickle=False)[rows[pos]].copy()
        event.update(status='MATERIALIZED_SELECTED_ROWS',rows_materialized=len(pos))
        assert context not in fold_held
        event=dict(fold=fold_index,context=context,role='TRAIN_source',kind='raw_matchedNTC',
            path=str(Path(record['bank_path'])/'raw_matchedNTC_model_F64.npy'),requested_rows=rows[pos].tolist(),status='OPEN_ATTEMPT')
        io_trace.append(event)
        C=np.load(Path(record['bank_path'])/'raw_matchedNTC_model_F64.npy',mmap_mode='r',allow_pickle=False)[rows[pos]].copy()
        event.update(status='MATERIALIZED_SELECTED_ROWS',rows_materialized=len(pos))
        measured=m['known'][rows[pos]].astype(bool)
        assert np.isfinite(T[measured]).all() and np.isfinite(C[measured]).all() and (T[measured]>=0).all() and (C[measured]>=0).all()
        # Same common-library availability boundary as literal V15 read_values.
        # Metadata n>=30 alone does not turn a zero observed common library into a valid label.
        numeric_valid=(np.nansum(T[:,axis['common_to_model']],axis=1)>0) & (np.nansum(C[:,axis['common_to_model']],axis=1)>0)
        valid[pos[~numeric_valid]]=False
        pos=pos[numeric_valid];T=T[numeric_valid];C=C[numeric_valid];measured=measured[numeric_valid]
        if len(pos):
            own=np.array([axis['model_exact_ENSG'].astype(str)!=targets[i] for i in pos])
            kk=measured & own & (axis['seen'][None,:]>0)
            bb=physical(T,C,n[pos],np.ones(len(pos),bool),own.astype(np.float32))
            assert bb.dtype==np.float32 and np.isfinite(bb[kk]).all()
            bb[~kk]=np.nan;beta[pos]=bb;known[pos]=kk
    return beta,known,n,valid,rows

def grouped_predictions(math,K,k,logk,beta,known,components,source_contexts,context_identifiable=True):
    """One exact mask subGram and multiRHS solve per pattern, scalar crosscheck."""
    assert beta.shape==known.shape and beta.ndim==2
    n,ng=beta.shape;out=np.zeros((len(ARMS),ng),np.float64);rknown=np.zeros(out.shape,bool)
    ncontext=np.zeros(ng,np.int16);ncomponent=np.zeros(ng,np.int16);learned=np.zeros(ng,bool)
    patterns,inverse=np.unique(np.packbits(known.T,axis=1),axis=0,return_inverse=True)
    table=[];scalar_checks=[];comp=np.asarray(components)
    for pi in range(len(patterns)):
        J=np.flatnonzero(inverse==pi);I=np.flatnonzero(known[:,J[0]])
        entry=dict(pattern_id=pi,gene_count=len(J),source_indices=I.tolist(),source_contexts=[source_contexts[i] for i in I],
            component_ids=comp[I].tolist(),weights=[],fallback='no_source_policy_queryC')
        if not len(I):table.append(entry);continue
        yy=beta[np.ix_(I,J)].astype(np.float64);assert np.isfinite(yy).all()
        cc=comp[I];w=math.component_weights(cc);mean=np.sum(w[:,None]*yy,axis=0)/w.sum()
        nc=len(set(cc.tolist()));ncontext[J]=len(I);ncomponent[J]=nc
        out[0,J]=out[1,J]=mean;rknown[0,J]=rknown[1,J]=rknown[4,J]=True
        lw=logk[I]+np.log(w);prob=np.exp(lw-lw.max());prob/=prob.sum()
        out[4,J]=prob@yy
        if 'K562' in source_contexts:
            ki=source_contexts.index('K562')
            if ki in I:out[2,J]=beta[ki,J];rknown[2,J]=True
        entry['weights']=w.tolist();entry['component_count']=nc
        if nc>=2 and context_identifiable:
            A=np.empty((len(I)+1,len(I)+1));A[:-1,:-1]=K[np.ix_(I,I)]+np.diag(1/w)
            A[:-1,-1]=1;A[-1,:-1]=1;A[-1,-1]=0
            sol=np.linalg.solve(A,np.vstack((yy,np.zeros((1,len(J))))))
            raw=k[I]@sol[:-1]+sol[-1];out[0,J]=np.clip(raw,-3,3);learned[J]=True
            entry.update(fallback='none',alpha_sum_max_abs=float(np.max(np.abs(sol[:-1].sum(0)))),
                coefficient_max_abs=float(np.max(np.abs(sol[:-1]))),tau_clip_count=int(np.count_nonzero(raw!=out[0,J])))
        else:entry['fallback']='singlecomponent_uniform_no_context_correction' if nc<2 else 'unidentifiable_NTC_geometry_uniform_abstention'
        # Preset positions, not high-error or favorable genes. Mathematical equivalence only.
        if context_identifiable:
            for g in J[:3]:
                scalar=math.context_response_prediction(K,k,beta[:,g],known[:,g],comp)
                error=abs(scalar['tau']-out[0,g]);assert error<=2e-12
                scalar_checks.append(dict(pattern_id=pi,gene_position=int(g),abs_error=float(error)))
        table.append(entry)
    assert np.isfinite(out).all()
    return out,rknown,ncontext,ncomponent,learned,inverse.astype(np.int16),table,scalar_checks

def actual_decode(math,tau,C,qknown,eligible):
    M=np.full((len(ARMS),10916),np.nan);rho=np.full_like(M,np.nan);actual=np.full_like(M,np.nan)
    J=np.flatnonzero(qknown)
    for ai in range(len(ARMS)):
        M[ai,J],rho[ai,J]=math.decode_original_raw_PC05(tau[ai,J],C[J],eligible[J])
        actual[ai,J]=np.log2((M[ai,J]+.5)/(C[J]+.5))
        np.testing.assert_array_equal(M[ai,qknown & ~eligible],C[qknown & ~eligible])
        np.testing.assert_array_equal(M[ai,qknown & (C==0)],C[qknown & (C==0)])
        assert np.isfinite(M[ai,J]).all() and (M[ai,J]>=0).all()
    np.testing.assert_array_equal(M[3,J],C[J])
    return M,rho,actual

def checked_source_geometry(math,X,components):
    """Only two identifiable absence-of-geometry conditions may abstain.

    Everything else is a wholepilot failure. Precheck duplicates the frozen
    source-only center/std/direct-offdistance math exactly, not a new model.
    No AssertionError from the actual helper is swallowed.
    """
    X=np.asarray(X,np.float64)
    if X.ndim!=2 or X.shape!=(len(components),5954) or len(X)<2:
        raise ValueError('Invalid sourceNTC geometry shape/count; wholepilot failure')
    if not np.isfinite(X).all():
        raise FloatingPointError('Invalid nonfinite sourceNTC geometry; wholepilot failure')
    w=math.component_weights(components);w/=w.sum()
    if not np.isfinite(w).all() or (w<=0).any():raise ValueError('Invalid source component weights')
    center=np.sum(w[:,None]*X,axis=0)
    sd=np.sqrt(np.sum(w[:,None]*(X-center)**2,axis=0))
    if not np.isfinite(center).all() or not np.isfinite(sd).all():
        raise FloatingPointError('Nonfinite sourceNTC center/std; wholepilot failure')
    varying=sd>0
    if not varying.any():
        return None,'ABSTAIN_NO_VARYING_SOURCE_NTC_FEATURES'
    Z=(X[:,varying]-center[varying])/sd[varying]
    if not np.isfinite(Z).all():raise FloatingPointError('Nonfinite standardized sourceNTC geometry')
    diff=Z[:,None,:]-Z[None,:,:];D=np.mean(diff*diff,axis=2);np.fill_diagonal(D,0.0)
    if not np.isfinite(D).all() or (D<0).any():raise FloatingPointError('Invalid source offdistance')
    off=D[np.triu_indices(len(Z),1)]
    if not np.any(off>0):
        return None,'ABSTAIN_NO_STRICTPOSITIVE_SOURCE_OFFDISTANCE'
    # Deliberately no catch: unforeseen helper assertions/errors fail the pilot.
    geometry=math.fit_ntc_geometry(X,components)
    return geometry,'FITTED_SOURCE_NTC_ONLY'

def predict():
    check_runtime();p=read(OUT/'PROTOCOL.json');assert not (OUT/'ALL120_PREDICTIONS_SEALED.json').exists()
    opaque=verify_bindings(p['bindings_SHA256']);m=read(META);records={r['context']:r for r in m['source_records']}
    rtstart=time.monotonic();check_runtime(p);rtseconds=time.monotonic()-rtstart
    assert p['arms']==list(ARMS) and p['code_SHA256']==sha(__file__)
    axis=load_axis();math=math_module();physical,AST=literal_prior();assert AST==p['literal_prior_AST_SHA256']
    slots=expected_slots(m);start=time.monotonic();reads=[];foldseals=[];source_io=[];NTC_reads=[]
    current_phase='BEGIN';current_context=None;current_fold=None
    write(OUT/'PREDICTION_LEDGER.json',dict(status='RUNNING',slots=slots))
    try:
        exact=axis['model_exact_ENSG'].astype(str);lookup={g:i for i,g in enumerate(exact) if g}
        H=np.array([lookup[g] for g in p['H5954']]);assert len(H)==5954
        component_of={c:i for i,cs in enumerate(p['components']) for c in cs}
        # All fixed global NTC panels validated on both declared axes BEFORE T.
        global_controls={};global_geometry={};geometry_receipts={}
        for c in p['context_order']:
            current_context=c;current_phase='GLOBAL_NTC_MODEL_QUERY_PREFLIGHT'
            global_controls[c]=query_api(c,records[c],axis)
            current_phase='GLOBAL_NTC_FULLQUALIFIED_GEOMETRY_PREFLIGHT'
            global_geometry[c],geometry_receipts[c]=geometry_api(c,records[c],p['H5954'])
            NTC_reads.append(dict(context=c,phase='COMPLETE_BOTH_NTC_AXES',source_response_values_read=0))
            resource_gate(start)
        current_context=None;current_phase='ALL10_GLOBAL_NTC_PREFLIGHT_COMPLETE'
        for fi,held in enumerate(p['components']):
            current_fold=fi
            src=[c for c in p['context_order'] if c not in held];components=[component_of[c] for c in src]
            X=[np.log1p(global_geometry[c]*50000/global_geometry[c].sum()) for c in src]
            current_phase='SOURCE_NTC_ONLY_GEOMETRY_FIT';current_context=None
            geometry,geometry_status=checked_source_geometry(math,np.stack(X),components)
            resource_gate(start)
            betas=[];knowns=[];support=[];available=[];positions=[]
            for c in src:
                current_context=c;current_phase='TRAIN_SOURCE_RESPONSE_SELECTED_ROWS'
                b,k,n,a,r=selected_source(c,held,records[c],axis,p['targets'],physical,source_io,fi)
                betas.append(b);knowns.append(k);support.append(n);available.append(a);positions.append(r)
                reads.append(dict(fold=fi,role='TRAIN_source',context=c,targets=12,
                    metadata_requested_expression_rows=sum(t['eligible_n30'] for t in records[c]['support']),
                    numerically_eligible_response_rows=int(a.sum()),held_component=list(held)))
                resource_gate(start)
            B=np.stack(betas);known=np.stack(knowns)
            current_context=None;current_phase='SOURCE_MODEL_SEAL'
            foldpath=OUT/f'fold{fi}_SOURCE_MODEL.npz'
            geo={} if geometry is None else {f'geometry_{k}':v for k,v in geometry.items()}
            np.savez_compressed(foldpath,source_contexts=np.asarray(src),components=np.asarray(components),
                source_beta_F32=B,source_known=known,source_n=np.stack(support),source_available=np.stack(available),
                source_row_positions=np.stack(positions),H_model_positions=H,
                H_qualified_raw_NTC_F64=np.stack([global_geometry[c] for c in src]),**geo)
            fseal=OUT/f'fold{fi}_MODEL_SEALED.json'
            write(fseal,dict(fold=fi,held_component=held,geometry_status=geometry_status,source_contexts=src,
                path=str(foldpath),SHA256=sha(foldpath),helper_path=str(HELPER),helper_SHA256=HELPER_SHA,
                actual_source_reads=[r for r in reads if r['fold']==fi],actual_source_file_IO=[r for r in source_io if r['fold']==fi],
                source_fullqualified_NTC_geometry=[geometry_receipts[c] for c in src],query_values_used_for_fit=False,
                PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json')))
            foldseals.append(dict(path=str(fseal),SHA256=sha(fseal)))
            for dst in held:
                current_context=dst;current_phase='NTC_ONLY_RECEIVER_INFERENCE'
                C,qknown=global_controls[dst] # preserved model-domain query; no target parameter
                G=global_geometry[dst] # same fullqualified geometry API as sources
                query_identifiable=True # all10 complete NTC preflight has passed, no proxy fill
                qdiag=dict(query_identifiable=query_identifiable,geometry_status=geometry_status,
                    fullqualified_input=geometry_receipts[dst])
                if geometry is not None and query_identifiable:
                    Q=np.log1p(G*50000/G.sum());k,qmetric=math.query_kernel(geometry,Q);qdiag.update(qmetric)
                    v=geometry['varying'];z=(Q[v]-geometry['center'][v])/geometry['sd'][v]
                    logk=-np.mean((geometry['source_Z']-z)**2,axis=1)/geometry['bandwidth'];K=geometry['K']
                else:K=np.eye(len(src));k=np.zeros(len(src));logk=np.zeros(len(src))
                for ti,tid in enumerate(p['targets']):
                    tau,rknown,nc,ng,learned,pattern,table,checks=grouped_predictions(math,K,k,logk,B[:,ti],known[:,ti],components,src,
                        context_identifiable=geometry is not None and query_identifiable)
                    if geometry is None or not query_identifiable:tau[0]=tau[1];tau[4]=tau[1];learned[:]=False
                    own=exact==tid;protected=axis['seen']<=0;eligible=qknown & ~own & ~protected
                    commonmask=np.zeros(10916,bool);commonmask[axis['common_to_model']]=True
                    protected43=commonmask & protected;assert int(protected43.sum())==43
                    policy_control=np.array([a=='zero_response' for a in ARMS])
                    # All arms preserve full axis and identical deployment eligibility. Unknown response -> policy0, flagged.
                    tau[:,~eligible]=0.;M,rho,actual=actual_decode(math,tau,C,qknown,eligible)
                    requested=tau.copy();requested[:,~eligible]=np.nan
                    path=OUT/'predictions'/f'f{fi}__{dst}__t{ti:02d}.npz';path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path,arms=np.asarray(ARMS),target_exact_ID=np.asarray(tid),receiver=np.asarray(dst),
                        model_genes=axis['genes'],model_exact_ENSG=exact,query_raw_C_F64=C,query_measured=qknown,
                        final_raw_mean_F64=M,requested_tau_rawPC05=requested,final_actual_rawPC05=actual,ratio=rho,
                        response_source_known=rknown,response_fallback_unknown=(~rknown & ~policy_control[:,None]),fixed_eligibility=eligible,
                        declared_zero_response_control=policy_control,own=own,protected_seen=protected,protected_common43=protected43,
                        outside_common_H0=~commonmask,source_context_count=nc,source_component_count=ng,
                        learned_context_correction_source_estimate=learned,active_context_correction=learned & eligible,
                        source_pattern_id=pattern)
                    rp=path.with_suffix('.receipt.json')
                    finite_lib=np.nansum(M,axis=1);q_lib=float(np.nansum(C))
                    write(rp,dict(slot=path.stem,PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'),NPZ_SHA256=sha(path),
                        fold=fi,receiver=dst,target_exact_ID=tid,query_API_has_target_argument=False,
                        query_control_path=records[dst]['global_NTC_geometry']['profile_path'],query_control_SHA256=records[dst]['global_NTC_geometry']['profile_SHA256'],
                        source_model_seal_path=str(fseal),source_model_seal_SHA256=sha(fseal),query_geometry=qdiag,
                        source_mask_patterns=table,scalar_equivalence_checks=checks,all10916_retained=True,
                        query_unmeasured_count=int((~qknown).sum()),query_measured_zero_count=int((qknown & (C==0)).sum()),
                        own_count=int(own.sum()),protected_seen_fullaxis_count=int(protected.sum()),protected_common43_count=43,
                        outside_common_H0_count=int((~commonmask).sum()),fixed_eligibility_count=int(eligible.sum()),
                        zero_response_is_declared_policy_control_not_observed_zero_response=True,
                        source_unknown_eligible_by_arm=[int(np.count_nonzero(eligible & ~rk)) for rk in rknown],
                        actual_cap16_by_arm=[int(np.count_nonzero(eligible & (rr==16))) for rr in rho],
                        actual_zero_floor_by_arm=[int(np.count_nonzero(eligible & (C>0) & (mm==0))) for mm in M],
                        observed_model_axis_library_by_arm=finite_lib.tolist(),query_observed_model_axis_library=q_lib,
                        library_change_by_arm=(finite_lib-q_lib).tolist(),official_metric_claim=False,
                        purpose_held_response_values_read_for_this_slot=0,program_wide_receiver_T0_claim=False))
                    record=next(s for s in slots if s['slot']==path.stem)
                    record.update(status='PREDICTED',path=str(path),SHA256=sha(path),receipt_path=str(rp),receipt_SHA256=sha(rp))
                    resource_gate(start)
                write(OUT/'PREDICTION_LEDGER.json',dict(status='RUNNING',slots=slots),replace=True)
                print(json.dumps(dict(fold=fi,receiver=dst,completed=sum(s['status']=='PREDICTED' for s in slots),numeric_seconds=time.monotonic()-start)),flush=True)
        assert len(slots)==120 and all(s['status']=='PREDICTED' for s in slots)
        res=resource_gate(start)
        write(OUT/'ALL120_PREDICTIONS_SEALED.json',dict(status='COMPLETE_ALL120_MEANS_ONLY',PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'),
            slots=slots,fold_model_seals=foldseals,legal_source_expression_reads=reads,actual_source_file_IO=source_io,
            all10_global_NTC_preflight=NTC_reads,fullqualified_geometry_inputs=geometry_receipts,resources=res,
            prephase_opaque_verification_seconds=opaque,prephase_runtime_binary_verification_seconds=rtseconds,
            Native_held_panel_response_reads=0,THP_numeric_IO=0,
            purpose_held_public_TRAIN_diagnostic_reads=0,program_wide_recipient_T0_claim=False,count_calls=0,judge_calls=0))
        print(json.dumps(dict(sealed=str(OUT/'ALL120_PREDICTIONS_SEALED.json'),SHA256=sha(OUT/'ALL120_PREDICTIONS_SEALED.json'))),flush=True)
    except Exception as exc:
        write(OUT/f'FAILED_PREDICT_{time.time_ns()}.json',dict(error=str(exc),traceback=traceback.format_exc(),
            numeric_seconds=time.monotonic()-start,prephase_opaque_verification_seconds=opaque,
            prephase_runtime_binary_verification_seconds=rtseconds,
            current_phase=current_phase,current_source_or_receiver_context=current_context,current_fold=current_fold,
            actual_source_reads=reads,source_expression_file_IO=source_io,completed_NTC_preflight=NTC_reads,
            complete_NTC_preflight_before_T=not source_io or len(NTC_reads)==10,
            expected_all120_slots=slots,partial_outputs_preserved=True,no_partial_population_scoring=True))
        raise

def observed_public_truth(context,record,axis,targets):
    assert context!='HepG2' and context not in {'Jurkat','RPE1','THP1'}
    m,lookup=response_metadata(context,record,axis);T=np.full((12,10916),np.nan);C=T.copy();known=np.zeros(T.shape,bool)
    valid=np.zeros(12,bool);n=np.full(12,np.nan);rows=np.array([lookup.get(t,-1) for t in targets])
    for i,r in enumerate(rows):
        if r>=0:n[i]=float(m['target_n'][r]);valid[i]=bool(m['available'][r] and m['eligible_n30'][r] and n[i]>=30)
    J=np.flatnonzero(valid)
    if len(J):
        T[J]=np.load(Path(record['bank_path'])/'raw_target_model_F64.npy',mmap_mode='r',allow_pickle=False)[rows[J]]
        C[J]=np.load(Path(record['bank_path'])/'raw_matchedNTC_model_F64.npy',mmap_mode='r',allow_pickle=False)[rows[J]]
        known[J]=m['known'][rows[J]]
        assert np.isfinite(T[known]).all() and np.isfinite(C[known]).all()
    return T,C,known,valid,n

def macro(rows,arm,weights=None):
    by={}
    for r in rows:
        if r['arm']!=arm or r['primary_rawPC05_MAE'] is None:continue
        w=1. if weights is None else weights.get(r['target_exact_ID'],0.)
        if w:by.setdefault((r['fold'],r['receiver']),[]).append((r['primary_rawPC05_MAE'],w))
    component={}
    for (fold,receiver),pairs in by.items():component.setdefault(fold,[]).append(float(np.average([v for v,w in pairs],weights=[w for v,w in pairs])))
    cm={str(k):float(np.mean(v)) for k,v in component.items()}
    return (float(np.mean(list(cm.values()))) if cm else None),cm

def diagnose():
    check_runtime();p=read(OUT/'PROTOCOL.json');sealpath=OUT/'ALL120_PREDICTIONS_SEALED.json';s=read(sealpath)
    assert s['status']=='COMPLETE_ALL120_MEANS_ONLY' and s['PROTOCOL_SHA256']==sha(OUT/'PROTOCOL.json')
    assert len(s['slots'])==120 and all(r['status']=='PREDICTED' for r in s['slots'])
    opaque=verify_bindings(p['bindings_SHA256'])
    rtstart=time.monotonic();check_runtime(p);rtseconds=time.monotonic()-rtstart
    for r in s['slots']:
        assert sha(r['path'])==r['SHA256'] and sha(r['receipt_path'])==r['receipt_SHA256']
    for r in s['fold_model_seals']:
        assert sha(r['path'])==r['SHA256']
        folded=read(r['path']);assert sha(folded['path'])==folded['SHA256']
    assert not (OUT/'TRAIN_DIAGNOSTIC_COMPLETE.json').exists()
    start=time.monotonic();axis=load_axis();m=read(META);records={r['context']:r for r in m['source_records']};rows=[];reads=[]
    try:
        truth={}
        for c in p['context_order']:
            if c!='HepG2':truth[c]=observed_public_truth(c,records[c],axis,p['targets']);reads.append(c);resource_gate(start)
        for rec in s['slots']:
            c=rec['receiver'];ti=rec['target_position'];z=arrays(rec['path'],('arms','query_raw_C_F64','query_measured','final_raw_mean_F64',
                'fixed_eligibility','response_source_known','own','protected_seen','final_actual_rawPC05'))
            assert list(z['arms'].astype(str))==list(ARMS);Cq=z['query_raw_C_F64'];M=z['final_raw_mean_F64']
            if c=='HepG2':T=None;km=None;valid=False;n=None
            else:T,Cmatch,km,v,ns=truth[c];valid=bool(v[ti]);n=float(ns[ti]) if np.isfinite(ns[ti]) else None
            mask=np.zeros(10916,bool) if not valid else z['fixed_eligibility'] & km[ti]
            # Exact same eligible observed domain for all five arms; unknown source fallbacks INCLUDED.
            if mask.any():
                y=np.log2((T[ti,mask]+.5)/(Cq[mask]+.5));assert np.isfinite(y).all()
                assert np.isfinite(M[:,mask]).all();nonzero=y!=0
            cm=axis['common_to_model'];profile_MAE=[None]*len(ARMS)
            exact_common_profile_available=bool(valid and z['query_measured'][cm].all() and km[ti,cm].all())
            if exact_common_profile_available and (M[:,cm].sum(1)>0).all() and T[ti,cm].sum()>0 and mask.any():
                pp=np.log1p(M[:,cm]*50000/M[:,cm].sum(1)[:,None]);tt=np.log1p(T[ti,cm]*50000/T[ti,cm].sum())
                assert np.isfinite(pp).all() and np.isfinite(tt).all()
                profile_MAE=np.mean(np.abs(pp[:,mask[cm]]-tt[mask[cm]]),axis=1).tolist()
            for ai,arm in enumerate(ARMS):
                row=dict(slot=rec['slot'],fold=rec['fold'],receiver=c,target_exact_ID=rec['target_exact_ID'],arm=arm,
                    public_TRAIN_truth_allowed=c!='HepG2',truth_support_n=n,truth_available=valid,
                    unsupported_truth_retained=not valid,primary_gene_count=int(mask.sum()),
                    primary_response_source_unknown_count=None if arm=='zero_response' else int(np.count_nonzero(mask & ~z['response_source_known'][ai])),
                    primary_rawPC05_MAE=None,primary_rawPC05_MSE=None,direction_accuracy_nonzero_truth=None,
                    cosine=None,primary_is_official_metric=False,
                    secondary_exact_common6157_bulk_logCP50K_profile_available=exact_common_profile_available,
                    secondary_common6157_bulk_logCP50K_profile_MAE=profile_MAE[ai])
                if mask.any():
                    pred=z['final_actual_rawPC05'][ai,mask];err=pred-y
                    row.update(primary_rawPC05_MAE=float(np.mean(np.abs(err))),primary_rawPC05_MSE=float(np.mean(err*err)),
                        direction_accuracy_nonzero_truth=float(np.mean(np.sign(pred[nonzero])==np.sign(y[nonzero]))) if nonzero.any() else None,
                        cosine=float(pred@y/np.sqrt((pred@pred)*(y@y))) if (pred@pred)>0 and (y@y)>0 else None,
                        primary_truth_zero_count=int((~nonzero).sum()))
                    matched=np.log2((T[ti,mask]+.5)/(Cmatch[ti,mask]+.5))
                    row['secondary_source_matchedC_rawPC05_MAE']=float(np.mean(np.abs(pred-matched)))
                rows.append(row)
            resource_gate(start)
        assert len(rows)==600
        scalars=OUT/'TRAIN_DIAGNOSTIC_ALL600_ROWS.json';write(scalars,rows)
        totals={a:dict(zip(('primary_macro_rawPC05_MAE','held_component_macro'),macro(rows,a))) for a in ARMS}
        rng=np.random.default_rng(20261005);CIs={}
        for base in ('equalcomponent_uniform','zero_response','positive_RBF'):
            values=[]
            for _ in range(1000):
                draw=rng.integers(0,12,size=12);w={t:int(np.count_nonzero(draw==i)) for i,t in enumerate(p['targets'])}
                ca,_=macro(rows,'context_KRR',w);ba,_=macro(rows,base,w)
                if ca is not None and ba is not None:values.append(ca-ba)
            CIs[base]=dict(delta=totals['context_KRR']['primary_macro_rawPC05_MAE']-totals[base]['primary_macro_rawPC05_MAE'],
                target_paired_descriptive_CI95=np.quantile(values,[.025,.975]).tolist(),bootstrap_n=1000,
                independent_background_replication_claim=False)
        result=dict(status='COMPLETE_FIXED12_PUBLIC_TRAIN_DIAGNOSTIC_ONLY',PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'),
            PREDICTION_SEAL_SHA256=sha(sealpath),rows_path=str(scalars),rows_SHA256=sha(scalars),rows=600,
            prediction_slots=120,public_TRAIN_slots=108,Hep_unscored_slots=12,all12_targets_retained=True,
            observed_public_TRAIN_truth_contexts_read=reads,Native_held_panel_response_IO=0,THP_numeric_IO=0,
            source_Hep_TRAIN_used_in_other_folds=True,program_wide_recipient_T0_claim=False,
            same_primary_gene_domain_for_allarms=True,source_unknown_policy_H0_included_in_primary=True,
            totals=totals,paired_target_descriptive=CIs,
            gate_pass_numerical_only=all(CIs[b]['delta']<0 for b in ('equalcomponent_uniform','zero_response')),
            gate='Bothmeancontrols must improve with coherent componentcoverage; numerical gate alone not count/overall promotion.',
            generator_changed=False,count_calls=0,judge_calls=0,resources=resource_gate(start),
            prephase_opaque_verification_seconds=opaque,prephase_runtime_binary_verification_seconds=rtseconds,
            unit='actual decoded rawPC.5 vsobservedT, SAMEglobalqueryC',
            prior_targets_previously_viewed_TRAIN=True,final_test_not_touched=True)
        write(OUT/'TRAIN_DIAGNOSTIC_COMPLETE.json',result)
        print(json.dumps(dict(totals=totals,paired=CIs,complete=str(OUT/'TRAIN_DIAGNOSTIC_COMPLETE.json'))),flush=True)
    except Exception as exc:
        write(OUT/f'FAILED_DIAGNOSE_{time.time_ns()}.json',dict(error=str(exc),traceback=traceback.format_exc(),
            partial_rows_preserved=rows,numeric_seconds=time.monotonic()-start))
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('draft','freeze','predict','diagnose'))
    phase=ap.parse_args().phase
    {'draft':draft,'freeze':freeze,'predict':predict,'diagnose':diagnose}[phase]()

if __name__=='__main__':main()
