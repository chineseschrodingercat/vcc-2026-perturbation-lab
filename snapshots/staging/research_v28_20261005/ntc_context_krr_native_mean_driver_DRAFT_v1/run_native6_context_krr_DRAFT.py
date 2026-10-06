"""Native six-role public19 mean-only driver. Draft phase performs no numeric IO.

Actual freeze/predict are separate parent-reviewed phases. No scorer, counts,
destination perturbation files, Native F4 response banks or THP numeric IO.
The three scientific functions are compiled unchanged from the pinned rev5 AST.
"""
from __future__ import annotations
import argparse
import ast
import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path('C:/00 - Competition')
HERE = Path(__file__).resolve().parent
OUT = ROOT/'runs/research_v28_20261005/ntc_context_krr_native_mean_v1'
EXE = ROOT/'envs/foundation312/Scripts/python.exe'
OLD = ROOT/'staging/research_v28_20261005/ntc_context_krr_fixed12_driver_DRAFT_v1/run_fixed12_context_krr_revision5_DRAFT.py'
OLD_RUN = ROOT/'runs/research_v28_20261005/ntc_context_krr_fixed12_revision5_reader_repair_v1'
OLD_PROTO = OLD_RUN/'PROTOCOL.json'
OLD_SEAL = OLD_RUN/'ALL120_PREDICTIONS_SEALED.json'
OLD_GATE = ROOT/'reports/research_v28_20261005/KRR_REVISION5_PREDICTION_INDEPENDENT_AUDIT_AGENT.json'
MATH = ROOT/'staging/research_v28_20261005/ntc_context_krr_collision_v1/context_krr_math_DRAFT.py'
META = ROOT/'staging/research_v28_20261005/ntc_context_krr_collision_v1/FIXED12_RAW_SOURCE_METADATA.json'
GEOMETRY_AUDIT = ROOT/'reports/research_v28_20261005/KRR_FULLQUALIFIED_H5954_GEOMETRY_AGENT_v1.json'
GATE = ROOT/'staging/newV20_native_source_selection/SOURCE_ROLE_GATE.json'
CLEAN = ROOT/'runs/research_v28_20261005/clean400_v1/PROTOCOL.json'
PUBLIC = ROOT/'runs/research_v28_20261005/ntc_context_krr_native_public19_v2_winapi_repair'
PUBLIC_PROTO = PUBLIC/'PUBLIC_SOURCE_PROTOCOL.json'
PUBLIC_SEAL = PUBLIC/'PUBLIC_SOURCE_LABELS_SEALED.json'
PUBLIC_AUDIT = PUBLIC/'SOURCE_INDEPENDENT_AUDIT.json'
PUBLIC_META = ROOT/'reports/research_v28_20261005/KRR_PUBLIC_QUERY19_METADATA_AGENT_revision2.json'
PUBLIC_CODE = ROOT/'staging/research_v28_20261005/ntc_context_krr_native_screen_DRAFT_v1/prepare_public_query19_sources_revision4_DRAFT.py'
LITERAL = ROOT/'staging/newV15_ntc_directed_source_selection/prepare_model_arrays.py'
COMPONENT_SOURCE = ROOT/'staging/research_v28_20261005/clean_direction_experiment.py'
ROOT_DECISION = ROOT/'reports/research_v28_20261005/CONTEXT_KRR400_ROOT_SCREEN_DECISION_CN.md'
INPUTS = ROOT/'staging/newV20_native_source_selection/prepared'
PROFILE_ROOT = ROOT/'staging/newV15_ntc_directed_source_selection/ntc_fullrawmeans_v1'

# The untouched rev5 functions index these five internal mathematical arms.
INTERNAL_ARMS = ('context_KRR', 'equalcomponent_uniform', 'corrected_K_if_lawful', 'zero_response', 'positive_RBF')
ARMS = ('context_KRR', 'equalcomponent_uniform', 'positive_RBF')
ARM_INDEX = (0, 1, 4)
PUBLIC_CONTEXTS = ('K562', 'HCT116', 'HEK293T', 'CD4_Rest', 'CD4_Stim8hr', 'CD4_Stim48hr')
NATIVE_NAMES = {'jurkat': 'Jurkat', 'rpe1': 'RPE1', 'hepg2': 'HepG2'}
SCIENCE = ('grouped_predictions', 'checked_source_geometry', 'actual_decode')
MAX_SECONDS = 120
MAX_BYTES = 256*1024**2
MAX_RSS = 2*1024**3


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def arrays(path, names):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in names}


def arrayhash(a):
    a = np.asarray(a)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def canonical_ast(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return node, hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def scientific_functions():
    """Compile exact pinned AST bodies with the original five-arm indexing."""
    ns = {'np': np, 'ARMS': INTERNAL_ARMS}
    hashes = {}
    for name in SCIENCE:
        node, hashes[name] = canonical_ast(OLD, name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(OLD), 'exec'), ns)
    return {name: ns[name] for name in SCIENCE}, hashes


def math_module():
    spec = importlib.util.spec_from_file_location('pinned_context_krr_math', MATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LAMBDA == 1.0
    return module


def loaded_python_DLL():
    from ctypes import wintypes
    assert os.name == 'nt'
    fn = ctypes.windll.kernel32.GetModuleFileNameW
    fn.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    fn.restype = wintypes.DWORD
    buf = ctypes.create_unicode_buffer(32768)
    n = fn(ctypes.pythonapi._handle, buf, len(buf))
    assert 0 < n < len(buf)
    p = Path(buf.value).resolve()
    assert p.name.lower() == 'python312.dll' and p.is_file()
    return p


def runtime():
    threadpool_limits(2)
    pools = threadpool_info()
    blas = [x for x in pools if x['user_api'] == 'blas']
    assert blas, 'Actual loaded BLAS must be bound'
    bins = {'python312_DLL': loaded_python_DLL(),
            'numpy_multiarray_umath': Path(importlib.import_module('numpy._core._multiarray_umath').__file__).resolve(),
            'numpy_umath_linalg': Path(importlib.import_module('numpy.linalg._umath_linalg').__file__).resolve()}
    for x in blas:
        p = Path(x['filepath']).resolve()
        bins['BLAS::'+str(p)] = p
    files = [EXE, MATH, OLD, Path(importlib.import_module('threadpoolctl').__file__).resolve()]
    files += list(bins.values())
    site = ROOT/'envs/foundation312/Lib/site-packages'
    for area in ('numpy/_core', 'numpy/linalg', 'scipy/sparse'):
        found = sorted((site/area).rglob('*.pyd'))
        assert found, area
        files += found
    distributions = {}
    for name in ('numpy', 'scipy', 'threadpoolctl'):
        d = importlib.metadata.distribution(name)
        distributions[name] = d.version
        for filename in ('METADATA', 'RECORD'):
            pp = [Path(d.locate_file(f)) for f in d.files if str(f).endswith('.dist-info/'+filename)]
            assert len(pp) == 1, (name, filename)
            files += pp
    deps = {str(p.resolve()): sha(p) for p in files}
    return dict(executable=str(Path(sys.executable).resolve()), python=sys.version,
                versions=distributions, CPU_threads=2, GPU=False,
                actual_loaded_binaries={k: dict(path=str(v), SHA256=sha(v)) for k, v in bins.items()},
                dependencies_SHA256=deps, threadpools=pools)


def memory_receipt():
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    value = PMC()
    value.cb = ctypes.sizeof(value)
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    assert fn(current(), ctypes.byref(value), value.cb)
    return dict(RSS_bytes=int(value.WorkingSetSize), peak_RSS_bytes=int(value.PeakWorkingSetSize))


def resources(start=None):
    r = memory_receipt()
    r.update(C_free_bytes=shutil.disk_usage(ROOT).free,
             new_output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0)
    if start is not None:
        r['numeric_seconds'] = time.monotonic()-start
        assert r['numeric_seconds'] <= MAX_SECONDS, 'Fixed numeric time budget exceeded'
    assert r['RSS_bytes'] < MAX_RSS and r['peak_RSS_bytes'] < MAX_RSS, 'Fixed RSS budget exceeded'
    assert r['C_free_bytes'] >= 50*1024**3, 'C free space below 50GiB'
    assert r['new_output_bytes'] <= MAX_BYTES, 'Fixed output budget exceeded'
    return r


def verify(deps):
    for path, value in deps.items():
        assert sha(path) == value, 'Dependency changed: '+path


def component_id(context):
    """Exact existing conservative provenance closure, including RPE+K merge."""
    tree = ast.parse(COMPONENT_SOURCE.read_text(encoding='utf-8-sig'))
    mapping = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for lhs in n.targets:
                if isinstance(lhs, ast.Name) and lhs.id in ('FAMILY', 'COMPONENT'):
                    mapping[lhs.id] = ast.literal_eval(n.value)
    family = mapping['FAMILY'][context]
    label = mapping['COMPONENT'][family]
    return {'Marson_CD4': 0, 'Orion': 1, 'Replogle_Mixscale_closure': 2, 'Nadig': 3}[label]


def axis_metadata(role):
    # These are axes and structural masks only; no labels or expression arrays.
    p = INPUTS/role/'inputs.npz'
    a = arrays(p, ('genes', 'model_exact_ENSG', 'common_to_model', 'seen'))
    assert len(a['genes']) == len(a['model_exact_ENSG']) == len(a['seen']) == 10916
    assert len(a['common_to_model']) == 6157 and int(a['seen'].sum()) == 6114
    assert np.isin(a['seen'], [0, 1]).all()
    return a


def source_profile_record(context, old_records):
    if context in old_records:
        ctl = dict(old_records[context]['global_NTC_geometry'])
    else:
        assert context in ('Jurkat', 'RPE1')
        d = PROFILE_ROOT/context
        ctl = dict(context=context, profile_path=str(d/'profile.npz'),
                   profile_SHA256=sha(d/'profile.npz'), complete_path=str(d/'COMPLETE.json'),
                   complete_SHA256=sha(d/'COMPLETE.json'))
    co = read(ctl['complete_path'])
    assert co['source_id'] == context and co['partition'] == 'train'
    assert co['source_perturbation_values_read'] == 0 and not co['targetmatched_C_averaged']
    assert sha(ctl['profile_path']) == ctl['profile_SHA256']
    assert sha(ctl['complete_path']) == ctl['complete_SHA256']
    return ctl


def metadata_contract():
    gate = read(GATE)
    clean = read(CLEAN)
    meta = read(META)
    qmeta = read(PUBLIC_META)
    old_protocol = read(OLD_PROTO)
    old_prediction_seal = read(OLD_SEAL)
    old_gate = read(OLD_GATE)
    assert old_protocol['code_SHA256'] == sha(OLD) and old_protocol['helper_SHA256'] == sha(MATH)
    assert old_prediction_seal['PROTOCOL_SHA256'] == sha(OLD_PROTO)
    assert old_gate['status'] == 'PASS' and old_gate['mean_vectors_replayed'] == 600
    assert old_gate['PROTOCOL_SHA256'] == sha(OLD_PROTO)
    assert old_gate['ALL120_PREDICTIONS_SEAL_SHA256'] == sha(OLD_SEAL)
    assert old_protocol['H5954'] == meta['fixed_global_NTC_geometry_axis']
    assert len(meta['fixed_global_NTC_geometry_axis']) == 5954
    assert len(set(meta['fixed_global_NTC_geometry_axis'])) == 5954
    assert len(qmeta['original_target_IDs']) == 19 and qmeta['original_instances'] == 48
    records = {r['context']: r for r in meta['source_records']}
    cases = []
    profiles = {}
    deps = {str(p): sha(p) for p in (Path(__file__).resolve(), OLD, MATH, META, GEOMETRY_AUDIT, GATE, CLEAN,
                                    PUBLIC_META, PUBLIC_CODE, LITERAL, COMPONENT_SOURCE,
                                    OLD_PROTO, OLD_SEAL, OLD_GATE, ROOT_DECISION)}
    slots = []
    reference_axis = None
    for case in clean['cases']:
        role = case['role']
        declared = gate['roles'][role]
        assert case['targets'] == declared['original_targets']
        assert case['allowed_response_contexts'] == declared['allowed_response_contexts']
        source_order = list(declared['allowed_response_contexts'])
        assert len(source_order) == 10 and len(set(source_order)) == 10
        assert set(PUBLIC_CONTEXTS) <= set(source_order)
        assert {NATIVE_NAMES[x.lower()] for x in declared['held_families'] if x.lower() in NATIVE_NAMES}.isdisjoint(source_order)
        assert set(source_order) & {'HepG2', 'Jurkat', 'RPE1'} == {NATIVE_NAMES[case['allowed_native']]}
        a = axis_metadata(role)
        if reference_axis is None:
            reference_axis = a
        else:
            for k in a:
                np.testing.assert_array_equal(a[k], reference_axis[k])
        deps[str(INPUTS/role/'inputs.npz')] = sha(INPUTS/role/'inputs.npz')
        leaves = case['leaves']
        # Bind B0 and count assets without decoding expression/theta/counts.
        for key in ('B0_effects', 'B0_complete', 'B0_checkpoint', 'genes.txt', 'generation_pool.npz',
                    'theta_seen', 'generator_complete', 'role_generator_complete', 'role_encoder', 'panel.json'):
            info = leaves[key]
            assert sha(info['path']) == info['sha256'], key
            deps[info['path']] = info['sha256']
        bm = arrays(leaves['B0_effects']['path'], ('genes', 'exact_ENSG', 'native_genes', 'targets',
                      'target_exact_ENSG', 'model_to_native', 'editable_mask', 'ownmask', 'new_generator_seen',
                      'effect_units', 'held_response_used', 'historical_fallback_used', 'source_contexts'))
        np.testing.assert_array_equal(bm['genes'], a['genes'])
        np.testing.assert_array_equal(bm['exact_ENSG'], a['model_exact_ENSG'])
        assert bm['targets'].astype(str).tolist() == case['targets']
        assert not bool(bm['held_response_used']) and not bool(bm['historical_fallback_used'])
        assert str(bm['effect_units']) == 'raw_log2_pseudocount_0.5'
        assert bm['editable_mask'].shape == (8, 10916)
        assert set(bm['source_contexts'].astype(str)) <= set(source_order)
        assert set(bm['target_exact_ENSG'].astype(str)) <= set(qmeta['original_target_IDs'])
        symbols = qmeta['query_ID_symbols']
        for symbol, tid in zip(case['targets'], bm['target_exact_ENSG'].astype(str)):
            assert symbols[tid] == symbol, 'Exact target mapping conflict'
        # NTC-only query uses original panel axis. Public gene identity metadata
        # validates model_to_native; destination profile numeric means are not read.
        recipient = NATIVE_NAMES[case['context']]
        identity_path = PROFILE_ROOT/recipient/'profile.npz'
        deps[str(identity_path)] = sha(identity_path)
        identity = arrays(identity_path, ('genes', 'provided_exact_ENSG', 'qualified_gene_keys', 'model_genes', 'model_exact_ENSG'))
        np.testing.assert_array_equal(identity['model_genes'], a['genes'])
        np.testing.assert_array_equal(identity['model_exact_ENSG'], a['model_exact_ENSG'])
        native = np.loadtxt(leaves['genes.txt']['path'], dtype=str, ndmin=1)
        np.testing.assert_array_equal(native, bm['native_genes'])
        pg = identity['genes'].astype(str)
        assert len(native) == len(set(native))
        symbol_positions = {}
        for pi, symbol in enumerate(pg):
            symbol_positions.setdefault(symbol, []).append(pi)
        # The fixed original Native panel already excludes both ambiguous
        # HSPA14 publisher columns. Preserve this exact panel; do not require
        # unrepresented original-profile symbols to be unique or add them.
        assert all(len(symbol_positions.get(g, [])) == 1 for g in native), 'Ambiguous/missing represented query gene'
        pp = {g: symbol_positions[g][0] for g in native}
        qualified = identity['qualified_gene_keys'][[pp[g] for g in native]].astype(str)
        positions = {}
        for ni, value in enumerate(qualified):
            positions.setdefault(value, []).append(ni)
        Hpositions = []
        for h in meta['fixed_global_NTC_geometry_axis']:
            assert len(positions.get(h, [])) == 1, 'Query H5954 missing/ambiguous '+h
            Hpositions.append(positions[h][0])
        mapping = bm['model_to_native'].astype(np.int64)
        assert mapping.shape == (10916,) and ((mapping >= -1) & (mapping < len(native))).all()
        mapped = mapping >= 0
        np.testing.assert_array_equal(a['genes'][mapped], native[mapping[mapped]])
        exact = a['model_exact_ENSG'].astype(str)
        query_known = mapped & (exact != '')
        np.testing.assert_array_equal(qualified[mapping[query_known]], exact[query_known])
        Hlookup = {g: i for i, g in enumerate(exact) if g}
        assert len(Hlookup) == int(np.count_nonzero(exact != ''))
        Hmodel = np.array([Hlookup[h] for h in meta['fixed_global_NTC_geometry_axis']], dtype=np.int64)
        np.testing.assert_array_equal(mapping[Hmodel], Hpositions)
        assert query_known[Hmodel].all()
        for context in source_order:
            if context not in profiles:
                profiles[context] = source_profile_record(context, records)
            ctl = profiles[context]
            deps[ctl['profile_path']] = ctl['profile_SHA256']
            deps[ctl['complete_path']] = ctl['complete_SHA256']
        cases.append(dict(role=role, context=case['context'], original_targets=list(case['targets']),
                          target_exact_ENSG=bm['target_exact_ENSG'].astype(str).tolist(),
                          allowed_response_contexts=source_order, source_components=[component_id(c) for c in source_order],
                          allowed_native=case['allowed_native'], held_families=case['held_families'],
                          base_case=case, axes_path=str(INPUTS/role/'inputs.npz'),
                          query_identity_metadata_path=str(identity_path),
                          H5954_native_positions=Hpositions, H5954_model_positions=Hmodel.tolist(),
                          query_model_to_native=mapping.tolist(), query_model_measured=query_known.tolist(),
                          metadata_only_query_measured_count=int(query_known.sum()),
                          inherited_effect_editable_count=int(np.count_nonzero(bm['editable_mask'])),
                          original_role_generator_seen_count=int(bm['new_generator_seen'].sum())))
        for ti, (symbol, tid) in enumerate(zip(case['targets'], bm['target_exact_ENSG'].astype(str))):
            for arm in ARMS:
                slots.append(dict(role=role, context=case['context'], target_ordinal=ti,
                                  target_symbol=symbol, target_exact_ENSG=tid, arm=arm))
    assert len(cases) == 6 and len(slots) == 144
    pending = []
    source_bind = None
    if PUBLIC_PROTO.is_file() and PUBLIC_SEAL.is_file():
        sp = read(PUBLIC_PROTO)
        ss = read(PUBLIC_SEAL)
        assert ss['all6_complete'] is True and ss['no_Native_response_IO'] is True and ss['THP_IO'] == 0
        assert ss['PROTOCOL_SHA256'] == sha(PUBLIC_PROTO)
        assert len(ss['records']) == 6
        source_bind = dict(protocol_path=str(PUBLIC_PROTO), protocol_SHA256=sha(PUBLIC_PROTO),
                           seal_path=str(PUBLIC_SEAL), seal_SHA256=sha(PUBLIC_SEAL), records=ss['records'])
        deps[str(PUBLIC_PROTO)] = sha(PUBLIC_PROTO)
        deps[str(PUBLIC_SEAL)] = sha(PUBLIC_SEAL)
        for r in ss['records']:
            context = r['context']
            assert context in PUBLIC_CONTEXTS
            assert sha(r['path']) == r['SHA256']
            assert sha(r['receipt_path']) == r['receipt_SHA256']
            deps[r['path']] = r['SHA256']
            deps[r['receipt_path']] = r['receipt_SHA256']
        assert {r['context'] for r in ss['records']} == set(PUBLIC_CONTEXTS)
        if PUBLIC_AUDIT.is_file():
            sa = read(PUBLIC_AUDIT)
            assert sa['status'] == 'PASS'
            assert sa['all6_sources_full19_targets'] is True
            assert sa['raw_T_source_control_rawPC_flags_rebuilt'] is True
            assert sa['all_input_output_oldprotected_start_end_exact'] is True
            assert sa['sourcebank_PROTOCOL_SHA256'] == sha(PUBLIC_PROTO)
            assert sa['sourcebank_ALL6_SEAL_SHA256'] == sha(PUBLIC_SEAL)
            for key in ('Native_response_IO', 'THP_IO', 'fits', 'count_calls', 'judge_calls'):
                assert sa[key] == 0
            assert sa['source_records'] == ss['records']
            source_bind.update(independent_audit_path=str(PUBLIC_AUDIT), independent_audit_SHA256=sha(PUBLIC_AUDIT))
            deps[str(PUBLIC_AUDIT)] = sha(PUBLIC_AUDIT)
        else:
            pending.append('Public19 independent full-source gate not yet present; freeze/predict prohibited')
    else:
        pending.append('Public19 protocol and all-six source-label seal not yet present; freeze/predict prohibited')
    return dict(cases=cases, expected_slots=slots, source_profile_records=profiles,
                public_source_binding=source_bind, bindings_SHA256=deps, pending=pending,
                original_target_IDs=qmeta['original_target_IDs'], H5954=meta['fixed_global_NTC_geometry_axis'],
                H5954_SHA256=meta['geometry_axis_SHA256'])


def draft(path):
    assert Path(sys.executable).resolve() == EXE.resolve(), 'Use pinned foundation312'
    c = metadata_contract()
    asts = {name: canonical_ast(OLD, name)[1] for name in SCIENCE}
    rt = runtime()
    c['bindings_SHA256'].update(rt['dependencies_SHA256'])
    obj = dict(schema='Native6_public19_same-source_KRR_mean_driver_DRAFT_v1', draft_only=True,
               actual_fit_authorized=False, code_path=str(Path(__file__).resolve()), code_SHA256=sha(__file__),
               expected_output=str(OUT), **c,
               arms=list(ARMS), internal_arms=list(INTERNAL_ARMS), formal_arm_indices=list(ARM_INDEX),
               pinned_function_AST_SHA256=asts, runtime_preview=rt,
               input_contract=dict(public_response='6 sealed public banks; exact19 IDs; beta F32 after one literal rawPC.5 clip3*n/(n+50)',
                   public_numeric_keys_to_read=['targets', 'model_genes', 'model_exact_ENSG', 'source_beta_F32',
                       'source_measured', 'response_known', 'source_n', 'source_eligible_n30', 'own_mask', 'seen',
                       'low_support', 'absent_target', 'rawbeta_zero_measured'],
                   public_T_C_value_decoding_by_this_driver=0,
                   query19_Native_and_Mixscale='All response rows unknown; no old F4 response bank access or same-target proxy',
                   source_geometry='Original role-allowed ten TRAIN globalNTC H5954 qualified original-axis rawmean profiles only',
                   destination_query='Original3000 generation_pool sparse integer NTC and pure gene-axis mapping only; target-free query API',
                   query_means='precise float64 rawsum/3000 for geometry and diagnostic decoder; old sparse dtype mean separately saved for count-ABI explanation',
                   count_ABI='effect10916 F32 only; all other B0 bundle keys/arrays exact. Original future NB rho uses original SparseNTC.mean→F64; no nominal mean lock'),
               science=dict(H=5954, lambda_value=1.0, standardization='Source-only component-balanced center/std; exact constants removed',
                   kernel='RBF exp(-mean squared standardized distance / median strictly positive source offdistance)',
                   components='Exact existing clean_direction_experiment FAMILY/COMPONENT provenance closure: CD4 three / Orion HCT+HEK / conservative K562+Mixscale+lawfulRPE closure / lawfulHepG2-or-Jurkat Nadig. Hep/Jur geometry has4 components, RPE geometry3. Not independent papers.',
                   model='Exact rev5 masked per-gene observed source subGram with component weights and unpenalized intercept; tau clipped±3 only as original',
                   positive_RBF='Original same source geometry and component weights; exp(logkernel+logweight−localmax), no epsilon',
                   unknown='No source support -> tau0 policy, explicitly unknown; one eligible component -> same uniform fallback',
                   eligible='Exact source support n>=30 and finite measured response; recipient only measured axis, own/seen/B0editable protection',
                   export='Three formal arms select internal indices0,1,4; all144slots retained before any count or judge',
                   controls='Internal zero/correctedK saved scalar/vector diagnostics only, no extra count budget',
                   comparison='Uniform vs KRR/positiveRBF isolates context weighting on the same expanded source. Versus B0 also changes source scope and mean model, cannot attribute solely to KRR.',
                   scope='Native three tuned development backgrounds; two roles/background are duplicate populations with distinct legal geometry pools, not six biological repeats or strict publisher-independent test',
                   source_only_preprocessing=True, destination_perturbation_reads=0, THP_numeric_IO=0, fit_budget_sweep=False),
               export_contract=dict(MEANS_SEALED='records18 role×arm, completedTrue; every record bundle_path/bundle_sha256/complete_path/complete_SHA256/source_model_seal_path/source_model_seal_SHA256',
                   bundle='Exactly originalB0 keys; only effect replaced. source_contexts remains original legal10 as ABI metadata; actual response public6/other4unknown in source model seal.',
                   effect_units='raw_log2_pseudocount_0.5', mean_only=True, counts=0, scorer_imports=False,
                   source_model_seal='3 unique lawful geometry/source models; source banks and kernel training axes/weights/learned-source provenance saved',
                   slot_arrays='Full requestedtau64, exportedtau32, sourceknown, supportcounts, kernels, nominal decoder M/rho and actualrawPC.5 from nominalM; no deployment normalizedmean claim',
                   independent_gate_filename='MEAN_INDEPENDENT_AUDIT.json'),
               resource=dict(CPU_threads=2, RSS_GiB=2, output_MiB=256, numeric_seconds=MAX_SECONDS, C_free_GiB=50,
                   opaque_hash_and_runtime_verification='before and after numeric phase, separately timed; no payload expression decoding in draft',
                   failure='Retain partial full144 ledger and FAILED receipt, no sealed partial candidate; no budget extension'),
               current_actual=dict(source_response_values_decoded=0, source_NTC_values_decoded=0,
                   query_NTC_values_decoded=0, fit_calls=0, mean_exports=0, count_calls=0, judge_calls=0))
    write_new(path, obj)
    print(json.dumps(dict(draft_path=str(Path(path).resolve()), draft_SHA256=sha(path), code_SHA256=sha(__file__),
                          pending=obj['pending'], fits=0, numeric_expression_reads=0)))


def freeze(draft_path):
    assert Path(sys.executable).resolve() == EXE.resolve()
    d = read(draft_path)
    assert d['code_SHA256'] == sha(__file__) and not d['pending']
    assert not OUT.exists(), 'Preserve existing output'
    verify(d['bindings_SHA256'])
    assert d['runtime_preview'] == runtime()
    resources()
    OUT.mkdir(parents=True)
    p = dict(d)
    p.update(schema='Native6_public19_same-source_KRR_mean_PROTOCOL_v1', draft_only=False,
             frozen=True, parent_requested_phase='freeze-only', parent_authorized_scope='Native6-public19-mean-only',
             freeze_executes_numeric_IO=False, created_utc=datetime.now(timezone.utc).isoformat(),
             draft_path=str(Path(draft_path).resolve()), draft_SHA256=sha(draft_path))
    write_new(OUT/'PROTOCOL.json', p)
    write_new(OUT/'EXPECTED_FULL144_LEDGER.json', dict(PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'), slots=p['expected_slots']))
    print(json.dumps(dict(PROTOCOL_path=str(OUT/'PROTOCOL.json'), PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'), fits=0)))


def source_ntc_geometry_api(context, record, H):
    """Source globalNTC only on frozen full qualified H; no target argument."""
    co = read(record['complete_path'])
    assert co['source_id'] == context and co['partition'] == 'train'
    assert co['source_perturbation_values_read'] == 0 and not co['targetmatched_C_averaged']
    z = arrays(record['profile_path'], ('qualified_gene_keys', 'measured', 'raw_mean_F64',
                'model_exact_ENSG', 'model_measured', 'raw_model10916_mean_F64'))
    positions = {}
    for i, key in enumerate(z['qualified_gene_keys'].astype(str)):
        positions.setdefault(key, []).append(i)
    assert all(len(positions.get(h, [])) == 1 for h in H), 'Incomplete/ambiguous source H geometry'
    J = np.array([positions[h][0] for h in H], np.int64)
    G = z['raw_mean_F64'][J].astype(np.float64)
    assert z['measured'][J].astype(bool).all() and np.isfinite(G).all() and (G >= 0).all() and G.sum() > 0
    exact = z['model_exact_ENSG'].astype(str)
    lookup = {h: i for i, h in enumerate(exact) if h}
    model_pos = np.array([lookup[h] for h in H], np.int64)
    overlap = z['model_measured'][model_pos].astype(bool)
    np.testing.assert_array_equal(G[overlap], z['raw_model10916_mean_F64'][model_pos[overlap]])
    locked = {x['context']: x for x in read(GEOMETRY_AUDIT)['records']}
    if context in locked:
        assert arrayhash(G) == locked[context]['geometry_raw_F64_bytes_SHA256']
        assert float(G.sum()) == locked[context]['H_raw_mean_library']
    return G, dict(context=context, profile_path=record['profile_path'], profile_SHA256=record['profile_SHA256'],
                   raw_H_F64_bytes_SHA256=arrayhash(G), H_library=float(G.sum()), original_axis_positions=J.tolist(),
                   geometry_original_fullqualified=True, model_overlap=int(overlap.sum()),
                   model_unmeasured_H_preserved=int(np.count_nonzero(~overlap)), source_T_values_decoded=0)


def query_ntc_api(pool_path, native_genes_path, model_to_native, model_measured, H_native_positions):
    """Only recipient NTC + axis mappings; no target/support/perturbation API."""
    genes = np.loadtxt(native_genes_path, dtype=str, ndmin=1)
    pool = sparse.load_npz(pool_path).tocsr()
    assert pool.shape == (3000, len(genes))
    assert np.isfinite(pool.data).all() and (pool.data >= 0).all()
    assert np.equal(pool.data, np.rint(pool.data)).all(), 'Query NTC counts not integer'
    precise = np.asarray(pool.astype(np.float64).sum(0)).ravel()/3000.0
    # This separate diagnostic preserves the original count decoder precision.
    legacy_sparse_mean = np.asarray(pool.mean(0)).ravel().astype(np.float64)
    mapping = np.asarray(model_to_native, np.int64)
    known = np.asarray(model_measured, bool)
    C = np.full(10916, np.nan, np.float64)
    C[known] = precise[mapping[known]]
    C_legacy = np.full(10916, np.nan, np.float64)
    C_legacy[known] = legacy_sparse_mean[mapping[known]]
    G = precise[np.asarray(H_native_positions, np.int64)]
    assert len(G) == 5954 and np.isfinite(G).all() and (G >= 0).all() and G.sum() > 0
    assert np.isfinite(C[known]).all() and np.isnan(C[~known]).all()
    return C, known, G, C_legacy, dict(pool_path=str(pool_path), pool_dtype=str(pool.dtype), cells=3000,
               pool_shape=list(pool.shape), precise_native_mean_SHA256=arrayhash(precise),
               model_mean_F64_SHA256=arrayhash(C), geometry_H_F64_SHA256=arrayhash(G),
               query_known_SHA256=arrayhash(known), original_count_ABI_mean_SHA256=arrayhash(C_legacy),
               precise_minus_legacy_max_abs=float(np.max(np.abs(C[known]-C_legacy[known]))),
               target_argument=False, perturbation_metadata_used=False)


def source_response_api(context, record, targets, axis):
    """Only newly sealed lawful public source beta; no Native/F4 label decoder."""
    assert context in PUBLIC_CONTEXTS, 'Forbidden query19 response source'
    z = arrays(record['path'], ('targets', 'model_genes', 'model_exact_ENSG', 'source_beta_F32',
               'source_measured', 'response_known', 'source_n', 'source_eligible_n30', 'own_mask',
               'seen', 'low_support', 'absent_target', 'rawbeta_zero_measured'))
    np.testing.assert_array_equal(z['targets'].astype(str), np.asarray(targets))
    np.testing.assert_array_equal(z['model_genes'], axis['genes'])
    np.testing.assert_array_equal(z['model_exact_ENSG'], axis['model_exact_ENSG'])
    np.testing.assert_array_equal(z['seen'], axis['seen'])
    B = z['source_beta_F32']
    known = z['response_known'].astype(bool)
    measured = z['source_measured'].astype(bool)
    assert B.shape == known.shape == measured.shape == (19, 10916) and B.dtype == np.float32
    assert np.isfinite(B[known]).all() and np.isnan(B[~known]).all()
    n = z['source_n'].astype(np.float64)
    eligible = z['source_eligible_n30'].astype(bool)
    assert n.shape == eligible.shape == (19,)
    assert np.isfinite(n[eligible]).all() and (n[eligible] >= 30).all()
    own = np.asarray(axis['model_exact_ENSG'].astype(str)[None, :] == np.asarray(targets)[:, None])
    np.testing.assert_array_equal(z['own_mask'].astype(bool), own)
    expected = measured & ~own & (axis['seen'][None, :] > 0) & eligible[:, None]
    np.testing.assert_array_equal(known, expected)
    assert not known[z['absent_target'].astype(bool)].any() and not known[z['low_support'].astype(bool)].any()
    np.testing.assert_array_equal(z['rawbeta_zero_measured'].astype(bool), known & (B == 0))
    assert np.all(np.abs(B[known]) <= np.repeat((3*n/(n+50))[:, None], 10916, axis=1)[known]+2e-7)
    return B, known, n, eligible, dict(context=context, bank_path=record['path'], bank_SHA256=record['SHA256'],
               receipt_path=record['receipt_path'], receipt_SHA256=record['receipt_SHA256'],
               beta_F32_SHA256=arrayhash(B), known_SHA256=arrayhash(known), n_SHA256=arrayhash(n),
               literal_shrink_already_applied=True, shrink_reapplied=False, original_T_C_payload_read=False)


def predict():
    assert Path(sys.executable).resolve() == EXE.resolve()
    p = read(OUT/'PROTOCOL.json')
    assert p['frozen'] is True and p['parent_authorized_scope'] == 'Native6-public19-mean-only'
    assert p['code_SHA256'] == sha(__file__) and not p['pending']
    assert not (OUT/'MEANS_SEALED.json').exists(), 'Preserve existing complete run'
    assert not (OUT/'STARTED.json').exists(), 'No implicit partial retry'
    pre = time.monotonic()
    verify(p['bindings_SHA256'])
    assert p['runtime_preview'] == runtime()
    opaque_seconds = time.monotonic()-pre
    scientific, asts = scientific_functions()
    assert asts == p['pinned_function_AST_SHA256']
    math = math_module()
    source_bank = {r['context']: r for r in p['public_source_binding']['records']}
    write_new(OUT/'STARTED.json', dict(PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'),
               started_utc=datetime.now(timezone.utc).isoformat(), parent_requested_phase='predict',
               no_implicit_retry=True, held_response_reads=0, count_calls=0, judge_calls=0))
    start = time.monotonic()
    phase = 'SOURCE_NTC_PREFLIGHT'
    current_role = None
    trace = []
    model_records = []
    means = []
    completed_slots = []
    try:
        # All source NTC profiles are validated before public beta decoding.
        raw_geometry = {}
        geometry_receipts = {}
        for context, record in p['source_profile_records'].items():
            assert context != 'THP1'
            raw_geometry[context], geometry_receipts[context] = source_ntc_geometry_api(context, record, p['H5954'])
            trace.append(dict(kind='source_global_NTC', context=context, path=record['profile_path'], source_T_read=False))
            resources(start)
        phase = 'PUBLIC_BETA_READ'
        axis = axis_metadata(p['cases'][0]['role'])
        public_arrays = {}
        for context in PUBLIC_CONTEXTS:
            public_arrays[context] = source_response_api(context, source_bank[context], p['original_target_IDs'], axis)
            trace.append(dict(kind='lawful_public_query19_beta', context=context, path=source_bank[context]['path'],
                              original_raw_T_C_payload_decoded=False, Native_response=False))
            resources(start)
        phase = 'SOURCE_GEOMETRY_MODELS'
        models = {}
        for case in p['cases']:
            key = tuple(case['allowed_response_contexts'])
            if key in models:
                continue
            src = list(key)
            cc = np.array(case['source_components'], np.int64)
            X = np.stack([np.log1p(raw_geometry[c]*50000/raw_geometry[c].sum()) for c in src])
            geometry, status = scientific['checked_source_geometry'](math, X, cc)
            B = np.full((10, 19, 10916), np.nan, np.float32)
            known = np.zeros(B.shape, bool)
            n = np.full((10, 19), np.nan, np.float64)
            eligible = np.zeros((10, 19), bool)
            for ci, context in enumerate(src):
                if context in public_arrays:
                    B[ci], known[ci], n[ci], eligible[ci] = public_arrays[context][:4]
                else:
                    assert context in ('HepG2', 'Jurkat', 'RPE1', 'HT29_IFNB', 'HT29_INS', 'K562_IFNB')
            assert np.isnan(B[~known]).all()
            mi = len(models)
            model_path = OUT/'source_models'/f'lawful_pool_{mi}.npz'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            common = dict(source_contexts=np.asarray(src), components=cc, targets=np.asarray(p['original_target_IDs']),
                model_genes=axis['genes'], model_exact_ENSG=axis['model_exact_ENSG'],
                H_exact_ENSG=np.asarray(p['H5954']), source_logCP50K=X,
                source_beta_F32=B, source_response_known=known, source_n=n, source_eligible_n30=eligible,
                geometry_status=np.asarray(status))
            if geometry is not None:
                common.update(center=geometry['center'], sd=geometry['sd'], varying=geometry['varying'],
                    source_Z=geometry['source_Z'], bandwidth=np.asarray(geometry['bandwidth']), K=geometry['K'])
            else:
                common.update(center=np.zeros(5954), sd=np.zeros(5954), varying=np.zeros(5954, bool),
                    source_Z=np.zeros((10, 0)), bandwidth=np.asarray(np.nan), K=np.eye(10))
            np.savez_compressed(model_path, **common)
            receipt = dict(model_index=mi, model_path=str(model_path), model_SHA256=sha(model_path),
                source_context_order=src, component_ids=cc.tolist(), geometry_status=status,
                role_members=[x['role'] for x in p['cases'] if tuple(x['allowed_response_contexts']) == key],
                actual_public_response_contexts=[c for c in src if c in PUBLIC_CONTEXTS],
                response_unknown_contexts=[c for c in src if c not in PUBLIC_CONTEXTS],
                source_NTC_receipts=[geometry_receipts[c] for c in src],
                public_response_receipts=[public_arrays[c][4] for c in src if c in PUBLIC_CONTEXTS],
                alpha=1, label_units='raw_log2_pseudocount_0.5 clip3 then n/(n+50) once F32',
                geometry_training_source_only=True, destination_T_read=False, Native_query19_T_read=False,
                partial_or_zero_filled_unknown_labels=False, source_NN_fit=False)
            models[key] = dict(geometry=geometry, B=B, known=known, n=n, eligible=eligible, components=cc,
                               source_contexts=src, receipt=receipt)
            model_records.append(receipt)
            resources(start)
        assert len(models) == 3
        model_seal = OUT/'SOURCE_MODELS_SEALED.json'
        write_new(model_seal, dict(schema='Native6_public19_KRR_source_models_seal_v1', completed=True,
                   PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'), model_records=model_records,
                   public_source_protocol_path=str(PUBLIC_PROTO), public_source_protocol_SHA256=sha(PUBLIC_PROTO),
                   public_source_seal_path=str(PUBLIC_SEAL), public_source_seal_SHA256=sha(PUBLIC_SEAL),
                   public_source_independent_audit_path=str(PUBLIC_AUDIT), public_source_independent_audit_SHA256=sha(PUBLIC_AUDIT),
                   original_function_AST_SHA256=asts, fit_calls=3, source_NN_fit_calls=0,
                   held_response_reads=0, Native_query19_response_reads=0, THP_numeric_IO=0,
                   no_final_metric_or_truth_used=True, count_calls=0, judge_calls=0))
        phase = 'NTC_ONLY_QUERY_AND_EXPORT'
        target_lookup = {t: i for i, t in enumerate(p['original_target_IDs'])}
        for case in p['cases']:
            current_role = case['role']
            resources(start)
            model = models[tuple(case['allowed_response_contexts'])]
            axis = axis_metadata(current_role)
            base = case['base_case']
            leaves = base['leaves']
            C, qknown, G, C_legacy, qreceipt = query_ntc_api(leaves['generation_pool.npz']['path'], leaves['genes.txt']['path'],
                  case['query_model_to_native'], case['query_model_measured'], case['H5954_native_positions'])
            trace.append(dict(kind='recipient_original3000NTC', role=current_role, path=leaves['generation_pool.npz']['path'],
                              target_argument=False, perturbation_metadata=False))
            geometry = model['geometry']
            if geometry is not None:
                Q = np.log1p(G*50000/G.sum())
                k, qdiag = math.query_kernel(geometry, Q)
                v = geometry['varying']
                z = (Q[v]-geometry['center'][v])/geometry['sd'][v]
                logk = -np.mean((geometry['source_Z']-z)**2, axis=1)/geometry['bandwidth']
                K = geometry['K']
            else:
                Q = np.log1p(G*50000/G.sum())
                K = np.eye(10)
                k = np.zeros(10)
                logk = np.zeros(10)
                qdiag = dict(geometry_status=model['receipt']['geometry_status'])
            with np.load(leaves['B0_effects']['path'], allow_pickle=False) as z:
                original = {n: np.asarray(z[n]).copy() for n in z.files}
            all_effects = np.zeros((3, 8, 10916), np.float32)
            slot_files = []
            for ti, (symbol, tid) in enumerate(zip(case['original_targets'], case['target_exact_ENSG'])):
                public_ti = target_lookup[tid]
                tau, rknown, nc, ng, learned, pattern, table, checks = scientific['grouped_predictions'](
                    math, K, k, logk, model['B'][:, public_ti], model['known'][:, public_ti],
                    model['components'], model['source_contexts'], context_identifiable=geometry is not None)
                editable = original['editable_mask'][ti].astype(bool)
                own = np.asarray(axis['model_exact_ENSG'].astype(str) == tid)
                assert not editable[own].any() and not editable[axis['seen'] <= 0].any()
                eligible = editable & qknown
                tau[:, ~eligible] = 0.0
                # Save model request and export quantization separately. Nominal
                # readout uses the actual exported F32 tau, but never claims a
                # final integer/normalized mean or count-ABI equality.
                exported_internal = tau.astype(np.float32)
                M, rho, decoded = scientific['actual_decode'](math, exported_internal.astype(np.float64), C, qknown, eligible)
                requested = tau.copy()
                requested[:, ~eligible] = np.nan
                semantic = rknown & eligible[None, :]
                for ai, ix in enumerate(ARM_INDEX):
                    all_effects[ai, ti] = exported_internal[ix]
                    assert not np.count_nonzero(all_effects[ai, ti, ~editable])
                slot_path = OUT/current_role/'slots'/f'{ti:02d}_{symbol}.npz'
                slot_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(slot_path, targets=np.asarray(symbol), target_exact_ENSG=np.asarray(tid),
                    model_genes=axis['genes'], model_exact_ENSG=axis['model_exact_ENSG'],
                    internal_arms=np.asarray(INTERNAL_ARMS), formal_arms=np.asarray(ARMS), formal_arm_indices=np.asarray(ARM_INDEX),
                    source_contexts=np.asarray(model['source_contexts']), components=model['components'],
                    source_beta_F32=model['B'][:, public_ti], source_response_known=model['known'][:, public_ti],
                    source_n=model['n'][:, public_ti], source_eligible_n30=model['eligible'][:, public_ti],
                    requested_tau_F64=requested, export_total_effect_F32=exported_internal,
                    prediction_response_known=semantic, unknown_response_policy0=eligible[None, :] & ~rknown,
                    source_context_count=nc, source_component_count=ng, context_correction_learned=learned,
                    grouped_mask_pattern=pattern, own_mask=own, inherited_B0_editable=editable,
                    recipient_NTC_model_measured=qknown, recipient_NTC_model_mean_F64=C,
                    original_count_ABI_sparse_mean_toF64=C_legacy, query_H5954_rawmean_F64=G,
                    query_H5954_logCP50K=Q, K=K, query_kernel=k, query_logkernel=logk,
                    nominal_decoder_M_from_preciseC_and_exporttau=M, nominal_decoder_ratio=rho,
                    decoded_rawPC05_from_nominalM=decoded, true_final_integer_mean_computed=np.asarray(False),
                    formal_count_ratio_computed=np.asarray(False))
                sr = dict(role=current_role, target_ordinal=ti, target_symbol=symbol, target_exact_ENSG=tid,
                          slot_path=str(slot_path), slot_SHA256=sha(slot_path), scalar_equivalence_checks=checks,
                          mask_patterns=table, effect_F32_SHA256=[arrayhash(all_effects[ai, ti]) for ai in range(3)])
                slot_files.append(sr)
                completed_slots.extend(x for x in p['expected_slots'] if x['role'] == current_role and x['target_ordinal'] == ti)
                resources(start)
            for ai, arm in enumerate(ARMS):
                bundle = OUT/current_role/arm/'effects.npz'
                bundle.parent.mkdir(parents=True, exist_ok=True)
                value = {name: a.copy() for name, a in original.items()}
                value['effect'] = all_effects[ai]
                for name in value:
                    if name != 'effect':
                        np.testing.assert_array_equal(value[name], original[name])
                assert set(value) == set(original)
                np.savez_compressed(bundle, **value)
                complete = bundle.parent/'complete.json'
                write_new(complete, dict(schema='Native6_public19_KRR_mean_bundle_complete_v1', completed=True,
                    PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'), role=current_role, arm=arm, internal_arm_index=ARM_INDEX[ai],
                    bundle_path=str(bundle), bundle_SHA256=sha(bundle), targets=case['original_targets'],
                    target_exact_ENSG=case['target_exact_ENSG'], baseline_bundle_path=leaves['B0_effects']['path'],
                    baseline_bundle_SHA256=leaves['B0_effects']['sha256'], all_non_effect_keys_arrays_B0_exact=True,
                    effect_F32_SHA256=arrayhash(value['effect']), units='raw_log2_pseudocount_0.5', raw_prior_alpha=1,
                    source_model_seal_path=str(model_seal), source_model_seal_SHA256=sha(model_seal),
                    source_model_path=model['receipt']['model_path'], source_model_SHA256=model['receipt']['model_SHA256'],
                    query_receipt=qreceipt, geometry_query_diagnostics=qdiag, slot_records=slot_files,
                    original_bundle_source_contexts_retained_as_ABI=True,
                    actual_response_sources=[c for c in model['source_contexts'] if c in PUBLIC_CONTEXTS],
                    source_unknown_contexts=[c for c in model['source_contexts'] if c not in PUBLIC_CONTEXTS],
                    held_response_reads=0, THP_numeric_IO=0, source_T_C_values_decoded_by_this_driver=0,
                    count_calls=0, judge_calls=0, nominal_decoder_readout_is_not_final_count_mean=True))
                means.append(dict(role=current_role, context=case['context'], arm=arm,
                    bundle_path=str(bundle), bundle_sha256=sha(bundle), complete_path=str(complete), complete_SHA256=sha(complete),
                    source_model_seal_path=str(model_seal), source_model_seal_SHA256=sha(model_seal)))
            write_new(OUT/current_role/'ROLE_COMPLETE.json', dict(role=current_role, completed=True,
                       PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'), slot_records=slot_files,
                       mean_records=[m for m in means if m['role'] == current_role], held_response_reads=0,
                       THP_numeric_IO=0, count_calls=0, judge_calls=0))
            write_new(OUT/f'PROGRESS_{len(means):02d}.json', dict(completed_mean_bundles=len(means),
                       completed_formal_slots=len(completed_slots), expected_formal_slots=144, current_role=current_role))
            print(json.dumps(dict(role=current_role, completed_mean_bundles=len(means), completed_slots=len(completed_slots))), flush=True)
        assert len(means) == 18 and len(completed_slots) == 144
        result_resource = resources(start)
        phase = 'POST_NUMERIC_BINDING_VERIFICATION'
        post = time.monotonic()
        verify(p['bindings_SHA256'])
        assert p['runtime_preview'] == runtime()
        post_seconds = time.monotonic()-post
        seal = OUT/'MEANS_SEALED.json'
        write_new(seal, dict(schema='Native6_public19_KRR_all144_mean_exports_sealed_v1', completed=True,
                   PROTOCOL_SHA256=sha(OUT/'PROTOCOL.json'), arms=list(ARMS), records=means,
                   expected_formal_slots=144, completed_formal_slots=144, expected_roles=6, mean_bundles=18,
                   source_model_seal_path=str(model_seal), source_model_seal_SHA256=sha(model_seal),
                   source_function_AST_SHA256=asts, held_response_reads=0, THP_numeric_IO=0,
                   count_calls=0, judge_calls=0, Native_query19_response_reads=0,
                   numeric_resource=result_resource, opaque_preverify_seconds=opaque_seconds,
                   opaque_postverify_seconds=post_seconds, all_start_end_bindings_runtime_verified=True,
                   actual_IO_trace=trace, no_native_mean_truth_selection=True,
                   all144_mean_vectors_before_any_count_or_judge=True,
                   future_count_gate='Independent saved mean gate before root originalNB400 all18 new count matrices; allcountseal before judge'))
        print(json.dumps(dict(completed=True, MEANS_SEALED_path=str(seal), MEANS_SEALED_SHA256=sha(seal),
                              resource=result_resource)), flush=True)
    except BaseException as e:
        try:
            rr = resources()
        except BaseException as ee:
            rr = dict(resource_receipt_error=repr(ee), raw_memory=memory_receipt())
        write_new(OUT/f'FAILED_{time.time_ns()}.json', dict(phase=phase, current_role=current_role, error=repr(e),
                   traceback=traceback.format_exc(), completed_mean_records=means,
                   completed_formal_slots=completed_slots, IO_trace=trace, resource=rr,
                   partial_preserved=True, no_partial_promotion=True, no_implicit_retry=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('draft', 'freeze', 'predict'))
    parser.add_argument('--draft-path', type=Path, default=HERE/'NATIVE6_MEAN_DRIVER_DRAFT.json')
    args = parser.parse_args()
    if args.phase == 'draft':
        draft(args.draft_path)
    elif args.phase == 'freeze':
        freeze(args.draft_path)
    else:
        predict()
