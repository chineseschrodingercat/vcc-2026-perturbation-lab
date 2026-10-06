"""Three fixed paired count seeds on unchanged clean400 B0 mean/theta/seen.

First-seed sealed counts and scores are SHA-reused. All 36 new integer-count
predictions are sealed before a single sequential vcc313 scoring process starts.
Only the existing three-background development benchmark is evaluated.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
import os
import shutil
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[key] = '4'
HERE = Path(__file__).resolve().parent
ROOT = Path('C:/00 - Competition')
OUT = ROOT / 'runs/research_v28_20261005/clean400_seedrep_v1'
SOURCE_OUT = ROOT / 'runs/research_v28_20261005/clean400_v1'
GENERATION_PYTHON = ROOT / 'envs/foundation312/Scripts/python.exe'
SCORING_PYTHON = ROOT / 'envs/vcc313/Scripts/python.exe'
import run_clean400 as cc
import numpy as np
import pandas as pd
from scipy import sparse
nr = cc.nr
ARMS = ('clean_B0_NB', 'clean_B0_Poisson', 'clean_B0_increment')
SEEDS = (20260928, 20260929, 20260930)
CELLS = 400
NEW_COUNT_OUTPUTS = 36
TOTAL_COUNT_OUTPUTS = 54


def fresh(path, value):
    cc.fresh(path, value)


def reserve(extra=0):
    cc.reserve(extra)


def owned_case(seed, role):
    return OUT / f'seed_{seed}' / role


def hardlink_checked(source, dest, expected):
    source, dest = Path(source), Path(dest)
    if nr.sha(source) != expected:
        raise ValueError('Source reuse bytes changed: ' + str(source))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        os.link(source, dest)
    if nr.sha(dest) != expected:
        raise ValueError('New reuse link bytes changed: ' + str(dest))


def generation_runtime():
    import scipy
    return {'python': str(Path(sys.executable).resolve()), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'threads': 4}


def verify_protocol(doc):
    if doc['schema'] != 'clean400_count_seedrep_v1' or tuple(doc['seeds']) != SEEDS or tuple(doc['arms']) != ARMS:
        raise ValueError('Independent seedrep protocol identity mismatch')
    for path, digest in doc['dependencies'].items():
        if nr.sha(path) != digest:
            raise ValueError('Seedrep frozen dependency drift: ' + path)
    cc.verify_frozen(doc['source_clean400_protocol'])
    for record in doc['pilot_reuse']:
        for item in record['files'].values():
            if nr.sha(item['path']) != item['sha256']:
                raise ValueError('Seedrep pilot/source result changed: ' + item['path'])
    return doc


def freeze():
    path = OUT / 'PROTOCOL.json'
    if path.exists():
        return verify_protocol(nr.read(path))
    if Path(sys.executable).resolve() != GENERATION_PYTHON.resolve():
        raise RuntimeError('Freeze with the original foundation312 generation environment')
    reserve()
    source = cc.freeze()
    pilot_seal = cc.validate_seal(source)
    source_complete = nr.read(SOURCE_OUT / 'COMPLETE.json')
    if source_complete['actual_integer_count_scores'] != 18 or not source_complete['completed']:
        raise ValueError('Original first-seed complete scores required')
    pilot = []
    for record in pilot_seal['records']:
        role, arm = record['role'], record['arm']
        directory = SOURCE_OUT / role
        files = {
            'prediction': cc.spec(record['prediction_path'], record['prediction_sha256']),
            'moments': cc.spec(record['moments_path'], record['moments_sha256']),
            'generation': cc.spec(directory / f'{arm}_generation.json')}
        score_dir = directory / f'scores/{arm}'
        if not (score_dir / 'summary.json').exists():
            raise ValueError('Incomplete first-seed score: ' + str(score_dir))
        for leaf in sorted(score_dir.iterdir()):
            if leaf.is_file():
                files['score/' + leaf.name] = cc.spec(leaf)
        pilot.append({'seed': SEEDS[0], 'role': role, 'arm': arm, 'files': files,
                      'source_generation_record': record, 'source_score_dir': str(score_dir)})
    dependencies = [Path(__file__), HERE / 'run_clean400.py', HERE / 'run_trace400.py',
                    ROOT / 'scripts/native_reproducer.py', SOURCE_OUT / 'PROTOCOL.json',
                    SOURCE_OUT / 'COMPLETE.json', SOURCE_OUT / 'PREDICTIONS_SEALED.json',
                    GENERATION_PYTHON, SCORING_PYTHON]
    site = ROOT / 'envs/foundation312/Lib/site-packages'
    for pattern in ('numpy-*.dist-info', 'scipy-*.dist-info', 'anndata-*.dist-info', 'h5py-*.dist-info'):
        for dist in sorted(site.glob(pattern)):
            dependencies += [p for p in (dist / 'METADATA', dist / 'RECORD') if p.exists()]
    dependencies += sorted((site / 'numpy/random').glob('*.pyd'))
    dependencies += sorted((site / 'numpy/_core').glob('*.pyd'))
    doc = {'schema': 'clean400_count_seedrep_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'arms': list(ARMS), 'seeds': list(SEEDS), 'cells_per_target': CELLS,
        'baseline': 'clean_B0_NB', 'mean_theta_seen': 'Unchanged retained V20 B0 double-held effect/F23 theta/seen',
        'source_clean400_protocol_sha256': nr.sha(SOURCE_OUT / 'PROTOCOL.json'),
        'source_clean400_protocol': source, 'pilot_reuse': pilot,
        'pilot_seed_previously_viewed': True, 'followup_two_count_seeds_predeclared_before_new_scores': True,
        'new_integer_count_outputs': NEW_COUNT_OUTPUTS, 'total_count_outputs': TOTAL_COUNT_OUTPUTS,
        'donor_sampling': 'Same original 3000 NTC row order; seed+100; 400 distinct donors per target, shared across all arms',
        'generator_rng': 'Original NB seed namespaces; imported Poisson701/increment702; unchanged effects/variance domains',
        'all_new_36_predictions_before_any_new_judge': True,
        'first_seed_scoring': 'Reuse all 18 saved predictions/moments/scores by verified SHA hardlinks; do not rejudge',
        'judge_processes': 1, 'judge_order': 'seed order then original six roles then all three arms, sequential',
        'judge_threads': 4, 'scorer_environment': str(SCORING_PYTHON),
        'generation_runtime': generation_runtime(), 'generation_environment': str(GENERATION_PYTHON),
        'cache_policy': 'New seed/role path per case; no existing cache copied or shared',
        'main_metric': 'Replicate-scaled six-metric overall, no requirement that all six metrics increase',
        'aggregation': 'Six metrics equal per role; two roles equal per background; three backgrounds equal; three seeds equal',
        'bootstrap': {'unit': 'Paired crossed background plus count-seed resampling, same indices across arms',
                      'method': 'Exact enumeration of 3^3 background draws times 3^3 count-seed draws (729 equally weighted draws)',
                      'interval': 'Percentile 2.5%/97.5%', 'cells_resampled_as_independent': False,
                      'roles_are_independent_repeats': False, 'count_seeds_are_biological_replicates': False},
        'benchmark_status': 'Previously tuned three-Native-background development benchmark; no untouched final claim',
        'strict_publisher_independence': False, 'final_THP1_opened': False,
        'new_model_fits': 0, 'official_calls': 0,
        'dependencies': {str(p.resolve()): nr.sha(p) for p in dependencies}}
    fresh(path, doc)
    handoff('FROZEN', 'Protocol frozen; next reuse pilot then generate both new seeds before any new scoring')
    return doc


def handoff(status, next_step, detail=None):
    value = {'status': status, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'phase_scope': 'Three fixed count seeds and all three arms only; no additional candidate/final/official work',
        'next_step': next_step, 'detail': detail, 'official_calls': 0, 'new_model_fits': 0,
        'code_sha256': nr.sha(__file__)}
    nr.write(OUT / 'HANDOFF.json', value)


def reuse_pilot(doc):
    records = []
    cases = {case['role']: case for case in doc['source_clean400_protocol']['cases']}
    for pilot in doc['pilot_reuse']:
        seed, role, arm = pilot['seed'], pilot['role'], pilot['arm']
        dest = owned_case(seed, role)
        nr.prepare_case(cases[role]['job'], dest)
        for key, item in pilot['files'].items():
            if key == 'prediction':
                target = dest / f'pred__{arm}.h5ad'
            elif key == 'moments':
                target = dest / f'{arm}_moments.npz'
            elif key == 'generation':
                target = dest / f'{arm}_SOURCE_generation.json'
            else:
                target = dest / 'scores' / arm / key.removeprefix('score/')
            hardlink_checked(item['path'], target, item['sha256'])
        source = pilot['source_generation_record']
        rec = {'completed': True, 'seed': seed, 'arm': arm, 'role': role, 'context': cases[role]['context'],
            'prediction_path': str(dest / f'pred__{arm}.h5ad'), 'prediction_sha256': pilot['files']['prediction']['sha256'],
            'moments_path': str(dest / f'{arm}_moments.npz'), 'moments_sha256': pilot['files']['moments']['sha256'],
            'protocol_sha256': nr.sha(OUT / 'PROTOCOL.json'), 'source_identity': source['source_identity'],
            'count_seed': seed, 'donor_seed': seed+100, 'cells_per_target': CELLS, 'cells': source['cells'],
            'reused_first_seed': True, 'scores_reused_no_new_judge': True,
            'source_clean400_protocol_sha256': doc['source_clean400_protocol_sha256'],
            'source_score_paths_in_original_configs_preserved': True, 'new_count_draws': 0, 'new_judges': 0,
            'source_files': pilot['files']}
        manifest = dest / f'{arm}_generation.json'
        if manifest.exists():
            if nr.read(manifest) != rec:
                raise ValueError('Pilot reuse receipt differs')
        else:
            fresh(manifest, rec)
        score_receipt = dest / f'scores/{arm}/REUSED_SCORE_RECEIPT.json'
        if not score_receipt.exists():
            fresh(score_receipt, {'seed': seed, 'role': role, 'arm': arm,
                'source_score_dir': pilot['source_score_dir'], 'source_files': pilot['files'],
                'rejudged': False, 'protocol_sha256': nr.sha(OUT / 'PROTOCOL.json')})
        records.append(rec)
    return records


def validate_seal(doc):
    seal = nr.read(OUT / 'PREDICTIONS_SEALED.json')
    expected = {(s, c['role'], a) for s in SEEDS for c in doc['source_clean400_protocol']['cases'] for a in ARMS}
    if len(seal['records']) != TOTAL_COUNT_OUTPUTS or {(r['seed'], r['role'], r['arm']) for r in seal['records']} != expected:
        raise ValueError('All 54 exact seed/role/arm predictions must be sealed')
    if seal['protocol_sha256'] != nr.sha(OUT / 'PROTOCOL.json'):
        raise ValueError('Seedrep prediction seal protocol changed')
    for r in seal['records']:
        if r['protocol_sha256'] != seal['protocol_sha256'] or nr.sha(r['prediction_path']) != r['prediction_sha256'] or nr.sha(r['moments_path']) != r['moments_sha256']:
            raise ValueError('Seedrep sealed counts/moments changed')
    return seal


def generate():
    doc = freeze()
    if Path(sys.executable).resolve() != GENERATION_PYTHON.resolve() or generation_runtime() != doc['generation_runtime']:
        raise RuntimeError('Keep exact foundation312 count-generation runtime')
    if (OUT / 'PREDICTIONS_SEALED.json').exists():
        validate_seal(doc)
        return
    lock = OUT / 'GENERATING.lock'
    with lock.open('x', encoding='utf8') as f:
        f.write(str(os.getpid()))
    try:
        import anndata as ad
        ad.settings.allow_write_nullable_strings = True
        start = time.monotonic()
        records = reuse_pilot(doc)
        handoff('GENERATING', 'Generate both new seeds, no judge until all 36 new outputs are sealed', {'reused_pilot_outputs': len(records)})
        for seed in SEEDS[1:]:
            for case in doc['source_clean400_protocol']['cases']:
                dest = owned_case(seed, case['role'])
                nr.prepare_case(case['job'], dest)
                gen = cc.CleanPanelGenerator(case)
                gen.count_seed = seed
                rng = np.random.default_rng(seed+100)
                gen.donors = {t: rng.choice(gen.pool.shape[0], CELLS, replace=False) for t in gen.order}
                expected_reference = {}
                for arm in ARMS:
                    prediction = dest / f'pred__{arm}.h5ad'
                    manifest = dest / f'{arm}_generation.json'
                    if manifest.exists():
                        rec = nr.read(manifest)
                        if rec['protocol_sha256'] != nr.sha(OUT / 'PROTOCOL.json') or nr.sha(prediction) != rec['prediction_sha256'] or nr.sha(rec['moments_path']) != rec['moments_sha256']:
                            raise ValueError('Existing seedrep generated artifact changed')
                        with np.load(rec['moments_path'], allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['targets'].astype(str), np.asarray(gen.targets))
                            for t, e in zip(gen.targets, z['expected_and_sample_mean'][:, 0]):
                                if t in expected_reference:
                                    np.testing.assert_array_equal(e, expected_reference[t])
                                else:
                                    expected_reference[t] = e.copy()
                        records.append(rec)
                        continue
                    if prediction.exists():
                        raise ValueError('Preserve unsealed partial prediction: ' + str(prediction))
                    reserve(CELLS * len(gen.targets) * len(gen.genes) * 8 + 16 * 2**20)
                    blocks, labels, diags, moments = [], [], [], []
                    for target in gen.targets:
                        if arm == 'clean_B0_NB':
                            result = gen.build_target(target)
                            block, diag = result.block, result.diagnostics
                            expected = result.moments['conditional_given_H0_expected_gene_mean']
                        else:
                            mode = 'formal_Poisson' if arm == 'clean_B0_Poisson' else 'formal_increment'
                            block, diag, helper_expected = cc.modified_block(gen, target, mode)
                            expected = gen.last_h0_conditional_expected_mean(target)
                            diag['protected_mean_readout_float64_repair_max_abs'] = float(np.max(np.abs(expected-helper_expected)))
                        expected = np.asarray(expected, np.float64)
                        if target in expected_reference:
                            np.testing.assert_array_equal(expected, expected_reference[target])
                        else:
                            expected_reference[target] = expected.copy()
                        sample = np.asarray(block.mean(0)).ravel().astype(np.float64)
                        diag.update(mode=arm, cells=CELLS, mean_source='legal_double_held_V20_B0_alpha1',
                            bulk_lognorm_drift_RMS=float(np.sqrt(np.mean((cc.normalized_profile(sample)-cc.normalized_profile(expected))**2))),
                            sample_library_mean=float(sample.sum()), conditional_expected_library_mean=float(expected.sum()),
                            same_conditional_mean_across_clean_arms_verified=True)
                        blocks.append(block)
                        labels.extend([target] * CELLS)
                        diags.append(diag)
                        moments.append([expected, sample])
                    matrix = sparse.vstack(blocks, format='csr')
                    obj = ad.AnnData(matrix,
                        obs=pd.DataFrame({'target_gene': labels}, index=[f'{arm}_{i}' for i in range(len(labels))]),
                        var=pd.DataFrame(index=gen.genes))
                    obj.write_h5ad(prediction, compression='gzip')
                    mpath = dest / f'{arm}_moments.npz'
                    np.savez_compressed(mpath, targets=np.asarray(gen.targets), genes=gen.genes,
                                        expected_and_sample_mean=np.asarray(moments))
                    rec = {'completed': True, 'seed': seed, 'arm': arm, 'role': case['role'], 'context': case['context'],
                        'prediction_path': str(prediction), 'prediction_sha256': nr.sha(prediction),
                        'moments_path': str(mpath), 'moments_sha256': nr.sha(mpath),
                        'protocol_sha256': nr.sha(OUT / 'PROTOCOL.json'), 'source_identity': gen.source_identity,
                        'count_seed': seed, 'donor_seed': seed+100, 'cells_per_target': CELLS, 'cells': matrix.shape[0],
                        'targets': diags, 'generation_runtime': generation_runtime(),
                        'reused_first_seed': False, 'truth_read_by_generation': False, 'new_model_fits': 0, 'official_calls': 0}
                    fresh(manifest, rec)
                    records.append(rec)
                    print('SEEDREP_COUNT', seed, case['role'], arm, matrix.shape, flush=True)
                    handoff('GENERATING', 'Finish all 36 new count outputs before any new judge',
                            {'completed_new_outputs': len(records)-18, 'last_seed': seed, 'last_role': case['role'], 'last_arm': arm})
                    del matrix, obj, blocks
                    reserve()
        fresh(OUT / 'PREDICTIONS_SEALED.json', {'completed': True, 'records': records,
            'protocol_sha256': nr.sha(OUT / 'PROTOCOL.json'), 'seconds': time.monotonic()-start,
            'total_outputs': TOTAL_COUNT_OUTPUTS, 'reused_outputs': 18, 'new_outputs': NEW_COUNT_OUTPUTS,
            'all_new_predictions_frozen_before_new_judge': True, 'new_judges_at_seal': 0, 'official_calls': 0})
        validate_seal(doc)
        handoff('COUNTS_SEALED', 'Run one vcc313 process sequentially with 4 threads for both new seeds', {'new_outputs': 36, 'reused_pilot_outputs': 18})
    finally:
        lock.unlink()


def paired_bootstrap(matrix):
    draws = list(itertools.product(range(3), repeat=3))
    distribution = np.asarray([matrix[np.ix_(b, s)].mean() for b in draws for s in draws], dtype=np.float64)
    return {'paired_macro_delta': float(matrix.mean()), 'crossed_background_seed_percentile95': np.quantile(distribution, [.025, .975]).tolist(),
            'fraction_positive_bootstrap_draws': float(np.mean(distribution > 0)), 'equally_weighted_bootstrap_draws': len(distribution),
            'minimum_draw': float(distribution.min()), 'maximum_draw': float(distribution.max()),
            'unit': 'Three backgrounds and three stochastic count seeds, paired across arms; roles/cells are not resampled as independent'}


def collect():
    doc = freeze()
    rows = []
    for seed in SEEDS:
        for case in doc['source_clean400_protocol']['cases']:
            dest = owned_case(seed, case['role'])
            for arm in ARMS:
                path = dest / f'scores/{arm}/summary.json'
                if not path.exists():
                    continue
                comp = pd.read_csv(dest / f'scores/{arm}/overall_components.csv')
                summary = nr.read(path)
                rows.append({'seed': seed, 'role': case['role'], 'context': case['context'], 'arm': arm,
                    **summary, **{f'scaled__{r.metric}': float(r.replicate_scaled) for r in comp.itertuples()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'completed_scores.csv', index=False)
    per_seed = []
    per_background_seed = []
    for seed in SEEDS:
        for arm in ARMS:
            group = frame[(frame.seed==seed)&(frame.arm==arm)] if not frame.empty else frame
            if len(group) != 6:
                continue
            bybg = group.groupby('context').replicate_overall.mean().to_dict()
            baseline = frame[(frame.seed==seed)&(frame.arm=='clean_B0_NB')].groupby('context').replicate_overall.mean().to_dict()
            delta = float(np.mean([bybg[c]-baseline[c] for c in nr.CONTEXTS])) if len(baseline)==3 else None
            per_seed.append({'seed': seed, 'arm': arm, 'overall': float(np.mean(list(bybg.values()))), 'delta_vs_NB': delta,
                'per_background': bybg, 'six_scaled': {m: float(group[f'scaled__{m}'].mean()) for m in nr.METRICS}})
            for context, value in bybg.items():
                per_background_seed.append({'seed': seed, 'arm': arm, 'context': context, 'overall': value,
                    'delta_vs_same_seed_same_background_NB': value-baseline[context] if context in baseline else None})
    macro = []
    bootstrap = []
    for arm in ARMS:
        entries = [e for e in per_seed if e['arm']==arm]
        if len(entries)==3:
            macro.append({'arm': arm, 'three_seed_macro_overall': float(np.mean([e['overall'] for e in entries])),
                'three_seed_macro_delta_vs_NB': float(np.mean([e['delta_vs_NB'] for e in entries])),
                'per_background_three_seed_mean': {c:float(np.mean([e['per_background'][c] for e in entries])) for c in nr.CONTEXTS},
                'all_three_seed_overall_deltas_positive': all(e['delta_vs_NB']>0 for e in entries) if arm!='clean_B0_NB' else None})
    if len(macro)==3:
        for arm in ARMS[1:]:
            matrix = np.array([[next(x['delta_vs_same_seed_same_background_NB'] for x in per_background_seed if x['seed']==s and x['context']==c and x['arm']==arm) for s in SEEDS] for c in nr.CONTEXTS])
            bootstrap.append({'arm': arm, 'backgrounds': list(nr.CONTEXTS), 'seeds': list(SEEDS), 'paired_delta_matrix': matrix.tolist(), **paired_bootstrap(matrix)})
    progress = {'completed': len(frame)==54, 'completed_score_cases': len(frame), 'reused_first_seed_score_cases': 18,
        'new_score_cases_completed': max(0,len(frame)-18), 'per_seed': per_seed, 'per_background_seed': per_background_seed,
        'three_seed_macro': macro, 'paired_crossed_bootstrap': bootstrap,
        'only_promotion_metric': 'Same-protocol overall', 'all_arms_retained': True,
        'independent_units': 'Three previously tuned Native backgrounds; count seeds measure generator Monte Carlo variability, not biological replication; six roles not independent',
        'new_model_fits': 0, 'official_calls': 0, 'final_THP1_opened': False,
        'protocol_sha256': nr.sha(OUT/'PROTOCOL.json')}
    nr.write(OUT / 'PROGRESS.json', progress)
    return frame, progress


def score():
    if Path(sys.executable).resolve() != SCORING_PYTHON.resolve():
        raise RuntimeError('Score only using pinned vcc313')
    doc = freeze()
    validate_seal(doc)
    nr.engine()
    lock = OUT / 'SCORING.lock'
    with lock.open('x',encoding='utf8') as f:
        f.write(str(os.getpid()))
    try:
        for seed in SEEDS[1:]:
            for case in doc['source_clean400_protocol']['cases']:
                reserve(2*2**30)
                dest = owned_case(seed,case['role'])
                names = [a for a in ARMS if not(dest/f'scores/{a}/summary.json').exists()]
                if names:
                    nr.score_case(dest,names,4)
                _, progress = collect()
                print('SEEDREP_PROGRESS',seed,case['role'],json.dumps(progress['per_seed']),flush=True)
                handoff('SCORING','Finish remaining predeclared new seed/role scores, single sequential process',
                        {'completed_score_cases':progress['completed_score_cases'],'new_score_cases_completed':progress['new_score_cases_completed'],'last_seed':seed,'last_role':case['role']})
        frame,progress = collect()
        if len(frame)!=54:
            raise ValueError('All fixed 54 seed/role/arm score cases required')
        if not(OUT/'COMPLETE.json').exists():
            fresh(OUT/'COMPLETE.json',{'completed':True,'actual_new_integer_count_outputs':36,'actual_reused_count_outputs':18,
                'actual_new_judge_cases':36,'reused_judge_cases':18,'judge_processes':1,'judge_threads':4,
                'code_sha256':nr.sha(__file__),**progress})
        handoff('COMPLETE','Return results to root; do not start another candidate or final test',progress['three_seed_macro'])
    finally:
        lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','generate','score','collect'))
    args=parser.parse_args();os.chdir(ROOT)
    try:
        if args.phase=='freeze':
            print(json.dumps(freeze(),ensure_ascii=False))
        elif args.phase=='generate':
            generate()
        elif args.phase=='score':
            score()
        else:
            print(json.dumps(collect()[1],ensure_ascii=False))
    except Exception as error:
        fresh(OUT/f'FAILED_{args.phase}_{time.time_ns()}.json',{'error':repr(error),'traceback':traceback.format_exc(),'new_model_fits':0,'official_calls':0})
        handoff('FAILED_PRESERVED','Inspect preserved failure; do not overwrite/reseed partial outputs',{'phase':args.phase,'error':repr(error)})
        raise
