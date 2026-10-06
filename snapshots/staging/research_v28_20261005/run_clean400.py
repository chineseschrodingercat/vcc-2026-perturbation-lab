"""Independent Native400 development branch: legal double-held V20 B0 means.

Three complete-background development panels, two double-held roles each. This
repeatedly tuned development benchmark is not an untouched final test or proof
of publisher/study-independent generalization. No network or official access.
All 18 integer-count predictions are sealed before destination truth is scored.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[key] = '4'
ROOT = Path('C:/00 - Competition')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'runs/research_v28_20261005/clean400_v1'
V20 = ROOT / 'staging/newV20_native_source_selection'
B0_MANIFEST = V20 / 'completed_arms/B0.json'
MACHINE = V20 / 'TRAINER_MACHINE_CONTRACT.json'
GENERATOR_COMPLETE = ROOT / 'models/V20_native_f23_generator/GENERATOR_COMPLETE.json'
V20_PROTOCOL = ROOT / 'configs/V20_NATIVE_F23_BRIDGE_PROTOCOL.json'
sys.path.insert(0, str(ROOT / 'scripts'))
import native_reproducer as nr
import numpy as np
import pandas as pd
from scipy import sparse
# Import only these two count mathematical helpers; never use trace freeze/arms.
from run_trace400 import modified_block, normalized_profile

ARMS = ('clean_B0_NB', 'clean_B0_Poisson', 'clean_B0_increment')
COUNT_SEED = 20260928
DONOR_SEED = COUNT_SEED + 100
CELLS = 400
EXPECTED_OUTPUTS = 18
VCC_PYTHON = ROOT / 'envs/vcc313/Scripts/python.exe'


def fresh(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def reserve(extra_bytes=0):
    if shutil.disk_usage(ROOT).free < 50 * 2**30 + int(extra_bytes):
        raise RuntimeError('C storage reserve would fall below 50 GiB')


def spec(path, expected=None):
    path = Path(path).resolve()
    digest = nr.sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Retained asset hash mismatch: ' + str(path))
    return {'path': str(path), 'sha256': digest}


def verify_frozen(protocol):
    if protocol['schema'] != 'Native400_clean_B0_v1' or tuple(protocol['arms']) != ARMS:
        raise ValueError('Wrong independent clean400 protocol')
    for path, digest in protocol['dependencies'].items():
        if nr.sha(path) != digest:
            raise ValueError('Frozen clean400 dependency drift: ' + path)
    for case in protocol['cases']:
        for item in case['leaves'].values():
            if nr.sha(item['path']) != item['sha256']:
                raise ValueError('Frozen clean400 input drift: ' + item['path'])
    return protocol


def freeze():
    lock_path = OUT / 'PROTOCOL.json'
    if lock_path.exists():
        return verify_frozen(nr.read(lock_path))
    reserve()
    manifest = nr.read(B0_MANIFEST)
    machine = nr.read(MACHINE)
    generator = nr.read(GENERATOR_COMPLETE)
    if not manifest['completed'] or manifest['arm'] != 'B0' or len(manifest['bundles']) != 6:
        raise ValueError('Expected all six completed original double-held B0 bundles')
    if manifest['truth_read_by_trainer'] or machine['physical_K_prior_alpha'] != 1:
        raise ValueError('Wrong alpha or destination-truth training boundary')
    spec(MACHINE, manifest['trainer_machine_contract_SHA256'])
    spec(V20_PROTOCOL, manifest['protocol_sha256'])
    if not generator['completed'] or generator['actual_fits'] != 3:
        raise ValueError('New F23-only generator fits incomplete')
    jobs = {job['key']: job for job in nr.jobs('inner')}
    cases = []
    for bundle in manifest['bundles']:
        job = jobs[bundle['job_key']]
        role = machine['roles'][bundle['role']]
        panel = nr.relocated(job['historical_panel'])
        info = nr.read(panel / 'panel.json')
        complete = nr.read(bundle['complete_path'])
        if job['role'] != bundle['role'] or job['context'] != role['destination']:
            raise ValueError('B0/job destination identity mismatch')
        if len(info['targets']) != 8 or info['targets'] != role['original_targets']:
            raise ValueError('Original complete eight-target population changed')
        if info.get('controls_disjoint') is not True:
            raise ValueError('Original scoring NTC and donor-pool disjointness metadata changed')
        held = set(role['held_families'])
        if role['destination_family'] not in held or 'THP1' not in held or role['allowed_native_family'] in held:
            raise ValueError('Broken double-held training/source boundary')
        if len(role['forbidden_native_families']) != 2:
            raise ValueError('Expected two forbidden Native families per model role')
        if complete['raw_prior_alpha'] != 1 or complete['effect_units'] != 'raw_log2_pseudocount_0.5':
            raise ValueError('B0 must retain its physical K prior alpha1, raw UMI PC0.5')
        for flag in ('held_response_used', 'historical_fallback_used'):
            if complete[flag]:
                raise ValueError('B0 bundle violates clean source boundary: ' + flag)
        if not complete['original_eight_targets_and_population_unchanged']:
            raise ValueError('B0 provenance does not retain the original population')
        binding = complete['generator_binding']
        if binding['model_role'] != role['model_role']:
            raise ValueError('B0 theta/seen role mismatch')
        gs = generator['roles'][binding['model_role']]
        if gs['allowed_native'] != role['allowed_native']:
            raise ValueError('Generator training Native differs from legal mean source')
        decoder_complete = nr.read(gs['complete_path'])
        if job['context'] in decoder_complete['decoder_contexts']:
            raise ValueError('Destination used by F23 generator fitting')
        if sorted(decoder_complete['decoder_contexts']) != sorted(['k562', role['allowed_native']]):
            raise ValueError('Unexpected F23 generator training context')
        if not decoder_complete['no_destination_perturbation_IO'] or not decoder_complete['no_F4_fitting']:
            raise ValueError('Generator metadata lacks the frozen F23 source boundary')
        if Path(binding['dispersion_seen_path']).resolve() != Path(gs['dispersion_seen_path']).resolve():
            raise ValueError('Mean export and generator use different theta/seen assets')
        leaves = {leaf: spec(panel / leaf) for leaf in (
            'real.h5ad', 'panel.json', 'genes.txt', 'generation_pool.npz',
            'references/reference.json', 'references/anchor_aggregate.parquet')}
        for path in sorted((panel / 'references').rglob('*')):
            if path.is_file():
                leaves['reference/' + str(path.relative_to(panel / 'references')).replace('\\', '/')] = spec(path)
        ref = nr.read(panel / 'references/reference.json')
        if not ref['replicate_usable'] or ref['real_sha256'] != leaves['real.h5ad']['sha256']:
            raise ValueError('Original real/anchor reference cannot be reused')
        leaves.update(
            B0_effects=spec(bundle['bundle_path'], bundle['bundle_sha256']),
            B0_complete=spec(bundle['complete_path'], bundle['complete_SHA256']),
            B0_checkpoint=spec(bundle['checkpoint_path'], bundle['checkpoint_SHA256']),
            theta_seen=spec(gs['dispersion_seen_path'], binding['dispersion_seen_SHA256']),
            generator_complete=spec(GENERATOR_COMPLETE, binding['generator_complete_SHA256']),
            role_generator_complete=spec(gs['complete_path'], gs['complete_sha256']),
            role_encoder=spec(gs['encoder_path'], gs['encoder_sha256']))
        cases.append({'job': job, 'role': bundle['role'], 'context': job['context'],
                      'targets': info['targets'], 'model_role': role['model_role'],
                      'allowed_native': role['allowed_native'], 'held_families': sorted(held),
                      'allowed_response_contexts': role['allowed_response_contexts'],
                      'original_truth_population': info['target_cells'],
                      'prediction_cells_per_target': CELLS, 'leaves': leaves})
    if len({case['role'] for case in cases}) != 6 or {case['context'] for case in cases} != set(nr.CONTEXTS):
        raise ValueError('Clean400 is missing an original complete Native role/background')
    dependencies = [Path(__file__), HERE / 'run_trace400.py', ROOT / 'scripts/native_reproducer.py',
        B0_MANIFEST, MACHINE, GENERATOR_COMPLETE, V20_PROTOCOL, V20 / 'SOURCE_ROLE_GATE.json',
        V20 / 'v20_effect_export.py', V20 / 'train_v20.py', V20 / 'prepare_mean_arrays.py',
        V20 / 'prepare_query_arrays.py', V20 / 'v20_boundary.py',
        nr.ARCHIVE / 'runs/literal_nb_v28/inner_jobs.json',
        nr.ARCHIVE / 'runs/context_v4/gene_axis.txt',
        nr.ARCHIVE / 'runs/context_v4/normalization_core.txt',
        ROOT / 'data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz',
        nr.SOURCE / 'literal_nb_v28.py', nr.SOURCE / 'latent_counts_v26.py',
        nr.SOURCE / 'residual_v6_common.py', nr.SOURCE / 'generate_counts.py',
        nr.SOURCE / 'prepare_response_inputs_v23.py', nr.SOURCE / 'score_residual_v6.py',
        nr.SOURCE / 'diagnose_v4.py', VCC_PYTHON]
    package = ROOT / 'envs/vcc313/Lib/site-packages/cell_eval2'
    if not package.is_dir():
        raise FileNotFoundError('Pinned vcc313 cell_eval2 package absent')
    dependencies += sorted(package.rglob('*.py'))
    site = package.parent
    for pattern in ('cell_eval2-*.dist-info', 'scanpy-*.dist-info', 'anndata-*.dist-info',
                    'numpy-*.dist-info', 'scipy-*.dist-info'):
        for dist in sorted(site.glob(pattern)):
            dependencies += [leaf for leaf in (dist / 'METADATA', dist / 'RECORD') if leaf.exists()]
    protocol = {'schema': 'Native400_clean_B0_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Isolate count distribution at fixed legal alpha1 double-held B0 mean response',
        'arms': list(ARMS), 'primary_baseline': 'clean_B0_NB', 'cell_count': CELLS,
        'count_seed': COUNT_SEED, 'donor_seed': DONOR_SEED,
        'heldout_backgrounds_for_model_fit': True, 'double_held_Native_families': True,
        'benchmark_status': 'Repeatedly tuned complete-background development benchmark',
        'untouched_final_test': False, 'strict_publisher_or_study_independent': False,
        'model_mean': 'Retained V20 B0 complete eight-target effects; no new model forward or fitting',
        'raw_prior_alpha': 1, 'legacy_672_prior_used': False,
        'raw_prior_formula': 'clip(log2((T_K+0.5)/(C_K+0.5)),-3,3)*n/(n+50), n>=30, own/unknown zero',
        'mean_unit': 'Raw UMI mean log2 ratio, PC0.5, not CP10K pseudocount',
        'theta_seen': 'Corresponding V20_native_f23_generator assets fit only K562 plus the allowed Native F23 source',
        'generation_pool': 'Original 3000 NTC pool, original row order, 400 distinct paired donors per target/arm',
        'real_population': 'All original complete target cells and original 1000 scoring NTC, unchanged',
        'anchors': 'Original generic baseline and five-split replicate anchors, unchanged; no refitting',
        'generator_baseline': 'Original literal build_target N1 AST with new F23 theta/seen; same fixed B0 mean',
        'Poisson': 'X*rho on editable genes, Poisson resampling; protected H0 unchanged, namespace701',
        'increment': 'rho<1 Binomial(X,rho); rho>1 X+Poisson(X*(rho-1)); rho=1 exactX, namespace702',
        'count_math_source': 'run_trace400.modified_block; only mathematical helpers imported',
        'all_18_predictions_before_judge': True, 'scorer': 'cell_eval2==0.16.0 vcc2026 CPU scanpy, native_reproducer.score_case',
        'scoring_interpreter': str(VCC_PYTHON), 'metric_weights': {m: 1/6 for m in nr.METRICS},
        'aggregation': 'Equal six metrics within role; two roles equal per background; three backgrounds equal',
        'only_primary_promotion_metric': 'replicate_overall relative to same-protocol clean_B0_NB',
        'cross_protocol_differences_allowed': False, 'fixed_next_count_seeds': [20260928, 20260929, 20260930],
        'destination_truth_fitting': False, 'final_THP1_truth_opened': False,
        'training': 0, 'official_calls': 0,
        'dependencies': {str(path.resolve()): nr.sha(path) for path in dependencies}, 'cases': cases}
    fresh(lock_path, protocol)
    return protocol


class CleanPanelGenerator:
    """Original count AST, populated exclusively with clean B0/F23 inputs."""
    def __init__(self, case):
        self.job = case['job']
        self.panel = nr.relocated(self.job['historical_panel'])
        self.info = nr.read(self.panel / 'panel.json')
        self.targets = list(case['targets'])
        self.destination = case['context']
        self.role = case['model_role']
        self.model_seed = int(self.job['model_seed'])
        self.count_seed = COUNT_SEED
        self.genes = np.loadtxt(case['leaves']['genes.txt']['path'], dtype=str, ndmin=1)
        self.output_genes = np.loadtxt(nr.ARCHIVE / 'runs/context_v4/gene_axis.txt', dtype=str, ndmin=1)
        self.pool = sparse.load_npz(case['leaves']['generation_pool.npz']['path']).tocsr()
        if self.pool.shape != (3000, len(self.genes)):
            raise ValueError('Original 3000 NTC donor pool/axis changed')
        if not np.isfinite(self.pool.data).all() or (self.pool.data < 0).any() or not np.equal(self.pool.data, np.rint(self.pool.data)).all():
            raise ValueError('NTC pool counts invalid')
        self.mean = np.asarray(self.pool.mean(0)).ravel().astype(np.float64)
        with np.load(case['leaves']['B0_effects']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['genes'].astype(str), self.output_genes)
            np.testing.assert_array_equal(z['native_genes'].astype(str), self.genes)
            np.testing.assert_array_equal(z['targets'].astype(str), np.asarray(self.targets))
            if bool(z['held_response_used']) or bool(z['historical_fallback_used']):
                raise ValueError('B0 numeric bundle violates held-out boundary')
            if str(z['effect_units']) != 'raw_log2_pseudocount_0.5':
                raise ValueError('Wrong B0 effect units')
            if not set(z['source_contexts'].astype(str)).issubset(case['allowed_response_contexts']):
                raise ValueError('B0 contains an undeclared/forbidden response source')
            self.effects = np.asarray(z['effect'], dtype=np.float32)
            effect_own = np.asarray(z['ownmask'], dtype=np.float32)
            effect_seen = z['new_generator_seen'].astype(bool)
            editable = z['editable_mask'].astype(bool)
        if self.effects.shape != (len(self.targets), len(self.output_genes)) or not np.isfinite(self.effects).all():
            raise ValueError('Invalid complete B0 response matrix')
        if np.count_nonzero(self.effects[~editable]):
            raise ValueError('Protected B0 effect coordinates are nonzero')
        with np.load(case['leaves']['theta_seen']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['gene_symbols'].astype(str), self.output_genes)
            self.theta = z['theta'].astype(np.float64)
            seen = z['seen'].astype(np.float32)
            if not np.isin(z['seen'], [0, 1]).all():
                raise ValueError('Invalid F23 seen mask')
        if not np.isfinite(self.theta).all() or (self.theta <= 0).any():
            raise ValueError('Invalid F23 theta')
        np.testing.assert_array_equal(effect_seen, seen > 0)
        with np.load(case['leaves']['role_encoder']['path'], allow_pickle=False) as z:
            self.input_genes = z['genes'].astype(str)
        if len(self.input_genes) != 6119 or len(set(self.input_genes)) != 6119:
            raise ValueError('Wrong legal F23 generator structural input axis')
        native = {gene: i for i, gene in enumerate(self.genes)}
        output = {gene: i for i, gene in enumerate(self.output_genes)}
        self.input_native = np.asarray([native[g] for g in self.input_genes], dtype=np.int64)
        self.input_output = np.asarray([output[g] for g in self.input_genes], dtype=np.int64)
        own_ns = {'np': np}
        nr.extract(nr.SOURCE / 'prepare_response_inputs_v23.py', ('own_for',), own_ns)
        own = own_ns['own_for'](self.targets, self.output_genes)
        np.testing.assert_array_equal(effect_own, own)
        self.inputs = {'seen': seen, 'ownmask': own}
        self.order = list(map(str, self.info.get('generation_targets', self.targets)))
        self.sizes = {target: CELLS for target in self.order}
        rng = np.random.default_rng(DONOR_SEED)
        self.donors = {target: rng.choice(self.pool.shape[0], CELLS, replace=False) for target in self.order}
        self._scientific = nr.generator_functions()
        self.source_identity = {'mean_path': case['leaves']['B0_effects']['path'],
            'mean_sha256': case['leaves']['B0_effects']['sha256'], 'mean_workflow': 'V20_clean_double_held_B0_alpha1',
            'compact_theta_path': case['leaves']['theta_seen']['path'],
            'compact_theta_sha256': case['leaves']['theta_seen']['sha256'],
            'seen_path': case['leaves']['theta_seen']['path'], 'seen_sha256': case['leaves']['theta_seen']['sha256'],
            'input_gene_identity_path': case['leaves']['role_encoder']['path'], 'raw_prior_alpha': 1,
            'held_families': case['held_families'], 'allowed_native': case['allowed_native'],
            'legacy_effect_theta_encoder_numeric_loaded': False, 'destination_truth_loaded': False}

    def build_target(self, target, arm='N1'):
        result = self._scientific['build_target'](self, target, arm)
        self.last_result = result
        return result

    def last_h0_conditional_expected_mean(self, target):
        """Use original float64 moment precision, including protected H0 counts.

        The imported helper reads sparse float32 means for protected genes. Its
        count mathematics remains unchanged; only this diagnostic mean readout
        is recomputed using the same float64 formula as the original NB AST.
        """
        result = self.last_result
        if result.target != target or result.diagnostics['arm'] != 'H0':
            raise ValueError('Expected the exact H0 used by the imported count helper')
        editable = result.moments['editable_native_positions']
        protected = np.ones(len(self.genes), dtype=bool)
        protected[editable] = False
        conditional = self.pool[self.donors[target]].toarray().astype(np.float64)
        conditional *= result.moments['count_ratio_native'][None, :]
        conditional[:, protected] = result.block[:, protected].toarray().astype(np.float64)
        return conditional.mean(0)


def validate_seal(protocol):
    seal = nr.read(OUT / 'PREDICTIONS_SEALED.json')
    expected = {(case['role'], arm) for case in protocol['cases'] for arm in ARMS}
    records = seal['records']
    if len(records) != EXPECTED_OUTPUTS or {(r['role'], r['arm']) for r in records} != expected:
        raise ValueError('All exact 18 clean400 predictions must be sealed before scoring')
    if seal['protocol_sha256'] != nr.sha(OUT / 'PROTOCOL.json'):
        raise ValueError('Prediction seal belongs to another protocol')
    for record in records:
        if record['protocol_sha256'] != seal['protocol_sha256'] or nr.sha(record['prediction_path']) != record['prediction_sha256']:
            raise ValueError('Sealed prediction bytes/protocol changed')
        if nr.sha(record['moments_path']) != record['moments_sha256']:
            raise ValueError('Sealed moment diagnostics changed')
    return seal


def generate():
    protocol = freeze()
    if (OUT / 'PREDICTIONS_SEALED.json').exists():
        validate_seal(protocol)
        return
    lock = OUT / 'GENERATING.lock'
    with lock.open('x', encoding='utf8') as handle:
        handle.write(str(os.getpid()))
    try:
        import anndata as ad
        ad.settings.allow_write_nullable_strings = True
        records = []
        start = time.monotonic()
        for case in protocol['cases']:
            dest = OUT / case['role']
            nr.prepare_case(case['job'], dest)
            gen = CleanPanelGenerator(case)
            conditional_reference = {}
            for arm in ARMS:
                output = dest / f'pred__{arm}.h5ad'
                manifest = dest / f'{arm}_generation.json'
                if manifest.exists():
                    record = nr.read(manifest)
                    if record['protocol_sha256'] != nr.sha(OUT / 'PROTOCOL.json') or nr.sha(output) != record['prediction_sha256']:
                        raise ValueError('Existing clean count bytes/protocol changed')
                    if nr.sha(record['moments_path']) != record['moments_sha256']:
                        raise ValueError('Existing clean moment bytes changed')
                    with np.load(record['moments_path'], allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['targets'].astype(str), np.asarray(gen.targets))
                        np.testing.assert_array_equal(z['genes'].astype(str), gen.genes)
                        saved_expected = z['expected_and_sample_mean'][:, 0]
                    for target, expected in zip(gen.targets, saved_expected):
                        if target in conditional_reference:
                            np.testing.assert_array_equal(expected, conditional_reference[target])
                        else:
                            conditional_reference[target] = expected.copy()
                    records.append(record)
                    continue
                if output.exists():
                    raise RuntimeError('Preserve unsealed existing clean count: ' + str(output))
                reserve(CELLS * len(gen.targets) * len(gen.genes) * 8 + 16 * 2**20)
                blocks, labels, diags, moments = [], [], [], []
                for target in gen.targets:
                    if arm == 'clean_B0_NB':
                        result = gen.build_target(target)
                        block, diag = result.block, result.diagnostics
                        expected = result.moments['conditional_given_H0_expected_gene_mean']
                    else:
                        mode = 'formal_Poisson' if arm == 'clean_B0_Poisson' else 'formal_increment'
                        block, diag, helper_expected = modified_block(gen, target, mode)
                        expected = gen.last_h0_conditional_expected_mean(target)
                        diag['protected_mean_readout_float64_repair_max_abs'] = float(np.max(np.abs(expected-helper_expected)))
                    expected = np.asarray(expected, dtype=np.float64)
                    if target in conditional_reference:
                        np.testing.assert_array_equal(expected, conditional_reference[target])
                    else:
                        conditional_reference[target] = expected.copy()
                    sample = np.asarray(block.mean(0)).ravel().astype(np.float64)
                    diag.update(mode=arm, cells=CELLS, mean_source='legal_double_held_V20_B0_alpha1',
                        bulk_lognorm_drift_RMS=float(np.sqrt(np.mean((normalized_profile(sample)-normalized_profile(expected))**2))),
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
                obj.write_h5ad(output, compression='gzip')
                moment_path = dest / f'{arm}_moments.npz'
                np.savez_compressed(moment_path, targets=np.asarray(gen.targets), genes=gen.genes,
                                    expected_and_sample_mean=np.asarray(moments))
                record = {'completed': True, 'arm': arm, 'role': case['role'], 'context': case['context'],
                    'cells': matrix.shape[0], 'cells_per_target': CELLS,
                    'protocol_sha256': nr.sha(OUT / 'PROTOCOL.json'),
                    'prediction_path': str(output), 'prediction_sha256': nr.sha(output),
                    'moments_path': str(moment_path), 'moments_sha256': nr.sha(moment_path),
                    'source_identity': gen.source_identity, 'targets': diags,
                    'truth_read_by_generation': False, 'training': 0, 'official_calls': 0,
                    'count_seed': COUNT_SEED, 'donor_seed': DONOR_SEED}
                fresh(manifest, record)
                records.append(record)
                print('CLEAN400_COUNT', case['role'], arm, matrix.shape, flush=True)
                del obj, matrix, blocks
                reserve()
        if len(records) != EXPECTED_OUTPUTS:
            raise ValueError('Not all 18 clean predictions completed')
        fresh(OUT / 'PREDICTIONS_SEALED.json', {'completed': True, 'records': records,
            'protocol_sha256': nr.sha(OUT / 'PROTOCOL.json'), 'seconds': time.monotonic()-start,
            'all_predictions_frozen_before_judge': True, 'actual_count_outputs': len(records),
            'training': 0, 'official_calls': 0})
        validate_seal(protocol)
    finally:
        lock.unlink()


def collect():
    protocol = freeze()
    rows = []
    for case in protocol['cases']:
        dest = OUT / case['role']
        for arm in ARMS:
            summary = dest / f'scores/{arm}/summary.json'
            if not summary.exists():
                continue
            comp = pd.read_csv(dest / f'scores/{arm}/overall_components.csv')
            rows.append({'arm': arm, 'role': case['role'], 'context': case['context'],
                **nr.read(summary), **{f'scaled__{r.metric}': float(r.replicate_scaled) for r in comp.itertuples()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'completed_scores.csv', index=False)
    aggregates = []
    for arm in ARMS:
        group = frame[frame.arm == arm] if not frame.empty else frame
        if len(group) != 6:
            continue
        aggregates.append({'arm': arm,
            'overall': float(group.groupby('context').replicate_overall.mean().mean()),
            'per_background': group.groupby('context').replicate_overall.mean().to_dict(),
            'six_scaled': {m: float(group[f'scaled__{m}'].mean()) for m in nr.METRICS}})
    baseline = next((item for item in aggregates if item['arm'] == 'clean_B0_NB'), None)
    if baseline is not None:
        for item in aggregates:
            item['delta_vs_same_protocol_clean_B0_NB'] = item['overall'] - baseline['overall']
            item['per_background_delta_vs_same_protocol_clean_B0_NB'] = {
                c: v - baseline['per_background'][c] for c, v in item['per_background'].items()}
    description = None
    trace = ROOT / 'runs/research_v28_20261005/trace400_v1/PROGRESS.json'
    if trace.exists():
        trace_doc = nr.read(trace)
        description = {'label': 'Different trace400 protocol; descriptive only, no score subtraction',
            'path': str(trace), 'sha256': nr.sha(trace),
            'protocol_sha256': trace_doc['protocol_sha256'], 'aggregates': trace_doc['aggregates']}
    progress = {'complete': len(frame) == EXPECTED_OUTPUTS, 'completed_scores': len(frame),
        'aggregates': aggregates, 'primary_baseline': 'clean_B0_NB',
        'benchmark_status': protocol['benchmark_status'], 'untouched_final_test': False,
        'strict_publisher_or_study_independent': False, 'cross_protocol_differences_computed': False,
        'other_protocol_descriptive_only': description, 'protocol_sha256': nr.sha(OUT / 'PROTOCOL.json'),
        'official_calls': 0}
    nr.write(OUT / 'PROGRESS.json', progress)
    return frame, progress


def score(threads):
    if Path(sys.executable).resolve() != VCC_PYTHON.resolve():
        raise RuntimeError('Run scoring with the pinned vcc313 interpreter: ' + str(VCC_PYTHON))
    protocol = freeze()
    validate_seal(protocol)
    nr.engine()
    lock = OUT / 'SCORING.lock'
    with lock.open('x', encoding='utf8') as handle:
        handle.write(str(os.getpid()))
    try:
        for case in protocol['cases']:
            reserve()
            dest = OUT / case['role']
            names = [arm for arm in ARMS if not (dest / f'scores/{arm}/summary.json').exists()]
            if names:
                nr.score_case(dest, names, threads)
            _, progress = collect()
            print('CLEAN400_PROGRESS', json.dumps(progress['aggregates']), flush=True)
        frame, progress = collect()
        if len(frame) != EXPECTED_OUTPUTS:
            raise ValueError('All 18 count predictions must have final six-metric scores')
        if not (OUT / 'COMPLETE.json').exists():
            fresh(OUT / 'COMPLETE.json', {'completed': True,
                'actual_integer_count_scores': len(frame), **progress, 'training': 0})
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'generate', 'score', 'collect'))
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        if args.phase == 'freeze':
            print(json.dumps(freeze(), ensure_ascii=False))
        elif args.phase == 'generate':
            generate()
        elif args.phase == 'score':
            score(args.threads)
        else:
            print(json.dumps(collect()[1], ensure_ascii=False))
    except Exception as error:
        fresh(OUT / f'FAILED_{args.phase}_{time.time_ns()}.json',
              {'error': repr(error), 'traceback': traceback.format_exc(), 'official_calls': 0})
        raise
