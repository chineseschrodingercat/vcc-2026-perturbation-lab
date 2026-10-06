"""Frozen V28 Native replay and relocated thin count/judge adapter.

Historical files remain byte-identical. Scientific function bodies are compiled
directly from the archived implementation rather than reimplemented. Only input
paths and compact theta construction replace the obsolete directory layout.
Run light audit with foundation312; actual judging uses the pinned vcc313.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse

PROJECT = Path('C:/00 - Competition').resolve()
ARCHIVE = PROJECT / 'archive/legacy_20261003'
SOURCE = ARCHIVE / 'src/vcc2026'
OUTPUT = Path(__file__).resolve().parent
METRICS = (
    'pds_cosine', 'expr_mse_unbiased_capped_norm',
    'de_wilcoxon_direction_fidelity_yield_raw',
    'de_wilcoxon_direction_reach_raw', 'de_wilcoxon_sig_jaccard',
    'de_wilcoxon_lfc_nmae',
)
CONTEXTS = ('hepg2', 'jurkat', 'rpe1')
SEED = 20260928


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def relocated(value):
    """Map historical input paths in memory; never rewrite historical JSON."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT / path
    if path.is_relative_to(ARCHIVE):
        return path.resolve()
    if path.is_relative_to(PROJECT):
        candidate = ARCHIVE / path.relative_to(PROJECT)
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def owned(value):
    path = Path(value).resolve()
    if path == PROJECT or not path.is_relative_to(PROJECT) or path.is_relative_to(ARCHIVE):
        raise ValueError(f'New output must stay outside immutable archive: {path}')
    return path


def jobs(stage='inner'):
    source = ARCHIVE / f'runs/literal_nb_v28/{stage}_jobs.json'
    return [item for item in read(source) if item['arm'] == 'N1']


def find_job(key, stage=None):
    result = [item for stage_name in ((stage,) if stage else ('inner', 'outer'))
              for item in jobs(stage_name) if item['key'] == key]
    if len(result) != 1:
        raise ValueError(f'Expected one exact Native job for {key}, got {len(result)}')
    return result[0]


def extract(path, names, namespace, class_name=None):
    """Execute exact old AST bodies with explicit lightweight dependencies."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    candidates = tree.body
    if class_name:
        candidates = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    nodes = [node for node in candidates if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if set(node.name for node in nodes) != set(names):
        raise ValueError(f'Missing original scientific functions: {path} {names}')
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def engine():
    import cell_eval2
    from cell_eval2.score import _replicate_entries
    from cell_eval2.scoring import score_one
    from cell_eval2.scales import SCALES
    if cell_eval2.__version__ != '0.16.0':
        raise RuntimeError(f'Pinned Native judge version drift: {cell_eval2.__version__}')
    return cell_eval2, _replicate_entries, score_one, SCALES


def replay(out=OUTPUT / 'replay'):
    out = owned(out)
    _, replicate_entries, score_one, scales = engine()
    rows, cases, mismatch = [], [], 0.0
    for stage in ('inner', 'outer'):
        for job in jobs(stage):
            result_dir = relocated(job['score_output'])
            panel = relocated(job['historical_panel'])
            refdir = panel / 'references'
            reference = read(refdir / 'reference.json')
            aggregate_path = result_dir / 'aggregate.csv'
            table = pd.read_csv(aggregate_path)
            raw = {row.metric: float(row.mean) for row in table.itertuples() if row.metric in METRICS}
            if set(raw) != set(METRICS) or not reference['replicate_usable']:
                raise ValueError('Missing Native six-metric reference or aggregate')
            entries = replicate_entries(reference['baseline'], pl.read_parquet(refdir / 'anchor_aggregate.parquet'))
            normalized = {metric: score_one(raw[metric], entries[metric].base, entries[metric].scoring) for metric in METRICS}
            overall = float(np.mean(list(normalized.values())))
            summary = read(result_dir / 'summary.json')
            components = pd.read_csv(result_dir / 'overall_components.csv').set_index('metric')
            error = max(abs(overall - summary['replicate_overall']),
                        max(abs(normalized[metric] - components.loc[metric, 'replicate_scaled']) for metric in METRICS))
            mismatch = max(mismatch, error)
            if error > 1e-12:
                raise AssertionError(f'Original Native aggregate replay mismatch: {job["key"]} {error}')
            # Surviving audited score leaves must still match the historical lock.
            lock = read(result_dir / 'retained_result_lock.json')['files_sha256']
            for leaf in ('aggregate.csv', 'summary.json', 'overall_components.csv', 'metrics.csv'):
                if sha(result_dir / leaf) != lock[leaf]:
                    raise AssertionError(f'Historical Native score leaf changed: {result_dir / leaf}')
            info = read(panel / 'panel.json')
            source_paths = {
                'aggregate': aggregate_path, 'summary': result_dir / 'summary.json',
                'components': result_dir / 'overall_components.csv',
                'reference': refdir / 'reference.json',
                'anchor': refdir / 'anchor_aggregate.parquet',
                'panel': panel / 'panel.json', 'genes': panel / 'genes.txt',
                'mean': relocated(job['effect']),
                'theta': ARCHIVE / f'retained/native_theta/{job["model_role"]}/seed_{job["model_seed"]}/dispersion_seen.npz',
            }
            cases.append({
                **{key: job[key] for key in ('key', 'role', 'context', 'model_role', 'model_seed', 'count_seed', 'teacher_train_seed', 'teacher_tag')},
                'stage': stage, 'targets': info['targets'], 'generated_target_cells': info.get('generation_target_cells', info['target_cells']),
                'native_genes': len(np.loadtxt(panel / 'genes.txt', dtype=str, ndmin=1)),
                'reference_real_sha256': reference['real_sha256'],
                'source_leaves': {key: {'path': str(path), 'sha256': sha(path)} for key, path in source_paths.items()},
                'replicate_scale_entries': {metric: asdict(entries[metric]) for metric in METRICS},
                'baseline_replicate_overall': overall,
            })
            rows.append({'stage': stage, **{key: job[key] for key in ('key', 'role', 'context', 'model_seed', 'count_seed', 'teacher_train_seed')},
                         **raw, 'replicate_overall': overall, **{f'normalized__{key}': value for key, value in normalized.items()}})
    frame = pd.DataFrame(rows)
    inner = frame[frame.stage == 'inner'].groupby('context').replicate_overall.mean()
    main_outer = frame[(frame.stage == 'outer') & (frame.model_seed == 20261002)]
    outer = main_outer.groupby('context').replicate_overall.mean()
    repeat = frame[(frame.stage == 'outer') & (frame.model_seed == 20261009)].groupby('context').replicate_overall.mean()
    expected_inner, expected_outer = 0.25932824671778637, 0.2281364395524066
    if abs(float(inner.mean()) - expected_inner) > 1e-12 or abs(float(outer.mean()) - expected_outer) > 1e-12:
        raise AssertionError('Frozen legacy equal-background benchmark did not reproduce')
    contract = {
        'schema': 'new_V2_Native_legacy_V28_score_contract_1',
        'primary_goal': {'benchmark': 'Native_inner_six_roles', 'metric': 'replicate_overall', 'target_exclusive': 0.35,
                         'baseline': expected_inner, 'count_seed': SEED, 'model_seed': 20261002, 'teacher_train_seed': SEED,
                         'aggregation': 'Equal six metrics within role; mean two roles within each destination; mean three destinations'},
        'outer_guard': {'baseline': expected_outer, 'model_seed': 20261002, 'count_seeds': [20260928, 20260929, 20260930],
                        'teacher_train_seed': SEED, 'report_separately': True,
                        'aggregation': 'Mean three count seeds within each destination; mean three destinations'},
        'repeat_guard': {'model_seed': 20261009, 'count_seed': SEED, 'teacher_train_seed': 20261005,
                         'replicate_overall': float(repeat.mean()), 'report_separately': True},
        'scorer': {'package': 'cell_eval2', 'version': '0.16.0', 'preset': 'vcc2026', 'device': 'cpu',
                   'pert_col': 'target_gene', 'de_backend': 'scanpy',
                   'scale': 'Historical per-panel generic-baseline to five-split-replicate entries; _replicate_entries then score_one',
                   'scale_anchors': 'Frozen saved Native panels; no refitting baseline/anchors',
                   'metrics': list(METRICS), 'metric_weight': 1 / 6,
                   'fixed_scale_secondary': 'low-random_high-1_v10; not the user target',
                   'source': str(SOURCE / 'score_residual_v6.py'), 'source_sha256': sha(SOURCE / 'score_residual_v6.py')},
        'data_boundary': {'heldout_destination_background': True, 'inner_target_fold': 1, 'outer_target_fold': 0,
                          'target_fold_definition': 'sha256(vcc-v4:+target) mod 5',
                          'fit_preprocessing': 'Training backgrounds/folds only; destination NTC allowed input',
                          'destination_perturbation_counts_for_fitting': False,
                          'benchmark_status': 'Repeatedly used development benchmark; proper outer reported separately',
                          'H1_Overall_comparable_to_primary': False, 'official_score_queries': 0},
        'cases': cases,
    }
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'baseline_replay.csv', index=False)
    write(out / 'score_contract.json', contract)
    value = {'completed': True, 'actual_native_aggregate_replays': len(frame), 'max_abs_replay_error': mismatch,
             'inner_overall': float(inner.mean()), 'inner_per_background': inner.to_dict(),
             'outer_main_overall': float(outer.mean()), 'outer_per_background': outer.to_dict(),
             'repeat_outer_overall': float(repeat.mean()), 'new_counts': 0, 'new_judges': 0, 'new_training': 0,
             'contract_sha256': sha(out / 'score_contract.json')}
    write(out / 'complete.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


@dataclass
class TargetResult:
    target: str
    block: object
    diagnostics: dict
    moments: dict


def generator_functions():
    """Original V28 scientific generation bodies, without old neural imports."""
    namespace = {'np': np, 'pd': pd, 'sparse': sparse, 'Path': Path, 'hashlib': hashlib,
                 'PSEUDOCOUNT': 0.5, 'MAX_MULTIPLIER': 16.0,
                 'ROUND_NAMESPACE': 1, 'NB_NAMESPACE': 3, 'FLOAT32_INTEGER_MAX': 2**24, 'ARMS': ('H0', 'N1')}
    extract(SOURCE / 'generate_counts.py', ('q', 'transform_counts', 'multiplier_from_effects'), namespace)
    # The original source gene set intentionally remains a shared immutable input.
    namespace['lru_cache'] = __import__('functools').lru_cache
    extract(SOURCE / 'residual_v6_common.py', ('positions', 'count_ratio', 'source_gene_set'), namespace)
    extract(SOURCE / 'latent_counts_v26.py', ('GenerationValidationError', 'array_sha', 'csr_sha', 'integer_sample', 'quantiles', '_corr'), namespace)
    namespace['base'] = SimpleNamespace(**{name: namespace[name] for name in ('integer_sample', 'quantiles', '_corr')}, TargetResult=TargetResult)
    extract(SOURCE / 'literal_nb_v28.py', ('build_target',), namespace, class_name='PanelGenerator')
    return namespace


class CompactPanelGenerator:
    """Construct only the arrays needed by the original V28 build_target body."""
    def __init__(self, job, effects_path=None, theta_scale=1.0):
        self.job = job
        self.panel = relocated(job['historical_panel'])
        self.info = read(self.panel / 'panel.json')
        self.targets = list(map(str, self.info['targets']))
        self.destination = self.info['context']
        self.genes = np.loadtxt(self.panel / 'genes.txt', dtype=str, ndmin=1)
        self.output_genes = np.loadtxt(ARCHIVE / 'runs/context_v4/gene_axis.txt', dtype=str, ndmin=1)
        self.pool = sparse.load_npz(self.panel / 'generation_pool.npz').tocsr()
        self.mean = np.asarray(self.pool.mean(0)).ravel().astype(np.float64)
        path = Path(effects_path) if effects_path else relocated(job['effect'])
        with np.load(path, allow_pickle=False) as z:
            if 'genes' in z and 'targets' in z:
                if not np.array_equal(z['genes'].astype(str), self.output_genes):
                    raise ValueError('New effects must use exact original 10,916-gene axis')
                names = z['targets'].astype(str).tolist()
            else:
                # Original V13 files predate explicit axes. Preserve the exact
                # old load_teacher_effect identity/reconstruction rule.
                if path.resolve() != relocated(job['effect']) or not {'effect', 'residual'}.issubset(z.files):
                    raise ValueError('Only the exact matched old V13 artifact can inherit old axes')
                names = self.targets
                prior_namespace = {'np': np, 'json': json, 'Path': relocated}
                extract(SOURCE / 'residual_v6_common.py', ('frozen_v1',), prior_namespace)
                prior = prior_namespace['frozen_v1'](self.targets, self.output_genes)
                if not np.array_equal(z['effect'].astype(np.float32), prior + z['residual'].astype(np.float32)):
                    raise ValueError('Legacy V13 full effect differs from its exact frozen prior plus saved residual')
            if set(names) != set(self.targets) or len(names) != len(set(names)):
                raise ValueError('Effect target identity differs from Native panel')
            keys = [key for key in ('effect', 'effects', 'prediction') if key in z]
            if not keys:
                raise ValueError('Missing total raw log2 effect array')
            arr = z[keys[0]]
            for key in keys[1:]:
                np.testing.assert_array_equal(z[key], arr)
            self.effects = np.asarray(arr[[names.index(target) for target in self.targets]], dtype=np.float32)
        if self.effects.shape != (len(self.targets), len(self.output_genes)) or not np.isfinite(self.effects).all():
            raise ValueError('Invalid total raw log2 effects')
        self.order = list(map(str, self.info.get('generation_targets', self.targets)))
        self.sizes = self.info.get('generation_target_cells', self.info['target_cells'])
        self.count_seed = int(job['count_seed'])
        self.model_seed = int(job['model_seed'])
        self.role = job['model_role']
        rng = np.random.default_rng(self.count_seed + 100)
        self.donors = {target: rng.choice(self.pool.shape[0], self.sizes[target], replace=False) for target in self.order}
        with np.load(PROJECT / 'data/processed/set_response_v22/encoder.npz', allow_pickle=False) as z:
            self.input_genes = z['genes'].astype(str)
        native = {gene: i for i, gene in enumerate(self.genes)}
        model = {gene: i for i, gene in enumerate(self.output_genes)}
        self.input_native = np.asarray([native[gene] for gene in self.input_genes], dtype=np.int64)
        self.input_output = np.asarray([model[gene] for gene in self.input_genes], dtype=np.int64)
        theta_path = ARCHIVE / f'retained/native_theta/{self.role}/seed_{self.model_seed}/dispersion_seen.npz'
        with np.load(theta_path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['gene_symbols'].astype(str), self.output_genes)
            self.theta = z['theta'].astype(np.float64) * float(theta_scale)
            seen = z['seen'].astype(np.float32)
        manifest = read(theta_path.parent / 'original_decoder_manifest.json')
        if self.destination in manifest['decoder_contexts']:
            raise ValueError('Native destination was used by compact theta fit')
        if not np.isfinite(self.theta).all() or (self.theta <= 0).any():
            raise ValueError('Invalid compact theta or theta multiplier')
        own_namespace = {'np': np}
        extract(SOURCE / 'prepare_response_inputs_v23.py', ('own_for',), own_namespace)
        self.inputs = {'seen': seen, 'ownmask': own_namespace['own_for'](self.targets, self.output_genes)}
        self._scientific = generator_functions()
        self.source_identity = {'mean_path': str(path), 'mean_sha256': sha(path), 'compact_theta_path': str(theta_path),
                                'compact_theta_sha256': sha(theta_path), 'theta_scale': float(theta_scale),
                                'original_build_target_source_sha256': sha(SOURCE / 'literal_nb_v28.py')}

    def build_target(self, target, arm='N1'):
        return self._scientific['build_target'](self, target, arm)


def prepare_case(job, out):
    """Immutable links for large inputs; copied editable reference/cache metadata."""
    out = owned(out)
    source = relocated(job['historical_panel'])
    out.mkdir(parents=True, exist_ok=True)
    for leaf in ('real.h5ad', 'generation_pool.npz', 'genes.txt', 'panel.json'):
        destination = out / leaf
        if destination.exists():
            if sha(destination) != sha(source / leaf):
                raise AssertionError(f'New case input drift: {destination}')
        else:
            os.link(source / leaf, destination)
    refs = out / 'references'
    if not refs.exists():
        shutil.copytree(source / 'references', refs)
    return out


def generate_case(job, out, name='V28_replay', effects_path=None, theta_scale=1.0, target_limit=None):
    """Actual counts; root owns scheduling. No judge truth read by generator."""
    out = prepare_case(job, out)
    prediction = out / f'pred__{name}.h5ad'
    if prediction.exists():
        raise FileExistsError(f'Preserve existing new counts: {prediction}')
    import anndata as ad
    ad.settings.allow_write_nullable_strings = True
    os.chdir(PROJECT)
    generator = CompactPanelGenerator(job, effects_path, theta_scale)
    base_case = effects_path is None and theta_scale == 1.0
    original = relocated(job['score_output']) / 'generation/targets'
    targets = generator.targets if target_limit is None else generator.targets[:target_limit]
    blocks, labels, records = [], [], []
    for target in targets:
        result = generator.build_target(target)
        if base_case:
            saved = read(original / f'{target}.json')
            for key in ('count_block_sha256', 'count_ratio_native_sha256', 'literal_B_editable_sha256',
                        'donor_pool_rows_sha256', 'protected_actual_count_block_sha256'):
                if result.diagnostics[key] != saved[key]:
                    raise AssertionError(f'Exact original V28 count replay differs: {target} {key}')
        blocks.append(result.block)
        labels.extend([target] * result.block.shape[0])
        records.append(result.diagnostics)
        print('NATIVE_GENERATED', job['role'], target, result.block.shape[0], flush=True)
        del result
    if target_limit is not None:
        # A target-limited replay verifies sampling only; it cannot be scored as full panel.
        target_path = out / f'{name}_target_replay.json'
        write(target_path, {'exact_original_count_replay': base_case, 'targets': records,
                            'new_full_count_matrix_written': False, 'source_identity': generator.source_identity})
        return target_path
    namespace = {'np': np, 'sparse': sparse, 'ad': ad, 'pd': pd}
    extract(SOURCE / 'benchmark_context_v4.py', ('save',), namespace)
    namespace['save'](prediction, blocks, labels, generator.genes)
    manifest = {'completed': True, 'job_key': job['key'], 'source_identity': generator.source_identity,
                'exact_original_count_replay': base_case, 'prediction_path': str(prediction),
                'prediction_sha256': sha(prediction), 'bytes': prediction.stat().st_size, 'targets': records,
                'original_teacher_effect_used': effects_path is None,
                'supplied_new_effect_file': effects_path is not None,
                'optimizer_updates_by_generator': 0,
                'destination_perturbation_truth_read_by_generator': False}
    write(out / f'{name}_generation.json', manifest)
    return prediction


def score_case(out, names, threads=2):
    """Run unchanged archived score_path with explicit relocated axes."""
    import anndata as ad
    from cell_eval2 import EvalConfig, compute_metrics, aggregate_metrics
    cell_eval2, replicate_entries, score_one, scales = engine()
    ad.settings.allow_write_nullable_strings = True
    out = owned(out)
    os.environ['VCC_SCORE_THREADS'] = str(threads)
    modeled = np.loadtxt(ARCHIVE / 'runs/context_v4/gene_axis.txt', dtype=str, ndmin=1)
    core = np.loadtxt(ARCHIVE / 'runs/context_v4/normalization_core.txt', dtype=str, ndmin=1)
    namespace = {'Path': Path, 'np': np, 'pd': pd, 'pl': pl, 'ad': ad, 'json': json,
                 'os': os, 'shutil': shutil, 'replace': replace, 'asdict': asdict,
                 'EvalConfig': EvalConfig, 'compute_metrics': compute_metrics,
                 'aggregate_metrics': aggregate_metrics, '__version__': cell_eval2.__version__,
                 '_replicate_entries': replicate_entries, 'score_one': score_one,
                 'SCALES': scales, 'SCALE': 'low-random_high-1_v10', 'METRICS': list(METRICS),
                 'SEED': SEED, 'sha': sha, 'dump': write, 'axes': lambda: (modeled, core)}
    extract(SOURCE / 'diagnose_v4.py', ('decomposition',), namespace)
    extract(SOURCE / 'score_residual_v6.py', ('cfg_for', 'raw_values', 'score_path'), namespace)
    namespace['score_path'](out, names, False)
    return {name: read(out / f'scores/{name}/summary.json') for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('replay', 'generate', 'score', 'target-replay'))
    parser.add_argument('--job-key')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--name', default='V28_replay')
    parser.add_argument('--effects', type=Path)
    parser.add_argument('--theta-scale', type=float, default=1.0)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.mode == 'replay':
        replay(args.output or OUTPUT / 'replay')
    elif args.mode in ('generate', 'target-replay'):
        job = find_job(args.job_key)
        destination = args.output or OUTPUT / 'counts' / job['key']
        result = generate_case(job, destination, args.name, args.effects, args.theta_scale,
                               target_limit=1 if args.mode == 'target-replay' else None)
        print(str(result), flush=True)
    else:
        if args.output is None:
            parser.error('--output is required for scoring an existing new Native case')
        print(json.dumps(score_case(args.output, [args.name], args.threads), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
