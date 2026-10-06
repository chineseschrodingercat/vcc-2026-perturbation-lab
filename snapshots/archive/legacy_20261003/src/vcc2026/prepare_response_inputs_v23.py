"""CPU source-only preparation for local V23 response mixtures.

This module has no CUDA/runtime configuration side effects. Destination labels
are read only from the immutable allowed V6 training cohorts. Prediction reads
only panel target identities, gene axes and generation NTC counts. Large source
arrays are shared by an in-process LRU rather than reloaded for each prediction.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from residual_v6_common import axes, CONTEXTS, dump, fold
from train_residual_v6 import load_rows

ROOT = Path('data/processed/response_mix_v23')
SOURCE = Path('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz')
H1 = Path('data/processed/v2/external_pseudobulk/vcc2025_h1.npz')
STRING = Path('data/raw/external/slim_embeddings/gene_string_embeddings.v0.3.h5')
ORIGIN = Path('models/residual_v7')
SCHEMA = 'v23_source_only_cpu_inputs_1'
ROLES = [f'inner_train_{line}' for line in CONTEXTS] + [f'outer_{line}' for line in CONTEXTS]


@lru_cache(maxsize=32)
def _sha_cached(path, size, mtime_ns):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def sha_file(path):
    path = Path(path)
    stat = path.stat()
    return _sha_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def origin_for(trainname):
    assert trainname in ROLES, trainname
    return ORIGIN / trainname / 'seed_20260928'


def lines_for(trainname):
    assert trainname in ROLES, trainname
    if trainname.startswith('inner_train_'):
        return [trainname.removeprefix('inner_train_'), 'orion_hct116']
    heldout = trainname.removeprefix('outer_')
    return [line for line in CONTEXTS if line != heldout] + ['orion_hct116']


@lru_cache(maxsize=1)
def source_arrays():
    """Allowed K562 observations only; no destination pseudobulk reads."""
    with np.load(SOURCE, allow_pickle=False) as data:
        result = {k: data[k] for k in ['gene_names', 'target_names',
                  'target_mean_counts', 'control_mean_counts', 'target_n_cells']}
    result['gene_names'] = result['gene_names'].astype(str)
    result['target_names'] = result['target_names'].astype(str)
    result['tmap'] = {target: i for i, target in enumerate(result['target_names'])}
    result['gmap'] = {gene: i for i, gene in enumerate(result['gene_names'])}
    return result


@lru_cache(maxsize=1)
def source_data():
    """Exactly reconstruct train_context_v4.load_data's K562 effect and ctrl."""
    genes, core = axes()
    source = source_arrays()
    means = source['target_mean_counts']
    control = source['control_mean_counts']
    core_columns = np.asarray([source['gmap'][gene] for gene in core])
    total = means[:, core_columns].sum(1)
    ctotal = control[core_columns].sum()
    destinations = {gene: i for i, gene in enumerate(genes)}
    keep = np.asarray([j for j, gene in enumerate(source['gene_names']) if gene in destinations])
    dest = np.asarray([destinations[gene] for gene in source['gene_names'][keep]])
    ctrl = np.zeros(len(genes), dtype=np.float32)
    effect = np.zeros((len(source['target_names']), len(genes)), dtype=np.float32)
    ctrl[dest] = np.log1p(50000. * control[keep] / ctotal)
    effect[:, dest] = np.log1p(50000. * means[:, keep] / np.maximum(total[:, None], 1e-12)) - ctrl[dest]
    for i, target in enumerate(source['target_names']):
        if target in destinations:
            effect[i, destinations[target]] = 0
    result = {'effect': effect, 'ctrl': ctrl, 'target_n_cells': source['target_n_cells'],
              'tmap': source['tmap']}
    return {'k562': result}


@lru_cache(maxsize=1)
def excluded_h1():
    with np.load(H1, allow_pickle=False) as data:
        return frozenset(data['target_names'].astype(str))


@lru_cache(maxsize=1)
def _string_embeddings():
    # Only source target identities are needed. Read the embedding file once,
    # preserving the original all-zero convention for genuinely missing IDs.
    names = source_arrays()['target_names']
    vectors = np.zeros((len(names), 64), dtype=np.float32)
    found = np.zeros(len(names), dtype=bool)
    with h5py.File(STRING, 'r') as handle:
        for i, target in enumerate(names):
            if target in handle:
                vectors[i] = handle[target][()]
                found[i] = True
    return vectors, found


def _features_path(trainname, features_origin=None):
    if features_origin is None:
        return origin_for(trainname) / 'features.npz'
    path = Path(features_origin)
    return path if path.is_file() else path / 'features.npz'


@lru_cache(maxsize=12)
def _feature_arrays(path, size, mtime_ns):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def feature_arrays(trainname, features_origin=None):
    path = _features_path(trainname, features_origin)
    stat = path.stat()
    return _feature_arrays(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def target_inputs(targets, features):
    """Reproduce target_features using cached K562 arrays and STRING vectors."""
    targets = np.asarray(targets, dtype=str)
    source = source_data()['k562']
    indices = np.asarray([source['tmap'][target] for target in targets])
    n = source['target_n_cells'][indices]
    response = source['effect'][indices] * (n / (n + 50))[:, None]
    raw_embeddings, source_found = _string_embeddings()
    embeddings = raw_embeddings[indices].copy()
    found = source_found[indices]
    embeddings[found] = (embeddings[found] - features['string_mean']) / features['string_std']
    latent = ((response - features['center']) @ features['basis'].T) / features['scale']
    return np.c_[latent, embeddings, np.log1p(n) / 10,
                 np.sqrt(np.mean(response * response, axis=1))].astype(np.float32)


@lru_cache(maxsize=1)
def source_prior():
    """Frozen V1 effects for all source identities; arithmetic matches V6."""
    genes, _ = axes()
    source = source_arrays()
    cfg = json.loads(Path('configs/baseline_v1.json').read_text())
    out = np.zeros((len(source['target_names']), len(genes)), dtype=np.float32)
    dest = np.asarray([j for j, gene in enumerate(genes) if gene in source['gmap']])
    columns = np.asarray([source['gmap'][genes[j]] for j in dest])
    means = source['target_mean_counts']
    controls = source['control_mean_counts']
    n = source['target_n_cells']
    for i in np.flatnonzero(n >= 30):
        out[i, dest] = np.clip(cfg['cross_context_scale'] * (n[i] / (n[i] + 50)) *
                      np.log2((means[i, columns] + .5) / (controls[columns] + .5)), -3, 3)
    return out


def raw_for(targets):
    mapping = source_arrays()['tmap']
    prior = source_prior()
    result = np.zeros((len(targets), prior.shape[1]), dtype=np.float32)
    valid = np.asarray([i for i, target in enumerate(targets) if str(target) in mapping], dtype=np.int64)
    if len(valid):
        indices = np.asarray([mapping[str(targets[i])] for i in valid])
        result[valid] = prior[indices]
    return result


def own_for(targets, genes):
    own = np.ones((len(targets), len(genes)), dtype=np.float32)
    positions = {gene: i for i, gene in enumerate(genes)}
    for i, target in enumerate(targets):
        if target in positions:
            own[i, positions[target]] = 0
    return own


def _training_lock(trainname):
    origin = origin_for(trainname)
    return {'schema': SCHEMA, 'trainname': trainname, 'train_contexts': lines_for(trainname),
            'features_sha256': sha_file(origin / 'features.npz'),
            'rows_sha256': sha_file(origin / 'training_rows.csv'),
            'source_sha256': sha_file(SOURCE), 'STRING_sha256': sha_file(STRING),
            'H1_identity_file_sha256': sha_file(H1),
            'input_code_sha256': sha_file(Path(__file__)),
            'training_labels_sha256': {line: sha_file(Path('runs/residual_v6/training_data') / f'{line}.npz')
                                      for line in lines_for(trainname)},
            'model_gene_axis_sha256': sha_file('runs/context_v4/gene_axis.txt'),
            'normalization_core_sha256': sha_file('runs/context_v4/normalization_core.txt')}


def load_training_inputs(trainname):
    """Return (old V6 rows, CPU numpy inputs, genes, immutable V7 origin).

    Inputs have z[N,130], h[N,64], raw[N,G], own[N,G], seen[G]. Only z/h/seen
    and row identities are cached on disk; source-prior and own masks are cheap
    to regenerate. The large mu/y/c arrays are never copied into the new cache.
    """
    origin = origin_for(trainname)
    rows = load_rows(lines_for(trainname))
    old = pd.read_csv(origin / 'training_rows.csv')
    assert rows['target'].tolist() == old.target.tolist()
    assert rows['line'].tolist() == old.context.tolist()
    assert all(fold(target) >= 2 and target not in excluded_h1() for target in rows['target'])
    assert old.fold.tolist() == [fold(target) for target in rows['target']]
    genes, _ = axes()
    features = feature_arrays(trainname)
    assert np.array_equal(features['source_ctrl'], source_data()['k562']['ctrl'])
    destination = ROOT / trainname
    destination.mkdir(parents=True, exist_ok=True)
    cache_path, lock_path = destination / 'inputs.npz', destination / 'complete.json'
    lock = _training_lock(trainname)
    if cache_path.exists() and lock_path.exists():
        complete = json.loads(lock_path.read_text(encoding='utf-8'))
        assert complete['input_lock'] == lock, 'Frozen V23 input provenance changed'
        assert complete['cache_sha256'] == sha_file(cache_path)
        with np.load(cache_path, allow_pickle=False) as cached:
            assert cached['targets'].tolist() == rows['target'].tolist()
            assert cached['contexts'].tolist() == rows['line'].tolist()
            z, h, seen = cached['z'], cached['h'], cached['seen']
    else:
        assert not cache_path.exists() and not lock_path.exists(), 'Incomplete V23 feature cache; inspect before replacing'
        started = time.monotonic()
        z = target_inputs(rows['target'], features)
        g = np.clip((rows['c'] - features['source_ctrl']) / 3, -2, 2).astype(np.float32)
        h = (g @ features['basis'].T / np.sqrt(len(genes))).astype(np.float32)
        seen = features['seen'].astype(np.float32)
        np.savez_compressed(cache_path, z=z, h=h, seen=seen,
                            targets=rows['target'], contexts=rows['line'])
        dump(lock_path, {'input_lock': lock, 'rows': len(rows['target']), 'genes': len(genes),
             'all_fit_targets_fold2_3_4_and_H1_excluded': True,
             'no_heldout_destination_outcome_reads': True,
             'cached_arrays': {'z': list(z.shape), 'h': list(h.shape), 'seen': list(seen.shape)},
             'no_duplicate_mu_y_mask_outside_cache': True,
             'cache_sha256': sha_file(cache_path), 'seconds': time.monotonic() - started})
    inputs = {'z': z, 'h': h, 'raw': raw_for(rows['target']),
              'own': own_for(rows['target'], genes), 'seen': seen}
    assert all(isinstance(value, np.ndarray) and np.isfinite(value).all() for value in inputs.values())
    return rows, inputs, genes, origin


def predict_inputs(trainname, panel, features_origin=None):
    """Return (CPU numpy z/h/seen/ownmask, raw V1); read only generation NTCs.

    features_origin may be an exact features.npz or a model directory holding
    the frozen copied features. None selects the immutable V7 role origin.
    """
    panel = Path(panel)
    genes, core = axes()
    info = json.loads((panel / 'panel.json').read_text(encoding='utf-8'))
    targets = np.asarray(info['targets'], dtype=str)
    native = np.loadtxt(panel / 'genes.txt', dtype=str)
    pool = sparse.load_npz(panel / 'generation_pool.npz')
    mu = np.asarray(pool.mean(0)).ravel().astype(np.float64)
    features = feature_arrays(trainname, features_origin)
    assert np.array_equal(features['source_ctrl'], source_data()['k562']['ctrl'])
    z = target_inputs(targets, features)
    gm = {gene: i for i, gene in enumerate(native)}
    # Preserve the original Python sum order and float64 NTC mean semantics.
    total = sum(mu[gm[gene]] for gene in core if gene in gm)
    ctrl = np.zeros(len(genes), dtype=np.float32)
    for i, gene in enumerate(genes):
        if gene in gm:
            ctrl[i] = np.log1p(mu[gm[gene]] * 50000 / total)
    g = np.tile(np.clip((ctrl - features['source_ctrl']) / 3, -2, 2), (len(targets), 1)).astype(np.float32)
    h = (g @ features['basis'].T / np.sqrt(len(genes))).astype(np.float32)
    inputs = {'z': z, 'h': h, 'seen': features['seen'].astype(np.float32),
              'ownmask': own_for(targets, genes)}
    return inputs, raw_for(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainname', choices=ROLES)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    roles = ROLES if args.all else [args.trainname]
    if roles == [None]:
        parser.error('Use --all or --trainname')
    started = time.monotonic()
    for role in roles:
        rows, inputs, genes, origin = load_training_inputs(role)
        print('V23_CPU_INPUTS', role, len(rows['target']),
              {key: list(value.shape) for key, value in inputs.items()},
              'seconds', round(time.monotonic() - started, 2), flush=True)
        del rows, inputs


if __name__ == '__main__':
    main()
