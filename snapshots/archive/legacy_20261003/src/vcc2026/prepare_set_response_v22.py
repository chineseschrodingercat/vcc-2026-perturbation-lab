"""Bounded local true-cell inputs for conditional set response V22.

Destination inputs are only the existing disjoint generation NTC pools. The
K562 encoder fits fold 2/3/4 source profiles plus source NTCs, excluding every
H1 identity; source fold 0/1 cells are covariates, never destination labels.
Cached cell subsamples and minibatch draws are technical views, not replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from foundation_io import field
from residual_v6_common import axes, dump, fold, SEED
from prepare_residual_v17 import cache as old_cache, SOURCES

ROOT = Path('data/processed/set_response_v22')
SOURCE_BULK = Path('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz')
H1_BULK = Path('data/processed/v2/external_pseudobulk/vcc2025_h1.npz')
LINES = ['hepg2', 'jurkat', 'rpe1', 'orion_hct116', 'k562']
SCHEMA = 'v22_true_sets_1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def _resource_check(root=ROOT, reserve_gib=75):
    free = shutil.disk_usage('.').free / 2**30
    size = sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file()) / 2**30
    if free < reserve_gib:
        raise RuntimeError(f'V22 requires C free >= {reserve_gib} GiB; got {free:.2f}')
    if size >= 3:
        raise RuntimeError(f'V22 cell cache exceeded 3 GiB: {size:.3f}')
    return {'free_GiB': free, 'cache_GiB': size}


def input_genes():
    """Structural intersection, with no destination perturbation outcome reads."""
    core = axes()[1]
    sets = [set(np.loadtxt(old_cache(line) / 'control_genes.txt', dtype=str)) for line in LINES]
    common = set.intersection(*sets)
    genes = np.asarray([gene for gene in core if gene in common], dtype=str)
    assert len(genes) == 6119
    return genes


def normalized(x):
    """Per-cell normalization over the 6119 actual common measured genes."""
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Nonfinite or negative input counts')
    total = x.sum(1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError('Zero common-axis library')
    return np.log1p(x * (50000 / total)).astype(np.float32)


def _control_counts(line, genes):
    path = old_cache(line)
    original = np.loadtxt(path / 'control_genes.txt', dtype=str)
    mapping = {g: i for i, g in enumerate(original)}
    columns = np.asarray([mapping[g] for g in genes])
    return sparse.load_npz(path / 'control_core_counts.npz')[:, columns]


def fit_encoder(root=ROOT, dimensions=64):
    """Fit source-only PCA on training source profiles and true source controls.

    Using mean profiles for the perturbation part makes this fit small and
    deterministic before the one-time raw-cell pass. It is not a foundation
    encoder and the fitted PCA is not claimed to use all cell variability.
    """
    root = Path(root)
    path = root / 'encoder.npz'
    if path.exists():
        with np.load(path) as data:
            encoder = {k: data[k] for k in data.files}
        assert encoder['basis'].shape[0] == dimensions
        return encoder
    genes = input_genes()
    with np.load(H1_BULK) as data:
        excluded = set(data['target_names'].astype(str))
    with np.load(SOURCE_BULK) as data:
        names = data['gene_names'].astype(str)
        mapping = {g: i for i, g in enumerate(names)}
        columns = np.asarray([mapping[g] for g in genes])
        targets = data['target_names'].astype(str)
        keep = (data['target_n_cells'] >= 30) & np.asarray([fold(t) >= 2 and t not in excluded for t in targets])
        training_targets = targets[keep]
        means = data['target_mean_counts'][keep][:, columns]
    controls = normalized(_control_counts('k562', genes))
    profiles = normalized(means)
    x = np.concatenate([controls, profiles])
    center = x.mean(0)
    scale = np.maximum(x.std(0), .1)
    standardized = (x - center) / scale
    with threadpool_limits(limits=2):
        _, singular, basis = randomized_svd(standardized, n_components=dimensions, n_iter=4, random_state=SEED)
    pc_scale = np.maximum(singular / np.sqrt(len(x)), .1).astype(np.float32)
    encoder = {'genes': genes, 'gene_center': center, 'gene_scale': scale,
               'basis': basis.astype(np.float32), 'pc_scale': pc_scale,
               'training_targets': training_targets}
    np.savez_compressed(path, **encoder)
    np.savetxt(root / 'input_genes.txt', genes, fmt='%s')
    np.savetxt(root / 'output_genes.txt', axes()[0], fmt='%s')
    dump(root / 'encoder_fit.json', {
        'source': 'K562', 'training_targets': len(training_targets),
        'training_folds': [2, 3, 4], 'all_H1_excluded': True,
        'source_controls': len(controls), 'source_mean_profiles': len(profiles),
        'input_genes': len(genes), 'PC_dimensions': dimensions,
        'normalization': 'Per cell/profile common measured gene sum -> 50000 -> log1p',
        'fit_rule': 'Source NTC cells plus K562 pseudobulk profiles of fold 2/3/4 non-H1 identities only; equal row weights, gene std floor0.1, randomized SVD4iterations. All destination inputs and heldout source identities transformed only.',
        'no_destination_outcomes_or_controls_in_fit': True,
        'input_axis_rule': 'Ordered structural intersection of actual measured existing NTC gene axes; no response-derived feature selection.',
        'missing_genes_excluded': ['C19orf48', 'PRRC2B', 'SOD2'],
        'source_bulk_sha256': sha_file(SOURCE_BULK),
        'source_controls_sha256': sha_file(old_cache('k562') / 'control_core_counts.npz'),
        'H1_identity_file_sha256': sha_file(H1_BULK),
    })
    print('V22_ENCODER', len(training_targets), len(genes), dimensions, flush=True)
    return encoder


def encode(x, encoder):
    h = normalized(x)
    with threadpool_limits(limits=2):
        pcs = ((h - encoder['gene_center']) / encoder['gene_scale']) @ encoder['basis'].T
    pcs /= encoder['pc_scale']
    if not np.isfinite(pcs).all():
        raise ValueError('Nonfinite V22 cell representation')
    return pcs.astype(np.float32)


def _native_obs(line):
    with h5py.File(SOURCES[line], 'r') as handle:
        obs = pd.DataFrame({'target': field(handle, 'obs/gene'),
                            'batch': field(handle, 'obs/gem_group'),
                            'guide': field(handle, 'obs/sgID_AB'),
                            'cell_id': field(handle, 'obs/cell_barcode')})
    return obs


def prepare_controls(encoder, source_obs=None, root=ROOT):
    root = Path(root)
    for line in LINES:
        destination = root / 'controls' / line
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / 'complete.json').exists():
            continue
        x = _control_counts(line, encoder['genes'])
        if line == 'orion_hct116':
            metadata = pd.read_csv(old_cache(line) / 'control_cells.csv', dtype=str)
            metadata['batch'] = metadata['sample']
            metadata['guide'] = metadata['guide_target']
            metadata['cell_id'] = metadata['cell_barcode']
            metadata['raw_source'] = 'Existing audited ORION true NTC cache'
            with np.load(old_cache(line) / 'metadata.npz') as old:
                np.savez_compressed(destination / 'target_batch_weights.npz',
                                    targets=old['targets'], batch_names=old['batch_names'],
                                    batch_weights=old['batch_weights'])
            audit_extra = {'target_control_matching': 'Sample NTC batch using immutable V17 target batch weights; then sample a true NTC within that batch.'}
        elif line == 'k562':
            original = pd.read_csv(old_cache(line) / 'control_cells.csv')
            if source_obs is None:
                source_obs = _native_obs('k562')
            metadata = source_obs.iloc[original['original_row'].to_numpy()].reset_index(drop=True)
            metadata['original_row'] = original['original_row'].to_numpy()
            assert metadata.cell_id.tolist() == original.cell_id.tolist()
            audit_extra = {'target_control_matching': 'Source NTC draws prefer same GEM batch as each sampled source perturbed cell.'}
        else:
            control_list = pd.read_csv(Path('runs/residual_v6/inner_panels') / line / 'controls.csv')
            original_rows = control_list.loc[control_list.role == 'generation', 'row'].to_numpy()
            reference_rows = control_list.loc[control_list.role == 'reference', 'row'].to_numpy()
            assert not set(original_rows) & set(reference_rows)
            obs = _native_obs(line)
            metadata = obs.iloc[original_rows].reset_index(drop=True)
            metadata['original_row'] = original_rows
            assert metadata.target.eq('non-targeting').all()
            audit_extra = {'reference_control_rows_disjoint': True,
                           'original_row_origin': str(Path('runs/residual_v6/inner_panels') / line / 'controls.csv')}
        assert len(metadata) == x.shape[0]
        metadata['cache_row'] = np.arange(len(metadata))
        metadata['role'] = 'input_NTC_only'
        metadata.to_csv(destination / 'cell_metadata.csv', index=False)
        pcs = encode(x, encoder)
        np.save(destination / 'cells.npy', pcs)
        np.save(destination / 'measurement_mask.npy', np.ones(len(encoder['genes']), dtype=np.uint8))
        dump(destination / 'complete.json', {'context': line, 'cells': len(pcs),
             'input_genes': len(encoder['genes']), 'PC_dimensions': pcs.shape[1],
             'source_counts': str(old_cache(line) / 'control_core_counts.npz'),
             'source_counts_sha256': sha_file(old_cache(line) / 'control_core_counts.npz'),
             'encoder_sha256': sha_file(root / 'encoder.npz'),
             'all_actual_input_genes_measured': True, 'no_destination_perturbed_cells_accessed': True,
             'sampling_unit': 'Technical cell subsamples within historical experiments',
             **audit_extra})
        print('V22_CONTROLS', line, len(pcs), flush=True)


def prepare_source(encoder, root=ROOT, cells_per_target=64, block_rows=4096, source_obs=None):
    root = Path(root)
    destination = root / 'source'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'complete.json').exists():
        return
    if source_obs is None:
        source_obs = _native_obs('k562')
    with np.load(H1_BULK) as data:
        excluded = set(data['target_names'].astype(str))
    with np.load(SOURCE_BULK) as data:
        eligible = set(data['target_names'][data['target_n_cells'] >= 30].astype(str)) - excluded
    selected_path = destination / 'selected_original_rows.npy'
    if selected_path.exists():
        selected = np.load(selected_path)
        targets = np.load(destination / 'targets.npy')
        offsets = np.load(destination / 'offsets.npy')
    else:
        grouped = pd.Series(np.arange(len(source_obs))).groupby(source_obs.target.to_numpy(), sort=False).indices
        targets = np.asarray(sorted(eligible & set(grouped)), dtype=str)
        selected_list, offsets_list = [], [0]
        rng = np.random.default_rng(SEED + 2200)
        target_metadata = []
        for target in targets:
            indices = grouped[target]
            n = min(cells_per_target, len(indices))
            # Unbiased within-target sample; original guide and batch are
            # retained rather than represented as independent repeats.
            rows = rng.choice(indices, n, replace=False)
            selected_list.extend(rows)
            offsets_list.append(offsets_list[-1] + n)
            target_metadata.append({'target': target, 'fold': fold(target),
                 'source_cells_total': len(indices), 'cached_cells': n,
                 'source_role': 'encoder_training_identity' if fold(target) >= 2 else 'heldout_identity_input_only'})
        selected = np.asarray(selected_list, dtype=np.int64)
        offsets = np.asarray(offsets_list, dtype=np.int64)
        np.save(selected_path, selected)
        np.save(destination / 'targets.npy', targets)
        np.save(destination / 'offsets.npy', offsets)
        pd.DataFrame(target_metadata).to_csv(destination / 'target_metadata.csv', index=False)
        metadata = source_obs.iloc[selected].reset_index(drop=True)
        metadata['original_row'] = selected
        metadata['cache_row'] = np.arange(len(selected))
        metadata['fold'] = metadata.target.map(fold)
        metadata['source_role'] = np.where(metadata['fold'] >= 2, 'encoder_training_identity', 'heldout_identity_input_only')
        metadata.to_csv(destination / 'cell_metadata.csv', index=False)
    assert not set(targets) & excluded
    shape = (len(selected), encoder['basis'].shape[0])
    cells_path = destination / 'cells.npy'
    cells = np.lib.format.open_memmap(cells_path, mode='r+' if cells_path.exists() else 'w+', dtype=np.float32, shape=shape)
    count_blocks = (len(source_obs) + block_rows - 1) // block_rows
    done_path = destination / 'completed_blocks.npy'
    done = np.load(done_path) if done_path.exists() else np.zeros(count_blocks, dtype=bool)
    assert done.shape == (count_blocks,)
    order = np.argsort(selected)
    sorted_rows = selected[order]
    start_time = time.monotonic()
    seen = 0
    with h5py.File(SOURCES['k562'], 'r') as handle:
        native = field(handle, 'var/gene_name')
        unique = ~pd.Index(native).duplicated(keep=False)
        mapping = {g: i for i, g in enumerate(native) if unique[i]}
        columns = np.asarray([mapping[g] for g in encoder['genes']])
        raw = handle['X']
        for block, lo in enumerate(range(0, len(source_obs), block_rows)):
            if done[block]:
                continue
            hi = min(len(source_obs), lo + block_rows)
            left, right = np.searchsorted(sorted_rows, [lo, hi])
            if right > left:
                # One bounded contiguous read: native H5AD X is dense,
                # uncompressed; millions of individual random rows are slow.
                x = np.asarray(raw[lo:hi, :], dtype=np.float32)
                x = x[sorted_rows[left:right] - lo][:, columns]
                if np.any(x != np.rint(x)):
                    raise ValueError('Source cells are not raw integer UMI counts')
                cells[order[left:right]] = encode(x, encoder)
                seen += right - left
            done[block] = True
            if block % 32 == 0 or block + 1 == count_blocks:
                cells.flush()
                np.save(done_path, done)
                _resource_check(root)
                print('V22_SOURCE', block + 1, count_blocks, 'transformed', seen,
                      'seconds', round(time.monotonic() - start_time, 1), flush=True)
    cells.flush()
    np.save(done_path, done)
    del cells
    check = np.load(cells_path, mmap_mode='r')
    assert np.isfinite(check).all() and np.all(done)
    measured = np.ones(len(encoder['genes']), dtype=np.uint8)
    np.save(destination / 'measurement_mask.npy', measured)
    coverage = {}
    for line in LINES:
        if line == 'k562':
            continue
        with np.load(Path('runs/residual_v6/training_data') / f'{line}.npz') as data:
            row_targets = data['targets'].astype(str)
        missing = sorted(set(row_targets) - set(targets))
        coverage[line] = {'training_rows': len(row_targets), 'supported': len(row_targets) - len(missing), 'missing_targets': missing}
    dump(destination / 'complete.json', {'targets': len(targets), 'cells': len(selected),
          'cells_per_target_cap': cells_per_target, 'input_genes': len(encoder['genes']),
          'PC_dimensions': shape[1], 'source': SOURCES['k562'],
          'source_bytes': Path(SOURCES['k562']).stat().st_size,
          'source_published_download_manifest': 'data/raw/external/replogle_2022/k562_gwps/singlecell_download_manifest.json',
          'encoder_sha256': sha_file(root / 'encoder.npz'),
          'selected_original_rows_sha256': sha_file(selected_path),
          'cells_sha256': sha_file(cells_path), 'all_H1_identities_excluded_from_cache': True,
          'fold0_1_are_source_covariates_only': True, 'all_V6_training_coverage': coverage,
          'sampling': 'Deterministic uniform within-target without replacement, max64 cells; batch and guide identities retained per cell; source NTC can be batch matched at draw time.',
          'no_destination_perturbed_cells_accessed': True,
          'seconds_last_resume_pass': time.monotonic() - start_time})
    print('V22_SOURCE_COMPLETE', len(targets), len(selected), coverage, flush=True)


def prepare(root=ROOT, dimensions=64, cells_per_target=64, block_rows=4096):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _resource_check(root)
    config = {'schema': SCHEMA, 'dimensions': dimensions, 'cells_per_target': cells_per_target,
              'block_rows': block_rows, 'seed': SEED, 'no_cloud_service': True}
    config_path = root / 'config.json'
    if config_path.exists():
        assert read(config_path) == config, 'Different V22 preparation settings require another cache root'
    else:
        dump(config_path, config)
    if (root / 'complete.json').exists():
        return load_cache(root)
    encoder = fit_encoder(root, dimensions)
    source_obs = _native_obs('k562')
    prepare_controls(encoder, source_obs, root)
    prepare_source(encoder, root, cells_per_target, block_rows, source_obs)
    files = [p for p in root.rglob('*') if p.is_file() and p.name != 'complete.json']
    dump(root / 'complete.json', {'schema': SCHEMA, 'config': config,
         'encoder_fit': read(root / 'encoder_fit.json'),
         'controls': {line: read(root / 'controls' / line / 'complete.json') for line in LINES},
         'source': read(root / 'source' / 'complete.json'),
         'resources': _resource_check(root),
         'files_sha256': {str(p.relative_to(root)): sha_file(p) for p in files},
         'training_targets_fold2_3_4_and_H1_excluded': True,
         'never_reads_destination_perturbed_cells': True})
    print('V22_PREPARE_COMPLETE', root, flush=True)
    return load_cache(root)


def load_cache(root=ROOT):
    """Load small metadata plus read-only mmap arrays; no raw H5AD access."""
    root = Path(root)
    if not (root / 'complete.json').exists():
        raise FileNotFoundError(f'V22 cache incomplete: {root / "complete.json"}')
    targets = np.load(root / 'source' / 'targets.npy')
    result = {'root': root, 'controls': {}, 'control_metadata': {}, 'control_batch_groups': {},
              'source_cells': np.load(root / 'source' / 'cells.npy', mmap_mode='r'),
              'source_targets': targets,
              'source_offsets': np.load(root / 'source' / 'offsets.npy'),
              'source_target_map': {target: i for i, target in enumerate(targets)},
              'source_metadata': pd.read_csv(root / 'source' / 'cell_metadata.csv', dtype={'batch': str, 'guide': str, 'target': str}),
              'input_genes': np.loadtxt(root / 'input_genes.txt', dtype=str),
              'output_genes': np.loadtxt(root / 'output_genes.txt', dtype=str),
              'measurement_mask': {}, 'control_target_batch_weights': {},
              'complete': read(root / 'complete.json')}
    for line in LINES:
        path = root / 'controls' / line
        result['controls'][line] = np.load(path / 'cells.npy', mmap_mode='r')
        metadata = pd.read_csv(path / 'cell_metadata.csv', dtype={'batch': str, 'guide': str, 'target': str})
        result['control_metadata'][line] = metadata
        result['control_batch_groups'][line] = {b: np.asarray(ids) for b, ids in metadata.groupby('batch', sort=False).groups.items()}
        result['measurement_mask'][line] = np.load(path / 'measurement_mask.npy')
        weights = path / 'target_batch_weights.npz'
        if weights.exists():
            with np.load(weights) as data:
                result['control_target_batch_weights'][line] = {
                    'batch_names': data['batch_names'].astype(str),
                    'weights': data['batch_weights'],
                    'target_map': {t: i for i, t in enumerate(data['targets'])}}
    result['source_batch'] = result['source_metadata'].batch.to_numpy(str)
    return result


_DEFAULT_CACHE = None


def sample_sets(targets, lines, rng, set_size=16, cache=None):
    """Return destination NTC, source NTC, source perturbed cells and validity.

    All three arrays have B x set_size x 64. Unsupported targets use the
    identical source NTC draw as perturbation (zero response), valid=False.
    The caller makes the matched mean ablation from these exact draws.
    """
    global _DEFAULT_CACHE
    if cache is None:
        if _DEFAULT_CACHE is None:
            _DEFAULT_CACHE = load_cache()
        cache = _DEFAULT_CACHE
    targets = np.asarray(targets, dtype=str)
    lines = np.repeat(lines, len(targets)) if isinstance(lines, str) else np.asarray(lines, dtype=str)
    if len(lines) != len(targets):
        raise ValueError('One destination line required per target')
    dimensions = cache['source_cells'].shape[1]
    dest = np.empty((len(targets), set_size, dimensions), dtype=np.float32)
    ntc = np.empty_like(dest)
    pert = np.empty_like(dest)
    valid = np.zeros(len(targets), dtype=bool)
    source_ntc = cache['controls']['k562']
    source_batch_groups = cache['control_batch_groups']['k562']
    for i, (target, line) in enumerate(zip(targets, lines)):
        control = cache['controls'][line]
        matched = cache['control_target_batch_weights'].get(line)
        if matched is not None and target in matched['target_map']:
            probabilities = matched['weights'][matched['target_map'][target]].astype(np.float64)
            probabilities /= probabilities.sum()
            batches = rng.choice(matched['batch_names'], size=set_size, p=probabilities)
            destination_ids = np.asarray([rng.choice(cache['control_batch_groups'][line][batch]) for batch in batches])
        else:
            destination_ids = rng.choice(len(control), set_size, replace=len(control) < set_size)
        dest[i] = control[destination_ids]
        source_id = cache['source_target_map'].get(target)
        if source_id is None:
            ids = rng.choice(len(source_ntc), set_size, replace=len(source_ntc) < set_size)
            ntc[i] = source_ntc[ids]
            pert[i] = ntc[i]
            continue
        start, stop = cache['source_offsets'][source_id:source_id + 2]
        pert_ids = rng.choice(np.arange(start, stop), set_size, replace=stop - start < set_size)
        pert[i] = cache['source_cells'][pert_ids]
        ntc_ids = []
        for batch in cache['source_batch'][pert_ids]:
            candidates = source_batch_groups.get(batch)
            ntc_ids.append(rng.choice(candidates) if candidates is not None else rng.integers(len(source_ntc)))
        ntc[i] = source_ntc[np.asarray(ntc_ids)]
        valid[i] = True
    return dest, ntc, pert, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dimensions', type=int, default=64)
    parser.add_argument('--cells-per-target', type=int, default=64)
    parser.add_argument('--block-rows', type=int, default=4096)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    if args.probe_only:
        genes = input_genes()
        print(json.dumps({'input_genes': len(genes), 'output_genes': len(axes()[0]),
                          'controls': {line: _control_counts(line, genes).shape for line in LINES},
                          'resources': _resource_check(args.root)}, indent=2))
    else:
        prepare(args.root, args.dimensions, args.cells_per_target, args.block_rows)


if __name__ == '__main__':
    main()
