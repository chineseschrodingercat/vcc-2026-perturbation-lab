"""Export the locked V13 deployment recipe without changing its count generator."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from prepare_submission_v13 import ROOT, read, lock
from residual_v6_common import SEED, axes, frozen_v1, count_ratio, fold, source_gene_set, dump
from generate_counts import (file_sha256, make_obs, init_h5ad, append_csr, transform_counts,
                             verify_h5ad, multiplier_from_effects)
from train_context_v4 import load_data, target_features
import train_residual_v13 as v13
import train_residual_v7 as v7


def inputs_for(data, genes, core, targets, native, pool, params):
    _, z, _, _ = target_features(data, targets, genes, params['basis'], params['center'],
                                params['scale'], params['string_mean'], params['string_std'])
    mu = np.asarray(pool.mean(0)).ravel().astype(np.float64)
    gm = {g: i for i, g in enumerate(native)}
    total = sum(mu[gm[g]] for g in core if g in gm)
    assert total > 0
    ctrl = np.zeros(len(genes), np.float32)
    for i, gene in enumerate(genes):
        if gene in gm:
            ctrl[i] = np.log1p(mu[gm[gene]] * 50000 / total)
    g = np.tile(np.clip((ctrl - params['source_ctrl']) / 3, -2, 2), (len(targets), 1)).astype(np.float32)
    h = (g @ params['basis'].T / np.sqrt(len(genes))).astype(np.float32)
    own = np.ones((len(targets), len(genes)), np.float32)
    gp = {g: i for i, g in enumerate(genes)}
    for i, target in enumerate(targets):
        if target in gp:
            own[i, gp[target]] = 0
    return {k: torch.as_tensor(v) for k, v in dict(z=z, g=g, h=h, ownmask=own, seen=params['seen']).items()}


def main():
    lock()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    manifest = read(ROOT / 'inference_manifest.json')
    origin = Path(manifest['features_origin'])
    assert file_sha256(origin / 'features.npz') == manifest['features_sha256']
    for expert in manifest['experts']:
        assert file_sha256(Path(expert['checkpoint'])) == expert['sha256']
        assert file_sha256(Path(expert['origin'])/'features.npz') == manifest['features_sha256']
    validation = Path('data/raw/vcc2026/validation/files')
    native = pd.read_csv(validation/'gene_names.csv').gene_name.astype(str).tolist()
    targets = pd.read_csv(validation/'pert_counts.csv').target_gene.astype(str).tolist()
    assert len(native) == len(set(native)) == 18533 and len(targets) == len(set(targets)) == 300
    data, genes, core = load_data()
    assert np.array_equal(genes, axes()[0]) and np.array_equal(core, axes()[1])
    assert len(genes) == 10916 and set(genes).issubset(native)
    with np.load(origin/'features.npz') as z:
        params = {k: z[k] for k in z.files}
    rows = pd.read_csv(origin/'training_rows.csv')
    assert all(fold(t) >= 2 and t not in set(data['h1']['target_names']) for t in rows.target)
    source = data['k562']
    eligible = [i for i, t in enumerate(targets) if t in source['tmap'] and source['target_n_cells'][source['tmap'][t]] >= 30]
    unsupported = [t for i, t in enumerate(targets) if i not in eligible]
    supported = [targets[i] for i in eligible]
    base = frozen_v1(targets, genes)
    # Compare deployment feature construction to the exact CV implementation.
    panel = Path('runs/residual_v13/inner_panels/hepg2')
    reference, _ = v7.predict_features(origin, panel)
    test_inputs = inputs_for(data, genes, core, read(panel/'panel.json')['targets'],
                            np.loadtxt(panel/'genes.txt', dtype=str),
                            sparse.load_npz(panel/'generation_pool.npz'), params)
    parity = {k: torch.equal(reference[k], test_inputs[k]) for k in reference}
    assert all(parity.values()), parity
    # Confirm the frozen branch agrees with the actual previously submitted V1.
    legacy = Path('runs/baseline_v1/challenge_effects/context_A__k562_calibrated_direct.npz')
    with np.load(legacy) as z:
        assert z['target_symbols'].astype(str).tolist() == targets
        gm = {g:i for i,g in enumerate(genes)}
        assert all(g in gm for g in z['gene_symbols'])
        v1_difference = float(np.max(np.abs(base[:,[gm[g] for g in z['gene_symbols']]]-z['effect_log2fc'])))
        assert v1_difference < 1e-6, v1_difference
    output = ROOT/f"v13_{manifest['strategy']}_prediction.h5ad"
    partial = ROOT/f"v13_{manifest['strategy']}_prediction.partial.h5ad"
    report_path = ROOT/'prediction_manifest.json'
    if output.exists():
        old = read(report_path)
        assert file_sha256(output) == old['output']['sha256']
        print('REUSE_VERIFIED_EXPORT', output, flush=True)
        return
    assert not partial.exists(), 'Inspect existing partial output before retrying.'
    assert shutil.disk_usage('.').free >= 75*2**30
    effect_dir = ROOT/'challenge_effects'
    effect_dir.mkdir(exist_ok=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        model=f"V13 deployment selector / {manifest['strategy']}",
        selected_strategy=manifest['strategy'],
        actual_experts=manifest['experts'],
        inference_manifest_sha256=file_sha256(ROOT/'inference_manifest.json'),
        exporter_sha256=file_sha256(Path(__file__)), official_submission_performed=False,
        count_generator='Unmodified count_ratio + transform_counts; V13 CV sampling convention',
        sampling_seed=SEED, cells_per_target=400, output_genes=18533, modeled_genes=len(genes),
        eligible_targets=len(eligible), frozen_V1_fallback_targets=unsupported,
        unsupported_targets_are_not_imputed_training_observations=True,
        feature_parity_with_CV=parity, frozen_V1_effect_max_error=v1_difference,
        training_contexts=sorted(rows.context.unique().tolist()), training_rows=len(rows),
        training_fold_counts={str(k):int(v) for k,v in rows.fold.value_counts().items()},
        heldout_score_for_this_full_refit=None, contexts={},
        input_sha256={p.name:file_sha256(p) for p in [validation/'gene_names.csv',validation/'pert_counts.csv']})
    init_h5ad(partial, make_obs(list('ABC'), targets), native)
    row_offset = nnz = 0
    with h5py.File(partial, 'r+') as h5:
        for context in 'ABC':
            cp = validation/f'context_{context}.h5ad'
            controls = ad.read_h5ad(cp)
            assert controls.shape == (18400,18533) and controls.var_names.tolist() == native
            assert set(controls.obs.target_gene.astype(str)) == {'non-targeting'}
            if 'context' in controls.obs:
                assert set(controls.obs.context.astype(str)) == {context}
            pool = controls.X
            assert sparse.isspmatrix_csr(pool)
            assert np.isfinite(pool.data).all() and (pool.data >= 0).all()
            assert np.equal(pool.data, np.rint(pool.data)).all()
            inputs = inputs_for(data, genes, core, supported, native, pool, params)
            residual = np.zeros_like(base)
            for expert in manifest['experts']:
                if expert['engine'] == 'v13':
                    prediction = v13.predict
                elif expert['engine'] == 'v7':
                    prediction = v7.predict
                else:
                    from train_residual_v10 import predict as prediction
                r, _ = prediction(expert['origin'], expert['kind'], expert['step'], inputs)
                residual[eligible] += (expert['weight'] * expert['scale']) * r
            assert np.isfinite(residual).all()
            assert np.count_nonzero(residual[[i for i in range(len(targets)) if i not in eligible]]) == 0
            effects = base + residual
            np.savez_compressed(effect_dir/f'context_{context}.npz', target_symbols=np.array(targets),
                                gene_symbols=genes, effect_log2fc=effects, residual=residual)
            mu = np.asarray(pool.mean(0)).ravel().astype(np.float64)
            rng = np.random.default_rng(SEED+100)
            samples = [rng.choice(pool.shape[0], 400, replace=False) for _ in targets]
            stats = []
            for i, target in enumerate(targets):
                ratio = count_ratio(effects[i], mu, native, genes)
                rr = np.random.default_rng(np.random.SeedSequence([SEED,i,0]))
                block = transform_counts(pool[samples[i]].tocsr(copy=True), ratio, rr)
                libs = np.asarray(block.sum(1)).ravel()
                assert (libs > 0).all() and (libs <= 1000000).all()
                assert np.equal(block.data, np.rint(block.data)).all()
                nnz = append_csr(h5['X'], block, row_offset, nnz)
                row_offset += 400
                assert nnz <= 4750000000
                stats.append(dict(target=target, cells=400, unique_controls=len(np.unique(samples[i])),
                    nnz=int(block.nnz), min_library=float(libs.min()), max_library=float(libs.max()),
                    genes_at_multiplier_cap=int((ratio>=16).sum()),
                    residual_RMS=float(np.sqrt(np.mean(residual[i]**2)))))
                if (i+1) % 50 == 0:
                    assert shutil.disk_usage('.').free >= 50*2**30
                    print('COUNT_EXPORT',context,i+1,'/300',row_offset,'cells',nnz,'entries',flush=True)
            pd.DataFrame(stats).to_csv(ROOT/f'count_audit_{context}.csv',index=False)
            report['contexts'][context] = dict(control_file=str(cp), control_sha256=file_sha256(cp),
                control_cells=pool.shape[0], prediction_cells=120000,
                effect_sha256=file_sha256(effect_dir/f'context_{context}.npz'),
                residual_RMS=float(np.sqrt(np.mean(residual**2))),
                min_library=min(s['min_library'] for s in stats), max_library=max(s['max_library'] for s in stats))
            del controls,pool,inputs
        h5.flush()
    assert row_offset == 360000
    checked = verify_h5ad(partial,(360000,18533),native)
    checked['sha256'] = file_sha256(partial)
    checked['path'] = str(output)
    report['output'] = checked
    # Destination was asserted absent above; preserve partial on all failures.
    partial.rename(output)
    dump(report_path,report)
    print('V13_EXPORT_PASS',json.dumps(checked),flush=True)


if __name__ == '__main__':
    main()
