"""Sequential, resumable Native search under the immutable V28 contract."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import pandas as pd
import native_reproducer as native

ROOT = native.PROJECT
OUT = ROOT / 'runs/V2/native_search'
CONTRACT = ROOT / 'configs/V2_NATIVE_SCORE_CONTRACT.json'
BASELINE = ROOT / 'staging/newV2_native_lock/replay/baseline_replay.csv'
ARM = {
    'theta_quarter': {'theta_scale': .25},
    'theta_four': {'theta_scale': 4.},
    'source_fixed': {'theta_scale': 1., 'bundle': 'fixed'},
    'source_module_r4': {'theta_scale': 1., 'bundle': 'module_fixed_r4'},
}

def collect(arms):
    records = []
    for job in native.jobs('inner'):
        for arm in arms:
            path = OUT / job['key'] / f'scores/{arm}/summary.json'
            if path.exists():
                records.append({'arm': arm, 'key': job['key'], 'context': job['context'],
                                **native.read(path)})
    frame = pd.DataFrame(records)
    if frame.empty:
        return
    frame.to_csv(OUT / 'completed_cases.csv', index=False)
    baseline = pd.read_csv(BASELINE).query("stage == 'inner'")
    merged = frame.merge(baseline[['key', 'replicate_overall']], on='key',
                         suffixes=('', '_V28'), validate='many_to_one')
    merged['delta_vs_V28'] = merged.replicate_overall - merged.replicate_overall_V28
    merged.to_csv(OUT / 'paired_results.csv', index=False)
    aggregate = []
    for arm, group in merged.groupby('arm', sort=False):
        complete = len(group) == 6 and group.context.value_counts().to_dict() == {
            'hepg2': 2, 'jurkat': 2, 'rpe1': 2}
        aggregate.append({'arm': arm, 'completed_roles': len(group),
            'complete_six_roles': complete,
            'overall': float(group.groupby('context').replicate_overall.mean().mean()) if complete else None,
            'delta_vs_V28': float(group.groupby('context').delta_vs_V28.mean().mean()) if complete else None,
            'per_background': group.groupby('context').replicate_overall.mean().to_dict(),
            'raw_metrics': group[list(native.METRICS)].mean().to_dict() if complete else None})
    native.write(OUT / 'progress.json', {'status': 'complete' if len(frame) == 6*len(arms) else 'running',
        'baseline': .25932824671778637, 'target_exclusive': .35,
        'contract_sha256': native.sha(CONTRACT), 'official_score_queries': 0,
        'aggregates': aggregate})
    print('SEARCH_PROGRESS', json.dumps(aggregate, ensure_ascii=False), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=tuple(ARM), default=list(ARM))
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    contract = native.read(CONTRACT)
    if native.sha(CONTRACT) != 'e489efd753fc0395c87c756113504317d3d144aa23bdc2f474c99f13ccfc3219':
        raise AssertionError('Frozen contract drift')
    native.write(OUT / 'search_plan.json', {'arms': args.arms, 'parameters': ARM,
        'contract_sha256': native.sha(CONTRACT), 'adaptation': 'Readout dispersion or train-only source mean; never reference anchors',
        'count_seed': 20260928, 'scorer_threads': args.threads, 'perturbation_truth_for_fit': False,
        'frozen_original_source_sha': contract['scorer']['source_sha256']})
    collect(args.arms)
    for job in native.jobs('inner'):
        case = OUT / job['key']
        for arm in args.arms:
            score = case / f'scores/{arm}/summary.json'
            if score.exists():
                continue
            params = ARM[arm]
            prediction = case / f'pred__{arm}.h5ad'
            manifest = case / f'{arm}_generation.json'
            if prediction.exists() and not manifest.exists():
                raise RuntimeError(f'Incomplete count artifact requires inspection: {prediction}')
            effects = None
            if 'bundle' in params:
                effects = ROOT / f"staging/newV2_module_gate/native_bundles/{params['bundle']}/{job['key']}.npz"
                if not effects.exists():
                    raise FileNotFoundError(effects)
            started = time.time()
            print('START_NATIVE_CASE', job['context'], job['model_role'], arm, flush=True)
            if not prediction.exists():
                native.generate_case(job, case, name=arm, effects_path=effects,
                                     theta_scale=params['theta_scale'])
            result = native.score_case(case, [arm], threads=args.threads)[arm]
            print('COMPLETE_NATIVE_CASE', arm, result['replicate_overall'],
                  'seconds', round(time.time()-started, 2), flush=True)
            collect(args.arms)
    collect(args.arms)

if __name__ == '__main__':
    main()
