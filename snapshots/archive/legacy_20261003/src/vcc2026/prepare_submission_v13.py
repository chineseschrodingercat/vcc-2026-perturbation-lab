"""Isolated V13 deployment refit and count export, explicitly authorized 2026-09-28.

Historical experiments are read-only. Deployment configuration uses only the six
existing inner background roles, never outer scores or the official leaderboard.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from residual_v6_common import CONTEXTS, SEED, axes, dump, frozen_v1, generate, sha, count_ratio
from benchmark_residual_v7 import clone_panel, link, reuse_score
import train_residual_v13 as train

ROOT = Path('runs/submission_v13_20260928')
MODEL = Path('models/submission_v13_20260928')
OLD = Path('runs/residual_v13')
SEEDS = [20260928, 20261012]
SCALES = [.5, 1., 1.25]
PRIORITY = ['tiny_ref', 'continued_ref', 'tuned_single', 'seed_bag', 'diverse_bag', 'anchor_bag']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def choose(rows, key):
    best = max(r['overall'] for r in rows)
    return min([r for r in rows if r['overall'] >= best - .005], key=key)


def lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    dest = ROOT / 'deployment_rule.json'
    if dest.exists():
        rule = read(dest)
        for path, digest in rule['inputs_sha256'].items():
            assert sha(path) == digest, path
        return rule
    table = pd.read_csv(OLD / 'screen_candidates.csv')
    table = table.groupby(['kind', 'step'], as_index=False).overall.mean()
    candidates = []
    for kind in train.KINDS:
        candidates.append(choose(table[table.kind == kind].to_dict('records'), lambda r: r['step']))
    shortlist = []
    for _ in range(2):
        cfg = choose(candidates, lambda r: train.KINDS.index(r['kind']))
        shortlist.append(cfg)
        candidates.remove(cfg)
    inputs = [Path(__file__), train.PROTOCOL, Path('src/vcc2026/train_residual_v13.py'),
              Path('src/vcc2026/train_residual_v7.py'), Path('src/vcc2026/residual_v6_common.py'),
              Path('src/vcc2026/generate_counts.py'), OLD / 'screen_candidates.csv',
              OLD / 'screen_jobs.json', OLD / 'stability_jobs.json', OLD / 'selection.json']
    rule = {
        'locked_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User explicitly requested generation and official upload of the recommended V13 candidate.',
        'scope': 'One V13 submission; no V17 submission in this operation.',
        'shortlist': shortlist,
        'selection': 'Pool six whole-background inner roles equally. Reuse original V13 checkpoint/tie rules, top2 recipes, two seeds, scales0.5/1/1.25, single/seed_bag/diverse_bag and continued anchor. Fill only missing inner fits/scores. Select before full refit and upload. No outer/official scores used.',
        'priority': PRIORITY, 'training_seeds': SEEDS,
        'fit_contexts': CONTEXTS + ['orion_hct116'],
        'fit_targets': 'Immutable V6 folds2/3/4; all H1 identities excluded, including during full deployment refit.',
        'output_genes': 18533, 'residual_gene_axis': 10916,
        'unsupported_target_rule': 'Keep frozen V1; disable residual if target absent from K562 or has fewer than30 source cells. Do not invent a source response.',
        'generator': 'Unchanged residual_v6_common.count_ratio and transform_counts;400 NTC cells without replacement per target/context, original panel sampling convention, seed20260928.',
        'local_cv_overall_is_not_a_score_for_full_refit': True,
        'inputs_sha256': {str(p): sha(p) for p in inputs},
    }
    dump(dest, rule)
    print('LOCKED_GLOBAL_INNER_SHORTLIST', json.dumps(shortlist), flush=True)
    return rule


def inner_origin(line, seed, cfg):
    old = Path('models/residual_v13') / f'inner_train_{line}' / f'seed_{seed}'
    filename = f"{cfg['kind']}_{int(cfg['step'])}.pt"
    if (old / filename).exists():
        return old
    train.MODEL = MODEL / 'inner_completion'
    name = f'inner_train_{line}'
    origin = Path('models/residual_v7') / name / f'seed_{SEED}'
    train.fit(name, [line, 'orion_hct116'], origin, seed, [cfg['kind']])
    return train.MODEL / name / f'seed_{seed}'


def score_value(job):
    return read(Path(job['path']) / 'scores' / job['tag'] / 'summary.json')['replicate_overall']


def record_job(path, tag, residual, base, meta, source=None):
    pred = path / f'pred__{tag}.h5ad'
    effect_file = path / f'effects__{tag}.npz'
    if source:
        src = Path(source['path'])
        with np.load(src / f"effects__{source['tag']}.npz") as z:
            same = np.array_equal(z['effect'], base + residual)
        if same:
            link(src / f"pred__{source['tag']}.h5ad", pred)
            link(src / f"effects__{source['tag']}.npz", effect_file)
            reuse_score(src / 'scores' / source['tag'], path / 'scores' / tag, tag)
    if not pred.exists():
        generate(path, tag, base + residual, SEED)
    if not effect_file.exists():
        np.savez_compressed(effect_file, residual=residual, effect=base + residual)
    else:
        with np.load(effect_file) as z:
            assert np.array_equal(z['effect'], base + residual)
    return {'path': str(path), 'tag': tag, **meta}


def inner():
    rule = lock()
    train.init_runtime()
    existing = read(OLD / 'stability_jobs.json')
    screens = read(OLD / 'screen_jobs.json')
    jobs = []
    for line in CONTEXTS:
        clone_panel(OLD / 'inner_panels' / line, ROOT / 'inner_panels' / line)
    for fitline in CONTEXTS:
        origins = {(slot, seed): inner_origin(fitline, seed, cfg)
                   for slot, cfg in enumerate(rule['shortlist']) for seed in SEEDS}
        left = [c for c in CONTEXTS if c != fitline]
        for outer, context in [left, left[::-1]]:
            path = ROOT / 'inner_panels' / context
            rr = {}
            for slot, cfg in enumerate(rule['shortlist']):
                for seed in SEEDS:
                    origin = origins[slot, seed]
                    inputs, base = train.predict_features(origin, path)
                    rr[slot, seed] = train.predict(origin, cfg['kind'], int(cfg['step']), inputs)[0]
                    for scale in SCALES:
                        tag = f'o_{outer}__single_q{slot}_a{int(100*scale)}_t{seed}'
                        src = next((j for j in existing if j['outer'] == outer and j['context'] == context
                                    and j['strategy'] == 'tuned_single' and j['kind'] == cfg['kind']
                                    and j['step'] == int(cfg['step']) and j['scale'] == scale
                                    and j['training_seed'] == seed), None)
                        if src is None and seed == SEED and scale == 1:
                            src = next(j for j in screens if j['outer'] == outer and j['context'] == context
                                       and j['kind'] == cfg['kind'] and j['step'] == int(cfg['step']))
                        meta = dict(outer=outer, context=context, strategy='tuned_single', slot=slot,
                                    kind=cfg['kind'], step=int(cfg['step']), scale=scale, training_seed=seed)
                        jobs.append(record_job(path, tag, rr[slot, seed] * scale, base, meta, src))
                for scale in SCALES:
                    tag = f'o_{outer}__seed_bag_q{slot}_a{int(100*scale)}'
                    src = next((j for j in existing if j['outer'] == outer and j['context'] == context
                                and j['strategy'] == 'seed_bag' and j['kind'] == cfg['kind']
                                and j['step'] == int(cfg['step']) and j['scale'] == scale), None)
                    r = (rr[slot, SEEDS[0]] + rr[slot, SEEDS[1]]) * .5 * scale
                    meta = dict(outer=outer, context=context, strategy='seed_bag', slot=slot,
                                kind=cfg['kind'], step=int(cfg['step']), scale=scale)
                    jobs.append(record_job(path, tag, r, base, meta, src))
            for scale in SCALES:
                tag = f'o_{outer}__diverse_bag_a{int(100*scale)}'
                meta = dict(outer=outer, context=context, strategy='diverse_bag', slot=0,
                            kind='two_recipes', step=0, scale=scale)
                jobs.append(record_job(path, tag, sum(rr.values()) * .25 * scale, base, meta))
            dump(ROOT / 'inner_partial_jobs.json', jobs)
            print('INNER_DEPLOYMENT_READY', outer, context, len(jobs), flush=True)
    dump(ROOT / 'inner_jobs.json', jobs)


def score(stage):
    lock()
    jobs = read(ROOT / f'{stage}_jobs.json')
    def work(context):
        path = ROOT / 'inner_panels' / context
        names = list(dict.fromkeys(j['tag'] for j in jobs if j['context'] == context
                                  and not (path / 'scores' / j['tag'] / 'summary.json').exists()))
        if not names:
            return
        env = os.environ.copy()
        for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VCC_SCORE_THREADS']:
            env[name] = '2'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with (ROOT / 'logs' / f'score_{stage}_{context}.log').open('a', encoding='utf-8') as out:
            subprocess.run(['envs/vcc313/Scripts/python.exe', '-B', 'src/vcc2026/score_residual_v6.py',
                            '--path', str(path), '--models', *names], stdout=out,
                           stderr=subprocess.STDOUT, env=env, check=True)
        print('INNER_DEPLOYMENT_SCORED', stage, context, len(names), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(work, CONTEXTS))


def stable_selection():
    jobs = read(ROOT / 'inner_jobs.json')
    choices = []
    for strategy in ['tuned_single', 'seed_bag', 'diverse_bag']:
        options = []
        for slot in ([0] if strategy == 'diverse_bag' else [0, 1]):
            for scale in SCALES:
                js = [j for j in jobs if j['strategy'] == strategy and j['slot'] == slot and j['scale'] == scale]
                assert len(js) == (12 if strategy == 'tuned_single' else 6)
                cfg = {k: js[0][k] for k in ['strategy', 'slot', 'scale', 'kind', 'step']}
                options.append({**cfg, 'overall': float(np.mean([score_value(j) for j in js]))})
        choices.append(choose(options, lambda r: (r['scale'], r['step'], r['slot'])))
        pd.DataFrame(options).to_csv(ROOT / f'global_{strategy}_candidates.csv', index=False)
    return choices


def anchor():
    lock()
    stable = stable_selection()
    dump(ROOT / 'global_stability_selection.json', stable)
    picked = next(r for r in stable if r['strategy'] == 'seed_bag')
    jobs = read(ROOT / 'inner_jobs.json')
    chosen = [j for j in jobs if j['strategy'] == 'seed_bag' and j['slot'] == picked['slot'] and j['scale'] == picked['scale']]
    output = []
    for j in chosen:
        path = Path(j['path'])
        with np.load(path / f"effects__{j['tag']}.npz") as z:
            bag = z['residual'].copy()
            base = z['effect'] - z['residual']
        # Reconstruct the original frozen branch directly, avoiding cancellation.
        base = frozen_v1(read(path / 'panel.json')['targets'], axes()[0])
        src = OLD / 'inner_panels' / j['context']
        with np.load(src / f"effects__outer_{j['outer']}__continued_ref_s0_a100.npz") as z:
            ref = z['residual'].copy()
        for beta in [.25, .5, .75]:
            tag = f"o_{j['outer']}__anchor_b{int(beta*100)}"
            output.append(record_job(path, tag, (1-beta)*ref + beta*bag, base,
                                     dict(outer=j['outer'], context=j['context'], strategy='anchor_bag', beta=beta)))
    dump(ROOT / 'anchor_jobs.json', output)


def select():
    rule = lock()
    dest = ROOT / 'final_selection.json'
    if dest.exists():
        print('REUSE_FINAL_SELECTION', json.dumps(read(dest)['selected']), flush=True)
        return
    choices = read(ROOT / 'global_stability_selection.json')
    jobs = read(ROOT / 'anchor_jobs.json')
    opts = []
    for beta in [.25, .5, .75]:
        js = [j for j in jobs if j['beta'] == beta]
        assert len(js) == 6
        opts.append(dict(strategy='anchor_bag', beta=beta, overall=float(np.mean([score_value(j) for j in js]))))
    choices.append(choose(opts, lambda r: r['beta']))
    for ref in PRIORITY[:2]:
        values = [read(OLD / 'inner_panels' / context / 'scores' / f'outer_{outer}__{ref}_s0_a100' / 'summary.json')['replicate_overall']
                  for context in CONTEXTS for outer in CONTEXTS if context != outer]
        choices.append(dict(strategy=ref, overall=float(np.mean(values))))
    selected = choose(choices, lambda r: PRIORITY.index(r['strategy']))
    dump(dest, {'locked_utc': datetime.now(timezone.utc).isoformat(), 'selected': selected,
                'strategies': choices, 'shortlist': rule['shortlist'], 'uses_outer_scores': False,
                'uses_official_scores': False, 'deployment_rule_sha256': sha(ROOT / 'deployment_rule.json')})
    print('FINAL_V13_SELECTION', json.dumps(selected), flush=True)


def refit():
    rule = lock()
    selection = read(ROOT / 'final_selection.json')
    selected = selection['selected']
    train.init_runtime()
    origin = Path('models/residual_v7/final_candidate') / f'seed_{SEED}'
    strategy = selected['strategy']
    selected_bag = next(c for c in selection['strategies'] if c['strategy'] == 'seed_bag')
    cfg = selected_bag if strategy == 'anchor_bag' else selected
    experts = []
    if strategy not in PRIORITY[:2]:
        slots = [0, 1] if strategy == 'diverse_bag' else [cfg['slot']]
        seeds = [SEED] if strategy == 'tuned_single' else SEEDS
        train.MODEL = MODEL / 'full_refit'
        for seed in seeds:
            kinds = [rule['shortlist'][slot]['kind'] for slot in slots]
            train.fit('final_candidate', rule['fit_contexts'], origin, seed, kinds)
            for slot in slots:
                recipe = rule['shortlist'][slot]
                ck = train.MODEL / 'final_candidate' / f'seed_{seed}' / f"{recipe['kind']}_{int(recipe['step'])}.pt"
                weight = (selected['beta'] if strategy == 'anchor_bag' else 1.) / (len(slots)*len(seeds))
                experts.append(dict(engine='v13', origin=str(ck.parent), checkpoint=str(ck), sha256=sha(ck),
                                    kind=recipe['kind'], step=int(recipe['step']), scale=cfg['scale'], weight=weight))
    if strategy in ['continued_ref', 'anchor_bag']:
        import train_residual_v10 as continuation
        from benchmark_residual_v10 import choose as old_choose
        table = pd.read_csv('runs/residual_v10/inner_selection_candidates.csv')
        table = table[table.kind == 'continued'].groupby(['kind','step','scale'], as_index=False).overall.mean()
        table['beta'] = 0.
        oldcfg = old_choose(table.to_dict('records'))
        continuation.MODEL = MODEL / 'full_continuation'
        continuation.fit('final_candidate', rule['fit_contexts'], origin, 600, SEED, ['continued'])
        ck = continuation.MODEL / 'final_candidate' / f'seed_{SEED}' / f"continued_{oldcfg['step']}.pt"
        experts.append(dict(engine='v10', origin=str(ck.parent), checkpoint=str(ck), sha256=sha(ck),
                            kind='continued', step=oldcfg['step'], scale=oldcfg['scale'],
                            weight=1-selected['beta'] if strategy == 'anchor_bag' else 1.))
    if strategy == 'tiny_ref':
        ck = origin / 'tiny_transformer_600.pt'
        experts.append(dict(engine='v7', origin=str(origin), checkpoint=str(ck), sha256=sha(ck),
                            kind='tiny_transformer', step=600, scale=1., weight=1.))
    assert abs(sum(e['weight'] for e in experts)-1) < 1e-8
    dump(ROOT / 'inference_manifest.json', {'strategy': strategy, 'experts': experts,
         'features_origin': str(origin), 'features_sha256': sha(origin/'features.npz'),
         'training_rows_sha256': sha(origin/'training_rows.csv'),
         'formula': 'frozen_V1 + weighted scaled residuals, applied once before count generation',
         'selection_sha256': sha(ROOT/'final_selection.json'), 'heldout_score_for_full_refit': None})
    print('FULL_V13_REFIT_COMPLETE', strategy, len(experts), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['lock','inner','score_inner','anchor','score_anchor','select','refit'])
    stage = parser.parse_args().stage
    if stage.startswith('score_'):
        score(stage.removeprefix('score_'))
    else:
        globals()[stage]()


if __name__ == '__main__':
    main()
