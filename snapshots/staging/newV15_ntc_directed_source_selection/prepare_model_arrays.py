"""Prepare legal F23 model arrays; no model/optimizer/query truth/count/judge.

Whole-background exclusion is enforced BEFORE value-file IO. Sparse response
contexts retain availability masks and never shrink the shared target union.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import argparse
import hashlib
import json
import h5py
import numpy as np
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path('C:/00 - Competition')
HERE = Path(__file__).resolve().parent
GATE_PATH = HERE / 'SOURCE_ROLE_GATE_revision2.json'
AXES = ROOT / 'staging/newV12_supervised_label_selection_revision3_identity_fixed_common6157/shared_axes.npz'
EMBEDDING = ROOT / 'data/raw/external/slim_embeddings/gene_string_embeddings.v0.3.h5'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def gate_context(gate, role, context):
    roleinfo = next(x for x in gate['roles'] if x['role'] == role)
    source = next(x for x in gate['source_contexts'] if x['context_id'] == context)
    if source['family'] in roleinfo['blacklisted_response_families_before_IO'] or context not in roleinfo['allowed_response_contexts']:
        raise ValueError('Forbidden whole-background response bank before IO: ' + context)
    return roleinfo, source, Path(source['response_bank_path'])


def metadata(gate, role, context, axes):
    _, source, path = gate_context(gate, role, context)
    complete = json.loads((path / 'LABELS_COMPLETE.json').read_text(encoding='utf-8'))
    if not complete['frozen'] or complete['source_id'] != context:
        raise ValueError('Unfrozen or wrong source bank')
    if source['source_complete_SHA256'] is not None and sha(path / 'LABELS_COMPLETE.json') != source['source_complete_SHA256']:
        raise ValueError('Frozen source-bank complete drift')
    m = load_npz(path / 'metadata.npz')
    for sourcekey, axiskey in [('model_genes', 'model_genes'), ('model_exact_ENSG', 'model_exact_ENSG')]:
        np.testing.assert_array_equal(m[sourcekey], axes[axiskey])
    if len(set(m['target_exact_ENSG'].astype(str))) != len(m['target_exact_ENSG']):
        raise ValueError('Ambiguous target identity')
    # These banks already restrict target slots to frozen external or Native F23.
    # New HT29 banks must explicitly certify the same metadata-only F23 boundary.
    if (context.startswith('HT29_') or context == 'K562_IFNB') and complete.get('F23_only') is not True:
        raise ValueError('New sparse bank must certify F23_only before response IO')
    return m, {'context': context, 'family': source['family'], 'path': str(path),
               'complete_SHA256': sha(path / 'LABELS_COMPLETE.json'),
               'metadata_SHA256': sha(path / 'metadata.npz'),
               'complete': complete}


def cp(value):
    value = np.asarray(value, np.float64)
    total = value.sum(-1, keepdims=True)
    if not np.isfinite(value).all() or (value < 0).any() or (total <= 0).any():
        raise ValueError('Nonpositive or invalid observed common RNA library')
    return value * 50000 / total


def physical_prior(T, C, n, valid, own):
    known = np.isfinite(T) & np.isfinite(C) & (T >= 0) & (C >= 0)
    out = np.zeros(T.shape, np.float64)
    ratio = np.zeros(T.shape, np.float64)
    ratio[known] = np.log2((T[known] + .5) / (C[known] + .5))
    out = np.clip(ratio, -3, 3) * n[:, None] / (n[:, None] + 50)
    out *= known & valid[:, None] & own.astype(bool)
    return out.astype(np.float32)


def read_values(gate, role, context, ids, m, cm):
    _, _, path = gate_context(gate, role, context)  # indispensable before EVERY numeric file open
    n, ng = len(ids), len(m['model_genes'])
    lookup = {str(g): i for i, g in enumerate(m['target_exact_ENSG'])}
    rows = np.asarray([lookup.get(str(g), -1) for g in ids])
    present = rows >= 0
    valid = present.copy()
    valid[present] &= m['available'][rows[present]].astype(bool) & m['eligible_n30'][rows[present]].astype(bool)
    support = np.zeros(n, np.float64); support[present] = m['target_n'][rows[present]]
    valid &= support >= 30
    known = np.zeros((n, ng), bool); known[present] = m['known'][rows[present]].astype(bool)
    T = np.full((n, ng), np.nan, np.float64); C = T.copy()
    fullT = np.zeros(n, np.float64); fullC = fullT.copy()
    selected = np.flatnonzero(valid)
    for name, dest in [('raw_target_model_F64.npy', T), ('raw_matchedNTC_model_F64.npy', C),
                       ('fullTlib_F64.npy', fullT), ('fullClib_F64.npy', fullC)]:
        gate_context(gate, role, context)
        complete = json.loads((path / 'LABELS_COMPLETE.json').read_text(encoding='utf-8'))
        if sha(path / name) != complete['files_SHA'][name]:
            raise ValueError('Numeric source bank hash drift: ' + context + '/' + name)
        arr = np.load(path / name, mmap_mode='r', allow_pickle=False)
        dest[selected] = arr[rows[selected]]
    kk = known & valid[:, None]
    if not np.isfinite(T[kk]).all() or not np.isfinite(C[kk]).all() or (T[kk] < 0).any() or (C[kk] < 0).any():
        raise ValueError('Invalid physically known raw means: ' + context)
    if not np.isfinite(fullT[selected]).all() or not np.isfinite(fullC[selected]).all() or (fullT[selected] <= 0).any() or (fullC[selected] <= 0).any():
        raise ValueError('Invalid full provided RNA library: ' + context)
    commonT = np.nan_to_num(T[:, cm], nan=0); commonC = np.nan_to_num(C[:, cm], nan=0)
    valid &= (commonT.sum(1) > 0) & (commonC.sum(1) > 0)
    p = float(m['thinning_probability']); assert 0 < p <= 1
    mu = np.nan_to_num(C * p, nan=0).astype(np.float32)
    outside = np.maximum(fullC * p - mu.sum(1, dtype=np.float64), 0).astype(np.float32)
    # Network-only fill0 carries the unknown mask; not an assertion of observed zero.
    y = np.zeros((n, ng), np.float32)
    chosen = np.flatnonzero(valid)
    y[chosen] = np.log1p(np.nan_to_num(T[chosen], nan=0) * 50000 / commonT[chosen].sum(1)[:, None]).astype(np.float32)
    mu[~valid] = 0; y[~valid] = 0; outside[~valid] = 0
    return dict(T=T, C=C, commonT=commonT, commonC=commonC, n=support, available=valid,
                known=known & valid[:, None], mu=mu, y=y, outside=outside, fullTlib=fullT, fullClib=fullC,
                thinning_probability=p)


def embeddings(targets):
    values = np.zeros((len(targets), 64), np.float32); found = np.zeros(len(targets), bool)
    with h5py.File(EMBEDDING, 'r') as f:
        for i, target in enumerate(targets):
            if str(target) in f:
                values[i] = f[str(target)][()]; found[i] = True
    return values, found


def prepare(role):
    threadpool_limits(2)
    gate = json.loads(GATE_PATH.read_text(encoding='utf-8'))
    info = next(r for r in gate['roles'] if r['role'] == role)
    if sha(gate['root_contract_path']) != gate['root_contract_SHA256']:
        raise ValueError('Root source scope drift before preparation')
    for path, wanted in gate['dependencies_SHA256'].items():
        if sha(path) != wanted:
            raise ValueError('Root scope dependency drift before preparation: '+path)
    out = HERE / 'prepared_revision2' / role
    if out.exists():
        raise ValueError('Existing prepare preserved; do not silently overwrite')
    axes = load_npz(AXES); genes = axes['model_genes'].astype(str); exact = axes['model_exact_ENSG'].astype(str)
    cm = axes['common_to_model']; common = axes['common6157_exact_ENSG'].astype(str)
    seen = axes['shared_editable'].astype(np.float32)
    assert len(genes) == 10916 and len(cm) == 6157 and seen.sum() == 6114
    contexts = sorted(info['allowed_response_contexts'])
    met, audit = {}, {}
    candidates, symbol = set(), {}
    for c in contexts:
        met[c], audit[c] = metadata(gate, role, c, axes)
        valid = met[c]['available'].astype(bool) & met[c]['eligible_n30'].astype(bool)
        candidates |= set(met[c]['target_exact_ENSG'][valid].astype(str))
        for id_, s_ in zip(met[c]['target_exact_ENSG'], met[c]['target_symbol']):
            if str(id_) and str(id_) not in symbol:
                symbol[str(id_)] = str(s_)
    ids = np.asarray(sorted(candidates), str); targets = np.asarray([symbol[i] for i in ids], str)
    if len(ids) < 96:
        raise ValueError('Legal source union smaller than fixed distinct batch96')
    own = np.ones((len(ids), len(genes)), np.float32)
    for i, id_ in enumerate(ids):
        own[i, exact == id_] = 0
    K = read_values(gate, role, 'K562', ids, met['K562'], cm)
    pos = np.flatnonzero(K['available'])
    x = np.zeros((len(ids), len(cm)), np.float64)
    x[pos] = np.log1p(cp(K['commonT'][pos])) - np.log1p(cp(K['commonC'][pos]))
    ci = {g: j for j, g in enumerate(common)}
    for i, id_ in enumerate(ids):
        if id_ in ci:
            x[i, ci[id_]] = 0
    x *= (K['n'] / (K['n'] + 50))[:, None]
    x = x.astype(np.float32)
    center = x[pos].mean(0); _, singular, basis = randomized_svd(x[pos] - center, n_components=64, n_iter=4, random_state=20260928)
    scale = np.maximum(((x[pos] - center) @ basis.T).std(0), .05)
    emb, found = embeddings(targets)
    if not found.any():
        raise ValueError('No legitimate STRING features')
    emean = emb[found].mean(0); estd = np.maximum(emb[found].std(0), .05); emb[found] = (emb[found] - emean) / estd
    z = np.concatenate(((x - center) @ basis.T / scale, emb, (np.log1p(K['n']) / 10)[:, None], np.sqrt(np.mean(x * x, axis=1))[:, None]), axis=1).astype(np.float32)
    z[~K['available'], :64] = 0
    prior = physical_prior(K['T'], K['C'], K['n'], K['available'], own)
    # For K-unsupported targets the lawful global publisher K control is still known.
    Kctrl = np.broadcast_to(np.log1p(cp(K['commonC'][pos[:1]])), x.shape).copy()
    Kctrl[pos] = np.log1p(cp(K['commonC'][pos]))
    fields = {k: [] for k in ('mu', 'y', 'outside', 'mask', 'g', 'h')}; available, ns, units = [], [], []
    for c in contexts:
        b = K if c == 'K562' else read_values(gate, role, c, ids, met[c], cm)
        small = np.zeros_like(x); chosen = np.flatnonzero(b['available'])
        small[chosen] = np.clip((np.log1p(cp(b['commonC'][chosen])) - Kctrl[chosen]) / 3, -2, 2).astype(np.float32)
        small *= b['known'][:, cm]
        g = np.zeros((len(ids), len(genes)), np.float32); g[:, cm] = small
        fields['g'].append(g); fields['h'].append((small @ basis.T / np.sqrt(len(genes))).astype(np.float32))
        for k in ('mu', 'y', 'outside'):
            fields[k].append(b[k])
        fields['mask'].append((b['known'] * own * seen).astype(np.float32))
        available.append(b['available']); ns.append(b['n'])
        units.append({'context': c, 'fullTlib': b['fullTlib'], 'fullClib': b['fullClib'],
                      'observed_common_genes': b['known'][:, cm].sum(1), 'thinning_probability': b['thinning_probability']})
    labels = {k: np.stack(v) for k, v in fields.items()}; labels['context_names'] = np.asarray(contexts)
    inputs = dict(target_ids=ids, targets=targets, genes=genes, model_exact_ENSG=exact, common_to_model=cm,
                  z=z, ownmask=own, seen=seen, fixed_K562_raw=prior, K_input_source_n=K['n'],
                  K_input_available=K['available'], K_input_known=K['known'])
    transform = dict(basis=basis, center=center, scale=scale, singular=singular, string_mean=emean, string_std=estd,
                     common_exact_ENSG=common, common_to_model=cm, model_genes=genes, model_exact_ENSG=exact,
                     unique_fit_target_ids=ids[pos], string_fit_target_ids=ids[found])
    for obj in (inputs, labels, transform):
        for key, arr in obj.items():
            if np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
                raise ValueError('Nonfinite array ' + key)
    out.mkdir(parents=True)
    for name, obj in [('inputs.npz', inputs), ('labels.npz', labels), ('shared_transform.npz', transform)]:
        np.savez_compressed(out / name, **obj)
    np.savez_compressed(out / 'unit_metadata.npz', context_label_available=np.stack(available), source_n=np.stack(ns),
                        fullTlib=np.stack([u['fullTlib'] for u in units]), fullClib=np.stack([u['fullClib'] for u in units]),
                        observed_common_genes=np.stack([u['observed_common_genes'] for u in units]),
                        thinning_probability=np.asarray([u['thinning_probability'] for u in units]))
    files = ['inputs.npz', 'labels.npz', 'shared_transform.npz', 'unit_metadata.npz']
    report = dict(schema='V15_legal_F23_union_model_array_prepare1', status='MODEL_ARRAYS_ONLY_NO_QUERY_SCHEDULE_OR_MODEL',
                  role=role, receiver=info['receiver'], held_family=info['blacklisted_response_families_before_IO'][0],
                  context_names=contexts, target_union=len(ids), K_supported_training_targets=len(pos),
                  context_available_targets={c: int(v.sum()) for c, v in zip(contexts, available)},
                  source_role_gate_SHA256=sha(GATE_PATH), root_contract_SHA256=gate['root_contract_SHA256'],
                  axes_SHA256=sha(AXES), prepare_code_SHA256=sha(__file__), files_SHA256={f: sha(out / f) for f in files},
                  source_bank_metadata_audits={c: {k: v for k, v in audit[c].items() if k != 'complete'} for c in contexts},
                  physical_K_prior_alpha=1, old_fixed_prior_called=False, historical_effect_or_theta_read=False,
                  target_label_unit='common6157 CP50000 over physically observed common coordinates; missing explicit mask',
                  selector_training_NTC_read=False, held_response_values_read=False, source_F0_F1_parameters_fit=False,
                  actual_model_fit_infer_count_judge_calls=0, threadpool_runtime=threadpool_info())
    write(out / 'MODEL_ARRAYS_COMPLETE.json', report)
    (out/'target_union_ids.txt').write_text('\n'.join(ids)+'\n',encoding='utf-8')
    (out/'K_basis_fit_ids.txt').write_text('\n'.join(ids[pos])+'\n',encoding='utf-8')
    print(json.dumps({'role': role, 'targets': len(ids), 'contexts': len(contexts), 'complete_SHA256': sha(out / 'MODEL_ARRAYS_COMPLETE.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--role', choices=['whole_Jurkat', 'whole_HCT116', 'whole_HT29'], required=True)
    prepare(parser.parse_args().role)
