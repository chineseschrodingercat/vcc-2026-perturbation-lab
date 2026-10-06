"""Inference-only TRUE formal V28 adapter; no count, scorer, fit, or truth reads.

The scientific AST bodies and K-only preprocessing loop are identical to the
successful retained ABC mean replay. Call ``predict(targets, native_genes, pool)``
with a destination NTC matrix. The original final checkpoint has trained on the
Native contexts: results there are descriptive, not whole-background holdout.
"""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
REPLAY_ROOT = PROJECT / 'staging/reproduce_official_V28_before_pause'
BINDING_PATH = REPLAY_ROOT / 'BINDING_METADATA_revision2.json'
REPLAY_PATH = REPLAY_ROOT / 'run_replay_revision2.py'
EXPECTED_BINDING_SHA256 = '46d30caf798bc52d31198b3a5228417ecc25b77640fc7883f788ed4cbf7f67a8'
EXPECTED_STATE_SHA256 = 'c90af4836a62b44ca9903ddf24a0456089d911a6c07202842e7d05c713a09466'
DEPENDENCIES = ('context_code', 'input_code', 'model_code', 'ratio_code',
                'model_axis', 'normalization_axis', 'prior_config', 'features',
                'checkpoint', 'K_public_prior', 'STRING')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def node(path, name):
    """Copied verbatim from successful run_replay_revision2.py."""
    found = [n for n in ast.parse(path.read_text(encoding='utf8')).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    assert len(found) == 1
    return copy.deepcopy(found[0])


def array_sha(value):
    arr = np.asarray(value)
    if arr.dtype.hasobject:
        raise TypeError('Object arrays cannot be receipt-bound')
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(json.dumps(list(arr.shape)).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


class FormalMeanAdapter:
    """Frozen_v1 plus eligible tiny600 residual on the original 10,916 axis."""

    def __init__(self):
        import h5py
        import torch
        from torch import nn

        self.torch = torch
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        binding_sha = sha(BINDING_PATH)
        if binding_sha != EXPECTED_BINDING_SHA256:
            raise ValueError('Successful replay binding bytes changed')
        binding = json.loads(BINDING_PATH.read_text(encoding='utf8'))
        if binding['schema'] != 'V28_ABC900_mean_replay_metadata1':
            raise ValueError('Unexpected original replay binding schema')
        if sha(REPLAY_PATH) != binding['runner_sha256']:
            raise ValueError('Successful original replay runner bytes changed')
        paths = {key: Path(binding['assets'][key]['path']) for key in DEPENDENCIES}
        for key, path in paths.items():
            if sha(path) != binding['assets'][key]['sha256']:
                raise ValueError('Original dependency bytes changed: ' + key)
        self.genes = np.loadtxt(paths['model_axis'], dtype=str, ndmin=1)
        self.core = np.loadtxt(paths['normalization_axis'], dtype=str, ndmin=1)
        if len(self.genes) != 10916 or len(set(self.genes)) != len(self.genes):
            raise ValueError('Invalid original model gene axis')
        if len(set(self.core)) != len(self.core) or not set(self.core).issubset(self.genes):
            raise ValueError('Invalid original normalization axis')
        with np.load(paths['K_public_prior'], allow_pickle=False) as z:
            source = {k: np.asarray(z[k]) for k in ('target_names', 'gene_names',
                        'target_mean_counts', 'control_mean_counts', 'target_n_cells')}
        # Literal original K transform, as in the validated mean replay.
        load = node(paths['context_code'], 'load_data')
        body = [n for n in load.body if isinstance(n, ast.For)]
        if len(body) != 1 or ast.unparse(body[0].iter) != 'data.items()':
            raise ValueError('Original K preprocessing loop identity changed')
        ns = dict(np=np, TS=50000., data={'k562': source}, genes=self.genes,
                  core=self.core, index={g: i for i, g in enumerate(self.genes)})
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                     '<literal_K_only_load_data_loop>', 'exec'), ns)
        if set(ns['data']) != {'k562'}:
            raise ValueError('Unexpected source context added')
        self.source = source
        with np.load(paths['features'], allow_pickle=False) as z:
            self.params = {k: z[k] for k in z.files}
        with torch.inference_mode():
            ck = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
        if (ck['ng'], ck['kind'], ck['step'], ck['seed']) != (10916, 'tiny_transformer', 600, 20260928):
            raise ValueError('Checkpoint identity differs from TRUE formal V28')
        aliases = {
            'data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz': paths['K_public_prior'],
            'data/raw/external/slim_embeddings/gene_string_embeddings.v0.3.h5': paths['STRING'],
            'configs/baseline_v1.json': paths['prior_config'],
        }
        def resolve(p):
            key = str(p).replace('\\', '/')
            if key not in aliases:
                raise ValueError('Unbound path requested by scientific AST: ' + key)
            return aliases[key]
        class NumpyProxy:
            def __getattr__(self, k):
                return getattr(np, k)
            def load(self, p, *args, **kwargs):
                kwargs['allow_pickle'] = False
                return np.load(resolve(p), *args, **kwargs)
        ns.update(np=NumpyProxy(),
                  h5py=SimpleNamespace(File=lambda p, *a, **kw: h5py.File(resolve(p), *a, **kw)),
                  Path=resolve, json=json, torch=torch, nn=nn, math=math)
        extracted = [node(paths['context_code'], 'target_features'),
                     node(paths['ratio_code'], 'frozen_v1'),
                     node(paths['input_code'], 'inputs_for'),
                     node(paths['model_code'], 'JointResidual')]
        exec(compile(ast.fix_missing_locations(ast.Module(body=extracted, type_ignores=[])),
                     '<literal_V28_mean_only>', 'exec'), ns)
        self.model = ns['JointResidual'](10916, 'tiny_transformer')
        self.model.load_state_dict(ck['state_dict'], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.ns = ns
        self.state_sha = self._state_sha()
        if self.state_sha != EXPECTED_STATE_SHA256:
            raise ValueError('Model parameter hash differs from successful replay')
        training_rows = paths['features'].parent / 'training_rows.csv'
        complete = paths['features'].parent / 'tiny_transformer_complete.json'
        with training_rows.open(encoding='utf8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        training_contexts = sorted(set(row['context'] for row in rows))
        counts = {context: sum(row['context'] == context for row in rows) for context in training_contexts}
        if training_contexts != sorted(ck['train_contexts']):
            raise ValueError('Training row/checkpoint context identity mismatch')
        if any(int(row['fold']) < 2 for row in rows):
            raise ValueError('Final training rows include development target folds')
        dependency_receipt = {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}
        dependency_receipt.update(training_rows=dict(path=str(training_rows), sha256=sha(training_rows)),
                                  training_complete=dict(path=str(complete), sha256=sha(complete)),
                                  original_replay=dict(path=str(REPLAY_PATH), sha256=sha(REPLAY_PATH)),
                                  replay_binding=dict(path=str(BINDING_PATH), sha256=binding_sha))
        ast_hashes = {item.name: hashlib.sha256(ast.dump(item, include_attributes=False).encode()).hexdigest()
                      for item in extracted}
        ast_hashes['K_only_load_data_loop'] = hashlib.sha256(ast.dump(body[0], include_attributes=False).encode()).hexdigest()
        self.receipt = dict(schema='TRUE_formal_V28_mean_adapter_v1', adapter_sha256=sha(__file__),
                            dependencies=dependency_receipt, scientific_AST_sha256=ast_hashes,
                            parameter_sha256=self.state_sha, torch_version=torch.__version__,
                            checkpoint_kind=ck['kind'], checkpoint_step=ck['step'], checkpoint_seed=ck['seed'],
                            training_contexts=training_contexts, training_context_rows=counts,
                            training_target_folds=sorted(set(int(row['fold']) for row in rows)),
                            modeled_genes=len(self.genes), normalization_core_genes=len(self.core),
                            preprocessing='Original saved source PCA/center/scales/STRING statistics, never refit',
                            feature_unit='log1p(CP50K on fixed normalization core); context g=clip((NTC-source_ctrl)/3,-2,2)',
                            prior_unit='Raw UMI mean log2 ratio with pseudocount 0.5; original cross_context_scale and n/(n+50)',
                            residual_semantics='Eligible K target_n_cells>=30; bounded tiny600 residual; own target residual zero; unsupported residual zero',
                            native_validation_status='Descriptive only: hepg2/jurkat/rpe1 are checkpoint training backgrounds',
                            inference_reads='Public K prior, STRING, frozen checkpoint/features; destination NTC argument only',
                            truth_values_read=False, fit_count=0, count_draws=0, judges=0, official_access=0)

    def _state_sha(self):
        return hashlib.sha256(b''.join(x.detach().cpu().contiguous().numpy().tobytes()
                                      for x in self.model.state_dict().values())).hexdigest()

    def predict(self, targets, native_genes, pool):
        targets = np.asarray(list(targets), dtype=str)
        native_genes = np.asarray(list(native_genes), dtype=str)
        if targets.ndim != 1 or len(set(targets)) != len(targets):
            raise ValueError('Targets must have unique one-dimensional identities')
        if native_genes.ndim != 1 or len(set(native_genes)) != len(native_genes):
            raise ValueError('NTC gene axis must have unique one-dimensional identities')
        if pool.ndim != 2 or pool.shape[0] == 0 or pool.shape[1] != len(native_genes):
            raise ValueError('NTC matrix shape/axis mismatch')
        from scipy import sparse
        if sparse.issparse(pool):
            pool = pool.tocsr(copy=False)
            values = pool.data
            pool_receipt = dict(storage='csr', shape=list(pool.shape), data_sha256=array_sha(pool.data),
                                indices_sha256=array_sha(pool.indices), indptr_sha256=array_sha(pool.indptr))
        else:
            pool = np.asarray(pool)
            values = pool
            pool_receipt = dict(storage='dense', shape=list(pool.shape), array_sha256=array_sha(pool))
        if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.rint(values)).all():
            raise ValueError('Destination NTC must contain finite nonnegative integer counts')
        st = self.source['tmap']
        eligible = np.array([t in st and self.source['target_n_cells'][st[t]] >= 30 for t in targets], dtype=bool)
        base = self.ns['frozen_v1'](targets, self.genes)
        residual = np.zeros_like(base)
        if eligible.any():
            inputs = self.ns['inputs_for']({'k562': self.source}, self.genes, self.core,
                                           targets[eligible], native_genes, pool, self.params)
            with self.torch.no_grad():
                r, _, _ = self.model(**inputs)
            residual[eligible] = r.cpu().numpy()
        effect = base + residual
        if base.dtype != np.float32 or residual.dtype != np.float32 or not np.isfinite(effect).all():
            raise ValueError('Nonfinite or invalid mean output dtype')
        if np.max(np.abs(residual), initial=0) > .750001 or np.count_nonzero(residual[~eligible]):
            raise ValueError('Original formal residual eligibility/bound violated')
        if self._state_sha() != self.state_sha:
            raise ValueError('Frozen model parameters mutated during inference')
        receipt = dict(self.receipt, target_axis_sha256=array_sha(targets),
                       native_gene_axis_sha256=array_sha(native_genes), ntc_pool=pool_receipt,
                       eligible_targets=int(eligible.sum()), unsupported_targets=targets[~eligible].tolist(),
                       output_effect_sha256=array_sha(effect), output_residual_sha256=array_sha(residual),
                       parameters_after_sha256=self.state_sha, parameter_mutation=False)
        return dict(effect=effect, residual=residual, frozen_v1=base, genes=self.genes.copy(),
                    targets=targets, eligible_mask=eligible, receipt=receipt)


_DEFAULT_ADAPTER = None


def predict(targets, native_genes, pool):
    """Return dict(effect, residual, frozen_v1, genes, targets, eligible_mask, receipt)."""
    global _DEFAULT_ADAPTER
    if _DEFAULT_ADAPTER is None:
        _DEFAULT_ADAPTER = FormalMeanAdapter()
    return _DEFAULT_ADAPTER.predict(targets, native_genes, pool)
