"""V26 mean-matched direct NB cells from immutable V25 role checkpoints.

Only generation NTC counts, frozen source covariates, and trained weights are
read. No destination perturbation count, score, reference or real H5AD is read.
G0 calls the literal historical generator. Every new arm preserves its integer
protected block and matches its unrounded editable gene and donor totals by IPF.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from residual_v6_common import axes, count_ratio, dump, generate as old_generate
from generate_counts import transform_counts
from batch_cells_v24 import sample_sets
from prepare_set_response_v22 import load_cache, _control_counts
from prepare_residual_v17 import cache as old_control_cache
from prepare_response_inputs_v23 import predict_inputs
from latent_flow_v25 import LatentResponse, NBDecoder
from train_latent_flow_v25 import encoder_arrays, model_cfg, directory, P as V25

PROTOCOL = Path('configs/latent_counts_v26_protocol.json')
ARMS = ('G0', 'G1', 'G2', 'G3', 'G4')
ROUND_NAMESPACE = 1
NB_NAMESPACE = 3
MAX_SWEEPS = 500
IPF_TOLERANCE = 1e-6
FLOAT32_INTEGER_MAX = 2**24


class GenerationValidationError(ValueError):
    def __init__(self,message,*,failed_draw=None,rng=None,ipf_weights=None):
        super().__init__(message)
        self.failed_draw=failed_draw;self.rng=rng;self.ipf_weights=ipf_weights


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    x = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def csr_sha(value):
    x = value.tocsr(copy=True); x.sort_indices(); x.eliminate_zeros()
    h = hashlib.sha256()
    h.update(np.asarray(x.shape, np.int64).tobytes())
    for value in (x.indptr.astype(np.int64), x.indices.astype(np.int32), x.data.astype(np.float32)):
        h.update(value.tobytes())
    return h.hexdigest()


def _row_keys(matrix):
    x = matrix.tocsr(copy=True); x.sort_indices(); x.eliminate_zeros()
    if not np.all(np.isfinite(x.data)) or np.any(x.data < 0) or not np.array_equal(x.data, np.rint(x.data)):
        raise ValueError('Provenance matching requires actual nonnegative integer NTC counts')
    result = []
    for i in range(x.shape[0]):
        a, b = x.indptr[i:i+2]; h = hashlib.sha256()
        h.update(x.indices[a:b].astype(np.int32).tobytes())
        h.update(x.data[a:b].astype(np.int64).tobytes()); result.append(h.hexdigest())
    return result


def quantiles(value):
    x = np.asarray(value, np.float64)
    return {k: float(np.quantile(x, v)) for k, v in
            [('min',0), ('p01',.01), ('median',.5), ('p99',.99), ('max',1)]}


def _corr(left, right):
    a = np.asarray(left, np.float64); b = np.asarray(right, np.float64)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def ipf(weights, rows, columns, *, max_sweeps=MAX_SWEEPS, tolerance=IPF_TOLERANCE):
    """Fixed row/column scaling with zero marginals kept exactly zero.

    Targets are calculated solely from the unchanged generator on the same NTC
    donors. No positive pseudocount is added to decoded weights. Lack of support
    or failure to converge is an implementation failure, never a relaxed gate.
    """
    x = np.array(weights, dtype=np.float64, copy=True)
    r = np.asarray(rows, np.float64); c = np.asarray(columns, np.float64)
    if x.shape != (len(r), len(c)) or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Invalid IPF weights')
    if not np.isfinite(r).all() or not np.isfinite(c).all() or np.any(r < 0) or np.any(c < 0):
        raise ValueError('Invalid IPF target margins')
    total_error = abs(float(r.sum()-c.sum())) / max(1., float(r.sum()), float(c.sum()))
    if total_error > 1e-12:
        raise ValueError(f'IPF target totals incompatible: {total_error}')
    x[r == 0, :] = 0.; x[:, c == 0] = 0.
    if r.sum() == 0:
        return x, {'sweeps': 0, 'row_max_relative_residual': 0., 'column_max_relative_residual': 0.,
                   'converged': True, 'zero_rows': int((r==0).sum()), 'zero_columns': int((c==0).sum()),
                   'max_sweeps': int(max_sweeps), 'tolerance': float(tolerance)}
    for step in range(1, max_sweeps+1):
        rs = x.sum(1)
        if np.any((r > 0) & (rs <= 0)):
            raise ValueError('IPF has a positive row target without decoded support')
        x *= np.divide(r, rs, out=np.zeros_like(r), where=rs>0)[:, None]
        cs = x.sum(0)
        if np.any((c > 0) & (cs <= 0)):
            raise ValueError('IPF has a positive gene target without decoded support')
        x *= np.divide(c, cs, out=np.zeros_like(c), where=cs>0)[None, :]
        re = float(np.max(np.abs(x.sum(1)-r)/np.maximum(1., r), initial=0.))
        ce = float(np.max(np.abs(x.sum(0)-c)/np.maximum(1., c), initial=0.))
        if max(re, ce) <= tolerance:
            return x, {'sweeps': step, 'row_max_relative_residual': re, 'column_max_relative_residual': ce,
                       'converged': True, 'zero_rows': int((r==0).sum()), 'zero_columns': int((c==0).sum()),
                       'max_sweeps': int(max_sweeps), 'tolerance': float(tolerance)}
    raise GenerationValidationError(f'IPF failed after {max_sweeps} sweeps: row={re:.9g}, column={ce:.9g}',ipf_weights=x)


def integer_sample(mu, theta, count_seed, order_index, sampler):
    """True NB2 draws or unbiased stochastic round, with separate namespaces.

    G1/G2 have the same round stream and G3/G4 the same NB stream. NumPy NB uses
    rejection sampling, so the latter does NOT imply gene-wise shared uniforms.
    All zero means are exact zero. A tail violation is retained as an exception;
    values are not clipped and draws are not retried until they pass.
    """
    x = np.asarray(mu, np.float64)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Invalid expected means before integer sampling')
    namespace = ROUND_NAMESPACE if sampler == 'round' else NB_NAMESPACE
    rng = np.random.default_rng(np.random.SeedSequence([int(count_seed), int(order_index), 26, namespace]))
    before = rng.bit_generator.state
    if sampler == 'round':
        floor = np.floor(x); draw = floor + (rng.random(x.shape) < (x-floor))
    elif sampler == 'nb':
        th = np.asarray(theta, np.float64)
        if th.shape != (x.shape[1],) or np.any(~np.isfinite(th)) or np.any(th <= 0):
            raise ValueError('NB requires actual frozen positive gene-specific inverse dispersion')
        probability = th[None, :]/(th[None, :]+x)
        draw = rng.negative_binomial(th[None, :], probability)
    else:
        raise ValueError(sampler)
    if np.any(~np.isfinite(draw)) or np.any(draw < 0) or np.any(draw > FLOAT32_INTEGER_MAX):
        raise GenerationValidationError('Integer draw outside exact float32 range; do not clip or redraw',
            failed_draw=draw,rng={'rng_before':before,'rng_after':rng.bit_generator.state,
                                'seed_sequence_entropy':[int(count_seed),int(order_index),26,namespace]})
    draw = np.asarray(draw, np.float32)
    return draw, {'sampler':sampler, 'seed_sequence_entropy':[int(count_seed),int(order_index),26,namespace],
                  'rng_before':before, 'rng_after':rng.bit_generator.state,
                  'paired_gene_uniforms_claimed':sampler=='round',
                  'NB_rejection_sampler_gene_pairing_claimed':False}


def validate_protocol(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    assert value['primary_arm']=='G4'
    assert value['fixed_model']['candidate']=='flow_nb' and value['fixed_model']['checkpoint']==3200
    assert value['fixed_model']['new_full_fits']==0
    assert value['ipf']['max_iterations']==MAX_SWEEPS and value['ipf']['relative_tolerance']==IPF_TOLERANCE
    assert value['ipf']['dtype']=='float64'
    assert value['rng']['donor_seed_offset']==100 and value['rng']['protected_namespace']==[0]
    assert value['rng']['rounding_namespace']==[26,ROUND_NAMESPACE]
    assert value['rng']['nb_namespace']==[26,NB_NAMESPACE]
    assert value['count_guard']['maximum_exact_float32_integer']==FLOAT32_INTEGER_MAX
    assert value['count_guard']['maximum_row_library_exclusive']==1000000
    for key in ['views','cell_seed','view_stride','integration_steps']:
        assert value['inference'][key]==V25['inference'][key]
    assert value['inference']['set_size']==V25['model']['set_size']
    return value


@lru_cache(maxsize=1)
def source_cache():
    return load_cache()


@dataclass
class TargetResult:
    target: str
    block: sparse.csr_matrix
    diagnostics: dict
    moments: dict


class PanelGenerator:
    """One role/panel/weight/count-seed input state; all targets locked first."""

    def __init__(self, panel, effects, role, model_seed, count_seed, protocol_path=PROTOCOL, device='cuda'):
        self.panel = Path(panel); self.role = str(role); self.model_seed = int(model_seed)
        self.count_seed = int(count_seed); self.device = torch.device(device)
        self.protocol_path = Path(protocol_path)
        self.protocol=validate_protocol(self.protocol_path)
        self.info = json.loads((self.panel/'panel.json').read_text(encoding='utf-8'))
        self.targets = list(map(str, self.info['targets'])); self.destination = self.info['context']
        self.genes = np.loadtxt(self.panel/'genes.txt', dtype=str); self.output_genes = axes()[0]
        self.pool = sparse.load_npz(self.panel/'generation_pool.npz').tocsr()
        self.mean = np.asarray(self.pool.mean(0)).ravel().astype(np.float64)
        self.effects = np.asarray(effects, np.float32)
        if self.effects.shape != (len(self.targets),len(self.output_genes)) or not np.isfinite(self.effects).all():
            raise ValueError('V26 effect axes/shape differ from immutable V25 full total effects')
        if len(set(self.targets)) != len(self.targets) or len(set(self.genes)) != len(self.genes):
            raise ValueError('Duplicate target or native measured gene identity')
        self.order = list(map(str,self.info.get('generation_targets',self.targets)))
        self.sizes = self.info.get('generation_target_cells',self.info['target_cells'])
        rng = np.random.default_rng(self.count_seed+100)
        self.donor_rng_before = rng.bit_generator.state
        self.donors = {t:rng.choice(self.pool.shape[0],self.sizes[t],replace=False) for t in self.order}
        self.donor_rng_after = rng.bit_generator.state
        if not set(self.targets).issubset(self.donors):
            raise ValueError('Generation target ordering lacks an evaluated target')
        self.encoder = encoder_arrays(); self.input_genes = self.encoder['genes'].astype(str)
        native_map = {g:i for i,g in enumerate(self.genes)}; output_map = {g:i for i,g in enumerate(self.output_genes)}
        if not all(g in native_map for g in self.input_genes):
            raise ValueError('Actual native gene axis lacks a frozen common input gene')
        self.input_native = np.asarray([native_map[g] for g in self.input_genes],np.int64)
        self.input_output = np.asarray([output_map[g] for g in self.input_genes],np.int64)
        self.cache = source_cache(); self.pool_metadata = self._pool_metadata()
        self.dest = directory(self.role,'flow_nb',self.model_seed)
        self.decoder_dest = directory(self.role,'decoder',self.model_seed)
        ck = torch.load(self.dest/'checkpoint_3200.pt',map_location='cpu',weights_only=False)
        dck = torch.load(self.decoder_dest/'checkpoint_final.pt',map_location='cpu',weights_only=False)
        self.checkpoint_manifest = ck['manifest']; self.decoder_manifest = dck['manifest']
        if ck['step'] != 3200 or dck['step'] != 3000:
            raise ValueError('V26 must use the predeclared completed flow/decoder checkpoints')
        if self.destination in ck['manifest']['training_contexts'] or self.destination in dck['manifest']['decoder_contexts']:
            raise ValueError('Whole destination background was included in model fitting')
        if ck['manifest']['decoder_checkpoint_sha256'] != sha(self.decoder_dest/'checkpoint_final.pt'):
            raise ValueError('Decoder differs from the response-frozen decoder')
        if ck['manifest']['protocol_sha256'] != sha('configs/latent_flow_v25_protocol.json'):
            raise ValueError('Frozen V25 protocol hash drift')
        if ck['manifest']['model_sha256'] != sha('src/vcc2026/latent_flow_v25.py'):
            raise ValueError('Frozen V25 architecture hash drift')
        if ck['manifest']['trainer_sha256'] != sha('src/vcc2026/train_latent_flow_v25.py'):
            raise ValueError('Frozen V25 trainer hash drift')
        self.inputs, self.prior = predict_inputs(self.role,self.panel,features_origin=self.dest/'features.npz')
        if ck['manifest']['features_sha256'] != sha(self.dest/'features.npz'):
            raise ValueError('Frozen role feature hash drift')
        torch.set_num_threads(int(os.environ.get('VCC_TRAIN_THREADS',os.cpu_count() or 1)))
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(1.0)
        torch.set_float32_matmul_precision('high')
        self.model = LatentResponse(model_cfg()).to(self.device).eval(); self.model.load_state_dict(ck['state_dict'])
        self.decoder = NBDecoder(len(self.output_genes),model_cfg()).to(self.device).eval()
        self.decoder.load_state_dict(dck['state_dict'])
        for model in (self.model,self.decoder):
            for p in model.parameters(): p.requires_grad_(False)
        self.encoder_gpu = {k:torch.as_tensor(self.encoder[k],dtype=torch.float32,device=self.device)
                            for k in ['gene_center','gene_scale','basis','pc_scale']}
        self.common_mask = torch.zeros(len(self.output_genes),dtype=torch.bool,device=self.device)
        self.common_mask[self.input_output] = True
        self.theta = (torch.nn.functional.softplus(self.decoder.dispersion_unconstrained)+self.decoder.cfg['theta_floor']).detach().cpu().numpy().astype(np.float64)
        self.conditions = []; self.source_draws = []
        gpu = {k:torch.as_tensor(v,device=self.device) for k,v in self.inputs.items()}
        with torch.no_grad():
            for view in range(V25['inference']['views']):
                crng = np.random.default_rng(V25['inference']['cell_seed']+view*V25['inference']['view_stride'])
                before = crng.bit_generator.state
                _, sc, sp, valid, meta = sample_sets(self.targets,np.repeat(self.destination,len(self.targets)),crng,
                    set_size=V25['model']['set_size'],cache=self.cache,centered=False,return_metadata=True)
                source = [torch.as_tensor(v,dtype=torch.float32,device=self.device) for v in (sc,sp,valid)]
                self.conditions.append(self.model.condition(gpu['z'],gpu['h'],*source))
                self.source_draws.append({'view':view,'seed':V25['inference']['cell_seed']+view*V25['inference']['view_stride'],
                    'rng_before':before,'rng_after':crng.bit_generator.state,
                    **{k:v for k,v in meta.items() if k!='destination_lines'}})
        self.dependencies = self._dependencies()

    def _pool_metadata(self):
        """Recover row identity by validated integer count fingerprints, not order."""
        old = _control_counts(self.destination,self.input_genes).tocsr()
        meta = self.cache['control_metadata'][self.destination]
        keys = _row_keys(old); lookup = {}
        for row,key in enumerate(keys):
            if key in lookup:
                raise ValueError('Ambiguous NTC count fingerprint cannot identify a unique biological raw row')
            lookup[key] = row
        actual = self.pool[:,self.input_native].tocsr(); ids=[]
        for key in _row_keys(actual):
            if key not in lookup:
                raise ValueError('Actual scoring donor pool differs from provenance-locked V22 NTC counts')
            ids.append(lookup[key])
        ids = np.asarray(ids,np.int64)
        if len(set(ids)) != len(ids) or (actual != old[ids]).nnz:
            raise ValueError('NTC pool row provenance is not one-to-one/count-exact')
        result = meta.iloc[ids].reset_index(drop=True).copy()
        result['NTC_cache_row'] = ids; result['pool_row'] = np.arange(len(ids))
        if not result.target.eq('non-targeting').all() or not result.role.eq('input_NTC_only').all():
            raise ValueError('Scoring donor provenance is not NTC-only')
        if (self.panel/'controls.csv').exists():
            controls = pd.read_csv(self.panel/'controls.csv')
            generation = controls.loc[controls.role=='generation','row'].to_numpy(np.int64)
            if not np.array_equal(generation,result.original_row.to_numpy(np.int64)):
                raise ValueError('Panel controls.csv disagrees with count-verified original-row provenance')
        original=Path('runs/residual_v6/inner_panels')/self.destination/'controls.csv'
        controls=pd.read_csv(original)
        generation=controls.loc[controls.role=='generation','row'].to_numpy(np.int64)
        reference=controls.loc[controls.role=='reference','row'].to_numpy(np.int64)
        if not np.array_equal(generation,result.original_row.to_numpy(np.int64)):
            raise ValueError('Original V6 generation rows disagree with count-verified raw NTC provenance')
        if set(reference)&set(result.original_row):
            raise ValueError('NTC generation and historical reference raw rows overlap')
        return result

    def _dependencies(self):
        paths = [self.panel/'panel.json',self.panel/'generation_pool.npz',self.panel/'genes.txt',
                 self.dest/'checkpoint_3200.pt',self.dest/'features.npz',self.dest/'manifest.json',
                 self.decoder_dest/'checkpoint_final.pt',self.decoder_dest/'manifest.json',
                 Path('configs/latent_flow_v25_protocol.json'),Path('data/processed/set_response_v22/encoder.npz'),
                 Path('data/processed/set_response_v22/complete.json'),Path('data/processed/set_response_v22/input_genes.txt'),
                 Path('src/vcc2026/latent_counts_v26.py'),Path('src/vcc2026/latent_flow_v25.py'),
                 Path('src/vcc2026/train_latent_flow_v25.py'),Path('src/vcc2026/prepare_response_inputs_v23.py'),
                 Path('src/vcc2026/batch_cells_v24.py'),Path('src/vcc2026/prepare_set_response_v22.py'),
                 Path('src/vcc2026/generate_counts.py'),Path('src/vcc2026/residual_v6_common.py'),
                 Path('src/vcc2026/benchmark_context_v4.py')]
        if self.protocol_path.exists():paths.append(self.protocol_path)
        if (self.panel/'controls.csv').exists():paths.append(self.panel/'controls.csv')
        paths.append(Path('runs/residual_v6/inner_panels')/self.destination/'controls.csv')
        for line in ['k562',self.destination]:
            base=Path('data/processed/set_response_v22/controls')/line
            paths.extend([base/'cells.npy',base/'cell_metadata.csv',base/'measurement_mask.npy'])
            if (base/'target_batch_weights.npz').exists():paths.append(base/'target_batch_weights.npz')
            paths.extend([old_control_cache(line)/'control_core_counts.npz',
                          old_control_cache(line)/'control_genes.txt'])
        base=Path('data/processed/set_response_v22/source')
        paths.extend([base/n for n in ['cells.npy','cell_metadata.csv','targets.npy','offsets.npy']])
        return {str(path):sha(path) for path in paths}

    def _encode(self,counts):
        values=torch.as_tensor(np.asarray(counts,np.float32),device=self.device)
        totals=values.sum(-1,keepdim=True)
        if not torch.all(totals>0):raise ValueError('Zero common-axis library in actual scoring donor')
        logged=torch.log1p(values*(50000/totals))
        previous=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
        try:pc=((logged-self.encoder_gpu['gene_center'])/self.encoder_gpu['gene_scale'])@self.encoder_gpu['basis'].T/self.encoder_gpu['pc_scale']
        finally:torch.set_float32_matmul_precision(previous)
        return pc

    def _decoded(self,index,donor_counts):
        """Use real scoring donors; only source-condition draws inherit V25 sets."""
        actual=torch.as_tensor(donor_counts,dtype=torch.float32,device=self.device)
        mean=torch.as_tensor(self.mean[self.input_native],dtype=torch.float32,device=self.device)
        prior=torch.as_tensor(self.prior[index,self.input_output],dtype=torch.float32,device=self.device)
        wanted=((mean+.5)*torch.exp2(prior)-.5).clamp_min(0)
        ratio=torch.where(mean>0,wanted/mean.clamp_min(1e-12),torch.ones_like(mean)).clamp(0,16)
        x0=self._encode((actual*ratio).detach().cpu().numpy())
        library=actual.sum(-1)
        base=[];end=[];states=[]
        with torch.no_grad():
            for condition in self.conditions:
                c=condition[index:index+1]
                endpoint=self.model.endpoint(x0[None],c,V25['inference']['integration_steps'])[0]
                base.append(self.decoder(x0,library,self.common_mask).mu.cpu().numpy())
                end.append(self.decoder(endpoint,library,self.common_mask).mu.cpu().numpy())
                states.append(endpoint.cpu().numpy())
        # Four expected count views; no sampled cells are averaged.
        return (np.mean(np.stack(base),axis=0,dtype=np.float64),
                np.mean(np.stack(end),axis=0,dtype=np.float64),x0.cpu().numpy(),states)

    def build_target(self,target,arm):
        if arm not in ARMS:raise ValueError(arm)
        index=self.targets.index(str(target));donors=self.donors[target];order_index=self.order.index(target)
        donor=self.pool[donors].tocsr(copy=True)
        ratio=count_ratio(self.effects[index],self.mean,self.genes,self.output_genes)
        old_rng=np.random.default_rng(np.random.SeedSequence([self.count_seed,order_index,0]))
        g0=transform_counts(donor.copy(),ratio,old_rng)
        native_map={g:i for i,g in enumerate(self.genes)}
        seen=np.asarray(self.inputs['seen']).astype(bool)
        own=np.asarray(self.inputs['ownmask'][index]).astype(bool)
        editable_output=self.input_output[seen[self.input_output]&own[self.input_output]]
        editable_native=np.asarray([native_map[self.output_genes[g]] for g in editable_output],np.int64)
        expected_old=donor.toarray().astype(np.float64)*ratio[None,:]
        expected=expected_old.copy(); sampler_info={'sampler':'historical_sparse_stochastic_round',
            'seed_sequence_entropy':[self.count_seed,order_index,0], 'NB_rejection_sampler_gene_pairing_claimed':False}
        ipf_info={'sweeps':0,'row_max_relative_residual':0.,'column_max_relative_residual':0.,'converged':True,
                  'max_sweeps':MAX_SWEEPS,'tolerance':IPF_TOLERANCE}
        latent={}; block=g0
        if arm!='G0':
            before=time.monotonic()
            base,end,x0,endpoints=self._decoded(index,donor[:,self.input_native].toarray().astype(np.float32))
            decoded=base if arm in ('G1','G3') else end
            anchor=expected_old[:,editable_native]
            calibrated,ipf_info=ipf(decoded[:,editable_output],anchor.sum(1),anchor.sum(0))
            expected[:,editable_native]=calibrated
            sampler='round' if arm in ('G1','G2') else 'nb'
            integer,sampler_info=integer_sample(calibrated,self.theta[editable_output],self.count_seed,order_index,sampler)
            dense=g0.toarray(); dense[:,editable_native]=integer
            block=sparse.csr_matrix(dense,dtype=np.float32);block.eliminate_zeros();block.sort_indices()
            latent={'x0_PC_covariance':np.cov(x0,rowvar=False,ddof=0),
                    'endpoint_PC_covariance':np.mean([np.cov(v,rowvar=False,ddof=0) for v in endpoints],axis=0),
                    'decoded_seconds':time.monotonic()-before}
        protected=np.ones(len(self.genes),bool);protected[editable_native]=False
        if (block[:,protected]!=g0[:,protected]).nnz:raise AssertionError('Protected integer block changed')
        if np.any(~np.isfinite(block.data)) or np.any(block.data<=0) or not np.array_equal(block.data,np.rint(block.data)) or np.any(block.data>FLOAT32_INTEGER_MAX):
            raise ValueError('Generated counts violate old integer/nonnegative/exact float32 guard')
        library=np.asarray(block.sum(1),np.float64).ravel();g0lib=np.asarray(g0.sum(1),np.float64).ravel()
        if np.any(library<=0) or np.any(library>=1e6):raise ValueError('Generated library violates unchanged row guard')
        sampled=block.toarray().astype(np.float64);g0dense=g0.toarray().astype(np.float64)
        conditional_variance=(expected-np.floor(expected))*(1-(expected-np.floor(expected)))
        conditional_p0=np.maximum(1-expected,0)
        if arm in ('G3','G4'):
            mu=expected[:,editable_native];th=self.theta[editable_output]
            conditional_variance[:,editable_native]=mu+mu**2/th[None,:]
            conditional_p0[:,editable_native]=np.exp(-th[None,:]*np.log1p(mu/th[None,:]))
        expected_mean=expected.mean(0);expected_variance=conditional_variance.mean(0)+expected.var(0)
        # A new arm preserves an already realized G0 block outside E. These
        # extra moments condition on that old round draw; the original fields
        # retain the shared unrounded two-stage benchmark expectation.
        conditional_expected=expected.copy()
        conditional_noise=conditional_variance.copy()
        conditional_zero=conditional_p0.copy()
        if arm!='G0':
            conditional_expected[:,protected]=g0dense[:,protected]
            conditional_noise[:,protected]=0.
            conditional_zero[:,protected]=(g0dense[:,protected]==0).astype(np.float64)
        moments={'native_genes':self.genes,'editable_native_positions':editable_native,
                 'editable_output_positions':editable_output,'donor_pool_rows':donors,
                 'expected_gene_mean':expected_mean,'G0_expected_gene_mean':expected_old.mean(0),
                 'expected_gene_variance':expected_variance,'conditional_mu_gene_variance':expected.var(0),
                 'expected_gene_zero_fraction':conditional_p0.mean(0),'sample_gene_mean':sampled.mean(0),
                 'sample_gene_variance':sampled.var(0),'sample_gene_zero_fraction':(sampled==0).mean(0),
                 'G0_sample_gene_mean':g0dense.mean(0),'G0_sample_gene_variance':g0dense.var(0),
                 'G0_sample_gene_zero_fraction':(g0dense==0).mean(0),
                 'actual_native_library':library,'G0_actual_native_library':g0lib,
                 'expected_native_library':expected.sum(1),'G0_expected_native_library':expected_old.sum(1),
                 'expected_editable_library':expected[:,editable_native].sum(1),
                 'G0_expected_editable_library':expected_old[:,editable_native].sum(1),
                 'sample_editable_library':sampled[:,editable_native].sum(1),
                 'frozen_theta_editable':self.theta[editable_output],
                 'conditional_given_G0_expected_gene_mean':conditional_expected.mean(0),
                 'conditional_given_G0_expected_gene_variance':conditional_noise.mean(0)+conditional_expected.var(0),
                 'conditional_given_G0_expected_gene_zero_fraction':conditional_zero.mean(0),
                 'conditional_given_G0_expected_native_library':conditional_expected.sum(1)}
        for key,value in latent.items():
            if not key.endswith('_seconds'):moments[key]=value
        editable_new_nonzero=(sampled[:,editable_native]>0)&(donor[:,editable_native].toarray()==0)
        diag={'target':target,'arm':arm,'cells':len(donors),'native_genes':len(self.genes),
              'editable_genes':len(editable_native),'protected_genes':int(protected.sum()),
              'IPF':ipf_info,'sampler':sampler_info,'protected_integer_equality':True,
              'integer_nonnegative_finite_exact_float32':True,'row_library_guard_passed':True,
              'expected_gene_mean_max_absolute_difference_G0':float(np.max(np.abs(expected_mean-expected_old.mean(0)),initial=0)),
              'expected_native_library_max_relative_difference_G0':float(np.max(np.abs(expected.sum(1)-expected_old.sum(1))/np.maximum(1,expected_old.sum(1)),initial=0)),
              'sample_gene_mean_MAE_vs_expected':float(np.mean(np.abs(sampled.mean(0)-expected_mean))),
              'editable_sample_gene_mean_MAE_vs_expected':float(np.mean(np.abs(sampled[:,editable_native].mean(0)-expected_mean[editable_native]))) if len(editable_native) else 0.,
              'sample_native_zero_fraction':float((sampled==0).mean()),
              'G0_sample_native_zero_fraction':float((g0dense==0).mean()),
              'sample_native_mean_gene_variance':float(sampled.var(0).mean()),
              'G0_sample_native_mean_gene_variance':float(g0dense.var(0).mean()),
              'actual_native_library':quantiles(library),'G0_actual_native_library':quantiles(g0lib),
              'actual_native_library_MAE_vs_G0':float(np.mean(np.abs(library-g0lib))),
              'actual_native_library_mean_difference_G0':float(np.mean(library-g0lib)),
              'actual_library_correlation_G0':_corr(library,g0lib),
              'editable_new_nonzero_coordinates':int(editable_new_nonzero.sum()),
              'editable_new_nonzero_fraction_of_original_zeros':float(editable_new_nonzero.sum()/max(1,(donor[:,editable_native].toarray()==0).sum())),
              'theta_editable':quantiles(self.theta[editable_output]) if len(editable_output) else None,
              'count_block_sha256':csr_sha(block),'G0_count_block_sha256':csr_sha(g0),
              'donor_pool_rows_sha256':array_sha(donors),'editable_output_positions_sha256':array_sha(editable_output),
              'original_NTC_common_library':quantiles(np.asarray(donor[:,self.input_native].sum(1)).ravel()),
              'new_mean_conditional_on_frozen_model_not_destination_truth':True,
              'full_axis_expectation_fields':'Shared unrounded G0 benchmark expectation; conditional_given_G0 fields hold protected realized integers fixed',
              'sample_gene_mean_MAE_vs_conditional_given_G0_expectation':float(np.mean(np.abs(sampled.mean(0)-conditional_expected.mean(0)))),
              'latent_diagnostics':{k:v for k,v in latent.items() if k.endswith('_seconds')}}
        return TargetResult(target,block,diag,moments)

    def iter_target(self,arm):
        for target in self.targets:yield self.build_target(target,arm)

    def write_provenance(self,records_dir):
        records_dir=Path(records_dir);records_dir.mkdir(parents=True,exist_ok=True)
        offset=0;rows=[]
        for target in self.targets:
            ids=self.donors[target];part=self.pool_metadata.iloc[ids].copy()
            part['generated_target']=target;part['generated_row_id']=[f'c_{offset+i}' for i in range(len(ids))]
            part['generated_row_offset']=np.arange(offset,offset+len(ids));offset+=len(ids)
            rows.append(part)
        csv=records_dir/'donor_provenance.csv';pd.concat(rows,ignore_index=True).to_csv(csv,index=False)
        source_path=records_dir/'source_condition_draws.npz';values={}
        source_records=[]
        for view in self.source_draws:
            idx=view['view']
            for key in ['destination_rows','source_ntc_rows','source_perturbed_rows']:values[f'view{idx}_{key}']=view[key]
            source_records.append({k:v for k,v in view.items() if k not in ['destination_rows','source_ntc_rows','source_perturbed_rows']})
        np.savez_compressed(source_path,targets=np.asarray(self.targets,str),**values)
        source_info=records_dir/'source_condition_draws.json'
        dump(source_info,{'views':source_records,'source_control_metadata_sha256':sha('data/processed/set_response_v22/controls/k562/cell_metadata.csv'),
            'source_perturbation_metadata_sha256':sha('data/processed/set_response_v22/source/cell_metadata.csv'),
            'rows_are_cache_indices_resolvable_to_original_raw_rows':True,
            'destination_draws_consumed_to_preserve_V25_source_RNG_but_actual_scoring_donors_used':True})
        return csv,source_path,source_info


def generate(panel_work, name, effects, count_seed, *, model_role, model_seed, arm,
             records_dir, protocol_path=PROTOCOL):
    """Score adapter API. No training or outcome-dependent parameters are added."""
    panel=Path(panel_work);records=Path(records_dir);dest=panel/f'pred__{name}.h5ad'
    if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
    manifest=records/'generation_manifest.json'
    if manifest.exists():raise FileExistsError(f'Refusing to overwrite {manifest}')
    records.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();generator=PanelGenerator(panel,effects,model_role,model_seed,count_seed,protocol_path)
    csv,source_npz,source_json=generator.write_provenance(records)
    if arm=='G0':
        old_generate(panel,name,effects,count_seed)
    blocks=[];labels=[];diags=[];paths=[]
    for target in generator.targets:
        try:
            result=generator.build_target(target,arm)
        except Exception as error:
            failure={'failed':True,'target':target,'arm':arm,'exception_type':type(error).__name__,
                     'exception':str(error),'clipped_or_redrawn':False,
                     'dependencies_sha256':generator.dependencies,'model_role':model_role,
                     'model_seed':int(model_seed),'count_seed':int(count_seed)}
            if isinstance(error,GenerationValidationError):
                if error.failed_draw is not None:
                    np.savez_compressed(records/'failed_draw.npz',draw=error.failed_draw)
                    failure['failed_draw_sha256']=sha(records/'failed_draw.npz')
                if error.ipf_weights is not None:
                    np.savez_compressed(records/'failed_IPF_weights.npz',weights=error.ipf_weights)
                    failure['failed_IPF_weights_sha256']=sha(records/'failed_IPF_weights.npz')
                if error.rng is not None:failure['rng']=error.rng
            dump(records/'generation_failure.json',failure)
            raise
        target_dir=records/'targets';target_dir.mkdir(exist_ok=True)
        moments=target_dir/f'{result.target}.npz';np.savez_compressed(moments,**result.moments)
        info=target_dir/f'{result.target}.json';dump(info,result.diagnostics)
        paths.extend([str(moments),str(info)]);diags.append(result.diagnostics)
        if arm!='G0':blocks.append(result.block);labels.extend([result.target]*result.block.shape[0])
        print('V26_GENERATED_TARGET',arm,result.target,result.block.shape[0],result.diagnostics['IPF']['sweeps'],flush=True)
    if arm!='G0':
        from benchmark_context_v4 import save
        save(dest,blocks,labels,generator.genes)
    targets_csv=records/'target_diagnostics.csv'
    flat=[]
    for row in diags:
        fields={k:v for k,v in row.items() if not isinstance(v,(dict,list))}
        fields.update({f'IPF_{k}':v for k,v in row['IPF'].items()})
        flat.append(fields)
    pd.DataFrame(flat).to_csv(targets_csv,index=False)
    value={'schema':'v26_mean_matched_direct_count_1','completed':True,'protocol_sha256':sha(protocol_path),
        'generator_sha256':sha('src/vcc2026/latent_counts_v26.py'),'arm':arm,'generator_changed':arm!='G0',
        'model_role':model_role,'model_seed':int(model_seed),'count_seed':int(count_seed),
        'fixed_flow_checkpoint_step':3200,'fixed_decoder_checkpoint_step':3000,'new_full_fits':0,
        'destination':generator.destination,'targets':generator.targets,'native_genes_sha256':array_sha(generator.genes),
        'effects_array_sha256':array_sha(generator.effects),'generation_target_order':generator.order,
        'donor_rng_before':generator.donor_rng_before,'donor_rng_after':generator.donor_rng_after,
        'same_G0_donors_selected_before_any_new_sampler_draw':True,
        'donor_provenance_count_verified_on_all_6119_common_genes':True,
        'source_conditions':{'views':V25['inference']['views'],'cell_seed':V25['inference']['cell_seed'],
                            'view_stride':V25['inference']['view_stride'],'set_size':V25['model']['set_size']},
        'anchor':'actual scoring NTC donors × frozen V1 capped ratio; expected fractional counts encoded without round',
        'editable_axis':'input6119 intersect role-seen own-gene-mask and native measured genes',
        'mean_match':'G0 unrounded editable donor row totals and gene column totals; expected native totals preserve G0',
        'IPF':{'max_sweeps':MAX_SWEEPS,'tolerance':IPF_TOLERANCE,'dtype':'float64','failure_is_fatal':True},
        'sampler':'unchanged literal historical generator' if arm=='G0' else ('stochastic_round' if arm in ('G1','G2') else 'numpy_negative_binomial_NB2'),
        'NB_inverse_dispersion_source':'role-specific frozen decoder gene-specific softplus(theta)+floor',
        'NB_rejection_sampler_gene_pairing_claimed':False,'actual_row_totals_forced':False,
        'destination_perturbation_outcomes_read':False,'whole_destination_heldout_from_training':True,
        'expected_count_views_averaged_before_integer_sampling':True,
        'all_protected_integer_equal_G0':all(v['protected_integer_equality'] for v in diags),
        'all_IPF_converged':all(v['IPF']['converged'] for v in diags),
        'dependencies_sha256':generator.dependencies,'prediction_sha256':sha(dest),
        'donor_provenance_sha256':sha(csv),'source_draws_sha256':sha(source_npz),'source_draws_info_sha256':sha(source_json),
        'target_diagnostics_sha256':sha(targets_csv),'seconds':time.monotonic()-start,
        'prediction_path':str(dest),'diagnostics_dir':str(records),'target_diagnostic_files':paths}
    dump(manifest,value)
    return {'prediction_path':dest,'generator_changed':arm!='G0','generation_manifest_path':manifest,
            'diagnostics_paths':[targets_csv,source_npz,source_json,*map(Path,paths)],'provenance_path':csv}
