"""Small Transformer warm-start tuning; immutable data, V1 and generator."""
from pathlib import Path
import copy, json, time, shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from train_residual_v7 import JointResidual, predict_features, init_runtime
from train_residual_v6 import load_rows, generated_profile
from train_context_v4 import load_data, target_features
from residual_v6_common import frozen_v1, dump, sha, fold, SEED

MODEL = Path('models/residual_v10')
PROTOCOL = Path('configs/residual_v10_protocol.json')
OFFSET = 100003


class BiasResidual(JointResidual):
    def __init__(self, ng):
        super().__init__(ng, 'tiny_transformer')
        self.response_bias = nn.Parameter(torch.zeros(ng))

    def bounded_bias(self):
        return .1 * torch.tanh(self.response_bias / .1)

    def forward(self, z, g, h, seen, ownmask=None):
        blocks = [torch.cat([z[:,:64], z[:,128:]], dim=1), z[:,64:128], h]
        t = torch.stack([p(b) for p,b in zip(self.projections, blocks)], dim=1) + self.token_type
        for layer in self.layers:
            t = layer(t)
        q = self.readout(self.norm(t).mean(1))
        base = q @ self.decoder.T
        back = (q @ self.background_decoder.T) * g
        s = torch.sigmoid(self.gate(torch.cat([z,h], dim=1)))
        r = s * .75 * torch.tanh((base+back+self.bounded_bias()) / .75) * seen
        if ownmask is not None:
            r = r * ownmask
        return r, s, back*s


def model_for(kind, ng):
    return BiasResidual(ng) if kind == 'bounded_bias' else JointResidual(ng, 'tiny_transformer')


def fit(name, lines, origin, start_step, seed=SEED, recipes=('continued', 'bounded_bias')):
    assert Path('reports/delivery_manifest_residual_v9.json').exists(), 'Finish V9 before V10 data training'
    origin = Path(origin)
    dest = MODEL / name / f'seed_{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    expected = [k for recipe in recipes for k in (['continued','ema','swa'] if recipe == 'continued' else [recipe])]
    if all((dest / f'{k}_complete.json').exists() for k in expected):
        return
    checkpoint = origin / f'tiny_transformer_{start_step}.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ck['kind'] == 'tiny_transformer' and ck['seed'] == seed and ck['step'] == start_step
    assert ck['protocol_sha256'] == sha('configs/residual_v7_protocol.json')
    assert ck['V1_config_sha256'] == sha('configs/baseline_v1.json')
    data, genes, core = load_data()
    rows = load_rows(lines)
    oldrows = pd.read_csv(origin / 'training_rows.csv')
    assert list(rows['target']) == oldrows.target.tolist() and list(rows['line']) == oldrows.context.tolist()
    assert set(ck['train_contexts']) == set(lines)
    with np.load(origin / 'features.npz') as z:
        feature = {k:z[k] for k in z.files}
    for fname in ['features.npz', 'training_rows.csv']:
        if not (dest / fname).exists():
            shutil.copy2(origin / fname, dest / fname)
        assert sha(dest / fname) == sha(origin / fname)
    _, z, _, _ = target_features(data, rows['target'], genes, feature['basis'], feature['center'],
                                feature['scale'], feature['string_mean'], feature['string_std'])
    g = np.clip((rows['c']-feature['source_ctrl']) / 3, -2, 2).astype(np.float32)
    h = (g @ feature['basis'].T / np.sqrt(len(genes))).astype(np.float32)
    raw = frozen_v1(rows['target'], genes)
    seen = feature['seen']
    ix = {gene:i for i,gene in enumerate(genes)}
    assert all(fold(t) >= 2 and t not in set(data['h1']['target_names']) for t in rows['target'])
    own = np.ones(rows['mask'].shape, np.float32)
    for i,target in enumerate(rows['target']):
        if target in ix:
            own[i,ix[target]] = 0
    tx = {k:torch.as_tensor(rows[k], device='cuda', dtype=torch.float32) for k in ['mu','y','mask','outside']}
    tx.update(raw=torch.as_tensor(raw,device='cuda'), g=torch.as_tensor(g,device='cuda'),
              h=torch.as_tensor(h,device='cuda'), z=torch.as_tensor(z,device='cuda'),
              own=torch.as_tensor(own,device='cuda'))
    ts = torch.as_tensor(seen,device='cuda')
    byline = {line:np.flatnonzero(rows['line'] == line) for line in lines}
    del data, raw, g, h, z, own
    ctrl = torch.log1p(tx['mu'] * 50000 / (tx['mu'].sum(1)+tx['outside'])[:,None])
    common = {'ng':len(genes), 'seed':seed, 'train_contexts':lines,
        'V1_config_sha256':sha('configs/baseline_v1.json'), 'protocol_sha256':sha(PROTOCOL),
        'warm_start_checkpoint':checkpoint.as_posix(), 'warm_start_sha256':sha(checkpoint),
        'warm_start_step':start_step, 'optimizer_reset':True}
    for recipe in recipes:
        kinds = ['continued','ema','swa'] if recipe == 'continued' else [recipe]
        if all((dest / f'{k}_complete.json').exists() for k in kinds):
            continue
        torch.manual_seed(seed+OFFSET)
        rng = np.random.default_rng(seed+OFFSET)
        model = model_for(recipe,len(genes)).cuda()
        mismatch = model.load_state_dict(ck['state_dict'],strict=False)
        assert not mismatch.unexpected_keys
        assert mismatch.missing_keys == (['response_bias'] if recipe == 'bounded_bias' else [])
        reference = model_for('continued',len(genes)).cuda()
        reference.load_state_dict(ck['state_dict'])
        model.eval(); reference.eval()
        with torch.no_grad():
            args = [tx['z'][:2],tx['g'][:2],tx['h'][:2],ts,tx['own'][:2]]
            assert torch.equal(model(*args)[0],reference(*args)[0])
        del reference
        # Reset after construction/identity checks so both recipes see identical dropout draws.
        torch.manual_seed(seed+OFFSET)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.00005,weight_decay=.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1200,eta_min=.00001)
        ema = AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(.995)) if recipe == 'continued' else None
        swa = AveragedModel(model) if recipe == 'continued' else None
        if ema is not None:
            ema.update_parameters(model)
        trace = []
        begin = time.monotonic()
        for step in range(1,1201):
            model.train()
            size = max(2,96//len(lines))
            ids = np.concatenate([rng.choice(byline[line],size,replace=False) for line in lines])
            j = torch.as_tensor(ids,device='cuda')
            residual, gate, back = model(tx['z'][j],tx['g'][j],tx['h'][j],ts,tx['own'][j])
            pred = generated_profile(tx['mu'][j],tx['outside'][j],tx['raw'][j]+residual)
            mask = tx['mask'][j]
            mse = ((pred-tx['y'][j]).square()*mask).sum()/mask.sum()
            reg = .1*(residual.square()*mask).sum()/mask.sum() + .5*(back.square()*mask).sum()/mask.sum()
            bias_penalty = .5*(model.bounded_bias().square()*ts).sum()/ts.sum() if recipe == 'bounded_bias' else torch.zeros((),device='cuda')
            losses = []
            for li,line in enumerate(lines):
                sl = slice(li*size,(li+1)*size)
                sub = j[sl]
                measured = mask[sl].amin(0)
                for target in rows['target'][ids[sl]]:
                    if target in ix:
                        measured[ix[target]] = 0
                pe = nn.functional.normalize((pred[sl]-ctrl[sub])*measured,dim=1,eps=1e-8)
                ye = nn.functional.normalize((tx['y'][sub]-ctrl[sub])*measured,dim=1,eps=1e-8)
                losses.append(nn.functional.cross_entropy(pe@ye.T/.2,torch.arange(size,device='cuda')))
            disc = torch.stack(losses).mean()
            loss = mse+reg+bias_penalty+.05*disc
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step(); scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
                if step >= 200 and step % 20 == 0:
                    swa.update_parameters(model)
            if step == 1 or step % 200 == 0:
                rec = {'extra_step':step, 'total_steps':start_step+step, 'loss':float(loss.detach()),
                    'mse':float(mse.detach()), 'discrimination':float(disc.detach()),
                    'residual_penalty':float(reg.detach()), 'bias_penalty':float(bias_penalty.detach()),
                    'residual_RMS':float(residual.detach().square().mean().sqrt()),
                    'learning_rate':optimizer.param_groups[0]['lr'], 'seconds':time.monotonic()-begin}
                trace.append(rec)
                pd.DataFrame(trace).to_csv(dest / f'{recipe}_trace.csv',index=False)
                print('TRAIN_V10',name,seed,recipe,json.dumps(rec),flush=True)
            if step in [600,1200]:
                torch.save({**common,'state_dict':model.state_dict(),'kind':recipe,'step':step,
                    'total_training_steps':start_step+step,'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),'torch_rng_state':torch.get_rng_state(),
                    'cuda_rng_state':torch.cuda.get_rng_state(),'numpy_rng_state':copy.deepcopy(rng.bit_generator.state)},
                    dest / f'{recipe}_{step}.pt')
                if ema is not None:
                    for kind,avg in [('ema',ema),('swa',swa)]:
                        torch.save({**common,'state_dict':avg.module.state_dict(),'kind':kind,'step':step,
                            'total_training_steps':start_step+step,'averaged_updates':int(avg.n_averaged),
                            'averaging':'exponential_decay0.995' if kind == 'ema' else 'equal_every20_start200',
                            'optimizer_state_dict':None},dest / f'{kind}_{step}.pt')
        for kind in kinds:
            dump(dest / f'{kind}_complete.json',{**common,'steps':1200,'training_rows':len(rows['target']),
                'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.monotonic()-begin,
                'initial_response_matches_warm_start':True,'features_origin':origin.as_posix(),
                'feature_sha256':sha(dest / 'features.npz'),'shared_trajectory':recipe})
        del model, optimizer, scheduler, ema, swa
        torch.cuda.empty_cache()
    del tx
    torch.cuda.empty_cache()


def predict(origin,kind,step,inputs):
    ck = torch.load(Path(origin) / f'{kind}_{step}.pt',map_location='cpu',weights_only=False)
    model = model_for(kind,ck['ng']).eval()
    model.load_state_dict(ck['state_dict'])
    with torch.no_grad():
        residual,gate,_ = model(**inputs)
    r = residual.numpy()
    assert np.isfinite(r).all() and np.abs(r).max() <= .750001
    return r,gate.numpy()
