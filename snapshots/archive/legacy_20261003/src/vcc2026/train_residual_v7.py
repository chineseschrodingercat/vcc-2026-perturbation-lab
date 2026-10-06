"""Bounded, frozen-V1 early context fusion; immutable V6 inputs and generator."""
from pathlib import Path
import json, time, math, argparse, shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import sparse
from train_residual_v6 import load_rows, generated_profile, BoundedResidual
from train_context_v4 import load_data, target_features
from residual_v6_common import axes, frozen_v1, dump, sha, fold, SEED, CONTEXTS

MODEL=Path('models/residual_v7')
PROTOCOL=Path('configs/residual_v7_protocol.json')

class JointResidual(nn.Module):
    def __init__(self,ng,kind):
        super().__init__(); self.kind=kind
        if kind=='joint_mlp':
            self.encoder=nn.Sequential(nn.Linear(194,432),nn.SiLU(),nn.Dropout(.1),nn.Linear(432,432),nn.SiLU(),nn.Dropout(.1),nn.Linear(432,32))
        elif kind=='tiny_transformer':
            self.projections=nn.ModuleList([nn.Linear(66,128),nn.Linear(64,128),nn.Linear(64,128)])
            self.token_type=nn.Parameter(torch.randn(1,3,128)*.02)
            # Construct independently: avoid cloned encoder layers with identical initialization.
            self.layers=nn.ModuleList([nn.TransformerEncoderLayer(128,4,256,.1,'gelu',batch_first=True,norm_first=True) for _ in range(2)])
            self.norm=nn.LayerNorm(128);self.readout=nn.Linear(128,32)
        else:raise ValueError(kind)
        self.decoder=nn.Parameter(torch.zeros(ng,32));self.background_decoder=nn.Parameter(torch.zeros(ng,32))
        self.gate=nn.Linear(194,1);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,math.log(.1/.9))
    def forward(self,z,g,h,seen,ownmask=None):
        if self.kind=='joint_mlp':q=self.encoder(torch.cat([z,h],dim=1))
        else:
            blocks=[torch.cat([z[:,:64],z[:,128:]],dim=1),z[:,64:128],h]
            t=torch.stack([p(b) for p,b in zip(self.projections,blocks)],dim=1)+self.token_type
            for layer in self.layers:t=layer(t)
            q=self.readout(self.norm(t).mean(1))
        base=q@self.decoder.T;back=(q@self.background_decoder.T)*g
        s=torch.sigmoid(self.gate(torch.cat([z,h],dim=1)))
        r=s*.75*torch.tanh((base+back)/.75)*seen
        if ownmask is not None:r=r*ownmask
        return r,s,back*s

def model_for(kind,ng):return BoundedResidual(ng,130) if kind=='B' else JointResidual(ng,kind)

def fit(name,lines,origin,seed=SEED,kinds=('joint_mlp','tiny_transformer')):
    dest=MODEL/name/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    if all((dest/f'{k}_complete.json').exists() for k in kinds):return
    data,genes,core=load_data();rows=load_rows(lines);origin=Path(origin)
    oldrows=pd.read_csv(origin/'training_rows.csv')
    assert list(rows['target'])==oldrows.target.tolist() and list(rows['line'])==oldrows.context.tolist()
    with np.load(origin/'features.npz') as f:p={k:f[k] for k in f.files}
    for fname in ['features.npz','training_rows.csv']:
        if not (dest/fname).exists():shutil.copy2(origin/fname,dest/fname)
        assert sha(dest/fname)==sha(origin/fname)
    _,z,_,_=target_features(data,rows['target'],genes,p['basis'],p['center'],p['scale'],p['string_mean'],p['string_std'])
    g=np.clip((rows['c']-p['source_ctrl'])/3,-2,2).astype(np.float32);h=(g@p['basis'].T/np.sqrt(len(genes))).astype(np.float32)
    raw=frozen_v1(rows['target'],genes);seen=p['seen'];ix={g:i for i,g in enumerate(genes)}
    assert all(fold(t)>=2 and t not in set(data['h1']['target_names']) for t in rows['target'])
    own=np.ones(rows['mask'].shape,np.float32)
    for i,t in enumerate(rows['target']):
        if t in ix:own[i,ix[t]]=0
    tx={k:torch.as_tensor(rows[k],device='cuda',dtype=torch.float32) for k in ['mu','y','mask','outside']}
    tx.update(raw=torch.as_tensor(raw,device='cuda'),g=torch.as_tensor(g,device='cuda'),h=torch.as_tensor(h,device='cuda'),z=torch.as_tensor(z,device='cuda'),own=torch.as_tensor(own,device='cuda'))
    ts=torch.as_tensor(seen,device='cuda');byline={l:np.flatnonzero(rows['line']==l) for l in lines};del data,raw,g,h,z,own
    ctrlprofile=torch.log1p(tx['mu']*50000/(tx['mu'].sum(1)+tx['outside'])[:,None])
    for kind in kinds:
        if (dest/f'{kind}_complete.json').exists():continue
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=model_for(kind,len(genes)).cuda()
        with torch.no_grad():
            r,_,_=model(tx['z'][:2],tx['g'][:2],tx['h'][:2],ts,tx['own'][:2]);assert torch.count_nonzero(r)==0
            assert torch.equal(generated_profile(tx['mu'][:2],tx['outside'][:2],tx['raw'][:2]+r),generated_profile(tx['mu'][:2],tx['outside'][:2],tx['raw'][:2]))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.02)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1200,eta_min=.00005)
        trace=[];begin=time.monotonic()
        for step in range(1,1201):
            model.train();size=max(2,96//len(lines));ids=np.concatenate([rng.choice(byline[l],size,replace=False) for l in lines]);j=torch.as_tensor(ids,device='cuda')
            r,s,back=model(tx['z'][j],tx['g'][j],tx['h'][j],ts,tx['own'][j]);pred=generated_profile(tx['mu'][j],tx['outside'][j],tx['raw'][j]+r)
            m=tx['mask'][j];mse=((pred-tx['y'][j]).square()*m).sum()/m.sum()
            reg=.1*(r.square()*m).sum()/m.sum()+.5*(back.square()*m).sum()/m.sum()
            losses=[]
            for li,line in enumerate(lines):
                sl=slice(li*size,(li+1)*size);sub=j[sl];pm=m[sl].amin(0)
                for t in rows['target'][ids[sl]]:
                    if t in ix:pm[ix[t]]=0
                pe=nn.functional.normalize((pred[sl]-ctrlprofile[sub])*pm,dim=1,eps=1e-8)
                ye=nn.functional.normalize((tx['y'][sub]-ctrlprofile[sub])*pm,dim=1,eps=1e-8)
                losses.append(nn.functional.cross_entropy(pe@ye.T/.2,torch.arange(size,device='cuda')))
            disc=torch.stack(losses).mean();loss=mse+reg+.05*disc;assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step()
            if step==1 or step%200==0:
                rec={'step':step,'loss':float(loss.detach()),'mse':float(mse.detach()),'residual_penalty':float(reg.detach()),'discrimination':float(disc.detach()),'residual_RMS':float(r.detach().square().mean().sqrt()),'gate_mean':float(s.detach().mean()),'seconds':time.monotonic()-begin}
                trace.append(rec);pd.DataFrame(trace).to_csv(dest/f'{kind}_trace.csv',index=False);print('TRAIN_V7',name,seed,kind,json.dumps(rec),flush=True)
            if step in [600,1200]:
                torch.save({'state_dict':model.state_dict(),'ng':len(genes),'kind':kind,'step':step,'seed':seed,'train_contexts':lines,'V1_config_sha256':sha('configs/baseline_v1.json'),'protocol_sha256':sha(PROTOCOL)},dest/f'{kind}_{step}.pt')
        dump(dest/f'{kind}_complete.json',{'steps':1200,'training_rows':len(rows['target']),'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.monotonic()-begin,'zero_residual_test':True,'features_origin':str(origin),'feature_sha256':sha(dest/'features.npz'),'train_contexts':lines,'seed':seed})
        del model;torch.cuda.empty_cache()
    del tx;torch.cuda.empty_cache()

def predict_features(origin,path):
    origin=Path(origin);path=Path(path);data,genes,core=load_data();info=json.loads((path/'panel.json').read_text());targets=info['targets'];native=np.loadtxt(path/'genes.txt',dtype=str)
    pool=sparse.load_npz(path/'generation_pool.npz');mu=np.asarray(pool.mean(0)).ravel().astype(np.float64)
    with np.load(origin/'features.npz') as f:p={k:f[k] for k in f.files}
    _,z,_,_=target_features(data,targets,genes,p['basis'],p['center'],p['scale'],p['string_mean'],p['string_std'])
    gm={g:i for i,g in enumerate(native)};total=sum(mu[gm[g]] for g in core if g in gm);ctrl=np.zeros(len(genes),np.float32)
    for i,gene in enumerate(genes):
        if gene in gm:ctrl[i]=np.log1p(mu[gm[gene]]*50000/total)
    g=np.tile(np.clip((ctrl-p['source_ctrl'])/3,-2,2),(len(targets),1)).astype(np.float32);h=(g@p['basis'].T/np.sqrt(len(genes))).astype(np.float32)
    own=np.ones((len(targets),len(genes)),np.float32);gp={g:i for i,g in enumerate(genes)}
    for i,t in enumerate(targets):
        if t in gp:own[i,gp[t]]=0
    return {'z':torch.as_tensor(z),'g':torch.as_tensor(g),'h':torch.as_tensor(h),'seen':torch.as_tensor(p['seen']),'ownmask':torch.as_tensor(own)},frozen_v1(targets,genes)

def predict(origin,kind,step,inputs):
    ck=torch.load(Path(origin)/f'{kind}_{step}.pt',map_location='cpu',weights_only=False)
    model=model_for(kind,ck['ng']);model.load_state_dict(ck['state_dict']);model.eval()
    with torch.no_grad():r,s,_=model(**inputs)
    r=r.numpy();assert np.max(np.abs(r))<=.750001
    return r,s.numpy()

def init_runtime():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65);torch.set_float32_matmul_precision('high')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--counts',action='store_true');args=a.parse_args()
    if args.counts:
        print(json.dumps({k:sum(p.numel() for p in model_for(k,10916).parameters()) for k in ['B','joint_mlp','tiny_transformer']},indent=2))
