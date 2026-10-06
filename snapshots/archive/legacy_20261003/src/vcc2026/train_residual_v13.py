"""Frozen-V1 response tuning; objectives and recipes locked before evaluation."""
from pathlib import Path
import json, math, time, shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from residual_v6_common import dump, sha, fold, frozen_v1, SEED
from train_residual_v6 import load_rows, generated_profile
from train_context_v4 import load_data, target_features
from train_residual_v7 import init_runtime, predict_features

MODEL=Path('models/residual_v13'); PROTOCOL=Path('configs/residual_v13_protocol.json')
P=json.loads(PROTOCOL.read_text()); KINDS=list(P['recipes'])
def config(kind):return {**P['defaults'],**P['recipes'][kind]}

class TunedResidual(nn.Module):
    def __init__(self,ng,kind):
        super().__init__();self.cfg=c=config(kind);rank=c['rank']
        self.projections=nn.ModuleList([nn.Linear(66,128),nn.Linear(64,128),nn.Linear(64,128)])
        self.token_type=nn.Parameter(torch.randn(1,3,128)*.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(128,4,256,c['dropout'],'gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.norm=nn.LayerNorm(128);self.readout=nn.Linear(128,rank)
        self.decoder=nn.Parameter(torch.zeros(ng,rank));self.background_decoder=nn.Parameter(torch.zeros(ng,rank))
        self.gate=nn.Linear(194,1);nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias,math.log(c['gate_init']/(1-c['gate_init'])))
        if c['film']:
            self.film_scale=nn.Linear(64,rank);self.film_shift=nn.Linear(64,rank)
            for m in [self.film_scale,self.film_shift]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,z,g,h,seen,ownmask):
        blocks=[torch.cat([z[:,:64],z[:,128:]],1),z[:,64:128],h]
        t=torch.stack([p(b) for p,b in zip(self.projections,blocks)],1)+self.token_type
        for layer in self.layers:t=layer(t)
        q=self.readout(self.norm(t).mean(1))
        if self.cfg['film']:q=q*(1+.25*torch.tanh(self.film_scale(h)))+self.film_shift(h)
        back=(q@self.background_decoder.T)*g;val=q@self.decoder.T+back
        s=torch.sigmoid(self.gate(torch.cat([z,h],1)));bound=self.cfg['bound']
        return s*bound*torch.tanh(val/bound)*seen*ownmask,s,back*s

def fit(name,lines,origin,seed=SEED,kinds=None):
    kinds=KINDS if kinds is None else kinds;dest=MODEL/name/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    kinds=[k for k in kinds if not (dest/f'{k}_complete.json').exists()]
    if not kinds:return
    assert shutil.disk_usage('.').free/2**30>=50
    data,genes,_=load_data();rows=load_rows(lines);origin=Path(origin)
    old=pd.read_csv(origin/'training_rows.csv')
    assert old.target.tolist()==rows['target'].tolist() and old.context.tolist()==rows['line'].tolist()
    for f in ['features.npz','training_rows.csv']:
        if not (dest/f).exists():shutil.copy2(origin/f,dest/f)
        assert sha(dest/f)==sha(origin/f)
    with np.load(dest/'features.npz') as f:params={k:f[k] for k in f.files}
    _,z,_,_=target_features(data,rows['target'],genes,params['basis'],params['center'],params['scale'],params['string_mean'],params['string_std'])
    g=np.clip((rows['c']-params['source_ctrl'])/3,-2,2).astype(np.float32)
    h=(g@params['basis'].T/np.sqrt(len(genes))).astype(np.float32);raw=frozen_v1(rows['target'],genes)
    assert all(fold(t)>=2 and t not in set(data['h1']['target_names']) for t in rows['target'])
    ix={g:i for i,g in enumerate(genes)};own=np.ones_like(rows['mask'],np.float32)
    for j,t in enumerate(rows['target']):
        if t in ix:own[j,ix[t]]=0
    byline={line:np.flatnonzero(rows['line']==line) for line in lines}
    tx={k:torch.as_tensor(rows[k],device='cuda',dtype=torch.float32) for k in ['mu','y','mask','outside']}
    tx.update({k:torch.as_tensor(v,device='cuda') for k,v in {'z':z,'g':g,'h':h,'raw':raw,'ownmask':own}.items()})
    tx['control']=torch.log1p(tx['mu']*50000/(tx['mu'].sum(1)+tx['outside'])[:,None])
    seen=torch.as_tensor(params['seen'],device='cuda');del data,z,g,h,raw,own
    args=['z','g','h','ownmask']
    for kind in kinds:
        c=config(kind);torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=TunedResidual(len(genes),kind).cuda()
        with torch.no_grad():
            r,_,_=model(**{k:tx[k][:2] for k in args},seen=seen)
            assert torch.count_nonzero(r)==0
            assert torch.equal(generated_profile(tx['mu'][:2],tx['outside'][:2],tx['raw'][:2]+r),generated_profile(tx['mu'][:2],tx['outside'][:2],tx['raw'][:2]))
        opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.02)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2400,eta_min=c['min_lr'])
        begin=time.monotonic();trace=[];torch.cuda.reset_peak_memory_stats()
        for step in range(1,2401):
            model.train();size=max(2,96//len(lines));ids=np.concatenate([rng.choice(byline[line],size,replace=False) for line in lines]);j=torch.as_tensor(ids,device='cuda')
            r,s,back=model(**{k:tx[k][j] for k in args},seen=seen)
            pred=generated_profile(tx['mu'][j],tx['outside'][j],tx['raw'][j]+r);mask=tx['mask'][j]
            err=(pred-tx['y'][j]).square();mse=(err*mask).sum()/mask.sum()
            reg=c['reg']*(r.square()*mask).sum()/mask.sum()+c['bg_reg']*(back.square()*mask).sum()/mask.sum()
            losses=[]
            for li,line in enumerate(lines):
                sl=slice(li*size,(li+1)*size);sub=j[sl];pm=mask[sl].amin(0)
                for target in rows['target'][ids[sl]]:
                    if target in ix:pm[ix[target]]=0
                pe=F.normalize((pred[sl]-tx['control'][sub])*pm,dim=1,eps=1e-8)
                ye=F.normalize((tx['y'][sub]-tx['control'][sub])*pm,dim=1,eps=1e-8)
                losses.append(F.cross_entropy(pe@ye.T/.2,torch.arange(size,device='cuda')))
            disc=torch.stack(losses).mean();direction=torch.zeros((),device='cuda');signal=direction
            if c['signal_mse'] or c['direction']:
                change=tx['y'][j]-tx['control'][j];weight=((change.abs()-.05)/.45).clamp(0,1)*mask
                signal=(err*weight).sum()/weight.sum().clamp_min(1)
                direction=(F.softplus(-change.sign()*(pred-tx['control'][j])/.2)*weight).sum()/weight.sum().clamp_min(1)
            loss=mse+reg+c['disc']*disc+c['signal_mse']*signal+c['direction']*direction
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sch.step()
            if step==1 or step%400==0:
                rec={'step':step,'loss':float(loss.detach()),'mse':float(mse.detach()),'penalty':float(reg.detach()),'discrimination':float(disc.detach()),'direction':float(direction.detach()),'signal_mse':float(signal.detach()),'residual_RMS':float(r.detach().square().mean().sqrt()),'gate_mean':float(s.detach().mean()),'seconds':time.monotonic()-begin}
                trace.append(rec);pd.DataFrame(trace).to_csv(dest/f'{kind}_trace.csv',index=False)
                print('V13_TRAIN',name,seed,kind,json.dumps(rec),flush=True)
            if step in [1200,2400]:
                torch.save({'state_dict':model.state_dict(),'ng':len(genes),'kind':kind,'config':c,'step':step,'seed':seed,'train_contexts':lines,'protocol_sha256':sha(PROTOCOL),'features_sha256':sha(dest/'features.npz'),'optimizer_state':opt.state_dict(),'scheduler_state':sch.state_dict(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},dest/f'{kind}_{step}.pt')
        dump(dest/f'{kind}_complete.json',{'steps':2400,'seed':seed,'train_contexts':lines,'training_rows':len(rows['target']),'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.monotonic()-begin,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'zero_residual_test':True,'feature_origin':str(origin),'config':c})
        del model,opt,sch;torch.cuda.empty_cache()
    del tx;torch.cuda.empty_cache()

def predict(origin,kind,step,inputs):
    ck=torch.load(Path(origin)/f'{kind}_{step}.pt',map_location='cpu',weights_only=False)
    assert ck['config']==config(kind);model=TunedResidual(ck['ng'],kind);model.load_state_dict(ck['state_dict']);model.eval()
    with torch.no_grad():r,s,_=model(**inputs)
    assert r.abs().max()<=config(kind)['bound']+.000001
    return r.numpy(),s.numpy()
