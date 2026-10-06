"""Frozen submitted V1 with output-bounded zero-initialized residuals.

Only the small residual parameters are optimized. The loss sees the native-axis
normalized reachable means AFTER V1's inversion and multiplier cap.
"""
from pathlib import Path
import argparse,json,time,math
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.utils.extmath import randomized_svd
from train_context_v4 import load_data,target_features
from select_transfer_model import read_source,SOURCE_IDS
from residual_v6_common import *

EXCLUDED={'HSPA14','MKKS','PRRC2B','SOD2','SPATA13','TBCE','C8orf44','CCDC144NL','FCGR2C','PRSS50','ZNF286B'}
STEPS=[600,1200]

def prepare():
    out=ROOT/'training_data';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    genes,core=axes();src=read_source('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz')
    eligible=set(src['target_names'][src['target_n_cells']>=30]);h1=read_source('data/processed/v2/external_pseudobulk/vcc2025_h1.npz');exclude=set(h1['target_names']);counts={}
    for line in CONTEXTS+['orion_hct116']:
        if (out/f'{line}.npz').exists():continue
        if line=='orion_hct116':d=read_source('data/processed/supplemental/orion/HCT116_counts/pseudobulk.npz')
        else:d=read_source(f'data/processed/v1/external_pseudobulk/{SOURCE_IDS[line]}.npz')
        keep=np.array([t in eligible and t not in exclude and fold(t)>=2 and (line!='orion_hct116' or t not in EXCLUDED) for t in d['target_names']])&(d['target_n_cells']>=30)
        targets=d['target_names'][keep];mean=d['target_mean_counts'][keep].astype(np.float32);native=d['gene_names'].astype(str)
        if line=='orion_hct116':
            with np.load('data/processed/supplemental/orion/HCT116_counts/pseudobulk.npz') as z:ctrl=z['target_control_mean_counts'][keep].astype(np.float32)
            masknative=native!='HSPA14';mean=mean[:,masknative];ctrl=ctrl[:,masknative];native=native[masknative]
            p=min(1.,20000/float(d['control_mean_counts'].sum()))
        else:
            ctrl=np.broadcast_to(d['control_mean_counts'].astype(np.float32),mean.shape).copy()
            p=json.loads((ROOT/'inner_panels'/line/'panel.json').read_text())['thinning_probability']
        mean*=p;ctrl*=p;gp={g:i for i,g in enumerate(native)}
        si=np.array([gp[g] for g in genes if g in gp]);di=np.array([i for i,g in enumerate(genes) if g in gp]);corei=np.array([gp[g] for g in core if g in gp])
        mu=np.zeros((len(targets),len(genes)),np.float32);y=np.zeros_like(mu);c=np.zeros_like(mu);mask=np.zeros(mu.shape,np.uint8)
        mu[:,di]=ctrl[:,si];y[:,di]=np.log1p(mean[:,si]*50000/mean.sum(1)[:,None]);c[:,di]=np.log1p(ctrl[:,si]*50000/ctrl[:,corei].sum(1)[:,None]);mask[:,di]=1
        ix={g:i for i,g in enumerate(genes)}
        for j,t in enumerate(targets):
            if t in ix:mask[j,ix[t]]=0
        outside=np.maximum(ctrl.sum(1)-mu.sum(1),0).astype(np.float32)
        np.savez_compressed(out/f'{line}.npz',targets=targets,mu=mu,y=y,c=c,mask=mask,outside=outside,thinning_probability=p)
        counts[line]=len(targets);print('TRAINING_DATA',line,len(targets),flush=True)
    for line in CONTEXTS+['orion_hct116']:
        with np.load(out/f'{line}.npz') as z:counts[line]=len(z['targets'])
    assert counts['orion_hct116']==5253
    dump(out/'complete.json',{'rows':counts,'genes':len(genes),'supervision':'measured native-axis log1p normalized mean; no zero labels for unmeasured genes','HCT116_same_batch_controls':True,'outer_fold0_and_inner_fold1_fit_rows':0})

class BoundedResidual(nn.Module):
    def __init__(self,ng,nz):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(nz,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,32))
        self.decoder=nn.Parameter(torch.zeros(ng,32));self.background_decoder=nn.Parameter(torch.zeros(ng,32))
        self.gate=nn.Linear(nz+64,1);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,math.log(.1/.9))
    def forward(self,z,g,h,seen,ownmask=None):
        q=self.encoder(z);base=q@self.decoder.T;back=(q@self.background_decoder.T)*g
        s=torch.sigmoid(self.gate(torch.cat([z,h],dim=1)))
        r=s*.75*torch.tanh((base+back)/.75)*seen
        if ownmask is not None:r=r*ownmask
        return r,s,back*s

def generated_profile(mu,outside,raw):
    wanted=torch.clamp((mu+.5)*torch.exp2(raw)-.5,min=0)
    # Exactly the current multiplicative generator's reachable mean; missing and
    # zero-control genes are not misrepresented as observed nonzero activation.
    ratio=torch.where(mu>0,wanted/torch.clamp(mu,min=1e-12),torch.ones_like(mu)).clamp(0,16)
    counts=mu*ratio;total=counts.sum(1)+outside
    return torch.log1p(counts*50000/total[:,None].clamp_min(1e-12))

def load_rows(lines):
    values={k:[] for k in ['target','line','mu','y','c','mask','outside']}
    for line in lines:
        with np.load(ROOT/'training_data'/f'{line}.npz') as z:
            values['target'].append(z['targets']);values['line'].append(np.array([line]*len(z['targets'])))
            for k in ['mu','y','c','mask','outside']:values[k].append(z[k])
    return {k:np.concatenate(v) for k,v in values.items()}

def fit(name,lines,steps=1200):
    dest=MODEL/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    data,genes,core=load_data();rows=load_rows(lines);targets=sorted(set(rows['target']));src=data['k562'];jj=np.array([src['tmap'][t] for t in targets]);n=src['target_n_cells'][jj]
    x=src['effect'][jj]*(n/(n+50))[:,None];center=x.mean(0);_,_,basis=randomized_svd((x-center).astype(np.float32),n_components=64,n_iter=4,random_state=SEED)
    scale=np.maximum(((x-center)@basis.T).std(0),.05);_,_,em,es=target_features(data,targets,genes,basis,center,scale,None,None)
    _,z,_,_=target_features(data,rows['target'],genes,basis,center,scale,em,es)
    g=np.clip((rows['c']-src['ctrl'])/3,-2,2).astype(np.float32);h=(g@basis.T/np.sqrt(len(genes))).astype(np.float32)
    raw=frozen_v1(rows['target'],genes);seen=(rows['mask'].sum(0)>0).astype(np.float32)
    assert all(fold(t)>=2 and t not in set(data['h1']['target_names']) for t in rows['target'])
    np.savez_compressed(dest/'features.npz',basis=basis,center=center,scale=scale,string_mean=em,string_std=es,seen=seen,source_ctrl=src['ctrl'])
    pd.DataFrame({'target':rows['target'],'context':rows['line'],'fold':[fold(t) for t in rows['target']]}).to_csv(dest/'training_rows.csv',index=False)
    ix={g:i for i,g in enumerate(genes)};own=np.ones(rows['mask'].shape,np.float32)
    for i,t in enumerate(rows['target']):
        if t in ix:own[i,ix[t]]=0
    tx={k:torch.as_tensor(rows[k],device='cuda',dtype=torch.float32) for k in ['mu','y','mask','outside']}
    tx.update(raw=torch.as_tensor(raw,device='cuda'),g=torch.as_tensor(g,device='cuda'),h=torch.as_tensor(h,device='cuda'),z=torch.as_tensor(z,device='cuda'),own=torch.as_tensor(own,device='cuda'))
    ts=torch.as_tensor(seen,device='cuda');byline={l:np.flatnonzero(rows['line']==l) for l in lines};del data,raw,g,h,z,own
    ctrlprofile=torch.log1p(tx['mu']*50000/(tx['mu'].sum(1)+tx['outside'])[:,None])
    baselines=generated_profile(tx['mu'],tx['outside'],tx['raw'])
    budgets={'A':steps,'B':steps}
    for arm in ['A','B']:
        if (dest/f'{arm}_complete.json').exists():continue
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED);model=BoundedResidual(len(genes),tx['z'].shape[1]).cuda()
        with torch.no_grad():
            r,_,_=model(tx['z'][:2],tx['g'][:2],tx['h'][:2],ts,tx['own'][:2]);assert torch.count_nonzero(r)==0
            assert torch.equal(generated_profile(tx['mu'][:2],tx['outside'][:2],tx['raw'][:2]+r),baselines[:2])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.02);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1200,eta_min=.00005)
        trace=[];begin=time.monotonic()
        for step in range(1,steps+1):
            model.train();size=max(2,96//len(lines));ids=np.concatenate([rng.choice(byline[l],size,replace=False) for l in lines]);j=torch.as_tensor(ids,device='cuda')
            r,s,back=model(tx['z'][j],tx['g'][j],tx['h'][j],ts,tx['own'][j]);pred=generated_profile(tx['mu'][j],tx['outside'][j],tx['raw'][j]+r)
            m=tx['mask'][j];mse=((pred-tx['y'][j]).square()*m).sum()/m.sum();reg=.1*(r.square()*m).sum()/m.sum()+.5*(back.square()*m).sum()/m.sum()
            disc=torch.zeros((),device='cuda')
            if arm=='B':
                # Only within-background negatives; identical data/order as A.
                losses=[]
                for li,l in enumerate(lines):
                    sl=slice(li*size,(li+1)*size);sub=j[sl];pm=m[sl].amin(0)
                    for t in rows['target'][ids[sl]]:
                        if t in ix:pm[ix[t]]=0
                    pe=(pred[sl]-ctrlprofile[sub])*pm;ye=(tx['y'][sub]-ctrlprofile[sub])*pm
                    pe=nn.functional.normalize(pe,dim=1,eps=1e-8);ye=nn.functional.normalize(ye,dim=1,eps=1e-8)
                    logits=pe@ye.T/.2;losses.append(nn.functional.cross_entropy(logits,torch.arange(size,device='cuda')))
                disc=torch.stack(losses).mean()
            loss=mse+reg+.05*disc;assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step()
            if step==1 or step%100==0:
                rec={'step':step,'loss':float(loss.detach()),'mse':float(mse.detach()),'residual_penalty':float(reg.detach()),'discrimination':float(disc.detach()),'residual_RMS':float(r.detach().square().mean().sqrt()),'gate_mean':float(s.detach().mean()),'seconds':time.monotonic()-begin};trace.append(rec)
                pd.DataFrame(trace).to_csv(dest/f'{arm}_trace.csv',index=False);print('TRAIN_V6',name,arm,json.dumps(rec),flush=True)
            if step in STEPS or step==steps:
                torch.save({'state_dict':model.state_dict(),'ng':len(genes),'nz':tx['z'].shape[1],'arm':arm,'step':step,'train_contexts':lines,'V1_config_sha256':sha('configs/baseline_v1.json'),'protocol_sha256':sha('configs/residual_v6_protocol.json')},dest/f'{arm}_{step}.pt')
        dump(dest/f'{arm}_complete.json',{'steps':steps,'training_rows':len(rows['target']),'seconds':time.monotonic()-begin});del model;torch.cuda.empty_cache()
    dump(dest/'complete.json',{'training_contexts':lines,'n_rows':len(rows['target']),'n_targets':len(targets),'steps':steps,'frozen_V1':True,'zero_residual_test':True})
    del tx;torch.cuda.empty_cache()

def predict(name,arm,step,path):
    data,genes,core=load_data();path=Path(path);info=json.loads((path/'panel.json').read_text());targets=info['targets'];native=np.loadtxt(path/'genes.txt',dtype=str)
    pool=sparse.load_npz(path/'generation_pool.npz');mu=np.asarray(pool.mean(0)).ravel().astype(np.float64);dest=MODEL/name
    with np.load(dest/'features.npz') as f:p={k:f[k] for k in f.files}
    _,z,_,_=target_features(data,targets,genes,p['basis'],p['center'],p['scale'],p['string_mean'],p['string_std'])
    gm={g:i for i,g in enumerate(native)};total=sum(mu[gm[g]] for g in core if g in gm);ctrl=np.zeros(len(genes),np.float32)
    for i,gene in enumerate(genes):
        if gene in gm:ctrl[i]=np.log1p(mu[gm[gene]]*50000/total)
    g=np.tile(np.clip((ctrl-p['source_ctrl'])/3,-2,2),(len(targets),1)).astype(np.float32);h=(g@p['basis'].T/np.sqrt(len(genes))).astype(np.float32)
    ck=torch.load(dest/f'{arm}_{step}.pt',map_location='cpu',weights_only=False);model=BoundedResidual(ck['ng'],ck['nz']);model.load_state_dict(ck['state_dict']);model.eval()
    with torch.no_grad():r,s,_=model(torch.as_tensor(z),torch.as_tensor(g),torch.as_tensor(h),torch.as_tensor(p['seen']));r=r.numpy()
    gp={g:i for i,g in enumerate(genes)}
    for i,t in enumerate(targets):
        if t in gp:r[i,gp[t]]=0
    return frozen_v1(targets,genes)+r,r,s.numpy()

def inner_build():
    for outer in CONTEXTS:
        for inner in [l for l in CONTEXTS if l!=outer]:
            name=f'outer_{outer}__inner_{inner}';fitlines=[l for l in CONTEXTS if l not in [outer,inner]]+['orion_hct116'];fit(name,fitlines)
            path=ROOT/'inner_panels'/inner
            for arm in ['A','B']:
                for step in STEPS:
                    tag=f'{name}__{arm}_{step}'
                    if (path/f'pred__{tag}.h5ad').exists():continue
                    effect,r,s=predict(name,arm,step,path);generate(path,tag,effect)
                    np.savez_compressed(path/f'effects__{tag}.npz',effect=effect,residual=r,gate=s)
                    print('INNER_PREDICTED',tag,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','inner','fit']);ap.add_argument('--name');ap.add_argument('--lines',nargs='+');ap.add_argument('--steps',type=int,default=1200);args=ap.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65);torch.set_float32_matmul_precision('high')
    if args.mode=='prepare':prepare()
    elif args.mode=='inner':inner_build()
    else:fit(args.name,args.lines,args.steps)
