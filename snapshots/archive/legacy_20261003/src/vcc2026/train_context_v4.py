"""Masked union-gene, context-conditioned residual transfer; strictly grouped splits.

K562 response is an observed covariate. Destination outcomes of outer targets,
H1 targets, and the entire held-out destination context never enter fitting.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, time
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.utils.extmath import randomized_svd
from select_transfer_model import read_source, SOURCE_IDS, sha256

ROOT=Path('.')
OUT=Path('runs/context_v4')
MODELS=Path('models/context_v4')
SEED=20260928
LINES=['hepg2','jurkat','rpe1']
TS=50000.


def fold(t):
    return int(hashlib.sha256(('vcc-v4:'+str(t)).encode()).hexdigest(),16)%5


def load_data():
    data={line:read_source(f'data/processed/v1/external_pseudobulk/{sid}.npz') for line,sid in SOURCE_IDS.items()}
    data['h1']=read_source('data/processed/v2/external_pseudobulk/vcc2025_h1.npz')
    official=np.loadtxt('data/raw/vcc2026/validation/files/gene_names.csv',dtype=str,skiprows=1)
    union=set.union(*(set(data[l]['gene_names']) for l in ['k562']+LINES))
    genes=np.array([g for g in official if g in union])
    core=set.intersection(*(set(d['gene_names']) for d in data.values())) & set(official)
    core=np.array([g for g in official if g in core])
    index={g:i for i,g in enumerate(genes)}
    for line,d in data.items():
        own={g:i for i,g in enumerate(d['gene_names'])}
        gi=np.array([own[g] for g in core]); total=d['target_mean_counts'][:,gi].sum(1)
        ctotal=d['control_mean_counts'][gi].sum()
        keep=np.array([j for j,g in enumerate(d['gene_names']) if g in index]);dest=np.array([index[g] for g in d['gene_names'][keep]])
        d['effect']=np.zeros((len(d['target_names']),len(genes)),np.float32)
        d['ctrl']=np.zeros(len(genes),np.float32);d['mask']=np.zeros(len(genes),np.float32)
        d['ctrl'][dest]=np.log1p(TS*d['control_mean_counts'][keep]/ctotal)
        d['effect'][:,dest]=np.log1p(TS*d['target_mean_counts'][:,keep]/np.maximum(total[:,None],1e-12))-d['ctrl'][dest]
        d['mask'][dest]=1
        d['raw_effect']=np.zeros_like(d['effect'])
        d['raw_effect'][:,dest]=np.log2((d['target_mean_counts'][:,keep]+.5)/(d['control_mean_counts'][keep]+.5))
        d['tmap']={t:i for i,t in enumerate(d['target_names'])}
        for i,t in enumerate(d['target_names']):
            if t in index:
                d['effect'][i,index[t]]=0;d['raw_effect'][i,index[t]]=0
    return data,genes,core


class ResidualTransfer(nn.Module):
    def __init__(self,n_genes,input_dim,alpha,conditioned=True):
        super().__init__();self.conditioned=conditioned
        self.alpha=nn.Parameter(torch.tensor(float(alpha)))
        self.encoder=nn.Sequential(nn.Linear(input_dim,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,32))
        self.decoder=nn.Parameter(torch.randn(n_genes,32)*.002)
        self.decoder_context=nn.Parameter(torch.zeros(n_genes,32))
        self.bias=nn.Parameter(torch.zeros(n_genes))
        self.bias_context=nn.Parameter(torch.zeros(n_genes))
        self.gate=nn.Parameter(torch.tensor(0.))

    def forward(self,x,z,g,seen):
        q=self.encoder(z)
        residual=q@self.decoder.T+self.bias
        base=self.alpha.clamp(0,2)*x
        if self.conditioned:
            residual=residual+(q@self.decoder_context.T+self.bias_context)*g
            base=base*(1+self.gate.tanh()*.5*g)
        return base+residual*seen


def prepare_rows(data,genes,lines,excluded):
    source=data['k562']; st=source['tmap']; gp={g:i for i,g in enumerate(genes)}
    x=[];y=[];mask=[];g=[];labels=[];contexts=[];rawx=[];rawy=[]
    for line in lines:
        d=data[line]
        for i,t in enumerate(d['target_names']):
            if t not in st or t in excluded or d['target_n_cells'][i]<30 or source['target_n_cells'][st[t]]<30:continue
            si=st[t];rel=source['target_n_cells'][si]/(source['target_n_cells'][si]+50)
            x.append(source['effect'][si]*rel);y.append(d['effect'][i]);m=d['mask'].copy()
            if t in gp:m[gp[t]]=0
            mask.append(m);g.append(np.clip((d['ctrl']-source['ctrl'])/3,-2,2))
            labels.append(t);contexts.append(line)
            rawx.append(source['raw_effect'][si]*rel);rawy.append(d['raw_effect'][i])
    return {'x':np.array(x),'y':np.array(y),'mask':np.array(mask),'g':np.array(g),
            'target':np.array(labels),'line':np.array(contexts),'rawx':np.array(rawx),'rawy':np.array(rawy)}


def target_features(data,targets,genes,basis,center,scale,emean,estd):
    source=data['k562'];st=source['tmap'];rows=np.array([st[t] for t in targets]);n=source['target_n_cells'][rows]
    x=source['effect'][rows]*(n/(n+50))[:,None]
    emb=np.zeros((len(targets),64),np.float32);found=np.zeros(len(targets),bool)
    with h5py.File('data/raw/external/slim_embeddings/gene_string_embeddings.v0.3.h5','r') as f:
        for i,t in enumerate(targets):
            if t in f:emb[i]=f[t][()];found[i]=True
    if emean is None:
        emean=emb[found].mean(0);estd=np.maximum(emb[found].std(0),.05)
    emb[found]=(emb[found]-emean)/estd
    latent=((x-center)@basis.T)/scale
    z=np.c_[latent,emb, np.log1p(n)/10, np.sqrt(np.mean(x*x,axis=1))].astype(np.float32)
    return x,z,emean,estd


def fit_fold(name,train_lines,data,genes,core,steps=1200):
    out=MODELS/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        print('reuse',name,flush=True);return
    excluded=set(data['h1']['target_names'])
    rows=prepare_rows(data,genes,train_lines,excluded)
    splits=np.array([fold(t) for t in rows['target']]);tr=splits>=2;va=splits==1
    assert not set(rows['target'][tr])&set(rows['target'][~tr])
    assert not set(rows['target'][tr])&excluded
    train_targets=sorted(set(rows['target'][tr]));source=data['k562'];si=np.array([source['tmap'][t] for t in train_targets])
    n=source['target_n_cells'][si];x=source['effect'][si]*(n/(n+50))[:,None];center=x.mean(0)
    _,_,basis=randomized_svd((x-center).astype(np.float32),n_components=64,n_iter=4,random_state=SEED)
    scale=np.maximum(((x-center)@basis.T).std(0),.05)
    _,_,emean,estd=target_features(data,train_targets,genes,basis,center,scale,None,None)
    _,z,_,_=target_features(data,rows['target'],genes,basis,center,scale,emean,estd)
    w=np.zeros(len(rows['x']),np.float64)
    for line in train_lines:w[tr & (rows['line']==line)]=1/max(1,sum(tr & (rows['line']==line)))
    wm=w[:,None]*rows['mask'];xx=np.sum(wm*rows['x']**2);xy=np.sum(wm*rows['x']*rows['y'])
    alpha=float(np.clip(xy/xx,0,2))
    rawalpha=float(np.clip(np.sum(wm*rows['rawx']*rows['rawy'])/np.sum(wm*rows['rawx']**2),0,1.5))
    seen=(rows['mask'][tr].sum(0)>0).astype(np.float32)
    mean=np.divide(np.sum(wm*rows['y'],axis=0),wm.sum(0),out=np.zeros(len(genes)),where=wm.sum(0)>0).astype(np.float32)
    np.savez_compressed(out/'features.npz',genes=genes,core=core,basis=basis,center=center,scale=scale,
        string_mean=emean,string_std=estd,seen=seen,mean=mean,alpha=alpha,raw_alpha=rawalpha)
    pd.DataFrame({'target':rows['target'],'line':rows['line'],'fold':splits,
                  'used_for_fit':tr,'used_for_checkpoint':va}).to_csv(out/'groups.csv',index=False)
    metadata={'fold':name,'training_contexts':train_lines,'training_target_context_rows':int(tr.sum()),
        'training_unique_targets':len(train_targets),'checkpoint_target_context_rows':int(va.sum()),
        'genes':len(genes),'training_measured_genes':int(seen.sum()),'core_normalization_genes':len(core),
        'alpha':alpha,'raw_alpha':rawalpha,'excluded_h1_targets':sorted(excluded),'seed':SEED,'steps':steps}
    (out/'config.json').write_text(json.dumps(metadata,indent=2))
    tx={k:torch.as_tensor(rows[k],device='cuda') for k in ['x','y','mask','g']}
    tz=torch.as_tensor(z,device='cuda');ts=torch.as_tensor(seen,device='cuda')
    tind={line:np.flatnonzero(tr & (rows['line']==line)) for line in train_lines}
    vid=np.flatnonzero(va);begin=time.monotonic()
    for variant,conditioned in [('residual_no_context',False),('context_residual',True)]:
        if (out/f'{variant}.pt').exists() and (out/f'{variant}_trace.csv').exists():continue
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
        model=ResidualTransfer(len(genes),z.shape[1],alpha,conditioned).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.02)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps,eta_min=.00005)
        best=np.inf;trace=[]
        for step in range(1,steps+1):
            model.train();ids=np.concatenate([rng.choice(tind[l],max(1,128//len(train_lines)),replace=True) for l in train_lines])
            ids=torch.as_tensor(ids,device='cuda');p=model(tx['x'][ids],tz[ids],tx['g'][ids],ts)
            m=tx['mask'][ids];loss=(((p-tx['y'][ids])**2)*m).sum()/m.sum()
            # Penalize deviations from the established source-response model.
            reg=.0005*(model.decoder.square().mean()+model.decoder_context.square().mean())
            loss=loss+reg
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step()
            if step==1 or step%100==0 or step==steps:
                model.eval();scores=[]
                with torch.no_grad():
                    for line in train_lines:
                        ids0=vid[rows['line'][vid]==line];err=mass=0.
                        for b in range(0,len(ids0),128):
                            j=torch.as_tensor(ids0[b:b+128],device='cuda');pp=model(tx['x'][j],tz[j],tx['g'][j],ts)
                            err+=float((((pp-tx['y'][j])**2)*tx['mask'][j]).sum());mass+=float(tx['mask'][j].sum())
                        scores.append(err/mass)
                vl=float(np.mean(scores));rec={'step':step,'training_loss':float(loss.detach()),'validation_mse':vl,
                    'seconds':round(time.monotonic()-begin,2),'learned_alpha':float(model.alpha.detach())}
                trace.append(rec)
                if vl<best:
                    best=vl;torch.save({'state_dict':model.state_dict(),'conditioned':conditioned,'input_dim':z.shape[1],
                        'n_genes':len(genes),'alpha':alpha,'step':step,'validation_mse':vl},out/f'{variant}.pt')
                pd.DataFrame(trace).to_csv(out/f'{variant}_trace.csv',index=False)
                print(json.dumps({'fold':name,'model':variant,**rec}),flush=True)
        del model;torch.cuda.empty_cache()
    (out/'complete.json').write_text(json.dumps({**metadata,'elapsed_seconds':time.monotonic()-begin},indent=2))


def predict(data,genes,core,foldname,line,targets,control_genes=None,control_means=None):
    path=MODELS/foldname
    with np.load(path/'features.npz') as f:params={k:f[k] for k in f.files}
    x,z,_,_=target_features(data,targets,genes,params['basis'],params['center'],params['scale'],params['string_mean'],params['string_std'])
    src=data['k562'];gmap={g:i for i,g in enumerate(genes)}
    if control_genes is None:
        ctrl=data[line]['ctrl'];mu=data[line]['control_mean_counts'];control_genes=data[line]['gene_names']
    else:
        mu=control_means;mp={g:i for i,g in enumerate(control_genes)};tot=sum(mu[mp[g]] for g in core if g in mp)
        ctrl=np.zeros(len(genes),np.float32)
        for i,g in enumerate(genes):
            if g in mp:ctrl[i]=np.log1p(TS*mu[mp[g]]/tot)
    g=np.tile(np.clip((ctrl-src['ctrl'])/3,-2,2),(len(targets),1)).astype(np.float32)
    out={'shrink_only':x*float(params['alpha']), 'mean_response':np.tile(params['mean'],(len(targets),1)),
         'zero':np.zeros_like(x)}
    for variant in ['residual_no_context','context_residual']:
        ck=torch.load(path/f'{variant}.pt',map_location='cpu',weights_only=False)
        model=ResidualTransfer(ck['n_genes'],ck['input_dim'],ck['alpha'],ck['conditioned'])
        model.load_state_dict(ck['state_dict']);model.eval()
        with torch.no_grad():out[variant]=model(torch.as_tensor(x),torch.as_tensor(z),torch.as_tensor(g),torch.as_tensor(params['seen'])).numpy()
    if (path/'context_se_frozen.pt').exists():
        from train_se_residual_v4 import predict_se
        out['context_se_frozen']=predict_se(foldname,line,x,z,g,params['seen'])
    # Original raw-UMI baseline is refitted on the same non-test target/context rows.
    si=np.array([src['tmap'][t] for t in targets]);n=src['target_n_cells'][si]
    raw=src['raw_effect'][si]*(n/(n+50))[:,None]*float(params['raw_alpha'])
    out['v1_raw']=np.clip(raw,-3,3)
    for variant,arr in out.items():
        for i,t in enumerate(targets):
            if t in gmap:arr[i,gmap[t]]=0
    return out,params


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1200);ap.add_argument('--fold',default='all');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65);torch.set_float32_matmul_precision('high')
    data,genes,core=load_data()
    protocol={'version':'v4.0','created_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
        'modeled_output_genes':len(genes),'normalization_core_genes':len(core),
        'gene_rule':'Union of official genes measured unambiguously in historical K562/HepG2/Jurkat/RPE1; absent measurements masked, never used as zero labels',
        'effect':'log1p(50000 * mean_raw_UMI / sum(mean_raw_UMI on fixed shared core)) minus control',
        'source':'K562 observed response times n/(n+50); minimum 30 original source cells',
        'validation':'Whole destination context held out. SHA256(vcc-v4:target) modulo 5: 0 outer evaluation, 1 checkpoint validation, 2/3/4 fitting. All H1 targets excluded from fitting/checkpoint selection.',
        'data_interpretation':'Contexts and target identities are grouped. Cells are subsamples, not independent biological replicates. H1 remains development validation.',
        'architecture':'K562 source response plus rank-32 residual MLP using training-only source SVD64 and STRING64. Gene-wise control expression differences condition baseline/residual. No context one-hot.',
        'models':['v1_raw','shrink_only','residual_no_context','context_residual','mean_response','zero'],
        'steps_per_model':args.steps,'checkpoint_criterion':'equal-context masked normalized-expression MSE on fold1 targets of training contexts',
        'samplers':['round','posterior'],'selection':'Inspect all six official raw metrics and equal-metric local rank composite across held-out contexts. Rank composite is not an official scaled score.',
        'full_panel_selection':'No leaderboard submission in this run; H1 alone cannot select winner',
        'no_unseen_wide_gene_recovery':'Raw H5AD inspection found no wider raw/layers matrix in historical sources',
        'inputs':{l:sha256(f'data/processed/v1/external_pseudobulk/{sid}.npz') for l,sid in SOURCE_IDS.items()},
        'code_sha256':sha256(__file__)}
    p=OUT/'protocol.json'
    if not p.exists():p.write_text(json.dumps(protocol,indent=2)+'\n')
    np.savetxt(OUT/'gene_axis.txt',genes,fmt='%s');np.savetxt(OUT/'normalization_core.txt',core,fmt='%s')
    folds={f'holdout_{line}':[l for l in LINES if l!=line] for line in LINES};folds['final']=LINES
    for name,ls in folds.items():
        if args.fold in ['all',name]:fit_fold(name,ls,data,genes,core,args.steps)
    print('TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
