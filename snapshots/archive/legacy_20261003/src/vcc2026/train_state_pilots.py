"""Train two actual Arc State set-to-set models, with group-held-out validation."""
from pathlib import Path
import argparse,json,time,contextlib
import numpy as np
import pandas as pd
import torch
from state.tx.models.state_transition import StateTransitionPerturbationModel

DATA=Path('data/processed/v3/state_cells');SEED=20260927

def model_kwargs(variant):
    hidden=128 if variant=='string' else 256
    return dict(input_dim=1024,hidden_dim=hidden,output_dim=1024,pert_dim=64 if variant=='string' else 128,
      transformer_backbone_key='llama',transformer_backbone_kwargs={'hidden_size':hidden,'intermediate_size':4*hidden,
      'num_hidden_layers':2 if variant=='string' else 4,'num_attention_heads':4 if hidden==128 else 8,
      'num_key_value_heads':4 if hidden==128 else 8,'vocab_size':8,'bidirectional_attention':True},
      cell_set_len=32,embed_key='X_hvg',output_space='gene',dropout=.05,n_encoder_layers=2,n_decoder_layers=2,
      predict_residual=True,distributional_loss='energy',loss='energy')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['string','string_source'],required=True)
    ap.add_argument('--steps',type=int,default=4000);args=ap.parse_args()
    if not (DATA/'complete.json').exists():raise RuntimeError('Data preparation incomplete')
    out=Path(f'models/foundation_v3/state_{args.variant}');out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65);torch.manual_seed(SEED)
    torch.set_float32_matmul_precision('high');rng=np.random.default_rng(SEED)
    with np.load(DATA/'features.npz') as z:
        emb=z['string_embedding'].copy();targets=z['targets'].astype(str)
        if args.variant=='string_source':emb=np.concatenate([emb,z['source_embedding']],axis=1)
    emap={t:emb[i] for i,t in enumerate(targets)};zero=np.zeros(emb.shape[1],np.float32)
    groups=pd.read_csv(DATA/'all_groups.csv');lines=list(groups.line.unique())
    arrays={line:np.load(DATA/f'{line}.npy',mmap_mode='r') for line in lines}
    control={line:groups[(groups.line==line)&(groups.split=='control')].iloc[0] for line in lines}
    train={line:groups[(groups.line==line)&(groups.split=='train')].to_dict('records') for line in lines}
    val=groups[groups.split=='validation'].sample(n=min(128,(groups.split=='validation').sum()),random_state=SEED).to_dict('records')
    def batch(items,random):
        controls=[];pert=[];truth=[]
        for g in items:
            c=control[g['line']];a=arrays[g['line']]
            idx=random.integers(int(c.start),int(c.start+c.n),size=32)
            ti=random.integers(int(g['start']),int(g['start']+g['n']),size=32)
            controls.append(np.asarray(a[idx]));truth.append(np.asarray(a[ti]));pert.append(np.tile(emap.get(g['target'],zero),(32,1)))
        return ({'ctrl_cell_emb':torch.as_tensor(np.concatenate(controls),device='cuda'),
                 'pert_emb':torch.as_tensor(np.concatenate(pert),device='cuda')},
                torch.as_tensor(np.stack(truth),device='cuda'))
    kwargs=model_kwargs(args.variant)
    with contextlib.redirect_stdout((out/'architecture.txt').open('w')):model=StateTransitionPerturbationModel(**kwargs).cuda()
    cfg={'model_class':'state.tx.models.state_transition.StateTransitionPerturbationModel','arc_state_version':'0.11.1',
      'variant':args.variant,'model_kwargs':kwargs,'seed':SEED,'optimizer':'AdamW','learning_rate':.0003,'weight_decay':.0001,
      'maximum_steps':args.steps,'cell_sets_per_step':8,'cells_per_set':32,'control_training_probability':.15,
      'loss':'official energy distance + 0.1 * pseudobulk MSE in log-normalized HVG space',
      'selection':'lowest fixed 128 held-out target/context group loss at steps 250,500,...; no H1 outcomes',
      'precision':'float32 TF32 enabled','trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
      'training_groups':int((groups.split=='train').sum()),'validation_groups_total':int((groups.split=='validation').sum())}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');pd.DataFrame(val).to_csv(out/'validation_groups.csv',index=False)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.steps,eta_min=.00003)
    best=float('inf');trace=[];begin=time.monotonic()
    for step in range(1,args.steps+1):
        model.train();items=[]
        for _ in range(8):
            line=lines[int(rng.integers(len(lines)))]
            items.append(control[line].to_dict() if rng.random()<.15 else train[line][int(rng.integers(len(train[line])))])
        x,y=batch(items,rng);opt.zero_grad(set_to_none=True)
        pred=model(x).reshape(8,32,1024)
        loss=model._compute_distribution_loss(pred,y).mean()+.1*torch.mean((pred.mean(1)-y.mean(1))**2)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();scheduler.step()
        if step%250==0 or step==1:
            model.eval();scores=[];vrng=np.random.default_rng(SEED+1)
            with torch.no_grad():
                for i in range(0,len(val),8):
                    xx,yy=batch(val[i:i+8],vrng);pp=model(xx).reshape(-1,32,1024)
                    scores.append(float((model._compute_distribution_loss(pp,yy).mean()+.1*torch.mean((pp.mean(1)-yy.mean(1))**2)).item()))
            v=float(np.mean(scores));record={'step':step,'train_loss':loss.item(),'validation_loss':v,'seconds':round(time.monotonic()-begin,1),
                'max_gpu_allocated_GB':torch.cuda.max_memory_allocated()/1e9}
            trace.append(record);pd.DataFrame(trace).to_csv(out/'training_trace.csv',index=False);print(json.dumps(record),flush=True)
            if v<best:
                best=v;torch.save({'state_dict':model.state_dict(),'model_kwargs':kwargs,'step':step,'validation_loss':v},out/'best.pt')
    (out/'complete.json').write_text(json.dumps({'best_validation_loss':best,'completed_steps':args.steps,'elapsed_seconds':time.monotonic()-begin},indent=2)+'\n')

if __name__=='__main__':main()
