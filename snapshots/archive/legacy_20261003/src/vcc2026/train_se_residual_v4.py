"""Add frozen SE control representation to v4, preserving existing checkpoints."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import torch
from train_context_v4 import (load_data,prepare_rows,target_features,ResidualTransfer,fold,MODELS,SEED,sha256)

AUDIT=Path('runs/context_v4/pretrained_audit')


def representation(train_lines):
    cells={}
    for line in ['hepg2','jurkat','rpe1','h1']:
        with np.load(AUDIT/f'{line}_se_cells.npz') as z:cells[line]=z['embedding'].copy()
    rng=np.random.default_rng(SEED);projection=np.linalg.qr(rng.normal(size=(2048,32)))[0].astype(np.float32)
    projected={line:e@projection for line,e in cells.items()}
    train=np.concatenate([projected[l] for l in train_lines])
    center=train.mean(0);scale=np.maximum(train.std(0),.005)
    h={l:((e-center)/scale).mean(0).astype(np.float32) for l,e in projected.items()}
    return h,projection,center,scale


def predict_se(foldname,line,x,z,g,seen):
    path=MODELS/foldname
    ck=torch.load(path/'context_se_frozen.pt',map_location='cpu',weights_only=False)
    model=ResidualTransfer(ck['n_genes'],ck['input_dim'],ck['alpha'],True)
    model.load_state_dict(ck['state_dict']);model.eval()
    with np.load(path/'se_features.npz') as p:
        lines=list(p['lines']);h=p['context_features'][lines.index(line)]
    zz=np.c_[z,np.tile(h,(len(z),1))].astype(np.float32)
    with torch.no_grad():p=model(torch.as_tensor(x),torch.as_tensor(zz),torch.as_tensor(g),torch.as_tensor(seen)).numpy()
    return p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1200);args=ap.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65);torch.set_float32_matmul_precision('high')
    data,genes,core=load_data()
    for name in ['holdout_hepg2','holdout_jurkat','holdout_rpe1','final']:
        path=MODELS/name;dest=path/'context_se_frozen.pt'
        if (path/'se_complete.json').exists():continue
        cfg=json.loads((path/'config.json').read_text());ls=cfg['training_contexts']
        assert cfg['steps']==args.steps
        with np.load(path/'features.npz') as z:params={k:z[k] for k in z.files}
        rows=prepare_rows(data,genes,ls,set(data['h1']['target_names']))
        splits=np.array([fold(t) for t in rows['target']]);tr=splits>=2;va=splits==1
        previous=pd.read_csv(path/'groups.csv')
        assert np.array_equal(rows['target'],previous.target) and np.array_equal(rows['line'],previous.line)
        assert np.array_equal(tr,previous.used_for_fit) and np.array_equal(va,previous.used_for_checkpoint)
        x,z,_,_=target_features(data,rows['target'],genes,params['basis'],params['center'],params['scale'],params['string_mean'],params['string_std'])
        h,proj,hcenter,hscale=representation(ls);hlines=list(h)
        np.savez_compressed(path/'se_features.npz',lines=np.array(hlines),context_features=np.array([h[l] for l in hlines]),
            projection=proj,center=hcenter,scale=hscale,standardization_training_lines=np.array(ls))
        zz=np.c_[z,np.array([h[l] for l in rows['line']])].astype(np.float32)
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
        model=ResidualTransfer(len(genes),zz.shape[1],float(params['alpha']),True).cuda()
        # Same initialization for all parameters shared with the expression-only architecture.
        torch.manual_seed(SEED);reference=ResidualTransfer(len(genes),z.shape[1],float(params['alpha']),True)
        state=reference.state_dict();expanded=model.state_dict()
        for k,v in state.items():
            if k=='encoder.0.weight':
                expanded[k][:,:z.shape[1]]=v.to('cuda');expanded[k][:,z.shape[1]:]=0.
            else:expanded[k].copy_(v.to('cuda'))
        model.load_state_dict(expanded);del reference,state,expanded
        # Zero initialization for h columns starts at exactly the same source/residual model.
        torch.manual_seed(SEED)
        tx={k:torch.as_tensor(rows[k],device='cuda') for k in ['x','y','mask','g']}
        tz=torch.as_tensor(zz,device='cuda');ts=torch.as_tensor(params['seen'],device='cuda')
        tind={l:np.flatnonzero(tr & (rows['line']==l)) for l in ls};vid=np.flatnonzero(va)
        opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.02)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.steps,eta_min=.00005)
        best=np.inf;trace=[];begin=time.monotonic()
        for step in range(1,args.steps+1):
            model.train();ids=np.concatenate([rng.choice(tind[l],max(1,128//len(ls)),replace=True) for l in ls])
            ids=torch.as_tensor(ids,device='cuda');p=model(tx['x'][ids],tz[ids],tx['g'][ids],ts);m=tx['mask'][ids]
            loss=(((p-tx['y'][ids])**2)*m).sum()/m.sum()+.0005*(model.decoder.square().mean()+model.decoder_context.square().mean())
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();schedule.step()
            if step==1 or step%100==0 or step==args.steps:
                model.eval();scores=[]
                with torch.no_grad():
                    for l in ls:
                        idx=vid[rows['line'][vid]==l];err=mass=0.
                        for b in range(0,len(idx),128):
                            j=torch.as_tensor(idx[b:b+128],device='cuda');pp=model(tx['x'][j],tz[j],tx['g'][j],ts)
                            err+=float((((pp-tx['y'][j])**2)*tx['mask'][j]).sum());mass+=float(tx['mask'][j].sum())
                        scores.append(err/mass)
                val=float(np.mean(scores));rec={'step':step,'training_loss':float(loss.detach()),'validation_mse':val,
                    'seconds':round(time.monotonic()-begin,2),'learned_alpha':float(model.alpha.detach())};trace.append(rec)
                if val<best:
                    best=val;torch.save({'state_dict':model.state_dict(),'n_genes':len(genes),'input_dim':zz.shape[1],
                        'alpha':float(params['alpha']),'step':step,'validation_mse':val},dest)
                pd.DataFrame(trace).to_csv(path/'context_se_frozen_trace.csv',index=False)
                print(json.dumps({'fold':name,'model':'context_se_frozen',**rec}),flush=True)
        meta={'candidate':'context_se_frozen','steps':args.steps,'training_rows':int(tr.sum()),'same_group_split_as_context_v4':True,
            'same_loss_optimizer_lr_schedule_as_context_v4':True,'output_genes':len(genes),'rank':32,'SE_trainable_parameters':0,
            'residual_input_features':int(zz.shape[1]),'added_context_dimensions':32,
            'trainable_residual_parameters':sum(p.numel() for p in model.parameters()),
            'h_projection':'fixed seed orthogonal random projection 2048->32; scale/center fitted on training-context control cells only',
            'SE_hash':json.loads((AUDIT/'encoder_audit.json').read_text())['model_sha256'],
            'generator':'unchanged stochastic rounding; posterior generator comparison stays separate',
            'best_inner_validation_mse':best,'checkpoint_sha256':sha256(dest)}
        (path/'se_complete.json').write_text(json.dumps(meta,indent=2)+'\n')
        del model,tx,tz;torch.cuda.empty_cache()
    print('SE_RESIDUAL_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
