"""Actual V25 role-specific count decoder and prior-anchored latent training."""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F
from residual_v6_common import axes as original_axes, dump, sha
from prepare_response_inputs_v23 import load_training_inputs, predict_inputs, lines_for
from prepare_set_response_v22 import load_cache, _control_counts
from batch_cells_v24 import sample_sets
from latent_flow_v25 import (LatentResponse, NBDecoder, PCAInverseDecoder, masked_nb_nll,
    readout_delta, merge_prior, flow_matching_loss, deterministic_endpoint_loss)

PROTOCOL=Path('configs/latent_flow_v25_protocol.json')
P=json.loads(PROTOCOL.read_text(encoding='utf-8'))
ROOT=Path('runs/latent_flow_v25');MODEL=Path('models/latent_flow_v25')
CACHE=Path(P['cache']['root'])

@lru_cache(maxsize=1)
def axes():
    return original_axes()

def model_cfg():
    c=P['model']
    return {**c,'latent_dim':c['cell_dim'],'condition_hidden':c['hidden'],
            'velocity_hidden':c['width'],'z_dim':130,'h_dim':64,
            'velocity_layers':2,'decoder_layers':2}

def runtime():
    torch.set_num_threads(int(os.environ.get('VCC_TRAIN_THREADS',os.cpu_count() or 1)))
    torch.cuda.set_per_process_memory_fraction(1.0)
    torch.set_float32_matmul_precision('high')

def directory(role,candidate,seed):
    candidate='flow_nb' if candidate=='flow_pca' else candidate
    return MODEL/role/f'seed_{seed}'/candidate

@lru_cache(maxsize=1)
def encoder_arrays():
    with np.load('data/processed/set_response_v22/encoder.npz') as z:return {k:z[k] for k in z.files}

@lru_cache(maxsize=1)
def controls():
    cache=load_cache();encoder=encoder_arrays();cache['input_counts']={};cache['pool_input_means']={}
    for line in cache['controls']:
        counts=_control_counts(line,encoder['genes']).tocsr()
        assert len(cache['controls'][line])==counts.shape[0]
        cache['input_counts'][line]=counts
        cache['pool_input_means'][line]=np.asarray(counts.mean(0)).ravel().astype(np.float32)
    return cache

@lru_cache(maxsize=4)
def count_cells(line):
    path=CACHE/line
    counts=sparse.load_npz(path/'counts.npz').tocsr()
    pc=np.load(path/'cells.npy',mmap_mode='r')
    metadata=pd.read_csv(path/'metadata.csv',dtype={'target':str,'guide':str,'batch':str})
    mask=np.load(path/'measurement_mask.npy').astype(bool)
    assert counts.shape==(len(metadata),len(axes()[0])) and pc.shape==(len(metadata),64)
    targets={str(t):np.asarray(ids) for t,ids in metadata.groupby('target',sort=False).groups.items() if str(t)!='non-targeting'}
    val=metadata['decoder_validation'].astype(str).str.lower().isin(['true','1']).to_numpy()
    library=np.asarray(counts.sum(1)).ravel().astype(np.float32)
    assert np.all(library>0) and np.all(counts.data>=0) and np.array_equal(counts.data,np.rint(counts.data))
    assert np.array_equal(library,metadata['library_output'].to_numpy(np.float32))
    return {'counts':counts,'pc':pc,'metadata':metadata,'mask':mask,'library':library,
            'targets':targets,'train':np.flatnonzero(~val),'validation':np.flatnonzero(val)}

def decoder_lines(role):
    return ['k562']+[x for x in lines_for(role) if x!='orion_hct116']

def checkpoint(dest,model,opt,scheduler,step,rngs,trace,exposures,elapsed,manifest):
    value={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
           'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'step':step,
           'rngs':{k:r.bit_generator.state for k,r in rngs.items()},'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state(),'trace':trace,'exposures':exposures,
           'elapsed':elapsed,'manifest':manifest}
    torch.save(value,dest/'resume.tmp.pt');(dest/'resume.tmp.pt').replace(dest/'resume.pt')
    return value

def restore(dest,model,opt,scheduler,rngs,manifest):
    if not (dest/'resume.pt').exists():return 1,[],None,0.
    ck=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
    assert ck['manifest']==manifest
    model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
    for key,rng in rngs.items():rng.bit_generator.state=ck['rngs'][key]
    torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
    return ck['step']+1,ck['trace'],ck['exposures'],ck['elapsed']

def provenance(role,candidate,seed,config,model):
    return {'protocol_sha256':sha(PROTOCOL),'trainname':role,'candidate':candidate,'seed':seed,
        'resolved_model_config':model_cfg(),'parameters':sum(p.numel() for p in model.parameters()),
        'steps':config['steps'],'batch_size':config['batch_size'],
        'trainer_sha256':sha(Path(__file__)),'model_sha256':sha('src/vcc2026/latent_flow_v25.py'),
        'cache_manifest_sha256':sha(CACHE/'complete.json'),
        'encoder_sha256':sha('data/processed/set_response_v22/encoder.npz'),
        'encoder_matmul_precision':'highest float32; model matmul precision high',
        'training_contexts':lines_for(role),'decoder_contexts':decoder_lines(role),
        'whole_destination_heldout':True,'all_response_targets_fold234_nonH1':True}

def manifest_write(dest,value):
    if (dest/'manifest.json').exists():assert json.loads((dest/'manifest.json').read_text())==value
    else:dump(dest/'manifest.json',value)

def resources_begin(dest):
    path=dest/'resource_attempts.json';v=json.loads(path.read_text()) if path.exists() else []
    v.append({'CPU_threads':torch.get_num_threads(),'full_GPU_memory_allowed':True,
              'resume':(dest/'resume.pt').exists(),'started_unix':time.time()});dump(path,v);return v

def decoder_diagnostics(model,data,dest):
    result={}
    model.eval()
    with torch.no_grad():
        for line,source in data.items():
            valid=source['validation'];rng=np.random.default_rng(P['cache']['seed']+sum(map(ord,line)))
            ids=rng.choice(valid,min(512,len(valid)),replace=False)
            pc=torch.as_tensor(np.asarray(source['pc'][ids]),device='cuda')
            truth=torch.as_tensor(source['counts'][ids].toarray(),dtype=torch.float32,device='cuda')
            lib=torch.as_tensor(source['library'][ids],device='cuda')
            mask=torch.as_tensor(source['mask'],device='cuda')
            decoded=model(pc,lib,mask)
            loss=masked_nb_nll(truth,decoded.mu,decoded.theta,mask,validate_counts=False)
            prob_zero=(decoded.theta/(decoded.theta+decoded.mu)).pow(decoded.theta)
            x=truth[:,mask];mu=decoded.mu[:,mask];pz=prob_zero[:,mask]
            predicted_variance=(mu+mu.square()/decoded.theta[mask]).mean(0)+mu.var(0,correction=0)
            actual_mean=x.mean(0);actual_variance=x.var(0,correction=0)
            correlation=torch.corrcoef(torch.stack([actual_mean,mu.mean(0)]))[0,1]
            metadata=source['metadata'];train=metadata.iloc[source['train']];val=metadata.iloc[source['validation']]
            result[line]={'validation_nll':float(loss),'diagnostic_cells':len(ids),
                'observed_zero_fraction':float((x==0).float().mean()),'predicted_zero_fraction':float(pz.mean()),
                'gene_mean_correlation':float(correlation),'gene_mean_MAE':float((actual_mean-mu.mean(0)).abs().mean()),
                'gene_variance_MAE':float((actual_variance-predicted_variance).abs().mean()),
                'train_cells':len(train),'validation_cells':len(val),'train_targets':int(train.target.nunique()),
                'validation_targets':int(val.target.nunique()),'train_guides':int(train.guide.nunique()),
                'validation_guides':int(val.guide.nunique()),'validation_is_count_reconstruction_only':True}
            result[line].update(target_overlap=len(set(train.target)&set(val.target)),
                guide_overlap=len(set(train.guide)-{'non-targeting'} & (set(val.guide)-{'non-targeting'})),
                targets_without_validation=int(metadata.target.nunique()-val.target.nunique()),
                targets_without_training=int(metadata.target.nunique()-train.target.nunique()))
    dump(dest/'decoder_diagnostics.json',result)

def fit_decoder(role,seed):
    runtime();dest=directory(role,'decoder',seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    c=P['decoder'];data={line:count_cells(line) for line in decoder_lines(role)}
    torch.manual_seed(seed);model=NBDecoder(len(axes()[0]),model_cfg()).cuda()
    # Fit initialization only on permitted decoder-training cells, with observed axes.
    probabilities=[]
    for source in data.values():
        totals=np.asarray(source['counts'][source['train']].sum(0)).ravel().astype(np.float64)
        q=(totals+1)/(totals.sum()+int(source['mask'].sum()));q[~source['mask']]=np.nan
        probabilities.append(q)
    bias=np.log(np.nan_to_num(np.nanmean(np.stack(probabilities),axis=0),nan=1/len(axes()[0]))+1e-12)
    with torch.no_grad():
        model.network[-1].bias.copy_(torch.as_tensor(bias,dtype=torch.float32,device='cuda'))
    manifest=provenance(role,'decoder',seed,c,model)
    manifest.update(likelihood='masked raw-count NB2',library_axis='actual measured output10916',
                    decoder_validation_is_standalone_reconstruction=True)
    manifest_write(dest,manifest);resources=resources_begin(dest)
    rngs={'count':np.random.default_rng(seed+300000)}
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['steps'],eta_min=c['min_lr'])
    first,trace,exposures,elapsed=restore(dest,model,opt,scheduler,rngs,manifest)
    if exposures is None:exposures={line:np.zeros(len(source['pc']),np.int64) for line,source in data.items()}
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(first,c['steps']+1):
        pc=[];counts=[];libs=[];masks=[];size=c['batch_size']//len(data)
        for line,source in data.items():
            ids=rngs['count'].choice(source['train'],size,replace=False);np.add.at(exposures[line],ids,1)
            pc.append(np.asarray(source['pc'][ids]));counts.append(source['counts'][ids].toarray())
            libs.append(source['library'][ids]);masks.append(np.broadcast_to(source['mask'],(size,len(axes()[0]))))
        x=torch.as_tensor(np.concatenate(pc),dtype=torch.float32,device='cuda')
        y=torch.as_tensor(np.concatenate(counts),dtype=torch.float32,device='cuda')
        lib=torch.as_tensor(np.concatenate(libs),device='cuda');mask=torch.as_tensor(np.concatenate(masks),device='cuda')
        model.train();out=model(x,lib,mask);loss=masked_nb_nll(y,out.mu,out.theta,mask,validate_counts=False)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
        assert torch.isfinite(norm);opt.step();scheduler.step()
        if step==1 or step%400==0 or step==c['steps']:
            r={'step':step,'loss':float(loss.detach()),'seconds':elapsed+time.monotonic()-start};trace.append(r)
            dump(dest/'trace.json',trace);print('V25_DECODER',role,seed,json.dumps(r),flush=True)
        if step%400==0 or step==c['steps']:
            ck=checkpoint(dest,model,opt,scheduler,step,rngs,trace,exposures,elapsed+time.monotonic()-start,manifest)
            if step==c['steps']:torch.save(ck,dest/'checkpoint_final.pt')
    training_seconds=elapsed+time.monotonic()-start
    decoder_diagnostics(model,data,dest)
    for line,v in exposures.items():
        pd.DataFrame({'cache_row':np.arange(len(v)),'exposures':v}).to_csv(dest/f'exposures_{line}.csv',index=False)
    dump(dest/'complete.json',{**manifest,'row_exposures':sum(int(v.sum()) for v in exposures.values()),
         'seconds':training_seconds,'fit_and_diagnostics_wall_seconds':elapsed+time.monotonic()-start,'resource_attempts':resources,
         'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30})

def frozen_decoder(role,seed,readout='nb'):
    if readout=='pca':decoder=PCAInverseDecoder(encoder_arrays(),axes()[0])
    else:
        ck=torch.load(directory(role,'decoder',seed)/'checkpoint_final.pt',map_location='cpu',weights_only=False)
        assert ck['manifest']['protocol_sha256']==sha(PROTOCOL)
        decoder=NBDecoder(len(axes()[0]),model_cfg());decoder.load_state_dict(ck['state_dict'])
    decoder=decoder.cuda().eval()
    for parameter in decoder.parameters():parameter.requires_grad_(False)
    return decoder

def common_mask(device='cuda'):
    genes=axes()[0];keep=set(encoder_arrays()['genes'].astype(str))
    return torch.as_tensor([g in keep for g in genes],dtype=torch.bool,device=device)

def sampled_inputs(targets,lines,rng,raw,encoder_gpu,pool_means=None):
    cache=controls();size=P['model']['set_size']
    dc,sc,sp,valid,metadata=sample_sets(targets,lines,rng,set_size=size,cache=cache,centered=False,return_metadata=True)
    encoder=encoder_arrays();positions={g:i for i,g in enumerate(axes()[0])}
    input_indices=np.asarray([positions[g] for g in encoder['genes'].astype(str)])
    arrays=np.empty((len(targets),size,len(input_indices)),np.float32);mu=np.empty((len(targets),len(input_indices)),np.float32)
    for line in dict.fromkeys(lines):
        at=np.flatnonzero(np.asarray(lines)==line);x=cache['input_counts'][line]
        arrays[at]=x[metadata['destination_rows'][at].ravel()].toarray().reshape(len(at),size,-1)
        mu[at]=(cache['pool_input_means'][line] if pool_means is None else pool_means[line])
    counts=torch.as_tensor(arrays,device='cuda');mean=torch.as_tensor(mu,device='cuda')
    input_raw=raw[:,torch.as_tensor(input_indices,device='cuda')]
    wanted=((mean+.5)*torch.exp2(input_raw)-.5).clamp_min(0)
    ratio=torch.where(mean>0,wanted/mean.clamp_min(1e-12),torch.ones_like(mean)).clamp(0,16)
    prior_counts=counts*ratio[:,None,:]
    totals=prior_counts.sum(-1,keepdim=True);assert torch.all(totals>0)
    logged=torch.log1p(prior_counts*(50000/totals))
    precision=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:x0=((logged-encoder_gpu['gene_center'])/encoder_gpu['gene_scale'])@encoder_gpu['basis'].T/encoder_gpu['pc_scale']
    finally:torch.set_float32_matmul_precision(precision)
    library=counts.sum(-1);assert torch.all(library>0)
    return x0,library,[torch.as_tensor(v,dtype=torch.float32,device='cuda') for v in (sc,sp,valid)],metadata

def population_endpoints(targets,lines,rng,x0):
    values=np.zeros(tuple(x0.shape),np.float32);valid=np.zeros(len(targets),bool);ids=np.full(tuple(x0.shape[:2]),-1,np.int64)
    for i,(target,line) in enumerate(zip(targets,lines)):
        if line=='orion_hct116':continue
        source=count_cells(str(line));pool=source['targets'][str(target)]
        chosen=rng.choice(pool,x0.shape[1],replace=len(pool)<x0.shape[1]);ids[i]=chosen
        values[i]=source['pc'][chosen];valid[i]=True
    return torch.as_tensor(values,device='cuda'),torch.as_tensor(valid,device='cuda'),ids

def mean_losses(raw,tx,j,byline):
    from train_residual_v6 import generated_profile
    c=P['training'];pred=generated_profile(tx['mu'][j],tx['outside'][j],raw);mask=tx['mask'][j]
    err=(pred-tx['y'][j]).square();mse=(err*mask).sum()/mask.sum()
    change=tx['y'][j]-tx['control'][j];sw=((change.abs()-.05)/.45).clamp(0,1)*mask
    signal=(err*sw).sum()/sw.sum().clamp_min(1)
    direction=(F.softplus(-change.sign()*(pred-tx['control'][j])/c['temperature'])*sw).sum()/sw.sum().clamp_min(1)
    size=len(j)//len(byline);terms=[]
    for li,line in enumerate(byline):
        sl=slice(li*size,(li+1)*size);pm=mask[sl].amin(0)*tx['own'][j[sl]].amin(0)
        pe=F.normalize((pred[sl]-tx['control'][j[sl]])*pm,dim=1,eps=1e-8)
        ye=F.normalize((tx['y'][j[sl]]-tx['control'][j[sl]])*pm,dim=1,eps=1e-8)
        terms.append(F.cross_entropy(pe@ye.T/c['temperature'],torch.arange(size,device='cuda')))
    disc=torch.stack(terms).mean();penalty=((raw-tx['raw'][j]).square()*mask).sum()/mask.sum()
    return {'mse':mse,'signal':signal,'direction':direction,'disc':disc,'penalty':penalty}

def fit(role,candidate,seed):
    assert candidate in ('det_nb','flow_nb');runtime();c=P['training']
    dest=directory(role,candidate,seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    rows,inputs,genes,origin=load_training_inputs(role)
    for name in ['features.npz','training_rows.csv']:
        if not (dest/name).exists():shutil.copy2(origin/name,dest/name)
        assert sha(dest/name)==sha(origin/name)
    torch.manual_seed(seed);model=LatentResponse(model_cfg()).cuda();decoder=frozen_decoder(role,seed)
    tx={k:torch.as_tensor(rows[k],dtype=torch.float32,device='cuda') for k in ['mu','y','mask','outside']}
    tx.update({k:torch.as_tensor(v,dtype=torch.float32,device='cuda') for k,v in inputs.items() if k!='seen'})
    seen=torch.as_tensor(inputs['seen'],device='cuda');measurement=common_mask()
    tx['control']=torch.log1p(tx['mu']*50000/(tx['mu'].sum(1)+tx['outside'])[:,None])
    byline={line:np.flatnonzero(rows['line']==line) for line in dict.fromkeys(rows['line'])}
    encoder_gpu={k:torch.as_tensor(encoder_arrays()[k],dtype=torch.float32,device='cuda') for k in ['gene_center','gene_scale','basis','pc_scale']}
    manifest=provenance(role,candidate,seed,c,model)
    manifest.update(features_sha256=sha(dest/'features.npz'),training_rows_sha256=sha(dest/'training_rows.csv'),
        decoder_checkpoint_sha256=sha(directory(role,'decoder',seed)/'checkpoint_final.pt'),
        population_supervision_contexts=[x for x in byline if x!='orion_hct116'],
        editable_common_genes=int(measurement.sum()),prior_anchor='old pool population mean multiplier, expected counts')
    manifest_write(dest,manifest);resources=resources_begin(dest)
    rngs={'row':np.random.default_rng(seed),'cell':np.random.default_rng(seed+100000),
          'population':np.random.default_rng(seed+200000),'time':np.random.default_rng(seed+400000)}
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['steps'],eta_min=c['min_lr'])
    first,trace,exposures,elapsed=restore(dest,model,opt,scheduler,rngs,manifest)
    if exposures is None:exposures=np.zeros(len(rows['target']),np.int64)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(first,c['steps']+1):
        size=c['batch_size']//len(byline);ids=np.concatenate([rngs['row'].choice(v,size,replace=False) for v in byline.values()])
        np.add.at(exposures,ids,1);j=torch.as_tensor(ids,device='cuda')
        x0,library,source,_=sampled_inputs(rows['target'][ids],rows['line'][ids],rngs['cell'],tx['raw'][j],encoder_gpu)
        x1,pairmask,popids=population_endpoints(rows['target'][ids],rows['line'][ids],rngs['population'],x0)
        times=torch.as_tensor(rngs['time'].random(tuple(x0.shape[:2])).astype(np.float32),device='cuda')
        model.train();cond=model.condition(tx['z'][j],tx['h'][j],*source)
        endpoint=model.deterministic(x0,cond) if candidate=='det_nb' else model.endpoint(x0,cond,P['model']['integration_steps'])
        delta=readout_delta(endpoint,x0,library,measurement,decoder)
        raw=merge_prior(tx['raw'][j],delta,seen,tx['own'][j],measurement,P['model']['correction_bound'])
        terms=mean_losses(raw,tx,j,byline)
        population=(deterministic_endpoint_loss(model,x0,x1,cond,pairmask) if candidate=='det_nb'
                    else flow_matching_loss(model,x0,x1,cond,times,pairmask))
        loss=terms['mse']+c['signal_mse']*terms['signal']+c['direction']*terms['direction']+c['discrimination']*terms['disc']+c['effect_penalty']*terms['penalty']+c['population_loss']*population
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip']);assert torch.isfinite(norm)
        opt.step();scheduler.step()
        if step==1 or step%400==0:
            r={'step':step,'loss':float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},
                'population':float(population.detach()),'Native_population_rows':int(pairmask.sum()),
                'seconds':elapsed+time.monotonic()-start};trace.append(r);dump(dest/'trace.json',trace)
            print('V25_RESPONSE',role,candidate,seed,json.dumps(r),flush=True)
        if step%400==0:
            ck=checkpoint(dest,model,opt,scheduler,step,rngs,trace,exposures,elapsed+time.monotonic()-start,manifest)
            if step in c['checkpoints']:torch.save(ck,dest/f'checkpoint_{step}.pt')
    pd.DataFrame({'target':rows['target'],'context':rows['line'],'exposures':exposures}).to_csv(dest/'exposures.csv',index=False)
    dump(dest/'complete.json',{**manifest,'row_exposures':int(exposures.sum()),'seconds':elapsed+time.monotonic()-start,
         'resource_attempts':resources,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,
         'reserved_GPU_GiB':torch.cuda.max_memory_reserved()/2**30})

def predict(role,candidate,seed,step,panel,destination,output):
    runtime();dest=directory(role,candidate,seed);ck=torch.load(dest/f'checkpoint_{step}.pt',map_location='cpu',weights_only=False)
    info=json.loads((Path(panel)/'panel.json').read_text())
    assert ck['manifest']['protocol_sha256']==sha(PROTOCOL) and destination not in ck['manifest']['training_contexts']
    inp,prior=predict_inputs(role,panel,features_origin=dest/'features.npz')
    gpu={k:torch.as_tensor(v,device='cuda') for k,v in inp.items()};rawprior=torch.as_tensor(prior,device='cuda')
    model=LatentResponse(model_cfg()).cuda();model.load_state_dict(ck['state_dict']);model.eval()
    decoder=frozen_decoder(role,seed,P['candidates'][candidate]['readout']);measurement=common_mask()
    egpu={k:torch.as_tensor(encoder_arrays()[k],dtype=torch.float32,device='cuda') for k in ['gene_center','gene_scale','basis','pc_scale']}
    from prepare_response_inputs_v23 import feature_arrays
    pool=sparse.load_npz(Path(panel)/'generation_pool.npz');native=np.loadtxt(Path(panel)/'genes.txt',dtype=str)
    mapping={g:i for i,g in enumerate(native)};mean=np.asarray(pool[:,[mapping[g] for g in encoder_arrays()['genes']]].mean(0)).ravel()
    targets=np.asarray(info['targets'],str);values=[]
    with torch.no_grad():
        for view in range(P['inference']['views']):
            rng=np.random.default_rng(P['inference']['cell_seed']+view*P['inference']['view_stride'])
            x0,library,source,_=sampled_inputs(targets,np.repeat(destination,len(targets)),rng,rawprior,egpu,{destination:mean})
            condition=model.condition(gpu['z'],gpu['h'],*source)
            endpoint=(model.deterministic(x0,condition) if candidate=='det_nb'
                      else model.endpoint(x0,condition,P['inference']['integration_steps']))
            delta=readout_delta(endpoint,x0,library,measurement,decoder)
            values.append(merge_prior(rawprior,delta,gpu['seen'],gpu['ownmask'],measurement,P['model']['correction_bound']).cpu().numpy())
    effect=np.mean(values,axis=0).astype(np.float32)
    editable=inp['seen']*inp['ownmask']*measurement.cpu().numpy()
    effect=np.where(editable!=0,effect,prior).astype(np.float32);assert np.isfinite(effect).all()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,effect=effect,targets=targets,genes=axes()[0])
    dump(output.with_suffix('.json'),{'trainname':role,'candidate':candidate,'seed':seed,'step':step,
         'checkpoint_sha256':sha(dest/f'checkpoint_{step}.pt'),'panel_sha256':sha(Path(panel)/'panel.json'),
         'decoder_readout':P['candidates'][candidate]['readout'],'decoder_checkpoint_sha256':sha(directory(role,'decoder',seed)/'checkpoint_final.pt'),
         'population_mean_from_actual_panel':True,'effect_semantics':'full frozen V1 plus bounded latent-count correction',
         'views':P['inference']['views'],'min':float(effect.min()),'max':float(effect.max())})
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['decoder','fit','predict'])
    parser.add_argument('--trainname',required=True);parser.add_argument('--seed',type=int,default=P['training']['seed'])
    parser.add_argument('--candidate',choices=list(P['candidates']));parser.add_argument('--step',type=int)
    parser.add_argument('--panel');parser.add_argument('--destination');parser.add_argument('--output');q=parser.parse_args()
    if q.mode=='decoder':fit_decoder(q.trainname,q.seed)
    elif q.mode=='fit':fit(q.trainname,q.candidate,q.seed)
    else:predict(q.trainname,q.candidate,q.seed,q.step,q.panel,q.destination,q.output)
