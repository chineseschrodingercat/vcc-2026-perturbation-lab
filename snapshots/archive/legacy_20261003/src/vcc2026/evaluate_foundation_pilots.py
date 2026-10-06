"""Compare executed Stack/State predictions on the same 24-target H1 panel."""
from pathlib import Path
import argparse,json,contextlib
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from foundation_io import adata
from benchmark_h1_cells import read_counts,thin,save_adata,score
from generate_counts import transform_counts
from select_transfer_model import profile,project

SEED=20260927
BENCH=Path('runs/model_selection_v2/h1_single_cell_benchmark')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['stack','state'],required=True)
    ap.add_argument('--build-only',action='store_true');args=ap.parse_args()
    out=Path(f'runs/foundation_v3/{args.family}_h1_benchmark');out.mkdir(parents=True,exist_ok=True)
    bman=json.loads((BENCH/'manifest.json').read_text());targets=bman['targets']
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        if not args.build_only:score(out,manifest)
        return
    a=ad.read_h5ad('data/raw/external/vcc2025_h1/adata_Training.h5ad',backed='r');genes=np.asarray(a.var_names,dtype=str)
    split=pd.read_csv(BENCH/'ntc_split.csv');rows=split.query('role=="generation"').raw_row_index.to_numpy()
    # Replay only the original NTC RNG stream so fitted NTC moments exactly
    # match the v2 benchmark. Reuse its zero-effect cells for paired generation.
    rng=np.random.default_rng(SEED)
    allntc=np.flatnonzero(np.asarray(a.obs.target_gene,dtype=str)=='non-targeting');rng.shuffle(allntc)
    assert set(allntc[2000:10000])==set(rows)
    refraw=read_counts(a,allntc[:2000]);thin(refraw,bman['thinning_probability'],rng);del refraw
    pool=thin(read_counts(a,rows),bman['thinning_probability'],rng);a.file.close()
    mu=np.asarray(pool.mean(0)).ravel();gp={g:i for i,g in enumerate(genes)}
    with np.load('runs/model_selection_v2/h1_confirmation/prediction__norm_scalar.npz') as z:
        core=z['gene_symbols'].astype(str);effect=z['effect_bulk'];lookup={t:i for i,t in enumerate(z['target_symbols'].astype(str))}
        normeffect=np.stack([effect[lookup[t]] for t in targets])
    coreidx=np.array([gp[g] for g in core]);ratios={};metadata={}
    desired=project(normeffect,mu[coreidx])[1]
    normratio=np.ones((len(targets),len(genes)),np.float32)
    normratio[:,coreidx]=np.clip(desired/np.maximum(mu[coreidx],1e-8),0,16)
    if args.family=='stack':
        source=Path('runs/foundation_v3/stack_gwps_steps3')
        if not (source/'complete.json').exists():raise RuntimeError('Stack run incomplete')
        ntc=ad.read_h5ad(source/'generated__non-targeting.h5ad');vocab=np.asarray(ntc.var_names,dtype=str)
        vg={g:i for i,g in enumerate(vocab)};modeled=np.array([i for i,g in enumerate(core) if g in vg])
        outputidx=coreidx[modeled];ix=np.array([vg[g] for g in core[modeled]])
        sc=np.asarray(ntc.X.mean(0)).ravel()[ix]
        dest=ad.read_h5ad(source/'destination_ntc.h5ad');dc=np.asarray(dest.X.mean(0)).ravel()[ix]
        raw=[];corrected=[]
        for t in targets:
            p=ad.read_h5ad(source/f'generated__{t}.h5ad');pm=np.asarray(p.X.mean(0)).ravel()[ix]
            raw.append(profile(pm)-profile(dc));corrected.append(profile(pm)-profile(sc))
        for name,eff in [('stack_direct',np.stack(raw)),('stack_corrected',np.stack(corrected))]:
            want=project(eff,mu[outputidx])[1];r=np.ones_like(normratio)
            r[:,outputidx]=np.clip(want/np.maximum(mu[outputidx],1e-8),0,16);ratios[name]=r
        ratios['stack25_norm75']=np.exp(.25*np.log(np.maximum(ratios['stack_corrected'],1e-6))+.75*np.log(np.maximum(normratio,1e-6)))
        metadata={'steps':3,'model_gene_coverage':len(modeled),'artifact_correction':'Subtract predicted NTC normalized pseudobulk rather than observed NTC',
            'blend':'Fixed 25% corrected Stack + 75% normalized transfer in log count multiplier space; weights not optimized on H1',
            'checkpoint':'models/pretrained/stack_large_aligned/bc_large_aligned.ckpt'}
    else:
        import torch
        from state.tx.models.state_transition import StateTransitionPerturbationModel
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65)
        with np.load('data/processed/v3/state_cells/features.npz') as z:
            stgenes=z['genes'].astype(str);stcore=z['core_genes'].astype(str);string=z['string_embedding'];source=z['source_embedding'];sttargets=z['targets'].astype(str)
        sti={t:i for i,t in enumerate(sttargets)};outidx=np.array([gp[g] for g in stgenes]);denidx=np.array([gp[g] for g in stcore])
        sample=rng.choice(pool.shape[0],1024,replace=False);px=pool[sample]
        total=np.asarray(px[:,denidx].sum(1)).ravel()[:,None]
        x=np.log1p(px[:,outidx].toarray()*10000/np.maximum(total,1)).astype(np.float32)
        tx=torch.as_tensor(x,device='cuda');metadata={'modeled_genes':len(outidx),'artifact_correction':'Ratio of predicted mean normalized counts for target to predicted NTC; +0.1 pseudocount',
            'blend':'Fixed 25% State + 75% normalized transfer in log count multiplier space; weights not optimized on H1','destination_input':'1024 H1 NTC cells only'}
        for variant in ['string','string_source']:
            folder=Path(f'models/foundation_v3/state_{variant}')
            if not (folder/'complete.json').exists():raise RuntimeError(f'State {variant} training incomplete')
            ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            with contextlib.redirect_stdout((out/f'architecture_{variant}.txt').open('w')):m=StateTransitionPerturbationModel(**ckpt['model_kwargs']).cuda()
            m.load_state_dict(ckpt['state_dict']);m.eval()
            emb=string if variant=='string' else np.concatenate([string,source],1)
            def predict(e):
                values=[]
                with torch.inference_mode():
                    for start in range(0,len(tx),256):
                        b=tx[start:start+256];pe=torch.as_tensor(np.tile(e,(len(b),1)),device='cuda')
                        p=m({'ctrl_cell_emb':b,'pert_emb':pe})
                        values.append(torch.expm1(torch.clamp(p,0,12)).cpu().numpy())
                return np.concatenate(values).mean(0)
            control=predict(np.zeros(emb.shape[1],np.float32));r=np.ones_like(normratio)
            for i,t in enumerate(targets):
                pe=emb[sti[t]] if t in sti else np.zeros(emb.shape[1],np.float32)
                response=predict(pe);r[i,outidx]=np.clip((response+.1)/(control+.1),0,16)
            ratios[f'state_{variant}']=r
            ratios[f'state_{variant}25_norm75']=np.exp(.25*np.log(np.maximum(r,1e-6))+.75*np.log(np.maximum(normratio,1e-6)))
            del m,ckpt;torch.cuda.empty_cache()
    files={};paired=ad.read_h5ad(BENCH/'pred__zero__round.h5ad')
    pairedlabels=np.asarray(paired.obs.target_gene,dtype=str)
    for model,r in ratios.items():
        blocks=[];labels=[]
        for i,t in enumerate(targets):
            base=paired.X[np.flatnonzero(pairedlabels==t)].copy()
            assert base.shape[0]==400
            x=transform_counts(base,r[i],np.random.default_rng(np.random.SeedSequence([SEED,i,0])))
            blocks.append(x);labels += [t]*400
        files[model]=save_adata(out/f'pred__{model}.h5ad',blocks,labels,genes)
        np.savez_compressed(out/f'count_multipliers__{model}.npz',target_symbols=np.asarray(targets,dtype=str),gene_symbols=genes,multipliers=r)
        print('Generated',model,files[model],flush=True)
    manifest={**bman,'models':list(ratios),'real_path':str(BENCH/'real.h5ad'),'prediction_files':files,
        'cache_real_path':str(BENCH/'cache_real'),
        'comparison_note':'Same fixed targets, real cells, NTC generation cells, gene axis, evaluator, NTC moments and count-rounding seeds as v2. Local raw metrics, no leaderboard scaling.',
        'foundation_inference':metadata,'outcome_use':'Post-v2 exploratory architecture comparison; no H1 outcomes used for fitting model parameters, fixed blend weights'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not args.build_only:score(out,manifest)

if __name__=='__main__':main()
