"""Execute the publisher's Stack-Large-Aligned model on frozen H1 targets.

The donor is K562; the only destination input is held-out H1 NTC cells.
Source and destination are aligned to the published vocabulary; absent source
genes are zero-padded, as required by the official inference implementation.
This is recorded as a domain/coverage limitation, not treated as real zeros.
"""
from pathlib import Path
import argparse,json,time,pickle,hashlib
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy import sparse
from foundation_io import metadata,dense_rows,align,adata
from benchmark_h1_cells import read_counts,thin

ROOT=Path.cwd()
SEED=20260927
# Stack 0.1.3 expects NumPy arrays from Index.values; pandas 3 otherwise
# returns a StringArray whose astype(str) is incompatible with np.char.upper.
pd.set_option('future.infer_string',False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='gwps',choices=['gwps','essential'])
    ap.add_argument('--limit',type=int,default=24);ap.add_argument('--steps',type=int,default=3)
    args=ap.parse_args()
    out=ROOT/f'runs/foundation_v3/stack_{args.source}_steps{args.steps}'
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/f'data/raw/external/replogle_2022/k562_{args.source}/K562_{args.source}_raw_singlecell_01.h5ad'
    labels,srcgenes=metadata(source)
    bench=ROOT/'runs/model_selection_v2/h1_single_cell_benchmark'
    bman=json.loads((bench/'manifest.json').read_text())
    targets=bman['targets'][:args.limit]
    counts=pd.Series(labels).value_counts()
    targets=[t for t in targets if counts.get(t,0)>=30]
    pretrained=ROOT/'models/pretrained/stack_large_aligned'
    with (pretrained/'basecount_1000per_15000max.pkl').open('rb') as f:genes=np.asarray(pickle.load(f),dtype=str)
    source_available=set(srcgenes[~pd.Index(srcgenes).duplicated(keep=False)])
    rng=np.random.default_rng(SEED)
    h=ad.read_h5ad(ROOT/'data/raw/external/vcc2025_h1/adata_Training.h5ad',backed='r')
    split=pd.read_csv(bench/'ntc_split.csv');poolrows=split.query('role=="generation"').raw_row_index.to_numpy()
    rows=rng.choice(poolrows,400,replace=False)
    hx=thin(read_counts(h,rows),bman['thinning_probability'],rng).toarray()
    destx,destidx=align(hx,h.var_names,genes);h.file.close()
    dest=adata(destx,genes)
    dest.write_h5ad(out/'destination_ntc.h5ad',compression='gzip')
    protocol={'model':'ArcInstitute/Stack-Large-Aligned','hf_revision':'b09f085dac03d170b078a5c72f550ae93686e544',
      'checkpoint_sha256':'f93cf6f42f36c8a85dc570d92e801c1fc3e1f45d55741e6bb250d178a6b6ad36',
      'source':str(source.relative_to(ROOT)),'targets':targets,'seed':SEED,'steps':args.steps,'mode':'vanilla',
      'prompt_ratio':.25,'context_ratio':.4,'context_ratio_min':.2,'batch_size':1,'num_workers':0,
      'source_covered_vocab_genes':len(source_available&set(genes)),'vocab_genes':len(genes),
      'gene_padding':'Official zero padding for absent measurements; predictions must be independently validated',
      'precision':'float32; published weights unchanged','destination_raw_ntc_rows':rows.tolist(),
      'source_rows':{},'gpu_memory_fraction':.70,'target_truth_used_for_inference':False}
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.set_per_process_memory_fraction(.70)
    from stack.model_finetune import ICL_FinetunedModel
    # The official checkpoint is hash-verified before this run. Load CPU first,
    # discard optimizer state, then move only model weights onto the GPU.
    ckpt=torch.load(pretrained/'bc_large_aligned.ckpt',map_location='cpu',weights_only=False)
    model=ICL_FinetunedModel(**ckpt['hyper_parameters']['model_config'])
    model.load_state_dict({k.removeprefix('model.'):v for k,v in ckpt['state_dict'].items()},strict=True)
    del ckpt
    model=model.cuda().eval()
    protocol['parameter_count']=sum(p.numel() for p in model.parameters())
    for t in ['non-targeting']+targets:
        output=out/f'generated__{t}.h5ad'
        if output.exists():continue
        ids=np.flatnonzero(labels==t)
        selected=rng.choice(ids,400,replace=len(ids)<400)
        sx=dense_rows(source,selected)
        sx,_=align(sx,srcgenes,genes);base=adata(sx,genes)
        protocol['source_rows'][t]=selected.tolist()
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        begin=time.monotonic()
        with torch.inference_mode():
            result=model.get_incontext_generation(base,dest,str(pretrained/'basecount_1000per_15000max.pkl'),
                num_steps=args.steps,prompt_ratio=.25,context_ratio=.4,context_ratio_min=.2,
                mode='vanilla',batch_size=1,num_workers=0,random_seed=SEED,show_progress=False)
        x=sparse.csr_matrix(result,dtype=np.float32)
        if x.shape!=(400,len(genes)) or not np.isfinite(x.data).all() or np.any(x.data<0):
            raise ValueError('Invalid Stack result')
        result_ad=adata(x,genes);result_ad.obs['target_gene']=t
        result_ad.write_h5ad(output,compression='gzip',compression_opts=1)
        print(json.dumps({'target':t,'seconds':round(time.monotonic()-begin,2),'shape':x.shape,'nnz':x.nnz,
          'median_library':float(np.median(np.asarray(x.sum(axis=1)).ravel())),
          'max_gpu_allocated_GB':torch.cuda.max_memory_allocated()/1e9}),flush=True)
    (out/'complete.json').write_text(json.dumps({'targets':targets,'parameters':protocol['parameter_count'],'status':'completed'},indent=2)+'\n')

if __name__=='__main__':main()
