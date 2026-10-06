"""Materialize a fully fitted normalized-transfer challenger, preserving v1."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import anndata as ad
from select_transfer_model import read_source,profile,project,sha256

def main():
    root=Path.cwd();out=Path('runs/transfer_v2/challenge_effects');out.mkdir(parents=True,exist_ok=True)
    modelout=Path('models/transfer_v2');modelout.mkdir(parents=True,exist_ok=True)
    fitted=Path('runs/model_selection_v2/final_suite.npz')
    with np.load(fitted) as z:alpha=float(z['alpha']);training_targets=z['training_targets'].astype(str)
    genes=np.loadtxt('runs/model_selection_v2/gene_axis.txt',dtype=str)
    src=read_source('data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz')
    gp={g:i for i,g in enumerate(src['gene_names'])};gi=np.array([gp[g] for g in genes]);ti={t:i for i,t in enumerate(src['target_names'])}
    val=Path('data/raw/vcc2026/validation/files');targets=pd.read_csv(val/'pert_counts.csv').target_gene.to_numpy(dtype=str)
    if set(targets)&set(training_targets):raise ValueError('Competition target used in transfer calibration')
    effect=np.zeros((len(targets),len(genes)),np.float32);support=np.zeros(len(targets),int)
    sc=src['control_mean_counts'][gi];coverage=[]
    for i,t in enumerate(targets):
        j=ti.get(t);n=0 if j is None else int(src['target_n_cells'][j]);support[i]=n
        coverage.append({'target':t,'source_cells':n,'direct_response_used':n>=30,'fallback_reason':'absent' if j is None else ('low_support' if n<30 else '')})
        if n>=30:
            effect[i]=alpha*n/(n+50)*(profile(src['target_mean_counts'][j,gi])-profile(sc))
            if t in set(genes):effect[i,np.flatnonzero(genes==t)]=0
    config={'model_id':'normalized_k562_transfer','alpha':alpha,'source_reliability_prior':50,'minimum_source_cells':30,
        'source_features':'K562 normalized pseudobulk log1p(50000*mu/sum_6202_genes) minus NTC profile',
        'formula':'delta = alpha * n/(n+50) * source_delta; project through destination NTC with nonnegative counts and conserved core library',
        'modeled_genes':len(genes),'covered_targets':int((support>=30).sum()),'official_targets':300,
        'unmodeled_rule':'Unmodeled genes and unsupported target effects retain destination NTC counts',
        'target_own_gene':'Excluded from fitting and source response input; output zero effect before compositional projection',
        'parameter_source':str(fitted),'parameter_source_sha256':sha256(fitted),
        'status':'additional fully fitted challenger; no claim of universal superiority or official score',
        'calibration_outcomes':['HepG2','Jurkat','RPE1'],'h1_outcomes_used_to_fit_parameters':False}
    pd.DataFrame(coverage).to_csv(modelout/'target_coverage.csv',index=False)
    np.savez_compressed(modelout/'parameters.npz',alpha=np.array(alpha),genes=genes,source_control=sc,
        target_symbols=targets,source_support=support,source_effect_scaled=effect,training_targets=training_targets)
    shutil.copyfile(fitted,modelout/'all_transfer_candidate_parameters.npz')
    (modelout/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for context in 'ABC':
        path=out/f'context_{context}__normalized_k562_transfer.npz'
        if path.exists():raise FileExistsError(path)
        a=ad.read_h5ad(val/f'context_{context}.h5ad');lookup={g:i for i,g in enumerate(a.var_names)}
        mu=np.asarray(a.X.mean(0)).ravel()[[lookup[g] for g in genes]]
        desired=project(effect,mu)[1]
        lfc=np.log2((desired+.5)/(mu[None,:]+.5));lfc[support<30]=0
        np.savez_compressed(path,target_symbols=targets,gene_symbols=genes,effect_log2fc=lfc.astype(np.float32))
        metadata={**config,'prediction_context':context,'effect_definition':'log2((target_mean+0.5)/(control_mean+0.5))'}
        path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(context,lfc.shape,alpha,flush=True);del a

if __name__=='__main__':main()
