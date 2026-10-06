"""Decompose official FID on the existing, explicitly developmental H1 panel."""
from pathlib import Path
from dataclasses import replace, asdict
import json
import numpy as np
import pandas as pd
import polars as pl
from cell_eval2 import EvalConfig, compute_metrics, aggregate_metrics, __version__


def decomposition(real, pred, modeled, targets):
    r = pd.DataFrame(real.to_dicts()).rename(columns={'p_adj':'real_padj','log2_fold_change':'real_lfc'})
    p = pd.DataFrame(pred.to_dicts()).rename(columns={'p_adj':'pred_padj','log2_fold_change':'pred_lfc'})
    d = r[['target','feature','real_padj','real_lfc']].merge(
        p[['target','feature','pred_padj','pred_lfc']], on=['target','feature'], how='left', validate='one_to_one')
    rows=[]
    for t in targets:
        v=d[(d.target==t)&(d.feature!=t)].copy()
        rs=v.real_padj < .05
        adjud=v.real_lfc.notna() & (v.real_lfc != 0)
        ps=(v.pred_padj < .05) & adjud
        correct=ps & (np.sign(v.pred_lfc)==np.sign(v.real_lfc))
        nr,npred,nk=int(rs.sum()),int(ps.sum()),int(correct.sum())
        coverage=min(1,npred/nr) if nr else 1.
        precision=nk/npred if npred else np.nan
        rows.append({'target':t,'real_sig':nr,'pred_sig_adjudicable':npred,'correct_direction':nk,
            'direction_precision':precision,'coverage_term':coverage,
            'fid_reconstructed':nk/max(nr,npred) if max(nr,npred) else np.nan,
            'overlap_sig':int((rs & (v.pred_padj < .05)).sum()),
            'real_sig_unmodeled':int((rs & ~v.feature.isin(modeled)).sum()),
            'pred_sig_unmodeled':int((ps & ~v.feature.isin(modeled)).sum())})
    return pd.DataFrame(rows)


def main():
    base=Path('runs/model_selection_v2/h1_single_cell_benchmark')
    out=Path('runs/context_v4/diagnostics_h1_v1');out.mkdir(parents=True,exist_ok=True)
    cfg=EvalConfig.from_preset('vcc2026')
    cfg=replace(cfg,pert_col='target_gene',device='cpu',num_threads=4,gather_threads=4,
        cache_real=str(base/'cache_real'),outdir=str(out),de=replace(cfg.de,backend='scanpy'))
    (out/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
    if not (out/'de_pred.parquet').exists():
        m=compute_metrics(base/'eval_input__v1_raw_scalar__round.h5ad',base/'real.h5ad',config=cfg,write_de=True)
        m.write_csv(out/'metrics.csv');aggregate_metrics(m).write_csv(out/'aggregate.csv')
    with np.load('runs/model_selection_v2/h1_confirmation/prediction__zero.npz') as z: modeled=z['gene_symbols']
    targets=json.loads((base/'protocol.json').read_text())['targets']
    d=decomposition(pl.read_parquet(out/'de_real.parquet'),pl.read_parquet(out/'de_pred.parquet'),modeled,targets)
    d.to_csv(out/'fid_decomposition.csv',index=False)
    summary={'cell_eval2_version':__version__,'panel':'Existing 24-target H1 development panel; not untouched test',
        'n_targets':len(d),'modeled_genes':len(modeled),'mean_fid':float(d.fid_reconstructed.mean()),
        'mean_direction_precision_when_called':float(d.direction_precision.mean()),
        'mean_coverage':float(d.coverage_term.mean()),'median_real_sig':float(d.real_sig.median()),
        'median_pred_sig':float(d.pred_sig_adjudicable.median()),
        'pooled_unmodeled_fraction_of_real_sig':float(d.real_sig_unmodeled.sum()/d.real_sig.sum()),
        'mean_unmodeled_fraction_of_real_sig':float((d.real_sig_unmodeled/d.real_sig.replace(0,np.nan)).mean()),
        'coverage_limited_targets':int((d.pred_sig_adjudicable<d.real_sig).sum())}
    expected=pd.read_csv(base/'aggregate__v1_raw_scalar__round.csv')
    value=float(expected.loc[expected.metric=='de_wilcoxon_direction_fidelity_yield_raw','mean'].iloc[0])
    assert abs(value-summary['mean_fid'])<1e-8,(value,summary['mean_fid'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
