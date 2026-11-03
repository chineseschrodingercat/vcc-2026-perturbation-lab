"""Measured-union extension of lean_transfer; independent from the core baseline.

The fixed common measured core defines library normalization. Response outputs
use the union of measured genes, and every ensemble denominator is gene-specific.
Missing measurements never become observed zero effects. No challenge submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lean_transfer as lt


def measured_union(sources, allowed=None):
    values=set().union(*(set(s.gmap) for s in sources))
    return np.asarray([g for g in allowed if g in values] if allowed is not None else sorted(values),dtype=str)


def background_effect(sources, core, genes, targets, min_cells=30):
    if len({s.background for s in sources})!=1:
        raise ValueError("One biological background is required")
    gmap={g:i for i,g in enumerate(genes)}
    effect=np.zeros((len(targets),len(genes)),np.float32)
    support=np.zeros_like(effect)
    for source in sorted(sources,key=lambda s:s.source_id):
        cg=np.asarray([source.gmap[g] for g in core])
        available=[g for g in genes if g in source.gmap]
        sg=np.asarray([source.gmap[g] for g in available])
        og=np.asarray([gmap[g] for g in available])
        rows=np.asarray([source.tmap.get(t,-1) for t in targets])
        valid=rows>=0
        n=np.zeros(len(targets))
        n[valid]=source.n_cells[rows[valid]]
        ri=np.flatnonzero(valid & (n>=min_cells))
        if not len(ri) or not len(og):
            continue
        raw=source.target_mean[rows[ri]]
        total=raw[:,cg].sum(axis=1)
        ctotal=source.control_mean[cg].sum()
        if (total<=0).any() or ctotal<=0:
            raise ValueError(f"{source.source_id}: empty common core library")
        current=np.log1p(lt.SCALE*raw[:,sg]/total[:,None])-np.log1p(lt.SCALE*source.control_mean[sg]/ctotal)
        ix=np.ix_(ri,og)
        replace=n[ri,None]>support[ix]
        effect[ix]=np.where(replace,current,effect[ix])
        support[ix]=np.where(replace,n[ri,None],support[ix])
    return effect,support


def predict_delta(sources,core,genes,targets,query_control,model="weighted_shrink",alpha=1.0,
                  temperature=1.0,prior_cells=50.0,min_cells=30):
    if model=="zero":
        shape=(len(targets),len(genes))
        return np.zeros(shape),np.zeros(shape,bool),{"backgrounds":[],"covered_targets":0}
    groups={}
    for s in sources:
        groups.setdefault(s.background,[]).append(s)
    if model=="k562_direct":
        groups={k:v for k,v in groups.items() if k.upper().replace("-","")=="K562"}
        if not groups:
            raise ValueError("K562 unavailable after background exclusion")
    outmap={g:i for i,g in enumerate(genes)}
    cp=np.asarray([outmap[g] for g in core])
    query=lt.normalize(query_control[cp])
    prediction=np.zeros((len(targets),len(genes)),np.float64)
    denominator=np.zeros_like(prediction)
    audit=[]
    for background,members in sorted(groups.items()):
        effect,support=background_effect(members,core,genes,targets,min_cells)
        control,control_id=lt.representative_control(members,core)
        distance=float(np.mean((lt.normalize(control)-query)**2))
        weight=1.0 if model=="k562_direct" else float(np.exp(-min(temperature*distance,50)))
        measured=support>0
        reliability=np.ones_like(support) if model=="k562_direct" else np.divide(support,support+prior_cells,out=np.zeros_like(support),where=(support+prior_cells)>0)
        prediction+=weight*reliability*effect*measured
        denominator+=weight*measured
        audit.append({"background":background,"sources":[s.source_id for s in members],
                      "control_source":control_id,"context_weight":weight,"control_distance":distance,
                      "covered_targets":int(measured.any(axis=1).sum()),
                      "covered_target_gene_pairs":int(measured.sum())})
    measured=denominator>0
    np.divide(prediction,denominator,out=prediction,where=measured)
    if model!="k562_direct":
        prediction*=alpha
    return prediction,measured,{"backgrounds":audit,"covered_targets":int(measured.any(axis=1).sum()),
                                "covered_target_gene_pairs":int(measured.sum()),
                                "same_background_rule":"one vote per gene; highest-support measured source per target/gene"}


def project(delta,control,core_positions,measured):
    core_mass=float(control[core_positions].sum())
    if core_mass<=0:
        raise ValueError("Empty query normalization core")
    baseline=np.log1p(lt.SCALE*control/core_mass)
    desired=np.expm1(np.clip(baseline[None,:]+delta,0,20))*core_mass/lt.SCALE
    new_core=desired[:,core_positions].sum(axis=1)
    factor=np.divide(core_mass,new_core,out=np.ones(len(desired)),where=new_core>0)
    desired=np.where(measured,desired*factor[:,None],control[None,:])
    unchanged=np.all(delta==0,axis=1)
    desired[unchanged]=control
    actual=np.log1p(lt.SCALE*desired/desired[:,core_positions].sum(axis=1)[:,None])-baseline
    actual[unchanged]=0
    return actual,desired


def error_rows(pred,truth,truth_mask,pred_mask,targets,genes):
    index={g:i for i,g in enumerate(genes)}
    rows=[]
    for i,target in enumerate(targets):
        valid=truth_mask[i].copy()
        if not valid.any():
            continue
        p,y=pred[i,valid],truth[i,valid]
        denom=float(np.linalg.norm(p)*np.linalg.norm(y))
        excluding=valid.copy()
        if target in index:
            excluding[index[target]]=False
        if not excluding.any():
            raise ValueError("No response genes remain after excluding the target")
        rows.append(dict(target=str(target),proxy_delta_mse=float(np.mean((p-y)**2)),
                         proxy_delta_mse_excluding_target=float(np.mean((pred[i,excluding]-truth[i,excluding])**2)),
                         proxy_delta_cosine=float(np.dot(p,y)/denom) if denom else None,
                         measured_truth_genes=int(valid.sum()),
                         measured_training_genes=int((valid & pred_mask[i]).sum())))
    return rows


def cv(args):
    sources=lt.load_sources(args.sources)
    locks=lt.source_locks(sources)
    backgrounds=sorted({s.background for s in sources})
    if len(backgrounds)<2:
        raise ValueError("Need at least two independent biological backgrounds")
    allowed=lt.read_axis(args.genes,{"gene_name","gene","gene_symbol","gene_symbols"}) if args.genes else None
    core=lt.common_genes(sources,allowed,minimum=args.min_genes)
    union=measured_union(sources,allowed)
    alphas=[float(v) for v in args.alphas.split(",")]
    temps=[float(v) for v in args.temperatures.split(",")]
    configs=[dict(id="zero",model="zero",alpha=0.,temperature=0.),
             dict(id="k562_direct",model="k562_direct",alpha=1.,temperature=0.)]
    configs += [dict(id=f"union_a{a:g}_t{t:g}",model="weighted_shrink",alpha=a,temperature=t) for a in alphas for t in temps]
    rows,folds=[],[]
    for heldout in backgrounds:
        train=[s for s in sources if s.background!=heldout]
        test=[s for s in sources if s.background==heldout]
        query_source=max(test,key=lambda s:(s.control_n,s.source_id))
        genes=np.asarray([g for g in union if g in query_source.gmap])
        gene_positions={g:i for i,g in enumerate(genes)}
        cp=np.asarray([gene_positions[g] for g in core])
        query=query_source.control_mean[[query_source.gmap[g] for g in genes]]
        targets=sorted({str(t) for s in test for t,n in zip(s.targets,s.n_cells) if n>=args.min_cells},
                       key=lambda t:hashlib.sha256((str(args.seed)+":"+t).encode()).digest())[:args.max_targets]
        targets=np.asarray(targets,str)
        if not len(targets):
            continue
        predictions,audits={},{}
        for config in configs:
            key=(config["model"],config["temperature"])
            if key in predictions:
                continue
            if config["model"]=="k562_direct" and not any(s.background.upper().replace("-","")=="K562" for s in train):
                continue
            predictions[key]=predict_delta(train,core,genes,targets,query,model=config["model"],alpha=1,
                temperature=config["temperature"],prior_cells=args.prior_cells,min_cells=args.min_cells)
            audits[str(key)]=predictions[key][2]
        # No held-out response was passed into prediction above.
        truth,truth_n=background_effect(test,core,genes,targets,args.min_cells)
        truth_mask=truth_n>0
        for config in configs:
            key=(config["model"],config["temperature"])
            if key not in predictions:
                continue
            delta,mask,_=predictions[key]
            scale=config["alpha"] if config["model"]=="weighted_shrink" else 1
            projected,_=project(delta*scale,query,cp,mask)
            for row in error_rows(projected,truth,truth_mask,mask,targets,genes):
                rows.append(dict(heldout_background=heldout,candidate=config["id"],**row))
        folds.append(dict(heldout_background=heldout,excluded_sources=[s.source_id for s in test],
                          training_sources=[s.source_id for s in train],query_control_source=query_source.source_id,
                          query_measured_output_genes=len(genes),targets=len(targets),candidates=audits))
        print(f"UNION LOCO {heldout}: {len(core)} normalization genes; {len(genes)} measured response genes; {len(targets)} targets",flush=True)
    if not rows:
        raise ValueError("No eligible fold targets")
    summaries=[]
    for config in configs:
        subset=[r for r in rows if r["candidate"]==config["id"]]
        if not subset:
            continue
        fold_names=sorted({r["heldout_background"] for r in subset})
        stats={}
        for metric in ("proxy_delta_mse","proxy_delta_mse_excluding_target","proxy_delta_cosine"):
            fold_values=[]
            for fold in fold_names:
                values=[r[metric] for r in subset if r["heldout_background"]==fold and r[metric] is not None]
                if values:
                    fold_values.append(float(np.mean(values)))
            stats[metric]=float(np.mean(fold_values)) if fold_values else None
        summaries.append({**config,**stats,"folds":fold_names})
    eligible=[s for s in summaries if s["model"]!="k562_direct" and len(s["folds"])==len(folds)]
    selected=min(eligible,key=lambda s:(s["proxy_delta_mse_excluding_target"],s["alpha"],s["temperature"]))
    fitted={k:selected[k] for k in ("id","model","alpha","temperature")}
    fitted.update(schema="vcc_restart_measured_union_v1",core_genes=core.tolist(),output_genes=union.tolist(),
                  source_ids=[s.source_id for s in sources],source_backgrounds={s.source_id:s.background for s in sources},
                  source_locks=locks,
                  prior_cells=args.prior_cells,min_cells=args.min_cells,
                  selection_metric="equal-background mean proxy_delta_mse_excluding_target",
                  status="same-fold external LOCO tuning result; not an unbiased generalization estimate or official VCC score")
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"proxy_per_target.csv",index=False)
    lt.write_json(output/"selected_model.json",fitted)
    lt.write_json(output/"cv_report.json",dict(schema=fitted["schema"],common_normalization_genes=len(core),
                  union_output_genes=len(union),folds=folds,candidates=summaries,
                  caveats=["Only genes actually measured in held-out data enter truth metrics.",
                           "Training absence is a baseline fallback, never a measured zero response.",
                           "Each background gets one vote with a gene-wise measured denominator.",
                           "K562 direct is only reported on folds with K562 available.",
                           "LOCO selection uses these same proxy folds, not the official six-metric score."]))
    print(json.dumps({"selected":selected,"normalization_genes":len(core),"output_genes":len(union),"candidates":summaries}),flush=True)


def effects(args):
    sources=lt.load_sources(args.sources)
    model=json.loads(Path(args.model_json).read_text(encoding="utf-8"))
    lt.validate_source_locks(sources,model)
    if sorted(s.source_id for s in sources)!=sorted(model["source_ids"]):
        raise ValueError("Source identities changed; recalibrate this candidate")
    bundle,manifest,contexts,paths,official_genes,targets=lt.read_bundle(args.bundle)
    core=np.asarray(model["core_genes"],str)
    if not set(core).issubset(official_genes) or any(not set(core).issubset(s.gmap) for s in sources):
        raise ValueError("Calibration normalization core unavailable; rerun cv with official --genes")
    genes=np.asarray([g for g in official_genes if g in set(model["output_genes"])])
    index={g:i for i,g in enumerate(genes)}
    cp=np.asarray([index[g] for g in core])
    official_index={g:i for i,g in enumerate(official_genes)}
    positions=np.asarray([official_index[g] for g in genes])
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    for context in contexts:
        path=output/f"context_{context}.npz"
        if path.exists():
            raise FileExistsError(path)
        label_audit={}
        mean,variance,n,columns=lt.control_statistics(paths[context],context,official_genes,
                                                    manifest_binding=paths,label_audit=label_audit)
        query=mean[positions]
        delta,mask,audit=predict_delta(sources,core,genes,targets,query,model=model["model"],alpha=model["alpha"],
                    temperature=model["temperature"],prior_cells=model["prior_cells"],min_cells=model["min_cells"])
        _,desired=project(delta,query,cp,mask)
        raw_lfc=np.log2((desired+0.5)/(query[None,:]+0.5))
        raw_lfc[~mask]=0
        np.savez_compressed(path,effect_log2fc=raw_lfc.astype(np.float32),gene_symbols=genes,target_symbols=targets,
                            effect_space=np.asarray("raw_pseudocount_log2fc"),pseudocount=np.asarray(0.5),measured_mask=mask)
        lt.write_json(path.with_suffix(".json"),dict(context=context,source_model=str(Path(args.model_json).resolve()),
                      common_normalization_genes=len(core),output_genes=len(genes),n_controls=n,**audit,**label_audit))
        print(f"union effects {context}: {len(targets)} targets x {len(genes)} genes",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("cv");q.add_argument("--sources",nargs="+",required=True);q.add_argument("--output-dir",required=True)
    q.add_argument("--genes");q.add_argument("--min-genes",type=int,default=100);q.add_argument("--min-cells",type=int,default=30)
    q.add_argument("--max-targets",type=int,default=1000);q.add_argument("--prior-cells",type=float,default=50)
    q.add_argument("--alphas",default="0.25,0.5,0.75,1");q.add_argument("--temperatures",default="0,1,4")
    q.add_argument("--seed",type=int,default=20261103);q.set_defaults(func=cv)
    q=sub.add_parser("effects");q.add_argument("--sources",nargs="+",required=True);q.add_argument("--model-json",required=True)
    q.add_argument("--bundle",required=True);q.add_argument("--output-dir",required=True);q.set_defaults(func=effects)
    args=p.parse_args();args.func(args)


if __name__=="__main__":
    main()
