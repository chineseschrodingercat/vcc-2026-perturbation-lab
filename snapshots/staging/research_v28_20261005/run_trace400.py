"""Paired 400-cell development trace. No network, training, or submissions.

Historical Native and unified formal V28 are separate anchors, neither is a
new clean background-generalization estimate. Truth is opened by scoring only.
"""
from pathlib import Path
import argparse, json, os, sys, time, traceback, shutil
from datetime import datetime, timezone
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ[k]='4'
ROOT=Path('C:/00 - Competition')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'runs/research_v28_20261005/trace400_v1'
sys.path.insert(0,str(ROOT/'scripts'))
import native_reproducer as nr
import numpy as np
import pandas as pd
from scipy import sparse
ARMS=('historical_NB','formal_NB','formal_Poisson','formal_increment')

def fresh(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def reserve():
    if shutil.disk_usage(ROOT).free<50*2**30:raise RuntimeError('C storage reserve below50GiB')

def freeze():
    p=OUT/'PROTOCOL.json'
    if p.exists():
        pdoc=nr.read(p)
        for path,h in pdoc['dependencies'].items():assert nr.sha(path)==h, 'Frozen dependency drift:'+path
        for case in pdoc['cases']:
            for spec in case['leaves'].values():assert nr.sha(spec['path'])==spec['sha256'], 'Frozen data drift:'+spec['path']
        return pdoc
    reserve();jobs=nr.jobs('inner');cases=[]
    for j in jobs:
        panel=nr.relocated(j['historical_panel']);info=nr.read(panel/'panel.json')
        leaves={k:{'path':str(panel/k),'sha256':nr.sha(panel/k)} for k in ('real.h5ad','panel.json','genes.txt','generation_pool.npz','references/reference.json','references/anchor_aggregate.parquet')}
        for k,q in [('historical_mean',nr.relocated(j['effect'])),('historical_theta',nr.ARCHIVE/f'retained/native_theta/{j["model_role"]}/seed_{j["model_seed"]}/dispersion_seen.npz')]:
            leaves[k]={'path':str(q),'sha256':nr.sha(q)}
        cases.append({'job':j,'targets':info['targets'],'historical_prediction_cells_per_target':info.get('generation_target_cells',info['target_cells']),
                      'fixed_prediction_cells_per_target':400,'leaves':leaves})
    dependencies=[ROOT/'scripts/native_reproducer.py',Path(__file__),HERE/'formal_mean_adapter.py',
        nr.ARCHIVE/'runs/submission_v28_20261001/dispersion.npz',
        nr.ARCHIVE/'models/residual_v7/final_candidate/seed_20260928/features.npz',
        nr.ARCHIVE/'models/residual_v7/final_candidate/seed_20260928/tiny_transformer_600.pt',
        nr.SOURCE/'literal_nb_v28.py',nr.SOURCE/'score_residual_v6.py',nr.SOURCE/'residual_v6_common.py',
        nr.SOURCE/'generate_counts.py',nr.SOURCE/'latent_counts_v26.py',nr.SOURCE/'diagnose_v4.py',
        nr.ARCHIVE/'runs/context_v4/gene_axis.txt',nr.ARCHIVE/'runs/context_v4/normalization_core.txt',
        nr.ARCHIVE/'configs/baseline_v1.json',ROOT/'data/processed/set_response_v22/encoder.npz',
        ROOT/'data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz']
    dependencies+=sorted((ROOT/'envs/vcc313/Lib/site-packages/cell_eval2').rglob('*.py'))
    pdoc={'schema':'V28_trace400_v1','created_utc':datetime.now(timezone.utc).isoformat(),
       'purpose':'Re-establish paired400 development anchors and isolate generator noise on fixed formal means',
       'primary_generalization_protocol':False,'historical_Native_is_repeated_development':True,
       'formal_checkpoint_trained_Native':True,'legacy_prior_scale_fitted_Native':True,
       'arms':list(ARMS),'count_seed':20260928,'donor_seed':20261028,'cell_count':400,
       'targets':'all original8 per case; no exclusions or changes after scores',
       'real_population':'all original frozen target cells and1000 scoring NTC; no new sampling',
       'generation_pool':'original3000 NTC disjoint from scoring NTC; same400 donors per target/arm',
       'generator_baseline':'original AST build_target N1; formal uses original deployment theta/seen, historical uses original role theta/seen',
       'Poisson':'same literal conditional means X*rho; Poisson resample editable genes; preserves protected H0',
       'increment':'rho<1 Binomial(X,rho), rho>1 X+Poisson(X*(rho-1)), rho=1 exactX; preserves protectedH0',
       'why_retest_distribution':'new fixed400 deployment-size protocol and true formal mean weights; historical count-noise variants not asserted novel',
       'scorer':'cell_eval2==0.16.0 vcc2026 CPU scanpy; exact archived score_path',
       'anchor':'original same real hash and five-split reference; reused unchanged across allarms',
       'aggregation':'six metrics equal within role; two roles equal within destination; three destinations equal',
       'count_seeds_for_next_stage':[20260928,20260929,20260930],
       'direction_truth':'no heldout truth fitted; diagnostics only after all predictions sealed',
       'final_test':'THP1 held sealed; old Native outer and mod1 are not untouched final tests',
       'dependencies':{str(p):nr.sha(p) for p in dependencies},
       'cases':cases,'official_calls':0,'training':0}
    fresh(p,pdoc);return pdoc

def normalized_profile(x):
    v=np.asarray(x,dtype=np.float64).ravel();return np.log1p(v/max(v.sum(),1e-300)*50000.)

def modified_block(gen,target,mode):
    r=gen.build_target(target,'H0');m=r.moments
    edit=m['editable_native_positions']; donor=gen.pool[gen.donors[target]].toarray().astype(np.int64)
    ratio=m['count_ratio_native'];X=donor[:,edit];mu=X*ratio[None,edit]
    rng=np.random.default_rng(np.random.SeedSequence([gen.count_seed,gen.order.index(target),701 if mode=='formal_Poisson' else 702]))
    if mode=='formal_Poisson':Y=rng.poisson(mu)
    else:
        Y=X.copy();down=ratio[edit]<1;up=ratio[edit]>1
        Y[:,down]=rng.binomial(X[:,down],ratio[edit][None,down])
        Y[:,up]+=rng.poisson(X[:,up]*(ratio[edit][None,up]-1.))
        assert np.array_equal(Y[:,~(down|up)],X[:,~(down|up)])
    dense=r.block.toarray();dense[:,edit]=Y
    block=sparse.csr_matrix(dense,dtype=np.float32);block.eliminate_zeros();block.sort_indices()
    protect=np.ones(len(gen.genes),dtype=bool);protect[edit]=False
    assert (block[:,protect]!=r.block[:,protect]).nnz==0
    lib=np.asarray(block.sum(1)).ravel()
    assert (lib>0).all() and (lib<1e6).all() and np.isfinite(block.data).all()
    assert (block.data>0).all() and (block.data<2**24).all() and np.array_equal(block.data,np.rint(block.data))
    diag={'target':target,'mode':mode,'count_block_sha256':nr.generator_functions()['csr_sha'](block),
          'editable_genes':len(edit),'literal_conditional_mean_same_as_NB':True,'protected_H0_preserved':True,
          'donor_rows_sha256':nr.generator_functions()['array_sha'](gen.donors[target])}
    expected=np.asarray(r.block.mean(0)).ravel().astype(np.float64)
    expected[edit]=mu.mean(0)
    return block,diag,expected

def generate():
    protocol=freeze()
    if (OUT/'PREDICTIONS_SEALED.json').exists():return
    from formal_mean_adapter import FormalMeanAdapter
    adapter=FormalMeanAdapter()
    import anndata as ad
    ad.settings.allow_write_nullable_strings=True
    formal_path=nr.ARCHIVE/'runs/submission_v28_20261001/dispersion.npz'
    feature_path=nr.ARCHIVE/'models/residual_v7/final_candidate/seed_20260928/features.npz'
    with np.load(formal_path,allow_pickle=False) as z:formal_theta=z['theta'].astype(np.float64);formal_genes=z['gene_symbols'].astype(str)
    with np.load(feature_path,allow_pickle=False) as z:formal_seen=z['seen'].astype(np.float32)
    all_records=[];start=time.monotonic()
    for case in protocol['cases']:
        j=case['job'];dest=OUT/j['role'];nr.prepare_case(j,dest)
        effect=dest/'formal_mean.npz'
        if effect.exists():
            receipt=nr.read(dest/'formal_mean_receipt.json')
            assert nr.sha(effect)==receipt['effect_file_sha256']
        if not effect.exists():
            pool=sparse.load_npz(nr.relocated(j['historical_panel'])/'generation_pool.npz').tocsr()
            native_genes=np.loadtxt(nr.relocated(j['historical_panel'])/'genes.txt',dtype=str,ndmin=1)
            pred=adapter.predict(case['targets'],native_genes,pool)
            np.savez_compressed(effect,targets=pred['targets'],genes=pred['genes'],effects=pred['effect'],residual=pred['residual'])
            receipt=pred['receipt'];receipt['effect_file_sha256']=nr.sha(effect)
            fresh(dest/'formal_mean_receipt.json',receipt)
        for arm in ARMS:
            reserve();output=dest/f'pred__{arm}.h5ad';manifest=dest/f'{arm}_generation.json'
            if manifest.exists():
                assert nr.sha(output)==nr.read(manifest)['prediction_sha256'];all_records.append(nr.read(manifest));continue
            if output.exists():raise RuntimeError('Unsealed count requires inspection:'+str(output))
            gen=nr.CompactPanelGenerator(j,None if arm=='historical_NB' else effect)
            gen.sizes={t:400 for t in gen.order};rng=np.random.default_rng(gen.count_seed+100)
            gen.donors={t:rng.choice(gen.pool.shape[0],400,replace=False) for t in gen.order}
            if arm!='historical_NB':
                np.testing.assert_array_equal(formal_genes,gen.output_genes)
                gen.theta=formal_theta.copy();gen.inputs['seen']=formal_seen.copy()
                gen.source_identity.update(compact_theta_path=str(formal_path),compact_theta_sha256=nr.sha(formal_path),seen_path=str(feature_path),seen_sha256=nr.sha(feature_path),mean_workflow='formal_unified_V28')
            blocks=[];labels=[];records=[];moments=[]
            for target in gen.targets:
                if arm in ('formal_NB','historical_NB'):
                    r=gen.build_target(target);b=r.block;d=r.diagnostics;expected=r.moments['conditional_given_H0_expected_gene_mean']
                else:b,d,expected=modified_block(gen,target,arm)
                sample=np.asarray(b.mean(0)).ravel().astype(np.float64)
                d.update(cells=400,bulk_lognorm_drift_RMS=float(np.sqrt(np.mean((normalized_profile(sample)-normalized_profile(expected))**2))),
                   sample_library_mean=float(sample.sum()),conditional_expected_library_mean=float(expected.sum()))
                blocks.append(b);labels.extend([target]*400);records.append(d);moments.append([expected,sample])
            mat=sparse.vstack(blocks,format='csr');obj=ad.AnnData(mat,obs=pd.DataFrame({'target_gene':labels},index=[f'{arm}_{i}' for i in range(len(labels))]),var=pd.DataFrame(index=gen.genes))
            obj.write_h5ad(output,compression='gzip');np.savez_compressed(dest/f'{arm}_moments.npz',targets=np.asarray(gen.targets),genes=gen.genes,expected_and_sample_mean=np.asarray(moments))
            rec={'arm':arm,'role':j['role'],'context':j['context'],'completed':True,'cells':mat.shape[0],
                 'protocol_sha256':nr.sha(OUT/'PROTOCOL.json'),'prediction_path':str(output),'prediction_sha256':nr.sha(output),
                 'effects_path':gen.source_identity['mean_path'],'source_identity':gen.source_identity,'targets':records,
                 'truth_read_by_generation':False,'count_seed':gen.count_seed,'donor_seed':gen.count_seed+100}
            fresh(manifest,rec);all_records.append(rec);print('COUNT400',j['role'],arm,mat.shape,flush=True)
    fresh(OUT/'PREDICTIONS_SEALED.json',{'completed':True,'records':all_records,'seconds':time.monotonic()-start,
             'all_predictions_frozen_before_judge':True,'actual_count_outputs':len(all_records),'training':0,'official_calls':0})

def collect():
    rows=[]
    for case in freeze()['cases']:
        j=case['job'];dest=OUT/j['role']
        for arm in ARMS:
            summary=dest/f'scores/{arm}/summary.json'
            if not summary.exists():continue
            comp=pd.read_csv(dest/f'scores/{arm}/overall_components.csv')
            rows.append({'arm':arm,'role':j['role'],'context':j['context'],**nr.read(summary),
                         **{f'scaled__{r.metric}':float(r.replicate_scaled) for r in comp.itertuples()}})
    df=pd.DataFrame(rows);df.to_csv(OUT/'completed_scores.csv',index=False)
    aggregates=[]
    for arm in ARMS:
        g=df[df.arm==arm] if not df.empty else df
        if len(g)!=6:continue
        aggregates.append({'arm':arm,'overall':float(g.groupby('context').replicate_overall.mean().mean()),
               'per_background':g.groupby('context').replicate_overall.mean().to_dict(),
               'six_scaled':{m:float(g[f'scaled__{m}'].mean()) for m in nr.METRICS}})
    nr.write(OUT/'PROGRESS.json',{'complete':len(df)==24,'completed_scores':len(df),'aggregates':aggregates,
                'not_clean_generalization':True,'protocol_sha256':nr.sha(OUT/'PROTOCOL.json'),'official_calls':0})
    return df,aggregates

def score(threads):
    assert (OUT/'PREDICTIONS_SEALED.json').exists()
    seal=nr.read(OUT/'PREDICTIONS_SEALED.json')
    for r in seal['records']:assert nr.sha(r['prediction_path'])==r['prediction_sha256']
    lock=OUT/'SCORING.lock'
    with lock.open('x',encoding='utf8') as f:f.write(str(os.getpid()))
    try:
        for case in freeze()['cases']:
            dest=OUT/case['job']['role'];names=[a for a in ARMS if not(dest/f'scores/{a}/summary.json').exists()]
            if names:nr.score_case(dest,names,threads)
            _,a=collect();print('PROGRESS400',json.dumps(a),flush=True)
        df,ag=collect();fresh(OUT/'COMPLETE.json',{'completed':True,'actual_integer_count_scores':len(df),'aggregates':ag,
               'protocol_sha256':nr.sha(OUT/'PROTOCOL.json'),'official_calls':0})
    finally:lock.unlink()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('freeze','generate','score','collect'));ap.add_argument('--threads',type=int,default=4);a=ap.parse_args()
    os.chdir(ROOT)
    try:
        if a.phase=='freeze':print(json.dumps(freeze(),ensure_ascii=False))
        elif a.phase=='generate':generate()
        elif a.phase=='score':score(a.threads)
        else:print(collect()[1])
    except Exception as e:
        fresh(OUT/f'FAILED_{a.phase}_{time.time_ns()}.json',{'error':repr(e),'traceback':traceback.format_exc(),'official_calls':0});raise
