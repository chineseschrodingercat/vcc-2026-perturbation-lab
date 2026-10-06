"""V20 lawful TRAIN-only study/family OOF direction calibration diagnostic.

Only reads audited V20 TRAIN prepared arrays. No query inference/Native held
truth, final THP, count generation, cell_eval2 or official operation.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
import argparse, hashlib, json, shutil, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path('C:/00 - Competition')
HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'staging/newV20_native_source_selection'
PROTOCOL = HERE / 'direction_train_protocol.json'
OUT = HERE / 'direction_train_actual'
ROLES = {
    'HepG2': ('outer_rpe1__inner_jurkat', ('Jurkat','RPE1')),
    'Jurkat': ('outer_rpe1__inner_hepg2', ('HepG2','RPE1')),
    'RPE1': ('outer_jurkat__inner_hepg2', ('HepG2','Jurkat')),
}
FAMILY = {'K562':'K562','K562_IFNB':'K562','HT29_INS':'HT29','HT29_IFNB':'HT29',
          'CD4_Rest':'CD4','CD4_Stim8hr':'CD4','CD4_Stim48hr':'CD4',
          'HCT116':'HCT116','HEK293T':'HEK293T','HepG2':'HepG2','Jurkat':'Jurkat','RPE1':'RPE1'}
# Conservative provenance closure: K562 bridges Replogle and Mixscale;
# same-family / same-study connected components are removed together in OOF.
COMPONENT = {'K562':'Replogle_Mixscale_closure','HT29':'Replogle_Mixscale_closure',
             'RPE1':'Replogle_Mixscale_closure','HCT116':'Orion','HEK293T':'Orion',
             'HepG2':'Nadig','Jurkat':'Nadig','CD4':'Marson_CD4'}
FEATURES = ['log1p_abs_source_mean_log1pCP50K_response',
            'abs_equal_component_sign_vote',
            'abs_source_mean_over_mean_abs_component_response',
            'known_source_components_divided_by_3']

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def arr(p,keys):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def stable_hash(s): return int(hashlib.sha256(s.encode()).hexdigest(),16)

def freeze():
    if PROTOCOL.exists(): raise FileExistsError('Frozen experiment already exists')
    tm=read(PARENT/'TRAINER_MACHINE_CONTRACT.json')
    inputs={}
    for pool,(role,forbidden) in ROLES.items():
        spec=tm['roles'][role]
        assert spec['allowed_native_family']==pool and set(spec['forbidden_native_families'])==set(forbidden)
        p=PARENT/'prepared'/role
        inputs[pool]=dict(role=role,path=str(p),forbidden_native_families=list(forbidden),
            files={n:sha(p/n) for n in ('inputs.npz','labels.npz','unit_metadata.npz','shared_transform.npz')})
    proto=dict(schema='V28_clean_TRAIN_direction_logistic_diagnostic1',frozen=True,
        code_sha256=sha(__file__),parent_machine_sha256=sha(PARENT/'TRAINER_MACHINE_CONTRACT.json'),inputs=inputs,
        normalization='original V20 ratio-of-raw-means common6157 CP50000; delta log1p target - log1p control',
        genes='512 TRAIN-seen genes fixed by sha256(direction-v1-gene|exact_ENSG), own excluded by inherited mask',
        target_split='sha256(direction-v1-target|exact_ENSG) mod 5: 0 validation, 1/2/3/4 fit',
        oof='held family and its full family-study connected component removed before constructing source features',
        component_map=COMPONENT,family_map=FAMILY,source_aggregation='equal conditions within family; equal families within component; equal available components',
        truth='point-estimate direction of held TRAIN family averaged across its known conditions; not significant-DE direction or latent causal truth',
        feature_names=FEATURES,logistic=dict(C=1.0,solver='lbfgs',max_iter=200,tol=1e-6,class_weight=None),
        weights='equal target; equal held component within target; equal held families within component; equal known genes in family',
        metrics='target/study weighted Brier/ECE10/AUROC; fixed TRAIN-probability quartiles; target-cluster paired Brier bootstrap1000',
        bootstrap=dict(seed=20261005,replicates=1000,unit='exact target, preserving all family/gene records; conditional on observed source studies'),
        direction_prediction='source-consensus sign; calibrated confidence is not yet calibrated for V20 B0 model sign',
        true_cluster_bootstrap_SE='unknown: no guide/GEM/donor expression read; not fabricated from pooled cell SEM',
        original_prior_and_generator_unchanged=True,actual_count_or_scoring_calls=0,forbidden=['query Native perturbations','THP values','official API'],
        selection='one fixed logistic fit per lawful pool; no hyperparameter scan or selected gene scan',minimum_source_components=2)
    write(PROTOCOL,proto)
    print(json.dumps(dict(frozen=str(PROTOCOL),sha256=sha(PROTOCOL))))

def mean_known(values,known,axis=0):
    n=known.sum(axis)
    return np.divide(np.where(known,values,0).sum(axis),n,out=np.zeros(n.shape,np.float64),where=n>0),n>0

def dataset(pool,spec,proto):
    p=Path(spec['path'])
    for n,h in spec['files'].items():assert sha(p/n)==h,(pool,n)
    m=arr(p/'inputs.npz',('target_ids','targets','genes','model_exact_ENSG','common_to_model','seen'))
    u=arr(p/'unit_metadata.npz',('context_label_available',))
    with np.load(p/'labels.npz',allow_pickle=False) as z:
        names=z['context_names'].astype(str).tolist()
        assert not set(names)&set(spec['forbidden_native_families']) and 'THP1' not in names
        assert set(names)&{'HepG2','Jurkat','RPE1'}=={pool}
        gi=sorted(np.flatnonzero(m['seen']>0),key=lambda j:stable_hash('direction-v1-gene|'+str(m['model_exact_ENSG'][j])))[:512]
        gi=np.asarray(gi)
        mu=z['mu']; denominator=mu[:,:,m['common_to_model']].sum(-1,dtype=np.float64)
        ctrl=np.log1p(np.divide(mu[:,:,gi]*50000,denominator[:,:,None],out=np.zeros((len(names),len(m['target_ids']),len(gi)),np.float64),where=denominator[:,:,None]>0))
        del mu
        y=z['y']; delta=y[:,:,gi].astype(np.float64)-ctrl; del y,ctrl
        mask=z['mask']; known=mask[:,:,gi]>0; del mask
    known &= u['context_label_available'][:,:,None]
    families=sorted(set(FAMILY[c] for c in names));fvalues={};fknown={}
    for family in families:
        ix=[i for i,c in enumerate(names) if FAMILY[c]==family]
        fvalues[family],fknown[family]=mean_known(delta[ix],known[ix])
    groups=sorted(set(COMPONENT[f] for f in families));cvalues={};cknown={}
    for group in groups:
        fs=[f for f in families if COMPONENT[f]==group]
        cvalues[group],cknown[group]=mean_known(np.stack([fvalues[f] for f in fs]),np.stack([fknown[f] for f in fs]))
    rec=[];disposition=[]
    for family in families:
        held=COMPONENT[family];sources=[g for g in groups if g!=held]
        values=np.stack([cvalues[g] for g in sources]);sk=np.stack([cknown[g] for g in sources]);count=sk.sum(0)
        pred,_=mean_known(values,sk)
        absmean,_=mean_known(np.abs(values),sk)
        vote,_=mean_known(np.sign(values),sk)
        agreement=np.divide(np.abs(pred),absmean,out=np.zeros(pred.shape),where=absmean>0)
        truth=fvalues[family]
        valid=fknown[family]&(count>=proto['minimum_source_components'])&(np.abs(pred)>1e-12)&(np.abs(truth)>1e-12)
        ti,gji=np.nonzero(valid)
        x=np.column_stack((np.log1p(np.abs(pred[valid])),np.abs(vote[valid]),agreement[valid],count[valid]/3)).astype(np.float64)
        target_ids=m['target_ids'][ti].astype(str)
        y=(np.sign(pred[valid])==np.sign(truth[valid])).astype(np.int8)
        split=np.asarray([stable_hash('direction-v1-target|'+t)%5==0 for t in target_ids])
        rec.append(dict(X=x,y=y,target=ti,split=split,family=np.full(len(y),family),component=np.full(len(y),held),
            baseline=np.clip((1+agreement[valid])/2,0,1)))
        disposition.append(dict(family=family,held_component=held,source_components=sources,records=len(y),
            fit_targets=len(set(target_ids[~split])),validation_targets=len(set(target_ids[split])),
            known_response_records=int(fknown[family].sum()),low_source_or_zero_records=int(fknown[family].sum()-valid.sum())))
    data={k:np.concatenate([r[k] for r in rec]) for k in rec[0]}
    weight=np.zeros(len(data['y']),np.float64)
    # Each target contributes equally even when it lacks sparse Mixscale families.
    for t in np.unique(data['target']):
        tix=data['target']==t;gs=np.unique(data['component'][tix])
        for group in gs:
            gix=tix&(data['component']==group);fs=np.unique(data['family'][gix])
            for family in fs:
                ix=gix&(data['family']==family);weight[ix]=1/(len(gs)*len(fs)*ix.sum())
    data['weight']=weight
    return data,dict(pool=pool,contexts=names,components=groups,targets=m['target_ids'].astype(str).tolist(),
        gene_exact_ids=m['model_exact_ENSG'][gi].astype(str).tolist(),genes=m['genes'][gi].astype(str).tolist(),
        dispositions=disposition,records=len(weight),fit_records=int((~data['split']).sum()),validation_records=int(data['split'].sum()))

def metrics(y,p,w):
    w=w/w.sum();brier=float(np.sum(w*(p-y)**2))
    bins=[];ece=0
    b=np.minimum((p*10).astype(int),9)
    for j in range(10):
        ix=b==j
        if not ix.any():continue
        mass=w[ix].sum();mean=float(np.average(p[ix],weights=w[ix]));truth=float(np.average(y[ix],weights=w[ix]))
        ece+=mass*abs(mean-truth);bins.append(dict(bin=j,records=int(ix.sum()),weight=float(mass),confidence=mean,accuracy=truth))
    auc=float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None
    return dict(Brier=brier,ECE10=float(ece),AUROC=auc,source_consensus_direction_accuracy=float(np.dot(w,y)),calibration_bins=bins)

def bootstrap_brier(data,prob,base,seed):
    target=data['target'];w=data['weight'];y=data['y'];ids=np.unique(target)
    a=np.asarray([np.average((prob[target==t]-y[target==t])**2-(base[target==t]-y[target==t])**2,weights=w[target==t]) for t in ids])
    rng=np.random.default_rng(seed)
    boot=np.mean(a[rng.integers(0,len(a),size=(1000,len(a)))],axis=1)
    return dict(target_clusters=len(ids),mean_delta=float(a.mean()),percentile95=np.quantile(boot,[.025,.975]).tolist(),
                scope='target cluster paired resampling; source studies fixed, no biological-replicate inference')

def actual():
    begin=time.monotonic();threadpool_limits(1)
    proto=read(PROTOCOL)
    assert proto['frozen'] and proto['code_sha256']==sha(__file__)
    if OUT.exists():raise FileExistsError('Preserve existing experiment')
    assert shutil.disk_usage(ROOT).free>=50*2**30
    OUT.mkdir();write(OUT/'STARTED.json',dict(protocol_sha256=sha(PROTOCOL),code_sha256=sha(__file__),no_held_truth_reads=True))
    reports={}
    for pooli,(pool,spec) in enumerate(proto['inputs'].items()):
        data,meta=dataset(pool,spec,proto);train=~data['split'];val=data['split']
        assert train.any() and val.any() and len(np.unique(data['y'][train]))==2
        w=data['weight'][train];x=data['X'][train];y=data['y'][train]
        center=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-center)**2,axis=0,weights=w));scale=np.maximum(scale,1e-6)
        model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=200,tol=1e-6,class_weight=None)
        model.fit((x-center)/scale,y,sample_weight=w*len(w)/w.sum())
        constant=float(np.average(y,weights=w))
        trainp=model.predict_proba((x-center)/scale)[:,1];quart=np.quantile(trainp,[.25,.75]).tolist()
        calibrator=dict(schema='TRAIN_only_source_consensus_direction_calibrator1',pool=pool,feature_names=FEATURES,
            center=center.tolist(),scale=scale.tolist(),coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),classes=model.classes_.tolist(),
            fit_target_ids=[meta['targets'][i] for i in np.unique(data['target'][train])],
            validation_target_ids=[meta['targets'][i] for i in np.unique(data['target'][val])],
            source_component_map=COMPONENT,known_component_minimum=2,train_accuracy_constant=constant,train_probability_quartiles=quart,
            optimizer_iterations=model.n_iter_.tolist(),protocol_sha256=sha(PROTOCOL),inputs_sha256=spec['files'],
            calibrated_quantity='Probability held TRAIN family response direction agrees with lawful source-consensus direction',
            unsupported_claims=['V20_B0 prediction confidence','significant DE correctness','biological uncertainty','local overall gain'])
        dest=OUT/pool;dest.mkdir();write(dest/'CALIBRATOR_FROZEN_BEFORE_VALIDATION.json',calibrator)
        # All probabilities/metrics use frozen parameters; no fit to validation target labels.
        pred=model.predict_proba((data['X'][val]-center)/scale)[:,1];truth=data['y'][val];weight=data['weight'][val]
        vd={k:v[val] for k,v in data.items()};fixed=np.full(len(pred),constant)
        report=dict(metadata=meta,fit=metrics(y,trainp,w),validation_logistic=metrics(truth,pred,weight),
            validation_constant=metrics(truth,fixed,weight),validation_uncalibrated_consensus=metrics(truth,vd['baseline'],weight),
            paired_Brier_vs_constant=bootstrap_brier(vd,pred,fixed,20261005+pooli),
            paired_Brier_vs_uncalibrated_consensus=bootstrap_brier(vd,pred,vd['baseline'],20261015+pooli),
            validation_component={},calibrator_sha256=sha(dest/'CALIBRATOR_FROZEN_BEFORE_VALIDATION.json'))
        for group in np.unique(vd['component']):
            ix=vd['component']==group
            report['validation_component'][str(group)]=metrics(truth[ix],pred[ix],weight[ix])
        low=pred<=quart[0];high=pred>=quart[1]
        report['validation_fixed_TRAIN_quartiles']=dict(low_records=int(low.sum()),high_records=int(high.sum()),
            low_direction_accuracy=float(np.average(truth[low],weights=weight[low])) if low.any() else None,
            high_direction_accuracy=float(np.average(truth[high],weights=weight[high])) if high.any() else None)
        np.savez_compressed(dest/'VALIDATION_PREDICTIONS.npz',probability=pred,correct_direction=truth,weight=weight,
            target_index=vd['target'],held_family=vd['family'],held_component=vd['component'],features=vd['X'])
        write(dest/'REPORT.json',report);reports[pool]=report
        print(json.dumps(dict(pool=pool,records=meta['records'],fit_targets=len(calibrator['fit_target_ids']),
            validation_targets=len(calibrator['validation_target_ids']),Brier=report['validation_logistic']['Brier'],
            delta_constant=report['paired_Brier_vs_constant']['mean_delta'],AUROC=report['validation_logistic']['AUROC']),ensure_ascii=False),flush=True)
        del data,vd,x,y,w,model
    write(OUT/'COMPLETE.json',dict(completed=True,protocol_sha256=sha(PROTOCOL),code_sha256=sha(__file__),
        lawful_TRAIN_pools=list(reports),new_logistic_fits=3,query_Native_perturbation_reads=0,THP_value_reads=0,
        count_generation_calls=0,cell_eval2_judges=0,official_calls=0,source_prior_generator_unchanged=True,
        seconds=time.monotonic()-begin,report_sha256={p:sha(OUT/p/'REPORT.json') for p in reports},
        calibration_sha256={p:sha(OUT/p/'CALIBRATOR_FROZEN_BEFORE_VALIDATION.json') for p in reports}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('freeze','actual'));a=ap.parse_args()
    freeze() if a.mode=='freeze' else actual()
