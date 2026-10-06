"""Run predeclared count interventions with the unchanged Native local judge."""
from __future__ import annotations
import argparse, importlib.util, json, os, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('POLARS_MAX_THREADS','2')
import pandas as pd
import native_reproducer as native

def collect(plan, out):
    records,components=[],[]
    for item in plan['cases']:
        job=native.find_job(item['job_key'])
        case=out/job['key'];leaf=case/f"scores/{item['arm']}/summary.json"
        if not leaf.exists():continue
        summary=native.read(leaf)
        baseline=native.read(native.relocated(job['score_output'])/'summary.json')
        records.append({'arm':item['arm'],'key':job['key'],'context':job['context'],
            'model_seed':job['model_seed'],'count_seed':job['count_seed'],**summary,
            'V28_overall':baseline['replicate_overall'],
            'delta_vs_V28':summary['replicate_overall']-baseline['replicate_overall']})
        frame=pd.read_csv(case/f"scores/{item['arm']}/overall_components.csv")
        frame['arm']=item['arm'];frame['key']=job['key'];frame['context']=job['context']
        components.append(frame)
    if not records:return
    frame=pd.DataFrame(records);frame.to_csv(out/'completed_cases.csv',index=False)
    pd.concat(components,ignore_index=True).to_csv(out/'six_metric_components.csv',index=False)
    aggregate=[]
    for arm,group in frame.groupby('arm',sort=False):
        expected=[i for i in plan['cases'] if i['arm']==arm]
        done=len(group)==len(expected)
        value=group.groupby('context').replicate_overall.mean()
        delta=group.groupby('context').delta_vs_V28.mean()
        aggregate.append({'arm':arm,'complete':done,'cases':len(group),'expected_cases':len(expected),
            'overall':float(value.mean()) if done else None,'delta_vs_V28':float(delta.mean()) if done else None,
            'per_background':value.to_dict(),
            'raw_metrics':group[list(native.METRICS)].mean().to_dict() if done else None})
    state={'status':'complete' if len(records)==len(plan['cases']) else 'running',
        'benchmark':plan['benchmark'],'score_contract_sha256':plan['score_contract_sha256'],
        'official_score_queries':0,'aggregates':aggregate}
    native.write(out/'progress.json',state)
    print('NATIVE_BATCH_PROGRESS',json.dumps(state,ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=6);args=parser.parse_args()
    plan=native.read(args.plan);out=native.owned(plan['output']);out.mkdir(parents=True,exist_ok=True)
    contract=native.PROJECT/'configs/V2_NATIVE_SCORE_CONTRACT.json'
    if native.sha(contract)!=plan['score_contract_sha256']:raise AssertionError('Judge contract changed')
    native.write(out/'executed_plan.json',{'plan':plan,'plan_sha256':native.sha(args.plan)})
    collect(plan,out)
    for item in plan['cases']:
        job=native.find_job(item['job_key']);arm=item['arm'];case=out/job['key']
        if item.get('effects_sha256') and native.sha(item['effects']) != item['effects_sha256']:
            raise AssertionError('Frozen effect bundle changed after plan creation')
        if (case/f'scores/{arm}/summary.json').exists():continue
        pred=case/f'pred__{arm}.h5ad';manifest=case/f'{arm}_generation.json'
        if pred.exists():
            if not manifest.exists() or native.sha(pred)!=native.read(manifest)['prediction_sha256']:
                raise AssertionError('Incomplete or changed new prediction; inspect before resume')
        else:
            print('NATIVE_BATCH_GENERATE',job['role'],arm,flush=True)
            start=time.time()
            if item.get('generator','literal')=='literal':
                native.generate_case(job,case,arm,effects_path=item.get('effects'),theta_scale=item.get('theta_scale',1.))
            elif item['generator']=='state':
                path=native.PROJECT/'staging/newV3_ntc_denoising/native_adapter.py'
                spec=importlib.util.spec_from_file_location('state_native_adapter',path)
                adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
                adapter.generate_case(job,case,arm,artifact_root=item.get('artifact_root'),
                    emission=item.get('emission','frozen'),state_pairing=item.get('state_pairing','true'))
            elif item['generator']=='carrier':
                path=native.PROJECT/'staging/newV4_carrier_readout/native_adapter.py'
                spec=importlib.util.spec_from_file_location('carrier_native_adapter',path)
                adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
                adapter.generate_case(job,case,arm,allocation_arm=item['allocation_arm'])
            elif item['generator']=='barycenter':
                path=native.PROJECT/'staging/newV4_count_barycenter/native_adapter.py'
                spec=importlib.util.spec_from_file_location('barycenter_native_adapter',path)
                adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
                adapter.generate_case(job,case,arm,weight=item['weight'])
            else:raise ValueError('Unknown predeclared generator')
            print('COUNT_SECONDS',round(time.time()-start,3),flush=True)
        result=native.score_case(case,[arm],args.threads)[arm]
        print('NATIVE_BATCH_SCORED',job['role'],arm,result['replicate_overall'],flush=True)
        collect(plan,out)
    collect(plan,out)

if __name__=='__main__':main()
