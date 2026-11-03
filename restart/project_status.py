"""Show the actual restart state without opening credentials or matrix payloads."""
from pathlib import Path
import csv
import json
from datetime import datetime, timezone


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def main():
    root=Path(__file__).resolve().parents[1]
    entries=read(root/'restart/download_manifest.json')
    downloads=[]
    for x in entries:
        path=root/x['path']
        receipt=read(path.with_suffix(path.suffix+'.receipt.json'))
        actual=path.stat().st_size if path.exists() else 0
        complete=bool(receipt and receipt.get('verified') and actual==x['bytes']
                      and receipt.get('mtime_ns')==path.stat().st_mtime_ns)
        downloads.append({'id':x['id'],'bytes':actual,'expected_bytes':x['bytes'],
                          'complete':complete,'publisher_checksum_checked':receipt.get('publisher_checksum_checked',False) if receipt else False,
                          'sha256':receipt.get('sha256') if receipt else None})
    sources=[]
    for path in sorted((root/'data/processed/restart').glob('*.json')):
        x=read(path)
        sources.append({k:x.get(k) for k in ['source_id','background','source_rows_used','targets','genes','control_n_cells','input_sha256']})
    experiments={}
    for name in ['core_cv','union_cv']:
        folder=root/'outputs/restart'/name
        model=read(folder/'selected_model.json')
        report=read(folder/'cv_report.json')
        if model and report:
            selected=next(c for c in report['candidates'] if c['id']==model['id'])
            zero=next(c for c in report['candidates'] if c['id']=='zero')
            by_background={}
            with (folder/'proxy_per_target.csv').open(newline='') as f:
                for row in csv.DictReader(f):
                    if row['candidate'] not in ['zero',model['id']]:
                        continue
                    values=by_background.setdefault(row['heldout_background'],{}).setdefault(row['candidate'],[])
                    values.append(float(row['proxy_delta_mse_excluding_target']))
            by_background={bg:{candidate:sum(values)/len(values) for candidate,values in table.items()} for bg,table in by_background.items()}
            experiments[name]={'selected':{k:model[k] for k in ['id','model','alpha','temperature']},
                'normalization_genes':report.get('common_normalization_genes',report.get('common_measured_genes')),
                'output_genes_before_official_axis':report.get('union_output_genes',report.get('common_measured_genes')),
                'zero_proxy_mse':zero['proxy_delta_mse_excluding_target'],
                'selected_proxy_mse':selected['proxy_delta_mse_excluding_target'],
                'selected_proxy_cosine':selected['proxy_delta_cosine'],
                'source_locks_count':len(model.get('source_locks',{})),
                'per_background_proxy_mse':by_background,
                'interpretation':'External background-held-out development tuning; same folds used to select hyperparameters. Not an official score or independent performance estimate.'}
    official={phase:all((root/'data/official'/phase/name).exists() for name in
              ['manifest.json','gene_names.csv','pert_counts.csv',*[f'context_{c}.h5ad' for c in contexts]])
              for phase,contexts in [('validation','ABC'),('final','DEF')]}
    storage=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    audits={}
    for name in ['extra_feng_audit','extra_cd4t_audit']:
        value=read(root/'reports'/f'{name}.json')
        if value:
            audits[name]={'status':value['status'],'datasets':[{'id':x['id'],'status':x['status'],
                'targets':x.get('selected_unique_targets',x.get('target_coverage',{}).get('unique_observed')),
                'genes':x.get('selected_unique_genes',x.get('n_genes'))} for x in value['datasets']]}
    smoke_checks={name:read(root/'reports'/f'{name}.json') for name in
                  ['real_count_smoke','recovered_v28_smoke','recovered_v28_gpu_smoke']}
    benchmark=read(root/'reports/count_worker_benchmark.json')
    benchmark_summary={k:benchmark.get(k) for k in ['status','generator','cells_per_run',
        'timing_scope','all_streams_identical','worker_summary','speedup_6_over_1','limitations']} if benchmark else None
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'project_root':str(root),
            'authorized_storage_cap_bytes':1_000_000_000_000,'project_bytes':storage,
            'downloaded_bytes':sum(x['bytes'] for x in downloads),'completed_downloads':sum(x['complete'] for x in downloads),
            'planned_downloads':len(downloads),'downloads':downloads,'sources':sources,'experiments':experiments,
            'extra_data_audits':audits,'engineering_checks':smoke_checks,
            'resource_profile':read(root/'reports/RUNTIME_PROFILE.json'),
            'count_worker_benchmark':benchmark_summary,
            'official_input_bundles_present':official,
            'official_final_artifacts':[str(p.relative_to(root)) for p in (root/'outputs/restart').rglob('*.vcc')],
            'competition_submissions_performed_by_restart':False}
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports/RESTART_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Project: {root}')
    print(f'Downloads: {result["completed_downloads"]}/{len(downloads)} complete; {result["downloaded_bytes"]/1e9:.3f} GB')
    print(f'Project storage: {storage/1e9:.3f} GB / 1000 GB allowed')
    print(f'Prepared source banks: {len(sources)}; distinct backgrounds: {len({x["background"] for x in sources})}')
    for name,x in experiments.items():
        print(f'{name}: {x["selected"]["id"]}; proxy MSE {x["selected_proxy_mse"]:.8f}; zero {x["zero_proxy_mse"]:.8f}; input locks {x["source_locks_count"]}')
    print(f'Official validation inputs: {official["validation"]}; final inputs: {official["final"]}')
    print('No competition submission is performed by this restart workflow.')
    print('Full status: reports/RESTART_STATUS.json')


if __name__=='__main__':
    main()
