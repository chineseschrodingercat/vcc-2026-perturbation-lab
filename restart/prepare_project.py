"""Aggregate verified source data and run a compact background-held-out comparison."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from runtime_profile import get_runtime_profile,thread_environment

SOURCES = [
    ('k562_gwps','K562','data/raw/external/replogle_2022/k562_gwps/K562_gwps_raw_bulk_01.h5ad','raw-bulk'),
    ('k562_essential','K562','data/raw/external/replogle_2022/k562_essential/K562_essential_raw_bulk_01.h5ad','raw-bulk'),
    ('rpe1_bulk','RPE1','data/raw/external/replogle_2022/rpe1/rpe1_raw_bulk_01.h5ad','raw-bulk'),
    ('hepg2_sc','HepG2','data/raw/external/scperturb/NadigOConner2024_hepg2.h5ad','singlecell'),
    ('jurkat_sc','Jurkat','data/raw/external/scperturb/NadigOConner2024_jurkat.h5ad','singlecell'),
    ('rpe1_sc','RPE1','data/raw/external/scperturb/ReplogleWeissman2022_rpe1.h5ad','singlecell'),
]


def record(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-targets',type=int,default=512)
    p.add_argument('--skip-cv',action='store_true')
    p.add_argument('--core-only',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    script=root/'restart'/'lean_transfer.py'
    logs=root/'logs'
    logs.mkdir(exist_ok=True)
    state=logs/'preparation_job.json'
    profile=get_runtime_profile()
    record(state,{'state':'running','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'runtime_profile':profile})
    selected=SOURCES[:3] if a.core_only else SOURCES
    ingest_env=thread_environment(profile['ingest_threads'],profile=profile)
    science_env=thread_environment(profile['scientific_threads'],profile=profile)
    print(json.dumps({'runtime_profile':profile}),flush=True)

    def ingest(source):
        sid,bg,relative,kind=source
        input_path=root/relative
        receipt_path=input_path.with_suffix(input_path.suffix+'.receipt.json')
        if not input_path.exists() or not receipt_path.exists():
            raise FileNotFoundError(f'{sid}: download has not completed; run RESUME_DOWNLOADS.cmd first')
        receipt=json.loads(receipt_path.read_text())
        if not receipt.get('verified') or input_path.stat().st_size!=receipt['expected_bytes']:
            raise ValueError(f'{sid}: missing valid download receipt')
        output=root/'data'/'processed'/'restart'/f'{sid}.npz'
        meta=output.with_suffix('.json')
        if output.exists():
            if not meta.exists():
                raise ValueError(f'{sid}: uncommitted partial aggregate; inspect {output}')
            report=json.loads(meta.read_text())
            if report['source_id']!=sid or report['background']!=bg or report['input_bytes']!=input_path.stat().st_size:
                raise ValueError(f'{sid}: existing aggregate belongs to different source')
            print(f'reuse {sid}',flush=True)
        else:
            command=[sys.executable,'-u',str(script),'ingest','--input',str(input_path),'--output',str(output),
                     '--source-id',sid,'--background',bg,'--kind',kind,'--chunk-rows','2048']
            if kind=='raw-bulk':
                command += ['--target-col','construct-index','--gene-col','gene_name','--n-col','num_cells_filtered','--control-col','core_control']
            else:
                command += ['--target-col','gene','--gene-col','_index','--control-label','non-targeting']
            with (logs/f'ingest_{sid}.log').open('w',encoding='utf-8') as out:
                subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True,env=ingest_env,cwd=root)
            report=json.loads(meta.read_text())
            report['input_sha256']=receipt['sha256']
            report['source_checksum_receipt']=str(receipt_path.relative_to(root))
            record(meta,report)
            print(f'completed {sid}: {report["targets"]} targets, {report["genes"]} genes',flush=True)
        return str(output)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=profile['ingest_workers']) as pool:
            prepared=list(pool.map(ingest,selected))
        output=root/'outputs'/'restart'/'core_cv'
        if not a.skip_cv:
            command=[sys.executable,'-u',str(script),'cv','--sources',*prepared,
                     '--output-dir',str(output),'--max-targets',str(a.max_targets)]
            with (logs/'core_cv.log').open('w',encoding='utf-8') as out:
                subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True,cwd=root,env=science_env)
        record(state,{'state':'complete','pid':os.getpid(),'sources':prepared,'core_cv':str(output),
                      'finished_utc':datetime.now(timezone.utc).isoformat(),
                      'official_data_ready':(root/'data/official/final/manifest.json').exists(),
                      'submission_performed':False,'runtime_profile':profile})
        print('Preparation and requested local comparison completed.',flush=True)
    except Exception as exc:
        record(state,{'state':'failed','pid':os.getpid(),'error':f'{type(exc).__name__}: {exc}',
                      'details':'See logs/ingest_*.log or logs/core_cv.log','submission_performed':False,'runtime_profile':profile})
        raise


if __name__=='__main__':
    main()
