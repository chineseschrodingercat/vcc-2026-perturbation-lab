"""Download official inputs after a local VCC login; never submits a prediction."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    cli = Path(sys.executable).with_name('vcc.exe' if sys.platform == 'win32' else 'vcc')
    subprocess.run([str(cli), 'whoami'], check=True)
    subprocess.run([str(cli), 'datasets', 'list'], check=True)
    for phase, dataset, zipname, contexts in [
        ('final','controls-final','vcc_2026_controls_final.zip', ['D','E','F']),
        ('validation','controls','vcc_2026_controls.zip', ['A','B','C'])]:
        folder = root/'data'/'official'/phase
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(cli),'datasets','download',dataset,'-d',str(folder)],check=True)
        archive = folder/zipname
        wanted = {f'context_{x}.h5ad' for x in contexts} | {'gene_names.csv','pert_counts.csv','manifest.json'}
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != len(set(z.namelist())):
                raise ValueError('Duplicate official ZIP members')
            if not wanted.issubset(z.namelist()):
                raise ValueError(f'Official bundle layout changed: {z.namelist()}')
            required = sum(z.getinfo(name).file_size for name in wanted)
            if required > 20*10**9 or shutil.disk_usage(root).free-required < 30*10**9:
                raise ValueError('Unexpected extraction size or insufficient disk')
            for name in sorted(wanted):
                dest = folder/name
                partial = folder/(name+'.extracting')
                with z.open(name) as src, partial.open('wb') as out:
                    shutil.copyfileobj(src,out,4*1024**2)
                partial.replace(dest)
        h=hashlib.sha256()
        with archive.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024**2),b''):
                h.update(block)
        (folder/'download_receipt.json').write_text(json.dumps({
            'dataset_id':dataset,'archive_sha256':h.hexdigest(),
            'expected_contexts':contexts,'extracted_files':sorted(wanted),
            'note':'CLI verifies official download checksum; ZIP extraction checks entry CRC. No submission made.'
        },indent=2)+'\n')
        print(json.dumps({'phase':phase,'directory':str(folder),'state':'ready'}),flush=True)


if __name__=='__main__':
    main()
