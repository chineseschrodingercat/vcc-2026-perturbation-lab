"""Verify and restore reviewed Release model assets. Never replaces different bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

MAX_UNCOMPRESSED_BYTES = 256 * 2**20
SHA = re.compile(r'^[a-f0-9]{64}$')

def safe_path(workspace: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative or '\x00' in relative:
        raise ValueError('Non-portable asset path')
    parts=relative.split('/')
    if relative.startswith('/') or any(x in ['', '.', '..'] for x in parts):
        raise ValueError('Unsafe asset path')
    root=workspace.resolve(); current=root
    for part in parts:
        current=current/part
        if current.is_symlink():raise ValueError('Asset path contains a symbolic link')
    if not current.resolve().is_relative_to(root):raise ValueError('Asset escapes workspace')
    return current

def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def restore(archive: Path, workspace: Path, dry_run: bool = False) -> dict:
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(set(names))!=len(names):raise ValueError('Duplicate archive member')
        if sum(x.file_size for x in z.infolist())>MAX_UNCOMPRESSED_BYTES:raise ValueError('Archive exceeds model-only resource bound')
        if any(x.is_dir() for x in z.infolist()):raise ValueError('Unexpected directory archive entry')
        manifest=json.loads(z.read('RESTORE_MANIFEST.json'))
        if manifest.get('schema_version')!=1 or not isinstance(manifest.get('assets'),list):raise ValueError('Unsupported restore manifest')
        expected={'RESTORE_MANIFEST.json'}; prepared=[]; paths=set()
        for x in manifest['assets']:
            path=x['path'];sha=x['sha256'];blob=x['blob']
            dest=safe_path(workspace,path)
            if path in paths:raise ValueError('Duplicate restore path')
            paths.add(path)
            if x.get('role')!='model-asset' or not SHA.fullmatch(sha) or type(x['bytes']) is not int or x['bytes']<0:
                raise ValueError('Invalid model asset specification')
            if not re.fullmatch(r'blobs/'+sha+r'\.(pt|npz)',blob):raise ValueError('Unexpected content-addressed model blob')
            if PurePosixPath(path).suffix not in ['.pt','.npz'] or 'raw' in path.split('/'):
                raise ValueError('Only reviewed fitted model assets may be restored')
            b=z.read(blob)
            if len(b)!=x['bytes'] or hashlib.sha256(b).hexdigest()!=sha:raise ValueError('Model asset hash/size mismatch')
            expected.add(blob)
            if dest.exists() and (not dest.is_file() or dest.stat().st_size!=x['bytes'] or file_sha(dest)!=sha):
                raise ValueError('Refusing to overwrite a conflicting destination: '+path)
            prepared.append((dest,b,sha))
        if set(names)!=expected:raise ValueError('Archive has unexpected or missing members')
        created=0;identical=0
        for dest,b,sha in prepared:
            if dest.exists():identical+=1;continue
            if dry_run:continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive create also refuses a new conflict appearing after preflight.
            with dest.open('xb') as f:f.write(b)
            if file_sha(dest)!=sha:raise ValueError('Post-write integrity failure')
            created+=1
        return {'verified':True,'family':manifest['family'],'logical_paths':len(prepared),'created':created,'already_identical':identical,'dry_run':dry_run,'RNA_or_scientific_execution':False}

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    try: print(json.dumps(restore(a.archive,a.workspace,a.dry_run),indent=2));return 0
    except (ValueError,KeyError,OSError,zipfile.BadZipFile) as e:p.exit(2,'error: '+str(e)+'\n')

if __name__=='__main__':raise SystemExit(main())
