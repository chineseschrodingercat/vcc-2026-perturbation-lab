"""Bounded, resumable public dataset downloads for the VCC 2026 restart."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request
from datetime import datetime, timezone

LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def project_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def download(entry, root, budget):
    relative = Path(entry['path'])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Unsafe destination: {relative}; root={root}; resolved={path}')
    expected = int(entry['bytes'])
    if expected <= 0 or expected > budget:
        raise ValueError('An exact positive source size within budget is required')
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix + '.receipt.json')
    statefile = root / 'logs' / 'downloads' / (entry['id'] + '.json')
    base = {'id': entry['id'], 'path': str(relative), 'url': entry['url'],
            'source_page': entry.get('source_page'), 'expected_bytes': expected}
    if path.exists():
        if path.stat().st_size != expected:
            raise ValueError(f'Existing completed file has a conflicting size: {path}')
        previous = json.loads(receipt.read_text()) if receipt.exists() else {}
        if previous.get('verified') and previous.get('mtime_ns') == path.stat().st_mtime_ns:
            atomic_json(statefile, {**base, 'state': 'complete', 'downloaded_bytes': expected,
                                   'verified': True, 'reused': True, 'updated_utc': now()})
            return {'id': entry['id'], 'state': 'complete', 'reused': True}
        temporary = path
    else:
        temporary = path.with_suffix(path.suffix + '.part')
        last_error = None
        for attempt in range(1, 7):
            start = temporary.stat().st_size if temporary.exists() else 0
            if start == expected:
                break
            if start > expected:
                raise ValueError(f'Partial file exceeds declared source size: {temporary}')
            try:
                headers = {'User-Agent': 'VCC2026-research-restart/1.0', 'Accept-Encoding': 'identity'}
                if start:
                    headers['Range'] = f'bytes={start}-'
                request = urllib.request.Request(entry['url'], headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status == 206:
                        content_range = response.headers.get('Content-Range', '')
                        if not content_range.startswith(f'bytes {start}-') or not content_range.endswith(f'/{expected}'):
                            raise ValueError(f'Unexpected range: {content_range}')
                        mode = 'ab'
                    elif response.status == 200:
                        start = 0
                        mode = 'wb'
                    else:
                        raise ValueError(f'Unexpected HTTP status {response.status}')
                    length = response.headers.get('Content-Length')
                    if length and int(length) + start != expected:
                        raise ValueError(f'Source size changed: {length}+{start} != {expected}')
                    done = start
                    t0 = last = time.monotonic()
                    with temporary.open(mode) as handle:
                        while True:
                            block = response.read(4 * 1024**2)
                            if not block:
                                break
                            done += len(block)
                            if done > expected:
                                raise ValueError('Server sent more bytes than registered')
                            if shutil.disk_usage(root).free < 30 * 10**9:
                                raise OSError('Stopping with 30 GB disk reserve')
                            handle.write(block)
                            t = time.monotonic()
                            if t - last > 5:
                                atomic_json(statefile, {**base, 'state': 'downloading', 'attempt': attempt,
                                    'downloaded_bytes': done, 'bytes_per_second': (done-start)/max(t-t0,.01),
                                    'updated_utc': now()})
                                last = t
                    if done != expected:
                        raise IOError(f'Incomplete transfer: {done}/{expected}')
                    break
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
                atomic_json(statefile, {**base, 'state': 'retrying' if attempt < 6 else 'failed',
                    'attempt': attempt, 'error': last_error, 'updated_utc': now()})
                if attempt == 6:
                    raise
                time.sleep(min(5 * 2**(attempt-1), 60))
        if not temporary.exists() or temporary.stat().st_size != expected:
            raise IOError(last_error or 'Transfer did not complete')
    atomic_json(statefile, {**base, 'state': 'verifying', 'downloaded_bytes': expected, 'updated_utc': now()})
    hashes = {'sha256': digest(temporary, 'sha256')}
    if entry.get('md5'):
        hashes['md5'] = digest(temporary, 'md5')
    for alg in ['sha256', 'md5']:
        if entry.get(alg) and hashes[alg].lower() != entry[alg].lower():
            atomic_json(statefile, {**base, 'state': 'failed', 'error': f'{alg} mismatch', 'updated_utc': now()})
            raise ValueError(f'{entry["id"]}: {alg} mismatch; source file left for inspection')
    if temporary != path:
        os.replace(temporary, path)
    record = {**base, **hashes, 'verified': True,
              'publisher_checksum_checked': bool(entry.get('sha256') or entry.get('md5')),
              'mtime_ns': path.stat().st_mtime_ns, 'completed_utc': now()}
    atomic_json(receipt, record)
    atomic_json(statefile, {**base, 'state': 'complete', 'downloaded_bytes': expected, 'verified': True,
                           'updated_utc': now()})
    with LOCK:
        print(json.dumps({'id': entry['id'], 'state': 'complete', 'bytes': expected}), flush=True)
    return {'id': entry['id'], 'state': 'complete'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--max-bytes', type=int, default=1_000_000_000_000)
    p.add_argument('--ids', nargs='*')
    a = p.parse_args()
    root = a.root.resolve()
    entries = json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    if isinstance(entries, dict):
        entries = entries['files']
    if a.ids:
        entries = [x for x in entries if x['id'] in a.ids]
        if set(a.ids) != {x['id'] for x in entries}:
            raise ValueError('Unknown requested dataset ID')
    if len({x['id'] for x in entries}) != len(entries) or len({x['path'] for x in entries}) != len(entries):
        raise ValueError('Duplicate dataset identity or path')
    additional = sum(max(0, int(x['bytes']) - ((root/x['path']).stat().st_size if (root/x['path']).exists() else
                       (root/(x['path']+'.part')).stat().st_size if (root/(x['path']+'.part')).exists() else 0))
                     for x in entries)
    used = project_bytes(root)
    if used + additional > a.max_bytes:
        raise ValueError(f'1 TB project budget exceeded: {used}+{additional}>{a.max_bytes}')
    if shutil.disk_usage(root).free - additional < 30*10**9:
        raise OSError('Insufficient free disk with 30 GB reserve')
    # Establish destination directories before concurrent Windows path resolution.
    for entry in entries:
        dest = root / entry['path']
        if Path(entry['path']).is_absolute() or '..' in Path(entry['path']).parts:
            raise ValueError('Non-portable destination path')
        dest.parent.mkdir(parents=True, exist_ok=True)
    status = root/'logs'/'download_job.json'
    results = []
    atomic_json(status, {'pid': os.getpid(), 'state': 'running', 'started_utc': now(),
                         'planned_bytes': sum(int(x['bytes']) for x in entries), 'max_bytes': a.max_bytes,
                         'files': [x['id'] for x in entries]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(a.workers,4))) as executor:
        tasks = {executor.submit(download,x,root,a.max_bytes):x for x in entries}
        for task in concurrent.futures.as_completed(tasks):
            entry = tasks[task]
            try:
                results.append(task.result())
            except Exception as exc:
                result = {'id': entry['id'], 'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                results.append(result)
                print(json.dumps(result), flush=True)
    failures = [x for x in results if x['state'] != 'complete']
    atomic_json(status, {'pid': os.getpid(), 'state': 'complete' if not failures else 'partial_failure',
                         'finished_utc': now(), 'results': results, 'max_bytes': a.max_bytes})
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
