from pathlib import Path
import json, os, subprocess, sys
root = Path(__file__).resolve().parents[1]
log = root / 'logs'
log.mkdir(exist_ok=True)
job = log / 'download_launcher.json'
if job.exists():
    old = json.loads(job.read_text())
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(old['pid']))
    if h:
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        if code.value == 259:
            print(json.dumps({'already_running': True, 'pid': old['pid']}))
            raise SystemExit(0)
with (log/'downloads.stdout.log').open('ab',buffering=0) as out, (log/'downloads.stderr.log').open('ab',buffering=0) as err:
    p = subprocess.Popen([sys.executable, '-u', str(root/'restart/download_data.py'), '--root', str(root), '--manifest', str(root/'restart/download_manifest.json'), '--workers', '3'], cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP, close_fds=True)
    job.write_text(json.dumps({'pid':p.pid,'manifest':'download_manifest.json'},indent=2))
    print(json.dumps({'pid':p.pid,'log':str(log/'downloads.stdout.log')}))
