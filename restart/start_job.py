"""Launch one named local background job, retaining its log and process id."""
from pathlib import Path
import argparse
import ctypes
import json
import os
import subprocess
import sys
from runtime_profile import get_runtime_profile,thread_environment


def process_running(pid):
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
        kernel.GetExitCodeProcess.restype=ctypes.c_int
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        kernel.CloseHandle.restype=ctypes.c_int
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            return False
        exit_code=ctypes.c_uint32()
        kernel.GetExitCodeProcess(handle,ctypes.byref(exit_code))
        kernel.CloseHandle(handle)
        return exit_code.value==259
    try:
        os.kill(int(pid),0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name')
    p.add_argument('script')
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if not a.name.replace('_','').isalnum():
        raise ValueError('Use a simple job name')
    script=(root/a.script).resolve()
    if not script.is_relative_to(root) or not script.is_file():
        raise ValueError('Script must be in this project')
    logs=root/'logs'
    logs.mkdir(exist_ok=True)
    pidfile=logs/f'{a.name}.launcher.json'
    if pidfile.exists():
        old=json.loads(pidfile.read_text())
        if process_running(old['pid']):
            print(json.dumps({'already_running':True,'pid':old['pid'],'name':a.name}))
            return
    profile=get_runtime_profile()
    env=thread_environment(profile['scientific_threads'],profile=profile)
    with (logs/f'{a.name}.stdout.log').open('ab',buffering=0) as out,(logs/f'{a.name}.stderr.log').open('ab',buffering=0) as err:
        command=[sys.executable,'-u',str(script),*a.arguments]
        flags={'creationflags':subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
        child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            close_fds=True,env=env,**flags)
    pidfile.write_text(json.dumps({'pid':child.pid,'command':command,'name':a.name,'runtime_profile':profile},indent=2))
    print(json.dumps({'name':a.name,'pid':child.pid,'stdout':str(logs/f'{a.name}.stdout.log'),'runtime_profile':profile}))


if __name__=='__main__':
    main()
