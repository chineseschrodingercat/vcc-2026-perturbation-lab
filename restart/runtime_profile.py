"""Report bounded local CPU/memory concurrency without allocating a memory pool.

On a 16-logical-CPU machine the defaults are 12 BLAS threads for one scientific
job, four ingest processes with three threads each, and six count workers with
a process BLAS thread limit of two. V28 may use CUDA through device=auto.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path

GIB=1024**3
MEMORY_TARGET=100*GIB
MEMORY_HEADROOM=16*GIB


def memory_information():
    """Read physical/available memory; no subprocess, arrays or reservations."""
    try:
        import psutil
        info=psutil.virtual_memory()
        result={"total_bytes":int(info.total),"available_bytes":int(info.available),"source":"psutil"}
    except (ImportError,OSError,AttributeError):
        result={"total_bytes":None,"available_bytes":None,"source":"unavailable"}
        if os.name=="nt":
            class MemoryStatus(ctypes.Structure):
                _fields_=[("length",ctypes.c_uint32),("load",ctypes.c_uint32),
                          ("total_phys",ctypes.c_uint64),("avail_phys",ctypes.c_uint64),
                          ("total_page",ctypes.c_uint64),("avail_page",ctypes.c_uint64),
                          ("total_virtual",ctypes.c_uint64),("avail_virtual",ctypes.c_uint64),
                          ("avail_extended",ctypes.c_uint64)]
            info=MemoryStatus();info.length=ctypes.sizeof(info)
            kernel=ctypes.WinDLL("kernel32",use_last_error=True)
            kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(MemoryStatus)]
            kernel.GlobalMemoryStatusEx.restype=ctypes.c_int
            if kernel.GlobalMemoryStatusEx(ctypes.byref(info)):
                result={"total_bytes":int(info.total_phys),"available_bytes":int(info.avail_phys),
                        "source":"GlobalMemoryStatusEx"}
        elif Path("/proc/meminfo").is_file():
            fields={line.split(":",1)[0]:int(line.split()[1])*1024
                    for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
            result={"total_bytes":fields.get("MemTotal"),"available_bytes":fields.get("MemAvailable"),
                    "source":"/proc/meminfo"}
        else:
            try:
                page=os.sysconf("SC_PAGE_SIZE")
                result={"total_bytes":os.sysconf("SC_PHYS_PAGES")*page,
                        "available_bytes":os.sysconf("SC_AVPHYS_PAGES")*page,"source":"sysconf"}
            except (OSError,ValueError,AttributeError):
                pass
    # Respect a Linux container's memory ceiling if it is tighter than the host.
    maximum=Path("/sys/fs/cgroup/memory.max")
    current=Path("/sys/fs/cgroup/memory.current")
    if maximum.is_file() and current.is_file():
        try:
            text=maximum.read_text().strip()
            if text!="max":
                limit=int(text);remaining=max(0,limit-int(current.read_text().strip()))
                if result["total_bytes"] is None or limit<result["total_bytes"]:
                    result["total_bytes"]=limit
                    result["available_bytes"]=min(remaining,result["available_bytes"]) if result["available_bytes"] is not None else remaining
                    result["source"]+=" + cgroup v2"
        except (OSError,ValueError):
            pass
    return result


def logical_cpu_count():
    if hasattr(os,"process_cpu_count"):
        count=os.process_cpu_count() or 1
    elif hasattr(os,"sched_getaffinity"):
        count=len(os.sched_getaffinity(0))
    else:
        count=os.cpu_count() or 1
    quota=Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        try:
            capacity,period=quota.read_text().strip().split()
            if capacity!="max":
                count=min(count,max(1,(int(capacity)+int(period)-1)//int(period)))
        except (OSError,ValueError):
            pass
    return max(1,int(count))


def get_runtime_profile(*,logical_cpus=None,memory_info=None):
    logical=logical_cpu_count() if logical_cpus is None else max(1,int(logical_cpus))
    memory=memory_information() if memory_info is None else dict(memory_info)
    threads=max(1,(logical*3)//4)
    ingest_workers=min(4,max(1,threads//3))
    count_workers=min(6,max(1,threads//2))
    available=memory.get("available_bytes")
    budget=min(MEMORY_TARGET,max(0,int(available)-MEMORY_HEADROOM)) if available is not None else None
    return {"schema":"vcc_runtime_profile_v1","recorded_utc":datetime.now(timezone.utc).isoformat(),"logical_cpus":logical,
            "scientific_threads":threads,"ingest_workers":ingest_workers,
            "ingest_threads":max(1,threads//ingest_workers),"count_workers":count_workers,
            "count_threads":max(1,threads//count_workers),
            "memory":memory,"memory_target_bytes":MEMORY_TARGET,
            "memory_headroom_bytes":MEMORY_HEADROOM,"memory_budget_bytes":budget,
            "memory_budget_is_os_limit":False,
            "allocation_policy":"On-demand bounded buffers; no memory reservation or preallocation",
            "gpu_policy":"Frozen V28 device=auto selects CUDA if available; CPU pipelines stay CPU"}


def thread_environment(threads,*,profile=None,base=None):
    threads=max(1,int(threads))
    env=dict(os.environ if base is None else base)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
                "NUMEXPR_MAX_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
        env[key]=str(threads)
    env.update(OMP_DYNAMIC="FALSE",MKL_DYNAMIC="FALSE",PYTHONUTF8="1")
    if profile is not None:
        env["VCC_RUNTIME_PROFILE"]=json.dumps(profile,separators=(",",":"))
        if profile.get("memory_budget_bytes") is not None:
            env["VCC_MEMORY_BUDGET_BYTES"]=str(profile["memory_budget_bytes"])
    return env


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,help="Also persist the measured runtime profile as JSON")
    args=parser.parse_args()
    profile=get_runtime_profile()
    rendered=json.dumps(profile,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.output.with_name(args.output.name+".tmp")
        temporary.write_text(rendered,encoding="utf-8")
        temporary.replace(args.output)
    print(rendered,end="")
