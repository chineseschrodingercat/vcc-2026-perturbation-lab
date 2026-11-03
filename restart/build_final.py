"""Build and locally validate a final VCC artifact; never upload or submit it.

Default: re-run measured-union LOCO calibration on the official gene axis,
generate D/E/F predictions, and package them with the installed official CLI.
The v28 recipe is a reconstructed historical fallback, not independent LOCO.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from runtime_profile import get_runtime_profile,thread_environment

SOURCE_IDS=("k562_gwps","k562_essential","rpe1_bulk","hepg2_sc","jurkat_sc","rpe1_sc")
PROJECT_BUDGET=1_000_000_000_000
GENERATION_RESERVE=80_000_000_000


def utc():
    return datetime.now(timezone.utc).isoformat()


def project_bytes(root):
    total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[name for name in dirs if not (Path(directory)/name).is_symlink()]
        for name in files:
            path=Path(directory)/name
            if not path.is_symlink():
                total+=path.stat().st_size
    return total


def space_check(root,reserve=0):
    used=project_bytes(root)
    free=shutil.disk_usage(root).free
    if used+reserve>PROJECT_BUDGET:
        raise RuntimeError(f"Project budget: {used:,} used + {reserve:,} reserved exceeds 1,000,000,000,000 bytes")
    if free<reserve:
        raise RuntimeError(f"Insufficient disk: {free:,} free; {reserve:,} bytes required before generation")
    return {"project_bytes":used,"disk_free_bytes":free,"project_budget_bytes":PROJECT_BUDGET,"reserved_bytes":reserve}


def missing_message(root,missing):
    print("Final artifact was not built. Required inputs are missing:",flush=True)
    for path in missing:
        print(f"  {path}",flush=True)
    print(f"Official-data login/download helper: {root/'LOGIN_AND_GET_OFFICIAL_DATA.cmd'}",flush=True)
    print("Complete source preparation if processed NPZs are missing. No official inputs were fabricated; no submission was made.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--recipe",choices=("union","core","v28"),default="union")
    parser.add_argument("--max-targets",type=int,default=512,help="Metadata-selected targets per external LOCO background")
    parser.add_argument("--generator",choices=("nb","poisson"),default="nb")
    parser.add_argument("--seed",type=int,default=20261103)
    parser.add_argument("--device",default="auto",help="Frozen V28 device; auto selects CUDA when available")
    parser.add_argument("--workers",type=int,help="Count-generation workers; runtime default is 6 on a 16-logical-CPU machine")
    parser.add_argument("--vcc",type=Path,help="Official local vcc executable; defaults beside the active Python")
    args=parser.parse_args()
    if args.max_targets<1:
        parser.error("--max-targets must be positive")
    if args.workers is not None and args.workers<1:
        parser.error("--workers must be positive")
    root=args.root.resolve()
    bundle=root/"data/official/final"
    scripts=Path(__file__).resolve().parent
    sources=[root/"data/processed/restart"/f"{name}.npz" for name in SOURCE_IDS]
    cli=args.vcc or Path(sys.executable).with_name("vcc.exe" if os.name=="nt" else "vcc")
    if not cli.is_file() and args.vcc is None:
        found=shutil.which("vcc")
        if found:
            cli=Path(found)
    cli=cli.resolve()
    assets={
        "checkpoint":root/"archive/legacy_20261003/models/residual_v7/final_candidate/seed_20260928/tiny_transformer_600.pt",
        "features":root/"archive/legacy_20261003/models/residual_v7/final_candidate/seed_20260928/features.npz",
        "string_embeddings":root/"data/raw/external/slim_embeddings/gene_string_embeddings.v0.3.h5",
        "gene_axis":root/"snapshots/archive/legacy_20261003/runs/context_v4/gene_axis.txt",
        "normalization_core":root/"snapshots/archive/legacy_20261003/runs/context_v4/normalization_core.txt",
    }
    required_sources=sources[:1] if args.recipe=="v28" else sources
    required=[bundle/"manifest.json",bundle/"gene_names.csv",bundle/"pert_counts.csv",*required_sources,cli,scripts/"lean_transfer.py"]
    if args.recipe=="union":
        required.append(scripts/"lean_union.py")
    if args.recipe=="v28":
        required.extend([scripts/"legacy_v28_adapter.py",*assets.values()])
    missing=[path for path in required if not path.is_file()]
    if missing:
        missing_message(root,missing)
        return 2
    profile=get_runtime_profile()
    if args.workers is not None:
        if args.workers>profile['logical_cpus']:
            parser.error("--workers must not exceed the available logical CPU count")
        profile['count_workers']=args.workers
        profile['count_threads']=max(1,profile['scientific_threads']//args.workers)
    os.environ.update(thread_environment(profile['scientific_threads'],profile=profile))
    # Heavy libraries are imported only after missing-input preflight, so --help
    # and a clear blocked result do not require a fully built scientific runtime.
    import lean_transfer as lt
    try:
        bundle,official_manifest,contexts,control_paths,genes,targets=lt.read_bundle(bundle)
        if set(contexts)!={"D","E","F"}:
            raise ValueError(f"This final-only entry point requires manifest contexts D/E/F; found {contexts}")
        initial_space=space_check(root,GENERATION_RESERVE)
    except Exception as exc:
        print(f"Final input preflight failed: {type(exc).__name__}: {exc}",flush=True)
        print(f"Official-data helper: {root/'LOGIN_AND_GET_OFFICIAL_DATA.cmd'}",flush=True)
        print("No build directory or submission was created.",flush=True)
        return 2
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run=root/"outputs/restart"/f"final_{stamp}_{args.recipe}"
    run.mkdir(parents=True,exist_ok=False)
    receipt_path=run/"build_receipt.json"
    receipt={"status":"running","started_utc":utc(),"recipe":args.recipe,"run_directory":str(run),
             "official_bundle":str(bundle),"contexts":contexts,"seed":args.seed,"generator":args.generator,
             "max_targets_per_background":args.max_targets,"budget":initial_space,"steps":[],
             "runtime_profile":profile,"requested_v28_device":args.device,
             "official_manifest_sha256":lt.sha256(bundle/"manifest.json"),
             "gene_names_sha256":lt.sha256(bundle/"gene_names.csv"),
             "pert_counts_sha256":lt.sha256(bundle/"pert_counts.csv"),
             "submission_performed":False,"submission_allowance_consumed":False,
             "evaluation_status":"historical reconstructed fallback; not independent LOCO" if args.recipe=="v28"
                                 else "external LOCO proxy tuning on the official gene axis; not an official score"}
    lt.write_json(receipt_path,receipt)
    env=thread_environment(profile['scientific_threads'],profile=profile)
    count_env=thread_environment(profile['count_threads'],profile=profile)

    def step(name,command,environment=None):
        log=run/f"{name}.log"
        item={"name":name,"started_utc":utc(),"argv":[str(x) for x in command],"log":str(log),"status":"running"}
        receipt["steps"].append(item);lt.write_json(receipt_path,receipt)
        print(f"{name}: {log}",flush=True)
        with log.open("w",encoding="utf-8") as stream:
            result=subprocess.run([str(x) for x in command],cwd=root,env=env if environment is None else environment,stdout=stream,stderr=subprocess.STDOUT)
        item.update(finished_utc=utc(),returncode=result.returncode,status="complete" if result.returncode==0 else "failed")
        lt.write_json(receipt_path,receipt)
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}; inspect {log}")
        space_check(root)

    try:
        step("vcc_version",[cli,"--version"])
        h5ad=run/"final_prediction.h5ad"
        packaged=run/"final_prediction.vcc"
        effects=run/"effects"
        if args.recipe in ("union","core"):
            cv_script=scripts/("lean_union.py" if args.recipe=="union" else "lean_transfer.py")
            cv_dir=run/"calibration"
            step("calibrate",[sys.executable,"-u",cv_script,"cv","--sources",*sources,
                              "--genes",bundle/"gene_names.csv","--output-dir",cv_dir,
                              "--max-targets",str(args.max_targets),"--seed",str(args.seed)])
            model=cv_dir/"selected_model.json"
            selected=json.loads(model.read_text(encoding="utf-8"))
            if not selected.get("source_locks"):
                raise RuntimeError("Calibration did not produce source content locks; deploy the current scripts and rerun")
            receipt["source_locks"]=selected["source_locks"]
            receipt["selected_model"]={"path":str(model),"sha256":lt.sha256(model),"candidate":selected["id"]}
            if args.recipe=="union":
                step("effects",[sys.executable,"-u",scripts/"lean_union.py","effects","--bundle",bundle,
                                 "--sources",*sources,"--model-json",model,"--output-dir",effects])
                prediction_args=["--effects-dir",effects]
            else:
                prediction_args=["--sources",*sources,"--model-json",model]
        else:
            from legacy_v28_adapter import LegacyV28Adapter
            # Only K562 GWPS is a live response source for this frozen fallback.
            receipt["source_locks"]=lt.source_locks(lt.load_sources([sources[0]]))
            receipt["frozen_assets"]={name:{"path":str(path),"sha256":lt.sha256(path)} for name,path in assets.items()}
            adapter=LegacyV28Adapter(source_npz=sources[0],device=args.device,**assets)
            receipt['actual_v28_device']=adapter.device
            effects.mkdir()
            label_sources={}
            for context in contexts:
                audit={}
                mean,variance,n,columns=lt.control_statistics(control_paths[context],context,genes,
                    manifest_binding=control_paths,label_audit=audit)
                prediction=adapter.predict(targets,genes,mean)
                prediction.metadata.update(context=context,context_label_source=audit["context_label_source"],
                                           evaluation_status=receipt["evaluation_status"])
                prediction.save(effects/f"context_{context}.npz")
                label_sources[context]=audit
            receipt["context_label_sources"]=label_sources
            lt.write_json(receipt_path,receipt)
            prediction_args=["--effects-dir",effects]
        space_check(root,GENERATION_RESERVE)
        step("generate_counts",[sys.executable,"-u",scripts/"lean_transfer.py","predict","--bundle",bundle,
                                *prediction_args,"--output",h5ad,"--generator",args.generator,"--seed",str(args.seed),
                                "--workers",str(profile['count_workers'])],environment=count_env)
        step("official_prep",[cli,"prep",h5ad,"--pert-col","target_gene","--context-col","context",
                              "-g",bundle/"gene_names.csv","--perts",bundle/"pert_counts.csv","-o",packaged])
        if not packaged.is_file() or packaged.stat().st_size==0:
            raise RuntimeError("Official prep returned without a nonempty .vcc artifact")
        receipt.update(status="complete",finished_utc=utc(),final_space=space_check(root),
                       h5ad={"path":str(h5ad),"bytes":h5ad.stat().st_size},
                       vcc={"path":str(packaged),"bytes":packaged.stat().st_size,"sha256":lt.sha256(packaged)})
        lt.write_json(receipt_path,receipt)
        print(json.dumps({"status":"complete","vcc":str(packaged),"receipt":str(receipt_path),
                          "submission_performed":False,"submission_allowance_consumed":False},indent=2),flush=True)
        return 0
    except Exception as exc:
        receipt.update(status="failed",finished_utc=utc(),error=f"{type(exc).__name__}: {exc}")
        lt.write_json(receipt_path,receipt)
        print(f"Local build failed: {exc}\nReceipt: {receipt_path}\nNo submission was made.",flush=True)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
