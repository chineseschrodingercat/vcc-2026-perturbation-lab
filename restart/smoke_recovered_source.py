"""Verify restored V28 inference on real external data; this is not validation."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from legacy_v28_adapter import LegacyV28Adapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto')
    p.add_argument('--max-targets',type=int,default=128)
    p.add_argument('--report',default='reports/recovered_v28_gpu_smoke.json')
    a=p.parse_args()
    if a.max_targets<1:
        p.error('--max-targets must be positive')
    root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(12)
    modeldir=root/'archive/legacy_20261003/models/residual_v7/final_candidate/seed_20260928'
    predictor=LegacyV28Adapter(
        source_npz=root/'data/processed/restart/k562_gwps.npz',
        checkpoint=modeldir/'tiny_transformer_600.pt',features=modeldir/'features.npz',
        string_embeddings=root/'data/raw/external/slim_embeddings/gene_string_embeddings.v0.3.h5',
        gene_axis=root/'snapshots/archive/legacy_20261003/runs/context_v4/gene_axis.txt',
        normalization_core=root/'snapshots/archive/legacy_20261003/runs/context_v4/normalization_core.txt',
        device=a.device)
    gpu=str(predictor.device).startswith('cuda')
    if gpu:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start=time.perf_counter()
    with np.load(root/'data/processed/restart/rpe1_bulk.npz',allow_pickle=False) as z:
        targets=z['target_names'][:a.max_targets]
        result=predictor.predict(targets,z['gene_names'],z['control_mean_counts'])
    if gpu:
        torch.cuda.synchronize()
    report={**result.metadata,'shape':list(result.effect_log2fc.shape),
        'all_finite':bool(np.isfinite(result.effect_log2fc).all()),
        'elapsed_inference_seconds':time.perf_counter()-start,
        'torch_version':torch.__version__,'device':predictor.device,
        'gpu_name':torch.cuda.get_device_name() if gpu else None,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if gpu else 0,
        'scope':'Restored source + real external control profile inference only. RPE1 was in historical V28 training, so this is not held-out validation or a training-speed benchmark.'}
    if not report['all_finite']:
        raise ValueError('Restored inference produced non-finite effects')
    output=(root/a.report).resolve()
    if not output.is_relative_to(root):
        raise ValueError('Report must stay in the project')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
