"""Generate VCC 2026 raw-count predictions from fitted log2 fold changes.

One target/context is transformed at a time. The output is a standard sparse
AnnData H5AD, written incrementally so the full 360,000-cell panel does not need
to fit in RAM while it is being generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


CONTEXTS = ("A", "B", "C")
DEFAULT_MODEL_ID = "fixed_half_blend"
PSEUDOCOUNT = 0.5
CELLS_PER_TARGET = 400
MAX_MULTIPLIER = 16.0
HDF5_CHUNK_ENTRIES = 1 << 20


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def q(array: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(array)),
        "p01": float(np.percentile(array, 1)),
        "median": float(np.median(array)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def load_effects(path: Path, official_targets: list[str], official_genes: list[str]):
    with np.load(path, allow_pickle=False) as data:
        targets = data["target_symbols"].astype(str).tolist()
        genes = data["gene_symbols"].astype(str).tolist()
        effects = data["effect_log2fc"].copy()
    if targets != official_targets:
        raise ValueError(f"{path}: target order differs from official pert_counts.csv")
    if len(genes) != len(set(genes)) or not set(genes).issubset(official_genes):
        raise ValueError(f"{path}: duplicate or nonofficial effect gene")
    if effects.shape != (len(targets), len(genes)) or not np.isfinite(effects).all():
        raise ValueError(f"{path}: invalid effect matrix")
    gene_to_position = {g: i for i, g in enumerate(official_genes)}
    positions = np.array([gene_to_position[g] for g in genes], dtype=np.int32)
    return positions, effects, genes


def make_obs(contexts: list[str], targets: list[str]) -> pd.DataFrame:
    labels: list[str] = []
    context_labels: list[str] = []
    names: list[str] = []
    for context in contexts:
        for target in targets:
            labels.extend([target] * CELLS_PER_TARGET)
            context_labels.extend([context] * CELLS_PER_TARGET)
            names.extend(f"{context}_{target}_{j:03d}" for j in range(CELLS_PER_TARGET))
    obs = pd.DataFrame(index=pd.Index(names, name=None))
    obs["target_gene"] = pd.Categorical(labels, categories=targets)
    obs["context"] = pd.Categorical(context_labels, categories=CONTEXTS)
    return obs


def init_h5ad(path: Path, obs: pd.DataFrame, genes: list[str]) -> None:
    # Write all lightweight AnnData metadata with the library. Replace the empty
    # CSR group with appendable datasets, retaining the documented H5AD schema.
    frame = ad.AnnData(
        X=sparse.csr_matrix((len(obs), len(genes)), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes)),
    )
    frame.write_h5ad(path)
    del frame
    with h5py.File(path, "r+") as h5:
        del h5["X"]
        group = h5.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.array([len(obs), len(genes)], dtype=np.int64)
        for name, dtype in (("data", np.float32), ("indices", np.int32)):
            group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(HDF5_CHUNK_ENTRIES,),
                dtype=dtype,
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
        group.create_dataset("indptr", shape=(len(obs) + 1,), dtype=np.int64)
        group["indptr"][0] = 0


def append_csr(group: h5py.Group, block: sparse.csr_matrix, row_offset: int, nnz: int) -> int:
    end = nnz + block.nnz
    for name, values in (("data", block.data), ("indices", block.indices)):
        dataset = group[name]
        dataset.resize((end,))
        dataset[nnz:end] = values
    group["indptr"][row_offset + 1 : row_offset + block.shape[0] + 1] = (
        block.indptr[1:].astype(np.int64, copy=False) + nnz
    )
    return end


def transform_counts(
    block: sparse.csr_matrix, multiplier: np.ndarray, rng: np.random.Generator
) -> sparse.csr_matrix:
    if block.dtype != np.float32:
        block = block.astype(np.float32)
    values = block.data.astype(np.float64) * multiplier[block.indices]
    floor = np.floor(values)
    rounded = floor + (rng.random(len(values)) < (values - floor))
    if not np.isfinite(rounded).all() or (rounded < 0).any() or (rounded > 2**24).any():
        raise ValueError("Transformed counts are outside exact float32 integer range")
    block.data = rounded.astype(np.float32)
    block.eliminate_zeros()
    block.sort_indices()
    return block


def multiplier_from_effects(
    control_mean: np.ndarray, modeled_positions: np.ndarray, effect: np.ndarray
) -> tuple[np.ndarray, dict]:
    mu = control_mean[modeled_positions]
    wanted = np.maximum((mu + PSEUDOCOUNT) * np.exp2(effect) - PSEUDOCOUNT, 0.0)
    ratio = np.ones(len(mu), dtype=np.float64)
    positive = mu > 0
    ratio[positive] = wanted[positive] / mu[positive]
    nonrecoverable = (~positive) & (wanted > 0)
    too_large = ratio > MAX_MULTIPLIER
    ratio = np.clip(ratio, 0.0, MAX_MULTIPLIER)
    full = np.ones(len(control_mean), dtype=np.float64)
    full[modeled_positions] = ratio
    diagnostics = {
        "modeled_genes": int(len(mu)),
        "positive_control_mean": int(positive.sum()),
        "nonrecoverable_zero_control_mean": int(nonrecoverable.sum()),
        "multiplier_capped": int(too_large.sum()),
        "multiplier_quantiles": q(ratio),
    }
    return full, diagnostics


def verify_h5ad(path: Path, expected_shape: tuple[int, int], genes: list[str]) -> dict:
    with h5py.File(path, "r") as h5:
        data = h5["X/data"]
        indices = h5["X/indices"]
        ptr = h5["X/indptr"]
        if len(data) != len(indices) or int(ptr[-1]) != len(data):
            raise ValueError("Final CSR pointers and values differ")
        if tuple(h5["X"].attrs["shape"]) != expected_shape:
            raise ValueError("Final CSR shape differs")
        nnz = len(data)
    backed = ad.read_h5ad(path, backed="r")
    try:
        if backed.shape != expected_shape or backed.var_names.astype(str).tolist() != genes:
            raise ValueError("AnnData shape or gene axis differs")
        if backed.obs["target_gene"].isna().any() or backed.obs["context"].isna().any():
            raise ValueError("Missing labels in obs")
        first = backed.X[: min(CELLS_PER_TARGET, expected_shape[0])]
        last_start = expected_shape[0] - min(CELLS_PER_TARGET, expected_shape[0])
        last = backed.X[last_start : expected_shape[0]]
        for matrix in (first, last):
            if not sparse.isspmatrix_csr(matrix):
                raise ValueError("Expected CSR from backed AnnData")
            if not np.isfinite(matrix.data).all() or (matrix.data <= 0).any():
                raise ValueError("Invalid stored count")
            if not np.equal(matrix.data, np.rint(matrix.data)).all():
                raise ValueError("Fractional stored count")
        return {"shape": list(backed.shape), "nnz": nnz, "bytes": path.stat().st_size}
    finally:
        backed.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--smoke-context", choices=CONTEXTS, default="A")
    parser.add_argument("--smoke-target", default="ABCD1")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--effect-dir", type=Path, default=Path("runs/baseline_v1/challenge_effects"))
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("Output must be inside the competition root")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    manifest = output.with_name(output.stem + ".manifest.json")
    if partial.exists() or manifest.exists():
        raise FileExistsError("Partial output or manifest exists; inspect it before retrying")
    output.parent.mkdir(parents=True, exist_ok=True)

    validation = root / "data/raw/vcc2026/validation/files"
    official_genes = pd.read_csv(validation / "gene_names.csv")["gene_name"].astype(str).tolist()
    official_targets = pd.read_csv(validation / "pert_counts.csv")["target_gene"].astype(str).tolist()
    if len(official_genes) != 18533 or len(set(official_genes)) != 18533:
        raise ValueError("Unexpected official gene list")
    if len(official_targets) != 300 or len(set(official_targets)) != 300:
        raise ValueError("Unexpected official target list")
    selected_contexts = list(CONTEXTS) if args.mode == "full" else [args.smoke_context]
    if args.mode == "smoke":
        if args.smoke_target not in official_targets:
            raise ValueError("Smoke target is absent from official panel")
        selected_targets = [args.smoke_target]
    else:
        selected_targets = official_targets
    total_cells = len(selected_contexts) * len(selected_targets) * CELLS_PER_TARGET
    # A deliberately conservative space guard; precise predicted size is reported
    # by the smoke run. A full H5AD and a later .vcc may coexist under the root.
    required_free_gib = 75 if args.mode == "full" else 51
    free_gib = shutil.disk_usage(output.parent).free / 2**30
    if free_gib < required_free_gib:
        raise OSError(f"Need at least {required_free_gib} GiB free; found {free_gib:.1f}")

    obs = make_obs(selected_contexts, selected_targets)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "seed": args.seed,
        "model_id": args.model_id,
        "model_effect_definition": "log2((target_mean+0.5)/(control_mean+0.5))",
        "generation_method": "For each target, sample 400 distinct real NTC cells without replacement from its context; invert the model's pseudocount log2FC against that context's NTC mean to obtain per-gene count multipliers, cap at 16, multiply existing nonzero raw counts, and stochastically round. Implicit zeros remain zero.",
        "limitation": "Existing zero entries are not activated; very rare induced genes can have distorted detection frequency. Unmodeled official genes retain control-like counts (effect=0 fallback).",
        "cells_per_target": CELLS_PER_TARGET,
        "pseudocount": PSEUDOCOUNT,
        "max_multiplier": MAX_MULTIPLIER,
        "n_genes": len(official_genes),
        "n_cells": total_cells,
        "software_versions": {
            "anndata": version("anndata"),
            "h5py": version("h5py"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "context_order": selected_contexts,
        "target_order": selected_targets,
        "input_sha256": {
            "gene_names.csv": file_sha256(validation / "gene_names.csv"),
            "pert_counts.csv": file_sha256(validation / "pert_counts.csv"),
        },
        "contexts": {},
    }
    init_h5ad(partial, obs, official_genes)
    cumulative_nnz = 0
    row_offset = 0
    try:
        with h5py.File(partial, "r+") as h5:
            group = h5["X"]
            for context in selected_contexts:
                input_path = validation / f"context_{context}.h5ad"
                effect_path = root / args.effect_dir / f"context_{context}__{args.model_id}.npz"
                effect_sidecar = effect_path.with_suffix(".json")
                effect_meta = json.loads(effect_sidecar.read_text(encoding="utf-8"))
                if effect_meta.get("model_id") != args.model_id or effect_meta.get("prediction_context") != context:
                    raise ValueError(f"{effect_sidecar}: model/context metadata mismatch")
                if "0.5" not in effect_meta.get("effect_definition", ""):
                    raise ValueError(f"{effect_sidecar}: expected pseudocount 0.5 in effect definition")
                modeled_positions, effects, modeled_genes = load_effects(
                    effect_path, official_targets, official_genes
                )
                report["input_sha256"][str(input_path.relative_to(root))] = file_sha256(input_path)
                report["input_sha256"][str(effect_path.relative_to(root))] = file_sha256(effect_path)
                report["input_sha256"][str(effect_sidecar.relative_to(root))] = file_sha256(effect_sidecar)
                controls = ad.read_h5ad(input_path)
                if list(controls.var_names) != official_genes or controls.n_obs != 18400:
                    raise ValueError(f"{context}: control axis or cell count differs")
                if not sparse.isspmatrix_csr(controls.X):
                    raise ValueError(f"{context}: control matrix is not CSR")
                if set(controls.obs["target_gene"].astype(str)) != {"non-targeting"}:
                    raise ValueError(f"{context}: unexpected control target labels")
                control_x = controls.X
                control_mean = np.asarray(control_x.mean(axis=0), dtype=np.float64).ravel()
                context_stats = {
                    "control_cells": controls.n_obs,
                    "control_nnz": int(control_x.nnz),
                    "modeled_genes": len(modeled_genes),
                    "target_count": len(selected_targets),
                    "targets": {},
                }
                for target in selected_targets:
                    target_index = official_targets.index(target)
                    rng = np.random.default_rng(
                        np.random.SeedSequence(args.seed, spawn_key=(CONTEXTS.index(context), target_index))
                    )
                    chosen = rng.choice(controls.n_obs, size=CELLS_PER_TARGET, replace=False)
                    batch = control_x[chosen].tocsr(copy=True)
                    multiplier, mdiag = multiplier_from_effects(
                        control_mean, modeled_positions, effects[target_index]
                    )
                    batch = transform_counts(batch, multiplier, rng)
                    library_sizes = np.asarray(batch.sum(axis=1), dtype=np.float64).ravel()
                    if np.any(library_sizes <= 0) or np.any(library_sizes > 1_000_000):
                        raise ValueError(f"{context}/{target}: invalid library size")
                    cumulative_nnz = append_csr(group, batch, row_offset, cumulative_nnz)
                    row_offset += CELLS_PER_TARGET
                    realized_mean = np.asarray(batch.mean(axis=0), dtype=np.float64).ravel()
                    realized_log2fc = np.log2(
                        (realized_mean[modeled_positions] + PSEUDOCOUNT)
                        / (control_mean[modeled_positions] + PSEUDOCOUNT)
                    )
                    mdiag.update(
                        {
                            "nnz": int(batch.nnz),
                            "library_size": q(library_sizes),
                            "effect_log2fc_mae": float(np.mean(np.abs(realized_log2fc - effects[target_index]))),
                            "control_rows_sampled_unique": int(len(set(chosen))),
                        }
                    )
                    context_stats["targets"][target] = mdiag
                report["contexts"][context] = context_stats
                print(f"{context}: {len(selected_targets)} targets, cumulative {cumulative_nnz:,} nonzeros", flush=True)
                del controls, control_x
            h5.flush()
        if row_offset != total_cells:
            raise ValueError("Written row count differs from planned row count")
        checked = verify_h5ad(partial, (total_cells, len(official_genes)), official_genes)
        report["output"] = checked
        report["output"]["relative_path"] = str(output.relative_to(root))
        report["output"]["sha256"] = file_sha256(partial)
        os.replace(partial, output)
        manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"PASS: {checked['shape']}, {checked['nnz']:,} nonzeros, {checked['bytes']:,} bytes")
        print(output)
    except Exception:
        # Preserve the partial file for diagnosis; never silently overwrite it.
        raise


if __name__ == "__main__":
    main()
