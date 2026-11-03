#!/usr/bin/env python3
"""Read-only audit of optional VCC response summaries; never train or transform effects.

Requires numpy, pandas and h5py. Writes one small JSON report. Gzip inputs are
streamed; CD4T matrices/layers are inspected through HDF5 metadata only.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SOURCES = [
    ("feng2026_ipsc_genomewide_lfc", "feng", "data/raw/external/feng_ipsc/GenomeWideScreen_LFC_byGene.tsv.gz"),
    ("feng2026_ipsc_targeted_lfc", "feng", "data/raw/external/feng_ipsc/TargetedScreen_LFC_byGene.tsv.gz"),
    ("zhu_dann2026_primary_cd4t_de", "cd4t", "data/raw/external/primary_cd4t/GWCD4i.DE_stats.h5ad"),
]

QUALITY_ROLES = {
    "ontarget_significant": "Evidence for intended target knockdown; candidate inclusion criterion.",
    "ontarget_effect_size": "Knockdown magnitude; verify direction and scale before thresholding.",
    "low_target_gex": "Low baseline target expression; on-target estimate may be unreliable.",
    "neighboring_gene_KD": "Possible cis co-target/off-target effect; flag for sensitivity analysis.",
    "distal_offtarget_flag": "Possible distal off-target effect; candidate exclusion/sensitivity flag.",
    "single_guide_estimate": "Only one guide supports the estimate; weaker replication evidence.",
    "n_guides": "Number of guides supporting this perturbation-condition estimate.",
    "guide_n_signif_ontarget": "Number of guides with significant intended-target knockdown.",
    "guide_correlation_signif": "Agreement between guide profiles on significant genes.",
    "guide_correlation_all": "Agreement between guide profiles on all genes.",
    "donor_correlation_all_mean": "Cross-donor reproducibility diagnostic; not new independent profiles.",
    "donor_correlation_all_min": "Worst cross-donor agreement diagnostic.",
    "donor_correlation_hits_mean": "Cross-donor agreement on hit genes.",
    "keep_for_DE": "Pseudobulk eligibility flag if present; not assumed to exist in the DE summary.",
    "keep_effective_guides": "Pseudobulk guide-efficacy flag if present.",
    "keep_min_cells": "Pseudobulk cell-count eligibility flag if present.",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def load_axis(path):
    if path is None:
        return None
    path = Path(path)
    if path.suffix.lower() == ".json":
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(values, dict):
            choices = ["gene_names", "genes", "gene_order", "var_names", "targets", "perturbations", "perts"]
            key = next((k for k in choices if isinstance(values.get(k), list)), None)
            if key is None:
                raise ValueError(f"Cannot identify a list axis in {path}")
            values = values[key]
    elif path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        if values.ndim != 1:
            raise ValueError(f"Expected one-dimensional string axis: {path}")
        values = values.tolist()
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            delimiter = "\t" if "\t" in sample.splitlines()[0] else "," if "," in sample.splitlines()[0] else None
            rows = list(csv.reader(handle, delimiter=delimiter)) if delimiter else [[line.strip()] for line in handle]
        rows = [r for r in rows if r and any(s.strip() for s in r)]
        headers = {"gene", "genes", "gene_name", "gene_names", "gene_symbol", "symbol", "target", "targets", "perturbation", "perturbations", "pert", "pert_name"}
        first = [s.strip().lower() for s in rows[0]] if rows else []
        col = next((i for i, h in enumerate(first) if h in headers), 0)
        if any(h in headers for h in first):
            rows = rows[1:]
        values = [r[col] for r in rows if len(r) > col]
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list axis: {path}")
    return {str(v).strip() for v in values if v is not None and str(v).strip()}


def coverage(seen, requested):
    result = {"unique_observed": len(seen)}
    if requested is not None:
        missing = sorted(requested - seen)
        result.update(requested=len(requested), overlap=len(requested & seen),
                      missing_count=len(missing), missing_preview=missing[:100],
                      missing_preview_truncated=len(missing) > 100)
    return result


class NumericSummary:
    """Exact moments/range; quantiles from a deterministic uniform bottom-k sample."""
    def __init__(self, capacity=100000, seed=20261103):
        self.capacity, self.rng = capacity, np.random.default_rng(seed)
        self.n = self.nonfinite = self.negative = self.zero = self.positive = 0
        self.total = self.total_sq = 0.0
        self.low = self.high = None
        self.values, self.keys = np.empty(0), np.empty(0)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values)
        self.nonfinite += int((~valid).sum())
        values = values[valid]
        if not len(values):
            return
        self.n += len(values)
        self.negative += int((values < 0).sum())
        self.zero += int((values == 0).sum())
        self.positive += int((values > 0).sum())
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())
        self.low = float(values.min()) if self.low is None else min(self.low, float(values.min()))
        self.high = float(values.max()) if self.high is None else max(self.high, float(values.max()))
        keys = self.rng.random(len(values))
        if len(self.keys) == self.capacity:
            keep = keys < self.keys.max()
            keys, values = keys[keep], values[keep]
        keys, values = np.concatenate([self.keys, keys]), np.concatenate([self.values, values])
        if len(keys) > self.capacity:
            keep = np.argpartition(keys, self.capacity - 1)[:self.capacity]
            keys, values = keys[keep], values[keep]
        self.keys, self.values = keys, values

    def report(self):
        probs = [0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1]
        quantiles = {str(q): float(v) for q, v in zip(probs, np.quantile(self.values, probs))} if len(self.values) else {}
        return {"finite_count": self.n, "nonfinite_or_nonnumeric_count": self.nonfinite,
                "negative": self.negative, "zero": self.zero, "positive": self.positive,
                "min_exact": self.low, "max_exact": self.high,
                "mean": self.total / self.n if self.n else None,
                "std_population": float(np.sqrt(max(0.0, self.total_sq / self.n - (self.total / self.n) ** 2))) if self.n else None,
                "quantiles": quantiles, "quantile_sample_size": len(self.values),
                "quantile_method": "exact when finite_count <= sample capacity, otherwise uniform bottom-k sample; endpoints above are exact"}


def audit_feng(path, targets, genes, chunksize, sample_size):
    columns = ["Target", "Expressed_Gene_Symbol", "Expressed_Gene_Ens_ID", "lfc", "pval_adj"]
    rows = selected = missing_keys = duplicates = invalid_p = 0
    all_targets, all_genes, selected_targets, selected_genes = set(), set(), set(), set()
    gene_ids, seen_bits = {}, {}
    effects, on_target = NumericSummary(sample_size), NumericSummary(min(sample_size, 10000))
    # pandas' gzip reader supports concatenated gzip members in the author files.
    for chunk in pd.read_csv(path, sep="\t", compression="gzip", usecols=columns,
                             dtype=str, chunksize=chunksize, keep_default_na=False):
        rows += len(chunk)
        t = chunk["Target"].str.strip()
        g = chunk["Expressed_Gene_Symbol"].str.strip()
        all_targets.update(t[t.ne("")].unique())
        all_genes.update(g[g.ne("")].unique())
        chosen = pd.Series(True, index=chunk.index)
        if targets is not None:
            chosen &= t.isin(targets)
        if genes is not None:
            chosen &= g.isin(genes)
        frame = chunk.loc[chosen].copy()
        frame["Target"], frame["Expressed_Gene_Symbol"] = t[chosen], g[chosen]
        selected += len(frame)
        numeric = pd.to_numeric(frame["lfc"], errors="coerce").to_numpy(dtype=float)
        effects.update(numeric)
        on_target.update(numeric[frame["Target"].eq(frame["Expressed_Gene_Symbol"]).to_numpy()])
        pvalues = pd.to_numeric(frame["pval_adj"], errors="coerce").to_numpy(dtype=float)
        invalid_p += int((~np.isfinite(pvalues) | (pvalues < 0) | (pvalues > 1)).sum())
        key_valid = frame["Target"].ne("") & frame["Expressed_Gene_Symbol"].ne("")
        missing_keys += int((~key_valid).sum())
        frame = frame.loc[key_valid]
        selected_targets.update(frame["Target"].unique())
        selected_genes.update(frame["Expressed_Gene_Symbol"].unique())
        for gene in frame["Expressed_Gene_Symbol"].unique():
            if gene not in gene_ids:
                gene_ids[gene] = len(gene_ids)
        # Exact duplicate count without retaining millions of Python tuple keys.
        # One packed bitset per target is only ~6 MB at 6673 x 6471 dimensions.
        for target, group in frame.groupby("Target", sort=False):
            ids = group["Expressed_Gene_Symbol"].map(gene_ids).to_numpy(dtype=np.int64)
            unique = np.unique(ids)
            duplicates += len(ids) - len(unique)
            packed = np.zeros((len(gene_ids) + 7) // 8, dtype=np.uint8)
            np.bitwise_or.at(packed, unique // 8, (1 << (unique % 8)).astype(np.uint8))
            bits = int.from_bytes(packed.tobytes(), "little")
            previous = seen_bits.get(target, 0)
            duplicates += (bits & previous).bit_count()
            seen_bits[target] = bits | previous
    return {"audit": "feng_streamed_complete", "rows_scanned": rows, "rows_selected": selected,
            "selection": "all rows" if targets is None and genes is None else "optional requested target/gene axes",
            "all_target_coverage": coverage(all_targets, targets), "all_gene_coverage": coverage(all_genes, genes),
            "selected_unique_targets": len(selected_targets), "selected_unique_genes": len(selected_genes),
            "selected_rows_with_missing_keys": missing_keys,
            "selected_duplicate_target_gene_rows": duplicates,
            "duplicate_definition": "Number of selected valid-key rows beyond the first occurrence of the same Target + Expressed_Gene_Symbol, including chunk boundaries.",
            "selected_unique_target_gene_pairs": sum(x.bit_count() for x in seen_bits.values()),
            "effect_statistics_selected": effects.report(), "on_target_effect_statistics_selected": on_target.report(),
            "invalid_adjusted_pvalue_rows_selected": invalid_p,
            "effect_semantics": "Covariate-adjusted mean log-expression residuals; not ordinary raw-pseudobulk log2FC. No exp/exp2, scale conversion, effect filtering or model integration was performed.",
            "usable_for_training": False, "decision": "Candidate only: validate sign, scale and whole-context holdout utility before model integration."}


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_column(group, name):
    """Read a small AnnData metadata column, not an expression matrix."""
    import h5py
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        if h5py.check_string_dtype(obj.dtype) is not None:
            return np.asarray(obj.asstr()[()])
        return np.asarray(obj[()])
    encoding = decode(obj.attrs.get("encoding-type", ""))
    if encoding == "categorical" or {"categories", "codes"}.issubset(obj.keys()):
        cats, codes = read_column(obj, "categories"), np.asarray(obj["codes"][()], dtype=np.int64)
        out = np.empty(len(codes), dtype=object)
        out[:] = None
        valid = codes >= 0
        out[valid] = cats[codes[valid]]
        return out
    if "values" in obj and "mask" in obj:
        out = read_column(obj, "values").astype(object)
        out[np.asarray(obj["mask"][()], dtype=bool)] = None
        return out
    raise ValueError(f"Unsupported metadata encoding for {obj.name}: {encoding}")


def schema(obj):
    import h5py
    if isinstance(obj, h5py.Dataset):
        # AnnData permits X=None encoded as an HDF5 null dataspace while
        # retaining actual response matrices in layers.
        return {"encoding": decode(obj.attrs.get("encoding-type", "array")),
                "shape": list(obj.shape) if obj.shape is not None else None,
                "dtype": str(obj.dtype)}
    return {"encoding": decode(obj.attrs.get("encoding-type", "group")),
            "shape": [int(x) for x in obj.attrs.get("shape", [])],
            "dtype": str(obj["data"].dtype) if "data" in obj else None,
            "stored_entries": int(obj["data"].size) if "data" in obj else None}


def metadata_summary(values):
    s = pd.Series(values)
    result = {"rows": len(s), "missing": int(s.isna().sum())}
    if s.dropna().nunique() <= 12:
        result["value_counts"] = {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}
    numeric = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if len(finite):
        result["numeric"] = {"finite": len(finite), "min": float(finite.min()),
                             "median": float(np.median(finite)), "max": float(finite.max())}
    return result


def audit_cd4t(path, targets, genes):
    import h5py
    with h5py.File(path, "r") as handle:
        if "obs" not in handle or "var" not in handle:
            raise ValueError("Expected AnnData obs and var groups")
        obs, var = handle["obs"], handle["var"]
        obs_index = decode(obs.attrs.get("_index", "_index"))
        var_index = decode(var.attrs.get("_index", "_index"))
        n_obs, n_vars = len(obs[obs_index]), len(var[var_index])
        target_col = next((s for s in ["target_contrast_gene_name", "perturbed_gene_name", "perturbation"] if s in obs), None)
        context_col = next((s for s in ["culture_condition", "context"] if s in obs), None)
        gene_col = next((s for s in ["gene_name", "gene_symbol"] if s in var), var_index)
        target_values = pd.Series(read_column(obs, target_col), dtype="string") if target_col else None
        contexts = pd.Series(read_column(obs, context_col), dtype="string") if context_col else None
        gene_values = pd.Series(read_column(var, gene_col), dtype="string")
        observed_genes = set(gene_values.dropna().astype(str)) - {""}
        observed_targets = set(target_values.dropna().astype(str)) - {""} if target_values is not None else set()
        quality = {field: {"present": field in obs, "role": role,
                          **({"observed": metadata_summary(read_column(obs, field))} if field in obs else {})}
                   for field, role in QUALITY_ROLES.items()}
        count_fields = {s: metadata_summary(read_column(obs, s)) for s in ["n_cells_target", "n_cells", "n_up_genes", "n_down_genes", "n_total_de_genes"] if s in obs}
        by_context, duplicate_keys = {}, None
        if contexts is not None and target_values is not None:
            frame = pd.DataFrame({"context": contexts, "target": target_values})
            duplicate_keys = int(frame.dropna().duplicated(["context", "target"]).sum())
            for context, part in frame.groupby("context", dropna=False):
                by_context[str(context)] = {"rows": len(part), "unique_targets": int(part["target"].nunique()),
                                           "requested_target_overlap": len(set(part["target"].dropna().astype(str)) & targets) if targets is not None else None}
        layers = {key: schema(value) for key, value in handle.get("layers", {}).items()}
        return {"audit": "cd4t_metadata_only", "n_observations": n_obs, "n_genes": n_vars,
                "obs_columns": sorted(k for k in obs if k != obs_index),
                "var_columns": sorted(k for k in var if k != var_index),
                "target_column": target_col, "context_column": context_col, "gene_column": gene_col,
                "target_coverage": coverage(observed_targets, targets), "gene_coverage": coverage(observed_genes, genes),
                "duplicated_gene_symbol_rows": int(gene_values.dropna().duplicated().sum()),
                "duplicate_context_target_rows": duplicate_keys, "contexts": by_context,
                "X_schema": schema(handle["X"]) if "X" in handle else None, "layers": layers,
                "required_layer_log_fc_present": "log_fc" in layers,
                "quality_fields": quality, "count_fields": count_fields,
                "matrix_values_read": False, "effect_finiteness_validated": False,
                "effect_semantics": "Published layers['log_fc'] is DESeq2 log2FC. File contains differential-expression summaries, not raw cells; it cannot directly supply a cell-level count distribution.",
                "usable_for_training": False,
                "decision": "Metadata audit only. Before inclusion, audit selected effect values, apply documented quality decisions, harmonize genes and assess whole-context holdout utility."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", "--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("download_manifest.json"))
    parser.add_argument("--targets", type=Path, help="Optional one-column target axis: CSV/TSV/TXT/JSON/NPY")
    parser.add_argument("--genes", type=Path, help="Optional expressed-gene axis: CSV/TSV/TXT/JSON/NPY")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--onlycompleted", "--only-completed", action="store_true", help="Skip unfinished files, retaining explicit pending entries in the report")
    parser.add_argument("--source", choices=["all", "feng", "cd4t"], default="all")
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--sample-size", type=int, default=100000)
    args = parser.parse_args()
    if args.chunksize <= 0 or args.sample_size <= 0:
        parser.error("chunksize and sample-size must be positive")
    root = args.project_root.resolve()
    output = args.output or root / "reports" / "extra_data_audit.json"
    targets, genes = load_axis(args.targets), load_axis(args.genes)
    entries = json.loads(args.manifest.read_text(encoding="utf-8-sig")) if args.manifest.exists() else []
    entries = entries.get("datasets", []) if isinstance(entries, dict) else entries
    manifest = {x["id"]: x for x in entries}
    report = {"started_utc": now(), "project_root": str(root), "manifest": str(args.manifest),
              "scope": "Read-only source audit; no training, predictions, effect exponentiation or downloads.",
              "filters": {"targets_path": str(args.targets) if args.targets else None, "target_count": len(targets) if targets is not None else None,
                          "genes_path": str(args.genes) if args.genes else None, "gene_count": len(genes) if genes is not None else None},
              "datasets": []}
    failed = False
    for identifier, kind, default_path in SOURCES:
        if args.source != "all" and kind != args.source:
            continue
        entry = manifest.get(identifier, {})
        path = root / entry.get("path", default_path)
        partial = path.with_suffix(path.suffix + ".part")
        expected = entry.get("bytes")
        result = {"id": identifier, "path": str(path), "expected_bytes": expected}
        actual = path.stat().st_size if path.is_file() else None
        result["actual_bytes"] = actual
        pending = not path.is_file() or partial.exists() or (expected is not None and actual != expected)
        if pending:
            result.update(status="pending", skipped=True,
                          reason="File absent, .part present, or byte count differs from the download manifest; no contents read.",
                          partial_bytes=partial.stat().st_size if partial.exists() else None)
            report["datasets"].append(result)
            print(json.dumps({"id": identifier, "status": "pending"}), flush=True)
            continue
        receipt = path.with_suffix(path.suffix + ".receipt.json")
        if receipt.exists():
            try:
                saved = json.loads(receipt.read_text(encoding="utf-8-sig"))
                result["download_receipt_matches_file"] = bool(saved.get("verified") and saved.get("mtime_ns") == path.stat().st_mtime_ns)
                result["publisher_checksum_checked_by_downloader"] = saved.get("publisher_checksum_checked")
            except (OSError, ValueError) as exc:
                result["receipt_read_error"] = str(exc)
        else:
            result["download_receipt_matches_file"] = False
        start = time.monotonic()
        print(json.dumps({"id": identifier, "status": "auditing"}), flush=True)
        try:
            result.update(audit_feng(path, targets, genes, args.chunksize, args.sample_size) if kind == "feng" else audit_cd4t(path, targets, genes))
            result["status"] = "audited"
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            failed = True
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        report["datasets"].append(result)
        print(json.dumps({"id": identifier, "status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    report["completed_utc"] = now()
    report["status"] = "error" if failed else "pending_inputs" if any(x["status"] == "pending" for x in report["datasets"]) else "audit_complete"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output)
    print(json.dumps({"report": str(output), "status": report["status"]}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
