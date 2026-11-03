"""Small CPU-only VCC restart baseline; never submits to the challenge.

Commands: ingest raw cells/construct means, compare leave-background-out proxies,
and generate a sparse integer-count H5AD from an official controls bundle.
All modelling happens on explicitly shared, measured genes. Missing genes are
not asserted to be zero. Different files from one background never create
independent validation folds or extra ensemble votes.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

SCALE = 50_000.0
SCHEMA = "vcc_restart_lean_transfer_v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def choose_column(frame, requested, candidates, role):
    if requested != "auto":
        if requested == "_index":
            return np.asarray(frame.index.astype(str)), "_index"
        if requested not in frame:
            raise ValueError(f"{role} column {requested!r} absent; available: {list(frame.columns)}")
        return frame[requested].to_numpy(), requested
    for key in candidates:
        if key in frame:
            return frame[key].to_numpy(), key
    if role == "gene":
        return np.asarray(frame.index.astype(str)), "_index"
    raise ValueError(f"Cannot infer {role}; pass the column explicitly. Columns: {list(frame.columns)}")


def strings(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=str)


def matrix_slice(a, matrix, start, end, columns=None):
    if matrix == "X":
        value = a.X[start:end]
    elif matrix == "raw.X":
        if a.raw is None:
            raise ValueError("raw.X requested but absent")
        value = a.raw.X[start:end]
    else:
        key = matrix.removeprefix("layers/")
        value = a.layers[key][start:end]
    if hasattr(value, "to_memory"):
        value = value.to_memory()
    if columns is not None:
        value = value[:, columns]
    return value


def check_raw(values, require_integer):
    values = values.data if sparse.issparse(values) else np.asarray(values)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts/construct means must be finite and nonnegative")
    # A raw_bulk row is a construct MEAN, so its fractional values are valid.
    if require_integer and not np.equal(values, np.rint(values)).all():
        raise ValueError("Single-cell input is not raw integer counts; choose the counts matrix/layer explicitly")


def ingest(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    input_path = Path(args.input).resolve()
    a = ad.read_h5ad(input_path, backed="r")
    try:
        var = a.raw.var if args.matrix == "raw.X" else a.var
        genes, gene_col = choose_column(var, args.gene_col,
                                      ("gene_name", "gene_symbol", "gene_symbols", "feature_name", "feature_symbol"), "gene")
        genes = np.char.strip(strings(genes))
        unique = ~pd.Index(genes).duplicated(keep=False)
        unique &= ~np.isin(genes, ["", "nan", "None", "<NA>"])
        columns = np.flatnonzero(unique)
        selected_genes = genes[columns]
        if not len(selected_genes):
            raise ValueError("No unambiguous gene symbols")
        source_construct_rows_missing_ensembl=0
        if args.target_col == "construct-index":
            labels=[]
            for name in strings(a.obs_names):
                # The guide label can be P1/P1P2 or one/many ENST transcript IDs.
                # Identity is the explicit gene field between numeric row ID and
                # guide descriptor; validate the terminal ENSG identifier.
                match=re.match(r"^\d+_([^_]+)_.+_(?:ENSG\d+(?:\.\d+)?|nan)$",name)
                if match and name.endswith("_nan"):
                    source_construct_rows_missing_ensembl+=1
                labels.append("non-targeting" if "non-targeting" in name.split("_") else (match.group(1) if match else name))
            labels=np.asarray(labels,dtype=str)
            target_col="construct-index"
        else:
            labels, target_col = choose_column(a.obs, args.target_col,
                                              ("target_gene", "gene", "perturbed_gene_name", "perturbation"), "target")
        labels = np.char.strip(strings(labels))
        if np.isin(labels, ["", "nan", "None", "<NA>"]).any():
            raise ValueError("Missing perturbation labels; filter or repair the source explicitly")
        keep = np.ones(a.n_obs, dtype=bool)
        for item in args.filter:
            col, value = item.split("=", 1)
            if col not in a.obs:
                raise ValueError(f"Filter column {col} absent")
            keep &= strings(a.obs[col].to_numpy()) == value
        controls = np.isin(labels, args.control_label)
        recognized_controls=controls.copy()
        excluded_noncore_controls=0
        if args.control_col:
            if args.control_col not in a.obs:
                raise ValueError(f"Control flag column absent: {args.control_col}")
            raw_flags=a.obs[args.control_col].to_numpy()
            if not np.isin(raw_flags,[True,False,0,1]).all():
                raise ValueError("Control flag must contain true/false or 1/0")
            controls=np.asarray(raw_flags,dtype=bool)
            if args.target_col=="construct-index" and (controls & ~recognized_controls).any():
                raise ValueError("core_control flags a named perturbation; source semantics require inspection")
            # Non-core NTCs must never be recast as biological perturbations.
            excluded_noncore_controls=int((keep & recognized_controls & ~controls).sum())
            keep &= ~recognized_controls | controls
        if args.target_col == "construct-index":
            unmatched=(labels==strings(a.obs_names)) & ~controls & keep
            if unmatched.any():
                raise ValueError(f"Unrecognized perturbation construct names: {strings(a.obs_names)[unmatched][:10]}")
        labels = labels.astype(object)
        labels[controls] = "__CONTROL__"
        labels = strings(labels)
        if not (controls & keep).any():
            raise ValueError(f"No controls matching {args.control_label}; example labels: {np.unique(labels)[:20]}")
        groups = np.unique(labels[keep])
        group_map = {g: i for i, g in enumerate(groups)}
        codes = np.asarray([group_map.get(g, -1) for g in labels], dtype=np.int64)
        if args.kind == "raw-bulk":
            if args.n_col not in a.obs:
                raise ValueError(f"Construct means require obs[{args.n_col!r}] weights")
            weights = np.asarray(a.obs[args.n_col], dtype=np.float64)
            if not np.isfinite(weights[keep]).all() or (weights[keep] <= 0).any():
                raise ValueError("raw_bulk n_cells must be finite and positive")
        else:
            weights = np.ones(a.n_obs, dtype=np.float64)
        # Memory is O(number of perturbations x measured genes), never O(cells x genes).
        sums = np.zeros((len(groups), len(columns)), dtype=np.float64)
        n_cells = np.zeros(len(groups), dtype=np.float64)
        rows_read = 0
        for start in range(0, a.n_obs, args.chunk_rows):
            end = min(start + args.chunk_rows, a.n_obs)
            valid = keep[start:end]
            if not valid.any():
                continue
            x = matrix_slice(a, args.matrix, start, end, columns)[valid]
            check_raw(x, require_integer=args.kind == "singlecell")
            ids, w = codes[start:end][valid], weights[start:end][valid]
            np.add.at(n_cells, ids, w)
            if sparse.issparse(x):
                assignment = sparse.csr_matrix((w, (ids, np.arange(len(ids)))), shape=(len(groups), len(ids)))
                added = (assignment @ x).tocoo()
                np.add.at(sums, (added.row, added.col), added.data)
            else:
                x = np.asarray(x, dtype=np.float64)
                for gid in np.unique(ids):
                    mask = ids == gid
                    sums[gid] += np.sum(x[mask] * w[mask, None], axis=0)
            rows_read += len(ids)
            if start % (20 * args.chunk_rows) == 0:
                print(f"ingest {args.source_id}: {end:,}/{a.n_obs:,} rows", flush=True)
        control_index = group_map["__CONTROL__"]
        take = np.flatnonzero(groups != "__CONTROL__")
        if not len(take):
            raise ValueError("Source has no perturbations after filtering")
        control_mean = sums[control_index] / n_cells[control_index]
        target_mean = sums[take] / n_cells[take, None]
        if control_mean.sum() <= 0 or (target_mean.sum(axis=1) <= 0).any():
            raise ValueError("Empty control or perturbation pseudobulk")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, gene_names=selected_genes, target_names=groups[take],
                            target_mean_counts=target_mean.astype(np.float32),
                            control_mean_counts=control_mean.astype(np.float32),
                            target_n_cells=n_cells[take], control_n_cells=n_cells[control_index],
                            measured_mask=np.ones(len(columns), dtype=bool),
                            source_id=np.asarray(args.source_id), background=np.asarray(args.background),
                            schema=np.asarray(SCHEMA))
        report = dict(schema=SCHEMA, input=str(input_path), source_id=args.source_id,
                      background=args.background, kind=args.kind, matrix=args.matrix,
                      target_col=target_col, gene_col=gene_col, control_labels=args.control_label, control_col=args.control_col,
                      excluded_noncore_control_rows=excluded_noncore_controls,
                      source_construct_rows_missing_ensembl=source_construct_rows_missing_ensembl,
                      filters=args.filter, input_bytes=input_path.stat().st_size,
                      source_rows_used=rows_read, genes=len(columns),
                      ambiguous_or_empty_gene_columns_excluded=int((~unique).sum()),
                      targets=len(take), control_n_cells=float(n_cells[control_index]),
                      min_target_n_cells=float(n_cells[take].min()), max_target_n_cells=float(n_cells[take].max()),
                      aggregation="n_cells-weighted construct means" if args.kind == "raw-bulk" else "sum raw single cells / cell count",
                      output=str(output.resolve()))
        write_json(output.with_suffix(".json"), report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        a.file.close()


@dataclass
class Source:
    source_id: str
    background: str
    genes: np.ndarray
    targets: np.ndarray
    target_mean: np.ndarray
    control_mean: np.ndarray
    n_cells: np.ndarray
    control_n: float
    measured: np.ndarray
    path: str = ""

    def __post_init__(self):
        self.gmap = {str(g): i for i, g in enumerate(self.genes) if self.measured[i]}
        self.tmap = {str(t): i for i, t in enumerate(self.targets)}


def load_source(path):
    with np.load(path, allow_pickle=False) as z:
        if "background" not in z:
            raise ValueError(f"{path}: source must record its background to prevent same-line leakage")
        source = Source(str(z["source_id"].item()), str(z["background"].item()),
                        z["gene_names"].astype(str), z["target_names"].astype(str),
                        z["target_mean_counts"].astype(np.float64), z["control_mean_counts"].astype(np.float64),
                        z["target_n_cells"].astype(np.float64), float(z["control_n_cells"].item()),
                        z["measured_mask"].astype(bool) if "measured_mask" in z else np.ones(len(z["gene_names"]), bool),
                        str(Path(path).resolve()))
    if len(set(source.genes)) != len(source.genes) or len(set(source.targets)) != len(source.targets):
        raise ValueError(f"{path}: duplicate gene/target axis")
    if source.target_mean.shape != (len(source.targets), len(source.genes)) or source.control_mean.shape != (len(source.genes),):
        raise ValueError(f"{path}: shape mismatch")
    if source.n_cells.shape != (len(source.targets),) or source.measured.shape != (len(source.genes),):
        raise ValueError(f"{path}: metadata shape mismatch")
    check_raw(source.control_mean[source.measured], False)
    check_raw(source.target_mean[:, source.measured], False)
    if not np.isfinite(source.n_cells).all() or (source.n_cells <= 0).any() or source.control_n <= 0:
        raise ValueError(f"{path}: invalid support")
    return source


def load_sources(paths):
    sources = [load_source(path) for path in paths]
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source_id values are forbidden; use unique identities for different files")
    return sources


def source_locks(sources):
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source_id values cannot form a calibration lock")
    return {source.source_id: {"id": source.source_id, "background": source.background,
                              "npz_sha256": sha256(source.path)} for source in sources}


def validate_source_locks(sources, model):
    expected = model.get("source_locks")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Model has no source content locks; rerun cv before prediction")
    actual = source_locks(sources)
    if set(actual) != set(expected):
        raise ValueError("Source IDs differ from calibration locks; rerun cv")
    for source_id, lock in actual.items():
        saved = expected[source_id]
        if not isinstance(saved, dict) or any(saved.get(key) != lock[key] for key in ("id", "background", "npz_sha256")):
            raise ValueError(f"Source lock mismatch for {source_id}: ID, background or NPZ contents changed; rerun cv")


def common_genes(sources, destination_genes=None, minimum=10):
    if not sources:
        raise ValueError("No source data")
    genes = set(sources[0].gmap)
    for source in sources[1:]:
        genes &= set(source.gmap)
    if destination_genes is not None:
        genes &= set(destination_genes)
        ordered = [g for g in destination_genes if g in genes]
    else:
        ordered = sorted(genes)
    if len(ordered) < minimum:
        raise ValueError(f"Only {len(ordered)} common measured genes; required >= {minimum}")
    return np.asarray(ordered, dtype=str)


def normalize(mean):
    x = np.asarray(mean, dtype=np.float64)
    denominator = x.sum(axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (x < 0).any() or (denominator <= 0).any():
        raise ValueError("Invalid common-gene library")
    return np.log1p(SCALE * x / denominator)


def representative_control(sources, genes):
    source = max(sources, key=lambda s: (s.control_n, s.source_id))
    indices = [source.gmap[g] for g in genes]
    return source.control_mean[indices], source.source_id


def background_effect(sources, genes, targets, min_cells=30):
    """Each target uses its highest-support source within this background.

    This avoids counting overlapping GWPS/essential samples twice. Controls are
    always taken from the same source as that target's response.
    """
    if len({s.background for s in sources}) != 1:
        raise ValueError("background_effect must receive exactly one biological background")
    effect = np.zeros((len(targets), len(genes)), dtype=np.float64)
    support = np.zeros(len(targets), dtype=np.float64)
    used = np.full(len(targets), "", dtype=object)
    for source in sorted(sources, key=lambda s: s.source_id):
        gi = np.asarray([source.gmap[g] for g in genes])
        rows = np.asarray([source.tmap.get(t, -1) for t in targets])
        valid = rows >= 0
        n = np.zeros(len(targets))
        n[valid] = source.n_cells[rows[valid]]
        valid &= (n >= min_cells) & (n > support)
        selected = np.flatnonzero(valid)
        if len(selected):
            effect[selected] = normalize(source.target_mean[rows[selected]][:, gi]) - normalize(source.control_mean[gi])
            support[selected] = n[selected]
            used[selected] = source.source_id
    return effect, support, used.astype(str)


def predict_delta(sources, genes, targets, query_control, model="weighted_shrink", alpha=0.5,
                  temperature=1.0, prior_cells=50.0, min_cells=30):
    if model == "zero":
        return np.zeros((len(targets), len(genes))), {"backgrounds": [], "covered_targets": 0, "query_uses_controls_only": True}
    groups = {}
    for source in sources:
        groups.setdefault(source.background, []).append(source)
    if model == "k562_direct":
        groups = {k: v for k, v in groups.items() if k.upper().replace("-", "") == "K562"}
        if not groups:
            raise ValueError("K562 direct unavailable after whole-background exclusion")
    q = normalize(query_control)
    total = np.zeros((len(targets), len(genes)), dtype=np.float64)
    denom = np.zeros(len(targets), dtype=np.float64)
    audit = []
    for background, members in sorted(groups.items()):
        effect, support, used = background_effect(members, genes, targets, min_cells)
        control, control_source = representative_control(members, genes)
        distance = float(np.mean((normalize(control) - q) ** 2))
        weight = 1.0 if model == "k562_direct" else float(np.exp(-min(temperature * distance, 50)))
        available = support > 0
        reliability = np.ones(len(targets)) if model == "k562_direct" else np.divide(support,support+prior_cells,out=np.zeros_like(support),where=(support+prior_cells)>0)
        total += effect * (weight * reliability * available)[:, None]
        denom += weight * available
        audit.append(dict(background=background, source_ids=[s.source_id for s in members],
                          control_source=control_source, control_distance=distance, weight=weight,
                          covered_targets=int(available.sum()), target_source_ids=used.tolist()))
    np.divide(total, denom[:, None], out=total, where=denom[:, None] > 0)
    if model != "k562_direct":
        total *= alpha
    return total, dict(backgrounds=audit, covered_targets=int((denom > 0).sum()),
                       query_uses_controls_only=True, same_background_rule="highest-support source per target; one ensemble vote")


def project_delta(delta, query_control):
    """Invert log1p-normalized effects and preserve this measured core's library."""
    control = np.asarray(query_control, dtype=np.float64)
    wanted = np.expm1(np.clip(normalize(control)[None, :] + delta, 0, 20))
    mass = wanted.sum(axis=1, keepdims=True)
    np.divide(wanted, mass, out=wanted, where=mass > 0)
    empty = mass[:, 0] <= 0
    wanted[empty] = control / control.sum()
    wanted *= control.sum()
    actual_delta = normalize(wanted) - normalize(control)
    unchanged=np.all(delta==0,axis=1)
    wanted[unchanged]=control
    actual_delta[unchanged]=0
    return actual_delta, wanted


def summarize_errors(pred, truth, targets, genes):
    keep = np.ones_like(truth, dtype=bool)
    gi = {g: i for i, g in enumerate(genes)}
    for i, target in enumerate(targets):
        if target in gi:
            keep[i, gi[target]] = False
    # Own-target removal in the secondary metric prevents a trivially known knockdown
    # from dominating a claim about the broader transcriptional response.
    rows = []
    for i, target in enumerate(targets):
        p, y = pred[i], truth[i]
        norm = float(np.linalg.norm(p) * np.linalg.norm(y))
        cosine = float(np.dot(p, y) / norm) if norm > 0 else None
        k = keep[i]
        rows.append(dict(target=str(target), proxy_delta_mse=float(np.mean((p-y)**2)),
                         proxy_delta_mse_excluding_target=float(np.mean((p[k]-y[k])**2)),
                         proxy_delta_cosine=cosine, truth_effect_norm=float(np.linalg.norm(y))))
    return rows


def cv(args):
    sources = load_sources(args.sources)
    locks = source_locks(sources)
    backgrounds = sorted({s.background for s in sources})
    if len(backgrounds) < 2:
        raise ValueError("At least two distinct backgrounds are needed for LOCO")
    destination_genes=read_axis(args.genes,{"gene_name","gene","gene_symbol","gene_symbols"}) if args.genes else None
    genes = common_genes(sources, destination_genes=destination_genes, minimum=args.min_genes)
    alphas = [float(v) for v in args.alphas.split(",")]
    temperatures = [float(v) for v in args.temperatures.split(",")]
    if any(a < 0 for a in alphas) or any(t < 0 for t in temperatures):
        raise ValueError("Alpha and temperature must be nonnegative")
    configs = [dict(id="zero", model="zero", alpha=0.0, temperature=0.0),
               dict(id="k562_direct", model="k562_direct", alpha=1.0, temperature=0.0)]
    configs += [dict(id=f"weighted_a{alpha:g}_t{temperature:g}", model="weighted_shrink", alpha=alpha, temperature=temperature)
                for alpha in alphas for temperature in temperatures]
    rows, fold_reports = [], []
    for heldout in backgrounds:
        train = [s for s in sources if s.background != heldout]
        test = [s for s in sources if s.background == heldout]
        assert all(s.background != heldout for s in train)
        target_union = sorted({str(t) for s in test for t, n in zip(s.targets, s.n_cells) if n >= args.min_cells})
        # Deterministic metadata-only selection; no response magnitude filtering.
        targets = sorted(target_union, key=lambda t: hashlib.sha256((str(args.seed)+":"+t).encode()).digest())[:args.max_targets]
        targets = np.asarray(targets, dtype=str)
        if not len(targets):
            continue
        query_control, control_source = representative_control(test, genes)
        predictions, audits = {}, {}
        # Held-out responses are not passed to any predictor/calibration routine.
        for config in configs:
            if config["model"] == "k562_direct" and not any(s.background.upper().replace("-", "") == "K562" for s in train):
                continue
            delta, audit = predict_delta(train, genes, targets, query_control,
                                         model=config["model"], alpha=config["alpha"], temperature=config["temperature"],
                                         prior_cells=args.prior_cells, min_cells=args.min_cells)
            predictions[config["id"]] = project_delta(delta, query_control)[0]
            audits[config["id"]] = audit
        # Only now access held-out outcomes to construct the evaluation labels.
        truth, n_truth, truth_sources = background_effect(test, genes, targets, args.min_cells)
        for config_id, prediction in predictions.items():
            for row in summarize_errors(prediction, truth, targets, genes):
                rows.append(dict(heldout_background=heldout, candidate=config_id, **row))
        fold_reports.append(dict(heldout_background=heldout, excluded_source_ids=[s.source_id for s in test],
                                 training_source_ids=[s.source_id for s in train], query_control_source=control_source,
                                 targets_evaluated=len(targets), truth_source_ids=truth_sources.tolist(),
                                 candidates=audits))
        print(f"LOCO {heldout}: {len(train)} source files, {len(targets)} targets, {len(genes)} genes", flush=True)
    if not rows:
        raise ValueError("No eligible validation targets")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "proxy_per_target.csv", index=False)
    summaries = []
    for config in configs:
        selected = [r for r in rows if r["candidate"] == config["id"]]
        if not selected:
            continue
        folds = sorted({r["heldout_background"] for r in selected})
        stats = {}
        for metric in ("proxy_delta_mse", "proxy_delta_mse_excluding_target", "proxy_delta_cosine"):
            fold_means = []
            for fold in folds:
                values = [r[metric] for r in selected if r["heldout_background"] == fold and r[metric] is not None]
                if values:
                    fold_means.append(float(np.mean(values)))
            stats[metric] = float(np.mean(fold_means)) if fold_means else None
        summaries.append({**config, **stats, "folds": folds, "n_targets_total": len(selected)})
    # K562-direct cannot score the K562-held-out fold, so do not rank it against
    # candidates evaluated on all folds. Its comparable per-fold rows are retained.
    eligible = [s for s in summaries if s["model"] != "k562_direct" and len(s["folds"]) == len(fold_reports)]
    selected = min(eligible, key=lambda s: (s["proxy_delta_mse_excluding_target"], s["alpha"], s["temperature"]))
    fitted = {k: selected[k] for k in ("id", "model", "alpha", "temperature")}
    fitted.update(prior_cells=args.prior_cells, min_cells=args.min_cells, min_genes=args.min_genes,
                  schema=SCHEMA, genes=genes.tolist(), source_ids=[s.source_id for s in sources],
                  source_backgrounds={s.source_id:s.background for s in sources},
                  source_locks=locks,
                  source_files=[s.path for s in sources], selection_metric="equal-background mean proxy_delta_mse_excluding_target",
                  status="chosen_on_these_external_LOCO_folds; not an independent performance estimate or official score")
    write_json(output / "selected_model.json", fitted)
    write_json(output / "cv_report.json", dict(schema=SCHEMA, common_measured_genes=len(genes),
               background_count=len(backgrounds), candidates=summaries, folds=fold_reports,
               caveats=["Proxy metrics use pseudobulk log1p composition, not the six official cell-eval2 metrics.",
                        "Hyperparameters were selected on these same folds; the selected value is a tuning result.",
                        "Source files sharing a biological background are excluded together and receive one vote.",
                        "Zero has undefined effect-direction cosine when its effect vector is zero; it is reported null.",
                        "A two-background run is only a minimal development check, not evidence of broad generalization."]))
    print(json.dumps({"selected": fitted, "candidates": summaries}, ensure_ascii=False), flush=True)


def read_axis(path, accepted_headers):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    rows = [r for r in rows if r and any(v.strip() for v in r)]
    if not rows:
        raise ValueError(f"Empty axis: {path}")
    first = [x.strip() for x in rows[0]]
    column = 0
    if any(x in accepted_headers for x in first):
        column = next(i for i, x in enumerate(first) if x in accepted_headers)
        rows = rows[1:]
    axis = [r[column].strip() for r in rows]
    if not axis or any(not x for x in axis) or len(set(axis)) != len(axis):
        raise ValueError(f"Empty or duplicate axis entries: {path}")
    return np.asarray(axis, dtype=str)


def read_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Official bundle manifest.json is required; context labels are never guessed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    contexts_value = manifest.get("contexts")
    files = {}
    if isinstance(contexts_value, dict):
        contexts = list(contexts_value)
        for label, value in contexts_value.items():
            if isinstance(value, str) and value.endswith(".h5ad"):
                files[label] = value
            elif isinstance(value, dict):
                files[label] = value.get("file") or value.get("filename") or value.get("path")
    elif isinstance(contexts_value, list):
        contexts = []
        for value in contexts_value:
            if isinstance(value, str):
                contexts.append(value)
            elif isinstance(value, dict):
                label = value.get("label") or value.get("context") or value.get("id") or value.get("name")
                if not label:
                    raise ValueError("Manifest context object has no label/context/id/name")
                contexts.append(str(label))
                files[str(label)] = value.get("file") or value.get("filename") or value.get("path")
            else:
                raise ValueError("Unsupported manifest contexts schema")
    else:
        raise ValueError("Manifest must explicitly specify contexts as a list or object")
    if len(contexts) != 3 or len(set(contexts)) != 3:
        raise ValueError("This VCC 2026 exporter requires exactly three distinct manifest contexts")
    paths = {}
    for label in contexts:
        path = (bundle / (files.get(label) or f"context_{label}.h5ad")).resolve()
        if not path.is_relative_to(bundle) or not path.is_file():
            raise ValueError(f"Missing or out-of-bundle control file for {label}: {path}")
        paths[label] = path
    genes = read_axis(bundle / "gene_names.csv", {"gene_name", "gene", "gene_symbol", "gene_symbols"})
    targets = read_axis(bundle / "pert_counts.csv", {"target_gene", "perturbation", "gene"})
    if len(genes) != 18533 or len(targets) != 300:
        raise ValueError(f"Official dimensions expected: 18533 genes and 300 targets, got {len(genes)}, {len(targets)}")
    if any(t.lower() in {"non-targeting", "ntc", "control", "__control__"} for t in targets):
        raise ValueError("Perturbation panel contains controls")
    return bundle, manifest, contexts, paths, genes, targets


def control_statistics(path, context, genes, chunk_rows=1024, *, manifest_binding=None, label_audit=None):
    a = ad.read_h5ad(path, backed="r")
    try:
        if "context" in a.obs:
            if set(strings(a.obs["context"].to_numpy())) != {context}:
                raise ValueError(f"{path}: control cells do not retain manifest context {context}")
            label_source="control.obs.context"
        else:
            # Some official control bundles omit the redundant observation
            # column. Only a binding returned by read_bundle can supply it.
            if (manifest_binding is None or context not in manifest_binding
                    or Path(manifest_binding[context]).resolve()!=Path(path).resolve()):
                raise ValueError(f"{path}: absent context labels require an explicit manifest-to-file binding")
            label_source="official manifest context-to-file binding; obs.context absent"
        if label_audit is not None:
            label_audit["context_label_source"]=label_source
        if "target_gene" not in a.obs or set(strings(a.obs["target_gene"].to_numpy())) != {"non-targeting"}:
            raise ValueError(f"{path}: expected only non-targeting control cells")
        g = strings(a.var_names)
        if len(set(g)) != len(g) or set(g) != set(genes):
            raise ValueError(f"{path}: official control gene axis mismatch")
        gmap = {g:i for i,g in enumerate(g)}
        columns = np.asarray([gmap[g] for g in genes])
        sums = np.zeros(len(genes), dtype=np.float64)
        sums2 = np.zeros(len(genes), dtype=np.float64)
        for start in range(0, a.n_obs, chunk_rows):
            x = matrix_slice(a, "X", start, min(start+chunk_rows, a.n_obs), columns)
            check_raw(x, True)
            sums += np.asarray(x.sum(axis=0)).ravel()
            sums2 += np.asarray(x.astype(np.float64).power(2).sum(axis=0)).ravel() if sparse.issparse(x) else np.sum(np.asarray(x,dtype=np.float64)**2, axis=0)
        if a.n_obs < 2:
            raise ValueError("Too few official control cells")
        mean = sums/a.n_obs
        variance = np.maximum((sums2-a.n_obs*mean**2)/(a.n_obs-1), 0)
        return mean, variance, a.n_obs, columns
    finally:
        a.file.close()


def init_output(path, contexts, targets, genes, cells_per_target=400):
    n = len(contexts) * len(targets) * cells_per_target
    obs = pd.DataFrame(index=pd.Index([f"{c}_{t}_{i:03d}" for c in contexts for t in targets for i in range(cells_per_target)]))
    obs["context"] = pd.Categorical(np.repeat(contexts, len(targets)*cells_per_target), categories=contexts)
    obs["target_gene"] = pd.Categorical(np.tile(np.repeat(targets, cells_per_target), len(contexts)), categories=targets)
    a = ad.AnnData(sparse.csr_matrix((n,len(genes)), dtype=np.int32), obs=obs,
                   var=pd.DataFrame(index=pd.Index(genes)))
    a.write_h5ad(path)
    with h5py.File(path, "r+") as f:
        del f["X"]
        x = f.create_group("X")
        x.attrs.update({"encoding-type":"csr_matrix", "encoding-version":"0.1.0", "shape":np.asarray([n,len(genes)],np.int64)})
        for name, dtype in (("data",np.int32),("indices",np.int32)):
            x.create_dataset(name,shape=(0,),maxshape=(None,),dtype=dtype,chunks=(1<<20,),compression="gzip",compression_opts=1,shuffle=True)
        x.create_dataset("indptr",shape=(n+1,),dtype=np.int64)
        x["indptr"][0]=0


def append_block(x, block, row_offset, nnz):
    block = sparse.csr_matrix(block)
    # Validate original values before narrowing; a fractional float or a large
    # integer must not become an apparently valid small int32 count.
    check_raw(block, True)
    if (block.data > 1_000_000).any():
        raise ValueError("Generated entry exceeds the per-cell count cap")
    row_totals = np.asarray(block.astype(np.float64).sum(axis=1)).ravel()
    if ((row_totals <= 0) | (row_totals > 1_000_000)).any():
        raise ValueError("Every generated cell must have >0 and <=1,000,000 counts")
    block = block.astype(np.int32)
    block.eliminate_zeros()
    block.sort_indices()
    end = nnz+block.nnz
    if end > 4_750_000_000:
        raise ValueError("Generated stored-entry count exceeds official density limit")
    for key, value in (("data",block.data),("indices",block.indices)):
        x[key].resize((end,))
        x[key][nnz:end]=value
    x["indptr"][row_offset+1:row_offset+block.shape[0]+1]=block.indptr[1:].astype(np.int64)+nnz
    return end


def sample_counts(mean, control_mean, control_variance, rng, n=400, generator="nb", theta_floor=0.1):
    if not np.isfinite(mean).all() or (mean < 0).any() or not 0 < mean.sum() <= 900_000:
        raise ValueError("Invalid or implausibly large requested count mean")
    if generator == "poisson":
        values = rng.poisson(mean, size=(n,len(mean)))
    else:
        # Preserve the control's gene-wise NB dispersion, not a fabricated exact
        # match to perturbed variance. Low-mean/underdispersed genes use Poisson.
        excess = control_variance-control_mean
        over = (excess > 1e-8) & (control_mean > 0)
        theta = np.full(len(mean), 1e6)
        theta[over] = np.clip(control_mean[over]**2/excess[over], theta_floor, 1e6)
        rates = rng.gamma(theta, mean/np.maximum(theta,1e-12), size=(n,len(mean)))
        values = rng.poisson(rates)
    return sparse.csr_matrix(values)


def sample_target_blocks(targets, mean, variance, positions, desired, gene_positions, *,
                         seed, context_index, generator="nb", theta_floor=0.1,
                         on_target_fraction=None, workers=1, n=400):
    """Yield target count blocks in panel order with bounded parallel sampling.

    Workers only sample in memory. The caller remains the sole HDF5 writer.
    Preserve the original per-context/per-target SeedSequence, so changing the
    worker count cannot change counts for an unchanged bundle and parameters.
    """
    if workers < 1:
        raise ValueError("workers must be positive")

    def sample_one(item):
        ti, target = item
        wanted = mean.copy()
        wanted[positions] = desired[ti]
        if on_target_fraction is not None and target in gene_positions:
            gi = gene_positions[target]
            wanted[gi] = mean[gi] * on_target_fraction
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(context_index, ti)))
        block = sample_counts(wanted, mean, variance, rng, n=n,
                              generator=generator, theta_floor=theta_floor)
        return ti, target, block

    items = iter(enumerate(targets))
    if workers == 1:
        for item in items:
            yield sample_one(item)
        return

    # executor.map eagerly submits the full iterable on older Python versions.
    # Keep at most 2*workers futures, including completed blocks awaiting write.
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vcc-counts")
    pending = deque()
    try:
        for _ in range(2 * workers):
            item = next(items, None)
            if item is None:
                break
            pending.append(pool.submit(sample_one, item))
        while pending:
            future = pending.popleft()
            yield future.result()
            del future
            # Refill only after the caller has consumed/written this block.
            item = next(items, None)
            if item is not None:
                pending.append(pool.submit(sample_one, item))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def load_external_effects(path, targets, official_genes, control_mean):
    with np.load(path,allow_pickle=False) as z:
        effect = z["effect_log2fc"].astype(np.float64)
        genes, source_targets = z["gene_symbols"].astype(str), z["target_symbols"].astype(str)
        if "effect_space" in z and str(z["effect_space"].item()) != "raw_pseudocount_log2fc":
            raise ValueError(f"{path}: incompatible effect space")
        if "pseudocount" in z and float(z["pseudocount"].item()) != 0.5:
            raise ValueError(f"{path}: expected raw-count pseudocount 0.5")
    if len(set(genes)) != len(genes) or not set(genes).issubset(official_genes):
        raise ValueError(f"{path}: duplicate/nonofficial effect genes")
    if len(set(source_targets)) != len(source_targets) or set(source_targets) != set(targets):
        raise ValueError(f"{path}: effect target axis does not exactly match official panel")
    if effect.shape != (len(source_targets),len(genes)) or not np.isfinite(effect).all():
        raise ValueError(f"{path}: invalid effect matrix")
    tp={t:i for i,t in enumerate(source_targets)}
    gp={g:i for i,g in enumerate(official_genes)}
    effect=effect[[tp[t] for t in targets]]
    positions=np.asarray([gp[g] for g in genes])
    if (np.abs(effect)>20).any():
        raise ValueError(f"{path}: implausibly large log2FC; inspect effects before count generation")
    desired=np.maximum((control_mean[positions][None,:]+0.5)*np.exp2(effect)-0.5,0)
    return positions,desired


def predict(args):
    workers = getattr(args, "workers", 1)
    if workers < 1:
        raise ValueError("workers must be positive")
    bundle, manifest, contexts, paths, genes, targets = read_bundle(args.bundle)
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(output.stem+".partial.h5ad")
    if output.exists() or partial.exists():
        raise FileExistsError("Output or partial exists; inspect it before writing a new output path")
    if args.effects_dir:
        sources=[]
        fitted={"model":"external_effects", "effect_space":"raw_pseudocount_log2fc", "pseudocount":0.5}
        core=None
    else:
        if not args.sources or not args.model_json:
            raise ValueError("Provide --sources and --model-json, or --effects-dir")
        sources=load_sources(args.sources)
        fitted=json.loads(Path(args.model_json).read_text(encoding="utf-8"))
        validate_source_locks(sources,fitted)
        core=common_genes(sources,genes,minimum=fitted.get("min_genes",10))
        if set(core) != set(fitted["genes"]):
            raise ValueError("Model common gene axis changed since calibration; rerun cv with these sources")
        if sorted(s.source_id for s in sources) != sorted(fitted["source_ids"]):
            raise ValueError("Source set changed since calibration; rerun cv")
    gmap={g:i for i,g in enumerate(genes)}
    init_output(partial,contexts,targets,genes)
    report=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),bundle=str(bundle),
                manifest_sha256=sha256(bundle/"manifest.json"),genes_sha256=sha256(bundle/"gene_names.csv"),
                targets_sha256=sha256(bundle/"pert_counts.csv"),contexts=contexts,shape=[len(contexts)*len(targets)*400,len(genes)],
                cells_per_target=400,seed=args.seed,generator=args.generator,theta_floor=args.theta_floor,
                count_workers=workers,max_pending_count_tasks=2*workers if workers>1 else 1,
                count_seed_scheme="SeedSequence(seed, spawn_key=(manifest_context_index, panel_target_index))",
                model=fitted,on_target_fraction=args.on_target_fraction,context_reports={},
                status="local prediction only; not prepared or submitted to VCC")
    nnz,row_offset=0,0
    with h5py.File(partial,"r+") as f:
        for ci,context in enumerate(contexts):
            label_audit={}
            mean,var,n_controls,columns=control_statistics(paths[context],context,genes,
                                                          manifest_binding=paths,label_audit=label_audit)
            if args.effects_dir:
                effect_path=Path(args.effects_dir)/f"context_{context}.npz"
                positions,desired=load_external_effects(effect_path,targets,genes,mean)
                audit={"external_effects":str(effect_path.resolve()),"sha256":sha256(effect_path)}
            else:
                positions=np.asarray([gmap[g] for g in core])
                delta,audit=predict_delta(sources,core,targets,mean[positions],
                    model=fitted["model"],alpha=fitted["alpha"],temperature=fitted["temperature"],
                    prior_cells=fitted["prior_cells"],min_cells=fitted["min_cells"])
                _,desired=project_delta(delta,mean[positions])
            context_report=dict(n_controls=n_controls,modeled_genes=len(positions),**audit,**label_audit)
            max_cell_total=0
            with closing(sample_target_blocks(targets,mean,var,positions,desired,gmap,
                         seed=args.seed,context_index=ci,generator=args.generator,
                         theta_floor=args.theta_floor,on_target_fraction=args.on_target_fraction,
                         workers=workers)) as blocks:
                for ti,target,block in blocks:
                    max_cell_total=max(max_cell_total,int(np.asarray(block.sum(axis=1)).max()))
                    nnz=append_block(f["X"],block,row_offset,nnz)
                    row_offset+=400
                    del block
                    if (ti+1)%25==0:
                        f.flush()
                        print(f"predict {context}: {ti+1}/{len(targets)} targets; {nnz:,} stored entries",flush=True)
            context_report["max_cell_total"]=max_cell_total
            report["context_reports"][context]=context_report
        if row_offset != report["shape"][0] or int(f["X/indptr"][-1]) != nnz:
            raise ValueError("Output row count/CSR pointer mismatch")
    # Reopen through AnnData to check schema and all observation labels.
    verified=ad.read_h5ad(partial,backed="r")
    try:
        if list(verified.shape) != report["shape"] or list(verified.var_names) != list(genes):
            raise ValueError("AnnData round-trip dimension/gene check failed")
        counts=verified.obs.groupby(["context","target_gene"],observed=True).size()
        if len(counts)!=len(contexts)*len(targets) or not (counts==400).all():
            raise ValueError("AnnData round-trip per-context target counts failed")
    finally:
        verified.file.close()
    partial.rename(output)
    report.update(output=str(output),bytes=output.stat().st_size,stored_entries=nnz)
    write_json(output.with_suffix(".manifest.json"),report)
    print(json.dumps({"output":str(output),"shape":report["shape"],"bytes":report["bytes"],"stored_entries":nnz,
                      "next_step":"Run vcc prep against this same official bundle; this script never submits"}),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("ingest",help="Aggregate a raw single-cell or construct-mean H5AD")
    q.add_argument("--input",required=True);q.add_argument("--output",required=True)
    q.add_argument("--source-id",required=True);q.add_argument("--background",required=True)
    q.add_argument("--kind",choices=["raw-bulk","singlecell"],required=True)
    q.add_argument("--target-col",default="auto",help="Column, auto, or construct-index for Replogle raw_bulk names");q.add_argument("--gene-col",default="auto")
    q.add_argument("--n-col",default="n_cells");q.add_argument("--matrix",default="X")
    q.add_argument("--control-col",help="Explicit boolean control flag, e.g. core_control, after source semantics are confirmed")
    q.add_argument("--control-label",action="append",default=None)
    q.add_argument("--filter",action="append",default=[],help="Optional exact obs filter COLUMN=VALUE; repeatable")
    q.add_argument("--chunk-rows",type=int,default=512);q.set_defaults(func=ingest)
    q=sub.add_parser("cv",help="Background-held-out proxy comparison and transparent shrinkage selection")
    q.add_argument("--sources",nargs="+",required=True);q.add_argument("--output-dir",required=True)
    q.add_argument("--genes",help="Optional official gene_names.csv; rerun cv with it before official prediction")
    q.add_argument("--min-cells",type=int,default=30);q.add_argument("--prior-cells",type=float,default=50)
    q.add_argument("--min-genes",type=int,default=100);q.add_argument("--max-targets",type=int,default=1000)
    q.add_argument("--alphas",default="0.25,0.5,0.75,1.0");q.add_argument("--temperatures",default="0,1,4")
    q.add_argument("--seed",type=int,default=20261103);q.set_defaults(func=cv)
    q=sub.add_parser("predict",help="Generate one official-shaped sparse count H5AD without uploading")
    q.add_argument("--bundle",required=True);q.add_argument("--output",required=True)
    q.add_argument("--sources",nargs="+");q.add_argument("--model-json")
    q.add_argument("--effects-dir",help="Optional context_LABEL.npz raw-pseudocount log2FC predictions")
    q.add_argument("--generator",choices=["nb","poisson"],default="nb")
    q.add_argument("--theta-floor",type=float,default=0.1)
    q.add_argument("--workers",type=int,default=1,help="Parallel count samplers; at most 2*workers blocks pending, with a single ordered HDF5 writer")
    q.add_argument("--on-target-fraction",type=float,default=None,help="Explicit optional residual target fraction; not calibrated by cv")
    q.add_argument("--seed",type=int,default=20261103);q.set_defaults(func=predict)
    return p


def main():
    args=parser().parse_args()
    if args.command=="ingest" and args.control_label is None:
        args.control_label=["non-targeting","non_targeting","control","NTC","ntc"]
    if args.command=="predict":
        if args.workers<1:
            raise ValueError("workers must be positive")
        if args.theta_floor<=0 or (args.on_target_fraction is not None and not 0<=args.on_target_fraction<=1):
            raise ValueError("theta-floor must be positive; on-target-fraction must lie in [0,1]")
    args.func(args)


if __name__=="__main__":
    main()
