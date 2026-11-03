"""Measure 1 versus 6 count workers on real RPE1-derived predicted means.

No raw-cell reread, output H5AD, model fit, official input, or submission is used.
Default: 12 deterministic targets, 400 cells each, two alternating runs per
worker count.  Timings include the same ordered validation/digest consumer for
both modes, but exclude source loading and mean prediction.  If the processed
RPE1 file has no genuine variance array, Poisson is used and reported explicitly.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

import numpy as np

import lean_transfer as lt
import lean_union as lu


class RSSProbe:
    """Sample this process, including its sampler threads, every 20 ms."""

    def __init__(self):
        self.stop = threading.Event()
        self.thread = None
        self.process = None
        self.before = self.peak = self.after = None
        try:
            import psutil
            self.process = psutil.Process()
            self.method = "psutil RSS sampled every 20 ms"
        except ImportError:
            self.method = "unavailable: psutil is not installed"

    def __enter__(self):
        if self.process is not None:
            self.before = self.peak = self.process.memory_info().rss

            def sample():
                while not self.stop.wait(0.02):
                    self.peak = max(self.peak, self.process.memory_info().rss)

            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.process is not None:
            self.stop.set()
            self.thread.join()
            self.after = self.process.memory_info().rss
            self.peak = max(self.peak, self.after)

    def result(self):
        return {"method": self.method, "before_bytes": self.before,
                "sampled_peak_bytes": self.peak, "after_bytes": self.after,
                "increase_over_before_bytes": None if self.peak is None else self.peak - self.before}


def array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(json.dumps({"shape": list(array.shape), "dtype": array.dtype.str}, sort_keys=True).encode())
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def block_digest(block):
    if block.format != "csr" or not block.has_canonical_format:
        raise ValueError("Sampler returned a noncanonical CSR matrix")
    lt.check_raw(block, True)
    libraries = np.asarray(block.sum(axis=1)).ravel()
    if np.any(libraries <= 0) or np.any(libraries > 1_000_000):
        raise ValueError("Invalid generated library size")
    h = hashlib.sha256()
    h.update(np.asarray(block.shape, dtype="<i8").tobytes())
    for name in ("data", "indices", "indptr"):
        value = np.ascontiguousarray(getattr(block, name))
        h.update(name.encode() + b":" + value.dtype.str.encode() + b"\0")
        h.update(memoryview(value).cast("B"))
    return {"sha256": h.hexdigest(), "shape": list(block.shape), "nnz": int(block.nnz),
            "min_library_counts": int(libraries.min()), "max_library_counts": int(libraries.max()),
            "total_counts": int(libraries.sum())}


def run_once(workers, targets, mean, variance, desired, genes, *, seed, generator):
    gc.collect()
    positions = np.arange(len(genes), dtype=np.int64)
    gene_positions = {g: i for i, g in enumerate(genes)}
    records = []
    with RSSProbe() as rss:
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        stream = lt.sample_target_blocks(
            targets, mean, variance, positions, desired, gene_positions,
            seed=seed, context_index=0, generator=generator, workers=workers, n=400)
        with closing(stream):
            for index, target, block in stream:
                if index != len(records) or str(target) != str(targets[index]):
                    raise ValueError("Sampler changed target stream order")
                if block.shape != (400, len(genes)):
                    raise ValueError("Unexpected count block dimensions")
                records.append({"index": int(index), "target": str(target), **block_digest(block)})
                del block
        wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
    if len(records) != len(targets):
        raise ValueError("Incomplete target stream")
    stream_sha = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"workers": workers, "wall_seconds": wall, "process_cpu_seconds": cpu,
            "cells_per_second": len(targets) * 400 / wall, "rss": rss.result(),
            "stream_sha256": stream_sha, "target_blocks": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-json", type=Path, default=Path("outputs/restart/union_cv/selected_model.json"))
    parser.add_argument("--targets", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path, default=Path("reports/count_worker_benchmark.json"))
    args = parser.parse_args()
    if args.targets < 2 or args.repeats < 1:
        parser.error("--targets must be >=2 and --repeats >=1")
    root = args.root.resolve()
    model_path = args.model_json if args.model_json.is_absolute() else root / args.model_json
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"Benchmark already exists; choose a new --output: {output}")
    prepared_start = time.perf_counter()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("schema") != "vcc_restart_measured_union_v1":
        raise ValueError("Use a saved measured-union model, not a common-core or legacy model")
    source_paths = [root / "data/processed/restart" / f"{name}.npz" for name in model["source_ids"]]
    sources = lt.load_sources(source_paths)
    lt.validate_source_locks(sources, model)
    queries = [s for s in sources if s.source_id == "rpe1_sc"]
    if len(queries) != 1:
        raise ValueError("The saved calibration source set must contain exactly one rpe1_sc")
    query = queries[0]
    training = [s for s in sources if s.background != query.background]
    if not training:
        raise ValueError("No sources remain after excluding the entire RPE1 background")
    genes = np.asarray([g for g in query.genes if g in query.gmap], str)
    positions = np.asarray([query.gmap[g] for g in genes])
    mean = query.control_mean[positions].copy()
    core = np.asarray(model["core_genes"], str)
    gp = {g: i for i, g in enumerate(genes)}
    if any(g not in gp for g in core):
        raise ValueError("RPE1 lacks a saved normalization-core gene")
    core_positions = np.asarray([gp[g] for g in core])
    train_targets = {str(t) for s in training for t, n in zip(s.targets, s.n_cells) if n >= model["min_cells"]}
    available = [str(t) for t, n in zip(query.targets, query.n_cells)
                 if n >= model["min_cells"] and str(t) in train_targets]
    targets = sorted(available, key=lambda t: hashlib.sha256(f"{args.seed}:{t}".encode()).digest())[:args.targets]
    if len(targets) != args.targets:
        raise ValueError(f"Only {len(targets)} supported RPE1 targets; requested {args.targets}")
    targets = np.asarray(targets, str)
    delta, measured, audit = lu.predict_delta(
        training, core, genes, targets, mean, model=model["model"], alpha=model["alpha"],
        temperature=model["temperature"], prior_cells=model["prior_cells"], min_cells=model["min_cells"])
    _, desired = lu.project(delta, mean, core_positions, measured)
    if not np.isfinite(desired).all() or np.any(desired < 0):
        raise ValueError("Actual predicted means are invalid")

    # Current processed sources store means but no second moment. Inspect the
    # real file; never relabel an invented variance as an observed statistic.
    variance, variance_key = None, None
    with np.load(query.path, allow_pickle=False) as bank:
        npz_keys = list(bank.files)
        names = [name for name in ("control_variance_counts", "control_variance") if name in bank]
        if len(names) > 1:
            raise ValueError("Multiple variance arrays are present; resolve their meaning explicitly")
        if names:
            variance_key = names[0]
            value = np.asarray(bank[variance_key], np.float64)
            if value.shape != query.control_mean.shape or not np.isfinite(value[positions]).all() or np.any(value[positions] < 0):
                raise ValueError("Stored control variance is not gene-aligned, finite and nonnegative")
            variance = value[positions]
    generator = "nb" if variance is not None else "poisson"
    prediction_seconds = time.perf_counter() - prepared_start
    # One actual-count warmup excludes import/initial allocator startup, while
    # thread-pool startup remains inside every timed parallel run.
    warmup = lt.sample_counts(desired[0], mean, variance, np.random.default_rng(args.seed), n=400, generator=generator)
    block_digest(warmup)
    del warmup
    order = [worker for rep in range(args.repeats) for worker in ((1, 6) if rep % 2 == 0 else (6, 1))]
    report = {
        "schema": "vcc_restart_actual_count_worker_benchmark_v1", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Real external RPE1 control means and held-background union predictions; engineering throughput only",
        "official_submission_performed": False, "raw_cells_reread": False,
        "model_path": str(model_path), "model_sha256": lt.sha256(model_path),
        "source_locks": model["source_locks"], "query_source": query.source_id,
        "query_npz_keys": npz_keys, "query_control_cells": query.control_n,
        "excluded_training_background": query.background,
        "targets": targets.tolist(), "genes": len(genes), "cells_per_target": 400,
        "cells_per_run": len(targets) * 400, "seed": args.seed, "context_index": 0,
        "generator": generator, "variance_key": variance_key,
        "variance_note": "Stored raw-control variance used" if variance is not None else "No variance stored; Poisson uses actual means only, variance is None",
        "desired_means_sha256": array_digest(desired), "control_means_sha256": array_digest(mean),
        "prediction_audit": audit, "source_loading_and_prediction_seconds": prediction_seconds,
        "timing_scope": "Count sampling + ordered count validation and CSR hashing; excludes loading, prediction and HDF5 compression/writing",
        "equality_rule": "Exact ordered canonical CSR shape/data/indices/indptr SHA256; no tolerance or reseeding by worker",
        "run_order": order, "runs": [], "numpy_version": np.__version__,
        "python_version": sys.version, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "code_sha256": {p.name: lt.sha256(p) for p in (Path(__file__), Path(lt.__file__), Path(lu.__file__))},
    }
    lt.write_json(output, report)
    try:
        reference = None
        for index, workers in enumerate(order):
            run = run_once(workers, targets, mean, variance, desired, genes, seed=args.seed, generator=generator)
            if reference is None:
                reference = run["target_blocks"]
            run["exact_digest_match"] = run["target_blocks"] == reference
            report["runs"].append(run)
            lt.write_json(output, report)
            if not run["exact_digest_match"]:
                raise AssertionError("Worker count changed an ordered count stream")
            print(json.dumps({"run": index + 1, "workers": workers, "wall_seconds": run["wall_seconds"],
                              "exact_digest_match": True, "rss": run["rss"]}), flush=True)
        summaries = {}
        for workers in (1, 6):
            runs = [r for r in report["runs"] if r["workers"] == workers]
            walls = [r["wall_seconds"] for r in runs]
            peaks = [r["rss"]["sampled_peak_bytes"] for r in runs if r["rss"]["sampled_peak_bytes"] is not None]
            summaries[str(workers)] = {"runs": len(runs), "median_wall_seconds": statistics.median(walls),
                "min_wall_seconds": min(walls), "max_wall_seconds": max(walls),
                "median_cells_per_second": len(targets) * 400 / statistics.median(walls),
                "max_sampled_process_rss_bytes": max(peaks) if peaks else None}
        speedup = summaries["1"]["median_wall_seconds"] / summaries["6"]["median_wall_seconds"]
        report.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
            all_streams_identical=True, worker_summary=summaries, speedup_6_over_1=speedup,
            limitations=["Twelve targets are a short runtime check, not a full 900-target throughput forecast.",
                         "HDF5 compression and disk writing are excluded and may limit end-to-end acceleration.",
                         "A Poisson benchmark cannot establish the speedup of the NB Gamma-Poisson path.",
                         "RSS is sampled in one process; allocator-retained pages affect later baseline and peak readings."])
        lt.write_json(output, report)
        print(json.dumps({"status": "complete", "generator": generator, "all_streams_identical": True,
                          "speedup_6_over_1": speedup, "worker_summary": summaries, "report": str(output)}), flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        lt.write_json(output, report)
        raise


if __name__ == "__main__":
    main()
