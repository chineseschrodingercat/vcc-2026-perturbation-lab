"""Identical V22 cell draws, optionally centered by NTC-only GEM estimates.

No new raw-cell or destination perturbation outcomes are read. Batch means
use only cached K562 NTCs, with pseudo-count shrinkage toward their global
mean. Missing NTC batches use the global mean. Destination NTCs are unchanged.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_set_response_v22 import ROOT as OLD_ROOT, load_cache as old_load_cache

SHRINKAGE_NTC = 20.0


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def add_ntc_centering(cache, shrinkage=SHRINKAGE_NTC):
    """Return a new cache dictionary; all original arrays remain read-only."""
    if shrinkage <= 0:
        raise ValueError("NTC shrinkage must be positive")
    result = dict(cache)
    cells = cache["controls"]["k562"]
    metadata = cache["control_metadata"]["k562"]
    if len(metadata) != len(cells) or not metadata.target.eq("non-targeting").all():
        raise ValueError("Centering inputs must all be K562 NTC cells")
    if "role" in metadata and not metadata.role.eq("input_NTC_only").all():
        raise ValueError("Unexpected source NTC provenance role")
    if not np.isfinite(cells).all():
        raise ValueError("Nonfinite source NTC PCs")
    global_mean = np.asarray(cells, dtype=np.float64).mean(axis=0)
    names, counts, batch_means, shrunk = [], [], [], []
    for batch, rows in cache["control_batch_groups"]["k562"].items():
        rows = np.asarray(rows, dtype=np.int64)
        if not len(rows):
            continue
        if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(cells)):
            raise ValueError("Invalid NTC batch row group")
        name = str(batch)
        if not metadata.iloc[rows].batch.astype(str).eq(name).all():
            raise ValueError("NTC batch group disagrees with its row metadata")
        mean = np.asarray(cells[rows], dtype=np.float64).mean(axis=0)
        n = len(rows)
        names.append(name); counts.append(n); batch_means.append(mean)
        shrunk.append((n * mean + shrinkage * global_mean) / (n + shrinkage))
    names = np.asarray(names, dtype=str)
    lookup = {name: i for i, name in enumerate(names)}
    # Last row is the declared fallback; integer codes avoid a full 600k x 64 copy.
    center_table = np.concatenate((np.asarray(shrunk), global_mean[None]), axis=0).astype(np.float32)
    fallback = len(names)
    source_codes = np.asarray([lookup.get(str(b), fallback) for b in cache["source_batch"]], dtype=np.int32)
    ntc_codes = np.asarray([lookup.get(str(b), fallback) for b in metadata.batch], dtype=np.int32)
    result["ntc_centering"] = {
        "shrinkage_ntc": float(shrinkage), "batch_names": names,
        "batch_counts": np.asarray(counts, dtype=np.int64),
        "batch_means": np.asarray(batch_means, dtype=np.float64),
        "global_mean": global_mean, "center_table": center_table,
        "source_codes": source_codes, "ntc_codes": ntc_codes,
        "fallback_code": fallback,
    }
    return result


@lru_cache(maxsize=1)
def load_batch_cache(root=OLD_ROOT, shrinkage=SHRINKAGE_NTC):
    return add_ntc_centering(old_load_cache(root), shrinkage)


def sample_sets(targets, lines, rng, set_size=16, cache=None, centered=False,
                return_metadata=False):
    """Return the old four arrays, with identical selected rows and RNG state.

    ``return_metadata=True`` adds a fifth dictionary of selected cache row IDs.
    The absent-target source perturbation is the exact same NTC draw, before
    and after centering. NTC and perturbation are centered by their own GEM.
    """
    if cache is None:
        cache = load_batch_cache()
    if centered and "ntc_centering" not in cache:
        cache = add_ntc_centering(cache)
    targets = np.asarray(targets, dtype=str)
    lines = np.repeat(lines, len(targets)) if isinstance(lines, str) else np.asarray(lines, dtype=str)
    if len(lines) != len(targets) or set_size < 1:
        raise ValueError("Positive set size and one destination line per target required")
    dimensions = cache["source_cells"].shape[1]
    dest = np.empty((len(targets), set_size, dimensions), dtype=np.float32)
    ntc = np.empty_like(dest); pert = np.empty_like(dest)
    valid = np.zeros(len(targets), dtype=bool)
    dest_rows = np.empty((len(targets), set_size), dtype=np.int64)
    ntc_rows = np.empty_like(dest_rows)
    pert_rows = np.full_like(dest_rows, -1)
    source_ntc = cache["controls"]["k562"]
    source_batch_groups = cache["control_batch_groups"]["k562"]
    for i, (target, line) in enumerate(zip(targets, lines)):
        control = cache["controls"][line]
        matched = cache["control_target_batch_weights"].get(line)
        if matched is not None and target in matched["target_map"]:
            probabilities = matched["weights"][matched["target_map"][target]].astype(np.float64)
            probabilities /= probabilities.sum()
            batches = rng.choice(matched["batch_names"], size=set_size, p=probabilities)
            destination_ids = np.asarray([rng.choice(cache["control_batch_groups"][line][batch]) for batch in batches])
        else:
            destination_ids = rng.choice(len(control), set_size, replace=len(control) < set_size)
        dest[i] = control[destination_ids]; dest_rows[i] = destination_ids
        source_id = cache["source_target_map"].get(target)
        if source_id is None:
            ids = rng.choice(len(source_ntc), set_size, replace=len(source_ntc) < set_size)
            ntc[i] = source_ntc[ids]; ntc_rows[i] = ids
            pert[i] = ntc[i]
            continue
        start, stop = cache["source_offsets"][source_id:source_id + 2]
        pert_ids = rng.choice(np.arange(start, stop), set_size, replace=stop - start < set_size)
        pert[i] = cache["source_cells"][pert_ids]; pert_rows[i] = pert_ids
        ntc_ids = []
        for batch in cache["source_batch"][pert_ids]:
            candidates = source_batch_groups.get(batch)
            ntc_ids.append(rng.choice(candidates) if candidates is not None else rng.integers(len(source_ntc)))
        ntc_ids = np.asarray(ntc_ids)
        ntc[i] = source_ntc[ntc_ids]; ntc_rows[i] = ntc_ids
        valid[i] = True
    if centered:
        center = cache["ntc_centering"]
        ntc -= center["center_table"][center["ntc_codes"][ntc_rows]]
        present = np.flatnonzero(valid)
        if len(present):
            pert[present] -= center["center_table"][center["source_codes"][pert_rows[present]]]
        pert[~valid] = ntc[~valid]
    arrays = (dest, ntc, pert, valid)
    if not return_metadata:
        return arrays
    return (*arrays, {"destination_rows": dest_rows, "source_ntc_rows": ntc_rows,
                     "source_perturbed_rows": pert_rows, "destination_lines": lines})


def write_centering_provenance(cache, output_dir):
    """Save immutable NTC-only means, counts, row identities and input hashes."""
    if "ntc_centering" not in cache:
        cache = add_ntc_centering(cache)
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    root = Path(cache["root"])
    center = cache["ntc_centering"]
    values = {key: center[key] for key in ("batch_names", "batch_counts", "batch_means",
                                         "global_mean", "center_table")}
    values["shrinkage_ntc"] = np.asarray(center["shrinkage_ntc"])
    values["fallback_code"] = np.asarray(center["fallback_code"])
    npz = output / "source_ntc_centering.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as stored:
            assert set(stored.files) == set(values)
            assert all(np.array_equal(stored[key], value) for key, value in values.items())
    else:
        np.savez_compressed(npz, **values)
    metadata = cache["control_metadata"]["k562"]
    csv = output / "source_ntc_rows.csv"
    text = metadata.to_csv(index=False, lineterminator="\n")
    if csv.exists():
        assert csv.read_text(encoding="utf-8") == text
    else:
        csv.write_text(text, encoding="utf-8", newline="")
    record = {"schema": "v24_K562_cached_NTC_batch_centering_1",
              "source": "K562 NTC cells only", "source_NTC_cells": len(metadata),
              "batch_count": len(center["batch_names"]), "PC_dimensions": len(center["global_mean"]),
              "shrinkage_ntc": center["shrinkage_ntc"],
              "formula": f"(nNTC * batch_NTC_PC_mean + {center['shrinkage_ntc']:g} * global_NTC_PC_mean) / (nNTC + {center['shrinkage_ntc']:g})",
              "mean_accumulation_dtype": "float64", "subtraction_dtype": "float32",
              "fallback": "global K562 NTC PC mean for missing NTC batch",
              "source_cells_using_fallback": int(np.sum(center["source_codes"] == center["fallback_code"])),
              "source_NTC_using_fallback": int(np.sum(center["ntc_codes"] == center["fallback_code"])),
              "destination_NTC_transformation": "unchanged",
              "no_destination_perturbation_response_read": True,
              "inputs_sha256": {str(path): sha_file(path) for path in
                  (root / "controls/k562/cells.npy", root / "controls/k562/cell_metadata.csv",
                   root / "encoder.npz", root / "source/cell_metadata.csv")},
              "artifacts_sha256": {str(path): sha_file(path) for path in (npz, csv)}}
    path = output / "source_ntc_centering.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == record
    else:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest = output / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8")) == record
    else:
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
