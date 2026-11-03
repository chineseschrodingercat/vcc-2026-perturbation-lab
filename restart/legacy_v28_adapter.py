"""Portable inference for the archived VCC 2026 V28 mean predictor.

This is a reconstructed adapter, not a claim of bit-exact reproduction of the
deleted preprocessing or the historical submission.  It does not train, sample
cells, score a submission, or contact the competition service.

The frozen model is a small V7 Transformer residual on top of a calibrated K562
raw-count effect.  Output effects use log2((perturbed_mean + 0.5) /
(control_mean + 0.5)); they are NOT normalized-expression log fold changes.

Library use::

    predictor = LegacyV28Adapter(source_npz=..., checkpoint=..., features=...,
                                string_embeddings=..., gene_axis=...,
                                normalization_core=..., device="cpu")
    result = predictor.predict(targets, control_gene_names, control_mean_counts)
    result.save("context_A.npz")

Only the five documented source NPZ keys are mandatory.  ``measured_mask``
may be gene-wise or target-by-gene; ``control_measured_mask`` is optional.
Unknown measurements remain explicit masks.  Source normalization requires the
entire saved core; a target with missing core measurements receives no neural
residual.  Query-core omissions are a hard error rather than a changed unit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CHECKPOINT_SHA256 = "a2b996eacf498ce63ff584be51d84bc52c77c87ec040f96a8a30d43a91d6dce2"
FEATURES_SHA256 = "a6cb72bcdcd0e0a5f01e9bce5a4a09d284fd74924373b69af363eebd694aa372"
STRING_SHA256 = "789416877b8701ef6f800106d26bf7bb97ea8e72744e6ab93e24933a717f247d"
GENE_AXIS_SHA256 = "74c1f3fde69301a0d3afb55224fbb4de375811e983c3dc734306ab49dfe63271"
CORE_AXIS_SHA256 = "889820f15bd8a966e3982b1b7f36645d0352584615c52bb7a046e87c9164ed66"
GENES = 10916
CORE_GENES = 6122
PSEUDOCOUNT = 0.5
CALIBRATION = 0.6720111954846246
MIN_SOURCE_CELLS = 30
SUPPORT_PRIOR = 50.0


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: str | Path, expected: str) -> Path:
    path = Path(path)
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"Frozen asset SHA256 differs: {path}; got {observed}")
    return path


def _identities(values: Sequence[str] | np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values).astype(str)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if len(set(values.tolist())) != len(values) or any(not x.strip() for x in values):
        raise ValueError(f"{name} contains duplicate or blank identities")
    return values


def load_features(path: str | Path) -> dict[str, np.ndarray]:
    path = _verify(path, FEATURES_SHA256)
    shapes = {"basis": (64, GENES), "center": (GENES,), "scale": (64,),
              "string_mean": (64,), "string_std": (64,), "seen": (GENES,),
              "source_ctrl": (GENES,)}
    with np.load(path, allow_pickle=False) as archive:
        params = {name: archive[name].copy() for name in shapes}
    for name, shape in shapes.items():
        if params[name].shape != shape or not np.isfinite(params[name]).all():
            raise ValueError(f"Invalid frozen feature array: {name}")
    if np.any(params["scale"] <= 0) or np.any(params["string_std"] <= 0):
        raise ValueError("Frozen feature scales must be positive")
    if not np.isin(params["seen"], [0.0, 1.0]).all():
        raise ValueError("Frozen seen mask is not binary")
    return params


def build_model(n_genes: int = GENES):
    """The archived tiny Transformer architecture, without historical imports."""
    import torch
    from torch import nn

    class FrozenJointResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.projections = nn.ModuleList([
                nn.Linear(66, 128), nn.Linear(64, 128), nn.Linear(64, 128)])
            self.token_type = nn.Parameter(torch.zeros(1, 3, 128))
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(128, 4, 256, 0.1, "gelu",
                                           batch_first=True, norm_first=True)
                for _ in range(2)])
            self.norm = nn.LayerNorm(128)
            self.readout = nn.Linear(128, 32)
            self.decoder = nn.Parameter(torch.zeros(n_genes, 32))
            self.background_decoder = nn.Parameter(torch.zeros(n_genes, 32))
            self.gate = nn.Linear(194, 1)

        def forward(self, z, g, h, seen, ownmask):
            blocks = [torch.cat([z[:, :64], z[:, 128:]], dim=1), z[:, 64:128], h]
            tokens = torch.stack([p(b) for p, b in zip(self.projections, blocks)], dim=1)
            tokens = tokens + self.token_type
            for layer in self.layers:
                tokens = layer(tokens)
            q = self.readout(self.norm(tokens).mean(1))
            base = q @ self.decoder.T
            background = (q @ self.background_decoder.T) * g
            gate = torch.sigmoid(self.gate(torch.cat([z, h], dim=1)))
            residual = gate * 0.75 * torch.tanh((base + background) / 0.75) * seen
            return residual * ownmask, gate

    # Model construction must not perturb a caller's training random stream.
    with torch.random.fork_rng(devices=[]):
        return FrozenJointResidual()


def load_model(checkpoint: str | Path, device: str = "cpu"):
    import torch

    path = _verify(checkpoint, CHECKPOINT_SHA256)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu")
    # This verified release contains tensors and primitive metadata only.
    # No unrestricted pickle fallback is performed.
    record = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"kind": "tiny_transformer", "ng": GENES,
                "step": 600, "seed": 20260928}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Unexpected checkpoint {key}: {record.get(key)!r}")
    model = build_model(GENES)
    model.load_state_dict(record["state_dict"], strict=True)
    model.eval().to(device)
    info = {**expected, "checkpoint_sha256": CHECKPOINT_SHA256,
            "train_contexts": record.get("train_contexts", []), "device": str(device),
            "torch_version": torch.__version__, "weights_only": True,
            "parameters": sum(p.numel() for p in model.parameters())}
    return model, info


@dataclass
class SourceBank:
    gene_names: np.ndarray
    target_names: np.ndarray
    target_mean_counts: np.ndarray
    control_mean_counts: np.ndarray
    target_n_cells: np.ndarray
    measured_mask: np.ndarray
    control_measured_mask: np.ndarray
    source_sha256: str

    @classmethod
    def read(cls, path: str | Path) -> "SourceBank":
        with np.load(path, allow_pickle=False) as archive:
            genes = _identities(archive["gene_names"], "source gene_names")
            targets = _identities(archive["target_names"], "source target_names")
            # Retain archived source precision where it exists. New aggregations
            # can be float64; neither case is called historical byte parity.
            means = np.asarray(archive["target_mean_counts"]).copy()
            controls = np.asarray(archive["control_mean_counts"]).copy()
            n = np.asarray(archive["target_n_cells"], dtype=np.float64)
            shape = (len(targets), len(genes))
            if means.shape != shape or controls.shape != (len(genes),) or n.shape != (len(targets),):
                raise ValueError("Source NPZ dimensions do not match its identities")
            measured = np.asarray(archive["measured_mask"], dtype=bool) if "measured_mask" in archive else np.ones(len(genes), bool)
            if measured.shape not in [(len(genes),), shape]:
                raise ValueError("measured_mask must be gene-wise or target-by-gene")
            control_measured = np.asarray(archive["control_measured_mask"], dtype=bool) if "control_measured_mask" in archive else (measured.copy() if measured.ndim == 1 else np.ones(len(genes), bool))
            if control_measured.shape != (len(genes),):
                raise ValueError("control_measured_mask must be gene-wise")
            measured = np.broadcast_to(measured, shape).copy() & np.isfinite(means)
            control_measured = control_measured & np.isfinite(controls)
            if np.any(means[measured] < 0) or np.any(controls[control_measured] < 0):
                raise ValueError("Source raw means must be nonnegative where measured")
            if not np.isfinite(n).all() or np.any(n < 0):
                raise ValueError("Invalid source target cell counts")
        return cls(genes, targets, means, controls, n, measured,
                   control_measured, sha256(path))


@dataclass
class EffectPrediction:
    target_symbols: np.ndarray
    gene_symbols: np.ndarray
    effect_log2fc: np.ndarray
    base_log2fc: np.ndarray
    residual_log2fc: np.ndarray
    source_measured_mask: np.ndarray
    query_measured_mask: np.ndarray
    residual_applied_mask: np.ndarray
    source_support_n: np.ndarray
    metadata: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Prediction output must have a .npz extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.with_suffix(path.suffix + ".json").exists():
            raise FileExistsError(f"Refusing to overwrite prediction: {path}")
        np.savez_compressed(
            path, target_symbols=self.target_symbols, gene_symbols=self.gene_symbols,
            effect_log2fc=self.effect_log2fc, base_log2fc=self.base_log2fc,
            residual_log2fc=self.residual_log2fc,
            source_measured_mask=self.source_measured_mask,
            query_measured_mask=self.query_measured_mask,
            residual_applied_mask=self.residual_applied_mask,
            source_support_n=self.source_support_n,
            effect_space=np.asarray("raw_pseudocount_log2fc"),
            pseudocount=np.asarray(PSEUDOCOUNT),
            metadata_json=np.asarray(json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)))
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        return path


class LegacyV28Adapter:
    """Frozen V28 means using an explicit, newly prepared K562 source bank."""

    def __init__(self, *, source_npz: str | Path, checkpoint: str | Path,
                 features: str | Path, string_embeddings: str | Path,
                 gene_axis: str | Path, normalization_core: str | Path,
                 device: str = "cpu"):
        self.genes = _identities(np.loadtxt(_verify(gene_axis, GENE_AXIS_SHA256), dtype=str), "model genes")
        self.core = _identities(np.loadtxt(_verify(normalization_core, CORE_AXIS_SHA256), dtype=str), "normalization core")
        if len(self.genes) != GENES or len(self.core) != CORE_GENES:
            raise ValueError("Frozen axes have unexpected dimensions")
        self.params = load_features(features)
        self.source = SourceBank.read(source_npz)
        self.string_path = _verify(string_embeddings, STRING_SHA256)
        self.model, self.model_info = load_model(checkpoint, device)
        self.device = self.model_info["device"]
        self.source_genes = {g: i for i, g in enumerate(self.source.gene_names)}
        self.source_targets = {t: i for i, t in enumerate(self.source.target_names)}
        self.model_genes = {g: i for i, g in enumerate(self.genes)}
        missing_core = [g for g in self.core if g not in self.source_genes or not self.source.control_measured_mask[self.source_genes[g]]]
        if missing_core:
            raise ValueError(f"K562 bank lacks {len(missing_core)} frozen-core control measurements, e.g. {missing_core[:10]}; do not renormalize on a smaller core")
        self.core_source = np.asarray([self.source_genes[g] for g in self.core])
        pairs = [(j, self.source_genes[g]) for j, g in enumerate(self.genes) if g in self.source_genes]
        self.model_columns = np.asarray([j for j, _ in pairs])
        self.source_columns = np.asarray([j for _, j in pairs])
        self.source_control_total = self.source.control_mean_counts[self.core_source].sum()
        if not np.isfinite(self.source_control_total) or self.source_control_total <= 0:
            raise ValueError("K562 frozen-core control library is not positive")

    def predict(self, targets: Sequence[str], control_gene_names: Sequence[str],
                control_mean_counts: np.ndarray, *,
                control_measured_mask: np.ndarray | None = None,
                batch_size: int = 128) -> EffectPrediction:
        import h5py
        import torch

        targets = _identities(targets, "prediction targets")
        native = _identities(control_gene_names, "control gene_names")
        control = np.asarray(control_mean_counts)
        if control.shape != native.shape:
            raise ValueError("Control mean and gene axis differ")
        known = np.ones(len(native), bool) if control_measured_mask is None else np.asarray(control_measured_mask, bool)
        if known.shape != native.shape:
            raise ValueError("Control measurement mask and gene axis differ")
        known = known & np.isfinite(control)
        if np.any(control[known] < 0):
            raise ValueError("Control raw means must be nonnegative")
        native_map = {g: i for i, g in enumerate(native)}
        missing_core = [g for g in self.core if g not in native_map or not known[native_map[g]]]
        if missing_core:
            raise ValueError(f"Query controls lack {len(missing_core)} frozen-core measurements, e.g. {missing_core[:10]}")
        control_total = sum(control[native_map[g]] for g in self.core)
        if not np.isfinite(control_total) or control_total <= 0:
            raise ValueError("Query frozen-core control library is not positive")
        query_known = np.array([g in native_map and known[native_map[g]] for g in self.genes])
        # An absent query measurement is neutral in the descriptor, not a zero
        # observed count. Its residual coordinate is masked below.
        ctrl_profile = self.params["source_ctrl"].copy()
        for j in np.flatnonzero(query_known):
            ctrl_profile[j] = np.log1p(control[native_map[self.genes[j]]] * 50000.0 / control_total)
        g_one = np.clip((ctrl_profile - self.params["source_ctrl"]) / 3.0, -2.0, 2.0).astype(np.float32)
        h_one = (g_one @ self.params["basis"].T / np.sqrt(GENES)).astype(np.float32)
        count = len(targets)
        base = np.zeros((count, GENES), np.float32)
        residual = np.zeros_like(base)
        source_known = np.zeros((count, GENES), bool)
        applied = np.zeros((count, GENES), bool)
        support = np.zeros(count, np.float64)
        normalized_effect = np.zeros_like(base)
        eligible, missing_core_targets = [], []
        source_ctrl = np.zeros(GENES, np.float32)
        mc, sc = self.model_columns, self.source_columns
        control_good = self.source.control_measured_mask[sc]
        source_ctrl[mc[control_good]] = np.log1p(
            50000.0 * self.source.control_mean_counts[sc[control_good]] / self.source_control_total)
        for i, target in enumerate(targets):
            row = self.source_targets.get(target)
            if row is None:
                continue
            support[i] = self.source.target_n_cells[row]
            good = self.source.measured_mask[row, sc] & control_good
            source_known[i, mc[good]] = True
            if support[i] < MIN_SOURCE_CELLS:
                continue
            shrink = support[i] / (support[i] + SUPPORT_PRIOR)
            destination, columns = mc[good], sc[good]
            raw = np.log2((self.source.target_mean_counts[row, columns] + PSEUDOCOUNT) /
                          (self.source.control_mean_counts[columns] + PSEUDOCOUNT))
            base[i, destination] = np.clip(CALIBRATION * shrink * raw, -3.0, 3.0)
            if not self.source.measured_mask[row, self.core_source].all():
                missing_core_targets.append(str(target))
                continue
            total = self.source.target_mean_counts[row, self.core_source].sum()
            if not np.isfinite(total) or total <= 0:
                missing_core_targets.append(str(target))
                continue
            effect = np.log1p(50000.0 * self.source.target_mean_counts[row, columns] / total) - source_ctrl[destination]
            normalized_effect[i, destination] = effect
            if target in self.model_genes:
                normalized_effect[i, self.model_genes[target]] = 0.0
            eligible.append(i)
        source_ctrl_error = source_ctrl - self.params["source_ctrl"]
        source_ctrl_measured = np.zeros(GENES, bool)
        source_ctrl_measured[mc[control_good]] = True
        embedding_found = np.zeros(count, bool)
        gates = []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if eligible:
            embeddings = np.zeros((len(eligible), 64), np.float32)
            with h5py.File(self.string_path, "r") as stream:
                for k, i in enumerate(eligible):
                    if targets[i] in stream:
                        value = np.asarray(stream[str(targets[i])][()])
                        if value.shape != (64,) or not np.isfinite(value).all():
                            raise ValueError(f"Invalid STRING vector for {targets[i]}")
                        embeddings[k] = value
                        embedding_found[i] = True
            present = embedding_found[eligible]
            embeddings[present] = (embeddings[present] - self.params["string_mean"]) / self.params["string_std"]
            # The historical normalized effect was float32, followed by a
            # float64 support shrink before the SVD projection. Preserve that
            # order rather than rounding the shrunken effect in place.
            x = normalized_effect[eligible] * (support[eligible] / (support[eligible] + SUPPORT_PRIOR))[:, None]
            latent = ((x - self.params["center"]) @ self.params["basis"].T) / self.params["scale"]
            z = np.c_[latent, embeddings, np.log1p(support[eligible]) / 10.0,
                      np.sqrt(np.mean(x * x, axis=1))].astype(np.float32)
            if not np.isfinite(z).all():
                raise ValueError("Non-finite source target features")
            for start in range(0, len(eligible), batch_size):
                ids = eligible[start:start + batch_size]
                length = len(ids)
                own = np.broadcast_to(query_known, (length, GENES)).astype(np.float32).copy()
                for k, i in enumerate(ids):
                    if targets[i] in self.model_genes:
                        own[k, self.model_genes[targets[i]]] = 0.0
                arrays = {"z": z[start:start + length],
                          "g": np.tile(g_one, (length, 1)),
                          "h": np.tile(h_one, (length, 1)),
                          "seen": self.params["seen"], "ownmask": own}
                tensors = {k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in arrays.items()}
                with torch.inference_mode():
                    result, gate = self.model(**tensors)
                values = result.cpu().numpy()
                if not np.isfinite(values).all() or np.max(np.abs(values)) > 0.750001:
                    raise ValueError("Invalid or unbounded frozen-model residual")
                residual[ids] = values
                applied[ids] = own.astype(bool) & self.params["seen"].astype(bool)[None, :]
                gates.extend(gate.cpu().numpy().ravel().tolist())
        effects = base + residual
        effects[:, ~query_known] = 0.0
        base[:, ~query_known] = 0.0
        metadata = {
            "model": "legacy_v28_reconstructed", "bit_exact_reproduction": False,
            "effect_space": "raw_pseudocount_log2fc", "pseudocount": PSEUDOCOUNT,
            "formula": "calibrated K562 PC0.5 effect + frozen V7 bounded residual",
            "calibration": CALIBRATION, "source_min_cells": MIN_SOURCE_CELLS,
            "support_prior_cells": SUPPORT_PRIOR, "base_effect_clip": [-3.0, 3.0],
            "input_source_sha256": self.source.source_sha256,
            "frozen_features_sha256": FEATURES_SHA256,
            "frozen_string_sha256": STRING_SHA256,
            "model_gene_count": GENES, "normalization_core_gene_count": CORE_GENES,
            "requested_targets": count, "neural_supported_targets": len(eligible),
            "source_absent_targets": [str(t) for t in targets if t not in self.source_targets],
            "source_low_support_targets": [str(t) for t, n in zip(targets, support) if t in self.source_targets and n < MIN_SOURCE_CELLS],
            "source_incomplete_core_targets": missing_core_targets,
            "query_missing_model_genes": int((~query_known).sum()),
            "source_control_profile_rmse_vs_saved": float(np.sqrt(np.mean(source_ctrl_error[source_ctrl_measured] ** 2))),
            "source_control_profile_max_abs_vs_saved": float(np.max(np.abs(source_ctrl_error[source_ctrl_measured]))),
            "string_missing_supported_targets": [str(targets[i]) for i in eligible if not embedding_found[i]],
            "mean_gate": float(np.mean(gates)) if gates else None,
            "residual_rms": float(np.sqrt(np.mean(residual.astype(np.float64) ** 2))),
            "query_controls_used_for_background_only": True,
            "count_generator_performed": False,
            **self.model_info,
        }
        return EffectPrediction(targets, self.genes.copy(), effects, base, residual,
                                source_known, query_known, applied, support, metadata)


def smoke_check_assets(checkpoint: str | Path, features: str | Path,
                       device: str = "cpu") -> dict[str, Any]:
    """Load actual frozen weights and check finite, bounded, masked inference."""
    import torch

    params = load_features(features)
    model, info = load_model(checkpoint, device)
    own = torch.ones((2, GENES), device=info["device"])
    own[0, 0] = 0
    own[1, 1] = 0
    with torch.inference_mode():
        residual, gate = model(
            z=torch.zeros((2, 130), device=info["device"]),
            g=torch.zeros((2, GENES), device=info["device"]),
            h=torch.zeros((2, 64), device=info["device"]),
            seen=torch.as_tensor(params["seen"], device=info["device"]), ownmask=own)
    values = residual.cpu().numpy()
    assert values.shape == (2, GENES) and np.isfinite(values).all()
    assert np.max(np.abs(values)) <= 0.750001
    assert values[0, 0] == values[1, 1] == 0
    assert np.count_nonzero(values[:, params["seen"] == 0]) == 0
    assert torch.isfinite(gate).all() and torch.all((gate > 0) & (gate < 1))
    return {"ok": True, "scope": "actual frozen-weight smoke inference, not scientific scoring",
            "shape": list(values.shape), "max_abs_residual": float(np.max(np.abs(values))),
            "own_gene_mask_pass": True, "unseen_gene_mask_pass": True, **info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="Check released model weights; no data download")
    smoke.add_argument("--checkpoint", required=True)
    smoke.add_argument("--features", required=True)
    smoke.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "smoke":
        print(json.dumps(smoke_check_assets(args.checkpoint, args.features, args.device), indent=2))


if __name__ == "__main__":
    main()
