"""Nested target-group and destination-context selection for K562 transfer.

Training is in the pseudobulk representation used by the VCC2026 PDS metric.
The shared gene restriction makes these development metrics local proxies,
not competition scores. H1 outcomes are not read by this script.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

SOURCE_IDS = {"k562": "replogle_k562_gwps", "hepg2": "nadig_hepg2",
              "jurkat": "nadig_jurkat", "rpe1": "replogle_rpe1"}
LINES = ["hepg2", "jurkat", "rpe1"]
TS = 50000.0
SEED = 20260927


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split_id(target):
    return int(hashlib.sha256(("vcc-v2:" + str(target)).encode()).hexdigest(), 16) % 5


def profile(mean):
    x = np.asarray(mean, dtype=np.float64)
    return np.log1p(TS * x / x.sum(axis=-1, keepdims=True)).astype(np.float32)


def project(effect, control):
    """Normalize feasible pseudobulk predictions, preserving the control library size."""
    p = np.maximum(np.expm1(np.clip(profile(control)[None, :] + effect, 0, 15)), 0)
    total = p.sum(axis=1, keepdims=True)
    p = np.divide(p, total, out=np.tile(control / control.sum(), (len(p), 1)), where=total > 0)
    counts = p * control.sum()
    return (profile(counts) - profile(control)).astype(np.float32), counts.astype(np.float32)


@dataclass
class Pair:
    line: str
    targets: np.ndarray
    genes: np.ndarray
    x: np.ndarray
    y: np.ndarray
    raw_x: np.ndarray
    raw_y: np.ndarray
    control: np.ndarray
    source_control: np.ndarray
    n_source: np.ndarray
    n_truth: np.ndarray

    def take(self, keep):
        return Pair(self.line, self.targets[keep], self.genes, self.x[keep], self.y[keep],
                    self.raw_x[keep], self.raw_y[keep], self.control, self.source_control,
                    self.n_source[keep], self.n_truth[keep])


def read_source(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ["target_names", "gene_names", "target_mean_counts",
                                             "control_mean_counts", "target_n_cells"]}


def make_pair(source, dest, line, genes, minimum=30):
    sp = {g: i for i, g in enumerate(source["gene_names"])}
    dp = {g: i for i, g in enumerate(dest["gene_names"])}
    sg = np.array([sp[g] for g in genes]); dg = np.array([dp[g] for g in genes])
    st = {t: i for i, t in enumerate(source["target_names"])}
    dt = {t: i for i, t in enumerate(dest["target_names"])}
    targets = np.array(sorted(t for t in set(st) & set(dt)
                             if source["target_n_cells"][st[t]] >= minimum
                             and dest["target_n_cells"][dt[t]] >= minimum), dtype=str)
    si = np.array([st[t] for t in targets]); di = np.array([dt[t] for t in targets])
    sm = source["target_mean_counts"][si][:, sg]
    dm = dest["target_mean_counts"][di][:, dg]
    sc = source["control_mean_counts"][sg]; dc = dest["control_mean_counts"][dg]
    support = source["target_n_cells"][si]
    rel = (support / (support + 50.0)).astype(np.float32)
    x = (profile(sm) - profile(sc)) * rel[:, None]
    y = profile(dm) - profile(dc)
    rx = np.log2((sm + .5) / (sc + .5)) * rel[:, None]
    ry = np.log2((dm + .5) / (dc + .5))
    gp = {g: i for i, g in enumerate(genes)}
    for i, t in enumerate(targets):
        if t in gp:
            for a in [x, y, rx, ry]:
                a[i, gp[t]] = 0.0
    return Pair(line, targets, genes, x, y, rx, ry, dc, sc, support, dest["target_n_cells"][di])


def expression_gate(source_control, dest_control):
    # Sensitivity d log(1 + mu) / d log(mu), at each context's baseline.
    s = -np.expm1(-profile(source_control))
    d = -np.expm1(-profile(dest_control))
    return np.clip(d / np.maximum(s, .05), 0, 2).astype(np.float32)


def fit_suite(pairs):
    x = np.concatenate([p.x for p in pairs]).astype(np.float64)
    y = np.concatenate([p.y for p in pairs]).astype(np.float64)
    # Contexts receive equal total weight; individual target cells are not replicates.
    w = np.concatenate([np.full(len(p.x), len(x) / (len(pairs) * len(p.x))) for p in pairs])
    sw = np.sqrt(w)[:, None]
    xx = np.sum(w[:, None] * x * x, axis=0)
    xy = np.sum(w[:, None] * x * y, axis=0)
    alpha = float(np.clip(xy.sum() / max(xx.sum(), 1e-12), 0, 2))
    rho = float(np.median(xx[xx > 0]))
    slope = np.clip((xy + rho * alpha) / (xx + rho), 0, 2).astype(np.float32)
    mx = np.average(x, axis=0, weights=w); my = np.average(y, axis=0, weights=w)
    xc = x - mx; yc = y - my
    xxc = np.sum(w[:, None] * xc * xc, axis=0)
    xyc = np.sum(w[:, None] * xc * yc, axis=0)
    cslope = np.clip((xyc + rho * alpha) / (xxc + rho), 0, 2).astype(np.float32)
    gate_x = np.concatenate([p.x * expression_gate(p.source_control, p.control) for p in pairs])
    gate_alpha = float(np.clip(np.sum(w[:, None] * gate_x * y) /
                               np.sum(w[:, None] * gate_x * gate_x), 0, 2))
    rx = np.concatenate([p.raw_x for p in pairs]); ry = np.concatenate([p.raw_y for p in pairs])
    raw_alpha = float(np.clip(np.sum(w[:, None] * rx * ry) / np.sum(w[:, None] * rx * rx), 0, 1.5))
    # Fit the input basis using training rows only; never the test targets or context.
    _, singular, vt = randomized_svd((xc * sw).astype(np.float32), n_components=128,
                                    n_iter=5, random_state=SEED)
    suite = {"alpha": alpha, "raw_alpha": raw_alpha, "gate_alpha": gate_alpha,
             "slope": slope, "centered_slope": cslope,
             "mean_x": mx.astype(np.float32), "mean_y": my.astype(np.float32),
             "basis": vt.astype(np.float32), "singular": singular.astype(np.float32)}
    for rank, penalty in [(32, 10), (64, 100), (128, 100)]:
        z = xc @ vt[:rank].T
        scale = np.sqrt(np.average(z * z, axis=0, weights=w))
        scale = np.maximum(scale, 1e-5)
        z /= scale
        coef = np.linalg.solve((z * sw).T @ (z * sw) + penalty * np.eye(rank),
                               (z * sw).T @ (yc * sw))
        key = f"lowrank_{rank}_a{penalty}"
        suite[key + "_coef"] = coef.astype(np.float32)
        suite[key + "_scale"] = scale.astype(np.float32)
    suite["training_targets"] = np.array(sorted(set(np.concatenate([p.targets for p in pairs]))), dtype=str)
    suite["training_lines"] = np.array([p.line for p in pairs], dtype=str)
    return suite


def predict_suite(suite, pair):
    x = pair.x
    raw_counts = np.maximum((pair.control + .5) * np.exp2(np.clip(suite["raw_alpha"] * pair.raw_x, -3, 3)) - .5, 0)
    predictions = {"zero": np.zeros_like(x),
                   "v1_raw_scalar": profile(raw_counts) - profile(pair.control),
                   "mean_response": np.tile(suite["mean_y"], (len(x), 1)),
                   "norm_scalar": suite["alpha"] * x,
                   "expression_gate": suite["gate_alpha"] * x * expression_gate(pair.source_control, pair.control),
                   "gene_shrink": x * suite["slope"],
                   "gene_centered": (x - suite["mean_x"]) * suite["centered_slope"] + suite["mean_y"]}
    for rank, penalty in [(32, 10), (64, 100), (128, 100)]:
        key = f"lowrank_{rank}_a{penalty}"
        z = (x - suite["mean_x"]) @ suite["basis"][:rank].T
        predictions[key] = (z / suite[key + "_scale"]) @ suite[key + "_coef"] + suite["mean_y"]
    predictions["hybrid_64"] = .5 * (predictions["gene_shrink"] + predictions["lowrank_64_a100"])
    predictions["hybrid_128"] = .5 * (predictions["gene_shrink"] + predictions["lowrank_128_a100"])
    return {k: project(v, pair.control)[0] if k != "zero" else v for k, v in predictions.items()}


def evaluate(pred, pair, pds=True):
    y = pair.y.copy().astype(np.float64); pred = pred.copy().astype(np.float64)
    mask = np.ones(y.shape, dtype=bool); gp = {g: i for i, g in enumerate(pair.genes)}
    for i, t in enumerate(pair.targets):
        if t in gp:
            mask[i, gp[t]] = False
    n = mask.sum(axis=1)
    y[~mask] = 0; pred[~mask] = 0
    mse = np.sum((pred - y) ** 2, axis=1) / n
    zero = np.sum(y ** 2, axis=1) / n
    def cosine(a, b):
        d = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return np.divide(np.sum(a*b, axis=1), d, out=np.full(len(a), np.nan), where=d > 1e-12)
    cos = cosine(pred, y)
    pc = pred - (pred.sum(axis=1) / n)[:, None]
    yc = y - (y.sum(axis=1) / n)[:, None]
    pc[~mask] = 0; yc[~mask] = 0
    cor = cosine(pc, yc)
    summary = {"n_targets": len(y), "n_genes": y.shape[1], "mse": float(mse.mean()),
               "zero_mse": float(zero.mean()), "relative_mse": float(mse.mean()/zero.mean()),
               "pearson": float(np.nanmean(cor)) if np.isfinite(cor).any() else None,
               "cosine": float(np.nanmean(cos)) if np.isfinite(cos).any() else None}
    ranks = np.full(len(y), np.nan)
    if pds and len(y) > 1:
        keep = ~np.isin(pair.genes, pair.targets)
        a = pred[:, keep]; b = y[:, keep]
        a /= np.maximum(np.linalg.norm(a, axis=1)[:, None], 1e-12)
        b /= np.maximum(np.linalg.norm(b, axis=1)[:, None], 1e-12)
        sim = a @ b.T
        own = np.diag(sim)[:, None]
        # Tie tolerance handles exact zero/shared-profile numerical ties only.
        higher = (sim > own + 1e-10).sum(axis=1)
        tie = (np.abs(sim-own) <= 1e-10).sum(axis=1)
        ranks = 1 - (higher + .5*(tie-1))/(len(y)-1)
        summary["pds_proxy"] = float(ranks.mean())
    table = pd.DataFrame({"target": pair.targets, "mse": mse, "zero_mse": zero,
                          "pearson": cor, "cosine": cos, "pds_proxy": ranks,
                          "source_n": pair.n_source, "truth_n": pair.n_truth})
    return summary, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("runs/model_selection_v2"))
    args = ap.parse_args(); root = args.root.resolve(); out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    srcdir = root / "data/processed/v1/external_pseudobulk"
    raw = {line: read_source(srcdir / f"{sid}.npz") for line, sid in SOURCE_IDS.items()}
    official = np.loadtxt(root / "data/raw/vcc2026/validation/files/gene_names.csv", dtype=str, skiprows=1)
    shared = set.intersection(*(set(r["gene_names"]) for r in raw.values()))
    genes = np.array([g for g in official if g in shared])
    pairs = {line: make_pair(raw["k562"], raw[line], line, genes) for line in LINES}
    splits = pd.concat([pd.DataFrame({"line": line, "target": p.targets,
                                      "split": [split_id(t) for t in p.targets]}) for line,p in pairs.items()])
    splits.to_csv(out / "target_split_manifest.csv", index=False)
    np.savetxt(out / "gene_axis.txt", genes, fmt="%s")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "official_score": False,
              "protocol_sha256": sha256(root / "configs/model_selection_v2_protocol.json"),
              "n_shared_genes": len(genes), "folds": {}, "input_hashes": {
                  line: sha256(srcdir / f"{sid}.npz") for line,sid in SOURCE_IDS.items()}}
    all_outer = []; all_inner = []
    # Each single-line inner fit is reused for both outer folds where it is allowed.
    inner_cache = {}
    for train_line in LINES:
        p = pairs[train_line]
        tr = p.take(np.array([split_id(t) >= 2 for t in p.targets]))
        print(f"inner fit {train_line}: {len(tr.targets)} targets", flush=True)
        suite = fit_suite([tr])
        for val_line in LINES:
            if val_line == train_line:
                continue
            val = pairs[val_line].take(np.array([split_id(t) == 1 for t in pairs[val_line].targets]))
            assert not set(tr.targets) & set(val.targets)
            preds = predict_suite(suite, val)
            scores = {m: evaluate(pred,val,pds=False)[0] for m,pred in preds.items()}
            inner_cache[(train_line,val_line)] = scores
            for m,sc in scores.items():
                all_inner.append({"train_context": train_line,"validation_context":val_line,"model":m,**sc})
    pd.DataFrame(all_inner).to_csv(out/"inner_selection.csv",index=False)
    for heldout in LINES:
        other = [line for line in LINES if line != heldout]
        scores = {m: np.mean([inner_cache[(other[0],other[1])][m]["relative_mse"],
                              inner_cache[(other[1],other[0])][m]["relative_mse"]])
                  for m in inner_cache[(other[0],other[1])]}
        selected = min(scores, key=scores.get)
        train = [pairs[line].take(np.array([split_id(t) != 0 for t in pairs[line].targets])) for line in other]
        test = pairs[heldout].take(np.array([split_id(t) == 0 for t in pairs[heldout].targets]))
        assert not set(test.targets) & set(np.concatenate([p.targets for p in train]))
        print(f"outer fit heldout={heldout}: selected={selected}, test={len(test.targets)}", flush=True)
        suite = fit_suite(train)
        preds = predict_suite(suite, test)
        fold = {"selected_from_inner": selected, "inner_relative_mse": scores,
                "training_contexts": other, "training_target_rows":sum(len(p.targets) for p in train),
                "test_targets":len(test.targets), "models":{}}
        for m,pred in preds.items():
            sc,df = evaluate(pred,test)
            fold["models"][m] = sc
            df.to_csv(out/f"outer_{heldout}__{m}.csv",index=False)
            all_outer.append({"heldout_context":heldout,"model":m,"nested_selected":m==selected,**sc})
        np.savez_compressed(out/f"outer_{heldout}__selected.npz",target_symbols=test.targets,
                            gene_symbols=genes,effect_bulk=preds[selected],control=test.control)
        np.savez_compressed(out/f"outer_{heldout}__suite.npz",**suite)
        report["folds"][heldout] = fold
    pd.DataFrame(all_outer).to_csv(out/"outer_results.csv",index=False)
    final_selection = pd.DataFrame(all_inner).groupby("model")["relative_mse"].mean().sort_values()
    final_model = str(final_selection.index[0])
    report["final_selection_inner_only"] = final_selection.to_dict()
    report["final_model"] = final_model
    print(f"final model before H1 inspection: {final_model}",flush=True)
    suite = fit_suite(list(pairs.values()))
    np.savez_compressed(out/"final_suite.npz", **suite)
    report["final_suite_sha256"] = sha256(out/"final_suite.npz")
    report["h1_used_for_selection_or_training"] = False
    (out/"selection.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(pd.DataFrame(all_outer)[["heldout_context","model","relative_mse","pearson","pds_proxy","nested_selected"]].to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
