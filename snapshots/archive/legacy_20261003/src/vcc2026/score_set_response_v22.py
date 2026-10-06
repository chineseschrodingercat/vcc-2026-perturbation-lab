"""Score V22 total raw log2 effects with the unchanged V6 count/metric engines.

Run this launcher in foundation312 (or vcc313). Official scoring always runs in
the project's pinned vcc313 interpreter. All writes stay under
runs/set_response_v22; historical panels, real caches, and references are inputs.

Example:
  python src/vcc2026/score_set_response_v22.py --effects prediction.npz \
    --role outer_hepg2__inner_rpe1 --output runs/set_response_v22/scores/example \
    --count-seed 20260928

An explicit --panel can replace role-based historical-panel resolution.
NPZ must contain targets, genes, and effect/effects/prediction: TOTAL raw log2FC
on the historical 10,916-gene model axis, not a residual or normalized profile.
The supplied total output is consumed directly; this scorer never adds V1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "runs" / "set_response_v22"
HISTORY = PROJECT / "runs" / "residual_v17"
SCORER = PROJECT / "src" / "vcc2026" / "score_residual_v6.py"
SCORE_PYTHON = PROJECT / "envs" / "vcc313" / "Scripts" / "python.exe"
SEED = 20260928
CONTEXTS = ("hepg2", "jurkat", "rpe1")
METRICS = (
    "pds_cosine", "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else PROJECT / path).resolve()


def output_path(path: str | Path) -> Path:
    result = absolute(path)
    if result == ROOT or not result.is_relative_to(ROOT):
        raise ValueError(f"Output must be a new child directory of {ROOT}: {result}")
    return result


def parse_role(role: str | None) -> tuple[str, str | None] | None:
    if role is None:
        return None
    match = re.fullmatch(r"outer_(hepg2|jurkat|rpe1)(?:__inner_(hepg2|jurkat|rpe1))?", role)
    if not match:
        raise ValueError(f"Unknown whole-background evaluation role: {role}")
    outer, inner = match.groups()
    if outer == inner:
        raise ValueError("Outer and inner held-out backgrounds must differ")
    return outer, inner


def resolve_panel(role: str, count_seed: int = SEED, teacher_train_seed: int = SEED) -> Path:
    """Resolve V13-reference panels retained in the immutable V17 run."""
    outer, inner = parse_role(role)
    if inner:
        return HISTORY / "inner_panels" / inner
    return (HISTORY / "outer_panels" / outer /
            f"train_seed_{teacher_train_seed}" / f"seed_{count_seed}")


def identifiers(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional identifier array")
    result = np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=str)
    if any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} contains empty or duplicate identifiers")
    return result


def load_effects(path: Path, panel: Path, role: str | None = None) -> tuple[np.ndarray, dict]:
    info = read_json(panel / "panel.json")
    modeled = np.loadtxt(PROJECT / "runs/context_v4/gene_axis.txt", dtype=str, ndmin=1)
    with np.load(path, allow_pickle=False) as data:
        if "targets" not in data or "genes" not in data:
            raise ValueError("Prediction NPZ requires explicit targets and genes")
        targets = identifiers(data["targets"], "targets")
        genes = identifiers(data["genes"], "genes")
        keys = [key for key in ("effect", "effects", "prediction") if key in data]
        if not keys:
            raise ValueError("Missing total raw log2FC: expected effect, effects, or prediction")
        effect = np.asarray(data[keys[0]])
        if any(not np.array_equal(effect, data[key]) for key in keys[1:]):
            raise ValueError("Effect aliases disagree; refusing to guess which is total log2FC")
    if not np.array_equal(genes, modeled):
        raise ValueError("genes must exactly match the historical 10,916-gene model axis")
    if effect.shape != (len(targets), len(genes)):
        raise ValueError(f"Effect shape {effect.shape} disagrees with explicit identifier axes")
    if not np.issubdtype(effect.dtype, np.number) or not np.isrealobj(effect) or not np.isfinite(effect).all():
        raise ValueError("Effects must be finite real numeric values")
    wanted = np.asarray(info["targets"], dtype=str)
    if set(targets) != set(wanted):
        missing = sorted(set(wanted) - set(targets))
        extra = sorted(set(targets) - set(wanted))
        raise ValueError(f"Target identity mismatch: missing={missing}, extra={extra}")
    positions = {target: index for index, target in enumerate(targets)}
    order = np.asarray([positions[target] for target in wanted], dtype=np.int64)
    reordered = not np.array_equal(targets, wanted)
    original_dtype = str(effect.dtype)
    effect = np.asarray(effect[order], dtype=np.float32)
    if not np.isfinite(effect).all():
        raise ValueError("Effects are nonfinite after the historical float32 conversion")
    parsed = parse_role(role)
    if parsed:
        outer, inner = parsed
        if info["context"] != (inner or outer):
            raise ValueError("Evaluation role and panel background disagree")
        # This verifies target-fold identity; it does not inspect outcome values.
        from residual_v6_common import fold
        expected_fold = 1 if inner else 0
        if any(fold(target) != expected_fold for target in wanted):
            raise ValueError(f"Panel targets do not all belong to expected held-out fold {expected_fold}")
    return effect, {
        "context": info["context"], "targets": wanted.tolist(), "genes": len(genes),
        "shape": list(effect.shape), "effect_key": keys[0],
        "semantics": "total raw log2 pseudocount effect; consumed directly with no V1 addition",
        "source_dtype": original_dtype, "generator_dtype": "float32",
        "target_rows_reordered_by_exact_identifier": reordered,
        "effect_min": float(effect.min()), "effect_max": float(effect.max()),
    }


def copy_immutable(source: Path, destination: Path, hardlink: bool = False) -> None:
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Existing V22 input differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def prepare_panel(source: Path, destination: Path) -> None:
    """Link read-only count inputs; COPY mutable caches so old results stay intact."""
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("real.h5ad", "generation_pool.npz", "genes.txt", "panel.json"):
        copy_immutable(source / filename, destination / filename, hardlink=True)
    for item in (source / "references").rglob("*"):
        if item.is_file():
            copy_immutable(item, destination / "references" / item.relative_to(source / "references"))
    cache_source = source / "cache_real"
    if source.name.startswith("seed_"):
        shared = source.parent / f"seed_{SEED}" / "cache_real"
        if shared.exists():
            cache_source = shared
    if cache_source.exists() and not (destination / "cache_real").exists():
        shutil.copytree(cache_source, destination / "cache_real")


def teacher_tag(role: str | None, panel: Path, override: str | None = None) -> str | None:
    if override:
        return override
    parsed = parse_role(role)
    if parsed:
        return f"outer_{parsed[0]}__v13_ref" if parsed[1] else "v13_ref"
    return "v13_ref" if (panel / "scores/v13_ref/summary.json").exists() else None


def score_models(panel: Path, models: list[str], log: Path, threads: int) -> None:
    env = os.environ.copy()
    env.update(VCC_SCORE_THREADS=str(threads), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    check = subprocess.run(
        [str(SCORE_PYTHON), "-c", "import cell_eval2; print(cell_eval2.__version__)"],
        cwd=PROJECT, env=env, capture_output=True, text=True, check=True,
    )
    expected = read_json(panel / "references/reference.json")["cell_eval2"]
    if check.stdout.strip() != expected or expected != "0.16.0":
        raise RuntimeError(f"Pinned cell_eval2 version mismatch: {check.stdout.strip()} versus {expected}")
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            [str(SCORE_PYTHON), str(SCORER), "--path", str(panel), "--models", *models],
            cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True,
        )


def evaluate(args: argparse.Namespace) -> dict:
    effect_path = absolute(args.effects)
    output = output_path(args.output)
    panel = absolute(args.panel) if args.panel else resolve_panel(
        args.role, args.count_seed, args.teacher_train_seed)
    if panel.is_relative_to(ROOT):
        raise ValueError("Historical --panel must remain outside the V22 output tree")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.model) or args.model in (".", "..", "v13_ref"):
        raise ValueError("Use a safe unique model tag; v13_ref is reserved for the teacher")
    effect, validation = load_effects(effect_path, panel, args.role)
    reference_path = panel / "references/reference.json"
    reference = read_json(reference_path)
    real_hash = sha256(panel / "real.h5ad")
    if reference["real_sha256"] != real_hash or not reference["replicate_usable"]:
        raise ValueError("Historical panel references are inconsistent or unusable")
    tag = teacher_tag(args.role, panel, args.teacher_tag) if args.teacher != "none" else None
    teacher_summary = panel / "scores" / tag / "summary.json" if tag else None
    teacher_effect = panel / f"effects__{tag}.npz" if tag else None
    if tag and (not teacher_summary.exists() or not teacher_effect.exists()):
        raise FileNotFoundError(f"Missing matched V13 teacher artifacts for {panel}: {tag}")
    lock = {
        "effect_path": str(effect_path), "effect_sha256": sha256(effect_path),
        "historical_panel": str(panel), "role": args.role, "model": args.model,
        "count_seed": args.count_seed, "teacher_train_seed": args.teacher_train_seed,
        "real_sha256": real_hash, "panel_sha256": sha256(panel / "panel.json"),
        "generation_pool_sha256": sha256(panel / "generation_pool.npz"),
        "reference_sha256": sha256(reference_path),
        "model_axis_sha256": sha256(PROJECT / "runs/context_v4/gene_axis.txt"),
        "teacher_tag": tag,
        "teacher_effect_sha256": sha256(teacher_effect) if tag else None,
        "score_engine_sha256": sha256(SCORER),
        "generator_sha256": sha256(PROJECT / "src/vcc2026/generate_counts.py"),
        "shared_generator_sha256": sha256(PROJECT / "src/vcc2026/residual_v6_common.py"),
        "cell_eval2": reference["cell_eval2"], "validation": validation,
    }
    if args.validate_only:
        return {"validated": True, "writes_performed": False, **lock}
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "input_lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise RuntimeError("Existing output belongs to different inputs; choose a new V22 output directory")
    if not lock_path.exists():
        write_json(lock_path, lock)
    work = output / "panel"  # Never named seed_*: original scorer's cache rule stays local.
    prepare_panel(panel, work)
    if args.prepare_only:
        return {"prepared": True, "counts_generated": False, "output": str(output), "panel": str(work)}
    if shutil.disk_usage(output).free < 75 * 2**30:
        raise RuntimeError("At least 75 GiB free space required by historical count-generation policy")
    from residual_v6_common import generate
    candidate = work / f"pred__{args.model}.h5ad"
    if not candidate.exists():
        generate(work, args.model, effect, args.count_seed)
    names = [args.model]
    source_seed = int(panel.name.removeprefix("seed_")) if panel.name.startswith("seed_") else SEED
    if tag and not (work / "scores/v13_ref/summary.json").exists():
        if source_seed == args.count_seed:
            shutil.copytree(panel / "scores" / tag, work / "scores/v13_ref")
            record = read_json(work / "scores/v13_ref/summary.json")
            record["model"] = "v13_ref"
            write_json(work / "scores/v13_ref/summary.json", record)
            write_json(work / "scores/v13_ref/v22_reuse.json", {
                "source": str(panel / "scores" / tag), "same_real_sha256": real_hash,
                "same_count_seed": source_seed, "same_reference_sha256": lock["reference_sha256"],
                "teacher_effect_sha256": lock["teacher_effect_sha256"],
            })
        else:
            with np.load(teacher_effect, allow_pickle=False) as data:
                teacher = np.asarray(data["effect"], dtype=np.float32)
            if teacher.shape != effect.shape or not np.isfinite(teacher).all():
                raise ValueError("Historical teacher effects have an incompatible shape")
            if not (work / "pred__v13_ref.h5ad").exists():
                generate(work, "v13_ref", teacher, args.count_seed)
            names.append("v13_ref")
    score_models(work, names, output / "score.log", args.threads)
    result_dir = work / "scores" / args.model
    summary = read_json(result_dir / "summary.json")
    if set(METRICS) - set(summary) or summary["replicate_overall"] is None:
        raise RuntimeError("Scoring did not produce all six raw metrics and normalized overall")
    for filename in ("summary.json", "overall_components.csv", "fid_decomposition.csv",
                     "aggregate.csv", "metrics.csv"):
        shutil.copy2(result_dir / filename, output / filename)
    evaluation = {
        "completed": True, "output": str(output), "prediction": str(candidate),
        "summary": summary, "input_lock": str(lock_path),
        "score_log": str(output / "score.log"), "generator_changed": False,
        "historical_outputs_modified": False,
    }
    if tag:
        teacher_record = read_json(work / "scores/v13_ref/summary.json")
        evaluation["v13_teacher"] = teacher_record
        evaluation["delta_raw"] = {metric: summary[metric] - teacher_record[metric] for metric in METRICS}
        evaluation["delta_replicate_overall"] = summary["replicate_overall"] - teacher_record["replicate_overall"]
        write_json(output / "teacher_summary.json", teacher_record)
    write_json(output / "evaluation.json", evaluation)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effects", "--effect-npz", dest="effects", required=True)
    parser.add_argument("--panel", "--panel-path", dest="panel")
    parser.add_argument("--role", help="outer_x__inner_y or outer_x; required when --panel is omitted")
    parser.add_argument("--output", "--output-dir", dest="output", required=True)
    parser.add_argument("--count-seed", "--countseed", dest="count_seed", type=int, default=SEED)
    parser.add_argument("--teacher-train-seed", type=int, default=SEED)
    parser.add_argument("--model", default="candidate")
    parser.add_argument("--teacher", choices=("auto", "none"), default="auto")
    parser.add_argument("--teacher-tag")
    parser.add_argument("--threads", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true", help="Read and validate; no writes or count generation")
    modes.add_argument("--prepare-only", action="store_true", help="Prepare isolated panel/cache/references; no predictions")
    args = parser.parse_args()
    if not args.panel and not args.role:
        parser.error("Provide --panel or --role")
    if args.threads < 1 or args.count_seed < 0:
        parser.error("threads must be positive and count-seed must be nonnegative")
    # Shared historical helpers use project-relative paths.
    os.chdir(PROJECT)
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
