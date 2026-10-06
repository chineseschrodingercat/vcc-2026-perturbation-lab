"""Registry validation and deterministic packaging, using only the standard library.

These helpers inspect saved files. They never run a model, query a leaderboard,
download data, or infer that a score difference is a stable improvement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
from typing import Any
import zipfile


class ArchiveError(ValueError):
    """The registry, evidence, or requested operation is inconsistent."""


_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
_EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    "raw", "data", "checkpoints", "artifacts", "build", "dist",
}
_EXCLUDED_SUFFIXES = {
    ".npy", ".npz", ".h5", ".hdf5", ".h5ad", ".pt", ".pth", ".ckpt",
    ".pkl", ".pickle", ".pyc", ".pyo", ".gz", ".zip", ".tar", ".7z",
}
_COMMON_DIRS = ("docs", "schemas", "schema", "src", "tests", ".github", "scripts")
_COMMON_STATIC_PATHS = ("assets/model-assets.json", "provenance/original-files.json")
_COMMON_FILES = {
    "pyproject.toml", ".gitignore", ".gitattributes", "Makefile",
    "CITATION.cff", "LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE",
}
SCORE_METRICS = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
)
_REPLAY_TOLERANCE = 1e-10


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a canonical portable relative path, refusing traversal and links."""
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ArchiveError("Asset path must be a non-empty string without NUL.")
    pieces = relative.split("/")
    if ("\\" in relative or ":" in relative or
            any(piece in {"", ".", ".."} for piece in pieces) or
            PureWindowsPath(relative).is_absolute() or PureWindowsPath(relative).drive):
        raise ArchiveError(f"Unsafe or non-canonical relative path: {relative!r}")
    root = root.resolve()
    result = root.joinpath(*pieces)
    try:
        result.resolve().relative_to(root)
    except ValueError as exc:
        raise ArchiveError(f"Asset escapes the configured root: {relative!r}") from exc
    current = root
    for piece in pieces:
        current = current / piece
        if current.is_symlink():
            raise ArchiveError(f"Symbolic links are not accepted as assets: {relative!r}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveError(f"JSON root must be an object: {path}")
    return value


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None and not value.endswith(".")


def _schema_one(value: dict[str, Any], label: str) -> None:
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ArchiveError(f"{label}: schema_version must be integer 1.")


def _asset_records(value: Any, root: Path, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ArchiveError(f"{label}: expected an array of assets.")
    seen: set[str] = set()
    for record in value:
        if not isinstance(record, dict):
            raise ArchiveError(f"{label}: each asset must be an object.")
        path = record.get("path")
        safe_path(root, path)
        if path.casefold() in seen:
            raise ArchiveError(f"{label}: duplicate path {path!r}.")
        seen.add(path.casefold())
        if not isinstance(record.get("sha256"), str) or not _SHA.fullmatch(record["sha256"]):
            raise ArchiveError(f"{label}: {path!r} needs a SHA256 digest.")
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise ArchiveError(f"{label}: {path!r} needs a non-negative integer byte count.")
        if not isinstance(record.get("role"), str) or not record["role"].strip():
            raise ArchiveError(f"{label}: {path!r} needs a non-empty role.")
    return value


def load_registry(repo: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate index/manifest identities and structure without opening assets."""
    repo = repo.resolve()
    index = _read_json(safe_path(repo, "milestones/index.json"))
    _schema_one(index, "index")
    entries = index.get("milestones")
    if not isinstance(entries, list):
        raise ArchiveError("index: milestones must be an array.")
    manifests: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not _valid_id(entry.get("id")):
            raise ArchiveError("index: each entry needs a portable lowercase milestone id.")
        milestone_id = entry["id"]
        if milestone_id in manifests:
            raise ArchiveError(f"index: duplicate milestone id {milestone_id!r}.")
        if not isinstance(entry.get("title"), str) or not entry["title"].strip():
            raise ArchiveError(f"index: {milestone_id!r} needs a title.")
        manifest_path = entry.get("manifest")
        path = safe_path(repo, manifest_path)
        if manifest_path in seen_paths:
            raise ArchiveError(f"index: duplicate manifest path {manifest_path!r}.")
        seen_paths.add(manifest_path)
        manifest = _read_json(path)
        _schema_one(manifest, milestone_id)
        if manifest.get("id") != milestone_id or manifest.get("title") != entry["title"]:
            raise ArchiveError(f"{milestone_id}: index and manifest identity/title disagree.")
        if not isinstance(manifest.get("evidence_scope"), str) or not manifest["evidence_scope"].strip():
            raise ArchiveError(f"{milestone_id}: evidence_scope must be a non-empty string.")
        protocol = manifest.get("score_protocol_id")
        if protocol is not None and (not isinstance(protocol, str) or not protocol.strip()):
            raise ArchiveError(f"{milestone_id}: score_protocol_id must be a string or null.")
        overall = manifest.get("overall")
        if overall is not None and (type(overall) not in (int, float) or not math.isfinite(overall)):
            raise ArchiveError(f"{milestone_id}: overall must be a finite number or null.")
        if "overall" not in manifest or "score_protocol_id" not in manifest:
            raise ArchiveError(f"{milestone_id}: overall and score_protocol_id are required, even if null.")
        _asset_records(manifest.get("files"), repo, f"{milestone_id}.files")
        # External paths are workspace-relative. A disposable root permits the
        # same lexical safety checks without opening the scientific files.
        _asset_records(manifest.get("external_assets"), repo, f"{milestone_id}.external_assets")
        manifests[milestone_id] = manifest
    return index, manifests


def select_milestones(manifests: dict[str, dict[str, Any]], ids: list[str] | None) -> list[str]:
    selected = list(manifests) if not ids else ids
    if len(selected) != len(set(selected)):
        raise ArchiveError("Milestone ids must not be repeated.")
    unknown = [milestone_id for milestone_id in selected if milestone_id not in manifests]
    if unknown:
        raise ArchiveError(f"Unknown milestone id(s): {', '.join(unknown)}")
    return selected


def check_asset(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = safe_path(root, record["path"])
    result: dict[str, Any] = {"path": record["path"], "role": record["role"]}
    if not path.is_file():
        return {**result, "status": "missing"}
    actual_bytes = path.stat().st_size
    if actual_bytes != record["bytes"]:
        return {**result, "status": "size_mismatch", "expected_bytes": record["bytes"], "actual_bytes": actual_bytes}
    actual_sha = sha256_file(path)
    if actual_sha.lower() != record["sha256"].lower():
        return {**result, "status": "hash_mismatch", "expected_sha256": record["sha256"].lower(), "actual_sha256": actual_sha}
    return {**result, "status": "ok", "bytes": actual_bytes, "sha256": actual_sha}


def verify(repo: Path, ids: list[str] | None = None) -> dict[str, Any]:
    _, manifests = load_registry(repo)
    selected = select_milestones(manifests, ids)
    results = {milestone_id: [check_asset(repo, record) for record in manifests[milestone_id]["files"]]
               for milestone_id in selected}
    return {"schema_version": 1, "operation": "verify", "ok": all(
        result["status"] == "ok" for items in results.values() for result in items),
        "milestones": results, "scope": "Bundled source and evidence only; no scientific rerun."}


def doctor(repo: Path, workspace: Path, ids: list[str] | None = None) -> dict[str, Any]:
    _, manifests = load_registry(repo)
    selected = select_milestones(manifests, ids)
    results = {milestone_id: [check_asset(workspace, record) for record in manifests[milestone_id]["external_assets"]]
               for milestone_id in selected}
    return {"schema_version": 1, "operation": "doctor", "ok": all(
        result["status"] == "ok" for items in results.values() for result in items),
        "milestones": results,
        "listed_assets_only": True, "full_pipeline_ready_not_certified": True,
        "checked_asset_count": sum(len(items) for items in results.values()),
        "scope": "Only listed external asset existence, byte count, and streaming SHA256 are checked; no RNA decoding or model execution. An empty or successful asset check does not certify full pipeline readiness."}


def compare(repo: Path, left: str, right: str) -> dict[str, Any]:
    _, manifests = load_registry(repo)
    select_milestones(manifests, [left, right])
    a, b = manifests[left], manifests[right]
    if a["score_protocol_id"] is None or b["score_protocol_id"] is None:
        raise ArchiveError("Comparison requires a known score protocol for both milestones.")
    if a["score_protocol_id"] != b["score_protocol_id"]:
        raise ArchiveError("Scores cannot be compared across different scoring protocols.")
    if a["evidence_scope"] != b["evidence_scope"]:
        raise ArchiveError("Scores cannot be compared across different evidence scopes.")
    if a["overall"] is None or b["overall"] is None:
        raise ArchiveError("Comparison requires an overall score for both milestones.")
    return {"schema_version": 1, "operation": "compare", "left": left, "right": right,
            "score_protocol_id": a["score_protocol_id"], "evidence_scope": a["evidence_scope"],
            "left_overall": a["overall"], "right_overall": b["overall"],
            "delta_right_minus_left": b["overall"] - a["overall"],
            "interpretation": "Descriptive saved-score comparison; this is not evidence of a stable promotion."}


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ArchiveError(f"{label}: expected a finite numeric value.")
    return float(value)


def replay_score(repo: Path, milestone_id: str) -> dict[str, Any]:
    """Replay only saved six-metric component arithmetic, never RNA scoring.

    Anchors, raw metrics, significance tables, and scaling transformations are
    not recomputed. The output establishes archived arithmetic consistency,
    rather than independently reproducing the scientific scoring pipeline.
    """
    _, manifests = load_registry(repo)
    select_milestones(manifests, [milestone_id])
    manifest = manifests[milestone_id]
    specification = manifest.get("score_replay")
    if not isinstance(specification, dict):
        raise ArchiveError(
            f"{milestone_id}: saved-component replay is unavailable; the archive does not contain a verified replay specification.")
    if not verify(repo, [milestone_id])["ok"]:
        raise ArchiveError("Bundled evidence verification failed; saved components were not replayed.")
    if specification.get("scale") != "replicate_scaled":
        raise ArchiveError("Saved-component replay requires the fixed replicate_scaled scale.")
    weights = specification.get("metric_weights")
    if not isinstance(weights, dict) or set(weights) != set(SCORE_METRICS):
        raise ArchiveError("Saved-component replay requires exactly the six unique official metrics.")
    weights = {metric: _finite_number(weights[metric], f"weight.{metric}") for metric in SCORE_METRICS}
    if any(abs(weight - 1 / 6) > 1e-12 for weight in weights.values()) or abs(math.fsum(weights.values()) - 1) > 1e-12:
        raise ArchiveError("Saved-component replay weights must sum to one and assign 1/6 to each metric.")
    seeds = specification.get("seeds")
    contexts = specification.get("contexts")
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ArchiveError("score_replay.seeds must explicitly declare unique integer seeds.")
    if (not isinstance(contexts, list) or len(contexts) != 3 or
            any(not isinstance(context, str) or not context.strip() for context in contexts) or
            len(set(contexts)) != 3):
        raise ArchiveError("score_replay.contexts must explicitly declare three distinct backgrounds.")
    saved_cases = specification.get("cases")
    if not isinstance(saved_cases, list) or len(saved_cases) != len(seeds) * 6:
        raise ArchiveError("Saved-component replay requires six cases per declared seed.")
    expected = _finite_number(specification.get("expected_overall"), "score_replay.expected_overall")
    if manifest["overall"] is None or abs(expected - manifest["overall"]) > _REPLAY_TOLERANCE:
        raise ArchiveError("Replay expected_overall and milestone overall disagree.")
    bundled = {record["path"] for record in manifest["files"]}
    seen_cases: set[tuple[int, str]] = set()
    role_contexts: dict[str, str] = {}
    seen_component_paths: set[str] = set()
    seen_summary_paths: set[str] = set()
    case_results: list[dict[str, Any]] = []
    largest_difference = 0.0

    def require_match(actual: float, saved: float, label: str) -> None:
        nonlocal largest_difference
        difference = abs(actual - saved)
        largest_difference = max(largest_difference, difference)
        if difference > _REPLAY_TOLERANCE:
            raise ArchiveError(f"{label}: saved arithmetic disagrees by {difference:.17g} (tolerance {_REPLAY_TOLERANCE:g}).")

    for case in saved_cases:
        if not isinstance(case, dict):
            raise ArchiveError("Each score replay case must be an object.")
        role, context, seed = case.get("role"), case.get("context"), case.get("seed")
        if not isinstance(role, str) or not role.strip() or context not in contexts or type(seed) is not int or seed not in seeds:
            raise ArchiveError("Replay case role, context, or seed is missing or outside the declared configuration.")
        key = (seed, role)
        if key in seen_cases:
            raise ArchiveError(f"Duplicate replay case: seed {seed}, role {role!r}.")
        seen_cases.add(key)
        if role in role_contexts and role_contexts[role] != context:
            raise ArchiveError(f"Replay role {role!r} changes background across seeds.")
        role_contexts[role] = context
        component_path, summary_path = case.get("components_path"), case.get("summary_path")
        for relative, seen, label in ((component_path, seen_component_paths, "component"),
                                       (summary_path, seen_summary_paths, "summary")):
            safe_path(repo, relative)
            if relative not in bundled:
                raise ArchiveError(f"Replay {label} file is not SHA-pinned in milestone files: {relative!r}")
            if relative in seen:
                raise ArchiveError(f"Replay {label} path is repeated: {relative!r}")
            seen.add(relative)
        rows: dict[str, dict[str, float]] = {}
        with safe_path(repo, component_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = {"metric", "raw", "fixed_scaled", "replicate_scaled", "baseline", "replicate"}
            if (reader.fieldnames is None or not columns.issubset(reader.fieldnames) or
                    len(reader.fieldnames) != len(set(reader.fieldnames))):
                raise ArchiveError(f"{component_path}: required saved-component columns are missing.")
            for row in reader:
                if None in row:
                    raise ArchiveError(f"{component_path}: CSV row has more fields than its header.")
                metric = row.get("metric")
                if metric not in SCORE_METRICS or metric in rows:
                    raise ArchiveError(f"{component_path}: unknown or repeated metric {metric!r}.")
                values: dict[str, float] = {}
                for column in columns - {"metric"}:
                    try:
                        value = float(row[column])
                    except (ValueError, TypeError) as exc:
                        raise ArchiveError(f"{component_path}: {metric}.{column} is missing or non-numeric.") from exc
                    if not math.isfinite(value):
                        raise ArchiveError(f"{component_path}: {metric}.{column} is non-finite.")
                    values[column] = value
                rows[metric] = values
        if set(rows) != set(SCORE_METRICS):
            raise ArchiveError(f"{component_path}: all six unique metrics are required.")
        summary = _read_json(safe_path(repo, summary_path))
        for metric in SCORE_METRICS:
            require_match(rows[metric]["raw"], _finite_number(summary.get(metric), f"{summary_path}.{metric}"),
                          f"{component_path} raw {metric}")
        weighted = {scale: math.fsum(weights[metric] * rows[metric][scale] for metric in SCORE_METRICS)
                    for scale in ("fixed_scaled", "replicate_scaled")}
        require_match(weighted["fixed_scaled"], _finite_number(summary.get("fixed_overall"), f"{summary_path}.fixed_overall"),
                      f"{summary_path} fixed_overall")
        require_match(weighted["replicate_scaled"], _finite_number(summary.get("replicate_overall"), f"{summary_path}.replicate_overall"),
                      f"{summary_path} replicate_overall")
        case_results.append({"role": role, "context": context, "seed": seed,
                             "overall": weighted["replicate_scaled"],
                             "fixed_overall": weighted["fixed_scaled"],
                             "six_scaled": {metric: rows[metric]["replicate_scaled"] for metric in SCORE_METRICS}})

    if len(role_contexts) != 6:
        raise ArchiveError("Replay must retain the same six roles across all declared seeds.")
    per_seed: list[dict[str, Any]] = []
    for seed in sorted(seeds):
        per_context: dict[str, float] = {}
        for context in sorted(contexts):
            items = [case for case in case_results if case["seed"] == seed and case["context"] == context]
            if len(items) != 2 or len({case["role"] for case in items}) != 2:
                raise ArchiveError(f"Seed {seed}, background {context!r}: exactly two distinct roles are required.")
            per_context[context] = math.fsum(case["overall"] for case in items) / 2
        per_seed.append({"seed": seed, "overall": math.fsum(per_context.values()) / 3,
                         "per_background": per_context})
    macro = math.fsum(item["overall"] for item in per_seed) / len(seeds)
    direct = math.fsum(case["overall"] for case in case_results) / len(case_results)
    require_match(macro, direct, "Equal-case mean versus background/seed macro")
    require_match(macro, expected, "Milestone expected_overall")
    six_scaled = {metric: math.fsum(case["six_scaled"][metric] for case in case_results) / len(case_results)
                  for metric in SCORE_METRICS}
    require_match(math.fsum(weights[metric] * six_scaled[metric] for metric in SCORE_METRICS), macro,
                  "Mean six-metric weighted overall versus macro")
    return {"schema_version": 1, "operation": "saved-component replay", "milestone_id": milestone_id,
            "ok": True, "scale": "replicate_scaled", "metric_weights": weights,
            "cases": sorted(case_results, key=lambda case: (case["seed"], case["context"], case["role"])),
            "case_count": len(case_results), "seeds": sorted(seeds), "contexts": sorted(contexts),
            "per_seed": per_seed, "per_background": {
                context: math.fsum(item["per_background"][context] for item in per_seed) / len(seeds)
                for context in sorted(contexts)}, "macro_overall": macro, "direct_case_mean": direct,
            "six_scaled": six_scaled, "expected_overall": expected,
            "max_absolute_difference": largest_difference, "tolerance": _REPLAY_TOLERANCE,
            "scope": "Saved CSV/summary arithmetic only; raw metrics, anchors, scaling, RNA, DE tests, and p-values were not recomputed.",
            "interpretation": "Arithmetic consistency of archived evidence; not a scientific scoring rerun or proof of stable promotion. Backgrounds, roles, and count seeds are not independent biological replicates."}


def excluded_from_release(relative: str) -> bool:
    pieces = relative.lower().split("/")
    basename = pieces[-1]
    return (relative.lower() == "assets/releases.json"
            or any(piece in _EXCLUDED_DIRS or piece.endswith(".egg-info") for piece in pieces[:-1])
            or basename == ".env" or basename.startswith(".env.")
            or basename in {".ds_store", "thumbs.db", "credentials", "credentials.json"}
            or Path(basename).suffix in _EXCLUDED_SUFFIXES)


def _common_paths(repo: Path) -> set[str]:
    found: set[str] = set()
    for path in repo.iterdir():
        if path.is_file() and (path.name in _COMMON_FILES or path.suffix.lower() == ".md"):
            found.add(path.relative_to(repo).as_posix())
    for name in _COMMON_DIRS:
        directory = safe_path(repo, name)
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    found.add(path.relative_to(repo).as_posix())
    for relative in _COMMON_STATIC_PATHS:
        if safe_path(repo, relative).is_file():
            found.add(relative)
    return {relative for relative in found if not excluded_from_release(relative)}


def build_release(repo: Path, milestone_id: str, output_dir: str = "artifacts") -> dict[str, Any]:
    """Package pinned milestone files and repository guidance; never raw assets."""
    repo = repo.resolve()
    index, manifests = load_registry(repo)
    select_milestones(manifests, [milestone_id])
    manifest = manifests[milestone_id]
    checked = verify(repo, [milestone_id])
    if not checked["ok"]:
        raise ArchiveError("Bundled evidence verification failed; release was not created.")
    destination = safe_path(repo, output_dir)
    try:
        destination.relative_to(safe_path(repo, "artifacts"))
    except ValueError as exc:
        raise ArchiveError("Release output must be within the repository artifacts/ directory.") from exc
    explicit = {record["path"] for record in manifest["files"]}
    entry = next(item for item in index["milestones"] if item["id"] == milestone_id)
    explicit.add(entry["manifest"])
    forbidden = sorted(relative for relative in explicit if excluded_from_release(relative))
    if forbidden:
        raise ArchiveError(f"Manifest requests excluded release assets: {', '.join(forbidden)}")
    paths = sorted(explicit | _common_paths(repo))
    payloads = {relative: safe_path(repo, relative).read_bytes() for relative in paths}
    # Recheck the exact bytes being archived, not just an earlier disk read.
    # This catches a concurrent modification between verification and packing.
    for record in manifest["files"]:
        content = payloads[record["path"]]
        if (len(content) != record["bytes"] or
                hashlib.sha256(content).hexdigest() != record["sha256"].lower()):
            raise ArchiveError(f"Asset changed while preparing the release: {record['path']!r}")
    try:
        packed_manifest = json.loads(payloads[entry["manifest"]].decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveError("Manifest changed while preparing the release.") from exc
    if packed_manifest != manifest:
        raise ArchiveError("Manifest changed while preparing the release.")
    # A release remains independently navigable: it references only the selected
    # milestone, rather than other manifests not included in this archive.
    payloads["milestones/index.json"] = json_bytes({
        "schema_version": 1, "milestones": [entry]})
    contents = [{"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                for relative, content in sorted(payloads.items())]
    payloads["RELEASE_CONTENTS.json"] = json_bytes({
        "schema_version": 1, "milestone_id": milestone_id,
        "packaging": "Sorted paths, fixed 1980 timestamp, regular-file 0644 permissions, ZIP_DEFLATED level 9.",
        "scope": "Saved source and evidence; external scientific assets are excluded.",
        "provenance_scope": "provenance/original-files.json, when included, describes the complete source repository context. Only paths listed here are present in this selected-milestone archive.",
        "files": contents})
    destination.mkdir(parents=True, exist_ok=True)
    archive = safe_path(repo, f"{output_dir}/{milestone_id}-source-evidence.zip")
    temporary = safe_path(repo, f"{output_dir}/.{archive.name}.tmp")
    receipt_path = safe_path(repo, f"{output_dir}/{milestone_id}-release.json")
    checksum_path = safe_path(repo, f"{output_dir}/{archive.name}.sha256")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
            for relative, content in sorted(payloads.items()):
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                handle.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = {"schema_version": 1, "operation": "build-release", "milestone_id": milestone_id,
              "archive": archive.relative_to(repo).as_posix(), "bytes": archive.stat().st_size,
              "sha256": sha256_file(archive), "archive_files": len(payloads),
              "scope": "Source and saved evidence only; portable scientific reruns are not asserted."}
    receipt_path.write_bytes(json_bytes(result))
    checksum_path.write_text(
        f"{result['sha256']}  {archive.name}\n", encoding="utf-8", newline="\n")
    return result
