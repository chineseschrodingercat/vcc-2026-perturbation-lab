"""Command-line interface for saved competition milestones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .core import ArchiveError, build_release, compare, doctor, load_registry, replay_score, verify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vcc-milestones",
        description="Inspect, verify, and package saved milestones without running scientific experiments.")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root (default: current directory).")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List indexed milestones and saved-score scope.")
    verify_parser = subparsers.add_parser("verify", help="Check bundled file hashes and byte counts.")
    verify_parser.add_argument("ids", nargs="*", help="Milestone ids; omit to check all.")
    verify_parser.add_argument("--verbose", action="store_true", help="Print every checked file rather than compact counts and failures.")
    doctor_parser = subparsers.add_parser("doctor", help="Check external asset existence, bytes, and hashes only.")
    doctor_parser.add_argument("--workspace", type=Path, required=True, help="Original scientific workspace root.")
    doctor_parser.add_argument("ids", nargs="*", help="Milestone ids; omit to check all.")
    release_parser = subparsers.add_parser("build-release", help="Build a deterministic source/evidence ZIP.")
    release_parser.add_argument("id", help="Milestone id.")
    release_parser.add_argument("--output-dir", default="artifacts", help="Repo-relative folder within artifacts/.")
    compare_parser = subparsers.add_parser("compare", help="Compare saved overall scores only under matching protocol and scope.")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    replay_parser = subparsers.add_parser("replay-score", help="Replay SHA-pinned saved-component arithmetic, without scientific scoring.")
    replay_parser.add_argument("id", help="Milestone id with a verified score_replay specification.")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            _, manifests = load_registry(args.repo)
            result = {"schema_version": 1, "operation": "list", "milestones": [{
                key: manifest[key] for key in ("id", "title", "evidence_scope", "score_protocol_id", "overall")}
                for manifest in manifests.values()]}
        elif args.command == "verify":
            result = verify(args.repo, args.ids)
            if not args.verbose:
                summaries = {}
                for milestone_id, items in result["milestones"].items():
                    failures = [item for item in items if item["status"] != "ok"]
                    summaries[milestone_id] = {
                        "checked_files": len(items), "passed_files": len(items) - len(failures),
                        "failed_files": len(failures), "failures": failures}
                result = {**result, "milestones": summaries,
                          "checked_files": sum(item["checked_files"] for item in summaries.values()),
                          "failed_files": sum(item["failed_files"] for item in summaries.values()),
                          "detail": "Use --verbose to include every checked file."}
        elif args.command == "doctor":
            result = doctor(args.repo, args.workspace, args.ids)
        elif args.command == "build-release":
            result = build_release(args.repo, args.id, args.output_dir)
        elif args.command == "replay-score":
            result = replay_score(args.repo, args.id)
        else:
            result = compare(args.repo, args.left, args.right)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if result.get("ok", True) else 1
    except (ArchiveError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
