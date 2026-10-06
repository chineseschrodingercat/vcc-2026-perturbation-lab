"""Data-free checks for archive integrity and scientific-comparison boundaries."""

from __future__ import annotations

import hashlib
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import zipfile

from vcc_milestones.core import (
    ArchiveError, build_release, compare, doctor, json_bytes, load_registry,
    safe_path, verify, replay_score, SCORE_METRICS,
)
from vcc_milestones.cli import main


class ArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        self.asset = b"Saved audit evidence\r\n"
        self._write("evidence/a/result.txt", self.asset)
        self._write("README.md", b"# A small scientific archive\n")
        self._write("src/tool.py", b"print('archive only')\n")
        self.entries: list[dict] = []
        self._add_milestone("a", 0.2)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, relative: str, content: bytes) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def _record(self, path: str, content: bytes, role: str = "audit") -> dict:
        return {"path": path, "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content), "role": role}

    def _add_milestone(self, milestone_id: str, overall: float | None,
                       protocol: str | None = "fixed-400", scope: str = "development") -> None:
        entry = {"id": milestone_id, "title": f"Milestone {milestone_id}",
                 "manifest": f"milestones/{milestone_id}/manifest.json"}
        self.entries.append(entry)
        manifest = {"schema_version": 1, "id": milestone_id, "title": entry["title"],
                    "evidence_scope": scope, "score_protocol_id": protocol,
                    "overall": overall,
                    "files": [self._record("evidence/a/result.txt", self.asset)],
                    "external_assets": []}
        self._write(entry["manifest"], json_bytes(manifest))
        self._write("milestones/index.json", json_bytes({"schema_version": 1, "milestones": self.entries}))

    def _change_manifest(self, milestone_id: str, **fields) -> None:
        relative = f"milestones/{milestone_id}/manifest.json"
        content = json.loads((self.repo / relative).read_text(encoding="utf-8"))
        content.update(fields)
        self._write(relative, json_bytes(content))

    def _pin_changed_asset(self, path: str) -> None:
        manifest_path = self.repo / "milestones/a/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            if record["path"] == path:
                content = (self.repo / path).read_bytes()
                record.update(self._record(path, content))
        manifest_path.write_bytes(json_bytes(manifest))

    def _add_replay(self, seeds: tuple[int, ...] = (101,)) -> dict:
        manifest_path = self.repo / "milestones/a/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases: list[dict] = []
        scores: list[float] = []
        for seed in seeds:
            for context_ordinal, context in enumerate(("hepg2", "jurkat", "rpe1")):
                for role_ordinal in range(2):
                    offset = context_ordinal * 0.01 + role_ordinal * 0.001 + seed * 0.00001
                    role = f"{context}-role{role_ordinal}"
                    prefix = f"evidence/a/replay/{seed}/{role}"
                    lines = ["metric,raw,fixed_scaled,replicate_scaled,baseline,replicate"]
                    raw: dict[str, float] = {}
                    fixed: list[float] = []
                    scaled: list[float] = []
                    for ordinal, metric in enumerate(SCORE_METRICS):
                        raw[metric] = 0.4 + ordinal * 0.01
                        fixed.append(0.1 + offset + ordinal * 0.005)
                        scaled.append(0.2 + offset + ordinal * 0.005)
                        lines.append(f"{metric},{raw[metric]},{fixed[-1]},{scaled[-1]},0.0,1.0")
                    components = ("\n".join(lines) + "\n").encode("utf-8")
                    weighted = math.fsum(value / 6 for value in scaled)
                    summary = json_bytes({**raw, "fixed_overall": math.fsum(value / 6 for value in fixed),
                                          "replicate_overall": weighted})
                    scores.append(weighted)
                    for suffix, content in (("components.csv", components), ("summary.json", summary)):
                        relative = f"{prefix}/{suffix}"
                        self._write(relative, content)
                        manifest["files"].append(self._record(relative, content))
                    cases.append({"role": role, "context": context, "seed": seed,
                                  "components_path": f"{prefix}/components.csv",
                                  "summary_path": f"{prefix}/summary.json"})
        expected = math.fsum(scores) / len(scores)
        replay = {"scale": "replicate_scaled", "metric_weights": {metric: 1 / 6 for metric in SCORE_METRICS},
                  "seeds": list(seeds), "contexts": ["hepg2", "jurkat", "rpe1"],
                  "cases": cases, "expected_overall": expected}
        manifest.update(overall=expected, score_replay=replay)
        manifest_path.write_bytes(json_bytes(manifest))
        return replay

    def test_verify_detects_same_size_hash_corruption(self) -> None:
        self.assertTrue(verify(self.repo)["ok"])
        self._write("evidence/a/result.txt", b"X" + self.asset[1:])
        result = verify(self.repo)
        self.assertFalse(result["ok"])
        self.assertEqual(result["milestones"]["a"][0]["status"], "hash_mismatch")

    def test_verify_detects_missing_and_wrong_size(self) -> None:
        (self.repo / "evidence/a/result.txt").unlink()
        self.assertEqual(verify(self.repo)["milestones"]["a"][0]["status"], "missing")
        self._write("evidence/a/result.txt", b"too short")
        self.assertEqual(verify(self.repo)["milestones"]["a"][0]["status"], "size_mismatch")

    def test_registry_rejects_identity_disagreement_and_duplicates(self) -> None:
        self._change_manifest("a", id="other")
        with self.assertRaisesRegex(ArchiveError, "identity/title"):
            load_registry(self.repo)
        self._change_manifest("a", id="a")
        self._write("milestones/index.json", json_bytes({
            "schema_version": 1, "milestones": self.entries + self.entries}))
        with self.assertRaisesRegex(ArchiveError, "duplicate milestone"):
            load_registry(self.repo)

    def test_paths_reject_traversal_absolute_windows_and_noncanonical_forms(self) -> None:
        for path in ("../secret", "a/../secret", "/etc/passwd", "C:/secret",
                     "C:\\secret", "\\\\host\\share", "a//b", "./a", "a/./b", "a\x00b"):
            with self.subTest(path=path), self.assertRaises(ArchiveError):
                safe_path(self.repo, path)
        self._change_manifest("a", files=[self._record("../secret", self.asset)])
        with self.assertRaises(ArchiveError):
            load_registry(self.repo)

    def test_compare_refuses_protocol_or_scope_mismatch(self) -> None:
        self._add_milestone("b", 0.3, protocol="historical-variable-size")
        with self.assertRaisesRegex(ArchiveError, "different scoring protocols"):
            compare(self.repo, "a", "b")
        self._change_manifest("b", score_protocol_id="fixed-400", evidence_scope="in-domain")
        with self.assertRaisesRegex(ArchiveError, "different evidence scopes"):
            compare(self.repo, "a", "b")

    def test_compare_requires_known_scores_and_is_descriptive(self) -> None:
        self._add_milestone("b", None, protocol=None)
        with self.assertRaisesRegex(ArchiveError, "known score protocol"):
            compare(self.repo, "a", "b")
        self._change_manifest("b", score_protocol_id="fixed-400")
        with self.assertRaisesRegex(ArchiveError, "overall score"):
            compare(self.repo, "a", "b")
        self._change_manifest("b", overall=0.3)
        result = compare(self.repo, "a", "b")
        self.assertAlmostEqual(result["delta_right_minus_left"], 0.1)
        self.assertIn("not evidence of a stable promotion", result["interpretation"])

    def test_doctor_hashes_external_asset_without_decoding(self) -> None:
        # Deliberately invalid RNA-format content: only bytes are inspected.
        blob = b"not a decodable RNA matrix\x00\xff"
        self._write("workspace/data/raw/sample.h5ad", blob)
        self._change_manifest("a", external_assets=[self._record("data/raw/sample.h5ad", blob, "external-required")])
        result = doctor(self.repo, self.repo / "workspace")
        self.assertTrue(result["ok"])
        self.assertIn("no RNA decoding", result["scope"])
        self.assertTrue(result["listed_assets_only"])
        self.assertTrue(result["full_pipeline_ready_not_certified"])
        self.assertEqual(result["checked_asset_count"], 1)
        (self.repo / "workspace/data/raw/sample.h5ad").unlink()
        self.assertFalse(doctor(self.repo, self.repo / "workspace")["ok"])

    def test_empty_doctor_does_not_certify_pipeline_readiness(self) -> None:
        result = doctor(self.repo, self.repo / "workspace")
        self.assertTrue(result["ok"])
        self.assertEqual(result["checked_asset_count"], 0)
        self.assertTrue(result["listed_assets_only"])
        self.assertTrue(result["full_pipeline_ready_not_certified"])

    def test_verify_cli_is_compact_and_verbose_is_explicit(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--repo", str(self.repo), "verify"])
        self.assertEqual(status, 0)
        compact = json.loads(output.getvalue())
        self.assertEqual(compact["checked_files"], 1)
        self.assertEqual(compact["failed_files"], 0)
        self.assertEqual(compact["milestones"]["a"]["passed_files"], 1)
        self.assertNotIn("sha256", output.getvalue())
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--repo", str(self.repo), "verify", "--verbose"])
        self.assertEqual(status, 0)
        verbose = json.loads(output.getvalue())
        self.assertEqual(verbose["milestones"]["a"][0]["sha256"], hashlib.sha256(self.asset).hexdigest())

    def test_release_contains_restore_entrypoint_and_model_registry(self) -> None:
        self._write("scripts/restore_model_assets.py", b"# Saved model-asset restore helper\n")
        self._write("assets/model-assets.json", b'{"schema_version":1,"assets":[]}\n')
        self._write("assets/releases.json", b'{"archive_sha256":"self-reference"}\n')
        self._write("provenance/original-files.json", b'{"scope":"complete-repository"}\n')
        result = build_release(self.repo, "a")
        with zipfile.ZipFile(self.repo / result["archive"]) as archive:
            names = archive.namelist()
            self.assertIn("scripts/restore_model_assets.py", names)
            self.assertIn("assets/model-assets.json", names)
            self.assertIn("provenance/original-files.json", names)
            self.assertNotIn("assets/releases.json", names)
            metadata = json.loads(archive.read("RELEASE_CONTENTS.json"))
            self.assertIn("complete source repository context", metadata["provenance_scope"])

    def test_zip_is_deterministic_and_excludes_raw_binary_and_secrets(self) -> None:
        self._add_milestone("b", 0.3)
        for relative in ("docs/raw/counts.csv", "docs/data/value.txt", "docs/checkpoints/x.pt",
                         "docs/array.npy", "docs/matrix.npz", "docs/expression.h5ad",
                         "docs/.env", "docs/.env.production", "docs/.git/config"):
            self._write(relative, b"must not be archived")
        first = build_release(self.repo, "a")
        before = (self.repo / first["archive"]).read_bytes()
        second = build_release(self.repo, "a")
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(before, (self.repo / second["archive"]).read_bytes())
        with zipfile.ZipFile(self.repo / first["archive"]) as archive:
            names = archive.namelist()
            self.assertEqual(names, sorted(names))
            self.assertEqual(archive.read("evidence/a/result.txt"), self.asset)
            self.assertFalse(any(name.startswith("docs/") for name in names))
            index = json.loads(archive.read("milestones/index.json"))
            self.assertEqual([entry["id"] for entry in index["milestones"]], ["a"])
            for info in archive.infolist():
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(info.external_attr >> 16, 0o100644)
            extracted = self.repo / "extracted"
            archive.extractall(extracted)
        self.assertTrue(verify(extracted)["ok"])

    def test_release_refuses_forbidden_explicit_asset_and_external_output(self) -> None:
        self._write("raw/counts.txt", self.asset)
        self._change_manifest("a", files=[self._record("raw/counts.txt", self.asset)])
        with self.assertRaisesRegex(ArchiveError, "excluded release assets"):
            build_release(self.repo, "a")
        self._change_manifest("a", files=[self._record("evidence/a/result.txt", self.asset)])
        with self.assertRaisesRegex(ArchiveError, "artifacts/"):
            build_release(self.repo, "a", "other-output")

    def test_saved_component_replay_handles_eighteen_cases(self) -> None:
        self._add_replay((101, 102, 103))
        result = replay_score(self.repo, "a")
        self.assertTrue(result["ok"])
        self.assertEqual(result["case_count"], 18)
        self.assertEqual(result["seeds"], [101, 102, 103])
        self.assertEqual(len(result["per_background"]), 3)
        self.assertLess(result["max_absolute_difference"], 1e-10)
        self.assertIn("not a scientific scoring rerun", result["interpretation"])

    def test_saved_component_replay_handles_six_cases(self) -> None:
        self._add_replay()
        result = replay_score(self.repo, "a")
        self.assertEqual(result["case_count"], 6)
        self.assertEqual(result["seeds"], [101])
        self.assertAlmostEqual(result["macro_overall"], result["expected_overall"])

    def test_saved_component_replay_refuses_unavailable_history(self) -> None:
        with self.assertRaisesRegex(ArchiveError, "replay is unavailable"):
            replay_score(self.repo, "a")

    def test_saved_component_replay_detects_missing_metric_after_hash_update(self) -> None:
        replay = self._add_replay()
        path = replay["cases"][0]["components_path"]
        lines = (self.repo / path).read_text(encoding="utf-8").splitlines()
        self._write(path, ("\n".join(lines[:-1]) + "\n").encode("utf-8"))
        self._pin_changed_asset(path)
        self.assertTrue(verify(self.repo)["ok"])
        with self.assertRaisesRegex(ArchiveError, "six unique metrics"):
            replay_score(self.repo, "a")

    def test_saved_component_replay_detects_wrong_summary_after_hash_update(self) -> None:
        replay = self._add_replay()
        path = replay["cases"][0]["summary_path"]
        summary = json.loads((self.repo / path).read_text(encoding="utf-8"))
        summary["replicate_overall"] += 0.01
        self._write(path, json_bytes(summary))
        self._pin_changed_asset(path)
        self.assertTrue(verify(self.repo)["ok"])
        with self.assertRaisesRegex(ArchiveError, "replicate_overall.*disagrees"):
            replay_score(self.repo, "a")

    def test_saved_component_replay_refuses_missing_roles_and_wrong_weights(self) -> None:
        replay = self._add_replay()
        replay["cases"][0]["context"] = "jurkat"
        self._change_manifest("a", score_replay=replay)
        with self.assertRaisesRegex(ArchiveError, "two distinct roles"):
            replay_score(self.repo, "a")
        replay["cases"][0]["context"] = "hepg2"
        replay["metric_weights"][SCORE_METRICS[0]] = 0.5
        self._change_manifest("a", score_replay=replay)
        with self.assertRaisesRegex(ArchiveError, "weights must sum to one"):
            replay_score(self.repo, "a")

    def test_saved_component_replay_refuses_nonfinite(self) -> None:
        replay = self._add_replay()
        path = replay["cases"][0]["components_path"]
        contents = (self.repo / path).read_text(encoding="utf-8")
        self._write(path, contents.replace("0.4,", "nan,", 1).encode("utf-8"))
        self._pin_changed_asset(path)
        with self.assertRaisesRegex(ArchiveError, "non-finite"):
            replay_score(self.repo, "a")

    def test_saved_component_replay_detects_wrong_expected_macro(self) -> None:
        replay = self._add_replay()
        replay["expected_overall"] += 0.01
        self._change_manifest("a", overall=replay["expected_overall"], score_replay=replay)
        with self.assertRaisesRegex(ArchiveError, "Milestone expected_overall.*disagrees"):
            replay_score(self.repo, "a")


if __name__ == "__main__":
    unittest.main()
