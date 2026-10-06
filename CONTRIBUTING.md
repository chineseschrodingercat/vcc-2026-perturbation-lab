# Extending the archive

Keep a new experiment small enough that its result has a clear interpretation, and keep the existing record intact.

## Add a milestone

1. Choose a namespaced ID: `clean400-method-v2`, `native-method-v1`, or a new benchmark namespace. Short V-numbers alone are ambiguous after the historical restart.
2. Copy reviewed source, frozen configuration, and complete scalar evidence into a new snapshot location. Exclude RNA populations, per-cell predictions, environments, credentials, and upload receipts.
3. Add `milestones/<id>/README.md`, `results.json`, `release-notes.md`, and `manifest.json`. Register it in `milestones/index.json`. The manifest records SHA256, bytes, role, protocol ID, evidence scope, overall, external inputs, and explicit replay cases.
4. Record full cases, all six metrics, backgrounds, seeds, independent validation units, unsuccessful arms, reuse, and missing assets. Distinguish count-seed repeats from model or biological repeats.
5. Put new fitted weights in a separate reviewed Release asset with a content-addressed restore manifest. Extend `assets/model-assets.json` without replacing existing blobs.
6. Run the tests, `verify`, and `replay-score`. Build the deterministic source ZIP and inspect its file list. Record its digest in `assets/releases.json`.
7. Open a focused PR explaining the hypothesis, the exact change, unchanged comparator, validation, result, and limitation. Tag an accepted snapshot and attach its source ZIP and model assets to a Release.

A different population, target set, gene axis, normalization unit, sampling rule, or scoring anchor means a new scoring protocol. Re-evaluate its control under the same protocol before interpreting a gain. Keep the primary promotion criterion overall; individual metrics explain the change.

## Correct an existing record

Do not silently change a released scientific file. Create a new revision with the original hash, corrected hash, reason, affected conclusions, and replacement release. Pure documentation clarifications can use a normal PR if they do not alter frozen source or result files. Imported historical tags are not reconstructed original Git history.

## Verification commands

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python -m vcc_milestones verify
python -m vcc_milestones replay-score YOUR_ID
python -m vcc_milestones build-release YOUR_ID
```

CI checks the archive on Windows and Linux without downloading scientific data or running models. A passing CI run establishes repository integrity and arithmetic replay, not new biological validation.

Research execution remains paused unless explicitly resumed by the project owner. Packaging, documentation, and integrity fixes do not authorize training or submissions.
