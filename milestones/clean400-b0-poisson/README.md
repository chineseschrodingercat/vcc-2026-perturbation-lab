# Clean400 · B0 Poisson

Fixed400 generator ablation. Imported on October 6, 2026.

Holds B0 mean fixed and replaces NB with Poisson. Positive macro average is unstable: two of three count-seed deltas are negative, and the paired interval includes zero.

Saved local overall: **0.269863085057**. Protocol: `clean400-double-held-development-c8a719bda010e352`.

Paired descriptive 95% interval for the change from B0–NB: **[-0.007600, 0.010883]**. Backgrounds and count seeds are the resampling units.

## What is preserved

- [Machine-readable result](results.json): saved scores, scope, seeds/backgrounds where applicable, and limitations.
- [Manifest](manifest.json): exact source/evidence hashes, bytes, known external inputs, and explicit replay cases.
- Original source, configurations, competing-arm records, and component tables under the referenced `snapshots/` paths. Original bytes and historical path bindings are preserved.
- Model asset availability: **shared clean400-model-assets.zip from the B0–NB release**. See [the data and asset guide](../../docs/DATA_AND_ASSETS.md).

## Verify this snapshot

```powershell
$env:PYTHONPATH = 'src'
python -m vcc_milestones verify clean400-b0-poisson
python -m vcc_milestones replay-score clean400-b0-poisson
```

Verification and saved-score replay are portable. A fresh scientific rerun is not newly validated: original external data, role-specific inputs, exact environments, and archival path restoration are still required. No raw count matrices or per-cell prediction populations are bundled. [Reproduction details](../../docs/REPRODUCIBILITY.md).

Status: **candidate; unstable gain**. This archive import makes no new promotion. Research remains paused.
