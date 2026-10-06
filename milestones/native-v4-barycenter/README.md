# V4 · N1 barycenter weight 0.75

Original Native carrier mixture. Imported on October 6, 2026.

Preserves N1 weight 0.75 from a three-weight development sweep. Higher screen overall did not establish stable subsequent improvement. Target-NTC calibrations and latent rates remain external.

Saved local overall: **0.269914398708**. Protocol: `native6-historical-n1-contract-e489efd753fc0395`.

## What is preserved

- [Machine-readable result](results.json): saved scores, scope, seeds/backgrounds where applicable, and limitations.
- [Manifest](manifest.json): exact source/evidence hashes, bytes, known external inputs, and explicit replay cases.
- Original source, configurations, competing-arm records, and component tables under the referenced `snapshots/` paths. Original bytes and historical path bindings are preserved.
- Model asset availability: **uses matched-role V28 means and external target-NTC fitted state; no separate new mean checkpoint**. See [the data and asset guide](../../docs/DATA_AND_ASSETS.md).

## Verify this snapshot

```powershell
$env:PYTHONPATH = 'src'
python -m vcc_milestones verify native-v4-barycenter
python -m vcc_milestones replay-score native-v4-barycenter
```

Verification and saved-score replay are portable. A fresh scientific rerun is not newly validated: original external data, role-specific inputs, exact environments, and archival path restoration are still required. No raw count matrices or per-cell prediction populations are bundled. [Reproduction details](../../docs/REPRODUCIBILITY.md).

Status: **development experiment; not promoted**. This archive import makes no new promotion. Research remains paused.
