# V3 · early baseline and foundation-model comparison

Early model comparison, September 2026. Imported on October 6, 2026.

Source-and-records archive. The retained selection favors simple complete candidates; deleted State/Stack weights and missing result CSV are listed explicitly. No compatible overall is reconstructed.

No comparable overall is available.

## What is preserved

- [Machine-readable result](results.json): saved scores, scope, seeds/backgrounds where applicable, and limitations.
- [Manifest](manifest.json): exact source/evidence hashes, bytes, known external inputs, and explicit replay cases.
- Original source, configurations, competing-arm records, and component tables under the referenced `snapshots/` paths. Original bytes and historical path bindings are preserved.
- Model asset availability: **none; not recoverable**. See [the data and asset guide](../../docs/DATA_AND_ASSETS.md).

## Verify this snapshot

```powershell
$env:PYTHONPATH = 'src'
python -m vcc_milestones verify legacy-v3
```

Verification and saved-score replay are portable. A fresh scientific rerun is not newly validated: original external data, role-specific inputs, exact environments, and archival path restoration are still required. No raw count matrices or per-cell prediction populations are bundled. [Reproduction details](../../docs/REPRODUCIBILITY.md).

Status: **historical archive; weights incomplete**. This archive import makes no new promotion. Research remains paused.
