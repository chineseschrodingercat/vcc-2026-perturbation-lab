# Clean400 · B0 negative binomial

Fixed400 paired control. Imported on October 6, 2026.

Lawful double-held B0 response model with original NB generation. Three distinct training-pool models produce six role exports. Three count seeds reuse those fixed means.

Saved local overall: **0.267185041654**. Protocol: `clean400-double-held-development-c8a719bda010e352`.

## What is preserved

- [Machine-readable result](results.json): saved scores, scope, seeds/backgrounds where applicable, and limitations.
- [Manifest](manifest.json): exact source/evidence hashes, bytes, known external inputs, and explicit replay cases.
- Original source, configurations, competing-arm records, and component tables under the referenced `snapshots/` paths. Original bytes and historical path bindings are preserved.
- Model asset availability: **clean400-model-assets.zip**. See [the data and asset guide](../../docs/DATA_AND_ASSETS.md).

## Verify this snapshot

```powershell
$env:PYTHONPATH = 'src'
python -m vcc_milestones verify clean400-b0-nb
python -m vcc_milestones replay-score clean400-b0-nb
```

Verification and saved-score replay are portable. A fresh scientific rerun is not newly validated: original external data, role-specific inputs, exact environments, and archival path restoration are still required. No raw count matrices or per-cell prediction populations are bundled. [Reproduction details](../../docs/REPRODUCIBILITY.md).

Status: **fixed control**. This archive import makes no new promotion. Research remains paused.
