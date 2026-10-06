# Clean400 · context kernel ridge with NB

Fixed400 context transfer. Imported on October 6, 2026.

Uses source/context kernel ridge response estimation and retains the original NB renderer. Positive macro gain, but consistent Jurkat losses and an interval crossing zero. Not promoted.

Saved local overall: **0.271666379388**. Protocol: `clean400-double-held-development-c8a719bda010e352`.

Paired descriptive 95% interval for the change from B0–NB: **[-0.018462, 0.027278]**. Backgrounds and count seeds are the resampling units.

## What is preserved

- [Machine-readable result](results.json): saved scores, scope, seeds/backgrounds where applicable, and limitations.
- [Manifest](manifest.json): exact source/evidence hashes, bytes, known external inputs, and explicit replay cases.
- Original source, configurations, competing-arm records, and component tables under the referenced `snapshots/` paths. Original bytes and historical path bindings are preserved.
- Model asset availability: **shared clean400-model-assets.zip from the B0–NB release; includes kernel support/source-summary representation**. See [the data and asset guide](../../docs/DATA_AND_ASSETS.md).

## Verify this snapshot

```powershell
$env:PYTHONPATH = 'src'
python -m vcc_milestones verify clean400-context-krr-nb
python -m vcc_milestones replay-score clean400-context-krr-nb
```

Verification and saved-score replay are portable. A fresh scientific rerun is not newly validated: original external data, role-specific inputs, exact environments, and archival path restoration are still required. No raw count matrices or per-cell prediction populations are bundled. [Reproduction details](../../docs/REPRODUCIBILITY.md).

Status: **highest validated clean snapshot; not promoted**. This archive import makes no new promotion. Research remains paused.
