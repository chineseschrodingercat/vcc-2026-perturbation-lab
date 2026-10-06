# V2 · dispersion multiplier four

Original Native dispersion ablation. Imported on October 6, 2026.

Reuses matched-role V28 response predictions and applies theta×4. No new mean-model training. The overall is lower than historical V28. Complete alternatives remain in the evidence.

Saved local overall: **0.256119747512**. Protocol: `native6-historical-n1-contract-e489efd753fc0395`.

## What is preserved

- [Machine-readable result](results.json): saved scores, scope, seeds/backgrounds where applicable, and limitations.
- [Manifest](manifest.json): exact source/evidence hashes, bytes, known external inputs, and explicit replay cases.
- Original source, configurations, competing-arm records, and component tables under the referenced `snapshots/` paths. Original bytes and historical path bindings are preserved.
- Model asset availability: **shared historical teachers remain external; unified V28 assets are not substituted for CV teachers**. See [the data and asset guide](../../docs/DATA_AND_ASSETS.md).

## Verify this snapshot

```powershell
$env:PYTHONPATH = 'src'
python -m vcc_milestones verify native-v2-theta-four
python -m vcc_milestones replay-score native-v2-theta-four
```

Verification and saved-score replay are portable. A fresh scientific rerun is not newly validated: original external data, role-specific inputs, exact environments, and archival path restoration are still required. No raw count matrices or per-cell prediction populations are bundled. [Reproduction details](../../docs/REPRODUCIBILITY.md).

Status: **development experiment; not promoted**. This archive import makes no new promotion. Research remains paused.
