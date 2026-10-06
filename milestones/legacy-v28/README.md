# V28 · retained N1 baseline and deployment recipe

Retained N1 baseline, October 2026. Imported on October 6, 2026.

The historical six-role CV baseline remains 0.259328246718. Unified deployment weights reuse the global V7 tiny Transformer (600 steps, seed 20260928) plus a frozen K562 prior and deployed gene-specific dispersion. These are separate workflows.

Saved local overall: **0.259328246718**. Protocol: `native6-historical-n1-contract-e489efd753fc0395`.

Previously saved official deployment overall: **0.06333542864645086**. The saved local 0.2281364395524066 is another historical CV workflow. Neither number is substituted for the Native6 result, and no official query was made during this import. The formal checkpoint SHA256 is `a2b996eacf498ce63ff584be51d84bc52c77c87ec040f96a8a30d43a91d6dce2`.

## What is preserved

- [Machine-readable result](results.json): saved scores, scope, seeds/backgrounds where applicable, and limitations.
- [Manifest](manifest.json): exact source/evidence hashes, bytes, known external inputs, and explicit replay cases.
- Original source, configurations, competing-arm records, and component tables under the referenced `snapshots/` paths. Original bytes and historical path bindings are preserved.
- Model asset availability: **legacy-v28-model-assets.zip**. See [the data and asset guide](../../docs/DATA_AND_ASSETS.md).

## Verify this snapshot

```powershell
$env:PYTHONPATH = 'src'
python -m vcc_milestones verify legacy-v28
python -m vcc_milestones replay-score legacy-v28
```

Verification and saved-score replay are portable. A fresh scientific rerun is not newly validated: original external data, role-specific inputs, exact environments, and archival path restoration are still required. No raw count matrices or per-cell prediction populations are bundled. [Reproduction details](../../docs/REPRODUCIBILITY.md).

Status: **retained historical baseline**. This archive import makes no new promotion. Research remains paused.
