# Reproducibility: three distinct levels

This archive makes the evidence easy to inspect while keeping its numerical limitations explicit.

| Level | Available from a fresh clone? | What the check establishes |
| :-- | :-- | :-- |
| File and registry verification | Yes; Python 3.10+ standard library | The included source and evidence match the published file hashes |
| Saved-component score replay | Yes; six scored milestones | The fixed weighted aggregation of saved components reproduces each archived overall |
| Model inference, count generation, and a fresh scientific score | Requires external inputs and original environments | Must reproduce the numerical prediction and count-space metrics; not newly run or certified during this import |

The early V3 archive has no recoverable comparable overall and intentionally does not support score replay.

## Inspect and verify

```powershell
$env:PYTHONPATH = 'src'
python -m vcc_milestones list
python -m vcc_milestones verify
python -m vcc_milestones replay-score legacy-v28
python -m vcc_milestones replay-score clean400-context-krr-nb
python -m unittest discover -s tests -v
```

Replay uses the full six-component CSV per case, checks both saved fixed and replicate scaled overall values against the corresponding summary, and reproduces equal roles within a background, equal backgrounds, and equal count seeds. Missing metrics, mismatched summaries, incomplete populations, or non-finite values fail. The packaged scalar results retain negative components rather than discarding them.

Replay is an arithmetic evidence check. It uses already calculated component tables. It does not independently rederive differential expression, MSE sampling correction, normalization, score anchors, or direction significance ranking from cells. The archived independent audits document which of those checks were actually performed during research.

## Restore model assets

Download the named model ZIP from [GitHub Releases](https://github.com/chineseschrodingercat/vcc-2026-perturbation-lab/releases). The V28 release has `legacy-v28-model-assets.zip`; the B0–NB release hosts the shared `clean400-model-assets.zip`. All logical paths and their hashes are in [the asset registry](../assets/model-assets.json).

```powershell
python scripts/restore_model_assets.py --archive 'downloads/legacy-v28-model-assets.zip' --workspace 'D:/vcc-external-workspace'
python -m vcc_milestones doctor --workspace 'D:/vcc-external-workspace' legacy-v28
```

The restorer verifies each content-addressed blob and recreates the documented relative paths. It accepts identical existing files and refuses to replace differing files. `doctor` checks **the listed external assets**, including data dependencies, without decoding RNA. Missing data after restoring weights is expected. Full original dependency receipts also remain in snapshot protocols; the curated external list is not an assertion that every historical pipeline can be rerun from these two ZIP files.

## Scientific environments and original sources

Historical model work used separate Python environments: Python 3.12 for model/generator code and Python 3.13 for the scientific judge. The archived locks are under `snapshots/archive/legacy_20261003/configs/`. They describe their historical environments; installation on a different machine has not been newly validated. The later fixed400 branch also records interpreter and loaded library bindings in its original protocols; those bindings take precedence over a generic old lock.

The judge is pinned to `cell_eval2==0.16.0`. Scoring follows six metrics with fixed anchors and equal weights. MSE aggregates its valid numerators and denominators before forming the ratio; per-target MSE values must not simply be averaged. The scoring population uses full retained real cells and the fixed score-NTC selection, while the generator uses a 3,000-cell NTC donor pool to draw 400 prediction cells. In the old Native protocol, prediction counts varied around 128–256; the 3,000 NTC cells were never 3,000 generated prediction cells per target.

The source snapshots intentionally keep original absolute paths, historical dependency hashes, and archived status text. Never launch an old exporter as a shortcut: some refit a decoder or contain submission operations. See [protocols](PROTOCOLS.md) and the version page first. A future portable scientific adapter must be a new, tested version; changing the frozen source silently would break the provenance contract.

## Build an inspectable source release

```powershell
python -m vcc_milestones build-release clean400-context-krr-nb
```

This creates a deterministic ZIP under `artifacts/` plus a SHA256 sidecar and receipt. It includes the selected manifest and its source/evidence dependencies with an isolated index. Raw data, model binaries, local caches, and secrets are excluded. Model ZIPs are separate release assets. Tags identify **imports of research milestones**, not recreated historical Git commits or historical publication dates.
