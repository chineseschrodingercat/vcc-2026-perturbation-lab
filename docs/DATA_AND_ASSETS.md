# Data and model assets

Git holds text source, configuration, score summaries, metric-component tables, small axes, and audit receipts. Releases hold the selected fitted model assets. Neither location holds raw RNA, processed RNA populations, real perturbation cells, NTC donor pools, per-cell predictions, or the multi-gigabyte competition package.

## Uploaded model families

| Family | Included | Where to download |
| :-- | :-- | :-- |
| `legacy-v28-model-assets` | Unified V7 tiny Transformer checkpoint, fitted feature transformation, deployed dispersion | [Historical V28 release](https://github.com/chineseschrodingercat/vcc-2026-perturbation-lab/releases/tag/legacy-v28) |
| `clean400-model-assets` | Three unique B0 checkpoints, three fitted preprocessing encoders, three dispersion arrays, three fitted context-KRR models | [B0–NB release](https://github.com/chineseschrodingercat/vcc-2026-perturbation-lab/releases/tag/clean400-b0-nb) |

The clean family maps 15 logical paths onto 12 unique blobs. B0 has six role bindings but only three different checkpoints: each lawful training pool is shared by two role exports. Count-seed repetitions reuse those fixed models. The V28 checkpoint is the **unified deployment model**; it is not a bundle of the six historical CV teachers.

Model feature transformations and encoders are fitted preprocessing state, not raw cell data. KRR files contain the kernel support representation: necessary source CP50K expression summaries, source-effect support dictionaries, source counts, and fitted parameters. These are RNA-derived aggregate model inputs; they contain no original cell-level UMI matrix. Full raw source-response banks and NTC-derived cell-level latent rates remain external. No third-party pretrained State, Stack, TxPert, or STRING payload is redistributed.

Each ZIP contains `RESTORE_MANIFEST.json` and `blobs/<SHA256>.<extension>`. [The registry](../assets/model-assets.json) records exact bytes, hash, role, and original relative path. The restoration helper verifies content before writing and refuses a conflicting destination.

## External inputs

The complete original protocols remain in the source snapshots. Each version also lists a curated set of external dependencies in `manifest.json`. These lists support local integrity checks, but they are not a substitute for the full historical dependency receipts.

- V28 needs the original axes, source mean/prior inputs, features, role-specific teachers for CV, gene-specific dispersion, real panels, NTC pools, and score references. The unified model asset ZIP covers three retained deployment assets; it does not cover every CV input.
- V2 theta×4 reuses V28's matched-role means and multiplies dispersion by four. There is no separately trained V2 mean checkpoint.
- V4 barycenter additionally needs target-NTC fitted calibrations and latent rate arrays. Those are external NTC-derived data, approximately 73 MB per background for the latent array alone.
- B0 fixed400 uses the lawful training-pool checkpoints and fixed generator state. It needs the original preprocessed training sources and Native panels for a full rerun.
- Context KRR uses the same NB renderer and a fitted public-source context model. The underlying public response banks and NTC profiles are external; they are not included as model assets.
- V3's deleted checkpoints cannot be restored by downloading current data. Its preserved selection record and source are an archive of what was tried, with explicit missing assets.

Original source origins and accession details are preserved in `snapshots/archive/legacy_20261003/configs/source_registry_v3.json` and the later protocols. Retrieve only the files needed by a selected recipe. Honor each source's terms. The original workspace continues to own its data; no source file was deleted or edited during packaging.

## Excluded payloads

The exclusion rules cover `.h5ad`, `.h5`, per-cell `.npy`/`.npz`, model `.pt` files in Git, archives, environments, caches, credentials, upload responses, and transient processes. The two reviewed model ZIPs are explicit exceptions for **Releases only**. Historical JSON receipts may reference files which remain external or were removed during earlier cleanup. Such references are evidence of the old workflow, not bundled availability.

Raw data are not required to verify this repository or replay its saved aggregate scores. They are required to validate a new scientific improvement.
