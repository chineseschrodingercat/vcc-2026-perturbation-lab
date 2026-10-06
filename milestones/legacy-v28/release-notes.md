# V28 · retained N1 baseline and deployment recipe

Imported research snapshot; tag `legacy-v28`. The tag records this archive import, not a reconstructed historical Git commit.

The historical six-role CV baseline remains 0.259328246718. Unified deployment weights reuse the global V7 tiny Transformer (600 steps, seed 20260928) plus a frozen K562 prior and deployed gene-specific dispersion. These are separate workflows.

Saved local overall: **0.259328246718**. Protocol: `native6-historical-n1-contract-e489efd753fc0395`.

Previously saved official deployment overall: **0.06333542864645086**. The saved local 0.2281364395524066 is another historical CV workflow. Neither number is substituted for the Native6 result, and no official query was made during this import. The formal checkpoint SHA256 is `a2b996eacf498ce63ff584be51d84bc52c77c87ec040f96a8a30d43a91d6dce2`.

Includes a deterministic source/evidence ZIP with SHA256 sidecar. Model assets: legacy-v28-model-assets.zip. Original data and per-cell predictions are excluded. Missing early assets and external scientific dependencies are documented.

Validated during packaging: byte integrity and, where available, independent saved-component arithmetic replay. No training, count generation, fresh cell-level scoring, official requests, or submissions occurred.

See `milestones/legacy-v28/README.md` for the scope, full original evidence, and reproduction boundary.
