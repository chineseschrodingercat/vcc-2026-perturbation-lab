# Optional response summaries

`audit_extra_signatures.py` produces a small read-only JSON audit. It does not
train, generate predictions, exponentiate effects, or change a source file.

Run from the project directory after downloads complete (or run earlier to see
explicit `pending` entries):

```powershell
python restart/audit_extra_signatures.py --project-root . --manifest restart/download_manifest.json --onlycompleted
```

If the scripts are stored elsewhere, substitute their actual relative path.
Default output: `reports/extra_data_audit.json`. Use `--output` to preserve another
report, `--source feng` / `--source cd4t` to audit one family, and optional
`--targets path/to/pert_counts.csv --genes path/to/gene_names.csv` to limit the
Feng pair/effect audit to an existing axis. CSV, TSV, TXT, JSON lists and ordinary
one-dimensional string NPY files are supported. No pickle loading is enabled.

## What is checked

- **Feng iPSC:** streams gzip chunks; records input/selected target and gene
  coverage, finite effect counts, exact range/moments, sampled quantiles,
  on-target effect statistics and invalid adjusted p-values. Duplicate
  target–gene rows are counted exactly across chunk boundaries with compact
  bitsets. Quantiles use a reproducible uniform sample; they are explicitly
  labelled approximate when the finite population exceeds that sample.
- **CD4T:** reads only HDF5 `obs`/`var` columns and matrix/layer schema. Reports
  target/condition counts, gene overlap, duplicate keys and quality fields with
  their interpretation. It never materializes the 16.8 GB matrix. Effect values
  and their finiteness are **not** certified by this metadata-only pass.
- **Incomplete transfers:** absent files, `.part` siblings and size mismatches
  are marked `pending` and are never opened. Existing download receipts are
  reported when their verification/mtime matches; the audit does not rehash
  multi-gigabyte sources. Pending inputs return exit code 0; an actual audit
  error returns 1.

## Interpretation before adding data to a model

Feng's reported `lfc` is a mean **covariate-adjusted log-expression residual**
from a model fitted on controls. It is not interchangeable with a raw-pseudobulk
or DESeq2 log2 fold change. The author's code separately computes an unadjusted
effect with its own log base. Keep Feng as a candidate feature/prior until its
sign, scale and whole-cell-context holdout benefit are checked. The script never
uses `exp2(lfc)` or automatically admits it into training.

The CD4T author's `layers['log_fc']` is DESeq2 log2FC. The file contains response
summaries, not single cells. It cannot supply a raw count population or its
dispersion. Fields such as `low_target_gex`, `distal_offtarget_flag`,
`ontarget_significant`, guide counts and guide/donor concordance need explicit
quality decisions before using those effects. The audit reports their presence
and observed values without silently filtering observations.

For this restart, obtain a working baseline from the six small
Replogle/Nadig/RPE1 sources first. Add these summaries only when a controlled
whole-context comparison supports them. RPE1 raw bulk and RPE1 single cells
derive from the same screen and must stay on the same side of a holdout.

## Source records

- Feng summaries and MIT licence: <https://api.figshare.com/v2/articles/26819743>
- Feng effect computation: <https://github.com/claudiafeng123/crispri_scrnaseq_hipsci/blob/main/6b_calc_lfcs_transcriptome_wide_by_gene.R>
- Primary CD4T schema and MIT licence: <https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq>
- Original CD4T download prefix: <https://github.com/emdann/GWT_perturbseq_analysis_2025/blob/master/README.md>

Upstream file sizes and hashes are recorded in `download_manifest.json` and
`external_candidates.json`. A multipart S3 ETag is not an MD5 checksum.
