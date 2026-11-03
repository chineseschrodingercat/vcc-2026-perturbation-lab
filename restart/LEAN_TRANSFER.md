# Lean restart baseline

This lightweight workflow uses CPU aggregation/calibration and bounded parallel
count generation; the recovered V28 model can use CUDA. Use the prepared project
environment, currently Python 3.13 on the Windows host, with NumPy, SciPy,
pandas, anndata and h5py; V28 additionally requires the installed PyTorch build.
The workflow never uploads a prediction or changes challenge credentials.

## Local resource profile

`runtime_profile.py` measures available CPU/memory and records the execution
policy without reserving or filling RAM. On the current 16-logical-CPU machine
the defaults are 12 BLAS threads for one scientific job, four concurrent ingest
processes with three threads each, and six count-generation workers with the
generator process's BLAS thread limit set to two. The memory planning budget is
`min(100 GiB, max(0, available RAM - 16 GiB))`. This is a recorded working budget,
not an operating-system memory limit; streaming and bounded queues control the
actual allocations. CPU worker/thread defaults scale down on smaller machines.

```powershell
python restart/runtime_profile.py --output logs/runtime_profile.json
```

The launcher, source preparation and final builder save this profile in their
job/receipt metadata. `build_final.py --device auto` is the default for V28;
the existing model loader selects CUDA when available and records the actual
device. Parallelism changes resource use, not the scientific model, random seed
scheme or evaluation definitions. Throughput improvements must be measured on
the actual task; no speed multiplier is assumed.

## One-command local final build

After official login/download and source preparation are complete:

```powershell
python restart/build_final.py --root E:\VCC2026_Restart_20261103
```

The default `--recipe union` reruns external LOCO calibration on the official
gene axis, using up to 512 targets per background, freezes source content locks,
generates final D/E/F effects and sparse raw counts, and runs the installed
official `vcc prep`. Optional recipes are `--recipe core` and `--recipe v28`;
V28 is explicitly a reconstructed historical fallback, not an independently
validated LOCO model.
Union/core require all six prepared source files. V28 requires only the K562
GWPS source and its frozen model/feature/embedding/axis assets. The default final
builder passes the runtime's count-worker setting; `--workers 1` requests the
serial sampling path for a direct reproducibility comparison.

All required inputs must already exist. Missing files produce a list and the
`LOGIN_AND_GET_OFFICIAL_DATA.cmd` helper path, without creating placeholder
official data or a build directory. The script requires final manifest
contexts D/E/F, preserves their order and file mapping, enforces the project's
1,000,000,000,000-byte budget, and requires 80,000,000,000 bytes of available
headroom before count generation. Outputs and logs go into a new timestamped
`outputs/restart/final_<time>_<recipe>/` directory. The result contains
`final_prediction.vcc` and `build_receipt.json`; failures record their stage and
log. No upload command is executed and no submission allowance is consumed.

Where an official control file lacks `obs.context`, its label can come only
from the verified manifest-to-file mapping returned by `read_bundle`. If that
column exists it must agree exactly. The label source is recorded in output
metadata. This compatibility path has a synthetic check; actual current
official inputs still have to pass the real pipeline after download.

## 1. Construct-mean inputs

The Replogle `raw_bulk` matrices contain **per-cell construct means**, not raw
single-cell observations and not construct sums. Fractional entries are valid.
Use the filtered cell count as the weight when combining guides for one gene:

```powershell
python lean_transfer.py ingest --input K562_gwps_raw_bulk_01.h5ad --output processed/replogle_k562_gwps.npz --source-id replogle_k562_gwps --background K562 --kind raw-bulk --target-col construct-index --gene-col gene_name --n-col num_cells_filtered --control-col core_control
python lean_transfer.py ingest --input K562_essential_raw_bulk_01.h5ad --output processed/replogle_k562_essential.npz --source-id replogle_k562_essential --background K562 --kind raw-bulk --target-col construct-index --gene-col gene_name --n-col num_cells_filtered --control-col core_control
python lean_transfer.py ingest --input rpe1_raw_bulk_01.h5ad --output processed/replogle_rpe1.npz --source-id replogle_rpe1 --background RPE1 --kind raw-bulk --target-col construct-index --gene-col gene_name --n-col num_cells_filtered --control-col core_control
```

`construct-index` recognizes source names such as
`10023_ZC3H18_P1P2_ENSG...`. All named non-targeting rows are identified as
controls. With `--control-col core_control`, only the source's selected control
rows form the baseline; other non-targeting rows are **excluded**, never
converted into targets. Omit that flag to pool all identified NTCs. The JSON
sidecar records the policy and excluded count. Preserve the same `K562`
background for both K562 files.

For a raw single-cell H5AD:

```powershell
python lean_transfer.py ingest --input hepg2_raw_singlecell.h5ad --output processed/hepg2.npz --source-id nadig_hepg2 --background HepG2 --kind singlecell --target-col gene --gene-col gene_name --control-label non-targeting
```

Choose the exact source columns after inspecting metadata. `--matrix X`,
`--matrix layers/counts`, or `--matrix raw.X` selects the matrix. A source that
stores multiple contexts can be selected with repeatable exact
`--filter COLUMN=VALUE`. Single-cell counts are checked for integrality;
construct means are not. Ambiguous duplicated gene symbols are excluded and
reported. Input is read in row chunks; accumulation holds only the
target-by-gene pseudobulk arrays.

## 2. Compare transparent candidates

```powershell
python lean_transfer.py cv --sources processed/replogle_k562_gwps.npz processed/replogle_k562_essential.npz processed/replogle_rpe1.npz --output-dir runs/lean_cv
```

Once the official bundle is present, rerun with
`--genes data/official/final/gene_names.csv` so the normalization axis is exactly
available for prediction. Add HepG2/Jurkat NPZs to the source list and rerun the
same command when ingestion finishes.

Candidates are zero effect, K562 direct transfer, and context-weighted transfer
with a small shrinkage grid. Effects are differences of log1p pseudobulk
composition normalized to 50,000 counts over explicitly shared measured genes.
Context similarity uses only the query's control mean. Each source response
uses its own matched source controls. Within one biological background, each
target uses the highest-support source; the background receives one ensemble
vote. K562 GWPS and essential never count as independent backgrounds.

Each LOCO fold excludes **all** files from the held-out background. Query
features use that background's controls; held-out responses are read only to
form evaluation labels after candidate predictions exist. Output files:

- `proxy_per_target.csv`: per-target effect MSE and cosine, with an additional
  MSE excluding the target gene itself.
- `cv_report.json`: fold memberships, actual candidate outcomes and coverage.
- `selected_model.json`: the small candidate selected by equal-background mean
  MSE excluding the target gene.

Both CV commands lock each source's unique ID, biological background and full
NPZ SHA256 in `selected_model.json`. Prediction/effect generation rejects
duplicate IDs, changed source contents or backgrounds, and old model files
without these locks. After reaggregating any source, rerun CV; previously
generated pseudobulk files do not need to be reaggregated merely to add locks.

These are **development proxy metrics, not official VCC scores**. Selection
uses these same folds, so its chosen result is a tuning result, not an unbiased
generalization estimate. K562-direct cannot be evaluated with K562 held out;
its comparable per-fold rows are retained, but it is not ranked using an
incomplete set of folds. With only K562 and RPE1 there are only two independent
backgrounds.

The conservative baseline predicts only genes measured in every source; genes
outside that core retain the query control mean. Adding sources may reduce
this core. Coverage is explicit in the report; this route is a recoverable
baseline candidate, not a claim of the strongest available model.

### Measured-union candidate

`lean_union.py` addresses that coverage limitation separately, without changing
the core-only baseline. It retains the common core only as the library-size
denominator, but predicts over the measured union. Each gene's ensemble
denominator includes only backgrounds that measured that target/gene pair.
Unknown genes retain the query baseline. Within a background, the highest
supported source for each target/gene is used, so duplicate K562 files never
receive two votes. Evaluation uses only genes actually measured in the held-out
background and represented in that fold's query control file.

```powershell
python lean_union.py cv --sources processed/replogle_k562_gwps.npz processed/replogle_k562_essential.npz processed/replogle_rpe1.npz --output-dir runs/union_cv
```

Add `--genes <official gene_names.csv>` once available, and rerun before final
prediction if the earlier normalization core included nonofficial genes.
The generated CV outputs use the same names as the core baseline but stay in
their own directory. They retain the same proxy/selection caveats. A larger
output axis is coverage, not demonstrated prediction improvement; compare
actual fold metrics on the same gene subset before making any quality claim.

After calibration and official controls are available:

```powershell
python lean_union.py effects --bundle data/official/final --sources processed/replogle_k562_gwps.npz processed/replogle_k562_essential.npz processed/replogle_rpe1.npz --model-json runs/union_cv/selected_model.json --output-dir runs/union_effects
python lean_transfer.py predict --bundle data/official/final --effects-dir runs/union_effects --output predictions/union_final.h5ad --generator nb --seed 20261103 --workers 6
```

This converts the union prediction back into the explicitly labelled
raw-pseudocount log2FC format consumed by the same streamed count exporter.

## 3. Generate an official-shaped count matrix

```powershell
python lean_transfer.py predict --bundle data/official/final --sources processed/replogle_k562_gwps.npz processed/replogle_k562_essential.npz processed/replogle_rpe1.npz --model-json runs/lean_cv/selected_model.json --output predictions/lean_final.h5ad --generator nb --seed 20261103 --workers 6
```

The bundle must contain `manifest.json`, `gene_names.csv`, `pert_counts.csv`
and all three control H5ADs. Context names come from the manifest and must agree
with every control cell's `context` label. Neither A/B/C nor D/E/F is hardcoded.
The output has 18,533 genes, 300 targets per context, exactly 400 cells per
target/context, and no submitted controls.

The NB generator transfers the control's gene-wise dispersion to the predicted
mean. This is an explicit simple variance assumption, not a fitted prediction
of perturbed-cell variance. `--generator poisson` is available as a separate
candidate. Counts are integer, finite, nonnegative and CSR, with official row,
cell-total and stored-entry checks. `--workers` uses a bounded sampling queue
with at most twice the worker count of pending perturbation blocks. A single
writer appends results to HDF5 in the original context/target order, so the
360,000-cell matrix is never held in RAM. Each block keeps the same seed derived
from the base seed and its context/target indices, independent of completion
order. The direct `predict` CLI defaults to one worker; the final builder applies
the measured runtime profile. A partial file remains distinguishable if a run
fails. Outputs are never silently overwritten.

The optional `--on-target-fraction 0.2` forces the target gene to 20% of its
query control mean. It is **not fitted by this CV**, is off by default, and is
recorded if explicitly chosen.

Use the official CLI to validate and package separately:

```powershell
vcc prep predictions/lean_final.h5ad -g data/official/final/gene_names.csv --perts data/official/final/pert_counts.csv -o predictions/lean_final.vcc
```

### Frozen V28 adapter output

An independent adapter can supply `context_D.npz`, `context_E.npz`, and
`context_F.npz` (or whichever manifest labels are present) using:

- `effect_log2fc`: target-by-gene array;
- `gene_symbols`, `target_symbols`: exact axis labels;
- `effect_space = "raw_pseudocount_log2fc"`, `pseudocount = 0.5`.

Generate from these instead with `--effects-dir <directory>` and omit
`--sources` / `--model-json`. The raw-count mean is
`max((control_mean + 0.5) * 2**effect - 0.5, 0)`; unmatched official genes retain
their control mean. The exporter validates target identity and effect space.

## Semantic checks

`python -m unittest -v test_lean_transfer test_lean_union` checks weighted construct means,
non-core NTC exclusion, sparse single-cell aggregation, background isolation,
one-vote handling, missing-gene masks, exact zero projection, seed
reproducibility and streamed H5AD labels/counts. These small checks validate
implementation semantics; they do not measure biological performance. Union
checks additionally cover gene-specific denominators, unmeasured-gene fallback,
and complete held-out-background exclusion in the generated CV artifacts.
