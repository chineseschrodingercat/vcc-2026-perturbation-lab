# Scoring and validation protocols

There are three namespaces in this import. The same short version label must never stand in for all three.

| Namespace | Benchmark | Interpretation |
| :-- | :-- | :-- |
| `legacy-*` | Early history and retained historical recipe | Exact version identity and recovery boundary |
| `native-*` | Original Native six-role development population and N1 renderer | Compare with the matched-role historical V28 baseline |
| `clean400-*` | Fixed 400 generated cells per target, double-held model roles | Compare only within the fixed400 group |

The project restarted its short V-number sequence after October 3. `legacy-v28` is the original N1 model recipe, not the later new-V28 state-transfer diagnostic. Namespaced IDs resolve that collision.

## Six metric components

| Metric key | Meaning in the retained scoring workflow |
| :-- | :-- |
| `pds_cosine` | Perturbation-response similarity |
| `expr_mse_unbiased_capped_norm` | Expression error with the official sampling correction |
| `de_wilcoxon_direction_fidelity_yield_raw` | Direction fidelity and significant-response yield |
| `de_wilcoxon_direction_reach_raw` | Direction reach, sensitive to significance ordering |
| `de_wilcoxon_sig_jaccard` | Significant-gene overlap |
| `de_wilcoxon_lfc_nmae` | Fold-change absolute error |

The local overall is the equal-weight aggregate of six scaled components, with the fixed role/background aggregation. Metric keys ending in `raw` are historical names; the overall still uses their saved **scaled** values. Zero scaled MSE does not mean zero raw error. Negative components remain in the tables.

Every scored manifest identifies a protocol and contains explicit score-replay cases. The CLI refuses comparisons across protocol or evidence scope. Protocol equality permits a descriptive comparison; it does not establish independent validation.

## Fixed400 details

The complete shared evaluation fingerprint is `c8a719bda010e352af4dd7c433313c795eab5460e709897a95b474cd9091e35c`. Its original receipt is in `snapshots/reports/research_v28_20261005/SHARED_EVALUATION_AUDIT.json`.

It retains three developed Native backgrounds, two roles per background, eight original target slots per role, full retained gene axes, the original own-gene exclusion, fixed NTC/reference selection, and count seeds 20260928/29/30. There are 48 target-role slots per seed; overlapping roles and individual cells are not independent model-validation repeats.

Target NTC may enter prediction; destination perturbation truth enters scoring. The clean branch excludes both held Native families when constructing its training pool. It does not certify publisher-independent generalization, and repeated development use means these Native backgrounds are not an untouched final test. The final THP truth remains unopened in this branch.

The paired uncertainty summaries resample backgrounds and count seeds, not individual cells. They are descriptive intervals for this small developed benchmark. KRR–NB's interval spans zero, with Jurkat losses in every seed. B0–Poisson's interval also spans zero. Neither is presented as a stable promotion.

## Historical V28

Historical Native CV and unified A/B/C deployment are separate workflows. The deployment reused a global V7 tiny Transformer plus K562 prior and fitted dispersion. Its archived official value is recorded only as historical evidence, obtained before this import. No official query or submission was made to create the repository.

The formal in-domain fixed400 V28 reconstruction can reach a higher local value under its own protocol. That is an identity/trace diagnostic, not a clean cross-background milestone or a comparable 0.35 achievement. It is intentionally absent from the clean ranking.
