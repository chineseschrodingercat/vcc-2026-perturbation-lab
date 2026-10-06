# What the experiments changed

The retained sequence separates three questions: predicting an average response, calibrating its reliability, and generating single-cell counts. A new result should name which question it changes.

## Historical baselines

V3 compared simple response transfer with trained State pilots, pretrained Stack inference, and SLIM-style bilinear models. The recovered selection record retained simple complete candidates because the foundation variants did not establish a consistent six-metric replacement. Some large early artifacts were deleted before this import; that loss is visible in the V3 manifest.

V28 retained the original response predictor with an N1 negative-binomial generator. The unified deployment reused the globally selected V7 predictor rather than refitting a new mean model. This makes it a useful fixed historical anchor, but it does not make historical CV and official deployment scores interchangeable.

## Later generator and mean experiments

V2 theta×4 changes dispersion while preserving the mean prediction. It falls below V28 on the old Native protocol. V4 barycenter combines an NTC-conditioned carrier with the original N1 contribution; the weight-0.75 development result is higher, but later checks did not establish stable generalization. Full competing-arm summaries remain bundled to show where that snapshot was selected.

The clean fixed400 branch uses lawful double-held source pools and fixes the count population. B0–NB is its paired control; B0–Poisson changes the renderer at the same B0 mean. Context KRR changes the mean using source/context structure while retaining NB generation. Three count seeds were then paired without refitting the mean models.

KRR–NB improves the macro overall over B0, but that average conceals consistent Jurkat losses. Its interval includes zero, so the current snapshot remains an unpromoted candidate. B0–Poisson also has an unstable positive macro average.

## Negative follow-up results

The subsequent KRR–Poisson factorial combination completed all 72 logical score cases and independent count/scalar audits. Its three-seed overall was **0.264268**, below B0–NB and KRR–NB. The gain of each ingredient did not add up into a better combination. It was not promoted.

The 0.5 post-decoder mixture of B0 and KRR also failed its fixed single-seed screen: overall **0.270395**, below B0 **0.270591** and KRR **0.275481** under that screen. It was closed without a weight sweep. Original result and completion receipts are included under `snapshots/runs/research_v28_20261005/`.

These failures matter for the next version: do not rerun the same mechanism without a new hypothesis, repaired implementation, or new data evidence. Keep mean, calibration, and renderer changes separately attributable. Any direction-confidence method must eventually be judged from final integer cells; editing a p-value table is not a model improvement.

## Deliberately outside this milestone set

The retained historical V129 has H1 two-fold CD4 raw-metric evidence but no compatible Native overall, and it requires additional external derived profiles. It is not a seventh score competing with these tables. Later restarted V-number experiments and literature/data preparation remain in the original workspace. This repository packages the seven selected milestones, not every transient research artifact.
