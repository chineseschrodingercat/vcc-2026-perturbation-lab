<div align="center">

# Virtual Cell Challenge 2026

**Perturbation prediction across cellular contexts — with an inspectable research history.**

[![Evidence archive](https://img.shields.io/badge/evidence-7%20milestones-147D92)](milestones/index.json)
[![Archive integrity](https://github.com/chineseschrodingercat/vcc-2026-perturbation-lab/actions/workflows/archive-integrity.yml/badge.svg)](https://github.com/chineseschrodingercat/vcc-2026-perturbation-lab/actions)
[![Verification](https://img.shields.io/badge/verification-Python%203.10%2B-3776AB)](docs/REPRODUCIBILITY.md)
[![Data policy](https://img.shields.io/badge/raw%20data-external-6B7280)](docs/DATA_AND_ASSETS.md)

[Milestones](docs/MILESTONES.md) · [Reproduce the evidence](docs/REPRODUCIBILITY.md) · [Model assets](docs/DATA_AND_ASSETS.md) · [Research notes](docs/RESEARCH_NOTES.md)

</div>

This repository preserves Minhao Liu's work on predicting CRISPRi responses in cellular backgrounds where only non-targeting control cells are available. It brings the early model comparisons, the retained V28 baseline, and two later experiment groups into a coherent archive that can be checked, discussed, and extended.

The practical question behind the work is simple: **which parts of a perturbation response transfer to a new cellular context, and how should that response become a population of integer-count cells?** We study the predicted mean, its calibration, and the cell generator separately so that a score change has an interpretable cause.

Some ideas improved a development score; several did not survive paired checks. Both outcomes belong here. A release is a preserved research snapshot, not a claim that its method generalizes or that it replaced the baseline.

> **Status:** research is paused. This October 6, 2026 import packages existing work; it does not restart training, generation, scoring, or competition submissions. Raw and processed RNA populations are external. No new official score was requested during packaging.

## The milestones

### Historical anchors

| Version | What it preserves | Reproduction boundary |
| :-- | :-- | :-- |
| [Early V3](milestones/legacy-v3/README.md) | Simple transfer baselines and the State / Stack / SLIM comparison | Source and recovered records; several original weights no longer exist |
| [Historical V28](milestones/legacy-v28/README.md) | Retained N1 recipe, matched-role CV evidence, unified deployment weights | Model assets available; numerical reruns require the original external inputs and scientific environments |

### Earlier Native six-role development benchmark

| Snapshot | Local overall | Interpretation |
| :-- | --: | :-- |
| [V4 barycenter, N1 weight 0.75](milestones/native-v4-barycenter/README.md) | 0.269914 | Development sweep winner; subsequent validation did not establish stable improvement |
| [Historical V28 N1](milestones/legacy-v28/README.md) | 0.259328 | Retained historical baseline |
| [V2, dispersion ×4](milestones/native-v2-theta-four/README.md) | 0.256120 | Controlled generator ablation; below V28 |

### Fixed 400-cell, double-held development benchmark

| Snapshot | Three-count-seed mean overall | Paired change from B0–NB | Interpretation |
| :-- | --: | --: | :-- |
| [Context KRR–NB](milestones/clean400-context-krr-nb/README.md) | 0.271666 | +0.004481 | Highest validated snapshot in this group; descriptive interval includes zero |
| [B0–Poisson](milestones/clean400-b0-poisson/README.md) | 0.269863 | +0.002678 | Positive macro average; two of three count seeds decline |
| [B0–NB](milestones/clean400-b0-nb/README.md) | 0.267185 | — | Fixed paired control |

**Compare rows within a benchmark, not between these tables.** Cell populations, generated cell counts, models, and validation scope differ. Three count seeds quantify generator variability; they are not three independent model fits or biological replicates. These backgrounds have been used for development. The final held-out background remains unopened in this research branch.

V28's local CV value also does **not** describe its unified A/B/C deployment checkpoint. That distinction, together with previously saved official evidence, is documented in its version page. There is no combined local/official leaderboard in this repository.

## Start in two minutes

The archive tools need Python 3.10 or later and no scientific libraries.

```bash
git clone https://github.com/chineseschrodingercat/vcc-2026-perturbation-lab.git
cd vcc-2026-perturbation-lab
python -m pip install -e .
vcc-milestones list
vcc-milestones verify
vcc-milestones replay-score clean400-context-krr-nb
```

For an installation-free PowerShell session:

```powershell
$env:PYTHONPATH = 'src'
python -m vcc_milestones verify
python -m vcc_milestones replay-score clean400-context-krr-nb
```

`verify` checks the bundled bytes against their recorded SHA256 hashes. `replay-score` reconstructs the saved overall from all six saved metric components and the fixed role/background/seed aggregation. It does not regenerate cells, calculate new p-values, or run the scientific judge. A full numerical rerun requires the external data and the pinned scientific environments described in [Reproducibility](docs/REPRODUCIBILITY.md).

## How the archive is organized

```text
milestones/       Version pages, results, manifests, and release notes
snapshots/        Original source/configuration/evidence, preserved byte for byte
assets/           Small manifests for model assets hosted in GitHub Releases
provenance/       Original-file mapping and import hashes
docs/            Protocols, data contracts, limitations, and extension workflow
src/             Portable archive verification and packaging tools
scripts/         Safe model-asset restoration helper
tests/           Integrity, path-safety, comparison, and score-replay checks
.github/         Windows/Linux CI and focused Issue / PR templates
```

The original scientific scripts are **archival source**, not the portable tools. They retain their Windows paths and exact historical environment checks. This preserves their provenance without implying that changing a path is enough to reproduce every experiment.

## Adding the next version

Give the next experiment a new namespaced ID, record its data/split/protocol, add a version manifest and complete results, then run verification and replay before cutting a release. Keep existing milestone files immutable; record corrections in a new revision. [Contributing](CONTRIBUTING.md) describes the small set of steps and the evidence expected for a new result.

Model binaries belong in release attachments; datasets, single-cell predictions, caches, environment folders, and credentials do not belong in Git. [The data and asset guide](docs/DATA_AND_ASSETS.md) explains what was retained, excluded, or lost before this import.

---

Maintained by **Minhao Liu**, Texas Tech University. This is an independent research archive for the Virtual Cell Challenge; it is not an official challenge repository. Project code and third-party dependencies retain their respective rights. See [NOTICE](NOTICE).
