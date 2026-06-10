# Changelog

All notable changes to the SpatialTranscriptFormer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Created `CHANGELOG.md` documenting project history, milestones, and design choices.
- Documented the role of Moran's I (diagnostic target validation and spatial representation collapse detection) in [PATHWAY_MAPPING.md](docs/PATHWAY_MAPPING.md) and [spatial_stats.py](src/spatial_transcript_former/data/spatial_stats.py).

### Changed
- Refactored baseline models (`HE2RNA`, `ViT_ST` in [regression.py](src/spatial_transcript_former/models/regression.py)) to accept `num_pathways` instead of `num_genes` and directly regress pathway activities.
- Corrected console script entry points in [pyproject.toml](pyproject.toml) to map to `recipes/hest/` instead of `data/`.
- Updated [setup.ps1](setup.ps1) and [setup.sh](setup.sh) to suggest `stf-compute-pathways` instead of `stf-build-vocab`.
- Cleaned up parameter descriptions and docstrings in [dataset.py](src/spatial_transcript_former/recipes/hest/dataset.py), [trainer.py](src/spatial_transcript_former/training/trainer.py), and [checkpoint.py](src/spatial_transcript_former/checkpoint.py).
- Completely updated documentation files ([DATALOADER.md](docs/DATALOADER.md), [MODELS.md](docs/MODELS.md), [SC_BEST_PRACTICES.md](docs/SC_BEST_PRACTICES.md), [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md), [TESTING.md](docs/TESTING.md), [PRECOMPUTED_WORKFLOW.md](docs/PRECOMPUTED_WORKFLOW.md), [DATA_FORMAT.md](docs/DATA_FORMAT.md)) to reflect the pathway-exclusive paradigm and remove legacy gene-reconstruction references.

### Removed
- Deleted obsolete gene vocabulary builder script `build_vocab.py`.
- Deleted obsolete gene availability analysis document [GENE_ANALYSIS.md](docs/GENE_ANALYSIS.md).

---

## [0.2.0] - 2026-06

### Added
- Integrated multi-loss framework containing Concordance Correlation Coefficient (CCC), Huber loss, and CLIP-style contrastive loss to improve target convergence and model robustness.
- Added direct supervision head for pre-computed pathway targets, eliminating circular dependency issues from older auxiliary pathway loss architectures.
- Created public inference API and model wrapping framework.
- Introduced Moran's I diagnostics for Spatially Variable Gene (SVG) selection and spatial pattern evaluation.
- Added licensing disclaimers and specific attribution details for MSigDB Hallmark gene sets (CC BY 4.0), HEST-1k dataset, and third-party foundation models (CTransPath, Phikon).

### Fixed
- Resolved `TypeError` in transformer encoder by placing `enable_nested_tensor=False` in PyTorch's `TransformerEncoder` constructor.
- Configured pytest warnings filter in `pyproject.toml` to suppress non-critical output noise (e.g. deprecations from third-party libraries).

---

## [0.1.0] - 2026-03

### Added
- Initialized core package architecture, modules, test suite, and scripts.
- Implemented the quad-flow interaction system (early fusion of spatial transcriptomics and whole-slide histology features).
- Added `LocalPatchMixer` module (Scatter-Gather depthwise 2D convolutions) to introduce localized spatial inductive biases into slide spot processing.
- Added support for pre-computing histology feature extraction (e.g. using CTransPath) and building KD-Tree representations for spatial neighbor retrieval.
- Developed an interactive Matplotlib visualization widget to overlay predicted pathway activities on histology slide coordinates.
- Set up GitHub Actions CI workflow for automated testing.
