# SpatialTranscriptFormer Framework

> [!WARNING]
> **Work in Progress**: This project is under active development. Core architectures, CLI flags, and data formats are subject to major changes.

<!-- -->

> [!TIP]
> **Framework Release**: SpatialTranscriptFormer has been restructured from a research codebase into a robust framework. You can now use the Python API to train on your own spatial transcriptomics data with custom backbones and architectures.

**SpatialTranscriptFormer** is a modular deep learning framework designed to bridge histology and biological pathways. It leverages transformer architectures to directly predict spatially-resolved **biological pathway activity scores** from H&E image patches, providing interpretable maps of the tissue microenvironment.

## Python API: Quick Start

The framework is designed to be integrated programmatically into your scanpy/AnnData workflows:

```python
from spatial_transcript_former import SpatialTranscriptFormer, Predictor, FeatureExtractor
from spatial_transcript_former.predict import inject_predictions

# 1. Load model and create feature extractor
model = SpatialTranscriptFormer.from_pretrained("./checkpoints/stf_small/")
extractor = FeatureExtractor(backbone="phikon", device="cuda")
predictor = Predictor(model, device="cuda")

# 2. Extract features from image patches
#    image_patches: (N, 3, 224, 224) float tensor in [0, 1]
#    coords:        (N, 2) tensor of spatial coordinates (from your WSI tiling)
features = extractor.extract_batch(image_patches, batch_size=64)  # → (N, 768)

# 3. Predict per-spot pathway activity from extracted features
predictions = predictor.predict_wsi(features, coords, return_dense=True)  # → (1, N, P)

# 4. Integrate with Scanpy (one pathway-activity vector per spot)
inject_predictions(adata, coords, predictions[0], pathway_names=model.pathway_names)
```

For more details, see the **[Python API Reference](docs/API.md)**.

## Key Technical Pillars

- **Modular Architecture**: Decoupled backbones, interaction modules, and pathway output heads.
- **Quad-Flow Interaction**: Configurable attention between Pathways and Histology patches (`p2p`, `p2h`, `h2p`, `h2h`).
- **Pathway-Exclusive Prediction**: Directly predicts biological pathway activity scores (e.g., 50 MSigDB Hallmark pathways) — no intermediate gene reconstruction step.
- **Offline Pathway Targets**: Ground-truth pathway activities are pre-computed offline (`stf-compute-pathways`) from raw gene expression using QC → CP10k normalisation → mean pathway aggregation. This eliminates the circular auxiliary loss used in previous versions.
- **Spatial Pattern Coherence**: Optimised using a composite **MSE + PCC (Pearson Correlation) loss**.
- **Foundation Model Ready**: Native support for **CTransPath**, **Phikon**, **Hibou**, **PLIP**, and **GigaPath**.

---

## License

SpatialTranscriptFormer is released under the **[Apache License 2.0](LICENSE)** — you are free to use, modify, and redistribute it (including commercially), provided you retain the copyright/license notices and state significant changes. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

> [!IMPORTANT]
> Apache-2.0 covers **this repository's source code only**. It does **not** grant rights to the third-party components this framework relies on at runtime, which keep their own licenses and are not redistributed here:
>
> - **Foundation-model backbones** (e.g. CTransPath, Phikon, GigaPath, Hibou) — each has its own license; some are gated or prohibit commercial use. Choose one whose license fits your use case (e.g. Apache-2.0 models such as Hibou, Virchow, or H-Optimus-0 for commercial use).
> - **HEST-1k dataset** (Mahmood Lab) — **CC BY-NC-SA 4.0** (non-commercial, share-alike): research/benchmarking only; obtain independent rights for commercial or clinical use.
> - **MSigDB Hallmark gene sets** (Broad Institute, v6.0–v7.5.1 / v2022.1+) — **CC BY 4.0**, attribution required; some subsets carry extra terms (e.g. KEGG). © 2004–2025 Broad Institute, Inc., MIT, and the Regents of the University of California.

## Attribution & Provenance

This is original work by Benjamin Isaac Wilson. The pathway↔histology interaction framing was **conceptually inspired by** SURVPATH (Jaume et al., 2024) — no SURVPATH source code is used or adapted. For the design contributions and third-party attributions, see the [Attribution & Design Notes](docs/IP_STATEMENT.md) and [NOTICE](NOTICE). If you use this work in academic research, please cite this repository and its author.

---

## Installation

This project requires [Conda](https://docs.conda.io/en/latest/).

1. Clone the repository.
2. Run the automated setup script:
   - On Windows: `.\setup.ps1`
   - On Linux/HPC: `bash setup.sh`

## Exemplar Recipe: HEST-1k Benchmark

The `SpatialTranscriptFormer` repository includes a complete, out-of-the-box CLI pipeline as an exemplar for reproducing our benchmarks on the [HEST-1k dataset](https://huggingface.co/datasets/MahmoodLab/hest).

### 1. Dataset Access & Preprocessing

```bash
# Download a specific subset
stf-download --organ Breast --disease Cancer --tech Visium --local_dir hest_data
```

### 2. Pre-Compute Pathway Activity Targets

Before training, compute the offline pathway activity matrix for each sample. This step applies per-spot QC and CP10k normalisation, then aggregates gene expression into MSigDB Hallmark pathway scores as the per-spot mean over each pathway's member genes.

```bash
stf-compute-pathways --data-dir hest_data
```

See the **[Pathway Mapping docs](docs/PATHWAY_MAPPING.md)** for a full description of the scoring methodology and available CLI options.

### 3. Training with Presets

```bash
# Recommended: Run the Interaction model (Small)
python scripts/run_preset.py --preset stf_small
```

### 4. Inference & Visualization

```bash
stf-predict --run-dir checkpoints --sample-id MEND29 --output-dir results
```

Visualization plots and spatial pathway activation maps will be saved to the `./results` directory. For the full guide, see the **[HEST Recipe Docs](src/spatial_transcript_former/recipes/hest/README.md)**.

## Documentation

### Framework APIs & Usage

- **[Python API Reference](docs/API.md)**: Full documentation for `Trainer`, `Predictor`, and `SpatialDataset`.
- **[Bring Your Own Data Guide](src/spatial_transcript_former/recipes/custom/README.md)**: Templates and examples for training on your own non-HEST spatial transcriptomics data.
- **[HEST Recipe Docs](src/spatial_transcript_former/recipes/hest/README.md)**: Detailed documentation for the included HEST-1k dataset recipe.
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Complete list of configuration flags and preset configurations for HEST models.

### Theory & Interpretability

- **[Models & Architecture](docs/MODELS.md)**: Deep dive into the pathway-exclusive prediction architecture, quad-flow interaction logic, and network scaling.
- **[Pathway Mapping](docs/PATHWAY_MAPPING.md)**: Offline pathway scoring methodology, QC pipeline, and MSigDB integration.
- **[SVG Exploratory Analysis](docs/SVG_HEST_EXPLORATORY_ANALYSIS.md)**: Detailed report on spatially variable pathway analysis across 95 HEST samples and data-driven target curation.
- **[Data Structure](docs/DATA_FORMAT.md)**: Detailed breakdown of the HEST data structure on disk, metadata conventions, and preprocessing invariants.

## Development

### Running Tests

```bash
# Run all tests (Pytest wrapper)
.\test.ps1
```

The test suite is organised into a hierarchical directory structure under `tests/`:

| Directory | Coverage Area |
| :--- | :--- |
| `tests/data/` | Data integrity, pathway scoring, augmentation |
| `tests/models/` | Backbone loading, interaction logic, model compilation |
| `tests/training/` | Loss functions, trainer loop, checkpoints, config |
| `tests/recipes/hest/` | HEST dataset loading and splitting |
| `tests/test_api.py` | End-to-end Python API integration |

## Development Roadmap

Active research and development is tracked in the **[Research & Improvement Roadmap](docs/SC_BEST_PRACTICES.md)**. Key directions are summarised below.

### Near-term

- **Extended knowledge base integration** — The offline pathway scoring step currently supports MSigDB Hallmarks via GMT files. The architecture is designed to be database-agnostic; future work will add first-class support for [decoupleR](https://decoupler-py.readthedocs.io) + [PROGENy](https://saezlab.github.io/progeny/) (Saez lab) and [LIANA+](https://liana-py.readthedocs.io) ligand-receptor databases as alternative scoring backends.
- **Visium HD & Xenium support** — Architecturally trivial; blocked only by data availability.

### Medium-term

- **Evaluation on the 2025 Nat. Comms. benchmark suite** (11 methods, 28 metrics, 5 datasets).
- **Pluggable scoring backends** — Allow `stf-compute-pathways` to accept any biological network (CollecTRI TF regulons, custom GMT files) without changing the model architecture.

### Longer-term

- **Clinical integration** — Using predicted spatial pathway activation maps as features for patient risk assessment and prognosis tracking in an end-to-end pipeline.

> [!NOTE]
> **Call for Collaborators:** Rigorous risk assessment models require large clinical cohorts with spatial transcriptomics and survival outcomes, which we currently lack access to. We are open to investigating *any* disease of interest. If you have access to such cohorts and are interested in exploring how spatially-resolved pathway activation correlates with patient prognosis, we would love to partner with you.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our coding standards and the process for submitting pull requests. Contributions are accepted under the project's Apache-2.0 license (inbound = outbound).
