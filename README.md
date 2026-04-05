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

# 3. Predict pathway activity scores from extracted features
predictions = predictor.predict_wsi(features, coords)  # → (1, P)

# 4. Integrate with Scanpy
inject_predictions(adata, coords, predictions[0], pathway_names=model.pathway_names)
```

For more details, see the **[Python API Reference](docs/API.md)**.

## Key Technical Pillars

- **Modular Architecture**: Decoupled backbones, interaction modules, and pathway output heads.
- **Quad-Flow Interaction**: Configurable attention between Pathways and Histology patches (`p2p`, `p2h`, `h2p`, `h2h`).
- **Pathway-Exclusive Prediction**: Directly predicts biological pathway activity scores (e.g., 50 MSigDB Hallmark pathways) — no intermediate gene reconstruction step.
- **Offline Pathway Targets**: Ground-truth pathway activities are pre-computed offline (`stf-compute-pathways`) from raw gene expression using QC → CP10k normalisation → z-score → mean pathway aggregation. This eliminates the circular auxiliary loss used in previous versions.
- **Spatial Pattern Coherence**: Optimised using a composite **MSE + PCC (Pearson Correlation) loss**.
- **Foundation Model Ready**: Native support for **CTransPath**, **Phikon**, **Hibou**, **PLIP**, and **GigaPath**.

---

## License

This project is protected by a **Proprietary Source Code License**. See the [LICENSE](LICENSE) file for full details.

- ✅ **Permitted**: Evaluation for employment, Academic Research, and Non-Profit use.
- 🤝 **For-Profit Use**: Permitted only under a **negotiated agreement** with the author.
  - **Note on Foundation Models**: This architecture is backbone-agnostic. Any negotiated commercial agreement covers *only* the SpatialTranscriptFormer source code and IP. It does **not** grant commercial rights to use restricted third-party weights (e.g., CTransPath, Phikon). To use this system commercially, you must select a foundation model with a compatible open or commercial license (e.g., Hibou, Virchow, or H-Optimus-0).
  - **Note on HEST-1k Dataset**: The benchmark data used in this project is sourced from the **HEST-1k** dataset (Mahmood Lab), which is licensed under **CC BY-NC-SA 4.0**. This data is strictly for non-commercial research and cannot be used for commercial training or clinical deployment without explicit permission from the original authors.
  - **Note on MSigDB**: This project uses data from the **Molecular Signatures Database (MSigDB)** (versions v6.0–v7.5.1, and v2022.1+). The contents are protected by copyright © 2004–2025 Broad Institute, Inc., MIT, and Regents of the University of California, and are distributed under the **CC BY 4.0** license. While this allows for commercial use, users must provide appropriate attribution. Note that individual gene sets within MSigDB may be subject to additional terms from third-party sources (e.g., KEGG).
- ❌ **Prohibited**: Redistribution and unauthorized commercial exploitation.

## Intellectual Property

The core architectural innovations, including the **SpatialTranscriptFormer** interaction logic and spatial masking strategies, are the unique Intellectual Property of the author. For a detailed breakdown, see the [IP Statement](docs/IP_STATEMENT.md).

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

Before training, compute the offline pathway activity matrix for each sample. This step applies per-spot QC, CP10k normalisation, and z-scoring before aggregating gene expression into MSigDB Hallmark pathway scores.

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
stf-predict --data-dir A:\hest_data --sample-id MEND29 --model-path checkpoints/best_model.pth --model-type interaction
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

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our coding standards and the process for submitting pull requests. Note that this project is under a proprietary license; contributions involve an assignment of rights for non-academic use.
