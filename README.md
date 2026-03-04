# SpatialTranscriptFormer

> [!WARNING]
> **Work in Progress**: This project is under active development. Core architectures, CLI flags, and data formats are subject to major changes.

**SpatialTranscriptFormer** bridges histology and biological pathways through a high-performance transformer architecture. By modeling the dense interplay between morphological features and gene expression signatures, it provides an interpretable and spatially-coherent mapping of the tissue microenvironment.

## Key Technical Pillars

- **Quad-Flow Interaction**: Configurable attention between Pathways and Histology patches (`p2p`, `p2h`, `h2p`, `h2h`).
- **Pathway Bottleneck**: Interpretable gene expression prediction via 50 MSigDB Hallmark tokens.
- **Spatial Pattern Coherence**: Optimized using a composite **MSE + PCC (Pearson Correlation) loss** to prevent spatial collapse and ensure accurate morphology-expression mapping.
- **Foundation Model Ready**: Native support for **CTransPath**, **Phikon**, **Hibou**, and **GigaPath**.
- **Biologically Informed Initialization**: Gene reconstruction weights derived from known hallmark memberships.

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

## Installation

This project requires [Conda](https://docs.conda.io/en/latest/).

1. Clone the repository.
2. Run the automated setup script:
3. On Windows: `.\setup.ps1`
   - On Linux/HPC: `bash setup.sh`

## Usage

### Dataset Access

The model uses the **HEST1k** dataset. You can download specific subsets (by organ, technology, etc.) or the entire dataset using the `stf-download` utility:

```bash
# List available filtering options
stf-download --list-options

# Download a specific subset (e.g., Breast Cancer samples from Visium)
stf-download --organ Breast --disease Cancer --tech Visium --local_dir hest_data

# Download all human samples
stf-download --species "Homo sapiens" --local_dir hest_data
```

> [!NOTE]
> The HEST dataset is gated on Hugging Face. Ensure you have accepted the terms at [MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest) and are logged in via `huggingface-cli login`.

### Train Models

We provide presets for baseline models and scaled versions of the SpatialTranscriptFormer.

```bash
# Recommended: Run the Interaction model (Small)
python scripts/run_preset.py --preset stf_small

# Run the lightweight Tiny version
python scripts/run_preset.py --preset stf_tiny

# Run baselines
python scripts/run_preset.py --preset he2rna_baseline
```

For a complete list of configurations, see the [Training Guide](docs/TRAINING_GUIDE.md).

### Real-Time Monitoring

Monitor training progress, loss curves, and **prediction variance (collapse detector)** via the web dashboard:

```bash
python scripts/monitor.py --run-dir runs/stf_interaction_l4
```

### Inference & Visualization

Generate spatial maps comparing Ground Truth vs Predictions:

```bash
stf-predict --data-dir A:\hest_data --sample-id MEND29 --model-path checkpoints/best_model.pth --model-type interaction
```

Visualization plots will be saved to the `./results` directory.

## Documentation

- [Models](docs/MODELS.md): Detailed model architectures and scaling parameters.
- [Data Structure](docs/DATA_STRUCTURE.md): Organization of HEST data on disk.
- [Pathway Mapping](docs/PATHWAY_MAPPING.md): Clinical interpretability and pathway integration.
- [Gene Analysis](docs/GENE_ANALYSIS.md): Modeling strategies for high-dimensional gene space.

## Development

### Running Tests

```bash
# Run all tests (Pytest wrapper)
.\test.ps1
```

## Future Directions & Clinical Collaborations

A major future direction for **SpatialTranscriptFormer** is to integrate this architecture into an **end-to-end pipeline for patient risk assessment** and prognosis tracking. By leveraging the model's predicted expression and pathway activations, we aim to build a downstream risk prediction module that allows users to directly evaluate how spatially-resolved expression relates to patient survival.

> [!NOTE]
> **Call for Collaborators:** Rigorous risk assessment models require vast datasets of clinical metadata and survival outcomes, which we currently lack access to. We are open to investigating *any* disease of interest! If you have access to large clinical cohorts and are interested in exploring how spatial pathway activation correlates with patient prognosis, we would love to partner with you.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our coding standards and the process for submitting pull requests. Note that this project is under a proprietary license; contributions involve an assignment of rights for non-academic use.
