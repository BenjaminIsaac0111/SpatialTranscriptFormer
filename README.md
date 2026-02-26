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

**Before running any commands**, you must activate the conda environment:

```bash
conda activate SpatialTranscriptFormer
```

### Download HEST Data

> [!CAUTION]
> **Authentication Required**: The HEST dataset is gated. You must accept the terms of use at [MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest) and authenticate with your Hugging Face account to download the data.

Please provide your token using ONE of the following methods before running the download tool:

1. **Persistent Login**: Run `huggingface-cli login` and paste your access token when prompted.
2. **Environment Variable**: Set the `HF_TOKEN` environment variable in your active terminal session.

Once authenticated, download specific subsets using filters or the entire dataset:

```bash
# Option 1: Download the ENTIRE HEST dataset (requires confirmation)
stf-download --local_dir hest_data

# Option 2: Download a specific subset (e.g., Bowel Cancer)
stf-download --organ Bowel --disease Cancer --local_dir hest_data

# Option 3: Filter by technology (e.g., Visium)
stf-download --tech Visium --local_dir hest_data
```

To see all available organs in the metadata:

```bash
stf-download --list_organs
```

### Train Models

We provide presets for baseline models and scaled versions of the SpatialTranscriptFormer.

```bash
# Recommended: Run the Interaction model with 4 transformer layers
python scripts/run_preset.py --preset stf_interaction_l4

# Run the lightweight 2-layer version
python scripts/run_preset.py --preset stf_interaction_l2

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

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our coding standards and the process for submitting pull requests. Note that this project is under a proprietary license; contributions involve an assignment of rights for non-academic use.
