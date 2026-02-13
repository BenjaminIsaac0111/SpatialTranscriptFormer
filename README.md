# SpatialTranscriptFormer

A transformer-based model for spatial transcriptomics.

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
   - On Windows: `.\setup.ps1`
   - On Linux/HPC: `bash setup.sh`

## Usage

After installation, the following command-line tools are available in your `SpatialTranscriptFormer` environment:

### Download HEST Data

Download specific subsets using filters or patterns:

```bash
# List available organs
stf-download --list_organs

# Download only the Bowel Cancer subset (including ST data and WSIs)
stf-download --organ Bowel --disease Cancer --local_dir hest_data

# Download any other organ
stf-download --organ Kidney
```

### Split Dataset

Perform patient-stratified splitting on the metadata:

```powershell
stf-split HEST_v1_3_0.csv --val_ratio 0.2
```

### Train Models

Train baseline models (HE2RNA, ViT) or the proposed interaction model. For a complete list of configurations and examples, see the [Training Guide](docs/TRAINING_GUIDE.md).

```bash
# Option 1: Using the standard command
stf-train --data-dir A:\hest_data --model he2rna --epochs 20

# Option 2: Using the preset launcher (recommended for complex models)
python scripts/run_preset.py --preset stf_interaction --epochs 30
```

### Inference & Visualization

Generate spatial maps comparing Ground Truth vs Predictions for specific samples:

```bash
stf-predict --data-dir A:\hest_data --sample-id MEND29 --model-path checkpoints/best_model_he2rna.pth --model-type he2rna
```

Visualization plots will be saved to the `./results` directory.

## Documentation

For detailed information on the data and code implementation, see:

- [Data Structure](docs/DATA_STRUCTURE.md): Organization of HEST data on disk.
- [Dataloader](docs/DATALOADER.md): Technical implementation of the PyTorch dataset and loaders.
- [Gene Analysis](docs/GENE_ANALYSIS.md): Analysis of available genes and modeling strategies.
- [Pathway Mapping](docs/PATHWAY_MAPPING.md): Strategies for clinical interpretability and pathway integration.
- [Latent Discovery](docs/LATENT_DISCOVERY.md): Unsupervised discovery of biological pathways from data.
- [Models](docs/MODELS.md): Model architectures and literature references.

## Development

### Running Tests

Use the included test wrapper:

```bash
# Run all tests
.\test.ps1
```
