# Testing Guide

This project includes a comprehensive test suite organized into a hierarchical directory structure that mirrors the source package layout.

## Running Tests

Ensure you have the `SpatialTranscriptFormer` conda environment activated:

```bash
conda activate SpatialTranscriptFormer
pytest tests/
```

Or using the provided wrapper scripts:

```powershell
# Windows
.\test.ps1
```

```bash
# Linux / HPC
bash test.sh
```

## Test Suite Structure

Tests are organized under `tests/` in subdirectories that reflect the source package:

| Directory | Test Files | Coverage Area |
| :--- | :--- | :--- |
| `tests/data/` | `test_data_integrity.py`, `test_pathways.py`, `test_augmentation.py`, `test_visualization.py` | Gene vocabulary, pathway scoring, data augmentation, visualization utilities |
| `tests/models/` | `test_backbones.py`, `test_interactions.py`, `test_compilation.py` | Backbone loading, interaction model logic, `torch.compile` compatibility |
| `tests/training/` | `test_losses.py`, `test_trainer.py`, `test_checkpoints.py`, `test_config.py` | Loss functions (MSE, PCC, composite), training loop, checkpoint serialization |
| `tests/recipes/hest/` | HEST-specific dataset loading and split logic | HEST dataset and splitting |
| `tests/test_api.py` | End-to-end Python API integration test | Full pipeline: model load → inference → AnnData injection |

## Key Test Areas

### Pathway Scoring (`tests/data/test_pathways.py`)

Tests for the offline pathway activity computation pipeline (`compute_pathway_activities.py`):

- Per-spot QC filtering (min UMIs, min genes, max MT%)
- CP10k normalisation correctness
- Z-scoring and mean pathway aggregation
- Moran I calculation
- `.h5` output format and barcode alignment

### Loss Functions (`tests/training/test_losses.py`)

Tests for all loss components used in training:

- `MaskedMSELoss`: Masked and unmasked MSE correctness
- `PCCLoss`: Batch-wise and spatial correlation; N=1 edge case
- `CompositeLoss`: MSE + PCC combination with configurable alpha

### Model Architecture (`tests/models/test_interactions.py`)

- Forward pass shapes for all interaction modes (`p2p`, `p2h`, `h2p`, `h2h`)
- Attention mask correctness
- Dense vs. global prediction output shapes

## Contributor Guidelines: Adding New Tests

When adding new functionality, add corresponding tests in the appropriate `tests/` subdirectory.

- **Framework**: Use `pytest` style tests (plain functions with `assert` statements).
- **Mocking**: Mock external calls (e.g., HuggingFace Hub downloads) to keep tests fast and offline-capable.
- **Fixtures**: Shared fixtures are defined in `tests/conftest.py`.

### Mutation Testing

Standard tests verify expected behaviour; **mutation testing** verifies that your tests are actually catching bugs. We encourage mutation testing for critical components using **[cosmic-ray](https://github.com/sixty-north/cosmic-ray)**:

```bash
pip install cosmic-ray
# Initialize config for your module, then run mutation tests and review survival report
```

If you have ideas for integrating mutation testing into CI, please open a discussion.
