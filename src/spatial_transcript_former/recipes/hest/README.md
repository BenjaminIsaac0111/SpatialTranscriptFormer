# HEST-1k Recipe (Exemplar)

This directory serves as a comprehensive **exemplar** for training `SpatialTranscriptFormer` on the **HEST-1k** benchmark dataset.

While the core `SpatialTranscriptFormer` framework is dataset-agnostic, this recipe provides a complete, out-of-the-box pipeline for reproducing our benchmarks, including data downloading, preprocessing, and specialized dataloaders.

## Components

- **`dataset.py`**: Contains `HEST_Dataset` and `HEST_FeatureDataset`, which subclass `SpatialDataset` to handle the specific `.h5ad` structure and metadata conventions of the HEST dataset.
- **`io.py`**: Utilities for reading spatial graphs, coordinates, and `.h5ad` matrices.
- **`utils.py`**: HEST-specific dataset setup routines, splitting logic, and vocabulary loading.
- **`download.py`**: Logic for fetching subsets of the gated HEST dataset from Hugging Face.

## Usage

For complete CLI usage and training preset commands, refer to the main **[README.md](../../../../README.md)** and the **[Training Guide](../../../../docs/TRAINING_GUIDE.md)**.
