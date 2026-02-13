# HEST Data Structure

This document describes how the HEST (Heterogeneous Evaluation of Spatial Transcriptomics) dataset is organized on disk and the purpose of each component.

## Root Directory Organization

The dataset is typically stored in a `hest_data/` directory (e.g., `A:\hest_data`).

```text
hest_data/
├── HEST_v1_3_0.csv        # Central metadata registry
├── README.md              # Dataset documentation from source
├── patches/               # H5 files containing image patches and local coordinates
│   ├── MEND29.h5
│   └── ...
├── st/                    # H5AD files containing gene expression data
│   ├── MEND29.h5ad
│   └── ...
├── wsis/                  # Whole Slide Images (WSIs) in .tif or .svs format
├── cellvit_seg/           # Nuclear segmentation maps (processed from CellViT)
├── spatial_plots/         # Visualization plots (optional)
└── ...
```

## Core Components

### 1. Metadata Registry (`HEST_v1_3_0.csv`)
The source of truth for all samples. It contains:
- `id`: Unique identifier for the sample (e.g., `MEND29`).
- `organ`: Primary organ (e.g., `Bowel`, `Breast`).
- `species`: Species (e.g., `Homo sapiens`).
- `st_technology`: Technology used (e.g., `Visium`, `Xenium`).
- `patient`: Patient ID for stratified splitting.

### 2. Patches (`patches/*.h5`)
Each H5 file corresponds to one sample and contains:
- `img`: `(N, 224, 224, 3)` array of image patches (uint8).
- `coords`: `(N, 2)` array of local (x, y) coordinates for each patch.
- `barcode`: `(N, 1)` array of spot barcodes corresponding to the ST data.

### 3. Spatial Transcriptomics (`st/*.h5ad`)
The transcriptome data preserved in AnnData format:
- `X`: Sparse or dense matrix of gene expression counts.
- `obs`: Observations metadata (containing barcodes in `_index` or `index`).
- `var`: Variable metadata (containing gene names in `_index` or `index`).

### 4. Segmentation & WSIs
- **WSIs**: High-resolution scans found in `wsis/`.
- **Segmentation**: Pre-computed nuclei masks found in `cellvit_seg/` or technology-specific folders like `xenium_seg/`.

## Data Handling in SpatialTranscriptFormer

The model primarily interfaces with the `patches/` and `st/` directories. During training, the `HEST_Dataset` class aligns the image patches in the `.h5` file with the transcriptomic profiles in the `.h5ad` file using the barcodes as keys.
