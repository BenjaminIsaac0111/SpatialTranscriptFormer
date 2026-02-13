# HEST Dataloader Documentation

The `SpatialTranscriptFormer` uses a custom PyTorch dataloader designed for memory-efficient loading of large-scale spatial transcriptomics datasets.

## Core Implementation Details

The implementation is located in `src/spatial_transcript_former/data/dataset.py`.

### 1. `HEST_Dataset` Class

This class implements the standard `torch.utils.data.Dataset` interface.

- **Lazy Loading**: To avoid overwhelming memory, it uses lazy loading for H5 file handles. File objects are initialized only when the first item is requested (typically within a worker process).
- **Indexing**: It supports an optional `indices` map, which allows it to represent a subset of the original data (e.g., after filtering for valid ST spots) without duplicating arrays in memory.
- **Transformation**: Images are permuted from `(H, W, C)` to `(C, H, W)` and normalized to `[0, 1]`.

### 2. `load_gene_expression_matrix`

This utility function handles the complex process of aligning image patches to gene expression data.

- **Barcode Alignment**: Since not every image patch in an `.h5` file necessarily has a corresponding transcriptomic profile in the `.h5ad` file, the function performs a lookup using the spot barcodes.
- **Gene Selection**: It can either:
  1. Select the top `N` most expressed genes from a single sample.
  2. Align the current sample to a predefined list of global gene names (filling missing genes with zeros).
- **Sparse Support**: It handles both dense and sparse (CSR) matrix formats in the `.h5ad` file.

### 3. `get_hest_dataloader`

The high-level orchestrator that creates a unified dataloader for multiple samples.

- **Sample Concatenation**: It iterates through multiple sample IDs and creates individual `HEST_Dataset` instances, which are then combined using `torch.utils.data.ConcatDataset`.
- **Global Gene Lock**: The first sample found "locks" the gene list (usually the top 1000 genes). Every subsequent sample in the loop is then aligned to this specific set of genes to ensure consistent input dimensions for the model.

## Usage Example

```python
from spatial_transcript_former.data import get_hest_dataloader

# IDs from your metadata split
train_ids = ['MEND29', 'TENX156', ...]

dataloader = get_hest_dataloader(
    root_dir="A:/hest_data",
    ids=train_ids,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    num_genes=1000
)

for patches, gene_counts in dataloader:
    # patches shape: (BS, 3, 224, 224)
    # gene_counts shape: (BS, 1000)
    ...
```

## Stratified Splitting

For robust evaluation, we use `split_hest_patients` in `src/spatial_transcript_former/data/splitting.py`. This ensures that all samples from a single patient go into the same split (Train/Val/Test), preventing data leakage due to biological similarities between slides from the same donor.
