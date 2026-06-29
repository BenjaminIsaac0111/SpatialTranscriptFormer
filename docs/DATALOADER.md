# HEST Dataloader Documentation

The `SpatialTranscriptFormer` uses custom PyTorch dataloaders designed for memory-efficient loading of large-scale spatial transcriptomics datasets. The framework supports two loading paths: loading raw histology patches or loading pre-extracted feature vectors.

## Core Implementation Details

The implementation is located in [dataset.py](../src/spatial_transcript_former/recipes/hest/dataset.py).

### 1. Raw-Patch Loading Path

This path is used when training or evaluating directly on pixel-space images.

*   **`HEST_Dataset` Class**: Loads raw histology patches from a HEST `.h5` file. It supports:
    *   **Lazy File Access**: File handles are created lazily inside each worker process to avoid pickling issues during multiprocessing.
    *   **Neighbourhood Context**: Can retrieve a patch along with its $K$ nearest neighbours.
    *   **Dihedral Augmentation**: Randomly rotates or flips patch pixels and coordinates in sync.
*   **`get_hest_dataloader`**: High-level orchestrator that creates a `DataLoader` over raw patches for a list of sample IDs, combining individual datasets using `ConcatDataset`.
*   **Returned Tuples**: Yields `(patches, None, pathway_acts, rel_coords, mask)`. The second slot is the legacy gene-counts slot (always `None` now); `pathway_acts` holds the pre-computed targets (or `None` when no `pathway_targets_dir` is given).

### 2. Pre-Computed Feature Loading Path

This is the default path used by the SpatialTranscriptFormer training pipeline (`--precomputed`), as it avoids repeated backbone inference.

*   **`HEST_FeatureDataset` Class**: Loads pre-extracted feature vectors (e.g. CTransPath, Phikon) from `.pt` files and aligns them to pre-computed pathway activity targets from `.h5` files.
    *   **Spot barcode alignment**: Filters features to keep only spots that passed quality control (QC) in the corresponding `.h5ad` file.
    *   **Stationary Coordinate Normalisation**: Normalises coordinates relative to the slide's centroid and standard deviation so coordinates are invariant to batching.
    *   **Patch Mode**: Returns a single spot feature vector, its local neighbourhood features (optionally with random dropout augmentation), pre-computed pathway targets, and relative coordinates.
    *   **Whole-Slide Mode**: Returns all spots on the slide as a single sequence.
*   **`get_hest_feature_dataloader`**: Builds a `DataLoader` over the feature datasets.
    *   In **patch mode**, yields standard batched tensors `(feats, None, pathway_acts, coords, mask)`.
    *   In **whole-slide mode**, pads variable-length slides to the longest slide in the batch and appends a boolean padding mask. Yields `(padded_feats, None, padded_pathways, padded_coords, mask)`.

---

## Usage Example (Pre-Computed Features)

```python
from spatial_transcript_former.recipes.hest.dataset import get_hest_feature_dataloader

# Pre-selected training sample IDs
train_ids = ['MEND29', 'TENX156', ...]

dataloader = get_hest_feature_dataloader(
    root_dir="./hest_data",
    ids=train_ids,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    n_neighbors=6,
    pathway_targets_dir="./hest_data/pathway_activities"
)

for feats, _, pathway_acts, rel_coords, mask in dataloader:
    # feats shape:        (BS, 1 + n_neighbors, feature_dim)
    # pathway_acts shape: (BS, num_pathways)
    # rel_coords shape:   (BS, 1 + n_neighbors, 2)
    # mask shape:         (BS, 1 + n_neighbors) bool, True = padded
    ...
```

---

## Patient-Aware Stratified Splitting

To prevent data leakage due to biological similarities between multiple slides from the same donor, splits are stratified by patient. The splitting logic is located in [splitting.py](../src/spatial_transcript_former/recipes/hest/splitting.py) and exposed via the `stf-split` command.
