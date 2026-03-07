# Bring Your Own Data

This guide shows how to implement a custom dataset for SpatialTranscriptFormer using the `SpatialDataset` contract.

## Data Contract

Every dataset must subclass `SpatialDataset` and return 3-tuples from `__getitem__`:

```python
(features, gene_counts, rel_coords)
```

| Field | Shape | Description |
| --- | --- | --- |
| `features` | `(S, D)` | Patch embeddings (`S` = 1 + neighbours, `D` = backbone dim) |
| `gene_counts` | `(G,)` | Gene expression targets for the centre patch |
| `rel_coords` | `(S, 2)` | Spatial coordinates relative to centre (centre = `[0, 0]`) |

## Minimal Example

```python
import torch
import numpy as np
from spatial_transcript_former.data.base import SpatialDataset

class MyVisiumDataset(SpatialDataset):
    """Custom dataset for your own spatial transcriptomics data."""

    def __init__(self, features, gene_matrix, coords, gene_names=None):
        """
        Args:
            features: (N, D) pre-extracted backbone features
            gene_matrix: (N, G) gene expression matrix
            coords: (N, 2) spatial coordinates
            gene_names: optional list of G gene symbols
        """
        self._features = torch.as_tensor(features, dtype=torch.float32)
        self._genes = torch.as_tensor(gene_matrix, dtype=torch.float32)
        self._coords = torch.as_tensor(coords, dtype=torch.float32)
        self._gene_names = gene_names

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):
        # Centre patch feature (unsqueeze to get shape (1, D))
        feat = self._features[idx].unsqueeze(0)

        # Gene expression target
        genes = self._genes[idx]

        # Relative coordinate (centre is always [0, 0])
        rel_coord = torch.zeros(1, 2)

        return feat, genes, rel_coord

    @property
    def num_genes(self):
        return self._genes.shape[1]

    @property
    def gene_names(self):
        return self._gene_names
```

## Using with the Trainer

```python
from torch.utils.data import DataLoader, random_split
from spatial_transcript_former import SpatialTranscriptFormer, Predictor
from spatial_transcript_former.training.engine import train_one_epoch, validate
from spatial_transcript_former.training.losses import CompositeLoss

# 1. Create your dataset
dataset = MyVisiumDataset(features, gene_matrix, coords, gene_names=my_genes)

# 2. Split
train_ds, val_ds = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# 3. Build model
model = SpatialTranscriptFormer(
    num_genes=dataset.num_genes,
    backbone_name="phikon",
    pretrained=False,    # backbone weights not needed for pre-extracted features
    use_spatial_pe=True,
).to(device)

# 4. Train
criterion = CompositeLoss(alpha=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                           whole_slide=False)
    metrics = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch}: loss={loss:.4f}, val={metrics['val_loss']:.4f}")

# 5. Save for inference
from spatial_transcript_former import save_pretrained
save_pretrained(model, "./my_model/", gene_names=my_genes)
```

## Preparing Your Data

### From AnnData / Scanpy

```python
import scanpy as sc
import numpy as np

adata = sc.read_h5ad("my_experiment.h5ad")

# Gene expression matrix
gene_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
gene_names = list(adata.var_names)

# Spatial coordinates
coords = adata.obsm["spatial"]

# Pre-extract features using FeatureExtractor
from spatial_transcript_former import FeatureExtractor
extractor = FeatureExtractor(backbone="phikon", device="cuda")
# ... extract patches from WSI and run through extractor
```

### From Raw Patches

If you have image patches as tensors:

```python
from spatial_transcript_former import FeatureExtractor

extractor = FeatureExtractor(backbone="phikon", device="cuda")
features = extractor.extract_batch(patch_tensor, batch_size=64)  # (N, 768)
```
