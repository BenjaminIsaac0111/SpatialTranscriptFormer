# Python API Reference

The SpatialTranscriptFormer package exposes a clean API for loading trained models, running inference on new data, and integrating with the Scanpy/AnnData ecosystem.

```python
from spatial_transcript_former import (
    SpatialTranscriptFormer,   # Core model
    Predictor,                 # Inference wrapper
    FeatureExtractor,          # Backbone feature extraction
    save_pretrained,           # Save checkpoint directory
    load_pretrained,           # Load checkpoint directory
    inject_predictions,        # AnnData integration
)
```

---

## Quick Start

### End-to-End Inference (New Data)

```python
from spatial_transcript_former import SpatialTranscriptFormer, Predictor, FeatureExtractor
from spatial_transcript_former.predict import inject_predictions
import scanpy as sc

# 1. Load model from checkpoint directory
model = SpatialTranscriptFormer.from_pretrained("./checkpoints/my_run/")
print(model.gene_names[:3])  # ['TP53', 'EGFR', 'MYC']

# 2. Extract features from raw patches
extractor = FeatureExtractor(backbone="phikon", device="cuda")
features = extractor.extract_batch(image_tensor, batch_size=64)  # (N, 768)

# 3. Predict gene expression
predictor = Predictor(model, device="cuda")
predictions = predictor.predict_wsi(features, coords)  # (1, G)

# 4. Inject into AnnData for Scanpy analysis
adata = sc.AnnData(obs=pd.DataFrame(index=[f"spot_{i}" for i in range(N)]))
inject_predictions(adata, coords, predictions[0], gene_names=model.gene_names)
sc.pl.spatial(adata, color="TP53")
```

### Saving a Trained Model

```python
from spatial_transcript_former import save_pretrained

# After training, export a self-contained checkpoint
save_pretrained(model, "./release/v1/", gene_names=gene_list)
```

This creates:

```
release/v1/
├── config.json        # Architecture parameters
├── model.pth          # Model weights (state_dict)
└── gene_names.json    # Ordered gene symbols
```

---

## API Reference

### `SpatialTranscriptFormer`

The core transformer model. Predicts gene expression from histology patch features and spatial coordinates.

#### `SpatialTranscriptFormer.__init__(...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_genes` | `int` | *required* | Number of output genes |
| `num_pathways` | `int` | `50` | Number of pathway bottleneck tokens |
| `backbone_name` | `str` | `"resnet50"` | Backbone identifier (`resnet50`, `phikon`, `ctranspath`, etc.) |
| `pretrained` | `bool` | `True` | Load pretrained backbone weights |
| `token_dim` | `int` | `256` | Common embedding dimension |
| `n_heads` | `int` | `4` | Number of attention heads |
| `n_layers` | `int` | `2` | Number of transformer layers |
| `dropout` | `float` | `0.1` | Dropout probability |
| `pathway_init` | `Tensor` | `None` | `(P, G)` biological pathway membership matrix |
| `use_spatial_pe` | `bool` | `True` | Enable learned spatial positional encodings |
| `output_mode` | `str` | `"counts"` | Output head: `"counts"` (Softplus) or `"zinb"` (Zero-Inflated NB) |
| `interactions` | `list[str]` | all | Attention interactions: `p2p`, `p2h`, `h2p`, `h2h` |

#### `SpatialTranscriptFormer.from_pretrained(checkpoint_dir, device="cpu", **kwargs)`

Load a model from a checkpoint directory created by `save_pretrained`.

```python
model = SpatialTranscriptFormer.from_pretrained("./checkpoint/", device="cuda")
model.gene_names  # List[str] or None
```

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_dir` | `str` | Path to directory with `config.json` + `model.pth` |
| `device` | `str` | Torch device (`"cpu"`, `"cuda"`) |
| `**kwargs` | | Override any `config.json` value (e.g. `dropout=0.0`) |

**Returns:** `SpatialTranscriptFormer` in eval mode with `.gene_names` attribute.

---

### `Predictor`

Stateful inference wrapper. Manages device placement, eval mode, and optional AMP.

#### `Predictor.__init__(model, device="cpu", use_amp=False)`

```python
predictor = Predictor(model, device="cuda", use_amp=True)
```

#### `Predictor.predict_patch(image, return_pathways=False)`

Single-patch inference from a raw image tensor.

```python
result = predictor.predict_patch(image)   # image: (1, 3, 224, 224) or (3, 224, 224)
# result: (1, num_genes)
```

> **Note:** When the model uses spatial PE, a zero-coordinate is automatically injected — no need to provide coordinates for single patches.

#### `Predictor.predict_wsi(features, coords, return_pathways=False, return_dense=False)`

Whole-slide inference from pre-extracted feature embeddings.

```python
# Global prediction (one vector per slide)
result = predictor.predict_wsi(features, coords)           # (1, G)

# Dense prediction (one vector per patch)
result = predictor.predict_wsi(features, coords, return_dense=True)  # (1, N, G)
```

| Parameter | Type | Description |
|---|---|---|
| `features` | `Tensor` | `(N, D)` or `(1, N, D)` embeddings |
| `coords` | `Tensor` | `(N, 2)` or `(1, N, 2)` spatial coordinates |
| `return_pathways` | `bool` | Also return pathway scores |
| `return_dense` | `bool` | Per-patch predictions instead of global |

> **Validation:** Raises `ValueError` with a clear message if the feature dimension doesn't match the model's expected backbone dimension.

#### `Predictor.predict(features, coords=None, **kwargs)`

Unified entry point — auto-dispatches:

- 4D tensor `(B, 3, H, W)` → `predict_patch`
- 2D tensor `(N, D)` → `predict_wsi` (requires `coords`)

---

### `FeatureExtractor`

Wraps a backbone model and its normalization transform for one-line feature extraction.

#### `FeatureExtractor.__init__(backbone="resnet50", device="cpu", pretrained=True, transform=None)`

```python
extractor = FeatureExtractor(backbone="phikon", device="cuda")
extractor.feature_dim   # 768
extractor.backbone_name # "phikon"
```

| Backbone | `feature_dim` | Source |
|---|---|---|
| `resnet50` | 2048 | torchvision |
| `ctranspath` | 768 | HuggingFace (CTransPath) |
| `phikon` | 768 | Owkin Phikon (HuggingFace) |
| `vit_b_16` | 768 | torchvision |
| `gigapath` | 1536 | ProvGigaPath *(gated)* |
| `hibou-b` | 768 | Hibou-B *(gated)* |
| `hibou-l` | 1024 | Hibou-L *(gated)* |

#### `extractor(images)` / `extractor.extract_batch(images, batch_size=64)`

```python
features = extractor(images)                           # (N, D) — all at once
features = extractor.extract_batch(images, batch_size=64)  # batched, returns on CPU
```

Images should be float tensors in `[0, 1]` range, shape `(N, 3, H, W)`.

---

### `save_pretrained(model, save_dir, gene_names=None)`

Save a self-contained checkpoint directory.

```python
save_pretrained(model, "./release/v1/", gene_names=["TP53", "EGFR", ...])
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `SpatialTranscriptFormer` | Trained model instance |
| `save_dir` | `str` | Output directory (created if needed) |
| `gene_names` | `list[str]` | Optional ordered gene symbols (must match `num_genes`) |

**Raises:** `ValueError` if `gene_names` length doesn't match `num_genes`.

### AnnData & Scanpy — A Primer

If you're coming from a pure deep-learning background, AnnData and Scanpy may be unfamiliar. They are the standard data format and analysis toolkit in single-cell and spatial biology — the equivalent of what Pandas DataFrames are for tabular ML.

#### What is AnnData?

An `AnnData` object is a structured container for observations × variables matrices, designed for genomics. Think of it as a spreadsheet with labelled sidecars:

```
                     var (genes)
                ┌──────────────────┐
                │  TP53  EGFR  MYC │
           ┌────┼──────────────────┤
 obs       │ s0 │  0.3   1.2  0.8 │  ← adata.X  (the main data matrix)
 (spots/   │ s1 │  0.1   0.5  1.1 │
  cells)   │ s2 │  0.9   0.2  0.4 │
           └────┴──────────────────┘
```

| Slot | What it stores | Our usage |
|---|---|---|
| `adata.X` | Main matrix `(N, G)` | Predicted gene expression |
| `adata.obs` | Per-observation metadata | Spot/cell barcodes, cluster labels |
| `adata.var` | Per-variable metadata | Gene symbols as the index |
| `adata.obsm["spatial"]` | Observation-level embeddings | `(N, 2)` spatial coordinates |
| `adata.obsm["spatial_pathways"]` | Additional embeddings | `(N, P)` pathway scores |
| `adata.uns` | Unstructured metadata | Pathway names, model config |

#### What is Scanpy?

[Scanpy](https://scanpy.readthedocs.io/) (`sc`) is the analysis library that operates on AnnData objects. Once predictions are inside an `adata`, you instantly get access to:

```python
import scanpy as sc

# Spatial plotting — visualise gene expression on tissue coordinates
sc.pl.spatial(adata, color="TP53")

# Clustering — find groups of spots with similar expression
sc.tl.leiden(adata)

# Differential expression — find marker genes per cluster
sc.tl.rank_genes_groups(adata, groupby="leiden")

# Dimensionality reduction
sc.tl.pca(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color="leiden")
```

#### Why does this matter?

By injecting predictions into AnnData, our model's output becomes **instantly compatible** with the entire Scanpy ecosystem — clustering, differential testing, spatial plotting, trajectory analysis — without any custom code. Biologists can take our predictions and run their standard workflows immediately.

---

### `inject_predictions(adata, coords, predictions, ...)`

Inject predictions into an AnnData object for Scanpy integration.

```python
inject_predictions(
    adata,
    coords,                              # → adata.obsm["spatial"]
    predictions,                         # → adata.X
    gene_names=["TP53", "EGFR", ...],    # → adata.var_names
    pathway_scores=pathway_activations,  # → adata.obsm["spatial_pathways"]
    pathway_names=["APOPTOSIS", ...],    # → adata.uns["pathway_names"]
)
```

| Parameter | Type | Description |
|---|---|---|
| `adata` | `AnnData` | Target AnnData object |
| `coords` | `ndarray` | `(N, 2)` spatial coordinates |
| `predictions` | `ndarray` | `(N, G)` gene predictions |
| `gene_names` | `list[str]` | Optional gene symbols |
| `pathway_scores` | `ndarray` | Optional `(N, P)` pathway scores |
| `pathway_names` | `list[str]` | Optional pathway names |

> **Lazy loading:** `anndata` is only imported when this function is called, so it's not required for basic inference.

---

## Checkpoint Directory Format

```
checkpoint/
├── config.json         # Architecture (JSON)
├── model.pth           # Weights (PyTorch state_dict)
└── gene_names.json     # Gene symbols (JSON array, optional)
```

**`config.json` example:**

```json
{
  "num_genes": 460,
  "num_pathways": 50,
  "backbone_name": "phikon",
  "token_dim": 256,
  "n_heads": 4,
  "n_layers": 2,
  "dropout": 0.1,
  "use_spatial_pe": true,
  "output_mode": "counts",
  "interactions": ["h2h", "h2p", "p2h", "p2p"]
}
```

**`gene_names.json` example:**

```json
["TP53", "EGFR", "MYC", "BRCA1", ...]
```

---

## Training API

The training pipeline lives in the `spatial_transcript_former.training` subpackage. You can use it via the **CLI** or **programmatically** in your own scripts.

### CLI Quick Start

Training is launched via the `stf-train` entry point (or `python -m spatial_transcript_former.train`):

```bash
# Minimal: train on precomputed features with whole-slide mode
stf-train \
    --model interaction \
    --backbone phikon \
    --data-dir /path/to/hest \
    --precomputed \
    --whole-slide \
    --use-spatial-pe \
    --pathway-init \
    --loss mse_pcc \
    --epochs 100 \
    --lr 1e-4 \
    --warmup-epochs 10

# Resume from checkpoint
stf-train --model interaction --resume --output-dir ./checkpoints
```

### CLI Arguments

#### Data

| Flag | Default | Description |
| --- | --- | --- |
| `--data-dir` | from config | Root HEST data directory |
| `--feature-dir` | auto | Explicit pre-extracted feature directory |
| `--num-genes` | 1000 | Number of output genes |
| `--precomputed` | off | Use pre-extracted backbone features |
| `--whole-slide` | off | Dense whole-slide prediction mode |
| `--organ` | all | Filter samples by organ type |
| `--max-samples` | all | Limit samples (for debugging) |

#### Model

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `he2rna` | Architecture: `interaction`, `he2rna`, `vit_st`, `attention_mil`, `transmil` |
| `--backbone` | `resnet50` | Backbone: `resnet50`, `phikon`, `ctranspath`, `vit_b_16`, etc. |
| `--num-pathways` | 50 | Pathway bottleneck tokens |
| `--token-dim` | 256 | Embedding dimension |
| `--n-heads` | 4 | Attention heads |
| `--n-layers` | 2 | Transformer layers |
| `--use-spatial-pe` | off | Learned spatial positional encoding |
| `--interactions` | all | Attention mask: `p2p p2h h2p h2h` |
| `--pathway-init` | off | Initialize gene head from MSigDB Hallmarks |

#### Training

| Flag | Default | Description |
| --- | --- | --- |
| `--epochs` | 10 | Total training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--warmup-epochs` | 10 | Linear warmup before cosine annealing |
| `--weight-decay` | 0.0 | AdamW weight decay |
| `--grad-accum-steps` | 1 | Gradient accumulation steps |
| `--use-amp` | off | Mixed precision (FP16) |
| `--compile` | off | `torch.compile` the model |
| `--resume` | off | Resume from latest checkpoint |

#### Loss

| Flag | Default | Description |
| --- | --- | --- |
| `--loss` | `mse_pcc` | Loss function: `mse`, `pcc`, `mse_pcc`, `zinb`, `poisson`, `logcosh` |
| `--pcc-weight` | 1.0 | PCC term weight in `mse_pcc` |
| `--pathway-loss-weight` | 0.0 | Auxiliary pathway PCC loss weight (0 = disabled) |

---

### Programmatic Training

For custom training loops, use the building blocks directly:

```python
from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.training.engine import train_one_epoch, validate
from spatial_transcript_former.training.losses import CompositeLoss
from spatial_transcript_former.training.experiment_logger import ExperimentLogger
from spatial_transcript_former.training.checkpoint import save_checkpoint, load_checkpoint

# 1. Build model
model = SpatialTranscriptFormer(
    num_genes=460,
    backbone_name="phikon",
    pretrained=True,
    token_dim=256,
    n_layers=2,
    use_spatial_pe=True,
).to(device)

# 2. Loss & Optimizer
criterion = CompositeLoss(alpha=1.0)  # MSE + PCC
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3. Training loop
for epoch in range(100):
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device,
        whole_slide=True, scaler=scaler, grad_accum_steps=4,
    )
    val_metrics = validate(
        model, val_loader, criterion, device,
        whole_slide=True, use_amp=True,
    )
    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_metrics['val_loss']:.4f}")
```

---

### Loss Functions (`training.losses`)

All losses accept `(B, G)` patch-level or `(B, N, G)` dense inputs, with optional `mask` for padded positions.

| Class | Formula / Description |
| --- | --- |
| `MaskedMSELoss` | Standard MSE with optional padding mask |
| `PCCLoss` | `1 - mean(PCC)` — gene-wise spatial Pearson correlation |
| `CompositeLoss` | `MSE + α · (1 - PCC)` — balances magnitude and spatial pattern |
| `ZINBLoss` | Zero-Inflated Negative Binomial NLL — for raw count data |
| `MaskedHuberLoss` | Huber (SmoothL1) — robust to outlier spots |
| `AuxiliaryPathwayLoss` | Wraps any base loss + PCC on pathway bottleneck scores |

### Training Engine (`training.engine`)

| Function | Description |
| --- | --- |
| `train_one_epoch(model, loader, criterion, optimizer, device, ...)` | One epoch of training. Handles gradient accumulation, AMP, and both patch/WSI modes. Returns average loss. |
| `validate(model, loader, criterion, device, ...)` | Validation pass. Returns `dict` with `val_loss`, `val_mae`, `val_pcc`, `pred_variance`, and optional `attn_correlation`. |

### Experiment Logger (`training.experiment_logger`)

Offline-friendly logger (no W&B dependency). Writes metrics to SQLite and a JSON summary.

```python
logger = ExperimentLogger(output_dir, config_dict)
logger.log_epoch(epoch, {"train_loss": 0.1, "val_loss": 0.2, "val_pcc": 0.65})
logger.finalize(best_val_loss=0.15)
```

| Output File | Contents |
| --- | --- |
| `training_logs.sqlite` | Per-epoch metrics table |
| `results_summary.json` | Config + final metrics + runtime |

### Checkpoint Lifecycle

During training, checkpoints are managed by `training.checkpoint` (the *internal* module — distinct from the public `save_pretrained`):

| Function | Purpose |
| --- | --- |
| `save_checkpoint(model, optimizer, scaler, schedulers, ...)` | Saves full training state for `--resume` |
| `load_checkpoint(model, optimizer, scaler, schedulers, ...)` | Restores training state |

After training is complete, use the public `save_pretrained` to export a clean, inference-ready checkpoint:

```python
from spatial_transcript_former import save_pretrained

# Export for inference (strips optimizer/scheduler state)
save_pretrained(model, "./release/v1/", gene_names=gene_list)
```
