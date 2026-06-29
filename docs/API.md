# Python API Reference

The SpatialTranscriptFormer package exposes a clean API for training, inference, and integration with the Scanpy/AnnData ecosystem. The model predicts spatially-resolved **biological pathway activity scores** (not gene expression) from histology features.

```python
from spatial_transcript_former import (
    SpatialTranscriptFormer,   # Core model
    Trainer,                   # High-level training orchestrator
    Predictor,                 # Inference wrapper
    FeatureExtractor,          # Backbone feature extraction
    save_pretrained,           # Save a checkpoint directory
    load_pretrained,           # Load a checkpoint directory
    inject_predictions,        # AnnData integration
)
```

---

## Quick Start

### End-to-End Inference (New Data)

```python
import pandas as pd
import scanpy as sc
from spatial_transcript_former import SpatialTranscriptFormer, Predictor, FeatureExtractor
from spatial_transcript_former.predict import inject_predictions

# 1. Load a model from a checkpoint directory (config.json + model.pth)
model = SpatialTranscriptFormer.from_pretrained("./checkpoints/my_run/")
print(model.pathway_names[:3])  # e.g. ['HALLMARK_HYPOXIA', ...] or None

# 2. Extract features from raw patches
extractor = FeatureExtractor(backbone="phikon", device="cuda")
features = extractor.extract_batch(image_tensor, batch_size=64)  # (N, 768)

# 3. Predict per-spot pathway activity from the features
predictor = Predictor(model, device="cuda")
preds = predictor.predict_wsi(features, coords, return_dense=True)  # (1, N, P)

# 4. Inject into AnnData for Scanpy analysis (one activity vector per spot)
adata = sc.AnnData(obs=pd.DataFrame(index=[f"spot_{i}" for i in range(len(coords))]))
inject_predictions(adata, coords, preds[0], pathway_names=model.pathway_names)
sc.pl.spatial(adata, color="HALLMARK_HYPOXIA")
```

### Saving a Trained Model

```python
from spatial_transcript_former import save_pretrained

# After training, export a self-contained checkpoint
save_pretrained(model, "./release/v1/", pathway_names=pathway_list)
```

This creates:

```
release/v1/
├── config.json          # Architecture parameters (+ pathway_format_version)
├── model.pth            # Model weights (state_dict)
└── pathway_names.json   # Ordered pathway names (optional)
```

---

## API Reference

### `SpatialTranscriptFormer`

The core transformer. Predicts pathway activity scores from histology patch features and spatial coordinates.

#### `SpatialTranscriptFormer(...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_pathways` | `int` | `50` | Number of pathway tokens **and** the output dimension |
| `backbone_name` | `str` | `"resnet50"` | Backbone identifier (`resnet50`, `phikon`, `ctranspath`, `gigapath`, `hibou-b`, …) |
| `pretrained` | `bool` | `True` | Load pretrained backbone weights |
| `token_dim` | `int` | `256` | Transformer embedding dimension |
| `n_heads` | `int` | `4` | Number of attention heads |
| `n_layers` | `int` | `2` | Transformer layers (minimum 2 when `h2h` is disabled) |
| `dropout` | `float` | `0.1` | Dropout probability |
| `use_spatial_pe` | `bool` | `True` | Enable learned 2-D spatial positional encodings (requires `rel_coords` at call time) |
| `interactions` | `list[str]` | all four | Attention quadrants to enable: `p2p`, `p2h`, `h2p`, `h2h` |

> The model predicts pathway activity **directly** via a scaled dot-product + Softplus head — there is no `num_genes`, gene-reconstruction head, or `output_mode`. See [MODELS.md](MODELS.md).

#### `forward(x, rel_coords=None, mask=None, return_dense=False, return_attention=False)`

| Arg | Description |
|---|---|
| `x` | `(B, 3, H, W)` image patch, **or** `(B, S, D)` pre-computed features |
| `rel_coords` | `(B, S, 2)` slide-stationary coordinates (required when `use_spatial_pe=True`) |
| `mask` | `(B, S)` bool padding mask (`True` = padded/ignore) |
| `return_dense` | If `True`, return per-patch predictions instead of a pooled slide vector |
| `return_attention` | If `True`, also return a list of per-layer attention maps |

**Returns:** `(B, num_pathways)`, or `(B, S, num_pathways)` if `return_dense=True`. If `return_attention=True`, returns `(scores, attentions)`.

#### `SpatialTranscriptFormer.from_pretrained(checkpoint_dir, device="cpu", **kwargs)`

Load a model from a checkpoint directory created by `save_pretrained`.

```python
model = SpatialTranscriptFormer.from_pretrained("./checkpoint/", device="cuda")
model.pathway_names  # list[str] or None
```

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_dir` | `str` | Directory with `config.json` + `model.pth` |
| `device` | `str` | Torch device (`"cpu"`, `"cuda"`) |
| `**kwargs` | | Override any `config.json` value (e.g. `dropout=0.0`) |

**Returns:** `SpatialTranscriptFormer` in eval mode with a `.pathway_names` attribute.

---

### `Predictor`

Stateful inference wrapper. Manages device placement, eval mode, and optional AMP.

#### `Predictor(model, device="cpu", use_amp=False)`

```python
predictor = Predictor(model, device="cuda", use_amp=True)
predictor.pathway_names  # forwarded from the model, or None
```

#### `Predictor.predict_patch(image)`

Single-patch inference from a raw image tensor.

```python
result = predictor.predict_patch(image)   # image: (1, 3, 224, 224) or (3, 224, 224)
# result: (1, num_pathways)
```

> When the model uses spatial PE, a zero coordinate is injected automatically — no coordinates are needed for single patches.

#### `Predictor.predict_wsi(features, coords, return_dense=False)`

Whole-slide inference from pre-extracted feature embeddings. Coordinates are re-centred/standardised internally to match the training-time slide-stationary scaling.

```python
result = predictor.predict_wsi(features, coords)                    # (1, P)  — pooled slide vector
result = predictor.predict_wsi(features, coords, return_dense=True) # (1, N, P) — per-spot
```

| Parameter | Type | Description |
|---|---|---|
| `features` | `Tensor` | `(N, D)` or `(1, N, D)` embeddings |
| `coords` | `Tensor` | `(N, 2)` or `(1, N, 2)` spatial coordinates |
| `return_dense` | `bool` | Per-patch predictions instead of a pooled slide vector |

> Raises `ValueError` with a clear message if the feature dimension doesn't match the model's expected backbone dimension.

#### `Predictor.predict(features, coords=None, **kwargs)`

Unified entry point — dispatches a `(B, 3, H, W)` image to `predict_patch`, otherwise to `predict_wsi` (requires `coords`).

---

### `FeatureExtractor`

Wraps a backbone model and its normalization transform for one-line feature extraction.

#### `FeatureExtractor(backbone="resnet50", device="cpu", pretrained=True, transform=None)`

```python
extractor = FeatureExtractor(backbone="phikon", device="cuda")
extractor.feature_dim    # 768
extractor.backbone_name  # "phikon"
```

| Backbone | `feature_dim` | Source |
|---|---|---|
| `resnet50` | 2048 | torchvision |
| `ctranspath` | 768 | CTransPath |
| `phikon` | 768 | Owkin Phikon |
| `vit_b_16` | 768 | torchvision |
| `gigapath` | 1536 | Prov-GigaPath *(gated)* |
| `hibou-b` | 768 | Hibou-B *(gated)* |
| `hibou-l` | 1024 | Hibou-L *(gated)* |

#### `extractor(images)` / `extractor.extract_batch(images, batch_size=64)`

```python
features = extractor(images)                              # (N, D) — all at once, on device
features = extractor.extract_batch(images, batch_size=64) # (N, D) — batched, returned on CPU
```

Images should be float tensors in `[0, 1]`, shape `(N, 3, H, W)`.

---

### `save_pretrained(model, save_dir, pathway_names=None)`

Save a self-contained checkpoint directory.

```python
save_pretrained(model, "./release/v1/", pathway_names=["HALLMARK_HYPOXIA", ...])
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `SpatialTranscriptFormer` | Trained model instance |
| `save_dir` | `str` | Output directory (created if needed) |
| `pathway_names` | `list[str]` | Optional ordered pathway names (must match `num_pathways`) |

**Raises:** `ValueError` if `pathway_names` length doesn't match `num_pathways`.

### `load_pretrained(checkpoint_dir, device="cpu", **override_kwargs)`

Reconstruct a model from `config.json` + `model.pth` (and optional `pathway_names.json`). `**override_kwargs` override config values. **Raises** `ValueError` if the checkpoint's `pathway_format_version` doesn't match the current preprocessing format (re-train against current targets).

> `SpatialTranscriptFormer.from_pretrained(...)` is a thin wrapper around `load_pretrained`.

### AnnData & Scanpy — A Primer

If you're coming from a pure deep-learning background, AnnData and Scanpy may be unfamiliar. They are the standard data format and analysis toolkit in single-cell and spatial biology — the equivalent of what Pandas DataFrames are for tabular ML.

#### What is AnnData?

An `AnnData` object is a structured container for observations × variables matrices, designed for genomics. Think of it as a spreadsheet with labelled sidecars:

```
                  var (pathways)
             ┌──────────────────────┐
             │  HYPOXIA  EMT  MYC   │
        ┌────┼──────────────────────┤
 obs    │ s0 │   0.3    1.2   0.8   │  ← adata.X  (the main data matrix)
 (spots)│ s1 │   0.1    0.5   1.1   │
        │ s2 │   0.9    0.2   0.4   │
        └────┴──────────────────────┘
```

| Slot | What it stores | Our usage |
|---|---|---|
| `adata.X` | Main matrix `(N, P)` | Predicted pathway activity |
| `adata.obs` | Per-observation metadata | Spot/cell barcodes, cluster labels |
| `adata.var` | Per-variable metadata | Pathway names as the index |
| `adata.obsm["spatial"]` | Observation-level embeddings | `(N, 2)` spatial coordinates |
| `adata.uns` | Unstructured metadata | Model config, run info |

#### What is Scanpy?

[Scanpy](https://scanpy.readthedocs.io/) (`sc`) is the analysis library that operates on AnnData objects. Once predictions are inside an `adata`, you get clustering, differential testing, spatial plotting, and trajectory analysis for free:

```python
import scanpy as sc

sc.pl.spatial(adata, color="HALLMARK_HYPOXIA")  # spatial pathway-activity map
sc.tl.leiden(adata)                              # cluster spots by activity profile
sc.tl.rank_genes_groups(adata, groupby="leiden")  # marker pathways per cluster
```

By injecting predictions into AnnData, the model's output becomes instantly compatible with the entire Scanpy ecosystem.

---

### `inject_predictions(adata, coords, pathway_scores, pathway_names=None)`

Inject pathway-activity predictions into an AnnData object for Scanpy integration.

```python
inject_predictions(
    adata,
    coords,                                  # → adata.obsm["spatial"]
    pathway_scores,                          # → adata.X   (N, P)
    pathway_names=["HALLMARK_HYPOXIA", ...], # → adata.var_names
)
```

| Parameter | Type | Description |
|---|---|---|
| `adata` | `AnnData` | Target object (must have `N` observations matching `coords`) |
| `coords` | `ndarray` | `(N, 2)` spatial coordinates |
| `pathway_scores` | `ndarray` | `(N, P)` predicted pathway activity |
| `pathway_names` | `list[str]` | Optional `P` pathway names (set as `adata.var_names`) |

> **Lazy loading:** `anndata` is imported only when this function is called, so it isn't required for basic inference.

---

## Checkpoint Directory Format

```
checkpoint/
├── config.json          # Architecture (JSON)
├── model.pth            # Weights (PyTorch state_dict)
└── pathway_names.json   # Pathway names (JSON array, optional)
```

**`config.json` example:**

```json
{
  "num_pathways": 50,
  "backbone_name": "phikon",
  "token_dim": 256,
  "n_heads": 4,
  "n_layers": 2,
  "dropout": 0.1,
  "use_spatial_pe": true,
  "interactions": ["h2h", "h2p", "p2h", "p2p"],
  "pathway_format_version": 2
}
```

`pathway_format_version` records the preprocessing/target schema the model was trained against; `load_pretrained` refuses checkpoints whose version doesn't match the current pipeline.

---

## Training API

The training pipeline lives in the `spatial_transcript_former.training` subpackage. Use it via the **CLI** or **programmatically**.

### CLI Quick Start

Training is launched via the `stf-train` entry point (or `python -m spatial_transcript_former.train`):

```bash
stf-train \
    --model interaction \
    --backbone phikon \
    --data-dir /path/to/hest \
    --precomputed \
    --whole-slide \
    --use-spatial-pe \
    --loss mse_ccc \
    --epochs 100 \
    --lr 1e-4 \
    --warmup-epochs 10
```

See the **[Training Guide](TRAINING_GUIDE.md)** for the full, authoritative flag reference and HEST recipes. Pathway targets must be pre-computed first with `stf-compute-pathways` (see [PATHWAY_MAPPING.md](PATHWAY_MAPPING.md)).

### Trainer (High-Level)

`Trainer` handles LR scheduling (linear warmup → cosine), AMP, checkpointing, SQLite logging, and early stopping. All arguments after `criterion` are **keyword-only**.

```python
from spatial_transcript_former import SpatialTranscriptFormer, Trainer
from spatial_transcript_former.training import CompositeLoss, EarlyStoppingCallback

model = SpatialTranscriptFormer(num_pathways=50, backbone_name="phikon")

trainer = Trainer(
    model=model,
    train_loader=train_dl,
    val_loader=val_dl,
    criterion=CompositeLoss(alpha=1.0, pcc_type="ccc"),
    epochs=100,
    warmup_epochs=10,
    device="cuda",
    output_dir="./checkpoints",
    model_name="interaction",
    use_amp=True,
    whole_slide=True,
    callbacks=[EarlyStoppingCallback(patience=15)],
)
results = trainer.fit()                    # {"best_val_loss", "epochs_completed", "history"}
trainer.save_pretrained("./release/v1/", pathway_names=pathway_list)
```

#### Trainer Parameters

| Parameter | Default | Description |
|---|---|---|
| `model` | *required* | Any `nn.Module` |
| `train_loader` / `val_loader` | *required* | Training / validation `DataLoader` |
| `criterion` | *required* | Loss function |
| `optimizer` | `None` | Custom optimizer (default: `AdamW` from `lr` / `weight_decay`) |
| `lr` | `1e-4` | Learning rate (when no custom optimizer) |
| `weight_decay` | `0.0` | Weight decay (when no custom optimizer) |
| `epochs` | `100` | Total epochs |
| `warmup_epochs` | `10` | Linear warmup before cosine annealing |
| `device` | `"cuda"` | Torch device |
| `output_dir` | `"./checkpoints"` | Directory for checkpoints/logs |
| `model_name` | `"model"` | Used in checkpoint filenames (`best_model_<name>.pth`) |
| `use_amp` | `False` | Mixed precision (FP16) |
| `grad_accum_steps` | `1` | Gradient accumulation |
| `whole_slide` | `False` | Dense whole-slide training mode |
| `val_whole_slide` | `None` | Whole-slide mode for validation (defaults to `whole_slide`) |
| `callbacks` | `None` | List of `TrainerCallback` instances |
| `resume` | `False` | Resume from a checkpoint in `output_dir` |

#### Callbacks

Subclass `TrainerCallback` to hook into the loop:

```python
from spatial_transcript_former.training import TrainerCallback

class WandbCallback(TrainerCallback):
    def on_epoch_end(self, trainer, epoch, metrics):
        wandb.log(metrics, step=epoch)
```

| Hook | When |
|---|---|
| `on_train_begin(trainer)` | Start of `fit()` |
| `on_epoch_begin(trainer, epoch)` | Before each epoch |
| `on_epoch_end(trainer, epoch, metrics)` | After validation |
| `on_train_end(trainer, results)` | End of `fit()` |
| `should_stop(trainer, epoch, metrics)` | Return `True` to stop early |

Built-in: `EarlyStoppingCallback(patience=15, min_delta=0.0)` (monitors `val_loss`).

### Programmatic Training (Low-Level)

```python
from spatial_transcript_former.training import train_one_epoch, validate, CompositeLoss

criterion = CompositeLoss(alpha=1.0, pcc_type="ccc")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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

`validate(...)` returns a dict: `val_loss`, `val_mae`, `val_baseline_mae`, `val_pcc`, `val_ccc`, `val_pcc_per_pathway`, `val_ccc_per_pathway`, `pred_variance`, `spatial_coherence`, `attn_correlation` (entries are `None` when not applicable).

### Loss Functions (`training.losses`)

All accept `(B, P)` patch-level or `(B, N, P)` dense inputs, with optional `mask` (padded positions) and `pathway_mask` (invalid/zero-variance pathways). `CLIPAlignmentLoss` takes `mask` only.

| Class | Formula / Description |
|---|---|
| `MaskedMSELoss` | MSE over valid positions |
| `MaskedHuberLoss` | Huber / Smooth-L1 — robust to outlier spots (`delta=1.0`) |
| `PCCLoss` | `1 − mean(PCC)` — per-pathway spatial Pearson correlation (scale-invariant) |
| `CCCLoss` | `1 − mean(CCC)` — concordance correlation; penalises mean/variance offsets too |
| `CLIPAlignmentLoss` | Batch-discriminative anti-collapse regulariser in pathway space (opt-in) |
| `CompositeLoss` | `MSE/Huber + α·(1 − PCC/CCC) [+ clip_weight·L_CLIP]` |

`CompositeLoss(alpha=1.0, eps=1e-8, mse_type="mse"|"huber", pcc_type="pcc"|"ccc", clip_weight=0.0, clip_temperature=0.07)`. The CLI `--loss` values map onto these: `mse`, `pcc`, `ccc`, `mse_pcc`, `mse_ccc`, `mse_ccc_clip`, `mse_huber`.

### Experiment Logger (`training.experiment_logger`)

Offline-friendly logger (no W&B dependency). Writes per-epoch metrics to SQLite and a JSON summary.

```python
from spatial_transcript_former.training import ExperimentLogger

logger = ExperimentLogger(output_dir, config_dict)
logger.log_epoch(epoch, {"train_loss": 0.1, "val_loss": 0.2, "val_ccc": 0.65})
logger.finalize(best_val_loss=0.15)
```

| Output File | Contents |
|---|---|
| `training_logs.sqlite` | Per-epoch metrics (table `metrics`, columns added dynamically) |
| `results_summary.json` | `config` + final metrics + runtime |

### Checkpoint Lifecycle

During training, full training state (for `--resume`) is managed by `training.checkpoint` — the *internal* module, distinct from the public `save_pretrained`:

| Function | Purpose |
|---|---|
| `save_checkpoint(model, optimizer, scaler, schedulers, epoch, best_val_metric, output_dir, model_name)` | Writes `latest_model_<name>.pth` (full training state) |
| `load_checkpoint(model, optimizer, scaler, schedulers, output_dir, model_name, device)` | Restores it → `(start_epoch, best_val_metric, loaded_schedulers)` |

After training, use the public `save_pretrained` to export a clean, inference-ready checkpoint directory.

---

## Bring Your Own Data

All datasets implement the `SpatialDataset` contract (in `data.base`). `__getitem__` must return a **5-tuple**:

```python
(features, gene_counts, pathway_targets, rel_coords, mask)
# features:        (S, D) tensor — patch embeddings (S = 1 + neighbours), or an image tensor
# gene_counts:     legacy slot — pass None (the model is pathway-exclusive)
# pathway_targets: (P,) tensor — the supervised target (or None for inference-only)
# rel_coords:      (S, 2) tensor — coordinates relative to the centre patch
# mask:            (S,) bool tensor — True = padded, or None
```

### Minimal Implementation

```python
import torch
from spatial_transcript_former.data.base import SpatialDataset

class MyVisiumDataset(SpatialDataset):
    def __init__(self, features, pathways, coords):
        self._features = torch.as_tensor(features, dtype=torch.float32)
        self._pathways = torch.as_tensor(pathways, dtype=torch.float32)
        self._coords = torch.as_tensor(coords, dtype=torch.float32)
        self.num_pathways = self._pathways.shape[1]

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):
        feat = self._features[idx].unsqueeze(0)   # (1, D)
        rel = torch.zeros(1, 2)                    # centre = [0, 0]
        return feat, None, self._pathways[idx], rel, None
```

### Training Your Custom Dataset

```python
from torch.utils.data import DataLoader, random_split
from spatial_transcript_former import SpatialTranscriptFormer, Trainer
from spatial_transcript_former.training import CompositeLoss, EarlyStoppingCallback
from spatial_transcript_former.recipes.hest.dataset import collate_fn_patch

dataset = MyVisiumDataset(features, pathways, coords)
train_ds, val_ds = random_split(dataset, [0.8, 0.2])

model = SpatialTranscriptFormer(num_pathways=dataset.num_pathways, backbone_name="phikon")

trainer = Trainer(
    model=model,
    train_loader=DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn_patch),
    val_loader=DataLoader(val_ds, batch_size=64, collate_fn=collate_fn_patch),
    criterion=CompositeLoss(),
    epochs=100,
    callbacks=[EarlyStoppingCallback(patience=15)],
)
results = trainer.fit()
trainer.save_pretrained("./my_model/")
```

See [`recipes/custom/README.md`](../src/spatial_transcript_former/recipes/custom/README.md) for the full guide.
