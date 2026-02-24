# Training Guide

This guide provides command-line recipes for training different architectures and configurations using `spatial_transcript_former.train`.

## Prerequisites

Ensure you have the `hest_data` and your conda environment is active.

```bash
conda activate SpatialTranscriptFormer
```

---

## 1. Single Patch Regression (Baselines)

Predicts gene expression for a single 224x224 patch.

### HE2RNA (ResNet50)

The simplest baseline. Uses raw images.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model he2rna \
    --backbone resnet50 \
    --batch-size 64 \
    --epochs 20
```

### ViT-ST

Uses a Vision Transformer backbone.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model vit_st \
    --backbone vit_b_16 \
    --batch-size 32 \
    --epochs 20
```

---

## 2. Whole-Slide MIL (Multiple Instance Learning)

Aggregates all patches from a slide to predict the average expression. Recommended to use **precomputed features** for speed.

### Attention MIL (Weak Supervision)

Uses a gated attention mechanism to weigh patches. Trained with bag-level (slide-average) targets.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model attention_mil \
    --backbone ctranspath \
    --whole-slide \
    --precomputed \
    --weak-supervision \
    --use-amp \
    --log-transform \
    --epochs 50
```

### TransMIL (Weak Supervision)

Uses a Transformer with Nystrom attention for cross-patch correlation.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model transmil \
    --backbone ctranspath \
    --whole-slide \
    --precomputed \
    --weak-supervision \
    --use-amp \
    --log-transform \
    --epochs 50
```

---

## 3. Spatial TranscriptFormer (Multimodal Interaction)

The core model of this repository. Captures dense interactions between pathways and histology.

### Standard Configuration (Dense Supervision)

Uses precomputed features and local spatial masking. Predicts expression at every spot.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model interaction \
    --backbone ctranspath \
    --use-nystrom \
    --num-pathways 50 \
    --precomputed \
    --whole-slide \
    --use-amp \
    --log-transform \
    --epochs 100
```

### With Biological Pathway Initialization

Initialize the gene reconstruction weights from MSigDB Hallmark gene sets (50 biological pathways). This provides a biologically-grounded starting point.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model interaction \
    --backbone ctranspath \
    --use-nystrom \
    --precomputed \
    --whole-slide \
    --pathway-init \
    --use-amp \
    --log-transform \
    --epochs 100
```

> **Note**: `--pathway-init` overrides `--num-pathways` to 50 (the number of Hallmark gene sets). The GMT file is cached in `.cache/` after first download.

### Robust Counting: ZINB + Auxiliary Loss

For raw count data with high sparsity, using the ZINB distribution and auxiliary pathway supervision is recommended.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model interaction \
    --backbone ctranspath \
    --pathway-init \
    --loss zinb \
    --pathway-loss-weight 0.5 \
    --lr 5e-5 \
    --batch-size 4 \
    --whole-slide \
    --precomputed \
    --epochs 200
```

### Choosing Interaction Modes

By default, the model runs in **Full Interaction** mode (`p2p p2h h2p h2h`) where all token types attend to each other. You can selectively disable interactions using the `--interactions` flag for ablation or to enforce specific architectural constraints.

For example, to use the **Pathway Bottleneck** (blocking patch-to-patch attention for interpretability):

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model interaction \
    --interactions p2p p2h h2p \
    --precomputed \
    --whole-slide
```

Available interaction tokens: `p2p`, `p2h`, `h2p`, `h2h`. Default is all four (Full Interaction).

---

## 4. HPC Batch Experiments

The `hpc/array_train.slurm` script runs all three whole-slide experiments as a SLURM array job:

| Index | Model | Supervision | Key Flags |
| :--- | :--- | :--- | :--- |
| 0 | SpatialTranscriptFormer | Dense | `--whole-slide` |
| 1 | AttentionMIL | Weak | `--whole-slide --weak-supervision` |
| 2 | TransMIL | Weak | `--whole-slide --weak-supervision` |

Submit with:

```bash
sbatch hpc/array_train.slurm
```

### Collecting Results

After experiments complete, aggregate all `results_summary.json` files into a comparison table:

```bash
python hpc/collect_results.py --results-dir runs/ws_experiments
```

This produces a sorted comparison table and `comparison.csv`.

---

## 5. Experiment Logging

Each training run automatically produces:

| File | Description |
| :--- | :--- |
| `training_log.csv` | Per-epoch metrics (train_loss, val_loss, attn_correlation) |
| `results_summary.json` | Full config + final metrics + runtime |
| `best_model_<name>.pth` | Best checkpoint (by val loss) |
| `latest_model_<name>.pth` | Latest checkpoint (for resume) |

Resume an interrupted run with `--resume`:

```bash
python -m spatial_transcript_former.train --resume --output-dir runs/my_experiment ...
```

---

## 6. Key Arguments Reference

| Argument | Description | Recommendation |
| :--- | :--- | :--- |
| `--precomputed` | Use saved features instead of raw H&E images. | Use for fast experimentation. |
| `--whole-slide` | Dense prediction across the whole slide. | Required for MIL and slide-level STF. |
| `--weak-supervision` | Bag-level training for MIL models. | Use with `attention_mil` or `transmil`. |
| `--pathway-init` | Initialize gene_reconstructor from MSigDB Hallmarks. | Use with `interaction` model. |
| `--feature-dir` | Explicit path to precomputed features directory. | Overrides auto-detection. |
| `--loss` | Loss function: `mse`, `pcc`, `mse_pcc`, `zinb`. | `mse_pcc` or `zinb` recommended. |
| `--pathway-loss-weight` | Weight ($\lambda$) for auxiliary pathway supervision. | Set `0.5` or `1.0` with `interaction` model. |
| `--interactions` | Enabled attention quadrants: `p2p`, `p2h`, `h2p`, `h2h`. | Default: `all` (Full Interaction). |
| `--log-transform` | Apply log1p to gene expression targets. | Recommended for raw count data. |
| `--num-genes` | Number of HVGs to predict (default: 1000). | Match your `global_genes.json`. |
| `--mask-radius` | Euclidean distance for spatial attention gating. | Usually between 200 and 800. |
| `--n-neighbors` | Number of context neighbors to load. | Set `> 0` for hybrid/GNN models. |
| `--use-amp` | Mixed precision training. | Recommended on modern GPUs. |
| `--grad-accum-steps` | Gradient accumulation steps. | Use when memory is limited. |
| `--compile` | Use `torch.compile` for speed. | Recommended on Linux/A100. |
| `--resume` | Resume from latest checkpoint. | Use after interruption. |

---

## Tips for Success

1. **Feature Extraction**: Run `hpc/prepare_data.slurm` or `scripts/extract_features.py` before training with `--precomputed`.
2. **Output**: Checkpoints, logs, and JSON summaries are saved to `--output-dir` (default: `./checkpoints`).
3. **Debug Mode**: Use `--max-samples 3 --epochs 1` to verify your setup before a full run.
4. **Results Aggregation**: Use `hpc/collect_results.py` to compare experiments across multiple runs.
