# Training Guide (HEST Benchmark Recipe)

> [!NOTE]
> This guide provides command-line recipes specifically for the **HEST-1k benchmark dataset**. If you are looking to train on your own data using the core API, please see the **[Python API Reference](API.md)**.

## Prerequisites

Ensure you have the `hest_data` and your conda environment is active.

```bash
conda activate SpatialTranscriptFormer
```

### Pre-Compute Pathway Activity Targets

Before any training run, you must pre-compute the pathway activity targets from raw expression data. This step applies per-spot QC, CP10k normalisation, and z-scoring.

```bash
stf-compute-pathways --data-dir hest_data
```

This will produce `.h5` files in `hest_data/pathway_activities/` which are consumed by the trainer automatically. See **[Pathway Mapping](PATHWAY_MAPPING.md)** for details on QC thresholds and scoring methodology.

---

## 1. Single Patch Regression (Baselines)

Predicts gene expression for a single 224x224 patch. No cross attention interactions between patches or pathways.

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

Aggregates all patches from a slide to predict the average expression. Recommended to use **precomputed features** from the `stf-compute-features` CLI tool for speed. Foundation models like ctranspath can be used as backbones.

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

Uses precomputed features and local spatial masking. Predicts pathway activity scores at every spot.

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
    --epochs 100
```

### Using PROGENy-Style Priors (14 Pathways) - Currently Not Implemented

Use `--pathway-prior progeny` to set the number of pathway tokens to 14, matching the dimensionality of PROGENy-style signalling pathway databases. This is useful when your pre-computed targets were derived from a 14-pathway database rather than the 50 MSigDB Hallmarks.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model interaction \
    --backbone ctranspath \
    --pathway-prior progeny \
    --precomputed \
    --whole-slide \
    --use-amp \
    --epochs 100
```

> **Note**: `--pathway-prior progeny` sets `--num-pathways` to 14 automatically. Ensure your pre-computed pathway targets match this dimensionality.

### Recommended: Using Presets

For most cases, it is recommended to use the provided presets:

```bash
# Tiny (2 layers, 256 dim)
python scripts/run_preset.py --preset stf_tiny

# Small (4 layers, 384 dim) - Recommended
python scripts/run_preset.py --preset stf_small

# Medium (6 layers, 512 dim)
python scripts/run_preset.py --preset stf_medium

# Large (12 layers, 768 dim)
python scripts/run_preset.py --preset stf_large
```

#### Disease-Specific Priors

To learn representations constrained to specific disease phenotypes, you can filter the pre-computed pathway targets and pathway tokens using the `--pathways` argument. This is useful when you want to focus on a specific subset of pathways relevant to your disease of interest.

The `crc` presets in `scripts/run_preset.py` demonstrate this, reducing from 50 Hallmarks to a CRC-relevant subset.

```bash
# Small CRC Variant
python scripts/run_preset.py --preset stf_crc_small
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

### Collecting Results (Currently broken!)

After experiments complete, aggregate all `results_summary.json` files into a comparison table:

```bash
python hpc/collect_results.py --results-dir runs/ws_experiments
```

This produces a sorted comparison table and `comparison.csv`.

---

## 5. Experiment Logging

Each training run automatically produces a directory of outputs:

| File | Description |
| :--- | :--- |
| `training_logs.sqlite` | Per-epoch metrics (train_loss, val_loss, attn_correlation) |
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
| `--pathway-targets-dir` | Directory of pre-computed `.h5` pathway activity files. | Defaults to `<data-dir>/pathway_activities`. |
| `--feature-dir` | Explicit path to precomputed features directory. | Overrides auto-detection. |
| `--loss` | Loss function: `mse`, `pcc`, `mse_pcc`. | `mse_pcc` recommended. |
| `--pcc-weight` | Weight ($\alpha$) for PCC term in composite loss. | Default 1.0. |
| `--pathway-prior` | Pathway prior for token count (`hallmarks`=50). | Use `hallmarks` for MSigDB Hallmarks. |
| `--interactions` | Enabled attention quadrants: `p2p`, `p2h`, `h2p`, `h2h`. | Default: `all` (Full Interaction). |
| `--plot-pathways-list` | Names of pathways to visualize as heatmaps during validation. | e.g. `HYPOXIA ANGIOGENESIS` |
| `--num-pathways` | Number of pathway bottleneck tokens (overridden by `--pathway-prior`). | Match your pre-computed targets. |
| `--mask-radius` | Euclidean distance for spatial attention gating. | Usually between 200 and 800. |
| `--n-neighbors` | Number of context neighbors to load. | Set `> 0` for hybrid/GNN models. |
| `--use-amp` | Mixed precision training. | Recommended on modern GPUs. |
| `--grad-accum-steps` | Gradient accumulation steps. | Use when memory is limited. |
| `--compile` | Use `torch.compile` for speed. | Recommended on Linux/A100. |
| `--resume` | Resume from latest checkpoint. | Use after interruption. |

---

## Tips for Success

1. **Pathway Pre-Computation**: Always run `stf-compute-pathways --data-dir hest_data` before training. The trainer will error if pathway targets are missing.
2. **Feature Extraction**: Run `hpc/prepare_data.slurm` or `scripts/extract_features.py` before training with `--precomputed`.
3. **Output**: Checkpoints, logs, and JSON summaries are saved to `--output-dir` (default: `./checkpoints`).
4. **Debug Mode**: Use `--max-samples 3 --epochs 1` to verify your setup before a full run.
5. **Results Aggregation**: Use `hpc/collect_results.py` to compare experiments across multiple runs.
