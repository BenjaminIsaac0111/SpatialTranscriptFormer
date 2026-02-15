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

### Attention MIL

Uses a gated attention mechanism to weigh patches.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model attention_mil \
    --whole-slide \
    --precomputed \
    --epochs 50
```

### TransMIL

Uses a Transformer with Nystrom attention for cross-patch correlation.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model transmil \
    --whole-slide \
    --precomputed \
    --epochs 50
```

---

## 3. Spatial TranscriptFormer (Multimodal Interaction)

The core model of this repository. Captures dense interactions between pathways and histology.

### Standard Configuration

Uses precomputed features and local spatial masking.

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model interaction \
    --use-nystrom \
    --mask-radius 400.0 \
    --num-pathways 50 \
    --precomputed \
    --epochs 30
```

### Advanced: Multimodal Masking (Ablation)

Test the model's robustness by masking specific quadrants (e.g., top-left and bottom-right).

```bash
python -m spatial_transcript_former.train \
    --data-dir A:\hest_data \
    --model interaction \
    --masked-quadrants top_left bottom_right \
    --precomputed
```

---

## 4. Key Arguments Reference

| Argument | Description | Recommendation |
| :--- | :--- | :--- |
| `--precomputed` | Use saved features instead of raw H&E images. | Use for fast experimentation. |
| `--whole-slide` | Dense prediction across the whole slide. | Use for MIL or global analysis. |
| `--num-genes` | Number of HVGs to predict (default: 1000). | Match your `load_gene_expression_matrix` config. |
| `--mask-radius` | Euclidean distance for spatial attention gating. | Usually between 200 and 800. |
| `--n-neighbors` | Number of context neighbors to load. | Set `> 0` for models using spatial context. |
| `--compile` | Use `torch.compile` for speed. | Recommended on Linux/A100. |

---

## Tips for Success

1. **Feature Extraction**: Run `scripts/extract_features.py` before training with `--precomputed`.
2. **Output**: Checkpoints and validation plots are saved to `./checkpoints` by default. Use `--output-dir` to change this.
3. **Debug Mode**: Use `--max-samples 2` and `--epochs 1` to verify your setup before running a full scale training.
