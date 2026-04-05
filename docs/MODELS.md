# SpatialTranscriptFormer: Architecture & Design

This document describes the architecture, design philosophy, and training objectives of the **SpatialTranscriptFormer** model, along with the baseline models implemented for comparison.

---

## 1. Problem Statement

**Goal**: Predict spatially-resolved **biological pathway activity scores** directly from histology images.

**Data**: Spatial transcriptomics datasets (a subset of HEST-1k, filtered to human samples) where each tissue section has:

- A whole-slide histology image (H&E)
- Per-spot gene expression counts with spatial coordinates
- **Pre-computed pathway activity targets** (50 MSigDB Hallmark pathways) derived offline via QC → CP10k normalisation → z-scoring → mean gene aggregation

**Design Choice**: Rather than predicting ~1,000 individual genes, the model directly predicts pathway activity scores. These targets are biologically meaningful, spatially smoother than individual genes, and computed from an interpretable offline pipeline — eliminating the circular auxiliary loss used in previous versions.

---

## 2. SpatialTranscriptFormer (Proposed Model)

### 2.1 Design Philosophy

The SpatialTranscriptFormer models the **interaction between biological pathways and histology** via four configurable information flows:

1. **P↔P (Pathway self-interaction)**: Pathways refine each other's representations, capturing biological co-dependencies — e.g., EMT and inflammatory response pathways often co-activate in tumour microenvironments.

2. **P→H (Pathway queries Histology)**: Pathway tokens query patch features with cross-attention, asking *"does this tissue region show morphological evidence of this biological process?"* — e.g., does a patch look consistent with angiogenesis or epithelial-mesenchymal transition?

3. **H→P (Histology reads Pathways)**: Patch tokens attend to pathway tokens, receiving biological context — e.g., *"this patch is in a region where the inflammatory response pathway is highly active."* This contextualises the visual features with global biological state.

4. **H↔H (Patch self-interaction)**: Patches attend to each other, enabling the model to capture spatial relationships between tissue regions directly.

By default, the model operates in **Full Interaction** mode where all four information flows are active. Users can selectively disable any combination using the `--interactions` flag to explore architectural variants:

```bash
# Default: Small Interaction (CTransPath, 4 layers)
python scripts/run_preset.py --preset stf_small
```

> [!TIP]
> The **Pathway Bottleneck** variant (disabling `h2h`) is particularly useful for **interpretability** — all spatial interactions are mediated by named biological pathways — and for **anti-collapse** — preventing patches from averaging into identical representations.

Three additional design principles support these interactions:

- **Frozen Foundation Model Backbone** — The visual backbone (CTransPath, Phikon, etc.) is a pre-trained pathology feature extractor. It is never fine-tuned. The model learns only the pathway-histology interactions, keeping training lightweight.

- **Dense Spatial Supervision** — Unlike weak MIL (which uses slide-level labels), we supervise at the **spot level** using pre-computed pathway activity targets. Every patch receives a ground-truth activity vector, enabling the model to learn spatially-resolved pathway activation patterns.

- **Offline Target Decoupling** — Pathway activity targets are pre-computed once from raw expression data (see [`PATHWAY_MAPPING.md`](PATHWAY_MAPPING.md)) and stored as `.h5` files. This cleanly separates biological knowledge integration from model training.

## 2.2 Spatial Learning

The spatial relationships of gene expression are central to this model. It is not sufficient to predict correct expression magnitudes at each spot independently — the model must capture **where** on the tissue pathways are active and how that spatial pattern varies across the slide. Two mechanisms enforce this:

1. **Positional Encoding** — Each patch token receives a 2D sinusoidal encoding of its (x, y) coordinate on the tissue. This means the pathway tokens, when they attend to patches, can distinguish *where* each patch is. A pathway token can learn that EMT is localised at the tumour-stroma boundary, not uniformly across the slide.

2. **PCC Loss (Spatial Pattern Coherence)** — The Pearson Correlation component in the composite loss measures whether the *spatial pattern* of each gene's predicted expression matches the ground truth pattern, independently of scale. A model that predicts the same value everywhere scores PCC = 0, even if the mean is correct. This directly penalises spatial collapse.

Together, these ensure the model learns *spatially-varying* pathway activation maps rather than slide-level averages.

### 2.3 Architecture

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                           SpatialTranscriptFormer                             │
│                                                                               │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐           │
│  │  Frozen      │   │ Image        │   │ + Spatial PE             │           │
│  │  Backbone    │──>│ Projection   │──>│   (2D Learned MLP)       │           │
│  │  (CTransPath)│   │ (Linear)     │   │                          │           │
│  └─────────────┘   └──────────────┘   └────────┬─────────────────┘           │
│                                                  │ Patch Tokens (S, D)        │
│  ┌──────────────────────────┐                    │                            │
│  │  Learnable Pathway       │                    │                            │
│  │  Tokens (P, D)           │────────┐           │                            │
│  └──────────────────────────┘        ▼           ▼                            │
│                              ┌─────────────────────────────┐                  │
│                              │  Transformer Encoder         │                  │
│                              │  Sequence: [Pathways|Patches]│                  │
│                              │                              │                  │
│                              │  Full Interaction (default): │                  │
│                              │  • P↔P ✅  P→H ✅           │                  │
│                              │  • H→P ✅  H↔H ✅           │                  │
│                              │                              │                  │
│                              │  Configurable via            │                  │
│                              │  --interactions flag         │                  │
│                              └──────────┬───────────────────┘                 │
│                                         │                                     │
│                              ┌──────────▼──────────────────┐                  │
│                              │  Cosine Similarity Scoring   │                  │
│                              │                              │                  │
│                              │  norm_patch @ norm_pathway.T │                  │
│                              │  → Pathway Scores (S, P)     │                  │
│                              └──────────┬───────────────────┘                 │
│                                         │                                     │
│                              ┌──────────▼──────────────────┐                  │
│                              │  Loss: MSE + PCC             │                  │
│                              │  vs. pre-computed pathway    │                  │
│                              │  activities (from .h5 files) │                  │
│                              └─────────────────────────────┘                  │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Key Components

#### Frozen Backbone (Feature Extraction)

Pre-computed features from a pathology foundation model. (The backbone is not fine-tuned during training, though this might change!)

| Backbone | Feature Dim | Source |
| :--- | :--- | :--- |
| **CTransPath** | 768 | [Wang et al. (2022)](https://arxiv.org/abs/2111.13324) |
| **GigaPath** | 1536 | [Microsoft Prov-GigaPath](https://hf.co/prov-gigapath/prov-gigapath) |
| **Hibou** | 768 / 1024 | [HistAI Hibou](https://hf.co/histai) |
| **Phikon** | 768 | [Owkin Phikon](https://hf.co/owkin/phikon) |
| **ResNet-50** | 2048 | Torchvision (ImageNet) |

#### Pathway Tokenizer

Learnable embeddings $T \in \mathbb{R}^{P \times D}$ representing biological pathways. These act as [CLS]-like bottleneck tokens, analogous to the "perceiver" cross-attention pattern.

#### Sinusoidal Positional Encoding (2D)

Patch tokens receive spatial location information via 2D sinusoidal embeddings:
$$H_{PE} = H_{proj} + \text{PE}_{2D}(x, y)$$
This encodes the absolute position of each patch on the tissue slide, enabling the model to learn spatially-varying pathway activation patterns.

#### Configurable Interaction Masking

The transformer uses a custom attention mask that controls information flow between token groups. By default, **all interactions are enabled** (Full Interaction). You can selectively disable any combination using the `--interactions` flag:

| Quadrant | Token Flow | Description |
| :--- | :--- | :--- |
| **p2p** | Pathway ↔ Pathway | Pathways refine each other |
| **p2h** | Pathway → Histology | Pathways gather spatial info from patches |
| **h2p** | Histology → Pathway | Patches receive contextualised pathway signals |
| **h2h** | Histology ↔ Histology | Patches attend to each other directly |

> [!NOTE]
> Disabling `h2h` creates the **Pathway Bottleneck** variant, where all inter-patch communication must flow through the pathway tokens. This requires **minimum 2 transformer layers**: Layer 1 lets pathways gather information from patches, and Layer 2 lets patches read the contextualised pathway tokens.

#### Cosine Similarity Scoring

Pathway scores are computed via L2-normalized cosine similarity with a learnable temperature parameter $\tau$ (following CLIP):
$$s_{ij} = \cos(\hat{h}_i, \hat{p}_j) \times \tau$$
where $\hat{h}_i$ and $\hat{p}_j$ are the L2-normalized processed patch and pathway tokens respectively. This produces scores in $[-\tau, +\tau]$ with meaningful relative differences, avoiding the saturation that occurs with raw dot-products.

#### Direct Pathway Prediction

The model's output *is* the pathway activity score vector. There is no intermediate gene reconstruction layer. The cosine similarity scores between (processed) patch tokens and pathway tokens directly serve as the prediction:
$$\hat{s}_{i,k} = \cos(\hat{h}_i, \hat{p}_k)$$
where $\hat{h}_i$ and $\hat{p}_k$ are the L2-normalised patch and pathway tokens, respectively. These are supervised against pre-computed pathway activities (see [PATHWAY_MAPPING.md](PATHWAY_MAPPING.md)).

### 2.4 Training Modes

| Mode | Input | Output | Supervision |
| :--- | :--- | :--- | :--- |
| **Dense (whole-slide)** | All patches from a slide | Per-patch gene predictions $(B, S, G)$ | Masked MSE+PCC at each spot |
| **Global** | All patches from a slide | Slide-level prediction $(B, G)$ | Mean-pooled expression |

---

## 3. Baseline Models

### HE2RNA (ResNet-50)

- **Reference**: [Schmauch et al. (2020), Nature Communications](https://www.nature.com/articles/s41467-020-17679-z)
- Direct regression from patch features to gene expression via a single linear layer.

### Attention-MIL

- **Reference**: [Ilse et al. (2018), ICML](https://arxiv.org/abs/1802.04712)
- Learns gated attention weights to aggregate patches into a slide-level representation.

### TransMIL

- **Reference**: [Shao et al. (2021), NeurIPS](https://arxiv.org/abs/2106.00908)
- Nyström-based transformer for capturing long-range patch correlations.

---

## 4. Loss Functions

All losses in the current codebase operate on **pathway activity scores** (B, P) or (B, N, P), where P is the number of pathways. The targets are pre-computed offline — not derived from in-flight gene expression.

| Key | Name | Description |
| :--- | :--- | :--- |
| `mse` | `MaskedMSELoss` | Mean squared error; penalises magnitude errors at each spot |
| `pcc` | `PCCLoss` | 1 − PCC; penalises deviations in spatial pattern shape (scale-invariant) |
| `mse_pcc` | `CompositeLoss` | MSE + α(1 − PCC); balances magnitude accuracy and spatial coherence |

### Composite Loss (Recommended)

The **MSE + PCC composite** (`mse_pcc`, default) is the recommended objective:

$$\mathcal{L} = \text{MSE}(\hat{s}, s) + \alpha \cdot (1 - \text{PCC}(\hat{s}, s))$$

- **MSE** ensures the predicted activity magnitudes are accurate.
- **PCC** ensures the *spatial pattern* of each pathway across the tissue matches the ground truth, regardless of scale. A model that predicts the same activity value everywhere scores PCC = 0 even if the mean is correct.

The `--pcc-weight` flag controls $\alpha$ (default: 1.0).

> [!NOTE]
> The previous versions of this codebase included a `ZINBLoss` (for raw count data) and an `AuxiliaryPathwayLoss` (multi-task learning against on-the-fly pathway pseudo-targets). Both have been removed. The model now predicts pathway activity scores directly against pre-computed targets, making a single clean MSE+PCC objective sufficient.

---

## 5. CLI Flags (Model Configuration)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model interaction` | — | Select SpatialTranscriptFormer |
| `--backbone` | `resnet50` | Foundation model backbone |
| `--token-dim` | 256 | Transformer embedding dimension |
| `--n-heads` | 4 | Number of attention heads |
| `--n-layers` | 2 | Transformer layers (minimum 2) |
| `--num-pathways` | 50 | Number of pathway bottleneck tokens |
| `--pathway-prior` | `hallmarks` | Pathway prior for token count (`hallmarks` = 50, `progeny` = 14) |
| `--pathway-targets-dir` | `<data-dir>/pathway_activities` | Directory of pre-computed `.h5` pathway activity files |
| `--loss` | `mse_pcc` | Loss function: `mse`, `pcc`, `mse_pcc` |
| `--pcc-weight` | 1.0 | Weight for PCC term in composite loss ($\alpha$) |
| `--interactions` | `all` | Enabled interaction quadrants (`p2p p2h h2p h2h`) |
| `--return-attention` | off | Return attention maps from forward pass (for diagnostics) |
| `--n-neighbors` | 0 | Number of context neighbors (for hybrid/GNN models) |
