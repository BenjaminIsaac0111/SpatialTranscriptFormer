# SpatialTranscriptFormer: Architecture & Design

This document describes the architecture, design philosophy, and training objectives of the **SpatialTranscriptFormer** model, along with the baseline models implemented for comparison.

---

## 1. Problem Statement

**Goal**: Predict spatially-resolved gene expression from histology images.

**Data**: Spatial transcriptomics datasets (a subset of HEST-1k, filtered to bowel cancer from human patients) where each tissue section has:

- A whole-slide histology image (H&E)
- Per-spot gene expression counts with spatial coordinates

**Challenge**: Directly predicting ~1000 genes from image patches is high-dimensional, noisy, and biologically uninterpretable. We need a structured bottleneck that compresses the gene space into biologically meaningful abstractions.

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

- **Dense Spatial Supervision** — Unlike weak MIL (which uses slide-level labels), we supervise at the **spot level** using spatial transcriptomics. Every patch receives ground-truth expression, enabling the model to learn spatially-resolved pathway activation patterns.

- **Biological Initialisation** — The gene reconstruction weights are initialised from MSigDB Hallmark gene sets, providing a biologically-grounded starting point that the model refines during training.

## 2.2 Spatial Learning

The spatial relationships of gene expression are central to this model. It is not sufficient to predict correct expression magnitudes at each spot independently — the model must capture **where** on the tissue pathways are active and how that spatial pattern varies across the slide. Two mechanisms enforce this:

1. **Positional Encoding** — Each patch token receives a 2D sinusoidal encoding of its (x, y) coordinate on the tissue. This means the pathway tokens, when they attend to patches, can distinguish *where* each patch is. A pathway token can learn that EMT is localised at the tumour-stroma boundary, not uniformly across the slide.

2. **PCC Loss (Spatial Pattern Coherence)** — The Pearson Correlation component in the composite loss measures whether the *spatial pattern* of each gene's predicted expression matches the ground truth pattern, independently of scale. A model that predicts the same value everywhere scores PCC = 0, even if the mean is correct. This directly penalises spatial collapse.

Together, these ensure the model learns *spatially-varying* pathway activation maps rather than slide-level averages.

### 2.3 Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SpatialTranscriptFormer                            │
│                                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐          │
│  │  Frozen      │   │ Image        │   │ + Spatial PE             │          │
│  │  Backbone    │──>│ Projection   │──>│   (2D Learned)           │          │
│  │  (CTransPath)│   │ (Linear)     │   │                          │          │
│  └─────────────┘   └──────────────┘   └────────┬─────────────────┘          │
│                                                  │ Patch Tokens (S, D)       │
│  ┌──────────────────────────┐                    │                           │
│  │  Learnable Pathway       │                    │                           │
│  │  Tokens (P, D)           │────────┐           │                           │
│  │  (MSigDB Hallmarks)      │        │           │                           │
│  └──────────────────────────┘        ▼           ▼                           │
│                             ┌─────────────────────────────┐                  │
│                             │  Transformer Encoder         │                  │
│                             │  Sequence: [Pathways|Patches]│                  │
│                             │                              │                  │
│                             │  Full Interaction (default):  │                  │
│                             │  • P↔P ✅  P→H ✅            │                  │
│                             │  • H→P ✅  H↔H ✅            │                  │
│                             │                              │                  │
│                             │  Configurable via             │                  │
│                             │  --interactions flag          │                  │
│                             └──────────┬──────────────────┘                  │
│                                        │                                     │
│                             ┌──────────▼──────────────────┐                  │
│                             │  Cosine Similarity Scoring   │                  │
│                             │  with Learnable Temperature  │                  │
│                             │                              │                  │
│                             │  scores = cos(patch, pathway)│                  │
│                             │           × τ                │                  │
│                             └──────────┬──────────────────┘                  │
│                                        │ Pathway Scores (S, P)               │
│                            ┌───────────┴───────────┐                         │
│                            │                       │                         │
│                            ▼                       ▼                         │
│             ┌──────────────────────┐  ┌───────────────────────────┐          │
│             │  Gene Reconstructor  │  │  Auxiliary Pathway Loss   │          │
│             │  (Linear: P → G)     │  │  PCC(scores, target_pw)   │          │
│             │  Init: MSigDB        │  │  weighted by λ_aux        │          │
│             └──────────┬───────────┘  └───────────┬───────────────┘          │
│                        │                          │                          │
│                        ▼                          ▼                          │
│             Gene Expression (S, G)     ℒ_aux = λ(1 − PCC)                   │
│                        │                          │                          │
│                        └──────────┬───────────────┘                          │
│                                   ▼                                          │
│                        ℒ_total = ℒ_gene + ℒ_aux                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Key Components

#### Frozen Backbone (Feature Extraction)

Pre-computed features from a pathology foundation model. (The backbone is never fine-tuned, though this might change!)

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

#### Gene Reconstruction (Biologically-Informed)

A linear layer $W_{recon} \in \mathbb{R}^{G \times P}$ maps pathway scores to gene expression:
$$\hat{y}_g = \sum_{k=1}^P s_k \cdot W_{gk} + b_g$$

When `--pathway-init` is enabled, $W_{recon}$ is initialised from the MSigDB Hallmark gene sets as a binary membership matrix, giving the model a biologically-grounded starting point where each pathway is connected only to its known member genes.

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

| `mse` | Masked MSE | Magnitude accuracy at each spot |
| `pcc` | 1 − PCC | Spatial pattern coherence per gene (scale-invariant) |
| `mse_pcc` | MSE + α(1 − PCC) | Balances absolute magnitude and spatial shape |
| `zinb` | ZINB NLL | Zero-Inflated Negative Binomial negative log-likelihood |

### ZINB Loss

The Zero-Inflated Negative Binomial (ZINB) loss is designed for raw, highly dispersed count data. It models the data using three parameters:

- **$\pi$ (pi)**: Probability of zero-inflation (technical dropout).
- **$\mu$ (mu)**: Mean of the negative binomial distribution.
- **$\theta$ (theta)**: Inverse dispersion (clumping) parameter.

The model outputs these parameters, and the loss computes the negative log-likelihood of the ground truth counts given this distribution.

### Proposed Auxiliary Pathway Loss

To prevent bottleneck collapse and provide a direct gradient signal to the pathway tokens, we use the `AuxiliaryPathwayLoss`. This loss compares the model's internal pathway scores against "ground truth" pathway activations computed from the gene expression targets via MSigDB membership.

To prevent highly-expressed housekeeping genes from dominating the pathway's spatial pattern, the ground-truth targets are computed using **Z-score spatial normalization**:

1. Every gene's spatial expression pattern is standardized (mean=0, variance=1) across the tissue slide.
2. The normalized genes are projected onto the binary MSigDB pathway matrix.
3. The resulting pathway scores are **mean-aggregated** (divided by the number of known member genes in each pathway) rather than raw-summed.

This ensures every gene—including critical but lowly-expressed transcription factors—gets an equal vote in determining where a pathway is active.

The total objective becomes:
$$\mathcal{L} = \mathcal{L}_{gene} + \lambda_{aux} (1 - \text{PCC}(\text{pathway\_scores}, \text{target\_pathways}))$$

The `--log-transform` flag applies `log1p` to targets, mitigating the heavy-tailed gene expression distribution where housekeeping genes dominate MSE.

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
| `--pathway-init` | off | Initialize gene_reconstructor from MSigDB |
| `--loss mse_pcc` | `mse` | Loss function (`mse`, `pcc`, `mse_pcc`, `zinb`) |
| `--pcc-weight` | 1.0 | Weight for PCC term in composite loss |
| `--pathway-loss-weight` | 0.0 | Weight for auxiliary pathway loss ($\lambda_{aux}$) |
| `--interactions` | `all` | Enabled interaction quadrants (`p2p p2h h2p h2h`) |
| `--log-transform` | off | Apply log1p to targets |
| `--return-attention` | off | Return attention maps from forward pass (for diagnostics) |
| `--n-neighbors` | 0 | Number of context neighbors (for hybrid/GNN models) |
