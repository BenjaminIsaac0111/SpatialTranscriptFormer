# Model Zoo and Literature References

This document provides a summary of the models implemented in this project, their origins in literature, and the architectural details of the **SpatialTranscriptFormer**.

## 1. Regression Models (Patch-Level)

### HE2RNA (ResNet-50)

- **Reference**: [Schmauch et al. (2020). "A deep learning model to predict RNA-Seq expression of tumours from whole slide images." Nature Communications.](https://www.nature.com/articles/s41467-020-17679-z)
- **Description**: Uses a ResNet-50 backbone to extract features from histology patches and directly regresses a high-dimensional gene expression vector.
- **Equation**:
  $$\hat{y} = \text{FC}(\text{ResNet50}(x))$$
  Where $x$ is the histology patch and $\hat{y} \in \mathbb{R}^G$ is the predicted expression for $G$ genes.

### ViT-ST

- **Description**: An adaptation of the Vision Transformer (ViT) to the spatial transcriptomics task by replacing the classification head with a regression head.
- **Equation**:
  $$\hat{y} = \text{MLP}(\text{TransformerEncoder}(x_{tokens}))$$

---

## 2. Multiple Instance Learning (Slide-Level)

### Attention-MIL

- **Reference**: [Ilse et al. (2018). "Attention-based Deep Multiple Instance Learning." ICML.](https://arxiv.org/abs/1802.04712)
- **Description**: Learns attention weights for individual patches to aggregate them into a slide-level representation.
- **Aggregation**:
  $$z = \sum_{i=1}^N a_i h_i, \quad a_i = \frac{\exp(\mathbf{w}^\top \tanh(\mathbf{V} h_i^\top) \odot \text{sigm}(\mathbf{U} h_i^\top))}{\sum_{j=1}^N \exp(\mathbf{w}^\top \tanh(\mathbf{V} h_j^\top) \odot \text{sigm}(\mathbf{U} h_j^\top))}$$

### TransMIL

- **Reference**: [Shao et al. (2021). "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification." NeurIPS.](https://arxiv.org/abs/2106.00908)
- **Description**: Uses a Nyström-based linear transformer to capture correlations between patches across the entire slide.

### Weak Supervision for MIL

MIL models are trained with **bag-level supervision**: the model predicts a single slide-level expression vector and is supervised against the mean expression across all valid spots.

Given a slide with $N$ valid spots and gene expression matrix $\mathbf{Y} \in \mathbb{R}^{N \times G}$, the **bag-level target** is:
$$\bar{y}_g = \frac{1}{N} \sum_{i=1}^N y_{ig}$$

For padded batches with mask $\mathbf{m} \in \{0, 1\}^{N}$, this becomes:
$$\bar{y}_g = \frac{\sum_{i=1}^N (1 - m_i) \cdot y_{ig}}{\sum_{i=1}^N (1 - m_i)}$$

The training loss is then:
$$\mathcal{L}_{weak} = \frac{1}{G} \sum_{g=1}^G (\hat{y}_g - \bar{y}_g)^2$$

### Spatial Attention Correlation

To evaluate whether MIL attention maps recover spatial localisation of gene activity, we compute the **Pearson correlation** between attention weights and total gene expression at each spot:
$$\rho = \text{corr}\left(\mathbf{a},\ \sum_{g=1}^G \mathbf{y}_g\right)$$

where $\mathbf{a} \in \mathbb{R}^N$ is the vector of attention weights and $\sum_g \mathbf{y}_g \in \mathbb{R}^N$ is the total expression per spot. A high $\rho$ indicates the model has learned to attend to transcriptionally active regions.

### Dense Supervision (Masked MSE)

For models that support per-spot dense prediction in whole-slide mode (e.g., SpatialTranscriptFormer via `forward_dense`), we use a **masked MSE** that ignores padded positions:
$$\mathcal{L}_{dense} = \frac{\sum_{i=1}^N \sum_{g=1}^G (1 - m_i)(\hat{y}_{ig} - y_{ig})^2}{\sum_{i=1}^N (1 - m_i) \cdot G}$$

---

## 3. SpatialTranscriptFormer (Proposed Model)

The **SpatialTranscriptFormer** (formerly Pathway Interaction Model) introduces a biologically-informed bottleneck layer using Cross-Attention between Learned Pathways and Image Features.

### 3.1 Architecture Equations

#### 1. Image Encoding

Given an image patch $x$, we extract features $F$ using a backbone (e.g., ResNet or ViT):
$$F = \text{Encoder}(x), \quad F \in \mathbb{R}^{D}$$

#### 2. Pathway Tokenization

We initialize $K$ learnable pathway tokens $T_{path} \in \mathbb{R}^K \times D_{token}$. These tokens act as "queries" to search for relevant morphological features in the image.

#### 3. Interaction (Unified Early Fusion)

The interaction logic is unified via the **EarlyFusionBlock**, which facilitates a generalized attention mechanism between Pathway tokens ($P$) and Histology tokens ($H$). The model operates on a concatenated sequence $X = [P; H]$.

##### Quadrant Masking

To control the topology of the interaction and ensure scalability, we apply a mask $M$ to the attention matrix:

- **P2P (Pathway Self-Interaction)**: Pathways refine each other's latent signatures.
- **P2H (Cross-Attention)**: Pathways query the histology patches for morphological evidence.
- **H2P (Biological Feedback)**: Histology features can be influenced by pathway context.
- **H2H (Global Patch-to-Patch)**: Patches attend to other distal patches.

> [!IMPORTANT]
> **Default Configuration**: In the standard experiment, **H2H is masked**. This removes the $O(N_H^2)$ memory bottleneck, allowing the model to handle thousands of spots simultaneously while still capturing global pathway-level correlations and local morphology.

### 3.2 Spatial Inductive Biases

#### Sinusoidal Positional Encoding (2D)

We inject knowledge of absolute patch locations $(x, y)$ into the transformer latent space via 2D sinusoidal embeddings:
$$H_{PE} = H_{proj} + \text{PE}_{2D}(x, y)$$
This ensures the attention mechanism is aware of the relative distances and orientations between histology features.

#### Local Patch Mixing (Conv)

Because global `H2H` is typically masked for efficiency, we use a **LocalPatchMixer** to regain immediate spatial context. This module applies a depthwise convolution over the spatial grid *before* global interaction:
$$H_{mixed} = \text{GELU}(\text{DepthwiseConv2d}(H_{grid}))$$
This allows the model to capture high-bandwidth local morphology (e.g., 3x3 window) while the Transformer focuses on global biology-driven interactions.

### 3.3 Whole-Slide Dense Prediction (`forward_dense`)

When predicting gene expression for every spot across a slide, the model leverages its global context differently. Instead of taking a single coordinate as the 'target', it updates all histology tokens simultaneously:

1. **Global Context**: Pathways query the entire slide (or a large masked window).
2. **Dense Head**: Updated histology tokens $H'$ are projected back to gene space:
   $$\hat{y}_{dense} = H' \cdot T_{path}^\top \cdot W_{recon}$$
   This ensures that the predicted expression logic is consistent with the pathway bottleneck used during patch-level training.

---

## 4. Scalability: Nyström Approximation

To scale effectively to whole-slide contexts or extremely large neighborhoods ($N_H > 1000$), we utilize the **Nyström method** for approximating the self-attention matrix. This reduces the complexity from quadratic $O(N^2)$ to linear $O(N \cdot m)$.

### Theoretical Intuition

The core of Transformer attention is the kernel matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$. Nyström approximates $\mathbf{K}$ using a small set of **landmark points** $m \ll N$:
$$\mathbf{K} \approx \mathbf{C} \mathbf{W}^{+} \mathbf{C}^\top$$
Where:

- $\mathbf{C} \in \mathbb{R}^{N \times m}$ contains $m$ landmark columns.
- $\mathbf{W} \in \mathbb{R}^{m \times m}$ is the intersection matrix of those columns.
- $\mathbf{W}^{+}$ is the Moore-Penrose pseudo-inverse.

### Decomposition in Attention

In the context of attention $\text{Softmax}(\frac{QK^\top}{\sqrt{d}})V$, the Nyström approximation allows us to compute the interaction without ever forming the $N \times N$ matrix:
$$\tilde{A} = \text{Softmax}\left(\frac{Q \tilde{K}^\top}{\sqrt{d}}\right) \left[ \text{Softmax}\left(\frac{\tilde{K} \tilde{K}^\top}{\sqrt{d}}\right) \right]^{+} \text{Softmax}\left(\frac{\tilde{K} K^\top}{\sqrt{d}}\right)$$
Where $\tilde{K}$ are the pooled landmark features.

### Why it matters for Spatial Transcriptomics

1. **Whole Slide Context**: Standard transformers fail on WSIs (e.g., 20,000 patches). Nyström enables slide-level correlation (implemented in the **TransMIL** model).
2. **Dense Neighborhoods**: Allows the **SpatialTranscriptFormer** to consider hundreds of surrounding patches for a single prediction with minimal GPU memory overhead.

---

## 5. Backbone Zoo

The model supports multiple state-of-the-art pathology backbones. These are selected via the CLI using the `--backbone` flag.

| Backbone | Variant | Feature Dim | Source / Reference |
| :--- | :--- | :--- | :--- |
| **ResNet** | `resnet50` | 2048 | Torchvision (ImageNet) |
| **CTransPath** | `ctranspath` | 768 | [Wang et al. (2022)](https://arxiv.org/abs/2111.13324) |
| **GigaPath** | `gigapath` | 1536 | [Microsoft Prov-GigaPath](https://hf.co/prov-gigapath/prov-gigapath) |
| **Hibou** | `hibou-b/l` | 768 / 1024 | [HistAI Hibou](https://hf.co/histai) |
| **Phikon** | `phikon` | 768 | [Owkin Phikon](https://hf.co/owkin/phikon) |
| **PLIP** | `plip` | 512 | [Huang et al. (2023)](https://hf.co/vinid/plip) |

> [!NOTE]
> GigaPath and Hibou are **gated models**. You must accept the terms of use on their respective HuggingFace model pages before the code can download the weights.

---

## 6. Biologically-Informed Bottleneck

The final gene expression $\hat{y}$ is a linear combination of pathway activations $s_k$:
$$\hat{y}_g = \sum_{k=1}^K s_k \cdot W_{gk} + b_g$$

### MSigDB Hallmarks Initialization

When `--pathway-init` is enabled, $\mathbf{W}_{recon}$ is initialized from the **MSigDB Hallmark gene sets** (50 curated biological pathways). A binary membership matrix $\mathbf{M} \in \{0, 1\}^{P \times G}$ is constructed:
$$M_{kg} = \begin{cases} 1 & \text{if gene } g \in \text{pathway } k \\ 0 & \text{otherwise} \end{cases}$$

The gene reconstructor weight is then initialized as $\mathbf{W}_{recon} \leftarrow \mathbf{M}^\top \in \mathbb{R}^{G \times P}$. This gives the model a biologically-grounded starting point where each pathway token is connected only to its known member genes.

---

## 7. Loss Functions and Gene Imbalance

### The Gene Imbalance Problem

Gene expression follows a **heavy-tailed distribution**: HOUSEKEEPING genes dominate, while signalling genes are rare. With standard MSE, high-expression genes dominate the loss.

The `--log-transform` flag (`log1p`) is the primary mitigation. Additionally, the following loss objectives are supported:

| CLI Flag | Objective | Focus |
| :--- | :--- | :--- |
| `mse` | Mean Squared Error | Magnitude accuracy at each spot. |
| `pcc` | Pearson Correlation | Spatial pattern coherence per gene. |
| `mse_pcc` | Combined Loss | Balances absolute magnitude and spatial shape. |

The complete training objective with pathway sparsity is:
$$\mathcal{L} = \mathcal{L}_{task} + \lambda \|\mathbf{W}_{recon}\|_1$$
