# Model Zoo and Literature References

This document provides a summary of the models implemented in this project, their origins in literature, and the architectural equations for the proposed **SpatialTranscriptFormer**.

## 1. Regression Models (Patch-Level)

### HE2RNA (ResNet-50)

- **Reference**: [Schmauch et al. (2020). "A deep learning model to predict RNA-Seq expression of tumours from whole slide images." Nature Communications.](https://www.nature.com/articles/s41467-020-17679-z)
- **Description**: Uses a ResNet-50 backbone to extract features from histology patches and directly regresses a high-dimensional gene expression vector.
- **Equation**:
  $$\hat{y} = \text{FC}(\text{ResNet50}(x))$$
  Where $x$ is the histology patch and $\hat{y} \in \mathbb{R}^G$ is the predicted expression for $G$ genes.

### ViT-ST

- **Description**: An adaptation of the Vision Transformer (ViT) to the spatial transcriptomics task.
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

### Architecture Equations

#### 1. Image Encoding

Given an image patch $x$, we extract features $F$ using a backbone (e.g., ResNet or ViT):
$$F = \text{Encoder}(x), \quad F \in \mathbb{R}^{D}$$

#### 2. Pathway Tokenization

We initialize $K$ learnable pathway tokens $T_{path} \in \mathbb{R}^{K \times D_{token}}$. These tokens act as "queries" to search for relevant morphological features in the image.

#### 3. Interaction (Unified Early Fusion)

The interaction logic is unified via the **EarlyFusionBlock**, which facilitates a generalized attention mechanism between Pathway tokens ($P \in \mathbb{R}^{N_P \times d}$) and Histology tokens ($H \in \mathbb{R}^{N_H \times d$).

##### Pre-Interaction Enrichment (PE & Mixed Context)

Before attention, histology features $F$ are projected and enriched with spatial gradients and neighborhood morphology:

1. **Projection**: $H_0 = image\_proj(F)$
2. **Positional Encoding**: $H_{PE} = H_0 + \text{PE}_{2D}(x, y)$
3. **Local Mixing**: $H_{mixed} = \text{Mixer}(H_{PE})$ (only in `decoder` mode)

##### Interaction Forward Pass

The tokens are concatenated into a sequence $X = [P; H_{final}]$. The attention output $Z$ is:
$$Z = \text{Attention}(X, X, X; M)$$
Where $M$ is the quadrant mask. The final pathway activations are extracted from the first $N_P$ tokens:
$$Z_{path} = Z[1:N_P, :] \in \mathbb{R}^{N_P \times d}$$

##### Decomposition of Interaction Modes

The model supports two distinct interaction paths:

1. **Early Fusion (`jaume`)**:
   - Both modalities share the same transformer layers.
   - Pathways can refine each other ($Q_P K_P^\top$).
   - Pathways query histology ($Q_P K_H^\top$).

2. **Cross-Attention (`decoder`)**:
   - Pathways act as queries to attend to the histology neighborhood.
   - Equivalent to `jaume` with $A_{P \to P}$ and $A_{H \to P}$ quadrants masked.

##### Interaction Masking

For efficiency and to focus on multimodal interactions, we can apply an interaction mask $M$ to zero out specific quadrants:

- **Masking $A_{H \to H}$**: By setting this quadrant to $-\infty$, we eliminate the $O(N_H^2)$ complexity of patch-to-patch attention, reducing the memory bottleneck while maintaining context.
- **Spatial Radius Masking**: We further refine $A_{P \to H}$ by masking out patches beyond a radius $R$ from the center:
$$M_{spatial}(p_i, h_j) = \begin{cases} 0 & \text{if } \text{dist}(p_i, h_j) \leq R \\ -\infty & \text{otherwise} \end{cases}$$

#### 4. Spatial Inductive Biases

To better capture the spatial structure of tissue, the model incorporates two inductive biases:

##### Sinusoidal Positional Encoding (2D)

We inject knowledge of absolute patch locations $(x, y)$ into the transformer latent space via 2D sinusoidal embeddings:
$$memory = image\_proj(F) + \text{PE}_{2D}(x, y)$$
This ensures the attention mechanism is aware of the relative distances and orientations between histology features.

##### Local Patch Mixing (Scatter-Gather Conv)

For `decoder` mode, we introduce a **LocalPatchMixer** that uses a depthwise convolution over a local spatial grid:
$$memory_{mixed} = \text{GELU}(\text{DepthwiseConv2d}(memory_{grid}))$$
This allows the model to aggregate immediate neighborhood morphology (e.g., 3x3 window) through high-bandwidth spatial kernels before performing global cross-attention with pathway tokens.

### 5. Scalability: Nyström Approximation

To scale effectively to whole-slide contexts or extremely large neighborhoods ($N_H > 1000$), we utilize the **Nyström method** for approximating the self-attention matrix. This reduces the complexity from quadratic $O(N^2)$ to linear $O(N \cdot m)$.

#### Theoretical Intuition

The core of Transformer attention is the kernel matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$. Nyström approximates $\mathbf{K}$ using a small set of **landmark points** $m \ll N$:
$$\mathbf{K} \approx \mathbf{C} \mathbf{W}^{+} \mathbf{C}^\top$$
Where:

- $\mathbf{C} \in \mathbb{R}^{N \times m}$ contains $m$ landmark columns.
- $\mathbf{W} \in \mathbb{R}^{m \times m}$ is the intersection matrix of those columns.
- $\mathbf{W}^{+}$ is the Moore-Penrose pseudo-inverse.

#### Decomposition in Attention

In the context of attention $\text{Softmax}(\frac{QK^\top}{\sqrt{d}})V$, the Nyström approximation allows us to compute the interaction without ever forming the $N \times N$ matrix:
$$\tilde{A} = \text{Softmax}\left(\frac{Q \tilde{K}^\top}{\sqrt{d}}\right) \left[ \text{Softmax}\left(\frac{\tilde{K} \tilde{K}^\top}{\sqrt{d}}\right) \right]^{+} \text{Softmax}\left(\frac{\tilde{K} K^\top}{\sqrt{d}}\right)$$
Where $\tilde{K}$ are the pooled landmark features.

#### Why it matters for Spatial Transcriptomics

1. **Whole Slide Context**: Standard transformers fail on WSIs (e.g., 20,000 patches). Nyström enables slide-level correlation (implemented in the **TransMIL** model).
2. **Dense Neighborhoods**: Allows the **SpatialTranscriptFormer** to consider hundreds of surrounding patches for a single prediction with minimal GPU memory overhead.

#### 5. Gene Reconstruction (Biologically-Informed Bottleneck)

The final gene expression $\hat{y}$ is a linear combination of pathway activations $s_k$:
$$\hat{y}_g = \sum_{k=1}^K s_k \cdot W_{gk} + b_g$$

##### Matrix Decomposition

We can decompose the prediction into clinical (informed) and discovery (latent) components:
$$\hat{y} = \mathbf{S} \cdot (\mathbf{W}_{informed} \odot M_{mask} + \mathbf{W}_{latent}) + b$$

- $\mathbf{W}_{informed}$: Fixed weights or initialized from biological prior knowledge (e.g., MSigDB).
- $\mathbf{W}_{latent}$: Learned parameters for discovering novel gene-morphology relationships.

##### MSigDB Hallmarks Initialization

When `--pathway-init` is enabled, $\mathbf{W}_{recon}$ is initialized from the **MSigDB Hallmark gene sets** (50 curated biological pathways). A binary membership matrix $\mathbf{M} \in \{0, 1\}^{P \times G}$ is constructed:
$$M_{kg} = \begin{cases} 1 & \text{if gene } g \in \text{pathway } k \\ 0 & \text{otherwise} \end{cases}$$

The gene reconstructor weight is then initialized as:
$$\mathbf{W}_{recon} \leftarrow \mathbf{M}^\top \in \mathbb{R}^{G \times P}$$

This gives the model a biologically-grounded starting point where each pathway token is connected only to its known member genes. During training, gradients can refine these connections — adding novel gene-pathway associations and adjusting the strengths. With 1,000 global genes, the Hallmark sets typically cover ~54% of genes, leaving the remaining genes initially disconnected but learnable.

### Why this works

1. **Interpretability**: By inspecting $\mathbf{W}_{recon}$, we can determine which genes belong to which "latent pathway."
2. **Sparsity**: Applying L1 regularization ($L_{sparse} = \lambda \|\mathbf{W}_{recon}\|_1$) forces the model to learn distinct, sparse gene sets for each pathway.
3. **Morphology-Guided**: The Cross-Attention mechanism ensures that pathway activity is directly triggered by specific visual features in the histology.
4. **Biological Prior**: MSigDB initialization constrains the model to discover pathway activations grounded in known biology, enabling faster convergence and more clinically meaningful representations.

---

## 4. Loss Functions and Gene Imbalance

### The Gene Imbalance Problem

Gene expression follows a **heavy-tailed distribution**:

- **Housekeeping genes** (TMSB4X, TPT1, ACTB) can have counts in the thousands.
- **Signalling genes** (transcription factors, receptors) often have counts of 0–10.
- **Many genes** are zero-inflated (sparse across spots).

With standard MSE, the model disproportionately optimises for high-expression genes because their squared errors dominate the loss. Low-expression genes — which may be the most biologically interesting — are effectively ignored.

The `--log-transform` flag (`log1p`) is the primary mitigation, compressing the dynamic range from [0, 10000] to [0, 9.2]. Beyond this, we support multiple loss functions that address the problem from different angles.

### Mean Squared Error (MSE) — Magnitude Focus

$$\mathcal{L}_{MSE} = \frac{1}{N \cdot G} \sum_{i=1}^N \sum_{g=1}^G (\hat{y}_{ig} - y_{ig})^2$$

MSE measures **pointwise magnitude error**. Each spot is evaluated independently — there is no notion of spatial coherence. Higher-expression genes contribute proportionally more to the loss.

- **CLI**: `--loss mse` (default)
- **Strength**: Ensures absolute expression values are accurate.
- **Weakness**: Biased towards high-expression genes.

### Pearson Correlation Coefficient (PCC) — Spatial Pattern Focus

For each gene $g$, the Pearson correlation across all $N$ spots is:

$$\rho_g = \frac{\sum_{i=1}^N (\hat{y}_{ig} - \bar{\hat{y}}_g)(y_{ig} - \bar{y}_g)}{\sqrt{\sum_{i=1}^N (\hat{y}_{ig} - \bar{\hat{y}}_g)^2} \cdot \sqrt{\sum_{i=1}^N (y_{ig} - \bar{y}_g)^2}}$$

The PCC loss is then:

$$\mathcal{L}_{PCC} = 1 - \frac{1}{G} \sum_{g=1}^G \rho_g$$

PCC is **completely scale-invariant** — it mean-centres and normalises each gene, so it measures only the *spatial pattern*, not *magnitude*. A gene with range [0, 3] contributes equally to a gene with range [100, 5000].

- **CLI**: `--loss pcc`
- **Strength**: Every gene contributes equally regardless of expression level. Captures spatial coherence.
- **Weakness**: Ignores absolute magnitude — predictions could be scaled arbitrarily and still achieve PCC = 1.0.

### Comparison

| Property | MSE | PCC |
| :--- | :--- | :--- |
| **Optimises for** | Correct magnitude at each spot | Correct spatial pattern per gene |
| **Scale-invariant?** | No — biased to high-expression genes | Yes — all genes weighted equally |
| **Spatial awareness?** | None — each spot independent | Yes — evaluates across all spots |
| **Failure mode** | Ignores low-expression genes | Allows arbitrary magnitude scaling |

### Combined MSE + PCC

A combined objective captures both magnitude and spatial pattern:

$$\mathcal{L}_{combined} = \mathcal{L}_{MSE} + \alpha \cdot \mathcal{L}_{PCC}$$

- **CLI**: `--loss mse_pcc`
- **Logic**: The combined loss ensures that the model learns to predict both the correct absolute intensity and the correct spatial distribution of gene expression. This is particularly effective for rare cell types or signalling gradients.

### Full Training Objective

The complete loss for the SpatialTranscriptFormer with pathway bottleneck and sparsity regularisation:

$$\mathcal{L} = \underbrace{\mathcal{L}_{task}}_{\text{MSE, PCC, or MSE\_PCC}} + \underbrace{\lambda \|\mathbf{W}_{recon}\|_1}_{\text{Pathway Sparsity}}$$

- **CLI**: `--sparsity-lambda 0.05`
- The L1 term encourages the gene reconstructor to maintain sparse, pathway-like groupings, preserving the biological structure from MSigDB initialisation during training.
