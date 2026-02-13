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

---

## 3. SpatialTranscriptFormer (Proposed Model)

The **SpatialTranscriptFormer** (formerly Pathway Interaction Model) introduces a biologically-informed bottleneck layer using Cross-Attention between Learned Pathways and Image Features.

### Architecture Equations

#### 1. Image Encoding

Given an image patch $x$, we extract features $F$ using a backbone (e.g., ResNet or ViT):
$$F = \text{Encoder}(x), \quad F \in \mathbb{R}^{D}$$

#### 2. Pathway Tokenization

We initialize $K$ learnable pathway tokens $T_{path} \in \mathbb{R}^{K \times D_{token}}$. These tokens act as "queries" to search for relevant morphological features in the image.

#### 3. Interaction (Multimodal Fusion & Quadrant Masking)

The interaction can be formulated as a generalized attention mechanism between Pathway tokens ($P \in \mathbb{R}^{N_P \times d}$) and Histology tokens ($H \in \mathbb{R}^{N_H \times d}$).

##### Unified Self-Attention Formulation

The tokens are concatenated into a sequence $X = [P; H]$. The attention output $Z \in \mathbb{R}^{(N_P + N_H) \times d}$ is:
$$Z = \text{Attention}(X, X, X; M)$$
Where $M$ is the quadrant mask. The final pathway activations are extracted from the first $N_P$ tokens:
$$Z_{path} = Z[1:N_P, :] \in \mathbb{R}^{N_P \times d}$$

##### Decomposition of Interaction (Jaume vs. Decoder)

The model supports two distinct interaction paths that are functionally related:

1. **Early Fusion (`jaume`)**:
   $$Z_{path} = \text{Softmax}\left( \frac{Q_P K_P^\top + Q_P K_H^\top}{\sqrt{d}} \right) [V_P; V_H]$$
   - $Q_P K_P^\top$: Pathway-to-Pathway self-correction (allows pathways to refine each other).
   - $Q_P K_H^\top$: Pathway-to-Histology querying (extracts morphological evidence).

2. **Cross-Attention (`decoder`)**:
   $$Z_{path} = \text{Softmax}\left( \frac{Q_P K_H^\top}{\sqrt{d}} + M_{spatial} \right) V_H$$
   - This mode is equivalent to `jaume` if $A_{P \to P}$ and $A_{H \to P}$ are masked and $V_P$ is null.

##### Interaction Masking

For efficiency and to focus on multimodal interactions, we can apply an interaction mask $M$ to zero out specific quadrants:

- **Masking $A_{H \to H}$**: By setting this quadrant to $-\infty$, we eliminate the $O(N_H^2)$ complexity of patch-to-patch attention, reducing the memory bottleneck while maintaining context.
- **Spatial Radius Masking**: We further refine $A_{P \to H}$ by masking out patches beyond a radius $R$ from the center:
$$M_{spatial}(p_i, h_j) = \begin{cases} 0 & \text{if } \text{dist}(p_i, h_j) \leq R \\ -\infty & \text{otherwise} \end{cases}$$

### 4. Scalability: Nyström Approximation

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

### Why this works

1. **Interpretability**: By inspecting $\mathbf{W}_{recon}$, we can determine which genes belong to which "latent pathway."
2. **Sparsity**: Applying L1 regularization ($L_{sparse} = \lambda \|\mathbf{W}_{recon}\|_1$) forces the model to learn distinct, sparse gene sets for each pathway.
3. **Morphology-Guided**: The Cross-Attention mechanism ensures that pathway activity is directly triggered by specific visual features in the histology.
