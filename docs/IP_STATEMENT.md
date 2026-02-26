# Intellectual Property Statement

This document identifies the core technological and methodological contributions of Benjamin Isaac Wilson within the **SpatialTranscriptFormer** project. The Author retains all intellectual property rights to these components.

## Scientific Context & Attribution

While this implementation introduces novel spatial constraints and transcriptomics-specific bottlenecks, the core multimodal interaction logic is inspired by and derived from the **SURVPATH** architecture ([Jaume et al., 2024](https://arxiv.org/abs/2303.15545)). This work focuses on the **adaptation and extension** of those concepts to the domain of high-dimensional spatial gene expression regression.

## 1. Model Architecture (SpatialTranscriptFormer)

The primary innovation is the **multimodal bottleneck transformer** designed for spatial transcriptomics regression. Key IP components include:

- **Pathway-Histology Interaction Layer**: The specific implementation of Cross-Attention where learnable pathway tokens (queries) interact with high-resolution histology tokens (keys/values).
- **Quadrant-Based Interaction Masking**: The logic used to zero out specific attention quadrants (e.g., $A_{H \to H}$) to optimize memory while maintaining multimodal context.
- **Biologically-Informed Reconstruction Bottleneck**: The specific matrix decomposition approach where gene expression is reconstructed from a linear combination of pathway activations.

### Proposed Auxiliary Pathway Loss

To prevent bottleneck collapse and provide a direct gradient signal to the pathway tokens, we use the `AuxiliaryPathwayLoss`. This loss compares the model's internal pathway scores against "ground truth" pathway activations computed from the gene expression targets via MSigDB membership.

The total objective becomes:
$$\mathcal{L} = \mathcal{L}_{gene} + \lambda_{aux} (1 - \text{PCC}(\text{pathway\_scores}, \text{target\_pathways}))$$

The `--log-transform` flag applies `log1p` to targets, mitigating the heavy-tailed gene expression distribution where housekeeping genes dominate MSE.

The full training objective with pathway sparsity regularisation:
$$\mathcal{L} = \mathcal{L}_{task} + \lambda \|W_{recon}\|_1$$

## 2. Spatial Context Methodologies

- **Euclidean-Gated Attention**: The implementation of spatial distance-based masking ($M_{spatial}$) to constrain model focus to local morphological regions.
- **Coordinate-Aware Augmentation**: Specific strategies for utilizing relative patch coordinates as inputs/constraints in the transformer pipeline.

## 3. Data Integration & Alignment

- **Gene-Lock Alignment Logic**: The specific implementation of automated gene set alignment across heterogeneous spatial transcriptomics technologies (Visium, Xenium, etc.).
- **Multi-Resolution Patch Loading**: Specialized dataloader logic for handling whole-slide histology integration with localized gene expression matrices.

## 4. Visualization & Interpretability

- **Histology-Gene Overlay Heatmaps**: The specific implementation of scaling and aligning predicted gene expression distributions onto full-resolution WSIs for clinical interpretation.

---

For technical inquiries or discussion regarding these methodologies, please contact the Author.
