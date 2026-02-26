# Pathway Mapping for Clinical Relevance

Mapping predicted gene expression into biological pathways is key to making the `SpatialTranscriptFormer` clinically interpretable. Instead of looking at 1,000 individual genes, clinicians can look at the activity of specific processes (e.g., "Wnt Signaling" or "EMT").

## 1. Mapping Resources

We recommend using the following curated databases for mapping:

- **MSigDB Hallmark**: 50 gene sets that summarize specific biological states or processes. This is the "gold standard" for general cancer research because it's non-redundant and well-defined.
- **KEGG & Reactome**: More detailed, hierarchical pathways that describe specific biochemical reactions.
- **Gene Ontology (GO)**: Useful for finding genes associated with specific molecular functions or cellular components.

## 2. Technical Mapping Approach

There are three ways to implement this in the current architecture:

### A. Post-Hoc Enrichment (Diagnostics)

After the model makes predictions (N spots x G genes), we run a statistical test (e.g., Gene Set Enrichment Analysis or a simple hypergeometric test) to see which pathways are "upregulated" in specific spatial regions.

- **Tool**: `gseapy` or a custom mapping script.
- **Use Case**: Generating a "Pathway Activation Map" from a trained model's output.

### B. Interaction Model via Multi-Task Learning (MTL)

The **SpatialTranscriptFormer** interaction model inherently represents pathway activations as part of its attention mechanism and output process. Rather than a simple linear bottleneck, it utilizes learnable pathway tokens and Multi-Task Learning (MTL).

#### 1. Informed Supervision via Auxiliary Loss

In this mode, the network receives direct supervision on its pathway tokens, guided by established biological databases (e.g., MSigDB):

- **Architecture Flow**:
    1. **Interaction**: Learnable pathway tokens $P$ interact with Histology patch features $H$ via self-attention (e.g., $p2h$, $h2p$).
    2. **Activation**: Pathway scores $S \in \mathbb{R}^P$ are computed using a learnable temperature-scaled cosine similarity between the pathway tokens and image patch tokens.
    3. **Gene Reconstruction**: $\hat{y} = S \cdot \mathbf{W}_{recon} + b$, where $\mathbf{W}_{recon}$ is initialized using the binary pathway membership matrix $M$.
- **MTL Auxiliary Loss**: To prevent standard bottleneck collapse, an explicit auxiliary loss bridges the spatial representations directly to biological data. The pathway scores $S$ are supervised against a pathway ground truth ($Y_{genes} \cdot M^T$) using a Pearson Correlation Coefficient (PCC) loss.
  $$L_{total} = L_{gene} + \lambda_{pathway} (1 - PCC(S, Y_{genes} \cdot M^T))$$
- **Benefit**: The model is forced to explicitly align its internal interaction tokens with concrete biological pathways, granting direct interpretability.

#### 2. Data-Driven Discovery (Latent Projection)

In the absence of a biological prior, the model can learn its own "latent pathways".

- **Implementation**: $\mathbf{W}_{recon}$ is randomly initialized and the auxiliary pathway loss is disabled.
- **Sparsity Constraint**: We apply an L1 penalty to force the model to identify "canonical" sparse gene sets: $L_{total} = L_{gene} + \lambda_{sparsity} \|\mathbf{W}_{recon}\|_1$.
- **Benefit**: Can discover novel spatial-transcriptomic relationships that aren't yet captured in curated databases.

## 3. Generalizing to HEST1k Tissues

The model supports any dataset within the HEST1k collection (e.g., Breast, Kidney, Lung, Colon). Instead of being bound to a single disease context, users can leverage the `--custom-gmt` flag to map genes to pathways relevant to their specific investigation.

### Example: Profiling the Tumor Microenvironment

Regardless of the tissue of origin (e.g., Kidney versus Breast), researchers often track core functional states within the tumor microenvironment. A user might define a `.gmt` file to explicitly monitor:

| Pathway Concept | Hallmarks / Relevant Genes | Interpretive Value across Tissues |
| :--- | :--- | :--- |
| **Hypoxia & Angiogenesis** | `VEGFA`, `FLT1`, `HIF1A` | Identifies oxygen-deprived or highly vascularized tumor cores. |
| **Immune Infiltration** | `CD8A`, `GZMB`, `IFNG` | Maps regions of active anti-tumor immune response. |
| **Stromal / EMT** | `VIM`, `SNAI1`, `ZEB1` | Highlights desmoplastic stroma and invasion fronts. |
| **Proliferation** | `MKI67`, `PCNA`, `MYC` | Pinpoints highly active, dividing cell populations. |

By supplying these functional groupings via `--custom-gmt`, the model's MTL process explicitly aligns its spatial interaction tokens to monitor these exact states across any whole-slide image in the HEST1k dataset.

## 4. Implementation Status

### Implemented

- **MSigDB Hallmarks Initialization** (`--pathway-init` flag): Downloads the GMT file, matches genes against `global_genes.json`, and initializes `gene_reconstructor.weight` with the binary membership matrix. See [`pathways.py`](../src/spatial_transcript_former/data/pathways.py).
  - 50 Hallmark pathways (default fixed fallback when using `--pathway-init`).
  - GMT file cached in `.cache/` after first download.
- **Custom Pathway Definitions** (`--custom-gmt` flag): Users can override the default Hallmarks by providing a URL or local path to a `.gmt` file, enabling custom database integrations (e.g., KEGG, Reactome, or highly specific tissue masks).

- **Sparsity Regularization** (`--sparsity-lambda` flag): L1 penalty on `gene_reconstructor` weights to encourage pathway-like groupings when using data-driven (random) initialization.

### Usage

```bash
# With biological initialization (50 MSigDB Hallmarks)
python -m spatial_transcript_former.train \
    --model interaction --pathway-init ...

# With data-driven pathways + sparsity
python -m spatial_transcript_former.train \
    --model interaction --num-pathways 50 --sparsity-lambda 0.01 ...
```

- **Spatial Pathway Maps**: Visualize pathway activations as spatial heatmaps overlaid on histology using `stf-predict`. See the [README](../README.md) for inference instructions.

### Future Work

- **Post-Hoc Enrichment**: `gseapy` integration for pathway activation maps from model outputs without architectural bottlenecks.
- **End-to-End Risk Assessment Module**: Developing a downstream prediction system that takes the spatially-resolved pathway activations and gene expressions derived from the model and maps them directly to clinical risk and survival outcomes.
