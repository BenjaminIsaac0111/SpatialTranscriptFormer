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

### B. Pathway Bottleneck (Model Architecture)

The **SpatialTranscriptFormer** replaces the standard linear output head with a two-step projection that can be configured in two modes:

#### 1. Informed Projection (Prior Knowledge)

In this mode, the **Gene Reconstruction Matrix** $\mathbf{W}_{recon}$ is guided by established biological databases (MSigDB, KEGG).

- **Implementation**: $\mathbf{W}_{recon}$ is initialized as a binary mask $M \in \{0, 1\}^{G \times P}$ where $M_{gk} = 1$ if gene $g$ belongs to pathway $k$.
- **Benefit**: Predictions are guaranteed to be linear combinations of known biological processes, making them instantly interpretable by clinicians.

#### 2. Data-Driven Projection (Latent Discovery)

In this mode, the model learns its own "latent pathways" based on morphological patterns.

- **Implementation**: $\mathbf{W}_{recon}$ is randomly initialized and learned via backpropagation.
- **Sparsity Constraint**: We apply an L1 penalty to force the model to identify "canonical" gene sets: $L_{total} = L_{MSE} + \lambda \|\mathbf{W}_{recon}\|_1$.
- **Benefit**: Can discover novel spatial-transcriptomic relationships that aren't yet captured in curated databases.

- **Architecture Flow**:
    1. **Interaction**: Pathway tokens $P$ query the Histology $H$.
    2. **Activation**: A linear layer reduces $P_{tokens}$ to activation scores $S \in \mathbb{R}^P$.
    3. **Reconstruction**: $\hat{y} = S \cdot \mathbf{W}_{recon} + b$.

## 3. Clinical Application in Bowel Cancer

For colorectal cancer, we should prioritize monitoring these specific pathways:

| Pathway | Clinically Relevant Genes | Clinical Significance |
| :--- | :--- | :--- |
| **Wnt Signaling** | `CTNNB1`, `MYC`, `AXIN2` | Common driver in CRC (APC mutations) |
| **MMR / DNA Repair** | `MLH1`, `MSH2`, `MSH6` | MSI vs MSS status (Immunotherapy response) |
| **EMT** | `SNAI1`, `VIM`, `ZEB1` | Tumor invasion and metastasis risk |
| **Angiogenesis** | `VEGFA`, `FLT1` | Potential for anti-angiogenic therapy |

## 4. Next Steps

To implement this, we can:

1. Create a `mapping/` directory and download `.gmt` files from MSigDB.
2. Build a script to convert $(N \times G)$ prediction matrices into $(N \times P)$ pathway score matrices.
3. Visualize these scores as spatial heatmaps to show "Biological Hotspots."
