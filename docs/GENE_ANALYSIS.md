# Gene Analysis and Modeling Strategies

This document outlines the available gene sets for modeling in the Bowel Cancer dataset, based on an analysis of 84 Human Bowel samples across different technologies (Visium, Visium HD, and Xenium).

## Gene Availability by Platform

Biological analysis is constrained by the intersection of genes available across different spatial transcriptomics platforms.

| Scope | Sample Count | Common Genes | Recommendation |
| :--- | :--- | :--- | :--- |
| **All Human Bowel** | 84 (Visium + Xenium) | ~405 | Cross-platform benchmarking |
| **Visium Only** | 78 | ~1060 | In-depth spatial profiling |

### 1. The "Bowel Core" Gene Set (405 genes)

This set represents the intersection of the Xenium panel and the Visium whole-transcriptome capture.

- **Pros**: Allows the model to be trained on Visium data and evaluated on high-resolution Xenium data.
- **Cons**: Limited to a smaller subset of genes, which might miss important specific pathways.

### 2. The "Visium Pan-Bowel" Gene Set (1060 genes)

This set includes all genes present in every Visium sample in the HEST dataset.

- **Pros**: Provides a much larger feature space (predicting 1000+ genes).
- **Cons**: Cannot be directly evaluated on Xenium samples without imputation or subsetting.

## Implementation in the Dataloader

The current dataloader implementation in `src/spatial_transcript_former/data/dataset.py` uses a "Gene Lock" mechanism:

1. The first sample in the training loop determines the target gene list.
2. All subsequent samples are aligned to this list (missing genes are filled with zeros).

### Recommended Strategy

To ensure the best model stability, it is recommended to provide an explicit list of gene names to the `get_hest_dataloader` function instead of relying on the first sample's top genes.

```python
# Create a fixed gene list for the project
bowel_genes = [...] # The 405 or 1060 common genes

dataloader = get_hest_dataloader(
    ids=sample_ids,
    selected_gene_names=bowel_genes,
    ...
)
```

## How to find specific genes?

You can use the `inspection/analyze_gene_overlap.py` script to generate custom gene sets based on your specific sample filtering criteria.
