# Latent Pathway Discovery

The `SpatialTranscriptFormer` allows for the unsupervised discovery of biological processes directly from data. This "Data-Driven" approach doesn't rely on existing pathway databases like KEGG or MSigDB.

## Latent Pathway Discovery

The **SpatialTranscriptFormer** (specifically the `SpatialTranscriptFormer` class) can be used to discover unsupervised pathways from spatial transcriptomics data. The Pathway Bottleneck Layer forces the model to compress all image-to-gene mappings through a set of intermediate "factors." 

- Each **factor** acts as a learned pathway.
- The **Gene Reconstructor** matrix determines which genes belong to which latent pathway.
- By looking at the highest weights in this matrix, you can "decode" what biological process the model has discovered.

## Sparsity Regularization (L1)

To make discovered pathways cleaner and more interpretable, we use L1 regularization. This pushes low-contribution gene weights to zero, ensuring each latent factor is associated with a small, cohesive set of genes.

### Usage in Training

You can enable sparsity regularization using the `--sparsity-lambda` argument in `train.py`:

```bash
python src/spatial_transcript_former/train.py \
    --model interaction \
    --num-pathways 50 \
    --sparsity-lambda 0.001 \
    --data-dir A:/hest_data
```

## Interpreting Discovered Pathways

After training, you can inspect the `gene_reconstructor.weight` matrix `(G x P)` to name your pathways:

```python
# Pseudo-code for interpretation
weights = model.gene_reconstructor.weight.data # (GeneCount, PathwayCount)
for p in range(num_pathways):
    top_indices = torch.topk(weights[:, p], k=10).indices
    top_genes = [gene_names[i] for i in top_indices]
    print(f"Learned Pathway {p} Top Genes: {top_genes}")
```

### Example Clinical Insights
- **Factor A**: High weights for `VIM`, `SNAI1`, `ZEB1` -> Discovered **EMT** pathway.
- **Factor B**: High weights for `TFF3`, `CHGA`, `MUC2` -> Discovered **Secretory/Goblet** cell signatures.

By visualizing these factor scores as spatial heatmaps, you can see where these discovered biological processes are active in the tissue.
