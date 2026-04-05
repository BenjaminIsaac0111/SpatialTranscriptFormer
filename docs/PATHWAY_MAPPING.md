# Pathway Mapping for Clinical Relevance

Mapping predicted pathway activities directly to histological context is the core purpose of the `SpatialTranscriptFormer`. Instead of predicting ~1,000 individual gene expression values, the model outputs a compact vector of **biological pathway activity scores** (e.g., 50 MSigDB Hallmark pathways) at each tissue spot. This is a bit more directly interpretable than predicting gene expression: clinicians can compare spatial "Wnt Signalling" or "EMT" activation maps rather than sifting through thousands of gene-level predictions and mapping them to pathways.

---

## 1. How Pathway Targets are Computed

Pathway activity scores are pre-computed offline from raw HEST `.h5ad` files using the `stf-compute-pathways` CLI tool (`src/spatial_transcript_former/recipes/hest/compute_pathway_activities.py`). This decouples target generation from model training and means any biological database can be used as a scoring backend without changing the model. One thing to note is that the pathway targets are computed from the raw counts, not the normalized counts and moran 1 is also precomputed for now, though I may change this to be diffencialy learned (or both?).

### Processing Pipeline (per sample)

For each `.h5ad` file, the following steps are applied in order:

| Step | Operation | Rationale |
| :--- | :--- | :--- |
| **1. QC Filtering** | Remove low-quality spots (min UMIs, min detected genes, max MT%) on **raw counts** | QC before normalisation prevents low-quality spots from distorting library-size estimates |
| **2. CP10k Normalisation** | Scale each spot to 10,000 total counts, then apply `log1p` | Corrects for sequencing depth differences between spots |
| **3. Gene Z-Scoring** | Standardise each gene across surviving spots (mean=0, std=1) | Eliminates housekeeping gene dominance; every gene gets equal weight |
| **4. Pathway Aggregation** | For each pathway: take the mean z-score of its member genes present in the matrix | Produces a single, comparable activity score per pathway per spot |
| **5. Moran I** | Compute Moran's I for each gene on the raw counts | Computes spatial autocorrelation for each gene |

Pathways with fewer than `--min-genes` (default: 5) detected members are filled with zeros. Samples with fewer than `--min-pathways` (default: 25) scorable pathways are excluded entirely.

### Default QC Thresholds

These defaults follow standard scRNA-seq / spatial transcriptomics QC practice though they may need to be adjusted for different tissue types or fine tuned for different datasets.

| Parameter | Default | Flag |
| :--- | :--- | :--- |
| Min UMI count per spot | 500 | `--qc-min-umis` |
| Min detected genes per spot | 200 | `--qc-min-genes` |
| Max mitochondrial read fraction | 15% | `--qc-max-mt` |
| CP10k normalisation target | 10,000 | `--target-sum` |

### Output Format

Each sample is saved as a compressed HDF5 file at `<data_dir>/pathway_activities/<sample_id>.h5`:

```text
activities      float32 (n_spots, n_pathways)   # z-scored pathway activity matrix
barcodes        bytes   (n_spots,)               # spot barcode strings
pathway_names   bytes   (n_pathways,)            # pathway name labels
attrs:
  n_spots_before_qc   int     # total spots in raw h5ad
  n_spots_after_qc    int     # spots surviving QC
  qc_min_umis         int
  qc_min_genes        int
  qc_max_mt           float
  n_scored_pathways   int     # pathways meeting the min_genes threshold
```

These files are consumed at training time by `HEST_FeatureDataset` when `--pathway-targets-dir` is provided (which defaults to `<data_dir>/pathway_activities`).

### Usage

```bash
# Compute with defaults (human-only samples auto-detected from HEST metadata)
stf-compute-pathways --data-dir hest_data

# Custom QC thresholds
stf-compute-pathways --data-dir hest_data --qc-max-mt 0.10 --qc-min-umis 1000

# Specific samples only
stf-compute-pathways --data-dir hest_data --sample-ids MEND29 TENX88

# Overwrite existing outputs
stf-compute-pathways --data-dir hest_data --overwrite
```

---

## 2. Pathway Databases

### Current: MSigDB Hallmark Gene Sets

The default scoring backend uses the **50 MSigDB Hallmark gene sets**, which summarise distinct, well-defined biological states and processes. These are ideal for cancer research because they are non-redundant and clinically well-characterised.

- **License**: MSigDB Hallmark sets (v6.0–v7.5.1, v2022.1+) are subject to the **CC BY 4.0** license.
- **Copyright**: © 2004–2025 Broad Institute, Inc., MIT, and Regents of the University of California.
- The GMT file is downloaded automatically and cached in `.cache/` on first use.

### Future: Extended Knowledge Bases

The offline preprocessing pipeline is designed to be database-agnostic. Future work will add first-class support for:

- **[decoupleR](https://decoupler-py.readthedocs.io) + [PROGENy](https://saezlab.github.io/progeny/)** (Saez lab) — mechanistic signalling pathway scores (14 cancer-relevant pathways) directly inferred from expression data.
- **[LIANA+](https://liana-py.readthedocs.io)** (Saez lab) — ligand-receptor interaction scores for cell-cell communication.
- **[CollecTRI](https://github.com/saezlab/collectri)** — transcription factor regulon activity.
- **Custom GMT files** — already supported in the scoring layer via `--custom-gmt` (any GMT file, local or URL).

---

## 3. Why Offline Pre-Computation?

Previous versions of SpatialTranscriptFormer used an **`AuxiliaryPathwayLoss`** that computed pathway pseudo-targets on-the-fly *from the training signal itself*, then supervised the model's internal pathway tokens against those computed values. This approach seemed to have fundamental circularity problem: the pathway targets were derived from the same gene expression the model was trying to predict.

The current design eliminates this entirely:

| Aspect | Old (Auxiliary Loss) | New (Pre-computed Targets) |
| :--- | :--- | :--- |
| Target source | Computed in-flight from training labels | Computed once, offline, from raw expression |
| QC & normalisation | None | Per-spot QC → CP10k → z-score |
| Model output | Gene expression (via gene reconstructor) | Pathway activity scores directly |
| Loss objective | `L_gene + λ · (1 - PCC(scores, pseudo-targets))` | `MSE + PCC` against pre-computed activities |
| Interpretability | Indirect (pathway scores were internal and needed to be mapped back to pathways) | Direct (output *is* the pathway activity) |

---

## 4. Generalising to Other Tissue Types

Because pathways are pre-computed once and stored, you can easily swap the gene set for different biology. Use the `--custom-gmt` flag to point to any GTM file:

```bash
# Example: Score only EMT and immune infiltration pathways
stf-compute-pathways --data-dir hest_data --custom-gmt path/to/my_custom.gmt
```

This makes the pipeline applicable to any disease of interest without changing the model architecture.

---

## 5. Spatial Pathway Visualisation

After training, the `stf-predict` CLI and the `Predictor` Python API produce spatial heatmaps of pathway activation across the tissue. Each pathway gets its own overlay plot, allowing direct visual comparison between, e.g., hypoxia activation and VEGF signalling.

```bash
stf-predict --data-dir hest_data --sample-id MEND29 \
    --model-path checkpoints/best_model.pth \
    --model-type interaction
```

Plots are saved to `./results/` by default.
