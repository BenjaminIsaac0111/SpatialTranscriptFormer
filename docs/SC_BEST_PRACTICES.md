# Research & Improvement Roadmap

Best-practices alignment and literature-driven improvement leads for SpatialTranscriptFormer, with competitive context from the spatial transcriptomics literature.

---

## Current Strengths


These areas already follow industry best practices:

- **Global Gene Vocabulary** — `build_vocab.py` enforces a consistent feature space
  across all samples, preventing feature mismatch at training and inference time.
- **SVG-aware Gene Selection** — Moran's I scoring is implemented in
  `data/spatial_stats.py` and exposed via `stf-build-vocab --svg-weight`. Genes with
  high spatial autocorrelation are the strongest learning targets for a spatially-aware
  model.
- **Spatial Coherence Validation** — `spatial_coherence_score()` compares predicted
  vs. ground-truth Moran's I for the top-50 SVGs, logging a Pearson correlation as a
  validation metric alongside MSE/PCC. Computed automatically in `training/engine.py`.
- **Spatial Context via Neighbourhoods** — KD-tree-based neighbour aggregation in
  `HEST_FeatureDataset` provides local spatial context to patch features.
- **Coordinate Standardisation** — `normalize_coordinates()` prevents spatial scale
  bias between slides from different platforms (Visium, Visium HD, etc.).
- **Pathway-Aware Feature Selection** — MSigDB pathway gene prioritisation in the
  vocabulary builder ensures biologically relevant signal even with a limited gene
  budget.
- **Statistical Loss Modelling** — `ZINBLoss` accounts for the overdispersion and
  sparsity inherent in raw count data.
- **Histology-Gene Integration** — The quad-flow architecture follows recommended
  multi-modal integration patterns; pathway tokens act as a structured biological
  bottleneck analogous to the Perceiver cross-attention design.

---

## Technology & Competitive Landscape

*Source: Yue et al. (2023), Spatial transcriptomics guidebook; Nat. Comms. benchmark 2025.*

**Visium is the right primary target.** The guidebook classifies 22 ST technologies and
confirms 10x Visium (55 µm spots, 1–10 cells/spot) as the dominant commercial platform
driving the majority of downstream computational tool development. This validates
HEST-1k as the appropriate training corpus.

**The resolution ceiling is real but addressable.** Visium spots are multi-cellular by
design. GHIST (Nat. Methods, 2025), which leverages Xenium sub-cellular data, represents
the current performance ceiling. Scaling to Visium HD or Xenium is an architectural
non-event — it is purely a data availability and preprocessing question since the model
is already coordinate-agnostic.

**No method dominates across all criteria.** The 2025 Nat. Comms. benchmark evaluated
11 methods across 5 datasets using 28 metrics in five categories: prediction accuracy,
generalisability, translational potential, usability, and computational efficiency.
No single method wins across all categories. This is the framework for external
evaluation.

### Competitive Methods

| Method | Venue | Relative Strength |
|--------|-------|-------------------|
| **HisToGene** | — | Best generalisability across datasets |
| **DeepSpaCE** | — | Best generalisability across datasets |
| **Hist2ST** | NeurIPS 2023 | Generalisability; contrastive bi-modal training |
| **EGNv2** | — | Highest per-platform accuracy on ST data |
| **DeepPT** | — | Highest accuracy on Visium specifically |
| **GHIST** | Nat. Methods 2025 | Single-cell resolution via Xenium |

STF's differentiators in this landscape are **pathway-level interpretability** and
**auxiliary biological supervision** — both under-represented in the 2025 benchmark's
*translational potential* category, where competing methods score poorly.

### Prior Knowledge: Saezlab Ecosystem (EMBL Heidelberg)

*Session: March 2026. Sources: Schubert et al. 2018; Badia-I-Mompel et al. 2022;
Dimitrov et al. 2022/2024; Tanevski et al. 2022; Müller-Dott et al. 2023.*

The Saez-Rodriguez lab (EMBL Heidelberg / RWTH Aachen) maintains a family of
interoperable prior knowledge resources and statistical tools that are directly
relevant to the SpatialTranscriptFormer pathway bottleneck.

**PROGENy** (Schubert et al., Nat. Comms. 2018) — 14 cancer-relevant signalling
pathways (EGFR, MAPK, PI3K, TGFb, TNFa, Hypoxia, VEGF, WNT, p53, NFkB, JAK-STAT,
Androgen, Estrogen, Trail) derived empirically from a large compendium of perturbation
experiments. Each gene carries a **continuous signed weight** (direction and magnitude
of pathway response). This is fundamentally different from MSigDB Hallmarks, which
use binary unweighted membership. PROGENy weights can be used directly as the
initialisation matrix for the linear gene reconstructor (`--pathway-init`). Holland
et al. (Genome Biology, 2020) benchmarked PROGENy on scRNA-seq and Visium data,
confirming it performs well in the spot-level transcriptome regime.

**decoupleR** (Badia-I-Mompel et al., Bioinformatics Advances, 2022) — A Python/R
package that applies a family of statistical estimators (MLM, ULM, GSEA, AUCell,
consensus) to any (resource, method) combination. `get_progeny()` and `run_mlm()`
provide PROGENy activity scores out of the box. The official documentation includes
a Visium spatial transcriptomics notebook. This is the practical bridge for computing
weighted pathway activity scores from HEST-1k expression matrices — directly
applicable as `AuxiliaryPathwayLoss` targets instead of the current binary MSigDB
membership scores.

**CollecTRI** (Müller-Dott et al., Nucleic Acids Research, 2023) — The recommended
successor to DoRothEA for transcription factor regulons. Integrates 12 resources for
1186 human TFs with signed interactions, consistently outperforming other regulon
collections in perturbation benchmarks. TF activity tokens are a longer-term
architectural direction: complementary to pathway tokens, operating at a finer
regulatory resolution.

**LIANA+** (Dimitrov et al., Nature Cell Biology, 2024) — Extends LIANA's
ligand-receptor benchmarking to spatially resolved data using location-weighted cosine
similarity scoring. The spatially-weighted LR scoring is conceptually aligned with
the `h2p` / `p2h` attention flows in STF's quad-flow model. LIANA+ LR pairs from
OmniPath are the highest-quality resource for supervising intercellular communication
pathway tokens (see item 8).

**MISTY / mistyR** (Tanevski et al., Genome Biology, 2022) — A multiview spatial
analysis framework that decomposes how much of a marker's variance is explained by
intrinsic vs. juxtaposition vs. neighbourhood views. Not a prediction tool, but
useful as a **diagnostic**: running MISTY on HEST-1k training data would reveal
whether PROGENy pathway activities are primarily driven by local (intrinsic) or
neighbourhood (para) context, informing how strongly the `h2h` interaction flow
matters for pathway token learning.

**OmniPath** (Türei et al., Mol. Systems Biology, 2021) — The meta-database
underpinning the entire Saezlab ecosystem (PROGENy, DoRothEA, LIANA). Aggregates
100+ resources covering directed signed protein interactions, enzyme-PTM, protein
complexes, pathway annotations, and ligand-receptor pairs. Accessible via the
`omnipath` Python package as a single programmatic interface.

**SPathDB** (Li et al., Nucleic Acids Research, 2025) — Pre-computed pathway activity
scores for 114,998 pathways across 695 spatial transcriptomics slices from 84 public
datasets. Not a Saezlab product, but relevant as a reference atlas: STF pathway token
outputs can be validated against SPathDB ground-truth pathway activity maps for the
same tissue types.

Coverage trade-offs by resource:

| Resource | Pathways/TFs | Gene weights | Spatial validation |
|----------|-------------|-------------|-------------------|
| MSigDB Hallmarks (current) | 50 | Binary | No |
| PROGENy | 14 signalling | Continuous, signed | Yes (Visium, decoupleR) |
| CollecTRI | ~1186 TFs | Continuous, signed | Indirect |
| Combined (PROGENy + Hallmarks) | 64 total | Mixed | Partial |

The 14 PROGENy pathways cover signalling cascades; the 50 Hallmarks cover broader
biological programs (proliferation, metabolism, immunity). A **hybrid initialisation**
using PROGENy weights for signalling tokens and Hallmark membership for the remaining
tokens has not been published but is a natural next step.

---

## Improvement Roadmap

Status key: ✅ Implemented | 🔧 Open | 💡 Research lead

### Vocabulary & Preprocessing

**1. Mitochondrial gene filtering** 🔧 — High priority, low effort

`global_genes_stats.csv` shows MT-CO3 and MT-CO2 in the top-20 most expressed genes.
Mitochondrial genes (`MT-*`) reflect cell health and apoptotic state, not spatial
morphological patterns, and are universally filtered in standard ST preprocessing.
Their presence inflates expression-rank scores for non-informative targets.

*Fix*: Exclude genes matching the `MT-` prefix before ranking in `build_vocab.py`.

**2. SVG-weighted vocab rebuild** 🔧 — High priority, low effort

The current `global_genes.json` was built with expression-only ranking — the
`global_genes_stats.csv` has no `morans_i` column, confirming `--svg-weight=0.0` was
used. SVG selection is now validated standard practice (SpatialDE, SPARK, NNSVG are
catalogued as de facto tools in the guidebook).

*Fix*: Re-run `stf-build-vocab --svg-weight 0.5 --svg-k 6` after applying the
MT-gene filter above.

**3. Vectorise `morans_i_batch`** 🔧 — Medium priority, low effort

`spatial_stats.py:morans_i_batch` loops over G genes with individual Python calls.
For 38,839 genes across many samples this is the bottleneck for SVG-weighted runs.
One sparse matrix multiply replaces the loop:

```python
z   = expression - expression.mean(axis=0)   # (N, G)
lag = W.dot(z)                                # (N, G)
num = (z * lag).sum(axis=0)                   # (G,)
den = (z**2).sum(axis=0)                      # (G,)
scores = np.where(den > 1e-12, (n / W_sum) * num / den, 0.0)
```

**4. Dispersion-based (HVG) gene filtering** 🔧 — Medium priority, medium effort

Beyond total counts and spatial variability, filtering for **Highly Variable Genes**
using dispersion metrics (as in `sc.pp.highly_variable_genes`) focuses the model on
genes that carry biological variation between tissue states rather than static
structural signal. This is complementary to Moran's I: HVG captures across-sample
variation while Moran's I captures within-sample spatial structure.

**5. Standardised library-size normalisation** 🔧 — Medium-High priority, medium effort

The pipeline currently lacks a standardised CPM/CP10k normalisation step before
`log1p`. Without it, sequencing depth variation between spots biases predictions
toward highly-sequenced spots. Standard: normalise to 10,000 counts per spot, then
`log1p`.

**6. Per-spot quality control** 🔧 — Medium priority, medium effort

Explicit QC thresholds (minimum UMI count, minimum detected genes, maximum
mitochondrial fraction) in the dataset loading scripts would protect the model from
training on low-quality "noise" spots. The mitochondrial fraction threshold directly
complements the MT-gene vocabulary filter above.

---

### Training & Supervision

**Architectural direction: pathway activity as the primary task.** The current model is
trained to predict 1000 individual gene expression values, with pathway scores as an
auxiliary, interpretability-oriented output. This framing has two compounding problems:

1. **Noisy learning signal.** Even after log-normalisation and SVG-based vocabulary
   selection, MSE over 1000 genes is dominated by high-expression genes — not
   necessarily the spatially informative ones. Moran's I weighting fixes gene
   *selection* but not the *gradient* during training.

2. **The `AuxiliaryPathwayLoss` is circular.** Its ground-truth targets are computed
   from the same gene expression being predicted (binary MSigDB membership sums). It
   constrains the intermediate representation to match a re-parameterisation of the
   output, adding no independent biological information.

**The cleaner framing** is to invert the task hierarchy:

```
PRIMARY:    H&E  →  pathway activity maps    (spatially coherent, interpretable)
SECONDARY:  pathway scores  →  gene expression    (optional imputation head)
```

Pathway activity maps pre-computed offline via decoupleR + PROGENy are:
- **Spatially cleaner** than individual genes — Moran's I on pathway aggregates is
  typically 3–5× higher than on individual gene expression
- **Adaptable** — the offline preprocessing step can be swapped for any biologically
  informed prior (PROGENy for signalling, Hallmarks for broader programs, LIANA+ for
  intercellular communication, CollecTRI for TF activity) without changing the model
- **Genuinely supervised** — the pathway scores are first-class outputs, not
  constraints on an intermediate representation

The `AuxiliaryPathwayLoss` in its current form is superseded by this framing. The
gene reconstructor becomes an optional secondary head rather than the primary loss
target.

---

**7. Pre-compute pathway activity targets (decoupleR + PROGENy)** 💡 — High priority, medium effort

For each HEST sample, run `decoupler.run_mlm(expression, net=get_progeny())` to
produce a `(spots × 14)` signed pathway activity matrix. Cache alongside the existing
feature `.h5` files. These become the primary prediction targets replacing the
`AuxiliaryPathwayLoss`.

The decoupleR MLM estimator fits: `expression[:, g] = a_p · W[g, p] + b_p + ε` and
solves for `a_p` — the pathway activity at each spot — via OLS. The result is
continuous, signed, and validated against known pathway perturbation experiments. The
prior knowledge source (PROGENy, Hallmarks, LIANA+ LR pairs, CollecTRI TF regulons)
can be swapped at the preprocessing step without touching the model.

**8. Moran's I weighted gene loss** 🔧 — High priority, low effort

If gene prediction is retained as a secondary head, weight each gene's contribution
to the loss by its Moran's I score so that spatially coherent genes drive the
gradient:

```python
L_gene = sum(w_g * MSE(pred_g, target_g) for g in genes) / sum(w_g)
# where w_g = MoranI_g  (pre-computed during stf-build-vocab --svg-weight > 0)
```

This is the training-time counterpart to SVG-based vocabulary selection — both
ensure spatially informative genes dominate, one at selection time and one at
training time.

**9. PROGENy pathway token initialisation** 💡 — High priority, medium effort

Replace or supplement the binary MSigDB Hallmark membership matrix used in
`--pathway-init` with PROGENy's continuous signed footprint weights. Each gene
carries a directional weight reflecting its perturbation-validated response
magnitude, rather than binary membership.

Hybrid strategy: initialise the first 14 pathway tokens from PROGENy weights
(signalling), the remaining 36 from Hallmarks membership (metabolic, immune,
proliferation). Requires additions to `data/pathways.py` and a new
`--pathway-init progeny` option in `training/builder.py`.

**10. Cell–cell interaction pathway tokens (LIANA+)** 💡 — Low priority, high effort

Under the primary-task framing, a subset of pathway tokens could be supervised by
LIANA+ spatially-weighted ligand-receptor scores (from OmniPath) rather than
intracellular pathway activity. This would produce communication tokens encoding
intercellular signalling alongside the intracellular pathway tokens — a natural
extension of the adaptable preprocessing approach in item 7.

**11. Cell-type deconvolution secondary head** 💡 — Medium priority, high effort

Pre-computed cell2location or RCTD deconvolution proportions could serve as an
additional secondary prediction head alongside gene expression — useful for
datasets where cell-type spatial organisation is the primary question.

---

### Evaluation, Scale & Tooling

**12. Evaluate on the Nat. Comms. 2025 benchmark suite** 🔧 — High priority, medium effort

The 2025 benchmark provides a ready-made evaluation framework: 5 public datasets, 28
metrics, 5 categories. SpatialTranscriptFormer has not yet been evaluated on this
suite. Submitting results, particularly in the *translational potential* category
(where pathway interpretability is the differentiator), is the clearest path to
situating STF in the published competitive landscape.

**13. Visium HD / Xenium data** 💡 — Low priority, high effort

The architecture is platform-agnostic. Scaling to sub-cellular resolution platforms
(Xenium: ~10 µm, Stereo-seq: 0.22 µm) is blocked only by data availability and
preprocessing pipelines, not the model itself. GHIST demonstrates that Xenium-derived
training labels substantially improve resolution.

**14. Preprocessing data contract** 🔧 — Low priority, low effort

Explicitly documenting which normalisation is applied at which stage (per-spot QC →
library-size normalisation → log1p → gene selection) and how the vocabulary was built
(which `--svg-weight`, which `--pathways`, run date) prevents silent mismatches between
training and inference. A short header in each output folder (`vocab_config.json`)
would suffice.

---

## Quick-Reference Summary

| # | Direction | Priority | Effort | Status |
|---|-----------|----------|--------|--------|
| 1 | MT-gene filter in `build_vocab.py` | High | Low | 🔧 Open |
| 2 | Rebuild vocab with `--svg-weight 0.5` | High | Low | 🔧 Open |
| 3 | Vectorise `morans_i_batch` | Medium | Low | 🔧 Open |
| 4 | HVG dispersion-based gene filtering | Medium | Medium | 🔧 Open |
| 5 | Library-size normalisation (CP10k) | Medium-High | Medium | 🔧 Open |
| 6 | Per-spot QC thresholds | Medium | Medium | 🔧 Open |
| 7 | Pre-compute pathway targets (decoupleR + PROGENy) | High | Medium | 💡 Lead |
| 8 | Moran's I weighted gene loss | High | Low | 🔧 Open |
| 9 | PROGENy pathway token initialisation | High | Medium | 💡 Lead |
| 10 | Cell–cell interaction tokens (LIANA+) | Low | High | 💡 Lead |
| 11 | Cell-type deconvolution secondary head | Medium | High | 💡 Lead |
| 12 | Nat. Comms. 2025 benchmark evaluation | High | Medium | 🔧 Open |
| 13 | Scale to Visium HD / Xenium | Low | High | 💡 Lead |
| 14 | Preprocessing data contract doc | Low | Low | 🔧 Open |

---

## References

- Yue et al. (2023). A guidebook of spatial transcriptomic technologies, data resources
  and analysis approaches. *CSBJ*, 21, 940–955.
  [DOI](https://doi.org/10.1016/j.csbj.2023.01.016) |
  [PMC10781722](https://pmc.ncbi.nlm.nih.gov/articles/PMC10781722/)
- Benchmarking the translational potential of spatial gene expression prediction from
  histology. *Nat. Comms.* 2025, 16(1):1544.
  [DOI](https://doi.org/10.1038/s41467-025-56618-y) |
  [PMC11814321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11814321/)
- GHIST: Spatial gene expression at single-cell resolution from histology.
  *Nat. Methods* 2025.
  [PMC12446070](https://pmc.ncbi.nlm.nih.gov/articles/PMC12446070/)
- Deep learning in spatially resolved transcriptomics: a comprehensive technical view.
  *Briefings in Bioinformatics* 2024.
  [Oxford Academic](https://academic.oup.com/bib/article/25/2/bbae082/7628264)
- Single-cell Best Practices (Theis Lab): [sc-best-practices.org](https://www.sc-best-practices.org)
- Squidpy: [squidpy.readthedocs.io](https://squidpy.readthedocs.io)
- Schubert et al. (2018). PROGENy: perturbation-response genes reveal signaling footprints in cancer.
  *Nat. Comms.* 9:20. [DOI](https://doi.org/10.1038/s41467-017-02391-6) | [PMC5750219](https://pmc.ncbi.nlm.nih.gov/articles/PMC5750219/)
- Holland et al. (2020). Robustness of TF and pathway tools on single-cell RNA-seq.
  *Genome Biology* 21:36. [DOI](https://doi.org/10.1186/s13059-020-1949-z) | [PMC7017576](https://pmc.ncbi.nlm.nih.gov/articles/PMC7017576/)
- Badia-I-Mompel et al. (2022). decoupleR: ensemble methods to infer biological activities.
  *Bioinformatics Advances* 2(1):vbac016. [DOI](https://doi.org/10.1093/bioadv/vbac016) | [PMC9710656](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710656/)
- Müller-Dott et al. (2023). CollecTRI: expanding TF regulon coverage for activity estimation.
  *Nucleic Acids Research* 51(20):10934. [DOI](https://doi.org/10.1093/nar/gkad841) | [PMC10639077](https://pmc.ncbi.nlm.nih.gov/articles/PMC10639077/)
- Dimitrov et al. (2022). LIANA: comparison of cell-cell communication methods.
  *Nat. Comms.* 13:3224. [DOI](https://doi.org/10.1038/s41467-022-30755-0) | [PMC9184522](https://pmc.ncbi.nlm.nih.gov/articles/PMC9184522/)
- Dimitrov et al. (2024). LIANA+: all-in-one framework for cell-cell communication.
  *Nature Cell Biology* 26(9):1613. [DOI](https://doi.org/10.1038/s41556-024-01469-w) | [PMC11392821](https://pmc.ncbi.nlm.nih.gov/articles/PMC11392821/)
- Tanevski et al. (2022). MISTY: multiview framework for spatial relationships.
  *Genome Biology* 23:97. [DOI](https://doi.org/10.1186/s13059-022-02663-5) | [PMC9011939](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011939/)
- Türei et al. (2021). OmniPath: integrated intra- and intercellular signaling knowledge.
  *Mol. Systems Biology* 17(3):e9923. [DOI](https://doi.org/10.15252/msb.20209923) | [PMC7983032](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983032/)
- Li et al. (2025). SPathDB: spatial pathway activity atlas.
  *Nucleic Acids Research* 53(D1):D1205. [DOI](https://doi.org/10.1093/nar/gkae1041) | [PMC11701687](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701687/)
