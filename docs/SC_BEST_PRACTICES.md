# Single-cell Best Practices Analysis & Recommendations

This document provides a gap analysis comparing the **SpatialTranscriptFormer** project against the industry standard recommendations from [sc-best-practices.org](https://www.sc-best-practices.org). It identifies current strengths and prioritised recommendations for future development.

---

## ✅ Current Strengths

These are areas where the project already follows industry best practices:

* **Global Gene Vocabulary**: `build_vocab.py` ensures a consistent feature space across all samples, preventing feature mismatch during training and inference.
* **Spatial Context via Neighbourhoods**: The use of KD-trees in `HEST_FeatureDataset` to incorporate spatial neighbours aligns with best practices for spatially-aware deep learning.
* **Histology-Gene Integration**: The architecture (extracting features from histology to predict/interact with gene expression) follows the recommended multi-modal integration patterns.
* **Coordinate Standardisation**: `normalize_coordinates()` prevents spatial scale bias between slides from different technologies (e.g., standard Visium vs. Visium HD).
* **Pathway-Aware Feature Selection**: Prioritising MSigDB genes in the vocabulary builder ensures that biologically relevant signal is captured even when using limited gene sets.
* **Statistical Loss Modelling**: The implementation of `ZINBLoss` (Zero-Inflated Negative Binomial) accounts for the overdispersion and sparsity inherent in transcriptomic count data.

---

## 🚀 Recommended Improvements

The following items are recommended for future sprints to improve model robustness and biological accuracy.

### 1. SVG-aware Gene Selection (Moran's I)

**Priority: High**  
**Rationale**: Currently, genes are selected based on total expression or pathway membership. However, the model's primary task is to learn spatial patterns. Selecting genes based on **Spatially Variable Gene (SVG)** metrics like Moran's I (available in Squidpy) would prioritise genes that have learned spatial coherence over those that are just highly expressed (like housekeeping genes).

### 2. Standardised Preprocessing Pipeline

**Priority: Medium-High**  
**Rationale**: The current pipeline lacks a standardised library-size normalisation (e.g., CPM/CP10k) before the `log1p` transform. Consistent normalisation ensures that sequencing depth variation between spots does not bias the model's predictions.

### 3. Dispersion-based Gene Filtering

**Priority: Medium**  
**Rationale**: In addition to total counts, filtering for **Highly Variable Genes (HVG)** using dispersion metrics (as in `sc.pp.highly_variable_genes`) would ensure the model focuses on genes that carry biological variation between tissue states rather than static structural signal.

### 4. Per-spot Quality Control (QC)

**Priority: Medium**  
**Rationale**: Adding explicit QC thresholds (e.g., minimum UMI count, minimum detected genes, maximum mitochondrial fraction) to the dataset loading scripts would protect the model from training on low-quality "noise" spots.

### 5. Spatial Coherence Validation Metrics

**Priority: Medium**  
**Rationale**: Aggregate metrics like MSE or PCC don't capture whether the *spatial distribution* of predictions is realistic. Adding a validation step that compares the Moran's I of predicted vs. ground-truth expression would provide a much stronger biological validation signal.

### 6. Preprocessing Documentation

**Priority: Low/QoL**  
**Rationale**: Explicitly documenting the "data contract" (which normalisation is applied when, and how genes were selected) in a dedicated `PREPROCESSING.md` or as a standard header in output folders.

---

## References

* [Single-cell Best Practices (Theis Lab)](https://www.sc-best-practices.org)
* [Squidpy: Scalable framework for spatial omics](https://squidpy.readthedocs.io)
* [Scanpy: Single-cell analysis in Python](https://scanpy.readthedocs.io)
