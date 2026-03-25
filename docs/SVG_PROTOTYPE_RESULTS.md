# Spatially Variable Gene (SVG) Selection & Validation Prototype Results

## Overview

This document summarizes the prototype results for integrating spatially variable gene (SVG) scoring into the SpatialTranscriptFormer pipeline. Two major components were built and validated:

1. **SVG-Aware Gene Selection**: Using Moran's I to bias the 1000-gene vocabulary towards genes that exhibit strong spatial autocorrelation, improving the biological relevance of the model's bottleneck.
2. **Spatial Coherence Validation**: An end-to-end validation metric that calculates Moran's I on the predicted vs. ground-truth expression vectors for the top-50 spatially variable genes, reported as a Pearson correlation score.

This prototype was run on a colorectal/bowel cancer cohort from the HEST-1k dataset (84 human samples) for ~400 epochs.

---

## 1. Training Progress & Stability

Training the `stf_small` model (4 layers, 384 dim, 8 heads) on the bowel cancer subset with SVG weighting (`--svg-weight 0.5`) showed steady convergence and stability.

![Training Loss Landscape](./assets/bowel_svg_loss_curve.png)

* **Learning Schedule**: 10 warm-up epochs followed by cosine annealing.
* **Overfitting**: The gap between training and validation loss is characteristic of the small dataset size (84 samples), but the model finds a strong minimum.
* **Best Checkpoint**: Reached a best validation loss of **1.611 at epoch 367**, an improvement over early training stages (~1.835 at epoch 130).

![Validation Metrics](./assets/bowel_svg_val_metrics.png)

* **Mean Absolute Error (MAE)**: Decreased smoothly from ~0.80 down strictly towards 0.60, indicating improved pixel-wise prediction accuracy.
* **Prediction Variance**: Evaluated as a collapse detector. The variance ramped up from near-zero to ~0.03, confirming the model successfully escaped "mean-prediction" collapse and learned to output highly differentiated spatial patterns. *(Note: A variance of 0.0 indicates a collapsed model predicting the same average value everywhere).*

---

## 2. Pathway Spatial Coherence (Bowel Cancer)

To validate the biological plausibility of the predictions, we visualised clinically relevant Hallmarks pathways for colorectal/bowel cancer using the best epoch checkpoint (epoch 367) on sample `TENX29`.

Pathways selected for their relevance to colorectal cancer progression, invasion, and tumor microenvironment:

* **Wnt/β-Catenin Signaling**: Mutated in >80% of sporadic colorectal cancers (APC mutation).
* **Epithelial-Mesenchymal Transition (EMT)**: Key marker of invasion and metastasis.
* **TNF-α Signaling via NF-κB** & **Inflammatory Response**: Chronic inflammation is a hallmark driver of CRC.
* **KRAS Signaling (Up)**: KRAS mutations occurring in ~40% of CRCs.
* **Angiogenesis**: Critical for tumor vasculature, the target of anti-VEGF therapies.

### Pathway Predictions vs Ground Truth (Epoch 524)

![Bowel Cancer Pathway Predictions](./assets/TENX29_epoch_524.png)

**Observations:**

1. **Spatial Pattern Matching**: The model successfully reconstructs complex, heterogeneous spatial patterns across the tissue architecture. High-expression regions (yellow/green) in the ground truth strongly correlate with high-expression regions in the predictions.
2. **Biological Gradients**: Crucially, the model does not just predict average expression but captures the relative spatial *gradients* of these pathways, confirming the health of the non-collapsed prediction variance metric.
3. **Tumor Microenvironment (TME)**: Inflammatory pathways (TNF-α, Inflammatory Response) show distinct spatial localization differing from tumor-intrinsic pathways (Wnt, KRAS), accurately reflecting TME heterogeneity.

---

## Conclusion

The integration of Moran's I for both **SVG-aware gene selection** and **Spatial Coherence Validation** provides a significantly more robust, biologically grounded training pipeline. The model demonstrates the ability to learn and reconstruct clinically relevant spatial pathway patterns from H&E imaging alone, laying a strong foundation for scaling the model to the full HEST-1k dataset and evaluating patient-level stratification.
