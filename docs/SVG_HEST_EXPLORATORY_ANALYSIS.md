# SVG Exploratory Analysis & Pathway Curation Report

This report summarizes the data-driven analysis of Spatially Variable Pathways across the HEST dataset (95 samples, 85 valid human samples after QC) conducted to refine the targets for model training.

## 1. Methodology

The analysis was performed using a standalone utility (`scripts/analyze_svg.py`) that:
1.  **Strips common gene prefixes** (e.g., `GRCh38_`, `GRCm38_`) to ensure compatibility with MSigDB Hallmark gene sets.
2.  **Computes pathway activities** using a sum-aggregation method (normalized to 10k target sum).
3.  **Calculates Moran's I** for each of the 50 Hallmark pathways per sample.
4.  **Aggregates statistics** (mean, median, std, etc.) across all valid human samples.
5.  **Analyzes correlations** between spot-level pathway activities to understand redundancy.

## 2. Global Spatial Autocorrelation Results

The following plot shows the ranked mean Moran's I across 85 human samples for all 50 Hallmark pathways.

![Global SVG Analysis](./assets/reports/svg_analysis_full.png)

### Key Observations:
*   **Widespread Spatial Structure**: All 50 pathways exhibit positive spatial autocorrelation (Mean Moran's I > 0.15).
*   **High-Signal Pathways**: Top-ranked pathways include **MYC Targets V1** (0.665), **E2F Targets** (0.639), **G2M Checkpoint** (0.633), and **Oxidative Phosphorylation** (0.631).
*   **Variance vs. Spatiality**: High expression variance does not always equate to high spatial coherence. Some pathways vary significantly between spots but lack a spatially organized pattern.

---

## 3. CRC Pathway Curation

Based on these results, the curated list of pathways for Colorectal Cancer (CRC) was validated. While some pathways exhibit lower spatial autocorrelation than others, all 14 selected hallmarks exceed a significance baseline of **Mean Moran's I > 0.20** and are therefore retained for training.

| Status | Pathway | Mean Moran's I | % Samples > 0.05 |
| :--- | :--- | :--- | :--- |
| ✅ **Retained** | EPITHELIAL_MESENCHYMAL_TRANSITION | 0.602 | 98.8% |
| ✅ **Retained** | DNA_REPAIR | 0.554 | 91.8% |
| ✅ **Retained** | APOPTOSIS | 0.547 | 100.0% |
| ✅ **Retained** | P53_PATHWAY | 0.546 | 92.9% |
| ✅ **Retained** | HYPOXIA | 0.539 | 100.0% |
| ✅ **Retained** | APICAL_JUNCTION | 0.498 | 100.0% |
| ✅ **Retained** | INFLAMMATORY_RESPONSE | 0.487 | 100.0% |
| ✅ **Retained** | PI3K_AKT_MTOR_SIGNALING | 0.483 | 91.8% |
| ✅ **Retained** | KRAS_SIGNALING_UP | 0.469 | 98.8% |
| ✅ **Retained** | IL6_JAK_STAT3_SIGNALING | 0.408 | 98.8% |
| ✅ **Retained** | TGF_BETA_SIGNALING | 0.397 | 94.1% |
| ✅ **Retained** | ANGIOGENESIS | 0.339 | 94.1% |
| ✅ **Retained** | WNT_BETA_CATENIN_SIGNALING | 0.302 | 90.6% |
| ✅ **Retained** | KRAS_SIGNALING_DN | 0.250 | 95.3% |

### Rationalization:
Although pathways like **WNT/β-catenin** and **KRAS_DN** have lower Moran's I scores (0.30 and 0.25 respectively) compared to **EMT** (0.60), they remain significantly above the background noise floor (~0.15). Their relative spatial uniformity likely reflects constitutive activation by driver mutations (e.g., APC mutations making WNT "on" globally), but the remaining spatial gradients are biologically critical for capturing tumor margins and stroma-epithelial interactions.

---

## 4. Pathway Correlation & Redundancy

To ensure the model is learning distinct biological signals, we analyzed the correlation between spot-level activities of the 14 CRC pathways.

![Pathway Correlations](./assets/reports/pathway_correlations_full.png)

### Correlation Insights:
*   **Biological Axes**: Strong correlations exist between **Angiogenesis** and **EMT** (r=0.749), and between **TGF-β** and **Apoptosis** (r=0.668). These axes represent co-regulated spatial processes.
*   **Distinct Signals**: Despite these correlations, each pathway provides a unique biological "view" of the tissue. Retaining the full set allows the model to learn complex regulatory relationships rather than just isolated spatial patterns.

**Conclusion**: All 14 CRC pathways exhibit sufficient spatial structure and biological relevance to be included as training targets. This ensures the model learns a comprehensive representation of the CRC tissue microenvironment.

---

## 5. Technical Improvements

During this analysis, two critical fixes were implemented:
1.  **Gene Prefix Stripping**: Fixed an issue where samples like `TENX175` had all-zero pathway scores because gene names were prefixed with `GRCh38_`.
2.  **Sample Compatibility Check**: Added a check for Hallmark gene overlap to automatically skip mouse samples or low-density panels that cannot be accurately scored using human Hallmark sets.
