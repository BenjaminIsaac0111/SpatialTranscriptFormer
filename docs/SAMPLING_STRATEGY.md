# Sampling Strategy & Data Leakage Prevention

This document explains the patch sampling methodologies used in the Spatial TranscriptFormer and how data leakage is prevented.

## 1. Patch Sampling Strategies

The model uses distinct sampling strategies to capture both local tissue architecture and global context.

### A. Local Context (KDTree Neighbor Sampling)

**Purpose:** Captures the immediate spatial neighborhood of a patch, providing local tissue context (e.g., cell-cell interactions, microenvironment).
**Mechanism:**

1. **Selection:** A **KDTree** is built using the spatial coordinates of all patches in a slide.
2. **Query:** For each training patch (center), the KDTree queries for the `k` nearest neighbors (e.g., `k=6`).
3. **Result:** The model receives the center patch + `k` local neighbors as input.

### B. Global Context (Random Long-Range Sampling)

**Purpose:** Captures long-range dependencies across the entire tissue slide, enabling the model to learn global features (e.g., tumor heterogeneity, distant signaling) similar to whole-slide context (Jaume et al.).
**Mechanism:**

1. **Selection:** The `HEST_FeatureDataset` randomly samples `N` patches (e.g., `N=256`) from the **entire slide**.
2. **Replacement:** Random sampling with replacement is used for efficiency.
3. **Result:** These global patches are concatenated with the local neighbors, forming a mixed context window: `[Center, Neighbors..., Global...]`.

## 2. Preventing Data Leakage

Data leakage occurs when information from the validation set influences training. We prevent this through strict **Slide-Level Splitting**.

### Mechanism

- **Structure:** HEST data consists of Whole Slide Images (WSIs), each containing thousands of patches.
- **Split Unit:** We split data at the **Slide Level**, not the Patch Level.
  - **Train Set:** Contains all patches from Slide A, Slide B, ...
  - **Validation Set:** Contains all patches from Slide X, Slide Y, ...
- **No Overlap:** A slide (and thus its patient) serves exclusively either as training or validation data.
- **Consequence:** The model never sees patches from validation slides during training. This ensures that validation performance reflects generalization to **unseen slides/patients**, rather than memorization of patches from known slides.

### Why this is critical

If we split randomly at the *patch* level:

- Neighboring patches (which are highly correlated) could end up in different splits.
- The model would "leak" information from a training patch to its validation neighbor, artificially inflating performance.
- **Slide-Level Splitting eliminates this risk.**
