# Attribution, Provenance & Design Notes

> **Licensing note:** SpatialTranscriptFormer is released under the Apache
> License 2.0 (see [LICENSE](../LICENSE) and [NOTICE](../NOTICE)). This document
> is *descriptive* — it records design provenance and attributions for readers
> and reviewers. It does not itself grant or reserve any rights.

## Provenance

This implementation is original work. The multimodal interaction framing —
treating learnable biological-pathway tokens as a bottleneck that cross-attends
to histology patch tokens — was **conceptually inspired by** SURVPATH
([Jaume et al., 2024](https://arxiv.org/abs/2303.15545)). **No SURVPATH source
code is used, copied, or adapted here**; the model, losses, data pipeline, and
training code were written from scratch for the spatial-transcriptomics
pathway-activity task.

## Design contributions

The notable design choices of this project (described for readers/reviewers, not
as legal claims):

- **Pathway-exclusive prediction.** The model predicts pre-computed spatial
  pathway-activity targets directly, with no intermediate gene-reconstruction
  step. Targets are computed offline (QC → CP10k normalisation → mean pathway
  aggregation; see [PATHWAY_MAPPING.md](PATHWAY_MAPPING.md)), decoupling
  biological-knowledge integration from training and removing the circular
  auxiliary-loss dependency used in earlier prototypes.

- **Dot-product + Softplus scoring head.** Per-spot pathway scores are
  `softplus(scale · (patch · pathwayᵀ / √d) + bias)`, with learnable per-pathway
  scale and bias. Softplus yields non-negative scores matching the
  mean-log1p-CP10k targets; the √d scaling keeps affinities well-conditioned (as
  in scaled dot-product attention). This is *not* cosine similarity.

- **Configurable quad-flow interaction masking.** Attention between pathway (P)
  and histology (H) tokens is selectable per quadrant (`p2p`, `p2h`, `h2p`,
  `h2h`), enabling ablations and the "pathway bottleneck" variant (block `h2h`
  so all inter-patch information must flow through named pathways).

- **Slide-stationary spatial positional encoding.** Patch tokens receive a
  learned 2-D encoding of centred/standardised coordinates, so the model learns
  spatially-varying pathway maps rather than slide-level averages.

- **Moran's I as diagnostic + collapse detector.** Spatial autocorrelation is
  used to curate targets and to detect representation collapse during validation
  (correlation of predicted vs. ground-truth per-pathway Moran's I), rather than
  as a loss term. See [spatial_stats.py](../src/spatial_transcript_former/data/spatial_stats.py).

## Third-party attributions

See [NOTICE](../NOTICE) for the licenses and attribution requirements of the
datasets, gene sets, and foundation-model backbones this framework uses at
runtime (HEST-1k, MSigDB Hallmarks, CTransPath / Phikon / GigaPath / Hibou, …).

## Citing

If you use this work in academic research, please cite this repository and its
author, Benjamin Isaac Wilson, alongside the relevant third-party works listed
in [NOTICE](../NOTICE).
