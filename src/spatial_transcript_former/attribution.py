"""
Spatial-attribution utilities for the weak-vs-dense supervision experiment
(docs/EXPERIMENT_SPATIAL_ATTRIBUTION.md).

Every function here works on a *single held-out slide* at a time and returns
plain ``numpy`` arrays shaped ``(N, P)`` (or ``(N,)`` for shared-attention/
saliency), where ``N`` is the number of valid (non-padded) spots/patches and
``P`` the number of pathways — the common currency the eval script uses to
compare dense predictions, attention, and gradient saliency against the same
ground-truth per-spot pathway maps.

* ``pathway_attention_map``   — STF's per-pathway attention (pathway->patch).
* ``shared_attention_map``    — MIL's single shared attention map.
* ``gradient_saliency``       — pathway-specific saliency for *any* model.
* ``remove_shared_component`` — PC1 (shared tissue-density factor) removal.
* ``spatial_pattern_fidelity``— grade a signal map against ground truth.
"""

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from spatial_transcript_former.data.spatial_stats import spatial_coherence_score
from spatial_transcript_former.training.engine import SPATIAL_MODELS


def _prep_single_slide(feats, coords=None, mask=None):
    """Add a batch dim to unbatched ``(S, D)`` / ``(S, 2)`` / ``(S,)`` tensors."""
    if feats.dim() == 2:
        feats = feats.unsqueeze(0)
    if coords is not None and coords.dim() == 2:
        coords = coords.unsqueeze(0)
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return feats, coords, mask


def _valid_rows(arr, mask):
    """Filter a ``(S, ...)`` array down to non-padded rows using batch-0 of ``mask``."""
    if mask is None:
        return arr
    valid = ~mask[0].detach().cpu().numpy()
    return arr[valid]


def pathway_attention_map(model, feats, coords, mask=None, layer="last", reduce="mean"):
    """Extract the pathway->patch attention block from an STF forward pass.

    Args:
        model: A ``SpatialTranscriptFormer`` instance.
        feats: ``(1, S, D)`` or ``(S, D)`` patch features for one slide.
        coords: ``(1, S, 2)`` or ``(S, 2)`` slide-stationary coordinates.
        mask: ``(1, S)`` or ``(S,)`` bool padding mask (``True`` = padding).
        layer: ``"last"`` (default) uses the final fusion-engine layer for
            ``reduce="mean"``, or caps the rollout depth for
            ``reduce="rollout"``. An int selects a specific layer index /
            rollout depth instead.
        reduce: ``"mean"`` (default) — head-averaged attention weights from
            a single layer. ``"rollout"`` — attention rollout (Abnar &
            Zuidema 2020) through every layer up to ``layer``, which accounts
            for how attention composes across residual connections; a
            robustness variant of the default.

    Returns:
        np.ndarray: ``(N, P)`` pathway attention weights over the ``N`` valid
        patches, where ``P = model.num_pathways``.
    """
    feats_t, coords_t, mask_t = _prep_single_slide(feats, coords, mask)
    p = model.num_pathways

    with torch.no_grad():
        _, attentions = model(
            feats_t, rel_coords=coords_t, mask=mask_t, return_attention=True
        )
    # attentions: list[(B, H, T, T)], one per fusion-engine layer, T = P + S.

    if reduce == "mean":
        layer_idx = len(attentions) - 1 if layer == "last" else layer
        attn = attentions[layer_idx][0].mean(dim=0)  # (T, T), head-mean
        block = attn[:p, p:]  # (P, S) pathway -> patch
    elif reduce == "rollout":
        n_layers = len(attentions) if layer == "last" else layer + 1
        t = attentions[0].shape[-1]
        eye = torch.eye(t, device=attentions[0].device)
        rollout = eye
        for layer_attn in attentions[:n_layers]:
            head_mean = layer_attn[0].mean(dim=0)  # (T, T), still row-stochastic
            # Account for the residual connection around attention (Abnar &
            # Zuidema 2020) before composing across layers.
            head_mean = 0.5 * head_mean + 0.5 * eye
            head_mean = head_mean / head_mean.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            rollout = head_mean @ rollout
        block = rollout[:p, p:]  # (P, S)
    else:
        raise ValueError(f"Unknown reduce mode: {reduce!r}")

    block = block.detach().cpu().numpy().T  # (S, P)
    return _valid_rows(block, mask_t)


def shared_attention_map(model, feats, mask=None):
    """Extract the single shared attention map from a MIL baseline.

    Structurally, ``AttentionMIL``/``TransMIL`` have one attention weight per
    patch shared across every pathway output — this is a *single* map,
    unlike STF's per-pathway attention. Compare it against every pathway's
    ground truth in turn; the expectation (doc §7, §10) is that it tracks
    total pathway activity rather than pathway-specific structure.

    Args:
        model: ``AttentionMIL`` or ``TransMIL`` instance.
        feats: ``(1, S, D)`` or ``(S, D)`` patch features for one slide.
        mask: ``(1, S)`` or ``(S,)`` bool padding mask (``True`` = padding).

    Returns:
        np.ndarray: ``(N,)`` attention weight per valid patch.
    """
    feats_t, _, mask_t = _prep_single_slide(feats, None, mask)
    with torch.no_grad():
        _, attn = model(feats_t, return_attention=True)
    if attn.dim() == 3:  # AttentionMIL: (B, N, 1) -> (B, N)
        attn = attn.squeeze(-1)
    attn = attn[0].detach().cpu().numpy()
    return _valid_rows(attn, mask_t)


def gradient_saliency(model, feats, coords=None, mask=None, pathway_idx=0, reduce="l2"):
    """Gradient-based saliency for a single pathway's bag-level score.

    Computes ``d(bag score for pathway pathway_idx) / d(patch features)``,
    reduced to one scalar per patch. Unlike raw attention, this is
    inherently pathway-specific *even for MIL* (the backward pass is seeded
    from one pathway's output score), so it gives ``attn_mil``/``transmil`` a
    fair per-pathway map despite their attention being a single shared map
    (doc §7).

    Args:
        model: Any model whose forward pass accepts ``feats`` (optionally
            ``rel_coords``/``mask``) and returns a ``(B, P)`` bag-level score
            — STF in bag mode (``return_dense=False``), ``AttentionMIL``, or
            ``TransMIL``.
        feats: ``(1, S, D)`` or ``(S, D)`` patch features for one slide.
        coords: ``(1, S, 2)`` or ``(S, 2)`` coordinates; only used for
            spatial models (ignored otherwise).
        mask: ``(1, S)`` or ``(S,)`` bool padding mask (``True`` = padding).
        pathway_idx: Index of the pathway whose bag score to backprop from.
        reduce: ``"l2"`` (default) — ``||grad||_2`` per patch. ``"grad_input"``
            — signed ``grad . input`` per patch.

    Returns:
        np.ndarray: ``(N,)`` saliency score per valid patch.
    """
    feats_t, coords_t, mask_t = _prep_single_slide(feats, coords, mask)
    feats_t = feats_t.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    if isinstance(model, SPATIAL_MODELS):
        scores = model(feats_t, rel_coords=coords_t, mask=mask_t)  # (1, P), bag mode
    else:
        scores = model(feats_t)  # (1, P)

    target_score = scores[0, pathway_idx]
    target_score.backward()

    grad = feats_t.grad[0]  # (S, D)
    if reduce == "l2":
        saliency = grad.norm(dim=-1)
    elif reduce == "grad_input":
        saliency = (grad * feats_t.detach()[0]).sum(dim=-1)
    else:
        raise ValueError(f"Unknown reduce mode: {reduce!r}")

    saliency = saliency.detach().cpu().numpy()
    return _valid_rows(saliency, mask_t)


def remove_shared_component(maps, pc1_loadings=None):
    """Remove the rank-1 (PC1) shared spatial component from a signal matrix.

    Each pathway column is standardized (zero mean, unit variance) before
    PCA so pathways on different scales contribute comparably to PC1 —
    matching the diagnostic behind docs/EXPERIMENT_SPATIAL_ATTRIBUTION.md §6.

    Args:
        maps: ``(N, P)`` per-spot signal for ``P`` pathways. Ground-truth
            activity, model dense predictions, attention weights, and
            gradient saliency all use this same shape/convention.
        pc1_loadings: Optional ``(P,)`` array. If given, this fixed direction
            is projected out instead of being fit fresh from ``maps`` — used
            to remove *the ground truth's* shared factor from a model's
            signal map, so both sides of a fidelity comparison are
            residualised against the same axis (doc §8: "fit PC1 on ground
            truth; project the signal onto the same removed direction").

    Returns:
        tuple:
            residual (np.ndarray): ``(N, P)``, ``maps`` standardized and then
                with the (fit-or-given) PC1 direction projected out.
            pc1_scores (np.ndarray): ``(N,)``, each spot's projection onto
                the PC1 direction.
            pc1_loadings (np.ndarray): ``(P,)``, the (possibly newly-fit)
                unit-norm loading vector defining the removed direction.
    """
    maps = np.asarray(maps, dtype=np.float64)
    mean = maps.mean(axis=0, keepdims=True)
    std = maps.std(axis=0, keepdims=True)
    std_safe = np.where(std < 1e-12, 1.0, std)
    standardized = (maps - mean) / std_safe

    if pc1_loadings is None:
        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        pc1_loadings = vt[0]  # (P,), already unit-norm
        # Stable sign convention: dominant-loading pathway is positive.
        if pc1_loadings[np.argmax(np.abs(pc1_loadings))] < 0:
            pc1_loadings = -pc1_loadings
    else:
        pc1_loadings = np.asarray(pc1_loadings, dtype=np.float64)
        norm = np.linalg.norm(pc1_loadings)
        if norm > 1e-12:
            pc1_loadings = pc1_loadings / norm

    pc1_scores = standardized @ pc1_loadings  # (N,)
    residual = standardized - np.outer(pc1_scores, pc1_loadings)
    return residual, pc1_scores, pc1_loadings


def regress_out_covariate(maps, covariate):
    """Remove a measured confound from every pathway column by least squares.

    Preferred over :func:`remove_shared_component` whenever the confound is
    actually measured. Benchmarked on the HEST pathway targets against
    sequencing depth:

    ==========================  ==================  =========================
    method                      variance destroyed  residual depth |r|
    ==========================  ==================  =========================
    PC1 removal                 63.0%               0.077
    regressing out depth        51.3%               0.000
    ==========================  ==================  =========================

    PC1 is an unsupervised direction that merely *correlates* with the
    confound, so it also destroys biology aligned with it. Projecting out the
    measured covariate removes the confound exactly while keeping ~12
    percentage points more variance.

    Args:
        maps: ``(N, P)`` per-spot values for ``P`` pathways.
        covariate: ``(N,)`` measured confound (for sequencing depth, pass
            ``log1p(total_counts)`` — the relationship is log-linear).

    Returns:
        tuple:
            residual (np.ndarray): ``(N, P)``, column-standardised with the
                covariate direction projected out.
            beta (np.ndarray): ``(P,)`` fitted slope per pathway, on the
                standardised scale.
    """
    maps = np.asarray(maps, dtype=np.float64)
    cov = np.asarray(covariate, dtype=np.float64).ravel()

    mean = maps.mean(axis=0, keepdims=True)
    std = maps.std(axis=0, keepdims=True)
    standardized = (maps - mean) / np.where(std < 1e-12, 1.0, std)

    c = cov - cov.mean()
    denom = float(c @ c)
    if denom < 1e-12:  # constant covariate — nothing to remove
        return standardized, np.zeros(maps.shape[1])
    c = c / np.sqrt(denom)
    beta = standardized.T @ c
    return standardized - np.outer(c, beta), beta


def _zscore_columns(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std_safe


def spatial_pattern_fidelity(
    signal, target, coords, residual=True, pc1_loadings=None, k=6
):
    """Grade a model's per-pathway spatial signal against ground truth (doc §8).

    Args:
        signal: ``(N, P)`` model signal (dense prediction, attention, or
            gradient saliency) for one held-out slide.
        target: ``(N, P)`` ground-truth per-spot pathway activity for the
            same slide, same pathway ordering.
        coords: ``(N, 2)`` spatial coordinates for the same slide (used for
            the tertiary Moran's-I agreement metric).
        residual: If True (default), PC1 is fit on ``target`` and the same
            direction is projected out of both ``signal`` and ``target``
            before correlating — the honest headline metric (doc §6/§8). If
            False, correlates the raw (un-residualised, per-slide z-scored)
            maps instead.
        pc1_loadings: Optional pre-fit ``(P,)`` PC1 direction. If None and
            ``residual=True``, PC1 is fit fresh on ``target`` for this slide.
        k: KNN neighbours for the Moran's-I agreement metric.

    Returns:
        dict:
            pearson (float): Mean per-pathway Pearson r, averaged over
                pathways with non-zero variance in both signal and target.
            spearman (float): Mean per-pathway Spearman rho, same averaging.
            per_pathway (dict): ``{"pearson": (P,), "spearman": (P,)}``
                arrays (NaN for pathways skipped due to zero variance).
            morans_i_agreement (float): Pearson correlation between signal's
                and target's per-pathway Moran's I (tertiary metric, §8).
    """
    signal = np.asarray(signal, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    p = target.shape[1]

    if residual:
        target_use, _, loadings = remove_shared_component(
            target, pc1_loadings=pc1_loadings
        )
        signal_use, _, _ = remove_shared_component(signal, pc1_loadings=loadings)
    else:
        target_use = _zscore_columns(target)
        signal_use = _zscore_columns(signal)

    pearson_per = np.full(p, np.nan)
    spearman_per = np.full(p, np.nan)
    for j in range(p):
        t = target_use[:, j]
        s = signal_use[:, j]
        if np.std(t) < 1e-12 or np.std(s) < 1e-12:
            continue
        r, _ = pearsonr(s, t)
        rho, _ = spearmanr(s, t)
        pearson_per[j] = r if np.isfinite(r) else np.nan
        spearman_per[j] = rho if np.isfinite(rho) else np.nan

    mean_pearson = (
        float(np.nanmean(pearson_per))
        if np.isfinite(pearson_per).any()
        else float("nan")
    )
    mean_spearman = (
        float(np.nanmean(spearman_per))
        if np.isfinite(spearman_per).any()
        else float("nan")
    )

    try:
        # Use the same (possibly residualised) maps as the correlation above,
        # so this tertiary metric genuinely differs between raw and residual
        # modes rather than silently reusing the raw maps in both.
        morans_agreement = spatial_coherence_score(
            signal_use, target_use, coords, k=k, top_k_genes=p
        )
    except Exception:
        morans_agreement = 0.0

    return {
        "pearson": mean_pearson,
        "spearman": mean_spearman,
        "per_pathway": {"pearson": pearson_per, "spearman": spearman_per},
        "morans_i_agreement": morans_agreement,
    }
