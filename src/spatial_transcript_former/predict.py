"""
Inference API for SpatialTranscriptFormer.

Provides three user-facing components:

* :class:`FeatureExtractor` — wraps a backbone (ResNet, Phikon, …) to turn
  raw image patches into feature embeddings.
* :class:`Predictor` — wraps a trained :class:`SpatialTranscriptFormer` model
  to predict gene expression from features + spatial coordinates.
* :func:`inject_predictions` — injects predictions into an AnnData object
  for seamless Scanpy integration.

Additionally retains the :func:`plot_training_summary` helper used during
training validation.
"""

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


# ═══════════════════════════════════════════════════════════════════════
#  FeatureExtractor
# ═══════════════════════════════════════════════════════════════════════


class FeatureExtractor:
    """Extract feature embeddings from histology image patches.

    Wraps a backbone model (e.g. ResNet-50, Phikon, CTransPath) and its
    associated normalization transform so callers don't need to worry
    about model-specific preprocessing.

    Example::

        extractor = FeatureExtractor(backbone="phikon", device="cuda")
        # images: Tensor of shape (N, 3, 224, 224), uint8 or float [0, 1]
        features = extractor(images)          # → (N, D)
        features = extractor.extract_batch(images, batch_size=64)
    """

    # Standard ImageNet normalization — used by all current backbones
    _IMAGENET_NORM = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(
        self,
        backbone: str = "resnet50",
        device: str = "cpu",
        pretrained: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            backbone: Backbone identifier (see ``models.backbones``).
            device: Torch device string.
            pretrained: Whether to load pretrained backbone weights.
            transform: Optional custom normalization transform.  If
                ``None``, standard ImageNet normalization is applied.
        """
        from spatial_transcript_former.models.backbones import get_backbone

        self.backbone_name = backbone
        self.device = torch.device(device)
        self.model, self.feature_dim = get_backbone(backbone, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transform or self._IMAGENET_NORM

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of images.

        Args:
            images: ``(N, 3, H, W)`` float tensor in ``[0, 1]`` range.

        Returns:
            ``(N, D)`` feature tensor on the same device.
        """
        images = images.to(self.device)
        images = self.transform(images)
        return self.model(images)

    @torch.no_grad()
    def extract_batch(
        self,
        images: torch.Tensor,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Extract features in batches to manage GPU memory.

        Args:
            images: ``(N, 3, H, W)`` float tensor in ``[0, 1]``.
            batch_size: Number of images per forward pass.

        Returns:
            ``(N, D)`` concatenated feature tensor on CPU.
        """
        all_features = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            features = self(batch)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)


# ═══════════════════════════════════════════════════════════════════════
#  Predictor
# ═══════════════════════════════════════════════════════════════════════


class Predictor:
    """High-level inference wrapper for SpatialTranscriptFormer.

    Manages model state (eval mode, device, AMP), and provides
    convenience methods for single-patch and whole-slide inference.

    Example::

        model = SpatialTranscriptFormer.from_pretrained("./checkpoint/")
        predictor = Predictor(model, device="cuda")

        # Single patch
        genes = predictor.predict_patch(image_tensor)

        # Whole slide (pre-extracted features)
        genes = predictor.predict_wsi(features, coords)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        use_amp: bool = False,
    ):
        """
        Args:
            model: A trained :class:`SpatialTranscriptFormer` instance.
            device: Torch device string.
            use_amp: Enable automatic mixed precision for inference.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.use_amp = use_amp

        # Expose gene names if the model has them (set by from_pretrained)
        self.gene_names: Optional[List[str]] = getattr(model, "gene_names", None)

    @torch.no_grad()
    def predict_patch(
        self,
        image: torch.Tensor,
        return_pathways: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """Predict gene expression from a single image patch.

        Args:
            image: ``(1, 3, H, W)`` or ``(3, H, W)`` image tensor.
            return_pathways: Also return pathway activation scores.

        Returns:
            Gene expression tensor ``(1, G)`` or tuple with pathway scores.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # For single-patch mode, spatial PE needs a dummy coordinate
        rel_coords = None
        if getattr(self.model, "use_spatial_pe", False):
            rel_coords = torch.zeros(image.shape[0], 1, 2, device=self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            result = self.model(
                image,
                rel_coords=rel_coords,
                return_pathways=return_pathways,
            )
        return result

    @torch.no_grad()
    def predict_wsi(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        return_pathways: bool = False,
        return_dense: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """Predict gene expression from pre-extracted whole-slide features.

        Args:
            features: ``(N, D)`` or ``(1, N, D)`` feature embeddings.
            coords: ``(N, 2)`` or ``(1, N, 2)`` spatial coordinates.
            return_pathways: Also return pathway activation scores.
            return_dense: If True, return per-patch predictions ``(1, N, G)``.

        Returns:
            Gene expression tensor ``(1, G)`` or ``(1, N, G)`` if dense,
            optionally with pathway scores as a tuple.
        """
        # Ensure batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        # Validate feature dimension matches model expectation
        expected_dim = self.model.image_proj.in_features
        actual_dim = features.shape[-1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: model expects {expected_dim} "
                f"(backbone '{getattr(self.model, '_backbone_name', 'unknown')}'), "
                f"but got {actual_dim}. "
                f"Did you extract features with the correct backbone?"
            )

        features = features.to(self.device)
        coords = coords.to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            result = self.model(
                features,
                rel_coords=coords,
                return_pathways=return_pathways,
                return_dense=return_dense,
            )
        return result

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """Unified prediction entry point.

        Automatically dispatches to :meth:`predict_patch` (if input looks
        like a raw image) or :meth:`predict_wsi` (if input looks like
        pre-computed features).

        Args:
            features: Either ``(B, 3, H, W)`` image or ``(N, D)`` features.
            coords: Required for WSI mode, ignored for patch mode.
            **kwargs: Forwarded to the underlying predict method.
        """
        if features.dim() == 4 and features.shape[1] == 3:
            # Looks like an image tensor
            return self.predict_patch(features, **kwargs)
        else:
            if coords is None:
                raise ValueError(
                    "coords are required for feature-based prediction. "
                    "Pass spatial coordinates alongside pre-extracted features."
                )
            return self.predict_wsi(features, coords, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  Scanpy / AnnData Integration
# ═══════════════════════════════════════════════════════════════════════


def inject_predictions(
    adata,
    coords: np.ndarray,
    predictions: np.ndarray,
    gene_names: Optional[List[str]] = None,
    pathway_scores: Optional[np.ndarray] = None,
    pathway_names: Optional[List[str]] = None,
):
    """Inject SpatialTranscriptFormer predictions into an AnnData object.

    Registers spatial coordinates and gene/pathway predictions into the
    appropriate AnnData slots so that standard Scanpy spatial plotting
    and analysis tools work out of the box.

    Args:
        adata: A :class:`anndata.AnnData` instance.  Must have the same
            number of observations as ``coords`` rows.
        coords: ``(N, 2)`` spatial coordinates array.
        predictions: ``(N, G)`` predicted gene expression array.
        gene_names: Optional list of G gene symbols.  If provided, they
            are set as ``adata.var_names``.
        pathway_scores: Optional ``(N, P)`` pathway activation scores.
        pathway_names: Optional list of P pathway names.

    Returns:
        The modified ``adata`` (in-place).

    Example::

        import scanpy as sc
        from spatial_transcript_former.predict import inject_predictions

        adata = sc.AnnData(obs=pd.DataFrame(index=[f"spot_{i}" for i in range(N)]))
        inject_predictions(adata, coords, preds, gene_names=model.gene_names)

        sc.pl.spatial(adata, color="TP53")
    """
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise ImportError(
            "anndata is required for inject_predictions. "
            "Install it with: pip install anndata"
        )

    n_obs = adata.n_obs
    if coords.shape[0] != n_obs:
        raise ValueError(
            f"coords has {coords.shape[0]} rows but adata has {n_obs} observations"
        )
    if predictions.shape[0] != n_obs:
        raise ValueError(
            f"predictions has {predictions.shape[0]} rows but adata has {n_obs} observations"
        )

    # Convert torch tensors to numpy if needed
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    # 1. Spatial coordinates
    adata.obsm["spatial"] = coords

    # 2. Gene predictions → adata.X
    import pandas as pd

    if gene_names is not None:
        if len(gene_names) != predictions.shape[1]:
            raise ValueError(
                f"gene_names length ({len(gene_names)}) != prediction columns ({predictions.shape[1]})"
            )
        adata.var = pd.DataFrame(index=gene_names)

    adata.X = predictions

    # 3. Pathway scores (optional) → adata.obsm
    if pathway_scores is not None:
        if isinstance(pathway_scores, torch.Tensor):
            pathway_scores = pathway_scores.cpu().numpy()
        adata.obsm["spatial_pathways"] = pathway_scores

        if pathway_names is not None:
            adata.uns["pathway_names"] = pathway_names

    return adata


# ═══════════════════════════════════════════════════════════════════════
#  Training Visualization (existing utility)
# ═══════════════════════════════════════════════════════════════════════
# Kept here for backwards compatibility with training scripts.

import matplotlib.pyplot as plt


def plot_training_summary(
    coords: np.ndarray,
    pathway_pred: np.ndarray,
    pathway_truth: np.ndarray,
    pathway_names: List[str],
    sample_id: str = "Sample",
    histology_img: Optional[np.ndarray] = None,
    scalef: float = 1.0,
    save_path: str = "plot.png",
    plot_pathways_list: Optional[List[str]] = None,
):
    """
    Creates a detailed summary plot of pathway predictions vs ground truth.

    Args:
        coords: (N, 2) spatial coordinates.
        pathway_pred: (N, P) predicted pathway activations (spatial z-score).
        pathway_truth: (N, P) ground truth pathway activations (spatial z-score).
        pathway_names: List of P pathway names.
        sample_id: Identifier for the plot title.
        histology_img: Optional RGB image for background.
        scalef: Scale factor for histology image.
        save_path: Path to save the PNG.
        plot_pathways_list: Optional list of specific pathway names to plot. If None, plots the first 6 available.
    """
    # Format short names for easier matching
    short_names = [n.replace("HALLMARK_", "") for n in pathway_names]

    selected_indices = []
    plot_names = []

    # Determine which pathways to plot
    if plot_pathways_list is not None and len(plot_pathways_list) > 0:
        target_pathways = [p.replace("HALLMARK_", "") for p in plot_pathways_list]
    else:
        # Default to the first 6 pathways available if none specified
        target_pathways = short_names[:6]

    for target_pw in target_pathways:
        if target_pw in short_names:
            idx = short_names.index(target_pw)
            selected_indices.append(idx)
            plot_names.append(target_pw)
        elif f"HALLMARK_{target_pw}" in pathway_names:
            # Fallback exact match if user supplied the full name
            idx = pathway_names.index(f"HALLMARK_{target_pw}")
            selected_indices.append(idx)
            plot_names.append(target_pw)

    if not selected_indices:
        print(
            f"Warning: None of the requested pathways '{target_pathways}' were found in the model's output."
        )
        return

    num_plots = len(selected_indices)

    # 1. Apply style BEFORE creating subplots to ensure all text/axes inherit it properly
    plt.style.use("dark_background")

    # 2. Adjust width to be less extremely stretched (3 per plot instead of 5)
    width = max(10, 3 * num_plots)
    height = 8
    fig, axes = plt.subplots(
        2, num_plots, figsize=(width, height), squeeze=False, layout="constrained"
    )

    # Set the specific dark slate background color early
    fig.patch.set_facecolor("#0f172a")

    plt.suptitle(
        f"Pathway Validation (Spatial Z-Score): {sample_id}",
        fontsize=18,
        color="white",
        fontweight="bold",
    )

    for i, idx in enumerate(selected_indices):
        name = plot_names[i]

        # Pred and Truth for this pathway (z-score scale)
        p = pathway_pred[:, idx]
        t = pathway_truth[:, idx]

        # Vmin/Vmax for shared z-score scale
        vmin = min(p.min(), t.min())
        vmax = max(p.max(), t.max())

        for j, (vals, title) in enumerate([(t, "Truth"), (p, "Prediction")]):
            ax = axes[j, i]

            if histology_img is not None:
                ax.imshow(histology_img, alpha=0.4)
                # Apply scale factor to coordinates if histology is present
                c_plot = coords * scalef
            else:
                c_plot = coords

            sc = ax.scatter(
                c_plot[:, 0],
                c_plot[:, 1],
                c=vals,
                cmap="viridis",
                s=2,
                alpha=1.0,
                vmin=vmin,
                vmax=vmax,
                edgecolors="none",
            )

            # Titles on top row
            if j == 0:
                ax.set_title(name, fontsize=14, pad=10)
            # Row labels on first column
            if i == 0:
                ax.set_ylabel(title, fontsize=14, labelpad=10)

            ax.set_xticks([])
            ax.set_yticks([])

            # Remove bounding box for cleaner spatial visualization
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Make subplot background transparent or match figure
            ax.set_facecolor("#0f172a")

            if histology_img is not None:
                # Invert Y to match histology orientation
                ax.invert_yaxis()

    # Add a colorbar spanning the figure
    cbar = fig.colorbar(
        sc,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        shrink=0.5,
        aspect=40,
        pad=0.05,
    )
    cbar.set_label("Relative Expression (Spatial Z-Score)", fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved pathway summary to {save_path}")
