import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


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
