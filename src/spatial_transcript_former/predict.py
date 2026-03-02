import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

# Representative pathways to highlight in summary plots
CORE_PATHWAYS = [
    "APOPTOSIS",
    "DNA_REPAIR",
    "G2M_CHECKPOINT",
    "MTORC1_SIGNALING",
    "P53_PATHWAY",
    "MYC_TARGETS_V1",
]


def plot_training_summary(
    coords: np.ndarray,
    pathway_pred: np.ndarray,
    pathway_truth: np.ndarray,
    pathway_names: List[str],
    sample_id: str = "Sample",
    histology_img: Optional[np.ndarray] = None,
    scalef: float = 1.0,
    save_path: str = "plot.png",
):
    """
    Creates a detailed summary plot of pathway predictions vs ground truth.

    Args:
        coords: (N, 2) spatial coordinates.
        pathway_pred: (N, P) predicted pathway activations (absolute counts).
        pathway_truth: (N, P) ground truth pathway activations (absolute counts).
        pathway_names: List of P pathway names.
        sample_id: Identifier for the plot title.
        histology_img: Optional RGB image for background.
        scalef: Scale factor for histology image.
        save_path: Path to save the PNG.
    """
    # Find indices for core pathways
    short_names = [n.replace("HALLMARK_", "") for n in pathway_names]
    selected_indices = []
    plot_names = []

    for core_pw in CORE_PATHWAYS:
        if core_pw in short_names:
            idx = short_names.index(core_pw)
            selected_indices.append(idx)
            plot_names.append(core_pw)

    if not selected_indices:
        print(
            f"Warning: None of the core pathways {CORE_PATHWAYS} found in available names."
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
        f"Pathway Validation (Absolute Counts): {sample_id}",
        fontsize=18,
        color="white",
        fontweight="bold",
    )

    for i, idx in enumerate(selected_indices):
        name = plot_names[i]

        # Pred and Truth for this pathway (absolute scale)
        p = pathway_pred[:, idx]
        t = pathway_truth[:, idx]

        # Vmin/Vmax for shared absolute scale
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
    cbar.set_label("Absolute Expression Level (Counts)", fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved pathway summary to {save_path}")
