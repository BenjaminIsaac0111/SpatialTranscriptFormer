import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from spatial_transcript_former.models.regression import HE2RNA, ViT_ST
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.data.dataset import (
    get_hest_dataloader,
    HEST_Dataset,
    load_gene_expression_matrix,
    load_global_genes,
)
import h5py


def plot_spatial_genes(
    coords, truth, pred, gene_names, sample_id, save_path=None, cmap="viridis"
):
    """
    Plots spatial maps of gene expression (dot plot format).
    """
    num_plots = min(len(gene_names), 5)
    fig, axes = plt.subplots(num_plots, 2, figsize=(12, 4 * num_plots))
    if num_plots == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_plots):
        gene_name = gene_names[i]

        # Truth
        # HEST coords are (x, y). Scatter takes (x, y).
        # imshow is y-down.
        sc = axes[i, 0].scatter(
            coords[:, 0],
            coords[:, 1],
            c=truth[:, i],
            cmap=cmap,
            s=10,
            edgecolors="none",
        )
        axes[i, 0].set_title(f"{sample_id} - {gene_name} (TRUTH)")
        axes[i, 0].invert_yaxis()  # Match image space
        axes[i, 0].set_aspect("equal")
        plt.colorbar(sc, ax=axes[i, 0], shrink=0.6)

        # Prediction
        sc = axes[i, 1].scatter(
            coords[:, 0], coords[:, 1], c=pred[:, i], cmap=cmap, s=10, edgecolors="none"
        )
        axes[i, 1].set_title(f"{sample_id} - {gene_name} (PRED)")
        axes[i, 1].invert_yaxis()  # Match image space
        axes[i, 1].set_aspect("equal")
        plt.colorbar(sc, ax=axes[i, 1], shrink=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    plt.close("all")


def plot_histology_overlay(
    image,
    coords,
    values,
    gene_names,
    sample_id,
    scalef=1.0,
    save_path=None,
    cmap="viridis",
):
    """
    Plots predictions as an overlay on the histology image.
    """
    num_plots = min(len(gene_names), 3)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    # Project coords to image pixels
    pixel_coords = coords * scalef

    for i in range(num_plots):
        gene_name = gene_names[i]
        ax = axes[i]

        # Display Histology
        ax.imshow(image)

        # Overlay Predictions
        # pixel_coords[:, 0] = X, pixel_coords[:, 1] = Y
        sc = ax.scatter(
            pixel_coords[:, 0],
            pixel_coords[:, 1],
            c=values[:, i],
            cmap=plt.get_cmap(cmap),
            alpha=0.4,
            s=15,
            edgecolors="none",
        )

        ax.set_title(f"{sample_id} - {gene_name} Overlay")
        ax.axis("off")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Overlay saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    plt.close("all")


# Fixed representative pathways for visualization (MSigDB Hallmark names)
CORE_PATHWAYS = [
    "EPITHELIAL_MESENCHYMAL_TRANSITION",
    "WNT_BETA_CATENIN_SIGNALING",
    "INFLAMMATORY_RESPONSE",
    "ANGIOGENESIS",
    "APOPTOSIS",
    "TNFA_SIGNALING_VIA_NFKB",
]


def plot_training_summary(
    coords,
    pathway_pred,
    pathway_truth,
    pathway_names,
    sample_id,
    histology_img=None,
    scalef=1.0,
    save_path=None,
    cmap="jet",
):
    """
    Compact landscape training visualization dashboard without z-scoring.
    Colorbars are explicitly externalized using make_axes_locatable so plots
    stay uniform in size.

    Args:
        coords: (N, 2) spot coordinates.
        pathway_pred: (N, P) predicted pathway activations.
        pathway_truth: (N, P) ground-truth pathway activations.
        pathway_names: List of pathway names (length P).
        sample_id: Sample identifier for titles.
        histology_img: Optional H&E image array.
        scalef: Scale factor for projecting coords onto histology.
        save_path: Where to save the figure.
        cmap: Colormap for scatter plots.
    """
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Find indices for our fixed pathways
    name_to_idx = {}
    if pathway_names is not None:
        for i, name in enumerate(pathway_names):
            short = name.replace("HALLMARK_", "")
            name_to_idx[short] = i

    display_pathways = []
    if pathway_names is not None:
        if len(pathway_names) <= 6:
            for pw in pathway_names:
                short = pw.replace("HALLMARK_", "")
                if short in name_to_idx:
                    display_pathways.append((short, name_to_idx[short]))
        else:
            for pw in CORE_PATHWAYS:
                if pw in name_to_idx:
                    display_pathways.append((pw, name_to_idx[pw]))

        if not display_pathways:
            for pw in pathway_names[:6]:
                short = pw.replace("HALLMARK_", "")
                if short in name_to_idx:
                    display_pathways.append((short, name_to_idx[short]))

    if not display_pathways:
        print("Warning: No viable pathways found in model to display. Skipping plot.")
        return

    n_per_row = 2
    n_pw = len(display_pathways)
    n_rows = int(np.ceil(n_pw / n_per_row))
    has_histology = histology_img is not None

    vis_coords = coords * scalef if has_histology else coords

    # Create figure: Width accommodates 1 Histology + (2 pathways * 3 cols each (Truth, Pred, Cbar))
    # Total ~7 logical columns
    fig = plt.figure(figsize=(24, 6 * n_rows), constrained_layout=False)
    fig.patch.set_facecolor("#1a1a2e")

    # Outer Grid: 1 col for Histology, 1 for all pathways
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1, 3.5],
        left=0.02,
        right=0.98,
        top=0.92,
        bottom=0.05,
        wspace=0.1,
    )

    # --- Left: Histology panel ---
    ax_hist = fig.add_subplot(outer[0, 0])
    if has_histology:
        ax_hist.imshow(histology_img)
    ax_hist.set_title("Histology", fontsize=16, color="white", pad=12)
    ax_hist.axis("off")
    ax_hist.set_facecolor("#0d0d1a")
    # Anchor to top so it doesn't float randomly if pathways are tall
    ax_hist.set_anchor("N")

    # --- Right: Pathway Grids ---
    # For each pathway, we want [GT | Pred | Colorbar] = 3 sub-columns.
    # Total columns = n_per_row * 3
    n_cols = n_per_row * 3
    # Configure width ratios: Maps get width 1, Colorbars get width 0.1
    col_widths = [1, 1, 0.1] * n_per_row

    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows,
        n_cols,
        subplot_spec=outer[0, 1],
        width_ratios=col_widths,
        hspace=0.35,
        wspace=0.15,
    )

    for idx, (pw_name, pw_idx) in enumerate(display_pathways):
        row = idx // n_per_row
        pw_col_base = (idx % n_per_row) * 3

        col_gt = pw_col_base
        col_pred = pw_col_base + 1
        col_cbar = pw_col_base + 2

        label = pw_name.replace("_", " ").title()
        if len(label) > 30:
            label = label[:27] + "..."

        truth_vals = pathway_truth[:, pw_idx]
        pred_vals = pathway_pred[:, pw_idx]

        # Both truth and pred are now in the same units (mean log1p expression
        # of pathway member genes), so shared bounds give a fair comparison
        vmin = min(truth_vals.min(), pred_vals.min())
        vmax = max(truth_vals.max(), pred_vals.max())

        sc = None
        for col, vals, suffix in [
            (col_gt, truth_vals, "Truth"),
            (col_pred, pred_vals, "Pred"),
        ]:
            ax = fig.add_subplot(inner[row, col])
            ax.set_facecolor("#0d0d1a")

            if has_histology:
                ax.imshow(histology_img, alpha=0.25)

            sc = ax.scatter(
                vis_coords[:, 0],
                vis_coords[:, 1],
                c=vals,
                cmap=cmap,
                s=6,
                edgecolors="none",
                vmin=vmin,
                vmax=vmax,
            )

            if not has_histology:
                ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{label}\n{suffix}", fontsize=12, color="white", pad=6)

        # Plot the colorbar in its dedicated axis so it NEVER shrinks the prediction map
        cax = fig.add_subplot(inner[row, col_cbar])
        cb = plt.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=9, colors="white")
        cb.outline.set_edgecolor("white")

    fig.suptitle(
        f"{sample_id}  —  Pathway Activation Summary",
        fontsize=20,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    if save_path:
        plt.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"Training summary saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Predict and Visualize Spatial Transcriptomics"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        default="he2rna",
        choices=["he2rna", "vit_st", "interaction"],
    )
    parser.add_argument("--num-genes", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument(
        "--n-neighbors", type=int, default=0, help="Number of spatial neighbors to use"
    )
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument(
        "--num-pathways",
        type=int,
        default=50,
        help="Number of pathways in the bottleneck",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone for feature extraction",
    )
    parser.add_argument(
        "--plot-pathways", action="store_true", help="Visualize pathway activations"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "pcc", "mse_pcc", "zinb", "poisson", "logcosh"],
        help="Loss function used for training (needed for model reconstruction)",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    patches_dir = os.path.join(args.data_dir, "patches")
    if not os.path.exists(patches_dir):
        patches_dir = args.data_dir
    st_dir = os.path.join(args.data_dir, "st")

    h5_path = os.path.join(patches_dir, f"{args.sample_id}.h5")
    h5ad_path = os.path.join(st_dir, f"{args.sample_id}.h5ad")

    # Load Model
    if args.model_type == "he2rna":
        model = HE2RNA(num_genes=args.num_genes, backbone=args.backbone)
    elif args.model_type == "vit_st":
        model = ViT_ST(num_genes=args.num_genes, model_name=args.backbone)
    elif args.model_type == "interaction":
        model = SpatialTranscriptFormer(
            num_genes=args.num_genes,
            token_dim=args.token_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            num_pathways=args.num_pathways,
            backbone_name=args.backbone,
            output_mode="zinb" if args.loss == "zinb" else "counts",
        )

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    # Handle possible torch.compile prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Load Sample
    with h5py.File(h5_path, "r") as f:
        patch_barcodes = f["barcode"][:].flatten()
        coords = f["coords"][:]

    # Load common genes for consistency
    try:
        common_gene_names = load_global_genes(args.data_dir, args.num_genes)
    except Exception as e:
        print(
            f"Warning: Could not load global genes, falling back to sample top genes: {e}"
        )
        common_gene_names = None

    gene_matrix, mask, gene_names = load_gene_expression_matrix(
        h5ad_path,
        patch_barcodes,
        selected_gene_names=common_gene_names,
        num_genes=args.num_genes,
    )

    coord_subset = coords[mask]
    indices = np.where(mask)[0]

    # Neighborhood computation if requested
    neighborhood_indices = None
    if args.n_neighbors > 0:
        from scipy.spatial import KDTree

        with h5py.File(h5_path, "r") as f:
            coords_all = f["coords"][:]
        tree = KDTree(coords_all)
        dists, idxs = tree.query(coord_subset, k=args.n_neighbors + 1)
        final_neighbors = []
        for i, center_idx in enumerate(indices):
            n_idxs = idxs[i]
            n_idxs = n_idxs[n_idxs != center_idx]
            final_neighbors.append(n_idxs[: args.n_neighbors])
        neighborhood_indices = np.array(final_neighbors)
    else:
        with h5py.File(h5_path, "r") as f:
            coords_all = f["coords"][:]

    dataset = HEST_Dataset(
        h5_path,
        coord_subset,
        gene_matrix,
        indices=indices,
        neighborhood_indices=neighborhood_indices,
        coords_all=coords_all,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_truth = []
    all_pathways = []

    print(f"Running inference on {len(dataset)} patches...")
    with torch.no_grad():
        for images, targets, rel_coords in loader:
            images = images.to(device)
            rel_coords = rel_coords.to(device)

            if isinstance(model, SpatialTranscriptFormer):
                # Request pathways if argument set
                output = model(
                    images, rel_coords=rel_coords, return_pathways=args.plot_pathways
                )
                if isinstance(output, tuple) and args.plot_pathways:
                    preds = output[0]
                    pathways = output[1]
                    all_pathways.append(pathways.cpu().numpy())
                else:
                    preds = output
            else:
                preds = model(images)

            # Unpack ZINB tuple if generated
            if isinstance(preds, tuple):
                preds = preds[1]  # Use mean component

            all_preds.append(preds.cpu().numpy())
            all_truth.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_truth = np.concatenate(all_truth, axis=0)
    if all_pathways:
        all_pathways = np.concatenate(all_pathways, axis=0)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Standard Plot
    save_dot_path = os.path.join(
        args.output_dir, f"{args.sample_id}_spatial_inference.png"
    )
    plot_spatial_genes(
        coord_subset,
        all_truth,
        all_preds,
        gene_names[:5],
        args.sample_id,
        save_path=save_dot_path,
        cmap="jet",
    )

    if args.plot_pathways and len(all_pathways) > 0:
        save_pathway_path = os.path.join(
            args.output_dir, f"{args.sample_id}_pathway_activations.png"
        )
        plot_spatial_pathways(
            coord_subset,
            all_pathways,
            args.sample_id,
            save_path=save_pathway_path,
            cmap="jet",
        )

    # Histology Overlay Plot
    try:
        # Construct h5ad path robustly
        h5ad_overlay_path = h5_path.replace(".h5", ".h5ad")
        if "patches" in h5ad_overlay_path:
            h5ad_overlay_path = h5ad_overlay_path.replace("patches", "st")

        if os.path.exists(h5ad_overlay_path):
            with h5py.File(h5ad_overlay_path, "r") as f:
                # Robust group access
                if "uns" in f and "spatial" in f["uns"]:
                    spatial = f["uns/spatial"]
                    sample_key = (
                        list(spatial.keys())[0] if len(spatial.keys()) > 0 else None
                    )
                    if sample_key:
                        img_group = spatial[sample_key]["images"]
                        img_key = (
                            "downscaled_fullres"
                            if "downscaled_fullres" in img_group
                            else list(img_group.keys())[0]
                        )
                        img = img_group[img_key][:]

                        scale_group = spatial[sample_key]["scalefactors"]
                        scale_key = (
                            "tissue_downscaled_fullres_scalef"
                            if "tissue_downscaled_fullres_scalef" in scale_group
                            else list(scale_group.keys())[0]
                        )
                        scalef = scale_group[scale_key][()]

                        save_overlay_path = os.path.join(
                            args.output_dir, f"{args.sample_id}_histology_overlay.png"
                        )
                        print(f"Generating histology overlay for {args.sample_id}...")
                        plot_histology_overlay(
                            img,
                            coord_subset,
                            all_preds,
                            gene_names,
                            args.sample_id,
                            scalef=scalef,
                            save_path=save_overlay_path,
                            cmap="jet",
                        )
                else:
                    print(
                        f"No 'uns/spatial' found in {h5ad_overlay_path}. Keys: {list(f.keys())}"
                    )
        else:
            print(f"H5AD file not found for overlay: {h5ad_overlay_path}")
    except Exception as e:
        print(f"Warning: Could not generate histology overlay: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
