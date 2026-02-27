import os
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from spatial_transcript_former.data.dataset import (
    load_gene_expression_matrix,
    HEST_FeatureDataset,
    HEST_Dataset,
    load_global_genes,
)
from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.predict import (
    plot_training_summary,
    plot_spatial_genes,
    plot_histology_overlay,
)


def _load_histology(h5ad_path):
    """Load the downscaled histology image from an h5ad file.

    Returns:
        tuple: (image_array, scale_factor) or (None, None) on failure.
    """
    try:
        with h5py.File(h5ad_path, "r") as f:
            if "uns" not in f or "spatial" not in f["uns"]:
                return None, None
            spatial = f["uns/spatial"]
            sample_key = list(spatial.keys())[0] if len(spatial.keys()) > 0 else None
            if sample_key is None:
                return None, None
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
            return img, scalef
    except Exception as e:
        print(f"Warning: Could not load histology: {e}")
        return None, None


def _compute_pathway_truth(gene_truth, gene_names, args=None):
    """Compute pathway ground truth from gene expression using MSigDB membership.

    For each pathway, computes the mean expression of its member genes.
    This is independent of model weights and consistent across epochs.

    Args:
        gene_truth: (N, G) gene expression matrix (log-transformed if applicable).
        gene_names: List of gene names (length G).
        args: Optional CLI args to extract pathway_init filters.

    Returns:
        tuple: (pathway_truth (N, P), pathway_names list) or (None, None).
    """
    try:
        from spatial_transcript_former.data.pathways import (
            get_pathway_init,
            MSIGDB_URLS,
        )

        filter_names = None
        urls = None
        if args is not None and getattr(args, "pathway_init", False):
            if getattr(args, "custom_gmt", None):
                urls = args.custom_gmt
            else:
                urls = [
                    MSIGDB_URLS["hallmarks"],
                    MSIGDB_URLS["c2_medicus"],
                    MSIGDB_URLS["c2_cgp"],
                ]
            filter_names = getattr(args, "pathways", None)

        pw_matrix, pw_names = get_pathway_init(
            gene_names, gmt_urls=urls, filter_names=filter_names, verbose=False
        )
        pw_np = pw_matrix.numpy()  # (P, G) binary membership
        member_counts = pw_np.sum(axis=1, keepdims=True).clip(min=1)  # (P, 1)
        # Mean expression of member genes per pathway
        pathway_truth = (gene_truth @ pw_np.T) / member_counts.T  # (N, P)
        return pathway_truth, pw_names
    except Exception as e:
        print(f"Warning: Could not compute pathway ground truth: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def _get_pathway_names():
    """Get MSigDB Hallmark pathway names.

    Returns:
        list: Pathway names, or None on failure.
    """
    try:
        from spatial_transcript_former.data.pathways import (
            download_msigdb_gmt,
            parse_gmt,
            MSIGDB_URLS,
        )

        url = MSIGDB_URLS["hallmarks"]
        filename = url.split("/")[-1]
        gmt_path = download_msigdb_gmt(url, filename, ".cache")
        pathway_dict = parse_gmt(gmt_path)
        return list(pathway_dict.keys())
    except Exception:
        return None


def run_inference_plot(model, args, sample_id, epoch, device):
    """
    Run inference on a single sample and save a unified pathway visualization.

    Produces a single figure per epoch showing histology + core
    pathways (ground truth vs prediction), where ground truth is computed by
    projecting true gene expression through the model's gene_reconstructor
    via pseudo-inverse so both live in the same activation space.

    Args:
        model (nn.Module): The model to use.
        args (argparse.Namespace): CLI arguments.
        sample_id (str): ID of the sample to plot.
        epoch (int): Current epoch (for filename).
        device (torch.device): Device to run inference on.
    """
    try:
        plot_pathways = getattr(args, "plot_pathways", False)
        if not plot_pathways:
            return

        log_transform = getattr(args, "log_transform", False)

        with torch.no_grad():
            model.eval()

            # Setup paths
            patches_dir = (
                os.path.join(args.data_dir, "patches")
                if os.path.isdir(os.path.join(args.data_dir, "patches"))
                else args.data_dir
            )
            st_dir = os.path.join(args.data_dir, "st")
            h5_path = os.path.join(patches_dir, f"{sample_id}.h5")
            h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

            # Load global genes
            try:
                common_gene_names = load_global_genes(args.data_dir, args.num_genes)
            except Exception:
                common_gene_names = None

            # Run inference
            preds = []
            pathways_list = []

            if args.precomputed:
                feat_dir_name = (
                    "he_features"
                    if args.backbone == "resnet50"
                    else f"he_features_{args.backbone}"
                )
                feature_path = os.path.join(
                    args.data_dir, feat_dir_name, f"{sample_id}.pt"
                )
                if not os.path.exists(feature_path):
                    feature_path = os.path.join(
                        args.data_dir, "patches", feat_dir_name, f"{sample_id}.pt"
                    )

                ds = HEST_FeatureDataset(
                    feature_path,
                    h5ad_path,
                    num_genes=args.num_genes,
                    selected_gene_names=common_gene_names,
                    n_neighbors=args.n_neighbors,
                    whole_slide_mode=args.whole_slide,
                    log1p=log_transform,
                )

                if args.whole_slide:
                    feats, gene_targets, slide_coords = ds[0]
                    feats = feats.unsqueeze(0).to(device)
                    slide_coords = slide_coords.unsqueeze(0).to(device)
                    if isinstance(model, SpatialTranscriptFormer):
                        output = model(
                            feats,
                            return_dense=True,
                            rel_coords=slide_coords,
                            return_pathways=True,
                        )
                        if isinstance(output, tuple):
                            out_preds = output[0]
                            if isinstance(out_preds, tuple):
                                out_preds = out_preds[1]
                            preds.append(out_preds.detach().cpu().squeeze(0))
                            pathways_list.append(output[1].detach().cpu().squeeze(0))
                        else:
                            preds.append(output.detach().cpu().squeeze(0))

                    # Use raw pixel coords from the .pt file for histology overlay
                    # These are guaranteed to be in the same order as the features
                    saved_data = torch.load(
                        feature_path, map_location="cpu", weights_only=True
                    )
                    raw_coords = saved_data["coords"]  # (N, 2)
                    raw_barcodes = saved_data["barcodes"]
                    del saved_data

                    # Compute the same mask the dataset used to filter
                    _, pt_mask, gene_names = load_gene_expression_matrix(
                        h5ad_path,
                        raw_barcodes,
                        selected_gene_names=common_gene_names,
                        num_genes=args.num_genes,
                    )
                    pt_mask_bool = np.array(pt_mask, dtype=bool)
                    coord_subset = raw_coords[pt_mask_bool].numpy()

                    # Pathway truth from the dataset's aligned gene matrix
                    gene_truth = gene_targets.numpy()
                    pathway_truth, pathway_names = _compute_pathway_truth(
                        gene_truth, gene_names, args=args
                    )
                else:
                    dl = DataLoader(ds, batch_size=32, shuffle=False)
                    for feats, _, rel_coords_batch in dl:
                        if isinstance(model, SpatialTranscriptFormer):
                            output = model(
                                feats.to(device),
                                rel_coords=rel_coords_batch.to(device),
                                return_pathways=True,
                            )
                            if isinstance(output, tuple):
                                pathways_list.append(output[1].cpu())
                                out_preds = output[0]
                                if isinstance(out_preds, tuple):
                                    out_preds = out_preds[1]
                                preds.append(out_preds.cpu())
                            else:
                                preds.append(output.cpu())
                        else:
                            preds.append(model(feats.to(device)).cpu())

                    # Non-whole-slide: use h5 file coords (same source as DataLoader)
                    with h5py.File(h5_path, "r") as f:
                        patch_barcodes = f["barcode"][:].flatten()
                        h5_coords = f["coords"][:]
                    gene_matrix, mask, gene_names = load_gene_expression_matrix(
                        h5ad_path,
                        patch_barcodes,
                        selected_gene_names=common_gene_names,
                        num_genes=args.num_genes,
                    )
                    coord_mask = np.array(mask, dtype=bool)
                    coord_subset = h5_coords[coord_mask]
                    gene_truth = np.log1p(gene_matrix) if log_transform else gene_matrix
                    pathway_truth, pathway_names = _compute_pathway_truth(
                        gene_truth, gene_names, args=args
                    )
            else:
                with h5py.File(h5_path, "r") as f:
                    patch_barcodes = f["barcode"][:].flatten()
                    h5_coords = f["coords"][:]
                gene_matrix, mask, gene_names = load_gene_expression_matrix(
                    h5ad_path,
                    patch_barcodes,
                    selected_gene_names=common_gene_names,
                    num_genes=args.num_genes,
                )
                coord_mask = np.array(mask, dtype=bool)
                coord_subset = h5_coords[coord_mask]
                ds = HEST_Dataset(
                    h5_path, coord_subset, gene_matrix, indices=np.where(mask)[0]
                )
                dl = DataLoader(ds, batch_size=32, shuffle=False)
                for imgs, _, rel_coords_batch in dl:
                    if isinstance(model, SpatialTranscriptFormer):
                        output = model(
                            imgs.to(device),
                            rel_coords=rel_coords_batch.to(device),
                            return_pathways=True,
                        )
                        if isinstance(output, tuple):
                            pathways_list.append(output[1].cpu())
                            out_preds = output[0]
                            if isinstance(out_preds, tuple):
                                out_preds = out_preds[1]
                            preds.append(out_preds.cpu())
                        else:
                            preds.append(output.cpu())
                    else:
                        preds.append(model(imgs.to(device)).cpu())
                gene_truth = np.log1p(gene_matrix) if log_transform else gene_matrix
                pathway_truth, pathway_names = _compute_pathway_truth(
                    gene_truth, gene_names, args=args
                )

            if not preds:
                print("Warning: No predictions generated. Skipping plot.")
                return

            # Compute pathway activations from gene predictions (same method as truth)
            # Both truth and pred are now: mean gene expression of pathway members
            gene_preds_np = torch.cat(preds, dim=0).numpy()
            pathway_pred, _ = _compute_pathway_truth(
                gene_preds_np, gene_names, args=args
            )

            if pathway_truth is None or pathway_pred is None:
                print("Warning: Could not compute pathway truth/pred. Skipping plot.")
                return

            # Load histology image
            histology_img, scalef = _load_histology(h5ad_path)

            # Generate unified figure
            save_path = os.path.join(
                args.output_dir, f"{sample_id}_epoch_{epoch+1}.png"
            )
            plot_training_summary(
                coord_subset,
                pathway_pred,
                pathway_truth,
                pathway_names,
                sample_id=sample_id,
                histology_img=histology_img,
                scalef=scalef,
                save_path=save_path,
                cmap=cmap,
            )

            # --- Optional Attention Map Visualization ---
            if getattr(args, "plot_attention", False):
                # Request attention maps from the model
                # We need features and rel_coords again
                if args.precomputed:
                    # Reuse feats and slide_coords from above
                    # This logic only works if whole_slide=True for now for simplicity
                    if args.whole_slide:
                        _, _, attn_maps = model(
                            feats,
                            rel_coords=slide_coords,
                            return_pathways=True,
                            return_attention=True,
                        )
                        save_attn_path = save_path.replace(".png", "_attention.png")
                        plot_attention_maps(
                            coord_subset,
                            attn_maps,
                            pathway_names,
                            sample_id,
                            histology_img=histology_img,
                            scalef=scalef,
                            save_path=save_attn_path,
                        )

    except Exception as e:
        print(f"Warning: Failed to generate validation plot: {e}")
        import traceback

        traceback.print_exc()


def plot_attention_maps(
    coords,
    attentions,
    pathway_names,
    sample_id,
    histology_img=None,
    scalef=1.0,
    save_path=None,
    cmap="magma",
):
    """
    Visualize attention maps for pathway tokens.

    Args:
        attentions: List of [B, H, T, T] tensors (one per layer).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # attentions[-1] is the last layer: (1, n_heads, T, T)
    # T = num_pathways + num_patches
    # We want attention from pathways to patches: attn[:, :, :P, P:]
    last_layer_attn = attentions[-1][0]  # (n_heads, T, T)
    n_heads = last_layer_attn.shape[0]
    p = len(pathway_names)

    # Average across heads or pick first head? Let's average for "publishable" stability.
    # (P, S) where S is num patches
    pathway_to_patch_attn = last_layer_attn[:, :p, p:].mean(dim=0).cpu().numpy()

    # Pick top 4 pathways to visualize
    display_idxs = [0, 1, 2, 3] if p >= 4 else list(range(p))

    fig = plt.figure(figsize=(18, 5 * len(display_idxs)))
    fig.patch.set_facecolor("#ffffff")
    gs = gridspec.GridSpec(len(display_idxs), 2, width_ratios=[1, 1.2])

    for i, pw_idx in enumerate(display_idxs):
        name = pathway_names[pw_idx].replace("HALLMARK_", "").replace("_", " ").title()
        weights = pathway_to_patch_attn[pw_idx]

        # Left: Attention Map
        ax = fig.add_subplot(gs[i, 0])
        if histology_img is not None:
            ax.imshow(histology_img, alpha=0.3)
            vis_coords = coords * scalef
        else:
            vis_coords = coords

        sc = ax.scatter(
            vis_coords[:, 0],
            vis_coords[:, 1],
            c=weights,
            cmap=cmap,
            s=8,
            alpha=0.8,
            edgecolors="none",
        )
        ax.set_title(f"Attention: {name}", fontsize=14)
        ax.axis("off")
        plt.colorbar(sc, ax=ax, shrink=0.8)

        # Right: Histology Detail (zoom into top attention region?)
        # For now, just show histology again with a different overlay or just histological context.
        ax2 = fig.add_subplot(gs[i, 1])
        if histology_img is not None:
            ax2.imshow(histology_img)
        ax2.set_title("Histological Context", fontsize=14)
        ax2.axis("off")

    plt.suptitle(f"Pathway Attention Maps - {sample_id}", fontsize=18, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Attention maps saved to {save_path}")
    plt.close(fig)
