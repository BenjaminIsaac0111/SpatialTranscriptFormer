import os
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from spatial_transcript_former.data.dataset import load_gene_expression_matrix, HEST_FeatureDataset, HEST_Dataset, load_global_genes
from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.predict import plot_training_summary


def _load_histology(h5ad_path):
    """Load the downscaled histology image from an h5ad file.

    Returns:
        tuple: (image_array, scale_factor) or (None, None) on failure.
    """
    try:
        with h5py.File(h5ad_path, 'r') as f:
            if 'uns' not in f or 'spatial' not in f['uns']:
                return None, None
            spatial = f['uns/spatial']
            sample_key = list(spatial.keys())[0] if len(spatial.keys()) > 0 else None
            if sample_key is None:
                return None, None
            img_group = spatial[sample_key]['images']
            img_key = 'downscaled_fullres' if 'downscaled_fullres' in img_group else list(img_group.keys())[0]
            img = img_group[img_key][:]
            scale_group = spatial[sample_key]['scalefactors']
            scale_key = 'tissue_downscaled_fullres_scalef' if 'tissue_downscaled_fullres_scalef' in scale_group else list(scale_group.keys())[0]
            scalef = scale_group[scale_key][()]
            return img, scalef
    except Exception as e:
        print(f"Warning: Could not load histology: {e}")
        return None, None


def _compute_pathway_truth(gene_truth, gene_names):
    """Compute pathway ground truth from gene expression using MSigDB membership.

    For each pathway, computes the mean expression of its member genes.
    This is independent of model weights and consistent across epochs.

    Args:
        gene_truth: (N, G) gene expression matrix (log-transformed if applicable).
        gene_names: List of gene names (length G).

    Returns:
        tuple: (pathway_truth (N, P), pathway_names list) or (None, None).
    """
    try:
        from spatial_transcript_former.data.pathways import get_pathway_init
        pw_matrix, pw_names = get_pathway_init(gene_names, verbose=False)
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
        from spatial_transcript_former.data.pathways import download_hallmarks_gmt, parse_gmt
        gmt_path = download_hallmarks_gmt(".cache")
        pathway_dict = parse_gmt(gmt_path)
        return list(pathway_dict.keys())
    except Exception:
        return None


def run_inference_plot(model, args, sample_id, epoch, device):
    """
    Run inference on a single sample and save a unified pathway visualization.

    Produces a single figure per epoch showing histology + fixed bowel-cancer
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
        plot_pathways = getattr(args, 'plot_pathways', False)
        if not plot_pathways:
            return

        log_transform = getattr(args, 'log_transform', False)

        with torch.no_grad():
            model.eval()

            # Setup paths
            patches_dir = os.path.join(args.data_dir, 'patches') if os.path.isdir(os.path.join(args.data_dir, 'patches')) else args.data_dir
            st_dir = os.path.join(args.data_dir, 'st')
            h5_path = os.path.join(patches_dir, f"{sample_id}.h5")
            h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

            with h5py.File(h5_path, 'r') as f:
                patch_barcodes = f['barcode'][:].flatten()
                coords = f['coords'][:]

            # Load global genes
            try:
                common_gene_names = load_global_genes(args.data_dir, args.num_genes)
            except Exception:
                common_gene_names = None

            gene_matrix, mask, gene_names = load_gene_expression_matrix(
                h5ad_path, patch_barcodes, selected_gene_names=common_gene_names, num_genes=args.num_genes
            )

            # Run inference
            preds = []
            pathways_list = []

            if args.precomputed:
                feat_dir_name = 'he_features' if args.backbone == 'resnet50' else f"he_features_{args.backbone}"
                feature_path = os.path.join(args.data_dir, feat_dir_name, f"{sample_id}.pt")
                if not os.path.exists(feature_path):
                    feature_path = os.path.join(args.data_dir, 'patches', feat_dir_name, f"{sample_id}.pt")

                ds = HEST_FeatureDataset(
                    feature_path, h5ad_path, num_genes=args.num_genes,
                    selected_gene_names=common_gene_names,
                    n_neighbors=args.n_neighbors, whole_slide_mode=args.whole_slide
                )

                if args.whole_slide:
                    feats, _, _ = ds[0]
                    feats = feats.unsqueeze(0).to(device)
                    if hasattr(model, 'forward_dense'):
                        output = model.forward_dense(feats, return_pathways=True)
                        if isinstance(output, tuple):
                            preds.append(output[0].detach().cpu().squeeze(0))
                            pathways_list.append(output[1].detach().cpu().squeeze(0))
                        else:
                            preds.append(output.detach().cpu().squeeze(0))
                else:
                    dl = DataLoader(ds, batch_size=32, shuffle=False)
                    for feats, _, rel_coords_batch in dl:
                        if isinstance(model, SpatialTranscriptFormer):
                            output = model(feats.to(device), rel_coords=rel_coords_batch.to(device), return_pathways=True)
                            if isinstance(output, tuple):
                                pathways_list.append(output[1].cpu())
                                preds.append(output[0].cpu())
                            else:
                                preds.append(output.cpu())
                        else:
                            preds.append(model(feats.to(device)).cpu())
            else:
                coord_subset = coords[mask]
                ds = HEST_Dataset(h5_path, coord_subset, gene_matrix, indices=np.where(mask)[0])
                dl = DataLoader(ds, batch_size=32, shuffle=False)
                for imgs, _, rel_coords_batch in dl:
                    if isinstance(model, SpatialTranscriptFormer):
                        output = model(imgs.to(device), rel_coords=rel_coords_batch.to(device), return_pathways=True)
                        if isinstance(output, tuple):
                            pathways_list.append(output[1].cpu())
                            preds.append(output[0].cpu())
                        else:
                            preds.append(output.cpu())
                    else:
                        preds.append(model(imgs.to(device)).cpu())

            if not preds or not pathways_list:
                print("Warning: No predictions/pathways generated. Skipping plot.")
                return

            pathways = torch.cat(pathways_list, dim=0).numpy()
            coord_mask = np.array(mask, dtype=bool)
            coord_subset = coords[coord_mask]

            # Compute pathway ground truth from MSigDB membership (fixed across epochs)
            gene_truth = np.log1p(gene_matrix) if log_transform else gene_matrix
            pathway_truth, pathway_names = _compute_pathway_truth(gene_truth, gene_names)

            if pathway_truth is None:
                print("Warning: Could not compute pathway truth. Skipping plot.")
                return

            # Load histology image
            histology_img, scalef = _load_histology(h5ad_path)

            # Generate unified figure
            save_path = os.path.join(args.output_dir, f"{sample_id}_epoch_{epoch+1}.png")
            plot_training_summary(
                coord_subset, pathways, pathway_truth, pathway_names,
                sample_id=sample_id, histology_img=histology_img, scalef=scalef,
                save_path=save_path
            )

    except Exception as e:
        print(f"Warning: Failed to generate validation plot: {e}")
        import traceback
        traceback.print_exc()
