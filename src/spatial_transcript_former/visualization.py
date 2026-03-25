import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from spatial_transcript_former.recipes.hest.utils import setup_dataloaders


def _load_histology(h5ad_path):
    """
    Load the downscaled histology image from an h5ad file.
    Returns: (image_array, scale_factor) or (None, 1.0) on failure.
    """
    try:
        import h5py

        with h5py.File(h5ad_path, "r") as f:
            if "uns" not in f or "spatial" not in f["uns"]:
                return None, 1.0
            spatial = f["uns/spatial"]
            sample_key = list(spatial.keys())[0] if len(spatial.keys()) > 0 else None
            if sample_key is None:
                return None, 1.0
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
            scalef = float(scale_group[scale_key][()])
            return img, scalef
    except Exception as e:
        print(f"Warning: Could not load histology: {e}")
        return None, 1.0


def _compute_pathway_truth(gene_truth, gene_names, args):
    """
    Compute pathway ground truth from gene expression using MSigDB membership.
    """
    try:
        from spatial_transcript_former.data.pathways import (
            get_pathway_init,
            MSIGDB_URLS,
        )

        urls = [MSIGDB_URLS["hallmarks"]]
        # Only use hallmarks for periodic visualization to keep it fast
        
        pw_matrix, pw_names = get_pathway_init(
            gene_names, 
            gmt_urls=urls, 
            verbose=False,
            filter_names=getattr(args, "pathways", None)
        )
        pw_np = pw_matrix.numpy()  # (P, G)

        # Z-score normalize gene spatial patterns to match AuxiliaryPathwayLoss
        gene_truth = gene_truth.astype(np.float64)
        means = np.mean(gene_truth, axis=0, keepdims=True)
        stds = np.std(gene_truth, axis=0, keepdims=True)
        stds[stds < 1e-6] = 1e-6  # prevent division by zero
        norm_genes = (gene_truth - means) / stds

        member_counts = pw_np.sum(axis=1, keepdims=True).clip(min=1)
        # Mean expression of normalized member genes per pathway
        pathway_truth = (norm_genes @ pw_np.T) / member_counts.T  # (N, P)
        return pathway_truth.astype(np.float32), pw_names
    except Exception as e:
        print(f"Warning: Could not compute pathway ground truth: {e}")
        return None, None


def run_inference_plot(model, args, sample_id, epoch, device):
    """
    Generates a high-quality spatial visualization of pathway predictions.
    """
    from spatial_transcript_former.predict import plot_training_summary

    # 1. Setup Data
    _, val_loader = setup_dataloaders(args, [], [sample_id])
    if val_loader is None:
        return

    model.eval()
    preds_list = []
    pathways_list = []
    targets_list = []
    coords_list = []
    masks_list = []

    # 2. Run Inference
    with torch.no_grad():
        for batch in val_loader:
            if args.whole_slide:
                image_features, target, coords, mask = batch
                image_features = image_features.to(device)
                coords = coords.to(device)
                mask = mask.to(device)
                target = target.to(device)
            else:
                image_features, target, coords = batch
                image_features = image_features.to(device)
                coords = coords.to(device)
                mask = torch.ones(target.shape[0], target.shape[1], device=device)
                target = target.to(device)

            # Forward pass
            if args.whole_slide:
                outputs = model(
                    image_features, rel_coords=coords, mask=mask, return_dense=True, return_pathways=True
                )
            else:
                outputs = model(image_features, rel_coords=coords, return_pathways=True)

            # The model might return a tuple if pathways are enabled
            if isinstance(outputs, tuple):
                pred_counts = outputs[0]
                pred_pathways = outputs[1] if len(outputs) > 1 else None
            else:
                pred_counts = outputs
                pred_pathways = None

            preds_list.append(pred_counts.cpu())
            if pred_pathways is not None:
                pathways_list.append(pred_pathways.cpu())
            targets_list.append(target.cpu())
            coords_list.append(coords.cpu())
            masks_list.append(mask.cpu())

            if args.whole_slide:
                break  # Whole slide is one batch

    # Concatenate results (for patch-based)
    all_preds = torch.cat(preds_list, dim=1 if args.whole_slide else 0)
    all_targets = torch.cat(targets_list, dim=1 if args.whole_slide else 0)
    all_coords = torch.cat(coords_list, dim=1 if args.whole_slide else 0)
    all_masks = torch.cat(masks_list, dim=1 if args.whole_slide else 0)

    if pathways_list:
        all_pathways = torch.cat(pathways_list, dim=1 if args.whole_slide else 0)
    else:
        all_pathways = None

    # Squeeze batch dim for processing
    pred_counts = all_preds.numpy()[0]
    target_genes = all_targets.numpy()[0]
    coords = all_coords.numpy()[0]
    mask = all_masks.numpy()[0]

    if all_pathways is not None:
        pathway_preds = all_pathways.numpy()[0]
    else:
        pathway_preds = None

    # Un-log if necessary to get absolute counts
    if getattr(args, "log_transform", False):
        pred_counts = np.expm1(pred_counts)
        target_genes = np.expm1(target_genes)
        if pathway_preds is not None:
            pathway_preds = np.expm1(pathway_preds)

    # 3. Filter Valid Spots
    if args.whole_slide:
        valid_idx = ~mask.astype(bool)
    else:
        valid_idx = mask.astype(bool)

    coords = coords[valid_idx]
    pred_counts = pred_counts[valid_idx]
    target_genes = target_genes[valid_idx]
    if pathway_preds is not None:
        pathway_preds = pathway_preds[valid_idx]

    if len(coords) == 0:
        return

    # 4. Compute Pathway Truth
    from spatial_transcript_former.data import GeneVocab

    vocab = GeneVocab.from_json(args.data_dir, num_genes=args.num_genes)
    gene_names = vocab.genes

    # Pathway truth calculation
    pathway_truth, pathway_names = _compute_pathway_truth(
        target_genes, gene_names, args
    )

    if pathway_truth is None:
        print("Warning: Could not compute pathway truth. Visualization skipped.")
        return

    # If the model didn't return pathways directly, use predicted genes to compute them
    if pathway_preds is None:
        pathway_preds, _ = _compute_pathway_truth(pred_counts, gene_names, args)

    # 5. Load Histology
    st_dir = os.path.join(args.data_dir, "st")
    h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
    histology_img, scalef = _load_histology(h5ad_path)

    # 6. Plot
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{sample_id}_epoch_{epoch}.png")

    plot_training_summary(
        coords,
        pathway_preds,
        pathway_truth,
        pathway_names,
        sample_id=sample_id,
        histology_img=histology_img,
        scalef=scalef,
        save_path=save_path,
        plot_pathways_list=getattr(args, "plot_pathways_list", None),
    )
