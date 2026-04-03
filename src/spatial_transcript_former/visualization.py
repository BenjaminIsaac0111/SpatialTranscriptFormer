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


def _get_pathway_names(args, num_expected: int):
    """Get pathway names to match the expected number of pathways."""
    try:
        from spatial_transcript_former.data.pathways import (
            get_pathway_init,
            MSIGDB_URLS,
        )
        from spatial_transcript_former.data import GeneVocab

        vocab = GeneVocab.from_json(args.data_dir, num_genes=args.num_genes)
        urls = [MSIGDB_URLS["hallmarks"]]
        _, pw_names = get_pathway_init(
            vocab.genes,
            gmt_urls=urls,
            verbose=False,
            filter_names=getattr(args, "pathways", None),
        )
        if len(pw_names) == num_expected:
            return pw_names
    except Exception:
        pass
    return [f"Pathway_{i}" for i in range(num_expected)]


def run_inference_plot(model, args, sample_id, epoch, device):
    """
    Generates a high-quality spatial visualization of pathway predictions.
    """
    from spatial_transcript_former.predict import plot_training_summary

    # 1. Setup Data
    _, val_loader, _ = setup_dataloaders(args, [], [sample_id])
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
                image_features, _, target, coords, mask = batch
                image_features = image_features.to(device)
                coords = coords.to(device)
                mask = mask.to(device)
                target = target.to(device)
            else:
                image_features, _, target, coords = batch
                image_features = image_features.to(device)
                coords = coords.to(device)
                mask = torch.ones(target.shape[0], target.shape[1], device=device)
                target = target.to(device)

            # Forward pass
            if args.whole_slide:
                outputs = model(
                    image_features,
                    rel_coords=coords,
                    mask=mask,
                    return_dense=True,
                )
            else:
                outputs = model(image_features, rel_coords=coords)

            preds_list.append(outputs.cpu())
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

    # Squeeze batch dim for processing
    pathway_preds = all_preds.numpy()[0]
    pathway_truth = all_targets.numpy()[0]
    coords = all_coords.numpy()[0]
    mask = all_masks.numpy()[0]

    # Un-log if necessary to get absolute counts
    if getattr(args, "log_transform", False):
        pathway_preds = np.expm1(pathway_preds)
        pathway_truth = np.expm1(pathway_truth)

    # 3. Filter Valid Spots
    if args.whole_slide:
        valid_idx = ~mask.astype(bool)
    else:
        valid_idx = mask.astype(bool)

    coords = coords[valid_idx]
    pathway_preds = pathway_preds[valid_idx]
    pathway_truth = pathway_truth[valid_idx]

    # The dataloader applies normalize_coordinates which breaks alignment with the full-res histology
    # image. Let's fetch the raw coordinates directly from the .pt file.
    try:
        from spatial_transcript_former.data.paths import resolve_feature_dir
        from spatial_transcript_former.recipes.hest.dataset import (
            load_gene_expression_matrix,
        )

        feat_dir = resolve_feature_dir(
            args.data_dir,
            getattr(args, "backbone", "resnet50"),
            getattr(args, "feature_dir", None),
        )
        pt_path = os.path.join(feat_dir, f"{sample_id}.pt")
        saved_data = torch.load(pt_path, map_location="cpu", weights_only=True)
        raw_coords = saved_data["coords"].numpy()
        barcodes = saved_data["barcodes"]

        st_dir = os.path.join(args.data_dir, "st")
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

        # We need the same valid mask used by the dataset
        _, pt_mask, _ = load_gene_expression_matrix(
            h5ad_path,
            barcodes,
            selected_gene_names=None,
            num_genes=1,
        )
        pt_mask_bool = np.array(pt_mask, dtype=bool)

        if len(raw_coords[pt_mask_bool]) == len(coords):
            coords = raw_coords[pt_mask_bool]
        else:
            print(
                f"Warning: Extracted raw coordinates length ({len(raw_coords[pt_mask_bool])}) "
                f"doesn't match expected valid coordinates length ({len(coords)})."
            )
    except Exception as e:
        print(
            f"Warning: Could not fetch raw coordinates for plotting. Defaults will be used. ({e})"
        )

    if len(coords) == 0:
        return

    # 4. Pathway Names
    pathway_names = _get_pathway_names(args, pathway_truth.shape[-1])

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
