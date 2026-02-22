import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


def check_alignment(
    sample_id="TENX29", output_path="visualizations/alignment_check.png"
):
    h5ad_path = f"A:/hest_data/st/{sample_id}.h5ad"
    feat_path = f"A:/hest_data/he_features_ctranspath/{sample_id}.pt"

    if not os.path.exists(h5ad_path):
        h5ad_path = f"z:/Projects/SpatialTranscriptFormer/data/st/{sample_id}.h5ad"
        feat_path = f"z:/Projects/SpatialTranscriptFormer/data/he_features_ctranspath/{sample_id}.pt"

    print(f"Checking alignment for {sample_id}...")

    # 1. Load Image and Scale
    with h5py.File(h5ad_path, "r") as f:
        spatial = f["uns/spatial"]
        sample_key = list(spatial.keys())[0]
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

        # Also get barcodes from H5AD to verify order
        if "obs" in f and "_index" in f["obs"]:
            st_barcodes = [
                b.decode("utf-8") if isinstance(b, bytes) else str(b)
                for b in f["obs"]["_index"][:]
            ]
        else:
            st_barcodes = []

    # 2. Load Features/Coords
    data = torch.load(feat_path, map_location="cpu", weights_only=True)
    coords = data["coords"].numpy()
    barcodes = data["barcodes"]
    barcodes = [b.decode("utf-8") if isinstance(b, bytes) else str(b) for b in barcodes]

    print(f"Image shape: {img.shape}, Scale: {scalef}")
    print(f"Num patches: {len(coords)}")

    # 3. Plot Full Slide
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(img)

    # Project coords to image space
    vis_coords = coords * scalef

    # Plot only a subset or all points with small size
    ax.scatter(
        vis_coords[:, 0],
        vis_coords[:, 1],
        c="cyan",
        s=1,
        alpha=0.5,
        label="Patch Centers",
    )

    # Highlight a few specific points to check orientation
    # Top-left most point
    tl_idx = np.argmin(coords[:, 0] + coords[:, 1])
    ax.scatter(
        vis_coords[tl_idx, 0],
        vis_coords[tl_idx, 1],
        c="red",
        s=50,
        edgecolors="white",
        label="Top-Left Point",
    )

    # Bottom-right most point
    br_idx = np.argmax(coords[:, 0] + coords[:, 1])
    ax.scatter(
        vis_coords[br_idx, 0],
        vis_coords[br_idx, 1],
        c="yellow",
        s=50,
        edgecolors="black",
        label="Bottom-Right Point",
    )

    ax.set_title(
        f"Alignment Check: {sample_id}\n(Red=TL, Yellow=BR) - Cyan points should form the tissue mask"
    )
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Alignment check plot saved to {output_path}")


if __name__ == "__main__":
    check_alignment()
