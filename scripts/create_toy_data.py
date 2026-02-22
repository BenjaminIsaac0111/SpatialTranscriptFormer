import os
import h5py
import numpy as np
import pandas as pd
import argparse


def create_toy_data(output_dir, num_samples=2, num_patches=100, num_genes=1000):
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = os.path.join(output_dir, "patches")
    st_dir = os.path.join(output_dir, "st")
    os.makedirs(patches_dir, exist_ok=True)
    os.makedirs(st_dir, exist_ok=True)

    metadata = []

    for i in range(num_samples):
        sample_id = f"TOY_SAMPLE_{i}"
        print(f"Generating {sample_id}...")

        # 1. Create .h5 (Patches)
        h5_path = os.path.join(patches_dir, f"{sample_id}.h5")
        with h5py.File(h5_path, "w") as f:
            # Random RGB images (C, H, W) -> HEST uses (H, W, C) usually or similar.
            # My dataset.py expects (H, W, C) from h5_file['img'] and permutes it.
            # Let's check dataset.py: patch = self.h5_file['img'][file_idx]; patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            # So we need (N, H, W, 3)
            imgs = np.random.randint(0, 255, (num_patches, 224, 224, 3), dtype=np.uint8)
            f.create_dataset("img", data=imgs)

            # Coords: Let's make a grid
            side = int(np.sqrt(num_patches))
            x, y = np.meshgrid(np.arange(side), np.arange(side))
            coords = np.stack([x.ravel(), y.ravel()], axis=1)
            # Pad if num_patches not square
            if len(coords) < num_patches:
                extra = np.zeros((num_patches - len(coords), 2))
                coords = np.concatenate([coords, extra], axis=0)
            f.create_dataset("coords", data=coords[:num_patches])

            # Barcodes
            barcodes = [f"patch_{j}".encode("utf-8") for j in range(num_patches)]
            f.create_dataset("barcode", data=barcodes)

        # 2. Create .h5ad (ST data)
        # We'll make genes spatially correlated: gene_0 ~ x, gene_1 ~ y
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
        with h5py.File(h5ad_path, "w") as f:
            # obs: barcodes
            obs = f.create_group("obs")
            obs.create_dataset("_index", data=barcodes)

            # var: gene names
            var = f.create_group("var")
            gene_names = [f"GENE_{j}".encode("utf-8") for j in range(num_genes)]
            var.create_dataset("_index", data=gene_names)

            # X: Expression matrix (N, G)
            # Correlate first few genes with coords for visibility in plots
            X = np.random.rand(num_patches, num_genes).astype(np.float32)
            c_norm = coords[:num_patches] / np.max(coords + 1)
            X[:, 0] = c_norm[:, 0]  # GENE_0 follows X
            X[:, 1] = c_norm[:, 1]  # GENE_1 follows Y
            X[:, 2] = (c_norm[:, 0] + c_norm[:, 1]) / 2  # GENE_2 follows diagonal

            # uns: spatial data for overlay testing
            uns = f.create_group("uns")
            spatial = uns.create_group("spatial")
            st_key = spatial.create_group("ST")

            # Dummy low-res image (100x100)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[:50, :50] = [255, 0, 0]  # Red quadrant
            img[50:, 50:] = [0, 255, 0]  # Green quadrant

            images = st_key.create_group("images")
            images.create_dataset("downscaled_fullres", data=img)

            scalefactors = st_key.create_group("scalefactors")
            scalefactors.create_dataset(
                "tissue_downscaled_fullres_scalef", data=10.0
            )  # Map 10x10 grid to 100x100 pixels

            f.create_dataset("X", data=X)

        metadata.append(
            {
                "id": sample_id,
                "species": "Homo sapiens",
                "organ": "Bowel",
                "disease": "Toy",
            }
        )

    # Save metadata CSV
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "HEST_v1_3_0.csv"), index=False)
    print(f"Toy data generated in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="A:/toy_hest_data")
    args = parser.parse_args()
    create_toy_data(args.output_dir)
