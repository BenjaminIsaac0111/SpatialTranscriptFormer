import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def diagnose_qc(h5ad_path, output_path, min_umis=500, min_genes=200, max_mt=0.15):
    print(f"Loading {h5ad_path}...")

    with h5py.File(h5ad_path, "r") as f:
        # Load barcodes and gene names
        barcodes = [
            b.decode("utf-8") if isinstance(b, bytes) else b
            for b in f["obs"]["_index"][:]
        ]
        gene_names = [
            g.decode("utf-8") if isinstance(g, bytes) else g
            for g in f["var"]["_index"][:]
        ]

        # Load expression matrix X
        X_group = f["X"]
        if isinstance(X_group, h5py.Group):
            data = X_group["data"][:]
            indices = X_group["indices"][:]
            indptr = X_group["indptr"][:]
            X = csr_matrix(
                (data, indices, indptr), shape=(len(barcodes), len(gene_names))
            )
        else:
            X = f["X"][:]

        # Spatial coordinates (usually in obsm/spatial)
        if "obsm" in f and "spatial" in f["obsm"]:
            coords = f["obsm"]["spatial"][:]
        else:
            # Fallback if spatial not found
            coords = np.zeros((len(barcodes), 2))
            print("Warning: Spatial coordinates not found in obsm/spatial")

    # Calculate QC metrics
    n_counts = np.array(X.sum(axis=1)).flatten()
    n_genes = np.array((X > 0).sum(axis=1)).flatten()

    # MT fraction
    # Robust MT detection: check for mt- or MT- anywhere, but prioritize common patterns
    mt_genes = [
        i
        for i, name in enumerate(gene_names)
        if "mt-" in name.lower() or "mt:" in name.lower()
    ]
    if mt_genes:
        print(f"Found {len(mt_genes)} mitochondrial genes.")
        mt_counts = np.array(X[:, mt_genes].sum(axis=1)).flatten()
        pct_counts_mt = mt_counts / (n_counts + 1e-9)
    else:
        pct_counts_mt = np.zeros_like(n_counts)
        print("No mitochondrial genes found.")

    # Filter mask
    pass_umi = n_counts >= min_umis
    pass_gene = n_genes >= min_genes
    pass_mt = pct_counts_mt <= max_mt

    keep_mask = pass_umi & pass_gene & pass_mt

    total_spots = len(barcodes)
    kept_spots = np.sum(keep_mask)

    print(f"Total spots: {total_spots}")
    print(f"Kept spots: {kept_spots} ({kept_spots/total_spots:.1%})")
    print(f"Filtered: {total_spots - kept_spots}")
    print(f"  - Low UMI (<{min_umis}): {np.sum(~pass_umi)}")
    print(f"  - Low Genes (<{min_genes}): {np.sum(~pass_gene)}")
    print(f"  - High MT (>{max_mt:.1%}): {np.sum(~pass_mt)}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. UMI vs Genes
    axes[0, 0].scatter(n_counts, n_genes, c=keep_mask, cmap="RdYlGn", alpha=0.5, s=10)
    axes[0, 0].axvline(
        min_umis, color="red", linestyle="--", label=f"Min UMI={min_umis}"
    )
    axes[0, 0].axhline(
        min_genes, color="blue", linestyle="--", label=f"Min Genes={min_genes}"
    )
    axes[0, 0].set_xlabel("Total UMI counts")
    axes[0, 0].set_ylabel("Number of detected genes")
    axes[0, 0].set_title("QC: UMI vs Genes")
    axes[0, 0].legend()

    # 2. MT Fraction distribution
    axes[0, 1].hist(pct_counts_mt, bins=50, color="gray", alpha=0.7)
    axes[0, 1].axvline(
        max_mt, color="red", linestyle="--", label=f"Max MT={max_mt:.0%}"
    )
    axes[0, 1].set_xlabel("Mitochondrial Fraction")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("QC: MT Fraction Distribution")
    axes[0, 1].legend()

    # 3. Spatial: Before (Total)
    axes[1, 0].scatter(
        coords[:, 0], coords[:, 1], c="lightgray", s=15, label="All Spots"
    )
    axes[1, 0].scatter(
        coords[keep_mask, 0], coords[keep_mask, 1], c="green", s=15, label="Pass QC"
    )
    axes[1, 0].set_title("Spatial Distribution: Kept vs Filtered")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].legend()

    # 4. Summary Table (as text)
    stats_text = (
        f"Sample: {os.path.basename(h5ad_path)}\n\n"
        f"Total Spots: {total_spots}\n"
        f"Kept Spots: {kept_spots} ({kept_spots/total_spots:.1%})\n"
        f"Filtered: {total_spots - kept_spots}\n\n"
        f"Thresholds:\n"
        f"Min UMI: {min_umis}\n"
        f"Min Genes: {min_genes}\n"
        f"Max MT: {max_mt:.0%}"
    )
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=14, family="monospace")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="MEND29")
    args = parser.parse_args()

    sample_id = args.sample
    h5ad_file = f"A:\\hest_data\\st\\{sample_id}.h5ad"
    output_file = f"qc_diagnosis_{sample_id}.png"

    if not os.path.exists(h5ad_file):
        print(f"Error: {h5ad_file} not found.")
    else:
        diagnose_qc(h5ad_file, output_file)
