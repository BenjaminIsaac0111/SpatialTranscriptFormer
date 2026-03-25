import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def calculate_qc_stats(h5ad_path, min_umis=500, min_genes=200, max_mt=0.15):
    with h5py.File(h5ad_path, "r") as f:
        barcodes = [
            b.decode("utf-8") if isinstance(b, bytes) else b
            for b in f["obs"]["_index"][:]
        ]
        gene_names = [
            g.decode("utf-8") if isinstance(g, bytes) else g
            for g in f["var"]["_index"][:]
        ]
        X_group = f["X"]
        if isinstance(X_group, h5py.Group):
            X = csr_matrix(
                (X_group["data"][:], X_group["indices"][:], X_group["indptr"][:]),
                shape=(len(barcodes), len(gene_names)),
            )
        else:
            X = f["X"][:]

    n_counts = np.array(X.sum(axis=1)).flatten()
    n_genes = np.array((X > 0).sum(axis=1)).flatten()
    mt_genes = [
        i
        for i, name in enumerate(gene_names)
        if "mt-" in name.lower() or "mt:" in name.lower()
    ]
    if mt_genes:
        mt_counts = np.array(X[:, mt_genes].sum(axis=1)).flatten()
        pct_counts_mt = mt_counts / (n_counts + 1e-9)
    else:
        pct_counts_mt = np.zeros_like(n_counts)

    keep_mask = (
        (n_counts >= min_umis) & (n_genes >= min_genes) & (pct_counts_mt <= max_mt)
    )
    return (
        len(barcodes),
        np.sum(keep_mask),
        np.sum(n_counts < min_umis),
        np.sum(n_genes < min_genes),
        np.sum(pct_counts_mt > max_mt),
    )


if __name__ == "__main__":
    st_dir = r"A:\hest_data\st"

    # Get all h5ad files
    samples = [
        f.replace(".h5ad", "") for f in os.listdir(st_dir) if f.endswith(".h5ad")
    ]
    samples.sort()

    print(f"Analyzing {len(samples)} samples...")
    print(
        f"{'Sample':<15} | {'Total':<6} | {'Kept':<6} | {'% Kept':<8} | {'Low UMI':<8} | {'Low Gene':<8} | {'High MT':<8}"
    )
    print("-" * 80)

    all_results = []
    for s in samples:
        path = os.path.join(st_dir, f"{s}.h5ad")
        try:
            total, kept, l_umi, l_gene, h_mt = calculate_qc_stats(path)
            pct = kept / total if total > 0 else 0
            print(
                f"{s:<15} | {total:<6} | {kept:<6} | {pct:7.1%} | {l_umi:<8} | {l_gene:<8} | {h_mt:<8}"
            )
            all_results.append((total, kept, pct))
        except Exception as e:
            print(f"Error processing {s}: {e}")

    if all_results:
        avg_kept = np.mean([r[2] for r in all_results])
        min_kept = np.min([r[2] for r in all_results])
        max_kept = np.max([r[2] for r in all_results])
        print("-" * 80)
        print(f"GLOBAL SUMMARY ({len(all_results)} samples):")
        print(f"Average Kept: {avg_kept:.1%}")
        print(f"Minimum Kept: {min_kept:.1%}")
        print(f"Maximum Kept: {max_kept:.1%}")
