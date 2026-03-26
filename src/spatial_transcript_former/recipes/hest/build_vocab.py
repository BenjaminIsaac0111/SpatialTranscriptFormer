import os
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys
from collections import defaultdict
from scipy.sparse import csr_matrix
from spatial_transcript_former.data.spatial_stats import morans_i_batch

# Add src to path
sys.path.append(os.path.abspath("src"))
from spatial_transcript_former.recipes.hest.io import (
    get_hest_data_dir,
    load_h5ad_metadata,
)
from spatial_transcript_former.config import get_config
from spatial_transcript_former.data.pathways import (
    download_msigdb_gmt,
    parse_gmt,
    MSIGDB_URLS,
)


def scan_h5ad_files(data_dir):
    """
    Find all .h5ad files in the configured data directory's 'st' subfolder.
    Works for both HEST and custom datasets.
    """
    st_dir = os.path.join(data_dir, "st")
    if not os.path.exists(st_dir):
        print(f"Directory not found: {st_dir}")
        print("Please ensure your data matches the structure in docs/DATA_FORMAT.md")
        return []

    sample_ids = [
        f.replace(".h5ad", "") for f in os.listdir(st_dir) if f.endswith(".h5ad")
    ]

    print(f"Found {len(sample_ids)} .h5ad samples in {st_dir}.")
    return sample_ids


def calculate_global_genes(
    data_dir,
    ids,
    num_genes=1000,
    target_pathways=None,
    svg_weight=0.0,
    svg_k=6,
):
    st_dir = os.path.join(data_dir, "st")
    if not ids:
        print("No samples provided for calculation.")
        return [], []

    print(f"Scanning {len(ids)} samples in {st_dir}...")
    if svg_weight > 0:
        print(f"SVG mode: weight={svg_weight}, k={svg_k}")

    gene_totals = defaultdict(float)
    # Moran's I accumulators (sum and count for averaging across samples)
    gene_morans_sum = defaultdict(float)
    gene_morans_count = defaultdict(int)

    for sample_id in tqdm(ids):
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

        try:
            # Use io utility for metadata
            meta = load_h5ad_metadata(h5ad_path)
            gene_names = meta["gene_names"]

            with h5py.File(h5ad_path, "r") as f:
                X = f["X"]
                if isinstance(X, h5py.Group):
                    data = X["data"][:]
                    indices = X["indices"][:]
                    indptr = X["indptr"][:]

                    n_obs = len(meta["barcodes"])
                    n_vars = len(gene_names)
                    mat = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
                    sums = np.array(mat.sum(axis=0)).flatten()
                elif isinstance(X, h5py.Dataset):
                    mat = X[:]
                    sums = np.sum(mat, axis=0)

                for i, gene in enumerate(gene_names):
                    gene_totals[gene] += float(sums[i])

                # --- SVG: compute Moran's I per gene for this sample ---
                if svg_weight > 0 and "obsm" in f and "spatial" in f["obsm"]:
                    coords = f["obsm"]["spatial"][:]
                    # Densify the expression matrix for Moran's I
                    if isinstance(mat, csr_matrix):
                        dense_mat = mat.toarray()
                    else:
                        dense_mat = np.asarray(mat)

                    mi_scores = morans_i_batch(dense_mat, coords, k=svg_k)

                    for i, gene in enumerate(gene_names):
                        gene_morans_sum[gene] += mi_scores[i]
                        gene_morans_count[gene] += 1

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")

    print(f"Aggregated counts for {len(gene_totals)} unique genes.")
    if svg_weight > 0:
        print(f"Computed Moran's I for {len(gene_morans_sum)} genes.")

    prioritized_genes = set()
    if target_pathways:
        print(f"Prioritizing genes from pathways: {target_pathways}")

        collections = ["hallmarks", "c2_kegg", "c2_medicus", "c2_cgp"]
        combined_dict = {}

        for coll in collections:
            url = MSIGDB_URLS[coll]
            filename = url.split("/")[-1]
            gmt_path = download_msigdb_gmt(
                url, filename, os.path.join(data_dir, ".cache")
            )
            combined_dict.update(parse_gmt(gmt_path))

        for p in target_pathways:
            if p in combined_dict:
                for pw_gene in combined_dict[p]:
                    if pw_gene in gene_totals:
                        prioritized_genes.add(pw_gene)
            else:
                print(f"Warning: Pathway {p} not found in MSIGDB dictionaries.")

        print(f"Found {len(prioritized_genes)} valid target pathway genes.")

    # --- Ranking: expression-only or hybrid ---
    all_genes = list(gene_totals.keys())

    if svg_weight > 0 and gene_morans_sum:
        # Compute average Moran's I per gene
        gene_morans_avg = {
            g: gene_morans_sum[g] / gene_morans_count[g]
            for g in all_genes
            if gene_morans_count.get(g, 0) > 0
        }

        # Rank by expression (lower rank = higher expression)
        expr_sorted = sorted(all_genes, key=lambda g: gene_totals[g], reverse=True)
        expr_rank = {g: r for r, g in enumerate(expr_sorted)}

        # Rank by Moran's I (lower rank = higher spatial variability)
        mi_sorted = sorted(
            all_genes, key=lambda g: gene_morans_avg.get(g, 0.0), reverse=True
        )
        mi_rank = {g: r for r, g in enumerate(mi_sorted)}

        # Hybrid score: weighted sum of ranks (lower = better)
        alpha = svg_weight
        hybrid_score = {
            g: (1 - alpha) * expr_rank[g] + alpha * mi_rank[g] for g in all_genes
        }
        sorted_all_genes = sorted(all_genes, key=lambda g: hybrid_score[g])

        # Build stats list with Moran's I column
        sorted_all = [
            (g, gene_totals[g], gene_morans_avg.get(g, 0.0)) for g in sorted_all_genes
        ]
        print(
            f"Hybrid ranking: expression weight={(1 - alpha):.1f}, "
            f"SVG weight={alpha:.1f}"
        )
    else:
        # Expression-only ranking (original behaviour)
        sorted_all = sorted(gene_totals.items(), key=lambda x: x[1], reverse=True)
        sorted_all_genes = [g for g, _ in sorted_all]
        # Pad stats tuples with 0.0 Moran's I for consistent CSV format
        sorted_all = [(g, c, 0.0) for g, c in sorted_all]

    top_genes = list(prioritized_genes)
    for g in sorted_all_genes:
        if len(top_genes) >= num_genes:
            break
        if g not in prioritized_genes:
            top_genes.append(g)

    print(
        f"Final set: {len(prioritized_genes)} pathway genes + "
        f"{len(top_genes) - len(prioritized_genes)} global genes"
    )

    return top_genes, sorted_all


def main():
    parser = argparse.ArgumentParser(
        description="Build Global Gene Vocabulary from .h5ad files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=get_config("data_dirs", ["hest_data"])[0],
        help="Root directory containing the 'st' subfolder",
    )
    parser.add_argument(
        "--num-genes",
        type=int,
        default=get_config("training.num_genes", 1000),
        help="Maximum number of global genes to select",
    )
    parser.add_argument(
        "--pathways",
        nargs="+",
        default=None,
        help="List of MSigDB pathway names to explicitly prioritize (e.g., HALLMARK_P53_PATHWAY)",
    )
    parser.add_argument(
        "--svg-weight",
        type=float,
        default=0.0,
        help="Weight for spatial variability (Moran's I) in gene ranking. "
        "0.0=expression-only (default), 1.0=SVG-only, 0.5=balanced.",
    )
    parser.add_argument(
        "--svg-k",
        type=int,
        default=6,
        help="Number of KNN neighbours for spatial weight matrix (default: 6).",
    )

    args = parser.parse_args()

    # Output directly to the specified data directory
    output_path = os.path.join(args.data_dir, "global_genes.json")

    ids = scan_h5ad_files(args.data_dir)

    if not ids:
        print("Vocabulary builder aborted.")
        sys.exit(1)

    top_genes, all_stats = calculate_global_genes(
        args.data_dir,
        ids,
        args.num_genes,
        target_pathways=args.pathways,
        svg_weight=args.svg_weight,
        svg_k=args.svg_k,
    )

    print(f"Saving top {len(top_genes)} genes to {output_path}")
    with open(output_path, "w") as f:
        json.dump(top_genes, f, indent=4)

    stats_df = pd.DataFrame(all_stats, columns=["gene", "total_counts", "morans_i"])
    stats_df.to_csv(output_path.replace(".json", "_stats.csv"), index=False)
    print("Saved stats to CSV.")


if __name__ == "__main__":
    main()
