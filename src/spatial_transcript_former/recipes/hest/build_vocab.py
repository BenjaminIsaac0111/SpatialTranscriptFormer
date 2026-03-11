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


def calculate_global_genes(data_dir, ids, num_genes=1000, target_pathways=None):
    st_dir = os.path.join(data_dir, "st")
    if not ids:
        print("No samples provided for calculation.")
        return [], []

    print(f"Scanning {len(ids)} samples in {st_dir}...")

    gene_totals = defaultdict(float)

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

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")

    print(f"Aggregated counts for {len(gene_totals)} unique genes.")

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

    # Sort all by total expression
    sorted_all = sorted(gene_totals.items(), key=lambda x: x[1], reverse=True)

    top_genes = list(prioritized_genes)
    for g, _ in sorted_all:
        if len(top_genes) >= num_genes:
            break
        if g not in prioritized_genes:
            top_genes.append(g)

    print(
        f"Final set: {len(prioritized_genes)} pathway genes + {len(top_genes) - len(prioritized_genes)} global genes"
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

    args = parser.parse_args()

    # Output directly to the specified data directory
    output_path = os.path.join(args.data_dir, "global_genes.json")

    ids = scan_h5ad_files(args.data_dir)

    if not ids:
        print("Vocabulary builder aborted.")
        sys.exit(1)

    top_genes, all_stats = calculate_global_genes(
        args.data_dir, ids, args.num_genes, target_pathways=args.pathways
    )

    print(f"Saving top {len(top_genes)} genes to {output_path}")
    with open(output_path, "w") as f:
        json.dump(top_genes, f, indent=4)

    stats_df = pd.DataFrame(all_stats, columns=["gene", "total_counts"])
    stats_df.to_csv(output_path.replace(".json", "_stats.csv"), index=False)
    print("Saved stats to CSV.")


if __name__ == "__main__":
    main()
