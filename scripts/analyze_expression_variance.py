import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import json


def analyze_sample(h5ad_path):
    print(f"Analyzing {h5ad_path}...")

    with h5py.File(h5ad_path, "r") as f:
        # Check standard AnnData structure
        if "X" in f:
            if isinstance(f["X"], h5py.Group):
                # Sparse format (CSR/CSC)
                data_group = f["X"]["data"][:]
                n_cells = (
                    f["obs"]["_index"].shape[0]
                    if "_index" in f["obs"]
                    else len(f["obs"])
                )
                n_genes = (
                    f["var"]["_index"].shape[0]
                    if "_index" in f["var"]
                    else len(f["var"])
                )

                print(f"Data is sparse, shape: ({n_cells}, {n_genes})")
                print(f"Non-zero elements: {len(data_group)}")

                # Analyze non-zero elements
                mean_val = np.mean(data_group)
                max_val = np.max(data_group)
                min_val = np.min(data_group)

                print(f"Non-zero Mean: {mean_val:.4f}")
                print(f"Max Expression: {max_val:.4f}")
                print(f"Min Expression: {min_val:.4f}")

            else:
                # Dense array
                X = f["X"][:]
                print(f"Data is dense, shape: {X.shape}")

                # Basic stats
                mean_exp = np.mean(X, axis=0)  # per gene mean
                var_exp = np.var(X, axis=0)  # per gene variance
                max_exp = np.max(X, axis=0)

                sparsity = np.sum(X == 0) / X.size
                print(f"Overall Sparsity (zeros): {sparsity:.2%}")

                print(
                    f"Gene Mean Range: {np.min(mean_exp):.4f} to {np.max(mean_exp):.4f}"
                )
                print(f"Gene Var Range: {np.min(var_exp):.4f} to {np.max(var_exp):.4f}")
                print(f"Overall Max Expression: {np.max(max_exp):.4f}")

                # Check for extreme differences in variance
                var_ratio = np.max(var_exp) / (np.min(var_exp) + 1e-8)
                print(f"Ratio of max/min gene variance: {var_ratio:.4e}")

                return {
                    "sparsity": sparsity,
                    "var_ratio": var_ratio,
                    "max_exp": np.max(max_exp),
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="A:\\hest_data",
        help="Path to HEST data directory",
    )
    args = parser.parse_args()

    st_dir = os.path.join(args.data_dir, "st")
    if not os.path.exists(st_dir):
        print(f"Error: Directory not found: {st_dir}")
        exit(1)

    # Get a few random samples
    samples = [f for f in os.listdir(st_dir) if f.endswith(".h5ad")]
    if not samples:
        print(f"No .h5ad files found in {st_dir}")

    # Analyze the first couple of samples
    for sample in samples[:3]:
        analyze_sample(os.path.join(st_dir, sample))
        print("-" * 50)
