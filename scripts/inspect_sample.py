import os
import sys
import argparse
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))
from spatial_transcript_former.data.io import get_hest_data_dir, load_h5ad_metadata
from spatial_transcript_former.config import get_config
from spatial_transcript_former.data.pathways import (
    download_msigdb_gmt,
    parse_gmt,
    MSIGDB_URLS,
)


def inspect_sample(sample_id, data_dir=None, check_pathways=False):
    if data_dir is None:
        try:
            data_dir = get_hest_data_dir()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    h5ad_path = os.path.join(data_dir, "st", f"{sample_id}.h5ad")
    if not os.path.exists(h5ad_path):
        # Try fallback
        h5ad_path = os.path.join(data_dir, f"{sample_id}.h5ad")
        if not os.path.exists(h5ad_path):
            print(f"Error: Could not find {sample_id}.h5ad in {data_dir}")
            return

    print("=" * 60)
    print(f"INSPECTING SAMPLE: {sample_id}")
    print(f"File: {h5ad_path}")
    print("=" * 60)

    try:
        metadata = load_h5ad_metadata(h5ad_path)
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        return

    # 1. Basic Stats
    n_barcodes = len(metadata.get("barcodes", []))
    n_genes = len(metadata.get("gene_names", []))
    print(f"Dimensions: {n_barcodes} spots x {n_genes} genes")

    # 2. Spatial Metadata
    spatial = metadata.get("spatial")
    if spatial:
        print("\nSpatial Metadata:")
        print(f"  Internal Sample ID: {spatial.get('sample_id')}")

        images = spatial.get("images", {})
        print(f"  Images available: {list(images.keys())}")
        for k, v in images.items():
            print(f"    - {k}: shape {v['shape']}, dtype {v['dtype']}")

        scalefactors = spatial.get("scalefactors", {})
        print(f"  Scale factors: {list(scalefactors.keys())}")
        for k, v in scalefactors.items():
            print(f"    - {k}: {v}")
    else:
        print("\nNo spatial metadata found in H5AD.")

    # 3. Pathway Coverage (Optional)
    if check_pathways:
        print("\nBiological Coverage (MSigDB Hallmarks):")
        try:
            url = MSIGDB_URLS["hallmarks"]
            gmt_path = download_msigdb_gmt(
                url, "h.all.v2024.1.Hs.symbols.gmt", os.path.join(data_dir, ".cache")
            )
            pathway_dict = parse_gmt(gmt_path)

            sample_genes = set(metadata["gene_names"])
            hallmark_genes = set()
            for g_list in pathway_dict.values():
                hallmark_genes.update(g_list)

            overlap = sample_genes.intersection(hallmark_genes)
            print(
                f"  Sample contains {len(overlap)} / {len(hallmark_genes)} Hallmark genes ({len(overlap)/len(hallmark_genes)*100:.1f}%)"
            )
        except Exception as e:
            print(f"  Could not check pathway coverage: {e}")

    # 4. Feature and Patch existence
    patches_path = os.path.join(data_dir, "patches", f"{sample_id}.h5")
    features_path = os.path.join(data_dir, "he_features_ctranspath", f"{sample_id}.pt")

    print("\nFile Availability:")
    print(f"  Raw Patches (.h5): {'[YES]' if os.path.exists(patches_path) else '[NO]'}")
    print(
        f"  Pre-computed Features (.pt): {'[YES]' if os.path.exists(features_path) else '[NO]'}"
    )

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect a HEST sample in detail.")
    parser.add_argument("id", type=str, help="Sample ID (e.g., TENX29)")
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Override data directory"
    )
    parser.add_argument(
        "--pathways", action="store_true", help="Check Hallmark gene coverage"
    )

    args = parser.parse_args()
    inspect_sample(args.id, args.data_dir, args.pathways)


if __name__ == "__main__":
    main()
