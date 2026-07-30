import os
import argparse
import fnmatch
import json
import logging
import zipfile
from typing import List, Optional

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ID = "MahmoodLab/hest"
METADATA_FILENAME = "HEST_v1_3_0.csv"


def download_metadata(local_dir: str, force: bool = False) -> str:
    """
    Ensure the HEST metadata CSV is present locally.
    Downloads it from Hugging Face if not found or if force is True.
    """
    logger.info(f"Checking for metadata {METADATA_FILENAME} in {local_dir}...")
    local_path = os.path.join(local_dir, METADATA_FILENAME)

    if os.path.exists(local_path) and not force:
        logger.info(f"Metadata found at {local_path}")
        return local_path

    try:
        logger.info(f"Downloading metadata from {REPO_ID}...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=METADATA_FILENAME,
            repo_type="dataset",
            local_dir=local_dir,
        )
        logger.info(f"Metadata downloaded to {path}")
        return path
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            logger.error("\n" + "=" * 60)
            logger.error("AUTHENTICATION REQUIRED: The HEST dataset is gated.")
            logger.error(
                "1. Accept the dataset terms at: https://huggingface.co/datasets/MahmoodLab/hest"
            )
            logger.error(
                "2. Get an access token from: https://huggingface.co/settings/tokens"
            )
            logger.error(
                "3. Run 'huggingface-cli login' OR set the HF_TOKEN environment variable."
            )
            logger.error("=" * 60 + "\n")
            import sys

            sys.exit(1)

        logger.error(f"Failed to download metadata: {e}")
        raise


def filter_samples(
    metadata_path: str,
    organ: Optional[str] = None,
    disease_state: Optional[str] = None,
    st_technology: Optional[str] = None,
) -> List[str]:
    """
    Filter HEST samples based on provided criteria.
    Returns a list of sample IDs.
    """
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        logger.error(f"Failed to read metadata file {metadata_path}: {e}")
        raise

    logger.info(f"Loaded metadata with {len(df)} total samples.")

    mask = pd.Series([True] * len(df))

    if organ:
        # Case-insensitive matching for convenience
        mask &= df["organ"].str.lower() == organ.lower()

    if disease_state:
        mask &= df["disease_state"].str.lower() == disease_state.lower()

    if st_technology:
        mask &= df["st_technology"].str.lower() == st_technology.lower()

    filtered_df = df[mask]
    logger.info(
        f"Found {len(filtered_df)} samples matching criteria: organ={organ}, disease={disease_state}, tech={st_technology}"
    )

    if filtered_df.empty:
        logger.warning("No samples found matching the criteria.")
        return []

    return filtered_df["id"].tolist()


def _resolve_and_fetch(repo_id, patterns, local_dir, repo_type="dataset"):
    """Resolve ``patterns`` against the repo, then fetch each file individually.

    Deliberately avoids ``snapshot_download`` here. HEST is a large repo, so
    ``huggingface_hub`` skips materialising the file list and hands a *generator*
    to ``tqdm.thread_map``; newer tqdm versions compute the total via
    ``_min_map_len``, which discards inputs without a length hint and then calls
    ``min()`` on an empty sequence:

        ValueError: min() arg is an empty sequence

    Both packages are unpinned, so that combination appears on fresh installs
    (it broke CI while working locally on tqdm 4.68.3). Resolving the file list
    ourselves sidesteps the interaction entirely and lets us fail with a useful
    message when the patterns match nothing, rather than a crash inside tqdm.

    Returns the list of files fetched.
    """
    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    matched = sorted(
        f for f in all_files if any(fnmatch.fnmatch(f, pat) for pat in patterns)
    )
    if not matched:
        raise FileNotFoundError(
            f"No files in {repo_id} matched any of {len(patterns)} patterns "
            f"(e.g. {patterns[:3]}). Check the sample IDs exist in the metadata."
        )

    logger.info(f"Resolved {len(matched)} files to download.")
    for i, filename in enumerate(matched, 1):
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=local_dir,
        )
        if i % 25 == 0 or i == len(matched):
            logger.info(f"  fetched {i}/{len(matched)}")
    return matched


def download_hest_subset(
    sample_ids: List[str],
    local_dir: str,
    additional_patterns: Optional[List[str]] = None,
):
    """
    Download specific samples and additional patterns from the HEST dataset.
    Downloads files matching the sample IDs across all subdirectories (e.g., st/, wsis/, cellvit_seg/).
    """
    if not sample_ids and not additional_patterns:
        logger.warning("No sample IDs or patterns provided to download.")
        return

    patterns = []
    # Construct patterns for each sample ID to match relevant files recursively
    # Matches: st/{id}.h5ad, wsis/{id}.tif, etc.
    for sample_id in sample_ids:
        patterns.append(f"**/{sample_id}.*")
        patterns.append(f"**/{sample_id}_*")

    if additional_patterns:
        patterns.extend(additional_patterns)

    # Always include metadata and readme if possible
    patterns.extend(["README.md", ".gitattributes", METADATA_FILENAME])

    # Remove duplicates
    patterns = list(set(patterns))

    logger.info(
        f"Starting download for {len(sample_ids)} samples (searching for ST data, WSIs, segmentation, etc.)..."
    )
    try:
        _resolve_and_fetch(REPO_ID, patterns, local_dir)
        logger.info("Download completed successfully.")
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            logger.error("\n" + "=" * 60)
            logger.error("AUTHENTICATION REQUIRED: The HEST dataset is gated.")
            logger.error(
                "1. Accept the dataset terms at: https://huggingface.co/datasets/MahmoodLab/hest"
            )
            logger.error(
                "2. Get an access token from: https://huggingface.co/settings/tokens"
            )
            logger.error(
                "3. Run 'huggingface-cli login' OR set the HF_TOKEN environment variable."
            )
            logger.error("=" * 60 + "\n")
            import sys

            sys.exit(1)

        logger.error(f"Download failed: {e}")
        raise

    # Post-processing: Unzip segmentation files in known directories
    # We check commonly used directories for zipped content
    seg_dirs = ["cellvit_seg", "xenium_seg", "tissue_seg"]

    for seg_dirname in seg_dirs:
        seg_dir = os.path.join(local_dir, seg_dirname)
        if os.path.exists(seg_dir):
            zip_files = [s for s in os.listdir(seg_dir) if s.endswith(".zip")]
            if zip_files:
                logger.info(f"Unzipping {len(zip_files)} files in {seg_dirname}...")
                for filename in tqdm(zip_files, desc=f"Unzipping {seg_dirname}"):
                    path_zip = os.path.join(seg_dir, filename)
                    try:
                        with zipfile.ZipFile(path_zip, "r") as zip_ref:
                            zip_ref.extractall(seg_dir)
                    except zipfile.BadZipFile:
                        logger.warning(f"Failed to unzip {filename}: Bad Zip File")
                    except Exception as e:
                        logger.warning(f"Failed to unzip {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download subsets of the HEST dataset based on metadata filters."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="hest_data",
        help="Local directory to save data (default: hest_data).",
    )
    parser.add_argument(
        "--organ", type=str, help="Filter by organ (e.g., Bowel, Kidney)."
    )
    parser.add_argument(
        "--disease",
        type=str,
        dest="disease_state",
        help="Filter by disease state (e.g., Cancer).",
    )
    parser.add_argument(
        "--tech",
        type=str,
        dest="st_technology",
        help="Filter by spatial technology (e.g., Visium).",
    )
    parser.add_argument(
        "--list_organs",
        action="store_true",
        help="List available organs in the metadata and exit.",
    )

    args = parser.parse_args()

    # Ensure local directory exists
    os.makedirs(args.local_dir, exist_ok=True)

    # Step 1: Get Metadata
    try:
        metadata_path = download_metadata(args.local_dir)
    except Exception:
        return

    # Helper: List organs if requested
    if args.list_organs:
        try:
            df = pd.read_csv(metadata_path)
            organs = sorted(df["organ"].dropna().unique().tolist())
            print("Available Organs:")
            for organ in organs:
                print(f" - {organ}")
        except Exception as e:
            logger.error(f"Could not list organs: {e}")
        return

    # Step 2: Filter Samples
    # If no filters provided, warn user (or download all? safer to warn)
    if not any([args.organ, args.disease_state, args.st_technology]):
        logger.warning(
            "No filters provided (organ, disease, tech). This would download the ENTIRE dataset."
        )
        confirm = input("Are you sure you want to download EVERYTHING? (y/n): ")
        if confirm.lower() != "y":
            logger.info("Aborting download.")
            return

    sample_ids = filter_samples(
        metadata_path,
        organ=args.organ,
        disease_state=args.disease_state,
        st_technology=args.st_technology,
    )

    if not sample_ids:
        logger.info("No samples to download.")
        return

    # Step 3: Download
    download_hest_subset(sample_ids, args.local_dir)


if __name__ == "__main__":
    main()
