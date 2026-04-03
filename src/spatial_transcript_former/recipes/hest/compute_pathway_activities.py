"""
Offline preprocessing: compute spatial pathway activity scores for HEST samples.

For each sample .h5ad, this script:
  1. Loads the raw gene expression matrix (spots x genes)
  2. Applies per-spot QC (min UMIs, min genes, max MT%) on raw counts
  3. Applies CP10k normalisation + log1p to surviving spots
  4. Z-scores each gene across spots, then computes per-pathway mean z-score
  5. Saves the resulting activity matrix to
     <data_dir>/pathway_activities/<sample_id>.h5

Pathway scores are computed from MSigDB Hallmark gene sets (50 pathways).
For each pathway, the score per spot is the mean z-scored expression of
member genes present in the expression matrix.

Non-human samples are auto-skipped via HEST metadata. Samples with
fewer than ``--min-pathways`` scored pathways are excluded.

The saved files are consumed at training time by HEST_FeatureDataset when
``pathway_targets_dir`` is provided.

Usage::

    stf-compute-pathways --data-dir hest_data
    stf-compute-pathways --data-dir hest_data --sample-ids MEND29 TENX88
    stf-compute-pathways --data-dir hest_data --qc-max-mt 0.10 --overwrite
    stf-compute-pathways --data-dir hest_data --no-species-filter
"""

import argparse
import os
import logging

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from spatial_transcript_former.data.pathways import (
    MSIGDB_URLS,
    download_msigdb_gmt,
    parse_gmt,
)

logger = logging.getLogger(__name__)


def _load_expression(
    h5ad_path: str,
    target_sum: int = 10_000,
    qc_min_umis: int = None,
    qc_min_genes: int = None,
    qc_max_mt: float = None,
):
    """Load a HEST .h5ad, apply spot QC on raw counts, then normalise.

    QC is applied BEFORE normalisation so that low-quality spots do not
    distort CP10k library-size estimates or downstream z-scores.

    Returns
    -------
    adata : anndata.AnnData
        CP10k-normalised, log1p-transformed AnnData containing only
        QC-passing spots.
    n_before : int
        Number of spots before QC.
    n_after : int
        Number of spots after QC.
    """
    import anndata as ad
    from scipy.sparse import issparse, csr_matrix

    adata = ad.read_h5ad(h5ad_path)
    n_before = adata.n_obs

    # --- Spot QC on raw counts ---
    if qc_min_umis is not None or qc_min_genes is not None or qc_max_mt is not None:
        raw = adata.X
        if issparse(raw):
            raw = raw.toarray()
        raw = raw.astype(np.float32)

        qc_mask = np.ones(n_before, dtype=bool)

        n_counts = raw.sum(axis=1)
        if qc_min_umis is not None:
            qc_mask &= n_counts >= qc_min_umis

        if qc_min_genes is not None:
            n_detected = (raw > 0).sum(axis=1)
            qc_mask &= n_detected >= qc_min_genes

        if qc_max_mt is not None:
            mt_prefixes = ["mt-", "mt:", "mt_", "grcm38_mt-", "hs_mt-"]
            gene_names_lower = [g.lower() for g in adata.var_names]
            mt_cols = [
                i
                for i, name in enumerate(gene_names_lower)
                if any(name.startswith(p) for p in mt_prefixes)
            ]
            if mt_cols:
                mt_counts = raw[:, mt_cols].sum(axis=1)
                pct_mt = mt_counts / (n_counts + 1e-9)
                qc_mask &= pct_mt <= qc_max_mt

        n_filtered = n_before - qc_mask.sum()
        if n_filtered > 0:
            sample = os.path.basename(h5ad_path).replace(".h5ad", "")
            logger.info(
                f"[{sample}] QC filtered {n_filtered}/{n_before} spots "
                f"({qc_mask.sum()}/{n_before} kept)"
            )
            adata = adata[qc_mask].copy()

    n_after = adata.n_obs

    # --- CP10k + log1p normalisation on surviving spots ---
    counts = adata.X
    if issparse(counts):
        counts = counts.toarray()
    counts = counts.astype(np.float32)
    lib_sizes = counts.sum(axis=1, keepdims=True).clip(min=1.0)
    counts = counts / lib_sizes * target_sum
    np.log1p(counts, out=counts)

    adata.X = csr_matrix(counts)

    return adata, n_before, n_after


def _load_hallmark_sets(cache_dir: str = ".cache"):
    """Download and parse MSigDB Hallmark gene sets.

    Returns
    -------
    pathway_dict : dict[str, list[str]]
        {pathway_name: [gene_symbols]}
    """
    url = MSIGDB_URLS["hallmarks"]
    filename = url.split("/")[-1]
    gmt_path = download_msigdb_gmt(url, filename, cache_dir)
    return parse_gmt(gmt_path)


def _score_pathways(expr_matrix, gene_names, pathway_dict, min_genes=5):
    """Score pathway activities via mean z-scored expression of member genes.

    Parameters
    ----------
    expr_matrix : np.ndarray, shape (n_spots, n_genes)
        Normalised expression matrix (CP10k + log1p).
    gene_names : list of str
        Gene symbols corresponding to columns of expr_matrix.
    pathway_dict : dict[str, list[str]]
        {pathway_name: [gene_symbols]} from parse_gmt.
    min_genes : int
        Minimum number of pathway member genes that must be present in the
        expression matrix for a pathway to be scored.

    Returns
    -------
    activities : np.ndarray, shape (n_spots, n_pathways), float32
        All pathways are included; those with fewer than ``min_genes``
        matched genes are filled with zeros.
    all_pathways : list of str
        All pathway names (same order as columns).
    n_scored : int
        Number of pathways that met the min_genes threshold.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_spots = expr_matrix.shape[0]

    # Z-score each gene across spots (zero-variance genes get z=0)
    means = expr_matrix.mean(axis=0)
    stds = expr_matrix.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    z_matrix = (expr_matrix - means) / stds

    all_pathways = list(pathway_dict.keys())
    activities = np.zeros((n_spots, len(all_pathways)), dtype=np.float32)
    n_scored = 0

    for i, (pw_name, pw_genes) in enumerate(pathway_dict.items()):
        col_indices = [gene_to_idx[g] for g in pw_genes if g in gene_to_idx]
        if len(col_indices) < min_genes:
            continue
        activities[:, i] = z_matrix[:, col_indices].mean(axis=1)
        n_scored += 1

    return activities, all_pathways, n_scored


def compute_pathway_activities_for_sample(
    h5ad_path: str,
    output_path: str,
    target_sum: int = 10_000,
    min_genes: int = 5,
    min_pathways: int = 25,
    qc_min_umis: int = None,
    qc_min_genes: int = None,
    qc_max_mt: float = None,
    overwrite: bool = False,
):
    """Compute and save Hallmark pathway activity scores for one HEST sample.

    Parameters
    ----------
    h5ad_path : str
        Path to the .h5ad expression file.
    output_path : str
        Where to write the resulting .h5 file.
    target_sum : int
        Library-size normalisation target (default 10 000 = CP10k).
    min_genes : int
        Minimum number of pathway member genes that must be present in the
        expression matrix for a pathway to be scored.
    min_pathways : int
        Minimum number of scored pathways required for a sample to be saved.
        Samples below this threshold are skipped.
    qc_min_umis : int or None
        Minimum total UMI count per spot (raw counts).
    qc_min_genes : int or None
        Minimum number of detected genes per spot (raw counts).
    qc_max_mt : float or None
        Maximum fraction of mitochondrial reads per spot.
    overwrite : bool
        Re-compute even if the output file already exists.
    """
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"Skipping {os.path.basename(h5ad_path)} — already computed.")
        return

    sample_name = os.path.basename(h5ad_path).replace(".h5ad", "")
    logger.info(f"[{sample_name}] Loading {h5ad_path}")
    adata, n_before, n_after = _load_expression(
        h5ad_path,
        target_sum=target_sum,
        qc_min_umis=qc_min_umis,
        qc_min_genes=qc_min_genes,
        qc_max_mt=qc_max_mt,
    )

    if n_after == 0:
        logger.warning(
            f"[{sample_name}] All {n_before} spots filtered by QC. Skipping."
        )
        return

    logger.info(
        f"[{sample_name}] Expression matrix: {n_after} spots x {adata.n_vars} genes "
        f"(CP{target_sum} + log1p, {n_before - n_after} spots removed by QC)"
    )

    pathway_dict = _load_hallmark_sets()
    total_pathways = len(pathway_dict)

    from scipy.sparse import issparse

    expr = adata.X
    if issparse(expr):
        expr = expr.toarray()
    expr = expr.astype(np.float32)
    gene_names = list(adata.var_names)

    activities, all_pathways, n_scored = _score_pathways(
        expr, gene_names, pathway_dict, min_genes=min_genes
    )

    if n_scored < min_pathways:
        logger.warning(
            f"[{sample_name}] Only {n_scored}/{total_pathways} pathways scored "
            f"(threshold: {min_pathways}). Skipping — insufficient pathway coverage."
        )
        return

    unscorable = total_pathways - n_scored
    if unscorable > 0:
        logger.warning(
            f"[{sample_name}] {unscorable} pathway(s) had fewer than {min_genes} "
            f"member genes — filled with zeros"
        )
    logger.info(f"[{sample_name}] Scored {n_scored}/{total_pathways} pathways")

    for i, pw in enumerate(all_pathways):
        col = activities[:, i]
        if col.any():
            logger.info(
                f"[{sample_name}]   {pw}: min={col.min():.3f}, "
                f"mean={col.mean():.3f}, max={col.max():.3f}"
            )

    barcodes = np.array(list(adata.obs_names), dtype="S")
    pathway_names = np.array(all_pathways, dtype="S")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("activities", data=activities, compression="gzip")
        f.create_dataset("barcodes", data=barcodes)
        f.create_dataset("pathway_names", data=pathway_names)
        # QC metadata for downstream auditing
        f.attrs["n_spots_before_qc"] = n_before
        f.attrs["n_spots_after_qc"] = n_after
        f.attrs["qc_min_umis"] = qc_min_umis or 0
        f.attrs["qc_min_genes"] = qc_min_genes or 0
        f.attrs["qc_max_mt"] = qc_max_mt or 1.0
        f.attrs["n_scored_pathways"] = n_scored

    file_size_kb = os.path.getsize(output_path) / 1024
    logger.info(
        f"[{sample_name}] Saved {activities.shape[0]} spots x {activities.shape[1]} pathways "
        f"-> {output_path} ({file_size_kb:.1f} KB)"
    )


def load_pathway_activities(
    h5_path: str,
    barcodes: list,
) -> tuple:
    """Load and barcode-align pathway activities for a sample.

    Parameters
    ----------
    h5_path : str
        Path to the .h5 file produced by this script.
    barcodes : list of bytes or str
        Ordered barcode list from the feature .pt file.  Activities are
        reordered to match this order; spots not found receive all-zero rows.

    Returns
    -------
    activities : np.ndarray, shape (N_barcodes, P), float32
        Pathway activity matrix aligned to ``barcodes``.  Missing spots are 0.
    pathway_names : list of str
        Pathway name labels.
    valid_mask : np.ndarray, shape (N_barcodes,), bool
        True for barcodes that were found in the activity file.
    """
    with h5py.File(h5_path, "r") as f:
        stored_acts = f["activities"][:]  # (M, P)
        stored_barcodes = f["barcodes"][:]  # bytes array
        pathway_names = [
            n.decode() if isinstance(n, bytes) else n for n in f["pathway_names"][:]
        ]

    # Build lookup: decoded barcode -> row index
    def _decode(b):
        return b.decode() if isinstance(b, bytes) else b

    barcode_to_row = {_decode(b): i for i, b in enumerate(stored_barcodes)}

    n = len(barcodes)
    p = stored_acts.shape[1]
    activities = np.zeros((n, p), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)

    for j, bc in enumerate(barcodes):
        key = _decode(bc)
        if key in barcode_to_row:
            activities[j] = stored_acts[barcode_to_row[key]]
            valid_mask[j] = True

    return activities, pathway_names, valid_mask


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Pre-compute Hallmark pathway activity scores for HEST samples."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root HEST data directory (contains st/ subdirectory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write .h5 files (default: <data-dir>/pathway_activities)",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Specific sample IDs to process (default: all .h5ad files in st/)",
    )
    parser.add_argument(
        "--target-sum",
        type=int,
        default=10_000,
        help="CP10k normalisation target (default: 10000)",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=5,
        help="Minimum member genes required per pathway (default: 5)",
    )
    parser.add_argument(
        "--min-pathways",
        type=int,
        default=25,
        help="Minimum scored pathways required per sample (default: 25)",
    )
    parser.add_argument(
        "--qc-min-umis",
        type=int,
        default=500,
        help="Minimum total UMI count per spot (default: 500)",
    )
    parser.add_argument(
        "--qc-min-genes",
        type=int,
        default=200,
        help="Minimum detected genes per spot (default: 200)",
    )
    parser.add_argument(
        "--qc-max-mt",
        type=float,
        default=0.15,
        help="Maximum mitochondrial read fraction per spot (default: 0.15)",
    )
    parser.add_argument(
        "--no-species-filter",
        action="store_true",
        help="Disable auto-filtering to human samples via HEST metadata",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-compute even if output already exists",
    )
    args = parser.parse_args()

    st_dir = os.path.join(args.data_dir, "st")
    output_dir = args.output_dir or os.path.join(args.data_dir, "pathway_activities")

    if not os.path.isdir(st_dir):
        raise FileNotFoundError(
            f"Expected HEST st/ directory at {st_dir}. "
            "Check --data-dir points to the root HEST data directory."
        )

    # Discover sample IDs
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        sample_ids = [f[:-5] for f in os.listdir(st_dir) if f.endswith(".h5ad")]
        sample_ids.sort()

        # Auto-filter to human samples via HEST metadata
        if not args.no_species_filter:
            metadata_path = os.path.join(args.data_dir, "HEST_v1_3_0.csv")
            if os.path.exists(metadata_path):
                df = pd.read_csv(metadata_path)
                human_ids = set(df[df["species"] == "Homo sapiens"]["id"])
                before = len(sample_ids)
                sample_ids = [s for s in sample_ids if s in human_ids]
                skipped = before - len(sample_ids)
                if skipped:
                    logger.info(
                        f"Skipped {skipped} non-human sample(s) via HEST metadata"
                    )
            else:
                logger.warning(
                    f"HEST metadata not found at {metadata_path} — "
                    f"processing all samples (use --no-species-filter to suppress)"
                )

    logger.info(
        f"Configuration: target_sum={args.target_sum}, min_genes={args.min_genes}, "
        f"min_pathways={args.min_pathways}, "
        f"qc_min_umis={args.qc_min_umis}, qc_min_genes={args.qc_min_genes}, "
        f"qc_max_mt={args.qc_max_mt}"
    )
    logger.info(f"Processing {len(sample_ids)} sample(s) -> {output_dir}")

    failed = []
    skipped_pathways = []
    for sample_id in tqdm(sample_ids, desc="Samples"):
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
        output_path = os.path.join(output_dir, f"{sample_id}.h5")

        if not os.path.exists(h5ad_path):
            logger.warning(f"Missing: {h5ad_path} — skipping")
            failed.append(sample_id)
            continue

        try:
            compute_pathway_activities_for_sample(
                h5ad_path=h5ad_path,
                output_path=output_path,
                target_sum=args.target_sum,
                min_genes=args.min_genes,
                min_pathways=args.min_pathways,
                qc_min_umis=args.qc_min_umis,
                qc_min_genes=args.qc_min_genes,
                qc_max_mt=args.qc_max_mt,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logger.error(f"Failed on {sample_id}: {e}")
            failed.append(sample_id)

    # Count outputs actually written
    written = [
        s
        for s in sample_ids
        if os.path.exists(os.path.join(output_dir, f"{s}.h5")) and s not in failed
    ]
    logger.info(
        f"Done: {len(written)}/{len(sample_ids)} samples saved, "
        f"{len(failed)} failed. Output: {output_dir}"
    )
    if failed:
        logger.warning(f"Failed samples: {failed}")


if __name__ == "__main__":
    main()
