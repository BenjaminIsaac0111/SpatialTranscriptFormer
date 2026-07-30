"""
Offline preprocessing: compute spatial pathway activity scores for HEST samples.

For each sample .h5ad, this script:
  1. Loads the raw gene expression matrix (spots x genes)
  2. Applies per-spot QC (min UMIs, min genes, max MT%) on raw counts
  3. Applies CP10k normalisation + log1p to surviving spots
  4. Computes per-pathway scores as the mean log1p CP10k expression of
     member genes (no per-slide normalisation — targets are slide-stationary)
  5. Saves the resulting activity matrix to
     <data_dir>/pathway_activities/<sample_id>.h5

Pathway scores are computed from MSigDB Hallmark gene sets (50 pathways).
The score for spot s and pathway p is the simple per-spot mean of the log1p
CP10k expression across the pathway's member genes that are present in the
sample. CP10k handles depth normalisation; no per-slide statistics enter the
score, so the same biological state in two slides yields the same target.

Non-human samples are auto-skipped via HEST metadata. Samples with
fewer than ``--min-pathways`` scored pathways are excluded.

The saved files are consumed at training time by HEST_FeatureDataset when
``pathway_targets_dir`` is provided. Files carry a ``format_version``
attribute; loaders refuse mismatched versions and ask for a recompute.

Usage::

    stf-compute-pathways --data-dir hest_data
    stf-compute-pathways --data-dir hest_data --sample-ids MEND29 TENX88
    stf-compute-pathways --data-dir hest_data --qc-max-mt 0.10 --overwrite
    stf-compute-pathways --data-dir hest_data --no-species-filter
"""

import argparse
import os
import logging
import re

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from spatial_transcript_former.data.spatial_stats import _build_knn_weights, morans_i

from spatial_transcript_former.data.pathways import (
    MSIGDB_URLS,
    download_msigdb_gmt,
    parse_gmt,
)

logger = logging.getLogger(__name__)


# File-format version stamped into each pathway-activities .h5 file.
# Bumped whenever the on-disk semantics of `activities` change.
#   v1: per-slide z-scored pathway-mean (deprecated — slide-relative drift).
#   v2: plain mean of log1p CP10k expression of pathway members.
PATHWAY_FILE_VERSION = 3


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
    total_counts = lib_sizes.ravel().astype(np.float32)
    counts = counts / lib_sizes * target_sum
    np.log1p(counts, out=counts)

    adata.X = csr_matrix(counts)

    return adata, n_before, n_after, total_counts


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


_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d+")

# Genome prefixes emitted by multi-reference CellRanger runs. The separator is
# *not* consistent across HEST: real files carry ``GRCh38_``, ``GRCh38__`` and
# even ``GRCh38______``, so match one-or-more underscores rather than a fixed
# literal. A generic fallback catches unseen genome names: two-or-more
# underscores is a reliable prefix marker, since HGNC symbols do not contain
# them.
_KNOWN_GENOME_RE = re.compile(
    r"^(?:GRCH38|GRCH37|HG19|HG38|MM10|GRCM38|WUHCOR1|SARS[-_]?COV[-_]?2|HS|MOUSE|HUMAN)_+"
)
_GENERIC_GENOME_RE = re.compile(r"^[A-Z0-9.\-]+__+")


def clean_gene_name(g):
    """Normalise a gene identifier to a bare uppercase HGNC symbol.

    Strips multi-genome CellRanger prefixes so that ``GRCh38______OR4F5``,
    ``GRCh38__OR4F5`` and ``OR4F5`` all resolve to the same symbol. Getting
    this wrong is silent: unstripped names simply fail to match any Hallmark
    member, the sample scores 0/50 pathways, and it is dropped with a warning
    while the batch summary still reports "0 failed".
    """
    g_upper = str(g).upper()
    stripped = _KNOWN_GENOME_RE.sub("", g_upper, count=1)
    if stripped == g_upper:
        stripped = _GENERIC_GENOME_RE.sub("", g_upper, count=1)
    return stripped or g_upper


# Columns HEST samples use to carry HGNC symbols when the index is Ensembl.
_SYMBOL_COLUMNS = (
    "SYMBOL",
    "symbol",
    "gene_symbol",
    "gene_name",
    "gene_symbols",
    "feature_name",
    "GeneSymbol",
)


def _resolve_gene_symbols(adata, sample_name=""):
    """Return HGNC symbols for ``adata``'s genes, whatever the index uses.

    HEST is not internally consistent: most slides index ``var`` by gene
    symbol, but some index by Ensembl gene ID and keep the symbol in a ``var``
    column. Hallmark sets are symbol-based, so an Ensembl-indexed slide matches
    **zero** pathway members, trips the ``min_pathways`` guard, and is dropped
    with only a warning — the batch summary still reports "0 failed". This
    silently shrinks the corpus, so resolve the symbols instead.
    """
    names = [str(g) for g in adata.var_names]
    head = names[: min(50, len(names))]
    if not head or sum(bool(_ENSEMBL_RE.match(g)) for g in head) / len(head) <= 0.5:
        return names  # already symbol-indexed

    for col in _SYMBOL_COLUMNS:
        if col in adata.var.columns:
            resolved = [str(x) for x in adata.var[col]]
            n_ok = sum(1 for g in resolved if g and g.lower() not in ("nan", "none"))
            if n_ok >= 0.5 * len(resolved):
                logger.info(
                    f"[{sample_name}] var_names are Ensembl IDs; using "
                    f"var['{col}'] for gene symbols ({n_ok}/{len(resolved)} resolved)."
                )
                return resolved

    logger.warning(
        f"[{sample_name}] var_names look like Ensembl IDs but no usable symbol "
        f"column was found (looked for {list(_SYMBOL_COLUMNS)}). Pathway "
        "scoring will match nothing and this sample will be skipped."
    )
    return names


def _score_pathways(expr_matrix, gene_names, pathway_dict, min_genes=5):
    """Score pathway activities as the mean log1p CP10k expression of member genes.

    The score for spot s and pathway p is the simple per-spot mean across the
    pathway's member genes that are present in ``gene_names``. This is depth-
    normalised (via the prior CP10k step) and slide-stationary by construction:
    no per-slide statistics enter the score, so the same biological state in
    two different slides yields the same target value.

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

    # Build the symbol lookup. Where a multi-genome reference maps two entries
    # onto the same cleaned symbol (e.g. human and mouse orthologs), keep the
    # first — CellRanger lists the primary genome first, so this prefers it
    # deterministically instead of silently taking whichever came last.
    gene_to_idx = {}
    for i, g in enumerate(gene_names):
        gene_to_idx.setdefault(clean_gene_name(g), i)
    n_spots = expr_matrix.shape[0]

    all_pathways = list(pathway_dict.keys())
    activities = np.zeros((n_spots, len(all_pathways)), dtype=np.float32)
    n_scored = 0

    for i, (pw_name, pw_genes) in enumerate(pathway_dict.items()):
        col_indices = [
            gene_to_idx[clean_gene_name(g)]
            for g in pw_genes
            if clean_gene_name(g) in gene_to_idx
        ]
        if len(col_indices) < min_genes:
            continue
        activities[:, i] = expr_matrix[:, col_indices].mean(axis=1)
        n_scored += 1

    return activities, all_pathways, n_scored


def _compute_pathway_morans_i(
    activities: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> np.ndarray:
    """Compute Moran's I for each pathway across spots.

    Parameters
    ----------
    activities : np.ndarray, shape (n_spots, n_pathways)
        Pathway activity matrix (output of ``_score_pathways``).
    coords : np.ndarray, shape (n_spots, 2)
        Spatial coordinates for each spot.
    k : int
        Number of nearest neighbours for the spatial weight graph.

    Returns
    -------
    morans : np.ndarray, shape (n_pathways,), float32
        Per-pathway Moran's I scores.  Values are clipped to [0, inf)
        so that only positively autocorrelated pathways receive weight;
        negatively autocorrelated or random pathways get weight 0.
    """
    n_spots, n_pathways = activities.shape
    if n_spots < k + 1:
        return np.zeros(n_pathways, dtype=np.float32)

    W = _build_knn_weights(coords, k=k)
    scores = np.empty(n_pathways, dtype=np.float32)
    for p in range(n_pathways):
        scores[p] = morans_i(activities[:, p], W)

    # Clip negative values — only positive spatial autocorrelation is useful
    np.clip(scores, 0.0, None, out=scores)
    return scores


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
    adata, n_before, n_after, total_counts = _load_expression(
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
    gene_names = _resolve_gene_symbols(adata, sample_name)

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

    # Only log individual pathway stats if verbose is enabled
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        for i, pw in enumerate(all_pathways):
            col = activities[:, i]
            if col.any():
                logger.debug(
                    f"[{sample_name}]   {pw}: min={col.min():.3f}, "
                    f"mean={col.mean():.3f}, max={col.max():.3f}"
                )

    # Compute per-pathway Moran's I spatial autocorrelation weights
    coords = (
        np.column_stack(
            [adata.obs["array_row"].values, adata.obs["array_col"].values]
        ).astype(np.float64)
        if "array_row" in adata.obs.columns
        else None
    )

    # Fallback: use obsm spatial coordinates if array_row/col not available
    if coords is None:
        for key in ["spatial", "X_spatial"]:
            if key in adata.obsm:
                coords = np.array(adata.obsm[key], dtype=np.float64)[:, :2]
                break

    pathway_morans = None
    if coords is not None and len(coords) >= 7:  # need k+1 spots minimum
        pathway_morans = _compute_pathway_morans_i(activities, coords, k=6)
        logger.info(
            f"[{sample_name}] Pathway Moran's I: "
            f"min={pathway_morans.min():.3f}, mean={pathway_morans.mean():.3f}, "
            f"max={pathway_morans.max():.3f}"
        )
    else:
        logger.warning(
            f"[{sample_name}] Could not compute pathway Moran's I "
            f"(no spatial coordinates or too few spots)"
        )

    barcodes = np.array(list(adata.obs_names), dtype="S")
    pathway_names = np.array(all_pathways, dtype="S")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("activities", data=activities, compression="gzip")
        f.create_dataset("barcodes", data=barcodes)
        f.create_dataset("pathway_names", data=pathway_names)
        if pathway_morans is not None:
            f.create_dataset("pathway_morans_i", data=pathway_morans)
        # Per-spot library size (total UMIs before CP10k). Stored because the
        # score correlates with it at |r|~0.93 -- it is the dominant confound,
        # and downstream consumers need it to regress the confound out.
        f.create_dataset("total_counts", data=total_counts)
        # File-format version — bumped when the semantics of `activities` change.
        f.attrs["format_version"] = PATHWAY_FILE_VERSION
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
    pathway_morans_i : np.ndarray or None, shape (P,), float32
        Per-pathway Moran's I diagnostic.  ``None`` if the field is absent.

    Raises
    ------
    ValueError
        If the file's ``format_version`` attribute is missing or does not
        match :data:`PATHWAY_FILE_VERSION`. Re-run ``stf-compute-pathways
        --overwrite`` to regenerate the file with the current schema.
    """
    with h5py.File(h5_path, "r") as f:
        version = f.attrs.get("format_version", None)
        if version is None or int(version) != PATHWAY_FILE_VERSION:
            raise ValueError(
                f"Pathway file {h5_path!r} has format_version="
                f"{version!r}, but this build expects "
                f"{PATHWAY_FILE_VERSION}. Re-run "
                "`stf-compute-pathways --overwrite` to regenerate."
            )
        stored_acts = f["activities"][:]  # (M, P)
        stored_barcodes = f["barcodes"][:]  # bytes array
        pathway_names = [
            n.decode() if isinstance(n, bytes) else n for n in f["pathway_names"][:]
        ]
        pathway_morans_i = (
            f["pathway_morans_i"][:].astype(np.float32)
            if "pathway_morans_i" in f
            else None
        )

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

    return activities, pathway_names, valid_mask, pathway_morans_i


def load_spot_depth(h5_path: str, barcodes: list) -> np.ndarray:
    """Load per-spot library size, aligned to ``barcodes``.

    Kept separate from :func:`load_pathway_activities` so that function's
    4-tuple return stays source-compatible with existing callers.

    Returns
    -------
    np.ndarray, shape (len(barcodes),), float32
        Total UMI counts per spot; ``0.0`` for barcodes absent from the file.
        Returns all-zeros if the file predates ``total_counts`` storage.
    """
    with h5py.File(h5_path, "r") as f:
        if "total_counts" not in f:
            return np.zeros(len(barcodes), dtype=np.float32)
        stored = f["total_counts"][:].astype(np.float32)
        stored_barcodes = f["barcodes"][:]

    def _decode(b):
        return b.decode() if isinstance(b, bytes) else b

    row_of = {_decode(b): i for i, b in enumerate(stored_barcodes)}
    out = np.zeros(len(barcodes), dtype=np.float32)
    for j, bc in enumerate(barcodes):
        i = row_of.get(_decode(bc))
        if i is not None:
            out[j] = stored[i]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Hallmark pathway activity scores for HEST samples."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root HEST data directory (contains st/ subdirectory and HEST_v1_3_0.csv)",
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging for every pathway score",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    st_dir = os.path.join(args.data_dir, "st")
    output_dir = args.output_dir or os.path.join(args.data_dir, "pathway_activities")

    if not os.path.isdir(st_dir):
        raise FileNotFoundError(
            f"\n\n[ERROR] Could not find the Spatial Transcriptomics data directory at:\n  {st_dir}\n\n"
            "The --data-dir should point to the root HEST directory that contains:\n"
            "  - st/                   (directory with .h5ad files)\n"
            "  - HEST_v1_3_0.csv       (metadata file)\n\n"
            f"You provided: --data-dir {args.data_dir}\n"
            "If your .h5ad files are elsewhere, please ensure the directory is named 'st'."
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

    processed = 0
    skipped_existing = 0
    failed = []

    for sample_id in tqdm(sample_ids, desc="Samples"):
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
        output_path = os.path.join(output_dir, f"{sample_id}.h5")

        if os.path.exists(output_path) and not args.overwrite:
            skipped_existing += 1
            if args.verbose:
                logger.debug(f"Skipping {sample_id} — already exists.")
            continue

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
            processed += 1
        except Exception as e:
            logger.error(f"Failed on {sample_id}: {e}")
            failed.append(sample_id)

    logger.info(
        f"Done: {processed} samples processed, {skipped_existing} skipped (existing), "
        f"{len(failed)} failed. Total samples: {len(sample_ids)}"
    )
    logger.info(f"Output directory: {output_dir}")
    if failed:
        logger.warning(f"Failed samples list: {failed}")


if __name__ == "__main__":
    main()
