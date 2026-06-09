"""
Exploratory analysis of Spatially Variable Genes/Pathways (SVGs).

Reads .h5ad files from the HEST data directory, computes pathway activity
scores and Moran's I spatial autocorrelation, then produces a summary
report showing which pathways exhibit genuine spatial structure vs noise.

This is a pure analysis tool — it does not modify any training data or
model configuration. Use the output to make informed decisions about
which pathways to include as training targets.

Usage::

    python scripts/analyze_svg.py --data-dir hest_data
    python scripts/analyze_svg.py --data-dir hest_data --sample-ids TENX175
    python scripts/analyze_svg.py --data-dir hest_data --threshold 0.05
    python scripts/analyze_svg.py --data-dir hest_data --plot
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
    _load_expression,
    _load_hallmark_sets,
    _score_pathways,
    _compute_pathway_morans_i,
)

logger = logging.getLogger(__name__)


def analyze_sample(h5ad_path, pathway_dict, min_genes=5, k=6):
    """Compute pathway activities and Moran's I for a single sample.

    Returns
    -------
    sample_id : str
    pathway_names : list of str
    morans : np.ndarray, shape (n_pathways,)
    n_spots : int
    n_scored : int
    """
    sample_id = os.path.basename(h5ad_path).replace(".h5ad", "")

    adata, n_before, n_after = _load_expression(
        h5ad_path,
        target_sum=10_000,
        qc_min_umis=500,
        qc_min_genes=200,
        qc_max_mt=0.15,
    )

    if n_after == 0:
        logger.warning(f"[{sample_id}] All spots filtered by QC. Skipping.")
        return None

    from scipy.sparse import issparse

    expr = adata.X
    if issparse(expr):
        expr = expr.toarray()
    expr = expr.astype(np.float32)
    gene_names = list(adata.var_names)

    # Strip common prefixes (GRCh38_, GRCm38_) often found in HEST/10x data
    stripped_names = []
    for g in gene_names:
        if "_" in g and (g.startswith("GRCh38_") or g.startswith("GRCm38_")):
            stripped_names.append(g.split("_", 1)[1])
        else:
            stripped_names.append(g)
    gene_names = stripped_names

    # Quick check for Hallmark coverage
    all_hallmark_genes = set()
    for genes in pathway_dict.values():
        all_hallmark_genes.update(genes)

    overlap = set(gene_names) & all_hallmark_genes
    if len(overlap) < 100:
        logger.warning(
            f"[{sample_id}] Insufficient Hallmark gene overlap ({len(overlap)}). Skipping (likely non-human or low-density panel)."
        )
        return None

    activities, all_pathways, n_scored = _score_pathways(
        expr, gene_names, pathway_dict, min_genes=min_genes
    )

    # Get spatial coordinates
    coords = (
        np.column_stack(
            [adata.obs["array_row"].values, adata.obs["array_col"].values]
        ).astype(np.float64)
        if "array_row" in adata.obs.columns
        else None
    )
    if coords is None:
        for key in ["spatial", "X_spatial"]:
            if key in adata.obsm:
                coords = np.array(adata.obsm[key], dtype=np.float64)[:, :2]
                break

    if coords is None or len(coords) < k + 1:
        logger.warning(f"[{sample_id}] No spatial coordinates. Skipping.")
        return None

    morans = _compute_pathway_morans_i(activities, coords, k=k)

    # Also compute per-pathway variance (to detect zero-variance pathways)
    variances = activities.var(axis=0)

    return {
        "sample_id": sample_id,
        "pathway_names": all_pathways,
        "activities": activities,  # (n_spots, n_pathways)
        "morans": morans,
        "variances": variances,
        "n_spots": n_after,
        "n_scored": n_scored,
    }


def print_report(results, threshold=None, pathways_filter=None):
    """Print a formatted SVG analysis report."""

    # Build a DataFrame: rows = pathways, columns = samples
    all_pathways = results[0]["pathway_names"]
    n_pathways = len(all_pathways)
    n_samples = len(results)

    morans_matrix = np.zeros((n_samples, n_pathways), dtype=np.float32)
    var_matrix = np.zeros((n_samples, n_pathways), dtype=np.float32)
    sample_ids = []

    for i, r in enumerate(results):
        morans_matrix[i] = r["morans"]
        var_matrix[i] = r["variances"]
        sample_ids.append(r["sample_id"])

    # Summary statistics per pathway
    df = pd.DataFrame(
        {
            "Pathway": all_pathways,
            "Mean_I": morans_matrix.mean(axis=0),
            "Median_I": np.median(morans_matrix, axis=0),
            "Std_I": morans_matrix.std(axis=0),
            "Min_I": morans_matrix.min(axis=0),
            "Max_I": morans_matrix.max(axis=0),
            "Mean_Var": var_matrix.mean(axis=0),
        }
    )

    # Count how many samples each pathway is above threshold
    if threshold is not None:
        df["Samples_Above"] = (morans_matrix >= threshold).sum(axis=0)
        df["Pct_Above"] = df["Samples_Above"] / n_samples * 100

    # Sort by mean Moran's I descending
    df = df.sort_values("Mean_I", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-indexed rank

    # Apply pathway filter if specified
    if pathways_filter:
        filter_set = set(pathways_filter)
        df["In_Filter"] = df["Pathway"].isin(filter_set)

    # --- Print Header ---
    print()
    print("=" * 90)
    print(f"  SVG ANALYSIS REPORT")
    print(f"  Samples: {n_samples}  |  Pathways: {n_pathways}")
    for r in results:
        print(
            f"    {r['sample_id']}: {r['n_spots']} spots, {r['n_scored']} pathways scored"
        )
    print("=" * 90)

    # --- Print Table ---
    print()
    shorten = lambda s, n=42: s[: n - 2] + ".." if len(s) > n else s

    header = f"{'Rank':>4}  {'Pathway':<44}  {'Mean':>6}  {'Med':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}  {'Var':>8}"
    if threshold is not None:
        header += f"  {'Above':>8}"
    if pathways_filter:
        header += f"  {'CRC':>3}"
    print(header)
    print("─" * len(header))

    for rank, row in df.iterrows():
        line = (
            f"{rank:>4}  {shorten(row['Pathway']):<44}  "
            f"{row['Mean_I']:>6.3f}  {row['Median_I']:>6.3f}  "
            f"{row['Std_I']:>6.3f}  {row['Min_I']:>6.3f}  "
            f"{row['Max_I']:>6.3f}  {row['Mean_Var']:>8.4f}"
        )
        if threshold is not None:
            line += f"  {int(row['Samples_Above']):>3}/{n_samples}"
        if pathways_filter:
            marker = " ✓" if row.get("In_Filter", False) else "  "
            line += f"  {marker}"
        print(line)

    # --- Threshold Summary ---
    if threshold is not None:
        print()
        svg_df = df[df["Mean_I"] >= threshold]
        non_svg = df[df["Mean_I"] < threshold]

        print(f"Pathways with mean Moran's I ≥ {threshold}: {len(svg_df)} / {len(df)}")
        print(f"Pathways below threshold (noise): {len(non_svg)} / {len(df)}")
        print()

        if len(svg_df) > 0:
            print("# Recommended SVG pathway list (copy-paste into run_preset.py):")
            print("SVG_PATHWAYS = [")
            for _, row in svg_df.iterrows():
                print(f'    "{row["Pathway"]}",')
            print("]")

    # --- CRC Filter Summary ---
    if pathways_filter:
        print()
        crc_df = df[df["In_Filter"]]
        print(f"CRC pathways in filter: {len(crc_df)} / {len(pathways_filter)}")
        if threshold is not None:
            crc_svg = crc_df[crc_df["Mean_I"] >= threshold]
            crc_noise = crc_df[crc_df["Mean_I"] < threshold]
            print(f"  Above threshold: {len(crc_svg)}")
            print(f"  Below threshold (noise): {len(crc_noise)}")
            if len(crc_noise) > 0:
                print(f"  Noisy CRC pathways:")
                for _, row in crc_noise.iterrows():
                    print(f"    - {row['Pathway']} (Mean I = {row['Mean_I']:.3f})")

    print()
    return df


def plot_svg_analysis(df, results, output_dir=None):
    """Generate exploratory plots of SVG statistics."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plots.")
        return

    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Spatially Variable Pathway Analysis", fontsize=14, fontweight="bold")

    # 1. Ranked bar chart of mean Moran's I
    ax = axes[0, 0]
    colors = ["#2ecc71" if v >= 0.05 else "#e74c3c" for v in df["Mean_I"]]
    ax.barh(range(len(df)), df["Mean_I"].values, color=colors, height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([p[:30] for p in df["Pathway"]], fontsize=6)
    ax.set_xlabel("Mean Moran's I")
    ax.set_title("Pathway Spatial Autocorrelation (ranked)")
    ax.axvline(x=0.05, color="gray", linestyle="--", alpha=0.7, label="Threshold=0.05")
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    # 2. Histogram of all Moran's I values
    ax = axes[0, 1]
    all_morans = np.concatenate([r["morans"] for r in results])
    ax.hist(all_morans, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(x=0.05, color="red", linestyle="--", label="Threshold=0.05")
    ax.set_xlabel("Moran's I")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Moran's I (all pathways, all samples)")
    ax.legend()

    # 3. Moran's I vs Expression Variance
    ax = axes[1, 0]
    ax.scatter(
        df["Mean_Var"], df["Mean_I"], c=colors, s=40, alpha=0.8, edgecolors="white"
    )
    ax.set_xlabel("Mean Expression Variance")
    ax.set_ylabel("Mean Moran's I")
    ax.set_title("Spatial Structure vs Expression Variability")
    for _, row in df.head(5).iterrows():
        ax.annotate(
            row["Pathway"][:25],
            (row["Mean_Var"], row["Mean_I"]),
            fontsize=6,
            alpha=0.7,
        )

    # 4. Per-sample heatmap (if multiple samples)
    ax = axes[1, 1]
    if len(results) > 1:
        morans_matrix = np.array([r["morans"] for r in results])
        # Sort columns by mean Moran's I (same as df order)
        sort_idx = np.argsort(
            [np.mean(morans_matrix[:, i]) for i in range(morans_matrix.shape[1])]
        )[::-1]
        im = ax.imshow(morans_matrix[:, sort_idx], aspect="auto", cmap="RdYlGn")
        ax.set_xlabel("Pathway (ranked)")
        ax.set_ylabel("Sample")
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels([r["sample_id"][:12] for r in results], fontsize=7)
        ax.set_title("Moran's I Heatmap (samples × pathways)")
        plt.colorbar(im, ax=ax, label="Moran's I")
    else:
        # Single sample: show bar chart per pathway
        r = results[0]
        sort_idx = np.argsort(r["morans"])[::-1]
        pathway_names = [r["pathway_names"][i][:25] for i in sort_idx]
        values = r["morans"][sort_idx]
        bar_colors = ["#2ecc71" if v >= 0.05 else "#e74c3c" for v in values]
        ax.bar(range(len(values)), values, color=bar_colors, width=0.8)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(pathway_names, rotation=90, fontsize=5)
        ax.set_ylabel("Moran's I")
        ax.set_title(f"Per-Pathway Moran's I ({r['sample_id']})")
        ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "svg_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.close()


def plot_correlation_heatmap(results, output_dir=None, crc_pathways=None):
    """Generate a pathway-pathway correlation heatmap from pooled spot activities.

    Concatenates spot-level activities across all samples, computes pairwise
    Pearson correlations between pathways, and plots a clustered heatmap.
    CRC pathways are highlighted and retained/dropped status is annotated.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
        from scipy.spatial.distance import squareform
    except ImportError:
        print("matplotlib/scipy not installed. Skipping correlation plot.")
        return

    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # Pool spot-level activities across all samples
    pathway_names = results[0]["pathway_names"]
    pooled = np.concatenate([r["activities"] for r in results], axis=0)
    n_total_spots = pooled.shape[0]

    # Compute Pearson correlation matrix
    # (columns = pathways, so corr across rows = spots)
    corr = np.corrcoef(pooled.T)  # (n_pathways, n_pathways)
    np.fill_diagonal(corr, 1.0)
    corr = np.nan_to_num(corr, nan=0.0)  # zero-variance pathways

    # Hierarchical clustering for ordering
    dist = 1 - corr
    dist = np.clip(dist, 0, 2)  # ensure valid distances
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    corr_ordered = corr[np.ix_(order, order)]
    names_ordered = [pathway_names[i] for i in order]

    # Determine CRC status for each pathway
    crc_retained = set()
    crc_dropped = set()
    if crc_pathways:
        # The dropped CRC pathways
        dropped = {
            "HALLMARK_TGF_BETA_SIGNALING",
            "HALLMARK_ANGIOGENESIS",
            "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
            "HALLMARK_KRAS_SIGNALING_DN",
        }
        crc_retained = set(crc_pathways) - dropped
        crc_dropped = set(crc_pathways) & dropped

    # Shorten pathway names for labels
    def short_name(name):
        return name.replace("HALLMARK_", "").replace("_", " ").title()[:30]

    # Color-code labels
    label_colors = []
    for name in names_ordered:
        if name in crc_retained:
            label_colors.append("#2ecc71")  # green = retained CRC
        elif name in crc_dropped:
            label_colors.append("#e74c3c")  # red = dropped CRC
        else:
            label_colors.append("#666666")  # grey = non-CRC

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr_ordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(names_ordered)))
    ax.set_yticks(range(len(names_ordered)))
    ax.set_xticklabels([short_name(n) for n in names_ordered], rotation=90, fontsize=7)
    ax.set_yticklabels([short_name(n) for n in names_ordered], fontsize=7)

    # Colour the tick labels
    for i, (xtick, ytick) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
        xtick.set_color(label_colors[i])
        ytick.set_color(label_colors[i])

    ax.set_title(
        f"Pathway Activity Correlations (Pearson, {n_total_spots:,} pooled spots from {len(results)} samples)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

    # Legend
    legend_elements = [
        Patch(facecolor="#2ecc71", label="CRC — Retained (spatial)"),
        Patch(facecolor="#e74c3c", label="CRC — Dropped (non-spatial)"),
        Patch(facecolor="#666666", label="Non-CRC pathway"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pathway_correlations.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Correlation plot saved to: {plot_path}")
    plt.close()

    # --- Print correlation summary for dropped CRC pathways ---
    if crc_dropped:
        print("\nCorrelations between DROPPED CRC pathways and RETAINED CRC pathways:")
        print(f"{'Dropped Pathway':<44} {'Retained Pathway':<44} {'Pearson r':>9}")
        print("─" * 100)
        for dropped_name in sorted(crc_dropped):
            if dropped_name not in pathway_names:
                continue
            di = pathway_names.index(dropped_name)
            pairs = []
            for retained_name in sorted(crc_retained):
                if retained_name not in pathway_names:
                    continue
                ri = pathway_names.index(retained_name)
                r_val = corr[di, ri]
                pairs.append((retained_name, r_val))
            # Sort by absolute correlation
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            for retained_name, r_val in pairs[:5]:  # top 5
                short_d = dropped_name.replace("HALLMARK_", "")
                short_r = retained_name.replace("HALLMARK_", "")
                print(f"  {short_d:<42} {short_r:<42} {r_val:>+9.3f}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory analysis of Spatially Variable Pathways (SVGs)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root HEST data directory (contains st/ subdirectory)",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Specific sample IDs to analyze (default: all .h5ad in st/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Moran's I threshold to classify SVGs (default: 0.05)",
    )
    parser.add_argument(
        "--crc",
        action="store_true",
        help="Highlight CRC-specific pathways in the report",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate exploratory plots (saved to --output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for plot output (default: <data-dir>/svg_analysis)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of spatial neighbours for Moran's I (default: 6)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    st_dir = os.path.join(args.data_dir, "st")
    if not os.path.isdir(st_dir):
        print(f"Error: Could not find st/ directory in {args.data_dir}")
        sys.exit(1)

    # Discover samples
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        sample_ids = [f[:-5] for f in os.listdir(st_dir) if f.endswith(".h5ad")]
        sample_ids.sort()

    if not sample_ids:
        print(f"No .h5ad files found in {st_dir}")
        sys.exit(1)

    print(f"Loading {len(sample_ids)} sample(s)...")
    pathway_dict = _load_hallmark_sets(cache_dir=os.path.join(args.data_dir, ".cache"))

    results = []
    for sample_id in sample_ids:
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
        if not os.path.exists(h5ad_path):
            logger.warning(f"Missing: {h5ad_path}")
            continue

        print(f"  Analyzing {sample_id}...", end=" ", flush=True)
        result = analyze_sample(h5ad_path, pathway_dict, k=args.k)
        if result is not None:
            results.append(result)
            print(f"OK ({result['n_spots']} spots, {result['n_scored']} pathways)")
        else:
            print("SKIPPED")

    if not results:
        print("No samples could be analyzed.")
        sys.exit(1)

    # CRC pathway list for filtering
    crc_pathways = None
    if args.crc:
        crc_pathways = [
            "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
            "HALLMARK_TGF_BETA_SIGNALING",
            "HALLMARK_KRAS_SIGNALING_UP",
            "HALLMARK_KRAS_SIGNALING_DN",
            "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
            "HALLMARK_ANGIOGENESIS",
            "HALLMARK_APICAL_JUNCTION",
            "HALLMARK_INFLAMMATORY_RESPONSE",
            "HALLMARK_IL6_JAK_STAT3_SIGNALING",
            "HALLMARK_APOPTOSIS",
            "HALLMARK_P53_PATHWAY",
            "HALLMARK_DNA_REPAIR",
            "HALLMARK_HYPOXIA",
        ]

    df = print_report(results, threshold=args.threshold, pathways_filter=crc_pathways)

    # Save to CSV file for easy viewing
    output_dir = args.output_dir or os.path.join(args.data_dir, "svg_analysis")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "svg_report.csv")
    try:
        df.to_csv(report_path, index=True)
        print(f"Report saved to: {report_path}")
    except PermissionError:
        # File might be open in another app
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = os.path.join(output_dir, f"svg_report_{ts}.csv")
        df.to_csv(alt_path, index=True)
        print(f"Report saved to: {alt_path}  (original was locked)")

    if args.plot:
        output_dir = args.output_dir or os.path.join(args.data_dir, "svg_analysis")
        plot_svg_analysis(df, results, output_dir=output_dir)
        plot_correlation_heatmap(
            results, output_dir=output_dir, crc_pathways=crc_pathways
        )


if __name__ == "__main__":
    main()
