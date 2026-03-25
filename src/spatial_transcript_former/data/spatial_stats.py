"""
Spatial statistics utilities for gene selection.

Provides lightweight, dependency-free Moran's I computation for
identifying spatially variable genes (SVGs) from spatial
transcriptomics data.

Moran's I measures spatial autocorrelation: whether nearby spots tend
to have similar (positive I) or dissimilar (negative I) expression
for a given gene. Genes with high Moran's I show distinct spatial
patterns and are the strongest learning targets for
SpatialTranscriptFormer.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix


def _build_knn_weights(coords: np.ndarray, k: int = 6) -> csr_matrix:
    """Build a row-normalised KNN spatial weight matrix.

    Args:
        coords: (N, 2) array of spatial coordinates.
        k: Number of nearest neighbours per spot.

    Returns:
        (N, N) sparse CSR matrix where ``W[i, j] = 1/k`` if j is one
        of the k nearest neighbours of i, else 0. Row-normalisation
        ensures that the weight contribution is independent of local
        spot density.
    """
    n = coords.shape[0]
    tree = KDTree(coords)
    # k+1 because the first neighbour returned is the point itself
    _, indices = tree.query(coords, k=min(k + 1, n))

    rows = []
    cols = []
    for i in range(n):
        neighbours = indices[i]
        neighbours = neighbours[neighbours != i][:k]
        for j in neighbours:
            rows.append(i)
            cols.append(j)

    data = np.ones(len(rows), dtype=np.float64) / k
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    return W


def morans_i(x: np.ndarray, W: csr_matrix) -> float:
    """Compute Moran's I for a single variable.

    .. math::

        I = \\frac{N}{W_{sum}} \\cdot
            \\frac{\\sum_i \\sum_j w_{ij} (x_i - \\bar{x})(x_j - \\bar{x})}
                  {\\sum_i (x_i - \\bar{x})^2}

    Args:
        x: (N,) array of values (e.g. gene expression per spot).
        W: (N, N) sparse spatial weight matrix.

    Returns:
        Moran's I statistic. Ranges roughly from -1 (perfect
        dispersion) through 0 (random) to +1 (perfect clustering).
        Returns 0.0 if variance is zero (constant gene).
    """
    n = len(x)
    x_mean = x.mean()
    z = x - x_mean

    denominator = np.sum(z**2)
    if denominator < 1e-12:
        return 0.0  # Constant expression → no spatial pattern

    # W @ z gives the spatially-lagged deviation for each spot
    lag = W.dot(z)
    numerator = np.sum(z * lag)

    W_sum = W.sum()
    if W_sum < 1e-12:
        return 0.0

    I = (n / W_sum) * (numerator / denominator)
    return float(I)


def morans_i_batch(
    expression: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> np.ndarray:
    """Compute Moran's I for all genes in an expression matrix.

    Args:
        expression: (N, G) dense expression matrix (spots × genes).
        coords: (N, 2) spatial coordinates for each spot.
        k: Number of nearest neighbours for the spatial weight graph.

    Returns:
        (G,) array of Moran's I scores, one per gene.
    """
    if expression.shape[0] < k + 1:
        # Too few spots to build a meaningful KNN graph
        return np.zeros(expression.shape[1], dtype=np.float64)

    W = _build_knn_weights(coords, k=k)
    n_genes = expression.shape[1]
    scores = np.empty(n_genes, dtype=np.float64)

    for g in range(n_genes):
        scores[g] = morans_i(expression[:, g], W)

    return scores


def spatial_coherence_score(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
    top_k_genes: int = 50,
) -> float:
    """Compare spatial structure of predictions vs ground truth.

    Computes Moran's I for both the predicted and ground-truth
    expression matrices, then returns the Pearson correlation between
    the two Moran's I vectors. A score near 1.0 means the model
    reproduces the correct spatial patterns; near 0 means random.

    To keep computation fast (this runs every validation epoch), only
    the ``top_k_genes`` with highest ground-truth spatial variability
    are evaluated.

    Args:
        predicted: (N, G) predicted expression matrix.
        ground_truth: (N, G) ground-truth expression matrix.
        coords: (N, 2) spatial coordinates.
        k: KNN neighbours for the spatial weight graph.
        top_k_genes: Number of top-Moran's-I genes to evaluate.

    Returns:
        Pearson correlation between predicted and ground-truth
        Moran's I vectors. Returns 0.0 if computation fails.
    """
    n_spots, n_genes = ground_truth.shape
    if n_spots < k + 1 or n_genes < 2:
        return 0.0

    W = _build_knn_weights(coords, k=k)

    # Compute Moran's I for ground truth
    mi_gt = np.empty(n_genes, dtype=np.float64)
    for g in range(n_genes):
        mi_gt[g] = morans_i(ground_truth[:, g], W)

    # Select top-K genes by ground-truth Moran's I (most spatially variable)
    top_indices = np.argsort(mi_gt)[-top_k_genes:]

    # Compute Moran's I for predictions on those genes only
    mi_pred = np.empty(len(top_indices), dtype=np.float64)
    mi_gt_top = mi_gt[top_indices]
    for i, g in enumerate(top_indices):
        mi_pred[i] = morans_i(predicted[:, g], W)

    # Pearson correlation between the two Moran's I vectors
    if np.std(mi_gt_top) < 1e-12 or np.std(mi_pred) < 1e-12:
        return 0.0

    corr = np.corrcoef(mi_gt_top, mi_pred)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0
