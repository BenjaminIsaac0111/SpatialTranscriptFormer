"""
MSigDB Hallmarks pathway utilities.

Downloads, parses, and converts MSigDB Hallmark gene sets into
a pathway membership matrix for initializing the SpatialTranscriptFormer.
"""

import json
import os
import torch
import urllib.request
from typing import Dict, List, Optional

# MSigDB Hallmarks GMT URL (v2024.1.Hs, gene symbols)
HALLMARKS_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"
HALLMARKS_FILENAME = "h.all.v2024.1.Hs.symbols.gmt"


def download_hallmarks_gmt(cache_dir: str = ".cache") -> str:
    """
    Download the MSigDB Hallmarks GMT file if not already cached.

    Returns:
        str: Path to the local GMT file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, HALLMARKS_FILENAME)

    if not os.path.exists(local_path):
        print(f"Downloading MSigDB Hallmarks from {HALLMARKS_URL}...")
        urllib.request.urlretrieve(HALLMARKS_URL, local_path)
        print(f"Saved to {local_path}")
    
    return local_path


def parse_gmt(gmt_path: str) -> Dict[str, List[str]]:
    """
    Parse a GMT file into a dict of {pathway_name: [gene_symbols]}.

    GMT format: each line is tab-separated:
        pathway_name \\t description \\t gene1 \\t gene2 \\t ...
    """
    pathways = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = parts[2:]  # Skip description at index 1
            pathways[name] = genes
    return pathways


def build_membership_matrix(
    pathway_dict: Dict[str, List[str]],
    gene_list: List[str],
    scale: float = 1.0
) -> torch.Tensor:
    """
    Build a binary membership matrix (num_pathways x num_genes).

    Args:
        pathway_dict: {pathway_name: [gene_symbols]} from parse_gmt().
        gene_list: Ordered list of gene symbols (e.g., from global_genes.json).
        scale: Value for member genes (default 1.0). Non-members are 0.

    Returns:
        torch.Tensor: Shape (num_pathways, num_genes).
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    num_pathways = len(pathway_dict)
    num_genes = len(gene_list)

    matrix = torch.zeros(num_pathways, num_genes)

    pathway_names = []
    for p_idx, (name, genes) in enumerate(pathway_dict.items()):
        pathway_names.append(name)
        matched = 0
        for gene in genes:
            if gene in gene_to_idx:
                matrix[p_idx, gene_to_idx[gene]] = scale
                matched += 1

    return matrix, pathway_names


def get_pathway_init(
    gene_list: List[str],
    cache_dir: str = ".cache",
    verbose: bool = True
) -> tuple:
    """
    Main entry point: download Hallmarks, match to gene list, return init matrix.

    Args:
        gene_list: Ordered list of gene symbols from global_genes.json.
        cache_dir: Directory to cache the downloaded GMT file.
        verbose: Print pathway coverage statistics.

    Returns:
        tuple: (membership_matrix [Tensor (50, G)], pathway_names [list of str])
    """
    gmt_path = download_hallmarks_gmt(cache_dir)
    pathway_dict = parse_gmt(gmt_path)

    matrix, pathway_names = build_membership_matrix(pathway_dict, gene_list)

    if verbose:
        total_genes = len(gene_list)
        covered = (matrix.sum(dim=0) > 0).sum().item()
        print(f"MSigDB Hallmarks: {len(pathway_names)} pathways")
        print(f"Gene coverage: {covered}/{total_genes} ({100*covered/total_genes:.1f}%)")
        for i, name in enumerate(pathway_names):
            n_matched = int(matrix[i].sum().item())
            short_name = name.replace("HALLMARK_", "")
            if verbose and n_matched > 0:
                print(f"  {short_name}: {n_matched} genes")

    return matrix, pathway_names
