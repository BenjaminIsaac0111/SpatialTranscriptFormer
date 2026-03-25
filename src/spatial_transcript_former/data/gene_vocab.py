"""
Gene vocabulary management for SpatialTranscriptFormer.

Provides :class:`GeneVocab` — the single source of truth for loading,
validating, and accessing the global gene list used across training,
inference, and visualization.
"""

import json
import warnings
from typing import List, Optional

from .paths import resolve_gene_vocab_path


class GeneVocab:
    """Immutable gene vocabulary loaded from ``global_genes.json``.

    Consolidates the gene-loading logic previously duplicated in
    ``train.py``, ``builder.py``, ``dataset.py``, and ``visualization.py``.

    Example::

        vocab = GeneVocab.from_json("A:/hest_data", num_genes=1000)
        print(vocab.num_genes)   # 1000
        print(vocab.genes[:5])   # ['TP53', 'EGFR', ...]
    """

    def __init__(self, genes: List[str]):
        """
        Args:
            genes: Ordered list of gene symbols.
        """
        if not genes:
            raise ValueError("Cannot create GeneVocab with an empty gene list.")
        self._genes = list(genes)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        *search_dirs: str,
        num_genes: int = 1000,
    ) -> "GeneVocab":
        """Load a gene vocabulary from ``global_genes.json``.

        Searches the provided directories (plus the project root and cwd)
        for the file, then truncates the list to ``num_genes``.

        Args:
            *search_dirs: Directories to search for ``global_genes.json``.
            num_genes: Maximum number of genes to retain.

        Returns:
            A new :class:`GeneVocab` instance.
        """
        path = resolve_gene_vocab_path(*search_dirs)

        with open(path, "r") as f:
            all_genes = json.load(f)

        if not isinstance(all_genes, list):
            raise RuntimeError(
                f"Expected a JSON list in {path}, got {type(all_genes).__name__}"
            )

        truncated = all_genes[:num_genes]
        actual = len(truncated)
        if actual < num_genes:
            warnings.warn(
                f"Requested {num_genes} genes but only {actual} available in {path}."
            )

        print(f"GeneVocab: loaded {actual} genes from {path}")
        return cls(truncated)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def genes(self) -> List[str]:
        """Ordered list of gene symbols."""
        return list(self._genes)

    @property
    def num_genes(self) -> int:
        """Number of genes in the vocabulary."""
        return len(self._genes)

    def __len__(self) -> int:
        return len(self._genes)

    def __contains__(self, gene: str) -> bool:
        return gene in self._genes

    def __repr__(self) -> str:
        return f"GeneVocab(num_genes={self.num_genes})"
