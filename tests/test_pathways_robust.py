import torch
import pytest
from spatial_transcript_former.data.pathways import build_membership_matrix


def test_build_membership_matrix_integrity():
    """Verify that the membership matrix correctly maps genes to pathways."""
    pathway_dict = {
        "PATHWAY_A": ["GENE_1", "GENE_2"],
        "PATHWAY_B": ["GENE_2", "GENE_3"],
    }
    gene_list = ["GENE_1", "GENE_2", "GENE_3", "GENE_4"]

    matrix, names = build_membership_matrix(pathway_dict, gene_list)

    assert names == ["PATHWAY_A", "PATHWAY_B"]
    assert matrix.shape == (2, 4)

    # Pathway A: GENE_1, GENE_2
    assert matrix[0, 0] == 1.0
    assert matrix[0, 1] == 1.0
    assert matrix[0, 2] == 0.0
    assert matrix[0, 3] == 0.0

    # Pathway B: GENE_2, GENE_3
    assert matrix[1, 0] == 0.0
    assert matrix[1, 1] == 1.0
    assert matrix[1, 2] == 1.0
    assert matrix[1, 3] == 0.0


def test_build_membership_matrix_empty():
    """Check behavior with no matches."""
    pathway_dict = {"EMPTY": ["XYZ"]}
    gene_list = ["ABC", "DEF"]
    matrix, names = build_membership_matrix(pathway_dict, gene_list)
    assert matrix.sum() == 0
    assert names == ["EMPTY"]


if __name__ == "__main__":
    pytest.main([__file__])
