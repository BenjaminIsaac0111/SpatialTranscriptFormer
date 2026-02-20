import pytest
import torch
from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.models.interaction import MultimodalFusion

def test_interaction_output_shape(mock_image_batch):
    """
    EDUCATIONAL: This test verifies that the SpatialTranscriptFormer correctly 
    maps a batch of histology images to a high-dimensional gene expression vector.
    
    Architecture:
    Input (x) -> Backbone -> Interaction (Cross-Attention) -> Gene Reconstructor -> Output (y)
    """
    num_genes = 1000
    model = SpatialTranscriptFormer(num_genes=num_genes)
    
    output = model(mock_image_batch)
    
    # Verify shape: (Batch Size, Number of Genes)
    assert output.shape == (mock_image_batch.shape[0], num_genes)

def test_multimodal_fusion_quadrant_masking():
    """
    EDUCATIONAL: This test verifies Jaume Et Al. Equation 1 quadrant masking.
    We verify that masking a quadrant (P2H: Pathway to Histology) effectively 
    blocks information flow, resulting in zero gradients for histology tokens.
    
    Matrix A = [ P2P  P2H ]
               [ H2P  H2H ]
    """
    dim, Np, Nh, B = 64, 5, 10, 2
    
    # Mode: Mask P2H (Pathways cannot see Histology)
    fusion = MultimodalFusion(dim, n_heads=4, n_layers=1, masked_quadrants=['P2H'])
    
    p_tokens = torch.randn(B, Np, dim, requires_grad=True)
    h_tokens = torch.randn(B, Nh, dim, requires_grad=True)
    
    # Forward pass: Results in Pathway tokens (Bottleneck)
    out = fusion(p_tokens, h_tokens)
    
    # Backpropagate signal from output
    out.sum().backward()
    
    # ASSERTION: If P2H is masked, histology tokens (H) should receive ZERO signal from the output
    assert h_tokens.grad is not None
    assert h_tokens.grad.abs().sum() < 1e-6, "Histology tokens should have no gradient if P2H quadrant is masked"

def test_sparsity_regularization_loss():
    """
    EDUCATIONAL: This test verifies the 'L1 Sparsity' calculation.
    Sparsity forces each pathway token to only contribute to a small, distinct 
    set of genes, creating a biologically-interpretable bottleneck.
    """
    num_genes = 100
    num_pathways = 10
    model = SpatialTranscriptFormer(num_genes=num_genes, num_pathways=num_pathways)
    
    sparsity_loss = model.get_sparsity_loss()
    
    # Expect a positive scalar (L1 norm of reconstruction weights)
    assert sparsity_loss > 0
    assert sparsity_loss.dim() == 0

def test_interaction_with_masking(mock_image_batch):
    """
    EDUCATIONAL: Verifies that the unified interaction model works with 
    different quadrant masking configurations.
    """
    num_genes = 100
    
    # Test with H2H masking (Default)
    model = SpatialTranscriptFormer(num_genes=num_genes, masked_quadrants=['H2H'])
    out = model(mock_image_batch)
    assert out.shape == (mock_image_batch.shape[0], num_genes)
    
    # Test with all quadrants open
    model_all = SpatialTranscriptFormer(num_genes=num_genes, masked_quadrants=[])
    out_all = model_all(mock_image_batch)
    assert out_all.shape == (mock_image_batch.shape[0], num_genes)
