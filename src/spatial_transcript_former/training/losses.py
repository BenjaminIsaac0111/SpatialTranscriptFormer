import torch
import torch.nn as nn

class PCCLoss(nn.Module):
    """
    Pearson Correlation Coefficient Loss.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, preds, target):
        """
        Args:
            preds: (B, G) or (B, N, G)
            target: (B, G) or (B, N, G)
        """
        # Flatten for correlation per sample (or per patch?)
        # Standard PCC is per-gene profile. 
        # If input is (B, N, G), we treat each patch as a sample?
        # Or do we correlate the entire slide's gene expression?
        # Typically in ST, PCC is calculated per-location (per-spot) across genes.
        
        if preds.dim() == 3: # (B, N, G)
            # Flatten to (B*N, G)
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.reshape(-1, target.shape[-1])
            
        # Center the data
        vx = preds - torch.mean(preds, dim=1, keepdim=True)
        vy = target - torch.mean(target, dim=1, keepdim=True)
        
        cost = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + self.eps)
        return 1 - torch.mean(cost)
