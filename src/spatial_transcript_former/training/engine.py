import torch
import torch.nn as nn
from tqdm import tqdm
from spatial_transcript_former.models import SpatialTranscriptFormer

def train_one_epoch(model, loader, criterion, optimizer, device, sparsity_lambda=0.0, whole_slide=False, scaler=None, grad_accum_steps=1):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on.
        sparsity_lambda (float): L1 sparsity penalty for gene reconstruction weights.
        whole_slide (bool): Whether to use whole slide training mode.
        scaler (torch.cuda.amp.GradScaler): Scaler for AMP.
        grad_accum_steps (int): Number of steps to accumulate gradients.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    
    optimizer.zero_grad()  # Initialize gradients
    
    pbar = tqdm(loader, desc="Training")
    
    if whole_slide:
        for batch_idx, (feats, genes, coords, mask) in enumerate(pbar):  # batch_idx is global step in epoch
            feats = feats.to(device)
            genes = genes.to(device)
            mask = mask.to(device)
            
            # Use AMP if scaler is provided
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                # Use forward_dense for whole-slide prediction
                if hasattr(model, 'forward_dense'):
                    preds = model.forward_dense(feats, mask=mask)  # (B, N, G)
                else:
                    # Fallback for models that don't support dense forward
                    preds = model(feats.squeeze(0))
                
                # We need to mask the loss calculation too!
                # genes is (B, N, G), preds is (B, N, G), mask is (B, N) True=Padding
                
                # Flatten everything to (B*N, G) and (B*N)
                loss = criterion(preds, genes) # This assumes criterion handles unreduced or we mask after
                
                # Standard MSELoss reduces by mean usually.
                # If we want to ignore padding, we must use reduction='none' and mask manually.
                # OR we can just zerout the loss where mask is True.
                
                # Check criterion reduction. If it's mean, it includes padding zeros which is wrong.
                # criterion = nn.MSELoss() usually.
                
                # Let's enforce masking if batch_size > 1
                if feats.shape[0] > 1 or mask.any():
                    # Recalculate loss with reduction='none'
                    # We assume criterion passed in is default (mean).
                    # We can't easily change it here without access to class.
                    # Best way: make sure padded genes are 0 and preds are close to 0?
                    # No, best is to compute manual MSE.
                    
                    diff = preds - genes
                    mse = diff ** 2 # (B, N, G)
                    
                    # Expand mask to (B, N, G)
                    # mask is True for padding. We want to keep False.
                    valid_mask = ~mask.unsqueeze(-1).expand_as(mse)
                    
                    mse = mse * valid_mask.float()
                    
                    # Sum and divide by number of valid elements
                    loss = mse.sum() / valid_mask.sum()
                
                if sparsity_lambda > 0 and hasattr(model, 'get_sparsity_loss'):
                    loss = loss + (sparsity_lambda * model.get_sparsity_loss())

                elif sparsity_lambda > 0:
                    # Manual sparsity if method not available
                    l1_loss = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                    loss = loss + (sparsity_lambda * l1_loss)
                    
                # Normalize loss for accumulation
                loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Log unscaled loss for reporting
            current_loss = loss.item() * grad_accum_steps
            running_loss += current_loss
            pbar.set_postfix({'loss': current_loss})
    else:
        for batch_idx, (images, targets, rel_coords) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            rel_coords = rel_coords.to(device)
            
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                if isinstance(model, SpatialTranscriptFormer):
                    outputs = model(images, rel_coords=rel_coords)
                else:
                    outputs = model(images)
                
                loss = criterion(outputs, targets)
                
                if sparsity_lambda > 0 and hasattr(model, 'get_sparsity_loss'):
                    loss = loss + (sparsity_lambda * model.get_sparsity_loss())
                    
                loss = loss / grad_accum_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
            
            current_loss = loss.item() * grad_accum_steps
            running_loss += current_loss
            pbar.set_postfix({'loss': current_loss})
            
    return running_loss / len(loader)

def validate(model, loader, criterion, device, whole_slide=False, use_amp=False):
    """
    Validate the model.

    Args:
        model (nn.Module): The model to validate.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run validation on.
        whole_slide (bool): Whether to use whole slide validation mode.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for item in tqdm(loader, desc="Validation"):
            # Handle both standard and whole-slide batches
            if len(item) == 3:
                data, targets, coords = item
            else:
                # Should not happen with new collate, but safety
                data, targets, coords = item[0], item[1], item[2]
            
            data = data.to(device)
            targets = targets.to(device)
            coords = coords.to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                if whole_slide:
                    # In whole slide mode, we only run inference on the first sample to save time/memory
                    # or we run on all but we need to handle batching if loader is batched.
                    # Given validation loader might imply batch_size=1 usually for simple eval.
                    
                    # We'll just take the first item from loader for visualization
                    # This block is for visualization, not for loss calculation over the whole dataset.
                    # The loss calculation for whole_slide is handled below.
                    try:
                        batch = next(iter(loader))
                        # Handle unpacked batch
                        if len(batch) == 4:
                            feats, genes, coords_vis, mask = batch
                        else:
                            # Fallback if collate hasn't kicked in? Should always be 4 now for WS
                            feats, genes, coords_vis = batch
                            mask = None
                            
                        feats = feats.to(device)
                        
                        # Just take first sample in batch
                        if feats.dim() == 3:
                            feats = feats[0].unsqueeze(0) # (1, N, D)
                            coords_vis = coords_vis[0] # (N, 2)
                        
                        # Run inference
                        with torch.no_grad():
                            # We can use forward_dense with no mask since it's 1 sample
                            if hasattr(model, 'forward_dense'):
                                preds = model.forward_dense(feats)
                            else:
                                preds = model(feats.squeeze(0))
                        
                        # ... plotting logic expects single sample ...
                        # We need to ensure we pass correct coords and preds to plot
                        # preds is (1, N, G) -> (N, G)
                        preds = preds.squeeze(0).cpu().numpy()
                    except StopIteration:
                        # Loader might be empty or exhausted if this is called multiple times
                        pass

                    if data.dim() == 2:
                        data = data.unsqueeze(0)
                    if targets.dim() == 3:
                        targets = targets.squeeze(0)
                        
                    if hasattr(model, 'forward_dense'):
                        preds = model.forward_dense(data)
                        outputs = preds.squeeze(0)
                    else:
                        outputs = model(data.squeeze(0))
                elif isinstance(model, SpatialTranscriptFormer):
                    outputs = model(data, rel_coords=coords)
                else:
                    outputs = model(data)
                
                loss = criterion(outputs, targets)
            running_loss += loss.item()
            
    return running_loss / len(loader)
