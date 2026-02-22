"""
Training engine for SpatialTranscriptFormer.

Provides train_one_epoch() and validate() functions that handle
both standard patch-level and whole-slide training modes.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from spatial_transcript_former.models import SpatialTranscriptFormer


def _optimizer_step(
    scaler, optimizer, loss, batch_idx, total_batches, grad_accum_steps
):
    """Shared gradient accumulation and optimizer step logic."""
    if scaler is not None:
        scaler.scale(loss).backward()
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        loss.backward()
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
            optimizer.step()
            optimizer.zero_grad()


def _compute_masked_mse(preds, targets, mask):
    """MSE loss ignoring padded positions."""
    diff = preds - targets
    mse = diff**2
    valid_mask = ~mask.unsqueeze(-1).expand_as(mse)
    return (mse * valid_mask.float()).sum() / valid_mask.sum()


def _compute_bag_target(genes, mask):
    """Average gene expression over valid spots (bag-level target for MIL)."""
    mask_float = (~mask).float().unsqueeze(-1)  # (B, N, 1)
    return (genes * mask_float).sum(dim=1) / mask_float.sum(dim=1)  # (B, G)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    sparsity_lambda=0.0,
    whole_slide=False,
    scaler=None,
    grad_accum_steps=1,
):
    """
    Train the model for one epoch.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")

    if whole_slide:
        for batch_idx, (feats, genes, coords, mask) in enumerate(pbar):
            feats, genes, coords, mask = (
                feats.to(device),
                genes.to(device),
                coords.to(device),
                mask.to(device),
            )

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                if hasattr(model, "forward_dense") and not getattr(
                    model, "weak_supervision", False
                ):
                    preds = model.forward_dense(feats, mask=mask, coords=coords)
                    loss = criterion(preds, genes, mask=mask)
                else:
                    preds = model(feats)
                    bag_target = _compute_bag_target(genes, mask)
                    loss = criterion(preds, bag_target)

                if sparsity_lambda > 0 and hasattr(model, "get_sparsity_loss"):
                    loss = loss + (sparsity_lambda * model.get_sparsity_loss())

                loss = loss / grad_accum_steps

            _optimizer_step(
                scaler, optimizer, loss, batch_idx, len(loader), grad_accum_steps
            )

            current_loss = loss.item() * grad_accum_steps
            running_loss += current_loss
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})
    else:
        for batch_idx, (images, targets, rel_coords) in enumerate(pbar):
            images, targets = images.to(device), targets.to(device)
            rel_coords = rel_coords.to(device)

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                if isinstance(model, SpatialTranscriptFormer):
                    outputs = model(images, rel_coords=rel_coords)
                else:
                    outputs = model(images)

                loss = criterion(outputs, targets)

                if sparsity_lambda > 0 and hasattr(model, "get_sparsity_loss"):
                    loss = loss + (sparsity_lambda * model.get_sparsity_loss())

                loss = loss / grad_accum_steps

            _optimizer_step(
                scaler, optimizer, loss, batch_idx, len(loader), grad_accum_steps
            )

            current_loss = loss.item() * grad_accum_steps
            running_loss += current_loss
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    return running_loss / len(loader)


def validate(model, loader, criterion, device, whole_slide=False, use_amp=False):
    """
    Validate the model.

    Returns:
        dict: {"val_loss": float, "attn_correlation": float or None}
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    pcc_list = []
    attn_correlations = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if whole_slide:
                feats, genes, coords, mask = batch
                feats, genes, coords, mask = (
                    feats.to(device),
                    genes.to(device),
                    coords.to(device),
                    mask.to(device),
                )
            else:
                images, genes, rel_coords = batch
                images, genes = images.to(device), genes.to(device)
                rel_coords = rel_coords.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                attn = None

                if whole_slide:
                    if hasattr(model, "forward_dense") and not getattr(
                        model, "weak_supervision", False
                    ):
                        outputs = model.forward_dense(feats, mask=mask, coords=coords)
                        targets = genes
                    else:
                        # MIL models: extract attention if supported
                        if (
                            hasattr(model, "forward")
                            and "return_attention" in model.forward.__code__.co_varnames
                        ):
                            outputs, attn = model(feats, return_attention=True)
                        else:
                            outputs = model(feats)
                        targets = _compute_bag_target(genes, mask)
                else:
                    targets = genes
                    if isinstance(model, SpatialTranscriptFormer):
                        outputs = model(images, rel_coords=rel_coords)
                    else:
                        outputs = model(images)

                loss = (
                    criterion(outputs, targets, mask=mask)
                    if whole_slide
                    and hasattr(model, "forward_dense")
                    and not getattr(model, "weak_supervision", False)
                    else criterion(outputs, targets)
                )

                # --- Interpretability Metrics (MAE & PCC) ---
                mae_diff = torch.abs(outputs - targets)
                if (
                    whole_slide
                    and hasattr(model, "forward_dense")
                    and not getattr(model, "weak_supervision", False)
                    and mask is not None
                ):
                    valid_mask = ~mask.unsqueeze(-1).expand_as(mae_diff)
                    mae_val = (mae_diff * valid_mask.float()).sum() / valid_mask.sum()

                    if torch.isfinite(outputs).all() and torch.isfinite(targets).all():
                        # Calculate Spatial PCC (across spots N, for each gene G independently)
                        # outputs/targets are (B, N, G) for whole_slide or (B, G) for patch
                        if whole_slide:
                            # Iterate over batches to correlate spatially for each slide
                            B = outputs.shape[0]
                            for b_idx in range(B):
                                p_slide = outputs[b_idx]  # (N, G)
                                t_slide = targets[b_idx]  # (N, G)

                                valid_idx = ~mask[b_idx]
                                p_slide = p_slide[valid_idx]  # (V, G)
                                t_slide = t_slide[valid_idx]  # (V, G)

                                if p_slide.shape[0] >= 2:
                                    vx = p_slide - p_slide.mean(dim=0, keepdim=True)
                                    vy = t_slide - t_slide.mean(dim=0, keepdim=True)
                                    num = torch.sum(vx * vy, dim=0)  # (G,)
                                    den = torch.sqrt(
                                        torch.sum(vx**2, dim=0) + 1e-8
                                    ) * torch.sqrt(torch.sum(vy**2, dim=0) + 1e-8)
                                    corr = num / den

                                    active_genes = torch.std(t_slide, dim=0) > 1e-6
                                    if active_genes.any():
                                        valid_corrs = corr[active_genes]
                                        valid_corrs = valid_corrs[
                                            torch.isfinite(valid_corrs)
                                        ]
                                        if len(valid_corrs) > 0:
                                            pcc_list.append(valid_corrs.mean().item())
                        else:
                            # Patch level (B, G). Correlate across the batch B (which is spatial patches)
                            vx = outputs - outputs.mean(dim=0, keepdim=True)
                            vy = targets - targets.mean(dim=0, keepdim=True)
                            num = torch.sum(vx * vy, dim=0)
                            den = torch.sqrt(
                                torch.sum(vx**2, dim=0) + 1e-8
                            ) * torch.sqrt(torch.sum(vy**2, dim=0) + 1e-8)
                            corr = num / den

                            active_genes = torch.std(targets, dim=0) > 1e-6
                            if active_genes.any():
                                valid_corrs = corr[active_genes]
                                valid_corrs = valid_corrs[torch.isfinite(valid_corrs)]
                                if len(valid_corrs) > 0:
                                    pcc_list.append(valid_corrs.mean().item())

                # Spatial Attention Correlation (MIL weak supervision study)
                if attn is not None and whole_slide:
                    if attn.dim() == 3:
                        attn = attn.squeeze(-1)
                    for b in range(attn.shape[0]):
                        valid_idx = ~mask[b]
                        a_b = attn[b][valid_idx]
                        g_total = genes[b][valid_idx].sum(dim=-1)
                        if a_b.std() > 0 and g_total.std() > 0:
                            corr = torch.corrcoef(torch.stack([a_b, g_total]))[0, 1]
                            attn_correlations.append(corr.item())

            running_loss += loss.item()
            running_mae += mae_val.item()

    avg_loss = running_loss / len(loader)
    avg_mae = running_mae / len(loader)
    avg_pcc = sum(pcc_list) / len(pcc_list) if pcc_list else None
    avg_corr = (
        sum(attn_correlations) / len(attn_correlations) if attn_correlations else None
    )

    if avg_pcc is not None:
        print(f"Validation MAE: {avg_mae:.4f} | PCC: {avg_pcc:.4f}")
    if avg_corr is not None:
        print(f"Spatial Attention Correlation: {avg_corr:.4f}")

    return {
        "val_loss": avg_loss,
        "val_mae": avg_mae,
        "val_pcc": avg_pcc,
        "attn_correlation": avg_corr,
    }
