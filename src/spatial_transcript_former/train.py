"""
Training script for SpatialTranscriptFormer and baselines.

Usage:
    stf-train --model interaction --data-dir /path/to/hest --precomputed --whole-slide
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from spatial_transcript_former.models import HE2RNA, ViT_ST, SpatialTranscriptFormer
from spatial_transcript_former.utils import set_seed
from spatial_transcript_former.training.losses import PCCLoss, CompositeLoss, MaskedMSELoss
from spatial_transcript_former.training.engine import train_one_epoch, validate
from spatial_transcript_former.training.experiment_logger import ExperimentLogger
from spatial_transcript_former.visualization import run_inference_plot
from spatial_transcript_former.data.utils import get_sample_ids, setup_dataloaders


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def setup_model(args, device):
    """Initialize and optionally compile the model."""
    if args.model == 'he2rna':
        model = HE2RNA(num_genes=args.num_genes, backbone=args.backbone, pretrained=args.pretrained)
    elif args.model == 'vit_st':
        model = ViT_ST(
            num_genes=args.num_genes,
            model_name=args.backbone if 'vit_' in args.backbone else 'vit_b_16',
            pretrained=args.pretrained
        )
    elif args.model == 'interaction':
        print(f"Initializing SpatialTranscriptFormer ({args.backbone}, pretrained={args.pretrained})")

        # Load biological pathway initialization if requested
        pathway_init = None
        if getattr(args, 'pathway_init', False):
            from spatial_transcript_former.data.pathways import get_pathway_init
            import json
            genes_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'global_genes.json')
            if not os.path.exists(genes_path):
                genes_path = 'global_genes.json'
            with open(genes_path) as f:
                gene_list = json.load(f)
            pathway_init, pathway_names = get_pathway_init(gene_list[:args.num_genes])

        model = SpatialTranscriptFormer(
            num_genes=args.num_genes,
            backbone_name=args.backbone,
            pretrained=args.pretrained,
            use_nystrom=args.use_nystrom,
            mask_radius=args.mask_radius,
            fusion_mode=args.fusion_mode,
            masked_quadrants=args.masked_quadrants,
            num_pathways=args.num_pathways,
            pathway_init=pathway_init,
            use_spatial_pe=args.use_spatial_pe
        )
    elif args.model == 'attention_mil':
        from spatial_transcript_former.models.mil import AttentionMIL
        model = AttentionMIL(output_dim=args.num_genes, backbone_name=args.backbone, pretrained=args.pretrained)
    elif args.model == 'transmil':
        from spatial_transcript_former.models.mil import TransMIL
        model = TransMIL(output_dim=args.num_genes, backbone_name=args.backbone, pretrained=args.pretrained)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.weak_supervision = getattr(args, 'weak_supervision', False)
    model = model.to(device)

    if args.compile:
        print(f"Compiling model (backend='{args.compile_backend}')...")
        try:
            model = torch.compile(model, backend=args.compile_backend)
        except Exception as e:
            print(f"Compilation failed: {e}. Using eager mode.")

    return model


def setup_criterion(args):
    """Create loss function from CLI args."""
    if args.loss == 'pcc':
        return PCCLoss()
    elif args.loss == 'mse_pcc':
        return CompositeLoss(alpha=args.pcc_weight)
    elif args.loss == 'poisson':
        return nn.PoissonNLLLoss(log_input=True)
    elif args.loss == 'logcosh':
        print("Using HuberLoss as proxy for LogCosh")
        return nn.HuberLoss()
    else:
        return MaskedMSELoss()


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, epoch, best_val_loss, output_dir, model_name):
    """Save training state for resuming."""
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    if scaler is not None:
        save_dict['scaler_state_dict'] = scaler.state_dict()

    torch.save(save_dict, os.path.join(output_dir, f'latest_model_{model_name}.pth'))


def load_checkpoint(model, optimizer, scaler, output_dir, model_name, device):
    """
    Load checkpoint if it exists.

    Returns:
        tuple: (start_epoch, best_val_loss)
    """
    ckpt_path = os.path.join(output_dir, f'latest_model_{model_name}.pth')
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Starting from scratch.")
        return 0, float('inf')

    print(f"Resuming from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        # Legacy checkpoint (raw state dict)
        model.load_state_dict(checkpoint)
        start_epoch = 0
        best_val_loss = float('inf')
        print("Loaded weights only (legacy checkpoint).")

    print(f"Resumed at epoch {start_epoch + 1}")
    return start_epoch, best_val_loss


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Spatial TranscriptFormer")

    # Data
    g = parser.add_argument_group('Data')
    g.add_argument('--data-dir', type=str, required=True, help='Root directory of HEST data')
    g.add_argument('--feature-dir', type=str, default=None, help='Explicit feature directory (overrides auto-detection)')
    g.add_argument('--num-genes', type=int, default=1000)
    g.add_argument('--max-samples', type=int, default=None, help='Limit samples for debugging')
    g.add_argument('--precomputed', action='store_true', help='Use pre-computed features')
    g.add_argument('--whole-slide', action='store_true', help='Dense whole-slide prediction')
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--log-transform', action='store_true', help='Log1p transform targets')

    # Loss
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'pcc', 'mse_pcc', 'poisson', 'logcosh'])
    parser.add_argument('--pcc-weight', type=float, default=1.0, help='Weight for PCC term in mse_pcc loss')

    # Model
    g = parser.add_argument_group('Model')
    g.add_argument('--model', type=str, default='he2rna',
                   choices=['he2rna', 'vit_st', 'interaction', 'attention_mil', 'transmil'])
    g.add_argument('--backbone', type=str, default='resnet50')
    g.add_argument('--no-pretrained', action='store_false', dest='pretrained')
    g.set_defaults(pretrained=True)
    g.add_argument('--num-pathways', type=int, default=50)
    g.add_argument('--fusion-mode', type=str, default='decoder', choices=['decoder', 'jaume'])
    g.add_argument('--use-nystrom', action='store_true')
    g.add_argument('--mask-radius', type=float, default=None)
    g.add_argument('--no-spatial-pe', action='store_false', dest='use_spatial_pe', help='Disable Spatial Positional Encoding')
    g.set_defaults(use_spatial_pe=True)

    # Training
    g = parser.add_argument_group('Training')
    g.add_argument('--epochs', type=int, default=10)
    g.add_argument('--batch-size', type=int, default=32)
    g.add_argument('--grad-accum-steps', type=int, default=1)
    g.add_argument('--lr', type=float, default=1e-4)
    g.add_argument('--weight-decay', type=float, default=0.0)
    g.add_argument('--sparsity-lambda', type=float, default=0.0)
    g.add_argument('--augment', action='store_true')
    g.add_argument('--use-amp', action='store_true')
    g.add_argument('--output-dir', type=str, default='./checkpoints')
    g.add_argument('--compile', action='store_true')
    g.add_argument('--resume', action='store_true')

    # Advanced
    g = parser.add_argument_group('Advanced')
    g.add_argument('--n-neighbors', type=int, default=0)
    g.add_argument('--use-global-context', action='store_true')
    g.add_argument('--global-context-size', type=int, default=128)
    g.add_argument('--compile-backend', type=str, default='inductor')
    g.add_argument('--masked-quadrants', type=str, nargs='+', default=None)
    g.add_argument('--plot-pathways', action='store_true')
    g.add_argument('--weak-supervision', action='store_true', help='Bag-level training for MIL')
    g.add_argument('--pathway-init', action='store_true', help='Initialize gene_reconstructor with MSigDB Hallmarks')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    set_seed(args.seed)

    # 1. Data
    final_ids = get_sample_ids(args.data_dir, args.precomputed, args.backbone, args.max_samples)
    np.random.shuffle(final_ids)
    split_idx = int(len(final_ids) * 0.8)
    train_ids, val_ids = final_ids[:split_idx], final_ids[split_idx:]
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    train_loader, val_loader = setup_dataloaders(args, train_ids, val_ids)

    # 2. Model, Loss, Optimizer
    model = setup_model(args, device)
    criterion = setup_criterion(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    print(f"Loss: {criterion.__class__.__name__}")

    # 3. Output & Logger
    os.makedirs(args.output_dir, exist_ok=True)
    config_dict = vars(args)
    logger = ExperimentLogger(args.output_dir, config_dict)

    # 4. Resume
    start_epoch, best_val_loss = 0, float('inf')
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scaler, args.output_dir, args.model, device
        )

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            sparsity_lambda=args.sparsity_lambda, whole_slide=args.whole_slide,
            scaler=scaler, grad_accum_steps=args.grad_accum_steps
        )

        val_metrics = validate(
            model, val_loader, criterion, device,
            whole_slide=args.whole_slide, use_amp=args.use_amp
        )
        val_loss = val_metrics["val_loss"]

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log epoch
        epoch_row = {"train_loss": train_loss, "val_loss": val_loss}
        if val_metrics.get("attn_correlation") is not None:
            epoch_row["attn_correlation"] = round(val_metrics["attn_correlation"], 4)
        logger.log_epoch(epoch + 1, epoch_row)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, f'best_model_{args.model}.pth')
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model -> {best_path}")

        # Save latest
        save_checkpoint(model, optimizer, scaler, epoch, best_val_loss, args.output_dir, args.model)

        # Periodic visualization
        if val_ids:
            run_inference_plot(model, args, val_ids[0], epoch, device)

    # 6. Finalize
    logger.finalize(best_val_loss)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
