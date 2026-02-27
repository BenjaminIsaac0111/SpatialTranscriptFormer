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

from spatial_transcript_former.config import get_config
from spatial_transcript_former.models import HE2RNA, ViT_ST, SpatialTranscriptFormer
from spatial_transcript_former.utils import set_seed
from spatial_transcript_former.training.losses import (
    PCCLoss,
    CompositeLoss,
    MaskedMSELoss,
    ZINBLoss,
)
from spatial_transcript_former.training.engine import train_one_epoch, validate
from spatial_transcript_former.training.experiment_logger import ExperimentLogger
from spatial_transcript_former.visualization import run_inference_plot
from spatial_transcript_former.data.utils import get_sample_ids, setup_dataloaders

# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------


def setup_model(args, device):
    """Initialize and optionally compile the model."""
    if args.model == "he2rna":
        model = HE2RNA(
            num_genes=args.num_genes, backbone=args.backbone, pretrained=args.pretrained
        )
    elif args.model == "vit_st":
        model = ViT_ST(
            num_genes=args.num_genes,
            model_name=args.backbone if "vit_" in args.backbone else "vit_b_16",
            pretrained=args.pretrained,
        )
    elif args.model == "interaction":
        print(
            f"Initializing SpatialTranscriptFormer ({args.backbone}, pretrained={args.pretrained})"
        )

        # Load biological pathway initialization if requested
        pathway_init = None
        if getattr(args, "pathway_init", False):
            from spatial_transcript_former.data.pathways import (
                get_pathway_init,
                MSIGDB_URLS,
            )
            import json

            genes_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "global_genes.json",
            )
            if not os.path.exists(genes_path):
                genes_path = "global_genes.json"
            with open(genes_path) as f:
                gene_list = json.load(f)

            if getattr(args, "custom_gmt", None):
                urls = args.custom_gmt
            elif getattr(args, "pathways", None):
                # If specific pathways requested but no custom GMT, search standard collections
                urls = [
                    MSIGDB_URLS["hallmarks"],
                    MSIGDB_URLS["c2_medicus"],
                    MSIGDB_URLS["c2_cgp"],
                ]
            else:
                # Default to just the 50 Hallmarks to prevent VRAM exhaustion
                urls = [MSIGDB_URLS["hallmarks"]]

            pathway_init, pathway_names = get_pathway_init(
                gene_list[: args.num_genes], gmt_urls=urls, filter_names=args.pathways
            )
            # Override num_pathways based on actual parsed paths
            args.num_pathways = len(pathway_names)
            print(f"Num pathways forced to {args.num_pathways} based on init dict")

        model = SpatialTranscriptFormer(
            num_genes=args.num_genes,
            backbone_name=args.backbone,
            pretrained=args.pretrained,
            token_dim=args.token_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            num_pathways=args.num_pathways,
            pathway_init=pathway_init,
            use_spatial_pe=args.use_spatial_pe,
            output_mode="zinb" if args.loss == "zinb" else "counts",
            interactions=getattr(args, "interactions", None),
        )
    elif args.model == "attention_mil":
        from spatial_transcript_former.models.mil import AttentionMIL

        model = AttentionMIL(
            output_dim=args.num_genes,
            backbone_name=args.backbone,
            pretrained=args.pretrained,
        )
    elif args.model == "transmil":
        from spatial_transcript_former.models.mil import TransMIL

        model = TransMIL(
            output_dim=args.num_genes,
            backbone_name=args.backbone,
            pretrained=args.pretrained,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.weak_supervision = getattr(args, "weak_supervision", False)
    model = model.to(device)

    if args.compile:
        print(f"Compiling model (backend='{args.compile_backend}')...")
        try:
            model = torch.compile(model, backend=args.compile_backend)
        except Exception as e:
            print(f"Compilation failed: {e}. Using eager mode.")

    return model


def setup_criterion(args, pathway_init=None):
    """Create loss function from CLI args.

    If ``pathway_init`` is provided and ``--pathway-loss-weight > 0``,
    wraps the base criterion with :class:`AuxiliaryPathwayLoss`.
    """
    if args.loss == "pcc":
        base = PCCLoss()
    elif args.loss == "mse_pcc":
        base = CompositeLoss(alpha=args.pcc_weight)
    elif args.loss == "zinb":
        base = ZINBLoss()
    elif args.loss == "poisson":
        base = nn.PoissonNLLLoss(log_input=True)
    elif args.loss == "logcosh":
        print("Using HuberLoss as proxy for LogCosh")
        base = nn.HuberLoss()
    else:
        base = MaskedMSELoss()

    pw_weight = getattr(args, "pathway_loss_weight", 0.0)
    if pathway_init is not None and pw_weight > 0:
        from spatial_transcript_former.training.losses import AuxiliaryPathwayLoss

        print(f"Wrapping criterion with AuxiliaryPathwayLoss (lambda={pw_weight})")
        return AuxiliaryPathwayLoss(pathway_init, base, lambda_pathway=pw_weight)

    return base


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------


def save_checkpoint(
    model, optimizer, scaler, epoch, best_val_loss, output_dir, model_name
):
    """Save training state for resuming."""
    save_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if scaler is not None:
        save_dict["scaler_state_dict"] = scaler.state_dict()

    torch.save(save_dict, os.path.join(output_dir, f"latest_model_{model_name}.pth"))


def load_checkpoint(model, optimizer, scaler, output_dir, model_name, device):
    """
    Load checkpoint if it exists.

    Returns:
        tuple: (start_epoch, best_val_loss)
    """
    ckpt_path = os.path.join(output_dir, f"latest_model_{model_name}.pth")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Starting from scratch.")
        return 0, float("inf")

    print(f"Resuming from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"Resumed at epoch {start_epoch + 1}")
    return start_epoch, best_val_loss


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train Spatial TranscriptFormer")

    # Data
    g = parser.add_argument_group("Data")
    g.add_argument(
        "--data-dir",
        type=str,
        default=get_config("data_dirs", ["hest_data"])[0],
        help="Root directory of HEST data",
    )
    g.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help="Explicit feature directory (overrides auto-detection)",
    )
    g.add_argument(
        "--num-genes", type=int, default=get_config("training.num_genes", 1000)
    )
    g.add_argument(
        "--max-samples", type=int, default=None, help="Limit samples for debugging"
    )
    g.add_argument(
        "--precomputed", action="store_true", help="Use pre-computed features"
    )
    g.add_argument(
        "--whole-slide", action="store_true", help="Dense whole-slide prediction"
    )
    g.add_argument("--seed", type=int, default=42)
    g.add_argument(
        "--log-transform", action="store_true", help="Log1p transform targets"
    )
    g.add_argument("--organ", type=str, default=None, help="Filter samples by organ")

    # Loss
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "pcc", "mse_pcc", "zinb", "poisson", "logcosh"],
    )
    parser.add_argument(
        "--pcc-weight",
        type=float,
        default=1.0,
        help="Weight for PCC term in mse_pcc loss",
    )
    parser.add_argument(
        "--pathway-loss-weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary pathway PCC loss (0 = disabled)",
    )

    # Model
    g = parser.add_argument_group("Model")
    g.add_argument(
        "--model",
        type=str,
        default="he2rna",
        choices=["he2rna", "vit_st", "interaction", "attention_mil", "transmil"],
    )
    g.add_argument("--backbone", type=str, default="resnet50")
    g.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    g.set_defaults(pretrained=True)
    g.add_argument("--num-pathways", type=int, default=50)
    g.add_argument("--token-dim", type=int, default=256)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-layers", type=int, default=2)
    g.add_argument(
        "--use-spatial-pe",
        action="store_true",
        help="Enable spatial positional encoding",
    )
    g.add_argument(
        "--interactions",
        nargs="+",
        default=None,
        help="Attention interactions to enable: p2p, p2h, h2p, h2h (default: all)",
    )

    # Training
    g = parser.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=get_config("training.epochs", 10))
    g.add_argument(
        "--batch-size", type=int, default=get_config("training.batch_size", 32)
    )
    g.add_argument("--grad-accum-steps", type=int, default=1)
    g.add_argument(
        "--lr", type=float, default=get_config("training.learning_rate", 1e-4)
    )
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--warmup-epochs", type=int, default=10)
    g.add_argument("--sparsity-lambda", type=float, default=0.0)
    g.add_argument("--augment", action="store_true")
    g.add_argument("--use-amp", action="store_true")
    g.add_argument(
        "--output-dir",
        type=str,
        default=get_config("training.output_dir", "./checkpoints"),
    )
    g.add_argument("--compile", action="store_true")
    g.add_argument("--resume", action="store_true")

    # Advanced
    g = parser.add_argument_group("Advanced")
    g.add_argument("--n-neighbors", type=int, default=0)
    g.add_argument("--use-global-context", action="store_true")
    g.add_argument("--global-context-size", type=int, default=128)
    g.add_argument("--compile-backend", type=str, default="inductor")
    g.add_argument("--plot-pathways", action="store_true")
    g.add_argument("--plot-attention", action="store_true")
    g.add_argument(
        "--weak-supervision", action="store_true", help="Bag-level training for MIL"
    )
    g.add_argument(
        "--pathway-init",
        action="store_true",
        help="Initialize gene_reconstructor with MSigDB Hallmarks",
    )
    g.add_argument(
        "--pathways",
        nargs="+",
        default=None,
        help="List of MSigDB pathway names to explicitly instantiate (e.g. HALLMARK_APOPTOSIS). If none are provided but --pathway-init is enabled, all pathways in the provided GMTs will be loaded.",
    )
    g.add_argument(
        "--custom-gmt",
        nargs="+",
        default=None,
        help="List of URLs or local paths to custom .gmt files for pathway initialization. Overrides standard MSigDB defaults if provided.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(args.seed)

    # Global gene count synchronization
    genes_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "global_genes.json",
    )
    if not os.path.exists(genes_path):
        genes_path = "global_genes.json"
    if os.path.exists(genes_path):
        import json

        with open(genes_path, "r") as f:
            gene_list = json.load(f)
        args.num_genes = min(args.num_genes, len(gene_list))
        print(f"Validated global gene count: {args.num_genes}")
    else:
        print(
            f"Warning: global_genes.json not found. Using requested num_genes={args.num_genes}"
        )

    # 1. Data
    final_ids = get_sample_ids(
        args.data_dir,
        precomputed=args.precomputed,
        backbone=args.backbone,
        feature_dir=args.feature_dir,
        max_samples=args.max_samples,
        organ=args.organ,
    )
    np.random.shuffle(final_ids)

    if len(final_ids) == 1:
        # Prevent empty train split if only testing 1 sample
        train_ids, val_ids = final_ids, final_ids
    else:
        split_idx = int(len(final_ids) * 0.8)
        train_ids, val_ids = final_ids[:split_idx], final_ids[split_idx:]

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    train_loader, val_loader = setup_dataloaders(args, train_ids, val_ids)

    # 2. Model, Loss, Optimizer
    model = setup_model(args, device)
    # Pass pathway_init to criterion so AuxiliaryPathwayLoss can use it
    pathway_init = getattr(model, "_pathway_init_matrix", None)
    criterion = setup_criterion(args, pathway_init=pathway_init).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # LR scheduler: cosine annealing with optional linear warmup
    warmup_epochs = args.warmup_epochs
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        return 1.0  # cosine scheduler handles the rest

    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None
    print(f"Loss: {criterion.__class__.__name__}")
    print(f"LR schedule: {warmup_epochs}-epoch warmup → cosine annealing to 1e-6")

    # 3. Output & Logger
    os.makedirs(args.output_dir, exist_ok=True)
    config_dict = vars(args)
    logger = ExperimentLogger(args.output_dir, config_dict)

    # 4. Resume
    start_epoch, best_val_loss = 0, float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scaler, args.output_dir, args.model, device
        )

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            sparsity_lambda=args.sparsity_lambda,
            whole_slide=args.whole_slide,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            whole_slide=args.whole_slide,
            use_amp=args.use_amp,
        )
        val_loss = val_metrics["val_loss"]

        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Step LR scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # Log epoch
        epoch_row = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_metrics.get("val_mae") is not None:
            epoch_row["val_mae"] = round(val_metrics["val_mae"], 4)
        if val_metrics.get("val_pcc") is not None:
            epoch_row["val_pcc"] = round(val_metrics["val_pcc"], 4)
        if val_metrics.get("pred_variance") is not None:
            epoch_row["pred_variance"] = round(val_metrics["pred_variance"], 6)
        if val_metrics.get("attn_correlation") is not None:
            epoch_row["attn_correlation"] = round(val_metrics["attn_correlation"], 4)
        logger.log_epoch(epoch + 1, epoch_row)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, f"best_model_{args.model}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model -> {best_path}")

        # Save latest
        save_checkpoint(
            model, optimizer, scaler, epoch, best_val_loss, args.output_dir, args.model
        )

        # Periodic visualization (only when --plot-pathways is set)
        if args.plot_pathways and val_ids:
            run_inference_plot(model, args, val_ids[0], epoch, device)

    # 6. Finalize
    logger.finalize(best_val_loss)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
