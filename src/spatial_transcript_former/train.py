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
from spatial_transcript_former.recipes.hest.utils import (
    get_sample_ids,
    setup_dataloaders,
)

from spatial_transcript_former.training.arguments import parse_args
from spatial_transcript_former.training.builder import setup_model, setup_criterion
from spatial_transcript_former.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

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
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=max(1, warmup_epochs)
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6
    )

    if warmup_epochs > 0:
        main_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        main_scheduler = cosine_scheduler

    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None
    print(f"Loss: {criterion.__class__.__name__}")
    print(f"LR schedule: {warmup_epochs}-epoch warmup → cosine annealing to 1e-6")

    # 3. Output & Logger
    os.makedirs(args.output_dir, exist_ok=True)
    config_dict = vars(args)
    logger = ExperimentLogger(args.output_dir, config_dict)

    # 4. Resume
    start_epoch, best_val_loss = 0, float("inf")
    schedulers = {"main": main_scheduler}
    if args.resume:
        start_epoch, best_val_loss, loaded_schedulers = load_checkpoint(
            model, optimizer, scaler, schedulers, args.output_dir, args.model, device
        )

        # Fallback for old checkpoints: manually step the scheduler to catch up
        if start_epoch > 0 and main_scheduler.last_epoch < start_epoch:
            print(
                f"Old checkpoint detected. Manually stepping scheduler {start_epoch} times to catch up..."
            )
            for _ in range(start_epoch):
                main_scheduler.step()

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
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
        main_scheduler.step()

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
        if val_metrics.get("spatial_coherence") is not None:
            epoch_row["spatial_coherence"] = round(val_metrics["spatial_coherence"], 4)
        if val_metrics.get("attn_correlation") is not None:
            epoch_row["attn_correlation"] = round(val_metrics["attn_correlation"], 4)

        # Hardware Resource Monitoring
        try:
            import psutil

            epoch_row["sys_cpu_percent"] = psutil.cpu_percent()
            epoch_row["sys_ram_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass

        if torch.cuda.is_available():
            epoch_row["sys_gpu_mem_mb"] = round(
                torch.cuda.memory_allocated() / (1024**2), 2
            )

        logger.log_epoch(epoch + 1, epoch_row)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, f"best_model_{args.model}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model -> {best_path}")

        # Save latest
        save_checkpoint(
            model,
            optimizer,
            scaler,
            schedulers,
            epoch,
            best_val_loss,
            args.output_dir,
            args.model,
        )

        # Periodic visualization
        if val_ids and (epoch + 1) % args.vis_interval == 0:
            vis_id = args.vis_sample if args.vis_sample else val_ids[0]
            print(f"Generating visualization for sample {vis_id}...")
            run_inference_plot(model, args, vis_id, epoch + 1, device)

    # 6. Finalize
    logger.finalize(best_val_loss)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
