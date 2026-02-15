import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

from spatial_transcript_former.models import HE2RNA, ViT_ST, SpatialTranscriptFormer
from spatial_transcript_former.utils import set_seed
from spatial_transcript_former.training.losses import PCCLoss
from spatial_transcript_former.training.engine import train_one_epoch, validate
from spatial_transcript_former.visualization import run_inference_plot
from spatial_transcript_former.data.utils import get_sample_ids, setup_dataloaders

def setup_model(args, device):
    """
    Initialize and optionally compile the model.

    Args:
        args (argparse.Namespace): CLI arguments.
        device (torch.device): Device to move model to.

    Returns:
        nn.Module: The initialized model.
    """
    if args.model == 'he2rna':
        model = HE2RNA(num_genes=args.num_genes, backbone=args.backbone, pretrained=args.pretrained)
    elif args.model == 'vit_st':
        model = ViT_ST(
            num_genes=args.num_genes, 
            model_name=args.backbone if 'vit_' in args.backbone else 'vit_b_16',
            pretrained=args.pretrained
        )
    elif args.model == 'interaction':
        print(f"Initializing SpatialTranscriptFormer with {args.backbone} backbone (pretrained={args.pretrained})...")
        model = SpatialTranscriptFormer(
            num_genes=args.num_genes, 
            backbone_name=args.backbone,
            pretrained=args.pretrained,
            use_nystrom=args.use_nystrom,
            mask_radius=args.mask_radius,
            fusion_mode=args.fusion_mode,
            masked_quadrants=args.masked_quadrants,
            num_pathways=args.num_pathways
        )
    elif args.model == 'attention_mil':
        from spatial_transcript_former.models.mil import AttentionMIL
        model = AttentionMIL(output_dim=args.num_genes, backbone_name=args.backbone, pretrained=args.pretrained)
    elif args.model == 'transmil':
        from spatial_transcript_former.models.mil import TransMIL
        model = TransMIL(output_dim=args.num_genes, backbone_name=args.backbone, pretrained=args.pretrained)
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")
        
    model = model.to(device)
    
    if args.compile:
        print(f"Compiling model with backend='{args.compile_backend}'...")
        try:
            model = torch.compile(model, backend=args.compile_backend)
        except Exception as e:
            print(f"Compilation error: {e}. Proceeding with eager mode.")
            
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Spatial TranscriptFormer Baselines")
    # Data arguments
    group_data = parser.add_argument_group('Data')
    group_data.add_argument('--data-dir', type=str, required=True, help='Root directory of HEST data')
    group_data.add_argument('--num-genes', type=int, default=1000, help='Number of genes to predict')
    group_data.add_argument('--max-samples', type=int, default=None, help='Limit sample count for debugging')
    group_data.add_argument('--precomputed', action='store_true', help='Use pre-computed features')
    group_data.add_argument('--whole-slide', action='store_true', help='Dense prediction on whole slide')
    group_data.add_argument('--seed', type=int, default=42, help='Random seed')
    group_data.add_argument('--log-transform', action='store_true', help='Log1p transform targets')
    
    # Loss
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'pcc', 'poisson', 'logcosh'], help='Loss function')
    
    # Model arguments
    group_model = parser.add_argument_group('Model')
    group_model.add_argument('--model', type=str, default='he2rna', choices=['he2rna', 'vit_st', 'interaction', 'attention_mil', 'transmil'], help='Architecture')
    group_model.add_argument('--backbone', type=str, default='resnet50', help='Feature extractor backbone')
    group_model.add_argument('--no-pretrained', action='store_false', dest='pretrained', help='Randomly initialize backbone')
    group_model.set_defaults(pretrained=True)
    group_model.add_argument('--num-pathways', type=int, default=50, help='Pathways in STF bottleneck')
    group_model.add_argument('--fusion-mode', type=str, default='decoder', choices=['decoder', 'jaume'])
    group_model.add_argument('--use-nystrom', action='store_true', help='Linear attention via Nystrom')
    group_model.add_argument('--mask-radius', type=float, default=None, help='Spatial mask radius')
    
    # Training arguments
    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--epochs', type=int, default=10)
    group_train.add_argument('--batch-size', type=int, default=32)
    group_train.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps')
    group_train.add_argument('--lr', type=float, default=1e-4)
    group_train.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (L2 penalty)')
    group_train.add_argument('--sparsity-lambda', type=float, default=0.0, help='L1 penalty')
    group_train.add_argument('--augment', action='store_true', help='Enable augmentations')
    group_train.add_argument('--use-amp', action='store_true', help='Use Mixed Precision Training')
    group_train.add_argument('--output-dir', type=str, default='./checkpoints')
    group_train.add_argument('--compile', action='store_true')
    group_train.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    
    # Advanced arguments
    group_adv = parser.add_argument_group('Advanced')
    group_adv.add_argument('--n-neighbors', type=int, default=0, help='Context neighbors')
    group_adv.add_argument('--use-global-context', action='store_true')
    group_adv.add_argument('--global-context-size', type=int, default=128)
    group_adv.add_argument('--compile-backend', type=str, default='inductor')
    group_adv.add_argument('--masked-quadrants', type=str, nargs='+', default=None)
    group_adv.add_argument('--plot-pathways', action='store_true', help='Visualize pathway activations during validation')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    
    # 1. Get Samples
    final_ids = get_sample_ids(args.data_dir, args.precomputed, args.backbone, args.max_samples)
    
    # 2. Split
    np.random.shuffle(final_ids)
    split_idx = int(len(final_ids) * 0.8)
    train_ids = final_ids[:split_idx]
    val_ids = final_ids[split_idx:]
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val.")
    
    # 3. Dataloaders
    train_loader, val_loader = setup_dataloaders(args, train_ids, val_ids)
    
    # 4. Model, Loss, Optimizer
    model = setup_model(args, device)
    
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'pcc':
        criterion = PCCLoss()
    elif args.loss == 'poisson':
        criterion = nn.PoissonNLLLoss(log_input=True) # Model needs to output log
    elif args.loss == 'logcosh':
        # Custom or simple implementation
        criterion = nn.MSELoss() # Fallback or implement
        # Actually torch doesn't have logcosh built-in usually, use Huber or L1
        criterion = nn.HuberLoss() 
        print("Using HuberLoss as proxy for LogCosh (Robust Regression)")
    else:
         criterion = nn.MSELoss()
         
    print(f"Using Loss: {criterion}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. Training Loop
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    log_exists = os.path.exists(log_path)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume Logic
    if args.resume:
        ckpt_path = os.path.join(args.output_dir, f'latest_model_{args.model}.pth')
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                 # Helper to load state dict handling potential prefix issues
                 state_dict = checkpoint['model_state_dict']
                 # Check if 'module.' prefix needs to be removed (if saved from DataParallel)
                 # Or added?
                 model.load_state_dict(state_dict)
                 
                 if 'optimizer_state_dict' in checkpoint:
                     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 if 'scaler_state_dict' in checkpoint and scaler is not None:
                     scaler.load_state_dict(checkpoint['scaler_state_dict'])
                 if 'epoch' in checkpoint:
                     start_epoch = checkpoint['epoch'] + 1
                     print(f"DEBUG: Found epoch {checkpoint['epoch']} in checkpoint. Setting start_epoch to {start_epoch}")
                 else:
                     print("DEBUG: No 'epoch' key in checkpoint.")
                 if 'best_val_loss' in checkpoint:
                     best_val_loss = checkpoint['best_val_loss']
                 
                 print(f"Resumed at epoch {start_epoch+1}")
            else:
                 # Legacy or simple save (just model weights)
                 model.load_state_dict(checkpoint)
                 print("Loaded model weights only (no optimizer/epoch state).")
        else:
            print(f"Resume requested but no checkpoint found at {ckpt_path}. Starting from scratch.")
            
    # Fallback: If start_epoch is still 0 and we have a log, try to infer from log
    # This handles both legacy checkpoints (no 'epoch' key) and crashes during first epoch of a resume
    if args.resume and start_epoch == 0 and log_exists:
         try:
             with open(log_path, 'r') as f:
                 lines = f.readlines()
                 if len(lines) > 1:
                     last_line = lines[-1]
                     try:
                        last_epoch = int(last_line.split(',')[0])
                        start_epoch = last_epoch
                        print(f"DEBUG: Inferred start_epoch {start_epoch} from training_log.csv (Legacy Resume)")
                     except ValueError:
                        pass
         except Exception as e:
             print(f"DEBUG: Failed to read epoch from log: {e}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            sparsity_lambda=args.sparsity_lambda, whole_slide=args.whole_slide,
            scaler=scaler, grad_accum_steps=args.grad_accum_steps
        )
        val_loss = validate(model, val_loader, criterion, device, whole_slide=args.whole_slide, use_amp=args.use_amp)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not log_exists and epoch == 0:
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
            writer.writerow([epoch + 1, train_loss, val_loss])
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, f'best_model_{args.model}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")
            
        # Save Latest (with state)
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        if scaler is not None:
            save_dict['scaler_state_dict'] = scaler.state_dict()
            
        torch.save(save_dict, os.path.join(args.output_dir, f'latest_model_{args.model}.pth'))
        
        # Periodic visualization
        if val_ids:
            run_inference_plot(model, args, val_ids[0], epoch, device)

if __name__ == '__main__':
    main()
