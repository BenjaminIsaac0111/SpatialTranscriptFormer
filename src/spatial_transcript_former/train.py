import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from spatial_transcript_former.models import HE2RNA, ViT_ST, SpatialTranscriptFormer
from spatial_transcript_former.data.dataset import get_hest_dataloader, load_gene_expression_matrix, get_hest_feature_dataloader, HEST_FeatureDataset
from spatial_transcript_former.predict import plot_spatial_genes, plot_histology_overlay
import h5py
import csv


def train_one_epoch(model, loader, criterion, optimizer, device, sparsity_lambda=0.0, whole_slide=False, scaler=None):
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

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    
    if whole_slide:
        for batch_idx, (feats, genes, coords) in enumerate(pbar):
            feats = feats.to(device)
            genes = genes.to(device)
            
            if feats.dim() == 2:
                feats = feats.unsqueeze(0)  # (1, N, D)
            if genes.dim() == 3:
                genes = genes.squeeze(0)  # (N, G)
            
            optimizer.zero_grad()
            
            # Use AMP if scaler is provided
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                # Use forward_dense for whole-slide prediction
                if hasattr(model, 'forward_dense'):
                    preds = model.forward_dense(feats)  # (1, N, G)
                    preds = preds.squeeze(0)  # (N, G)
                else:
                    # Fallback for models that don't support dense forward
                    preds = model(feats.squeeze(0))
                
                loss = criterion(preds, genes)
                
                if sparsity_lambda > 0 and hasattr(model, 'get_sparsity_loss'):
                    loss = loss + (sparsity_lambda * model.get_sparsity_loss())
                elif sparsity_lambda > 0:
                    # Manual sparsity if method not available
                    l1_loss = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                    loss = loss + (sparsity_lambda * l1_loss)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    else:
        for batch_idx, (images, targets, rel_coords) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            rel_coords = rel_coords.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                if isinstance(model, SpatialTranscriptFormer):
                    outputs = model(images, rel_coords=rel_coords)
                else:
                    outputs = model(images)
                
                loss = criterion(outputs, targets)
                
                if sparsity_lambda > 0 and hasattr(model, 'get_sparsity_loss'):
                    loss = loss + (sparsity_lambda * model.get_sparsity_loss())
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
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

def get_sample_ids(data_dir, precomputed=False, backbone='resnet50', max_samples=None):
    """
    Find and filter HEST sample IDs based on metadata and data availability.

    Args:
        data_dir (str): Root directory of HEST data.
        precomputed (bool): Whether to filter for samples with precomputed features.
        backbone (str): Backbone name to use for finding precomputed features.
        max_samples (int): Maximum number of samples to return.

    Returns:
        list: List of filtered sample IDs.
    """
    # Check for patches subdirectory
    patches_dir = os.path.join(data_dir, 'patches')
    search_dir = patches_dir if os.path.isdir(patches_dir) else data_dir

    all_files = [f for f in os.listdir(search_dir) if f.endswith('.h5')]
    all_ids = [f.replace('.h5', '') for f in all_files]
    
    if not all_ids:
        raise ValueError(f"No .h5 files found in {search_dir}")
        
    # Load metadata to filter samples
    metadata_path = os.path.join(data_dir, "HEST_v1_3_0.csv")
    if os.path.exists(metadata_path):
        import pandas as pd
        df = pd.read_csv(metadata_path)
        available_ids = set(all_ids)
        
        # Filter for existing files and Homo sapiens
        df_filtered = df[df['id'].isin(available_ids)]
        df_human = df_filtered[df_filtered['species'] == 'Homo sapiens']
        human_ids = df_human['id'].tolist()
        
        # Filter for Human Bowel
        df_bowel = df_human[df_human['organ'].str.contains('Bowel', case=False, na=False)]
        
        if not df_bowel.empty:
            print(f"Filtering for Human Bowel samples ({len(df_bowel)} found)...")
            final_ids = df_bowel['id'].tolist()
        else:
            print("No Human Bowel samples found, falling back to all Human samples...")
            final_ids = human_ids
            
        if not final_ids:
             print("Warning: No Human samples found. Using all files.")
             final_ids = all_ids
    else:
        print("Metadata not found, using all files.")
        final_ids = all_ids

    # Filter for precomputed features
    if precomputed:
        feat_dir_name = 'he_features' if backbone == 'resnet50' else f"he_features_{backbone}"
        feat_dir = os.path.join(data_dir, feat_dir_name)
        if not os.path.exists(feat_dir):
             feat_dir = os.path.join(data_dir, 'patches', feat_dir_name)
        
        if os.path.exists(feat_dir):
            available_pts = set(f.replace('.pt', '') for f in os.listdir(feat_dir) if f.endswith('.pt'))
            original_count = len(final_ids)
            final_ids = [fid for fid in final_ids if fid in available_pts]
            print(f"Refined to {len(final_ids)}/{original_count} based on features in {feat_dir}")
        else:
            print(f"Warning: Feature directory '{feat_dir_name}' not found.")

    if max_samples is not None:
        final_ids = final_ids[:max_samples]

    return final_ids

def setup_dataloaders(args, train_ids, val_ids):
    """
    Create training and validation dataloaders.

    Args:
        args (argparse.Namespace): CLI arguments.
        train_ids (list): IDs for training.
        val_ids (list): IDs for validation.
        augment (bool): Whether to enable augmentations.

    Returns:
        tuple: (train_loader, val_loader)
    """
    if args.precomputed:
        feat_dir_name = 'he_features' if args.backbone == 'resnet50' else f"he_features_{args.backbone}"
        feat_dir = os.path.join(args.data_dir, feat_dir_name)
        if not os.path.exists(feat_dir):
             feat_dir = os.path.join(args.data_dir, 'patches', feat_dir_name)
             
        if args.whole_slide:
            train_loader = get_hest_feature_dataloader(
                args.data_dir, train_ids, batch_size=args.batch_size, shuffle=True,
                num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                whole_slide_mode=True, augment=args.augment, feature_dir=feat_dir
            )
            val_loader = get_hest_feature_dataloader(
                args.data_dir, val_ids, batch_size=args.batch_size, shuffle=False,
                num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                whole_slide_mode=True, augment=False, feature_dir=feat_dir
            )
        else:
            train_loader = get_hest_feature_dataloader(
                args.data_dir, train_ids, batch_size=args.batch_size, shuffle=True,
                num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                use_global_context=args.use_global_context, global_context_size=args.global_context_size,
                augment=args.augment, feature_dir=feat_dir
            )
            val_loader = get_hest_feature_dataloader(
                args.data_dir, val_ids, batch_size=args.batch_size, shuffle=False,
                num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                use_global_context=args.use_global_context, global_context_size=args.global_context_size,
                augment=False, feature_dir=feat_dir
            )
    else:
        from torchvision import transforms
        
        # Base normalization
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        train_transform = norm
        if args.augment:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                norm
            ])
            print("Enabled image augmentations for training.")
            
        val_transform = norm
        
        if args.use_global_context:
            print("Warning: Global context only supported with pre-computed features.")
        train_loader = get_hest_dataloader(args.data_dir, train_ids, batch_size=args.batch_size, shuffle=True, 
                                          num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                                          transform=train_transform)
        val_loader = get_hest_dataloader(args.data_dir, val_ids, batch_size=args.batch_size, shuffle=False, 
                                        num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                                        transform=val_transform)
        
    return train_loader, val_loader

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

def run_inference_plot(model, args, sample_id, epoch, device):
    """
    Run inference on a single sample and save a visualization plot.

    Args:
        model (nn.Module): The model to use.
        args (argparse.Namespace): CLI arguments.
        sample_id (str): ID of the sample to plot.
        epoch (int): Current epoch (for filename).
        device (torch.device): Device to run inference on.
    """
    try:
        with torch.no_grad():
            model.eval()
            
            # Setup paths for this sample
            patches_dir = os.path.join(args.data_dir, 'patches') if os.path.isdir(os.path.join(args.data_dir, 'patches')) else args.data_dir
            st_dir = os.path.join(args.data_dir, 'st')
            h5_path = os.path.join(patches_dir, f"{sample_id}.h5")
            h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
            
            with h5py.File(h5_path, 'r') as f:
                patch_barcodes = f['barcode'][:].flatten()
                coords = f['coords'][:]
            
            gene_matrix, mask, gene_names = load_gene_expression_matrix(h5ad_path, patch_barcodes, num_genes=args.num_genes)
            
            preds = []
            if args.precomputed:
                feat_dir_name = 'he_features' if args.backbone == 'resnet50' else f"he_features_{args.backbone}"
                feature_path = os.path.join(args.data_dir, feat_dir_name, f"{sample_id}.pt")
                if not os.path.exists(feature_path):
                     feature_path = os.path.join(args.data_dir, 'patches', feat_dir_name, f"{sample_id}.pt")
                     
                ds = HEST_FeatureDataset(feature_path, h5ad_path, num_genes=args.num_genes, n_neighbors=args.n_neighbors, whole_slide_mode=args.whole_slide)
                
                if args.whole_slide:
                    feats, _, _ = ds[0]
                    feats = feats.unsqueeze(0).to(device)
                    if hasattr(model, 'forward_dense'):
                        preds.append(model.forward_dense(feats).detach().cpu().squeeze(0))
                else:
                    dl = DataLoader(ds, batch_size=32, shuffle=False)
                    for feats, _, rel_coords_batch in dl:
                        if isinstance(model, SpatialTranscriptFormer):
                            p = model(feats.to(device), rel_coords=rel_coords_batch.to(device)).cpu()
                        else:
                            p = model(feats.to(device)).cpu()
                        preds.append(p)
            else:
                from spatial_transcript_former.data.dataset import HEST_Dataset
                coord_subset = coords[mask]
                ds = HEST_Dataset(h5_path, coord_subset, gene_matrix, indices=np.where(mask)[0])
                dl = DataLoader(ds, batch_size=32, shuffle=False)
                for imgs, _, rel_coords_batch in dl:
                    if isinstance(model, SpatialTranscriptFormer):
                        preds.append(model(imgs.to(device), rel_coords=rel_coords_batch.to(device)).cpu())
                    else:
                        preds.append(model(imgs.to(device)).cpu())

            if preds:
                preds = torch.cat(preds, dim=0).numpy()
                coord_mask = np.array(mask, dtype=bool)
                coord_subset = coords[coord_mask]
                
                plot_spatial_genes(coord_subset, gene_matrix, preds, gene_names[:5], sample_id, 
                                   save_path=os.path.join(args.output_dir, f"{sample_id}_epoch_{epoch+1}.png"))
                
                # Histology Overlay
                try:
                    if os.path.exists(h5ad_path):
                        with h5py.File(h5ad_path, 'r') as f:
                            if 'uns' in f and 'spatial' in f['uns']:
                                spatial = f['uns/spatial']
                                sample_key = list(spatial.keys())[0] if len(spatial.keys()) > 0 else None
                                if sample_key:
                                    img_group = spatial[sample_key]['images']
                                    img_key = 'downscaled_fullres' if 'downscaled_fullres' in img_group else list(img_group.keys())[0]
                                    img = img_group[img_key][:]
                                    
                                    scale_group = spatial[sample_key]['scalefactors']
                                    scale_key = 'tissue_downscaled_fullres_scalef' if 'tissue_downscaled_fullres_scalef' in scale_group else list(scale_group.keys())[0]
                                    scalef = scale_group[scale_key][()]
                                    
                                    save_overlay_path = os.path.join(args.output_dir, f"{sample_id}_overlay_epoch_{epoch+1}.png")
                                    plot_histology_overlay(img, coord_subset, preds, gene_names, sample_id, scalef=scalef, save_path=save_overlay_path)
                except Exception as e_overlay:
                    print(f"Warning: Could not create overlay: {e_overlay}")
    except Exception as e:
        print(f"Warning: Failed to generate validation plot: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train Spatial TranscriptFormer Baselines")
    # Data arguments
    group_data = parser.add_argument_group('Data')
    group_data.add_argument('--data-dir', type=str, required=True, help='Root directory of HEST data')
    group_data.add_argument('--num-genes', type=int, default=1000, help='Number of genes to predict')
    group_data.add_argument('--max-samples', type=int, default=None, help='Limit sample count for debugging')
    group_data.add_argument('--precomputed', action='store_true', help='Use pre-computed features')
    group_data.add_argument('--whole-slide', action='store_true', help='Dense prediction on whole slide')
    
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
    group_train.add_argument('--lr', type=float, default=1e-4)
    group_train.add_argument('--sparsity-lambda', type=float, default=0.0, help='L1 penalty')
    group_train.add_argument('--augment', action='store_true', help='Enable augmentations')
    group_train.add_argument('--use-amp', action='store_true', help='Use Mixed Precision Training')
    group_train.add_argument('--output-dir', type=str, default='./checkpoints')
    group_train.add_argument('--compile', action='store_true')
    
    # Advanced arguments
    group_adv = parser.add_argument_group('Advanced')
    group_adv.add_argument('--n-neighbors', type=int, default=0, help='Context neighbors')
    group_adv.add_argument('--use-global-context', action='store_true')
    group_adv.add_argument('--global-context-size', type=int, default=128)
    group_adv.add_argument('--compile-backend', type=str, default='inductor')
    group_adv.add_argument('--masked-quadrants', type=str, nargs='+', default=None)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. Training Loop
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    log_exists = os.path.exists(log_path)
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            sparsity_lambda=args.sparsity_lambda, whole_slide=args.whole_slide,
            scaler=scaler
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
            
        # Save Latest
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'latest_model_{args.model}.pth'))
        
        # Periodic visualization
        if val_ids:
            run_inference_plot(model, args, val_ids[0], epoch, device)

if __name__ == '__main__':
    main()
