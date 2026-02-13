import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

# Import get_backbone from your package
# Assuming this script is run as a module or with PYTHONPATH set correctly
try:
    from spatial_transcript_former.models.backbones import get_backbone
except ImportError:
    # Fallback for running as script from root
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from spatial_transcript_former.models.backbones import get_backbone

class PatchInferenceDataset(Dataset):
    """
    Simple dataset to load all patches from an H5 file for inference.
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        # Lazy loading
        self.h5_file = None
        self.coords = None
        self.barcodes = None
        self.length = 0
        
        # Open once to get length
        with h5py.File(h5_path, 'r') as f:
            if 'img' in f:
                self.length = f['img'].shape[0]
            
    def _open_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.coords = self.h5_file['coords'][:]
            self.barcodes = self.h5_file['barcode'][:].flatten()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self._open_file()
            
        img_data = self.h5_file['img'][idx] # (H, W, 3) or (H, W, C)
        
        # Convert to PIL/Tensor
        # HEST images are usually uint8 (H, W, 3)
        img_tensor = torch.from_numpy(img_data).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        coord = torch.tensor(self.coords[idx], dtype=torch.float32)
        
        # Barcode handling
        bc = self.barcodes[idx]
        if isinstance(bc, bytes):
            bc = bc.decode('utf-8')
        else:
            bc = str(bc)
            
        return img_tensor, coord, bc

def extract_features_for_slide(model, h5_path, output_path, batch_size=32, device='cuda', num_workers=4, transform=None):
    """
    Runs inference on a single slide and saves features.
    """
    # Validation
    if os.path.exists(output_path):
        print(f"Skipping {os.path.basename(h5_path)}, already exists.")
        return

    if transform is None:
        # Fallback to standard ImageNet mean/std
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
    
    dataset = PatchInferenceDataset(h5_path, transform=transform)
    # Note: Barcodes are strings, so default_collate might fail or stack them as list. 
    # We'll handle barcodes separately if needed, or just let DataLoader return a tuple/list of strings.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    all_features = []
    all_coords = []
    all_barcodes = []
    
    model.eval()
    with torch.no_grad():
        for imgs, coords, barcodes in tqdm(loader, desc=f"Processing {os.path.basename(h5_path)}", leave=False):
            imgs = imgs.to(device)
            
            # Forward pass
            features = model(imgs) # (B, D)
            
            all_features.append(features.cpu())
            all_coords.append(coords)
            all_barcodes.extend(barcodes)
            
    # Concatenate
    if not all_features:
        print(f"Warning: No features extracted for {h5_path}")
        return

    all_features = torch.cat(all_features, dim=0)
    all_coords = torch.cat(all_coords, dim=0)
    
    # Save to .pt
    # Include barcodes to ensure alignment
    torch.save({
        'features': all_features,
        'coords': all_coords,
        'barcodes': all_barcodes
    }, output_path)
    
    print(f"Saved {all_features.shape[0]} features to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract Features from HEST patches")
    parser.add_argument('--data-dir', type=str, required=True, help='Root directory containing patches/ subdirectory')
    parser.add_argument('--output-dir', type=str, default='he_features', help='Output directory for feature files')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model name')
    parser.add_argument('--batch-size', type=int, default=128, help='Inference batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num workers for dataloader')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of slides to process (for testing)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine default output based on backbone
    if args.output_dir == 'he_features':
        # Auto-name output dir if default is used
        if args.backbone != 'resnet50':
            args.output_dir = f"he_features_{args.backbone}"
    
    # Setup paths
    patches_dir = os.path.join(args.data_dir, 'patches')
    if not os.path.exists(patches_dir):
        # Maybe data-dir is already the patches dir
        if os.path.basename(args.data_dir) == 'patches':
            patches_dir = args.data_dir
        else:
            raise ValueError(f"Could not find patches directory at {patches_dir}")
            
    # Output dir
    # If output_dir is relative, make it relative to data_dir? 
    # Or just use the arg as is. Let's make it relative to data_dir if it's just a name
    if not os.path.isabs(args.output_dir):
        full_output_dir = os.path.join(args.data_dir, args.output_dir)
    else:
        full_output_dir = args.output_dir
        
    os.makedirs(full_output_dir, exist_ok=True)
    print(f"features will be saved to: {full_output_dir}")
    
    # Load Model
    print(f"Loading backbone: {args.backbone}...")
    model, dim = get_backbone(args.backbone, pretrained=True)
    model.to(device)
    model.eval()
    
    # Iterate slides
    h5_files = [f for f in os.listdir(patches_dir) if f.endswith('.h5')]
    if args.limit:
        h5_files = h5_files[:args.limit]
        
    print(f"Found {len(h5_files)} slides to process.")
    
    # Backbone-specific normalization
    if args.backbone == 'ctranspath':
        # CTransPath often uses slightly different normalization or none 
        # but original code often uses mean=0.485, 0.456, 0.406, std=0.229, 0.224, 0.225
        # However, some implementations use [0.5, 0.5, 0.5]
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
    elif 'hibou' in args.backbone or 'phikon' in args.backbone:
        # DINOv2 / iBOT models usually use standard ImageNet
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
    else:
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])

    for filename in tqdm(h5_files, desc="Total Progress"):
        file_path = os.path.join(patches_dir, filename)
        output_name = filename.replace('.h5', '.pt')
        output_path = os.path.join(full_output_dir, output_name)
        
        try:
            extract_features_for_slide(
                model, 
                file_path, 
                output_path, 
                batch_size=args.batch_size, 
                device=device,
                num_workers=args.num_workers,
                transform=transform
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print("Done!")

if __name__ == '__main__':
    main()
