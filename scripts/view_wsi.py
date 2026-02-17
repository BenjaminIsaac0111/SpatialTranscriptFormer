import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import sys
from scipy.spatial import KDTree
import torch

# Add src to path
sys.path.append(os.path.abspath('src'))
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.data.dataset import HEST_FeatureDataset, load_global_genes
from spatial_transcript_former.data.pathways import get_pathway_init

def load_histology(h5ad_path):
    """Load the downscaled histology image and scale factor."""
    try:
        with h5py.File(h5ad_path, 'r') as f:
            if 'uns' not in f or 'spatial' not in f['uns']:
                print("Error: 'uns/spatial' not found in h5ad.")
                return None, None
            spatial = f['uns/spatial']
            sample_key = list(spatial.keys())[0] if len(spatial.keys()) > 0 else None
            
            if sample_key is None:
                print("Error: No sample key found in spatial.")
                return None, None
                
            img_group = spatial[sample_key]['images']
            # Try keys in order of preference
            possible_keys = ['downscaled_fullres', 'hires', 'lowres']
            img_key = next((k for k in possible_keys if k in img_group), list(img_group.keys())[0])
            
            img = img_group[img_key][:]
            
            # Scalefactors
            scale_group = spatial[sample_key]['scalefactors']
            scale_key = f'tissue_{img_key}_scalef'
            if scale_key not in scale_group:
                 # Fallback
                 scale_key = list(scale_group.keys())[0]
            
            scalef = scale_group[scale_key][()]
            return img, scalef
    except Exception as e:
        print(f"Error loading histology: {e}")
        return None, None

def load_model_and_predict(checkpoint_path, args, sample_id, device):
    """Load model and return pathway activations for the sample."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Infer config (simplistic)
    if 'pathway_tokenizer.pathway_embeddings' in state_dict:
        _, num_pathways, token_dim = state_dict['pathway_tokenizer.pathway_embeddings'].shape
    else:
        num_pathways = 50
        token_dim = 512 # guess
    
    print(f"Inferred: {num_pathways} pathways, {token_dim} dim")

    # Reconstruct model (assuming default architecture options for now)
    # Ideally we'd pickle the args, but we'll use defaults + CLI overrides if we had them
    # For now, hardcode the most likely config or rely on the user to match?
    # We'll use the visualize_top_patches approach
    model = SpatialTranscriptFormer(
        num_genes=args.num_genes,
        num_pathways=num_pathways,
        backbone_name=args.backbone,
        token_dim=token_dim,
        use_nystrom=args.use_nystrom,
        fusion_mode='decoder', # Default
        use_spatial_pe=True # Default true now
    )
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Notes: Missing keys {len(missing)}")
    
    model.to(device)
    model.eval()

    # Data Loading for Features
    feat_dir_name = 'he_features' if args.backbone == 'resnet50' else f"he_features_{args.backbone}"
    feature_path = os.path.join(args.data_dir, feat_dir_name, f"{sample_id}.pt")
    if not os.path.exists(feature_path):
        feature_path = os.path.join(args.data_dir, 'patches', feat_dir_name, f"{sample_id}.pt")
    
    st_dir = os.path.join(args.data_dir, 'st')
    h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

    # Load global genes logic
    try:
        common_gene_names = load_global_genes(args.data_dir, args.num_genes)
    except:
        common_gene_names = None

    ds = HEST_FeatureDataset(
        feature_path, h5ad_path, num_genes=args.num_genes,
        selected_gene_names=common_gene_names,
        whole_slide_mode=True
    )

    feats, _, coords = ds[0]
    feats = feats.unsqueeze(0).to(device)
    coords = coords.unsqueeze(0).to(device)

    print("Running inference on whole slide...")
    with torch.no_grad():
        # inference without coords to match visualization.py behavior (avoids PE grid artifacts)
        _, pathway_activations = model.forward_dense(feats, return_pathways=True)
    
    # (1, N, P) -> (N, P)
    activations = pathway_activations.squeeze(0).cpu().numpy()
    
    # Get pathway names
    from spatial_transcript_former.data.pathways import download_hallmarks_gmt, parse_gmt
    gmt_path = download_hallmarks_gmt(os.path.join(args.data_dir, '.cache'))
    pathway_dict = parse_gmt(gmt_path)
    pathway_names = list(pathway_dict.keys())
    
    return activations, pathway_names

def main():
    parser = argparse.ArgumentParser(description="Interactive WSI Viewer for HEST")
    parser.add_argument('--data-dir', type=str, default='A:\\hest_data', help='Data directory')
    parser.add_argument('--sample-id', type=str, required=True, help='Sample ID (e.g., TENX29)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint for overlays')
    
    # Model args needed for reconstruction
    parser.add_argument('--backbone', type=str, default='ctranspath')
    parser.add_argument('--num-genes', type=int, default=1000)
    parser.add_argument('--use-nystrom', action='store_true')
    
    args = parser.parse_args()

    sample_id = args.sample_id
    patches_dir = os.path.join(args.data_dir, 'patches') if os.path.isdir(os.path.join(args.data_dir, 'patches')) else args.data_dir
    st_dir = os.path.join(args.data_dir, 'st')
    h5_path = os.path.join(patches_dir, f"{sample_id}.h5")
    h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

    if not os.path.exists(h5_path):
        print(f"Error: .h5 file not found at {h5_path}")
        return
    if not os.path.exists(h5ad_path):
        print(f"Error: .h5ad file not found at {h5ad_path}")
        return

    # 1. Load Downscaled Image
    print(f"Loading histology from {h5ad_path}...")
    hist_img, scalef = load_histology(h5ad_path)
    if hist_img is None:
        return
    
    # 2. Load Patch Coordinates
    print(f"Loading patch coordinates from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:] # (N, 2)
    
    # Scale coordinates
    scaled_coords = coords * scalef
    tree = KDTree(scaled_coords)

    # 3. Optional: Run Inference
    activations = None
    pathway_names = []
    if args.checkpoint:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            activations, pathway_names = load_model_and_predict(args.checkpoint, args, sample_id, device)
            print(f"Inference complete. Activations: {activations.shape}")
        except Exception as e:
            print(f"Error running inference: {e}")
            activations = None

    # 4. Setup Visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    # Adjust layout: Left 25% for pathways list, Right 75% for image
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.05)
    
    # Image Axes
    ax.imshow(hist_img)
    scatter = ax.scatter(scaled_coords[:, 0], scaled_coords[:, 1], s=5, c='r', alpha=0.3, label='Patches')
    ax.set_title(f"Sample: {sample_id} (Left-click to view patch)")
    ax.axis('off')

    h5_handle = h5py.File(h5_path, 'r')

    # 5. Controls
    if activations is not None:
        # Pathway Selector
        button_ax = plt.axes([0.02, 0.05, 0.25, 0.85]) 
        button_ax.set_title("Select Pathway", fontsize=10)
        
        display_names = ['(None)'] + sorted(pathway_names)
        
        from matplotlib.widgets import RadioButtons, CheckButtons, Slider
        radio = RadioButtons(button_ax, display_names, active=0)
        
        # Helper to style radio buttons if supported
        if hasattr(radio, 'circles'):
            for circle in radio.circles:
                circle.set_radius(0.005)
        for label in radio.labels:
            label.set_fontsize(8)

        # Smoothing Checkbox
        smooth_ax = plt.axes([0.02, 0.92, 0.25, 0.05])
        check = CheckButtons(smooth_ax, ['Smooth'], [False])

        # Alpha Slider
        alpha_ax = plt.axes([0.3, 0.02, 0.4, 0.03])
        alpha_slider = Slider(
            ax=alpha_ax,
            label='Alpha',
            valmin=0.0,
            valmax=1.0,
            valinit=0.6,
        )
        
        current_pathway = None
        contours = None
        
        def update_viz(val=None):
            nonlocal contours
            # Clear previous
            if contours:
                try:
                    for c in contours.collections:
                        c.remove()
                except AttributeError:
                    # Matplotlib 3.8+: ContourSet is a single Collection
                    contours.remove()
                contours = None
            
            # Reset scatter
            scatter.set_array(None)
            scatter.set_color('r')
            scatter.set_alpha(0.3)
            scatter.set_sizes([5] * len(scaled_coords))
            
            if current_pathway is None or current_pathway == '(None)':
                ax.set_title(f"Sample: {sample_id}")
                fig.canvas.draw_idle()
                return

            print(f"Visualizing: {current_pathway}")
            idx = pathway_names.index(current_pathway)
            vals = activations[:, idx]
            
            is_smooth = check.get_status()[0]
            alpha_val = alpha_slider.val
            
            if is_smooth:
                # Hide scatter (make it invisible but kept for clicking)
                scatter.set_visible(False)
                
                # Triangulate and Contour
                # We need to triangulate on the fly? or once?
                # Just let tricontourf handle it.
                cntr = ax.tricontourf(scaled_coords[:, 0], scaled_coords[:, 1], vals, 
                                      levels=20, cmap='jet', alpha=alpha_val)
                contours = cntr
            else:
                scatter.set_visible(True)
                scatter.set_array(vals)
                scatter.set_cmap('jet')
                scatter.set_alpha(alpha_val)
                scatter.set_sizes([15] * len(vals))
            
            ax.set_title(f"Sample: {sample_id} | Overlay: {current_pathway}")
            fig.canvas.draw_idle()

        def change_pathway(label):
            nonlocal current_pathway
            current_pathway = label
            update_viz()
            
        def toggle_smooth(label):
            update_viz()

        radio.on_clicked(change_pathway)
        check.on_clicked(toggle_smooth)
        alpha_slider.on_changed(update_viz)
        
    else:
        print("No activations loaded (no checkpoint), skipping pathway selector.")
        radio = None 
        check = None 
        alpha_slider = None 

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button != 1: 
            return
            
        click_coords = [event.xdata, event.ydata]
        dist, idx = tree.query(click_coords)
        
        pixel_radius = (256 * scalef) / 2
        
        if dist > pixel_radius * 2: 
            return
            
        print(f"Loading patch {idx} at {coords[idx]}...")
        patch = h5_handle['img'][idx]
        
        new_fig, new_ax = plt.subplots(figsize=(5, 5))
        new_ax.imshow(patch)
        
        title = f"Patch {idx}\nCoords: {coords[idx]}"
        
        # Check current selection from radio buttons if available
        if radio and radio.value_selected != '(None)':
             p_name = radio.value_selected
             if p_name in pathway_names:
                 p_idx = pathway_names.index(p_name)
                 score = activations[idx, p_idx]
                 title += f"\n{p_name}: {score:.4f}"
                
        new_ax.set_title(title)
        new_ax.axis('off')
        new_fig.show()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    print("Viewer launched.")
    print(" - Select a pathway from the list on the left.")
    print(" - Click dots to view patches.")
    plt.show()
    
    h5_handle.close()

if __name__ == "__main__":
    main()
