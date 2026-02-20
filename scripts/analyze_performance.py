print("DEBUG: Script started (pre-imports)")
import argparse
import os
import sys
try:
    import torch
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from scipy.stats import pearsonr
    print("DEBUG: Standard imports successful")
except Exception as e:
    print(f"DEBUG: Import failed: {e}")
    sys.exit(1)

# Add src to path
sys.path.append(os.path.abspath('src'))
print(f"DEBUG: sys.path: {sys.path}")

try:
    print("DEBUG: Importing SpatialTranscriptFormer...")
    from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
    print("DEBUG: Importing HEST_FeatureDataset...")
    from spatial_transcript_former.data.dataset import HEST_FeatureDataset, load_global_genes
    print("DEBUG: Importing get_sample_ids...")
    from spatial_transcript_former.data.utils import get_sample_ids
    print("DEBUG: Importing pathways...")
    from spatial_transcript_former.data.pathways import get_pathway_init, MSIGDB_URLS
    print("DEBUG: Importing utils...")
    from spatial_transcript_former.utils import set_seed
    print("DEBUG: Custom imports successful")
except Exception as e:
    print(f"DEBUG: Custom import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def load_model(checkpoint_path, args, device):
    """Load model and infer configuration."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Infer config
    if 'pathway_tokenizer.pathway_embeddings' in state_dict:
        _, num_pathways, token_dim = state_dict['pathway_tokenizer.pathway_embeddings'].shape
    else:
        num_pathways = 50
        token_dim = 512
    
    print(f"Inferred: {num_pathways} pathways, {token_dim} dim")

    model = SpatialTranscriptFormer(
        num_genes=args.num_genes,
        num_pathways=num_pathways,
        backbone_name=args.backbone,
        token_dim=token_dim,
        use_nystrom=args.use_nystrom,
        use_spatial_pe=True
    )
    
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Notes: Missing keys {len(missing)}")
    
    model.to(device)
    model.eval()
    return model, num_pathways

def evaluate_sample(model, sample_id, args, device, membership, gene_names, pathway_names):
    """Run inference and compute metrics for a single sample."""
    # Paths
    feat_dir_name = 'he_features' if args.backbone == 'resnet50' else f"he_features_{args.backbone}"
    feature_path = os.path.join(args.data_dir, feat_dir_name, f"{sample_id}.pt")
    if not os.path.exists(feature_path):
        feature_path = os.path.join(args.data_dir, 'patches', feat_dir_name, f"{sample_id}.pt")
    
    st_dir = os.path.join(args.data_dir, 'st')
    h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

    # Load Data
    try:
        ds = HEST_FeatureDataset(
            feature_path, h5ad_path, num_genes=args.num_genes,
            selected_gene_names=gene_names,
            whole_slide_mode=True,
            log1p=True 
        )
    except Exception as e:
        print(f"Skipping {sample_id}: {e}")
        return None

    feats, targets, coords = ds[0]
    feats = feats.unsqueeze(0).to(device)
    targets = targets.numpy() # (N, G)
    
    # Inference
    with torch.no_grad():
        preds, pathway_activations = model.forward_dense(feats, return_pathways=True)
    
    preds = preds.squeeze(0).cpu().numpy() # (N, G)
    pred_pathways = pathway_activations.squeeze(0).cpu().numpy() # (N, P)
    
    # DEBUG: Check Scale
    print(f"DEBUG: Sample {sample_id}")
    print(f"  Preds:   Min={preds.min():.4f}, Max={preds.max():.4f}, Mean={preds.mean():.4f}")
    print(f"  Targets: Min={targets.min():.4f}, Max={targets.max():.4f}, Mean={targets.mean():.4f}")
    
    # Compute Ground Truth Pathways: (N, G) @ (G, P) -> (N, P)
    truth_pathways = targets @ membership.T
    
    metrics = {}

    # --- Gene Level Metrics ---
    # MAE per gene
    mae = np.mean(np.abs(preds - targets), axis=0) # (G,)
    
    # Mean Expression
    mean_pred = np.mean(preds, axis=0)
    mean_truth = np.mean(targets, axis=0)
    
    # R2 per gene (skipping constant genes)
    r2 = []
    for i in range(preds.shape[1]):
        ss_res = np.sum((targets[:, i] - preds[:, i]) ** 2)
        ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        if ss_tot < 1e-6:
            r2.append(0.0)
        else:
            r2.append(1 - (ss_res / ss_tot))
    r2 = np.array(r2)

    metrics['genes'] = {
        'names': gene_names,
        'mae': mae,
        'mean_pred': mean_pred,
        'mean_truth': mean_truth,
        'r2': r2
    }

    # --- Pathway Level Metrics ---
    pcc_scores = {}
    for i, name in enumerate(pathway_names):
        if np.std(pred_pathways[:, i]) < 1e-6 or np.std(truth_pathways[:, i]) < 1e-6:
             pcc_scores[name] = 0.0
        else:
             r, _ = pearsonr(pred_pathways[:, i], truth_pathways[:, i])
             pcc_scores[name] = r
    
    metrics['pathways'] = pcc_scores
             
    return metrics

def main():
    print("Starting analysis script...")
    parser = argparse.ArgumentParser(description="Analyze Model Performance")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='A:\\hest_data')
    parser.add_argument('--backbone', type=str, default='ctranspath')
    parser.add_argument('--num-genes', type=int, default=1000)
    parser.add_argument('--use-nystrom', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='both', choices=['genes', 'pathways', 'both'])
    parser.add_argument('--pathway-init', action='store_true')
    parser.add_argument('--pathways', nargs='+', default=None)
    
    args = parser.parse_args()
    print(f"Arguments parsed. Checkpoint: {args.checkpoint}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    set_seed(args.seed)

    # 1. Setup Data Split (Validation Set)
    print("Getting sample IDs...")
    final_ids = get_sample_ids(args.data_dir, True, args.backbone, None)
    np.random.shuffle(final_ids)
    split_idx = int(len(final_ids) * 0.8)
    val_ids = final_ids[split_idx:]
    print(f"Analyzing {len(val_ids)} validation samples.")
    
    if len(val_ids) == 0:
        print("Error: No validation samples found.")
        return

    # 2. Load Resources (Genes, Pathways)
    print("Loading global genes...")
    try:
        gene_names = load_global_genes(args.data_dir, args.num_genes) # List[str]
        print(f"Loaded {len(gene_names)} genes.")
    except Exception as e:
        print(f"Error loading genes: {e}")
        return

    print("Loading pathways...")
    from spatial_transcript_former.data.pathways import get_pathway_init, MSIGDB_URLS
    urls = [MSIGDB_URLS['hallmarks'], MSIGDB_URLS['c2_kegg'], MSIGDB_URLS['c2_medicus'], MSIGDB_URLS['c2_cgp']]
    
    membership, pathway_names = get_pathway_init(
        gene_names,
        gmt_urls=urls,
        filter_names=args.pathways,
        cache_dir=os.path.join(args.data_dir, '.cache')
    )
    membership = membership.numpy() # (P, G)
    print(f"Built membership matrix: {membership.shape}")

    # 3. Load Model
    print("Loading model...")
    model, num_pathways = load_model(args.checkpoint, args, device)
    
    if num_pathways != membership.shape[0]:
        print(f"WARNING: Model was trained with {num_pathways} pathways but we parsed {membership.shape[0]}.")
        
    print("Model loaded.")

    # 4. Run Analysis
    gene_stats = {'mae': [], 'mean_pred': [], 'mean_truth': [], 'r2': []}
    pathway_stats = {name: [] for name in pathway_names}
    
    print("Evaluating...")
    for sid in tqdm(val_ids):
        metrics = evaluate_sample(model, sid, args, device, membership, gene_names, pathway_names)
        if metrics:
            # Aggregate Gene Stats
            if args.mode in ['genes', 'both']:
                gene_stats['mae'].append(metrics['genes']['mae'])
                gene_stats['mean_pred'].append(metrics['genes']['mean_pred'])
                gene_stats['mean_truth'].append(metrics['genes']['mean_truth'])
                gene_stats['r2'].append(metrics['genes']['r2'])
            
            # Aggregate Pathway Stats
            if args.mode in ['pathways', 'both']:
                for name, r in metrics['pathways'].items():
                    pathway_stats[name].append(r)

    # 5. Reporting
    if args.mode in ['genes', 'both']:
        # Average across samples -> Now we have (G,) arrays
        avg_mae = np.mean(np.array(gene_stats['mae']), axis=0)
        avg_pred = np.mean(np.array(gene_stats['mean_pred']), axis=0)
        avg_truth = np.mean(np.array(gene_stats['mean_truth']), axis=0)
        avg_r2 = np.mean(np.array(gene_stats['r2']), axis=0)
        
        df_genes = pd.DataFrame({
            'Gene': gene_names,
            'MAE': avg_mae,
            'Pred_Mean_Log1p': avg_pred,
            'Truth_Mean_Log1p': avg_truth,
            'Bias': avg_pred - avg_truth,
            'R2': avg_r2
        })
        df_genes = df_genes.sort_values('R2', ascending=False)
        
        print("\n" + "="*60)
        print("TOP 10 BEST PREDICTED GENES (by R2)")
        print("="*60)
        print(df_genes.head(10).to_string(index=False))
        
        print("\n" + "="*60)
        print("GENE EXPRESSION STATS (Global)")
        print("="*60)
        print(f"Mean MAE: {df_genes['MAE'].mean():.4f}")
        print(f"Mean R2:  {df_genes['R2'].mean():.4f}")
        print(f"Avg Bias: {df_genes['Bias'].mean():.4f} (Pos = Overestimation)")
        
        df_genes.to_csv(args.checkpoint.replace('.pth', '_genes.csv'), index=False)

    if args.mode in ['pathways', 'both']:
        results = []
        for name, scores in pathway_stats.items():
            if scores:
                mean_r = np.mean(scores)
                results.append({'Pathway': name, 'Mean_PCC': mean_r})
        
        df_path = pd.DataFrame(results).sort_values('Mean_PCC', ascending=False)
        print("\n" + "="*60)
        print("TOP 10 PATHWAYS (PCC)")
        print("="*60)
        print(df_path.head(10).to_string(index=False))
        df_path.to_csv(args.checkpoint.replace('.pth', '_pathways.csv'), index=False)


if __name__ == "__main__":
    main()

