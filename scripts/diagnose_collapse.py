"""
Diagnostic script for detecting model collapse in SpatialTranscriptFormer.

Loads a checkpoint and a sample, then checks:
1. Pathway token diversity (are all pathway embeddings collapsing to the same vector?)
2. Prediction variance (are all output genes/patches getting the same value?)
3. ZINB parameter health (are pi/mu/theta in sensible ranges?)
4. Gradient flow (are gradients reaching all layers?)
5. Attention entropy (are attention weights uniform or peaked?)

Usage:
    python scripts/diagnose_collapse.py --checkpoint runs/stf_interaction_zinb/best_model_interaction.pth --data-dir A:/hest_data
"""

import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.data.dataset import (
    HEST_FeatureDataset,
    load_global_genes,
)
from spatial_transcript_former.data.pathways import get_pathway_init, MSIGDB_URLS


def load_model(
    checkpoint_path, device, num_genes=1000, backbone="ctranspath", loss="zinb"
):
    """Load model from checkpoint."""
    gene_list_path = "global_genes.json"
    if not os.path.exists(gene_list_path):
        gene_list_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "global_genes.json",
        )
    with open(gene_list_path) as f:
        gene_list = json.load(f)

    urls = [MSIGDB_URLS["hallmarks"]]
    pathway_init, pathway_names = get_pathway_init(gene_list[:num_genes], gmt_urls=urls)
    num_pathways = len(pathway_names)

    model = SpatialTranscriptFormer(
        num_genes=num_genes,
        backbone_name=backbone,
        pretrained=False,
        num_pathways=num_pathways,
        pathway_init=pathway_init,
        output_mode="zinb" if loss == "zinb" else "counts",
    )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle torch.compile prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("_orig_mod.", "")
        new_state_dict[key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    return model, pathway_names


def check_pathway_diversity(model):
    """Check if pathway tokens have collapsed to the same vector."""
    pw = model.pathway_tokens.data.squeeze(0)  # (P, D)

    # Pairwise cosine similarity
    pw_norm = F.normalize(pw, dim=-1)
    sim_matrix = pw_norm @ pw_norm.T  # (P, P)

    # Off-diagonal similarities
    mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
    off_diag = sim_matrix[mask]

    mean_sim = off_diag.mean().item()
    max_sim = off_diag.max().item()
    min_sim = off_diag.min().item()

    # Check variance of each pathway token
    pw_std = pw.std(dim=-1)  # (P,)
    dead_pathways = (pw_std < 1e-6).sum().item()

    print("\n" + "=" * 60)
    print("1. PATHWAY TOKEN DIVERSITY")
    print("=" * 60)
    print(f"   Num pathways: {pw.shape[0]}, Token dim: {pw.shape[1]}")
    print(f"   Mean pairwise cosine sim: {mean_sim:.4f}")
    print(f"   Max pairwise cosine sim:  {max_sim:.4f}")
    print(f"   Min pairwise cosine sim:  {min_sim:.4f}")
    print(f"   Dead pathways (zero std): {dead_pathways}")

    if mean_sim > 0.9:
        print("   ⚠️  WARNING: Pathway tokens are highly similar — possible collapse!")
    elif mean_sim > 0.7:
        print("   ⚠️  CAUTION: Pathway tokens are somewhat similar")
    else:
        print("   ✅ Pathway tokens appear diverse")

    return mean_sim


def check_gene_reconstructor(model):
    """Check if gene_reconstructor weights are degenerate."""
    W = model.gene_reconstructor.weight.data  # (G, P)

    col_std = W.std(dim=0)  # Per-pathway variance across genes
    row_std = W.std(dim=1)  # Per-gene variance across pathways

    dead_pathways = (col_std < 1e-6).sum().item()
    dead_genes = (row_std < 1e-6).sum().item()

    # Check sparsity (fraction of near-zero weights)
    sparsity = (W.abs() < 1e-4).float().mean().item()

    print("\n" + "=" * 60)
    print("2. GENE RECONSTRUCTOR WEIGHTS")
    print("=" * 60)
    print(f"   Shape: {W.shape}")
    print(f"   Weight range: [{W.min().item():.4f}, {W.max().item():.4f}]")
    print(f"   Weight std: {W.std().item():.4f}")
    print(f"   Dead pathways (zero col std): {dead_pathways}/{W.shape[1]}")
    print(f"   Dead genes (zero row std):    {dead_genes}/{W.shape[0]}")
    print(f"   Sparsity (|w| < 1e-4):        {sparsity:.2%}")

    if dead_pathways > W.shape[1] * 0.5:
        print("   ⚠️  WARNING: >50% of pathways are dead in gene_reconstructor!")
    else:
        print("   ✅ Gene reconstructor weights appear healthy")


def check_predictions(model, feats, coords, device):
    """Run a forward pass and check prediction diversity."""
    model.eval()
    with torch.no_grad():
        feats_t = feats.unsqueeze(0).to(device)
        coords_t = coords.unsqueeze(0).to(device)

        output = model(
            feats_t, return_dense=True, rel_coords=coords_t, return_pathways=True
        )

        if isinstance(output, tuple):
            gene_output = output[0]
            pathway_scores = output[1]
        else:
            gene_output = output
            pathway_scores = None

    print("\n" + "=" * 60)
    print("3. PREDICTION DIVERSITY")
    print("=" * 60)

    if isinstance(gene_output, tuple):
        # ZINB output: (pi, mu, theta)
        pi, mu, theta = gene_output
        pi, mu, theta = pi.squeeze(0), mu.squeeze(0), theta.squeeze(0)

        print(f"   ZINB mode detected")
        print(f"   mu range:    [{mu.min().item():.4f}, {mu.max().item():.4f}]")
        print(f"   mu mean:     {mu.mean().item():.4f}")
        print(f"   mu std:      {mu.std().item():.4f}")
        print(
            f"   pi range:    [{pi.min().item():.4f}, {pi.max().item():.4f}]  (dropout probability)"
        )
        print(f"   pi mean:     {pi.mean().item():.4f}")
        print(
            f"   theta range: [{theta.min().item():.4f}, {theta.max().item():.4f}]  (dispersion)"
        )
        print(f"   theta mean:  {theta.mean().item():.4f}")

        # Spatial variance of mu (do predictions vary across patches?)
        mu_spatial_std = mu.std(dim=0).mean().item()  # avg per-gene spatial std
        print(f"\n   Spatial std of mu (per-gene avg): {mu_spatial_std:.6f}")

        # Per-patch variance (do predictions vary across genes?)
        mu_gene_std = mu.std(dim=1).mean().item()
        print(f"   Gene std of mu (per-patch avg):   {mu_gene_std:.6f}")

        if mu_spatial_std < 1e-4:
            print(
                "   ⚠️  WARNING: Almost zero spatial variation — model is predicting the same thing everywhere!"
            )
        elif mu_spatial_std < 1e-2:
            print("   ⚠️  CAUTION: Very low spatial variation")
        else:
            print("   ✅ Spatial variation appears present")

        if pi.mean().item() > 0.9:
            print(
                "   ⚠️  WARNING: pi is very high — model thinks almost everything is a dropout!"
            )
        elif pi.mean().item() < 0.01:
            print("   ✅ pi is low — model is not over-relying on zero inflation")
    else:
        preds = gene_output.squeeze(0)  # (N, G)
        print(
            f"   Prediction range: [{preds.min().item():.4f}, {preds.max().item():.4f}]"
        )
        spatial_std = preds.std(dim=0).mean().item()
        print(f"   Spatial std (per-gene avg): {spatial_std:.6f}")

        if spatial_std < 1e-4:
            print("   ⚠️  WARNING: Model is predicting the same thing everywhere!")

    if pathway_scores is not None:
        pw = pathway_scores.squeeze(0)  # (N, P)
        print(f"\n   Pathway scores shape: {pw.shape}")
        print(
            f"   Pathway scores range: [{pw.min().item():.4f}, {pw.max().item():.4f}]"
        )
        print(f"   Pathway scores std:   {pw.std().item():.4f}")

        # Per-pathway spatial variance
        pw_spatial_std = pw.std(dim=0)  # (P,)
        print(
            f"   Per-pathway spatial std: min={pw_spatial_std.min().item():.4f}, max={pw_spatial_std.max().item():.4f}, mean={pw_spatial_std.mean().item():.4f}"
        )

        dead_pw = (pw_spatial_std < 1e-6).sum().item()
        print(f"   Dead pathways (zero spatial var): {dead_pw}/{pw.shape[1]}")


def check_attention_interactions(model, feats, coords, device):
    """Analyze attention maps to see how different token groups are interacting."""
    model.eval()
    with torch.no_grad():
        feats_t = feats.unsqueeze(0).to(device)
        coords_t = coords.unsqueeze(0).to(device)

        # SpatialTranscriptFormer.forward returns (gene_expr, pathway_scores, attentions)
        # when return_pathways=True and return_attention=True
        results = model(
            feats_t,
            rel_coords=coords_t,
            return_pathways=True,
            return_attention=True,
        )

        # Unpack results: (expr, p_scores, attentions)
        if len(results) == 3:
            attentions = results[2]
        else:
            # Fallback for unexpected return signature (shouldn't happen with my changes)
            print(
                "   ⚠️  Could not extract attention maps (unexpected return signature)"
            )
            return

        p = model.num_pathways
        s = feats.shape[0]  # number of spots

        print("\n" + "=" * 60)
        print("5. ATTENTION INTERACTIONS")
        print("=" * 60)
        print(f"   Interactions enabled: {model.interactions}")

        for i, attn in enumerate(attentions):
            # attn is (B, T, T) where T = P + S
            attn = attn.squeeze(0)  # (T, T)

            # Interaction quadrants
            p2p_m = attn[:p, :p].mean().item()
            p2h_m = attn[:p, p:].mean().item()
            h2p_m = attn[p:, :p].mean().item()
            h2h_m = attn[p:, p:].mean().item()

            # Diagnostic for sparse attention: off-diagonal h2h
            h2h_off_diag = attn[p:, p:].clone()
            h2h_off_diag.fill_diagonal_(0)
            h2h_off_diag_mean = h2h_off_diag.mean().item()

            print(f"   LAYER {i}:")
            print(f"      p2p (Path -> Path): {p2p_m:.6f}")
            print(f"      p2h (Path -> Spot): {p2h_m:.6f}")
            print(f"      h2p (Spot -> Path): {h2p_m:.6f}")
            print(
                f"      h2h (Spot -> Spot): {h2h_m:.6f} (off-diag: {h2h_off_diag_mean:.6f})"
            )

            # Warnings for "dead" interaction types
            if h2p_m < 1e-4 and "h2p" in model.interactions:
                print(
                    f"      ⚠️  CAUTION: Spots are barely attending to pathways in Layer {i}"
                )
            if p2h_m < 1e-4 and "p2h" in model.interactions:
                print(
                    f"      ⚠️  CAUTION: Pathways are barely attending to spots in Layer {i}"
                )
            if h2h_off_diag_mean < 1e-7 and "h2h" in model.interactions:
                print(
                    f"      ⚠️  WARNING: Spot-to-spot attention is nearly zero despite being enabled!"
                )


def check_gradient_flow(model, feats, coords, device):
    """Check if gradients reach all parts of the model."""
    model.train()
    model.zero_grad()

    feats_t = feats.unsqueeze(0).to(device)
    coords_t = coords.unsqueeze(0).to(device)

    output = model(feats_t, return_dense=True, rel_coords=coords_t)

    if isinstance(output, tuple):
        loss = output[1].sum()  # mu
    else:
        loss = output.sum()

    loss.backward()

    print("\n" + "=" * 60)
    print("4. GRADIENT FLOW")
    print("=" * 60)

    layers = {
        "pathway_tokens": model.pathway_tokens,
        "image_proj.weight": model.image_proj.weight,
        "gene_reconstructor.weight": model.gene_reconstructor.weight,
        "fusion_engine.layers[0].self_attn.in_proj_weight": model.fusion_engine.layers[
            0
        ].self_attn.in_proj_weight,
    }

    if hasattr(model, "pi_reconstructor"):
        layers["pi_reconstructor.weight"] = model.pi_reconstructor.weight
        layers["theta_reconstructor.weight"] = model.theta_reconstructor.weight

    for name, param in layers.items():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_max = param.grad.abs().max().item()
            print(f"   {name:45s} grad_norm={grad_norm:.6f}  grad_max={grad_max:.6f}")
            if grad_norm < 1e-10:
                print(f"   ⚠️  DEAD gradient in {name}!")
        else:
            print(f"   {name:45s} ⚠️  NO GRADIENT!")


def main():
    parser = argparse.ArgumentParser(description="Diagnose model collapse")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--num-genes", type=int, default=1000)
    parser.add_argument("--backbone", type=str, default="ctranspath")
    parser.add_argument("--loss", type=str, default="zinb")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model, pathway_names = load_model(
        args.checkpoint, device, args.num_genes, args.backbone, args.loss
    )
    print(f"Loaded {len(pathway_names)} pathways")

    # Load a sample
    print("Loading sample data...")
    common_gene_names = load_global_genes(args.data_dir, args.num_genes)

    patches_dir = os.path.join(args.data_dir, "patches")
    if not os.path.isdir(patches_dir):
        patches_dir = args.data_dir
    st_dir = os.path.join(args.data_dir, "st")

    feat_dir = os.path.join(args.data_dir, f"patches/he_features_{args.backbone}")
    if not os.path.isdir(feat_dir):
        feat_dir = os.path.join(args.data_dir, f"he_features_{args.backbone}")

    # Find first available sample
    sample_files = [f for f in os.listdir(feat_dir) if f.endswith(".pt")]
    if not sample_files:
        print("ERROR: No feature files found!")
        sys.exit(1)

    sample_id = sample_files[0].replace(".pt", "")
    feature_path = os.path.join(feat_dir, sample_files[0])
    h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

    print(f"Using sample: {sample_id}")

    ds = HEST_FeatureDataset(
        feature_path,
        h5ad_path,
        num_genes=args.num_genes,
        selected_gene_names=common_gene_names,
        whole_slide_mode=True,
    )

    feats, targets, coords = ds[0]
    print(f"Features shape: {feats.shape}, Coords shape: {coords.shape}")

    # Run diagnostics
    check_pathway_diversity(model)
    check_gene_reconstructor(model)
    check_predictions(model, feats, coords, device)
    check_attention_interactions(model, feats, coords, device)
    check_gradient_flow(model, feats, coords, device)

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
