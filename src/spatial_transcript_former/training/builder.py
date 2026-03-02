import os
import torch
import torch.nn as nn
from spatial_transcript_former.models import HE2RNA, ViT_ST, SpatialTranscriptFormer
from spatial_transcript_former.training.losses import (
    PCCLoss,
    CompositeLoss,
    MaskedMSELoss,
    ZINBLoss,
)


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
