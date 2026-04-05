"""
Public-facing checkpoint serialization for SpatialTranscriptFormer.

Saves and loads a self-contained checkpoint directory containing:
    - config.json    — architecture hyper-parameters
    - model.pth      — model weights (state_dict)
    - pathway_names.json — ordered list of gene symbols (optional)
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch

# Keys serialized into config.json.  These correspond to
# SpatialTranscriptFormer.__init__ arguments (minus runtime-only
# arguments like ``pathway_init`` and ``pretrained``).
_CONFIG_KEYS = [
    "num_pathways",
    "backbone_name",
    "token_dim",
    "n_heads",
    "n_layers",
    "dropout",
    "use_spatial_pe",
    "interactions",
]


def _model_config(model) -> Dict[str, Any]:
    """Extract serializable architecture config from a live model."""
    from spatial_transcript_former.models.interaction import (
        SpatialTranscriptFormer,
    )

    if not isinstance(model, SpatialTranscriptFormer):
        raise TypeError(f"Expected SpatialTranscriptFormer, got {type(model).__name__}")

    # Reconstruct config from the live model's attributes / constructor args.
    num_pathways = model.num_pathways
    token_dim = model.image_proj.out_features
    backbone_name = _infer_backbone_name(model)

    # Transformer encoder introspection
    first_layer = model.fusion_engine.layers[0]
    n_heads = first_layer.self_attn.num_heads
    n_layers = len(model.fusion_engine.layers)
    dropout = first_layer.dropout.p if hasattr(first_layer, "dropout") else 0.1

    use_spatial_pe = model.use_spatial_pe
    interactions = sorted(model.interactions)

    return {
        "num_pathways": num_pathways,
        "backbone_name": backbone_name,
        "token_dim": token_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "use_spatial_pe": use_spatial_pe,
        "interactions": interactions,
    }


def _infer_backbone_name(model) -> str:
    """Best-effort inference of backbone name from stored attribute or class."""
    # If we explicitly stored it (set by from_pretrained or user code):
    if hasattr(model, "_backbone_name"):
        return model._backbone_name
    # Fallback: inspect the backbone module's class name
    backbone_cls = type(model.backbone).__name__.lower()
    if "resnet" in backbone_cls:
        return "resnet50"
    if "ctrans" in backbone_cls:
        return "ctranspath"
    if "phikon" in backbone_cls or "dinov2" in backbone_cls:
        return "phikon"
    return "unknown"


# ── Public API ────────────────────────────────────────────────────────


def save_pretrained(
    model,
    save_dir: str,
    pathway_names: Optional[List[str]] = None,
) -> None:
    """Save a SpatialTranscriptFormer checkpoint directory.

    Creates ``save_dir`` containing:
        - ``config.json``     — architecture parameters
        - ``model.pth``       — ``state_dict``
        - ``pathway_names.json`` — ordered pathway names (if provided)

    Args:
        model: A :class:`SpatialTranscriptFormer` instance.
        save_dir: Directory to write files into (created if needed).
        pathway_names: Optional ordered list of pathway names matching the
            model's ``num_genes`` output dimension.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Config
    config = _model_config(model)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 2. Weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    # 3. Pathway names (optional)
    if pathway_names is not None:
        if len(pathway_names) != config["num_pathways"]:
            raise ValueError(
                f"pathway_names length ({len(pathway_names)}) does not match "
                f"model num_pathways ({config['num_pathways']})"
            )
        with open(os.path.join(save_dir, "pathway_names.json"), "w") as f:
            json.dump(pathway_names, f)

    print(f"Saved pretrained checkpoint to {save_dir}")


def load_pretrained(
    checkpoint_dir: str,
    device: str = "cpu",
    **override_kwargs,
):
    """Load a SpatialTranscriptFormer from a pretrained checkpoint directory.

    Reads ``config.json`` to reconstruct architectural parameters, then
    loads ``model.pth`` weights and (optionally) ``pathway_names.json``.

    Args:
        checkpoint_dir: Path to directory containing ``config.json`` and
            ``model.pth``.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        **override_kwargs: Override any config values (e.g. ``dropout=0.0``
            for deterministic inference).

    Returns:
        SpatialTranscriptFormer: The loaded model in eval mode with
            ``gene_names`` attribute set (or ``None``).
    """
    from spatial_transcript_former.models.interaction import (
        SpatialTranscriptFormer,
    )

    config_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "model.pth")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"config.json not found in {checkpoint_dir}. "
            "Use save_pretrained() to create a valid checkpoint directory."
        )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"model.pth not found in {checkpoint_dir}.")

    # 1. Read config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Apply any user overrides
    config.update(override_kwargs)

    # Don't load pretrained backbone weights — we're loading our own
    config["pretrained"] = False

    # 2. Instantiate
    model = SpatialTranscriptFormer(**config)

    # Store backbone name for future save_pretrained round-trips
    model._backbone_name = config.get("backbone_name", "unknown")

    # 3. Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 4. Pathway names (optional)
    pathway_names_path = os.path.join(checkpoint_dir, "pathway_names.json")
    if os.path.isfile(pathway_names_path):
        with open(pathway_names_path, "r") as f:
            model.pathway_names = json.load(f)
    else:
        model.pathway_names = None

    print(
        f"Loaded SpatialTranscriptFormer from {checkpoint_dir} "
        f"({config['num_pathways']} pathways)"
    )
    return model
