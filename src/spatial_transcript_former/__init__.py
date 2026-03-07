"""
SpatialTranscriptFormer — predict gene expression from histology.

Core public API::

    from spatial_transcript_former import (
        SpatialTranscriptFormer,  # the model
        Predictor,               # inference wrapper
        FeatureExtractor,        # backbone feature extraction
        Trainer,                 # high-level training orchestrator
        save_pretrained,         # checkpoint serialization
        inject_predictions,      # AnnData integration
    )
"""

from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.predict import (
    FeatureExtractor,
    Predictor,
    inject_predictions,
)
from spatial_transcript_former.checkpoint import save_pretrained, load_pretrained
from spatial_transcript_former.training.trainer import Trainer

__all__ = [
    "SpatialTranscriptFormer",
    "Predictor",
    "FeatureExtractor",
    "Trainer",
    "save_pretrained",
    "load_pretrained",
    "inject_predictions",
]
