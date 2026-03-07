"""
Training subpackage for SpatialTranscriptFormer.

Exposes the high-level :class:`Trainer` and the lower-level building blocks.
"""

from .trainer import Trainer, TrainerCallback, EarlyStoppingCallback
from .engine import train_one_epoch, validate
from .losses import (
    CompositeLoss,
    PCCLoss,
    MaskedMSELoss,
    MaskedHuberLoss,
    ZINBLoss,
    AuxiliaryPathwayLoss,
)
from .experiment_logger import ExperimentLogger

__all__ = [
    # High-level
    "Trainer",
    "TrainerCallback",
    "EarlyStoppingCallback",
    # Engine
    "train_one_epoch",
    "validate",
    # Losses
    "CompositeLoss",
    "PCCLoss",
    "MaskedMSELoss",
    "MaskedHuberLoss",
    "ZINBLoss",
    "AuxiliaryPathwayLoss",
    # Logging
    "ExperimentLogger",
]
