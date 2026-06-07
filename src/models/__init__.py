"""
Model package for multimodal biomedical monitoring.

Available models:
- CNNTransformerLite: Lightweight model for edge deployment (primary)
- CNNTransformerImproved: Enhanced model with SE blocks, residual connections

The primary training-time architecture is `CNNTransformerLite`, used for
multi-task prediction (activity / stress / arrhythmia).
"""

from .cnn_transformer_lite import CNNTransformerLite
from .cnn_transformer_improved import (
    CNNTransformerImproved,
    create_improved_model,
    MultiTaskLoss,
    FocalLoss,
    LabelSmoothingLoss,
)

__all__ = [
    "CNNTransformerLite",
    "CNNTransformerImproved",
    "create_improved_model",
    "MultiTaskLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
]

