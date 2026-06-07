"""
Legacy model architecture that matches the saved checkpoint structure.

This module provides compatibility with previously trained models that have
a different architecture than the current cnn_transformer_lite.py
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CustomTransformerLayer(nn.Module):
    """Custom transformer layer matching saved checkpoint structure."""

    def __init__(self, d_model: int = 64, nhead: int = 4, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class CustomTransformer(nn.Module):
    """Custom transformer matching saved checkpoint structure."""

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        # Single layer components (legacy)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Stacked layers
        self.layers = nn.ModuleList([
            CustomTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class LegacyCNNTransformerLite(nn.Module):
    """
    Legacy model architecture matching saved checkpoint.

    This model matches the structure of comprehensive_trained_model.pth
    """

    def __init__(
        self,
        n_channels: int = 11,
        n_samples: int = 1000,
        activity_classes: int = 8,
        stress_classes: int = 4,
        arrhythmia_classes: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.d_model = d_model

        # CNN layers matching checkpoint structure
        # idx 0: Conv, idx 1: BN, idx 5: Conv, idx 6: BN, idx 10: Conv, idx 11: BN
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3),  # 0
            nn.BatchNorm1d(16),  # 1
            nn.ReLU(),  # 2
            nn.MaxPool1d(2),  # 3
            nn.Dropout(dropout),  # 4
            nn.Conv1d(16, 32, kernel_size=5, padding=2),  # 5
            nn.BatchNorm1d(32),  # 6
            nn.ReLU(),  # 7
            nn.MaxPool1d(2),  # 8
            nn.Dropout(dropout),  # 9
            nn.Conv1d(32, 32, kernel_size=3, padding=1),  # 10
            nn.BatchNorm1d(32),  # 11
            nn.ReLU(),  # 12
        )

        # Channel projection
        self.channel_projection = nn.Linear(32, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = PositionalEncoding(d_model, max_len=n_samples)

        # Transformer
        self.transformer = CustomTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Task weights (learnable)
        self.task_weights = nn.ParameterDict({
            'activity': nn.Parameter(torch.tensor(1.0)),
            'stress': nn.Parameter(torch.tensor(1.0)),
            'arrhythmia': nn.Parameter(torch.tensor(1.0)),
        })

        # Task heads
        hidden = 32
        self.task_heads = nn.ModuleDict({
            'activity': nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, activity_classes)
            ),
            'stress': nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, stress_classes)
            ),
            'arrhythmia': nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, arrhythmia_classes)
            ),
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # CNN feature extraction
        feat = self.cnn_layers(x)  # [B, 32, T']

        # Project and reshape for transformer
        feat = feat.transpose(1, 2)  # [B, T', 32]
        feat = self.channel_projection(feat)  # [B, T', d_model]

        # Add positional encoding
        feat = self.pos_encoding(feat)

        # Transformer
        feat = self.transformer(feat)  # [B, T', d_model]

        # Global average pooling
        pooled = feat.mean(dim=1)  # [B, d_model]

        # Task predictions
        return {
            'activity': self.task_heads['activity'](pooled),
            'stress': self.task_heads['stress'](pooled),
            'arrhythmia': self.task_heads['arrhythmia'](pooled),
        }

    def forward_tuple(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(x)
        return out['activity'], out['stress'], out['arrhythmia']


def load_legacy_model(model_path: str, device: torch.device = None) -> LegacyCNNTransformerLite:
    """Load a legacy model from checkpoint."""
    if device is None:
        device = torch.device('cpu')

    model = LegacyCNNTransformerLite()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
