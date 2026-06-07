"""
CNN + Transformer-Lite model for multimodal biomedical monitoring.

This repo expects this module to exist (e.g., `test_accuracy_4_cases.py`,
`scripts/export_weights_esp32.py`, `scripts/convert_to_tflite.py`).

Notes
-----
- The embedded deployment uses a simplified CNN inference graph (no transformer).
  The export scripts extract weights for the CNN + projection + task heads.
- Layer naming/structure is chosen to match the export script expectations:
  - `self.cnn_layers` is an `nn.Sequential` where conv/bn layers appear at
    indices 0/1, 4/5, 8/9 respectively.
  - `self.channel_projection` is a `Linear(32 -> d_model)`.
  - `self.task_heads` is a ModuleDict with keys: activity/stress/arrhythmia,
    each a 2-layer MLP where Linear layers are at indices 0 and 3.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


def _sinusoidal_positional_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Non-trainable sinusoidal positional encoding. Shape: [1, seq_len, d_model]."""
    if seq_len <= 0:
        return torch.zeros(1, 0, d_model, device=device)
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, S, D]


class CNNTransformerLite(nn.Module):
    """
    Lightweight multi-task model.

    Input:  x in shape [B, C, T], where default C=11, T=1000.
    Output: (activity_logits, stress_logits, arrhythmia_logits)
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
        use_transformer: bool = True,
    ):
        super().__init__()

        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.d_model = int(d_model)
        self.use_transformer = bool(use_transformer) and int(num_layers) > 0

        # CNN front-end (keeps length with padding; matches embedded spec)
        # 1000 -> pool2 -> 500 -> pool2 -> 250
        self.cnn_layers = nn.Sequential(
            # idx 0/1 expected by export script
            nn.Conv1d(self.n_channels, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # idx 4/5 expected by export script
            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # idx 8/9 expected by export script
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=max(0.0, float(dropout))),
        )

        # Projection used by both training-time and embedded export
        self.channel_projection = nn.Linear(32, self.d_model)
        self.proj_act = nn.ReLU(inplace=True)
        self.proj_drop = nn.Dropout(p=max(0.0, float(dropout)))

        if self.use_transformer:
            # Prefer batch_first=True when available (PyTorch >= 1.9)
            try:
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    batch_first=True,
                    activation="relu",
                    norm_first=True,
                )
            except TypeError:
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    activation="relu",
                )
            self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        else:
            self.transformer_encoder = None

        # Task heads (structure expected by export script)
        hidden = 32
        self.task_heads = nn.ModuleDict(
            {
                "activity": nn.Sequential(
                    nn.Linear(self.d_model, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=max(0.0, float(dropout))),
                    nn.Linear(hidden, int(activity_classes)),
                ),
                "stress": nn.Sequential(
                    nn.Linear(self.d_model, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=max(0.0, float(dropout))),
                    nn.Linear(hidden, int(stress_classes)),
                ),
                "arrhythmia": nn.Sequential(
                    nn.Linear(self.d_model, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=max(0.0, float(dropout))),
                    nn.Linear(hidden, int(arrhythmia_classes)),
                ),
            }
        )

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input window to a single d_model embedding per sample.
        Returns: [B, d_model]
        """
        # x: [B, C, T]
        feat = self.cnn_layers(x)  # [B, 32, 250] (for default input)

        if not self.use_transformer or self.transformer_encoder is None:
            # Embedded-style path: global average pool then projection
            pooled = feat.mean(dim=-1)  # [B, 32]
            z = self.channel_projection(pooled)  # [B, d_model]
            z = self.proj_act(z)
            z = self.proj_drop(z)
            return z

        # Training-time path: project per timestep then transformer over time
        seq = feat.transpose(1, 2)  # [B, S, 32] where S ~ 250
        seq = self.channel_projection(seq)  # [B, S, d_model]
        seq = self.proj_act(seq)
        seq = self.proj_drop(seq)

        pe = _sinusoidal_positional_encoding(seq_len=seq.shape[1], d_model=self.d_model, device=seq.device)
        seq = seq + pe

        enc = self.transformer_encoder(seq)  # [B, S, d_model] (batch_first) or [S,B,D]
        if enc.dim() == 3 and enc.shape[0] == seq.shape[0]:
            # batch_first
            return enc.mean(dim=1)
        # fallback (not batch_first)
        return enc.mean(dim=0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass (repo default).

        Many scripts in this repo expect a dict output with keys:
        `activity`, `stress`, `arrhythmia`.
        """
        z = self._embed(x)
        return {
            "activity": self.task_heads["activity"](z),
            "stress": self.task_heads["stress"](z),
            "arrhythmia": self.task_heads["arrhythmia"](z),
        }

    def forward_tuple(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning a fixed tuple (activity, stress, arrhythmia).

        Use this for export paths (e.g., ONNX) that expect ordered outputs.
        """
        out = self.forward(x)
        return out["activity"], out["stress"], out["arrhythmia"]

