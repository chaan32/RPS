"""Temporal deep-learning risk model for Fusion V2."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalRiskPredictor(nn.Module):
    """GRU-based multi-label risk predictor.

    Input:
      x: (B, T, F), a BEV/world-coordinate feature sequence.

    Output:
      logits: (B, 2)
        [:, 0] = worker vs forklift risk
        [:, 1] = worker vs dropzone risk
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        out, _ = self.gru(h)
        return self.head(out[:, -1, :])
