"""Simple PyTorch MLP Regressor for Diagnostic Bottleneck Analysis.

Flattens node feature matrix (64 nodes x 10 features = 640 dimensions)
and completely ignores graph connectivity edges.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLPRegressor(nn.Module):
    """Simple 3-layer MLP Regressor operating on flattened node features."""

    def __init__(self, in_features: int = 640, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Flattened node feature tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor:
            Continuous scalar prediction tensor of shape (batch_size,).
        """
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.fc3(x).squeeze(-1)
        return out
