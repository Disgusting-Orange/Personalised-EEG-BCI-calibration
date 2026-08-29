"""PyTorch Geometric Graph Attention Network (GATv2) Regressor for Subject-Level EEG Motor Imagery Regression.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool


class GATRegressor(nn.Module):
    """Publication-quality PyG Graph Attention Network (GATv2) Regressor."""

    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 32,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()
        self.dropout = dropout
        self.pooling_type = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=1, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Final Conv layer
        if num_layers > 1:
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        out_dim = hidden_channels if num_layers > 1 else hidden_channels * heads

        # Regression Head
        self.fc1 = nn.Linear(out_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph Readout Pooling
        if self.pooling_type == "add":
            h_graph = global_add_pool(x, batch)
        elif self.pooling_type == "max":
            h_graph = global_max_pool(x, batch)
        else:
            h_graph = global_mean_pool(x, batch)

        # MLP Head
        h = F.relu(self.fc1(h_graph))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.fc2(h).squeeze(-1)

        return out
