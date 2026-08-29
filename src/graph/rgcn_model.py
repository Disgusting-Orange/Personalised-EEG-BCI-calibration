"""PyTorch Geometric Relational Graph Convolutional Network (RGCN) Regressor.

Processes multi-relational graphs with 5 frequency-band relations (Delta, Theta, Alpha, Beta, Gamma).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool

class RGCNRegressor(nn.Module):
    """PyG Relational GCN Regressor for continuous BCI target prediction."""

    def __init__(
        self,
        in_channels: int = 20,
        hidden_channels: int = 64,
        num_relations: int = 5,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.fc1 = nn.Linear(hidden_channels, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for multi-relational graph convolution."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h_graph = global_mean_pool(x, batch)

        h = F.relu(self.fc1(h_graph))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.fc2(h).squeeze(-1)

        return out
