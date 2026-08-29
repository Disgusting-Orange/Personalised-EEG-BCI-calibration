"""PyTorch Geometric GraphSAGE Regressor for Subject-Level EEG Motor Imagery Regression.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool


class SAGERegressor(nn.Module):
    """Publication-quality PyG GraphSAGE Regressor."""

    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 48,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: str = "mean",
        use_jk: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.pooling_type = pooling
        self.use_jk = use_jk
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Linear Head
        multiplier = num_layers if use_jk else 1
        fc1_in = (hidden_channels * multiplier * 2) if pooling == "concat" else (hidden_channels * multiplier)
        self.fc1 = nn.Linear(fc1_in, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass computing continuous scalar prediction per subject graph.

        Parameters
        ----------
        x:
            Node feature tensor of shape (N_total, in_channels).
        edge_index:
            Edge index COO tensor of shape (2, E).
        edge_weight:
            Optional edge weight tensor of shape (E,).
        batch:
            Graph assignment vector mapping nodes to graphs in batch.

        Returns
        -------
        torch.Tensor:
            Continuous scalar prediction tensor of shape (num_graphs,).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        layer_outs = []
        for conv, bn in zip(self.convs, self.bns):
            # Note: SAGEConv uses edge_index (and handles edge_weight if passed, or standard connectivity)
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outs.append(x)

        if self.use_jk:
            x = torch.cat(layer_outs, dim=-1)

        # Graph Readout Pooling
        if self.pooling_type == "concat":
            h_mean = global_mean_pool(x, batch)
            h_max = global_max_pool(x, batch)
            h_graph = torch.cat([h_mean, h_max], dim=-1)
        elif self.pooling_type == "max":
            h_graph = global_max_pool(x, batch)
        else:
            h_graph = global_mean_pool(x, batch)

        # MLP Head
        h = F.relu(self.fc1(h_graph))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.fc2(h).squeeze(-1)

        return out
