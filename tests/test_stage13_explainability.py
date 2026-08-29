from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph.explainability import explain_single_graph
from graph.gat_model import GATRegressor
from graph.gcn_model import GCNRegressor


class Stage13ExplainabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        # Synthetic 64-node graph
        x = torch.randn(64, 10)
        edge_index = torch.randint(0, 64, (2, 200))
        edge_weight = torch.rand(200)
        y = torch.tensor([0.65])
        self.data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

    def test_explain_single_graph_gcn(self) -> None:
        model = GCNRegressor(in_channels=10, hidden_channels=32, num_layers=2)
        res = explain_single_graph(model, self.data, epochs=10)

        self.assertEqual(res["node_mask"].shape, (64, 10))
        self.assertEqual(res["edge_mask"].shape[0], 200)
        self.assertEqual(res["node_importances"].shape, (64,))
        self.assertEqual(res["band_importances"].shape, (10,))
        self.assertFalse(torch.isnan(torch.tensor(res["node_mask"])).any())

    def test_explain_single_graph_gat(self) -> None:
        model = GATRegressor(in_channels=10, hidden_channels=16, heads=2, num_layers=2)
        res = explain_single_graph(model, self.data, epochs=10)

        self.assertEqual(res["node_mask"].shape, (64, 10))
        self.assertEqual(res["edge_mask"].shape[0], 200)
        self.assertFalse(torch.isnan(torch.tensor(res["node_mask"])).any())


if __name__ == "__main__":
    unittest.main()
