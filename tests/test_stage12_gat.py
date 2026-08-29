from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph.gat_model import GATRegressor


class Stage12GATTests(unittest.TestCase):
    def test_gat_model_forward(self) -> None:
        g1 = Data(x=torch.randn(64, 10), edge_index=torch.randint(0, 64, (2, 200)), edge_weight=torch.rand(200), y=torch.tensor([0.65]))
        g2 = Data(x=torch.randn(64, 10), edge_index=torch.randint(0, 64, (2, 200)), edge_weight=torch.rand(200), y=torch.tensor([0.72]))
        batch = Batch.from_data_list([g1, g2])

        model = GATRegressor(in_channels=10, hidden_channels=16, heads=2, num_layers=2)
        out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

        self.assertEqual(out.shape, (2,))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
