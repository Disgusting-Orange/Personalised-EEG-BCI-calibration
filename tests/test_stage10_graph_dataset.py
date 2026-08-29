from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph.builder import build_subject_graph
from graph.dataset import EEGGraphDataset
from graph.validator import validate_graph, validate_graph_dataset_directory


class Stage10GraphDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = {
            "connectivity_metric": "wpli",
            "frequency_band": "alpha",
            "sparsification_density": 0.15,
            "node_feature_type": "concatenated",
            "ensure_connected": True,
        }

    def test_build_subject_graph_s001(self) -> None:
        data = build_subject_graph("S001", self.config)
        self.assertIsInstance(data, Data)
        self.assertEqual(data.num_nodes, 64)
        self.assertEqual(data.x.shape, (64, 10))
        self.assertEqual(data.edge_index.shape[0], 2)
        self.assertEqual(data.edge_weight.shape[0], data.edge_index.shape[1])
        self.assertTrue(data.y.item() > 0.0 and data.y.item() <= 1.0)
        self.assertEqual(data.subject_id, "S001")

    def test_sparsification_densities(self) -> None:
        densities = [0.10, 0.15, 0.20, 1.0]
        edge_counts = []
        for d in densities:
            cfg = dict(self.config)
            cfg["sparsification_density"] = d
            data = build_subject_graph("S001", cfg)
            edge_counts.append(data.edge_index.shape[1])
            self.assertEqual(data.num_nodes, 64)

        # Higher density must yield more edges
        self.assertTrue(edge_counts[0] < edge_counts[1])
        self.assertTrue(edge_counts[1] < edge_counts[2])
        self.assertTrue(edge_counts[2] < edge_counts[3])

    def test_graph_validation_pass(self) -> None:
        data = build_subject_graph("S001", self.config)
        res = validate_graph(data)
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["isolated_node_count"], 0)
        self.assertEqual(len(res["errors"]), 0)

    def test_pyg_dataset_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = EEGGraphDataset(
                root=tmpdir,
                subjects=["S001", "S002", "S003"],
                config=self.config,
            )
            self.assertEqual(len(ds), 3)
            self.assertTrue((Path(tmpdir) / "pyg_dataset.pt").exists())
            self.assertTrue((Path(tmpdir) / "dataset_manifest.json").exists())

            v_res = validate_graph_dataset_directory(tmpdir, expected_subjects=3)
            self.assertEqual(v_res["overall_status"], "PASS")
            self.assertEqual(v_res["passed_graphs"], 3)


if __name__ == "__main__":
    unittest.main()
