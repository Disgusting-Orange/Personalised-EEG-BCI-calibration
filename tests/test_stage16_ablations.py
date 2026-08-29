"""Unit test suite for Stage 16 Scientific Ablation Suite."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.graph.ablation_runner import (
    compute_metrics,
    get_or_build_graph_dataset,
    load_subject_ids,
)
from src.graph.ablation_viz import (
    generate_ablation_latex_table,
    generate_publication_ablation_figure,
)


class TestStage16Ablations(unittest.TestCase):
    """Test cases verifying Stage 16 ablation components."""

    def test_compute_metrics(self):
        y_true = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        y_pred = np.array([0.52, 0.58, 0.71, 0.79, 0.88])
        res = compute_metrics(y_true, y_pred)
        self.assertIn("mae", res)
        self.assertIn("r2", res)
        self.assertIn("pearson_r", res)
        self.assertGreater(res["r2"], 0.8)

    def test_load_subject_ids(self):
        ids = load_subject_ids()
        self.assertEqual(len(ids), 109)
        self.assertIn("S001", ids)

    def test_get_or_build_graph_dataset_s001(self):
        dataset = get_or_build_graph_dataset(["S001"], "wpli", "alpha", 0.20)
        self.assertEqual(len(dataset), 1)
        data = dataset[0]
        self.assertEqual(data.num_nodes, 64)
        self.assertEqual(data.x.shape, (64, 10))

    def test_table_and_figure_generators(self):
        dummy_data = [
            {"Experiment": "topology_density", "Variant": "top10", "R2": 0.1520, "MAE": 0.0910, "RMSE": 0.1140, "Pearson_r": 0.4120, "Pearson_p": 0.001, "Spearman_r": 0.4010, "Spearman_p": 0.001},
            {"Experiment": "topology_density", "Variant": "top20", "R2": 0.2685, "MAE": 0.0831, "RMSE": 0.1045, "Pearson_r": 0.5312, "Pearson_p": 0.0001, "Spearman_r": 0.5185, "Spearman_p": 0.0001},
            {"Experiment": "connectivity_metric", "Variant": "wPLI", "R2": 0.2685, "MAE": 0.0831, "RMSE": 0.1045, "Pearson_r": 0.5312, "Pearson_p": 0.0001, "Spearman_r": 0.5185, "Spearman_p": 0.0001},
            {"Experiment": "connectivity_metric", "Variant": "PLV", "R2": 0.1712, "MAE": 0.0901, "RMSE": 0.1128, "Pearson_r": 0.4321, "Pearson_p": 0.001, "Spearman_r": 0.4210, "Spearman_p": 0.001},
            {"Experiment": "frequency_band", "Variant": "Delta", "R2": 0.0612, "MAE": 0.0965, "RMSE": 0.1205, "Pearson_r": 0.2810, "Pearson_p": 0.01, "Spearman_r": 0.2710, "Spearman_p": 0.01},
            {"Experiment": "pooling_strategy", "Variant": "Mean", "R2": 0.2095, "MAE": 0.0878, "RMSE": 0.1102, "Pearson_r": 0.4782, "Pearson_p": 0.0001, "Spearman_r": 0.4650, "Spearman_p": 0.0001},
            {"Experiment": "loss_function", "Variant": "MSE", "R2": 0.2285, "MAE": 0.0865, "RMSE": 0.1089, "Pearson_r": 0.4912, "Pearson_p": 0.0001, "Spearman_r": 0.4810, "Spearman_p": 0.0001},
            {"Experiment": "jumping_knowledge", "Variant": "Without_JK", "R2": 0.2412, "MAE": 0.0851, "RMSE": 0.1072, "Pearson_r": 0.5042, "Pearson_p": 0.0001, "Spearman_r": 0.4910, "Spearman_p": 0.0001},
            {"Experiment": "learning_rate_scheduler", "Variant": "Plateau", "R2": 0.2542, "MAE": 0.0842, "RMSE": 0.1058, "Pearson_r": 0.5185, "Pearson_p": 0.0001, "Spearman_r": 0.5050, "Spearman_p": 0.0001},
        ]
        df_dummy = pd.DataFrame(dummy_data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tex_file = tmp_path / "test_table.tex"
            fig_file = tmp_path / "test_fig.png"

            latex_str = generate_ablation_latex_table(df_dummy, tex_file)
            self.assertTrue(tex_file.exists())
            self.assertIn("top20", latex_str)

            generate_publication_ablation_figure(df_dummy, fig_file, dpi=100)
            self.assertTrue(fig_file.exists())


if __name__ == "__main__":
    unittest.main()
