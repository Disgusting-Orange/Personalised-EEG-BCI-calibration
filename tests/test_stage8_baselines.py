from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baselines.evaluator import evaluate_model_loso, load_or_create_loso_splits
from baselines.models import create_pipeline, get_hpo_grid
from baselines.stats import compute_bootstrap_cis, compute_regression_metrics, paired_model_comparison


class Stage8BaselinesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rng = np.random.default_rng(42)
        # Synthetic 10 subjects x 20 features
        cls.X_synth = cls.rng.normal(size=(10, 20))
        cls.y_synth = cls.rng.uniform(0.5, 0.9, size=10)
        cls.feature_names = [f"feat_{i+1:02d}" for i in range(20)]
        cls.subject_ids = [f"S{i+1:03d}" for i in range(10)]

    def test_pipeline_creation(self) -> None:
        models = ["dummy", "ridge", "lasso", "elasticnet", "svr", "rf", "xgboost"]
        for m_id in models:
            pipe = create_pipeline(m_id, random_seed=42)
            self.assertIn("scaler", pipe.named_steps)
            self.assertIn("regressor", pipe.named_steps)

            grid = get_hpo_grid(m_id)
            if m_id != "dummy":
                self.assertTrue(len(grid) > 0)
            else:
                self.assertEqual(len(grid), 0)

    def test_regression_metrics_computation(self) -> None:
        y_true = np.array([0.6, 0.7, 0.8, 0.65, 0.75])
        y_pred = np.array([0.62, 0.68, 0.79, 0.64, 0.76])

        m = compute_regression_metrics(y_true, y_pred)
        self.assertIn("mae", m)
        self.assertIn("rmse", m)
        self.assertIn("r2", m)
        self.assertIn("pearson_r", m)
        self.assertIn("spearman_r", m)
        self.assertTrue(m["mae"] >= 0.0)
        self.assertTrue(m["rmse"] >= m["mae"])
        self.assertTrue(m["r2"] <= 1.0)
        self.assertTrue(m["pearson_r"] > 0.9)

    def test_bootstrap_cis(self) -> None:
        y_true = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75, 0.85, 0.95])
        y_pred = y_true + self.rng.normal(0, 0.02, size=10)

        cis = compute_bootstrap_cis(y_true, y_pred, n_bootstraps=100, seed=42)
        self.assertIn("mae_ci", cis)
        self.assertIn("rmse_ci", cis)
        self.assertIn("r2_ci", cis)
        self.assertIn("pearson_ci", cis)
        self.assertTrue(cis["mae_ci"][0] <= cis["mae_ci"][1])

    def test_paired_comparison(self) -> None:
        y_true = np.array([0.6, 0.7, 0.8, 0.65, 0.75])
        y_pred_a = y_true + 0.01
        y_pred_b = y_true + 0.10

        comp = paired_model_comparison(y_true, y_pred_a, y_pred_b)
        self.assertIn("stat", comp)
        self.assertIn("p_value", comp)
        self.assertTrue(comp["mean_diff"] < 0)

    def test_loso_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_file = Path(tmpdir) / "test_loso.json"
            splits = load_or_create_loso_splits(10, splits_file)
            self.assertEqual(len(splits), 10)
            self.assertEqual(len(splits[0]["train_idx"]), 9)
            self.assertEqual(len(splits[0]["test_idx"]), 1)
            self.assertTrue(splits_file.exists())

    def test_loso_evaluation_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                "random_seed": 42,
                "inner_cv_folds": 3,
                "bootstrap_iterations": 50,
                "loso_splits_path": str(Path(tmpdir) / "loso_splits.json"),
                "hpo_grids": {
                    "ridge": {"regressor__alpha": [0.1, 1.0]}
                }
            }

            res_dummy = evaluate_model_loso("dummy", self.X_synth, self.y_synth, self.feature_names, self.subject_ids, cfg)
            self.assertEqual(res_dummy["model_id"], "dummy")
            self.assertEqual(len(res_dummy["oof_predictions"]), 10)
            self.assertIn("mae", res_dummy["metrics"])

            res_ridge = evaluate_model_loso("ridge", self.X_synth, self.y_synth, self.feature_names, self.subject_ids, cfg)
            self.assertEqual(res_ridge["model_id"], "ridge")
            self.assertEqual(len(res_ridge["oof_predictions"]), 10)
            self.assertIn("mae", res_ridge["metrics"])
            self.assertTrue(len(res_ridge["feature_importance"]) == 20)


if __name__ == "__main__":
    unittest.main()
