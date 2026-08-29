from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph.statistical_tests import (
    bootstrap_confidence_intervals,
    compute_cohens_dz,
    compute_loso_fold_robustness,
    compute_rank_biserial,
    paired_model_comparisons,
    target_permutation_test,
)


class Stage14StatisticalTests(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.y_true = np.random.uniform(0.4, 0.8, 109)
        self.y_pred_gcn = self.y_true + np.random.normal(0.0, 0.05, 109)
        self.y_pred_svr = self.y_true + np.random.normal(0.0, 0.08, 109)

    def test_compute_cohens_dz(self) -> None:
        dz = compute_cohens_dz(self.y_pred_svr, self.y_pred_gcn)
        self.assertIsInstance(dz, float)

    def test_compute_rank_biserial(self) -> None:
        err_a = np.abs(self.y_true - self.y_pred_svr)
        err_b = np.abs(self.y_true - self.y_pred_gcn)
        r_rb = compute_rank_biserial(err_a, err_b)
        self.assertIsInstance(r_rb, float)
        self.assertTrue(-1.0 <= r_rb <= 1.0)

    def test_paired_model_comparisons(self) -> None:
        preds = {"gcn": self.y_pred_gcn, "svr": self.y_pred_svr}
        df_res = paired_model_comparisons(self.y_true, preds, ref_model="gcn")
        self.assertEqual(len(df_res), 1)
        self.assertIn("p_wilcoxon", df_res.columns)
        self.assertIn("p_ttest", df_res.columns)
        self.assertIn("p_adj_holm", df_res.columns)
        self.assertIn("cohens_dz", df_res.columns)
        self.assertIn("rank_biserial_r", df_res.columns)

    def test_bootstrap_confidence_intervals(self) -> None:
        res = bootstrap_confidence_intervals(self.y_true, self.y_pred_gcn, n_bootstrap=100, seed=42)
        self.assertIn("mae_ci", res)
        self.assertIn("r2_ci", res)
        self.assertEqual(len(res["mae_ci"]), 2)
        self.assertTrue(res["mae_ci"][0] <= res["mae"] <= res["mae_ci"][1])

    def test_target_permutation_test(self) -> None:
        res = target_permutation_test(self.y_true, self.y_pred_gcn, n_permutations=50, seed=42)
        self.assertEqual(res["n_permutations"], 50)
        self.assertTrue(0.0 <= res["perm_pearson_pvalue"] <= 1.0)

    def test_compute_loso_fold_robustness(self) -> None:
        preds = {"gcn": self.y_pred_gcn, "svr": self.y_pred_svr}
        df_rob = compute_loso_fold_robustness(self.y_true, preds)
        self.assertEqual(len(df_rob), 2)
        self.assertIn("mean_abs_error", df_rob.columns)
        self.assertIn("variance_abs_error", df_rob.columns)


if __name__ == "__main__":
    unittest.main()
