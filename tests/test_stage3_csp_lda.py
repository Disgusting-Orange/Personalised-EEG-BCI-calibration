from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_decoding.csp_lda import generate_stratified_folds, run_csp_lda_oof


CV_CONFIG = {"strategy": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42}


class CspLdaTests(unittest.TestCase):
    def test_fold_generation_is_deterministic_and_disjoint(self) -> None:
        y = np.array([0, 1] * 10)
        first = generate_stratified_folds(y, CV_CONFIG)
        second = generate_stratified_folds(y, CV_CONFIG)
        for (first_train, first_test), (second_train, second_test) in zip(first, second):
            np.testing.assert_array_equal(first_train, second_train)
            np.testing.assert_array_equal(first_test, second_test)
            self.assertEqual(np.intersect1d(first_train, first_test).size, 0)

    def test_oof_predictions_assign_every_trial_once(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 4, 40))
        X[10:, 0, :] += 0.5
        y = np.array([0] * 10 + [1] * 10)
        folds, predictions, probabilities, fold_metrics = run_csp_lda_oof(
            X,
            y,
            csp_config={"n_components": 2, "log": True, "norm_trace": False},
            lda_config={"solver": "lsqr", "shrinkage": "auto"},
            cv_config=CV_CONFIG,
            subject_id="S001",
        )
        self.assertTrue(np.all(folds > 0))
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertEqual(len(fold_metrics), 5)


if __name__ == "__main__":
    unittest.main()
