from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resting_state.connectivity import (
    CONNECTIVITY_ESTIMATORS,
    BaseConnectivityEstimator,
    CoherenceEstimator,
    PLVIEstimator,
    WPLIEstimator,
    run_stage7_subject,
)


class Stage7ConnectivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        info = mne.create_info(
            [f"EEG{i+1:02d}" for i in range(64)],
            sfreq=128.0,
            ch_types="eeg",
        )
        rng = np.random.default_rng(42)
        # Synthetic 10 epochs x 64 channels x 256 samples
        data = rng.normal(size=(10, 64, 256)) * 1e-6
        cls.synthetic_epochs = mne.EpochsArray(data, info, verbose=False)
        cls.epochs_data = cls.synthetic_epochs.get_data()
        cls.config_path = Path(__file__).resolve().parents[1] / "configs" / "stage7_functional_connectivity.yaml"

    def test_wpli_estimator(self) -> None:
        wpli_est = WPLIEstimator()
        adj = wpli_est.compute(self.epochs_data, 128.0, 8.0, 13.0)
        self.assertEqual(adj.shape, (64, 64))
        self.assertTrue(np.allclose(adj, adj.T, atol=1e-5))
        np.testing.assert_array_equal(adj.diagonal(), np.zeros(64))
        self.assertTrue(np.all(adj >= 0.0) and np.all(adj <= 1.0))

    def test_plv_estimator(self) -> None:
        plv_est = PLVIEstimator()
        adj = plv_est.compute(self.epochs_data, 128.0, 8.0, 13.0)
        self.assertEqual(adj.shape, (64, 64))
        self.assertTrue(np.allclose(adj, adj.T, atol=1e-5))
        np.testing.assert_allclose(adj.diagonal(), np.ones(64), atol=1e-5)
        self.assertTrue(np.all(adj >= 0.0) and np.all(adj <= 1.0))

    def test_coherence_estimator(self) -> None:
        coh_est = CoherenceEstimator()
        adj = coh_est.compute(self.epochs_data, 128.0, 8.0, 13.0)
        self.assertEqual(adj.shape, (64, 64))
        self.assertTrue(np.allclose(adj, adj.T, atol=1e-5))
        np.testing.assert_allclose(adj.diagonal(), np.ones(64), atol=1e-5)
        self.assertTrue(np.all(adj >= 0.0) and np.all(adj <= 1.0))

    def test_strategy_pattern_registration(self) -> None:
        self.assertIn("wpli", CONNECTIVITY_ESTIMATORS)
        self.assertIn("plv", CONNECTIVITY_ESTIMATORS)
        self.assertIsInstance(CONNECTIVITY_ESTIMATORS["wpli"], BaseConnectivityEstimator)
        self.assertIsInstance(CONNECTIVITY_ESTIMATORS["plv"], BaseConnectivityEstimator)

    def test_run_stage7_subject_s001(self) -> None:
        s001_epochs = Path(__file__).resolve().parents[1] / "outputs" / "preprocessed" / "epochs" / "S001_R01_epochs-epo.fif"
        if s001_epochs.exists():
            result = run_stage7_subject("S001", self.config_path)
            self.assertEqual(result["subject_id"], "S001")
            self.assertEqual(result["status"], "PASS")
            self.assertIn("R01", result["runs"])
            self.assertIn("wpli", result["runs"]["R01"]["metrics"])
            self.assertIn("plv", result["runs"]["R01"]["metrics"])


if __name__ == "__main__":
    unittest.main()
