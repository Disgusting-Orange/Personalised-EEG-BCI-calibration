from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resting_state.spectral import (
    compute_epoch_psd,
    export_node_features,
    extract_band_powers,
    run_stage6_subject,
)


class Stage6SpectralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Create synthetic Epochs object with 64 channels, 128 Hz sfreq, 2.0s duration
        info = mne.create_info(
            [f"EEG{i+1:02d}" for i in range(64)],
            sfreq=128.0,
            ch_types="eeg",
        )
        rng = np.random.default_rng(42)
        data = rng.normal(size=(10, 64, 256)) * 1e-6  # 10 epochs, 64 channels, 256 samples
        cls.synthetic_epochs = mne.EpochsArray(data, info, verbose=False)
        cls.config_path = Path(__file__).resolve().parents[1] / "configs" / "stage6_spectral_features.yaml"

    def test_psd_computation_shape_and_positivity(self) -> None:
        psd, freqs = compute_epoch_psd(self.synthetic_epochs)
        self.assertEqual(psd.shape[0], 64)
        self.assertTrue(np.all(psd >= 0))
        self.assertTrue(np.all(freqs >= 1.0) and np.all(freqs <= 40.0))

    def test_band_powers_relative_sum_to_unity(self) -> None:
        psd, freqs = compute_epoch_psd(self.synthetic_epochs)
        band_powers = extract_band_powers(psd, freqs)
        rel_powers = band_powers["relative"]
        self.assertEqual(set(rel_powers.keys()), {"delta", "theta", "alpha", "beta", "gamma"})
        
        # Check relative powers sum to ~1.0
        rel_sums = np.sum(list(rel_powers.values()), axis=0)
        np.testing.assert_allclose(rel_sums, 1.0, atol=0.05)

    def test_node_features_export_shape(self) -> None:
        psd, freqs = compute_epoch_psd(self.synthetic_epochs)
        band_powers = extract_band_powers(psd, freqs)
        node_features = export_node_features(band_powers, feature_type="relative")
        self.assertEqual(node_features.shape, (64, 5))
        self.assertTrue(np.all(node_features >= 0))

    def test_run_stage6_on_s001_preprocessed(self) -> None:
        s001_epochs = Path(__file__).resolve().parents[1] / "outputs" / "preprocessed" / "epochs" / "S001_R01_epochs-epo.fif"
        if s001_epochs.exists():
            result = run_stage6_subject("S001", self.config_path)
            self.assertEqual(result["subject_id"], "S001")
            self.assertEqual(result["status"], "PASS")
            self.assertIn("R01", result["runs"])
            self.assertIn("R02", result["runs"])
            self.assertEqual(result["runs"]["R01"]["node_features_shape"], [64, 5])


if __name__ == "__main__":
    unittest.main()
