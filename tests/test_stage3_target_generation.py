from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_decoding.target_generation import aggregate_target


class TargetAggregationTests(unittest.TestCase):
    def test_primary_target_is_pooled_balanced_accuracy(self) -> None:
        metrics = aggregate_target(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertIn("macro_f1", metrics)
        self.assertIn("kappa", metrics)


if __name__ == "__main__":
    unittest.main()
