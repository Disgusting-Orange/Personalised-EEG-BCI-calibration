from __future__ import annotations

import sys
from pathlib import Path
import unittest

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_decoding.event_epochs import create_mi_event_epochs, resolve_approved_mi_task


class EventEpochTests(unittest.TestCase):
    def test_t1_t2_are_mapped_and_t0_is_excluded(self) -> None:
        info = mne.create_info(["C3", "C4"], sfreq=20.0, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((2, 200)), info, verbose=False)
        raw.set_annotations(mne.Annotations([1.0, 3.0, 5.0], [0.0, 0.0, 0.0], ["T0", "T1", "T2"]))
        epochs, manifest = create_mi_event_epochs(
            raw,
            subject_id="S001",
            run_id="R04",
            event_mapping={"T1": {"label": 0, "class_name": "left_fist"}, "T2": {"label": 1, "class_name": "right_fist"}},
            epoch_config={"tmin_seconds": 0.0, "tmax_seconds": 1.0, "baseline": None, "reject_by_annotation": True},
        )
        self.assertEqual(len(epochs), 2)
        self.assertEqual(manifest["event_code"].tolist(), ["T1", "T2"])
        self.assertEqual(manifest["true_label"].tolist(), [0, 1])

    def test_task_resolution_rejects_incompatible_runs(self) -> None:
        config = {
            "task_definitions": {
                "imagined_left_right_fist": {
                    "role": "motor_imagery",
                    "runs": ["R04"],
                    "event_mapping": {"T1": {"label": 0}, "T2": {"label": 1}},
                }
            },
            "run_mapping": {"R04": {"role": "motor_imagery", "task_definition": "other_task"}},
        }
        with self.assertRaises(ValueError):
            resolve_approved_mi_task(config, "imagined_left_right_fist")


if __name__ == "__main__":
    unittest.main()
