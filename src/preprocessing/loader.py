"""Dataset discovery and EDF metadata loading utilities.

This module intentionally does not implement filtering, ICA, epoching,
artifact rejection, or saving operations. It is limited to locating the
EEGMMIDB dataset, enumerating subjects/runs, validating expected files,
and loading EDF metadata through MNE.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import mne

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "eegmmidb"


def locate_dataset(dataset_root: Optional[str | os.PathLike[str]] = None) -> Path:
    """Return the dataset root path.

    Parameters
    ----------
    dataset_root:
        Optional explicit path to the dataset root. When omitted, the function
        uses the repository-local EEGMMIDB raw dataset directory.
    """

    candidate = Path(dataset_root) if dataset_root is not None else None
    if candidate is None:
        candidate = Path(os.environ.get("EEGMMIDB_DATASET_ROOT", DEFAULT_DATASET_ROOT))

    if not candidate.exists():
        raise FileNotFoundError(f"Dataset root not found: {candidate}")
    return candidate.resolve()


def enumerate_subjects(dataset_root: Optional[str | os.PathLike[str]] = None) -> list[str]:
    """List subject directories that match the EEGMMIDB naming convention."""

    root = locate_dataset(dataset_root)
    subject_dirs = [
        entry.name
        for entry in sorted(root.iterdir())
        if entry.is_dir() and re.fullmatch(r"S\d{3}", entry.name)
    ]
    return subject_dirs


def enumerate_runs(subject_dir: str | os.PathLike[str], dataset_root: Optional[str | os.PathLike[str]] = None) -> list[str]:
    """List run identifiers for a subject directory.

    Returns run IDs such as ``R01`` or ``R14`` based on the EDF filenames.
    """

    root = locate_dataset(dataset_root)
    subject_path = Path(subject_dir)
    if not subject_path.is_absolute():
        subject_path = root / subject_path

    if not subject_path.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_path}")

    runs = []
    for path in sorted(subject_path.glob("*.edf")):
        match = re.search(r"(R\d{2})\.edf$", path.name)
        if match:
            runs.append(match.group(1))
    return runs


def validate_expected_files(subject_dir: str | os.PathLike[str], run_id: str, dataset_root: Optional[str | os.PathLike[str]] = None) -> dict[str, Path | bool]:
    """Validate that EDF and event-sidecar files are present for a run."""

    root = locate_dataset(dataset_root)
    subject_path = Path(subject_dir)
    if not subject_path.is_absolute():
        subject_path = root / subject_path

    edf_path = subject_path / f"{subject_path.name}{run_id}.edf"
    event_path = subject_path / f"{subject_path.name}{run_id}.edf.event"
    return {
        "subject_dir": subject_path,
        "edf_exists": edf_path.exists(),
        "event_exists": event_path.exists(),
        "edf_path": edf_path,
        "event_path": event_path,
    }


def load_edf_metadata(
    subject_id: str,
    run_id: str,
    dataset_root: Optional[str | os.PathLike[str]] = None,
    preload: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    """Load an EDF file with MNE and return metadata only.

    No filtering, artifact removal, or saving is performed.
    """

    root = locate_dataset(dataset_root)
    subject_path = root / subject_id
    edf_path = subject_path / f"{subject_id}{run_id}.edf"
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=verbose)

    metadata = {
        "subject_id": subject_id,
        "run_id": run_id,
        "file_path": str(edf_path),
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "n_channels": int(raw.info["nchan"]),
        "channel_names": list(raw.ch_names),
        "n_times": int(raw.n_times),
        "duration_seconds": float(raw.times[-1]) if len(raw.times) else 0.0,
        "annotations": list(raw.annotations.description),
    }

    raw.close()
    return metadata
