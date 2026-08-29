"""Preprocessing framework scaffolding for Stage 2A/B."""

from .epochs import create_fixed_length_epochs, run_epoch_pipeline, save_epochs
from .filters import (
    apply_bandpass_filter,
    apply_notch_filter,
    load_raw,
    normalize_eegmmidb_channels_and_set_montage,
    run_filter_pipeline,
    save_filtered_raw,
)
from .loader import (
    enumerate_runs,
    enumerate_subjects,
    load_edf_metadata,
    locate_dataset,
    validate_expected_files,
)
from .utils import get_logger

__all__ = [
    "apply_bandpass_filter",
    "apply_notch_filter",
    "create_fixed_length_epochs",
    "enumerate_runs",
    "enumerate_subjects",
    "load_edf_metadata",
    "load_raw",
    "locate_dataset",
    "normalize_eegmmidb_channels_and_set_montage",
    "run_epoch_pipeline",
    "run_filter_pipeline",
    "save_epochs",
    "save_filtered_raw",
    "validate_expected_files",
    "get_logger",
]
