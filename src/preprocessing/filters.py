"""Reusable EEG filtering pipeline for Stage 2B.

This module implements a minimal, research-safe filtering workflow for MNE Raw
objects. It supports optional band-pass and notch filtering, metadata-aware
logging, and deterministic output saving without modifying the original raw
files.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import mne

from .loader import locate_dataset
from .utils import get_logger

RawLike = Union[mne.io.BaseRaw, str, os.PathLike[str]]


def normalize_eegmmidb_channels_and_set_montage(
    raw: mne.io.BaseRaw,
    montage_name: str = "standard_1020",
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Normalize EEGMMIDB channel names and apply standard montage.

    Strips trailing dots from channel names, maps them to canonical MNE
    case-standard channel names, and sets the standard montage so that 3D
    digitization coordinates (raw.info['dig']) are properly populated.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    montage_lower_map = {ch.lower(): ch for ch in montage.ch_names}

    rename_dict = {}
    for ch in raw.ch_names:
        clean_name = ch.strip().rstrip(".")
        if clean_name.lower() in montage_lower_map:
            rename_dict[ch] = montage_lower_map[clean_name.lower()]

    if rename_dict:
        raw.rename_channels(rename_dict)

    raw.set_montage(montage, on_missing="warn", verbose=verbose)
    return raw


def load_raw(file_path: RawLike, *, preload: bool = True, verbose: bool = False) -> mne.io.BaseRaw:
    """Load an EDF file into an MNE Raw object.

    Parameters
    ----------
    file_path:
        Path to an EDF file or an existing Raw object.
    preload:
        Whether to load the data into memory.
    verbose:
        Whether to emit MNE verbose output.
    """

    if isinstance(file_path, mne.io.BaseRaw):
        raw = file_path.copy()
    else:
        raw = mne.io.read_raw_edf(str(file_path), preload=preload, verbose=verbose)
        if not preload:
            raw.load_data(verbose=verbose)
        raw = normalize_eegmmidb_channels_and_set_montage(raw, verbose=verbose)

    if raw.info.get("dig") is None:
        raw = normalize_eegmmidb_channels_and_set_montage(raw, verbose=verbose)

    return raw


def apply_bandpass_filter(
    raw: mne.io.BaseRaw,
    *,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    method: str = "fir",
    phase: str = "zero",
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Apply a zero-phase FIR band-pass filter to a Raw object."""

    filtered = raw.copy().load_data(verbose=verbose)
    filtered.filter(l_freq=l_freq, h_freq=h_freq, method=method, phase=phase, verbose=verbose)
    return filtered


def apply_notch_filter(
    raw: mne.io.BaseRaw,
    *,
    frequencies: Optional[list[float]] = None,
    method: str = "fir",
    phase: str = "zero",
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Apply a zero-phase notch filter to a Raw object."""

    filtered = raw.copy().load_data(verbose=verbose)
    if frequencies is None:
        frequencies = [50.0, 60.0]
    filtered.notch_filter(freqs=frequencies, method=method, phase=phase, verbose=verbose)
    return filtered


def run_filter_pipeline(
    raw_or_path: RawLike,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
    save_filtered: bool = False,
    output_dir: Optional[str | os.PathLike[str]] = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Run the filtering pipeline on a Raw object or EDF path.

    The pipeline loads the data, applies band-pass filtering first, then notch
    filtering, logs the operation, and optionally saves the output to disk.
    """

    if logger is None:
        logger = get_logger("preprocessing.filters")

    start_time = time.perf_counter()
    config = config or {}
    bandpass_cfg = config.get("bandpass", {})
    notch_cfg = config.get("notch", {})

    if isinstance(raw_or_path, mne.io.BaseRaw):
        raw = raw_or_path.copy().load_data(verbose=verbose)
    else:
        raw = load_raw(raw_or_path, preload=True, verbose=verbose)

    subject_name = subject or "unknown"
    run_name = run or Path(str(raw_or_path)).stem if not isinstance(raw_or_path, mne.io.BaseRaw) else "unknown"

    logger.info(
        "Starting filtering pipeline for subject=%s run=%s sfreq=%.2f",
        subject_name,
        run_name,
        raw.info["sfreq"],
    )

    filtered = raw
    if bandpass_cfg.get("enabled", False):
        filtered = apply_bandpass_filter(
            filtered,
            l_freq=bandpass_cfg.get("l_freq", 1.0),
            h_freq=bandpass_cfg.get("h_freq", 40.0),
            verbose=verbose,
        )
    if notch_cfg.get("enabled", False):
        filtered = apply_notch_filter(
            filtered,
            frequencies=notch_cfg.get("frequencies", [50.0, 60.0]),
            verbose=verbose,
        )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Filtering complete for subject=%s run=%s elapsed=%.3fs settings=%s",
        subject_name,
        run_name,
        elapsed,
        {
            "bandpass": bandpass_cfg,
            "notch": notch_cfg,
        },
    )

    if save_filtered:
        output_path = save_filtered_raw(
            filtered,
            subject=subject_name,
            run=run_name,
            output_dir=output_dir,
            overwrite=overwrite,
            logger=logger,
        )
        logger.info("Saved filtered output to %s", output_path)

    return filtered


def save_filtered_raw(
    raw: mne.io.BaseRaw,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    output_dir: Optional[str | os.PathLike[str]] = None,
    overwrite: bool = False,
    logger: Optional[Any] = None,
) -> Path:
    """Persist a filtered Raw object to disk using a deterministic file name."""

    if logger is None:
        logger = get_logger("preprocessing.filters")

    base_dir = Path(output_dir) if output_dir is not None else Path("outputs") / "preprocessed" / "filtered"
    base_dir.mkdir(parents=True, exist_ok=True)

    subject_name = subject or "unknown"
    run_name = run or "unknown"
    output_path = base_dir / f"{subject_name}_{run_name}_filtered_raw.fif"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Filtered output already exists and overwrite=False: {output_path}")

    raw.save(output_path, overwrite=overwrite, verbose=False)
    logger.info("Saved filtered Raw to %s", output_path)
    return output_path
