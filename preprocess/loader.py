"""
loader.py
---------
Recursive EDF discovery and loading with robust error handling.

PhysioNet EEG Motor Movement/Imagery dataset specifics
-------------------------------------------------------
* 64-channel EEG, 10-10 electrode placement (modified)
* Sampling rate: 160 Hz
* Channel names end in "." in the EDF header → stripped to match
  standard 10-10 names expected by MNE's montage lookup.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import mne
import numpy as np


# ── Channel name normalisation ─────────────────────────────────────────────────

# PhysioNet header appends "." to every channel name (e.g. "Fc5.")
# Map them to clean 10-10 names.
_PHYSIONET_64CH = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
    "Fp1", "Fpz", "Fp2",
    "Af7", "Af3", "Afz", "Af4", "Af8",
    "F7",  "F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",  "F8",
    "Ft7", "Ft8",
    "T7",  "T8",  "T9",  "T10",
    "Tp7", "Tp8",
    "P7",  "P5",  "P3",  "P1",  "Pz",  "P2",  "P4",  "P6",  "P8",
    "Po7", "Po3", "Poz", "Po4", "Po8",
    "O1",  "Oz",  "O2",
    "Iz",
]


def _normalise_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Strip trailing dots from PhysioNet channel names and apply standard
    10-10 capitalisation so MNE can match them to a montage.
    """
    rename_map: dict = {}
    for ch in raw.ch_names:
        clean = ch.strip().rstrip(".")
        # capitalise first letter, lower-case rest (e.g. "FC5" → "Fc5")
        clean = clean[0].upper() + clean[1:].lower() if len(clean) > 1 else clean.upper()
        if clean != ch:
            rename_map[ch] = clean
    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def _set_montage(raw: mne.io.BaseRaw, logger: logging.Logger) -> mne.io.BaseRaw:
    """Apply the standard_1020 montage; log any channels that don't match."""
    montage = mne.channels.make_standard_montage("standard_1020")
    try:
        raw.set_montage(montage, on_missing="warn")
    except Exception as exc:
        logger.warning(f"Could not set montage: {exc}")
    return raw


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_edf_files(data_dir: Path) -> List[Path]:
    """
    Recursively find all ``*.edf`` files under *data_dir*
    (excludes ``*.edf.event`` annotation files).

    Returns a sorted list of absolute Paths.
    """
    edf_files = sorted(
        p for p in data_dir.rglob("*.edf")
        if not p.name.endswith(".edf.event")
    )
    return edf_files


def discover_subject_runs(
    data_dir: Path,
    subjects: Optional[List[str]] = None,
    runs: Optional[List[str]] = None,
) -> List[Path]:
    """
    Return EDF paths filtered to *subjects* and/or *runs*.

    Parameters
    ----------
    data_dir : root data directory (contains S001/, S002/, …)
    subjects : e.g. ["S001", "S002"]  — None means all subjects
    runs     : e.g. ["R03", "R04"]    — None means all runs

    Returns
    -------
    Sorted list of matching Paths.
    """
    all_files = discover_edf_files(data_dir)
    result = []
    for p in all_files:
        stem = p.stem            # "S001R03"
        subj = stem[:4]          # "S001"
        run  = stem[4:]          # "R03"
        if subjects and subj not in subjects:
            continue
        if runs and run not in runs:
            continue
        result.append(p)
    return sorted(result)


# ── Loading ────────────────────────────────────────────────────────────────────

def load_raw_edf(
    edf_path: Path,
    logger: logging.Logger,
    preload: bool = True,
) -> Optional[mne.io.BaseRaw]:
    """
    Load a single EDF file and return an MNE ``Raw`` object.

    Handles corrupt / unreadable files gracefully — returns ``None``
    and logs the exception rather than raising.

    Steps applied at load time:
    1. Read EDF with ``mne.io.read_raw_edf``
    2. Strip trailing dots from channel names
    3. Set standard 10-20 montage
    4. Set channel type to EEG for all channels

    Parameters
    ----------
    edf_path : absolute Path to the EDF file
    logger   : logger instance for this subject
    preload  : whether to load data into RAM immediately

    Returns
    -------
    mne.io.BaseRaw or None if loading failed
    """
    logger.info(f"Loading: {edf_path.name}")
    try:
        raw = mne.io.read_raw_edf(
            str(edf_path),
            preload=preload,
            verbose=False,
            stim_channel="auto",
        )
    except Exception as exc:
        logger.error(f"Failed to read {edf_path.name}: {type(exc).__name__}: {exc}")
        return None

    # Validate minimum content
    if raw.n_times == 0:
        logger.error(f"{edf_path.name}: empty recording — skipping.")
        return None
    if len(raw.ch_names) == 0:
        logger.error(f"{edf_path.name}: no channels found — skipping.")
        return None

    # Normalise channel names
    raw = _normalise_channel_names(raw)

    # Force all channels to EEG type (dataset has no dedicated EOG/ECG channels)
    channel_types = {ch: "eeg" for ch in raw.ch_names}
    raw.set_channel_types(channel_types)

    # Attach montage
    raw = _set_montage(raw, logger)

    logger.info(
        f"  Loaded: {len(raw.ch_names)} ch, "
        f"{raw.info['sfreq']:.0f} Hz, "
        f"{raw.n_times / raw.info['sfreq']:.1f}s"
    )
    return raw


def load_subject_runs(
    edf_paths: List[Path],
    logger: logging.Logger,
    concatenate: bool = False,
) -> Iterator[Tuple[Path, Optional[mne.io.BaseRaw]]]:
    """
    Generator that yields (path, raw_or_None) for every EDF in *edf_paths*.

    Parameters
    ----------
    edf_paths   : list of EDF paths to process
    logger      : logger instance
    concatenate : if True, concatenate all successful raws into one
                  and yield a single (None, concatenated_raw)
                  instead of individual tuples

    Yields
    ------
    (Path, mne.io.BaseRaw | None)
    """
    loaded_raws = []
    for path in edf_paths:
        raw = load_raw_edf(path, logger)
        if not concatenate:
            yield path, raw
        elif raw is not None:
            loaded_raws.append(raw)

    if concatenate and loaded_raws:
        logger.info(f"Concatenating {len(loaded_raws)} runs …")
        combined = mne.concatenate_raws(loaded_raws)
        yield None, combined


# ── Sanity check ──────────────────────────────────────────────────────────────

def validate_raw(raw: mne.io.BaseRaw, logger: logging.Logger) -> bool:
    """
    Run basic sanity checks on a loaded Raw object.

    Returns True if the recording looks usable, False otherwise.
    """
    sfreq = raw.info["sfreq"]
    n_ch  = len(raw.ch_names)
    dur_s = raw.n_times / sfreq

    if sfreq < 100:
        logger.warning(f"Unusually low sampling rate: {sfreq} Hz")
    if n_ch < 10:
        logger.warning(f"Very few channels ({n_ch})")
    if dur_s < 5.0:
        logger.warning(f"Very short recording ({dur_s:.1f}s) — may yield no epochs")
        return False

    # Check for NaNs / Infs
    data = raw.get_data()
    if not np.isfinite(data).all():
        n_bad = (~np.isfinite(data)).sum()
        logger.warning(f"Non-finite values detected: {n_bad} samples")
        # Impute with zeros for robustness
        data[~np.isfinite(data)] = 0.0
        raw._data = data

    return True
