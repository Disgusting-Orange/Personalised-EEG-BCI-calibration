"""
epochs.py
---------
Bad-channel detection, common average re-referencing,
fixed-length epoching, and amplitude-based epoch rejection.

Output
------
mne.Epochs object containing clean 2-second non-overlapping epochs
ready for PLV / wPLI graph construction and GNN training.
"""

import logging
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np


# ── Bad-channel detection ─────────────────────────────────────────────────────

def detect_bad_channels(
    raw: mne.io.BaseRaw,
    std_multiplier: float,
    flat_std_threshold: float,
    logger: logging.Logger,
) -> List[str]:
    """
    Detect bad channels using two criteria:

    1. **High variance** : channel std > ``std_multiplier`` × median(all stds)
    2. **Flat line**     : channel std < ``flat_std_threshold``

    Parameters
    ----------
    raw               : Raw object to inspect (not mutated)
    std_multiplier    : multiplier on median std for "noisy" detection
    flat_std_threshold: std threshold below which a channel is "flat"
    logger            : logger

    Returns
    -------
    List of channel names identified as bad.
    """
    data = raw.get_data()  # (n_channels, n_times)
    ch_names = np.array(raw.ch_names)
    stds = data.std(axis=1)

    median_std = np.median(stds)
    
    logger.info(f"Median STD: {median_std*1e6:.2f} µV")
    
    top_idx = np.argsort(stds)[::-1][:10]
    
    logger.info("Top 10 channel STDs:")
    for idx in top_idx:
        logger.info(
            f"{ch_names[idx]} : {stds[idx]*1e6:.2f} µV"
        )
    
    high_var_mask = stds > (std_multiplier * median_std)
    flat_mask = stds < flat_std_threshold

    
    bad_high = list(ch_names[high_var_mask])
    bad_flat = list(ch_names[flat_mask])
    bads = sorted(set(bad_high + bad_flat))

    if bad_high:
        logger.info(f"High-variance channels: {bad_high}")
    if bad_flat:
        logger.info(f"Flat-line channels: {bad_flat}")
    if not bads:
        logger.info("No bad channels detected.")

    return bads


def mark_and_interpolate_bads(
    raw: mne.io.BaseRaw,
    bad_channels: List[str],
    logger: logging.Logger,
) -> mne.io.BaseRaw:
    """
    Mark channels as bad and interpolate them using spherical splines.

    Parameters
    ----------
    raw          : Raw object (mutated in-place and also returned)
    bad_channels : list of channel names to mark as bad
    logger       : logger

    Returns
    -------
    Raw with bad channels interpolated.
    """
    if not bad_channels:
        return raw

    # Merge with any existing bads
    existing = list(raw.info.get("bads", []))
    combined = sorted(set(existing + bad_channels))
    raw.info["bads"] = combined
    logger.info(f"Interpolating {len(combined)} bad channel(s): {combined}")

    try:
        raw.interpolate_bads(reset_bads=True, verbose=False)
        logger.info("Bad channel interpolation complete.")
    except Exception as exc:
        logger.warning(
            f"Interpolation failed: {exc}. "
            f"Dropping bad channels instead: {combined}"
        )

        try:
            raw.drop_channels(combined)
            logger.info(
                f"Dropped {len(combined)} bad channel(s): {combined}"
            )
        except Exception as drop_exc:
            logger.error(
                f"Failed to drop bad channels: {drop_exc}"
            )

    return raw


# ── Re-referencing ────────────────────────────────────────────────────────────

def apply_reference(
    raw: mne.io.BaseRaw,
    reference: str,
    logger: logging.Logger,
) -> mne.io.BaseRaw:
    """
    Apply EEG re-referencing.

    Parameters
    ----------
    raw       : Raw to re-reference (mutated in-place)
    reference : "average" for common average, or a channel name string
    logger    : logger

    Returns
    -------
    Re-referenced Raw
    """
    logger.info(f"Applying reference: {reference}")
    if reference == "average":
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    else:
        raw.set_eeg_reference(ref_channels=[reference], projection=False, verbose=False)
    return raw


# ── Epoching ──────────────────────────────────────────────────────────────────

def make_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    duration: float,
    overlap: float,
    tmin: float,
    reject: Dict[str, float],
    flat: Dict[str, float],
    logger: logging.Logger,
    task_label: str = "rest",
) -> Optional[mne.Epochs]:
    """
    Segment a continuous Raw recording into fixed-length non-overlapping epochs.

    Parameters
    ----------
    raw        : cleaned, re-referenced Raw
    duration   : epoch length in seconds
    overlap    : overlap between consecutive epochs in seconds (0 = non-overlapping)
    tmin       : epoch start offset (typically 0.0)
    reject     : peak-to-peak amplitude rejection dict e.g. {"eeg": 150e-6}
    flat       : flat-signal detection dict e.g. {"eeg": 1e-6}
    logger     : logger
    task_label : event description embedded in the Epochs metadata

    Returns
    -------
    mne.Epochs or None if no valid epochs were created
    """
    logger.info(
        f"Creating {duration}s fixed-length epochs "
        f"(overlap={overlap}s, reject={reject}) …"
    )

    events = mne.make_fixed_length_events(
        raw,
        id=1,
        duration=duration,
        overlap=overlap,
        first_samp=True,
    )

    if len(events) == 0:
        logger.warning("No events generated — recording too short for epochs.")
        return None

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"segment": 1},
        tmin=tmin,
        tmax=tmin + duration - (1.0 / raw.info["sfreq"]),
        baseline=None,
        reject=reject,
        flat=flat,
        preload=True,
        verbose=False,
        reject_by_annotation=True,
    )

    n_total = len(events)
    n_kept = len(epochs)
    n_dropped = n_total - n_kept
    logger.info(
        f"Epoching: {n_total} total → {n_kept} kept, "
        f"{n_dropped} dropped ({100*n_dropped/max(n_total,1):.1f}%)"
    )

    if n_kept == 0:
        drop_reasons = {}
    
        for drop_log in epochs.drop_log:
            for reason in drop_log:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
    
        logger.warning(f"Drop reasons: {drop_reasons}")
        logger.warning("All epochs rejected — check rejection thresholds.")
        return None

    # Attach task label as metadata
    import pandas as pd
    epochs.metadata = pd.DataFrame(
        {"task": [task_label] * n_kept},
        index=range(n_kept),
    )

    return epochs


# ── Epoch quality report ──────────────────────────────────────────────────────

def epoch_quality_report(
    epochs: mne.Epochs,
    logger: logging.Logger,
) -> Dict:
    """
    Compute basic quality metrics on the final epoch set.

    Returns
    -------
    dict with keys: n_epochs, n_channels, sfreq, duration_s,
                    mean_amplitude_uV, std_amplitude_uV
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    amplitude_uV = data * 1e6  # convert V → µV

    report = {
        "n_epochs":           data.shape[0],
        "n_channels":         data.shape[1],
        "n_times":            data.shape[2],
        "sfreq":              epochs.info["sfreq"],
        "duration_s":         data.shape[2] / epochs.info["sfreq"],
        "mean_amplitude_uV":  float(np.abs(amplitude_uV).mean()),
        "std_amplitude_uV":   float(amplitude_uV.std()),
        "peak_amplitude_uV":  float(np.abs(amplitude_uV).max()),
    }

    logger.info(
        f"Epoch quality — "
        f"N={report['n_epochs']}, "
        f"mean|amp|={report['mean_amplitude_uV']:.2f} µV, "
        f"peak={report['peak_amplitude_uV']:.2f} µV"
    )
    return report


# ── Saving ────────────────────────────────────────────────────────────────────

def save_epochs(
    epochs: mne.Epochs,
    output_path,
    overwrite: bool,
    logger: logging.Logger,
) -> bool:
    """
    Save epochs to disk in FIF format.

    Parameters
    ----------
    epochs      : preprocessed Epochs object
    output_path : Path where the file will be written
    overwrite   : if False and file exists, skip saving
    logger      : logger

    Returns
    -------
    True if saved, False if skipped.
    """
    output_path = output_path if hasattr(output_path, "suffix") else __import__("pathlib").Path(output_path)

    if output_path.exists() and not overwrite:
        logger.info(f"Output exists, skipping: {output_path.name}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(str(output_path), overwrite=True, verbose=False)
    logger.info(f"Saved: {output_path}")
    return True


def save_epochs_numpy(
    epochs: mne.Epochs,
    output_path,
    dtype: str,
    overwrite: bool,
    logger: logging.Logger,
) -> bool:
    """
    Save epochs as a compressed NumPy archive (.npz).

    Saved arrays:
      - ``data``      : float32 array (n_epochs, n_channels, n_times)
      - ``ch_names``  : channel name strings
      - ``sfreq``     : scalar sampling frequency
      - ``task``      : task label per epoch (from metadata)

    Parameters
    ----------
    epochs      : preprocessed Epochs
    output_path : Path with ``.npz`` suffix (auto-added if missing)
    dtype       : NumPy dtype string, e.g. "float32"
    overwrite   : skip if file exists and False
    logger      : logger

    Returns
    -------
    True if saved, False if skipped.
    """
    import pathlib
    output_path = pathlib.Path(output_path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")

    if output_path.exists() and not overwrite:
        logger.info(f"Output exists, skipping: {output_path.name}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = epochs.get_data().astype(dtype)
    ch_names = np.array(epochs.ch_names)
    sfreq = np.array(epochs.info["sfreq"])
    tasks = np.array(
        epochs.metadata["task"].values
        if epochs.metadata is not None
        else ["unknown"] * len(epochs)
    )

    np.savez_compressed(
        str(output_path),
        data=data,
        ch_names=ch_names,
        sfreq=sfreq,
        task=tasks,
    )
    logger.info(f"Saved NumPy: {output_path} | shape={data.shape} | dtype={dtype}")
    return True
