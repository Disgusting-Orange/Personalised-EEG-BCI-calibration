"""
filters.py
----------
Bandpass and notch filtering for EEG data.

Design choices
--------------
* Bandpass  : zero-phase FIR filter (MNE default) via ``raw.filter()``
* Notch     : zero-phase notch FIR at 60 Hz (+ harmonics up to Nyquist)
* A copy of the raw object is returned so the caller retains the
  unfiltered original for visualisation.
"""

import logging
from typing import List, Optional, Tuple

import mne
import numpy as np


def bandpass_filter(
    raw: mne.io.BaseRaw,
    low: float,
    high: float,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> mne.io.BaseRaw:
    """
    Apply zero-phase FIR bandpass filter.

    Parameters
    ----------
    raw    : Raw object (modified in place and also returned)
    low    : lower cutoff frequency in Hz
    high   : upper cutoff frequency in Hz
    n_jobs : number of parallel jobs (-1 = all cores)
    logger : logger instance

    Returns
    -------
    Filtered mne.io.BaseRaw (same object, in-place)
    """
    if logger:
        logger.info(f"Bandpass filter: {low}–{high} Hz")
    raw.filter(
        l_freq=low,
        h_freq=high,
        method="fir",
        fir_window="hamming",
        fir_design="firwin",
        n_jobs=n_jobs,
        verbose=False,
    )
    return raw


def notch_filter(
    raw: mne.io.BaseRaw,
    notch_freq: float,
    notch_width: float = 2.0,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> mne.io.BaseRaw:
    """
    Apply zero-phase notch filter at *notch_freq* Hz and its harmonics
    (up to the Nyquist frequency).

    Parameters
    ----------
    raw         : Raw object (modified in place and also returned)
    notch_freq  : fundamental notch frequency in Hz (e.g. 60 for US)
    notch_width : bandwidth (Hz) around each notch frequency
    n_jobs      : number of parallel jobs
    logger      : logger instance

    Returns
    -------
    Notch-filtered mne.io.BaseRaw
    """
    nyquist = raw.info["sfreq"] / 2.0
    freqs: List[float] = []
    harmonic = notch_freq
    while harmonic <= nyquist:
        freqs.append(harmonic)
        harmonic += notch_freq

    if not freqs:
        if logger:
            logger.warning(
                f"Notch frequency {notch_freq} Hz exceeds Nyquist "
                f"({nyquist} Hz) — skipping notch filter."
            )
        return raw

    if logger:
        logger.info(f"Notch filter at {freqs} Hz")

    raw.notch_filter(
        freqs=freqs,
        method="fir",
        fir_window="hamming",
        notch_widths=notch_width,
        n_jobs=n_jobs,
        verbose=False,
    )
    return raw


def apply_filters(
    raw: mne.io.BaseRaw,
    bandpass_low: float,
    bandpass_high: float,
    notch_freq: float,
    notch_width: float = 2.0,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw]:
    """
    Convenience wrapper that applies both filters and returns
    (raw_unfiltered, raw_filtered).

    The unfiltered copy is kept for visualisation / comparison purposes.

    Parameters
    ----------
    raw           : original loaded Raw (will be copied; not mutated)
    bandpass_low  : bandpass lower cutoff Hz
    bandpass_high : bandpass upper cutoff Hz
    notch_freq    : notch fundamental Hz
    notch_width   : notch bandwidth Hz
    n_jobs        : parallel jobs
    logger        : logger

    Returns
    -------
    (raw_original_copy, raw_filtered)
    """
    # Keep a copy of the raw signal for later comparison visualisation
    raw_original = raw.copy()

    raw_filtered = raw.copy()
    bandpass_filter(raw_filtered, bandpass_low, bandpass_high, n_jobs, logger)
    notch_filter(raw_filtered, notch_freq, notch_width, n_jobs, logger)

    return raw_original, raw_filtered


def high_pass_copy(
    raw: mne.io.BaseRaw,
    cutoff: float = 1.0,
    n_jobs: int = -1,
) -> mne.io.BaseRaw:
    """
    Return a high-pass filtered copy of *raw* (used for ICA fitting).
    ICA benefits from high-pass filtering at ≥1 Hz to reduce slow drifts
    that can dominate the decomposition.

    Parameters
    ----------
    raw    : source Raw object (not mutated)
    cutoff : high-pass cutoff in Hz

    Returns
    -------
    New mne.io.BaseRaw filtered at *cutoff* Hz
    """
    raw_hp = raw.copy()
    raw_hp.filter(
        l_freq=cutoff,
        h_freq=None,
        method="fir",
        fir_window="hamming",
        n_jobs=n_jobs,
        verbose=False,
    )
    return raw_hp


def compute_psd_summary(
    raw: mne.io.BaseRaw,
    fmin: float = 0.5,
    fmax: float = 50.0,
    n_fft: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average PSD across all channels using Welch's method.

    Returns
    -------
    freqs : 1-D array of frequency bins
    psds  : 2-D array (n_channels, n_freqs) in dB re 1 µV²/Hz
    """
    spectrum = raw.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        verbose=False,
    )
    psds, freqs = spectrum.get_data(return_freqs=True)
    # Convert to dB
    psds_db = 10 * np.log10(psds + 1e-30)
    return freqs, psds_db
