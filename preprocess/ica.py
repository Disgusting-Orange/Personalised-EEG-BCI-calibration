"""
ica.py
------
ICA-based artefact removal for eye blinks and muscle noise.

Strategy
--------
1. Fit ICA on a 1 Hz high-passed copy of the bandpass-filtered signal
   (high-pass removes slow drifts that dominate ICA decomposition).
2. Auto-detect EOG components via correlation with frontal channels
   (Fp1, Fp2 used as EOG proxies since the dataset has no dedicated
   EOG channel).
3. Auto-detect muscle (EMG) components via z-score on the high-frequency
   power of each component's time-series.
4. Apply the exclusion list back onto the original bandpass-filtered Raw.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np
from mne.preprocessing import ICA


# ── Core ICA fitting ──────────────────────────────────────────────────────────

def fit_ica(
    raw_filtered: mne.io.BaseRaw,
    n_components: int,
    method: str,
    max_iter: int,
    random_state: int,
    hp_cutoff: float,
    n_jobs: int,
    logger: logging.Logger,
) -> Tuple[ICA, mne.io.BaseRaw]:
    """
    Fit ICA on a high-passed copy of *raw_filtered*.

    Parameters
    ----------
    raw_filtered  : bandpass-filtered Raw (not mutated)
    n_components  : number of ICA components
    method        : "fastica" | "infomax" | "picard"
    max_iter      : maximum ICA iterations
    random_state  : RNG seed for reproducibility
    hp_cutoff     : high-pass cutoff applied before fitting (Hz)
    n_jobs        : parallel jobs
    logger        : logger instance

    Returns
    -------
    (fitted ICA object, high-passed Raw used for fitting)
    """
    logger.info(f"Fitting ICA ({method}, {n_components} components) …")

    # High-pass the filtered signal before ICA fitting
    raw_for_ica = raw_filtered.copy()
    raw_for_ica.filter(
        l_freq=hp_cutoff,
        h_freq=None,
        method="fir",
        fir_window="hamming",
        n_jobs=n_jobs,
        verbose=False,
    )

    ica = ICA(
        n_components=n_components,
        method=method,
        max_iter=max_iter,
        random_state=random_state,
        fit_params={"extended": True} if method == "infomax" else None,
    )

    try:
        ica.fit(raw_for_ica, verbose=False)
        if hasattr(ica, "pca_explained_variance_"):
            explained = (
                ica.pca_explained_variance_[:n_components].sum()
                / ica.pca_explained_variance_.sum()
            )
            logger.info(
                f"ICA fitted: {ica.n_components_} components, "
                f"explained variance: {explained:.1%}"
            )
        else:
            logger.info(
                f"ICA fitted: {ica.n_components_} components"
            )
    except Exception as exc:
        logger.error(f"ICA fitting failed: {exc}")
        raise

    return ica, raw_for_ica


# ── EOG artefact detection ────────────────────────────────────────────────────

def detect_eog_components(
    ica: ICA,
    raw_hp: mne.io.BaseRaw,
    eog_channels: List[str],
    threshold: float,
    logger: logging.Logger,
) -> List[int]:
    """
    Identify ICA components correlated with frontal (EOG-proxy) channels.

    Parameters
    ----------
    ica          : fitted ICA object
    raw_hp       : high-passed Raw (same one used for ICA fitting)
    eog_channels : list of channel names to use as EOG proxies
    threshold    : z-score threshold for detection
    logger       : logger

    Returns
    -------
    List of component indices flagged as EOG artefacts.
    """
    eog_indices: List[int] = []
    available = [ch for ch in eog_channels if ch in raw_hp.ch_names]

    if not available:
        logger.warning("No EOG proxy channels found — skipping EOG detection.")
        return eog_indices

    for ch in available:
        try:
            indices, scores = ica.find_bads_eog(
                raw_hp,
                ch_name=ch,
                threshold=threshold,
                verbose=False,
            )
            eog_indices.extend(indices)
            logger.info(
                f"EOG detection ({ch}): components {indices} "
                f"(scores {[f'{s:.2f}' for s in scores[indices] if len(scores) > 0]})"
            )
        except Exception as exc:
            logger.warning(f"EOG detection failed for {ch}: {exc}")

    # Deduplicate
    eog_indices = sorted(set(eog_indices))
    return eog_indices


# ── Muscle artefact detection ─────────────────────────────────────────────────

def detect_muscle_components(
    ica: ICA,
    raw_hp: mne.io.BaseRaw,
    threshold: float,
    logger: logging.Logger,
) -> List[int]:
    """
    Identify ICA components dominated by high-frequency (muscle) activity
    using MNE's ``find_bads_muscle`` if available, otherwise fall back to
    a kurtosis-based heuristic.

    Parameters
    ----------
    ica       : fitted ICA object
    raw_hp    : high-passed Raw
    threshold : z-score threshold
    logger    : logger

    Returns
    -------
    List of component indices flagged as muscle artefacts.
    """
    muscle_indices: List[int] = []

    # Preferred: MNE ≥1.0 has find_bads_muscle
    if hasattr(ica, "find_bads_muscle"):
        try:
            indices, scores = ica.find_bads_muscle(
                raw_hp,
                threshold=threshold,
                verbose=False,
            )
            muscle_indices.extend(indices)
            logger.info(f"Muscle detection: components {indices}")
            return sorted(set(muscle_indices))
        except Exception as exc:
            logger.warning(f"find_bads_muscle failed ({exc}), using kurtosis fallback.")

    # Fallback: kurtosis-based detection
    try:
        sources = ica.get_sources(raw_hp).get_data()  # (n_components, n_times)
        kurtosis = _kurtosis(sources)                  # (n_components,)
        z_kurt = (kurtosis - kurtosis.mean()) / (kurtosis.std() + 1e-10)
        muscle_indices = list(np.where(z_kurt > threshold)[0])
        logger.info(
            f"Muscle detection (kurtosis fallback): components {muscle_indices}"
        )
    except Exception as exc:
        logger.warning(f"Kurtosis-based muscle detection failed: {exc}")

    return sorted(set(muscle_indices))


def _kurtosis(data: np.ndarray) -> np.ndarray:
    """Compute excess kurtosis along axis=1 (time axis)."""
    mu = data.mean(axis=1, keepdims=True)
    sigma = data.std(axis=1, keepdims=True) + 1e-10
    z = (data - mu) / sigma
    kurt = np.mean(z ** 4, axis=1) - 3.0
    return kurt


# ── Application ───────────────────────────────────────────────────────────────

def apply_ica(
    ica: ICA,
    raw_filtered: mne.io.BaseRaw,
    exclude: List[int],
    logger: logging.Logger,
) -> mne.io.BaseRaw:
    """
    Apply ICA exclusion list to *raw_filtered* and return the cleaned Raw.

    Parameters
    ----------
    ica          : fitted ICA object
    raw_filtered : bandpass-filtered Raw (not mutated)
    exclude      : list of component indices to remove
    logger       : logger

    Returns
    -------
    Clean mne.io.BaseRaw with artefact components removed
    """
    if not exclude:
        logger.info("No ICA components to exclude — returning original.")
        return raw_filtered.copy()

    ica.exclude = exclude
    logger.info(f"Applying ICA: excluding components {exclude}")

    raw_clean = raw_filtered.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean


# ── High-level entry point ────────────────────────────────────────────────────

def run_ica_pipeline(
    raw_filtered: mne.io.BaseRaw,
    n_components: int,
    method: str,
    max_iter: int,
    random_state: int,
    eog_channels: List[str],
    eog_threshold: float,
    muscle_threshold: float,
    hp_cutoff: float,
    n_jobs: int,
    logger: logging.Logger,
    figure_dir: Optional[Path] = None,
    subject_id: str = "",
    run_id: str = "",
) -> Tuple[mne.io.BaseRaw, ICA, List[int]]:
    """
    Full ICA pipeline: fit → detect artefacts → apply.

    Parameters
    ----------
    raw_filtered      : bandpass+notch filtered Raw
    n_components      : ICA components to fit
    method            : ICA algorithm
    max_iter          : max iterations
    random_state      : RNG seed
    eog_channels      : EOG proxy channel names
    eog_threshold     : EOG z-score threshold
    muscle_threshold  : muscle z-score threshold
    hp_cutoff         : high-pass cutoff for ICA fitting
    n_jobs            : parallel jobs
    logger            : logger
    figure_dir        : if given, save ICA component plots here
    subject_id        : used in figure filenames
    run_id            : used in figure filenames

    Returns
    -------
    (raw_ica_cleaned, fitted_ica, excluded_components)
    """
    # 1. Fit
    ica, raw_hp = fit_ica(
        raw_filtered, n_components, method, max_iter,
        random_state, hp_cutoff, n_jobs, logger,
    )

    # 2. Detect
    eog_idx = detect_eog_components(
        ica, raw_hp, eog_channels, eog_threshold, logger,
    )
    muscle_idx = detect_muscle_components(
        ica, raw_hp, muscle_threshold, logger,
    )

    all_excluded = sorted(set(eog_idx + muscle_idx))
    logger.info(
        f"Total excluded: {len(all_excluded)} components "
        f"(EOG: {eog_idx}, Muscle: {muscle_idx})"
    )

    # 3. Apply
    raw_clean = apply_ica(ica, raw_filtered, all_excluded, logger)

    return raw_clean, ica, all_excluded
