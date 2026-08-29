"""Bad-channel detection and interpolation utilities for Stage 2C1.

This module provides a lightweight, configurable workflow for identifying
problematic channels in an MNE Raw object and interpolating them using MNE's
standard methods. It does not implement ICA or any other artifact removal
steps.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import mne
import numpy as np

from .utils import get_logger


def detect_bad_channels(
    raw: mne.io.BaseRaw,
    *,
    config: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
) -> list[str]:
    """Detect bad channels using configurable heuristics."""

    if logger is None:
        logger = get_logger("preprocessing.artifacts")

    config = config or {}
    bad_cfg = config.get("bad_channels", {})
    detection_method = bad_cfg.get("detection_method", "variance")
    max_bad_channels = int(bad_cfg.get("max_bad_channels", 4))

    if not bad_cfg.get("enabled", False):
        return []

    data = raw.get_data(picks="eeg")
    channel_names = raw.ch_names

    detected: list[str] = []

    if detection_method in {"flat", "all"}:
        flat_mask = np.isclose(data, 0.0).all(axis=1)
        for idx, name in enumerate(channel_names):
            if flat_mask[idx]:
                detected.append(name)

    if detection_method in {"variance", "all"}:
        variance = data.var(axis=1)
        variance_threshold = float(bad_cfg.get("variance_threshold", np.nanstd(variance) * 5.0 + np.nanmean(variance)))
        for idx, name in enumerate(channel_names):
            if variance[idx] > variance_threshold:
                detected.append(name)

    if detection_method in {"amplitude", "all"}:
        amplitude_threshold = float(bad_cfg.get("amplitude_threshold", 1e6))
        for idx, name in enumerate(channel_names):
            if np.max(np.abs(data[idx])) > amplitude_threshold:
                detected.append(name)

    unique_bad = sorted(set(detected))
    if len(unique_bad) > max_bad_channels:
        unique_bad = unique_bad[:max_bad_channels]

    logger.info("Detected bad channels: %s", unique_bad)
    return unique_bad


def mark_bad_channels(raw: mne.io.BaseRaw, bad_channels: list[str]) -> mne.io.BaseRaw:
    """Mark channels as bad in the Raw object metadata."""

    marked = raw.copy().load_data(verbose=False)
    if bad_channels:
        marked.info["bads"] = sorted(set(marked.info["bads"]) | set(bad_channels))
    return marked


def interpolate_bad_channels(
    raw: mne.io.BaseRaw,
    *,
    method: str = "spline",
    logger: Optional[Any] = None,
) -> mne.io.BaseRaw:
    """Interpolate bad channels using MNE interpolation."""

    if logger is None:
        logger = get_logger("preprocessing.artifacts")

    if not raw.info["bads"]:
        logger.info("No bad channels to interpolate")
        return raw.copy().load_data(verbose=False)

    interpolated = raw.copy().load_data(verbose=False)
    interpolated.interpolate_bads(reset_bads=True, verbose=False)
    logger.info("Interpolated bad channels: %s", interpolated.info["bads"])
    return interpolated


def run_bad_channel_pipeline(
    raw: mne.io.BaseRaw,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
    save_interpolated: bool = False,
    output_dir: Optional[str | os.PathLike[str]] = None,
    overwrite: bool = False,
) -> mne.io.BaseRaw:
    """Run the full bad-channel detection and interpolation workflow."""

    if logger is None:
        logger = get_logger("preprocessing.artifacts")

    config = config or {}
    bad_cfg = config.get("bad_channels", {})
    start_time = time.perf_counter()

    subject_name = subject or "unknown"
    run_name = run or "unknown"

    logger.info("Starting bad-channel pipeline for subject=%s run=%s", subject_name, run_name)

    detected = detect_bad_channels(raw, config=config, logger=logger)
    if detected:
        marked = mark_bad_channels(raw, detected)
        cleaned = interpolate_bad_channels(marked, method=bad_cfg.get("interpolation_method", "spline"), logger=logger)
        logger.info("Interpolation performed for %s", detected)
    else:
        cleaned = raw.copy().load_data(verbose=False)
        logger.info("No bad channels detected")

    elapsed = time.perf_counter() - start_time
    logger.info("Bad-channel pipeline complete for subject=%s run=%s elapsed=%.3fs", subject_name, run_name, elapsed)

    if save_interpolated and detected:
        output_path = save_interpolated_raw(
            cleaned,
            subject=subject_name,
            run=run_name,
            output_dir=output_dir,
            overwrite=overwrite,
            logger=logger,
        )
        logger.info("Saved interpolated output to %s", output_path)

    return cleaned


def save_interpolated_raw(
    raw: mne.io.BaseRaw,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    output_dir: Optional[str | os.PathLike[str]] = None,
    overwrite: bool = False,
    logger: Optional[Any] = None,
) -> Path:
    """Persist an interpolated Raw object to disk with a deterministic name."""

    if logger is None:
        logger = get_logger("preprocessing.artifacts")

    base_dir = Path(output_dir) if output_dir is not None else Path("outputs") / "preprocessed" / "interpolated"
    base_dir.mkdir(parents=True, exist_ok=True)

    subject_name = subject or "unknown"
    run_name = run or "unknown"
    output_path = base_dir / f"{subject_name}_{run_name}_interpolated_raw.fif"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Interpolated output already exists and overwrite=False: {output_path}")

    raw.save(output_path, overwrite=overwrite, verbose=False)
    logger.info("Saved interpolated Raw to %s", output_path)
    return output_path


def fit_ica(
    raw: mne.io.BaseRaw,
    *,
    config: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
) -> mne.preprocessing.ICA:
    """Fit an ICA model on a copy of the input Raw object."""

    if logger is None:
        logger = get_logger("preprocessing.artifacts")

    config = config or {}
    ica_cfg = config.get("ica", {})

    if not ica_cfg.get("enabled", False):
        raise ValueError("ICA is disabled in configuration")

    method = ica_cfg.get("method", "fastica")
    n_components = int(ica_cfg.get("n_components", 15))
    random_seed = int(ica_cfg.get("random_seed", 42))
    max_iter = int(ica_cfg.get("max_iter", 1000))

    working_raw = raw.copy().load_data(verbose=False)
    highpass = working_raw.info.get("highpass")
    if highpass is None or float(highpass) < 1.0:
        logger.warning(
            "ICA is being fit on data that are not high-pass filtered at 1 Hz or above; current highpass=%.3f Hz",
            float(highpass) if highpass is not None else float("nan"),
        )

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_seed,
        max_iter=max_iter,
        verbose=False,
    )
    ica.fit(working_raw, verbose=False)
    logger.info("Fitted ICA with method=%s n_components=%d", method, n_components)
    return ica


def detect_artifact_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    *,
    config: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
) -> list[int]:
    """Detect artifact-related ICA components automatically when possible."""

    if logger is None:
        logger = get_logger("preprocessing.artifacts")

    config = config or {}
    ica_cfg = config.get("ica", {})
    auto_eog = bool(ica_cfg.get("auto_detect_eog", True))
    auto_ecg = bool(ica_cfg.get("auto_detect_ecg", True))

    candidate_components: list[int] = []

    if auto_eog:
        try:
            eog_indices, _, _ = ica.find_bads_eog(raw, ch_name=None, verbose=False)
            candidate_components.extend(int(idx) for idx in eog_indices)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("EOG component detection failed: %s", exc)

    if auto_ecg:
        try:
            ecg_indices, _, _ = ica.find_bads_ecg(raw, ch_name=None, verbose=False)
            candidate_components.extend(int(idx) for idx in ecg_indices)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("ECG component detection failed: %s", exc)

    components = sorted(set(candidate_components))
    logger.info("Detected artifact components: %s", components)
    return components


def remove_artifact_components(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    components: list[int],
    *,
    logger: Optional[Any] = None,
) -> mne.io.BaseRaw:
    """Apply ICA to remove the specified components from the data."""

    if logger is None:
        logger = get_logger("preprocessing.artifacts")

    working_raw = raw.copy().load_data(verbose=False)
    cleaned = ica.apply(working_raw, exclude=components, verbose=False)
    logger.info("Removed ICA components: %s", components)
    return cleaned


def run_ica_pipeline(
    raw: mne.io.BaseRaw,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
    save_ica_model: bool = False,
    output_dir: Optional[str | os.PathLike[str]] = None,
    overwrite: bool = False,
) -> tuple[mne.io.BaseRaw, Optional[mne.preprocessing.ICA]]:
    """Run the ICA fitting, artifact detection, and component removal workflow."""

    if logger is None:
        logger = get_logger("preprocessing.artifacts")

    config = config or {}
    ica_cfg = config.get("ica", {})
    start_time = time.perf_counter()

    subject_name = subject or "unknown"
    run_name = run or "unknown"

    logger.info("Starting ICA pipeline for subject=%s run=%s", subject_name, run_name)

    if not ica_cfg.get("enabled", False):
        logger.info("ICA disabled; returning original Raw")
        return raw.copy().load_data(verbose=False), None

    ica = fit_ica(raw, config=config, logger=logger)
    components = detect_artifact_components(ica, raw, config=config, logger=logger)
    cleaned = remove_artifact_components(raw, ica, components, logger=logger)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "ICA pipeline complete for subject=%s run=%s elapsed=%.3fs components_removed=%s",
        subject_name,
        run_name,
        elapsed,
        components,
    )

    if save_ica_model and ica_cfg.get("save_ica_model", False):
        output_path = save_ica_model_to_disk(
            ica,
            subject=subject_name,
            run=run_name,
            output_dir=output_dir,
            overwrite=overwrite,
            logger=logger,
        )
        logger.info("Saved ICA model to %s", output_path)

    return cleaned, ica


def save_ica_model_to_disk(
    ica: mne.preprocessing.ICA,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    output_dir: Optional[str | os.PathLike[str]] = None,
    overwrite: bool = False,
    logger: Optional[Any] = None,
) -> Path:
    """Persist a fitted ICA object to disk using a deterministic file name."""

    if logger is None:
        logger = get_logger("preprocessing.artifacts")

    base_dir = Path(output_dir) if output_dir is not None else Path("outputs") / "preprocessed" / "ica"
    base_dir.mkdir(parents=True, exist_ok=True)

    subject_name = subject or "unknown"
    run_name = run or "unknown"
    output_path = base_dir / f"{subject_name}_{run_name}_ica.fif"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"ICA model already exists and overwrite=False: {output_path}")

    ica.save(output_path, overwrite=overwrite)
    logger.info("Saved ICA object to %s", output_path)
    return output_path
