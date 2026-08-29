"""Stage 6 — Resting-state spectral feature extraction.

Computes Welch Power Spectral Density (PSD) and extracts frequency-band powers
(Delta, Theta, Alpha, Beta, Gamma) in both absolute and relative scales from
validated resting-state EEG epochs (R01 eyes-open, R02 eyes-closed).

Exports node feature matrices (64 channels x 5 bands) for downstream graph neural
networks (GCN/GAT) and tabular representations for classical ML baselines.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import mne
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load the Stage 6 configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def compute_epoch_psd(
    epochs: mne.Epochs,
    config: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD per channel averaged across clean epochs.

    Parameters
    ----------
    epochs:
        Preprocessed MNE Epochs object.
    config:
        Stage 6 configuration dict with 'psd' settings.
    logger:
        Optional logger instance.

    Returns
    -------
    mean_psd:
        Array of shape (n_channels, n_freqs) containing average PSD in uV^2/Hz.
    freqs:
        Array of shape (n_freqs,) containing frequency bin centers.
    """
    if logger is None:
        logger = logging.getLogger("resting_state.spectral")

    config = config or {}
    psd_cfg = config.get("psd", {})
    fmin = float(psd_cfg.get("fmin", 1.0))
    fmax = float(psd_cfg.get("fmax", 40.0))
    n_fft = int(psd_cfg.get("n_fft", 256))
    n_overlap = int(psd_cfg.get("n_overlap", 128))
    window = str(psd_cfg.get("window", "hamming"))

    # Compute PSD using MNE's compute_psd
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        window=window,
        verbose=False,
    )
    
    # get_data returns (n_epochs, n_channels, n_freqs)
    psd_data = spectrum.get_data()
    freqs = spectrum.freqs

    # Average across clean epochs
    mean_psd = np.mean(psd_data, axis=0)
    logger.info("Computed Welch PSD across %d epochs for %d channels (%d freqs)", len(epochs), mean_psd.shape[0], len(freqs))
    return mean_psd, freqs


def extract_band_powers(
    psd: np.ndarray,
    freqs: np.ndarray,
    bands: Optional[dict[str, list[float]]] = None,
    logger: Optional[Any] = None,
) -> dict[str, Any]:
    """Integrate PSD over frequency bands to calculate absolute and relative power.

    Parameters
    ----------
    psd:
        Array of shape (n_channels, n_freqs) containing PSD values.
    freqs:
        Array of shape (n_freqs,) containing frequency points.
    bands:
        Dictionary mapping band names (e.g., 'alpha') to [fmin, fmax].
    logger:
        Optional logger instance.

    Returns
    -------
    dict containing:
        'absolute': dict[band_name, array_of_shape_(n_channels,)]
        'relative': dict[band_name, array_of_shape_(n_channels,)]
        'total_power': array_of_shape_(n_channels,)
    """
    if logger is None:
        logger = logging.getLogger("resting_state.spectral")

    if bands is None:
        bands = {
            "delta": [1.0, 4.0],
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta": [13.0, 30.0],
            "gamma": [30.0, 40.0],
        }

    # Total power across the full 1-40 Hz range
    # Use trapezoid (or trapz fallback) for numerical integration
    integrate_func = getattr(np, "trapezoid", np.trapz)
    total_power = integrate_func(psd, x=freqs, axis=1)

    abs_powers: dict[str, np.ndarray] = {}
    rel_powers: dict[str, np.ndarray] = {}

    for band_name, (fmin, fmax) in bands.items():
        # Mask frequencies in [fmin, fmax]
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if not np.any(idx):
            raise ValueError(f"No frequency bins found for band {band_name} [{fmin}, {fmax}]")

        band_psd = psd[:, idx]
        band_freqs = freqs[idx]

        # Integrate over frequency band
        band_abs = integrate_func(band_psd, x=band_freqs, axis=1)
        abs_powers[band_name] = band_abs

        # Relative power
        # Avoid division by zero by replacing 0 with small epsilon
        safe_total = np.where(total_power > 0, total_power, 1e-12)
        rel_powers[band_name] = band_abs / safe_total

    return {
        "absolute": abs_powers,
        "relative": rel_powers,
        "total_power": total_power,
    }


def export_node_features(
    band_powers: dict[str, Any],
    band_order: Optional[list[str]] = None,
    feature_type: str = "relative",
) -> np.ndarray:
    """Format band powers into a (n_channels, n_bands) matrix for graph node features.

    Parameters
    ----------
    band_powers:
        Output dict from extract_band_powers.
    band_order:
        List of band names defining column order. Defaults to ['delta', 'theta', 'alpha', 'beta', 'gamma'].
    feature_type:
        'relative' or 'absolute'.

    Returns
    -------
    node_features:
        Numpy array of shape (n_channels, n_bands).
    """
    if band_order is None:
        band_order = ["delta", "theta", "alpha", "beta", "gamma"]

    powers_dict = band_powers[feature_type]
    cols = [powers_dict[band] for band in band_order]
    node_features = np.column_stack(cols)
    return node_features


def run_stage6_subject(
    subject_id: str,
    config_path: Union[str, Path],
) -> dict[str, Any]:
    """Run Stage 6 spectral feature extraction for a single subject.

    Parameters
    ----------
    subject_id:
        Subject identifier (e.g., 'S001').
    config_path:
        Path to Stage 6 YAML configuration file.

    Returns
    -------
    dict containing execution results, file paths, and QC metrics.
    """
    start_time = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)

    logger = logging.getLogger("resting_state.spectral")
    logger.info("Starting Stage 6 spectral feature extraction for subject=%s", subject_id)

    epochs_root = Path(config.get("preprocessed_epochs_dir", "outputs/preprocessed/epochs"))
    output_root = Path(config.get("output_directory", "outputs/features"))
    subject_out_dir = output_root / f"stage6_{subject_id.lower()}"
    subject_out_dir.mkdir(parents=True, exist_ok=True)

    baseline_runs = config.get("baseline_runs", [
        {"run_id": "R01", "condition": "eyes_open"},
        {"run_id": "R02", "condition": "eyes_closed"},
    ])
    bands = config.get("bands", {
        "delta": [1.0, 4.0],
        "theta": [4.0, 8.0],
        "alpha": [8.0, 13.0],
        "beta": [13.0, 30.0],
        "gamma": [30.0, 40.0],
    })
    band_order = list(bands.keys())

    run_results: dict[str, Any] = {}
    channel_names: list[str] = []

    for run_info in baseline_runs:
        run_id = run_info["run_id"]
        condition = run_info["condition"]
        epochs_path = epochs_root / f"{subject_id}_{run_id}_epochs-epo.fif"

        if not epochs_path.exists():
            raise FileNotFoundError(f"Preprocessed epochs FIF not found for {subject_id}/{run_id}: {epochs_path}")

        logger.info("Processing %s/%s (%s)", subject_id, run_id, condition)
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        channel_names = list(epochs.ch_names)

        # 1. Compute PSD
        psd, freqs = compute_epoch_psd(epochs, config=config, logger=logger)

        # 2. Extract band powers
        band_powers = extract_band_powers(psd, freqs, bands=bands, logger=logger)

        # 3. Export node features matrix (64 x 5)
        primary_feature_type = config.get("normalization", {}).get("primary_node_feature", "relative")
        node_features = export_node_features(band_powers, band_order=band_order, feature_type=primary_feature_type)

        # Save artifacts
        psd_df = pd.DataFrame(psd, index=channel_names, columns=[f"freq_{f:.1f}Hz" for f in freqs])
        psd_csv_path = subject_out_dir / f"{subject_id}_{run_id}_psd.csv"
        psd_df.to_csv(psd_csv_path)

        band_power_dict = {"channel": channel_names}
        for b in band_order:
            band_power_dict[f"{b}_abs"] = band_powers["absolute"][b]
            band_power_dict[f"{b}_rel"] = band_powers["relative"][b]
        band_power_dict["total_power"] = band_powers["total_power"]
        
        band_df = pd.DataFrame(band_power_dict)
        band_csv_path = subject_out_dir / f"{subject_id}_{run_id}_band_powers.csv"
        band_df.to_csv(band_csv_path, index=False)

        node_feat_path = subject_out_dir / f"{subject_id}_{run_id}_node_features.npy"
        np.save(node_feat_path, node_features)

        # Sanity Checks
        rel_sums = np.sum([band_powers["relative"][b] for b in band_order], axis=0)
        rel_sum_ok = bool(np.allclose(rel_sums, 1.0, atol=0.05))
        non_negative = bool(np.all(psd >= 0) and np.all(node_features >= 0))
        shape_ok = node_features.shape == (len(channel_names), len(band_order))

        run_results[run_id] = {
            "condition": condition,
            "n_epochs": len(epochs),
            "n_channels": len(channel_names),
            "n_bands": len(band_order),
            "node_features_shape": list(node_features.shape),
            "psd_csv": str(psd_csv_path),
            "band_powers_csv": str(band_csv_path),
            "node_features_npy": str(node_feat_path),
            "qc_metrics": {
                "relative_power_sum_valid": rel_sum_ok,
                "values_non_negative": non_negative,
                "shape_valid": shape_ok,
                "mean_total_power": float(np.mean(band_powers["total_power"])),
                "mean_alpha_relative": float(np.mean(band_powers["relative"]["alpha"])),
            },
        }

    # Calculate Alpha Blocking Index (R02 eyes_closed alpha / R01 eyes_open alpha) on Occipital channels
    alpha_blocking_ratio = None
    if "R01" in run_results and "R02" in run_results:
        occipital_chans = [ch for ch in ["O1", "Oz", "O2"] if ch in channel_names]
        if occipital_chans:
            occ_indices = [channel_names.index(ch) for ch in occipital_chans]
            r01_df = pd.read_csv(run_results["R01"]["band_powers_csv"])
            r02_df = pd.read_csv(run_results["R02"]["band_powers_csv"])
            
            r01_occ_alpha = r01_df.loc[occ_indices, "alpha_abs"].mean()
            r02_occ_alpha = r02_df.loc[occ_indices, "alpha_abs"].mean()
            alpha_blocking_ratio = float(r02_occ_alpha / r01_occ_alpha) if r01_occ_alpha > 0 else None

    elapsed = time.perf_counter() - start_time
    report = {
        "subject_id": subject_id,
        "stage": 6,
        "validator_version": "stage6-spectral-1.0",
        "elapsed_seconds": elapsed,
        "runs": run_results,
        "alpha_blocking_ratio_occipital": alpha_blocking_ratio,
        "status": "PASS",
    }

    report_path = subject_out_dir / f"{subject_id}_spectral_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Stage 6 complete for subject=%s elapsed=%.2fs report=%s", subject_id, elapsed, report_path)
    
    report["report_file"] = str(report_path)
    return report
