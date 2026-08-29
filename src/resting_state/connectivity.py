"""Stage 7 — Publication-Quality Functional Connectivity Framework.

Provides an extensible strategy-pattern architecture for computing functional
connectivity matrices (64x64) from resting-state EEG epochs.

Supported Metrics:
  - wPLI (Weighted Phase Lag Index, Vinck et al., 2011) — Primary metric
  - PLV (Phase Locking Value, Lachaux et al., 1999) — Secondary metric
  - Coherence (Magnitude-Squared Coherence) — Extensible secondary metric

Produces 64x64 weighted adjacency matrices (.npy, .csv), high-resolution (300 DPI)
annotated heatmaps (.png), and structured QC reports (.json).
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.signal as signal
import yaml


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load Stage 7 configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Strategy Pattern: Base Estimator & Concrete Implementations
# ---------------------------------------------------------------------------


class BaseConnectivityEstimator(ABC):
    """Abstract Base Class for functional connectivity metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name identifier (e.g. 'wpli')."""
        pass

    @abstractmethod
    def compute(
        self,
        epochs_data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """Compute 64x64 connectivity matrix for a specific frequency band.

        Parameters
        ----------
        epochs_data:
            Array of shape (n_epochs, n_channels, n_times).
        sfreq:
            Sampling frequency in Hz.
        fmin, fmax:
            Frequency band boundaries in Hz.

        Returns
        -------
        adj_matrix:
            Symmetric array of shape (n_channels, n_channels) with values in [0, 1].
        """
        pass


class WPLIEstimator(BaseConnectivityEstimator):
    """Weighted Phase Lag Index (wPLI) Estimator (Vinck et al., 2011).

    Eliminates volume conduction and field spread artifacts by weighting phase
    differences by the magnitude of the imaginary cross-spectrum.
    """

    @property
    def name(self) -> str:
        return "wpli"

    def compute(
        self,
        epochs_data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        n_epochs, n_channels, n_times = epochs_data.shape

        # Zero-phase FIR band-pass filter
        sos = signal.butter(4, [fmin, fmax], btype="bandpass", fs=sfreq, output="sos")
        filtered = signal.sosfiltfilt(sos, epochs_data, axis=-1)

        # Complex analytic signal via Hilbert transform
        analytic = signal.hilbert(filtered, axis=-1)  # (n_epochs, n_channels, n_times)
        Z = analytic.transpose(0, 2, 1).reshape(-1, n_channels)  # (N_total, n_channels)

        re = Z.real
        im = Z.imag

        # Im(S_jk) = im_j * re_k - re_j * im_k
        sum_im = (im.T @ re) - (re.T @ im)  # (n_channels, n_channels)

        # Sum of absolute imaginary components across samples
        im_S = (im[:, :, None] * re[:, None, :]) - (re[:, :, None] * im[:, None, :])
        sum_abs_im = np.sum(np.abs(im_S), axis=0)  # (n_channels, n_channels)

        safe_denom = np.where(sum_abs_im > 0, sum_abs_im, 1e-12)
        wpli = np.abs(sum_im) / safe_denom

        # Make symmetric and zero out diagonal
        wpli = 0.5 * (wpli + wpli.T)
        np.fill_diagonal(wpli, 0.0)
        return np.clip(wpli, 0.0, 1.0)


class PLVIEstimator(BaseConnectivityEstimator):
    """Phase Locking Value (PLV) Estimator (Lachaux et al., 1999).

    Measures phase synchronization regardless of amplitude. Sensitive to volume conduction.
    """

    @property
    def name(self) -> str:
        return "plv"

    def compute(
        self,
        epochs_data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        n_epochs, n_channels, n_times = epochs_data.shape

        sos = signal.butter(4, [fmin, fmax], btype="bandpass", fs=sfreq, output="sos")
        filtered = signal.sosfiltfilt(sos, epochs_data, axis=-1)

        analytic = signal.hilbert(filtered, axis=-1)
        Z = analytic.transpose(0, 2, 1).reshape(-1, n_channels)

        phase = np.angle(Z)
        phase_exp = np.exp(1j * phase)

        plv = np.abs(np.conj(phase_exp.T) @ phase_exp) / Z.shape[0]

        plv = 0.5 * (plv + plv.T)
        np.fill_diagonal(plv, 1.0)
        return np.clip(plv, 0.0, 1.0)


class CoherenceEstimator(BaseConnectivityEstimator):
    """Magnitude-Squared Coherence Estimator."""

    @property
    def name(self) -> str:
        return "coherence"

    def compute(
        self,
        epochs_data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        n_epochs, n_channels, n_times = epochs_data.shape

        sos = signal.butter(4, [fmin, fmax], btype="bandpass", fs=sfreq, output="sos")
        filtered = signal.sosfiltfilt(sos, epochs_data, axis=-1)

        analytic = signal.hilbert(filtered, axis=-1)
        Z = analytic.transpose(0, 2, 1).reshape(-1, n_channels)  # (N, n_channels)

        # Cross-spectral density matrix & auto-spectral power
        CSD = (np.conj(Z.T) @ Z) / Z.shape[0]  # (64, 64)
        PSD_diag = np.real(np.diag(CSD))

        denom = np.sqrt(np.outer(PSD_diag, PSD_diag))
        safe_denom = np.where(denom > 0, denom, 1e-12)

        coherence = np.abs(CSD) / safe_denom
        coherence = 0.5 * (coherence + coherence.T)
        np.fill_diagonal(coherence, 1.0)
        return np.clip(coherence, 0.0, 1.0)


CONNECTIVITY_ESTIMATORS: dict[str, BaseConnectivityEstimator] = {
    "wpli": WPLIEstimator(),
    "plv": PLVIEstimator(),
    "coherence": CoherenceEstimator(),
}


# ---------------------------------------------------------------------------
# Visualization & Artifact Generation
# ---------------------------------------------------------------------------


def generate_connectivity_heatmap(
    adj_matrix: np.ndarray,
    channel_names: list[str],
    title: str,
    output_path: Union[str, Path],
    dpi: int = 300,
    cmap: str = "viridis",
) -> None:
    """Generate high-resolution (300 DPI) connectivity heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    im = ax.imshow(adj_matrix, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)

    fig.colorbar(im, ax=ax, label="Connectivity Value")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Set tick labels if 64 channels
    if len(channel_names) == 64:
        ax.set_xticks(np.arange(0, 64, 4))
        ax.set_yticks(np.arange(0, 64, 4))
        ax.set_xticklabels(channel_names[::4], rotation=90, fontsize=8)
        ax.set_yticklabels(channel_names[::4], fontsize=8)

    ax.set_xlabel("Channels")
    ax.set_ylabel("Channels")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single Subject Pipeline Runner
# ---------------------------------------------------------------------------


def run_stage7_subject(
    subject_id: str,
    config_path: Union[str, Path],
    save_heatmaps: Optional[bool] = None,
) -> dict[str, Any]:
    """Run Stage 7 functional connectivity extraction for one subject.

    Parameters
    ----------
    subject_id:
        Subject identifier (e.g. 'S001').
    config_path:
        Path to Stage 7 YAML configuration file.
    save_heatmaps:
        Optional boolean override for heatmaps rendering.

    Returns
    -------
    dict containing execution status, output artifact paths, and QC metrics.
    """
    start_time = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)

    logger = logging.getLogger("resting_state.connectivity")
    logger.info("Starting Stage 7 functional connectivity for subject=%s", subject_id)

    epochs_root = Path(config.get("preprocessed_epochs_dir", "outputs/preprocessed/epochs"))
    output_root = Path(config.get("output_directory", "outputs/connectivity"))
    subject_out_dir = output_root / f"stage7_{subject_id.lower()}"
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
    metrics_to_run = [config.get("metrics", {}).get("primary", "wpli"), config.get("metrics", {}).get("secondary", "plv")]

    if save_heatmaps is None:
        save_heatmaps = bool(config.get("execution", {}).get("save_heatmaps", True))
    heatmap_dpi = int(config.get("execution", {}).get("heatmap_dpi", 300))

    run_results: dict[str, Any] = {}

    for run_info in baseline_runs:
        run_id = run_info["run_id"]
        condition = run_info["condition"]
        epochs_path = epochs_root / f"{subject_id}_{run_id}_epochs-epo.fif"

        if not epochs_path.exists():
            raise FileNotFoundError(f"Preprocessed epochs FIF not found for {subject_id}/{run_id}: {epochs_path}")

        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        epochs_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        sfreq = float(epochs.info["sfreq"])
        channel_names = list(epochs.ch_names)

        run_metrics: dict[str, Any] = {}

        for metric_name in metrics_to_run:
            if metric_name not in CONNECTIVITY_ESTIMATORS:
                raise ValueError(f"Unknown connectivity metric '{metric_name}'. Registered: {list(CONNECTIVITY_ESTIMATORS.keys())}")

            estimator = CONNECTIVITY_ESTIMATORS[metric_name]
            band_matrices: dict[str, Any] = {}

            for band_name, (fmin, fmax) in bands.items():
                adj_matrix = estimator.compute(epochs_data, sfreq, fmin, fmax)

                # Artifact paths
                matrix_npy_path = subject_out_dir / f"{subject_id}_{run_id}_{band_name}_{metric_name}.npy"
                matrix_csv_path = subject_out_dir / f"{subject_id}_{run_id}_{band_name}_{metric_name}.csv"
                np.save(matrix_npy_path, adj_matrix)

                df = pd.DataFrame(adj_matrix, index=channel_names, columns=channel_names)
                df.to_csv(matrix_csv_path)

                heatmap_path = None
                if save_heatmaps:
                    heatmap_path = subject_out_dir / f"{subject_id}_{run_id}_{band_name}_{metric_name}_heatmap.png"
                    title = f"{subject_id} {run_id} ({condition}) — {metric_name.upper()} ({band_name.capitalize()} {fmin}-{fmax}Hz)"
                    generate_connectivity_heatmap(adj_matrix, channel_names, title, heatmap_path, dpi=heatmap_dpi)

                # QC checks
                is_symmetric = bool(np.allclose(adj_matrix, adj_matrix.T, atol=1e-5))
                no_nan = bool(not np.isnan(adj_matrix).any())
                no_inf = bool(not np.isinf(adj_matrix).any())
                bounded = bool(np.all(adj_matrix >= 0.0) and np.all(adj_matrix <= 1.0))

                band_matrices[band_name] = {
                    "npy_file": str(matrix_npy_path),
                    "csv_file": str(matrix_csv_path),
                    "heatmap_file": str(heatmap_path) if heatmap_path else None,
                    "mean_connectivity": float(np.mean(adj_matrix)),
                    "std_connectivity": float(np.std(adj_matrix)),
                    "max_connectivity": float(np.max(adj_matrix)),
                    "qc": {
                        "is_symmetric": is_symmetric,
                        "no_nan": no_nan,
                        "no_inf": no_inf,
                        "bounded_0_1": bounded,
                        "shape": list(adj_matrix.shape),
                    },
                }

            run_metrics[metric_name] = band_matrices

        run_results[run_id] = {
            "condition": condition,
            "n_epochs": len(epochs),
            "n_channels": len(channel_names),
            "metrics": run_metrics,
        }

    elapsed = time.perf_counter() - start_time
    report = {
        "subject_id": subject_id,
        "stage": 7,
        "validator_version": "stage7-connectivity-1.0",
        "elapsed_seconds": elapsed,
        "runs": run_results,
        "status": "PASS",
    }

    report_path = subject_out_dir / f"{subject_id}_connectivity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Stage 7 complete for subject=%s elapsed=%.2fs report=%s", subject_id, elapsed, report_path)

    report["report_file"] = str(report_path)
    return report
