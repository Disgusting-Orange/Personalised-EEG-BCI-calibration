"""Quality-control reporting for Stage 2E.

This module inspects already-produced preprocessing outputs and generates
summary metrics and lightweight plots without altering the EEG recordings or
Epochs objects.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml

from .utils import get_logger


def compute_qc_metrics(
    raw: Optional[mne.io.BaseRaw] = None,
    epochs: Optional[mne.Epochs] = None,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    bad_channels: Optional[list[str]] = None,
    ica_components_removed: Optional[list[int]] = None,
    logger: Optional[Any] = None,
) -> dict[str, Any]:
    """Compute read-only QC metrics from Raw and Epochs objects."""

    if logger is None:
        logger = get_logger("preprocessing.qc")

    subject_name = subject or "unknown"
    run_name = run or "unknown"

    metrics: dict[str, Any] = {
        "subject": subject_name,
        "run": run_name,
        "n_channels": 0,
        "sampling_frequency_hz": None,
        "recording_duration_seconds": None,
        "n_generated_epochs": 0,
        "n_usable_epochs": 0,
        "n_rejected_epochs": 0,
        "rejection_reasons": [],
        "n_bad_channels_detected": 0,
        "ica_components_removed": [],
        "channel_names": [],
        "data_shape": None,
    }

    if raw is not None:
        data = raw.get_data(picks="eeg")
        metrics.update(
            {
                "n_channels": int(data.shape[0]),
                "sampling_frequency_hz": float(raw.info["sfreq"]),
                "recording_duration_seconds": float(raw.times[-1]) if len(raw.times) else 0.0,
                "channel_names": list(raw.ch_names),
                "data_shape": [int(data.shape[0]), int(data.shape[1])],
            }
        )
        metrics["n_bad_channels_detected"] = len(bad_channels or [])

    if epochs is not None:
        generated_epochs = int(len(epochs.events))
        usable_epochs = int(len(epochs))
        rejected_epochs = max(0, generated_epochs - usable_epochs)
        metrics.update(
            {
                "n_generated_epochs": generated_epochs,
                "n_usable_epochs": usable_epochs,
                "n_rejected_epochs": rejected_epochs,
                "rejection_reasons": [],
            }
        )
        if hasattr(epochs, "drop_log") and epochs.drop_log:
            metrics["rejection_reasons"] = [str(entry) for entry in epochs.drop_log]

    if ica_components_removed is not None:
        metrics["ica_components_removed"] = list(ica_components_removed)

    logger.info("Computed QC metrics for subject=%s run=%s", subject_name, run_name)
    return metrics


def generate_qc_report(
    raw: Optional[mne.io.BaseRaw] = None,
    epochs: Optional[mne.Epochs] = None,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    bad_channels: Optional[list[str]] = None,
    ica_components_removed: Optional[list[int]] = None,
    logger: Optional[Any] = None,
) -> dict[str, Any]:
    """Generate a QC report dictionary from Raw and Epochs metadata."""

    metrics = compute_qc_metrics(
        raw=raw,
        epochs=epochs,
        subject=subject,
        run=run,
        bad_channels=bad_channels,
        ica_components_removed=ica_components_removed,
        logger=logger,
    )
    return {"subject": metrics["subject"], "run": metrics["run"], "metrics": metrics}


def save_qc_outputs(
    report: dict[str, Any],
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    output_dir: Optional[str | os.PathLike[str]] = None,
    save_plots: bool = False,
    report_format: str = "json",
    logger: Optional[Any] = None,
) -> list[Path]:
    """Save QC report files and optional lightweight plots."""

    if logger is None:
        logger = get_logger("preprocessing.qc")

    base_dir = Path(output_dir) if output_dir is not None else Path("outputs") / "preprocessed" / "qc"
    base_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    subject_name = subject or "unknown"
    run_name = run or "unknown"
    output_files: list[Path] = []

    report_path = base_dir / f"{subject_name}_{run_name}_qc_report.{report_format}"
    if report_format == "json":
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif report_format in {"yaml", "yml"}:
        report_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    else:
        report_path.write_text(str(report), encoding="utf-8")
    output_files.append(report_path)

    summary_path = base_dir / f"{subject_name}_{run_name}_summary.txt"
    summary_path.write_text(
        f"Subject: {subject_name}\nRun: {run_name}\nMetrics: {json.dumps(report['metrics'], indent=2)}\n",
        encoding="utf-8",
    )
    output_files.append(summary_path)

    if save_plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        metrics = report["metrics"]
        ax.bar(["epochs", "channels"], [metrics.get("n_generated_epochs", 0), metrics.get("n_channels", 0)])
        ax.set_title(f"QC summary for {subject_name} {run_name}")
        ax.set_ylabel("Count")
        fig.tight_layout()
        plot_path = plots_dir / f"{subject_name}_{run_name}_summary.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        output_files.append(plot_path)

    logger.info("Saved QC outputs to %s", base_dir)
    return output_files


def run_qc_pipeline(
    raw: Optional[mne.io.BaseRaw] = None,
    epochs: Optional[mne.Epochs] = None,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    bad_channels: Optional[list[str]] = None,
    ica_components_removed: Optional[list[int]] = None,
    config: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
    output_dir: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    """Run the full QC reporting workflow for a preprocessed Raw/Epochs pair."""

    if logger is None:
        logger = get_logger("preprocessing.qc")

    config = config or {}
    qc_cfg = config.get("quality_control", {})
    start_time = time.perf_counter()

    if not qc_cfg.get("enabled", False):
        logger.info("QC disabled; no output generated")
        return {"status": "disabled"}

    report = generate_qc_report(
        raw=raw,
        epochs=epochs,
        subject=subject,
        run=run,
        bad_channels=bad_channels,
        ica_components_removed=ica_components_removed,
        logger=logger,
    )
    output_files = save_qc_outputs(
        report,
        subject=subject,
        run=run,
        output_dir=qc_cfg.get("output_directory", output_dir),
        save_plots=bool(qc_cfg.get("save_plots", False)),
        report_format=qc_cfg.get("report_format", "json"),
        logger=logger,
    )

    elapsed = time.perf_counter() - start_time
    logger.info("QC pipeline complete for subject=%s run=%s elapsed=%.3fs files=%d", subject, run, elapsed, len(output_files))
    report["output_files"] = [str(path) for path in output_files]
    return report
