"""Fixed-length epoch extraction for Stage 2D.

This module converts an already preprocessed MNE Raw object into fixed-length
epochs using MNE's epoching utilities. It is intentionally limited to epoch
creation and saving, without quality-control or feature extraction.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import mne

from .utils import get_logger


def create_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    *,
    duration: float = 2.0,
    overlap: float = 0.0,
    reject_by_annotation: bool = True,
    baseline: Optional[tuple[float, float]] = None,
    preload: bool = True,
    logger: Optional[Any] = None,
) -> mne.Epochs:
    """Create fixed-length epochs from a preprocessed Raw object."""

    if logger is None:
        logger = get_logger("preprocessing.epochs")

    if baseline is None:
        baseline = None

    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=None,
        tmin=0.0,
        tmax=duration,
        baseline=baseline,
        preload=preload,
        reject_by_annotation=reject_by_annotation,
        verbose=False,
    )

    generated_epochs = int(len(epochs.events))
    usable_epochs = int(len(epochs))
    rejected_epochs = max(0, generated_epochs - usable_epochs)

    logger.info(
        "Created %d epochs with duration=%.2fs overlap=%.2fs (usable=%d rejected=%d)",
        generated_epochs,
        duration,
        overlap,
        usable_epochs,
        rejected_epochs,
    )
    return epochs


def save_epochs(
    epochs: mne.Epochs,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    output_dir: Optional[str | os.PathLike[str]] = None,
    overwrite: bool = False,
    logger: Optional[Any] = None,
) -> Path:
    """Persist an Epochs object to disk with a deterministic file name."""

    if logger is None:
        logger = get_logger("preprocessing.epochs")

    base_dir = Path(output_dir) if output_dir is not None else Path("outputs") / "preprocessed" / "epochs"
    base_dir.mkdir(parents=True, exist_ok=True)

    subject_name = subject or "unknown"
    run_name = run or "unknown"
    output_path = base_dir / f"{subject_name}_{run_name}_epochs-epo.fif"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Epoch file already exists and overwrite=False: {output_path}")

    epochs.save(output_path, overwrite=overwrite, verbose=False)
    logger.info("Saved epochs to %s", output_path)
    return output_path


def run_epoch_pipeline(
    raw: mne.io.BaseRaw,
    *,
    subject: Optional[str] = None,
    run: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
    save_epochs_output: bool = False,
    output_dir: Optional[str | os.PathLike[str]] = None,
    overwrite: bool = False,
) -> tuple[mne.Epochs, Optional[Path]]:
    """Run the fixed-length epoch extraction pipeline on a preprocessed Raw object."""

    if logger is None:
        logger = get_logger("preprocessing.epochs")

    config = config or {}
    epoch_cfg = config.get("epoch", {})
    start_time = time.perf_counter()

    subject_name = subject or "unknown"
    run_name = run or "unknown"

    if not epoch_cfg.get("enabled", False):
        logger.info("Epoch extraction disabled; returning None")
        return None, None  # type: ignore[return-value]

    duration = float(epoch_cfg.get("duration_seconds", 2.0))
    overlap = float(epoch_cfg.get("overlap_seconds", 0.0))
    reject_by_annotation = bool(epoch_cfg.get("reject_by_annotation", True))
    baseline = tuple(epoch_cfg.get("baseline", (None, 0.0))) if epoch_cfg.get("baseline") else None
    preload = bool(epoch_cfg.get("preload", True))

    logger.info(
        "Starting epoch pipeline for subject=%s run=%s duration=%.2fs overlap=%.2fs",
        subject_name,
        run_name,
        duration,
        overlap,
    )

    epochs = create_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        reject_by_annotation=reject_by_annotation,
        baseline=baseline,
        preload=preload,
        logger=logger,
    )

    generated_epochs = int(len(epochs.events))
    usable_epochs = int(len(epochs))
    rejected_epochs = max(0, generated_epochs - usable_epochs)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Epoch pipeline complete for subject=%s run=%s elapsed=%.3fs epochs_created=%d epochs_dropped=%d",
        subject_name,
        run_name,
        elapsed,
        generated_epochs,
        rejected_epochs,
    )

    output_path: Optional[Path] = None
    if save_epochs_output:
        output_path = save_epochs(
            epochs,
            subject=subject_name,
            run=run_name,
            output_dir=output_dir,
            overwrite=overwrite,
            logger=logger,
        )

    return epochs, output_path
