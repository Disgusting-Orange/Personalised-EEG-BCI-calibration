"""
pipeline.py
-----------
End-to-end per-run preprocessing orchestration.

Each call to ``preprocess_run`` processes a single EDF file through the
full pipeline and writes the output to disk.  ``preprocess_subject``
iterates over all runs for a subject.  ``preprocess_all`` iterates over
every subject in the dataset.

Pipeline stages (per run)
--------------------------
1.  Load EDF  (loader.load_raw_edf)
2.  Validate  (loader.validate_raw)
3.  Bad-channel detection (epochs.detect_bad_channels)
4.  Bad-channel interpolation (epochs.mark_and_interpolate_bads)
5.  Bandpass + notch filtering (filters.apply_filters)
6.  Downsampling to 128 Hz
7.  Common average re-reference  <--- MOVED HERE (before ICA)
8.  ICA artifact removal
9.  Fixed-length epoching + rejection
10. Visualisation
11. Save epochs
"""

import gc
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import mne
import numpy as np
from . import config as cfg
from .epochs import (
    apply_reference,
    detect_bad_channels,
    epoch_quality_report,
    make_fixed_length_epochs,
    mark_and_interpolate_bads,
    save_epochs,
    save_epochs_numpy,
)
from .filters import apply_filters
from .ica import run_ica_pipeline
from .loader import load_raw_edf, validate_raw
from .utils import (
    ensure_dirs,
    free_memory,
    log_memory,
    output_path_for,
    parse_edf_path,
    setup_logging,
    timer,
)
from .visualizer import (
    plot_channel_stds,
    plot_epoch_image,
    plot_ica_components,
    plot_preprocessing_summary,
    plot_psd_comparison,
    plot_raw_comparison,
)


# ── Per-run pipeline ──────────────────────────────────────────────────────────

def preprocess_run(
    edf_path: Path,
    output_dir: Path,
    figure_dir: Path,
    log_dir: Path,
    subject_logger: logging.Logger,
) -> Dict:
    """
    Preprocess a single EDF run end-to-end.

    Parameters
    ----------
    edf_path         : absolute path to one .edf file
    output_dir       : where preprocessed epochs are saved
    figure_dir       : where figures are saved (per-subject subfolder)
    log_dir          : log directory (already set up by caller)
    subject_logger   : logger shared across runs for this subject

    Returns
    -------
    dict with keys:
        status   : "success" | "skipped" | "failed"
        subject  : subject ID string
        run      : run ID string
        task     : task label
        n_epochs : int (0 on failure)
        message  : optional error message
    """
    meta = parse_edf_path(edf_path)
    subject_id = meta["subject"]
    run_id     = meta["run"]
    task       = cfg.TASK_MAP.get(run_id, "unknown")

    log = subject_logger
    log.info(f"{'='*60}")
    log.info(f"Run: {subject_id}{run_id}  |  Task: {task}")
    log.info(f"{'='*60}")

    result = {
        "status":   "failed",
        "subject":  subject_id,
        "run":      run_id,
        "task":     task,
        "n_epochs": 0,
        "message":  "",
    }

    # Output path check (skip early if not overwriting)
    out_path = output_path_for(output_dir, subject_id, run_id, task)
    if out_path.exists() and not cfg.OVERWRITE:
        log.info(f"Already preprocessed — skipping: {out_path.name}")
        result["status"] = "skipped"
        return result

    # Subject-specific figure directory
    subj_fig_dir = figure_dir / subject_id
    ensure_dirs(subj_fig_dir)

    try:
        # ── Stage 1: Load ────────────────────────────────────────────────────
        with timer("Load", log):
            raw = load_raw_edf(edf_path, log, preload=True)
        if raw is None:
            result["message"] = "EDF load failed"
            return result

        # ── Stage 2: Validate ────────────────────────────────────────────────
        if not validate_raw(raw, log):
            result["message"] = "Validation failed (recording too short)"
            return result

        log_memory(log, "after load")

        # ── Stage 3: Bad-channel detection ───────────────────────────────────
        with timer("Bad-channel detection", log):
            bad_chs = detect_bad_channels(
                raw,
                std_multiplier=cfg.BAD_CHANNEL_STD_MULT,
                flat_std_threshold=cfg.BAD_CHANNEL_FLAT_STD,
                logger=log,
            )

        # ── Stage 4: Channel std visualisation (before interpolation) ────────
        if cfg.SAVE_FIGURES:
            plot_channel_stds(
                raw, bad_chs, subj_fig_dir,
                subject_id, run_id, cfg.SHOW_FIGURES, log,
            )

        # ── Stage 5: Interpolate bad channels ────────────────────────────────
        with timer("Bad-channel interpolation", log):
            raw = mark_and_interpolate_bads(raw, bad_chs, log)

        # ── Stage 6: Bandpass + notch filtering ──────────────────────────────
        with timer("Filtering", log):
            raw_orig_copy, raw_filtered = apply_filters(
                raw,
                bandpass_low=cfg.BANDPASS_LOW,
                bandpass_high=cfg.BANDPASS_HIGH,
                notch_freq=cfg.NOTCH_FREQ,
                notch_width=cfg.NOTCH_WIDTH,
                n_jobs=cfg.N_JOBS,
                logger=log,
            )
        free_memory(raw)   # release unfiltered + uninterpolated copy

        # ── Stage 6.5: Downsample ────────────────────────────────────────────
        with timer("Downsampling", log):
            raw_filtered.resample(
                cfg.TARGET_SFREQ,
                npad="auto"
            )
        log.info(f"Resampled to {raw_filtered.info['sfreq']} Hz")

        # ── Stage 7: Common average re-reference (BEFORE ICA) ──────────────
        with timer("Re-referencing (before ICA)", log):
            raw_filtered = apply_reference(raw_filtered, cfg.REFERENCE, log)

        # ── Stage 8: ICA artifact removal ────────────────────────────────────
        with timer("ICA", log):
            raw_clean, fitted_ica, excluded_comps = run_ica_pipeline(
                raw_filtered=raw_filtered,
                n_components=cfg.ICA_N_COMPONENTS,
                method=cfg.ICA_METHOD,
                max_iter=cfg.ICA_MAX_ITER,
                random_state=cfg.ICA_RANDOM_STATE,
                eog_channels=cfg.ICA_EOG_CHANNELS,
                eog_threshold=cfg.ICA_EOG_THRESHOLD,
                muscle_threshold=cfg.ICA_MUSCLE_THRESHOLD,
                hp_cutoff=cfg.ICA_HIGH_PASS_FOR_FIT,
                n_jobs=cfg.N_JOBS,
                logger=log,
                figure_dir=subj_fig_dir,
                subject_id=subject_id,
                run_id=run_id,
            )

        log_memory(log, "after ICA")

        # ── Stage 9: ICA component visualisation ──────────────────────────────
        if cfg.SAVE_FIGURES:
            # Re-build high-passed copy for plotting (ica module already did
            # this internally but we need it here for properties plots)
            raw_hp_for_plot = raw_filtered.copy()
            raw_hp_for_plot.filter(
                l_freq=cfg.ICA_HIGH_PASS_FOR_FIT,
                h_freq=None,
                verbose=False,
                n_jobs=cfg.N_JOBS,
            )
            plot_ica_components(
                fitted_ica, raw_hp_for_plot, excluded_comps,
                subj_fig_dir, subject_id, run_id, cfg.SHOW_FIGURES, log,
            )
            free_memory(raw_hp_for_plot)

        # Peak amplitude logging (after ICA, already re-referenced)
        log.info(
            f"Peak amplitude after ICA: "
            f"{np.max(np.abs(raw_clean.get_data()))*1e6:.2f} µV"
        )

        # ── Stage 10: Epoching ────────────────────────────────────────────────
        with timer("Epoching", log):
            epochs = make_fixed_length_epochs(
                raw_clean,
                duration=cfg.EPOCH_DURATION,
                overlap=0.0,
                tmin=cfg.EPOCH_TMIN,
                reject=cfg.EPOCH_REJECT,
                flat=cfg.EPOCH_FLAT,
                logger=log,
                task_label=task,
            )

        if epochs is None or len(epochs) == 0:
            result["message"] = "No valid epochs after rejection"
            free_memory(raw_filtered, raw_clean)
            return result

        # ── Stage 11: Quality report ──────────────────────────────────────────
        qr = epoch_quality_report(epochs, log)
        result["n_epochs"] = qr["n_epochs"]

        # ── Stage 12: Visualisations ──────────────────────────────────────────
        if cfg.SAVE_FIGURES:
            plot_raw_comparison(
                raw_orig_copy, raw_filtered, raw_clean,
                cfg.VIZ_CHANNELS, cfg.VIZ_DURATION,
                subj_fig_dir, subject_id, run_id, cfg.SHOW_FIGURES, log,
            )
            plot_psd_comparison(
                raw_orig_copy, raw_filtered, raw_clean,
                cfg.BANDPASS_LOW, cfg.BANDPASS_HIGH,
                subj_fig_dir, subject_id, run_id, cfg.SHOW_FIGURES, log,
            )
            plot_epoch_image(
                epochs, cfg.VIZ_CHANNELS[2] if len(cfg.VIZ_CHANNELS) > 2 else epochs.ch_names[0],
                subj_fig_dir, subject_id, run_id, cfg.SHOW_FIGURES, log,
            )
            plot_preprocessing_summary(
                raw_orig_copy, raw_clean, epochs,
                bad_chs, excluded_comps,
                subj_fig_dir, subject_id, run_id, cfg.SHOW_FIGURES, log,
            )

        # Free large Raw objects before saving
        free_memory(raw_orig_copy, raw_filtered, raw_clean)

        # ── Stage 13: Save ────────────────────────────────────────────────────
        with timer("Save", log):
            if cfg.SAVE_FORMAT == "fif":
                save_epochs(epochs, out_path, cfg.OVERWRITE, log)
            else:
                save_epochs_numpy(
                    epochs,
                    out_path.with_suffix(".npz"),
                    cfg.NUMPY_DTYPE,
                    cfg.OVERWRITE,
                    log,
                )

        free_memory(epochs)
        gc.collect()

        result["status"] = "success"
        log.info(
            f"Run complete: {subject_id}{run_id} | "
            f"{result['n_epochs']} epochs | task={task}"
        )

    except Exception as exc:
        result["message"] = str(exc)
        log.error(
            f"Unexpected error in {subject_id}{run_id}:\n"
            f"{traceback.format_exc()}"
        )
        gc.collect()

    return result


# ── Per-subject pipeline ──────────────────────────────────────────────────────

def preprocess_subject(
    subject_id: str,
    data_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    log_dir: Path,
    runs: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Preprocess all (or selected) runs for one subject.

    Parameters
    ----------
    subject_id  : e.g. "S001"
    data_dir    : root data directory containing S001/, S002/, …
    output_dir  : root preprocessed output directory
    figure_dir  : root figure directory
    log_dir     : log directory
    runs        : if given, only process these run IDs e.g. ["R03","R04"]

    Returns
    -------
    List of per-run result dicts from ``preprocess_run``.
    """
    logger = setup_logging(log_dir, subject_id)
    logger.info(f"{'#'*60}")
    logger.info(f"Subject: {subject_id}")
    logger.info(f"{'#'*60}")

    subject_dir = data_dir / subject_id
    if not subject_dir.exists():
        logger.error(f"Subject directory not found: {subject_dir}")
        return []

    # Discover EDF files for this subject
    edf_files = sorted(
        p for p in subject_dir.glob("*.edf")
        if not p.name.endswith(".edf.event")
    )

    if runs:
        edf_files = [
            p for p in edf_files
            if any(p.stem.endswith(r) for r in runs)
        ]

    if not edf_files:
        logger.warning(f"No EDF files found for {subject_id}")
        return []

    logger.info(f"Found {len(edf_files)} run(s) for {subject_id}")

    results = []
    for edf_path in edf_files:
        run_result = preprocess_run(
            edf_path=edf_path,
            output_dir=output_dir,
            figure_dir=figure_dir,
            log_dir=log_dir,
            subject_logger=logger,
        )
        results.append(run_result)

    # Subject-level summary
    n_ok  = sum(1 for r in results if r["status"] == "success")
    n_sk  = sum(1 for r in results if r["status"] == "skipped")
    n_err = sum(1 for r in results if r["status"] == "failed")
    logger.info(
        f"Subject {subject_id} done: "
        f"{n_ok} success | {n_sk} skipped | {n_err} failed"
    )
    return results


# ── Full dataset pipeline ─────────────────────────────────────────────────────

def preprocess_all(
    data_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    log_dir: Path,
    subjects: Optional[List[str]] = None,
    runs: Optional[List[str]] = None,
) -> Dict:
    """
    Preprocess all subjects and runs in the dataset.

    Parameters
    ----------
    data_dir    : root data directory
    output_dir  : root output directory
    figure_dir  : root figure directory
    log_dir     : log directory
    subjects    : if given, only process these subject IDs
    runs        : if given, only process these run IDs

    Returns
    -------
    Summary dict with overall statistics and per-subject results.
    """
    ensure_dirs(output_dir, figure_dir, log_dir)
    pipeline_logger = setup_logging(log_dir)
    pipeline_logger.info("EEG Preprocessing Pipeline — START")
    pipeline_logger.info(f"Data dir  : {data_dir}")
    pipeline_logger.info(f"Output dir: {output_dir}")
    pipeline_logger.info(f"Figure dir: {figure_dir}")

    # Discover subjects
    if subjects:
        subject_ids = sorted(subjects)
    else:
        subject_ids = sorted(
            p.name for p in data_dir.iterdir()
            if p.is_dir() and p.name.startswith("S")
        )

    pipeline_logger.info(f"Subjects to process: {len(subject_ids)}")

    all_results = {}
    total = {"success": 0, "skipped": 0, "failed": 0, "total_epochs": 0}

    for subj in subject_ids:
        with timer(f"Subject {subj}", pipeline_logger):
            subj_results = preprocess_subject(
                subject_id=subj,
                data_dir=data_dir,
                output_dir=output_dir,
                figure_dir=figure_dir,
                log_dir=log_dir,
                runs=runs,
            )
        all_results[subj] = subj_results
        for r in subj_results:
            total[r["status"]] = total.get(r["status"], 0) + 1
            total["total_epochs"] += r.get("n_epochs", 0)

    pipeline_logger.info(
        f"\n{'='*60}\n"
        f"Pipeline complete\n"
        f"  Success : {total['success']}\n"
        f"  Skipped : {total['skipped']}\n"
        f"  Failed  : {total['failed']}\n"
        f"  Total epochs saved: {total['total_epochs']}\n"
        f"{'='*60}"
    )

    return {"summary": total, "per_subject": all_results}