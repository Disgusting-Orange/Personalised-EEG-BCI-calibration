"""
utils.py
--------
Shared utilities: logging, directory management, timing, memory tracking.
"""

import gc
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import mne
import psutil


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(
    log_dir: Path,
    subject_id: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and return a logger that writes to console *and* a per-subject
    (or global) log file.

    Parameters
    ----------
    log_dir    : directory to write log files in (created if absent)
    subject_id : if given, log file is ``<subject_id>.log``; else ``pipeline.log``
    level      : Python logging level

    Returns
    -------
    logging.Logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{subject_id}.log" if subject_id else "pipeline.log"
    log_file = log_dir / fname

    logger_name = f"eeg.{subject_id}" if subject_id else "eeg.pipeline"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid duplicate handlers if the function is called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Suppress MNE's own verbose output unless we want DEBUG
    mne.set_log_level("WARNING" if level > logging.DEBUG else "INFO")

    return logger


# ── Directory helpers ──────────────────────────────────────────────────────────

def ensure_dirs(*dirs: Path) -> None:
    """Create all listed directories (including parents) if they do not exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def output_path_for(
    output_dir: Path,
    subject_id: str,
    run_id: str,
    task: str,
    suffix: str = "_epo.fif",
) -> Path:
    """
    Return the canonical output path for a preprocessed epochs file.

    Structure: ``<output_dir>/<subject_id>/<subject_id>_<run_id>_<task><suffix>``
    """
    subject_dir = output_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{subject_id}_{run_id}_{task}{suffix}"
    return subject_dir / filename


# ── Timing ────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None):
    """Context manager that logs elapsed time for a code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        msg = f"{label} completed in {elapsed:.2f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)


# ── Memory ────────────────────────────────────────────────────────────────────

def log_memory(logger: logging.Logger, label: str = "") -> None:
    """Log current process RSS memory usage."""
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / 1_048_576
    tag = f"[{label}] " if label else ""
    logger.debug(f"{tag}Memory usage: {rss_mb:.1f} MB")


def free_memory(*objects) -> None:
    """Delete objects and run garbage collection to release memory."""
    for obj in objects:
        del obj
    gc.collect()


# ── EDF path parsing ──────────────────────────────────────────────────────────

def parse_edf_path(edf_path: Path) -> dict:
    """
    Extract subject ID and run ID from an EDF file path.

    Expected format: ``…/<SUBJECT_ID>/<SUBJECT_ID><RUN_ID>.edf``
    e.g.  ``S001/S001R03.edf``  →  subject="S001", run="R03"

    Returns
    -------
    dict with keys "subject", "run"
    """
    stem = edf_path.stem          # e.g. "S001R03"
    subject = stem[:4]            # "S001"
    run = stem[4:]                # "R03"
    return {"subject": subject, "run": run}
