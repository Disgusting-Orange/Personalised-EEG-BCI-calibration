"""Configuration, provenance, and output helpers for Stage 3."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import sklearn
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return data


def file_sha256(path: str | Path) -> str:
    """Return a SHA-256 digest for provenance."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def software_versions() -> dict[str, str]:
    """Collect the core runtime versions used by Stage 3."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "mne": mne.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def configure_execution_logger(output_directory: str | Path) -> logging.Logger:
    """Configure an isolated Stage 3 execution log."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mi_decoding.stage3")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler = logging.FileHandler(output_path / "execution.log", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a UTF-8 CSV without a pandas index."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a structured JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return output_path
