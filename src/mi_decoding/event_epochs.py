"""Event-locked MI epoch extraction using the frozen Stage 2 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

try:
    from src.preprocessing.artifacts import run_bad_channel_pipeline, run_ica_pipeline
    from src.preprocessing.filters import load_raw, run_filter_pipeline
    from src.preprocessing.qc import run_qc_pipeline
    from src.preprocessing.utils import get_logger
except ModuleNotFoundError:
    from preprocessing.artifacts import run_bad_channel_pipeline, run_ica_pipeline
    from preprocessing.filters import load_raw, run_filter_pipeline
    from preprocessing.qc import run_qc_pipeline
    from preprocessing.utils import get_logger


def preprocess_mi_run(
    edf_path: str | Path,
    *,
    subject_id: str,
    run_id: str,
    preprocessing_config: dict[str, Any],
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Apply frozen Stage 2 preprocessing to one MI run without modifying it.

    This intentionally occurs before CV as a fixed, validated preprocessing
    stage. It is not a supervised learning transformation.
    """

    logger = get_logger("mi_decoding.preprocessing")
    raw = load_raw(edf_path, preload=True, verbose=False)
    original_sfreq = float(raw.info["sfreq"])
    filtered = run_filter_pipeline(
        raw, subject=subject_id, run=run_id, config=preprocessing_config, logger=logger
    )
    bad_channel_cleaned = run_bad_channel_pipeline(
        filtered, subject=subject_id, run=run_id, config=preprocessing_config, logger=logger
    )
    cleaned, ica = run_ica_pipeline(
        bad_channel_cleaned,
        subject=subject_id,
        run=run_id,
        config=preprocessing_config,
        logger=logger,
    )
    qc_report = run_qc_pipeline(
        raw=cleaned,
        subject=subject_id,
        run=run_id,
        config=preprocessing_config,
        logger=logger,
    )
    provenance = {
        "input_edf": str(Path(edf_path)),
        "original_sampling_frequency_hz": original_sfreq,
        "output_sampling_frequency_hz": float(cleaned.info["sfreq"]),
        "channel_names": list(cleaned.ch_names),
        "ica_model_fitted": ica is not None,
        "qc_report": qc_report,
        "pre_cv_policy": "Frozen validated preprocessing is applied before CV; CSP and LDA are fit within CV folds.",
    }
    return cleaned, provenance


def create_mi_event_epochs(
    raw: mne.io.BaseRaw,
    *,
    subject_id: str,
    run_id: str,
    event_mapping: dict[str, dict[str, Any]],
    epoch_config: dict[str, Any],
) -> tuple[mne.Epochs, pd.DataFrame]:
    """Create event-locked binary MI epochs and a traceable trial manifest."""

    if set(event_mapping) != {"T1", "T2"}:
        raise ValueError("Stage 3 requires an explicit binary mapping for exactly T1 and T2.")

    annotation_event_id = {description: index + 1 for index, description in enumerate(sorted(event_mapping))}
    events, event_id = mne.events_from_annotations(raw, event_id=annotation_event_id, verbose=False)
    if len(events) == 0:
        raise ValueError(f"No T1/T2 events found for {subject_id} {run_id}.")

    inverse_event_id = {value: key for key, value in event_id.items()}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=float(epoch_config["tmin_seconds"]),
        tmax=float(epoch_config["tmax_seconds"]),
        baseline=epoch_config.get("baseline"),
        preload=True,
        reject_by_annotation=bool(epoch_config.get("reject_by_annotation", True)),
        verbose=False,
    )
    if len(epochs) == 0:
        raise ValueError(f"No usable MI epochs remain for {subject_id} {run_id}.")

    records: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(epochs.selection):
        event = events[source_index]
        event_code = inverse_event_id[int(event[2])]
        mapping = event_mapping[event_code]
        records.append(
            {
                "subject_id": subject_id,
                "run_id": run_id,
                "run_epoch_index": output_index,
                "annotation_event_index": int(source_index),
                "event_sample": int(event[0]),
                "event_time_seconds": float(event[0] / raw.info["sfreq"]),
                "event_code": event_code,
                "true_label": int(mapping["label"]),
                "class_name": str(mapping["class_name"]),
            }
        )
    return epochs, pd.DataFrame.from_records(records)


def resolve_approved_mi_task(run_mapping_config: dict[str, Any], task_definition: str) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Resolve one approved MI task and reject incompatible run reuse."""

    task_definitions = run_mapping_config.get("task_definitions", {})
    if task_definition not in task_definitions:
        raise ValueError(f"Unknown approved task definition: {task_definition}")
    task = task_definitions[task_definition]
    if task.get("role") != "motor_imagery":
        raise ValueError(f"Task definition {task_definition} is not motor imagery.")
    runs = [str(run_id) for run_id in task.get("runs", [])]
    event_mapping = task.get("event_mapping")
    if not runs or not isinstance(event_mapping, dict):
        raise ValueError(f"Task definition {task_definition} must define runs and a T1/T2 mapping.")
    if set(event_mapping) != {"T1", "T2"}:
        raise ValueError(f"Task definition {task_definition} must define only T1 and T2 mappings.")
    run_entries = run_mapping_config.get("run_mapping", {})
    for run_id in runs:
        run = run_entries.get(run_id, {})
        if run.get("role") != "motor_imagery" or run.get("task_definition") != task_definition:
            raise ValueError(f"Run {run_id} is not approved for task definition {task_definition}.")
    return runs, event_mapping, task
