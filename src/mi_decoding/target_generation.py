"""Stage 3 orchestration for one S001 MI decoding target."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .csp_lda import run_csp_lda_oof
from .event_epochs import create_mi_event_epochs, preprocess_mi_run, resolve_approved_mi_task
from .io import configure_execution_logger, file_sha256, load_yaml, software_versions, write_csv, write_json
from .metrics import compute_classification_metrics


def aggregate_target(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate the canonical Stage 3 target from pooled OOF predictions."""

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Target aggregation requires equal, non-empty true and predicted labels.")
    return compute_classification_metrics(y_true, y_pred)


def run_stage3_target_generation(config_path: str | Path) -> dict[str, Path]:
    """Generate and persist the Stage 3 target for the configured subject.

    The subject is read from the config ``subject_id`` field. Backward
    compatibility: a config with ``subject_id: S001`` reproduces the original
    S001-only behaviour and output filenames exactly.
    """

    config_path = Path(config_path)
    config = load_yaml(config_path)
    if config.get("stage") != 3:
        raise ValueError("Config stage must be 3.")

    output_directory = Path(config["output_directory"])
    logger = configure_execution_logger(output_directory)
    preprocessing_config_path = Path(config["preprocessing_config"])
    preprocessing_config = load_yaml(preprocessing_config_path)
    run_mapping_config_path = Path(config["run_mapping_config"])
    run_mapping_config = load_yaml(run_mapping_config_path)
    mi_runs, event_mapping, task_definition = resolve_approved_mi_task(
        run_mapping_config, str(config["task_definition"])
    )
    dataset_root = Path(config["dataset_root"])
    subject_id = str(config["subject_id"])

    epoch_data: list[np.ndarray] = []
    manifests: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []
    epoch_config = {
        **task_definition["epoch_window"],
        "baseline": task_definition["baseline"]["value"],
        "reject_by_annotation": bool(config["epoch"].get("reject_by_annotation", True)),
    }
    for run_id in mi_runs:
        edf_path = dataset_root / subject_id / f"{subject_id}{run_id}.edf"
        if not edf_path.exists():
            raise FileNotFoundError(f"Missing MI EDF input: {edf_path}")
        logger.info("Processing subject=%s run=%s", subject_id, run_id)
        raw, provenance = preprocess_mi_run(
            edf_path,
            subject_id=subject_id,
            run_id=str(run_id),
            preprocessing_config=preprocessing_config,
        )
        epochs, manifest = create_mi_event_epochs(
            raw,
            subject_id=subject_id,
            run_id=str(run_id),
            event_mapping=event_mapping,
            epoch_config=epoch_config,
        )
        epoch_data.append(epochs.get_data(copy=True))
        manifests.append(manifest)
        run_records.append({**provenance, "run_id": str(run_id), "n_usable_mi_epochs": int(len(epochs))})

    X = np.concatenate(epoch_data, axis=0)
    trial_manifest = pd.concat(manifests, ignore_index=True)
    trial_manifest.insert(0, "trial_id", np.arange(len(trial_manifest), dtype=int))
    y = trial_manifest["true_label"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Stage 3 requires both configured MI classes after epoch extraction.")

    fold_ids, predictions, probabilities, fold_metrics = run_csp_lda_oof(
        X,
        y,
        csp_config=config["csp"],
        lda_config=config["lda"],
        cv_config=config["cross_validation"],
        subject_id=subject_id,
    )
    target_metrics = aggregate_target(y, predictions)
    oof_predictions = trial_manifest.assign(
        fold=fold_ids,
        predicted_label=predictions,
        probability_class_1=probabilities,
    )
    target_row = pd.DataFrame([{"subject_id": subject_id, **target_metrics}])
    fold_metrics_frame = pd.DataFrame(fold_metrics)

    preprocessing_provenance = {
        "config_path": str(preprocessing_config_path),
        "config_sha256": file_sha256(preprocessing_config_path),
        "policy": "Preprocessing is intentionally performed before cross-validation as a fixed validated preprocessing stage, not as a supervised learning step.",
        "runs": run_records,
    }
    report = {
        "subject_id": subject_id,
        "stage": 3,
        "primary_target_metric": "balanced_accuracy",
        "metrics": target_metrics,
        "n_trials_total": int(len(y)),
        "n_trials_per_class": {str(label): int(np.sum(y == label)) for label in sorted(np.unique(y))},
        "cross_validation": config["cross_validation"],
        "decoder": {"csp": config["csp"], "lda": config["lda"]},
        "run_mapping_provenance": {
            "config_path": str(run_mapping_config_path),
            "config_sha256": file_sha256(run_mapping_config_path),
            "task_definition": config["task_definition"],
            "runs": mi_runs,
            "event_mapping": event_mapping,
            "epoch_window": task_definition["epoch_window"],
            "baseline": task_definition["baseline"],
        },
        "random_seed": int(config["random_seed"]),
        "preprocessing_provenance": preprocessing_provenance,
        "software_versions": software_versions(),
    }

    outputs = {
        "target_csv": write_csv(target_row, output_directory / "mi_targets.csv"),
        "oof_predictions_csv": write_csv(oof_predictions, output_directory / f"{subject_id}_oof_predictions.csv"),
        "fold_metrics_csv": write_csv(fold_metrics_frame, output_directory / f"{subject_id}_fold_metrics.csv"),
        "trial_manifest_csv": write_csv(trial_manifest.assign(fold=fold_ids), output_directory / f"{subject_id}_trial_manifest.csv"),
        "run_manifest_json": write_json({"subject_id": subject_id, "runs": run_records}, output_directory / f"{subject_id}_run_manifest.json"),
        "target_report_json": write_json(report, output_directory / f"{subject_id}_target_report.json"),
        "execution_log": output_directory / "execution.log",
    }
    logger.info("Stage 3 complete for subject=%s balanced_accuracy=%.6f", subject_id, target_metrics["balanced_accuracy"])
    return outputs
