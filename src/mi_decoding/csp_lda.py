"""Leakage-safe CSP + LDA decoding with deterministic stratified CV."""

from __future__ import annotations

from typing import Any

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .metrics import add_metric_context, compute_classification_metrics


def generate_stratified_folds(y: np.ndarray, cv_config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic, non-overlapping StratifiedKFold assignments."""

    if cv_config.get("strategy") != "StratifiedKFold":
        raise ValueError("Stage 3 supports StratifiedKFold only.")
    splitter = StratifiedKFold(
        n_splits=int(cv_config["n_splits"]),
        shuffle=bool(cv_config["shuffle"]),
        random_state=int(cv_config["random_state"]),
    )
    folds = [(train.copy(), test.copy()) for train, test in splitter.split(np.zeros(len(y)), y)]
    test_indices = np.concatenate([test for _, test in folds])
    if len(np.unique(test_indices)) != len(y) or set(test_indices) != set(range(len(y))):
        raise RuntimeError("Invalid fold assignment: each trial must occur once in held-out data.")
    for train, test in folds:
        if np.intersect1d(train, test).size:
            raise RuntimeError("Leakage detected: a trial appears in both train and test partitions.")
    return folds


def build_csp_lda_pipeline(csp_config: dict[str, Any], lda_config: dict[str, Any]) -> Pipeline:
    """Build the configurable decoder. Fitting occurs only inside each fold."""

    csp_parameters = {key: value for key, value in csp_config.items() if value is not None}
    lda_parameters = {key: value for key, value in lda_config.items() if value is not None}
    return Pipeline(
        [
            ("csp", CSP(**csp_parameters)),
            ("lda", LinearDiscriminantAnalysis(**lda_parameters)),
        ]
    )


def run_csp_lda_oof(
    X: np.ndarray,
    y: np.ndarray,
    *,
    csp_config: dict[str, Any],
    lda_config: dict[str, Any],
    cv_config: dict[str, Any],
    subject_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return fold IDs, held-out predictions, probabilities, and fold metrics."""

    if X.ndim != 3:
        raise ValueError("CSP input must have shape (n_trials, n_channels, n_times).")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of trials.")

    fold_ids = np.full(len(y), -1, dtype=int)
    predictions = np.full(len(y), -1, dtype=int)
    positive_probabilities = np.full(len(y), np.nan, dtype=float)
    fold_metrics: list[dict[str, Any]] = []

    for fold_id, (train_index, test_index) in enumerate(generate_stratified_folds(y, cv_config), start=1):
        decoder = build_csp_lda_pipeline(csp_config, lda_config)
        decoder.fit(X[train_index], y[train_index])
        y_pred = decoder.predict(X[test_index]).astype(int)
        probabilities = decoder.predict_proba(X[test_index])
        class_positions = {int(label): position for position, label in enumerate(decoder.classes_)}
        if 1 not in class_positions:
            raise RuntimeError("Expected binary class label 1 is absent from a training fold.")

        fold_ids[test_index] = fold_id
        predictions[test_index] = y_pred
        positive_probabilities[test_index] = probabilities[:, class_positions[1]]
        metrics = compute_classification_metrics(y[test_index], y_pred)
        fold_metrics.append(
            add_metric_context(
                metrics,
                subject_id=subject_id,
                fold=fold_id,
                n_train_trials=int(len(train_index)),
                n_test_trials=int(len(test_index)),
                n_train_class_0=int(np.sum(y[train_index] == 0)),
                n_train_class_1=int(np.sum(y[train_index] == 1)),
                n_test_class_0=int(np.sum(y[test_index] == 0)),
                n_test_class_1=int(np.sum(y[test_index] == 1)),
            )
        )

    if np.any(fold_ids < 0) or np.any(predictions < 0):
        raise RuntimeError("Incomplete out-of-fold prediction assignment.")
    return fold_ids, predictions, positive_probabilities, fold_metrics
