"""Classification metrics for Stage 3 MI decoding."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score


METRIC_COLUMNS = ("balanced_accuracy", "accuracy", "macro_f1", "kappa")


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the Stage 3 primary and supplementary classification metrics."""

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def add_metric_context(metrics: dict[str, float], **context: Any) -> dict[str, Any]:
    """Return a metric record with explicit context such as subject and fold."""

    return {**context, **metrics}
